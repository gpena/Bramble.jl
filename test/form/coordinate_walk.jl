module FormCoordinateWalkTests

using Test
using Bramble
using Random
using SparseArrays: SparseMatrixCSC, sparse, findnz, nonzeros, nnz
using Bramble:
               form,
               assemble,
               assemble_parallel!,
               allocate_system_matrix,
               resolve_form_ast,
               visit_bilinear_stencil,
               PatternSink,
               shift_op,
               D₋ₓ,
               D₊ₓ,
               Dcₓ,
               D₋ᵧ,
               D₊ᵧ,
               Dcᵧ

# The coordinate walk (`_form_coordinates`, form/bilinear_pattern.jl) is the one setup walk
# of every (term, block) unit: its coordinates build the sparsity pattern and, searched in the
# matrix, the positions every fill replays. Each form below is checked three ways: the
# coordinates name exactly the stored entries of the allocated matrix; every nonzero of an
# independently scattered matrix (the threaded sweep, which searches each entry) is among
# them; and the replayed values equal that sweep's.

# The set of `(row, col)` the walk wrote, and the unit bookkeeping it kept.
function _walk(f)
    p = Bramble._form_coordinates(f.trial_space, f.test_space, resolve_form_ast(f))
    return p, Set(zip(p.I, p.J))
end

_stored(A::SparseMatrixCSC) = Set(zip(findnz(A)[1], findnz(A)[2]))

function _check_form(f)
    p, coords = _walk(f)
    A = assemble(f)
    @test coords == _stored(A)

    # Every unit's `point_ptr` opens one slice per grid point (interior box first, so not in
    # linear-index order) and closes on its count.
    total = 0
    for u in eachindex(p.counts)
        ptr = p.ptrs[u]
        @test minimum(ptr) == 1
        @test ptr[end] == p.counts[u] + 1
        n = p.counts[u]
        h = p.halves[u]
        total += h == -1 ? n : (h == 0 ? 2n : n)
    end
    @test total == length(p.I)

    # The threaded sweep searches every entry itself, sharing no positions with the replay.
    C = copy(A)
    assemble_parallel!(C, f)
    I, J, V = findnz(C)
    @test all(((i, j),) -> (i, j) in coords, zip(I[V .!= 0], J[V .!= 0]))
    @test nonzeros(A) ≈ nonzeros(C)

    # A matrix the cache has never seen records afresh (the cache-miss path) and agrees.
    B = copy(A)
    fill!(nonzeros(B), 0.0)
    assemble!(B, f)
    @test nonzeros(B) == nonzeros(A)
    return p
end

@testset "Coordinate walk" begin
    Random.seed!(20260924)
    S = interval(0.0, 1.0) × interval(0.0, 1.0)
    Ωₕ = mesh(domain(S, :walls => boundary_symbols(S)), (9, 8), (false, false))
    Wₕ = gridspace(Ωₕ)
    Vₕ = gridspace(Ωₕ, Val(2))
    cₕ = element(Wₕ, rand(ndofs(Wₕ)) .+ 1.0)

    @testset "scalar" begin
        f = form(Wₕ, Wₕ, (u, v) -> innerₕ(D₋ₓ(u), D₋ₓ(v)) + innerₕ(D₋ᵧ(u), D₋ᵧ(v)))
        p = _check_form(f)
        # One unit per summand, walked alone.
        @test p.halves == [-1, -1]
        # The pattern `PatternSink` finds walking the whole fused sum names the same entries.
        ast = resolve_form_ast(f)
        pat = visit_bilinear_stencil(PatternSink(Int[], Int[]), ast, Wₕ, 0, 0)
        @test Set(zip(pat.I_vec, pat.J_vec)) == _walk(f)[2]
    end

    @testset "coefficient" begin
        _check_form(form(Wₕ, Wₕ, (u, v) -> innerₕ(cₕ * D₋ₓ(u), D₋ₓ(v)) + 2.0 * innerₕ(u, v)))
    end

    @testset "restricted" begin
        _check_form(form(Wₕ, Wₕ,
            (u, v) -> inner₊(D₋ₓ(u), D₋ₓ(v); markers = (:walls,)) + innerₕ(u, v)))
    end

    @testset "composite" begin
        _check_form(form(Vₕ, Vₕ, (u, v) -> innerₕ(εcₕ(u), εcₕ(v))))
    end

    @testset "coupled" begin
        _check_form(form(Vₕ, Vₕ,
            (u, v) -> innerₕ(D₋ₓ(u(1)), D₋ᵧ(v(2))) + innerₕ(u(2), v(1)) +
                      inner₊(∇ₕ(u(1)), ∇ₕ(v(1)))))
    end

    @testset "shift" begin
        _check_form(form(Wₕ, Wₕ,
            (u, v) -> innerₕ(shift_op(u, 1, 1), shift_op(v, 2, 1)) + innerₕ(Mₓ(u), D₊ᵧ(v))))
    end

    @testset "transposed pair" begin
        # Scalar: one unit writes both halves.
        f = form(Wₕ, Wₕ, (u, v) -> innerₕ(D₋ₓ(u), D₊ᵧ(v)) + innerₕ(D₊ᵧ(u), D₋ₓ(v)))
        p = _check_form(f)
        @test p.halves == [0]
        # The two halves are each other's transposes.
        n = p.counts[1]
        @test Set(zip(p.I[1:n], p.J[1:n])) == Set(zip(p.J[(n + 1):2n], p.I[(n + 1):2n]))

        # Composite, one leaf object: the transposed half lands in the second term's block.
        g = form(Vₕ, Vₕ, (u, v) -> innerₕ(Dcᵧ(u(1)), Dcₓ(v(2))) + innerₕ(Dcₓ(u(2)), Dcᵧ(v(1))))
        @test _check_form(g).halves == [0]

        # Two distinct leaf objects: one half on each leaf, neither searched twice.
        Sd = Bramble.CompositeGridSpace((Wₕ, gridspace(Ωₕ)))
        h = form(Sd, Sd, (u, v) -> innerₕ(Dcᵧ(u(1)), Dcₓ(v(2))) + innerₕ(Dcₓ(u(2)), Dcᵧ(v(1))))
        ph = _check_form(h)
        @test ph.halves == [1, 2]
        @test length(ph.I) == sum(ph.counts)
        A = assemble(h)
        assemble!(A, h)
        @test (@allocated assemble!(A, h)) == 0
    end

    @testset "1D" begin
        Ω1 = mesh(domain(interval(0.0, 1.0)), 21, false)
        W1 = gridspace(Ω1)
        f = form(W1, W1, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(u, v))
        _check_form(f)
        # The first fill is the replay `allocate_system_matrix` stored, diagonal in 1D.
        A = assemble(f)
        @test f.cache.valid && f.cache.A_id == objectid(A)
        @test all(s -> s.is_diagonal, f.cache.segments)
    end

    @testset "from 2D up every segment is flat" begin
        f = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
        assemble(f)
        @test !any(s -> s.is_diagonal, f.cache.segments)
    end
end

end # module FormCoordinateWalkTests
