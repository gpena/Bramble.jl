module TestFormMatrixFree

using Test
using Bramble
using Bramble: MatrixFreeOperator, VectorElement, trial_space, test_space, restrict_to, πₕ, jumpₓ, M₊ₓ, D₋ₓ, D₋ᵧ
using LinearAlgebra: mul!, norm
using Random

# `matrix_free_operator` (gpena/Bramble.jl#326) applies a bilinear form through the walk the
# assembly replays, so every check compares it against `assemble(a; dirichlet)` on the same
# vector. Meshes are non-uniform throughout: a uniform mesh hides a spacing read on the wrong
# side of a point.

const MF_SEED = 3261

_mf_agree(a, b) = isapprox(a, b; rtol = 1e-12, atol = 1e-12 * max(1.0, norm(b, Inf)))

# Inside functions so `@allocated` measures the call alone, not global-scope boxing.
_mf_alloc3(y, op, x) = (mul!(y, op, x); @allocated mul!(y, op, x))
_mf_alloc5(y, op, x) = (mul!(y, op, x, 0.5, 2.0); @allocated mul!(y, op, x, 0.5, 2.0))
_mf_alloc_times(op, x) = (op * x; @allocated op * x)

function _mf_spaces()
    Random.seed!(MF_SEED)
    return (
        gridspace(mesh(domain(interval(0.0, 1.0), :west => :left), 17, false)),
        gridspace(mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0), :west => :left), (9, 11), (false, true))),
        gridspace(mesh(
            domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0), :west => :left), (
                6, 7, 5), false))
    )
end

# (name, form, dirichlet): variable diffusion, jump/average/difference, a region restriction,
# a transposed pair, per dimension; then composite spaces with crossed components, one on a
# single leaf object and one on two, so both halves of the pair walk run.
function _mf_cases()
    out = Any[]
    for W in _mf_spaces()
        D = dim(W)
        κ = Rₕ(W, x -> 1 + sum(abs2, x))
        push!(out, ("$(D)D diffusion", form(W, W, (u, v) -> innerₕ(u, v) + inner₊(κ * ∇ₕ(u), ∇ₕ(v))), :boundary))
        push!(out, ("$(D)D jump-avg", form(W, W, (u, v) -> innerₕ(jumpₓ(u), M₊ₓ(v)) + innerₕ(D₋ₓ(u), v)), nothing))
        push!(out, (
            "$(D)D restricted", form(W, W, (u, v) -> innerₕ(u, v) + innerₕ(κ * u, restrict_to(:boundary, v))), nothing))
        push!(out, ("$(D)D pair", form(W, W, (u, v) -> innerₕ(D₋ₓ(u), v) + 2.0 * innerₕ(u, D₋ₓ(v))), (:west,)))
    end
    W = _mf_spaces()[2]
    V = W × W
    push!(out, ("composite",
        form(V, V, (u, v) -> innerₕ(u(1), v(1)) + inner₊(∇ₕ(u(2)), ∇ₕ(v(2))) + innerₕ(u(1), v(2))), :boundary))
    push!(out,
        ("composite pair",
            form(V, V, (u, v) -> innerₕ(D₋ᵧ(u(1)), v(2)) + 3.0 * innerₕ(u(2), D₋ᵧ(v(1))) + innerₕ(u(1), v(1))),
            nothing))
    W2 = gridspace(mesh(W))
    V2 = W × W2
    push!(out,
        ("two-leaf pair",
            form(V2, V2, (u, v) -> innerₕ(D₋ₓ(u(1)), v(2)) + innerₕ(u(2), D₋ₓ(v(1))) + innerₕ(u(2), v(2))),
            (:west, :boundary)))
    return out
end

_mf_op(a, dl) = dl === nothing ? matrix_free_operator(a) : matrix_free_operator(a; dirichlet = dl)
_mf_mat(a, dl) = dl === nothing ? assemble(a) : assemble(a; dirichlet = dl)

@testset "matrix-free operator (#326)" begin
    cases = _mf_cases()

    @testset "matrix-free: agrees with assemble" begin
        for (name, a, dl) in cases
            @testset "$name" begin
                A = _mf_mat(a, dl)
                op = _mf_op(a, dl)
                @test op isa MatrixFreeOperator{Float64}
                @test size(op) == size(A)
                @test eltype(op) == eltype(A)
                x = randn(size(A, 2))
                y0 = randn(size(A, 1))
                @test !iszero(A * x)
                @test _mf_agree(op * x, A * x)
                y = copy(y0)
                mul!(y, op, x)
                @test _mf_agree(y, A * x)
                y = copy(y0)
                mul!(y, op, x, 0.5, 2.0)
                @test _mf_agree(y, 0.5 * (A * x) + 2.0 * y0)
                # `β = 0` overwrites `y`: a `NaN` in it does not survive.
                y = fill(NaN, size(A, 1))
                mul!(y, op, x, -1.5, 0.0)
                @test _mf_agree(y, -1.5 * (A * x))

                uₕ = element(trial_space(a))
                parent(uₕ) .= x
                vₕ = op * uₕ
                @test vₕ isa VectorElement
                @test space(vₕ) === test_space(a)
                @test _mf_agree(parent(vₕ), A * x)
                wₕ = element(test_space(a))
                parent(wₕ) .= y0
                mul!(wₕ, op, uₕ, 0.5, 2.0)
                @test _mf_agree(parent(wₕ), 0.5 * (A * x) + 2.0 * y0)
                mul!(wₕ, op, uₕ)
                @test _mf_agree(parent(wₕ), A * x)
            end
        end
    end

    @testset "matrix-free: allocation-free mul!" begin
        for (name, a, dl) in cases
            @testset "$name" begin
                op = _mf_op(a, dl)
                x = randn(size(op, 2))
                y = randn(size(op, 1))
                @test _mf_alloc3(y, op, x) == 0
                @test _mf_alloc5(y, op, x) == 0
                # `op * x` allocates its result only, never a matrix read entry by entry.
                @test _mf_alloc_times(op, x) <= sizeof(y) + 256
                uₕ = element(trial_space(a))
                parent(uₕ) .= x
                wₕ = element(test_space(a))
                @test _mf_alloc3(wₕ, op, uₕ) == 0
                @test _mf_alloc5(wₕ, op, uₕ) == 0
            end
        end
    end

    @testset "entries, Dirichlet rows, live data" begin
        W = _mf_spaces()[1]
        κ = Rₕ(W, x -> 1 + x^2)
        c = Ref(2.0)
        a = form(W, W, (u, v) -> c * innerₕ(u, v) + inner₊(κ * ∇ₕ(u), ∇ₕ(v)))
        A = assemble(a; dirichlet = :boundary)
        op = matrix_free_operator(a; dirichlet = :boundary)
        @test [op[i, j] for i in axes(A, 1), j in axes(A, 2)] ≈ Matrix(A) atol = 1e-12
        # Dirichlet rows are identity rows: `(A * x)[i] = x[i]`, columns untouched.
        x = randn(size(A, 2))
        y = op * x
        @test y[1] == x[1] && y[end] == x[end]
        # A coefficient changed in place is read by the next product, as by `assemble!`.
        c[] = 5.0
        Rₕ!(κ, x -> 3 + x)
        @test _mf_agree(op * x, assemble(a; dirichlet = :boundary) * x)
        @test_throws DimensionMismatch mul!(zeros(3), op, x)
        # Only the matrix's rows are read from a `dirichlet` pair; the values are not.
        @test _mf_agree(matrix_free_operator(a; dirichlet = :boundary => 1.0) * x, op * x)
    end

    # More test rows than trial columns: a row in Γ_D past the last column has no diagonal,
    # so the assembled matrix leaves it zero and `x` must not be read there. `x` is a view
    # into a longer buffer, so an out-of-range read would pick up the `1e6` past its end.
    @testset "rectangular Dirichlet rows" begin
        Ωc = mesh(domain(interval(0.0, 1.0)), 9, false)
        Ωf = mesh(domain(interval(0.0, 1.0)), 13, false)
        Wc, Wf = gridspace(Ωc), gridspace(Ωf)
        V = Wc × Wc
        for (name, a) in (
            ("πₕ onto finer", form(Wc, Wf, (u, v) -> innerₕ(πₕ(u), v))),
            ("scalar to composite", form(Wc, V, (u, v) -> innerₕ(u, v(1)) + innerₕ(D₋ₓ(u), v(2))))
        )
            @testset "$name" begin
                A = assemble(a; dirichlet = :boundary)
                op = matrix_free_operator(a; dirichlet = :boundary)
                @test size(op, 1) > size(op, 2)
                buf = fill(1.0e6, size(A, 1))
                buf[1:size(A, 2)] .= randn(size(A, 2))
                x = view(buf, 1:size(A, 2))
                @test _mf_agree(op * x, A * x)
                y0 = randn(size(A, 1))
                y = copy(y0)
                mul!(y, op, x, 0.5, 2.0)
                @test _mf_agree(y, 0.5 * (A * x) + 2.0 * y0)
                @test [op[i, j] for i in axes(A, 1), j in axes(A, 2)] == Matrix(A)
            end
        end
    end

    # `dirichlet_components` limits Γ_D to the named leaves, as it does in `assemble`.
    @testset "dirichlet_components" begin
        W = _mf_spaces()[1]
        V = W × W
        a = form(V, V, (u, v) -> inner₊(∇ₕ(u(1)), ∇ₕ(v(1))) + innerₕ(u(2), v(2)) + innerₕ(u(1), v(2)))
        x = randn(ndofs(V))
        for comps in (1, 2, (1, 2), nothing)
            A = assemble(a; dirichlet = :boundary, dirichlet_components = comps)
            op = matrix_free_operator(a; dirichlet = :boundary, dirichlet_components = comps)
            @test _mf_agree(op * x, A * x)
        end
        @test !(matrix_free_operator(a; dirichlet = :boundary) * x ≈
                assemble(a; dirichlet = :boundary, dirichlet_components = 2) * x)
        @test_throws ArgumentError matrix_free_operator(
            a; dirichlet = :boundary, dirichlet_components = 3)
        Ws = _mf_spaces()[1]
        as = form(Ws, Ws, (u, v) -> innerₕ(u, v))
        @test_throws ArgumentError matrix_free_operator(
            as; dirichlet = :boundary, dirichlet_components = 2)
    end

    # Both `show` forms print one line, never the dense printer's one product per entry.
    @testset "show is a one-line summary" begin
        W = _mf_spaces()[1]
        op = matrix_free_operator(form(W, W, (u, v) -> innerₕ(u, v)); dirichlet = :boundary)
        s2 = repr(op)
        s3 = repr(MIME"text/plain"(), op)
        @test s2 == s3
        @test s2 == "17×17 MatrixFreeOperator{Float64} with Dirichlet rows on (:boundary,)"
        @test sprint(print, op) == s2
    end

    @testset "GpuPolicy refused (v4.4.0)" begin
        W = _mf_spaces()[1]
        a = form(W, W, (u, v) -> innerₕ(u, v))
        err = try
            matrix_free_operator(a; policy = Bramble.GpuKernel())
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("v4.4.0", sprint(showerror, err))
    end
end

end # module
