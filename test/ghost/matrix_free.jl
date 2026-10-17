module GhostMatrixFreeTests

# `matrix_free_operator` on a ghost-space form (#781): `mul!` applies `E_vᵀ A_pad E_u`
# without assembling it. Each product is checked against `assemble(a) * x` and against
# `E' * assemble(a.padded) * E` from the space's closure, on seeded non-uniform meshes in
# 1D to 3D, with Robin, conormal and Dirichlet relations, a corner-reading form, both
# Dirichlet routes, `α`/`β`, sentinel ghosts, warm allocations, and a CG solve against
# `pde_solve`.

using Test
using Bramble
import Bramble
using Bramble: D₋ₓ, D₊ᵧ, CpuThreaded, MatrixFreeOperator
using LinearAlgebra: mul!, norm
using LinearSolve: LinearProblem, KrylovJL_CG, solve
using Random: MersenneTwister
using SparseArrays: AbstractSparseMatrix, nzrange, rowvals

using ..TestUtils: WITH_3D_TESTS

const K = [2.0 1.0; 1.0 2.0]

# A seeded non-uniform mesh on [0, 1] × [0, 2] × [0, 3] (its first `D` axes).
function ghost_mesh(ns::NTuple{D, Int}; seed = 5) where {D}
    Bramble._seed_mesh1d_rng!(seed)
    Ωd = domain(foldl(×, ntuple(d -> interval(0.0, Float64(d)), D)))
    Ω = mesh(Ωd, D == 1 ? only(ns) : ns, D == 1 ? false : ntuple(_ -> false, D))
    Bramble._unseed_mesh1d_rng!()
    return Ω
end

relerr(x, y) = norm(x - y) / norm(y)
alloc(f::F) where {F} = (f(); f(); @allocated f())

laplacian(u, v) = innerₕ(-1.0 * Δₕ(u), v) + innerₕ(u, v)
gradient(u, v) = inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(u, v)
mixed(u, v) = innerₕ(D₋ₓ(D₊ᵧ(u)), v) + innerₕ(u, v)      # reads the corner ghosts

robin1(u) = (:xmin => u + ∂ₙ(u) => 2.0, :xmax => ∂ₙ(u) => x -> x[1])
dirichlet1(u) = (:xmin => u => 1.0, :xmax => ∂ₙ(u) => 0.5)
robin2(u) = (:xmin => ∂ₙ(u) => 1.0, :xmax => u + ∂ₙ(u) => x -> x[2],
    :ymin => ∂ₙ(u) => 0.0, :ymax => 2.0 * u + ∂ₙ(u) => 1.0)
conormal2(u) = (:xmin => ∂ₙ(u; K) => 1.0, :xmax => u + ∂ₙ(u; K) => x -> x[2],
    :ymin => ∂ₙ(u; K) => 0.0, :ymax => 2.0 * u + ∂ₙ(u; K) => 1.0)
dirichlet2(u) = (:xmin => ∂ₙ(u) => 1.0, :xmax => 2.0 * u + ∂ₙ(u) => 0.0,
    :ymin => ∂ₙ(u) => 0.5, :ymax => u => x -> x[1])
robin3(u) = (:xmin => ∂ₙ(u) => 1.0, :xmax => u + ∂ₙ(u) => x -> x[3],
    :ymin => ∂ₙ(u) => 0.0, :ymax => ∂ₙ(u) => 0.0, :zmin => u => x -> x[1],
    :zmax => ∂ₙ(u) => 1.0)

# The product of `matrix_free_operator(a; dirichlet)` on `C` against the assembled matrix
# and, without `dirichlet`, against `E' * A_pad * E`; then `α`/`β`, a `NaN`-filled `y`,
# buffers holding a sentinel, a complex `x`, `getindex`, the threaded policy and the warm
# allocations (`allocs = false` skips these).
function check_operator(C, f; dirichlet = nothing, seed = 1, allocs = true)
    a = form(C, C, f)
    A = assemble(a; dirichlet)
    M = matrix_free_operator(a; dirichlet)
    rng = MersenneTwister(seed)
    x = rand(rng, size(A, 2))
    @test size(M) == size(A)
    @test relerr(M * x, A * x) <= 1e-12
    if dirichlet === nothing
        E, _ = Bramble.closure(C)
        @test relerr(M * x, E' * (assemble(a.padded) * (E * x))) <= 1e-12
    end
    y0 = rand(rng, size(A, 1))
    y = copy(y0)
    mul!(y, M, x, 2.0, -0.5)
    @test relerr(y, 2.0 * (A * x) - 0.5 * y0) <= 1e-12
    fill!(y, NaN)
    mul!(y, M, x, 3.0, 0.0)
    @test relerr(y, 3.0 * (A * x)) <= 1e-12
    fill!(M.xpad, 1e300)
    fill!(M.ypad, 1e300)
    @test relerr(M * x, A * x) <= 1e-12
    xc = x .+ im .* rand(rng, length(x))
    @test relerr(M * xc, A * xc) <= 1e-12
    i, j = size(A, 1) ÷ 2, size(A, 2) ÷ 3
    @test M[i, j] ≈ A[i, j] atol = 1e-12 * norm(A, Inf)
    Mt = matrix_free_operator(a; dirichlet, policy = CpuThreaded())
    @test relerr(Mt * x, A * x) <= 1e-12
    if allocs
        @test alloc(() -> mul!(y, M, x)) == 0
        @test alloc(() -> mul!(y, M, x, 2.0, 1.0)) == 0
    end
    return M, A
end

@testset "Ghost matrix_free_operator" begin
    @testset "Matrix-free, not assembled" begin
        C = constrained(gridspace(ghost_mesh((9, 11)); ghosts = :boundary), robin2)
        M, A = check_operator(C, laplacian)
        @test M.padded isa MatrixFreeOperator
        @test !any(fl -> getfield(M, fl) isa AbstractSparseMatrix, fieldnames(typeof(M)))
    end

    @testset "1D Robin and Dirichlet relations" begin
        for (rel, ghosts) in ((robin1, :boundary), (dirichlet1, (:xmax,)))
            C = constrained(gridspace(ghost_mesh((15,)); ghosts), rel)
            check_operator(C, laplacian)
            check_operator(C, gradient)
        end
    end

    @testset "2D $(nameof(rel)), $(nameof(f))" for rel in (robin2, conormal2),
        f in (laplacian, gradient, mixed)

        C = constrained(gridspace(ghost_mesh((9, 11)); ghosts = :boundary), rel)
        check_operator(C, f)
    end

    # Route (b): the Dirichlet facet is a relation `u => g` and drops out of the unknowns.
    @testset "Dirichlet as u => g, $(nameof(f))" for f in (laplacian, mixed)
        C = constrained(gridspace(ghost_mesh((9, 11)); ghosts = (:xmin, :xmax, :ymin)),
            dirichlet2)
        check_operator(C, f)
    end

    # Route (a): `dirichlet` sets identity rows on a space whose unknowns are the physical
    # nodes, and refuses one that keeps ghosts as unknowns, as `assemble` does.
    @testset "Dirichlet as identity rows" begin
        C = constrained(gridspace(ghost_mesh((9, 11)); ghosts = :boundary), robin2)
        M, A = check_operator(C, laplacian; dirichlet = (:ymin, :xmax))
        a = form(C, C, laplacian)
        @test any(!iszero, M * ones(size(M, 2)) - matrix_free_operator(a) * ones(size(M, 2)))
        @test_throws ArgumentError matrix_free_operator(a; dirichlet = :ymin,
            dirichlet_components = 2)
        Wg = gridspace(ghost_mesh((9, 11)); ghosts = :boundary)
        @test_throws ArgumentError matrix_free_operator(form(Wg, Wg, laplacian);
            dirichlet = :ymin)
    end

    # Ghosts kept as unknowns hold a sentinel: the rows that read no ghost column of the
    # assembled matrix stay finite and equal its rows.
    @testset "Sentinel ghosts" begin
        Wg = gridspace(ghost_mesh((9, 11)); ghosts = :boundary)
        G = mesh(Wg)
        a = form(Wg, Wg, mixed)
        A = assemble(a)
        M = matrix_free_operator(a)
        E, _ = Bramble.closure(Wg)
        plin = LinearIndices(npoints(G, Tuple))
        phys = Set(plin[J] for J in Bramble.physical_indices(G))
        ghost = [!(only(rowvals(E)[nzrange(E, c)]) in phys) for c in axes(E, 2)]
        @test count(ghost) > 0
        x = rand(MersenneTwister(2), size(A, 2))
        x[ghost] .= 1e300
        rows = vec(sum(abs.(A[:, ghost]); dims = 2)) .== 0
        @test count(rows) > 0
        y = M * x
        @test all(isfinite, y[rows])
        xs = copy(x)
        xs[ghost] .= 0
        @test relerr(y[rows], (A * xs)[rows]) <= 1e-12
    end

    # The padded 3D walk allocates: inference loses the summands' tuple type in
    # `_concat_summand_stencils` on the 3D padded space (a runtime dispatch per point), in
    # the shared stencil evaluation rather than in this operator.
    # One operator applies one product at a time: a product entered while another holds
    # the padded buffers throws, and tasks racing on one operator never get a wrong
    # product, each either the right one or that `ArgumentError`.
    @testset "One product at a time" begin
        C = constrained(gridspace(ghost_mesh((9, 11)); ghosts = :boundary), robin2)
        a = form(C, C, gradient)
        A = assemble(a)
        M = matrix_free_operator(a)
        x = rand(MersenneTwister(4), size(A, 2))
        y = similar(x)
        M.busy[] = true
        @test_throws ArgumentError mul!(y, M, x)
        M.busy[] = false
        @test relerr(M * x, A * x) <= 1e-12          # released after the refusal
        @test relerr(M * x, A * x) <= 1e-12          # and after a product
        @test alloc(() -> mul!(y, M, x)) == 0
        if Threads.nthreads() > 1
            xs = [rand(MersenneTwister(k), size(A, 2)) for k in 1:8]
            wrong = Threads.Atomic{Int}(0)
            refused = Threads.Atomic{Int}(0)
            @sync for k in 1:8
                Threads.@spawn for _ in 1:200
                    try
                        relerr(M * xs[k], A * xs[k]) > 1e-12 && Threads.atomic_add!(wrong, 1)
                    catch e
                        e isa ArgumentError || rethrow()
                        Threads.atomic_add!(refused, 1)
                    end
                end
            end
            @test wrong[] == 0
            @test !M.busy[]
        end
    end

    WITH_3D_TESTS && @testset "3D Robin and Dirichlet relations" begin
        C = constrained(
            gridspace(ghost_mesh((5, 6, 4)); ghosts = (:xmin, :xmax, :ymin, :ymax, :zmax)),
            robin3)
        check_operator(C, laplacian; allocs = false)
        check_operator(C, gradient; allocs = false)
    end

    # The operator in a Krylov solve: CG on the symmetric positive definite reduced
    # operator gives `pde_solve`'s solution of the assembled system.
    @testset "CG solve against pde_solve" begin
        for rel in (robin2, dirichlet2)
            ghosts = rel === robin2 ? :boundary : (:xmin, :xmax, :ymin)
            C = constrained(gridspace(ghost_mesh((13, 15)); ghosts), rel)
            a = form(C, C, gradient)
            l = form(C, v -> innerₕ(x -> 1.0 + x[1] * x[2], v))
            A, F = assemble(a, l)
            u = pde_solve(A, F)
            sol = solve(LinearProblem(matrix_free_operator(a), F), KrylovJL_CG();
                reltol = 1e-13, abstol = 0.0, maxiters = 10_000)
            @test relerr(sol.u, u) <= 1e-9
        end
    end
end

end
