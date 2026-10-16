module GaugeSolveTests

# `pde_solve(A, F; gauge = Z)` against references that share no code with the bordered
# solve. With `F = A x` and a kernel of `A` spanned by the columns of `K`, every solution is
# `x + K c`, and the border `C` fixes `c` by `Cᵀ(x + K c) = 0`: the reference is
# `x − K (CᵀK)⁻¹ Cᵀ x`, a dense k × k solve. The physical part is unique when the kernel is
# ghost-only, so it is checked against `x` and against `pinv` too. The forms that produce a
# ghost-only kernel are ill conditioned (κ of the nonzero part near 1e12), so the tolerances
# are 1e-7 against `x` and 1e-6 against `pinv` (departure 4); a wrong border gives O(1) or a
# singular solve. Random and graded meshes only.

using Test
using Bramble
import Bramble
using LinearAlgebra: norm, pinv, issymmetric, I
using SparseArrays: SparseMatrixCSC, sparse, spzeros
using Random: Random
using ..TestUtils: WITH_3D_TESTS, WITH_SLOW_TESTS

const Δform = (u, v) -> innerₕ(Δₕ(u), Δₕ(v)) + innerₕ(u, v)

function random_mesh(seed, ranges, n)
    Bramble._seed_mesh1d_rng!(seed)
    try
        return mesh(domain(foldl(×, ranges)), n, ntuple(_ -> false, length(n)))
    finally
        Bramble._unseed_mesh1d_rng!()
    end
end

# The points `x^1.5` on [0, 1] and their double on [0, 2].
function graded_mesh_2d(n)
    Ω = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), n, (true, true))
    Bramble.change_points!(Ω, (collect(range(0.0, 1.0; length = n[1])) .^ 1.5,
        2 .* collect(range(0.0, 1.0; length = n[2])) .^ 1.5))
    return Ω
end

relerr(a, b) = norm(a - b, Inf) / norm(b, Inf)

# The solution of `A u = A x` in `x + span K` with `Cᵀu = 0`.
function projected_reference(x, K, C)
    Kd, Cd = Matrix(K), Matrix(C)
    return x - Kd * ((transpose(Cd) * Kd) \ (transpose(Cd) * x))
end

# One gauged solve on `Wₕ` checked against the manufactured `x`, `pinv` and the projected
# reference; returns the two physical-part errors.
function check_gauged(A, Wₕ, np; solver = :default, tol_x = 1e-7, tol_pinv = 1e-6)
    Z = Bramble.ghost_kernel(A, Wₕ)
    n = size(A, 1)
    x = rand(eltype(A), n)
    F = A * x
    u = pde_solve(A, F; gauge = Z, solver = solver)
    @test u isa Vector{eltype(A)} && length(u) == n
    @test norm(transpose(Z) * u, Inf) <= 1e-10 * norm(u, Inf)
    e1 = relerr(u[1:np], x[1:np])
    @test e1 <= tol_x
    ref = pinv(Matrix(A); rtol = 1e-14) * F
    e2 = relerr(u[1:np], ref[1:np])
    @test e2 <= tol_pinv
    # the ghost values too: the minimum of nothing but `Zᵀu = 0`
    @test relerr(u, projected_reference(x, Z, Z)) <= tol_pinv
    @test pde_solve(A, F; gauge = Z, solver = solver) == u
    return e1, e2
end

@testset "Gauged solve: random and graded 2D" begin
    Random.seed!(777)
    for Ω in (random_mesh(11, (interval(0.0, 1.0), interval(0.0, 2.0)), (6, 7)),
        graded_mesh_2d((9, 11)))
        W = gridspace(Ω; ghosts = :boundary)
        A = assemble(form(W, W, Δform))
        @test size(Bramble.ghost_kernel(A, W), 2) == 4
        A0 = copy(A)
        check_gauged(A, W, npoints(Ω))
        @test A == A0
    end
end

@testset "Gauged solve: the border is read" begin
    Random.seed!(778)
    Ω = random_mesh(11, (interval(0.0, 1.0), interval(0.0, 2.0)), (6, 7))
    W = gridspace(Ω; ghosts = :boundary)
    A = assemble(form(W, W, Δform))
    Z = Bramble.ghost_kernel(A, W)
    np = npoints(Ω)
    x = rand(size(A, 1))
    F = A * x
    u = pde_solve(A, F; gauge = Z)
    # a dense `Z` gives the same solution
    @test pde_solve(A, F; gauge = Matrix(Z)) ≈ u rtol = 1e-8
    # another border outside the range of `A`, its two entries per column weighted 1 and 2,
    # moves only the ghosts
    Z₂ = copy(Z)
    Z₂.nzval .*= [isodd(q) ? 1.0 : 2.0 for q in eachindex(Z₂.nzval)]
    u₂ = pde_solve(A, F; gauge = Z₂)
    @test norm(transpose(Z₂) * u₂, Inf) <= 1e-10 * norm(u₂, Inf)
    @test relerr(u₂[1:np], x[1:np]) <= 1e-7
    @test relerr(u₂, projected_reference(x, Z, Z₂)) <= 1e-6
    @test norm(u₂ - u, Inf) > 1e-3 * norm(u, Inf)
end

# A Neumann space keeping the ghosts of the xmin and ymax facets. `inner₊(∇ₕ(u), ∇ₕ(v))`
# puts the constants in the kernel, and the corner the two kept facets share puts one
# ghost-only vector there. The border is `[w Z]`, the kernel `[1 Z]`.
@testset "Gauged solve: mean_zero, pure Neumann" begin
    Random.seed!(779)
    Ω = random_mesh(17, (interval(0.0, 1.0), interval(0.0, 2.0)), (7, 6))
    W = gridspace(Ω; ghosts = :boundary)
    C = constrained(W, u -> (:xmax => ∂ₙ(u) => 0.0, :ymin => ∂ₙ(u) => 0.0))
    A = assemble(form(C, C, (u, v) -> innerₕ(Δₕ(u), Δₕ(v)) + inner₊(∇ₕ(u), ∇ₕ(v))))
    w = assemble(form(C, v -> innerₕ(x -> 1.0, v)))
    n, np = size(A, 1), npoints(Ω)
    @test norm(A * ones(n), Inf) <= 1e-10 * norm(A, Inf)
    Z = Bramble.ghost_kernel(A, C)
    @test size(Z, 2) == 1
    x = rand(n)
    F = A * x
    u = pde_solve(A, F; mean_zero = w, gauge = Z)
    @test abs(sum(w .* u)) <= 1e-10 * norm(u, Inf) * norm(w, 1)
    @test norm(transpose(Z) * u, Inf) <= 1e-10 * norm(u, Inf)
    K = [ones(n) Matrix(Z)]
    @test relerr(u, projected_reference(x, K, [w Matrix(Z)])) <= 1e-6
    # the physical part is `x` shifted to a zero mean, and `pinv`'s shifted likewise
    shift(v) = v[1:np] .- sum(w[1:np] .* v[1:np]) / sum(w[1:np])
    @test iszero(w[(np + 1):end])
    @test relerr(u[1:np], shift(x)) <= 1e-7
    @test relerr(u[1:np], shift(pinv(Matrix(A); rtol = 1e-14) * F)) <= 1e-6
    # the order of the keywords does not matter, nor a dense or sparse form of the border
    @test pde_solve(A, F; gauge = Z, mean_zero = w) == u
    @test pde_solve(A, F; mean_zero = sparse(w), gauge = Matrix(Z)) ≈ u rtol = 1e-6
end

@testset "Gauged solve: a gauge with no column" begin
    Random.seed!(780)
    Ω = random_mesh(11, (interval(0.0, 1.0), interval(0.0, 2.0)), (6, 7))
    W = gridspace(Ω; ghosts = (:xmin,))
    A = assemble(form(W, W, Δform))
    Z = Bramble.ghost_kernel(A, W)
    @test size(Z, 2) == 0
    x = rand(size(A, 1))
    @test relerr(pde_solve(A, A * x; gauge = Z), x) <= 1e-7
end

# A `Z` spanning half the kernel is not detected: the physical part is still the quotient
# solution, and only the two gauged components are fixed. Whether the factorisation
# notices is a matter of rounding and backend (with no column at all, UMFPACK throws
# `SingularException` here, while Accelerate solves).
@testset "Gauged solve: an under-spanning gauge" begin
    Random.seed!(786)
    Ω = random_mesh(11, (interval(0.0, 1.0), interval(0.0, 2.0)), (6, 7))
    W = gridspace(Ω; ghosts = :boundary)
    A = assemble(form(W, W, Δform))
    Z = Bramble.ghost_kernel(A, W)
    np = npoints(Ω)
    x = rand(size(A, 1))
    Zp = Z[:, 1:2]
    u = pde_solve(A, A * x; gauge = Zp)
    @test all(isfinite, u)
    @test relerr(u[1:np], x[1:np]) <= 1e-7
    @test norm(transpose(Zp) * u, Inf) <= 1e-8 * norm(u, Inf)
end

@testset "Gauged solve: complex A" begin
    Random.seed!(781)
    Ω = random_mesh(11, (interval(0.0, 1.0), interval(0.0, 2.0)), (6, 7))
    W = gridspace(Ω; ghosts = :boundary)
    Ac = assemble(form(W, W, (u, v) -> (1.0 + 2.0im) * innerₕ(Δₕ(u), Δₕ(v)) + innerₕ(u, v)))
    @test eltype(Ac) <: Complex
    check_gauged(Ac, W, npoints(Ω))
    # a complex border on a real `A`: the border is transposed, not conjugated
    A = assemble(form(W, W, Δform))
    Zc = Bramble.ghost_kernel(A, W) * (1.0 + 1.0im)
    x = rand(size(A, 1))
    u = pde_solve(A, A * x; gauge = Zc)
    @test eltype(u) == ComplexF64
    @test norm(transpose(Zc) * u, Inf) <= 1e-10 * norm(u, Inf)
    @test relerr(u, projected_reference(x, Zc, Zc)) <= 1e-6
end

# A form that never reads the ghosts: the kernel is every ghost and the rest is better
# conditioned (κ near 1e3 on this graded mesh), so a Float32 solve keeps four digits.
@testset "Gauged solve: Float32 and inference" begin
    Random.seed!(782)
    Ω = graded_mesh_2d((5, 6))
    W = gridspace(Ω; ghosts = :boundary)
    A = assemble(form(W, W, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(u, v)))
    Z = Bramble.ghost_kernel(A, W)
    @test size(Z, 2) == Bramble.nghosts(mesh(W))
    x = rand(size(A, 1))
    F = A * x
    u = pde_solve(A, F; gauge = Z)
    @test relerr(u, projected_reference(x, Z, Z)) <= 1e-10
    u32 = pde_solve(Float32.(A), Float32.(F); gauge = Float32.(Z))
    @test u32 isa Vector{Float32}
    @test relerr(u32, u) <= 1e-4
    @test (@inferred pde_solve(A, F; gauge = Z)) isa Vector{Float64}
    @test (@inferred pde_solve(A, F; gauge = Z, mean_zero = ones(size(A, 1)))) isa
          Vector{Float64}
    @test pde_solve(spzeros(0, 0), Float64[]; gauge = zeros(0, 2)) == Float64[]
end

@testset "Gauged solve: refusals" begin
    Ω = random_mesh(11, (interval(0.0, 1.0), interval(0.0, 2.0)), (6, 7))
    W = gridspace(Ω; ghosts = :boundary)
    A = assemble(form(W, W, Δform))
    Z = Bramble.ghost_kernel(A, W)
    F = A * ones(size(A, 1))
    w = ones(size(A, 1))
    @test_throws DimensionMismatch pde_solve(A, F; gauge = Z[1:(end - 1), :])
    @test_throws DimensionMismatch pde_solve(A, F; gauge = [Z; Z])
    @test_throws DimensionMismatch pde_solve(A, F; gauge = Z, mean_zero = w[1:(end - 1)])
    @test_throws DimensionMismatch pde_solve(A, F[1:(end - 1)]; gauge = Z)
    @test_throws DimensionMismatch pde_solve(A[:, 1:(end - 1)], F; gauge = Z[1:(end - 1), :])
    @test_throws ArgumentError pde_solve(A, F; gauge = Z, solver = :spqr)
    @test_throws ArgumentError pde_solve(A, F; gauge = Z, solver = :sparspak)
    @test_throws ArgumentError pde_solve(A, F; gauge = Z, mean_zero = w, solver = :sparspak)
    @test_throws ArgumentError pde_solve(A, F; gauge = Z, solver = :bogus)
    @test_throws ArgumentError pde_solve(A, F; gauge = Z, bogus = 1)
    # the border is indefinite: a symmetric `A` keeps the symmetric (pivoting) route only
    # for a real border
    @test issymmetric(A)
    @test Bramble._bordered_sym(A, Z) === :symmetric
    @test Bramble._bordered_sym(A, Z * im) === :unsymmetric
end

# The backends loaded in this session, then (slow group) a child on the test project that
# loads every backend the environment has. `:sparspak` refuses a gauge (see refusals).
@testset "Gauged solve: each solver backend" begin
    Random.seed!(783)
    Ω = random_mesh(11, (interval(0.0, 1.0), interval(0.0, 2.0)), (6, 7))
    W = gridspace(Ω; ghosts = :boundary)
    A = assemble(form(W, W, Δform))
    loaded = filter(((s, ext),) -> Base.get_extension(Bramble, ext) !== nothing,
        [(:suitesparse, :BrambleSuiteSparseExt), (:mumps, :BrambleMUMPSExt),
            (:accelerate, :BrambleAppleAccelerateExt)])
    for solver in (:default, first.(loaded)...)
        check_gauged(A, W, npoints(Ω); solver = solver)
    end
    if WITH_SLOW_TESTS
        code = raw"""
        using Bramble, LinearAlgebra, SparseArrays, Random, SuiteSparse, MUMPS
        Sys.isapple() && @eval using AppleAccelerate
        Random.seed!(784)
        Bramble._seed_mesh1d_rng!(11)
        Ω = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (6, 7), (false, false))
        Bramble._unseed_mesh1d_rng!()
        W = gridspace(Ω; ghosts = :boundary)
        A = assemble(form(W, W, (u, v) -> innerₕ(Δₕ(u), Δₕ(v)) + innerₕ(u, v)))
        Z = Bramble.ghost_kernel(A, W)
        np = npoints(Ω)
        x = rand(size(A, 1))
        F = A * x
        for s in (:default, :suitesparse, :mumps, (Sys.isapple() ? (:accelerate,) : ())...)
            u = pde_solve(A, F; gauge = Z, solver = s)
            g = norm(transpose(Z) * u, Inf) / norm(u, Inf)
            e = norm(u[1:np] - x[1:np], Inf) / norm(x[1:np], Inf)
            println(s, " ", g, " ", e)
        end
        """
        project = dirname(something(Base.active_project()))
        cmd = `$(Base.julia_cmd()) --project=$project --startup-file=no --threads=1 -e $code`
        lines = split(read(cmd, String), '\n'; keepempty = false)
        @test length(lines) == (Sys.isapple() ? 4 : 3)
        for line in lines
            s, g, e = split(line)
            @test parse(Float64, g) <= 1e-10
            @test parse(Float64, e) <= 1e-7
        end
    end
end

# 3D runs nightly: 52 kernel vectors on the edges and corners of a 4×5×6 random mesh.
WITH_3D_TESTS && @testset "Gauged solve: 3D" begin
    Random.seed!(785)
    Ω = random_mesh(7, (interval(0.0, 1.0), interval(0.0, 2.0), interval(0.0, 1.5)), (4, 5, 6))
    W = gridspace(Ω; ghosts = :boundary)
    A = assemble(form(W, W, Δform))
    @test size(Bramble.ghost_kernel(A, W), 2) == 4 * sum((4, 5, 6) .- 2) + 16
    check_gauged(A, W, npoints(Ω))
end

end
