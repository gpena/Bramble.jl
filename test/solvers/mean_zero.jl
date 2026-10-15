module SolversMeanZeroTests

using Test
using Bramble
using LinearAlgebra
using SparseArrays

# A pure-Neumann system: `A` annihilates constants, `w` holds the quadrature weights of the
# discrete average. The bordered solution is fixed by its defining equations, `wᵀu = 0` and
# `A u = F − λ w`, where `λ = 1ᵀF / 1ᵀw` because `1ᵀA = 0`.
function _neumann_system(Ω, f)
    W = gridspace(Ω)
    A = assemble(form(W, W, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v))))
    F = assemble(form(W, v -> innerₕ(f, v)))
    w = assemble(form(W, v -> innerₕ(x -> 1.0, v)))
    return A, F, w
end

function _is_bordered_solution(A, F, w, u)
    λ = sum(F) / sum(w)
    return abs(dot(w, u)) <= 1e-10 * norm(u) && norm(A * u - (F - λ * w)) <= 1e-9 * norm(F)
end

@testset "Mean-zero bordered solve (#205)" begin
    Ω = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 11), (false, false))
    A, F, w = _neumann_system(Ω, x -> cos(π * x[1]) * cos(π * x[2]))
    A0 = copy(A)
    u = pde_solve(A, F; mean_zero = w)
    @test _is_bordered_solution(A, F, w, u)
    @test norm(u) > 0.1 * norm(F) / norm(A, Inf)
    @test A == A0
    @test u isa Vector{Float64} && length(u) == size(A, 1)
end

@testset "Mean-zero sparse and dense w, F" begin
    Ω = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 8), (false, false))
    A, F, w = _neumann_system(Ω, x -> cos(π * x[1]))
    u = pde_solve(A, F; mean_zero = w)
    @test _is_bordered_solution(A, F, w, u)
    @test pde_solve(A, F; mean_zero = sparsevec(w)) ≈ u rtol = 1e-12
    @test pde_solve(A, sparsevec(F); mean_zero = w) ≈ u rtol = 1e-12
    # A different weight gives a different (shifted) solution: `w` is really read.
    w₂ = collect(range(1.0, 2.0; length = length(w)))
    u₂ = pde_solve(A, F; mean_zero = w₂)
    @test abs(dot(w₂, u₂)) <= 1e-10 * norm(u₂)
    @test norm(u₂ - u) > 1e-6 * norm(u)
end

@testset "Mean-zero 1D non-uniform mesh" begin
    Ω = mesh(domain(interval(-1.0, 2.0)), 17, false)
    A, F, w = _neumann_system(Ω, x -> cos(π * (x - 2.0) / 3.0))
    u = pde_solve(A, F; mean_zero = w)
    @test _is_bordered_solution(A, F, w, u)
    # The constant is fixed: shifting u by any constant breaks wᵀu = 0.
    @test abs(dot(w, u .+ 1.0)) > 0.5 * sum(w)
end

@testset "Neumann warning names mean_zero" begin
    Ω = mesh(domain(interval(0.0, 1.0)), 9, false)
    A, F, _ = _neumann_system(Ω, x -> cos(π * x))
    @test_logs (:warn, r"mean_zero") match_mode = :any (try
        pde_solve(A, F)
    catch
    end)
    @test_logs min_level = Base.CoreLogging.Warn pde_solve(A + I, F)
    @test_logs min_level = Base.CoreLogging.Warn pde_solve(A, F; mean_zero = ones(size(A, 1)))
end

@testset "Mean-zero refusals" begin
    Ω = mesh(domain(interval(0.0, 1.0)), 9, false)
    A, F, w = _neumann_system(Ω, x -> cos(π * x))
    @test_throws DimensionMismatch pde_solve(A, F; mean_zero = w[1:(end - 1)])
    @test_throws DimensionMismatch pde_solve(A, F[1:(end - 1)]; mean_zero = w)
    @test_throws DimensionMismatch pde_solve(A[:, 1:(end - 1)], F; mean_zero = w)
    @test_throws ArgumentError pde_solve(A, F; mean_zero = w, solver = :spqr)
    @test_throws ArgumentError pde_solve(A, F; mean_zero = w, solver = :bogus)
    @test_throws ArgumentError pde_solve(A, F; mean_zero = w, bogus = 1)
end

# A ghost-point Neumann matrix annihilates constants but is not symmetric: its boundary rows
# carry the doubled neighbour `-2/h²`. The bordered system is then not symmetric either.
function _ghost_neumann(N)
    h = 1 / N
    A = spdiagm(-1 => fill(-1.0, N), 0 => fill(2.0, N + 1), 1 => fill(-1.0, N)) / h^2
    A[1, 2] = -2 / h^2
    A[N + 1, N] = -2 / h^2
    w = fill(h, N + 1)
    w[1] = w[end] = h / 2
    F = cos.(π .* range(0.0, 1.0; length = N + 1))
    return A, F, w
end

@testset "Mean-zero nonsymmetric A" begin
    A, F, w = _ghost_neumann(8)
    @test !issymmetric(A) && norm(A * ones(size(A, 1)), Inf) == 0
    ref = (Matrix([A w; w' 0]) \ [F; 0])[1:(end - 1)]
    u = pde_solve(A, F; mean_zero = w)
    @test u ≈ ref rtol = 1e-10
    @test abs(dot(w, u)) <= 1e-10 * norm(u)
    # A symmetric route (Accelerate, MUMPS) would read one triangle of the border.
    @test Bramble._bordered_sym(A, w) === :unsymmetric
    @test Bramble._bordered_sym(A + sparse(A'), w) === :symmetric
    @test Bramble._bordered_sym(A + sparse(A'), complex.(w)) === :unsymmetric
end

@testset "Mean-zero element type and inference" begin
    A, F, w = _ghost_neumann(8)
    u32 = pde_solve(Float32.(A), Float32.(F); mean_zero = Float32.(w))
    @test u32 isa Vector{Float32}
    @test u32 ≈ pde_solve(A, F; mean_zero = w) rtol = 1e-4
    @test (@inferred pde_solve(A, F; mean_zero = w)) isa Vector{Float64}
    @test pde_solve(spzeros(0, 0), Float64[]; mean_zero = Float64[]) == Float64[]
    @test pde_solve(spzeros(0, 0), Float64[]; mean_zero = Float64[]) isa Vector{Float64}
    Ai = SparseMatrixCSC{Int, Int}([1 -1 0; -1 2 -1; 0 -1 1])
    ui = pde_solve(Ai, [1, 0, -1]; mean_zero = [1, 1, 1])
    @test ui isa Vector{Float64}
    @test ui ≈ [1.0, 0.0, -1.0] atol = 1e-12
end

end # module SolversMeanZeroTests
