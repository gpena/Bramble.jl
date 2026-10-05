# Nonlinear Poisson problem, -(α(u) u′)′ = g on a random non-uniform 1D mesh, solved twice:
# by Picard iteration (refill one matrix in place) and by Newton's method with a sparse
# automatic-differentiation Jacobian. Both must agree and match the exact solution.
using Bramble
using Bramble: allocate_system_matrix
using Random
using ForwardDiff, DifferentiationInterface
import SparseConnectivityTracer, SparseMatrixColorings

sol(x) = exp(x[1])
α(u) = 3 + 1 / (1 + u^2)
dαdu(u) = -2u / (1 + u^2)^2
rhs(x) = -dαdu(sol(x)) * sol(x)^2 - α(sol(x)) * sol(x)

Ω = domain(interval(0.0, 1.0))
Random.seed!(20260903)
Ωₕ = mesh(Ω, 40, false)
Wₕ = gridspace(Ωₕ)

gₕ = element(Wₕ)
avgₕ!(gₕ, rhs)
F = assemble(form(Wₕ, v -> innerₕ(gₕ, v)); dirichlet = dirichlet_constraints(Ω, :boundary => sol))

# Picard: the coefficient is a grid function the form captures, so the form is built once
# and only αvals changes between assemblies.
uₙ = element(Wₕ, 0.0)
αvals = element(Wₕ)
αvals .= α.(Mₕ(uₙ))                                  # α at cell midpoints (backward average)
a = form(Wₕ, Wₕ, (U, V) -> inner₊(αvals * ∇ₕ(U), ∇ₕ(V)))
A = allocate_system_matrix(a)
steps = Float64[]
for it in 1:200
    assemble!(A, a; dirichlet = :boundary)           # refills A, no new matrix
    unew = A \ F
    push!(steps, maximum(abs, unew .- parent(uₙ)))
    uₙ .= unew
    αvals .= α.(Mₕ(uₙ))
    steps[end] < 1e-12 && break
end

# Newton: the residual takes any element type, so ForwardDiff's dual numbers pass through.
const sparse_ad = AutoSparse(AutoForwardDiff();
    sparsity_detector = SparseConnectivityTracer.TracerSparsityDetector(),
    coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm())

function residual(u_vec::AbstractVector{T}) where {T}
    uₕ = element(Wₕ, T)                              # element type from the data, not Float64
    uₕ .= u_vec
    αₕ = α.(Mₕ(uₕ))
    A = assemble(form(Wₕ, Wₕ, (U, V) -> inner₊(αₕ * ∇ₕ(U), ∇ₕ(V))); dirichlet = :boundary)
    return A * u_vec .- F
end

u = zeros(ndofs(Wₕ))
prep = prepare_jacobian(residual, sparse_ad, u)
J = DifferentiationInterface.jacobian(residual, prep, sparse_ad, u)
residuals = Float64[]
for it in 1:20
    r = residual(u)
    push!(residuals, sqrt(sum(abs2, r)))
    residuals[end] < 1e-10 && break
    DifferentiationInterface.jacobian!(residual, J, prep, sparse_ad, u)
    u .-= J \ r
end

uₕ_newton = element(Wₕ)
uₕ_newton .= u
@assert steps[end] < 1e-12
@assert length(residuals) < 8 && residuals[end] < 1e-10
@assert maximum(abs, parent(uₕ_newton) .- parent(uₙ)) < 1e-8
@assert 1.0e-6 < norm₁ₕ(uₙ .- Rₕ(Wₕ, sol)) < 1.0e-2
