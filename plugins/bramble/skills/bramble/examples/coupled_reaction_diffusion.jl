# Two coupled nonlinear reaction-diffusion equations on a random non-uniform 2D mesh:
#
#   -Δu + u + u v = f₁,   -Δv + v - u v = f₂,   u = v = 0 on the boundary.
#
# Both unknowns live in one composite space Vₕ = Wₕ^Val(2); `p(1)`, `q(2)` pick components
# inside the form. Newton's method with a sparse ForwardDiff Jacobian solves the system.
using Bramble
using Random
using ForwardDiff, DifferentiationInterface
import SparseConnectivityTracer, SparseMatrixColorings

u_ex(x) = sin(π * x[1]) * sin(π * x[2])
v_ex(x) = sin(2π * x[1]) * sin(2π * x[2])
f1(x) = 2π^2 * u_ex(x) + u_ex(x) + u_ex(x) * v_ex(x)
f2(x) = 8π^2 * v_ex(x) + v_ex(x) - u_ex(x) * v_ex(x)

Random.seed!(20260903)
Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
Ωₕ = mesh(Ω, (24, 24), (false, false))
Wₕ = gridspace(Ωₕ)
Vₕ = Wₕ^Val(2)
bcs = dirichlet_constraints(Ω, :boundary => x -> 0.0)

f1ₕ, f2ₕ = element(Wₕ), element(Wₕ)
avgₕ!(f1ₕ, f1)
avgₕ!(f2ₕ, f2)
F = assemble(form(Vₕ, q -> innerₕ(f1ₕ, q(1)) + innerₕ(f2ₕ, q(2))); dirichlet = bcs)

function residual(w::AbstractVector{T}) where {T}
    wₕ = element(Vₕ, T)
    wₕ .= w
    u_c, v_c = components(wₕ)                        # views, no copy
    a = form(Vₕ, Vₕ,
        (p, q) -> inner₊(∇ₕ(p(1)), ∇ₕ(q(1))) + innerₕ(p(1), q(1)) + innerₕ(v_c * p(1), q(1)) +
                  inner₊(∇ₕ(p(2)), ∇ₕ(q(2))) + innerₕ(p(2), q(2)) - innerₕ(u_c * p(2), q(2)))
    return assemble(a; dirichlet = :boundary) * w .- F
end

const sparse_ad = AutoSparse(AutoForwardDiff();
    sparsity_detector = SparseConnectivityTracer.TracerSparsityDetector(),
    coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm())
w = zeros(ndofs(Vₕ))
prep = prepare_jacobian(residual, sparse_ad, w)
J = DifferentiationInterface.jacobian(residual, prep, sparse_ad, w)
residuals = Float64[]
for it in 1:20
    r = residual(w)
    push!(residuals, sqrt(sum(abs2, r)))
    residuals[end] < 1e-10 && break
    DifferentiationInterface.jacobian!(residual, J, prep, sparse_ad, w)
    w .-= J \ r
end

wₕ = element(Vₕ)
wₕ .= w
uₕ, vₕ = components(wₕ)
@assert length(residuals) < 8 && residuals[end] < 1e-10
@assert 1.0e-6 < norm₁ₕ(uₕ .- Rₕ(Wₕ, u_ex)) < 1.0e-1
@assert 1.0e-6 < norm₁ₕ(vₕ .- Rₕ(Wₕ, v_ex)) < 5.0e-1     # v oscillates twice as fast
