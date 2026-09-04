# Coupled nonlinear reaction-diffusion system

Two species, coupled through a quadratic reaction term, solved with Newton's method the same
way the [nonlinear Poisson example](poisson_nonlinear.md) does — except now the Jacobian
differentiates through a *composite* space's assembly, not a scalar one. Every number and
every plot below was produced by the code shown.

## Problem

```math
\begin{aligned}
-\Delta u + u + uv &= f_1 \\
-\Delta v + v - uv &= f_2
\end{aligned}
\qquad \text{in } \Omega = (0,1)^2, \qquad u = v = 0 \text{ on } \partial\Omega,
```

a predator-prey-shaped coupling without the time derivative: `u` grows through the
interaction term, `v` is depleted by it. The manufactured solutions vanish on the boundary
already, so homogeneous Dirichlet data is all that is needed:

```@example coupled
using Bramble

u_ex(x) = sin(π * x[1]) * sin(π * x[2])
v_ex(x) = sin(2π * x[1]) * sin(2π * x[2])
f1(x) = 2π^2 * u_ex(x) + u_ex(x) + u_ex(x) * v_ex(x)
f2(x) = 8π^2 * v_ex(x) + v_ex(x) - u_ex(x) * v_ex(x)

Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
Ωₕ = mesh(Ω, (24, 24), (false, false))
Wₕ = gridspace(Ωₕ)
Vₕ = Wₕ^Val(2)

bcs = dirichlet_constraints(Bramble.set(Ω), :boundary => x -> 0.0)
f1ₕ = element(Wₕ)
avgₕ!(f1ₕ, f1)
f2ₕ = element(Wₕ)
avgₕ!(f2ₕ, f2)
l = form(Vₕ, q -> innerₕ(f1ₕ, q(1)) + innerₕ(f2ₕ, q(2)))
F = assemble(l; dirichlet_conditions = bcs, dirichlet_labels = :boundary)
nothing # hide
```

## Newton's method on a composite residual

`uv` is quadratic in the unknowns, so it cannot sit inside a matrix independent of
`w = (u, v)` the way the linear terms can — but it *can* sit inside a matrix that depends on
the current guess, the same trick [the nonlinear Poisson example uses for a single
species](poisson_nonlinear.md#Fixed-point-(Picard)-iteration), extended to a second one.
Writing the coupling as `v_current * u(1)` in `u`'s own equation and `-u_current * u(2)` in
`v`'s reproduces `uv` and `-uv` exactly once the trial function is evaluated at the current
`w` — which is all `A(w) w` needs to do. Nothing here works out `∂(uv)/∂u` and `∂(uv)/∂v` by
hand; `ForwardDiff` differentiates through *how* `A` itself depends on `w` automatically:

```@example coupled
function coupled_matrix(wₕ)
    u_c, v_c = components(wₕ)
    a = form(Vₕ, Vₕ, (p, q) ->
        inner₊(∇₋ₕ(p(1)), ∇₋ₕ(q(1))) + innerₕ(p(1), q(1)) + innerₕ(v_c * p(1), q(1)) +
        inner₊(∇₋ₕ(p(2)), ∇₋ₕ(q(2))) + innerₕ(p(2), q(2)) - innerₕ(u_c * p(2), q(2)))
    return assemble(a; dirichlet_labels = :boundary)
end
nothing # hide
```

Its Jacobian is sparse for the same reason the [nonlinear Poisson
example's](poisson_nonlinear.md#Newton's-method) is — the reaction term couples `u` and `v`
only pointwise, so it adds nothing to the diffusion stencil's own reach — so the same sparse
AD setup applies unchanged, just over twice as many unknowns:

```@example coupled
using ForwardDiff, DifferentiationInterface
import SparseConnectivityTracer, SparseMatrixColorings

const sparse_ad = AutoSparse(AutoForwardDiff();
    sparsity_detector = SparseConnectivityTracer.TracerSparsityDetector(),
    coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm())

function residual(w::AbstractVector{T}) where {T}
    wₕ = element(Vₕ, T)
    wₕ .= w
    A = coupled_matrix(wₕ)
    return A * w .- F
end

w = zeros(ndofs(Vₕ))
prep = prepare_jacobian(residual, sparse_ad, w)
J = DifferentiationInterface.jacobian(residual, prep, sparse_ad, w)  # once, for its sparse structure
newton_residuals = Float64[]
for it in 1:20
    r = residual(w)
    push!(newton_residuals, sqrt(sum(abs2, r)))
    newton_residuals[end] < 1e-10 && break
    DifferentiationInterface.jacobian!(residual, J, prep, sparse_ad, w)
    w .-= J \ r
end
length(newton_residuals), newton_residuals
```

Quadratic convergence, same as the single-species case — the composite space changes what
the Jacobian differentiates through, not how well Newton converges once it has a correct one.

```@example coupled
wₕ = element(Vₕ)
wₕ .= w
uₕ, vₕ = components(wₕ)
uexact, vexact = Rₕ(Wₕ, u_ex), Rₕ(Wₕ, v_ex)
norm₁ₕ(uₕ .- uexact), norm₁ₕ(vₕ .- vexact)
```

## Visualizing the solution

Each species is its own 2D scalar field — `components(wₕ)` gives a view directly onto it, no
new solve or copy needed:

```@example coupled
include(joinpath(@__DIR__, "..", "solution_plot.jl")) # hide
heatmap_plot(uₕ; title = "Coupled reaction-diffusion, u") # hide
```

```@example coupled
heatmap_plot(vₕ; title = "Coupled reaction-diffusion, v") # hide
```

## Checking the answer

The same nested-random-mesh pattern as every other example, checking each species' own error
separately — a routing mistake would show up as one converging correctly while the other
silently used the wrong block, which a single combined error could hide. A *dense*
`ForwardDiff.jacobian` over two coupled species would cost `(2n)^2` against the scalar
examples' `n^2`, and was what forced this example to stay at three small refinement levels
before switching to sparse AD; with it, this reaches the same five levels the [linear coupled
example](convection_diffusion_linear.md) uses, tens of thousands of degrees of freedom, in
about a second per level:

```@example coupled
using Random
Random.seed!(20260903)

function coupled_series(; n0::Int = 5, levels::Int)
    Ωc = mesh(Ω, (n0, n0), (false, false))
    hs = Float64[]
    erru, errv = Float64[], Float64[]
    for level in 1:levels
        Wc = gridspace(Ωc)
        Vc = Wc^Val(2)
        bcs_c = dirichlet_constraints(Bramble.set(Ω), :boundary => x -> 0.0)
        f1_c = element(Wc)
        avgₕ!(f1_c, f1)
        f2_c = element(Wc)
        avgₕ!(f2_c, f2)
        l_c = form(Vc, q -> innerₕ(f1_c, q(1)) + innerₕ(f2_c, q(2)))
        F_c = assemble(l_c; dirichlet_conditions = bcs_c, dirichlet_labels = :boundary)

        Ac(wₕ) = begin
            u_c, v_c = components(wₕ)
            assemble(form(Vc, Vc, (p, q) ->
                    inner₊(∇₋ₕ(p(1)), ∇₋ₕ(q(1))) + innerₕ(p(1), q(1)) +
                    innerₕ(v_c * p(1), q(1)) +
                    inner₊(∇₋ₕ(p(2)), ∇₋ₕ(q(2))) + innerₕ(p(2), q(2)) -
                    innerₕ(u_c * p(2), q(2)));
                dirichlet_labels = :boundary)
        end
        rc(w::AbstractVector{T}) where {T} = begin
            wₕ = element(Vc, T)
            wₕ .= w
            Ac(wₕ) * w .- F_c
        end

        w_c = zeros(ndofs(Vc))
        prep_c = prepare_jacobian(rc, sparse_ad, w_c)
        J_c = DifferentiationInterface.jacobian(rc, prep_c, sparse_ad, w_c)
        for it in 1:20
            r = rc(w_c)
            sqrt(sum(abs2, r)) < 1e-10 && break
            DifferentiationInterface.jacobian!(rc, J_c, prep_c, sparse_ad, w_c)
            w_c .-= J_c \ r
        end
        w_ch = element(Vc)
        w_ch .= w_c
        u_ch, v_ch = components(w_ch)
        uexact_c, vexact_c = Rₕ(Wc, u_ex), Rₕ(Wc, v_ex)

        push!(hs, hₘₐₓ(Ωc))
        push!(erru, norm₁ₕ(u_ch .- uexact_c))
        push!(errv, norm₁ₕ(v_ch .- vexact_c))
        level < levels && iterative_refinement!(Ωc)
    end
    return hs, erru, errv
end

hs, erru, errv = coupled_series(; levels = 5)
order_u = log(erru[end - 1] / erru[end]) / log(hs[end - 1] / hs[end])
order_v = log(errv[end - 1] / errv[end]) / log(hs[end - 1] / hs[end])
(order_u, order_v)
```

```@example coupled
order_u > 1.9 && order_v > 1.9
```

```@example coupled
include(joinpath(@__DIR__, "..", "convergence_plot.jl")) # hide
convergence_plot([(hs, erru, "u", "#5B5FC7"), (hs, errv, "v", "#0E7C86")]; title = "Coupled nonlinear reaction, ‖·‖₁ₕ") # hide
```

Second order for both species, same rate as every other example — the composite space and
the quadratic coupling change how the residual and its Jacobian are built, not the
discretization's own accuracy once Newton has converged to it.
