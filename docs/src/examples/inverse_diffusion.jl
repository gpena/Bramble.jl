# # Recovering a diffusion coefficient (adjoint gradients)
#
# This example recovers a diffusion coefficient from noisy observations of a solution, the
# shape of a parameter-estimation problem or of training a network whose output feeds a PDE
# solve. [`pde_solve`](@ref)'s adjoint rule gives the gradient with respect to every
# parameter for one extra linear solve. The gradient needs `Enzyme`.
#
# ## Problem
#
# ```math
# -\kappa u'' = f \text{ in } (0,1), \qquad u(0) = u(1) = 0,
# ```
#
# with ``f(x) = \kappa_{\text{true}} \pi^2 \sin(\pi x)``, so the exact solution at any
# ``\kappa`` is ``u_{\text{exact}}(x, \kappa) = (\kappa_{\text{true}} / \kappa) \sin(\pi x)``,
# equal to ``\sin(\pi x)`` exactly at ``\kappa = \kappa_{\text{true}}``. The diffusion
# coefficient ``\kappa`` is unknown, and a handful of noisy point observations of ``u`` at
# the true ``\kappa`` are all that is given.
#
# ## Discretisation and the forward solve

using Bramble

κ_true = 0.7
f(x) = κ_true * pi^2 * sinpi(x[1])
uexact(x, κ) = (κ_true / κ) * sinpi(x[1])

Ωₕ = mesh(domain(interval(0.0, 1.0)), 41)
Wₕ = gridspace(Ωₕ)
fₕ = Rₕ(Wₕ, f)
l = form(Wₕ, v -> innerₕ(fₕ, v))

# `l` is built once and stays `Float64`-typed. `aκ` is rebuilt inside `forward` on every
# call, because `κ` scales the stiffness form itself.

function forward(κ::Real)
    aκ = form(Wₕ, Wₕ, (u, v) -> κ * inner₊(∇ₕ(u), ∇ₕ(v)))
    A, F = assemble(aκ, l; dirichlet = :boundary => x -> 0.0)
    return pde_solve(A, F)
end

# ## Synthetic observations
#
# Five interior points, `κ_true`'s exact solution there plus a small perturbation standing
# in for measurement noise. The perturbation is fixed, so the page renders the same numbers
# every build. The mesh is uniform, so these indices sit at
# ``x = 0.125, 0.25, 0.5, 0.75, 0.875``.

obs_idx = (6, 11, 21, 31, 36)
noise = (0.0021, -0.0035, 0.0012, -0.0018, 0.0027)
pts = points(Ωₕ)
u_obs = [uexact(pts[i], κ_true) + noise[k] for (k, i) in enumerate(obs_idx)]

loss(κ::Real) = sum(abs2, forward(κ)[obs_idx[k]] - u_obs[k] for k in eachindex(obs_idx))

# ## Gradient descent, driven by the adjoint
#
# `using Enzyme` and `import DifferentiationInterface as DI` are all the setup this needs:
# `BrambleEnzymeExt` defines a native `EnzymeRules` reverse rule for [`pde_solve`](@ref), so
# Enzyme's reverse mode reaches the adjoint directly, and DifferentiationInterface.jl (`DI`)
# calls it through an `AutoEnzyme` backend. Do *not* add
# `Enzyme.@import_rrule(typeof(pde_solve), ...)` here: besides defining a second rule for the
# same signature, that bridge corrupts Enzyme's shadow of a sparse `A` whenever the cotangent
# carries an explicit zero, and returns a wrong gradient with no error at all (`pde_solve`'s
# own docstring).
#
# `loss` closes over `Wₕ`/`l`, which Enzyme cannot prove read-only on its own, so the backend
# wraps the function `Const` (`function_annotation`) and sets `set_runtime_activity`, as in
# the [automatic differentiation tutorial](@ref tutorial_autodiff).

using Enzyme
import DifferentiationInterface as DI

backend = DI.AutoEnzyme(;
    mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
    function_annotation = Enzyme.Const)

κ = 0.3   # deliberately far from κ_true = 0.7
step = 0.02
for _ in 1:100
    g = DI.gradient(loss, backend, κ)
    global κ -= step * g
end

(κ, κ_true)

# Recovered to within the noise level the synthetic observations carry, from a starting guess
# more than a factor of two away.
@test isapprox(κ, κ_true; atol = 0.005)                                               #src
@test loss(κ) < loss(0.3)                                                             #src
fd_check = (loss(0.5 + 1e-6) - loss(0.5 - 1e-6)) / 2e-6                               #src
ad_check = DI.gradient(loss, backend, 0.5)                                            #src
@test ad_check ≈ fd_check rtol = 1e-6                                                 #src

# ## Cost of the adjoint
#
# `DI.gradient` costs one forward solve and one adjoint solve for any number of parameters.
# A coefficient field with one value per grid point costs the same two solves. Forward mode
# needs one solve per parameter: with `ForwardDiff` loaded, `pde_solve` runs on a `Dual`
# matrix, factors the `Float64` values once, and each partial is one solve against that
# factor.
#
# ## Where to go next
#
#   - [`pde_solve`](@ref) in the [API reference](../api.md).
#   - [Linear Poisson](poisson_linear.md) for the forward problem this inverts.
#   - [Automatic differentiation](../tutorials/autodiff.md) for the differentiation tools
#     this page relies on.
