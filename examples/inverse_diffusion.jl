# # Recovering a boundary condition (adjoint gradients)
#
# Every other worked example goes forward: known parameters in, a solution out. This one goes
# backward -- noisy observations of a solution in, the parameter that produced them out --
# the shape of a parameter-estimation problem, and of "a neural network predicts a PDE
# parameter, a real solve runs on it, the loss trains the network" (gpena/Bramble.jl#228's own
# motivation). What makes it tractable for more than a handful of parameters is
# [`pde_solve`](@ref)'s adjoint rule: one extra linear solve gives the gradient with respect
# to *every* parameter at once, instead of one extra solve *per* parameter.
#
# This page is generated from `docs/src/examples/inverse_diffusion.jl` by Literate.jl, and the
# same file runs under `test/examples/inverse_diffusion.jl` with the assertions that the page
# renders but does not check. Lines marked `#src` there are the assertions; they never reach
# the page. It needs `Enzyme` and runs behind the `ad`/`full` test groups, unlike the other
# example pages -- see that test file for why.
#
# ## Problem
#
# ```math
# -u'' = f \text{ in } (0,1), \qquad u(0) = u(1) = \theta,
# ```
#
# with ``f(x) = \pi^2 \sin(\pi x)``, so the exact solution for any ``\theta`` is
# ``u_{\text{exact}}(x) = \theta + \sin(\pi x)``. ``\theta`` -- a uniform offset applied to
# both ends of the domain -- is unknown; a handful of noisy point observations of ``u`` at the
# true ``\theta`` are all that is given.
#
# !!! note "Why a boundary value, not a diffusion coefficient"
#     The issue's own motivating example is recovering a *diffusion coefficient* -- `θ`
#     scaling the stiffness form itself, `θ K uₕ = F`. A gradient with respect to the
#     *Dirichlet boundary value*, as on this page, reaches only `F`, filled by
#     `dirichlet_bc!`'s own value-writing path; `assemble`'s stiffness half stays fixed and
#     `Float64`-typed throughout, so none of what follows applies to it.
#
#     A coefficient gradient instead makes `Enzyme` differentiate `assemble`'s own recording
#     pass, which is not currently supported: from a fully inferred call site a `Union` left
#     in the assembly path raises `IllegalTypeAnalysisException`, and above roughly a dozen
#     machine words of stencil -- any difference operator in 2D or 3D -- Enzyme cannot type
#     the mixed offset/weight tuple at all. [`pde_solve`](@ref)'s own docstring states both
#     limits and what does work instead; lifting them is follow-up work
#     (gpena/Bramble.jl#240).
#
# ## Solving it

using Bramble

θ_true = 0.7
uexact(x, θ) = θ + sinpi(x[1])

Ωₕ = mesh(domain(interval(0.0, 1.0)), 41, true)
Wₕ = gridspace(Ωₕ)
a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
fₕ = Rₕ(Wₕ, x -> pi^2 * sinpi(x[1]))
l = form(Wₕ, v -> innerₕ(fₕ, v))

# `a`/`l` are built once, fixed and `Float64`-typed for the rest of the page: only the
# Dirichlet boundary value below depends on `θ`.

function forward(θ::Real)
    A, F = assemble(a, l; dirichlet = :boundary => x -> θ)
    return Bramble.pde_solve(A, F)
end

# ## Synthetic observations
#
# Five interior points, `θ_true`'s exact solution there plus a small fixed perturbation
# standing in for measurement noise -- fixed rather than `Random`-drawn, so the page renders
# the same numbers every build.

obs_idx = (6, 11, 21, 31, 36)
noise = (0.0021, -0.0035, 0.0012, -0.0018, 0.0027)
pts = points(Ωₕ)
u_obs = [uexact(pts[i], θ_true) + noise[k] for (k, i) in enumerate(obs_idx)]

loss(θ::Real) = sum(abs2, forward(θ)[obs_idx[k]] - u_obs[k] for k in eachindex(obs_idx))

# ## Gradient descent, driven by the adjoint
#
# `using Enzyme` is all the setup this needs: `BrambleEnzymeExt` defines a native
# `EnzymeRules` reverse rule for [`pde_solve`](@ref), so `Enzyme.gradient` reaches the adjoint
# directly. Do *not* add `Enzyme.@import_rrule(typeof(pde_solve), ...)` here -- besides
# defining a second rule for the same signature, that bridge corrupts Enzyme's shadow of a
# sparse `A` whenever the cotangent carries an explicit zero, and returns a wrong gradient
# with no error at all (`pde_solve`'s own docstring, gpena/Bramble.jl#240).
#
# `loss` closes over `Wₕ`/`a`/`l`, which Enzyme cannot prove read-only on its own, so the
# function argument is wrapped `Const` -- the same annotation `docs/src/tutorials/autodiff.md`
# and `test/space/autodiff_heavy.jl` already document for the ordinary (non-solve) path.

using Enzyme

mode = Enzyme.set_runtime_activity(Enzyme.Reverse)

θ = 0.3   # deliberately far from θ_true = 0.7
history = Float64[θ]
step = 0.05
for _ in 1:60
    g = Enzyme.gradient(mode, Enzyme.Const(loss), θ)[1]
    global θ -= step * g
    push!(history, θ)
end

(θ, θ_true)

# Recovered to within the noise level the synthetic observations carry, from a starting guess
# more than a factor of two away.                                                     #src
@test isapprox(θ, θ_true; atol = 0.02)                                                #src
@test loss(θ) < loss(0.3)                                                             #src

# ## Cost: one extra solve, not one per parameter
#
# The point of the adjoint rule: `Enzyme.gradient` above costs one forward solve plus one
# adjoint solve, regardless of how many parameters `θ` were -- a coefficient *field* with one
# value per grid point would cost the same two solves, where forward-mode differentiation
# through the same model would cost one solve per parameter. A direct timing comparison against
# a naive `ForwardDiff`-through-the-solve baseline is not run on this page (that baseline fails
# outright here for an unrelated reason: UMFPACK's sparse factorisation only accepts
# `Float64`/`ComplexF64`, so `A \ F` for a `Dual`-valued `F` errors regardless of this scaling
# question -- `docs/src/tutorials/autodiff.md` §5 documents this and recommends a dense
# conversion or an iterative solver as the forward-mode workaround) -- see
# `test/ext/chainrules_ext.jl` for the scaling measurement itself: the adjoint rule's own
# pullback cost against `nθ` separate forward solves, the per-parameter cost a naive
# forward-mode scheme pays.
#
# ## See also
#
#   - [`pde_solve`](@ref) in the [API reference](../api.md).
#   - [Linear Poisson](poisson_linear.md) for the forward problem this inverts.
