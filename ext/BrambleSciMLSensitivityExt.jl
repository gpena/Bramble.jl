module BrambleSciMLSensitivityExt

using Bramble: Bramble, Semidiscretization, mass_matrix
using SciMLBase: SciMLBase, ODESolution
using SciMLSensitivity: SciMLSensitivity, InterpolatingAdjoint, BrownFullBasicInit
# `transpose` is `Base`'s own -- every module sees it already, so it is used unqualified
# below with no `using LinearAlgebra: transpose` at all (that import would itself fail
# `test/quality/explicit_imports.jl`'s "imports come from the true owner" check, since
# `LinearAlgebra` only re-exports it).

# `Bramble.adjoint_sensitivities`: the transient counterpart of `pde_solve`'s steady-state
# adjoint rule (gpena/Bramble.jl#228), for a `Semidiscretization`'s `ode_problem` trajectory
# (gpena/Bramble.jl#239). Where `pde_solve` earns a ~20-line hand-written `ChainRulesCore`/
# `EnzymeRules` rule because the steady adjoint is that small, the transient case does not:
# a discrete or continuous adjoint for a general time-stepping method is substantially more
# machinery, so this wraps `SciMLSensitivity.adjoint_sensitivities` instead of writing one.
#
# Composed directly against a `Semidiscretization`-backed `ode_problem`/`solve` trajectory
# while investigating #239 (posted to the issue), including the mutating residual and the
# singular (index-1 DAE) mass matrix a constrained problem has -- neither needed a bespoke
# workaround, once three caller-side gaps were closed:
#
# 1. `ode_problem(...; specialize = SciMLBase.FullSpecialize)`. Without it, the default
#    specialization builds a function wrapper for the `p`-eltype the *forward* solve ran at;
#    computing a `p`-vjp calls the residual with a *different* `p` eltype (`Dual`/`Tracker`
#    depending on `sensealg`), and the wrapper raises "No matching function wrapper was
#    found!" rather than differentiating. This is set at `ode_problem` construction, before
#    `solve` ever runs, so it cannot be fixed here after the fact -- callers must ask for it.
# 2. `initializealg = BrownFullBasicInit()`. The adjoint DAE inherits the forward problem's
#    algebraic structure: with `D` the constrained rows, `semidiscretize`'s convention gives
#    `A[D, D] = I`, `A[D, Dᶜ] = 0`, so `(Aᵀ)[D, D] = I` and the adjoint is index-1 too, but
#    its algebraic rows need `λ_D = ∂g/∂u_D - (A[Dᶜ, D]ᵀ λ_Dᶜ)`, not the raw `∂g/∂u` every
#    `sensealg` here seeds `λ(T)` with. The default `CheckInit` only verifies consistency
#    rather than restoring it, and rejects the seed with a `normresid` error; the alternative
#    default already documents as unsupported (`ShampineCollocationInit`, checked and about a
#    decimal place less accurate).
# 3. `du0` needs `Mᵀ`. `adjoint_sensitivities` returns `λ(0)`, but the Lagrangian's initial
#    boundary term is `λ(0)ᵀ M δu₀`, not `λ(0)ᵀ δu₀` -- it drops the mass matrix from the one
#    place it appears in the *initial-condition* gradient (the interior ODE's own `Mᵀλ' =
#    Aᵀλ - (∂g/∂u)ᵀ` is unaffected; only the boundary term at `t = 0` is). Measured on a
#    Bramble discretisation, where `M` is `innerₕ`'s diagonal `≈ h`: the raw `du0` was wrong
#    by 19.999996× on a uniform 21-point 1D mesh (`h⁻¹` exactly) and 19.807× on a
#    non-uniform one -- the non-uniform case is what confirms it is the full `Mᵀ`, not a
#    scalar mesh width, since a uniform mesh's `1/h` cannot be told apart from that. This is
#    silent: no error, no warning, and it lands squarely on "one parameter entering the
#    initial condition" -- gpena/Bramble.jl#239's own first acceptance criterion.
#
# The parameter gradient `dp` (`∫ λᵀ ∂F/∂p dt`) carries no `M` and needs no correction.
#
# Scope, deliberately: everything above holds `A` constant, the case every acceptance
# criterion in #239 asks for (an initial condition, a time-dependent Dirichlet value). With
# `state` set or `reassemble = true` the operator depends on `u(t)`, the backward pass needs
# the forward trajectory, and checkpointing engages -- untested here, and left as follow-up
# rather than pretended to come free.

"""
    Bramble.adjoint_sensitivities(sol::ODESolution, alg; kwargs...) -> (du0, dp)

Adjoint sensitivities of a [`Semidiscretization`](@ref)'s solved trajectory `sol` (from
[`ode_problem`](@ref)/`solve`) with respect to its initial condition (`du0`) and its `p`
(`dp`), via `SciMLSensitivity.adjoint_sensitivities` -- one backward solve for *every*
parameter at once, the same O(1)-in-parameter-count trade [`pde_solve`](@ref)'s own adjoint
rule makes for the steady case.

# Keywords
Every keyword `SciMLSensitivity.adjoint_sensitivities` takes, plus these two defaults chosen
for a `Semidiscretization`'s index-1 DAE specifically (both overridable):
- `sensealg`: `InterpolatingAdjoint(autojacvec = false)` -- `autojacvec = false` uses
  [`jacobian!`](@ref)'s own exact `-A` for the `u`-vjp, so no AD tool ever needs to
  differentiate through the residual's mutating buffers for that half.
- `initializealg`: `BrownFullBasicInit()` -- restores the adjoint's own algebraic
  consistency at `t = T` rather than merely checking it (the default `CheckInit` rejects a
  constrained problem's seed outright); see this file's own module-level comment for why.

# Requirements on `sol`
`sol.prob` must have been built with `ode_problem(sd, u₀, I; p = ..., specialize =
SciMLBase.FullSpecialize)` -- both `p` and `specialize` matter: `specialize` avoids a
function-wrapper error the moment a `p`-vjp is computed (see the note on `ode_problem`
itself), and without a real `p` there is nothing for `dp` to be a gradient with respect to.

# Examples

```julia
using Bramble, SciMLBase, SciMLSensitivity, OrdinaryDiffEqBDF

bcs = dirichlet_constraints(Ωₕ, I, :boundary => (x, t, p) -> p[1] * t)
sd = semidiscretize(a, l; dirichlet = bcs)
prob = ode_problem(sd, u₀, I; p = [0.7], specialize = SciMLBase.FullSpecialize)
sol = solve(prob, FBDF(); saveat = ts)

dgdu!(out, u, p, t, i) = (@. out = 2 * (u - obs[i]); nothing)
du0, dp = Bramble.adjoint_sensitivities(sol, FBDF(); t = ts, dgdu_discrete = dgdu!)
```

See also [`ode_problem`](@ref), [`pde_solve`](@ref) for the steady-state adjoint this
mirrors.
"""
function Bramble.adjoint_sensitivities(
        sol::ODESolution, alg;
        sensealg = InterpolatingAdjoint(autojacvec = false),
        initializealg = BrownFullBasicInit(),
        kwargs...
)
    du0, dp = SciMLSensitivity.adjoint_sensitivities(
        sol, alg; sensealg = sensealg, initializealg = initializealg, kwargs...
    )
    return _correct_initial_adjoint(du0, sol.prob.f.f), dp
end

# `Mᵀ` corrects the initial-condition gradient for exactly the case where Bramble handed
# SciMLBase a `Semidiscretization`'s own mass matrix -- `sol.prob.f.f` is the underlying
# callable an `ODEFunction` always wraps at `.f`. Anything else (a `SemidiscretizeRHS`,
# whose `M⁻¹` is already folded into the residual and carries no `mass_matrix` of its own;
# a plain user residual) is left untouched: there is no known convention here to correct
# for, and guessing one would be worse than doing nothing.
@inline _correct_initial_adjoint(du0, ::Any) = du0
@inline _correct_initial_adjoint(du0, sd::Semidiscretization) = transpose(mass_matrix(sd)) * du0

end
