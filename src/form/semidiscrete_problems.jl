# semidiscrete_problems.jl: the SciMLBase handoff for a `Semidiscretization`/
# `SemidiscretizeRHS` (`semidiscrete.jl`/`semidiscrete_rhs.jl`) -- `ode_function`,
# `ode_problem`, `adjoint_sensitivities`, `linear_problem`.

# --- SciMLBase handoff -------------------------------------------------------------- #
#
# The three entry points below need SciMLBase and are implemented in `BrambleSciMLExt`. Each
# forwards to an underscored fallback whose first argument is `::Any` rather than the
# concrete type, so the extension's method is a strict specialisation: an identical signature
# would overwrite a method during precompilation, which Julia refuses. Same idiom as
# `ast_sparsity_detector`/`_ast_sparsity_detector` and `export_vtk`/`_export_vtk`.

"""
    ode_function(sd::Semidiscretization; kwargs...) -> ODEFunction
    ode_function(a::BilinearForm, l::LinearForm; kwargs...) -> ODEFunction

Wrap a semidiscretisation as an `ODEFunction` carrying its mass matrix, Jacobian and
sparsity, ready for `OrdinaryDiffEq`.

The two-form method builds the [`Semidiscretization`](@ref) first, forwarding every keyword
to [`semidiscretize`](@ref).

# Keywords
- `jacobian`: `jacobian!` (the default) to hand the solver the exact `-A`, or `nothing` to let it build one by automatic differentiation from `jac_prototype`.
- `jac_prototype`: sparsity for the solver's Jacobian cache (default: [`jacobian_prototype`](@ref)`(sd)`).
- `tgrad`: analytical `∂f/∂t`, for a Rosenbrock method to use instead of differentiating through `t` (default: `nothing`, AD). Either SciMLBase's own `(dT, u, p, t) -> ...` signature or the Bramble-aware `(dT, sd, u, p, t) -> ...` one, told apart by arity.

The resulting system is a differential-algebraic one whenever any Dirichlet label is
constrained, since those rows of the mass matrix are zero. Solve it with a method that
admits a singular mass matrix -- `FBDF`, `QNDF`, `Rodas5P`, `RadauIIA5` -- not an explicit
one.

!!! note "Rosenbrock methods and `update_coefficients!`"
    A Rosenbrock method (`Rodas5P`, `Rosenbrock23`) also needs `∂f/∂t`, which it builds by
    differentiating through `t`. That works when the only time dependence is the Dirichlet
    data, since those values reach the assembled vector as its element type. An
    `update_coefficients!` hook writing into a `Float64` [`VectorElement`](@ref) -- the usual
    `t -> Rₕ!(fₕ, x -> f(x, t))` -- cannot take a `ForwardDiff.Dual` time, and the solve
    fails on the first step with a time-gradient error. Three ways out: pass an analytical
    `tgrad`, which bypasses the differentiation through `t` entirely; pass
    `Rodas5P(autodiff = AutoFiniteDiff())`; or use a BDF method, which needs no `∂f/∂t` at
    all. `FBDF` and `QNDF` are unaffected either way.

Requires [SciMLBase.jl](https://github.com/SciML/SciMLBase.jl); call `using SciMLBase` (or
any package that loads it, such as `OrdinaryDiffEq`) before calling this function.

See also [`ode_problem`](@ref), [`linear_problem`](@ref), [`nonlinear_problem`](@ref).
"""
function ode_function(sd::Semidiscretization; kwargs...)
    return _ode_function(sd; kwargs...)
end

function ode_function(
        a::BilinearForm, l::LinearForm;
        jacobian = jacobian!, jac_prototype = nothing, tgrad = nothing, kwargs...
)
    sd = semidiscretize(a, l; kwargs...)
    return _ode_function(sd; jacobian = jacobian, jac_prototype = jac_prototype, tgrad = tgrad)
end

function _ode_function(::Any; kwargs...)
    return error(
        "ode_function requires SciMLBase.jl. Add `using SciMLBase` (or a package that " *
        "loads it, such as OrdinaryDiffEq) before calling this function.",
    )
end

"""
    ode_problem(sd::Semidiscretization, u₀, I::CartesianProduct{1}; kwargs...) -> ODEProblem
    ode_problem(a::BilinearForm, l::LinearForm, u₀, I; kwargs...) -> ODEProblem

Build the `ODEProblem` stepping `sd` over the time domain `I`, from the initial condition
`u₀` -- a [`VectorElement`](@ref) or a plain vector. `I` may equally be a `(t₀, t₁)` tuple.

`u₀` is copied, never mutated, and the copy is made consistent with the Dirichlet rows at
`t₀` (see [`dirichlet_bc!`](@ref)), against `p` when one is given.

Keywords are those of [`ode_function`](@ref), plus:
- `p` (default `SciMLBase.NullParameters()`) for a residual whose `update_coefficients!` or
  Dirichlet conditions were given a parameter-dependent, `(t, p)`/`(x, t, p)` form (see
  [`semidiscretize`](@ref)'s own keywords).
- `specialize` (default `nothing`, `ODEProblem`'s own choice untouched) -- pass
  `SciMLBase.FullSpecialize` before handing the solved trajectory to
  `Bramble.adjoint_sensitivities`: without it, a `p`-vjp calls the residual with a
  differently-`eltype`-`p` than the forward solve used, which the default specialization
  cannot dispatch and fails with "No matching function wrapper was found!" rather than
  differentiating.

The two-form method also forwards its other keywords to [`semidiscretize`](@ref).

Requires [SciMLBase.jl](https://github.com/SciML/SciMLBase.jl).

# Examples

```julia
sd = semidiscretize(a, l; dirichlet = bcs)
prob = ode_problem(sd, Rₕ(Wₕ, x -> sinpi(x[1])), interval(0.0, 1.0))
sol = solve(prob, FBDF())
```

```julia
bcs = dirichlet_constraints(Ωₕ, I, :boundary => (x, t, p) -> p[1] * t)
sd = semidiscretize(a, l; dirichlet = bcs)
prob = ode_problem(sd, u₀, I; p = [0.7])
```

See also [`ode_function`](@ref), [`semidiscretize`](@ref).
"""
function ode_problem(sd::Semidiscretization, u₀, I; kwargs...)
    return _ode_problem(sd, u₀, I; kwargs...)
end

function ode_problem(
        a::BilinearForm,
        l::LinearForm,
        u₀,
        I;
        jacobian = jacobian!,
        jac_prototype = nothing,
        tgrad = nothing,
        p = nothing,
        specialize = nothing,
        kwargs...
)
    sd = semidiscretize(a, l; kwargs...)
    # `p`/`specialize` are pulled out here rather than left in `kwargs...`: `semidiscretize`
    # above has no keyword of its own for either (nothing about assembling `sd` needs them)
    # and no catch-all either, so an unrecognized keyword would raise. Each is omitted
    # entirely (not forwarded as `= nothing`) when the caller didn't ask for it, so
    # `_ode_problem`'s own defaults apply exactly as they did before either existed here.
    p_kwargs = p === nothing ? (;) : (; p = p)
    specialize_kwargs = specialize === nothing ? (;) : (; specialize = specialize)
    return _ode_problem(
        sd, u₀, I; jacobian = jacobian, jac_prototype = jac_prototype, tgrad = tgrad,
        p_kwargs..., specialize_kwargs...
    )
end

"""
    ode_problem(rhs::SemidiscretizeRHS, u₀, I; kwargs...) -> ODEProblem

Build the plain (non-mass-matrix) `ODEProblem` stepping `rhs` over the time domain `I`, from
the initial condition `u₀` -- a [`VectorElement`](@ref) or a plain vector, copied, never
mutated.

Unlike `ode_problem(sd::Semidiscretization, ...)`, there is no Dirichlet consistency step:
[`semidiscretize_rhs`](@ref) only ever builds `rhs` from a `Semidiscretization` with
`dirichlet = nothing`, so there are no boundary rows to make consistent.

Requires [SciMLBase.jl](https://github.com/SciML/SciMLBase.jl).

# Examples

```julia
sd = semidiscretize(a, l)
rhs = semidiscretize_rhs(sd)
prob = ode_problem(rhs, Rₕ(Wₕ, x -> sinpi(x[1])), interval(0.0, 1.0))
sol = solve(prob, Tsit5())
```

See also [`semidiscretize_rhs`](@ref), [`ode_problem`](@ref)`(::Semidiscretization, ...)`.
"""
function ode_problem(rhs::SemidiscretizeRHS, u₀, I; kwargs...)
    return _ode_problem(rhs, u₀, I; kwargs...)
end

function _ode_problem(::Any, u₀, I; kwargs...)
    return error(
        "ode_problem requires SciMLBase.jl. Add `using SciMLBase` (or a package that " *
        "loads it, such as OrdinaryDiffEq) before calling this function.",
    )
end

"""
    adjoint_sensitivities(sol::ODESolution, alg; kwargs...) -> (du0, dp)

Adjoint sensitivities of a [`Semidiscretization`](@ref)'s solved trajectory, via
`SciMLSensitivity.adjoint_sensitivities` with two Bramble-specific corrections applied.
Full documentation lives on `BrambleSciMLSensitivityExt`'s own method, the only one that
exists once `SciMLSensitivity` is loaded -- this stub exists so that method has a function to
extend, and so calling this without `SciMLSensitivity` loaded gives a clear error rather than
`UndefVarError`.

Deliberately not exported, unlike [`pde_solve`](@ref): `SciMLSensitivity` itself exports a
function of this exact name, so `using Bramble, SciMLSensitivity` together would collide on
the bare name regardless of what Bramble does. Call this one as `Bramble.adjoint_sensitivities`.
"""
function adjoint_sensitivities(sol, alg; kwargs...)
    return error(
        "adjoint_sensitivities requires SciMLSensitivity.jl. Add `using SciMLSensitivity` " *
        "before calling this function.",
    )
end

"""
    linear_problem(a::BilinearForm, l::LinearForm; kwargs...) -> LinearProblem

Assemble `a` and `l` into the `LinearProblem` that `LinearSolve.solve` takes, so the steady
system reaches the factorisations, iterative solvers and preconditioners in that stack
without being assembled by hand first.

# Keywords
- `dirichlet`, `dirichlet_components`: as [`assemble`](@ref) takes them.
- `symmetrize`: restore symmetry after imposing the conditions (default: `false`; see [`symmetrize!`](@ref)).

Requires [SciMLBase.jl](https://github.com/SciML/SciMLBase.jl), which defines
`LinearProblem`; `LinearSolve` itself is needed only to solve the result.

# Examples

```julia
using LinearSolve, IncompleteLU

prob = linear_problem(a, l; dirichlet = bcs)
sol = solve(prob, KrylovJL_GMRES())
```

See also [`assemble`](@ref), [`ode_problem`](@ref), [`nonlinear_problem`](@ref).
"""
function linear_problem(a::BilinearForm, l::LinearForm; kwargs...)
    return _linear_problem(a, l; kwargs...)
end

function _linear_problem(::Any, l; kwargs...)
    return error(
        "linear_problem requires SciMLBase.jl. Add `using SciMLBase` (or a package that " *
        "loads it, such as LinearSolve) before calling this function.",
    )
end
