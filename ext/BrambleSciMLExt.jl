module BrambleSciMLExt

using Bramble:
               Bramble,
               BilinearForm,
               LinearForm,
               Semidiscretization,
               SemidiscretizeRHS,
               SecondOrderSemidiscretization,
               assemble,
               jacobian!,
               jacobian_prototype,
               mass_matrix,
               block_mass_matrix,
               domain,
               interval,
               mesh,
               gridspace,
               Rₕ,
               form,
               inner₊,
               ∇₋ₕ,
               innerₕ,
               dirichlet_constraints,
               semidiscretize,
               semidiscretize_second_order,
               ode_problem,
               ode_function,
               second_order_ode_problem,
               second_order_ode_function,
               linear_problem,
               nonlinear_problem,
               trial_space,
               boundary_symbols
using SciMLBase:
                 SciMLBase,
                 LinearProblem,
                 LinearSolution,
                 ODEFunction,
                 ODEProblem,
                 DynamicalODEFunction,
                 SecondOrderODEProblem,
                 NonlinearFunction,
                 NonlinearProblem,
                 solve
using PrecompileTools: @setup_workload, @compile_workload

# Gated on SciMLBase alone -- the package `ODEFunction`, `ODEProblem`, `LinearProblem`,
# `NonlinearFunction` and `NonlinearProblem` are all defined in. `OrdinaryDiffEq`,
# `LinearSolve` and `NonlinearSolve` each depend on it, so loading any one of them loads this
# extension, and none needs to be a weak dependency of its own.
#
# Everything with substance lives in `src/form/semidiscrete.jl`: a `Semidiscretization` is
# already a callable with the `(du, u, p, t)` signature, already carries its mass matrix, and
# already knows its own Jacobian. This file only names those pieces the way SciMLBase does.

# `jacobian = nothing` leaves `jac` unset, which is what a solution-dependent operator wants:
# `jacobian!` is only a Picard linearisation there, and the solver should build the real one
# by (sparse) automatic differentiation from `jac_prototype` instead.
@inline _jac_closure(::Nothing, ::Semidiscretization) = nothing
@inline _jac_closure(jacobian::F, sd::Semidiscretization) where {F} = (J, u, p, t) -> jacobian(J, sd, u, p, t)

# `tgrad` accepts either SciMLBase's own `(dT, u, p, t)` signature or a Bramble-aware
# `(dT, sd, u, p, t)` one, told apart by arity the same way `_dirichlet_is_time_dependent`
# tells a time-dependent condition from a spatial one: a 4-argument method exists on the
# plain SciML signature, a 5-argument one on the Bramble-aware form, and never both.
@inline _tgrad_closure(::Nothing, ::Semidiscretization) = nothing
@inline function _tgrad_closure(tgrad::F, sd::Semidiscretization) where {F}
    return hasmethod(tgrad, Tuple{Any, Any, Any, Any, Any}) ?
           (dT, u, p, t) -> tgrad(dT, sd, u, p, t) : tgrad
end

function Bramble._ode_function(
        sd::Semidiscretization; jacobian = jacobian!, jac_prototype = nothing, tgrad = nothing
)
    prototype = jac_prototype === nothing ? jacobian_prototype(sd) : jac_prototype
    return ODEFunction(
        sd;
        mass_matrix = mass_matrix(sd),
        jac_prototype = prototype,
        jac = _jac_closure(jacobian, sd),
        tgrad = _tgrad_closure(tgrad, sd)
    )
end

# A `CartesianProduct{1}` is how a time domain is spelled everywhere else here --
# `dirichlet_constraints(Ωₕ, I, ...)` takes the same object -- so it is accepted alongside
# the `(t₀, t₁)` tuple SciMLBase expects.
@inline _tspan(I::Bramble.CartesianProduct{1}) = extrema(I)
@inline _tspan(tspan::Tuple{Number, Number}) = tspan

# The initial condition is copied before the algebraic rows are written into it: silently
# mutating a caller's `u₀` -- which is usually the `Rₕ` of the exact initial datum, and often
# reused to measure the error afterwards -- would be a poor trade for one allocation.
@inline _initial_vector(u₀::AbstractVector) = collect(u₀)
@inline _initial_vector(u₀) = collect(parent(u₀))

# `p` defaults to `SciMLBase.NullParameters()`, `ODEProblem`'s own default -- passed
# positionally rather than as a `kwargs...` entry, since `ODEProblem(f, u0, tspan; kwargs...)`
# (the method a plain callable `f`/`rhs` reaches, as opposed to the `@add_kwonly`-generated
# one an `AbstractODEFunction` reaches) does not bind a `p` keyword to that positional
# argument at all; it would silently land in `kwargs` instead and never reach the residual.
#
# `specialize` defaults to `nothing`, leaving `ODEProblem`'s own specialization choice
# untouched -- existing callers get exactly the problem they always did. Passing
# `SciMLBase.FullSpecialize` is what `Bramble.adjoint_sensitivities`
# (`BrambleSciMLSensitivityExt`) needs: without it, calling the residual with a
# differently-`eltype`-`p` than the forward solve used (exactly what computing a `p`-vjp
# does) fails with "No matching function wrapper was found!" rather than differentiating.
# The type-parameterized `ODEProblem{iip, specialize}` constructor is the only one that
# accepts a choice of specialization at all -- there is no keyword for it on the plain one.
function Bramble._ode_problem(
        sd::Semidiscretization, u₀, I;
        jacobian = jacobian!, jac_prototype = nothing, tgrad = nothing,
        p = SciMLBase.NullParameters(), specialize = nothing
)
    tspan = _tspan(I)
    u0 = _initial_vector(u₀)
    Bramble.dirichlet_bc!(u0, sd, first(tspan), p)
    f = Bramble._ode_function(
        sd; jacobian = jacobian, jac_prototype = jac_prototype, tgrad = tgrad
    )
    return specialize === nothing ? ODEProblem(f, u0, tspan, p) :
           ODEProblem{true, specialize}(f, u0, tspan, p)
end

# No `dirichlet_bc!` consistency step, unlike the `Semidiscretization` method above:
# `SemidiscretizeRHS` only ever wraps a `Semidiscretization` built with `dirichlet =
# nothing` (`Bramble.semidiscretize_rhs` checks), so there are no boundary rows to make
# consistent. `ODEProblem(rhs, u0, tspan, p)` carries no `mass_matrix` either -- `rhs`
# already folded `M⁻¹` in, which is the whole point.
function Bramble._ode_problem(
        rhs::SemidiscretizeRHS, u₀, I; p = SciMLBase.NullParameters(), kwargs...
)
    tspan = _tspan(I)
    u0 = _initial_vector(u₀)
    return ODEProblem(rhs, u0, tspan, p; kwargs...)
end

function Bramble._second_order_ode_function(sd::SecondOrderSemidiscretization)
    return DynamicalODEFunction(
        (dv, v, u, p, t) -> sd(dv, v, u, p, t);
        mass_matrix = block_mass_matrix(sd)
    )
end

function Bramble._second_order_ode_problem(sd::SecondOrderSemidiscretization, du₀, u₀, I; kwargs...)
    tspan = _tspan(I)
    u0 = _initial_vector(u₀)
    dv0 = _initial_vector(du₀)
    Bramble.dirichlet_bc!(u0, sd, first(tspan))
    f = Bramble._second_order_ode_function(sd)
    return SecondOrderODEProblem(f, dv0, u0, tspan)
end

function Bramble._linear_problem(
        a::BilinearForm,
        l::LinearForm;
        dirichlet = nothing,
        dirichlet_components = nothing,
        symmetrize::Bool = false
)
    A, F = assemble(
        a,
        l;
        dirichlet = dirichlet,
        dirichlet_components = dirichlet_components,
        symmetrize = symmetrize
    )
    return LinearProblem(A, F)
end

# `sol.u` is exactly the plain vector `element(Wₕ, ::AbstractVector)` already knows how to
# wrap, so both unwrapping spellings just reroute to it. `Wₕ` is narrowed to
# `AbstractSpaceType` -- a `LinearSolution` is itself an `AbstractVector`, so an untyped `Wₕ`
# here would be ambiguous with `element(Wₕ::AbstractSpaceType, v::AbstractVector)` above:
# neither method would be strictly more specific than the other.
Bramble.element(Wₕ::Bramble.AbstractSpaceType, sol::LinearSolution) = Bramble.element(Wₕ, sol.u)
Bramble.VectorElement(sol::LinearSolution, Wₕ::Bramble.AbstractSpaceType) = Bramble.element(Wₕ, sol.u)

"""
    solve(a::BilinearForm, l::LinearForm; dirichlet = nothing, dirichlet_components = nothing,
          symmetrize = false, solver = nothing, preconditioner = nothing, kwargs...) -> VectorElement

Assemble `a` and `l`, solve the resulting linear system, and hand back the solution as a
`VectorElement` over `a`'s trial space -- the convenience path around [`linear_problem`](@ref)
for a caller who wants the solved grid function directly, not the raw `LinearProblem` and a
bare coefficient vector to wrap by hand.

# Keywords
- `dirichlet`, `dirichlet_components`, `symmetrize`: forwarded to [`assemble`](@ref), exactly
  as [`linear_problem`](@ref) takes them.
- `solver`: the `LinearSolve` algorithm, e.g. `KrylovJL_GMRES()`. Default `nothing`: `solve`
  picks its own default.
- `preconditioner`: `:amg` to precondition an iterative `solver` with
  [`amg_preconditioner`](@ref) (requires
  [AlgebraicMultigrid.jl](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl)),
  `:ilu0` for [`ilu_preconditioner`](@ref) (requires
  [ILUZero.jl](https://github.com/mohamed82008/ILUZero.jl); a better fit than `:amg` for
  unsymmetric, convection-dominated forms, see [`ilu_preconditioner`](@ref)), an
  already-built object with `ldiv!` to use as-is, or `nothing` (default) for none. Passed as
  `LinearSolve`'s `Pl` -- a `LinearProblem`'s own `Pl`/`Pr` keywords are not honoured by its
  Krylov algorithms, only ones given to `solve` itself, so this keyword lives here rather
  than on [`linear_problem`](@ref).
- Every other keyword forwards to `LinearSolve.solve`.

# Examples

```julia
uₕ = solve(a, l; dirichlet = bcs)

using AlgebraicMultigrid
uₕ = solve(a, l; dirichlet = bcs, solver = KrylovJL_CG(), preconditioner = :amg)
```

See also [`linear_problem`](@ref), [`amg_preconditioner`](@ref), [`ilu_preconditioner`](@ref),
[`element`](@ref).
"""
function SciMLBase.solve(
        a::BilinearForm, l::LinearForm;
        dirichlet = nothing, dirichlet_components = nothing, symmetrize::Bool = false,
        solver = nothing, preconditioner = nothing, kwargs...
)
    prob = linear_problem(
        a, l; dirichlet = dirichlet, dirichlet_components = dirichlet_components,
        symmetrize = symmetrize
    )
    Pl = _preconditioner_operator(preconditioner, prob.A)
    solve_kwargs = Pl === nothing ? kwargs : (; Pl = Pl, kwargs...)
    sol = solver === nothing ? solve(prob; solve_kwargs...) : solve(prob, solver; solve_kwargs...)
    return Bramble.element(trial_space(a), sol)
end

# `preconditioner` is resolved here rather than inside `_linear_problem` -- a `LinearProblem`
# built with `Pl`/`Pr` in its own keywords does not actually reach a Krylov algorithm's
# solve; only a `Pl` given to `solve` itself does, verified against `LinearSolve` directly.
# `:amg`/`:ilu0` reach AMG/ILU(0) through `Bramble._amg_operator`/`Bramble._ilu_operator`,
# `BrambleAlgebraicMultigridExt`/`BrambleILUZeroExt`'s fallback-idiom counterparts to
# `_amg_preconditioner`/`_ilu_preconditioner` (solvers/amg_preconditioner.jl,
# solvers/ilu_preconditioner.jl) -- this extension calls them without ever depending on
# `AlgebraicMultigrid`/`ILUZero` itself.
_preconditioner_operator(::Nothing, ::AbstractMatrix) = nothing
function _preconditioner_operator(preconditioner::Symbol, A::AbstractMatrix)
    if preconditioner === :amg
        return Bramble._amg_operator(A)
    elseif preconditioner === :ilu0
        return Bramble._ilu_operator(A)
    else
        throw(
            ArgumentError("Unknown preconditioner: $preconditioner. Expected :amg, :ilu0, or nothing."),
        )
    end
end
_preconditioner_operator(preconditioner, ::AbstractMatrix) = preconditioner

# `u0` narrows to `AbstractVector` (a `VectorElement` already is one) rather than `residual`
# to `::Any` there -- an arbitrary user function has no Bramble type to narrow to, unlike
# `BilinearForm`/`Semidiscretization` above. Narrowing on either argument is a strict
# specialisation of `Bramble.jl`'s `_nonlinear_problem(residual, ::Any; kwargs...)` fallback,
# which is all the method-overwrite check needs (form/nonlinear_problem.jl explains).
#
# No `_jac_closure`-style wrapping needed: unlike `jacobian!` above, which needs `sd` closed
# over it, a `jacobian` passed here is already the user's own `jac(J, u, p)`/`J = jac(u, p)`,
# exactly `NonlinearFunction`'s own signature -- passed straight through.
function Bramble._nonlinear_problem(
        residual, u0::AbstractVector;
        jacobian = nothing, jac_prototype = nothing, kwargs...
)
    f = NonlinearFunction(residual; jac = jacobian, jac_prototype = jac_prototype)
    return NonlinearProblem(f, _initial_vector(u0); kwargs...)
end

# Warms `ode_problem`/`ode_function`/`linear_problem`/`nonlinear_problem` -- the four entry
# points this extension defines -- which only exist once `SciMLBase` is loaded, so only this
# extension's own precompile pass (not the core package's, in `src/precompile.jl`) ever
# reaches them. The mesh/space/form/`semidiscretize` setup above `@compile_workload` is
# already covered by the core workload; kept here only to build the `Semidiscretization`
# and `BilinearForm`/`LinearForm` pair these calls need.
if Bramble.PRECOMPILE_WORKLOAD
    @setup_workload begin
        I0 = interval(0.0, 1.0)
        Ωₕ = mesh(domain(I0, :boundary => boundary_symbols(I0)), 5, true)
        Wₕ = gridspace(Ωₕ)
        I_time = interval(0.0, 1.0)
        fₕ = Rₕ(Wₕ, x -> 1.0)
        a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
        l = form(Wₕ, v -> innerₕ(fₕ, v))
        bcs = dirichlet_constraints(Ωₕ, I_time, :boundary => (x, t) -> 0.0)
        sd = semidiscretize(a, l; dirichlet = bcs)
        u0 = Rₕ(Wₕ, x -> 0.0)
        sd2 = semidiscretize_second_order(a, l)
        du0 = Rₕ(Wₕ, x -> 0.0)

        @compile_workload begin
            ode_problem(sd, u0, I_time)
            ode_function(sd)
            second_order_ode_problem(sd2, du0, u0, I_time)
            second_order_ode_function(sd2)
            linear_problem(a, l)
            nonlinear_problem((u, p) -> u, u0)
        end
    end
end

end
