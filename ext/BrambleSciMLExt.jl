module BrambleSciMLExt

using Bramble:
               Bramble,
               BilinearForm,
               LinearForm,
               Semidiscretization,
               assemble,
               jacobian!,
               jacobian_prototype,
               mass_matrix,
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
               ode_problem,
               ode_function,
               linear_problem,
               boundary_symbols
using SciMLBase: SciMLBase, LinearProblem, ODEFunction, ODEProblem
using PrecompileTools: @setup_workload, @compile_workload

# Gated on SciMLBase alone -- the package `ODEFunction`, `ODEProblem` and `LinearProblem` are
# all defined in. `OrdinaryDiffEq` and `LinearSolve` both depend on it, so loading either
# loads this extension, and neither needs to be a weak dependency of its own.
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

function Bramble._ode_problem(
        sd::Semidiscretization, u₀, I;
        jacobian = jacobian!, jac_prototype = nothing, tgrad = nothing
)
    tspan = _tspan(I)
    u0 = _initial_vector(u₀)
    Bramble.dirichlet_bc!(u0, sd, first(tspan))
    f = Bramble._ode_function(
        sd; jacobian = jacobian, jac_prototype = jac_prototype, tgrad = tgrad
    )
    return ODEProblem(f, u0, tspan)
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

# Warms `ode_problem`/`ode_function`/`linear_problem` -- the three entry points this
# extension defines -- which only exist once `SciMLBase` is loaded, so only this
# extension's own precompile pass (not the core package's, in `src/precompile.jl`) ever
# reaches them. The mesh/space/form/`semidiscretize` setup above `@compile_workload` is
# already covered by the core workload; kept here only to build the `Semidiscretization`
# and `BilinearForm`/`LinearForm` pair these three calls need.
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

        @compile_workload begin
            ode_problem(sd, u0, I_time)
            ode_function(sd)
            linear_problem(a, l)
        end
    end
end

end
