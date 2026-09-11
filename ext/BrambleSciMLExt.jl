module BrambleSciMLExt

using Bramble:
    Bramble,
    BilinearForm,
    LinearForm,
    Semidiscretization,
    assemble,
    jacobian!,
    jacobian_prototype,
    mass_matrix
using SciMLBase: SciMLBase, LinearProblem, ODEFunction, ODEProblem

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
@inline _jac_closure(jacobian::F, sd::Semidiscretization) where {F} =
    (J, u, p, t) -> jacobian(J, sd, u, p, t)

function Bramble._ode_function(
    sd::Semidiscretization; jacobian=jacobian!, jac_prototype=nothing
)
    prototype = jac_prototype === nothing ? jacobian_prototype(sd) : jac_prototype
    return ODEFunction(
        sd;
        mass_matrix=mass_matrix(sd),
        jac_prototype=prototype,
        jac=_jac_closure(jacobian, sd),
    )
end

# A `CartesianProduct{1}` is how a time domain is spelled everywhere else here --
# `dirichlet_constraints(Ωₕ, I, ...)` takes the same object -- so it is accepted alongside
# the `(t₀, t₁)` tuple SciMLBase expects.
@inline _tspan(I::Bramble.CartesianProduct{1}) = extrema(I)
@inline _tspan(tspan::Tuple{Number,Number}) = tspan

# The initial condition is copied before the algebraic rows are written into it: silently
# mutating a caller's `u₀` -- which is usually the `Rₕ` of the exact initial datum, and often
# reused to measure the error afterwards -- would be a poor trade for one allocation.
@inline _initial_vector(u₀::AbstractVector) = collect(u₀)
@inline _initial_vector(u₀) = collect(parent(u₀))

function Bramble._ode_problem(
    sd::Semidiscretization, u₀, I; jacobian=jacobian!, jac_prototype=nothing
)
    tspan = _tspan(I)
    u0 = _initial_vector(u₀)
    Bramble.dirichlet_bc!(u0, sd, first(tspan))
    f = Bramble._ode_function(sd; jacobian=jacobian, jac_prototype=jac_prototype)
    return ODEProblem(f, u0, tspan)
end

function Bramble._linear_problem(
    a::BilinearForm,
    l::LinearForm;
    dirichlet=nothing,
    dirichlet_components=nothing,
    symmetrize::Bool=false,
)
    A, F = assemble(
        a,
        l;
        dirichlet=dirichlet,
        dirichlet_components=dirichlet_components,
        symmetrize=symmetrize,
    )
    return LinearProblem(A, F)
end

end
