# semidiscrete_constraints.jl: how a semidiscretization's Dirichlet boundary values are
# carried and applied. `AbstractSemidiscretization` is the shared supertype this all
# dispatches on, so that `Semidiscretization` (`semidiscrete.jl`) and
# `SecondOrderSemidiscretization` (`second_order_semidiscrete.jl`) reuse it instead of
# each carrying their own copy -- see its own docstring below for why it lives here.

# --- How the source's boundary values are reached at each step ---------------------- #
#
# Which of these a `Semidiscretization` stores is decided once, at construction, and kept as
# a field whose concrete type selects the `_assemble_source!` method -- rather than a flag
# tested per call inside the residual, which runs once per stage of every step.

"""
    NoConstraints

Source constraint carrier for a problem with no Dirichlet labels at all.
"""
struct NoConstraints end

"""
    LabelsOnly

Source constraint carrier for labels given without values (`dirichlet = :boundary`), which
constrain `u_h` to zero there.
"""
struct LabelsOnly end

"""
    StaticConstraints{C}

Source constraint carrier for time-independent boundary values `g(x)`.
"""
struct StaticConstraints{C}
    constraints::C
end

"""
    TimeDependentConstraints{C}

Source constraint carrier for boundary values `g(x, t)`, evaluated at each step through
`(dm::DomainMarkers)(t)`.
"""
struct TimeDependentConstraints{C}
    constraints::C
end

"""
    TimeParamDependentConstraints{C}

Source constraint carrier for boundary values `g(x, t, p)`, evaluated at each step through
`(dm::DomainMarkers)(t, p)` -- the same residual `p` [`semidiscretize`](@ref)'s `(du, u, p,
t)` call receives and, until now, never used. Kept as a separate carrier from
[`TimeDependentConstraints`](@ref) rather than a `p === nothing` branch inside it, matching
[`EvaluatedParametricDomainMarkers`](@ref)'s own reason for existing alongside
[`EvaluatedDomainMarkers`](@ref).
"""
struct TimeParamDependentConstraints{C}
    constraints::C
end

# Time/parameter dependence is read off the conditions' arity, the same test
# `_validate_time_dependent_arity` uses to check a time domain was passed honestly. A
# spatial condition `x -> ...` has no two- or three-argument method, and a `(x, t) -> ...`
# one has no three-argument method, so the three never collide.
function _dirichlet_is_time_dependent(constraints::ConstraintMarkers)
    conds = conditions(constraints)
    isempty(conds) && return false
    return all(m -> hasmethod(identifier(m), Tuple{Any, Any}), conds)
end

function _dirichlet_is_time_param_dependent(constraints::ConstraintMarkers)
    conds = conditions(constraints)
    isempty(conds) && return false
    return all(m -> hasmethod(identifier(m), Tuple{Any, Any, Any}), conds)
end

@inline _source_constraints(::Nothing, ::Nothing) = NoConstraints()
@inline _source_constraints(::Any, ::Nothing) = LabelsOnly()
@inline _source_constraints(::Any, constraints) =
    if _dirichlet_is_time_param_dependent(constraints)
        TimeParamDependentConstraints(constraints)
    elseif _dirichlet_is_time_dependent(constraints)
        TimeDependentConstraints(constraints)
    else
        StaticConstraints(constraints)
    end

# --- Walking the constrained rows --------------------------------------------------- #

# Mirrors `dirichlet_bc!`'s own route into `_each_marked` (dirichlet_constraints.jl) so the
# rows zeroed in the mass matrix are exactly the rows `assemble` writes `eₖ` into. Scanning
# for them instead would cost `ndofs` rather than the boundary cardinality.
function _each_dirichlet_row(f::F, space::ScalarGridSpace, labels, components) where {F}
    _validate_scalar_components(components)
    Ωₕ = mesh(space)
    for p in labels
        _each_marked(f, index_in_marker(Ωₕ, p), 0)
    end
    return nothing
end

function _each_dirichlet_row(f::F, space::CompositeGridSpace, labels, components) where {F}
    leaves = leaf_spaces_offsets(space)
    _validate_dirichlet_components(components, length(leaves))
    for p in labels
        for (mask, offset, _, active) in _leaf_entries(leaves, p, components)
            active || continue
            _each_marked(f, mask, offset)
        end
    end
    return nothing
end

# --- The semidiscretisation --------------------------------------------------------- #

"""
    AbstractSemidiscretization

Common supertype for [`Semidiscretization`](@ref) (first-order, `M u_h' = F(t) - A u_h`) and
[`SecondOrderSemidiscretization`](@ref) (second-order, `M ü_h + C u̇_h + K u_h = F(t)`).

Everything about how the source's boundary values are reached at each step --
[`NoConstraints`](@ref)/[`LabelsOnly`](@ref)/[`StaticConstraints`](@ref)/
[`TimeDependentConstraints`](@ref)/[`TimeParamDependentConstraints`](@ref),
`_source_constraints`, `_assemble_source!`, `_apply_initial_constraints!` -- only ever
touches `space`/`labels`/`components`/`constraints`/`source`, never the order-specific
operator matrices, so it is shared here
rather than duplicated for the second-order struct.
"""
abstract type AbstractSemidiscretization end

"""
    space(sd::AbstractSemidiscretization)

Return the test space the semidiscretisation was built on.
"""
@inline space(sd::AbstractSemidiscretization) = sd.space

# --- Consistent initial conditions -------------------------------------------------- #

"""
    dirichlet_bc!(u::AbstractVector, sd::AbstractSemidiscretization, t::Number, p = nothing) -> u

Write `sd`'s Dirichlet values at time `t` (and, for a parameter-dependent condition,
parameter `p`) into `u` and return it.

An index-1 differential-algebraic system needs its initial condition to satisfy the
algebraic rows already: a `u` disagreeing with `g(x, 0)` on the boundary is inconsistent,
and a stiff solver either rejects it or absorbs it into the first step.
[`ode_problem`](@ref)/[`second_order_ode_problem`](@ref) apply this to a copy of the initial
condition they are handed, forwarding whatever `p` the caller gave the `ODEProblem`.
"""
function dirichlet_bc!(u::AbstractVector, sd::AbstractSemidiscretization, t::Number, p = nothing)
    return _apply_initial_constraints!(u, sd, sd.constraints, p, t)
end

@inline _apply_initial_constraints!(
    u::AbstractVector, sd::AbstractSemidiscretization, ::NoConstraints, p, t
) = u

function _apply_initial_constraints!(
        u::AbstractVector, sd::AbstractSemidiscretization, ::LabelsOnly, p, t
)
    _each_dirichlet_row(sd.space, sd.labels, sd.components) do i
        return @inbounds u[i] = zero(eltype(u))
    end
    return u
end

@inline _apply_initial_constraints!(
    u::AbstractVector, sd::AbstractSemidiscretization, c::StaticConstraints, p, t
) = dirichlet_bc!(u, sd.space, c.constraints, sd.labels...; components = sd.components)

@inline _apply_initial_constraints!(
    u::AbstractVector, sd::AbstractSemidiscretization, c::TimeDependentConstraints, p, t
) = dirichlet_bc!(u, sd.space, c.constraints(t), sd.labels...; components = sd.components)

@inline _apply_initial_constraints!(
    u::AbstractVector, sd::AbstractSemidiscretization, c::TimeParamDependentConstraints, p, t
) = dirichlet_bc!(u, sd.space, c.constraints(t, p), sd.labels...; components = sd.components)

# --- Constraint descriptions --------------------------------------------------------- #
#
# Dispatches on the constraint-carrier types above, so it lives here rather than beside
# `Semidiscretization`'s own `Base.show` (`semidiscrete.jl`), which calls this.

function _constraints_description(sd::AbstractSemidiscretization)
    isempty(sd.labels) && return "none"
    return string(
        _constraint_description(sd.constraints),
        " on ",
        join((string(":", l) for l in sd.labels), ", ")
    )
end

@inline _constraint_description(::NoConstraints) = "none"
@inline _constraint_description(::LabelsOnly) = "zero"
@inline _constraint_description(::StaticConstraints) = "g(x)"
@inline _constraint_description(::TimeDependentConstraints) = "g(x, t)"
@inline _constraint_description(::TimeParamDependentConstraints) = "g(x, t, p)"
