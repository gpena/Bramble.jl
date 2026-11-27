# semidiscrete_constraints.jl: how a semidiscretization's Dirichlet boundary values are
# carried and applied. These functions take the semidiscretisation untyped and dispatch on
# the constraint carrier: `Semidiscretization` (`semidiscrete.jl`) is defined after this file.

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

# The three carriers that hold values, and the conditions each yields at `(t, p)`. They stay
# three types so the method is picked by type, with no flag read per call.
const _ValueConstraints = Union{
    StaticConstraints, TimeDependentConstraints, TimeParamDependentConstraints}

@inline _constraints_at(c::StaticConstraints, t, p) = c.constraints
@inline _constraints_at(c::TimeDependentConstraints, t, p) = c.constraints(t)
@inline _constraints_at(c::TimeParamDependentConstraints, t, p) = c.constraints(t, p)

# Time/parameter dependence is read off the conditions' arity through `_accepts`
# (geometry/marker.jl), which sees through a `Fix` or a composition to the function it
# wraps. A condition of `x` alone is spatial and passes unchanged through `_fix_time` and
# `_fix_time_param`; any other condition makes the set time-dependent. `(x, t, p) -> ...`
# and `(x, t = 0.0, p = 1) -> ...` are parametric. `(x, t) -> ...` and `(x, t = 0.0) -> ...`
# are time-dependent, and the parametric evaluation would call them with a `p` they cannot
# take. So a set holding both kinds throws here, at construction.
# A time-dependent set is stored with each verdict in its markers (`_classify_conditions`),
# so the per-step `conditions` reads it and folds for typed callbacks too.
@inline _dirichlet_is_time_dependent(constraints::ConstraintMarkers) = any(m -> !_is_spatial(identifier(m)), conditions(constraints))

@inline _dirichlet_is_time_param_dependent(constraints::ConstraintMarkers) = any(
    m -> _accepts(identifier(m), Val(3)), conditions(constraints))

function _validate_dirichlet_arity_mix(constraints::ConstraintMarkers)
    conds = conditions(constraints)
    parametric = filter(m -> _accepts(identifier(m), Val(3)), conds)
    isempty(parametric) && return nothing
    timed = filter(m -> !_is_spatial(identifier(m)) && !_accepts(identifier(m), Val(3)), conds)
    isempty(timed) && return nothing
    throw(
        ArgumentError(
        "semidiscretize: Dirichlet conditions cannot mix (x, t) and (x, t, p) arities; " *
        "labels $(map(label, timed)) accept (x, t) only, labels $(map(label, parametric)) " *
        "accept (x, t, p). Give every time-dependent condition the same arity " *
        "(a condition of `x` alone may stay beside them).",
    ),
    )
end

# Checks the arity only: a condition that takes none of `(x)`, `(x, t)`, `(x, t, p)` would
# otherwise be read as time-dependent and fail at the first residual call. The argument
# types are not checked, as the point type is unknown here, so `g(x::Int, t::Real)` on a
# `Float64` mesh passes and fails at the first residual call.
#
# `_accepts` sees a typed method only when no arity has an untyped one, so `g(x)` beside a
# typed `g(x::Real, t::Real)` would be read as spatial and drop `t`. That case is refused
# (`_validate_hidden_typed_arity`, geometry/marker.jl).
function _validate_dirichlet_arity_known(constraints::ConstraintMarkers)
    for m in conditions(constraints)
        f = identifier(m)
        (_accepts(f, Val(1)) || _accepts(f, Val(2)) || _accepts(f, Val(3))) || throw(
            ArgumentError(
            "semidiscretize: Dirichlet condition for label `:$(label(m))` must accept " *
            "(x), (x, t) or (x, t, p), got $(f), which has no method for any of them.",
        ),
        )
        _validate_hidden_typed_arity("semidiscretize", label(m), f)
    end
    return nothing
end

@inline _source_constraints(::Nothing, ::Nothing) = NoConstraints()
@inline _source_constraints(::Any, ::Nothing) = LabelsOnly()
@inline function _source_constraints(::Any, constraints)
    _validate_dirichlet_arity_known(constraints)
    _validate_dirichlet_arity_mix(constraints)
    return if _dirichlet_is_time_param_dependent(constraints)
        TimeParamDependentConstraints(_classify_conditions(constraints))
    elseif _dirichlet_is_time_dependent(constraints)
        TimeDependentConstraints(_classify_conditions(constraints))
    else
        StaticConstraints(constraints)
    end
end

# --- Walking the constrained rows --------------------------------------------------- #

# Mirrors `dirichlet_bc!`'s own route into `_each_marked` (dirichlet_constraints.jl) so the
# rows zeroed in the mass matrix are exactly the rows `assemble` writes `eₖ` into. Scanning
# for them instead would cost `ndofs` rather than the boundary cardinality.
function _each_dirichlet_row(f::F, space::ScalarGridSpace, labels, components) where {F}
    _validate_scalar_components(components)
    Ωₕ = mesh(space)
    for p in labels
        _each_marker_point(f, Ωₕ, p, 0)
    end
    return nothing
end

# Each selected leaf goes through `_each_marker_point`, as the scalar method does:
# `_leaf_entries` would call `index_in_marker`, which builds a `BitVector` for a face name
# that a remapped alias hides.
function _each_dirichlet_row(f::F, space::CompositeGridSpace, labels, components) where {F}
    leaves = leaf_spaces_offsets(space)
    _validate_dirichlet_components(components, length(leaves))
    for p in labels
        _each_selected_leaf(leaves, components) do sp, offset
            _each_marker_point(f, mesh(sp), p, offset)
        end
    end
    return nothing
end

# --- The semidiscretisation --------------------------------------------------------- #

# Everything about how the source's boundary values are reached at each step --
# `NoConstraints`/`LabelsOnly`/`StaticConstraints`/`TimeDependentConstraints`/
# `TimeParamDependentConstraints`, `_source_constraints`, `_assemble_source!`,
# `_apply_initial_constraints!` -- only ever touches `space`/`labels`/`components`/
# `constraints`/`source`, never the operator matrices, so it lives apart from the concrete
# type.

# --- Consistent initial conditions -------------------------------------------------- #

@inline _apply_initial_constraints!(
    u::AbstractVector, sd, ::NoConstraints, p, t
) = u

function _apply_initial_constraints!(
        u::AbstractVector, sd, ::LabelsOnly, p, t
)
    _each_dirichlet_row(sd.space, sd.labels, sd.components) do i
        return @inbounds u[i] = zero(eltype(u))
    end
    return u
end

@inline function _apply_initial_constraints!(
        u::AbstractVector, sd, c::_ValueConstraints, p, t
)
    return dirichlet_bc!(
        u, sd.space, _constraints_at(c, t, p), sd.labels...; components = sd.components
    )
end

# --- Constraint descriptions --------------------------------------------------------- #
#
# Dispatches on the constraint-carrier types above, so it lives here rather than beside
# `Semidiscretization`'s own `Base.show` (`semidiscrete.jl`), which calls this.

function _constraints_description(sd)
    # A ghost space has no labels: its relations are the constraints.
    sd.constraints isa GhostConstraints && return _constraint_description(sd.constraints)
    isempty(sd.labels) && return "none"
    return string(
        _constraint_description(sd.constraints),
        " on ",
        join((string(":", l) for l in sd.labels), ", ")
    )
end

@inline _constraint_description(::LabelsOnly) = "zero"
@inline _constraint_description(::StaticConstraints) = "g(x)"
@inline _constraint_description(::TimeDependentConstraints) = "g(x, t)"
@inline _constraint_description(::TimeParamDependentConstraints) = "g(x, t, p)"
