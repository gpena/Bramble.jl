# ghost_semidiscrete.jl: `semidiscretize` and `ode_problem` on forms posed on a constrained
# ghost space, whose relation data may depend on time, `(x, t) -> ...`.
#
# The padded values of the degrees of freedom `u` are `E u + d(t)`, `E` fixed and `d(t)` the
# affine part of the closure (`_lift_affine!`). With `A = Eᵀ A_pad E` and `M = Eᵀ M_pad E`,
# the method of lines on the padded system gives
#
#     M u' = Eᵀ (F_pad(t) − A_pad d(t)) − A u − Eᵀ M_pad d′(t),
#
# and the last term vanishes when `M_pad` reads no point where `d` depends on time, as the
# default `innerₕ(u, v)` mass does: it weighs the physical nodes only, and a node fixed by
# `u => g(x, t)` is not a degree of freedom (its row of `E` is empty). Any other mass is
# refused, since the term needs `∂ₜ g`, which no relation carries. Relations eliminate the
# ghosts and the Dirichlet nodes, so the system has no algebraic rows.
#
# The result is the plain `Semidiscretization`, so `ode_function`, `ode_problem`,
# `jacobian!` and `BrambleSciMLExt` take it unchanged: the operator is the ghost form,
# refilled through `_assemble_ghost!` under `reassemble`, and the constraints carrier
# `GhostConstraints` makes `_assemble_source!` write `Eᵀ (F_pad(t) − A_pad d(t))` with the
# banded load of `ghost_assembly.jl`, `A_pad` as the operator last filled it. Its buffers
# are preallocated, so a residual at the space's element type allocates nothing; a wider
# time (a `ForwardDiff.Dual`) gets buffers of its own type for that call. A `u` the padded
# state cannot hold (a `Dual`) goes to a twin, as on a plain space: a padded state of its
# type, the forms rebound to it, and the operator's own `A` and `A_pad`, which the load reads.
#
# Refused, each naming the alternative: the `dirichlet` keyword, a space keeping ghosts as
# unknowns, a time-dependent operator built by a function, a form on a ghost space with
# one on a plain space, and `semidiscretize_rhs`.
#
# See also: `Semidiscretization`, `GhostBilinearForm`, `constrained`, `lift`

"""
    GhostConstraints{DV}

Source constraint carrier of a [`Semidiscretization`](@ref) on a constrained ghost space:
the source at `t` is `Eᵀ (F_pad(t) − A_pad d(t))`, `d(t)` written into the padded buffer `d`
(unused on a space whose data do not depend on time, whose `d` is read directly).
"""
struct GhostConstraints{DV <: AbstractVector}
    d::DV
end

# The padded state a ghost form's coefficients read: `lift(Cₕ, u, t)`, refilled in place.
struct _GhostState{V <: VectorElement, W <: GhostGridSpace}
    v::V
    space::W
end

function _sync_state!(s::_GhostState, u, t)
    S = promote_type(eltype(u), _affine_eltype(s.space, t))
    promote_type(eltype(s.v), S) === eltype(s.v) || _throw_ghost_state_eltype(s.v, u, S)
    lift!(s.v, s.space, u, t)
    return nothing
end

@inline _state_element(s::_GhostState) = s.v
@inline _twin_state(s::_GhostState, v::VectorElement) = _GhostState(v, s.space)

# A ghost form assembles without the `dirichlet` keywords: its relations hold them.
_twin_matrix(sd, ::Val{true}, a::GhostBilinearForm) = assemble(a)

# The residual sends a non-plain `u` the state cannot hold to a twin (`_state_holds`), so
# a time, or a wider plain `u`, reaches this; `jacobian!` lifts any `u` into the state.
@noinline function _throw_ghost_state_eltype(v, u, S::Type)
    if _holds(eltype(v), eltype(u))
        by = "a ForwardDiff time: give the Rosenbrock solver an explicit tgrad, or use a " *
             "solver that does not differentiate in t (FBDF)."
    else
        by = "a u of element type $(nameof(eltype(u))): pass u at the state's type, " *
             "or, for a ForwardDiff Dual, evaluate the residual, which gives it a state " *
             "of its own type."
    end
    throw(
        ArgumentError(
        "the state of a semidiscretization on a ghost space receives lift(Cₕ, u, t), of " *
        "element type $(nameof(S)) here, and it holds $(eltype(v)). It cannot follow $by",
    ),
    )
end

"""
    semidiscretize(a::GhostBilinearForm, l::GhostLinearForm; mass, dirichlet,
        dirichlet_components, state, update_coefficients!, reassemble) -> Semidiscretization

Semidiscretise `M u' = F(t) − A u` for forms posed on one constrained ghost space `Cₕ`,
whose relation data may depend on time. With `(E, d(t))` the closure of `Cₕ`, `A = Eᵀ A_pad E`,
`M = Eᵀ M_pad E` and `F(t) = Eᵀ (F_pad(t) − A_pad d(t))`, so at a fixed `t` the residual is
`F − A u` of `assemble(a, l)` on the space rebuilt with the data frozen at `t`.

# Keywords
- `mass`: a form on `Cₕ` defining `M` (default `innerₕ(u, v)`, diagonal and positive). A mass that couples a degree of freedom to a point where `d` depends on time is refused: its term `Eᵀ M_pad d′(t)` needs the time derivative of the data.
- `state`: a grid function on `Bramble.padded_space(Cₕ)` that receives `lift(Cₕ, u, t)` before each assembly, for forms whose coefficients read it (default `nothing`). A `u` it cannot hold (a `ForwardDiff.Dual`) is lifted into a padded buffer of its own type, read by the forms rebound to it, as on a plain space.
- `update_coefficients!`, `reassemble`: as for plain forms.

The `dirichlet` and `dirichlet_components` keywords are refused: give a Dirichlet condition as `:F => u => g(x, t)` in
[`constrained`](@ref). So are a space that keeps ghosts as unknowns (their mass rows
vanish), `semidiscretize(build, l)` with a ghost load (use `update_coefficients!` and
`reassemble = true`), and [`semidiscretize_rhs`](@ref).
"""
Base.@constprop :aggressive function semidiscretize(
        a::GhostBilinearForm,
        l::GhostLinearForm;
        mass = nothing,
        dirichlet = nothing,
        dirichlet_components = nothing,
        state = nothing,
        (update_coefficients!) = nothing,
        reassemble::Bool = false
)
    (dirichlet === nothing && dirichlet_components === nothing) ||
        _throw_ghost_semidiscrete_dirichlet()
    Cₕ = l.test_space
    _check_ghost_semidiscrete_space(a, Cₕ, "a")
    _check_ghost_unknowns(Cₕ)

    A = assemble(a)
    mform = mass === nothing ? _default_mass_form(Cₕ) : mass
    mform isa GhostBilinearForm ||
        _throw_ghost_semidiscrete_mixed(("a", a), ("l", l), ("mass", mform))
    _check_ghost_semidiscrete_space(mform, Cₕ, "mass")
    M = assemble(mform)
    _check_ghost_mass(mform.A_pad, Cₕ)

    # Of `A`'s element type, as on a plain space: a Float32 space with a Float64 datum in
    # the load steps at Float32.
    F = similar(l.F_pad, eltype(A), ndofs(Cₕ))
    constraints = GhostConstraints(similar(Cₕ.d))

    return Semidiscretization(
        a,
        l,
        Cₕ,
        A,
        M,
        F,
        constraints,
        (),
        nothing,
        _ghost_state(state, Cₕ),
        _wrap_update(update_coefficients!),
        Val(reassemble),
        Dict{DataType, Any}()
    )
end

# --- The source --------------------------------------------------------------------- #

# `Eᵀ (F_pad(t) − A_pad d(t))`. `d(t)` goes into the carrier's buffer through the space's
# stored data buffer, or, for a time whose type they cannot hold, into buffers of that type.
function _assemble_source!(F::AbstractVector, sd, c::GhostConstraints, p, t)
    return _assemble_ghost_load!(F, sd.operator, sd.source, _ghost_affine!(c, sd.space, t))
end

@inline _ghost_affine!(c::GhostConstraints, Cₕ::GhostGridSpace, t) = _ghost_affine!(c, Cₕ,
    Cₕ.td, t)
@inline _ghost_affine!(::GhostConstraints, Cₕ::GhostGridSpace, ::Nothing, t) = Cₕ.d
@inline function _ghost_affine!(c::GhostConstraints, Cₕ::GhostGridSpace, td, t)
    S = _affine_eltype(Cₕ, t)
    S === eltype(c.d) && return _lift_affine!(c.d, Cₕ, t)
    return _lift_affine!(similar(c.d, S), Cₕ, t)
end

# Relations eliminate every algebraic row, so the initial condition needs no correction.
@inline _apply_initial_constraints!(u::AbstractVector, _, ::GhostConstraints, p, t) = u

@inline _constraint_description(::GhostConstraints) = "relations"

@inline function _reassemble_operator!(op::GhostBilinearForm, sd::Semidiscretization, x, t)
    return _assemble_ghost!(sd.operator_matrix, op)
end

# --- Checks at construction --------------------------------------------------------- #

@inline _ghost_state(::Nothing, ::GhostGridSpace) = nothing
function _ghost_state(state, Cₕ::GhostGridSpace)
    state isa VectorElement && state.space === padded_space(Cₕ) ||
        _throw_ghost_state_space(state)
    return _GhostState(state, Cₕ)
end

@noinline function _throw_ghost_state_space(state)
    on = state isa VectorElement ? state.space : typeof(state)
    throw(
        ArgumentError(
        "the state of a semidiscretization on a ghost space Cₕ receives lift(Cₕ, u, t), " *
        "a grid function on the padded space, got a $(typeof(state)) on $on: build it with " *
        "element(Bramble.padded_space(Cₕ)) and read that in the form's coefficients.",
    ),
    )
end

# Both sides of `a` (and of the mass) are the space of the load: the residual lifts `u` with
# the trial closure and restricts with the test closure, so another closure would be wrong.
function _check_ghost_semidiscrete_space(a::GhostBilinearForm, Cₕ::GhostGridSpace, name)
    a.trial_space === Cₕ && a.test_space === Cₕ || _throw_ghost_semidiscrete_space(name)
    return nothing
end

@noinline function _throw_ghost_semidiscrete_space(name)
    throw(
        ArgumentError(
        "semidiscretize: `$name` and `l` must be posed on one ghost space: the same " *
        "constrained space as trial and test space of `$name` and as test space of `l`.",
    ),
    )
end

# A degree of freedom whose column of `E` reads no physical point is a ghost left unknown:
# the mass weighs physical points only, so its row of `M` vanishes.
function _check_ghost_unknowns(Cₕ::GhostGridSpace)
    G = mesh(Cₕ)
    plin = LinearIndices(npoints(G, Tuple))
    physical = falses(length(plin))
    for J in physical_indices(G)
        physical[plin[J]] = true
    end
    E = Cₕ.E
    rv = rowvals(E)
    for r in axes(E, 2)
        any(k -> physical[rv[k]], nzrange(E, r)) || _throw_ghost_unknowns(Cₕ)
    end
    return nothing
end

@noinline function _throw_ghost_unknowns(Cₕ::GhostGridSpace)
    throw(
        ArgumentError(
        "semidiscretize: $(Cₕ) keeps ghosts as unknowns, whose rows of the mass matrix " *
        "vanish. Eliminate every ghost by a relation in constrained, giving a Dirichlet " *
        "condition as `:F => u => g(x, t)`.",
    ),
    )
end

# The padded points where `d` depends on time: those whose column of the stored `Lᵀ` reads
# a datum of a group (a datum in time) rather than a static value.
_time_points(::GhostGridSpace, ::Nothing) = falses(0)
function _time_points(Cₕ::GhostGridSpace, td)
    timed = falses(length(td.g0))
    for grp in td.groups
        timed[grp.first:(grp.first + length(grp.points) - 1)] .= true
    end
    colptr, rowval = Array(td.colptr), Array(td.rowval)
    return BitVector([any(k -> timed[rowval[k]], colptr[i]:(colptr[i + 1] - 1))
                      for i in 1:(length(colptr) - 1)])
end

# `Eᵀ M_pad d′(t)` vanishes when no stored nonzero `M_pad[i, j]` has `j` a point in time
# and row `i` of `E` nonempty.
function _check_ghost_mass(M_pad::SparseMatrixCSC, Cₕ::GhostGridSpace)
    timed = _time_points(Cₕ, Cₕ.td)
    isempty(timed) && return nothing
    Et = Cₕ.Et
    rv, nz = rowvals(M_pad), nonzeros(M_pad)
    for j in axes(M_pad, 2)
        timed[j] || continue
        for k in nzrange(M_pad, j)
            iszero(nz[k]) || isempty(nzrange(Et, rv[k])) || _throw_ghost_mass()
        end
    end
    return nothing
end

@noinline function _throw_ghost_mass()
    throw(
        ArgumentError(
        "semidiscretize: the mass reads a point where the relation data depend on time, " *
        "so the system would need Eᵀ M d′(t), the time derivative of the data, which no " *
        "relation carries. Use a mass that weighs the physical nodes only, such as the " *
        "default innerₕ(u, v).",
    ),
    )
end

# --- Refused ------------------------------------------------------------------------ #

@noinline function _throw_ghost_semidiscrete_dirichlet()
    throw(
        ArgumentError(
        "semidiscretize on a ghost space takes no `dirichlet` keyword: give the Dirichlet " *
        "condition as a relation `:F => u => g(x, t)` in constrained, which eliminates " *
        "its nodes.",
    ),
    )
end

@noinline function _throw_ghost_semidiscrete_mixed(forms::Tuple...)
    given = join(("$name::$(nameof(typeof(f)))" for (name, f) in forms), ", ")
    throw(
        ArgumentError(
        "semidiscretize takes forms on ghost spaces on every side or on none, got " *
        "$given: build the operator, the load and the mass on the same constrained " *
        "ghost space.",
    ),
    )
end

semidiscretize(a::GhostBilinearForm, l::LinearForm; kwargs...) = _throw_ghost_semidiscrete_mixed(("a", a), ("l", l))
semidiscretize(a::BilinearForm, l::GhostLinearForm; kwargs...) = _throw_ghost_semidiscrete_mixed(("a", a), ("l", l))

function semidiscretize(::Function, ::GhostLinearForm; kwargs...)
    throw(
        ArgumentError(
        "semidiscretize(build, l) does not take a ghost-space load: build the operator " *
        "once as a form on the ghost space, whose coefficients update_coefficients! " *
        "refills at each t, and pass reassemble = true.",
    ),
    )
end

function semidiscretize_rhs(::Semidiscretization{<:GhostBilinearForm})
    throw(
        ArgumentError(
        "semidiscretize_rhs does not take a semidiscretization on a ghost space: pass it " *
        "to ode_problem(sd, u₀, I), which hands its diagonal mass matrix to the solver.",
    ),
    )
end
