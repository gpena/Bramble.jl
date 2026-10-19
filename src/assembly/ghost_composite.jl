# ghost_composite.jl: composite spaces whose components are ghost spaces (#781).
#
# `C₁ × C₂` and `C^Val(n)` of `GhostGridSpace`s give a flat `CompositeGridSpace` of them.
# Its degrees of freedom are the components' in order (`component_range`), its padded space
# is `padded(C₁) × padded(C₂)` and its closure `(blockdiag(E₁, E₂), vcat(d₁, d₂))`: each
# component is constrained by its own relations, and no relation couples two components (a
# traction condition does: gpena/Bramble.jl#793). A form on it is the form on the padded
# composite with that closure as its sides (`_CompositeClosure`), a
# `GhostCompositeBilinearForm` or `GhostCompositeLinearForm`, so the reduction and the load
# of ghost_assembly.jl assemble it unchanged.
#
# `constrained(C₁ × C₂, u -> (:xmin => ∂ₙ(u(1)) => g, :xmax => u(2) => h, …))` constrains
# the composite as a whole: each relation names its component through `u(i)`, and the rows
# of all of them are solved once on the padded composite (ghost_relations.jl), so the
# `ConstrainedCompositeSpace` it returns stores a closure `(E, d)` of its own, which its
# forms, `closure` and `lift` read. With relations of one component each it equals
# `constrained(C₁, r₁) × constrained(C₂, r₂)`. Beyond the refusals below, it refuses data
# depending on time, the `dirichlet` rows and `GhostVector`.
#
# The components of a form's spaces must ghost the same facets of meshes of one size: a term
# coupling `u(1)` and `v(2)` reads both at one padded index, which names different points on
# padded grids of different shapes or shifted by one node. Components ghosting different
# facets are refused, coupled or not.
#
# A mixed difference such as `D₋ₓ(D₊ᵧ(u(1)))` that reads an absent padded point (beyond a
# corner between two ghosted facets) is refused by the walk's absent-read check on a padded
# composite, on diagonal blocks too, where the scalar ghost form takes the inward corner
# rule. Mending it belongs to the walk (stencil_eval.jl, bilinear_traversal.jl), not to this
# file. Such a term assembles where it reads no absent point.
#
# The `dirichlet` keyword, `dirichlet_bc!` and `symmetrize!` replace rows of the components
# `dirichlet_components` selects (all by default), each of whose unknowns must be its
# physical nodes in mesh order, as on a scalar ghost space; `u => g` relations give
# Dirichlet conditions per component in `constrained`.
#
# Refused with an `ArgumentError`: a composite mixing ghost and plain components or nesting
# a composite with a ghost component (where a form or `u(i)` walks its leaves:
# `_leaf_spaces_offsets` below), forms between a composite and a scalar space, a rank-one
# term, `dirichlet` rows of a component placed differently on the trial and test sides,
# `semidiscretize` and `ode_problem` with a composite form on either side, `reaction`, the
# matrix-free and Kronecker operators and the `fdm_*`, Jacobi, Chebyshev and AMG entry
# points that take the form, and `Rₕ`/`avgₕ`. `gmg_preconditioner` refuses the form as a
# level form of no kind it takes.
#
# See also: `GhostGridSpace`, `CompositeGridSpace`, `_CompositeClosure`

# A flat composite of ghost spaces, the one kind of composite with a ghost component that
# has a closure.
const _GhostCompositeSpace = CompositeGridSpace{<:Any,
    <:Tuple{GhostGridSpace, Vararg{GhostGridSpace}}}

# Either kind of composite of ghost spaces a form takes as a side.
const _GhostCompositeSide = Union{_GhostCompositeSpace, ConstrainedCompositeSpace}

# The ghost spaces of a side, unconstrained for a `ConstrainedCompositeSpace`.
@inline _ghost_components(Vₕ::_GhostCompositeSpace) = Vₕ.spaces
@inline _ghost_components(Vₕ::ConstrainedCompositeSpace) = Vₕ.space.spaces

# --- Building the space -------------------------------------------------------------- #

"""
    ×(C₁::GhostGridSpace, C₂::GhostGridSpace) -> CompositeGridSpace

The composite of two ghost spaces, flat as for plain spaces: `C₁ × C₂ × C₃` has three
components. Its degrees of freedom are those of `C₁` then those of `C₂`, and its closure
is block diagonal, each component constrained by its own relations, so forms on it
assemble a vector problem with per-component relations (see [`GhostGridSpace`](@ref)).
`Bramble.padded_space` and `Bramble.closure` give the padded composite and the closure,
and [`lift`](@ref) the padded grid functions of a solution.

A ghost space and a plain one do not combine: the product refuses with an `ArgumentError`.
Forms couple the components only when they ghost the same facets of meshes of one size.
"""
@inline ×(X::GhostGridSpace, Y::GhostGridSpace) = CompositeGridSpace((X, Y))
@inline ×(X::_GhostCompositeSpace, Y::GhostGridSpace) = CompositeGridSpace((X.spaces..., Y))
@inline ×(X::GhostGridSpace, Y::_GhostCompositeSpace) = CompositeGridSpace((X, Y.spaces...))
@inline ×(X::_GhostCompositeSpace, Y::_GhostCompositeSpace) = CompositeGridSpace((
    X.spaces..., Y.spaces...))

×(::GhostGridSpace, ::Union{ScalarGridSpace, CompositeGridSpace}) = _throw_ghost_composite()
×(::Union{ScalarGridSpace, CompositeGridSpace}, ::GhostGridSpace) = _throw_ghost_composite()
×(::_GhostCompositeSpace, ::ScalarGridSpace) = _throw_ghost_composite()
×(::ScalarGridSpace, ::_GhostCompositeSpace) = _throw_ghost_composite()
×(::_GhostCompositeSpace, ::CompositeGridSpace) = _throw_ghost_composite()
×(::CompositeGridSpace, ::_GhostCompositeSpace) = _throw_ghost_composite()

"""
    ^(Cₕ::GhostGridSpace, ::Val{N}) where N -> CompositeGridSpace{N}
    ^(Cₕ::GhostGridSpace, N::Int) -> CompositeGridSpace{N}

The composite of `N` copies of the ghost space `Cₕ`, as `Wₕ^N` of a plain space: `Cₕ^1` is
`Cₕ`, and the `Int` spelling takes `1 <= N <= 3`.
"""
@inline Base.:^(Cₕ::GhostGridSpace, ::Val{N}) where {N} = CompositeGridSpace(ntuple(_ -> Cₕ,
    Val(N)))
@inline Base.:^(Cₕ::GhostGridSpace, ::Val{1}) = Cₕ

Base.@constprop :aggressive function Base.:^(Cₕ::GhostGridSpace, N::Int)
    N == 1 && return Cₕ
    N == 2 && return Cₕ^Val(2)
    N == 3 && return Cₕ^Val(3)
    throw(
        ArgumentError(
        "Power N must satisfy 1 <= N <= 3 for Cₕ^N; got $N. Use Cₕ^Val(N) for N > 3."
    ),
    )
end

# The components and their offsets, as for a plain composite. A ghost space anywhere else,
# beside a plain one or inside a nested composite, is refused by the walk.
@inline leaf_spaces_offsets(Vₕ::_GhostCompositeSpace) = first(_ghost_leaves(Vₕ.spaces, 0))
@inline _ghost_leaves(::Tuple{}, offset::Int) = ((), offset)
@inline function _ghost_leaves(spaces::Tuple, offset::Int)
    C = first(spaces)
    tail, final = _ghost_leaves(Base.tail(spaces), offset + ndofs(C))
    return (((C, offset), tail...), final)
end

@inline first_space(Vₕ::_GhostCompositeSpace) = first(Vₕ.spaces)

_leaf_spaces_offsets(::GhostGridSpace, ::Int) = _throw_ghost_composite()
first_space(::GhostGridSpace) = _throw_ghost_composite()

"""
    padded_space(Vₕ::CompositeGridSpace) -> CompositeGridSpace

The padded space of a composite of ghost spaces: the composite of its components' padded
spaces, the space its forms assemble on and its [`lift`](@ref)s live on.
"""
@inline padded_space(Vₕ::_GhostCompositeSpace) = CompositeGridSpace(map(padded_space,
    Vₕ.spaces))

"""
    closure(Vₕ::CompositeGridSpace) -> (E::SparseMatrixCSC, d::Vector)
    closure(Vₕ::ConstrainedCompositeSpace) -> (E::SparseMatrixCSC, d::Vector)

The closure of a composite of ghost spaces: `E = blockdiag(E₁, E₂, …)` and
`d = vcat(d₁, d₂, …)` from the components' closures, so the padded values of its degrees of
freedom `u` are `E * u + d`, component by component. It is built on each call. On a
composite constrained as a whole, `constrained(C₁ × C₂, relations)`, it is the closure
`constrained` stored, whose `E` couples the components wherever a relation does.
"""
function closure(Vₕ::_GhostCompositeSpace)
    return (SparseArrays.blockdiag(map(C -> sparse(C.E), Vₕ.spaces)...), vcat(map(C -> C.d,
        Vₕ.spaces)...))
end
@inline closure(Vₕ::ConstrainedCompositeSpace) = (Vₕ.E, Vₕ.d)

# --- The closure as a form's side ---------------------------------------------------- #

function _CompositeClosure(Vₕ::_GhostCompositeSpace)
    foreach(_check_ghost_host, Vₕ.spaces)
    E, d = closure(Vₕ)
    return _CompositeClosure(Vₕ, padded_space(Vₕ), E, d, _closure_transpose(E))
end
function _CompositeClosure(Vₕ::ConstrainedCompositeSpace)
    foreach(_check_ghost_host, _ghost_components(Vₕ))
    return _CompositeClosure(Vₕ, Vₕ.padded, Vₕ.E, Vₕ.d, Vₕ.Et)
end

@inline padded_space(W::_CompositeClosure) = W.padded
@inline backend(W::_CompositeClosure) = backend(W.padded)
@inline ndofs(W::_CompositeClosure) = size(W.E, 2)
@inline _ghost_policy(W::_CompositeClosure) = execution_policy(W.padded)
@inline _is_time_dependent(W::_CompositeClosure) = _is_time_dependent(W.space)
@inline _is_time_dependent(Vₕ::_GhostCompositeSpace) = any(_is_time_dependent, Vₕ.spaces)
@inline _is_time_dependent(::ConstrainedCompositeSpace) = false   # refused by `constrained`
_affine_support(W::_CompositeClosure) = _affine_support(W.space)
_affine_support(Vₕ::_GhostCompositeSpace) = vcat(map(_affine_support, Vₕ.spaces)...)
_affine_support(Vₕ::ConstrainedCompositeSpace) = map(!iszero, Array(Vₕ.d))
Base.show(io::IO, W::_CompositeClosure) = show(io, W.space)

trial_space(a::GhostCompositeBilinearForm) = a.trial_space.space
test_space(a::GhostCompositeBilinearForm) = a.test_space.space
test_space(l::GhostCompositeLinearForm) = l.test_space.space

function Base.show(io::IO, a::GhostCompositeBilinearForm{D}) where {D}
    print(io, "GhostCompositeBilinearForm{$(D)D}(", a.trial_space, " → ", a.test_space, ")")
    return nothing
end

function Base.show(io::IO, l::GhostCompositeLinearForm{D}) where {D}
    print(io, "GhostCompositeLinearForm{$(D)D}(", l.test_space, ")")
    return nothing
end

# --- Forms --------------------------------------------------------------------------- #

"""
    form(Wₕ::CompositeGridSpace, Vₕ::CompositeGridSpace, f) -> GhostCompositeBilinearForm
    form(Vₕ::CompositeGridSpace, f) -> GhostCompositeLinearForm

Forms on composites of ghost spaces (`C₁ × C₂`, `C^Val(n)`): the form on their padded
spaces, reduced through the block-diagonal closure. `f` addresses the components as on a
plain composite, `u(1)`, `v(2)`. `assemble(a)`, `assemble(a, l)`, `assemble!` and their
`dirichlet` and `dirichlet_components` keywords act as on a scalar ghost space, the rows of
each selected component replaced through its physical mesh. `a(u, v)`, `l(v)` and
`evaluate!` give `vᴴ A u` and `Fᵀ v` as on a scalar ghost space.

Every component of both spaces must ghost the same facets of meshes of one size; an
`ArgumentError` names the fix otherwise. A rank-one term, a scalar space on the other side,
`semidiscretize`, and the matrix-free, Kronecker and multigrid operators are refused with
an `ArgumentError`.
"""
function form(Wₕ::_GhostCompositeSide, Vₕ::_GhostCompositeSide, f)
    _check_composite_ghosts(_ghost_components(Wₕ)..., _ghost_components(Vₕ)...)
    W, V = _CompositeClosure(Wₕ), _CompositeClosure(Vₕ)
    a = form(W.padded, V.padded, f)
    a isa RankOneForm && _throw_ghost_composite_unsupported("a rank-one term")
    return _ghost_bilinear_form(W, V, a)
end

function form(Vₕ::_GhostCompositeSide, f)
    _check_composite_ghosts(_ghost_components(Vₕ)...)
    V = _CompositeClosure(Vₕ)
    return _ghost_linear_form(V, form(V.padded, f))
end

# `vᴴ A u` and `Fᵀ v` without assembling, as on a scalar ghost space (ghost_contract.jl,
# which defines `_ClosureApplied`): the closures' linear parts lift `u` and `v` lazily.
function (a::GhostCompositeBilinearForm)(u, v)
    ud, vd = _mf_data(u), _mf_data(v)
    Wₕ, Vₕ = a.trial_space, a.test_space
    (length(ud) == ndofs(Wₕ) && length(vd) == ndofs(Vₕ)) ||
        _throw_composite_contraction_length(Wₕ, Vₕ, ud, vd)
    Base.require_one_based_indexing(ud, vd)
    return a.padded(_ClosureApplied{false}(Wₕ.Et, ud), _ClosureApplied{true}(Vₕ.Et, vd))
end

function (l::GhostCompositeLinearForm)(vₕ::VectorElement)
    _is_time_dependent(l.test_space) && _throw_ghost_time("l(v)")
    any(!iszero, l.test_space.d) && _throw_ghost_lift()
    vd = parent(vₕ)
    E = l.test_space.E
    length(vd) == size(E, 2) || _throw_composite_contraction_length(l.test_space,
        l.test_space, vd, vd)
    assemble!(l.F_pad, l.padded)
    T = promote_type(eltype(l.F_pad), eltype(E), eltype(vd))
    return _closure_contract(zero(T), SparseArrays.getcolptr(E), rowvals(E), nonzeros(E),
        l.F_pad, vd)
end

@noinline function (l::GhostCompositeLinearForm)(::AbstractVector)
    throw(
        ArgumentError(
        "a linear form contracts against an element of its test space, not a bare vector. " *
        "Name the space first, with l(element(test_space(l), v)).",
    ),
    )
end

# The load `assemble(l)` written into `scratch`, and its contraction with `vₕ`: `l(vₕ)`.
function evaluate!(scratch::AbstractVector, l::GhostCompositeLinearForm,
        vₕ::VectorElement)
    vd = parent(vₕ)
    length(vd) == ndofs(l.test_space) ||
        _throw_composite_contraction_length(l.test_space, l.test_space, vd, vd)
    assemble!(scratch, l)
    return _contract_values(scratch, vd)
end

@noinline function _throw_composite_contraction_length(Wₕ, Vₕ, ud, vd)
    throw(
        DimensionMismatch(
        "a form on composites of ghost spaces with $(ndofs(Vₕ))×$(ndofs(Wₕ)) degrees of " *
        "freedom cannot contract vectors of lengths $(length(ud)) and $(length(vd))",
    ),
    )
end

form(::_GhostCompositeSide, Vₕ, f) = _throw_ghost_composite_side()
form(Wₕ, ::_GhostCompositeSide, f) = _throw_ghost_composite_side()
form(::_GhostCompositeSide, ::GhostGridSpace, f) = _throw_ghost_composite_side()
form(::GhostGridSpace, ::_GhostCompositeSide, f) = _throw_ghost_composite_side()

@noinline function _throw_ghost_composite_side()
    throw(
        ArgumentError(
        "a form on a composite of ghost spaces takes such a composite on both sides: the " *
        "other side is a scalar or plain space. Build both as products of ghost spaces, " *
        "C₁ × C₂, of the same mesh and ghosts.",
    ),
    )
end

# One padded index names one point in every component: the same ghosted facets of meshes of
# one size.
function _check_composite_ghosts(C::GhostGridSpace, rest::GhostGridSpace...)
    G = mesh(C)
    for Cᵢ in rest
        Gᵢ = mesh(Cᵢ)
        Gᵢ === G || (Gᵢ.active == G.active && npoints(Gᵢ, Tuple) == npoints(G, Tuple)) ||
            _throw_composite_ghost_sets(G, Gᵢ)
    end
    return nothing
end

@noinline function _throw_composite_ghost_sets(G::GhostMesh, Gᵢ::GhostMesh)
    throw(
        ArgumentError(
        "the components of these composite ghost spaces ghost different facets " *
        "($(G.active) and $(Gᵢ.active) as per-axis (min, max) flags) or lie on meshes of " *
        "different sizes, so one padded index names different points in each. Build every " *
        "component from one mesh with the same `ghosts`, gridspace(Ωₕ; ghosts = ...), and " *
        "give each component its own relations on those facets in constrained.",
    ),
    )
end

@noinline function _throw_ghost_composite_unsupported(what::String)
    throw(
        ArgumentError(
        "$what on a composite of ghost spaces is not supported yet. Assemble the system, " *
        "A, F = assemble(a, l), and solve it with pde_solve(A, F) or a preconditioner " *
        "built from A, or pose the problem on each scalar ghost space.",
    ),
    )
end

function matrix_free_operator(::GhostCompositeBilinearForm; kwargs...)
    return _throw_ghost_composite_unsupported("matrix_free_operator")
end
function kronecker_operator(::GhostCompositeBilinearForm)
    return _throw_ghost_composite_unsupported("kronecker_operator")
end
# The solvers and preconditioners that take the form read it through those operators, or
# assemble it for one (amg); their functions are declared in solvers/, included later.
function fdm_solve(::GhostCompositeBilinearForm, ::AbstractVector; kwargs...)
    return _throw_ghost_composite_unsupported("fdm_solve")
end
function fdm_factorize(::GhostCompositeBilinearForm; kwargs...)
    return _throw_ghost_composite_unsupported("fdm_factorize")
end
function fdm_preconditioner(::GhostCompositeBilinearForm; kwargs...)
    return _throw_ghost_composite_unsupported("fdm_preconditioner")
end
function chebyshev_preconditioner(::GhostCompositeBilinearForm; kwargs...)
    return _throw_ghost_composite_unsupported("chebyshev_preconditioner")
end
function amg_preconditioner(::GhostCompositeBilinearForm; kwargs...)
    return _throw_ghost_composite_unsupported("amg_preconditioner")
end
function jacobi_preconditioner(::GhostCompositeBilinearForm; kwargs...)
    return _throw_ghost_composite_unsupported("jacobi_preconditioner")
end
# With any other operator, load or `build` beside it: a composite form on either side.
function semidiscretize(::GhostCompositeBilinearForm, ::GhostCompositeLinearForm; kwargs...)
    return _throw_ghost_composite_unsupported("semidiscretize")
end
function semidiscretize(::GhostCompositeBilinearForm, l; kwargs...)
    return _throw_ghost_composite_unsupported("semidiscretize")
end
function semidiscretize(a, ::GhostCompositeLinearForm; kwargs...)
    return _throw_ghost_composite_unsupported("semidiscretize")
end
function ode_problem(::GhostCompositeBilinearForm, ::GhostCompositeLinearForm, u₀, I;
        kwargs...)
    return _throw_ghost_composite_unsupported("ode_problem")
end
function ode_problem(::GhostCompositeBilinearForm, l, u₀, I; kwargs...)
    return _throw_ghost_composite_unsupported("ode_problem")
end
function ode_problem(a, ::GhostCompositeLinearForm, u₀, I; kwargs...)
    return _throw_ghost_composite_unsupported("ode_problem")
end

# A reaction reads the plain rows the relations replaced, as on a scalar ghost space.
function reaction(::GhostCompositeBilinearForm, ::GhostCompositeLinearForm, ::VectorElement;
        marker, dirichlet_components = nothing)
    return _throw_ghost_reaction()  # ghost_contract.jl
end

# `Rₕ` and `avgₕ` refuse as on a ghost component. Without these methods a composite whose
# components share one ghost mesh would take the plain scatter over its padded mesh.
function project!(uₕ::VectorElement{<:_GhostCompositeSpace}, ::ProjectionRule,
        ::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {N}
    return _throw_composite_projection(space(uₕ))
end
function project!(uₕ::VectorElement{<:_GhostCompositeSpace}, ::Tuple,
        ::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {N}
    return _throw_composite_projection(space(uₕ))
end

# A composite constrained as a whole is no `CompositeGridSpace`: it refuses at the entry
# points, as a ghost space does (ghost_gridspace.jl).
Rₕ(Vₕ::ConstrainedCompositeSpace, f; kwargs...) = _throw_composite_projection(Vₕ)
function Rₕ!(uₕ::VectorElement{<:ConstrainedCompositeSpace}, f; kwargs...)
    return _throw_composite_projection(uₕ.space)
end
avgₕ(Vₕ::ConstrainedCompositeSpace, f; kwargs...) = _throw_composite_projection(Vₕ)
function avgₕ!(uₕ::VectorElement{<:ConstrainedCompositeSpace}, f, rest...; kwargs...)
    return _throw_composite_projection(uₕ.space)
end

@noinline function _throw_composite_projection(Vₕ::_GhostCompositeSide)
    throw(
        ArgumentError(
        "Rₕ or avgₕ is not defined on a composite of ghost spaces, $(Vₕ), as on each " *
        "ghost space: what a restriction holds at the ghost points is not decided yet. " *
        "Apply it on the plain composite of gridspace(Ωₕ) instead, or lift a solution " *
        "with Bramble.lift(Vₕ, u).",
    ),
    )
end

# --- Dirichlet by row replacement, per component ------------------------------------- #

function _check_rows(a::GhostCompositeBilinearForm, labels, components)
    _check_whole_rows(test_space(a), labels)
    _check_whole_rows(trial_space(a), labels)
    return _check_component_rows(_row_leaves(test_space(a)), _row_leaves(trial_space(a)),
        labels, components)
end
function _check_rows(l::GhostCompositeLinearForm, labels, components)
    _check_whole_rows(test_space(l), labels)
    return _check_component_rows(_row_leaves(test_space(l)), nothing, labels, components)
end

# A composite constrained as a whole has no rows of a component to replace: its unknowns
# are not split into components' closures. With no labels there is nothing to replace, so
# its leaves are none.
@inline _row_leaves(Vₕ::_GhostCompositeSpace) = leaf_spaces_offsets(Vₕ)
@inline _row_leaves(::ConstrainedCompositeSpace) = ()
@inline _check_whole_rows(::_GhostCompositeSpace, labels) = nothing
@inline _check_whole_rows(::ConstrainedCompositeSpace, ::Nothing) = nothing
@inline function _check_whole_rows(::ConstrainedCompositeSpace, labels::Tuple)
    isempty(labels) || _throw_whole_rows()
    return nothing
end

@noinline function _throw_whole_rows()
    throw(
        ArgumentError(
        "`dirichlet`, `dirichlet_bc!` and `symmetrize!` replace rows by facet label, " *
        "and a composite constrained as a whole, constrained(C₁ × C₂, relations), does " *
        "not take them. Give each Dirichlet condition as a relation of its component in " *
        "constrained, `:F => u(i) => g`, and assemble with assemble(a, l).",
    ),
    )
end

# Each selected component, on the test side and, at the same position, the trial side, must
# have its physical nodes as unknowns; `components` names the test side's. A row replaced by
# `dirichlet` gets its 1 in the column of its own index, so on a form the selected component
# must also start at the same degree of freedom and hold as many on both sides: otherwise
# that column is another unknown, or none. `trial` is `nothing` for a load or a space alone.
@inline _check_component_rows(test, trial, ::Nothing, components) = nothing
function _check_component_rows(test::Tuple, trial, labels::Tuple, components)
    isempty(labels) && return nothing
    _validate_dirichlet_components(components, length(test))
    _check_leaf_rows(test, labels, components)
    trial === nothing || _check_paired_rows(test, trial, labels, components)
    return nothing
end

@inline _check_leaf_rows(::Tuple{}, labels, components, i::Int = 1) = nothing
@inline function _check_leaf_rows(leaves::Tuple, labels, components, i::Int = 1)
    C = first(first(leaves))
    _leaf_selected(components, i) && !C.rows.physical && _throw_rows_dirichlet((C,), labels)
    return _check_leaf_rows(Base.tail(leaves), labels, components, i + 1)
end

@inline _check_paired_rows(::Tuple{}, ::Tuple, labels, components, i::Int = 1) = nothing
@inline function _check_paired_rows(test::Tuple, trial::Tuple, labels, components,
        i::Int = 1)
    if _leaf_selected(components, i)
        isempty(trial) && _throw_unpaired_rows(i, first(test), nothing)
        Cu, ou = first(trial)
        Cv, ov = first(test)
        Cu.rows.physical || _throw_rows_dirichlet((Cu,), labels)
        (ou == ov && ndofs(Cu) == ndofs(Cv)) || _throw_unpaired_rows(i, first(test),
            first(trial))
    end
    return _check_paired_rows(Base.tail(test), isempty(trial) ? trial : Base.tail(trial),
        labels, components, i + 1)
end

@noinline function _throw_unpaired_rows(i::Int, (Cv, ov), trial)
    what = trial === nothing ? "the trial space has no component $i" :
           "component $i holds degrees of freedom $(ov + 1):$(ov + ndofs(Cv)) of the " *
           "test space and $(trial[2] + 1):$(trial[2] + ndofs(trial[1])) of the trial space"
    throw(
        ArgumentError(
        "`dirichlet` replaces a row of component $i by the identity at the column of the " *
        "same index, and $what, so that column is not the row's own unknown. Give the " *
        "condition as a relation `u => g` of the component in constrained, or build the " *
        "components before it alike on both sides.",
    ),
    )
end

# The entries of `_dirichlet_bc_rows!` for one label: each component's facet mask on its
# physical mesh, at its offset, and whether it is selected.
@inline _component_entries(::Tuple{}, label::Symbol, components, i::Int = 1) = ()
@inline function _component_entries(leaves::Tuple, label::Symbol, components, i::Int = 1)
    C, offset = first(leaves)
    entry = (_label_mask(physical_mesh(mesh(C)), C.rows, label), offset, ndofs(C),
        _leaf_selected(components, i))
    return (entry, _component_entries(Base.tail(leaves), label, components, i + 1)...)
end

function _component_rows!(A::AbstractMatrix, leaves::Tuple, labels::Tuple, components)
    for lbl in labels
        _dirichlet_bc_rows!(A, _component_entries(leaves, lbl, components))
    end
    return A
end

function _component_conditions!(v::AbstractVector, leaves::Tuple, bcs, labels::Tuple,
        components)
    _check_labels_have_conditions(conditions(bcs), labels)
    _each_selected_leaf(leaves, components) do C, offset
        _rows_conditions!(conditions(bcs), v, physical_mesh(mesh(C)), C.rows, labels, offset)
    end
    return v
end

function _component_symmetrize!(A::AbstractMatrix, F::AbstractVector, leaves::Tuple,
        labels::Tuple, components)
    _check_symmetrize_square(A)
    _each_selected_leaf(leaves, components) do C, offset
        Ωₕ = physical_mesh(mesh(C))
        for lbl in labels
            symmetrize!(A, F, _label_mask(Ωₕ, C.rows, lbl), offset)
        end
    end
    return nothing
end

function _form_rows!(A::AbstractMatrix, a::GhostCompositeBilinearForm, labels::Tuple,
        components)
    return _component_rows!(A, _row_leaves(test_space(a)), labels, components)
end

function _form_rows!(b::AbstractVector, l::GhostCompositeLinearForm, labels::Tuple, bcs,
        components)
    isempty(labels) && return b
    bcs === nothing && _throw_labels_without_conditions(labels)
    return _component_conditions!(b, _row_leaves(test_space(l)), bcs, labels, components)
end

function _form_symmetrize!(A, F, a::GhostCompositeBilinearForm, labels::Tuple, components)
    return _component_symmetrize!(A, F, _row_leaves(test_space(a)), labels, components)
end

"""
    dirichlet_bc!(A::AbstractMatrix, Vₕ::CompositeGridSpace, labels::Symbol...; components = nothing)
    dirichlet_bc!(v::AbstractVector, Vₕ::CompositeGridSpace, bcs, labels::Symbol...; components = nothing)
    symmetrize!(A, F, Vₕ::CompositeGridSpace, labels::Symbol...; components = nothing)

Row replacement by hand on a composite of ghost spaces, as the `dirichlet` keyword of
`assemble` does it: the rows of the nodes of `labels` (markers or facet names of the
physical mesh) in each component `components` selects (all by default). Each selected
component's unknowns must be its physical nodes: every ghost eliminated by a relation and
no node fixed by `u => g`; an `ArgumentError` names the route otherwise.
"""
function dirichlet_bc!(A::AbstractMatrix, Vₕ::_GhostCompositeSpace, labels::Symbol...;
        components = nothing)
    leaves = leaf_spaces_offsets(Vₕ)
    _check_component_rows(leaves, nothing, labels, components)
    return _component_rows!(A, leaves, labels, components)
end

function dirichlet_bc!(v::AbstractVector, Vₕ::_GhostCompositeSpace,
        bcs::Union{ConstraintMarkers, EvaluatedParametricDomainMarkers}, labels::Symbol...;
        components = nothing)
    leaves = leaf_spaces_offsets(Vₕ)
    _check_component_rows(leaves, nothing, labels, components)
    return _component_conditions!(v, leaves, bcs, labels, components)
end

function symmetrize!(A::AbstractMatrix, F::AbstractVector, Vₕ::_GhostCompositeSpace,
        labels::Symbol...; components = nothing)
    leaves = leaf_spaces_offsets(Vₕ)
    _check_component_rows(leaves, nothing, labels, components)
    return _component_symmetrize!(A, F, leaves, labels, components)
end

function dirichlet_bc!(::AbstractMatrix, ::ConstrainedCompositeSpace, labels::Symbol...;
        components = nothing)
    return _throw_whole_rows()
end
function dirichlet_bc!(::AbstractVector, ::ConstrainedCompositeSpace,
        ::Union{ConstraintMarkers, EvaluatedParametricDomainMarkers}, labels::Symbol...;
        components = nothing)
    return _throw_whole_rows()
end
function symmetrize!(::AbstractMatrix, ::AbstractVector, ::ConstrainedCompositeSpace,
        labels::Symbol...; components = nothing)
    return _throw_whole_rows()
end

# No form on these spaces holds a rank-one term (`form` refuses it).
function symmetrize!(::RankOneUpdate, ::AbstractVector, ::_GhostCompositeSpace,
        labels::Symbol...; components = nothing)
    return _throw_ghost_composite_unsupported("a rank-one term")
end

# --- Lifting a solution -------------------------------------------------------------- #

# Each component with its offsets into the degrees of freedom and into the padded values.
@inline _padded_leaves(Vₕ::_GhostCompositeSpace) = _padded_leaves(Vₕ.spaces, 0, 0)
@inline _padded_leaves(::Tuple{}, ::Int, ::Int) = ()
@inline function _padded_leaves(spaces::Tuple, offset::Int, poffset::Int)
    C = first(spaces)
    return ((C, offset, poffset),
        _padded_leaves(Base.tail(spaces), offset + ndofs(C), poffset + length(C.d))...)
end

@inline _padded_length(Vₕ::_GhostCompositeSpace) = sum(C -> length(C.d), Vₕ.spaces)

"""
    lift(Vₕ::CompositeGridSpace) -> VectorElement
    lift(Vₕ::CompositeGridSpace, u::AbstractVector) -> VectorElement
    lift(Vₕ::CompositeGridSpace, u::AbstractVector, t::Number) -> VectorElement

[`lift`](@ref) on a composite of ghost spaces (`C₁ × C₂`): the grid function on
`Bramble.padded_space(Vₕ)` whose component `i` is `lift(Cᵢ, uᵢ)` (`d`, `E u + d` or
`E u + d(t)`) for the slice `uᵢ` of `u` at `component_range(Vₕ, i)`. Each component is
checked, and refused, as `lift` on it is. `Bramble.lift!(uₕ, Vₕ, u)` and
`Bramble.lift!(uₕ, Vₕ, u, t)` refill a grid function `lift` returned.

On a composite constrained as a whole, `constrained(C₁ × C₂, relations)`, they give
`d` and `E u + d` from its stored closure, whose `E` may couple the components; its data
do not depend on time, so `t` changes nothing.
"""
function lift(Vₕ::_GhostCompositeSpace)
    foreach(C -> (_check_lift_fresh(C); _check_lift_static(C, "lift(Vₕ)")), Vₕ.spaces)
    return VectorElement(vcat(map(C -> C.d, Vₕ.spaces)...), padded_space(Vₕ))
end

@noinline function lift(Vₕ::_GhostCompositeSide, u, rest...)
    throw(
        ArgumentError(
        "lift on a composite of ghost spaces takes lift(Vₕ), lift(Vₕ, u) or " *
        "lift(Vₕ, u, t), u a vector of its $(ndofs(Vₕ)) degrees of freedom and t a " *
        "number; got $(1 + length(rest)) arguments after $(Vₕ).",
    ),
    )
end

function lift(Vₕ::_GhostCompositeSpace, u::AbstractVector)
    foreach(C -> _check_lift_static(C, "lift(Vₕ, u)"), Vₕ.spaces)
    _check_composite_lift(Vₕ, u)
    T = promote_type(eltype(u), map(C -> eltype(C.d), Vₕ.spaces)...)
    U = similar(first(Vₕ.spaces).d, T, _padded_length(Vₕ))
    return VectorElement(_lift_components!(U, _padded_leaves(Vₕ), u), padded_space(Vₕ))
end

function lift(Vₕ::_GhostCompositeSpace, u::AbstractVector, t::Number)
    _check_composite_lift(Vₕ, u)
    T = promote_type(eltype(u), map(C -> _affine_eltype(C, t), Vₕ.spaces)...)
    U = similar(first(Vₕ.spaces).d, T, _padded_length(Vₕ))
    return VectorElement(_lift_components!(U, _padded_leaves(Vₕ), u, t, Val(true)),
        padded_space(Vₕ))
end

function lift!(uₕ::VectorElement, Vₕ::_GhostCompositeSpace, u::AbstractVector)
    foreach(C -> _check_lift_static(C, "lift!(uₕ, Vₕ, u)"), Vₕ.spaces)
    _check_composite_lift(Vₕ, u)
    _check_composite_lift_into(uₕ, Vₕ, promote_type(eltype(u),
        map(C -> eltype(C.d), Vₕ.spaces)...))
    _lift_components!(parent(uₕ), _padded_leaves(Vₕ), u)
    return uₕ
end

function lift!(uₕ::VectorElement, Vₕ::_GhostCompositeSpace, u::AbstractVector, t::Number)
    _check_composite_lift(Vₕ, u)
    _check_composite_lift_into(uₕ, Vₕ, promote_type(eltype(u),
        map(C -> _affine_eltype(C, t), Vₕ.spaces)...))
    _lift_components!(parent(uₕ), _padded_leaves(Vₕ), u, t, Val(false))
    return uₕ
end

function _check_composite_lift(Vₕ::_GhostCompositeSide, u::AbstractVector)
    foreach(_check_lift_fresh, _ghost_components(Vₕ))
    length(u) == ndofs(Vₕ) || _throw_composite_lift_length(Vₕ, u)
    return nothing
end

function _check_composite_lift_into(uₕ::VectorElement, Vₕ::_GhostCompositeSide,
        ::Type{T}) where {T}
    uₕ.space === padded_space(Vₕ) || _throw_composite_lift_target(Vₕ)
    promote_type(eltype(uₕ), T) === eltype(uₕ) || _throw_lift_eltype(uₕ, T)
    return nothing
end

# `U = E u + d` (or `d(t)`), component by component, on the slices of `U` and `u`; with a
# time, `own` gives each component a data buffer of its own (`lift`) instead of its stored
# one (`lift!`).
@inline _lift_components!(U, ::Tuple{}, u) = U
@inline function _lift_components!(U, leaves::Tuple, u)
    C, offset, poffset = first(leaves)
    _lift_values!(view(U, (poffset + 1):(poffset + length(C.d))), C,
        view(u, (offset + 1):(offset + ndofs(C))))
    return _lift_components!(U, Base.tail(leaves), u)
end

@inline _lift_components!(U, ::Tuple{}, u, t, ::Val) = U
@inline function _lift_components!(U, leaves::Tuple, u, t, ::Val{own}) where {own}
    C, offset, poffset = first(leaves)
    g = own ? _affine_buffer(C, _affine_eltype(C, t)) : _stored_buffer(C.td, t)
    _lift_values!(view(U, (poffset + 1):(poffset + length(C.d))), C,
        view(u, (offset + 1):(offset + ndofs(C))), t, g)
    return _lift_components!(U, Base.tail(leaves), u, t, Val(own))
end

@noinline function _throw_composite_lift_length(Vₕ::_GhostCompositeSide, u)
    throw(
        ArgumentError(
        "lift takes one value per degree of freedom of $(Vₕ), $(ndofs(Vₕ)) of them, the " *
        "components' in order; got $(length(u)). Pass the solution of a system assembled " *
        "on Vₕ.",
    ),
    )
end

@noinline function _throw_composite_lift_target(Vₕ::_GhostCompositeSide)
    throw(
        ArgumentError(
        "lift!(uₕ, Vₕ, u) writes into a grid function on the padded space of $(Vₕ), and " *
        "uₕ lives on another space. Build it once with uₕ = Bramble.lift(Vₕ, u).",
    ),
    )
end

# `U = E u + d` from the stored closure of a composite constrained as a whole, over the
# whole padded composite: a relation coupling two components makes a component's padded
# values read the other's degrees of freedom, so no slice of `u` lifts alone.
function lift(Vₕ::ConstrainedCompositeSpace)
    foreach(_check_lift_fresh, _ghost_components(Vₕ))
    return VectorElement(copy(Vₕ.d), Vₕ.padded)
end

function lift(Vₕ::ConstrainedCompositeSpace, u::AbstractVector)
    _check_composite_lift(Vₕ, u)
    U = similar(Vₕ.d, promote_type(eltype(u), eltype(Vₕ.d)))
    return VectorElement(_lift_whole!(U, Vₕ, u), Vₕ.padded)
end
lift(Vₕ::ConstrainedCompositeSpace, u::AbstractVector, ::Number) = lift(Vₕ, u)

function lift!(uₕ::VectorElement, Vₕ::ConstrainedCompositeSpace, u::AbstractVector)
    _check_composite_lift(Vₕ, u)
    _check_composite_lift_into(uₕ, Vₕ, promote_type(eltype(u), eltype(Vₕ.d)))
    _lift_whole!(parent(uₕ), Vₕ, u)
    return uₕ
end
function lift!(uₕ::VectorElement, Vₕ::ConstrainedCompositeSpace, u::AbstractVector,
        ::Number)
    return lift!(uₕ, Vₕ, u)
end

function _lift_whole!(U::AbstractVector, Vₕ::ConstrainedCompositeSpace, u::AbstractVector)
    Et = Vₕ.Et
    _run_bands!(execution_policy(Vₕ.padded), _lift_band!, U, Vₕ.d,
        SparseArrays.getcolptr(Et), rowvals(Et), nonzeros(Et), u)
    return U
end

# --- Constrained as a whole ---------------------------------------------------------- #

"""
    constrained(Vₕ::CompositeGridSpace, relations) -> ConstrainedCompositeSpace

A composite of ghost spaces (`C₁ × C₂`, each built by `gridspace(Ωₕ; ghosts = ...)` with
the same ghosts) constrained as a whole. `relations(u)` returns the pairs
`facet => operator => data` of [`constrained`](@ref) on a ghost space, each written of one
component `u(i)`: `:xmin => ∂ₙ(u(1)) => g` relates the ghosts of `:xmin` in component 1,
`:ymin => u(2) => 0.0` fixes the nodes of `:ymin` in component 2. A component has at most
one relation of each kind per facet.

The relations of all the components are solved together on the padded composite, and the
result stores the closure `(E, d)` they give (`Bramble.closure`). Its degrees of freedom are
component 1's, then component 2's, …, each in the order `constrained` gives on the
component alone, so with relations of one component each it equals
`constrained(C₁, r₁) × constrained(C₂, r₂)`. Forms on it, `assemble(a, l)` and
[`lift`](@ref) work as on that product.

Refused with an `ArgumentError`, besides what `constrained` and the product refuse: a
relation naming no component or several, data depending on time (constrain the components
apart and take the product), the `dirichlet` keyword of `assemble`, `dirichlet_bc!` and
`symmetrize!` (give `u(i) => g` instead), and `GhostVector` and `update_ghosts!`.

# Examples

```julia
Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 9), (false, false))
Sₕ = gridspace(Ωₕ; ghosts = (:xmin, :xmax))
Vₕ = constrained(Sₕ × Sₕ, u -> (:xmin => ∂ₙ(u(1)) => 1.0, :xmax => ∂ₙ(u(1)) => 0.0,
    :xmin => ∂ₙ(u(2)) => 0.0, :xmax => u(2) + ∂ₙ(u(2)) => 0.0))
E, d = Bramble.closure(Vₕ)
```
"""
function constrained(Vₕ::_GhostCompositeSpace, relations)
    foreach(_check_unconstrained, Vₕ.spaces)
    _check_composite_ghosts(Vₕ.spaces...)
    P = padded_space(Vₕ)
    comps = map(((C, _, poffset),) -> (mesh(C), poffset), _padded_leaves(Vₕ))
    rels = _composite_relations(map(first, comps), relations(trial_function(P)))
    E, d = _composite_closure(P, comps, filter(!_is_dirichlet, rels),
        filter(_is_dirichlet, rels))
    T, be = eltype(P), backend(P)
    Ec = _closure_matrix(be, T, E)
    return ConstrainedCompositeSpace(Vₕ, P, Ec, _closure_data(be, T, d),
        _closure_transpose(Ec))
end

@noinline function constrained(Vₕ::ConstrainedCompositeSpace, relations)
    throw(
        ArgumentError(
        "constrained takes the composite of gridspace(Ωₕ; ghosts = ...) spaces, and " *
        "$(Vₕ) is already constrained. Give every relation in one call on the " *
        "unconstrained composite.",
    ),
    )
end

@inline padded_space(Vₕ::ConstrainedCompositeSpace) = Vₕ.padded
@inline ndofs(Vₕ::ConstrainedCompositeSpace) = size(Vₕ.E, 2)
@inline backend(Vₕ::ConstrainedCompositeSpace) = backend(Vₕ.padded)
@inline eltype(Vₕ::ConstrainedCompositeSpace) = eltype(Vₕ.padded)
@inline dim(Vₕ::ConstrainedCompositeSpace) = dim(Vₕ.padded)

function Base.show(io::IO, Vₕ::ConstrainedCompositeSpace{N}) where {N}
    print(io, "ConstrainedCompositeSpace{$N components, ", ndofs(Vₕ), " dofs}")
    return nothing
end
Base.summary(Vₕ::ConstrainedCompositeSpace) = sprint(show, Vₕ)

# A `GhostVector` holds one scalar ghost space's values, which a closure coupling the
# components does not split into.
@noinline function GhostVector(Vₕ::ConstrainedCompositeSpace, u...)
    return _throw_whole_ghost_vector("GhostVector")
end
@noinline function update_ghosts!(v, Vₕ::ConstrainedCompositeSpace, rest...)
    return _throw_whole_ghost_vector("update_ghosts!")
end

@noinline function _throw_whole_ghost_vector(call::String)
    throw(
        ArgumentError(
        "$call takes a scalar ghost space, and a composite constrained as a whole, " *
        "constrained(C₁ × C₂, relations), has no ghost vector of its own. Lift a " *
        "solution with Bramble.lift(Vₕ, u), whose padded values hold every component's " *
        "ghosts.",
    ),
    )
end
