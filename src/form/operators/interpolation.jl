# interpolation.jl
# The symbolic counterpart of interpolate_at/πₕ!/πₕ (space/operators/interpolation.jl):
# wraps a grid function's interpolant as a SourceFunction, so it composes with the rest of
# the AST layer exactly the way any other source does. This is a second method of the same
# `πₕ` the numeric layer defines (dispatch tells them apart by arity), not a separate name.

"""
	πₕ(uₕ::VectorElement) -> LazyOp

The interpolant of `uₕ`, as a symbolic source term — usable anywhere a source is, including
inside another operator: `innerₕ(D₋ₓ(πₕ(uₕ)), D₋ₓ(v))` differentiates the interpolated field
the same way `D₋ₓ` differentiates any other source, `innerₕ(M₋ₓ(πₕ(uₕ)), v)` averages it,
and so on. This is what lets a coupled form read a leaf's own grid function on a *different*
leaf's mesh — the case point 24 unlocked and point 25 exists to use.

Built as `source_function(x -> interpolate_at(uₕ, x), Val(D))`: a `SourceFunction`'s own
`local_stencil` already evaluates its function at the *current* point of whichever mesh is
being walked, so nothing else needs to know `uₕ` came from another leaf at all — the
interpolation happens once per point, inside `interpolate_at`, exactly where any other
source's function call would happen.
"""
function πₕ(uₕ::VectorElement{<:ScalarGridSpace{D}}) where {D}
    return source_function(x -> interpolate_at(uₕ, x), Val(D))
end

#===========================================================================#
# The interpolation *operator*: πₕ over a trial function (point 61).
#
# The source wrapper above needs concrete nodal values, so it cannot take a trial function —
# there is nothing to blend. What a bilinear form wants instead is the operator itself: for
# each point of the *test* mesh, the `2ᴰ` trial degrees of freedom of the cell containing it,
# with the corner weights. That is exactly one row of `interpolation_matrix`, produced a row
# at a time inside the assembly sweep rather than as a separate matrix — which is what keeps
# `assemble!` refilling at zero bytes, the thing a matrix product could not do.
#
# The entries name **absolute trial columns**, not offsets. Every other node's stencil says
# "this many points from here, on the mesh being walked"; an interpolation says "these dofs
# of the *other* mesh", and which ones depends on where the point falls, via `locate_cell`.
# `AbsoluteColumn` marks the difference so the three bilinear consumers can resolve each kind
# by dispatch rather than by a runtime flag.
#===========================================================================#

"""
    InterpolationNode{D, S, OpType} <: LazyOp{D}

The symbolic interpolation operator: `inner_op` lives on `src_space`, and this node evaluates
it at points of whatever mesh the assembly is walking.

Distinct from the source wrapper `πₕ(uₕ)`, which carries a grid function's values. This one
carries no values at all — it carries the *map*, and its stencil names trial columns.
"""
struct InterpolationNode{D, S, OpType <: LazyOp{D}} <: LazyOp{D}
    src_space::S
    inner_op::OpType
end

@noinline function _throw_interp_inner(op)
    throw(ArgumentError(
        "πₕ as a bilinear operator wraps a trial function directly — `πₕ(Wsrc, u)` or " *
        "`πₕ(Wsrc, u(2))` — and got $(typeof(op)). An operator applied *before* the " *
        "interpolation (`πₕ(Wsrc, D₋ₓ(u))`, differencing on the source mesh and then " *
        "interpolating) is a different operator and is not implemented; write the operator " *
        "outside instead, `D₋ₓ(πₕ(Wsrc, u))`, which differences on the mesh being " *
        "integrated over."))
end

"""
    πₕ(Wsrc::ScalarGridSpace{D}, op::LazyOp{D}) -> InterpolationNode

The interpolation operator from `Wsrc` onto whichever mesh the form integrates over, applied
to the trial function `op`.

This is the bilinear counterpart of `πₕ(uₕ)`: that one interpolates a grid function whose
values are already known, and belongs on the source side of a linear form; this one
interpolates the *unknown*, and so contributes matrix columns. `innerₕ(πₕ(Wsrc, u), v)`
assembles `Hᵥ · P`, with `P` the same matrix [`interpolation_matrix`](@ref) builds — computed
a row at a time during the sweep rather than as a matrix product, which is what lets
`assemble!` refill it allocating nothing.

Operators wrap it from the outside, acting on the mesh being integrated over:
`inner₊(D₋ₓ(πₕ(Wsrc, u)), D₋ₓ(v))` is ``D_x^\\top H_+ D_x P``. Writing an operator *inside*
is a different thing and is refused, since it would difference on the source mesh instead.

`op` must be a trial-function leaf, plain or indexed.
"""
function πₕ(Wsrc::ScalarGridSpace{D}, op::LazyOp{D}) where {D}
    op isa TrialFunction || op isa IndexedTrialFunction || _throw_interp_inner(op)
    return InterpolationNode{D, typeof(Wsrc), typeof(op)}(Wsrc, op)
end

# --- The stencil: absolute trial columns, with the corner weights ------------------- #

@inline function local_stencil(op::InterpolationNode{D}, space, I::CartesianIndex{D},
        markers, lin_idx::Int) where {D}
    return _interp_stencil(mesh(op.src_space), point(mesh(space), I), Val(D))
end

@inline function _interp_stencil(Ωsrc::AbstractMeshType{1}, x, ::Val{1})
    j, t = _interp_cell_frac(Ωsrc, x)
    return ((AbsoluteColumn(j), 1 - t), (AbsoluteColumn(j + 1), t))
end

@inline function _interp_stencil(Ωsrc::AbstractMeshType{D}, x, ::Val{D}) where {D}
    idx, ts = _interp_cell_frac(Ωsrc, x)
    li = LinearIndices(indices(Ωsrc))
    # the `2ᴰ` corners, decoded from the bits of `k - 1` so the tuple length is static
    return ntuple(Val(1 << D)) do k
        corner = CartesianIndex(ntuple(d -> ((k - 1) >> (d - 1)) & 1, Val(D)))
        (AbsoluteColumn(li[idx + corner]), _interp_corner_weight(ts, corner, Val(D)))
    end
end

# --- Traits: every walker that sees through a wrapper has to see through this one ---- #

is_symbolic(op::InterpolationNode) = is_symbolic(op.inner_op)

# It carries a trial function, so it is never a source however it is wrapped — which is what
# keeps `innerₕ` building a `BilinearProduct` for it (points 25, 60).
_is_source_only(::InterpolationNode) = false

function resolve_ast(op::InterpolationNode{D, S}) where {D, S}
    inner = resolve_ast(op.inner_op)
    return InterpolationNode{D, S, typeof(inner)}(op.src_space, inner)
end

trial_component_or_nothing(op::InterpolationNode) = trial_component_or_nothing(op.inner_op)
test_component_or_nothing(op::InterpolationNode) = test_component_or_nothing(op.inner_op)

@inline function component(op::InterpolationNode{D, S}, i::Int) where {D, S}
    inner = component(op.inner_op, i)
    return InterpolationNode{D, S, typeof(inner)}(op.src_space, inner)
end

_collect_region_labels(op::InterpolationNode) = _collect_region_labels(op.inner_op)

# The reach on the mesh being walked is the inner leaf's — a single point. The columns this
# node names are on the *other* mesh and are not offsets at all, so they have no place in an
# offset set; a matrix's colouring reads the evaluated stencil's test side instead
# (`_bilinear_colour_strides`), which is why that is sound here.
stencil_offsets(op::InterpolationNode) = stencil_offsets(op.inner_op)

Bramble.get_innermost_dim(op::InterpolationNode) = get_innermost_dim(op.inner_op)

# Two interpolations are the same shape only when they interpolate from the same space. The
# symmetry fast path compares the two sides of a product for structural equality, and an
# interpolation on one side only must not read as symmetric.
function _same_operator_shape(a::InterpolationNode{D}, b::InterpolationNode{D}) where {D}
    a.src_space === b.src_space && _same_operator_shape(a.inner_op, b.inner_op)
end

# --- The shift trait: which nodes carry something a relabelled offset cannot express -- #
#
# `stencil_shift_trait`'s base method (form/common.jl) says "translation invariant", which is
# what a trial or test function is however deeply wrapped. An interpolation is not: its
# entries name absolute columns picked by `locate_cell` from the point's own coordinates, and
# adding one to an offset says nothing about which columns the neighbour reaches. This ladder
# is what finds one under any tower of wrappers. Each method is decided by the operator's type
# alone, so the trait is a compile-time constant at every call and the branch it selects folds
# away.
#
# A *source* is the other node this is true of — point 68's own defect, on the same
# `local_stencil` relabelling assumption. Marking it here (point 71) is what let
# `_contracted_left_stencil` (form/operators/inner.jl) stop re-deriving every operator's
# masks and spacings by hand: a source-only subtree's own `local_stencil`, read through this
# trait, already re-evaluates at each neighbour the way a value contraction needs.
stencil_shift_trait(::InterpolationNode) = PointDependentStencil()
stencil_shift_trait(::SourceFunction) = PointDependentStencil()
stencil_shift_trait(::SourceVector) = PointDependentStencil()

stencil_shift_trait(op::BackwardDifference) = stencil_shift_trait(op.inner_op)
stencil_shift_trait(op::ForwardDifference) = stencil_shift_trait(op.inner_op)
stencil_shift_trait(op::CenteredDifference) = stencil_shift_trait(op.inner_op)
stencil_shift_trait(op::StarDifference) = stencil_shift_trait(op.inner_op)
stencil_shift_trait(op::CrossWeightedDifference) = stencil_shift_trait(op.inner_op)
stencil_shift_trait(op::BackwardAverage) = stencil_shift_trait(op.inner_op)
stencil_shift_trait(op::ForwardAverage) = stencil_shift_trait(op.inner_op)
stencil_shift_trait(op::ShiftNode) = stencil_shift_trait(op.inner_op)
stencil_shift_trait(op::JumpNode) = stencil_shift_trait(op.inner_op)
stencil_shift_trait(op::RegionRestriction) = stencil_shift_trait(op.inner_op)
stencil_shift_trait(op::OperatorScale) = stencil_shift_trait(op.inner_op)
stencil_shift_trait(op::GridFunctionScale) = stencil_shift_trait(op.inner_op)
function stencil_shift_trait(op::OperatorAdd)
    _combine_shift_traits(
        stencil_shift_trait(op.left_op), stencil_shift_trait(op.right_op))
end

# --- Which trial contributions interpolate, and from where --------------------------- #
#
# Two questions, and conflating them is a silent-wrong answer.
#
# `_all_trial_interpolated` asks whether **every** trial column the term contributes comes
# from an interpolation. Only then is the term exempt from the cross-mesh refusal
# (`_check_block_meshes`, point 69). A sum like `πₕ(Wsrc, u) + u` contributes absolute
# columns from one summand and ordinary offsets from the other, and the offsets still need
# the two leaves to share an index space. The first version of this file asked only "does an
# interpolation appear anywhere in the term", which exempted the bare `u` along with the
# interpolation and assembled it against wrong columns without a word — measured at 0.25
# absolute error on a 5&times;9 block, in range and therefore silent, which is exactly the
# failure point 69 exists to refuse.
#
# `_check_interp_spaces` then validates **each** interpolation in the term against the trial
# leaf rather than one of them: the columns are numbered in that node's own `Wsrc` and written
# into the trial leaf's column range, so every node has to agree with it, not just whichever
# one a walk happened to find first.
#
# Both are decided by the operator's type alone, so each rung folds to a constant.

# A node that contributes no trial column at all — a source, a test function — answers `true`
# vacuously: there is nothing there needing a mesh correspondence.
_all_trial_interpolated(::LazyOp) = false
_all_trial_interpolated(::InterpolationNode) = true
_all_trial_interpolated(::SourceFunction) = true
_all_trial_interpolated(::SourceVector) = true
_all_trial_interpolated(::TestFunction) = true
_all_trial_interpolated(::IndexedTestFunction) = true

_all_trial_interpolated(op::BackwardDifference) = _all_trial_interpolated(op.inner_op)
_all_trial_interpolated(op::ForwardDifference) = _all_trial_interpolated(op.inner_op)
_all_trial_interpolated(op::CenteredDifference) = _all_trial_interpolated(op.inner_op)
_all_trial_interpolated(op::StarDifference) = _all_trial_interpolated(op.inner_op)
_all_trial_interpolated(op::CrossWeightedDifference) = _all_trial_interpolated(op.inner_op)
_all_trial_interpolated(op::BackwardAverage) = _all_trial_interpolated(op.inner_op)
_all_trial_interpolated(op::ForwardAverage) = _all_trial_interpolated(op.inner_op)
_all_trial_interpolated(op::ShiftNode) = _all_trial_interpolated(op.inner_op)
_all_trial_interpolated(op::JumpNode) = _all_trial_interpolated(op.inner_op)
_all_trial_interpolated(op::RegionRestriction) = _all_trial_interpolated(op.inner_op)
_all_trial_interpolated(op::OperatorScale) = _all_trial_interpolated(op.inner_op)
_all_trial_interpolated(op::GridFunctionScale) = _all_trial_interpolated(op.inner_op)

# A sum needs *both* summands to interpolate — the whole point of this predicate.
function _all_trial_interpolated(op::OperatorAdd)
    _all_trial_interpolated(op.left_op) &&
        _all_trial_interpolated(op.right_op)
end

# Only the trial side of a product contributes columns, so only the trial side is asked. A
# linear product contributes none — its left factor is contracted away
# (`multiply_stencils_linear`), which is why a source interpolation belongs there and an
# operator one does not.
_all_trial_interpolated(op::BilinearProduct) = _all_trial_interpolated(op.left_op)
_all_trial_interpolated(op::LinearProduct) = true

# Validate every interpolation the term carries against the leaf whose columns it writes into.
_check_interp_spaces(::Any, trial_leaf) = nothing
function _check_interp_spaces(op::InterpolationNode, trial_leaf)
    _check_one_interp_space(
        op, op.src_space, trial_leaf)
end

_check_interp_spaces(op::BackwardDifference, t) = _check_interp_spaces(op.inner_op, t)
_check_interp_spaces(op::ForwardDifference, t) = _check_interp_spaces(op.inner_op, t)
_check_interp_spaces(op::CenteredDifference, t) = _check_interp_spaces(op.inner_op, t)
_check_interp_spaces(op::StarDifference, t) = _check_interp_spaces(op.inner_op, t)
_check_interp_spaces(op::CrossWeightedDifference, t) = _check_interp_spaces(op.inner_op, t)
_check_interp_spaces(op::BackwardAverage, t) = _check_interp_spaces(op.inner_op, t)
_check_interp_spaces(op::ForwardAverage, t) = _check_interp_spaces(op.inner_op, t)
_check_interp_spaces(op::ShiftNode, t) = _check_interp_spaces(op.inner_op, t)
_check_interp_spaces(op::JumpNode, t) = _check_interp_spaces(op.inner_op, t)
_check_interp_spaces(op::RegionRestriction, t) = _check_interp_spaces(op.inner_op, t)
_check_interp_spaces(op::OperatorScale, t) = _check_interp_spaces(op.inner_op, t)
_check_interp_spaces(op::GridFunctionScale, t) = _check_interp_spaces(op.inner_op, t)

function _check_interp_spaces(op::OperatorAdd, t)
    _check_interp_spaces(op.left_op, t)
    _check_interp_spaces(op.right_op, t)
    return nothing
end

_check_interp_spaces(op::BilinearProduct, t) = _check_interp_spaces(op.left_op, t)
_check_interp_spaces(op::LinearProduct, t) = nothing
