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
# A *source* is the other node this is true of, and it is deliberately left translation
# invariant here: point 68 already fixed the same defect for sources by a separate route —
# `_source_value` (form/common.jl), which a `LinearProduct` reaches through
# `_contracted_left_stencil` instead of through the shift at all — and marking sources
# point-dependent would change that behaviour rather than merely extend it. The two
# mechanisms answer one question and one of them should go; which is a decision of its own,
# with its own verification, not a side effect of this one.
stencil_shift_trait(::InterpolationNode) = PointDependentStencil()

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

# --- Which space a term interpolates from, if any ------------------------------------ #
#
# `nothing` when the term's trial entries are ordinary offsets, and the source space itself
# when they are absolute columns of another space (`AbsoluteColumn`, above). Two things want
# this, and both need more than a yes/no: whether the two leaves of a coupled block may be
# over different meshes at all, and — since they may — whether the space the interpolation
# names is in fact the trial leaf whose columns the block writes into
# (`_check_block_meshes`, point 69/61).
#
# Type-determined at every rung, so `_interp_src_space(term) === nothing` is a compile-time
# constant and the branches it guards fold away.
_interp_src_space(::LazyOp) = nothing
_interp_src_space(op::InterpolationNode) = op.src_space
_interp_src_space(op::BackwardDifference) = _interp_src_space(op.inner_op)
_interp_src_space(op::ForwardDifference) = _interp_src_space(op.inner_op)
_interp_src_space(op::CenteredDifference) = _interp_src_space(op.inner_op)
_interp_src_space(op::StarDifference) = _interp_src_space(op.inner_op)
_interp_src_space(op::CrossWeightedDifference) = _interp_src_space(op.inner_op)
_interp_src_space(op::BackwardAverage) = _interp_src_space(op.inner_op)
_interp_src_space(op::ForwardAverage) = _interp_src_space(op.inner_op)
_interp_src_space(op::ShiftNode) = _interp_src_space(op.inner_op)
_interp_src_space(op::JumpNode) = _interp_src_space(op.inner_op)
_interp_src_space(op::RegionRestriction) = _interp_src_space(op.inner_op)
_interp_src_space(op::OperatorScale) = _interp_src_space(op.inner_op)
_interp_src_space(op::GridFunctionScale) = _interp_src_space(op.inner_op)
function _interp_src_space(op::OperatorAdd)
    _either_interp_space(
        _interp_src_space(op.left_op), _interp_src_space(op.right_op))
end

# Only the trial side of a product contributes columns, so only the trial side is asked. A
# linear product contributes none at all — its left factor is contracted away
# (`multiply_stencils_linear`), which is why a source interpolation belongs there and an
# operator one does not.
_interp_src_space(op::BilinearProduct) = _interp_src_space(op.left_op)
_interp_src_space(op::LinearProduct) = nothing

@inline _either_interp_space(::Nothing, ::Nothing) = nothing
@inline _either_interp_space(a, ::Nothing) = a
@inline _either_interp_space(::Nothing, b) = b
# Both summands interpolate. Taking the left one is enough: the block they write into is one
# block, so if they named different spaces the check on the way in would fail for one of them
# whichever was reported.
@inline _either_interp_space(a, b) = a

# Whether the term's trial entries are absolute columns rather than offsets.
_bears_interpolation(term) = _interp_src_space(term) !== nothing
