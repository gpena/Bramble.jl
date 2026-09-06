# interpolation.jl
# Symbolic counterpart of interpolate_at / πₕ! / πₕ (space/operators/interpolation.jl):
# wraps a grid function's interpolant as a SourceFunction, so it composes with the rest of
# the AST layer exactly the way any other source does. This is a second method of the same
# `πₕ` the numeric layer defines (dispatch distinguishes them by arity), not a separate name.

"""
    πₕ(uₕ::VectorElement) -> LazyOp

The interpolant of `uₕ`, as a symbolic source term; usable anywhere a source is, including
inside another operator: `innerₕ(D₋ₓ(πₕ(uₕ)), D₋ₓ(v))` differentiates the interpolated field
the same way `D₋ₓ` differentiates any other source, `innerₕ(M₋ₓ(πₕ(uₕ)), v)` averages it,
and so on. This enables a coupled form to evaluate a leaf's grid function on a different
leaf's mesh.

Built as `source_function(x -> interpolate_at(uₕ, x), Val(D))`: a `SourceFunction`'s own
`local_stencil` evaluates its function at the current point of whichever mesh is
being walked, so `uₕ` can originate from another leaf without special handling; the
interpolation occurs once per point inside `interpolate_at`, where ordinary source
function calls occur.
"""
function πₕ(uₕ::VectorElement{<:ScalarGridSpace{D}}) where {D}
    return source_function(x -> interpolate_at(uₕ, x), Val(D))
end

#===========================================================================#
# The interpolation operator: πₕ over a trial function.
#
# The source wrapper above requires concrete nodal values, so it cannot take a trial function:
# there are no values to blend. What a bilinear form requires instead is the operator itself: for
# each point of the test mesh, the `2ᴰ` trial degrees of freedom of the cell containing it,
# along with the corner weights. That matches one row of `interpolation_matrix`, produced a row
# at a time during the assembly sweep rather than as a separate matrix, which keeps `assemble!`
# refilling at zero allocations.
#
# The entries name absolute trial columns, not relative offsets. Every other node's stencil says
# "this many points from here, on the mesh being walked"; an interpolation says "these dofs
# of the other mesh", determined by where the point falls via `locate_cell`.
# `AbsoluteColumn` marks this distinction so bilinear assembly routines resolve each kind
# by dispatch rather than by a runtime flag.
#===========================================================================#

"""
    InterpolationNode{D, S, OpType} <: LazyOp{D}

The symbolic interpolation operator: `inner_op` lives on `src_space`, and this node evaluates
it at points of whatever mesh the assembly is walking.

Distinct from the source wrapper `πₕ(uₕ)`, which carries a grid function's values. This node
carries no values; it carries the map, and its stencil names trial columns.
"""
struct InterpolationNode{D, S, OpType <: LazyOp{D}} <: LazyOp{D}
    src_space::S
    inner_op::OpType
end

@noinline function _throw_interp_inner(op)
    throw(ArgumentError(
        "πₕ as a bilinear operator wraps a trial function directly (`πₕ(Wsrc, u)` or " *
        "`πₕ(Wsrc, u(2))`), but received $(typeof(op)). An operator applied before the " *
        "interpolation (`πₕ(Wsrc, D₋ₓ(u))`, differencing on the source mesh and then " *
        "interpolating) is a different operator and is not implemented; write the operator " *
        "outside instead, `D₋ₓ(πₕ(Wsrc, u))`, which differences on the mesh being " *
        "integrated over."))
end

#=
    πₕ(Wsrc::ScalarGridSpace{D}, op::LazyOp{D}) -> InterpolationNode

The interpolation operator from `Wsrc` onto whichever mesh the form integrates over, applied
to the trial function `op`.

This is the bilinear counterpart of `πₕ(uₕ)`: that one interpolates a grid function whose
values are already known, and belongs on the source side of a linear form; this one
interpolates the unknown, and so contributes matrix columns. `innerₕ(πₕ(Wsrc, u), v)`
assembles `Hᵥ · P`, with `P` the same matrix `interpolation_matrix` builds: computed a row
at a time during the sweep rather than as a matrix product, which allows `assemble!`
to refill it with zero allocations.

Operators wrap it from the outside, acting on the mesh being integrated over:
`inner₊(D₋ₓ(πₕ(Wsrc, u)), D₋ₓ(v))` is `D_x^⊤ H_+ D_x P`. Writing an operator inside is a
different operation and is refused, since it would difference on the source mesh instead.

`op` must be a trial-function leaf, plain or indexed.
=#
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

# It carries a trial function, so it is never a source however it is wrapped:
# this ensures `innerₕ` constructs a `BilinearProduct` for it.
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

# The reach on the mesh being walked is the inner leaf's (a single point). The columns this
# node names are on the other mesh and are not offsets, so they have no place in an
# offset set; a bilinear term's colouring only ever reads its test factor's reach anyway
# (`stencil_offsets(::BilinearProduct)`, form/stencil_pattern.jl), and an interpolation
# names a trial-side space, not a test one.
stencil_offsets(op::InterpolationNode) = stencil_offsets(op.inner_op)

# Two interpolations are the same shape only when they interpolate from the same space. The
# symmetry fast path compares the two sides of a product for structural equality, and an
# interpolation on one side only must not read as symmetric.
function _same_operator_shape(a::InterpolationNode{D}, b::InterpolationNode{D}) where {D}
    a.src_space === b.src_space && _same_operator_shape(a.inner_op, b.inner_op)
end

# --- The shift trait: which nodes carry something a relabelled offset cannot express -- #
#
# `stencil_shift_trait`'s base method (form/common.jl) indicates translation invariance, which
# holds for a trial or test function regardless of wrapper depth. An interpolation is not: its
# entries name absolute columns determined by `locate_cell` from the point's own coordinates, and
# adding one to an offset indicates nothing about which columns the neighbour reaches. This ladder
# discovers non-translation-invariant nodes under arbitrary wrappers. Each method is determined by
# the operator type alone, allowing the trait to fold away at compile time.
#
# A source is also point-dependent. Marking it here allows `_contracted_left_stencil`
# (form/operators/inner.jl) to avoid re-deriving masks and spacings manually: a source-only
# subtree's own `local_stencil`, read through this trait, re-evaluates at each neighbour as
# required by value contraction.
stencil_shift_trait(::InterpolationNode) = PointDependentStencil()
stencil_shift_trait(::SourceFunction) = PointDependentStencil()
stencil_shift_trait(::SourceVector) = PointDependentStencil()
stencil_shift_trait(::SourceConstant) = PointDependentStencil()

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
# Two separate questions arise:
#
# `_all_trial_interpolated` verifies whether every trial column contributed by the term originates
# from an interpolation. Only in that case is the term exempt from the cross-mesh refusal
# (`_check_block_meshes`). A sum like `πₕ(Wsrc, u) + u` contributes absolute columns from one
# summand and ordinary offsets from the other; the offsets still require both leaves to share an
# index space.
#
# `_check_interp_spaces` then validates each interpolation in the term against the trial
# leaf: the columns are numbered in that node's own `Wsrc` and written into the trial leaf's
# column range, so every node must agree with it.
#
# Both queries are decided by the operator's type alone, allowing each rung to fold to a constant.

# A node that contributes no trial column at all (such as a source or test function) answers `true`
# vacuously, as no mesh correspondence is required.
_all_trial_interpolated(::LazyOp) = false
_all_trial_interpolated(::InterpolationNode) = true
_all_trial_interpolated(::SourceFunction) = true
_all_trial_interpolated(::SourceVector) = true
_all_trial_interpolated(::SourceConstant) = true
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

# A sum requires both summands to interpolate.
function _all_trial_interpolated(op::OperatorAdd)
    _all_trial_interpolated(op.left_op) &&
        _all_trial_interpolated(op.right_op)
end

# Only the trial side of a product contributes columns, so only the trial side is inspected. A
# linear product contributes none: its left factor is contracted away
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
