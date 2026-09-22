# interpolation.jl
# Symbolic counterpart of interpolate_at / πₕ! / πₕ (space/operators/interpolation.jl):
# wraps a grid function's interpolant as a SourceFunction, so it composes with the rest of
# the AST layer exactly the way any other source does. This is a second method of the same
# `πₕ` the numeric layer defines (dispatch distinguishes them by arity), not a separate name.

"""
    πₕ(uₕ::VectorElement; outside = :error) -> LazyOp

The interpolant of `uₕ`, as a symbolic source term; usable anywhere a source is, including
inside another operator: `innerₕ(D₋ₓ(πₕ(uₕ)), D₋ₓ(v))` differentiates the interpolated field
the same way `D₋ₓ` differentiates any other source, `innerₕ(Mₓ(πₕ(uₕ)), v)` averages it,
and so on. This enables a coupled form to evaluate a leaf's grid function on a different
leaf's mesh.

Built as `source_function(x -> interpolate_at(uₕ, x; outside), Val(D))`: a
`SourceFunction`'s own `local_stencil` evaluates its function at the current point of
whichever mesh is being walked, so `uₕ` can originate from another leaf without special
handling; the interpolation occurs once per point inside `interpolate_at`, where ordinary
source function calls occur. `outside` (gpena/Bramble.jl#223) is forwarded to
[`interpolate_at`](@ref) unchanged, fill values included -- this is a source (a function of
`uₕ`'s values), not a linear map, so unlike the operator [`πₕ`](@ref)`(op)` below it
carries no such restriction.
"""
function πₕ(uₕ::VectorElement{<:ScalarGridSpace{D}}; outside = :error) where {D}
    _validate_outside(outside)
    return source_function(x -> interpolate_at(uₕ, x; outside), Val(D))
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

`Side` is [`TrialSide`](@ref) or [`TestSide`](@ref), taken from the leaf `πₕ` wrapped. It
decides which leaf `_bind_interp_spaces` stamps into `src_space`, whether the stencil names
[`AbsoluteColumn`](@ref)s or [`AbsoluteRow`](@ref)s, and which leaf's grid the sweep walks:
always the other one, the side that stays native (gpena/Bramble.jl#263).

`src_space` is `nothing` until assembly binds it. `πₕ(op)` cannot name the space itself: on a
composite space the leaf a term interpolates from is only resolved block by block, so
`_bind_interp_spaces` stamps that leaf in at that point, once per block, and the
`S === Nothing` node exists only between construction and that binding.

Distinct from the source wrapper `πₕ(uₕ)`, which carries a grid function's values. This node
carries no values; it carries the map, and its stencil names degrees of freedom of the space
it interpolates from.

`outside` (gpena/Bramble.jl#223) is one of `:error`, `:clamp` or `:extrapolate` -- never a
fill value, since this node's stencil is a *linear* map (weighted trial columns), and a
constant independent of the trial unknowns cannot be written that way; see
[`interpolation_matrix`](@ref)'s docstring for the same restriction.
"""
struct InterpolationNode{D, S, OpType <: LazyOp{D}, Side} <: LazyOp{D}
    src_space::S
    inner_op::OpType
    outside::Symbol
end

"""
    TrialSide
    TestSide

Which side of a bilinear term an [`InterpolationNode`](@ref) sits on.

A trial-side interpolation names absolute *columns* and leaves the rows to the walked mesh; a
test-side one names absolute *rows* and leaves the columns to it (gpena/Bramble.jl#263). The
side is a type parameter rather than a field so the node stays a singleton the like-term
simplifier can fold, and so every walk that reads it folds away at compile time.

The walked mesh is the native side's in both cases: it is the one carrying the quadrature
weight the product integrates against.
"""
struct TrialSide end

@doc (@doc TrialSide)
struct TestSide end

# Which leaf a node of each side binds to, and which slot type its stencil entries name.
@inline _interp_slot(::Type{TrialSide}) = AbsoluteColumn
@inline _interp_slot(::Type{TestSide}) = AbsoluteRow

# The side as a value, read off the type. Every side-dependent query goes through this rather
# than dispatching on the parameter directly: `UnaryWrapper` (form/stencil_eval.jl) names
# `InterpolationNode{D}`, and a method fixing a *later* parameter than `D` is neither more
# nor less specific than that union, which makes the pair ambiguous. One method on the
# unparameterized type is not, and the branch still folds away, since `Side` is in the type.
@inline _interp_side(::InterpolationNode{D, S, O, Side}) where {D, S, O, Side} = Side

@noinline function _throw_interp_inner(op)
    throw(
        ArgumentError(
        "πₕ as a bilinear operator wraps a trial or test function directly (`πₕ(u)`, " *
        "`πₕ(u(2))`, `πₕ(v)`), but received $(typeof(op)). An operator applied before the " *
        "interpolation (`πₕ(D₋ₓ(u))`, differencing on the source mesh and then " *
        "interpolating) is a different operator and is not implemented; write the operator " *
        "outside instead, `D₋ₓ(πₕ(u))`, which differences on the mesh being integrated over.",
    ),
    )
end

# The side a leaf puts the interpolation on. Both leaf kinds are accepted, plain or indexed
# (gpena/Bramble.jl#263); anything else is the operator-inside-interpolation refusal above.
@inline _interp_side_of(::Union{TrialFunction, IndexedTrialFunction}) = TrialSide
@inline _interp_side_of(::Union{TestFunction, IndexedTestFunction}) = TestSide

"""
    πₕ(op::LazyOp{D}; outside = :error) -> InterpolationNode

The interpolation operator onto whichever mesh the form integrates over, applied to the
trial or the test function `op`. The space interpolated *from* is that function's own, and is
not written here: it is bound during assembly, once the concrete leaf is known
(`_bind_interp_spaces`), which is also what makes this work on a composite space, where the
leaf a term names is only resolved block by block.

This is the bilinear counterpart of `πₕ(uₕ)`: that one interpolates a grid function whose
values are already known, and belongs on the source side of a linear form; this one
interpolates the unknown, and so contributes matrix columns. `innerₕ(πₕ(u), v)` assembles
`Hᵥ · P`, with `P` the same matrix `interpolation_matrix` builds: computed a row at a time
during the sweep rather than as a matrix product, which allows `assemble!` to refill it with
zero allocations.

Operators wrap it from the outside, acting on the mesh being integrated over:
`inner₊(D₋ₓ(πₕ(u)), D₋ₓ(v))` is `D_x^⊤ H_+ D_x P`. Writing an operator inside is a different
operation and is refused, since it would difference on the source mesh instead.

`innerₕ(u, πₕ(w))` is the mirror (gpena/Bramble.jl#263): the test side interpolates, so the
*trial* mesh is the one integrated over and the entries name absolute rows instead of absolute
columns. The assembled matrix is `Pᵀ · H · (trial factor)`. A single term interpolating both
sides is refused -- one side has to stay native, since it is the side whose mesh carries the
quadrature weight.

`op` must be a trial- or test-function leaf, plain or indexed. `outside`
(gpena/Bramble.jl#223) accepts only `:error` (the default), `:clamp` and `:extrapolate` -- see
[`InterpolationNode`](@ref)'s own docstring for why a fill value is refused here.
"""
function πₕ(op::LazyOp{D}; outside = :error) where {D}
    _is_interp_leaf(op) || _throw_interp_inner(op)
    _validate_outside_linear(outside)
    return InterpolationNode{D, Nothing, typeof(op), _interp_side_of(op)}(
        nothing, op, outside
    )
end

@inline _is_interp_leaf(op) = op isa TrialFunction || op isa IndexedTrialFunction ||
                              op isa TestFunction || op isa IndexedTestFunction

# --- The stencil: absolute trial columns, with the corner weights ------------------- #

@inline function local_stencil(
        op::InterpolationNode{D, S, OpType, Side}, space, I::CartesianIndex{D}, markers,
        lin_idx::Int
) where {D, S, OpType, Side}
    return _interp_stencil(
        mesh(op.src_space), point(mesh(space), I), Val(D), op.outside, _interp_slot(Side)
    )
end

# An unbound node reaching the sweep means an assembly path walked a term without calling
# `_bind_interp_spaces` first. Caught by dispatch rather than by a `nothing` check inside
# the bound method, so the common path carries no test, and loudly, since the alternative
# is a `MethodError` from `mesh(nothing)` several frames deeper.
@noinline function local_stencil(
        ::InterpolationNode{D, Nothing}, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D}
    throw(
        ArgumentError(
        "an interpolation node reached the assembly sweep without a source space. Every " *
        "path that walks a bilinear term over a block binds one first with " *
        "`_bind_interp_spaces(term, trial_leaf)`; this one did not.",
    ),
    )
end

# `Slot` is `AbsoluteColumn` on the trial side and `AbsoluteRow` on the test side: the
# blend itself is the same map either way, and only which half of the matrix position it
# names differs (gpena/Bramble.jl#263).
@inline function _interp_stencil(
        Ωsrc::AbstractMeshType{1}, x, ::Val{1}, outside::Symbol, Slot
)
    j, t = _interp_cell_frac(Ωsrc, x, outside)
    return ((Slot(j), 1 - t), (Slot(j + 1), t))
end

@inline function _interp_stencil(
        Ωsrc::AbstractMeshType{D}, x, ::Val{D}, outside::Symbol, Slot
) where {D}
    idx, ts = _interp_cell_frac(Ωsrc, x, outside)
    li = LinearIndices(indices(Ωsrc))
    # the `2ᴰ` corners, decoded from the bits of `k - 1` so the tuple length is static
    return ntuple(Val(1 << D)) do k
        corner = CartesianIndex(ntuple(d -> ((k - 1) >> (d - 1)) & 1, Val(D)))
        (Slot(li[idx + corner]), _interp_corner_weight(ts, corner, Val(D)))
    end
end

# --- Traits: every walker that sees through a wrapper has to see through this one ---- #

# It carries a trial function, so it is never a source however it is wrapped:
# this ensures `innerₕ` constructs a `BilinearProduct` for it.
_is_source_only(::InterpolationNode) = false

function resolve_ast(op::InterpolationNode{D, S, OpType, Side}) where {D, S, OpType, Side}
    inner = resolve_ast(op.inner_op)
    return InterpolationNode{D, S, typeof(inner), Side}(op.src_space, inner, op.outside)
end

@inline function component(
        op::InterpolationNode{D, S, OpType, Side}, i::Int
) where {D, S, OpType, Side}
    inner = component(op.inner_op, i)
    return InterpolationNode{D, S, typeof(inner), Side}(op.src_space, inner, op.outside)
end

# `_collect_region_labels` for `InterpolationNode` comes from its `UnaryWrapper` membership
# (form/block_extract.jl); it recursed the same way and needed no override.

# The reach on the mesh being walked is the inner leaf's (a single point). The columns this
# node names are on the other mesh and are not offsets, so they have no place in an
# offset set; a bilinear term's colouring only ever reads its test factor's reach anyway
# (`stencil_offsets(::BilinearProduct)`, form/stencil_pattern.jl), and an interpolation
# names a trial-side space, not a test one.
stencil_offsets(op::InterpolationNode) = stencil_offsets(op.inner_op)

# Two interpolations are the same shape only when they interpolate from the same space
# under the same out-of-domain policy (gpena/Bramble.jl#223) -- :clamp and :extrapolate
# disagree exactly at the points that matter, so treating them as interchangeable here
# would let symmetry detection paper over a real difference. The symmetry fast path
# compares the two sides of a product for structural equality, and an interpolation on one
# side only must not read as symmetric.
# Two interpolations on opposite sides are never the same shape either: one names columns and
# the other rows, so the symmetry fast path must not read such a product as symmetric.
function _same_operator_shape(a::InterpolationNode{D}, b::InterpolationNode{D}) where {D}
    return _interp_side(a) === _interp_side(b) && a.src_space === b.src_space &&
           a.outside === b.outside && _same_operator_shape(a.inner_op, b.inner_op)
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
stencil_shift_trait(::DiracSource) = PointDependentStencil()

# A `GridFunctionScale` is point-dependent in its own right, whatever it wraps: the
# coefficient it reads varies from point to point exactly like a source's value does, so a
# neighbour's contribution needs the coefficient re-read there, not relabelled here. Without
# this, `UnaryWrapper`'s fallback (`stencil_shift_trait(op.inner_op)`, form/stencil_eval.jl)
# forwards to whatever the wrapped trial/test function reports -- translation-invariant --
# and `D₋ₓ(cₕ * u)` reads the coefficient at the point being visited instead of the point
# the difference's tap reaches (gpena/Bramble.jl#271).
#
# This line alone is not enough, and briefly worse than the bug it targets: the generic
# `PointDependentStencil` branch (form/common.jl) discards the operand's own stencil and
# re-evaluates the whole node at the shifted point, which for a `GridFunctionScale` loses
# the trial or test column the operand contributed -- `local_stencil(GridFunctionScale(c, u),
# ..., Ishift)` returns a single entry at offset zero, not the trial column shifted by
# `delta`. The two `shifted_inner_stencil` overrides in form/common.jl are what make this
# line correct: they shift the operand by its own rule and read the coefficient at the
# shifted point separately, instead of asking the trait's two stock branches to do both at
# once.
stencil_shift_trait(::GridFunctionScale) = PointDependentStencil()

function stencil_shift_trait(op::OperatorAdd)
    return _combine_shift_traits(
        stencil_shift_trait(op.left_op), stencil_shift_trait(op.right_op)
    )
end

# --- Which trial contributions interpolate, and from where --------------------------- #
#
# Two separate questions arise:
#
# `_all_trial_interpolated` verifies whether every trial column contributed by the term originates
# from an interpolation. Only in that case is the term exempt from the cross-mesh refusal
# (`_check_block_meshes`). A sum like `πₕ(u) + u` contributes absolute columns from one
# summand and ordinary offsets from the other; the offsets still require both leaves to share an
# index space.
#
# `_bind_interp_spaces` then supplies each interpolation with the leaf whose degrees of freedom
# it names -- the trial leaf for a trial-side node, the test leaf for a test-side one. Nothing
# validates the two against each other any more: the node is given that leaf and has no other
# space to disagree with, which is what dropping `πₕ`'s space argument bought
# (gpena/Bramble.jl#10).
#
# Both are decided by the operator's type alone, allowing each rung to fold to a constant.

# A node that contributes no trial column at all (such as a source or test function) answers `true`
# vacuously, as no mesh correspondence is required.
_all_trial_interpolated(::LazyOp) = false
_all_trial_interpolated(op::InterpolationNode) = _interp_side(op) === TrialSide
_all_trial_interpolated(::SourceFunction) = true
_all_trial_interpolated(::SourceVector) = true
_all_trial_interpolated(::SourceConstant) = true
_all_trial_interpolated(::DiracSource) = true
_all_trial_interpolated(::TestFunction) = true
_all_trial_interpolated(::IndexedTestFunction) = true

# The mirror question, for a test-side interpolation (gpena/Bramble.jl#263): whether every
# row the term scatters into is named by an interpolation. That is the other way a block may
# straddle two meshes without the two leaves having to share an index space.
_all_test_interpolated(::LazyOp) = false
_all_test_interpolated(op::InterpolationNode) = _interp_side(op) === TestSide
_all_test_interpolated(::SourceFunction) = true
_all_test_interpolated(::SourceVector) = true
_all_test_interpolated(::SourceConstant) = true
_all_test_interpolated(::DiracSource) = true
_all_test_interpolated(::TrialFunction) = true
_all_test_interpolated(::IndexedTrialFunction) = true

function _all_test_interpolated(op::OperatorAdd)
    return _all_test_interpolated(op.left_op) && _all_test_interpolated(op.right_op)
end

_all_test_interpolated(op::BilinearProduct) = _all_test_interpolated(op.right_op)
_all_test_interpolated(op::LinearProduct) = true

# Whether the term carries a test-side interpolation anywhere. This is what decides which
# leaf's grid the sweep walks: the native side's, since that is the side whose mesh supplies
# the quadrature weight the product integrates against. Decided by type alone, so the choice
# folds away at compile time.
_has_test_interp(::LazyOp) = false
_has_test_interp(op::InterpolationNode) = _interp_side(op) === TestSide

# The same question for the trial side, which only `_check_one_interpolated_side`
# (form/operators/inner.jl) asks: the walked leaf does not depend on it, since a trial-side
# interpolation leaves the test side native and that is where the sweep already walks.
_has_trial_interp(::LazyOp) = false
_has_trial_interp(op::InterpolationNode) = _interp_side(op) === TrialSide

function _has_trial_interp(op::OperatorAdd)
    return _has_trial_interp(op.left_op) || _has_trial_interp(op.right_op)
end

function _has_trial_interp(op::BilinearProduct)
    return _has_trial_interp(op.left_op) || _has_trial_interp(op.right_op)
end

_has_trial_interp(op::LinearProduct) = false

function _has_test_interp(op::OperatorAdd)
    return _has_test_interp(op.left_op) || _has_test_interp(op.right_op)
end

function _has_test_interp(op::BilinearProduct)
    return _has_test_interp(op.left_op) || _has_test_interp(op.right_op)
end

_has_test_interp(op::LinearProduct) = false

"""
    _walked_leaf(term, trial_leaf, test_leaf)

The leaf whose grid the assembly sweep walks for `term`, and whose weights and markers its
stencil sees.

The test leaf, as it has always been, unless the term interpolates on the test side: then the
rows are named absolutely and the trial leaf is the one that stays native, so it supplies the
grid, the quadrature weight and the columns (gpena/Bramble.jl#263).
"""
@inline function _walked_leaf(term, trial_leaf, test_leaf)
    return _has_test_interp(term) ? trial_leaf : test_leaf
end

# A sum requires both summands to interpolate.
function _all_trial_interpolated(op::OperatorAdd)
    return _all_trial_interpolated(op.left_op) && _all_trial_interpolated(op.right_op)
end

# Only the trial side of a product contributes columns, so only the trial side is inspected. A
# linear product contributes none: its left factor is contracted away
# (`multiply_stencils_linear`), which is why a source interpolation belongs there and an
# operator one does not.
_all_trial_interpolated(op::BilinearProduct) = _all_trial_interpolated(op.left_op)
_all_trial_interpolated(op::LinearProduct) = true

# Bind every interpolation the term carries to the leaf whose columns it writes into.
#
# `πₕ(u)` names no space: the columns it produces are numbered in the trial function's own
# space, which is exactly `blk.trial_leaf`, and on a composite space that leaf is only known
# once `blocks` has resolved the term's `component_idx`. This pass stamps it in, with the
# recursion shape of `resolve_ast` -- the fallback returns the term untouched, so a term
# carrying no interpolation rebuilds nothing.
#
# Every method is decided by the operator's type alone, so the walk folds away at compile
# time and a bound term is as concrete as the one it came from. Pattern discovery and
# execution must bind identically, or the pattern reserves entries the sweep never fills.
_bind_interp_spaces(op::Any, trial_leaf, test_leaf) = op

# Each node binds to the leaf whose degrees of freedom it names: the trial leaf for a
# trial-side interpolation, the test leaf for a test-side one (gpena/Bramble.jl#263). Both
# leaves are threaded through the whole walk, so one form may interpolate on either side in
# different terms.
function _bind_interp_spaces(
        op::InterpolationNode{D, S, OpType, TrialSide}, trial_leaf, test_leaf
) where {D, S, OpType}
    inner = _bind_interp_spaces(op.inner_op, trial_leaf, test_leaf)
    return InterpolationNode{D, typeof(trial_leaf), typeof(inner), TrialSide}(
        trial_leaf, inner, op.outside
    )
end

function _bind_interp_spaces(
        op::InterpolationNode{D, S, OpType, TestSide}, trial_leaf, test_leaf
) where {D, S, OpType}
    inner = _bind_interp_spaces(op.inner_op, trial_leaf, test_leaf)
    return InterpolationNode{D, typeof(test_leaf), typeof(inner), TestSide}(
        test_leaf, inner, op.outside
    )
end

function _bind_interp_spaces(op::OperatorAdd{D}, trial_leaf, test_leaf) where {D}
    left = _bind_interp_spaces(op.left_op, trial_leaf, test_leaf)
    right = _bind_interp_spaces(op.right_op, trial_leaf, test_leaf)
    return OperatorAdd{D, typeof(left), typeof(right)}(left, right)
end

# Both sides of a product are bound, since either may carry an interpolation: the trial side
# names columns and the test side rows. A `LinearProduct` contracts its left factor away and
# can hold no interpolation node at all (`_is_source_only` answers false for one, which is
# what makes `innerₕ` build a `BilinearProduct` instead), so it binds nothing.
function _bind_interp_spaces(
        op::BilinearProduct{D, InnerType}, trial_leaf, test_leaf
) where {D, InnerType}
    left = _bind_interp_spaces(op.left_op, trial_leaf, test_leaf)
    right = _bind_interp_spaces(op.right_op, trial_leaf, test_leaf)
    return BilinearProduct{D, InnerType, typeof(left), typeof(right)}(left, right)
end

function _bind_interp_spaces(ops::NTuple{N, Any}, trial_leaf, test_leaf) where {N}
    map(
        op -> _bind_interp_spaces(op, trial_leaf, test_leaf), ops
    )
end

# --- Expression rendering (gpena/Bramble.jl#274) ----------------------------------- #

# Operand only, per the plan's departure from the issue text: `src_space` is `nothing` until
# assembly binds it (`_bind_interp_spaces`) and carries no name a caller wrote, so rendering
# it would show either `nothing` or an internal leaf object instead of anything the caller
# recognizes.
expression(op::InterpolationNode) = "πₕ($(expression(op.inner_op)))"
