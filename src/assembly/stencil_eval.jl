# stencil_eval.jl

# ==============================================================================
# 1. Zero-Allocation Stencil Evaluators
# ==============================================================================

#=
The three-tap template (gpena/Bramble.jl#70).

Every difference, average and jump node evaluates to the same shape: ordered taps drawn
from {+1, 0, -1}, each the inner operator's stencil either as-is (tap 0) or relabelled to a
neighbour, scaled by a per-node weight and concatenated in order. Eight bodies wrote that
out; a node now declares `_stencil_taps` and `_stencil_weights` in its own file and this is
the only evaluator.

Two properties this has to preserve, both load-bearing:

  - **Taps are a compile-time ordered tuple.** They are `Val`s so `_tap_stencil` can
    dispatch the tap-0 case away from the shifted one -- `shifted_inner_stencil` needs its
    delta as a type. A node declares exactly its own taps and no padding: a zero-weight tap
    added to make shapes uniform would widen `stencil_offsets`, which sizes the sparsity
    pattern and the parallel colouring strides.
  - **Concatenation order is entry-for-entry what it was.** `concatenate_stencils` is
    `(left..., right...)`, so folding right-to-left gives the same flat order the
    hand-written left-nested calls did.

`stencil_offsets` (`form/stencil_pattern.jl`) now reads the same `_stencil_taps`, so a
node's reach and its stencil cannot disagree -- they used to be spelled out twice.
=#

# One tap. Tap 0 is the inner stencil itself; any other is the inner stencil relabelled to
# that neighbour, which is exactly what `shifted_inner_stencil` decides by trait.
@inline _tap_stencil(op, inner, space, I, markers, ::Val{Dim}, ::Val{0}, w) where {Dim} = scale_stencil(inner, w)

@inline _tap_stencil(
    op, inner, space, I, markers, ::Val{Dim}, ::Val{Delta}, w
) where {Dim, Delta} = scale_stencil(
    shifted_inner_stencil(op.inner_op, inner, space, I, markers, Val(Dim), Val(Delta)),
    w
)

# Recursion rather than a `foldl`, so the tuple length is consumed at compile time and the
# stencil tuple type stays concrete through every step.
@inline _fold_taps(op, inner, space, I, markers, vdim, taps::Tuple{Any}, ws::Tuple{Any}) = _tap_stencil(
    op, inner, space, I, markers, vdim, taps[1], ws[1])

@inline _fold_taps(op, inner, space, I, markers, vdim, taps::Tuple, ws::Tuple) = concatenate_stencils(
    _tap_stencil(op, inner, space, I, markers, vdim, taps[1], ws[1]),
    _fold_taps(op, inner, space, I, markers, vdim, Base.tail(taps), Base.tail(ws))
)

"""
    UnaryWrapper{D}

The AST nodes that wrap exactly one operand in an `inner_op` field.

Thirteen node types, and the reason they are worth naming together: a query about a term
usually has the same answer for a wrapper as for the operand inside it, so each such query
used to be registered against all thirteen by hand -- one line apiece, per query
(gpena/Bramble.jl#52). A query that forgot one inherited a fallback instead, and every
fallback in this family is a plausible wrong answer rather than an error:
`test_component_or_nothing` answering `nothing` sends a term to every block.

Written as a union of concrete types rather than reached through a generic child accessor,
deliberately. Dispatch resolves it at compile time and each method still reads `op.inner_op`
directly, so nothing on the assembly path pays for the generality.

Products and sums are not members. `BilinearProduct` and `LinearProduct` hold two operands
with different roles -- trial on the left, test on the right -- so a query about the test
component reads `right_op` alone, and `OperatorAdd` has to reconcile both sides. Those stay
written out, which is the point: what is left explicit is what genuinely differs.
"""
const UnaryWrapper{D} = Union{
    BackwardDifference{D},
    ForwardDifference{D},
    CenteredDifference{D},
    StarDifference{D},
    CrossWeightedDifference{D},
    JumpNode{D},
    BackwardAverage{D},
    ForwardAverage{D},
    CenteredAverage{D},
    ShiftNode{D},
    OperatorScale{D},
    GridFunctionScale{D},
    RegionRestriction{D},
    InterpolationNode{D}
}

# The queries that answer for a wrapper whatever they answer for its operand. One method
# each, where every one of these was previously registered against all thirteen wrapper
# types by hand (gpena/Bramble.jl#52).
#
# They live here rather than beside their own ladders because `UnaryWrapper` names
# `InterpolationNode`, which `ast/operators/interpolation.jl` defines -- so the union, and
# anything dispatching on it, has to come after every operator file. The node-specific
# overrides stay in their own files; `InterpolationNode` deviates from three of these and
# says so there.
#
# Two of these override a fallback declared on the *abstract* type rather than filling a
# gap: `stencil_shift_trait(::LazyOp)` answers translation-invariant and
# `_all_trial_interpolated(::LazyOp)` answers false, so a wrapper reaching either default
# is a wrong answer, not a missing one. That is the shape of mistake this collapse is meant
# to make impossible: there is now one method to get right per query, not thirteen.
stencil_shift_trait(op::UnaryWrapper) = stencil_shift_trait(op.inner_op)
_all_trial_interpolated(op::UnaryWrapper) = _all_trial_interpolated(op.inner_op)
_all_test_interpolated(op::UnaryWrapper) = _all_test_interpolated(op.inner_op)
_has_test_interp(op::UnaryWrapper) = _has_test_interp(op.inner_op)
_has_trial_interp(op::UnaryWrapper) = _has_trial_interp(op.inner_op)
_is_dirac(op::UnaryWrapper) = _is_dirac(op.inner_op)

"""
    TappedNode{D, Dim}

The nodes whose stencil is ordered taps from {+1, 0, -1} along `Dim` with per-node weights:
the one-sided and extended differences, the three averages, and the jump. `ShiftNode` is not
one of them -- it relabels its child's whole stencil rather than combining taps.
"""
const TappedNode{D, Dim} = Union{
    BackwardDifference{D, Dim},
    ForwardDifference{D, Dim},
    CenteredDifference{D, Dim},
    StarDifference{D, Dim},
    CrossWeightedDifference{D, Dim},
    BackwardAverage{D, Dim},
    ForwardAverage{D, Dim},
    CenteredAverage{D, Dim},
    JumpNode{D, Dim}
}

@inline function local_stencil(
        op::TappedNode{D, Dim}, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D, Dim}
    inner = local_stencil(op.inner_op, space, I, markers, lin_idx)
    return _fold_taps(
        op,
        inner,
        space,
        I,
        markers,
        Val(Dim),
        _stencil_taps(op),
        _stencil_weights(op, space, I)
    )
end

@inline local_stencil(
    ::TrialFunction{D}, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D} = ((zero_offset(Val(D)), 1),)
@inline local_stencil(
    ::TestFunction{D}, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D} = ((zero_offset(Val(D)), 1),)
@inline local_stencil(
    ::IndexedTrialFunction{D}, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D} = ((zero_offset(Val(D)), 1),)
@inline local_stencil(
    ::IndexedTestFunction{D}, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D} = ((zero_offset(Val(D)), 1),)

@inline function local_stencil(
        op::SourceFunction{D}, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D}
    m = mesh(space)
    x = point(m, I)
    return ((zero_offset(Val(D)), op.func(x)),)
end

@inline function local_stencil(
        op::SourceVector{D}, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D}
    return ((zero_offset(Val(D)), op.vec[lin_idx]),)
end

@inline function local_stencil(
        op::SourceConstant{D}, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D}
    return ((zero_offset(Val(D)), op.value),)
end

@inline function _point_strength_val(s)
    if s isa Function
        return s()
    elseif s isa Base.RefValue
        return s[]
    else
        return s
    end
end

# `outside = :extrapolate`, always: a `DiracSource` point is a modelling choice (where the
# source physically sits), not a query that #223's `outside` policy is about, and this
# keeps its behaviour exactly what it was before that policy existed -- #226's own note
# ("out-of-domain handling consistent with whatever #223 settles for interpolate_at") is
# deliberately left for a follow-up, not folded into #223's own scope (its own "Subsystems
# to Update" list never names this file).
@inline _interp_cell_frac_pt(Ωₕ::AbstractMeshType{1}, pt::NTuple{1}, ::Val{1}) = (
    CartesianIndex(_interp_cell_frac(Ωₕ, pt[1], :extrapolate)[1]),
    (_interp_cell_frac(Ωₕ, pt[1], :extrapolate)[2],))
@inline _interp_cell_frac_pt(Ωₕ::AbstractMeshType{D}, pt::NTuple{D}, ::Val{D}) where {D} = _interp_cell_frac(
    Ωₕ, pt, :extrapolate
)

@inline _is_corner_inbounds(corner::Tuple) = all(c -> 0 <= c <= 1, corner)

@inline function local_stencil(
        op::DiracSource{D, <:NTuple{D, Float64}}, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D}
    Ωₕ = mesh(space)
    idx, ts = _interp_cell_frac_pt(Ωₕ, op.points, Val(D))
    corner = Tuple(I - idx)
    strength_val = _point_strength_val(op.strengths)
    T = promote_type(Float64, typeof(strength_val))
    if _is_corner_inbounds(corner)
        w = _interp_corner_weight(ts, corner, Val(D))
        return ((zero_offset(Val(D)), strength_val * w),)
    else
        return ((zero_offset(Val(D)), zero(T)),)
    end
end

@inline function local_stencil(
        op::DiracSource{D, <:AbstractVector}, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D}
    Ωₕ = mesh(space)
    s_first = _point_strength_val(first(op.strengths))
    T = promote_type(Float64, typeof(s_first))
    acc = zero(T)
    for k in eachindex(op.points)
        pt = op.points[k]
        s = _point_strength_val(op.strengths[k])
        idx, ts = _interp_cell_frac_pt(Ωₕ, pt, Val(D))
        corner = Tuple(I - idx)
        if _is_corner_inbounds(corner)
            w = _interp_corner_weight(ts, corner, Val(D))
            acc += s * w
        end
    end
    return ((zero_offset(Val(D)), acc),)
end

# Every summand's stencil, concatenated once in `_summands` order -- entry for entry what
# pairwise `concatenate_stencils` down the tree gives, without a left-deep sum building one
# intermediate tuple type per node.
@inline function local_stencil(
        op::OperatorAdd, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D}
    return _flatten_tuples(map(t -> local_stencil(t, space, I, markers, lin_idx), _summands(op)))
end

@inline function local_stencil(
        op::OperatorScale, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D}
    inner = local_stencil(op.inner_op, space, I, markers, lin_idx)
    return scale_stencil(inner, op.scalar)
end

@inline function local_stencil(
        op::OperatorScale{D, <:Base.RefValue}, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D}
    inner = local_stencil(op.inner_op, space, I, markers, lin_idx)
    return scale_stencil(inner, op.scalar[])
end

@inline function local_stencil(
        op::GridFunctionScale, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D}
    inner = local_stencil(op.inner_op, space, I, markers, lin_idx)
    return scale_stencil(inner, _grid_function_value(op.grid_function, lin_idx))
end

@inline local_stencil(
    op::IdentityOperator{D}, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D} = ((zero_offset(Val(D)), 1),)
# No entries at all, the same "contributes nothing here" `RegionRestriction` answers with.
# A one-entry `((offset, 0),)` is a linear-form entry only: a bilinear walk reads each entry
# as `(off_u, off_v, weight)`, and a form whose whole AST simplified to zero (`0 * a`, or a
# runtime `k = 0`) crashed destructuring it with a `BoundsError`.
@inline local_stencil(
    op::ZeroOperator{D}, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D} = ()

# ==============================================================================
# 2. AST Resolution & Thunk Eval
# ==============================================================================

# Each child is resolved once and its value reused for the type parameter. Calling
# `resolve_ast` again inside `typeof(...)` doubles the work per level, `2^depth` calls on a
# left-deep sum: free without coverage, where the compiler removes the duplicate, but under
# `--code-coverage` a 27-term sum's `form` did not finish within 10 GB.
function resolve_ast(op::OperatorAdd{D}) where {D}
    left = resolve_ast(op.left_op)
    right = resolve_ast(op.right_op)
    return OperatorAdd{D, typeof(left), typeof(right)}(left, right)
end
function resolve_ast(op::OperatorScale{D}) where {D}
    inner = resolve_ast(op.inner_op)
    return OperatorScale{D, typeof(op.scalar), typeof(inner)}(op.scalar, inner)
end

function resolve_ast(op::GridFunctionScale{D, VType}) where {D, VType}
    inner = resolve_ast(op.inner_op)
    return GridFunctionScale{D, VType, typeof(inner)}(op.grid_function, inner)
end

function resolve_ast(op::GridFunctionScale{D, <:Function}) where {D}
    vec = op.grid_function()
    inner = resolve_ast(op.inner_op)
    return GridFunctionScale{D, typeof(vec), typeof(inner)}(vec, inner)
end

resolve_ast(ops::NTuple{N, Any}) where {N} = map(resolve_ast, ops)

# The two scaling wrappers' half of `_bind_interp_spaces` (ast/operators/interpolation.jl),
# beside their `resolve_ast` because they are the same walk. `GridFunctionScale`'s thunk
# form has already been evaluated by `resolve_ast` when binding runs, so one method covers
# both: the scale itself is carried across untouched.
function _bind_interp_spaces(op::OperatorScale{D}, trial_leaf, test_leaf) where {D}
    inner = _bind_interp_spaces(op.inner_op, trial_leaf, test_leaf)
    return OperatorScale{D, typeof(op.scalar), typeof(inner)}(op.scalar, inner)
end

function _bind_interp_spaces(
        op::GridFunctionScale{D, VType}, trial_leaf, test_leaf
) where {D, VType}
    inner = _bind_interp_spaces(op.inner_op, trial_leaf, test_leaf)
    return GridFunctionScale{D, VType, typeof(inner)}(op.grid_function, inner)
end
# The catch-all every node above without its own method falls through to: TrialFunction,
# TestFunction, IndexedTrialFunction, IndexedTestFunction, SourceFunction, SourceVector,
# SourceConstant, IdentityOperator, ZeroOperator, and anything else with nothing to resolve.
# gpena/Bramble.jl#62: those nine used to have their own identity methods here, each
# decorative -- this catch-all made every one of them redundant, since a node not listed
# was never an error, only silently unresolved. Left as one line rather than nine.
resolve_ast(op::Any) = op

# ==============================================================================
# 3. Symbolic AST Traits
# ==============================================================================

# Note: is_symbolic base function is declared in ast.jl

# One method for every node that wraps a single operand, instead of one registration per
# node type (gpena/Bramble.jl#52); see [`UnaryWrapper`](@ref).
is_symbolic(op::UnaryWrapper) = is_symbolic(op.inner_op)
is_symbolic(::TrialFunction) = true
is_symbolic(::TestFunction) = true
is_symbolic(::IndexedTrialFunction) = true
is_symbolic(::IndexedTestFunction) = true
is_symbolic(::SourceFunction) = true
is_symbolic(::SourceVector) = true
is_symbolic(::SourceConstant) = true
is_symbolic(::DiracSource) = true
is_symbolic(op::BilinearProduct) = true
is_symbolic(op::LinearProduct) = true

# A wrapper is source-only when what it wraps is. One method instead of one registration
# per node type (gpena/Bramble.jl#52); `InterpolationNode` overrides it below, since it
# carries a trial function however it is wrapped.
"""
    _is_source_only(op::LazyOp) -> Bool

Whether a `LazyOp` subtree is source-only: built entirely from sources
(`SourceFunction`/`SourceVector`) and the plain operators that wrap them, never bottoming
out in a `TrialFunction`/`IndexedTrialFunction` leaf.

`innerₕ`'s `l::Function`/`l::Number`/`l::VectorElement` overloads (`operators/inner.jl`)
never need this: those three types are never anything but a source, so wrapping them in a
`LinearProduct` is unconditional. The question only exists for an argument that already
arrived as a `LazyOp`: `πₕ(uₕ)` ([`interpolate_at`](@ref)) or `D₋ₓ(πₕ(uₕ))` are
sources too, just already wrapped, and the generic `innerₕ(::LazyOp, ::LazyOp)` used to build
a `BilinearProduct` regardless, which is the wrong AST shape for a `LinearForm`'s assembly
walk: a `BilinearProduct`'s stencil carries a pair of offsets (trial and test), where
`_scatter_term!` (`form/linear.jl`) expects one.

A missing case defaults to `false` (the fallback `::LazyOp` method below): conservative,
since that is exactly the behavior every node had before this predicate existed (always
`BilinearProduct`) for anything not explicitly listed as source-only.
"""
_is_source_only(op::UnaryWrapper) = _is_source_only(op.inner_op)
_is_source_only(::TrialFunction) = false
_is_source_only(::TestFunction) = false
_is_source_only(::IndexedTrialFunction) = false
_is_source_only(::IndexedTestFunction) = false
_is_source_only(::SourceFunction) = true
_is_source_only(::SourceVector) = true
_is_source_only(::SourceConstant) = true
_is_source_only(::DiracSource) = true

function _is_source_only(op::OperatorAdd)
    return _is_source_only(op.left_op) && _is_source_only(op.right_op)
end

# A product (whichever kind) is its own thing, not a bare source to route again.
_is_source_only(::BilinearProduct) = false
_is_source_only(::LinearProduct) = false

_is_source_only(::LazyOp) = false

# The value of a source-only subtree at a grid point: `_contracted_left_stencil`
# (ast/operators/inner.jl) reads it from the subtree's own `local_stencil`, correctly
# re-evaluated at every neighbour because a source is `PointDependentStencil`
# (ast/operators/interpolation.jl).

# ==============================================================================
# 4. Walking an OperatorAdd tree: shared by every router in linear.jl/bilinear.jl
# ==============================================================================

# Functions across `linear.jl`/`bilinear.jl`/`jacobian_pattern.jl` (`_check_block_meshes`,
# `_route_terms!`, `_route_terms_parallel!`, `_pattern_blocks_jacobian!`,
# `_assemble_blocks_parallel!`) walk a form's `OperatorAdd` tree to send each summand where
# it belongs, all with the same shape: recurse left, recurse right, done. Recursing the tree
# rather than flattening it into a vector of terms first preserves concrete types: a
# flattened vector is `Vector{Any}` and makes every term a dynamic read, whereas recursing
# keeps each term concretely typed at its own call and costs nothing.
#
# They differ only in how many arguments sit before `op` and what (if anything) the caller
# reads back; nothing does: every call site above these six is a bare statement, the
# return value always discarded. The three shapes below return whatever the six already
# return unread today. Three separate names rather than one overloaded on argument count:
# with `rest...`/untyped leading arguments, overloads sharing a name are genuinely ambiguous
# to the compiler (a call whose second and third arguments both happen to be `OperatorAdd`
# matches two of the three signatures at once). Aqua's ambiguity check catches this even
# though no real call here ever hits it, and three names sidesteps the question rather than
# resolving it with a disambiguating method nothing calls. Each is still picked at its call
# site by hand; what moves out is only the recursion body, and
# `@code_warntype` still sees ordinary calls to `f`, specialized on `F = typeof(f)` like any
# other higher-order call in Julia.

# The summands of a sum, left to right, flattened into one tuple.
@inline _summands(op::OperatorAdd) = (_summands(op.left_op)..., _summands(op.right_op)...)
@inline _summands(op) = (op,)

# The three walks below visit `_summands(op)` with one `map` rather than recursing
# left/right: recursing, every `OperatorAdd` node is its own method instance whose type
# carries its whole subtree, so a left-deep `a + b + c + ...` (what `foldl(+, terms)` and a
# hand-written sum build) is inferred over types of total size quadratic in the term count.
# 3D `innerₕ(εcₕ(u), εcₕ(v))` built with `foldl(+, ...)`, first assemble: 18.6–21.0 s
# recursing, 16.6–17.6 s over `_summands`. Only leaves reach `f`; the walk order is unchanged.

# `op` first, nothing to mutate: `_check_block_meshes`.
@inline function _visit_operator_add1(f::F, op::OperatorAdd, rest...) where {F}
    map(t -> f(t, rest...), _summands(op))
    return nothing
end

# `op` second, one mutated argument returned unchanged: `_route_terms!`,
# `_route_terms_parallel!`, `_assemble_blocks!`, `_assemble_blocks_parallel!`.
@inline function _visit_operator_add2(f::F, first_arg, op::OperatorAdd, rest...) where {F}
    map(t -> f(first_arg, t, rest...), _summands(op))
    return first_arg
end

# `op` third, two mutated arguments, nothing returned: `_pattern_blocks_jacobian!`.
@inline function _visit_operator_add3(f::F, a1, a2, op::OperatorAdd, rest...) where {F}
    map(t -> f(a1, a2, t, rest...), _summands(op))
    return nothing
end

# `op` first, an accumulator threaded through both children and returned:
# `_route_terms_contract` (`form/linear.jl`).
#
# The `_visit_operator_add*` members above each return one of their own arguments unchanged,
# so they walk for side effects and cannot fold a value. Contraction threads an accumulator,
# so it hand-recursed instead. The family had a hole and its one folding caller quietly
# stepped around it rather than the family gaining this (gpena/Bramble.jl#55). The
# accumulator is the second positional argument, mirroring `_visit_operator_add2`'s
# `first_arg` slot, so the walk order reads the same across the family.
@inline function _fold_operator_add(f::F, op::OperatorAdd, acc, rest...) where {F}
    acc = f(op.left_op, acc, rest...)
    return f(op.right_op, acc, rest...)
end
