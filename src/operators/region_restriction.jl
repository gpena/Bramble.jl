# region_restriction.jl
# RegionRestriction struct and spatial restriction logic for Bramble lazy AST

# --- Struct definition ------------------------------------------------------------- #

"""
    RegionRestriction{D, RegionType, OpType <: LazyOp{D}, Z} <: LazyOp{D}

AST node representing a spatial restriction of an operator to a specific mesh region or boundary.

# Arguments
- `region::RegionType`: Identifier for the region (e.g. `:interior`, `:boundary`, `:left`, `:right`, `:top`, `:bottom`), or a tuple of them.
  Assembly binds it to the walked mesh's marker id (`Int`, or `NTuple{N, Int}` for a tuple; see `_bind_marker_ids`).
- `inner_op::OpType`: Underlying operator being restricted.
- `zero_stencil::Z`: what the node answers outside its region: `nothing` for the empty
  stencil `()`, or, for a restricted source a `form` has probed (`_typed_restrictions`,
  assembly/linear.jl), a zero stencil of the type the source has inside its region.
"""
struct RegionRestriction{D, RegionType, OpType <: LazyOp{D}, Z} <: LazyOp{D}
    region::RegionType
    inner_op::OpType
    zero_stencil::Z
end

# Every builder but `form`'s probe makes an unprobed node: `()` outside its region.
@inline RegionRestriction{D, R, O}(region, inner_op) where {D, R, O} = RegionRestriction{
    D, R, O, Nothing}(region, inner_op, nothing)

# `op` rebuilt around `region` and `inner`, keeping its zero stencil: the rebuilds a walk
# makes (binding the region, host copies) must not lose what `form` probed.
@inline _rebuild_restriction(op::RegionRestriction{D}, region, inner) where {D} = RegionRestriction{
    D, typeof(region), typeof(inner), typeof(op.zero_stencil)}(region, inner, op.zero_stencil)

# --- User-facing API --------------------------------------------------------------- #

"""
    restrict_to(region, op::LazyOp{D}) -> RegionRestriction

Restrict the operator `op` to a specific mesh region or boundary identifier.

# Examples
```julia
# Restrict the trial function to the interior
restrict_to(:interior, U)

# Restrict to a boundary region
restrict_to(:left, U)
```
"""
function restrict_to(region, op::LazyOp{D}) where {D}
    return RegionRestriction{D, typeof(region), typeof(op)}(region, op)
end

# --- Zero-allocation stencil evaluators -------------------------------------------- #

# `markers` is optional throughout the stencil evaluators. Every other node accepts it and
# ignores it, and callers with nothing to restrict by pass `nothing`. Only this node reads
# it, so only this node decides what an absent table means: no point is marked. The
# `:interior` region is then the whole grid and every other region is empty. `_in_region`
# below answers the `nothing` case itself, so `_is_marked` only ever sees a real table.
#
# A real table is the walked mesh's marker word matrix (`_marker_words`, word × label), and
# the region a label id bound at the walk entry (`_bind_walk`, assembly/block_extract.jl):
# point `lin_idx` of label `id` is one word load and a shift, no `Dict` per point.
@inline function _is_marked(words::AbstractMatrix{UInt64}, id::Int, lin_idx::Int)
    i = lin_idx - 1
    return isodd(@inbounds(words[(i >> 6) + 1, id]) >> (i & 63))
end

# A tuple of regions represents a union, not an intersection: `restrict_to((:bottom, :left), u)`
# matches either region, consistent with tuple markers throughout the package (`Rₕ!`,
# `dirichlet_bc!`, numeric `innerₕ`). Chaining single-region `RegionRestriction`s instead
# would give the intersection (a different and rarely useful condition).
@inline function _is_marked(
        words::AbstractMatrix{UInt64}, ids::NTuple{N, Int}, lin_idx::Int
) where {N}
    return any(id -> _is_marked(words, id, lin_idx), ids)
end

@inline function local_stencil(
        op::RegionRestriction, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D}
    # A real marker table always carries its own `:interior` (`_ensure_geometric_markers!`
    # guarantees the label, geometric or user-redefined), so it is read directly like every
    # other region, no exception for `:interior` here. There is exactly one case that still
    # needs one: `markers === nothing`, the "no marker context at all" sentinel above, where
    # `:interior` is defined as the whole grid and every other region as empty. Read
    # directly, a real `:interior` used to be silently overridden by "not :boundary", which
    # discarded a deliberately redefined `:interior` even though the mesh warns that a
    # custom definition wins (mesh/marker.jl).
    if _in_region(op, markers, lin_idx)
        return local_stencil(op.inner_op, space, I, markers, lin_idx)
    else
        return _outside_stencil(op.zero_stencil)
    end
end

# Outside its region a node answers the empty stencil, or, once `form` has probed it, the
# zero stencil it stored: the same length and type as inside, built without calling the
# source, so the stencil has one concrete type at every point (gpena/Bramble.jl#639).
@inline _outside_stencil(::Nothing) = ()
@inline _outside_stencil(z::Tuple) = z

# A probed restriction (`Z <: Tuple`) under a chain of pointwise scales: outside the region
# the scales are not multiplied into the zero stencil but zeroed with it, so a coefficient
# not finite there (`u * restrict_to(:interior, f)` with `u` infinite on the boundary, or
# `u * (2 * restrict_to(:interior, f))`) still gives 0, as the empty stencil did, not
# `Inf * 0 = NaN`. Inside it, the scales are applied as above. The same holds outside a
# probed restriction nested under it, through any scales between them
# (`u * restrict_to(:interior, 2 * restrict_to(:half, f))`).
#
# The outermost scale of such a chain walks it once with `_probed_chain` and zeroes the
# result at the top: zeroing at each level would be too late, as an outer `Inf` would
# already have multiplied an inner zero. Whether a scale heads such a chain is decided by
# the operand's type (`_reaches_probed`), so the branch folds away at compile time. A chain
# broken by any other node (a shift, a sum, a difference) is not followed.
const _ProbedRestriction{D} = RegionRestriction{D, <:Any, <:Any, <:Tuple}
const _ChainLink{D} = Union{_ProbedRestriction{D}, OperatorScale{D}, GridFunctionScale{D}}

# Whether `op` reaches a probed restriction through scales and probed restrictions only.
@inline _reaches_probed(::_ProbedRestriction) = true
@inline _reaches_probed(op::Union{OperatorScale, GridFunctionScale}) = _reaches_probed(
    op.inner_op)
@inline _reaches_probed(::Any) = false

# A scale whose operand is no such chain is applied as `local_stencil` applies it anywhere.
@inline function _chain_stencil(op, space, I, markers, lin_idx)
    _reaches_probed(op.inner_op) || return scale_stencil(
        local_stencil(op.inner_op, space, I, markers, lin_idx), _scale_value(op, lin_idx))
    s, outside = _probed_chain(op, space, I, markers, lin_idx)
    return outside ? _zero_entries(s) : s
end

@inline _scale_value(op::OperatorScale, ::Int) = op.scalar
@inline _scale_value(op::OperatorScale{D, <:Base.RefValue}, ::Int) where {D} = op.scalar[]
@inline _scale_value(op::GridFunctionScale, lin_idx::Int) = _grid_function_value(
    op.grid_function, lin_idx)

# `op`'s stencil at the point, and whether a probed restriction in the chain of probed
# restrictions and pointwise scales from `op` down leaves the point outside its region. If
# so, the stencil is that restriction's zero stencil scaled by the scales above it, which a
# caller zeroes rather than reads. The stencil has the type `local_stencil(op, ...)` has.
@inline _probed_chain(op, space, I, markers, lin_idx) = (
    local_stencil(op, space, I, markers, lin_idx), false)
@inline function _probed_chain(op::_ProbedRestriction, space, I, markers, lin_idx)
    _in_region(op, markers, lin_idx) || return (op.zero_stencil, true)
    return _probed_chain(op.inner_op, space, I, markers, lin_idx)
end
@inline function _probed_chain(
        op::Union{OperatorScale, GridFunctionScale}, space, I, markers, lin_idx
)
    s, outside = _probed_chain(op.inner_op, space, I, markers, lin_idx)
    return (scale_stencil(s, _scale_value(op, lin_idx)), outside)
end

# A scale over a link of a chain: the `Ref` method only resolves the ambiguity with
# `local_stencil(::OperatorScale{D, <:Base.RefValue})` (assembly/stencil_eval.jl).
@inline local_stencil(
    op::OperatorScale{D, <:Any, <:_ChainLink{D}}, space, I::CartesianIndex{D}, markers,
    lin_idx::Int
) where {D} = _chain_stencil(op, space, I, markers, lin_idx)
@inline local_stencil(
    op::OperatorScale{D, <:Base.RefValue, <:_ChainLink{D}}, space, I::CartesianIndex{D},
    markers, lin_idx::Int
) where {D} = _chain_stencil(op, space, I, markers, lin_idx)
@inline local_stencil(
    op::GridFunctionScale{D, <:Any, <:_ChainLink{D}}, space, I::CartesianIndex{D},
    markers, lin_idx::Int
) where {D} = _chain_stencil(op, space, I, markers, lin_idx)

@inline _in_region(op::RegionRestriction, ::Nothing, ::Int) = op.region === :interior
@inline _in_region(op::RegionRestriction, words, lin_idx::Int) = _is_marked(
    words, op.region, lin_idx)

# The source read a `ShiftNode` makes (`operators/shift.jl`): the restricted source at `J`
# inside the region, outside it the zero stencil `form` stored, as `local_stencil` above
# (the empty stencil only for a node `form` never probed). The source is never called
# outside its region, where it may be undefined (throw, or return a non-number), and a
# probed shifted restricted source infers one concrete type (gpena/Bramble.jl#639).
@inline function _source_stencil_at(
        op::RegionRestriction, space, J::CartesianIndex, markers, lin_idx::Int
)
    if _in_region(op, markers, lin_idx)
        return _source_stencil_at(op.inner_op, space, J, markers, lin_idx)
    else
        return _outside_stencil(op.zero_stencil)
    end
end

# A tap reaching `delta` points away re-evaluates the restriction at that neighbour. Doing
# it through `local_stencil` above would return `()` or a full tuple depending on the
# neighbour's marker, so the tap's tuple length would vary from point to point and the
# assembly would lose type stability. Instead the operand is shifted by its own rule and
# zeroed outside the region, which keeps the tuple length fixed.
@inline function shifted_inner_stencil(
        inner_op::RegionRestriction, inner, space, I::CartesianIndex{D}, markers, ::Val{Dim}, delta
) where {D, Dim}
    m = mesh(space)
    lins = LinearIndices(indices(m))
    Ishift = _clamped_shift(m, I, Val(Dim), _shift_delta(delta))
    sub = local_stencil(inner_op.inner_op, space, I, markers, lins[I])
    shifted = shifted_inner_stencil(inner_op.inner_op, sub, space, I, markers, Val(Dim), delta)
    T = eltype(space)
    return scale_stencil(shifted, _in_region(inner_op, markers, lins[Ishift]) ? one(T) : zero(T))
end

# --- AST resolution ---------------------------------------------------------------- #

function resolve_ast(op::RegionRestriction{D, RegionType}) where {D, RegionType}
    inner = resolve_ast(op.inner_op)
    return _rebuild_restriction(op, op.region, inner)
end

function _bind_interp_spaces(
        op::RegionRestriction{D, RegionType}, trial_leaf, test_leaf
) where {D, RegionType}
    inner = _bind_interp_spaces(op.inner_op, trial_leaf, test_leaf)
    return _rebuild_restriction(op, op.region, inner)
end

# --- Expression rendering ----------------------------------- #

# `repr` rather than plain string interpolation: `"$(:boundary)"` prints `boundary`, dropping
# the leading colon, while `repr(:boundary)` prints `:boundary`, which is what a caller
# wrote. `repr` on the tuple form (`(:bottom, :left)`) keeps the
# colon on every element too, so one call covers both `RegionType`s this node is built with.
expression(op::RegionRestriction) = "Rₕ($(repr(op.region)), $(expression(op.inner_op)))"
