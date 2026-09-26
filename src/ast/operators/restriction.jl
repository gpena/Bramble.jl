# restriction.jl
# RegionRestriction struct and spatial restriction logic for Bramble lazy AST

# --- Struct definition ------------------------------------------------------------- #

"""
    RegionRestriction{D, RegionType, OpType <: LazyOp{D}} <: LazyOp{D}

AST node representing a spatial restriction of an operator to a specific mesh region or boundary.

# Arguments
- `region::RegionType`: Identifier for the region (e.g. `:interior`, `:boundary`, `:left`, `:right`, `:top`, `:bottom`).
- `inner_op::OpType`: Underlying operator being restricted.
"""
struct RegionRestriction{D, RegionType, OpType <: LazyOp{D}} <: LazyOp{D}
    region::RegionType
    inner_op::OpType
end

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

# `markers` is optional throughout the stencil evaluators: every other node accepts it and
# ignores it, and callers with nothing to restrict by pass `nothing`. Only this node reads
# it, so only this node determines what an absent table means: no point is marked. The
# `:interior` region is then the whole grid, and every other region is empty, which matches
# `haskey` returning `false` for a table that lacks the key.
@inline _is_marked(::Nothing, ::Symbol, ::Int) = false
@inline _is_marked(markers, region::Symbol, lin_idx::Int) = haskey(markers, region) && markers[region][lin_idx]

# A tuple of regions represents a union, not an intersection: `restrict_to((:bottom, :left), u)`
# matches either region, consistent with tuple markers throughout the package (`Rₕ!`,
# `dirichlet_bc!`, numeric `innerₕ`). Chaining single-region `RegionRestriction`s instead
# would give the intersection (a different and rarely useful condition).
@inline _is_marked(markers, regions::NTuple{N, Symbol}, lin_idx::Int) where {N} = any(r -> _is_marked(markers, r, lin_idx), regions)

@inline function local_stencil(
        op::RegionRestriction, space, I::CartesianIndex{D}, markers, lin_idx::Int
) where {D}
    # A real marker table always carries its own `:interior` (`_ensure_geometric_markers!`
    # guarantees the key, geometric or user-redefined), so it is read directly like every
    # other region, no exception for `:interior` here. There is exactly one case that still
    # needs one: `markers === nothing`, the "no marker context at all" sentinel above, where
    # `:interior` is defined as the whole grid rather than as `_is_marked`'s blanket `false`
    # for every region. Read directly, a real `:interior` used to be silently overridden by
    # "not :boundary", which discarded a deliberately redefined `:interior` even though the
    # mesh warns that a custom definition wins (mesh/marker.jl).
    if _in_region(op, markers, lin_idx)
        return local_stencil(op.inner_op, space, I, markers, lin_idx)
    else
        return ()
    end
end

@inline _in_region(op::RegionRestriction, markers, lin_idx::Int) = markers === nothing ?
                                                                   (op.region === :interior) :
                                                                   _is_marked(markers, op.region, lin_idx)

@inline _wraps_leaf(::RegionRestriction{D, R, <:_BareLeaf}) where {D, R} = true

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
    return RegionRestriction{D, RegionType, typeof(inner)}(op.region, inner)
end

function _bind_interp_spaces(
        op::RegionRestriction{D, RegionType}, trial_leaf, test_leaf
) where {D, RegionType}
    inner = _bind_interp_spaces(op.inner_op, trial_leaf, test_leaf)
    return RegionRestriction{D, RegionType, typeof(inner)}(op.region, inner)
end

# --- Expression rendering (gpena/Bramble.jl#274) ----------------------------------- #

# `repr` rather than plain string interpolation: `"$(:boundary)"` prints `boundary`, dropping
# the leading colon, while `repr(:boundary)` prints `:boundary`, which is what a caller wrote
# and what the CHECK below expects. `repr` on the tuple form (`(:bottom, :left)`) keeps the
# colon on every element too, so one call covers both `RegionType`s this node is built with.
expression(op::RegionRestriction) = "Rₕ($(repr(op.region)), $(expression(op.inner_op)))"
