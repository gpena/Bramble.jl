# restriction.jl
# Contains RegionRestriction struct and related logic for Bramble lazy AST

# ==============================================================================
# Struct Definitions
# ==============================================================================

"""
    RegionRestriction{D,RegionType,OpType<:LazyOp{D}} <: LazyOp{D}

An AST node representing a spatial restriction of an operator to a specific mesh region or boundary.

# Fields
- `region::RegionType`: The identifier for the region (e.g., `:interior`, `:boundary`, `:left`, `:right`, `:top`, `:bottom`).
- `inner_op::OpType`: The underlying operator being restricted.
"""
struct RegionRestriction{D, RegionType, OpType <: LazyOp{D}} <: LazyOp{D}
    region::RegionType
    inner_op::OpType
end

# ==============================================================================
# User-Facing API
# ==============================================================================

"""
    restrict_to(region, op::LazyOp{D}) where D

Restricts the operator `op` to a specific mesh region or boundary identifier.

# Examples
```julia
# Restrict the trial function to the interior
restrict_to(:interior, U)

# Restrict to a boundary region
restrict_to(:left, U)
```
"""
function restrict_to(region, op::LazyOp{D}) where {D}
    RegionRestriction{D, typeof(region), typeof(op)}(region, op)
end

# ==============================================================================
# Zero-Allocation Stencil Evaluators
# ==============================================================================

# `markers` is optional throughout the stencil evaluators — every other node takes it and
# ignores it, and callers with nothing to restrict by pass `nothing`. Only this node reads
# it, so only this node has to say what an absent table means: no point is marked. The
# `:interior` region is then the whole grid, and every other region is empty, which is what
# `haskey` returning `false` already gave for a table that simply lacked the key.
@inline _is_marked(::Nothing, ::Symbol, ::Int) = false
@inline _is_marked(markers, region::Symbol, lin_idx::Int) = haskey(markers, region) &&
                                                            markers[region][lin_idx]

# A tuple of regions is a union, not an intersection: `restrict_to((:bottom, :left), u)`
# means either counts, matching what several `markers = (...)` labels mean everywhere else
# in the package (`Rₕ!`, `dirichlet_bc!`, the numeric `innerₕ`). Chaining single-region
# `RegionRestriction`s instead would give the intersection, which is a different — and much
# less useful — condition.
@inline _is_marked(markers, regions::NTuple{N, Symbol}, lin_idx::Int) where {N} = any(
    r -> _is_marked(markers, r, lin_idx), regions)

@inline function local_stencil(
        op::RegionRestriction, space, I::CartesianIndex{D}, markers, lin_idx::Int) where {D}
    if op.region === :interior
        in_region = !_is_marked(markers, :boundary, lin_idx)
    else
        in_region = _is_marked(markers, op.region, lin_idx)
    end

    if in_region
        return local_stencil(op.inner_op, space, I, markers, lin_idx)
    else
        return ()
    end
end

# The value twin of the stencil above. Off-region the stencil is empty, which contributes
# nothing; a value contributes nothing by being zero, which the assembly's `+=` treats the
# same way. See `_source_value` in form/common.jl.
@inline function _source_value(
        op::RegionRestriction, space, I::CartesianIndex{D}, markers) where {D}
    lin = _source_lin(space, I)
    in_region = if op.region === :interior
        !_is_marked(markers, :boundary, lin)
    else
        _is_marked(markers, op.region, lin)
    end
    v = _source_value(op.inner_op, space, I, markers)
    return in_region ? v : zero(v)
end

# ==============================================================================
# AST Resolution
# ==============================================================================

# A restriction narrows where its child contributes without changing what it reaches, so
# the direction is its child's.
Bramble.get_innermost_dim(op::RegionRestriction) = get_innermost_dim(op.inner_op)

function resolve_ast(op::RegionRestriction{D, RegionType}) where {D, RegionType}
    RegionRestriction{D, RegionType, typeof(resolve_ast(op.inner_op))}(op.region, resolve_ast(op.inner_op))
end
