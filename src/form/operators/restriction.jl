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
    RegionRestriction{D, typeof(region), typeof(op)}(region, op)
end

# --- Zero-allocation stencil evaluators -------------------------------------------- #

# `markers` is optional throughout the stencil evaluators: every other node accepts it and
# ignores it, and callers with nothing to restrict by pass `nothing`. Only this node reads
# it, so only this node determines what an absent table means: no point is marked. The
# `:interior` region is then the whole grid, and every other region is empty, which matches
# `haskey` returning `false` for a table that lacks the key.
@inline _is_marked(::Nothing, ::Symbol, ::Int) = false
@inline _is_marked(markers, region::Symbol, lin_idx::Int) = haskey(markers, region) &&
                                                            markers[region][lin_idx]

# A tuple of regions represents a union, not an intersection: `restrict_to((:bottom, :left), u)`
# matches either region, consistent with tuple markers throughout the package (`Rₕ!`,
# `dirichlet_bc!`, numeric `innerₕ`). Chaining single-region `RegionRestriction`s instead
# would give the intersection (a different and rarely useful condition).
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

# --- AST resolution ---------------------------------------------------------------- #

# A restriction narrows where its child contributes without changing what it reaches, so
# the direction is its child's.
Bramble.get_innermost_dim(op::RegionRestriction) = get_innermost_dim(op.inner_op)

function resolve_ast(op::RegionRestriction{D, RegionType}) where {D, RegionType}
    RegionRestriction{D, RegionType, typeof(resolve_ast(op.inner_op))}(op.region, resolve_ast(op.inner_op))
end
