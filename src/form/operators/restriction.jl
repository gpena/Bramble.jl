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
struct RegionRestriction{D,RegionType,OpType<:LazyOp{D}} <: LazyOp{D}
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
restrict_to(region, op::LazyOp{D}) where D = RegionRestriction{D,typeof(region),typeof(op)}(region, op)


# ==============================================================================
# Zero-Allocation Stencil Evaluators
# ==============================================================================

@inline function local_stencil(op::RegionRestriction, space, I::CartesianIndex{D}, markers, lin_idx::Int) where D
	if op.region === :interior
		in_region = !(haskey(markers, :boundary) && markers[:boundary][lin_idx])
	else
		in_region = haskey(markers, op.region) && markers[op.region][lin_idx]
	end

	if in_region
		return local_stencil(op.inner_op, space, I, markers, lin_idx)
	else
		return ()
	end
end

# ==============================================================================
# AST Resolution
# ==============================================================================

resolve_ast(op::RegionRestriction{D,RegionType}) where {D,RegionType} = RegionRestriction{D,RegionType,typeof(resolve_ast(op.inner_op))}(op.region, resolve_ast(op.inner_op))
