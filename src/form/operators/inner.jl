# inner.jl
# Contains all inner product traits and logic for Bramble lazy AST

# ==============================================================================
# Struct Definitions
# ==============================================================================

"""
    AbstractInnerProduct

Abstract base type for inner product quadrature weights.
"""
abstract type AbstractInnerProduct end

"""
    InnerH <: AbstractInnerProduct

Quadrature weights for the standard \$L^2\$ inner product using trapezoidal integration.
"""
struct InnerH <: AbstractInnerProduct end

"""
    InnerPlus{Dim} <: AbstractInnerProduct

Quadrature weights for the modified \$L^2_+\$ inner product in a specific coordinate dimension `Dim`.
"""
struct InnerPlus{Dim} <: AbstractInnerProduct end

"""
    BilinearProduct{D,InnerType,LeftType,RightType} <: LazyOp{D}

An AST node representing a bilinear integration term \$(u, v)\$ in a bilinear form.
"""
struct BilinearProduct{D,InnerType<:AbstractInnerProduct,LeftType<:LazyOp{D},RightType<:LazyOp{D}} <: LazyOp{D}
	left_op::LeftType
	right_op::RightType
end

"""
    LinearProduct{D,InnerType,LeftType,RightType} <: LazyOp{D}

An AST node representing a linear integration term \$(f, v)\$ in a linear form.
"""
struct LinearProduct{D,InnerType<:AbstractInnerProduct,LeftType<:LazyOp{D},RightType<:LazyOp{D}} <: LazyOp{D}
	left_op::LeftType
	right_op::RightType
end

# ==============================================================================
# Weight Helpers
# ==============================================================================

@inline compute_weight(::InnerH, space, I::CartesianIndex{D}, lin_idx::Int) where D = weights(space, Innerh())[lin_idx]

@inline compute_weight(::InnerPlus{ActiveDim}, space, I::CartesianIndex{D}, lin_idx::Int) where {ActiveDim,D} = weights(space, Innerplus(), ActiveDim)[lin_idx]

# ==============================================================================
# User-Facing API & Overloads
# ==============================================================================

"""
    inner_plus(left::NTuple{D,LazyOp{D}}, right::NTuple{D,LazyOp{D}}) where D

Constructs the sum of directional modified \$L^2_+\$ inner products across all dimensions.
"""
function inner_plus(left::NTuple{D,LazyOp{D}}, right::NTuple{D,LazyOp{D}}) where D
	terms = ntuple(dim -> BilinearProduct{D,InnerPlus{dim},typeof(left[dim]),typeof(right[dim])}(left[dim], right[dim]), Val(D))
	return foldl(+, terms)
end

# Support both scalar and NTuple combinations in standard inner products:

"""
    innerₕ(left::LazyOp{D}, right::LazyOp{D}) where D

Constructs a symbolic \$L^2\$ bilinear inner product between `left` and `right`.
"""
innerₕ(left::LazyOp{D}, right::LazyOp{D}) where D = BilinearProduct{D,InnerH,typeof(left),typeof(right)}(left, right)

"""
    inner₊(left::LazyOp{1}, right::LazyOp{1})
    inner₊(left::NTuple{D,LazyOp{D}}, right::NTuple{D,LazyOp{D}}) where D

Constructs a symbolic modified \$L^2_+\$ inner product between `left` and `right`.
"""
inner₊(left::LazyOp{1}, right::LazyOp{1}) = BilinearProduct{1,InnerPlus{1},typeof(left),typeof(right)}(left, right)
inner₊(left::NTuple{D,LazyOp{D}}, right::NTuple{D,LazyOp{D}}) where D = inner_plus(left, right)

"""
    inner₊ₓ(left::LazyOp{D}, right::LazyOp{D}) where D
    inner₊ᵧ(left::LazyOp{D}, right::LazyOp{D}) where D
    inner₊₂(left::LazyOp{D}, right::LazyOp{D}) where D

Constructs directional modified \$L^2_+\$ inner products in x, y, and z directions.
"""
inner₊ₓ(left::LazyOp{D}, right::LazyOp{D}) where D = BilinearProduct{D,InnerPlus{1},typeof(left),typeof(right)}(left, right)
inner₊ᵧ(left::LazyOp{D}, right::LazyOp{D}) where D = BilinearProduct{D,InnerPlus{2},typeof(left),typeof(right)}(left, right)
inner₊₂(left::LazyOp{D}, right::LazyOp{D}) where D = BilinearProduct{D,InnerPlus{3},typeof(left),typeof(right)}(left, right)

@inline function source_number(l::Number, ::Val{D}) where D
	f = x -> l
	return SourceFunction{D, typeof(f)}(f)
end

# Linear Forms (e.g. innerₕ(f, v) where f is a Function, Number, or VectorElement and v is TestFunction)
innerₕ(l::Function, r::LazyOp{D}) where D = LinearProduct{D,InnerH,SourceFunction{D,typeof(l)},typeof(r)}(SourceFunction{D,typeof(l)}(l), r)
innerₕ(l::Number, r::LazyOp{D}) where D = let sf = source_number(l, Val(D)); LinearProduct{D,InnerH,typeof(sf),typeof(r)}(sf, r) end
innerₕ(l::VectorElement, r::LazyOp{D}) where D = LinearProduct{D,InnerH,SourceVector{D,typeof(l.data)},typeof(r)}(SourceVector{D,typeof(l.data)}(l.data), r)

inner₊(l::Function, r::LazyOp{D}) where D = LinearProduct{D,InnerPlus{1},SourceFunction{D,typeof(l)},typeof(r)}(SourceFunction{D,typeof(l)}(l), r)
inner₊(l::Number, r::LazyOp{D}) where D = let sf = source_number(l, Val(D)); LinearProduct{D,InnerPlus{1},typeof(sf),typeof(r)}(sf, r) end
inner₊(l::VectorElement, r::LazyOp{D}) where D = LinearProduct{D,InnerPlus{1},SourceVector{D,typeof(l.data)},typeof(r)}(SourceVector{D,typeof(l.data)}(l.data), r)

inner₊(l::NTuple{D,Function}, r::NTuple{D,LazyOp{D}}) where D = foldl(+, ntuple(dim -> LinearProduct{D,InnerPlus{dim},SourceFunction{D,typeof(l[dim])},typeof(r[dim])}(SourceFunction{D,typeof(l[dim])}(l[dim]), r[dim]), Val(D)))
inner₊(l::NTuple{D,Number}, r::NTuple{D,LazyOp{D}}) where D = foldl(+, ntuple(dim -> let sf = source_number(l[dim], Val(D)); LinearProduct{D,InnerPlus{dim},typeof(sf),typeof(r[dim])}(sf, r[dim]) end, Val(D)))
@inline function inner₊(l::NTuple{D,VectorElement}, r::NTuple{D,LazyOp{D}}) where D
	if is_symbolic(r)
		return foldl(+, ntuple(dim -> LinearProduct{D,InnerPlus{dim},SourceVector{D,typeof(l[dim].data)},typeof(r[dim])}(SourceVector{D,typeof(l[dim].data)}(l[dim].data), r[dim]), Val(D)))
	else
		res = similar(first(l).values)
		res .= 0
		inner₊!(res, l, r)
		return res
	end
end

inner₊ₓ(l::Function, r::LazyOp{D}) where D = LinearProduct{D,InnerPlus{1},SourceFunction{D,typeof(l)},typeof(r)}(SourceFunction{D,typeof(l)}(l), r)
inner₊ᵧ(l::Function, r::LazyOp{D}) where D = LinearProduct{D,InnerPlus{2},SourceFunction{D,typeof(l)},typeof(r)}(SourceFunction{D,typeof(l)}(l), r)
inner₊₂(l::Function, r::LazyOp{D}) where D = LinearProduct{D,InnerPlus{3},SourceFunction{D,typeof(l)},typeof(r)}(SourceFunction{D,typeof(l)}(l), r)

inner₊ₓ(l::Number, r::LazyOp{D}) where D = let sf = source_number(l, Val(D)); LinearProduct{D,InnerPlus{1},typeof(sf),typeof(r)}(sf, r) end
inner₊ᵧ(l::Number, r::LazyOp{D}) where D = let sf = source_number(l, Val(D)); LinearProduct{D,InnerPlus{2},typeof(sf),typeof(r)}(sf, r) end
inner₊₂(l::Number, r::LazyOp{D}) where D = let sf = source_number(l, Val(D)); LinearProduct{D,InnerPlus{3},typeof(sf),typeof(r)}(sf, r) end

inner₊ₓ(l::VectorElement, r::LazyOp{D}) where D = LinearProduct{D,InnerPlus{1},SourceVector{D,typeof(l.data)},typeof(r)}(SourceVector{D,typeof(l.data)}(l.data), r)
inner₊ᵧ(l::VectorElement, r::LazyOp{D}) where D = LinearProduct{D,InnerPlus{2},SourceVector{D,typeof(l.data)},typeof(r)}(SourceVector{D,typeof(l.data)}(l.data), r)
inner₊₂(l::VectorElement, r::LazyOp{D}) where D = LinearProduct{D,InnerPlus{3},SourceVector{D,typeof(l.data)},typeof(r)}(SourceVector{D,typeof(l.data)}(l.data), r)

# ==============================================================================
# Zero-Allocation Stencil Evaluators
# ==============================================================================

@inline function local_stencil(op::BilinearProduct{D,InnerType}, space, I::CartesianIndex{D}, markers, lin_idx::Int) where {D,InnerType}
	left_stencil = local_stencil(op.left_op, space, I, markers, lin_idx)
	right_stencil = local_stencil(op.right_op, space, I, markers, lin_idx)
	vol = compute_weight(InnerType(), space, I, lin_idx)
	return multiply_stencils_bilinear(left_stencil, right_stencil, vol)
end

@inline function local_stencil(op::LinearProduct{D,InnerType}, space, I::CartesianIndex{D}, markers, lin_idx::Int) where {D,InnerType}
	left_stencil = local_stencil(op.left_op, space, I, markers, lin_idx)
	right_stencil = local_stencil(op.right_op, space, I, markers, lin_idx)
	vol = compute_weight(InnerType(), space, I, lin_idx)
	return multiply_stencils_linear(left_stencil, right_stencil, vol)
end

# ==============================================================================
# AST Resolution
# ==============================================================================

resolve_ast(op::BilinearProduct{D,InnerType}) where {D,InnerType} = BilinearProduct{D,InnerType,typeof(resolve_ast(op.left_op)),typeof(resolve_ast(op.right_op))}(resolve_ast(op.left_op), resolve_ast(op.right_op))
resolve_ast(op::LinearProduct{D,InnerType}) where {D,InnerType} = LinearProduct{D,InnerType,typeof(resolve_ast(op.left_op)),typeof(resolve_ast(op.right_op))}(resolve_ast(op.left_op), resolve_ast(op.right_op))
