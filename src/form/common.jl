# common.jl

# ==============================================================================
# 1. Zero-Allocation Spacing & Tuple Utilities
# ==============================================================================

@inline _get_component(x::Tuple, dim::Int) = x[dim]
@inline _get_component(x::Number, dim::Int) = x

"""
    get_spacing(mesh, I, dim::Int)

Gets the grid spacing in a given coordinate direction `dim` at Cartesian index `I`.
"""
@inline get_spacing(mesh, I, dim::Int) = _get_component(spacing(mesh, I), dim)

"""
    get_forward_spacing(mesh, I, dim::Int)

Gets the forward grid spacing in a given coordinate direction `dim` at Cartesian index `I`.
"""
@inline get_forward_spacing(mesh, I, dim::Int) = _get_component(forward_spacing(mesh, I), dim)

"""
    get_half_spacing(mesh, I, dim::Int)

Gets the half-grid spacing in a given coordinate direction `dim` at Cartesian index `I`.
"""
@inline get_half_spacing(mesh, I, dim::Int) = _get_component(half_spacing(mesh, I), dim)

"""
    shift_offset(offset::NTuple{D,Int}, dim::Int, delta::Int) where D

Shifts a Cartesian offset tuple by `delta` in dimension `dim`.
"""
@inline shift_offset(offset::NTuple{D,Int}, dim::Int, delta::Int) where D = ntuple(i -> i == dim ? offset[i] + delta : offset[i], Val(D))

"""
    zero_offset(::Val{D}) where D

Returns a D-tuple of zeros.
"""
@inline zero_offset(::Val{D}) where D = ntuple(x -> 0, Val(D))

"""
    shift_stencil(inner::Tuple, ::Val{Dim}, delta)

Shifts all coordinates in a stencil tuple by `delta` in dimension `Dim`.
"""
@generated function shift_stencil(inner::Tuple, ::Val{Dim}, ::Val{Delta}) where {Dim,Delta}
	N = length(inner.parameters)
	exprs = Expr[]
	for i in 1:N
		push!(exprs, :((shift_offset(inner[$i][1], Dim, Delta), inner[$i][2])))
	end
	return Expr(:tuple, exprs...)
end


@generated function shift_stencil(inner::Tuple, ::Val{Dim}, delta::Int) where {Dim}
	N = length(inner.parameters)
	exprs = Expr[]
	for i in 1:N
		push!(exprs, :((shift_offset(inner[$i][1], Dim, delta), inner[$i][2])))
	end
	return Expr(:tuple, exprs...)
end

@generated function concatenate_stencils(left::Tuple, right::Tuple)
	N_left = length(left.parameters)
	N_right = length(right.parameters)
	exprs = Expr[]
	for i in 1:N_left
		push!(exprs, :(left[$i]))
	end
	for i in 1:N_right
		push!(exprs, :(right[$i]))
	end
	return Expr(:tuple, exprs...)
end

@generated function multiply_stencils_bilinear(left::Tuple, right::Tuple, vol::Number)
	N_l = length(left.parameters)
	N_r = length(right.parameters)
	exprs = Expr[]
	for i in 1:N_l, j in 1:N_r
		push!(exprs, :((left[$i][1], right[$j][1], left[$i][2] * right[$j][2] * vol)))
	end
	return Expr(:tuple, exprs...)
end

@generated function multiply_stencils_linear(left::Tuple, right::Tuple, vol::Number)
	N_l = length(left.parameters)
	N_r = length(right.parameters)
	exprs = Expr[]
	for i in 1:N_l, j in 1:N_r
		push!(exprs, :((right[$j][1], left[$i][2] * right[$j][2] * vol)))
	end
	return Expr(:tuple, exprs...)
end

@generated function scale_stencil(inner::Tuple, scalar::Number)
	N = length(inner.parameters)
	exprs = Expr[]
	for i in 1:N
		push!(exprs, :((Base.front(inner[$i])..., inner[$i][end] * scalar)))
	end
	return Expr(:tuple, exprs...)
end

# ==============================================================================
# 2. Abstract Syntax Tree (AST) Nodes
# ==============================================================================
# 2. Abstract Syntax Tree (AST) Nodes
# ==============================================================================

"""
    TrialFunction{D} <: LazyOp{D}

An AST node representing the symbolic trial function \$u\$ in a bilinear form.
"""
struct TrialFunction{D} <: LazyOp{D} end

"""
    TestFunction{D} <: LazyOp{D}

An AST node representing the symbolic test function \$v\$ in a form.
"""
struct TestFunction{D} <: LazyOp{D} end

"""
    SourceFunction{D,F} <: LazyOp{D}

An AST node representing a source term defined by a continuous function.
"""
struct SourceFunction{D,F} <: LazyOp{D}
	func::F
end

"""
    SourceVector{D,VType} <: LazyOp{D}

An AST node representing a source term defined by a discrete vector of values.
"""
struct SourceVector{D,VType<:AbstractVector} <: LazyOp{D}
	vec::VType
end

# ==============================================================================
# 3. Form API & Bramble Standard Mapping
# ==============================================================================

"""
    trial_function(::Val{D}) where D

Constructs a `TrialFunction` of dimension `D`.
"""
trial_function(::Val{D}) where D = TrialFunction{D}()

"""
    test_function(::Val{D}) where D

Constructs a `TestFunction` of dimension `D`.
"""
test_function(::Val{D}) where D = TestFunction{D}()

"""
    source_function(f, ::Val{D}) where D

Constructs a `SourceFunction` wrapping function `f`.
"""
source_function(f, ::Val{D}) where D = SourceFunction{D,typeof(f)}(f)


# Import modularized operator and product logic
include("operators/difference.jl")
include("operators/average.jl")
include("operators/restriction.jl")
include("operators/inner.jl")

# ==============================================================================
# 4. Zero-Allocation Stencil Evaluators
# ==============================================================================

@inline local_stencil(::TrialFunction{D}, space, I::CartesianIndex{D}, markers, lin_idx::Int) where D = ((zero_offset(Val(D)), 1.0),)
@inline local_stencil(::TestFunction{D}, space, I::CartesianIndex{D}, markers, lin_idx::Int) where D = ((zero_offset(Val(D)), 1.0),)

@inline function local_stencil(op::SourceFunction{D}, space, I::CartesianIndex{D}, markers, lin_idx::Int) where D
	m = mesh(space)
	x = point(m, I)
	return ((zero_offset(Val(D)), op.func(x)),)
end

@inline function local_stencil(op::SourceVector{D}, space, I::CartesianIndex{D}, markers, lin_idx::Int) where D
	return ((zero_offset(Val(D)), op.vec[lin_idx]),)
end

@inline function local_stencil(op::OperatorAdd, space, I::CartesianIndex{D}, markers, lin_idx::Int) where D
	left_stencil = local_stencil(op.left_op, space, I, markers, lin_idx)
	right_stencil = local_stencil(op.right_op, space, I, markers, lin_idx)
	return concatenate_stencils(left_stencil, right_stencil)
end

@inline function local_stencil(op::OperatorScale, space, I::CartesianIndex{D}, markers, lin_idx::Int) where D
	inner = local_stencil(op.inner_op, space, I, markers, lin_idx)
	return scale_stencil(inner, op.scalar)
end

@inline function local_stencil(op::GridFunctionScale, space, I::CartesianIndex{D}, markers, lin_idx::Int) where D
	inner = local_stencil(op.inner_op, space, I, markers, lin_idx)
	grid_fn = op.grid_function
	local_val = if grid_fn isa Function
		val = grid_fn()
		val isa Number ? val : val[lin_idx]
	else
		grid_fn isa Number ? grid_fn : grid_fn[lin_idx]
	end
	return scale_stencil(inner, local_val)
end

@inline local_stencil(op::IdentityOperator{D}, space, I::CartesianIndex{D}, markers, lin_idx::Int) where D = ((zero_offset(Val(D)), 1.0),)
@inline local_stencil(op::ZeroOperator{D}, space, I::CartesianIndex{D}, markers, lin_idx::Int) where D = ((zero_offset(Val(D)), 0.0),)

# ==============================================================================
# 5. AST Resolution & Thunk Eval
# ==============================================================================

resolve_ast(op::TrialFunction) = op
resolve_ast(op::TestFunction) = op
resolve_ast(op::SourceFunction) = op
resolve_ast(op::SourceVector) = op

resolve_ast(op::OperatorAdd{D}) where D = OperatorAdd{D,typeof(resolve_ast(op.left_op)),typeof(resolve_ast(op.right_op))}(resolve_ast(op.left_op), resolve_ast(op.right_op))
resolve_ast(op::OperatorScale{D}) where {D} = OperatorScale{D,typeof(op.scalar),typeof(resolve_ast(op.inner_op))}(op.scalar, resolve_ast(op.inner_op))

resolve_ast(op::GridFunctionScale{D,VType}) where {D,VType} = GridFunctionScale{D,VType,typeof(resolve_ast(op.inner_op))}(op.grid_function, resolve_ast(op.inner_op))

function resolve_ast(op::GridFunctionScale{D,<:Function}) where D
	vec = op.grid_function()
	return GridFunctionScale{D,typeof(vec),typeof(resolve_ast(op.inner_op))}(vec, resolve_ast(op.inner_op))
end

resolve_ast(op::IdentityOperator) = op
resolve_ast(op::ZeroOperator) = op

resolve_ast(ops::NTuple{N,Any}) where N = map(resolve_ast, ops)
resolve_ast(op::Any) = op

# ==============================================================================
# 6. Symbolic AST Traits
# ==============================================================================

# Note: is_symbolic base function is declared in linear_operators.jl

is_symbolic(::TrialFunction) = true
is_symbolic(::TestFunction) = true
is_symbolic(::SourceFunction) = true
is_symbolic(::SourceVector) = true
is_symbolic(op::BilinearProduct) = true
is_symbolic(op::LinearProduct) = true

is_symbolic(op::BackwardDifference) = is_symbolic(op.inner_op)
is_symbolic(op::ForwardDifference) = is_symbolic(op.inner_op)

is_symbolic(op::BackwardAverage) = is_symbolic(op.inner_op)
is_symbolic(op::ForwardAverage) = is_symbolic(op.inner_op)
is_symbolic(op::ShiftNode) = is_symbolic(op.inner_op)

is_symbolic(op::RegionRestriction) = is_symbolic(op.inner_op)