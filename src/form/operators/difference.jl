# difference.jl
# Discrete finite difference operators for Bramble lazy AST

# ==============================================================================
# Struct Definitions
# ==============================================================================

"""
    BackwardDifference{D,Dim,OpType<:LazyOp{D}} <: LazyOp{D}

An AST node representing a backward finite difference operator acting in dimension `Dim`.
"""
struct BackwardDifference{D, Dim, OpType <: LazyOp{D}} <: LazyOp{D}
    inner_op::OpType
end

"""
    ForwardDifference{D,Dim,OpType<:LazyOp{D}} <: LazyOp{D}

An AST node representing a forward finite difference operator acting in dimension `Dim`.
"""
struct ForwardDifference{D, Dim, OpType <: LazyOp{D}} <: LazyOp{D}
    inner_op::OpType
end

# ==============================================================================
# User-Facing API & Overloads
# ==============================================================================

"""
    grad_backward(op::LazyOp{D}) where D

Constructs a backward gradient operator tuple, yielding `D`-tuple of `BackwardDifference` operators.
"""
grad_backward(op::LazyOp{1}) = BackwardDifference{1, 1, typeof(op)}(op)
function grad_backward(op::LazyOp{D}) where {D}
    ntuple(dim -> BackwardDifference{D, dim, typeof(op)}(op), Val(D))
end

"""
    grad_forward(op::LazyOp{D}) where D

Constructs a forward gradient operator tuple, yielding `D`-tuple of `ForwardDifference` operators.
"""
grad_forward(op::LazyOp{1}) = ForwardDifference{1, 1, typeof(op)}(op)
function grad_forward(op::LazyOp{D}) where {D}
    ntuple(dim -> ForwardDifference{D, dim, typeof(op)}(op), Val(D))
end

# Add standard Bramble operator overloads mapped to the fast lazy AST:

"""
    ∇₋ₕ(op::LazyOp{D}) where D

Symbolic backward gradient operator.
"""
∇₋ₕ(op::LazyOp{D}) where {D} = grad_backward(op)

"""
    ∇₋ₕ(ops::Tuple)

Applies the backward gradient component-wise to a **tuple** of scalar symbolic
functions (e.g. the velocity components `(u1, u2)` of a composite space).
Returns a tuple of gradient tuples, one per component.
"""
∇₋ₕ(ops::Tuple) = map(grad_backward, ops)

"""
    ∇₊ₕ(op::LazyOp{D}) where D

Symbolic forward gradient operator.
"""
∇₊ₕ(op::LazyOp{D}) where {D} = grad_forward(op)

"""
    ∇₊ₕ(ops::Tuple)

Applies the forward gradient component-wise to a **tuple** of scalar symbolic
functions, as `∇₋ₕ` does. Returns a tuple of gradient tuples, one per component.
"""
∇₊ₕ(ops::Tuple) = map(grad_forward, ops)

# AST-based difference operators (distinct names)

"""
    D₋ₓ(op::LazyOp{D}) where D
    D₊ₓ(op::LazyOp{D}) where D
    D₋ᵧ(op::LazyOp{D}) where D
    D₊ᵧ(op::LazyOp{D}) where D
    D₋₂(op::LazyOp{D}) where D
    D₊₂(op::LazyOp{D}) where D

Symbolic finite difference operators in specified coordinate directions (x, y, z).
"""
D₋ₓ(op::LazyOp{D}) where {D} = BackwardDifference{D, 1, typeof(op)}(op)
D₊ₓ(op::LazyOp{D}) where {D} = ForwardDifference{D, 1, typeof(op)}(op)
D₋ᵧ(op::LazyOp{D}) where {D} = BackwardDifference{D, 2, typeof(op)}(op)
D₊ᵧ(op::LazyOp{D}) where {D} = ForwardDifference{D, 2, typeof(op)}(op)
D₋₂(op::LazyOp{D}) where {D} = BackwardDifference{D, 3, typeof(op)}(op)
D₊₂(op::LazyOp{D}) where {D} = ForwardDifference{D, 3, typeof(op)}(op)

# ==============================================================================
# Zero-Allocation Stencil Evaluators
# ==============================================================================

@inline function local_stencil(op::BackwardDifference{D, Dim}, space,
        I::CartesianIndex{D}, markers, lin_idx::Int) where {D, Dim}
    inner = local_stencil(op.inner_op, space, I, markers, lin_idx)
    m = mesh(space)
    h = get_spacing(m, I, Dim)

    mask = I[Dim] == 1 ? 0.0 : 1.0
    t1 = scale_stencil(inner, mask / h)

    inner_shifted = shift_stencil(inner, Val(Dim), Val(-1))
    t2 = scale_stencil(inner_shifted, -mask / h)

    return concatenate_stencils(t1, t2)
end

@inline function local_stencil(op::ForwardDifference{D, Dim}, space, I::CartesianIndex{D},
        markers, lin_idx::Int) where {D, Dim}
    inner = local_stencil(op.inner_op, space, I, markers, lin_idx)
    m = mesh(space)
    dims = npoints(m, Tuple)
    h = get_forward_spacing(m, I, Dim)

    mask = I[Dim] == dims[Dim] ? 0.0 : 1.0
    inner_shifted = shift_stencil(inner, Val(Dim), Val(1))
    t1 = scale_stencil(inner_shifted, mask / h)
    t2 = scale_stencil(inner, -mask / h)

    return concatenate_stencils(t1, t2)
end

# ==============================================================================
# AST Resolution
# ==============================================================================

function resolve_ast(op::BackwardDifference{D, Dim}) where {D, Dim}
    BackwardDifference{D, Dim, typeof(resolve_ast(op.inner_op))}(resolve_ast(op.inner_op))
end
function resolve_ast(op::ForwardDifference{D, Dim}) where {D, Dim}
    ForwardDifference{D, Dim, typeof(resolve_ast(op.inner_op))}(resolve_ast(op.inner_op))
end

# ==============================================================================
# Direct integration helpers for linear_operators.jl
# ==============================================================================

"""
	DifferenceNode{D, Dim}

Either one-sided difference node over a `D`-dimensional space, differencing along `Dim`.

The two carry the same parameters, so anything that reads only the *direction* off the
node is written against this alias and stays symmetric between them by construction.

Not everything can be: `inner₊` takes backward differences alone, because the staggered
weights it carries are the ones the summation-by-parts identity pairs with a backward
difference. Use this alias where the distinction genuinely does not arise.
"""
const DifferenceNode{D, Dim} = Union{BackwardDifference{D, Dim}, ForwardDifference{D, Dim}}

# `backward_difference_matrix` was never a function — no revision of the package defines
# it. The generated names for the differences *including* their 1/h weights, which is what
# these nodes stand for, are `backward_finite_difference` and `forward_finite_difference`,
# and they want the direction as a `Val`; `Dim` here is a plain `Int`.
#
# The scale is `1` rather than `1.0` so that it promotes against whatever element type the
# space has, instead of dragging a Float32 or an extended-precision assembly up to Float64.
function Bramble.get_derivative_matrix_and_scale(
        op::BackwardDifference{D, Dim}, W) where {D, Dim}
    return backward_finite_difference(W, Val(Dim)), 1
end

function Bramble.get_derivative_matrix_and_scale(
        op::ForwardDifference{D, Dim}, W) where {D, Dim}
    return forward_finite_difference(W, Val(Dim)), 1
end

Bramble.get_innermost_dim(op::DifferenceNode{D, Dim}) where {D, Dim} = Dim
