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

    mask = I[Dim] == 1 ? 0 : 1
    t1 = scale_stencil(inner, mask / h)

    inner_shifted = shifted_inner_stencil(op.inner_op, inner, space, I, markers,
        Val(Dim), Val(-1))
    t2 = scale_stencil(inner_shifted, -mask / h)

    return concatenate_stencils(t1, t2)
end

@inline function local_stencil(op::ForwardDifference{D, Dim}, space, I::CartesianIndex{D},
        markers, lin_idx::Int) where {D, Dim}
    inner = local_stencil(op.inner_op, space, I, markers, lin_idx)
    m = mesh(space)
    dims = npoints(m, Tuple)
    h = get_forward_spacing(m, I, Dim)

    mask = I[Dim] == dims[Dim] ? 0 : 1
    inner_shifted = shifted_inner_stencil(op.inner_op, inner, space, I, markers,
        Val(Dim), Val(1))
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

Bramble.get_innermost_dim(op::DifferenceNode{D, Dim}) where {D, Dim} = Dim

# ==============================================================================
# The remaining difference families
# ==============================================================================
#
# Three more differences, each a symbolic counterpart of an operator the space layer
# already provides. These nodes assemble through their stencils; the space layer's matrix
# forms of the same three operators are what those stencils are tested against.
#
# The boundary convention is the one the one-sided nodes already use: the offsets stay and
# the coefficients go to zero. A truncated point contributes nothing while the stencil
# keeps the same shape, which is what lets the assembly loop stay branch-free.
#
# Writing `h` for `spacing` (xᵢ - xᵢ₋₁) and `hf` for `forward_spacing` (xᵢ₊₁ - xᵢ), as the
# space layer's own docstrings do.

"""
    CenteredDifference{D,Dim,OpType<:LazyOp{D}} <: LazyOp{D}

An AST node for the centered difference along `Dim`,

```math
Dc(u)_i = \\frac{u_{i+1} - u_{i-1}}{h_i + h_{i+1}}
```

Truncated at both ends of `Dim`, having no neighbour on one side.
"""
struct CenteredDifference{D, Dim, OpType <: LazyOp{D}} <: LazyOp{D}
    inner_op::OpType
end

"""
    StarDifference{D,Dim,OpType<:LazyOp{D}} <: LazyOp{D}

An AST node for the starred forward difference along `Dim`,

```math
D^{*}_{+}(u)_i = \\frac{u_{i+1} - u_i}{(h_i + h_{i+1})/2}
```

The forward difference over the *averaged* spacing rather than the forward one, which is
what makes the discrete integration by parts close. Truncated at the far end of `Dim`.
"""
struct StarDifference{D, Dim, OpType <: LazyOp{D}} <: LazyOp{D}
    inner_op::OpType
end

"""
    CrossWeightedDifference{D,Dim,OpType<:LazyOp{D}} <: LazyOp{D}

An AST node for the cross-weighted centered difference along `Dim`,

```math
D_h(u)_i = \\frac{h_i}{h_i + h_{i+1}} D_{-}(u)_{i+1}
         + \\frac{h_{i+1}}{h_i + h_{i+1}} D_{-}(u)_i
```

The same two one-sided differences the centered difference combines, weighted by the
*opposite* spacings. That swap is what makes it second order on a non-uniform grid where
`Dc` is first, and the two coincide when the spacing is constant. Truncated at both ends.
"""
struct CrossWeightedDifference{D, Dim, OpType <: LazyOp{D}} <: LazyOp{D}
    inner_op::OpType
end

"""
    Dcₓ(op::LazyOp{D}) where D
    Dcᵧ(op::LazyOp{D}) where D
    Dc₂(op::LazyOp{D}) where D

Symbolic centered differences in the coordinate directions.
"""
Dcₓ(op::LazyOp{D}) where {D} = CenteredDifference{D, 1, typeof(op)}(op)
Dcᵧ(op::LazyOp{D}) where {D} = CenteredDifference{D, 2, typeof(op)}(op)
Dc₂(op::LazyOp{D}) where {D} = CenteredDifference{D, 3, typeof(op)}(op)

"""
    Dstar₊ₓ(op::LazyOp{D}) where D
    Dstar₊ᵧ(op::LazyOp{D}) where D
    Dstar₊₂(op::LazyOp{D}) where D

Symbolic starred forward differences in the coordinate directions.
"""
Dstar₊ₓ(op::LazyOp{D}) where {D} = StarDifference{D, 1, typeof(op)}(op)
Dstar₊ᵧ(op::LazyOp{D}) where {D} = StarDifference{D, 2, typeof(op)}(op)
Dstar₊₂(op::LazyOp{D}) where {D} = StarDifference{D, 3, typeof(op)}(op)

"""
    Dₕₓ(op::LazyOp{D}) where D
    Dₕᵧ(op::LazyOp{D}) where D
    Dₕ₂(op::LazyOp{D}) where D

Symbolic cross-weighted centered differences in the coordinate directions.
"""
Dₕₓ(op::LazyOp{D}) where {D} = CrossWeightedDifference{D, 1, typeof(op)}(op)
Dₕᵧ(op::LazyOp{D}) where {D} = CrossWeightedDifference{D, 2, typeof(op)}(op)
Dₕ₂(op::LazyOp{D}) where {D} = CrossWeightedDifference{D, 3, typeof(op)}(op)

"""
    Dcₕ(op::LazyOp{D}) where D
    Dstar₊ₕ(op::LazyOp{D}) where D
    ∇ₕ(op::LazyOp{D}) where D

The vector forms: every direction at once, as a `D`-tuple of nodes. In one dimension there
is only one direction, so the node itself is returned rather than a one-element tuple,
as `∇₋ₕ` and `∇₊ₕ` already do.
"""
Dcₕ(op::LazyOp{1}) = Dcₓ(op)
function Dcₕ(op::LazyOp{D}) where {D}
    ntuple(
        dim -> CenteredDifference{D, dim, typeof(op)}(op), Val(D))
end

Dstar₊ₕ(op::LazyOp{1}) = Dstar₊ₓ(op)
function Dstar₊ₕ(op::LazyOp{D}) where {D}
    ntuple(
        dim -> StarDifference{D, dim, typeof(op)}(op), Val(D))
end

∇ₕ(op::LazyOp{1}) = Dₕₓ(op)
function ∇ₕ(op::LazyOp{D}) where {D}
    ntuple(
        dim -> CrossWeightedDifference{D, dim, typeof(op)}(op), Val(D))
end

# --- Stencils --------------------------------------------------------------------- #

@inline function local_stencil(op::CenteredDifference{D, Dim}, space,
        I::CartesianIndex{D}, markers, lin_idx::Int) where {D, Dim}
    inner = local_stencil(op.inner_op, space, I, markers, lin_idx)
    m = mesh(space)
    dims = npoints(m, Tuple)

    # no neighbour on one side at either end
    mask = (I[Dim] == 1 || I[Dim] == dims[Dim]) ? 0 : 1
    c = mask / (get_spacing(m, I, Dim) + get_forward_spacing(m, I, Dim))

    forward = scale_stencil(
        shifted_inner_stencil(op.inner_op, inner, space, I, markers, Val(Dim), Val(1)), c)
    backward = scale_stencil(
        shifted_inner_stencil(op.inner_op, inner, space, I, markers, Val(Dim), Val(-1)), -c)
    return concatenate_stencils(forward, backward)
end

@inline function local_stencil(op::StarDifference{D, Dim}, space, I::CartesianIndex{D},
        markers, lin_idx::Int) where {D, Dim}
    inner = local_stencil(op.inner_op, space, I, markers, lin_idx)
    m = mesh(space)
    dims = npoints(m, Tuple)

    mask = I[Dim] == dims[Dim] ? 0 : 1
    # the averaged spacing, which is what the starred difference divides by
    c = 2 * mask / (get_spacing(m, I, Dim) + get_forward_spacing(m, I, Dim))

    forward = scale_stencil(
        shifted_inner_stencil(op.inner_op, inner, space, I, markers, Val(Dim), Val(1)), c)
    here = scale_stencil(inner, -c)
    return concatenate_stencils(forward, here)
end

# Expanding the definition over the two one-sided differences gives a three-point stencil.
# With S = h + hf,
#
#   D_h(u)_i = h/(S·hf) · u_{i+1} + (hf/(S·h) - h/(S·hf)) · u_i - hf/(S·h) · u_{i-1}
#
# which is where the two coefficients below come from: `a` is the weight of the forward
# neighbour and `b` the magnitude of the backward one.
@inline function local_stencil(op::CrossWeightedDifference{D, Dim}, space,
        I::CartesianIndex{D}, markers, lin_idx::Int) where {D, Dim}
    inner = local_stencil(op.inner_op, space, I, markers, lin_idx)
    m = mesh(space)
    dims = npoints(m, Tuple)

    mask = (I[Dim] == 1 || I[Dim] == dims[Dim]) ? 0 : 1
    h = get_spacing(m, I, Dim)
    hf = get_forward_spacing(m, I, Dim)
    total = h + hf

    a = mask * h / (total * hf)
    b = mask * hf / (total * h)

    forward = scale_stencil(
        shifted_inner_stencil(op.inner_op, inner, space, I, markers, Val(Dim), Val(1)), a)
    here = scale_stencil(inner, b - a)
    backward = scale_stencil(
        shifted_inner_stencil(op.inner_op, inner, space, I, markers, Val(Dim), Val(-1)), -b)
    return concatenate_stencils(concatenate_stencils(forward, here), backward)
end

# --- Traits ----------------------------------------------------------------------- #

"""
    ExtendedDifferenceNode{D, Dim}

The three difference nodes that are neither one-sided nor a jump, differencing along `Dim`.
Grouped so that everything reading only the direction off a node covers all of them at
once, as `DifferenceNode` does for the one-sided pair.
"""
const ExtendedDifferenceNode{D, Dim} = Union{CenteredDifference{D, Dim},
    StarDifference{D, Dim}, CrossWeightedDifference{D, Dim}}

Bramble.get_innermost_dim(op::ExtendedDifferenceNode{D, Dim}) where {D, Dim} = Dim

function resolve_ast(op::CenteredDifference{D, Dim}) where {D, Dim}
    inner = resolve_ast(op.inner_op)
    return CenteredDifference{D, Dim, typeof(inner)}(inner)
end
function resolve_ast(op::StarDifference{D, Dim}) where {D, Dim}
    inner = resolve_ast(op.inner_op)
    return StarDifference{D, Dim, typeof(inner)}(inner)
end
function resolve_ast(op::CrossWeightedDifference{D, Dim}) where {D, Dim}
    inner = resolve_ast(op.inner_op)
    return CrossWeightedDifference{D, Dim, typeof(inner)}(inner)
end
