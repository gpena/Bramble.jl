# average.jl
# Discrete averaging operators for Bramble lazy AST

# ==============================================================================
# Struct Definitions
# ==============================================================================

"""
    BackwardAverage{D,Dim,OpType<:LazyOp{D}} <: LazyOp{D}

An AST node representing a backward spatial averaging operator acting in dimension `Dim`.
"""
struct BackwardAverage{D, Dim, OpType <: LazyOp{D}} <: LazyOp{D}
    inner_op::OpType
end

"""
    ForwardAverage{D,Dim,OpType<:LazyOp{D}} <: LazyOp{D}

An AST node representing a forward spatial averaging operator acting in dimension `Dim`.
"""
struct ForwardAverage{D, Dim, OpType <: LazyOp{D}} <: LazyOp{D}
    inner_op::OpType
end

"""
    ShiftNode{D,Dim,OpType<:LazyOp{D}} <: LazyOp{D}

An AST node representing a stencil shift operation by `shift_amount` grid points in dimension `Dim`.
"""
struct ShiftNode{D, Dim, OpType <: LazyOp{D}} <: LazyOp{D}
    shift_amount::Int
    inner_op::OpType
end

# ==============================================================================
# User-Facing API & Overloads
# ==============================================================================

"""
    avg_backward(op::LazyOp{D}, dim::Int) where D

Applies a backward average operator to `op` in dimension `dim`.
"""
avg_backward(op::LazyOp{D}, dim::Int) where {D} = BackwardAverage{D, dim, typeof(op)}(op)

"""
    avg_forward(op::LazyOp{D}, dim::Int) where D

Applies a forward average operator to `op` in dimension `dim`.
"""
avg_forward(op::LazyOp{D}, dim::Int) where {D} = ForwardAverage{D, dim, typeof(op)}(op)

"""
    shift_op(op::LazyOp{D}, dim::Int, amount::Int) where D

Shifts the stencil of `op` by `amount` grid points in dimension `dim`.
"""
function shift_op(op::LazyOp{D}, dim::Int, amount::Int) where {D}
    ShiftNode{D, dim, typeof(op)}(amount, op)
end

# AST-based average operators (distinct names)

"""
    M₋ₓ(op::LazyOp{D}) where D
    M₊ₓ(op::LazyOp{D}) where D
    M₋ᵧ(op::LazyOp{D}) where D
    M₊ᵧ(op::LazyOp{D}) where D
    M₋₂(op::LazyOp{D}) where D
    M₊₂(op::LazyOp{D}) where D

Symbolic averaging operators in specified coordinate directions (x, y, z).
"""
M₋ₓ(op::LazyOp{D}) where {D} = BackwardAverage{D, 1, typeof(op)}(op)
M₊ₓ(op::LazyOp{D}) where {D} = ForwardAverage{D, 1, typeof(op)}(op)
M₋ᵧ(op::LazyOp{D}) where {D} = BackwardAverage{D, 2, typeof(op)}(op)
M₊ᵧ(op::LazyOp{D}) where {D} = ForwardAverage{D, 2, typeof(op)}(op)
M₋₂(op::LazyOp{D}) where {D} = BackwardAverage{D, 3, typeof(op)}(op)
M₊₂(op::LazyOp{D}) where {D} = ForwardAverage{D, 3, typeof(op)}(op)

"""
    vectorial_avg_backward(op::LazyOp{D}) where D

Applies backward spatial averaging component-wise across all dimensions.
"""
function vectorial_avg_backward(op::LazyOp{D}) where {D}
    ntuple(dim -> BackwardAverage{D, dim, typeof(op)}(op), Val(D))
end
vectorial_avg_backward(op::LazyOp{1}) = BackwardAverage{1, 1, typeof(op)}(op)

"""
    vectorial_avg_forward(op::LazyOp{D}) where D

Applies forward spatial averaging component-wise across all dimensions.
"""
function vectorial_avg_forward(op::LazyOp{D}) where {D}
    ntuple(dim -> ForwardAverage{D, dim, typeof(op)}(op), Val(D))
end
vectorial_avg_forward(op::LazyOp{1}) = ForwardAverage{1, 1, typeof(op)}(op)

"""
    M₋ₕ(op::LazyOp{D}) where D

Symbolic backward spatial averaging operator tuple.
"""
M₋ₕ(op::LazyOp{D}) where {D} = vectorial_avg_backward(op)

"""
    M₊ₕ(op::LazyOp{D}) where D

Symbolic forward spatial averaging operator tuple.
"""
M₊ₕ(op::LazyOp{D}) where {D} = vectorial_avg_forward(op)

# ==============================================================================
# Zero-Allocation Stencil Evaluators
# ==============================================================================

@inline function local_stencil(op::BackwardAverage{D, Dim}, space, I::CartesianIndex{D},
        markers, lin_idx::Int) where {D, Dim}
    inner = local_stencil(op.inner_op, space, I, markers, lin_idx)

    T = eltype(space)
    mask = I[Dim] == 1 ? zero(T) : T(1) / 2
    t1 = scale_stencil(inner, mask)

    inner_shifted = shifted_inner_stencil(op.inner_op, inner, space, I, markers,
        Val(Dim), Val(-1))
    t2 = scale_stencil(inner_shifted, mask)

    return concatenate_stencils(t1, t2)
end

@inline function local_stencil(op::ForwardAverage{D, Dim}, space, I::CartesianIndex{D},
        markers, lin_idx::Int) where {D, Dim}
    inner = local_stencil(op.inner_op, space, I, markers, lin_idx)
    m = mesh(space)
    dims = npoints(m, Tuple)

    T = eltype(space)
    mask = I[Dim] == dims[Dim] ? zero(T) : T(1) / 2
    inner_shifted = shifted_inner_stencil(op.inner_op, inner, space, I, markers,
        Val(Dim), Val(1))
    t1 = scale_stencil(inner_shifted, mask)
    t2 = scale_stencil(inner, mask)

    return concatenate_stencils(t1, t2)
end

@inline function local_stencil(op::ShiftNode{D, Dim}, space, I::CartesianIndex{D},
        markers, lin_idx::Int) where {D, Dim}
    inner = local_stencil(op.inner_op, space, I, markers, lin_idx)
    return _shift_node_stencil(
        stencil_shift_trait(op.inner_op), op, inner, space, I, markers)
end

@inline _shift_node_stencil(::TranslationInvariantStencil, op::ShiftNode{D, Dim}, inner,
    space, I::CartesianIndex{D}, markers) where {D, Dim} = shift_stencil(
    inner, Val(Dim), op.shift_amount)

# `shift_op` has no mask of its own: every other wrapper that reaches a neighbour
# (differences, averages, jumps) computes one first and multiplies a clamped boundary read by
# it, which is what makes `_clamped_shift`'s "clamp now, a zero mask absorbs it" contract safe
# for them. Nothing here would absorb it for a source: relabelling an offset is safe
# unclamped, since the caller's own bounds check drops the whole entry when the offset lands
# out of range, but a source has already been reduced to a value by the time this runs, with
# no offset left for that check. A source shifted off the grid therefore reads as zero here:
# an empty stencil, the same "missing neighbour is zero" convention the masked stencils use,
# mirroring how `RegionRestriction` already spells "contributes nothing here".
#
# An interpolation is not a source, and clamping is its own correct behaviour: `locate_cell`
# (`space/operators/interpolation.jl`) clamps every point it is given, in-grid or not, by
# design (`πₕ`'s own docstring calls this extrapolation along the boundary cell's slope, not
# a missing-neighbour convention to override). So only a source-only inner operand gets the
# in-grid check; anything else falls through to the ordinary clamped re-evaluation.
@inline function _shift_node_stencil(::PointDependentStencil, op::ShiftNode{D, Dim}, inner,
        space, I::CartesianIndex{D}, markers) where {D, Dim}
    if _is_source_only(op.inner_op)
        Ishift = I + _stencil_step(Val(Dim), Val(D)) * op.shift_amount
        _in_grid(space, Ishift) || return ()
        return local_stencil(op.inner_op, space, Ishift, markers,
            LinearIndices(indices(mesh(space)))[Ishift])
    else
        return shifted_inner_stencil(op.inner_op, inner, space, I, markers, Val(Dim),
            op.shift_amount)
    end
end

# ==============================================================================
# AST Resolution
# ==============================================================================

"""
    AverageNode{D, Dim}

Either average node over a `D`-dimensional space, averaging along `Dim`.

The pair carries the same parameters and differs only in which neighbour it reaches, so
anything reading the direction rather than choosing a stencil is written against this alias.
"""
const AverageNode{D, Dim} = Union{BackwardAverage{D, Dim}, ForwardAverage{D, Dim}}

# The direction an operator works along, for the nodes that name one. An average carries
# `Dim` exactly as a difference does, and a shift likewise; a restriction has whatever its
# child has. The gap was found by the precompilation workload, which calls the trait across
# every node kind and met a MethodError on the averages.
Bramble.get_innermost_dim(op::AverageNode{D, Dim}) where {D, Dim} = Dim
Bramble.get_innermost_dim(op::ShiftNode{D, Dim}) where {D, Dim} = Dim

function resolve_ast(op::BackwardAverage{D, Dim}) where {D, Dim}
    BackwardAverage{D, Dim, typeof(resolve_ast(op.inner_op))}(resolve_ast(op.inner_op))
end
function resolve_ast(op::ForwardAverage{D, Dim}) where {D, Dim}
    ForwardAverage{D, Dim, typeof(resolve_ast(op.inner_op))}(resolve_ast(op.inner_op))
end
function resolve_ast(op::ShiftNode{D, Dim}) where {D, Dim}
    ShiftNode{D, Dim, typeof(resolve_ast(op.inner_op))}(op.shift_amount, resolve_ast(op.inner_op))
end
