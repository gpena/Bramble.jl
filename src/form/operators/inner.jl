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
struct BilinearProduct{
    D, InnerType <: AbstractInnerProduct, LeftType <: LazyOp{D}, RightType <: LazyOp{D}} <:
       LazyOp{D}
    left_op::LeftType
    right_op::RightType
end

"""
    LinearProduct{D,InnerType,LeftType,RightType} <: LazyOp{D}

An AST node representing a linear integration term \$(f, v)\$ in a linear form.
"""
struct LinearProduct{
    D, InnerType <: AbstractInnerProduct, LeftType <: LazyOp{D}, RightType <: LazyOp{D}} <:
       LazyOp{D}
    left_op::LeftType
    right_op::RightType
end

# ==============================================================================
# Weight Helpers
# ==============================================================================

@inline compute_weight(::InnerH, space, I::CartesianIndex{D}, lin_idx::Int) where {D} = weights(
    space, Innerh())[lin_idx]

@inline compute_weight(::InnerPlus{ActiveDim}, space, I::CartesianIndex{D},
    lin_idx::Int) where {ActiveDim, D} = weights(space, Innerplus(), ActiveDim)[lin_idx]

# ==============================================================================
# User-Facing API & Overloads
# ==============================================================================

"""
    inner_plus(left::NTuple{D,LazyOp{D}}, right::NTuple{D,LazyOp{D}}) where D

Constructs the sum of directional modified \$L^2_+\$ inner products across all dimensions.
"""
function inner_plus(left::NTuple{D, LazyOp{D}}, right::NTuple{D, LazyOp{D}}) where {D}
    terms = ntuple(
        dim -> BilinearProduct{D, InnerPlus{dim}, typeof(left[dim]), typeof(right[dim])}(left[dim], right[dim]),
        Val(D))
    return foldl(+, terms)
end

# Support both scalar and NTuple combinations in standard inner products:

"""
    innerₕ(left::LazyOp{D}, right::LazyOp{D}) where D

Constructs a symbolic \$L^2\$ bilinear inner product between `left` and `right`.
"""
function innerₕ(left::LazyOp{D}, right::LazyOp{D}) where {D}
    BilinearProduct{D, InnerH, typeof(left), typeof(right)}(left, right)
end

"""
    inner₊(left::LazyOp{1}, right::LazyOp{1})
    inner₊(left::NTuple{D,LazyOp{D}}, right::NTuple{D,LazyOp{D}}) where D

Constructs a symbolic modified \$L^2_+\$ inner product between `left` and `right`.
"""
function inner₊(left::LazyOp{1}, right::LazyOp{1})
    BilinearProduct{1, InnerPlus{1}, typeof(left), typeof(right)}(left, right)
end
function inner₊(left::NTuple{D, LazyOp{D}}, right::NTuple{D, LazyOp{D}}) where {D}
    inner_plus(left, right)
end

"""
    inner₊(left::Union{IndexedTrialFunction{D},IndexedTestFunction{D}}, right::BackwardDifference{D,Dim}) where {D,Dim}
    inner₊(left::BackwardDifference{D,Dim}, right::Union{IndexedTrialFunction{D},IndexedTestFunction{D}}) where {D,Dim}

Direction-inferring `inner₊` for **coupled forms only**: uses the `InnerPlus{Dim}` weight
matching the dimension of the `BackwardDifference` operator. This is needed for
pressure-velocity coupling terms like `inner₊(p, D₋ₓ(v[1]))` and `inner₊(p, D₋ᵧ(v[2]))`,
where `p` is an `IndexedTrialFunction` (a symbolic scalar field).

These overloads are intentionally restricted to `IndexedTrialFunction`/`IndexedTestFunction`
leaves so they **do not** conflict with the standard `inner₊(D₋ₓ(u), D₋ₓ(v))` usage.
"""
function inner₊(left::Union{IndexedTrialFunction{D}, IndexedTestFunction{D}},
        right::BackwardDifference{D, Dim}) where {D, Dim}
    BilinearProduct{D, InnerPlus{Dim}, typeof(left), typeof(right)}(left, right)
end
function inner₊(left::BackwardDifference{D, Dim},
        right::Union{IndexedTrialFunction{D}, IndexedTestFunction{D}}) where {D, Dim}
    BilinearProduct{D, InnerPlus{Dim}, typeof(left), typeof(right)}(left, right)
end

"""
    inner₊(left::NTuple{N,<:Tuple}, right::NTuple{N,<:Tuple}) where N

Vector-field `inner₊`: sums per-component inner products.
Used when `left` and `right` are **tuples of gradient tuples**, e.g.
`inner₊(∇₋ₕ(u), ∇₋ₕ(v))` where `u = (u1, u2)` is a velocity tuple.
Each element pair `(left[k], right[k])` is a `D`-tuple of `LazyOp` (a gradient),
which dispatches to the existing `inner₊(::NTuple{D,LazyOp}, ::NTuple{D,LazyOp})`.

This overload is intentionally restricted to `NTuple{N,<:Tuple}` so it does **not**
interfere with `inner₊(NTuple{D,VectorElement}, NTuple{D,VectorElement})` handled by
the `@generated` method in `inner_product.jl`.
"""
function inner₊(left::NTuple{N, <:Tuple}, right::NTuple{N, <:Tuple}) where {N}
    foldl(+, map(inner₊, left, right))
end

"""
    inner₊ₓ(left::LazyOp{D}, right::LazyOp{D}) where D
    inner₊ᵧ(left::LazyOp{D}, right::LazyOp{D}) where D
    inner₊₂(left::LazyOp{D}, right::LazyOp{D}) where D

Constructs directional modified \$L^2_+\$ inner products in x, y, and z directions.
"""
function inner₊ₓ(left::LazyOp{D}, right::LazyOp{D}) where {D}
    BilinearProduct{D, InnerPlus{1}, typeof(left), typeof(right)}(left, right)
end
function inner₊ᵧ(left::LazyOp{D}, right::LazyOp{D}) where {D}
    BilinearProduct{D, InnerPlus{2}, typeof(left), typeof(right)}(left, right)
end
function inner₊₂(left::LazyOp{D}, right::LazyOp{D}) where {D}
    BilinearProduct{D, InnerPlus{3}, typeof(left), typeof(right)}(left, right)
end

@inline function source_number(l::Number, ::Val{D}) where {D}
    f = x -> l
    return SourceFunction{D, typeof(f)}(f)
end

# Linear Forms (e.g. innerₕ(f, v) where f is a Function, Number, or VectorElement and v is TestFunction)
function innerₕ(l::Function, r::LazyOp{D}) where {D}
    LinearProduct{D, InnerH, SourceFunction{D, typeof(l)}, typeof(r)}(
        SourceFunction{
            D, typeof(l)}(l), r)
end
function innerₕ(l::Number, r::LazyOp{D}) where {D}
    let sf = source_number(l, Val(D))
        LinearProduct{D, InnerH, typeof(sf), typeof(r)}(sf, r)
    end
end
function innerₕ(l::VectorElement, r::LazyOp{D}) where {D}
    LinearProduct{D, InnerH, SourceVector{D, typeof(l.data)}, typeof(r)}(
        SourceVector{D, typeof(l.data)}(l.data), r)
end

function inner₊(l::Function, r::LazyOp{D}) where {D}
    LinearProduct{D, InnerPlus{1}, SourceFunction{D, typeof(l)}, typeof(r)}(
        SourceFunction{
            D, typeof(l)}(l), r)
end
function inner₊(l::Number, r::LazyOp{D}) where {D}
    let sf = source_number(l, Val(D))
        LinearProduct{D, InnerPlus{1}, typeof(sf), typeof(r)}(sf, r)
    end
end
function inner₊(l::VectorElement, r::LazyOp{D}) where {D}
    LinearProduct{D, InnerPlus{1}, SourceVector{D, typeof(l.data)}, typeof(r)}(
        SourceVector{D, typeof(l.data)}(l.data), r)
end

function inner₊(l::NTuple{D, Function}, r::NTuple{D, LazyOp{D}}) where {D}
    foldl(+,
        ntuple(
            dim -> LinearProduct{
                D, InnerPlus{dim}, SourceFunction{D, typeof(l[dim])}, typeof(r[dim])}(
                SourceFunction{D, typeof(l[dim])}(l[dim]), r[dim]),
            Val(D)))
end
function inner₊(l::NTuple{D, Number}, r::NTuple{D, LazyOp{D}}) where {D}
    foldl(+,
        ntuple(
            dim -> let sf = source_number(l[dim], Val(D))
                LinearProduct{D, InnerPlus{dim}, typeof(sf), typeof(r[dim])}(sf, r[dim])
            end, Val(D)))
end
@inline function inner₊(l::NTuple{D, VectorElement}, r::NTuple{D, LazyOp{D}}) where {D}
    if is_symbolic(r)
        return foldl(+,
            ntuple(
                dim -> LinearProduct{
                    D, InnerPlus{dim}, SourceVector{D, typeof(l[dim].data)}, typeof(r[dim])}(
                    SourceVector{D, typeof(l[dim].data)}(l[dim].data), r[dim]),
                Val(D)))
    else
        res = similar(first(l).values)
        res .= 0
        inner₊!(res, l, r)
        return res
    end
end

function inner₊ₓ(l::Function, r::LazyOp{D}) where {D}
    LinearProduct{D, InnerPlus{1}, SourceFunction{D, typeof(l)}, typeof(r)}(
        SourceFunction{
            D, typeof(l)}(l), r)
end
function inner₊ᵧ(l::Function, r::LazyOp{D}) where {D}
    LinearProduct{D, InnerPlus{2}, SourceFunction{D, typeof(l)}, typeof(r)}(
        SourceFunction{
            D, typeof(l)}(l), r)
end
function inner₊₂(l::Function, r::LazyOp{D}) where {D}
    LinearProduct{D, InnerPlus{3}, SourceFunction{D, typeof(l)}, typeof(r)}(
        SourceFunction{
            D, typeof(l)}(l), r)
end

function inner₊ₓ(l::Number, r::LazyOp{D}) where {D}
    let sf = source_number(l, Val(D))
        LinearProduct{D, InnerPlus{1}, typeof(sf), typeof(r)}(sf, r)
    end
end
function inner₊ᵧ(l::Number, r::LazyOp{D}) where {D}
    let sf = source_number(l, Val(D))
        LinearProduct{D, InnerPlus{2}, typeof(sf), typeof(r)}(sf, r)
    end
end
function inner₊₂(l::Number, r::LazyOp{D}) where {D}
    let sf = source_number(l, Val(D))
        LinearProduct{D, InnerPlus{3}, typeof(sf), typeof(r)}(sf, r)
    end
end

function inner₊ₓ(l::VectorElement, r::LazyOp{D}) where {D}
    LinearProduct{D, InnerPlus{1}, SourceVector{D, typeof(l.data)}, typeof(r)}(
        SourceVector{D, typeof(l.data)}(l.data), r)
end
function inner₊ᵧ(l::VectorElement, r::LazyOp{D}) where {D}
    LinearProduct{D, InnerPlus{2}, SourceVector{D, typeof(l.data)}, typeof(r)}(
        SourceVector{D, typeof(l.data)}(l.data), r)
end
function inner₊₂(l::VectorElement, r::LazyOp{D}) where {D}
    LinearProduct{D, InnerPlus{3}, SourceVector{D, typeof(l.data)}, typeof(r)}(
        SourceVector{D, typeof(l.data)}(l.data), r)
end

# ==============================================================================
# Zero-Allocation Stencil Evaluators
# ==============================================================================

@inline function local_stencil(
        op::BilinearProduct{D, InnerType}, space, I::CartesianIndex{D},
        markers, lin_idx::Int) where {D, InnerType}
    left_stencil = local_stencil(op.left_op, space, I, markers, lin_idx)
    right_stencil = local_stencil(op.right_op, space, I, markers, lin_idx)
    vol = compute_weight(InnerType(), space, I, lin_idx)
    return multiply_stencils_bilinear(left_stencil, right_stencil, vol)
end

@inline function local_stencil(
        op::LinearProduct{D, InnerType}, space, I::CartesianIndex{D},
        markers, lin_idx::Int) where {D, InnerType}
    left_stencil = local_stencil(op.left_op, space, I, markers, lin_idx)
    right_stencil = local_stencil(op.right_op, space, I, markers, lin_idx)
    vol = compute_weight(InnerType(), space, I, lin_idx)
    return multiply_stencils_linear(left_stencil, right_stencil, vol)
end

# ==============================================================================
# AST Resolution
# ==============================================================================

function resolve_ast(op::BilinearProduct{D, InnerType}) where {D, InnerType}
    BilinearProduct{
        D, InnerType, typeof(resolve_ast(op.left_op)), typeof(resolve_ast(op.right_op))}(
        resolve_ast(op.left_op), resolve_ast(op.right_op))
end
function resolve_ast(op::LinearProduct{D, InnerType}) where {D, InnerType}
    LinearProduct{
        D, InnerType, typeof(resolve_ast(op.left_op)), typeof(resolve_ast(op.right_op))}(
        resolve_ast(op.left_op), resolve_ast(op.right_op))
end
