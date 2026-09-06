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

#=
`markers`, on every `innerₕ`/`inner₊`/`inner₊ₓ`/`inner₊ᵧ`/`inner₊₂` below (both bilinear
and linear forms), restricts the product to the union of the labelled regions: a mask on
which grid points the assembled term contributes to at all, the symbolic counterpart of the
numeric `markers` keyword on `space/inner_product.jl`'s versions of the same names.

Implemented by wrapping the built `BilinearProduct`/`LinearProduct` in `RegionRestriction`
(`restrict_to`). That node returns an empty stencil off-region and the term's own stencil on
it, which is the mask required, and is supported by all existing AST walkers: block routing
(`trial_component_or_nothing`/`test_component_or_nothing`), `resolve_ast`, `is_symbolic`.
=#

@inline _restrict_by_markers(prod::LazyOp{D}, ::NTuple{0, Symbol}) where {D} = prod

# A single marker unwraps to a bare `Symbol` region rather than a one-element tuple, so it
# matches `restrict_to`'s own convention exactly, including `:interior`, which is a keyword
# `RegionRestriction` special-cases only when `region` is literally a `Symbol`, not a tuple
# containing one.
@inline function _restrict_by_markers(prod::LazyOp{D}, markers::NTuple{1, Symbol}) where {D}
    RegionRestriction{D, Symbol, typeof(prod)}(markers[1], prod)
end

@inline function _restrict_by_markers(
        prod::LazyOp{D}, markers::NTuple{N, Symbol}) where {D, N}
    RegionRestriction{D, typeof(markers), typeof(prod)}(markers, prod)
end

"""
    inner_plus(left::NTuple{D, LazyOp{D}}, right::NTuple{D, LazyOp{D}}; markers = ()) -> LazyOp{D}

Constructs the sum of directional modified \$L^2_+\$ inner products across all dimensions.

Each dimension's term is a `LinearProduct` or a `BilinearProduct` independently, following
[`_is_source_only`](@ref) on `left[dim]` exactly as [`innerₕ`](@ref) does: a gradient tuple
of interpolated sources (`πₕ(u1), πₕ(u2)`) is source-only dimension by dimension.
"""
function inner_plus(left::NTuple{D, LazyOp{D}}, right::NTuple{D, LazyOp{D}};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {D, N}
    terms = ntuple(Val(D)) do dim
        if _is_source_only(left[dim])
            LinearProduct{D, InnerPlus{dim}, typeof(left[dim]), typeof(right[dim])}(
                left[dim], right[dim])
        else
            BilinearProduct{D, InnerPlus{dim}, typeof(left[dim]), typeof(right[dim])}(
                left[dim], right[dim])
        end
    end
    return _restrict_by_markers(foldl(+, terms), markers)
end

# Support both scalar and NTuple combinations in standard inner products:

"""
    innerₕ(left::LazyOp{D}, right::LazyOp{D}; markers = ()) -> LazyOp{D}

Constructs a symbolic \$L^2\$ inner product between `left` and `right`: a `LinearProduct`
(source × test) if `left` is source-only ([`_is_source_only`](@ref): a source, or a source
wrapped in differences/averages/shifts/jumps/restrictions/scales, never a trial function),
or a `BilinearProduct` (trial × test) otherwise.

This applies specifically when `left` is a `LazyOp`: a bare `Function`/`Number`/`VectorElement`
is unconditionally a source, so those overloads build a `LinearProduct` directly. When
`left` arrives already wrapped (`πₕ(uₕ)` or `D₋ₓ(πₕ(uₕ))`), this check ensures the correct
linear AST node is constructed.

`markers` restricts the assembled term to the union of the labelled regions: a mask on
which grid points it contributes to at all.
"""
function innerₕ(left::LazyOp{D}, right::LazyOp{D};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {D, N}
    prod = if _is_source_only(left)
        LinearProduct{D, InnerH, typeof(left), typeof(right)}(left, right)
    else
        BilinearProduct{D, InnerH, typeof(left), typeof(right)}(left, right)
    end
    _restrict_by_markers(prod, markers)
end

# There is deliberately no `innerₕ` over gradient tuples. `inner₊` has one because its
# weights are directional and the tuple is what supplies the directions; `InnerH` carries a
# single weight, so summing the components is a plain sum with nothing to infer, and it is
# written out at the call site rather than hidden behind the same spelling as the scalar
# product.

"""
    inner₊(left::LazyOp{1}, right::LazyOp{1}; markers = ())
    inner₊(left::NTuple{D,LazyOp{D}}, right::NTuple{D,LazyOp{D}}; markers = ()) where D

Constructs a symbolic modified \$L^2_+\$ inner product between `left` and `right`.

`markers` restricts the sum as it does for [`innerₕ`](@ref).
"""
function inner₊(left::NTuple{D, LazyOp{D}}, right::NTuple{D, LazyOp{D}};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {D, N}
    inner_plus(left, right; markers = markers)
end

"""
    inner₊(left::BackwardDifference{D,Dim}, right::BackwardDifference{D,Dim}) -> LazyOp{D}

`inner₊` of two backward differences taken along the same direction, which is the weight
the product carries: `InnerPlus{Dim}`.

In one dimension there is only one direction, so `inner₊(left, right)` already answers.
Above one dimension a bare `inner₊` of two operators names no direction, and the
weights are directional: the direction is read off the nodes. This is what makes
`inner₊(D₋ₓ(u), D₋ₓ(v))` mean what it reads as.

Backward differences only, as everywhere `inner₊` meets a difference: the weights are those
of the summation-by-parts identity, which pairs them with a backward difference.

Constructs a `LinearProduct` (source × test) if `left` is source-only ([`_is_source_only`](@ref)),
or a `BilinearProduct` otherwise, matching [`innerₕ`](@ref).

`markers` restricts the sum as it does for [`innerₕ`](@ref).
"""
function inner₊(left::BackwardDifference{D, Dim},
        right::BackwardDifference{D, Dim};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {D, Dim, N}
    prod = if _is_source_only(left)
        LinearProduct{D, InnerPlus{Dim}, typeof(left), typeof(right)}(left, right)
    else
        BilinearProduct{D, InnerPlus{Dim}, typeof(left), typeof(right)}(left, right)
    end
    _restrict_by_markers(prod, markers)
end

"""
    inner₊(left::LazyOp{D}, right::LazyOp{D}) -> LazyOp{D}

Symbolic `inner₊` of two operators neither of which names a direction.

In one dimension there is only one direction to name, so this is the product, with weight
`InnerPlus{1}`. Above one dimension the weights are directional and nothing here supplies
the direction, so it throws an `ArgumentError`.

`markers` restricts the sum as it does for [`innerₕ`](@ref).
"""
function inner₊(left::LazyOp{D}, right::LazyOp{D};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {D, N}
    return _inner₊_same_dim(Val(D), left, right, markers)
end

# Split on `Val(D)` rather than branching on `D == 1` at runtime: `D` is a type parameter,
# known at compile time, so the choice belongs at dispatch (gpena/Bramble.jl#59). Kept as an
# inner helper, not a second `inner₊` method on `LazyOp{1}`, `LazyOp{1}`: that concrete-D
# signature is no longer a subtype of the `BackwardDifference{D,Dim}`-paired overloads below
# (unlike this method's shared, still-generic-in-D one), and is genuinely ambiguous against
# them for D=1 -- confirmed by trying it first and watching precompilation fail on exactly
# that call shape.
function _inner₊_same_dim(::Val{1}, left, right, markers::NTuple{N, Symbol}) where {N}
    prod = if _is_source_only(left)
        LinearProduct{1, InnerPlus{1}, typeof(left), typeof(right)}(left, right)
    else
        BilinearProduct{1, InnerPlus{1}, typeof(left), typeof(right)}(left, right)
    end
    return _restrict_by_markers(prod, markers)
end

function _inner₊_same_dim(::Val{D}, left, right, markers::NTuple{N, Symbol}) where {D, N}
    return _inner₊_no_direction(left, right, D)
end

@noinline function _inner₊_no_direction(left, right, D::Int)
    throw(ArgumentError(
        "inner₊ of two symbolic operators in $D dimensions names no direction, and its " *
        "weights are directional. Write inner₊ₓ, inner₊ᵧ or inner₊₂ for a specific one, " *
        "pass gradient tuples such as inner₊(∇₋ₕ(u), ∇₋ₕ(v)) to sum over all of them, or " *
        "difference both sides along the same direction as in inner₊(D₋ₓ(u), D₋ₓ(v)). " *
        "Got $(typeof(left)) and $(typeof(right))."))
end

"""
    inner₊(left::LazyOp{D}, right::BackwardDifference{D,Dim}) -> LazyOp{D}
    inner₊(left::BackwardDifference{D,Dim}, right::LazyOp{D}) -> LazyOp{D}

`inner₊` where one side is a backward difference and the other is not: the difference names
the direction, so the product carries `InnerPlus{Dim}`.

This is what `inner₊(u, D₋ₓ(v))` means: the common form, and the one the coupled
pressure-velocity terms are written in, `inner₊(p, D₋ₓ(v[1]))` with `p` a symbolic scalar
field. It is not restricted to indexed leaves: a plain `TrialFunction` reads the
direction off the difference just as an `IndexedTrialFunction` does.

Backward differences only, as everywhere `inner₊` meets a difference.

A `LinearProduct` if `left` is source-only ([`_is_source_only`](@ref)), a `BilinearProduct`
otherwise, matching [`innerₕ`](@ref).

`markers` restricts the sum as it does for [`innerₕ`](@ref).
"""
function inner₊(left::LazyOp{D}, right::BackwardDifference{D, Dim};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {D, Dim, N}
    prod = if _is_source_only(left)
        LinearProduct{D, InnerPlus{Dim}, typeof(left), typeof(right)}(left, right)
    else
        BilinearProduct{D, InnerPlus{Dim}, typeof(left), typeof(right)}(left, right)
    end
    _restrict_by_markers(prod, markers)
end
function inner₊(left::BackwardDifference{D, Dim}, right::LazyOp{D};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {D, Dim, N}
    prod = if _is_source_only(left)
        LinearProduct{D, InnerPlus{Dim}, typeof(left), typeof(right)}(left, right)
    else
        BilinearProduct{D, InnerPlus{Dim}, typeof(left), typeof(right)}(left, right)
    end
    _restrict_by_markers(prod, markers)
end

"""
    inner₊(left::BackwardDifference{D,Dim1}, right::BackwardDifference{D,Dim2}) where {D,Dim1,Dim2}

Rejects `inner₊` of two backward differences taken along *different* directions.

Each side names a direction and they disagree, so there is no one weight the product
carries. This also has to be written out rather than left to dispatch: with a difference on
either side, the two single-sided methods above tie, and the pair would be an ambiguity
rather than an error the caller can read.
"""
@noinline function inner₊(left::BackwardDifference{D, Dim1},
        right::BackwardDifference{D, Dim2};
        markers::NTuple{M, Symbol} = NTuple{0, Symbol}()) where {D, Dim1, Dim2, M}
    throw(ArgumentError(
        "inner₊ of backward differences along different directions ($Dim1 and $Dim2) " *
        "names no single weight. Difference both sides along the same direction, or " *
        "write inner₊ₓ, inner₊ᵧ or inner₊₂ for the one you mean."))
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

`markers` restricts the whole sum, not each component separately, as it does for
[`innerₕ`](@ref).
"""
function inner₊(left::NTuple{N, <:Tuple}, right::NTuple{N, <:Tuple};
        markers::NTuple{M, Symbol} = NTuple{0, Symbol}()) where {N, M}
    _restrict_by_markers(foldl(+, map(inner₊, left, right)), markers)
end

"""
    inner₊ₓ(left::LazyOp{D}, right::LazyOp{D}) where D
    inner₊ᵧ(left::LazyOp{D}, right::LazyOp{D}) where D
    inner₊₂(left::LazyOp{D}, right::LazyOp{D}) where D

Constructs directional modified \$L^2_+\$ inner products in x, y, and z directions.

A `LinearProduct` if `left` is *source-only* ([`_is_source_only`](@ref)), a `BilinearProduct`
otherwise, exactly as [`innerₕ`](@ref) decides.

`markers` restricts the sum as it does for [`innerₕ`](@ref).
"""
function inner₊ₓ(left::LazyOp{D}, right::LazyOp{D};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {D, N}
    prod = if _is_source_only(left)
        LinearProduct{D, InnerPlus{1}, typeof(left), typeof(right)}(left, right)
    else
        BilinearProduct{D, InnerPlus{1}, typeof(left), typeof(right)}(left, right)
    end
    _restrict_by_markers(prod, markers)
end
function inner₊ᵧ(left::LazyOp{D}, right::LazyOp{D};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {D, N}
    prod = if _is_source_only(left)
        LinearProduct{D, InnerPlus{2}, typeof(left), typeof(right)}(left, right)
    else
        BilinearProduct{D, InnerPlus{2}, typeof(left), typeof(right)}(left, right)
    end
    _restrict_by_markers(prod, markers)
end
function inner₊₂(left::LazyOp{D}, right::LazyOp{D};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {D, N}
    prod = if _is_source_only(left)
        LinearProduct{D, InnerPlus{3}, typeof(left), typeof(right)}(left, right)
    else
        BilinearProduct{D, InnerPlus{3}, typeof(left), typeof(right)}(left, right)
    end
    _restrict_by_markers(prod, markers)
end

@inline function source_number(l::Number, ::Val{D}) where {D}
    return SourceConstant{D, typeof(l)}(l)
end

# Linear Forms (e.g. innerₕ(f, v) where f is a Function, Number, or VectorElement and v is
# TestFunction). `markers` restricts each the same way it does the bilinear forms above: a
# mask on which grid points the source term contributes to at all.
function innerₕ(l::Function, r::LazyOp{D};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {D, N}
    _restrict_by_markers(
        LinearProduct{D, InnerH, SourceFunction{D, typeof(l)}, typeof(r)}(
            SourceFunction{D, typeof(l)}(l), r),
        markers)
end
function innerₕ(l::Number, r::LazyOp{D};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {D, N}
    sf = source_number(l, Val(D))
    _restrict_by_markers(
        LinearProduct{D, InnerH, typeof(sf), typeof(r)}(sf, r), markers)
end
function innerₕ(l::VectorElement, r::LazyOp{D};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {D, N}
    _restrict_by_markers(
        LinearProduct{D, InnerH, SourceVector{D, typeof(l.data)}, typeof(r)}(
            SourceVector{D, typeof(l.data)}(l.data), r),
        markers)
end

function inner₊(l::Function, r::LazyOp{D};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {D, N}
    _restrict_by_markers(
        LinearProduct{D, InnerPlus{1}, SourceFunction{D, typeof(l)}, typeof(r)}(
            SourceFunction{D, typeof(l)}(l), r),
        markers)
end
function inner₊(l::Number, r::LazyOp{D};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {D, N}
    sf = source_number(l, Val(D))
    _restrict_by_markers(
        LinearProduct{D, InnerPlus{1}, typeof(sf), typeof(r)}(sf, r), markers)
end
function inner₊(l::VectorElement, r::LazyOp{D};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {D, N}
    _restrict_by_markers(
        LinearProduct{D, InnerPlus{1}, SourceVector{D, typeof(l.data)}, typeof(r)}(
            SourceVector{D, typeof(l.data)}(l.data), r),
        markers)
end

function inner₊(l::NTuple{D, Function}, r::NTuple{D, LazyOp{D}};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {D, N}
    _restrict_by_markers(
        foldl(+,
            ntuple(
                dim -> LinearProduct{
                    D, InnerPlus{dim}, SourceFunction{D, typeof(l[dim])}, typeof(r[dim])}(
                    SourceFunction{D, typeof(l[dim])}(l[dim]), r[dim]),
                Val(D))),
        markers)
end
function inner₊(l::NTuple{D, Number}, r::NTuple{D, LazyOp{D}};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {D, N}
    _restrict_by_markers(
        foldl(+,
            ntuple(
                dim -> let sf = source_number(l[dim], Val(D))
                    LinearProduct{D, InnerPlus{dim}, typeof(sf), typeof(r[dim])}(sf, r[dim])
                end, Val(D))),
        markers)
end
@inline function inner₊(l::NTuple{D, VectorElement}, r::NTuple{D, LazyOp{D}};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {D, N}
    if all(is_symbolic, r)
        terms = ntuple(Val(D)) do dim
            LinearProduct{D, InnerPlus{dim}, SourceVector{D, typeof(values(l[dim]))},
                typeof(r[dim])}(
                SourceVector{D, typeof(values(l[dim]))}(values(l[dim])), r[dim])
        end
        return _restrict_by_markers(foldl(+, terms), markers)
    else
        return _inner₊_numeric_tuple_unsupported(r)
    end
end

# The right-hand side in this branch carries no trial or test function, so there is
# nothing for the product to be a form in: `∇₋ₕ(IdentityOperator(Wₕ))` has no argument to
# differentiate.
@noinline function _inner₊_numeric_tuple_unsupported(r)
    throw(ArgumentError(
        "inner₊ of a tuple of grid functions against a tuple of non-symbolic operators " *
        "has no definition: the right-hand side carries no trial or test function, so " *
        "there is nothing for the product to be a form in. Got $(typeof(r)). Pair the " *
        "grid functions with a symbolic gradient such as ∇₋ₕ(u), or take the numeric " *
        "inner₊ of two grid functions directly."))
end

function inner₊ₓ(l::Function, r::LazyOp{D};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {D, N}
    _restrict_by_markers(
        LinearProduct{D, InnerPlus{1}, SourceFunction{D, typeof(l)}, typeof(r)}(
            SourceFunction{D, typeof(l)}(l), r),
        markers)
end
function inner₊ᵧ(l::Function, r::LazyOp{D};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {D, N}
    _restrict_by_markers(
        LinearProduct{D, InnerPlus{2}, SourceFunction{D, typeof(l)}, typeof(r)}(
            SourceFunction{D, typeof(l)}(l), r),
        markers)
end
function inner₊₂(l::Function, r::LazyOp{D};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {D, N}
    _restrict_by_markers(
        LinearProduct{D, InnerPlus{3}, SourceFunction{D, typeof(l)}, typeof(r)}(
            SourceFunction{D, typeof(l)}(l), r),
        markers)
end

function inner₊ₓ(l::Number, r::LazyOp{D};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {D, N}
    sf = source_number(l, Val(D))
    _restrict_by_markers(LinearProduct{D, InnerPlus{1}, typeof(sf), typeof(r)}(sf, r), markers)
end
function inner₊ᵧ(l::Number, r::LazyOp{D};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {D, N}
    sf = source_number(l, Val(D))
    _restrict_by_markers(LinearProduct{D, InnerPlus{2}, typeof(sf), typeof(r)}(sf, r), markers)
end
function inner₊₂(l::Number, r::LazyOp{D};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {D, N}
    sf = source_number(l, Val(D))
    _restrict_by_markers(LinearProduct{D, InnerPlus{3}, typeof(sf), typeof(r)}(sf, r), markers)
end

function inner₊ₓ(l::VectorElement, r::LazyOp{D};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {D, N}
    _restrict_by_markers(
        LinearProduct{D, InnerPlus{1}, SourceVector{D, typeof(l.data)}, typeof(r)}(
            SourceVector{D, typeof(l.data)}(l.data), r),
        markers)
end
function inner₊ᵧ(l::VectorElement, r::LazyOp{D};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {D, N}
    _restrict_by_markers(
        LinearProduct{D, InnerPlus{2}, SourceVector{D, typeof(l.data)}, typeof(r)}(
            SourceVector{D, typeof(l.data)}(l.data), r),
        markers)
end
function inner₊₂(l::VectorElement, r::LazyOp{D};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {D, N}
    _restrict_by_markers(
        LinearProduct{D, InnerPlus{3}, SourceVector{D, typeof(l.data)}, typeof(r)}(
            SourceVector{D, typeof(l.data)}(l.data), r),
        markers)
end

# ==============================================================================
# Zero-Allocation Stencil Evaluators
# ==============================================================================

# `_same_operator_shape` answers, for `innerₕ(L(u), L(v))` with the same `L`
# on both sides, that `op.left_op` and `op.right_op` are the same operator chain up to
# substituting `TrialFunction` for `TestFunction` at the leaves: exactly the
# condition under which `local_stencil(op.left_op, …)` and `local_stencil(op.right_op, …)`
# compute the identical tuple of `(offset, coefficient)` pairs. The two are then multiplied
# pairwise regardless, so half of those products (`left[i][2]*left[j][2]` and
# `left[j][2]*left[i][2]`) are the same number computed twice.
# `multiply_stencils_bilinear_symmetric` computes each such product once and reuses it for
# both `(i, j)` and `(j, i)`, so the fast path below runs `local_stencil` on one side only and
# still returns the same `N²`-entry tuple `multiply_stencils_bilinear` would have, just with
# `N(N+1)/2` multiplications behind it instead of `N²`.
#
# This check depends only on `op.left_op`/`op.right_op`'s structure, not on which space the
# trial and test argument range over: unlike `issymmetric`/`isposdef`,
# which additionally require the same space, it answers a strictly local question ("do the two
# sides compute the same numbers here") and is safe regardless.
@inline function local_stencil(
        op::BilinearProduct{D, InnerType}, space, I::CartesianIndex{D},
        markers, lin_idx::Int) where {D, InnerType}
    vol = compute_weight(InnerType(), space, I, lin_idx)
    if _same_operator_shape(op.left_op, op.right_op)
        stencil = local_stencil(op.left_op, space, I, markers, lin_idx)
        return multiply_stencils_bilinear_symmetric(stencil, vol)
    end
    left_stencil = local_stencil(op.left_op, space, I, markers, lin_idx)
    right_stencil = local_stencil(op.right_op, space, I, markers, lin_idx)
    return multiply_stencils_bilinear(left_stencil, right_stencil, vol)
end

# The left factor of a linear product is contracted to a scalar:
# `multiply_stencils_linear` keeps only the *right* operand's offsets and multiplies the
# coefficients, so the assembly sums the left stencil's coefficients and discards where each
# one sat. That is exact only when the left stencil is a single entry, at offset zero,
# carrying the factor's true value at this point: an invariant the code relies on, and which
# a source under an operator breaks: `D₋ₓ(f)` stencils as the same value at two offsets with
# opposite signs, so the sum is zero.
#
# Fixed by reading the source-only subtree's own `local_stencil` and discarding its offsets
# (`sum_stencil_values`). That reads correctly via `stencil_shift_trait`:
# a source is `PointDependentStencil`, so every neighbour a wrapping operator reaches is
# obtained by re-evaluating the subtree at that neighbour's own point
# (`shifted_inner_stencil`) rather than by relabelling the offset.
@noinline function _throw_source_not_point_dependent(op)
    throw(ArgumentError(
        "`_is_source_only` accepted $(typeof(op)) as a source, but `stencil_shift_trait` " *
        "does not mark it (or a node it wraps) `PointDependentStencil`. Contracting it would " *
        "relabel offsets instead of re-reading the source at each neighbour. " *
        "Add the missing `stencil_shift_trait` method next to the node's definition."))
end

# Called only on a `LinearProduct`'s own `left_op`, which is source-only by construction:
# every constructor above chooses `LinearProduct` over `BilinearProduct` precisely by
# checking `_is_source_only(left)` first. So the contraction below is unconditional.
@inline function _contracted_left_stencil(op, space, I::CartesianIndex{D},
        markers, lin_idx::Int) where {D}
    stencil_shift_trait(op) isa PointDependentStencil ||
        _throw_source_not_point_dependent(op)
    stencil = local_stencil(op, space, I, markers, lin_idx)
    return ((zero_offset(Val(D)), sum_stencil_values(stencil)),)
end

@inline function local_stencil(
        op::LinearProduct{D, InnerType}, space, I::CartesianIndex{D},
        markers, lin_idx::Int) where {D, InnerType}
    left_stencil = _contracted_left_stencil(op.left_op, space, I, markers, lin_idx)
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

# Disambiguation for the empty tuple.
#
# The overloads below are written over `NTuple{D, …}` for several element types: LazyOp
# nodes, VectorElements, Functions, Numbers; any two of them overlap at `D = 0`,
# where `Tuple{}` satisfies both and neither signature is more specific. That is ten
# ambiguous pairs, which Aqua fails on.
#
# One method for `Tuple{}` settles all of them. It throws rather than returning zero: an
# empty tuple carries no direction to integrate over, so reaching here means a caller
# built a form with no components, and a silent zero would hide that.
#=
Needs its own `markers` keyword, not just the positional disambiguation: with `markers` added
to every `NTuple{D,...}` overload above, they all tie again at `D = 0` specifically for the
*keyword-call* dispatch (`Tuple{}` matches `NTuple{0,LazyOp{0}}`, `NTuple{0,Function}`,
`NTuple{0,Number}`, `NTuple{0,VectorElement}` and `NTuple{0,<:Tuple}` identically), even
though the plain positional call already resolves through this method with no keywords at
all. Aqua's ambiguity check is what caught it.
=#
@noinline function inner₊(::Tuple{}, ::Tuple{};
        markers::NTuple{M, Symbol} = NTuple{0, Symbol}()) where {M}
    throw(ArgumentError("inner₊ needs at least one component; got two empty tuples"))
end
