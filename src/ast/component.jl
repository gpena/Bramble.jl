# component.jl
#
# Indexing a symbolic operator by component.
#
# `v(i)` is the i-th component of a test function, and the index distributes through
# whatever has been built on top of it: `(v + D₋ₓ(v))(1)` is `v(1) + D₋ₓ(v(1))`, and
# `D₋ₓ(v)(2)` is `D₋ₓ(v(2))`. So a coupled form can be written the way the mathematics is,
# with the component named once at the leaf or once at the outside, whichever reads better.
#
# The distribution is a rebuild of the tree with the trial and test leaves replaced by their
# indexed forms. Everything else is structural and passes through: an operator does not
# change because its argument names a component, and a source term does not have one to
# name.
#
# This is what makes the composite `innerₕ(uₕ, r)` shorthand possible:
# `Σ innerₕ(uₕ(c), r(c))` (because `r` may be any expression in the test function rather
# than only the bare leaf).

"""
    component(op::LazyOp, i::Int) -> LazyOp

The `i`-th component of the symbolic operator `op`: the same expression with its trial and
test leaves replaced by their indexed forms.

Reached through the functor, so `op(i)` is `component(op, i)`. The index distributes, so
`(v + D₋ₓ(v))(1)` and `v(1) + D₋ₓ(v(1))` are the same tree.
"""
function component end

# --- the leaves -------------------------------------------------------------------- #

@inline _check_component_range(i::Int, ::Nothing) = (
    i >= 1 || throw(ArgumentError("component index must be >= 1, got $i")); nothing)
@inline function _check_component_range(i::Int, N::Integer)
    1 <= i <= N || throw(
        ArgumentError(
        "component index $i is out of range for a $N-component function space. " *
        "Components are numbered 1 to $N.",
    ),
    )
    return nothing
end

@inline function component(op::TrialFunction{D, N}, i::Int) where {D, N}
    _check_component_range(i, N)
    return IndexedTrialFunction{D}(i)
end

@inline function component(op::TestFunction{D, N}, i::Int) where {D, N}
    _check_component_range(i, N)
    return IndexedTestFunction{D}(i)
end

# Already indexed: re-indexing replaces the index, so that `v(1)(2)` is `v(2)` rather than
# an error or a silent no-op.
@inline function component(::IndexedTrialFunction{D}, i::Int) where {D}
    i >= 1 || throw(ArgumentError("component index must be >= 1, got $i"))
    return IndexedTrialFunction{D}(i)
end

@inline function component(::IndexedTestFunction{D}, i::Int) where {D}
    i >= 1 || throw(ArgumentError("component index must be >= 1, got $i"))
    return IndexedTestFunction{D}(i)
end

# A source has no component to name, and neither identity nor zero depends on one.
@inline component(op::SourceFunction, ::Int) = op
@inline component(op::SourceVector, ::Int) = op
@inline component(op::SourceConstant, ::Int) = op
@inline component(op::DiracSource, ::Int) = op
@inline component(op::IdentityOperator, ::Int) = op
@inline component(op::ZeroOperator, ::Int) = op

# --- the operators, rebuilt around an indexed argument ----------------------------- #

for T in (
    :BackwardDifference,
    :ForwardDifference,
    :CenteredDifference,
    :StarDifference,
    :CrossWeightedDifference,
    :JumpNode,
    :BackwardAverage,
    :ForwardAverage,
    :CenteredAverage
)
    @eval @inline function component(op::$T{D, Dim}, i::Int) where {D, Dim}
        inner = component(op.inner_op, i)
        return $T{D, Dim, typeof(inner)}(inner)
    end
end

@inline function component(op::ShiftNode{D, Dim}, i::Int) where {D, Dim}
    inner = component(op.inner_op, i)
    return ShiftNode{D, Dim, typeof(inner)}(op.shift_amount, inner)
end

@inline function component(op::RegionRestriction{D, R}, i::Int) where {D, R}
    inner = component(op.inner_op, i)
    return RegionRestriction{D, R, typeof(inner)}(op.region, inner)
end

# Scaling passes through untouched: the scalar or grid function multiplying an operator is
# not what the index names.
@inline function component(op::OperatorScale{D, S}, i::Int) where {D, S}
    inner = component(op.inner_op, i)
    return OperatorScale{D, S, typeof(inner)}(op.scalar, inner)
end

@inline function component(op::GridFunctionScale{D, V}, i::Int) where {D, V}
    inner = component(op.inner_op, i)
    return GridFunctionScale{D, V, typeof(inner)}(op.grid_function, inner)
end

@inline function component(op::OperatorAdd{D}, i::Int) where {D}
    l = component(op.left_op, i)
    r = component(op.right_op, i)
    return OperatorAdd{D, typeof(l), typeof(r)}(l, r)
end

# --- the functor & bracket indexing ----------------------------------------------- #
#
# Defined on the abstract type, so every node answers and a new one inherits it. What it
# needs from a node is a `component` method, which is the list above.

@inline (op::LazyOp)(i::Integer) = component(op, Int(i))
@inline Base.getindex(op::LazyOp, i::Integer) = component(op, Int(i))

@noinline function _throw_unknown_components(op)
    throw(
        ArgumentError(
        "the number of components for $(typeof(op)) is not statically known. " *
        "Specify the component count explicitly with `components(op, N)` or construct the form with space context.",
    ),
    )
end

"""
    components(op::Union{TrialFunction, TestFunction}) -> Tuple
    components(op::LazyOp, N::Integer) -> Tuple
    components(op::LazyOp, space::AbstractSpaceType) -> Tuple

Returns an `NTuple` of components of the symbolic trial or test function, suitable for
tuple destructuring: `(u, v) = components(p)`.
"""
@inline function components(op::Union{TrialFunction{D, N}, TestFunction{D, N}}) where {D, N}
    N isa Integer || _throw_unknown_components(op)
    return ntuple(i -> component(op, i), Val(N))
end

@inline function components(op::LazyOp, N::Integer)
    N >= 1 || throw(ArgumentError("component count must be positive, got $N"))
    return ntuple(i -> component(op, i), Val(Int(N)))
end

@inline components(op::LazyOp, space::AbstractSpaceType) = components(op, leaf_count(space))
@inline components(op::Union{IndexedTrialFunction, IndexedTestFunction}) = (op,)

@inline Base.length(op::Union{TrialFunction{D, N}, TestFunction{D, N}}) where {D, N} = N isa Integer ? N :
                                                                                       _throw_unknown_components(op)
@inline Base.firstindex(::Union{TrialFunction, TestFunction}) = 1
@inline Base.lastindex(op::Union{TrialFunction{D, N}, TestFunction{D, N}}) where {D, N} = length(op)
@inline Base.eachindex(op::Union{TrialFunction{D, N}, TestFunction{D, N}}) where {D, N} = 1:length(op)

@inline function Base.iterate(
        op::Union{TrialFunction{D, N}, TestFunction{D, N}}, state::Int = 1
) where {D, N}
    N isa Integer || _throw_unknown_components(op)
    state > N && return nothing
    return (component(op, state), state + 1)
end

# --- the composite shorthand ------------------------------------------------------- #
#
# `innerₕ(uₕ, r)` where `uₕ` is a grid function of a composite space is the inner product of
# the product space: the sum over components of each component's own product,
#
#     innerₕ(uₕ, r) = Σ_c innerₕ(uₕ(c), r(c))
#
# and because the index distributes, `r` can be any expression in the test function rather
# than only the bare leaf: `innerₕ(uₕ, v + 2 * D₋ₓ(v) - Mₓ(v))` expands term by term and
# component by component.
#
# Without these methods the call did not fail; it took the scalar overload, wrapped the
# whole composite coefficient vector in one `SourceVector`, and assembled it into every
# block reading the first component's coefficients each time. Silently the wrong answer,
# which is why these are worth having rather than merely convenient.

for (f, W) in (
    (:innerₕ, :InnerH),
    (:inner₊, :(InnerPlus{1})),
    (:inner₊ₓ, :(InnerPlus{1})),
    (:inner₊ᵧ, :(InnerPlus{2})),
    (:inner₊₂, :(InnerPlus{3}))
)
    # `NC` is `l`'s *leaf* count (`length(comps)`, over `components`, which flattens any
    # nesting), not the space's own structural type parameter -- `component(r, c)` names
    # leaf `c` (form-level indexing is already leaf-based, see form/block_extract.jl), so
    # this has to walk the same leaves `comps` does, in the same order, for any `r`.
    @eval @inline function $f(
            l::VectorElement{<:CompositeGridSpace}, r::LazyOp{D}
    ) where {D}
        comps = components(l)
        return foldl(+, ntuple(c -> $f(comps[c], component(r, c)), Val(length(comps))))
    end

    # A tuple reads the same way, one entry per component, which is how `Rₕ` already takes
    # a composite source: `Rₕ(Vₕ, (f, g))`. So `innerₕ((f, g), v)` is the form-level spelling
    # of the same thing, and covers a tuple of numbers as readily as a tuple of functions (both
    # are sources the scalar case already accepts).
    #
    # Unlike the `VectorElement` method above, a tuple carries no space, so its length is
    # only a claim about how many components the form has. A claim that turns out wrong is
    # caught where the space is known, in `_route_terms!`, which used to drop such a term
    # in silence.
    @eval @inline function $f(l::NTuple{NC, Any}, r::LazyOp{D}) where {NC, D}
        return foldl(+, ntuple(c -> $f(l[c], component(r, c)), Val(NC)))
    end

    @eval @noinline function $f(::Tuple{}, ::LazyOp)
        throw(
            ArgumentError(
            "an empty tuple names no components, so there is nothing to sum. Give one " *
            "entry per component of the test space.",
        ),
        )
    end
end
