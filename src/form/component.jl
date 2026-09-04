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

@inline component(::TrialFunction{D}, i::Int) where {D} = IndexedTrialFunction{D}(i)
@inline component(::TestFunction{D}, i::Int) where {D} = IndexedTestFunction{D}(i)

# Already indexed: re-indexing replaces the index, so that `v(1)(2)` is `v(2)` rather than
# an error or a silent no-op.
@inline component(::IndexedTrialFunction{D}, i::Int) where {D} = IndexedTrialFunction{D}(i)
@inline component(::IndexedTestFunction{D}, i::Int) where {D} = IndexedTestFunction{D}(i)

# A source has no component to name, and neither identity nor zero depends on one.
@inline component(op::SourceFunction, ::Int) = op
@inline component(op::SourceVector, ::Int) = op
@inline component(op::SourceConstant, ::Int) = op
@inline component(op::IdentityOperator, ::Int) = op
@inline component(op::ZeroOperator, ::Int) = op

# --- the operators, rebuilt around an indexed argument ----------------------------- #

for T in (:BackwardDifference, :ForwardDifference, :CenteredDifference, :StarDifference,
    :CrossWeightedDifference, :JumpNode, :BackwardAverage, :ForwardAverage)
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

# --- the functor ------------------------------------------------------------------- #
#
# Defined on the abstract type, so every node answers and a new one inherits it. What it
# needs from a node is a `component` method, which is the list above.

@inline (op::LazyOp)(i::Int) = component(op, i)

# --- the composite shorthand ------------------------------------------------------- #
#
# `innerₕ(uₕ, r)` where `uₕ` is a grid function of a composite space is the inner product of
# the product space: the sum over components of each component's own product,
#
#     innerₕ(uₕ, r) = Σ_c innerₕ(uₕ(c), r(c))
#
# and because the index distributes, `r` can be any expression in the test function rather
# than only the bare leaf: `innerₕ(uₕ, v + 2 * D₋ₓ(v) - M₋ₓ(v))` expands term by term and
# component by component.
#
# Without these methods the call did not fail; it took the scalar overload, wrapped the
# whole composite coefficient vector in one `SourceVector`, and assembled it into every
# block reading the first component's coefficients each time. Silently the wrong answer,
# which is why these are worth having rather than merely convenient.

for (f, W) in ((:innerₕ, :InnerH), (:inner₊, :(InnerPlus{1})),
    (:inner₊ₓ, :(InnerPlus{1})), (:inner₊ᵧ, :(InnerPlus{2})), (:inner₊₂, :(InnerPlus{3})))
    @eval @inline function $f(l::VectorElement{<:CompositeGridSpace{NC}},
            r::LazyOp{D}) where {NC, D}
        comps = components(l)
        return foldl(+, ntuple(c -> $f(comps[c], component(r, c)), Val(NC)))
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
        throw(ArgumentError(
            "an empty tuple names no components, so there is nothing to sum. Give one " *
            "entry per component of the test space."))
    end
end
