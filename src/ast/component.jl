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

The index is part of the leaf's type, so a literal index such as `v(1) + v(2)` gives
`form` one concrete return type, and every function the form passes through, compiled on
its own (as Enzyme compiles it), sees that type too. A runtime component index (`v(i)` with
`i` a function argument) gives a leaf whose type inference cannot know, so `form` does not
infer a concrete type for it. The form assembles the same either way; where a concrete type
matters (automatic differentiation, a function barrier), use literal components.
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
@inline component(
    op::Union{SourceFunction, SourceVector, SourceConstant, DiracSource,
        IdentityOperator, ZeroOperator}, ::Int) = op

# --- the operators, rebuilt around an indexed argument ----------------------------- #

# Every wrapper (stencil nodes, scales, restrictions, interpolation) is rebuilt around its
# indexed operand, other fields kept: a scale's scalar or grid function is not what the
# index names. By value, as `v(1)(2)` re-indexes a leaf without changing its type.
@inline component(op::UnaryWrapper, i::Int) = _rewrap_value(op, component(op.inner_op, i))

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
#
# Every method below takes `markers` and passes it to each component's own product. A call
# with keywords dispatches only among the methods that declare keywords, so a method without
# one is invisible to `innerₕ(uₕ, v; markers = m)`, which then fell back to the scalar
# overload and the wrong answer above. `inner_Γ` gets its own method after the loop, with
# the keyword untyped as in operators/inner.jl, since a bare `Symbol` is a valid label there.
#
# `r` must not already name a component. `component(r, c)` re-indexes, so `innerₕ(fₕ, v(2))`
# used to read as `Σ_c innerₕ(fₕ(c), v(c))`, every component, whatever `v(2)` asked for
# (gpena/Bramble.jl#979). Each method below refuses such an `r` before indexing it.
# Bare `inner₊` of a bare test function names no direction past 1D, so the spelling the
# message suggests for it is a directional one, which forms in any dimension.
@noinline function _throw_indexed_shorthand(f::Symbol)
    g = f === :inner₊ ? "inner₊ₓ" : string(f)
    bare = f === :inner₊ ?
           " (bare `inner₊` reads its direction off a difference in `r`; past 1D a bare " *
           "test function needs `inner₊ₓ`, `inner₊ᵧ` or `inner₊₂`)" : ""
    throw(
        ArgumentError(
        "`$f(fₕ, r)` with a composite `fₕ` sums over every component, indexing `r` " *
        "itself, but `r` already names a component. Write `$g(fₕ(i), v(i))` for one " *
        "component, or pass the unindexed test function to sum over all of them$bare.",
    ),
    )
end

for (f, W) in (
    (:innerₕ, :InnerH),
    (:inner₊, :(InnerPlus{1})),
    (:inner₊ₓ, :(InnerPlus{1})),
    (:inner₊ᵧ, :(InnerPlus{2})),
    (:inner₊₂, :(InnerPlus{3}))
)
    # `NC` is `l`'s *leaf* count (`length(comps)`, over `components`, which flattens any
    # nesting), not the space's own structural type parameter -- `component(r, c)` names
    # leaf `c` (form-level indexing is already leaf-based, see assembly/block_extract.jl), so
    # this has to walk the same leaves `comps` does, in the same order, for any `r`.
    @eval @inline function $f(
            l::VectorElement{<:CompositeGridSpace}, r::LazyOp{D};
            markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
    ) where {D, N}
        _test_component_class(r) === nothing || _throw_indexed_shorthand($(QuoteNode(f)))
        comps = components(l)
        return foldl(
            +, ntuple(c -> $f(comps[c], component(r, c); markers), Val(length(comps)))
        )
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
    @eval @inline function $f(
            l::NTuple{NC, Any}, r::LazyOp{D};
            markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
    ) where {NC, D, N}
        _test_component_class(r) === nothing || _throw_indexed_shorthand($(QuoteNode(f)))
        return foldl(+, ntuple(c -> $f(l[c], component(r, c); markers), Val(NC)))
    end

    # The keyword keeps an empty tuple with `markers` here rather than in the method above
    # with `NC = 0`, where `foldl` would throw a reduction error naming nothing useful.
    @eval @noinline function $f(
            ::Tuple{}, ::LazyOp; markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
    ) where {N}
        throw(
            ArgumentError(
            "an empty tuple names no components, so there is nothing to sum. Give one " *
            "entry per component of the test space.",
        ),
        )
    end
end

# The composite `inner_Γ`: without it the scalar `VectorElement` method would read the first
# component into every block. No tuple method (a tuple is a loud `MethodError`).
@inline function inner_Γ(
        l::VectorElement{<:CompositeGridSpace}, r::LazyOp{D}; markers = ()
) where {D}
    _test_component_class(r) === nothing || _throw_indexed_shorthand(:inner_Γ)
    comps = components(l)
    return foldl(
        +, ntuple(c -> inner_Γ(comps[c], component(r, c); markers), Val(length(comps)))
    )
end
