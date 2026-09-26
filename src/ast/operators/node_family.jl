##############################################################################
#                                                                            #
#                  Declarative AST node families for the form layer          #
#                                                                            #
##############################################################################

#=
# node_family.jl

The form layer's counterpart of `space/operators/stencil.jl`'s `@operator_family`.

Every directional AST node is the same shape: a `struct Node{D, Dim, OpType}` holding one
`inner_op`, built by a per-coordinate alias, gathered by a tuple-valued one, and rebuilt by
`resolve_ast`. `difference.jl`, `average.jl` and `jump.jl` wrote those four things out by
hand, eight times over -- around forty one-line methods whose only variation was which
struct name appeared in them, and six `resolve_ast` methods that differed in nothing else at
all. `@node_family` is that text, once (gpena/Bramble.jl#74).

What stays hand-written is what actually differs per family: `_stencil_taps` and
`_stencil_weights`, which are the arithmetic, and `ShiftNode`, which carries a second field
and so is not this shape.

The dimensional entry point is the reason the collapse happens here rather than in its own
issue. `Dim` is a type parameter, so a node's direction has to be known to the compiler:
`D₋(op, 1)` and `D₋(op, 2)` return different types, and an `Int` argument would make the
return a `Union`. The form layer therefore takes the `Val` form alone, while the space layer
also accepts an `Int` or a `Symbol` -- there, the direction picks a value out of an array
rather than a type, so branching over literal `Val`s recovers a single return type. Both
halves are methods of the same function: `D₋(uₕ, Val(1))` differences a grid function and
`D₋(U, Val(1))` builds a node, and the argument says which.
=#

"""
    _node_dimensional_expr(node_name, alias_name, alias_stem, what; source = nothing)

Returns the expressions defining a node family's dimensional entry point,
`alias_name(op::LazyOp{D}, ::Val{Dim})`, as a `Vector{Expr}`.

`Val` only, and deliberately: `Dim` lands in the node's type, so an `Int` direction would
make the return type a `Union` of the three nodes it could be. The space layer's `Int` and
`Symbol` forms have no counterpart here.
"""
function _node_dimensional_expr(node_name, alias_name, alias_stem, what; source = nothing)
    doc_string = """
        $alias_name(op::LazyOp{D}, ::Val{Dim})

    The symbolic $what of `op` along direction `Dim`, as a `$node_name` node.

    The entry point [`$(alias_stem)ₓ`](@ref) and its siblings forward through, and the same
    function the space layer applies to a grid function: `$alias_name(uₕ, Val(1))` computes
    values, `$alias_name(U, Val(1))` builds the node that will compute them during assembly.

    Takes a `Val` and not an `Int`, unlike its space-layer namesake: `Dim` is a type
    parameter of the node, so a runtime direction would make the return type a `Union`.
    """

    method = :(@inline $(alias_name)(op::LazyOp{D}, ::Val{Dim}) where {D, Dim} = $(node_name){
        D, Dim, typeof(op)}(op))

    return Expr[
        _relocate!(
        Expr(:macrocall, GlobalRef(Core, Symbol("@doc")), source, doc_string, method),
        source
    ),
    ]
end

"""
    _node_alias_expr(alias_name, dispatch_name, direction_index, what; source = nothing)

Returns the expression defining one per-coordinate alias, `alias_name(op)` as
`dispatch_name(op, Val(direction_index))`, with its docstring attached.
"""
function _node_alias_expr(
        alias_name, dispatch_name, direction, direction_index, what; source = nothing
)
    doc_string = """
        $alias_name(op::LazyOp{D})

    The symbolic $what of `op` along the `$direction` direction.

    Alias for `$dispatch_name(op, Val($direction_index))`.
    """

    method = :(@inline $(alias_name)(op::LazyOp{D}) where {D} = $(dispatch_name)(
        op, Val($(direction_index))
    ))

    return _relocate!(
        Expr(:macrocall, GlobalRef(Core, Symbol("@doc")), source, doc_string, method),
        source
    )
end

"""
    _node_vectorial_expr(alias_name, dispatch_name, what; componentwise = false,
                         source = nothing)

Returns the expressions defining a node family's tuple-valued alias, as a `Vector{Expr}`:
`alias_name(op)` giving a `D`-tuple of nodes, and the bare node rather than a one-tuple in
one dimension, as `∇ₕ` and `∇₊ₕ` have always done.

`componentwise = true` adds the method that maps the alias over a tuple of scalar symbolic
functions, which is how a composite space's components reach it. Only the two gradients
carry it; nothing calls `Mₕ((u1, u2))`.

`ntuple` with a `Val` length unrolls, so `dim` is a literal in each unrolled body and the
`Val(dim)` it builds is a compile-time constant -- not the boxed `Val` of
gpena/Bramble.jl#146, which came from a `Val(i)` under a runtime loop bound.
"""
function _node_vectorial_expr(
        alias_name, dispatch_name, what; componentwise::Bool = false, source = nothing
)
    doc_string = """
        $alias_name(op::LazyOp{D})

    The symbolic $what of `op` along every coordinate, as a `D`-tuple of nodes. In one
    dimension there is one direction, so the node itself is returned rather than a
    one-element tuple.
    """

    one_d = :(@inline $(alias_name)(op::LazyOp{1}) = $(dispatch_name)(op, Val(1)))
    n_d = :(@inline $(alias_name)(op::LazyOp{D}) where {D} = ntuple(
        dim -> $(dispatch_name)(op, Val(dim)), Val(D)
    ))

    exprs = Expr[
        Expr(:macrocall, GlobalRef(Core, Symbol("@doc")), source, doc_string, n_d),
        one_d
    ]

    if componentwise
        tuple_doc = """
            $alias_name(ops::Tuple)

        Applies $alias_name component-wise to a tuple of scalar symbolic functions, such as
        the velocity components `(u1, u2)` of a composite space. Returns a tuple of results,
        one per component.
        """
        tuple_method = :($(alias_name)(ops::Tuple) = map($(alias_name), ops))
        push!(
            exprs,
            Expr(
                :macrocall, GlobalRef(Core, Symbol("@doc")), source, tuple_doc, tuple_method
            )
        )
    end

    return Expr[_relocate!(e, source) for e in exprs]
end

"""
    _node_resolve_expr(node_name; source = nothing)

Returns the expression defining `resolve_ast` for one node family.

Identical across the six families but for the struct name, and it resolves the inner
operator once rather than twice: `BackwardDifference` and `ForwardDifference` each called
`resolve_ast(op.inner_op)` in both the type parameter and the field before this was
generated, resolving the whole subtree twice per node.
"""
function _node_resolve_expr(node_name; source = nothing)
    return _relocate!(
        :(function resolve_ast(op::$(node_name){D, Dim}) where {D, Dim}
            inner = resolve_ast(op.inner_op)
            return $(node_name){D, Dim, typeof(inner)}(inner)
        end),
        source
    )
end

"""
    _node_bind_expr(node_name; source = nothing)

Returns the expression defining `_bind_interp_spaces` for one node family.

The interpolation binding pass (`ast/operators/interpolation.jl`) walks a term the way
`resolve_ast` does, and a directional node's share of that walk is the same line per family
as its `resolve_ast`: rebuild with the bound operand inside.
"""
function _node_bind_expr(node_name; source = nothing)
    return _relocate!(
        :(function _bind_interp_spaces(
                op::$(node_name){D, Dim}, trial_leaf, test_leaf
        ) where {D, Dim}
            inner = _bind_interp_spaces(op.inner_op, trial_leaf, test_leaf)
            return $(node_name){D, Dim, typeof(inner)}(inner)
        end),
        source
    )
end

# --- The family macro ---------------------------------------------------------------- #

"""
    @node_family(kwargs...)

Defines one AST node family's whole alias surface and its `resolve_ast`, declaratively.

Keywords, all optional except `node`, `stem` and `what`:

| Keyword | Meaning |
|:--|:--|
| `node` | the node struct, e.g. `BackwardDifference` |
| `stem` | the alias stem, e.g. `D₋`, giving `D₋ₓ`, `D₋ᵧ`, `D₋₂` |
| `what` | names the quantity in the generated prose, e.g. `"backward finite difference"` |
| `dispatch_alias` | the name carrying `f(op, ::Val{Dim})`; defaults to `stem` |
| `vectorial_alias` | the tuple-valued alias over every coordinate, e.g. `∇ₕ` |
| `componentwise` | `true` adds `alias(ops::Tuple)`; only the two gradients use it |
| `resolve` | `false` skips the generated `resolve_ast`, for a node that writes its own |

`_bind_interp_spaces` is generated unconditionally: it is the same rebuild as `resolve_ast`,
and a family that opts out of the latter still has to pass a bound operand through.

The space-layer counterpart is `@operator_family` in `space/operators/stencil.jl`, and the
generators it shares -- `_relocate!` for line attribution, and `_BRAMBLE_var2symbol` /
`_BRAMBLE_var2label` for the coordinate suffixes -- are the same ones used here.
"""
macro node_family(kwargs...)
    opts = Dict{Symbol, Any}()
    for kw in kwargs
        (kw isa Expr && kw.head === :(=)) ||
            error("@node_family takes `key = value` arguments, got $(kw)")
        opts[kw.args[1]] = kw.args[2]
    end

    node_name = opts[:node]
    stem = opts[:stem]
    what = _macro_string(opts[:what])
    dispatch_alias = get(opts, :dispatch_alias, stem)

    exprs = Expr[]
    append!(
        exprs,
        _node_dimensional_expr(
            node_name, dispatch_alias, stem, what; source = __source__
        )
    )

    for (i, suffix) in enumerate(_BRAMBLE_var2symbol)
        push!(
            exprs,
            _node_alias_expr(
                Symbol(stem, suffix),
                dispatch_alias,
                _BRAMBLE_var2label[i],
                i,
                what;
                source = __source__
            )
        )
    end

    if haskey(opts, :vectorial_alias)
        append!(
            exprs,
            _node_vectorial_expr(
                opts[:vectorial_alias],
                dispatch_alias,
                what;
                componentwise = get(opts, :componentwise, false) === true,
                source = __source__
            )
        )
    end

    get(opts, :resolve, true) === false ||
        push!(exprs, _node_resolve_expr(node_name; source = __source__))

    push!(exprs, _node_bind_expr(node_name; source = __source__))

    return esc(Expr(:block, exprs...))
end
