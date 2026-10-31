# rebind.jl

#=
A form rebuilt around another grid function (gpena/Bramble.jl#985): every node of its AST
that reads the element `old` reads `new` instead, `new` of any element type, a `Dual` one
among them. A semidiscretisation whose operator reads its own state needs this to give a
`Dual` state its own operator: the form holds the user's `Float64` element, and no copy of
values can put `Dual`s into it.

The walk descends every node with children, as `simplify_ast` (simplifier.jl) and
`_lower_signs` do, and rebuilds a node only where a child changed, so a form that never
reads `old` comes back as itself. Included after the assembly files, as its form methods
name the plain and ghost form types.
=#

"""
    _replace_grid_function(form, old => new)

`form` rebuilt so that its AST reads the grid function `new` wherever it read `old`: a
coefficient (`old * ∇ₕ(u)`), a source (`innerₕ(old, v)`) or an interpolant (`πₕ(old)`).
`old` is matched by identity (`===`), so only the object written in the expression is
replaced. `new` may have a wider element type than `old` (a `ForwardDiff.Dual` one), and
the form returned assembles to that type. A form that never reads `old` is returned
itself.

Takes a [`BilinearForm`](@ref) or [`LinearForm`](@ref), on plain or composite spaces, and
the forms on ghost spaces and on composites of them, which are rebuilt as `form` builds
them, buffers included: the form returned shares no buffer or cache with `form`. A ghost
form keeps refusing an element type `form` refuses there.

A grid function captured inside a Julia closure (`x -> sₕ(x)`, a source function) is
invisible to the AST and is not rebound, nor is a component view (`sₕ(1)`), a new object
at every call.
"""
# The passes `form` runs after `simplify_ast` read the types of the scalars and data, so
# they run again on the rebound tree: a constant scale beside a coefficient that turned
# complex is widened as `form` widens it, or every point of a refill boxes.
function _replace_grid_function(a::BilinearForm{D}, p::Pair) where {D}
    rebound = _rebind(a.ast, p.first, p.second)
    rebound === a.ast && return a
    W, V = a.trial_space, a.test_space
    E = promote_type(eltype(W), eltype(V))
    N = _join_eltype(_leaf_eltype(W), _leaf_eltype(V))
    ast = _split_restricted_products(_normalise_scalars(rebound, N), E)
    return BilinearForm{D, typeof(W), typeof(V), typeof(ast)}(W, V, ast, _AssemblyCache{D}(ast))
end

# As for a bilinear form, and the sources are lowered again, so a restricted one is probed
# at its new type (`_typed_restrictions`, assembly/linear.jl); lowered already, the rest
# stay as they are.
function _replace_grid_function(l::LinearForm{D}, p::Pair) where {D}
    rebound = _rebind(l.ast, p.first, p.second)
    rebound === l.ast && return l
    W = l.test_space
    ast = _split_restricted_products(_normalise_scalars(rebound, _leaf_eltype(W)), eltype(W))
    ast = _lower_sources_for_space(ast, W)
    return LinearForm{D, typeof(W), typeof(ast)}(W, ast)
end

# One method per ghost form, so the two sides are both scalar or both composite.
function _replace_grid_function(a::GhostBilinearForm, p::Pair)
    return _replace_ghost_padded(a, p)
end
function _replace_grid_function(a::GhostCompositeBilinearForm, p::Pair)
    return _replace_ghost_padded(a, p)
end

function _replace_ghost_padded(a, p::Pair)
    padded = _replace_grid_function(a.padded, p)
    padded === a.padded && return a
    return _ghost_bilinear_form(a.trial_space, a.test_space, padded)
end

function _replace_grid_function(l::_AnyGhostLinearForm, p::Pair)
    padded = _replace_grid_function(l.padded, p)
    padded === l.padded && return l
    return _ghost_linear_form(l.test_space, padded)
end

# --- The AST --------------------------------------------------------------------------- #

# A leaf, or a node holding neither a grid function nor a child.
@inline _rebind(op, old, new) = op

function _rebind(op::GridFunctionScale, old, new)
    inner = _rebind(op.inner_op, old, new)
    op.grid_function === old && return GridFunctionScale(new, inner)
    return inner === op.inner_op ? op : GridFunctionScale(op.grid_function, inner)
end

@inline function _rebind(op::SourceVector{D}, old, new) where {D}
    return op.vec === old ? SourceVector{D, typeof(new)}(new) : op
end

# `πₕ(old)`: the interpolant holds `old` and reads it at every assembly.
@inline function _rebind(op::SourceFunction{D, <:GridInterpolant}, old, new) where {D}
    f = op.func
    f.uₕ === old || return op
    g = GridInterpolant(new, f.outside)
    return SourceFunction{D, typeof(g)}(g)
end

# Not `_rewrap_inner`: it returns `op` for an operand of the same type, and `old => new` of
# one type changes the operand's value only.
function _rebind(op::UnaryWrapper, old, new)
    inner = _rebind(op.inner_op, old, new)
    return inner === op.inner_op ? op : _rebuilt_around(op, inner)
end

function _rebind(op::OperatorAdd, old, new)
    left = _rebind(op.left_op, old, new)
    right = _rebind(op.right_op, old, new)
    return left === op.left_op && right === op.right_op ? op : OperatorAdd(left, right)
end

function _rebind(op::BilinearProduct{D, W}, old, new) where {D, W}
    l, r = _rebind(op.left_op, old, new), _rebind(op.right_op, old, new)
    l === op.left_op && r === op.right_op && return op
    return BilinearProduct{D, W, typeof(l), typeof(r)}(op.inner, l, r)
end

function _rebind(op::LinearProduct{D, W}, old, new) where {D, W}
    l, r = _rebind(op.left_op, old, new), _rebind(op.right_op, old, new)
    l === op.left_op && r === op.right_op && return op
    return LinearProduct{D, W, typeof(l), typeof(r)}(op.inner, l, r)
end
