# simplifier.jl

#=
Algebraic simplification of the symbolic `+`/`*` layer built in `ast.jl`, run once after
`resolve_ast` and before a `BilinearForm`/`LinearForm` stores its AST (gpena/Bramble.jl#159).

Only three node types carry algebra: `OperatorAdd`, `OperatorScale`, `GridFunctionScale`.
Everything else -- differences, averages, jumps, shifts, restrictions, interpolation,
`BilinearProduct`/`LinearProduct`, and every leaf -- is semantic, not algebraic, and is left
untouched. A term such as `innerₕ(2 * u, v)` still buries its scalar in the trial slot; lifting
it out is gpena/Bramble.jl#159's rule 4, deferred. What this pass does fix, because both
patterns sit at the algebra layer the router itself walks (`_visit_operator_add*` in
`stencil_eval.jl` splits only on `OperatorAdd`; every other node is one routed term, one
mesh sweep):

  - `c * A + c * B` (same `c`, different `A`/`B`) factors to `c * (A + B)`: two routed terms
    become one.
  - `c1 * A + c2 * A` (same `A`) combines to `(c1 + c2) * A`: two routed terms become one.
  - `0 * A` collapses to a zero term with a one-point sparsity pattern instead of `A`'s full
    stencil, and `A + 0` / `0 + A` drops the zero term from the tree entirely.

Bit-exact with the unsimplified AST: every rule is an algebraic identity over the scalars and
grid functions involved, not an approximation.
=#

# --- Zero/identity recognition ------------------------------------------------------ #

# Synthesized only here, for a subtree whose concrete space is not known generically.
# Every consumer of a `ZeroOperator` (`local_stencil`, `stencil_offsets`, `component`,
# `_is_source_only`'s fallback) reads only its `D` type parameter, never `.space` --
# except `_same_operator_shape` (`symmetry.jl`), which compares `a.space === b.space`, and
# `nothing === nothing` settles that the same way two zero operators over the same space
# would: they are the same operator.
@inline _zero_of(::LazyOp{D}) where {D} = ZeroOperator{D,Nothing}(nothing)

@inline _is_zero_op(::LazyOp) = false
@inline _is_zero_op(::ZeroOperator) = true

# `(scalar, inner)` for a node the combining/factoring rules can read a coefficient from.
# A bare (non-`OperatorScale`) node is `1 * itself`. The scalar is returned exactly as
# stored -- a `Base.RefValue` is never dereferenced, since its value can change after the
# form is built (`β[] = 3.0`, gpena/Bramble.jl#159's own worked example).
@inline _scale_parts(op::OperatorScale) = (op.scalar, op.inner_op)
@inline _scale_parts(op::LazyOp) = (1, op)

# `c * A` for a coefficient just computed by combining/factoring -- collapsing back to the
# identity/zero cases a fresh `OperatorScale(c, A)` would otherwise re-introduce.
@inline function _wrap_scale(c::Number, A::LazyOp)
    iszero(c) && return _zero_of(A)
    isone(c) && return A
    return OperatorScale(c, A)
end
# A `RefValue` coefficient is never statically zero or one; wrap unconditionally.
@inline _wrap_scale(c::Base.RefValue, A::LazyOp) = OperatorScale(c, A)

# --- Structural equality ------------------------------------------------------------- #

"""
    _ast_equal(a, b) -> Bool

Whether two `LazyOp` subtrees are the same expression: same concrete node type, and every
field equal -- recursively for a field that is itself a `LazyOp`, by `===` otherwise.

`===` rather than `==` for a leaf field (a grid function, a closure, a component index) is
deliberate: two arrays that hold equal values right now are not the same operator if one is
later mutated in place (`Rₕ!(cₕ, ...)`) and the other is not, and two independently built
closures are never "the same" scaling function even if they happen to compute the same
thing. Missing an equal-but-distinct pair only forgoes an optimization; treating two
different subtrees as equal would silently change what the assembled form computes, which
is the one thing this pass may never do.
"""
@inline _ast_equal(::LazyOp, ::LazyOp) = false
@inline function _ast_equal(a::T, b::T) where {T<:LazyOp}
    for name in fieldnames(T)
        fa = getfield(a, name)
        fb = getfield(b, name)
        if fa isa LazyOp && fb isa LazyOp
            _ast_equal(fa, fb) || return false
        else
            fa === fb || return false
        end
    end
    return true
end

# --- The pass ------------------------------------------------------------------------ #

"""
    simplify_ast(op) -> LazyOp

Rewrite the algebraic layer of a resolved AST (`OperatorAdd`, `OperatorScale`,
`GridFunctionScale`) into a form that routes to fewer, cheaper mesh sweeps, without
changing what it computes. See the module comment at the top of this file for the rules.

Every other node type is a leaf as far as this pass is concerned and is returned unchanged;
`form(Wₕ, Vₕ, f)`/`form(Wₕ, f)` call this immediately after `resolve_ast`.
"""
@inline simplify_ast(op::LazyOp) = op

function simplify_ast(op::OperatorScale)
    inner = simplify_ast(op.inner_op)
    _is_zero_op(inner) && return inner  # `c * 0 == 0`, whatever `c` is.

    if op.scalar isa Number
        iszero(op.scalar) && return _zero_of(inner)
        isone(op.scalar) && return inner
        # `c1 * (c2 * A) -> (c1 * c2) * A`, only when both scalars are static numbers: a
        # `RefValue` on either side can change after construction, so folding through one
        # would bake in whatever value it happened to hold right now.
        if inner isa OperatorScale && inner.scalar isa Number
            return _wrap_scale(op.scalar * inner.scalar, inner.inner_op)
        end
    end

    return inner === op.inner_op ? op : OperatorScale(op.scalar, inner)
end

function simplify_ast(op::GridFunctionScale)
    inner = simplify_ast(op.inner_op)
    _is_zero_op(inner) && return inner
    return inner === op.inner_op ? op : GridFunctionScale(op.grid_function, inner)
end

function simplify_ast(op::OperatorAdd)
    left = simplify_ast(op.left_op)
    right = simplify_ast(op.right_op)

    _is_zero_op(left) && return right
    _is_zero_op(right) && return left

    cl, al = _scale_parts(left)
    cr, ar = _scale_parts(right)

    if _ast_equal(al, ar)
        # Combine like terms: `c1 * A + c2 * A -> (c1 + c2) * A`. Only when both
        # coefficients are static numbers -- summing across a `RefValue` would freeze a
        # value meant to keep changing.
        cl isa Number && cr isa Number && return _wrap_scale(cl + cr, al)
        # Same `RefValue` object scaling the same subtree on both sides: `c*A + c*A ==
        # 2*(c*A)`, true for whatever `c` holds at assembly time.
        cl === cr && return _wrap_scale(2, left)
    elseif (left isa OperatorScale || right isa OperatorScale) &&
        cl isa Number &&
        cr isa Number &&
        cl == cr
        # Factor a common static scalar out of two different subtrees: `c*A + c*B ->
        # c*(A+B)`. Neither side is zero here (caught above), so `cl == cr` implies both
        # are nonzero. Guarded on an actual `OperatorScale` being present so two already
        # bare, unrelated terms (`cl == cr == 1` always) are not rebuilt for nothing --
        # otherwise this pass would not be idempotent on its own output.
        return _wrap_scale(cl, OperatorAdd(al, ar))
    elseif (left isa OperatorScale || right isa OperatorScale) &&
        cl isa Base.RefValue &&
        cr isa Base.RefValue &&
        cl === cr
        # Same reasoning, for a shared dynamic coefficient.
        return OperatorScale(cl, OperatorAdd(al, ar))
    end

    return left === op.left_op && right === op.right_op ? op : OperatorAdd(left, right)
end
