# simplifier.jl

#=
Algebraic simplification of a resolved AST, run once after `resolve_ast` and before a
`BilinearForm`/`LinearForm` stores it (gpena/Bramble.jl#159).

Most of this rewrites only three node types: `OperatorAdd`, `OperatorScale` and
`GridFunctionScale` -- exactly what `ast.jl`'s `+`, `*` and `/` overloads build. That matters
because the router splits work on `OperatorAdd` alone (`_visit_operator_add*` in
`stencil_eval.jl`); every other node is one routed term, one mesh sweep, however large the
subtree underneath it. Fewer top-level `OperatorAdd` nodes is therefore fewer sweeps for the
same matrix or vector:

  - `c * A + c * B` (same `c`, different `A`/`B`) factors to `c * (A + B)`: two routed terms
    become one.
  - `c1 * A + c2 * A` (same `A`) combines to `(c1 + c2) * A`: two routed terms become one.
  - `0 * A` collapses to a zero term with a one-point sparsity pattern instead of `A`'s full
    stencil, and `A + 0` / `0 + A` drops the zero term from the tree entirely.

Two more rules reach one layer deeper, into `BilinearProduct`/`LinearProduct` (the nodes
`innerₕ`/`inner₊`/... build) and `ShiftNode`, because leaving them out would mean either a
correctness gap (a component-mixing sum inside one inner product currently has no valid
routing at all) or a documented dead end (a hidden scalar defeating symmetry detection):

  - `⟨c * u, v⟩` / `⟨u, c * v⟩` lifts the scalar out to `c * ⟨u, v⟩`, exposing it to the
    rules above and to `symmetry.jl`'s structural shape check, which only recognises
    `⟨L(u), L(v)⟩` when nothing else sits between the product and its arguments.
  - `⟨u, v(i) + v(j)⟩` (or the mirror on the trial side) distributes to
    `⟨u, v(i)⟩ + ⟨u, v(j)⟩` -- but *only* when `i ≠ j` (or one side names a component the
    other does not): that shape has no valid single-term routing today
    (`test_component_or_nothing`/`trial_component_or_nothing`, `block_extract.jl`, throw on
    it), so distributing is the only way to assemble it at all. A same-component sum, the
    ordinary `innerₕ(uₕ, v + 2 * D₋ₓ(v))` case, is left as the single term it already is.
  - `u_h * (v_h * A) -> (u_h .* v_h) * A`, precomputing the elementwise product once rather
    than evaluating both scalings at every grid point of every assembly.
  - `Shift₀(u) -> u`, and two nested shifts along the *same* dimension combine their amounts,
    `Shift_a(Shift_b(u)) -> Shift_{a+b}(u)` (so `Shift_k(Shift_{-k}(u)) -> Shift₀(u) -> u`).

What is not attempted: this stops at `BilinearProduct`/`LinearProduct`/`ShiftNode` and does
not descend into differences, averages, jumps, restrictions or interpolation -- a scalar or
shift buried one layer further in (`D₋ₓ(2 * u)`, say) is not reached. Nor does it fold
`πₕ(Wsrc, u)` away when `Wsrc` happens to be the space `u` is assembled against: that
equality is only known once a concrete trial space is available, which a context-free
rewrite over the expression alone does not have.

Bit-exact with the unsimplified AST: every rule is an algebraic identity over the scalars,
grid functions and components involved, not an approximation.
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

# `(scalar, inner)` for a node the combining/factoring/lifting rules can read a coefficient
# from. A bare (non-`OperatorScale`) node is `1 * itself`. The scalar is returned exactly as
# stored -- a `Base.RefValue` is never dereferenced, since its value can change after the
# form is built (`β[] = 3.0`, gpena/Bramble.jl#159's own worked example).
@inline _scale_parts(op::OperatorScale) = (op.scalar, op.inner_op)
@inline _scale_parts(op::LazyOp) = (1, op)

# `c * A` for a coefficient just computed by combining/factoring/lifting -- collapsing back
# to the identity/zero cases a fresh `OperatorScale(c, A)` would otherwise re-introduce.
@inline function _wrap_scale(c::Number, A::LazyOp)
    iszero(c) && return _zero_of(A)
    isone(c) && return A
    return OperatorScale(c, A)
end
# A `RefValue` coefficient is never statically zero or one; wrap unconditionally.
@inline _wrap_scale(c::Base.RefValue, A::LazyOp) = OperatorScale(c, A)

# Two coefficients just lifted out of the two arguments of one inner product, wrapped around
# the product they were lifted from.
@inline function _lift_scalars(cl, cr, inner::LazyOp)
    cl isa Number && cr isa Number && return _wrap_scale(cl * cr, inner)
    return _wrap_scale(cl, _wrap_scale(cr, inner))
end

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

# --- Component-routing safety -------------------------------------------------------- #

# Whether the router could route `a` and `b` as one opaque term: it recurses through any
# `UnaryWrapper` (including `OperatorScale`/`GridFunctionScale`) down to
# `trial_component_or_nothing`/`test_component_or_nothing`, and *throws*
# (`_throw_mixed_components`, `block_extract.jl`) if the two sides of a sum it meets along
# the way name different components. So a rewrite may hide `OperatorAdd(a, b)` inside a
# scale or a grid-function scaling only when this answers `false`; when it answers `true`,
# distributing over `a`/`b` is the only routing-safe shape; leaving them combined would not
# merely miss an optimisation, it would make an already-throwing pattern reachable through a
# path (a factored scalar, a scale wrapping a distributed sum) that never existed before.
@inline function _mixes_components(a::LazyOp, b::LazyOp)
    return trial_component_or_nothing(a) !== trial_component_or_nothing(b) ||
           test_component_or_nothing(a) !== test_component_or_nothing(b)
end

# --- The pass ------------------------------------------------------------------------ #

"""
    simplify_ast(op) -> LazyOp

Rewrite a resolved AST into a form that routes to fewer, cheaper mesh sweeps, and exposes a
few patterns the router could not assemble at all, without changing what it computes. See
the module comment at the top of this file for the rules.

Every node type not named there is a leaf as far as this pass is concerned and is returned
unchanged; `form(Wₕ, Vₕ, f)`/`form(Wₕ, f)` call this immediately after `resolve_ast`.
"""
@inline simplify_ast(op::LazyOp) = op

function simplify_ast(op::OperatorScale)
    inner = simplify_ast(op.inner_op)
    _is_zero_op(inner) && return inner  # `c * 0 == 0`, whatever `c` is.

    if inner isa OperatorAdd && _mixes_components(inner.left_op, inner.right_op)
        # `c * (A + B)` cannot route as one term when `A`/`B` name different components:
        # distributing is the only routing-safe shape, so it takes priority over the
        # factoring this would otherwise be (rule 2's mirror image).
        return simplify_ast(
            OperatorAdd(
                _wrap_scale(op.scalar, inner.left_op),
                _wrap_scale(op.scalar, inner.right_op),
            ),
        )
    end

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

    if inner isa OperatorAdd && _mixes_components(inner.left_op, inner.right_op)
        # Same reasoning as `OperatorScale` above: a grid-function coefficient distributes
        # over a component-mixing sum rather than hiding it from the router.
        return simplify_ast(
            OperatorAdd(
                GridFunctionScale(op.grid_function, inner.left_op),
                GridFunctionScale(op.grid_function, inner.right_op),
            ),
        )
    end

    # Fuse nested grid-function scalings into one precomputed array: `u_h * (v_h * A) ->
    # (u_h .* v_h) * A`. Paid once, here, rather than as two scalings at every grid point of
    # every assembly -- both `op.grid_function` and `inner.grid_function` are concrete
    # arrays by this point (`resolve_ast` has already called any thunk), so this is an
    # ordinary elementwise multiply, not a deferred one.
    inner isa GridFunctionScale &&
        return GridFunctionScale(op.grid_function .* inner.grid_function, inner.inner_op)

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
        # Combine like terms: `c1 * A + c2 * A -> (c1 + c2) * A`. Structurally equal `al`,
        # `ar` name the same component (or none) by construction -- `_ast_equal` already
        # compared every field, `component_idx` included -- so this is always routing-safe.
        #
        # Only when both coefficients are static numbers -- summing across a `RefValue`
        # would freeze a value meant to keep changing.
        cl isa Number && cr isa Number && return _wrap_scale(cl + cr, al)
        # Same `RefValue` object scaling the same subtree on both sides: `c*A + c*A ==
        # 2*(c*A)`, true for whatever `c` holds at assembly time.
        cl === cr && return _wrap_scale(2, left)
    elseif (left isa OperatorScale || right isa OperatorScale) &&
        cl isa Number &&
        cr isa Number &&
        cl == cr &&
        !_mixes_components(al, ar)
        # Factor a common static scalar out of two different subtrees: `c*A + c*B ->
        # c*(A+B)`. Neither side is zero here (caught above), so `cl == cr` implies both
        # are nonzero. Guarded on an actual `OperatorScale` being present so two already
        # bare, unrelated terms (`cl == cr == 1` always) are not rebuilt for nothing --
        # otherwise this pass would not be idempotent on its own output. Guarded on
        # `!_mixes_components` too: factoring `A`/`B` naming different components would
        # hide the exact shape the router cannot route as one term.
        return _wrap_scale(cl, OperatorAdd(al, ar))
    elseif (left isa OperatorScale || right isa OperatorScale) &&
        cl isa Base.RefValue &&
        cr isa Base.RefValue &&
        cl === cr &&
        !_mixes_components(al, ar)
        # Same reasoning, for a shared dynamic coefficient.
        return OperatorScale(cl, OperatorAdd(al, ar))
    end

    return left === op.left_op && right === op.right_op ? op : OperatorAdd(left, right)
end

# --- Inner products: scalar lifting and component distribution ---------------------- #

function simplify_ast(op::BilinearProduct{D,W}) where {D,W}
    left = simplify_ast(op.left_op)
    right = simplify_ast(op.right_op)

    _is_zero_op(left) && return _zero_of(op)
    _is_zero_op(right) && return _zero_of(op)

    # Distribute over a component-mixing sum on either side: `⟨u, v(i) + v(j)⟩ ->
    # ⟨u, v(i)⟩ + ⟨u, v(j)⟩` for `i ≠ j`. This is the only case that fires -- a
    # same-component (or component-free) sum is left as the single term it already is, so
    # `innerₕ(uₕ, v + 2 * D₋ₓ(v))` keeps assembling as one sweep.
    if right isa OperatorAdd && _mixes_components(right.left_op, right.right_op)
        return simplify_ast(
            OperatorAdd(
                BilinearProduct{D,W,typeof(left),typeof(right.left_op)}(
                    left, right.left_op
                ),
                BilinearProduct{D,W,typeof(left),typeof(right.right_op)}(
                    left, right.right_op
                ),
            ),
        )
    end
    if left isa OperatorAdd && _mixes_components(left.left_op, left.right_op)
        return simplify_ast(
            OperatorAdd(
                BilinearProduct{D,W,typeof(left.left_op),typeof(right)}(
                    left.left_op, right
                ),
                BilinearProduct{D,W,typeof(left.right_op),typeof(right)}(
                    left.right_op, right
                ),
            ),
        )
    end

    # Lift a scalar (static or `Ref`) out of either argument: `⟨c * u, v⟩ -> c * ⟨u, v⟩`,
    # `⟨u, c * v⟩ -> c * ⟨u, v⟩`. Exposes it to the rules above, and to `symmetry.jl`'s
    # `_same_operator_shape`, which only recognises `⟨L(u), L(v)⟩` when nothing else sits
    # between the product and its two arguments.
    cl, al = _scale_parts(left)
    cr, ar = _scale_parts(right)
    inner = BilinearProduct{D,W,typeof(al),typeof(ar)}(al, ar)
    return _lift_scalars(cl, cr, inner)
end

function simplify_ast(op::LinearProduct{D,W}) where {D,W}
    left = simplify_ast(op.left_op)
    right = simplify_ast(op.right_op)

    _is_zero_op(left) && return _zero_of(op)
    _is_zero_op(right) && return _zero_of(op)

    # As `BilinearProduct` above; only the test side can name a component here, since a
    # `LinearProduct`'s left side is always a source (`_is_source_only`).
    if right isa OperatorAdd && _mixes_components(right.left_op, right.right_op)
        return simplify_ast(
            OperatorAdd(
                LinearProduct{D,W,typeof(left),typeof(right.left_op)}(left, right.left_op),
                LinearProduct{D,W,typeof(left),typeof(right.right_op)}(
                    left, right.right_op
                ),
            ),
        )
    end

    cl, al = _scale_parts(left)
    cr, ar = _scale_parts(right)
    inner = LinearProduct{D,W,typeof(al),typeof(ar)}(al, ar)
    return _lift_scalars(cl, cr, inner)
end

# --- Stencil shifts: idempotence and additive composition ---------------------------- #

function simplify_ast(op::ShiftNode{D,Dim}) where {D,Dim}
    inner = simplify_ast(op.inner_op)
    op.shift_amount == 0 && return inner  # Shift₀(u) -> u

    if inner isa ShiftNode{D,Dim}
        # Shift_a(Shift_b(u)) -> Shift_{a+b}(u), along the *same* dimension only -- a shift
        # along a different dimension is a different operation and cannot fold into one
        # node. `a + (-a) = 0` collapses straight to `u`, matching the Shift₀ rule above
        # rather than building a zero-shift node and relying on a second pass to remove it.
        total = op.shift_amount + inner.shift_amount
        total == 0 && return inner.inner_op
        return ShiftNode{D,Dim,typeof(inner.inner_op)}(total, inner.inner_op)
    end

    return if inner === op.inner_op
        op
    else
        ShiftNode{D,Dim,typeof(inner)}(op.shift_amount, inner)
    end
end
