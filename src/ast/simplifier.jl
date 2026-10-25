# simplifier.jl

#=
Algebraic simplification of a resolved AST, run once after `resolve_ast` and before a
`BilinearForm`/`LinearForm` stores it (gpena/Bramble.jl#159).

Each rule below either measured faster or is needed for a form to assemble. The audit of
gpena/Bramble.jl#771 gives the forms, meshes, machine and spread in
`docs/notes/internals/simplify_rules.md`. Most of this rewrites only three node types,
`OperatorAdd`, `OperatorScale` and `GridFunctionScale`, exactly what `ast.jl`'s `+`, `*`
and `/` overloads build. That matters because the router splits work on `OperatorAdd` alone
(`_visit_operator_add*` in `stencil_eval.jl`); every other node is one routed term, one mesh
sweep, however large the subtree underneath it. The rules kept for speed, each with its gain
over the same form without it:

  - Zeros (rule 3): `0 * A` collapses to a zero term with a one-point sparsity pattern
    instead of `A`'s full stencil, and `A + 0` / `0 + A` drops the zero term from the tree
    entirely. Steady-state 2D `assemble!` 456 µs against 530 µs, 14% faster. Only for an
    `Integer` coefficient -- `0.0 * A` keeps its term. See `_wrap_scale` below for why
    (gpena/Bramble.jl#240): that collapse reads the coefficient's *value*, so allowing it on
    a `Float64` would make `form`'s return type depend on a number the compiler need not
    know.
  - Nested scales (rule 4): `c1 * (c2 * A)` folds to `(c1 * c2) * A` when both coefficients
    are static numbers. First 2D `form` and `assemble` 873 ms against 945 ms, steady-state
    3D 407 µs against 422 µs. Not across a `Ref`, whose value can change after the form is
    built, nor a `Rational` (`_folds`): `op / 2` builds `1 // 2`, and checked `Rational`
    products overflow where the floats they stand for do not (gpena/Bramble.jl#633).
  - Same mesh (`same_mesh`, not in this pass): `πₕ(u)` read on the mesh it interpolates
    from is the identity, on a plain mesh only (a ghost mesh's layer is read through the
    clamped physical nodes, `_same_mesh_rule`). Which two meshes meet is known only once
    a block binds its leaves, so this pass keeps `πₕ(u)` in the stored tree and the
    stencil decides (`_same_mesh_stencil`, operators/interpolation.jl): the point's own
    column, weight one, no `locate_cell`, and a pattern without the other corners'
    explicit zeros.
    Steady-state 2D `assemble!` 56-71 µs against 260-328 µs, 4.6x faster, and 3D
    200-236 µs against 410-511 µs, 2.1x (one thread, 61x57 and 17x15x16 points; the notes
    time the dropped-node prototype the `Union` ruled out). It is for `πₕ` written where
    two meshes may coincide (a form over the leaves of a composite space, some on one
    mesh, or two spaces sharing one); interpolating a space onto itself is not a form
    anyone writes for its own sake.

A sum is otherwise kept as written: `c * A + c * B`, `c1 * A + c2 * A` and two inner
products sharing an argument each stay two routed terms. Rules merging them (1, 2 and 7)
were measured and removed; of the candidate rules, four were measured and dropped and two
were not built (`simplify_rules.md`).

Four more rules are needed for correctness and carry no timing (rules 5, 6, 8 and 9). The
first three reach one layer deeper, into `BilinearProduct`/`LinearProduct` (the nodes
`innerₕ`/`inner₊`/... build):

  - Scalar lift (rule 5): `⟨c * u, v⟩` / `⟨u, c * v⟩` lifts the scalar out to
    `c * ⟨u, v⟩`, exposing it to the nested-scale fold above and to `symmetry.jl`'s
    structural shape check, which only recognises `⟨L(u), L(v)⟩` when nothing else sits
    between the product and its arguments.
  - Component distribution (rule 6): `⟨u, v(i) + v(j)⟩` (or the mirror on the trial side)
    distributes to `⟨u, v(i)⟩ + ⟨u, v(j)⟩` -- but *only* when `i ≠ j` (or one side names a
    component the other does not): that shape has no valid single-term routing
    (`test_component_or_nothing`/`trial_component_or_nothing`, `block_extract.jl`, throw on
    it), so distributing is the only way to assemble it at all. A same-component sum, the
    ordinary `innerₕ(uₕ, v + 2 * D₋ₓ(v))` case, is left as the single term it already is.
  - Dirac split (rule 8): `⟨a + δ, v⟩ -> ⟨a, v⟩ + ⟨δ, v⟩` when the sum holds a Dirac
    source, so each term gets its own weight, none for the Dirac one (`_split_dirac_sum`).
  - Wrapper distribution (rule 9): a difference, average, jump, shift or restriction over a
    component-mixing sum distributes over it, `D₋ₓ(v(1) + v(2)) -> D₋ₓ(v(1)) + D₋ₓ(v(2))`,
    so that sum assembles whatever wraps it (gpena/Bramble.jl#759).

A `ShiftNode` is simplified inside but kept as built: `Shift₀(u)` stays, and nested shifts
are not merged, since the amount is a runtime field (gpena/Bramble.jl#530).

What is not attempted otherwise: this stops at `BilinearProduct`/`LinearProduct`/
`ShiftNode` and does not descend into differences, averages, jumps, restrictions or
interpolation -- a scalar or shift buried one layer further in (`D₋ₓ(2 * u)`, say) is not
reached.

Bit-exact with the unsimplified AST for finite coefficients: every rule is an algebraic
identity over the scalars, grid functions and components involved. The one exception is the
`Integer` zero collapse, which drops a term whose coefficients hold `NaN` or `Inf`, where
`0 * NaN` would have given `NaN`.
=#

# --- Zero/identity recognition ------------------------------------------------------ #

# Synthesized only here, for a subtree whose concrete space is not known generically.
# Every consumer of a `ZeroOperator` (`local_stencil`, `stencil_offsets`, `component`,
# `_is_source_only`'s fallback) reads only its `D` type parameter, never `.space` --
# except `_same_operator_shape` (`symmetry.jl`), which compares `a.space === b.space`, and
# `nothing === nothing` settles that the same way two zero operators over the same space
# would: they are the same operator.
@inline _zero_of(::LazyOp{D}) where {D} = ZeroOperator{D, Nothing}(nothing)

# `(scalar, inner)` for a node the lifting rule can read a coefficient from. A bare
# (non-`OperatorScale`) node is `1 * itself`. The scalar is returned exactly as stored -- a
# `Base.RefValue` is never dereferenced, since its value can change after the form is built
# (`β[] = 3.0`, gpena/Bramble.jl#159's own worked example).
@inline _scale_parts(op::OperatorScale) = (op.scalar, op.inner_op)
@inline _scale_parts(op::LazyOp) = (1, op)

# `c * A` for a freshly computed coefficient, collapsing `0 * A` and `1 * A`. Only for an
# `Integer` `c` (gpena/Bramble.jl#240): the collapse reads the value, so it is type-stable
# only when `c` is a compile-time constant. A runtime `Float64` would make `form` a `Union`
# (and Enzyme rejects that). A runtime `Int` does the same, deliberately: pass `float(n)` or
# a `Ref`. `test/form/simplifier.jl` pins that `Union`.
@inline function _wrap_scale(c::Integer, A::LazyOp)
    iszero(c) && return _zero_of(A)
    isone(c) && return A
    return OperatorScale(c, A)
end
@inline _wrap_scale(c::Number, A::LazyOp) = OperatorScale(c, A)
# The sign `_MinusOne` (ast.jl) is neither zero nor one, read off its type.
@inline _wrap_scale(c::_MinusOne, A::LazyOp) = OperatorScale(c, A)
# A `RefValue` coefficient is never statically zero or one; wrap unconditionally.
@inline _wrap_scale(c::Base.RefValue, A::LazyOp) = OperatorScale(c, A)

# Whether a coefficient may be multiplied into another at simplify time, decided by its
# type. A `RefValue` may not: its value can change after the form is built. Nor may a
# `Rational` (`op / 2` is `1 // 2`, gpena/Bramble.jl#633): its arithmetic is checked, so
# `(1 // 2^40) * (1 // 2^40)` overflows and `1 // 0 - 1 // 0` throws, where the floats it
# stands for give `2.0^-80` and `NaN`. Left apart, each converts to the space's element type
# on its own when assembled.
@inline _folds(c) = c isa Number && !(c isa Rational)
@inline _folds(::_MinusOne) = true

# The product of two folding coefficients. Two `Integer`s use checked arithmetic, so
# `2^32 * (2^32 * u)` throws `OverflowError` instead of wrapping to `0 * u` and vanishing
# (gpena/Bramble.jl#704), as an overflowing `Rational` scale throws. Same type as `*`.
@inline _fold_mul(a, b) = a * b
@inline _fold_mul(a::Integer, b::Integer) = Base.Checked.checked_mul(a, b)
# The sign `_MinusOne` (ast.jl) folds as the `Int` `-1` it stands for, against each scalar
# type as `-1` would; two give the literal `1`, so `-(-u)` collapses to `u` from the types
# alone.
@inline _fold_mul(::_MinusOne, ::_MinusOne) = 1
@inline _fold_mul(::_MinusOne, b) = _fold_mul(-1, b)
@inline _fold_mul(a, ::_MinusOne) = _fold_mul(a, -1)

# Two coefficients just lifted out of the two arguments of one inner product, wrapped around
# the product they were lifted from.
@inline function _lift_scalars(cl, cr, inner::LazyOp)
    _folds(cl) && _folds(cr) && return _wrap_scale(_fold_mul(cl, cr), inner)
    return _wrap_scale(cl, _wrap_scale(cr, inner))
end

# --- Component-routing safety -------------------------------------------------------- #

# `_mixes_components` is a *classifying* query -- it must answer for the exact shape it is
# asked to tell apart, a component-mixing `OperatorAdd`, not refuse to look at it
# (gpena/Bramble.jl#235). `trial_component_or_nothing`/`test_component_or_nothing`
# (block_extract.jl) cannot be reused directly for that: they throw
# (`_throw_mixed_components`) the moment *either side itself* already mixes components,
# which a three-or-more-term sum's left-associated `+` makes true of an inner node on every
# recursive call past the first two terms -- `2.0 * (A + B + C)` parses as `2.0 * ((A + B)
# + C)`, and asking whether `(A + B)` and `C` route as one term meant first asking what
# component `(A + B)` itself names, which is exactly the question that node has no single
# answer to and is not what is being asked here.
#
# `_component_class` is the non-throwing mirror those two need: a third value, `Mixed()`,
# stands for "this subtree already names more than one component" instead of throwing.
# `_mixes_components` asks whether the two sides *combine* to `Mixed()`, so a side that
# already mixes makes the pair mix, even beside another mixing side: `Mixed()` is a singleton,
# so two of them are `===`, and comparing the classes alone let `(v(1) + v(2)) + (v(2) +
# v(1))` through undistributed (mixed *anything* is never routable as one term). This mirrors
# `trial_component_or_nothing`/`test_component_or_nothing`'s own recursion field for field,
# not just the top-level shape, so it agrees with them everywhere they do answer.
struct _MixedComponents end

@inline _component_combine(l, r) = l === r ? l : _MixedComponents()

@inline _test_component_class(op::IndexedTestFunction) = op.component_idx
@inline _test_component_class(op::UnaryWrapper) = _test_component_class(op.inner_op)
@inline _test_component_class(op::LinearProduct) = _test_component_class(op.right_op)
@inline _test_component_class(op::BilinearProduct) = _test_component_class(op.right_op)
@inline function _test_component_class(op::OperatorAdd)
    return _component_combine(
        _test_component_class(op.left_op), _test_component_class(op.right_op)
    )
end
@inline _test_component_class(::Any) = nothing

@inline _trial_component_class(op::IndexedTrialFunction) = op.component_idx
@inline _trial_component_class(op::UnaryWrapper) = _trial_component_class(op.inner_op)
@inline _trial_component_class(op::BilinearProduct) = _trial_component_class(op.left_op)
@inline function _trial_component_class(op::OperatorAdd)
    return _component_combine(
        _trial_component_class(op.left_op), _trial_component_class(op.right_op)
    )
end
@inline _trial_component_class(::Any) = nothing

# Whether the router could route `a` and `b` as one opaque term. So a rewrite may hide
# `OperatorAdd(a, b)` inside a scale or a grid-function scaling only when this answers
# `false`; when it answers `true`, distributing over `a`/`b` is the only routing-safe
# shape -- and, since it never throws, that distribution itself is what eventually turns
# every mixing subtree into single-component leaves, the recursion `simplify_ast` already
# performs (`OperatorScale`, `GridFunctionScale`, `BilinearProduct`) reaching a fixed point
# where `_component_class` on every remaining node is a plain `Int` or `nothing`, never
# `Mixed()` -- at which point `block_of`'s own throwing query is the one that runs, and only
# ever on a term that genuinely cannot route (one side named, the other not).
#
# The answer is a value, not a type: it compares the two `component_idx::Int` fields. A
# literal index (`v(1) + v(2)`) constant-folds, but a runtime component index (`v(i)` with
# `i` an argument) leaves it to run time, so each distribution gate below (`OperatorScale`,
# `GridFunctionScale`, `BilinearProduct`, `LinearProduct`, `_LinearWrapper`) makes `form`'s
# type a `Union` of the distributed and undistributed trees when a sum of runtime-indexed
# terms sits inside one inner product, under a difference, average, jump, shift or
# restriction, or under a scalar or grid-function coefficient shared by the whole sum, even
# when every term names the same `i` (gpena/Bramble.jl#529). Separate inner products, each
# with its own coefficient, never reach those gates and infer concretely. Kept on purpose:
# distributing every sum that names a component would infer concretely, but costs a sweep
# for every same-component sum (`v(1) + D₋ₓ(v(1))`).
@inline function _mixes_components(a::LazyOp, b::LazyOp)
    trial = _component_combine(_trial_component_class(a), _trial_component_class(b))
    test = _component_combine(_test_component_class(a), _test_component_class(b))
    return trial isa _MixedComponents || test isa _MixedComponents
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
    inner isa ZeroOperator && return inner  # `c * 0 == 0`, whatever `c` is.

    if inner isa OperatorAdd && _mixes_components(inner.left_op, inner.right_op)
        # `c * (A + B)` cannot route as one term when `A`/`B` name different components:
        # distributing is the only routing-safe shape, so it takes priority over keeping
        # the scaled sum as one term. Each piece is simplified again when that can change
        # it (`_resimplify_moves`); otherwise one generated body distributes the scale.
        _resimplify_moves(typeof(inner)) && return simplify_ast(
            OperatorAdd(
            _wrap_scale(op.scalar, inner.left_op),
            _wrap_scale(op.scalar, inner.right_op)
        ),
        )
        return _scale_over_sum(op.scalar, inner)
    end

    # `Integer`, not `Number`, and for the reason `_wrap_scale` spells out: collapsing on the
    # value of a `Float64` coefficient makes this method's return type depend on a number the
    # compiler may not know (gpena/Bramble.jl#240).
    if op.scalar isa Integer
        iszero(op.scalar) && return _zero_of(inner)
        isone(op.scalar) && return inner
    end

    if op.scalar isa Union{Number, _MinusOne}
        # `c1 * (c2 * A) -> (c1 * c2) * A`, only when both scalars are static numbers: a
        # `RefValue` on either side can change after construction, so folding through one
        # would bake in whatever value it happened to hold right now. Type-stable whatever
        # the values are: both operands' types are known here, so `_wrap_scale` dispatches
        # on a known type and the product's type follows from them alone. A `Rational` on
        # either side does not fold either (`_folds`).
        if inner isa OperatorScale && _folds(op.scalar) && _folds(inner.scalar)
            return _wrap_scale(_fold_mul(op.scalar, inner.scalar), inner.inner_op)
        end
    end

    return inner === op.inner_op ? op : OperatorScale(op.scalar, inner)
end

function simplify_ast(op::GridFunctionScale)
    inner = simplify_ast(op.inner_op)
    inner isa ZeroOperator && return inner

    if inner isa OperatorAdd && _mixes_components(inner.left_op, inner.right_op)
        # Same reasoning as `OperatorScale` above: a grid-function coefficient distributes
        # over a component-mixing sum rather than hiding it from the router.
        return simplify_ast(
            OperatorAdd(
            GridFunctionScale(op.grid_function, inner.left_op),
            GridFunctionScale(op.grid_function, inner.right_op)
        ),
        )
    end

    # Nested scalings `u_h * (v_h * A)` are left as two `GridFunctionScale`s, each reading
    # its own array at evaluation time, so in-place changes to either coefficient reach the
    # next `assemble!` (gpena/Bramble.jl#365). Fusing them into one `u_h .* v_h` array would
    # save a multiply per point but snapshot both coefficients at `form` time.
    return inner === op.inner_op ? op : GridFunctionScale(op.grid_function, inner)
end

# --- Sums: one generated body each (gpena/Bramble.jl#763) ---------------------------- #

# The expression simplifying the node of type `T` reached as `path`. The generators below
# call it, so it is defined before them.
function _simplified_sum_expr(@nospecialize(T), path)
    T <: OperatorAdd || return :(simplify_ast($path))
    l, r = gensym(:l), gensym(:r)
    return quote
        $l = $(_simplified_sum_expr(fieldtype(T, :left_op), :($path.left_op)))
        $r = $(_simplified_sum_expr(fieldtype(T, :right_op), :($path.right_op)))
        $(_sum_dropping_zeros(l, r))
    end
end

_sum_dropping_zeros(l, r) = :($l isa ZeroOperator ? $r :
                              $r isa ZeroOperator ? $l : OperatorAdd($l, $r))

# A sum is simplified from one generated body, not one method per partial sum: each term
# through `simplify_ast`, then every `+` rebuilt with a zero operand dropped, as a recursion
# over `left_op` and `right_op` would. A sum whose terms come back unchanged is rebuilt
# equal (`===`) to itself.
@generated function simplify_ast(op::OperatorAdd)
    return _simplified_sum_expr(op, :op)
end

function _distributed_expr(@nospecialize(T), path)
    T <: OperatorAdd || return :(_scale_piece(c, $path))
    l, r = gensym(:l), gensym(:r)
    return quote
        if _mixes_components($path.left_op, $path.right_op)
            $l = $(_distributed_expr(fieldtype(T, :left_op), :($path.left_op)))
            $r = $(_distributed_expr(fieldtype(T, :right_op), :($path.right_op)))
            $(_sum_dropping_zeros(l, r))
        else
            _scale_piece(c, $path)
        end
    end
end

# `simplify_ast(c * x)` for a sum `x` already simplified whose two operands mix components:
# `c` goes into every sum in `x` that mixes them, and each piece left is scaled by
# `_scale_piece`, which is what simplifying `OperatorAdd(c * x.left_op, c * x.right_op)`
# gives, one sum at a time, without simplifying the terms of `x` a second time. That holds
# only where simplifying `x` again returns it, which `_resimplify_moves` answers before
# this is called. One generated body for the whole of `x`.
@generated function _scale_over_sum(c, x::OperatorAdd)
    return _distributed_expr(x, :x)
end

# Whether simplifying again a tree of type `T`, itself the output of `simplify_ast`, may
# change it. Only a product with a scale operand can: the lift peels one scale off each
# operand per pass, so `⟨r * (2 * u), v⟩` lifts `r` and keeps `2 * u`, and a Dirac split
# keeps `k * f` inside `⟨k * f + δ, v⟩`'s pieces. Every other node it outputs is returned
# as it is, so a sum without such a product distributes a scale without simplifying its
# terms again (`_scale_over_sum`). Read off the type and folded at compile time; a field
# whose type is not concrete answers `true`.
Base.@nospecializeinfer Base.@assume_effects :foldable function _resimplify_moves(
        @nospecialize(T))
    T isa DataType || return true
    if T <: Union{BilinearProduct, LinearProduct}
        (fieldtype(T, :left_op) <: OperatorScale ||
         fieldtype(T, :right_op) <: OperatorScale) && return true
    end
    for i in 1:fieldcount(T)
        F = fieldtype(T, i)
        isconcretetype(F) || return true
        F <: LazyOp && _resimplify_moves(F) && return true
    end
    return false
end

# `simplify_ast(_wrap_scale(c, x))` for an `x` already simplified that is not a sum mixing
# components: dropped or kept whole for an `Integer` zero or one, a zero kept, a nested
# static scale folded (rule 4), anything else scaled.
@inline function _scale_piece(c, x::LazyOp)
    if c isa Integer
        iszero(c) && return _zero_of(x)
        isone(c) && return x
    end
    x isa ZeroOperator && return x
    if x isa OperatorScale && _folds(c) && _folds(x.scalar)
        return _wrap_scale(_fold_mul(c, x.scalar), x.inner_op)
    end
    return OperatorScale(c, x)
end

# --- Dirac sources inside a sum ------------------------------------------------------ #

# `local_stencil(::LinearProduct)` weights a source by the cell measure unless `_is_dirac`
# holds for the whole source (`operators/inner.jl`, beside the other `_is_dirac` methods;
# the wrapper method is in `assembly/stencil_eval.jl`). A sum of Dirac sources is a Dirac
# source too; a sum mixing one with a smooth source is neither, and `simplify_ast` below
# splits it at the top or rejects it under a wrapper. Both read the type alone.
@inline _is_dirac(op::OperatorAdd) = _is_dirac(op.left_op) && _is_dirac(op.right_op)

@inline _any_dirac(::DiracSource) = true
@inline _any_dirac(op::OperatorAdd) = _any_dirac(op.left_op) || _any_dirac(op.right_op)
@inline _any_dirac(op::UnaryWrapper) = _any_dirac(op.inner_op)
@inline _any_dirac(::Any) = false

@noinline function _throw_mixed_dirac_source()
    throw(ArgumentError("a source mixing a dirac term with a non-dirac term under an \
                         operator or grid-function factor is not supported: the dirac \
                         term would be scaled by the cell measure. Write the dirac term in \
                         its own inner product, e.g. `innerₕ(Mₓ(f), v) + innerₕ(Mₓ(δ), v)`."))
end

# --- Inner products: scalar lifting and component distribution ---------------------- #

function simplify_ast(op::BilinearProduct{D, W}) where {D, W}
    left = simplify_ast(op.left_op)
    right = simplify_ast(op.right_op)

    left isa ZeroOperator && return _zero_of(op)
    right isa ZeroOperator && return _zero_of(op)

    # Distribute over a component-mixing sum on either side: `⟨u, v(i) + v(j)⟩ ->
    # ⟨u, v(i)⟩ + ⟨u, v(j)⟩` for `i ≠ j`. This is the only case that fires -- a
    # same-component (or component-free) sum is left as the single term it already is, so
    # `innerₕ(uₕ, v + 2 * D₋ₓ(v))` keeps assembling as one sweep.
    if right isa OperatorAdd && _mixes_components(right.left_op, right.right_op)
        return simplify_ast(
            OperatorAdd(
            BilinearProduct{D, W, typeof(left), typeof(right.left_op)}(
                op.inner, left, right.left_op
            ),
            BilinearProduct{D, W, typeof(left), typeof(right.right_op)}(
                op.inner, left, right.right_op
            )
        ),
        )
    end
    if left isa OperatorAdd && _mixes_components(left.left_op, left.right_op)
        return simplify_ast(
            OperatorAdd(
            BilinearProduct{D, W, typeof(left.left_op), typeof(right)}(
                op.inner, left.left_op, right
            ),
            BilinearProduct{D, W, typeof(left.right_op), typeof(right)}(
                op.inner, left.right_op, right
            )
        ),
        )
    end

    # Lift a scalar (static or `Ref`) out of either argument: `⟨c * u, v⟩ -> c * ⟨u, v⟩`,
    # `⟨u, c * v⟩ -> c * ⟨u, v⟩`. Exposes it to the nested-scale fold, and to
    # `symmetry.jl`'s `_same_operator_shape`, which only recognises `⟨L(u), L(v)⟩` when
    # nothing else sits between the product and its two arguments.
    cl, al = _scale_parts(left)
    cr, ar = _scale_parts(right)
    inner = BilinearProduct{D, W, typeof(al), typeof(ar)}(op.inner, al, ar)
    return _lift_scalars(cl, cr, inner)
end

function simplify_ast(op::LinearProduct{D, W}) where {D, W}
    left = simplify_ast(op.left_op)
    right = simplify_ast(op.right_op)

    left isa ZeroOperator && return _zero_of(op)
    right isa ZeroOperator && return _zero_of(op)

    # As `BilinearProduct` above; only the test side can name a component here, since a
    # `LinearProduct`'s left side is always a source (`_is_source_only`).
    if right isa OperatorAdd && _mixes_components(right.left_op, right.right_op)
        return simplify_ast(
            OperatorAdd(
            LinearProduct{D, W, typeof(left), typeof(right.left_op)}(op.inner, left, right.left_op),
            LinearProduct{D, W, typeof(left), typeof(right.right_op)}(
                op.inner, left, right.right_op
            )
        ),
        )
    end

    cl, al = _scale_parts(left)
    cr, ar = _scale_parts(right)
    # `⟨a + b, v⟩ -> ⟨a, v⟩ + ⟨b, v⟩` when the sum holds a Dirac source, so each term gets
    # its own weight (none for the Dirac one).
    if al isa OperatorAdd && _any_dirac(al)
        return _lift_scalars(cl, cr, _split_dirac_sum(op, al, ar))
    end
    _any_dirac(al) && !_is_dirac(al) && _throw_mixed_dirac_source()
    inner = LinearProduct{D, W, typeof(al), typeof(ar)}(op.inner, al, ar)
    return _lift_scalars(cl, cr, inner)
end

# `⟨a, v⟩` distributed over every sum in `a` that holds a Dirac source, keeping each
# `OperatorScale` as a node around its product. Built from the types alone and never through
# `_wrap_scale`, whose value branch on a stored `Integer` (the `-1` of `δa - δb`) would leave
# `form` a `Union`; a sum with no Dirac term stays one product.
@inline function _split_dirac_sum(op::LinearProduct, a::OperatorAdd, ar)
    _any_dirac(a) || return _dirac_piece(op, a, ar)
    l = _split_dirac_sum(op, a.left_op, ar)
    return OperatorAdd(l, _split_dirac_sum(op, a.right_op, ar))
end
@inline function _split_dirac_sum(op::LinearProduct, a::OperatorScale, ar)
    _any_dirac(a.inner_op) || return _dirac_piece(op, a, ar)
    return OperatorScale(a.scalar, _split_dirac_sum(op, a.inner_op, ar))
end
@inline _split_dirac_sum(op::LinearProduct, a::LazyOp, ar) = _dirac_piece(op, a, ar)

@inline function _dirac_piece(op::LinearProduct{D, W}, a, ar) where {D, W}
    _any_dirac(a) && !_is_dirac(a) && _throw_mixed_dirac_source()
    return LinearProduct{D, W, typeof(a), typeof(ar)}(op.inner, a, ar)
end

# --- Stencil shifts: simplified inside, never folded --------------------------------- #

# A shift is kept as built, `Shift₀(u)` and nested shifts included: the amount is a field, so
# folding would choose the node type from a runtime value and `form` would infer a `Union`
# (gpena/Bramble.jl#530). Opposite senses never cancel: a shift reads 0 off the grid, so
# `S₊ₓ(S₋ₓ(u))` is 0 at the last point (gpena/Bramble.jl#352).
function simplify_ast(op::ShiftNode{D, Dim}) where {D, Dim}
    inner = simplify_ast(op.inner_op)
    _mixes(inner) && return _distribute_wrapper(op, inner)
    return if inner === op.inner_op
        op
    else
        ShiftNode{D, Dim, typeof(inner)}(op.shift_amount, inner)
    end
end

# --- Linear wrappers over a component-mixing sum ------------------------------------ #

# Differences, averages, jumps, shifts and restrictions are linear in their operand, entry
# for entry, so `D₋ₓ(v(1) + v(2))` is `D₋ₓ(v(1)) + D₋ₓ(v(2))` exactly. Left whole, a sum
# naming two components is one term `block_of` cannot route, while the same sum bare or
# under a scale is distributed above, so whether a form assembled hung on how it was
# grouped (gpena/Bramble.jl#759). Each of these now distributes over a mixing operand, all
# the way down a chain of them, as a scale does. An operand naming one component, or none,
# is returned as built and never simplified. `_mixes` reads `_component_class`, which is
# `nothing` or an `Int` by type for an operand with no sum, so such a form compiles as it
# did. A sum of indexed terms under a wrapper is told apart by its `Int` fields, as the
# other gates do, and its form infers the distributed tree too before the literal folds.
# `InterpolationNode` is linear too but is left whole. Its source space is bound per block,
# by the leaf it wraps (`_bind_interp_spaces`), and that path is not exercised here. The
# stencil nodes are the list `_summands_of` splits over too (stencil_nodes.jl); a shift
# keeps its own `simplify_ast` above, which simplifies its operand first.
const _LinearWrapper = Union{_StencilNode, RegionRestriction}

# Whether `op` names more than one component on either side. Only a sum can, so an operand
# whose type holds none (`_may_hold_sum`) answers `false` without compiling the class
# queries over it (gpena/Bramble.jl#763).
@inline _mixes(op) = _may_hold_sum(typeof(op)) &&
                     (_trial_component_class(op) isa _MixedComponents ||
                      _test_component_class(op) isa _MixedComponents)

# Whether a node of type `T` may hold an `OperatorAdd`, read off the type alone and folded
# at compile time; a field whose type is not concrete answers `true`.
Base.@nospecializeinfer Base.@assume_effects :foldable function _may_hold_sum(
        @nospecialize(T))
    T isa DataType || return true
    T <: OperatorAdd && return true
    for i in 1:fieldcount(T)
        F = fieldtype(T, i)
        isconcretetype(F) || return true
        F <: LazyOp && _may_hold_sum(F) && return true
    end
    return false
end

function simplify_ast(op::_LinearWrapper)
    _mixes(op.inner_op) || return op
    return _distribute_wrapper(op, op.inner_op)
end

# `op` distributed over its mixing operand `inner`. A sum is split as it stands, each piece
# simplified on its own, so the branch a same-component sum compiles but never takes stays
# small. Anything else (a scale, another wrapper) is simplified first and split if that
# leaves a sum; a mixing operand that does not (a wrapper this pass does not split) is
# rewrapped, and `block_of` rejects it as before.
function _distribute_wrapper(op, inner::OperatorAdd)
    return OperatorAdd(
        simplify_ast(_rewrap_inner(op, inner.left_op)),
        simplify_ast(_rewrap_inner(op, inner.right_op))
    )
end
function _distribute_wrapper(op, inner)
    s = simplify_ast(inner)
    return s isa OperatorAdd ? _distribute_wrapper(op, s) : _rewrap_inner(op, s)
end

# --- The sign, lowered --------------------------------------------------------------- #

# `op` with each sign `_MinusOne` (ast.jl) left after `simplify_ast` stored as the `Int`
# `-1` it stands for, every other node as it was. `form` calls it on the simplified tree, so
# what assembles, checks symmetry, splits into Kronecker factors or prints holds the `Int`
# `-1` and the sign never meets Base's arithmetic. Read off the types: a node holding no sign
# comes back as itself (`_rewrap_inner`'s shortcut, a sum or product rebuilt equal).
@inline _lower_signs(op) = op
@inline function _lower_signs(op::OperatorScale{D, _MinusOne}) where {D}
    return OperatorScale(-1, _lower_signs(op.inner_op))
end
@inline _lower_signs(op::UnaryWrapper) = _rewrap_inner(op, _lower_signs(op.inner_op))
@inline function _lower_signs(op::OperatorAdd)
    return OperatorAdd(_lower_signs(op.left_op), _lower_signs(op.right_op))
end
@inline function _lower_signs(op::BilinearProduct{D, W}) where {D, W}
    l, r = _lower_signs(op.left_op), _lower_signs(op.right_op)
    return BilinearProduct{D, W, typeof(l), typeof(r)}(op.inner, l, r)
end
@inline function _lower_signs(op::LinearProduct{D, W}) where {D, W}
    l, r = _lower_signs(op.left_op), _lower_signs(op.right_op)
    return LinearProduct{D, W, typeof(l), typeof(r)}(op.inner, l, r)
end
