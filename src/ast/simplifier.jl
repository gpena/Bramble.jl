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
    become one. Only for an `Integer` `c`, for the same reason the zero collapse below is --
    "same `c`" is decided by comparing the two coefficients, and a comparison of two runtime
    numbers is not something the compiler can settle. Nor does a shared `Ref`: two `Ref`s of
    one type are told apart only by object identity, which is a run-time value too
    (gpena/Bramble.jl#527). Nor when `A` or `B` names any component (`_component_free`),
    so a component-mixing sum is never hidden inside the scale (gpena/Bramble.jl#529).
  - `c1 * A + c2 * A` (same `A`) combines to `(c1 + c2) * A`: two routed terms become one.
  - `0 * A` collapses to a zero term with a one-point sparsity pattern instead of `A`'s full
    stencil, and `A + 0` / `0 + A` drops the zero term from the tree entirely. Only for an
    `Integer` coefficient -- `0.0 * A` keeps its term. See `_wrap_scale` below for why
    (gpena/Bramble.jl#240): that collapse reads the coefficient's *value*, so allowing it on
    a `Float64` would make `form`'s return type depend on a number the compiler need not
    know.

Three more rules reach one layer deeper, into `BilinearProduct`/`LinearProduct` (the nodes
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
  - `⟨Au, Bv⟩ + ⟨Au, Cv⟩ -> ⟨Au, (B + C)v⟩`, the mirror `⟨Au, Bv⟩ + ⟨Cu, Bv⟩ ->
    ⟨(A + C)u, Bv⟩`, and `⟨f, Av⟩ + ⟨f, Bv⟩ -> ⟨f, (A + B)v⟩` for a linear form, anywhere in
    a sum (`_absorb`). The shared argument must be a singleton node (`_statically_equal`, as
    for like terms), both products must name no component, and each coefficient moves onto
    its own unshared argument, so no coefficient is compared. Fewer products is fewer
    compiled terms: see `_factor`.
  - `Shift₀(u) -> u`, and two nested shifts along the *same* dimension and in the same sense
    combine their amounts, `Shift_a(Shift_b(u)) -> Shift_{a+b}(u)` for `a` and `b` of one
    sign. Opposite senses stay nested: a shift reads 0 off the grid, so `Shift_k(Shift_{-k}(u))`
    is not `u` at the boundary.

What is not attempted: this stops at `BilinearProduct`/`LinearProduct`/`ShiftNode` and does
not descend into differences, averages, jumps, restrictions or interpolation -- a scalar or
shift buried one layer further in (`D₋ₓ(2 * u)`, say) is not reached. Nor does it fold
`πₕ(u)` away when the space it interpolates from is the one `u` is assembled against: that
equality is only known once a concrete trial space is available, which a context-free
rewrite over the expression alone does not have. Since gpena/Bramble.jl#10 that space is
supplied by `_bind_interp_spaces` at block time, so the fold is available to a pass that
runs there -- not to this one.

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
@inline _zero_of(::LazyOp{D}) where {D} = ZeroOperator{D, Nothing}(nothing)

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
#
# Only for an `Integer` coefficient, and that restriction is the whole point
# (gpena/Bramble.jl#240). Collapsing on the *value* of `c` makes the node type this returns
# -- and so the `AST` type parameter of the `BilinearForm`/`LinearForm` built from it --
# depend on a number the compiler may not know: with `c` a runtime `Float64`, `form` infers
# as `Union{BilinearForm{...,ZeroOperator}, BilinearForm{...,OperatorScale},
# BilinearForm{...,<inner>}}` instead of one concrete type. That costs every caller a
# dynamic dispatch through the assembly engine, and it is exactly the `Union` Enzyme's
# strict-aliasing type analysis rejects with `IllegalTypeAnalysisException`, which is what
# blocks a gradient with respect to an operator's own coefficient.
#
# An `Integer` coefficient keeps the collapse because that is where it is worth having: `1`
# is what `_scale_parts` reports for every bare node (so every lifted inner product would
# otherwise gain a `1 *` wrapper it never had), `0` is what `A - A` and `A + 0 * B` reduce
# to, and no AD backend differentiates an `Integer`. A floating-point coefficient is treated
# as an opaque runtime value instead -- write `0 * A` / `1 * A`, not `0.0 * A` / `1.0 * A`,
# to get the structural collapse.
#
# The restriction buys stability only because an `Integer` coefficient is, in practice, a
# literal: `iszero`/`isone` are then constant-folded and this method has one return type per
# call site. That is a real contract, not an accident, so state it plainly -- **an `Integer`
# coefficient must be a compile-time constant**. A genuinely runtime one (`n::Int` read from a
# parameter) cannot be folded and this method returns
# `Union{ZeroOperator{D,Nothing}, typeof(A), OperatorScale{D,typeof(c),typeof(A)}}`, which is
# the same instability the `Float64` case was restricted to avoid. There is no way to ask, from
# inside a function, whether a value is known to inference, and removing the value branch
# altogether would cost `0 * A` and `1 * A` their collapse -- both documented, and `1` is what
# `_scale_parts` reports for every bare node. So the hole stays, deliberately, with a narrow
# fix available to the caller: pass a runtime scalar as a `Float64` (`float(n)`) or wrap it in
# a `Ref`. `test/form/simplifier.jl` pins the `Union` so a future change to this trade-off has
# to be made on purpose.
@inline function _wrap_scale(c::Integer, A::LazyOp)
    iszero(c) && return _zero_of(A)
    isone(c) && return A
    return OperatorScale(c, A)
end
@inline _wrap_scale(c::Number, A::LazyOp) = OperatorScale(c, A)
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
@inline function _ast_equal(a::T, b::T) where {T <: LazyOp}
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

# Whether `_ast_equal(a, b)` is settled by the two types alone, with no field read. The
# like-term rule is gated on this *before* it asks `_ast_equal` anything: the rule returns a
# different *node type* depending on the answer, so an answer the compiler cannot fold makes
# `form`'s return type a `Union` of the rewritten and unrewritten trees -- the same defect
# gpena/Bramble.jl#240 fixed for coefficient *values*, one level down, in the subtrees those
# coefficients scale. It bit every sum of two same-shaped terms carrying runtime data, not
# only the duplicate expressions it was once thought to: `innerₕ(g₁ * u, v) + innerₕ(g₂ * u, v)`
# for distinct grid functions `g₁`, `g₂` inferred as `Union{OperatorAdd, OperatorScale}`, and
# `IllegalTypeAnalysisException` is what Enzyme makes of that.
#
# `issingletontype` is exactly the property needed: a type with no non-singleton fields has
# one value, so any two of its instances are `===` and `_ast_equal` is constant-`true` after
# inlining. Anything else -- a `GridFunctionScale` holding an array, a `DiracSource`, an
# `IndexedTrialFunction` whose `component_idx` is a field rather than a type parameter -- is
# left as the sum it was written as. That costs a routed term and changes no number.
#
# The pure-operator trees `form` actually builds *are* singletons: a `BilinearProduct` over
# `TrialFunction`/`TestFunction` and any stack of difference or average wrappers has singleton
# fields all the way down, so `innerₕ(u, v) + innerₕ(u, v)` and its static-coefficient
# multiples still combine. `Ref`-coefficient sums do not: see the like-term branch below.
@inline _statically_equal(::LazyOp, ::LazyOp) = false
@inline _statically_equal(::T, ::T) where {T <: LazyOp} = Base.issingletontype(T)

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
# `Mixed() !== <anything>`, including another `Mixed()` from an unrelated subtree, so two
# differently-mixing sides still compare as "differ" -- which is the answer `_mixes_components`
# needs (mixed *anything* is never routable as one opaque term). This mirrors
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
# `GridFunctionScale`, `BilinearProduct`, `LinearProduct`) makes `form`'s type a `Union` of
# the distributed and undistributed trees when a sum of runtime-indexed terms sits inside one
# inner product or under a scalar or grid-function coefficient shared by the whole sum, even
# when every term names the same `i` (gpena/Bramble.jl#529). Separate inner products, each
# with its own coefficient, never reach those gates and infer concretely. Kept on purpose:
# distributing every sum that is not `_component_free` would infer concretely, but costs a
# sweep for every same-component sum (`v(1) + D₋ₓ(v(1))`). The factor rule in
# `simplify_ast(::OperatorAdd)` uses `_component_free` instead, which is decided by type.
@inline function _mixes_components(a::LazyOp, b::LazyOp)
    return _trial_component_class(a) !== _trial_component_class(b) ||
           _test_component_class(a) !== _test_component_class(b)
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
            _wrap_scale(op.scalar, inner.right_op)
        ),
        )
    end

    # `Integer`, not `Number`, and for the reason `_wrap_scale` spells out: collapsing on the
    # value of a `Float64` coefficient makes this method's return type depend on a number the
    # compiler may not know (gpena/Bramble.jl#240).
    if op.scalar isa Integer
        iszero(op.scalar) && return _zero_of(inner)
        isone(op.scalar) && return inner
    end

    if op.scalar isa Number
        # `c1 * (c2 * A) -> (c1 * c2) * A`, only when both scalars are static numbers: a
        # `RefValue` on either side can change after construction, so folding through one
        # would bake in whatever value it happened to hold right now. Type-stable whatever
        # the values are: both operands' types are known here, so `_wrap_scale` dispatches
        # on a known type and the product's type follows from them alone.
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

function simplify_ast(op::OperatorAdd)
    left = simplify_ast(op.left_op)
    right = simplify_ast(op.right_op)

    _is_zero_op(left) && return right
    _is_zero_op(right) && return left

    cl, al = _scale_parts(left)
    cr, ar = _scale_parts(right)

    if _statically_equal(al, ar) && _ast_equal(al, ar)
        # Combine like terms: `c1 * A + c2 * A -> (c1 + c2) * A`. `_ast_equal` stays the
        # definition of "the same subtree"; `_statically_equal` in front of it is the gate on
        # whether that question may be *asked* at all. This branch and the fall-through below
        # return different node types, so the decision has to be one inference can fold
        # (gpena/Bramble.jl#240 -- see `_statically_equal`'s own comment for the `Union` this
        # otherwise produces). Past a `true` gate the type is a singleton and `_ast_equal`
        # folds to `true` as well, so the pair costs one constant, not two comparisons.
        #
        # Routing-safe because a singleton type has no runtime `component_idx` for the two
        # sides to disagree on -- `IndexedTrialFunction`/`IndexedTestFunction` carry theirs as
        # a field, so they are never singletons and never reach here.
        #
        # Only when both coefficients are static numbers -- summing across a `RefValue`
        # would freeze a value meant to keep changing. With a `RefValue` on either side
        # control falls through to the `OperatorAdd` at the bottom. Not even the same `Ref`
        # on both sides combines: `cl === cr` on two `RefValue`s of one type is object
        # identity, a run-time value, so `form` would infer a `Union`
        # (gpena/Bramble.jl#527).
        cl isa Number && cr isa Number && return _wrap_scale(cl + cr, al)
    elseif (left isa OperatorScale || right isa OperatorScale) &&
           cl isa Integer &&
           cr isa Integer &&
           cl == cr &&
           _component_free(al) &&
           _component_free(ar)
        # Factor a common static scalar out of two different subtrees: `c*A + c*B ->
        # c*(A+B)`. Neither side is zero here (caught above), so `cl == cr` implies both
        # are nonzero. Guarded on an actual `OperatorScale` being present so two already
        # bare, unrelated terms (`cl == cr == 1` always) are not rebuilt for nothing --
        # otherwise this pass would not be idempotent on its own output. Guarded on
        # `_component_free` too, the policy `_factor_products` applies: factoring `A`/`B`
        # naming different components would hide the shape the router cannot route as one
        # term, and `_component_free` settles that by type. `!_mixes_components` would let
        # same-component terms factor, but it compares two `Int` fields, so a runtime
        # component index would make `form` infer a `Union` (gpena/Bramble.jl#529).
        #
        # `Integer`, not `Number`, for the reason `_wrap_scale` gives (gpena/Bramble.jl#240):
        # whether this rule fires is decided by comparing two coefficients, so with `Float64`
        # coefficients the *type* of what this method returns -- `OperatorScale` here,
        # `OperatorAdd` at the bottom -- depends on a comparison the compiler cannot make.
        # That is a `Union` in `form`'s return type for every sum of runtime-scaled terms,
        # and `IllegalTypeAnalysisException` under Enzyme. `2 * A + 2 * B` still factors;
        # `2.0 * A + 2.0 * B` assembles as the two terms it was written as. So does a
        # shared `Ref`: object identity of two `RefValue`s of one type is just as invisible
        # to inference (gpena/Bramble.jl#527). Write `β * (A + B)` for one sweep.
        return _wrap_scale(cl, OperatorAdd(al, ar))
    end

    # Factor a shared inner-product argument out of `right` and whichever summand of `left`
    # shares one: `⟨Au, Bv⟩ + ⟨Au, Cv⟩ -> ⟨Au, (B + C)v⟩` and the mirror on the test side.
    # `left` is already factored, so a left-deep sum of many terms reaches every match.
    merged = _absorb(left, right)
    merged === nothing || return merged

    return left === op.left_op && right === op.right_op ? op : OperatorAdd(left, right)
end

# --- Shared inner-product arguments ------------------------------------------------- #

# Whether `op` names no component on either side, settled by its type: the two classes are
# `nothing` by type or an `Int` field, never compared at run time. Factoring is limited to
# component-free products, so it can never assemble a component-mixing sum as one term.
@inline _component_free(op::LazyOp) = _trial_component_class(op) === nothing && _test_component_class(op) === nothing

# `X` carried under the coefficient the term `t` had, decided by `t`'s type and not by the
# coefficient's value, so an `Integer` read from a stored `OperatorScale` never reaches
# `_wrap_scale`'s value branch and the result keeps one type.
@inline _rescale(t::OperatorScale, X::LazyOp) = OperatorScale(t.scalar, X)
@inline _rescale(::LazyOp, X::LazyOp) = X

# The product of `a` and `b` with their shared argument factored out, or `nothing`. Which
# one it returns depends only on the argument types: the shared argument must pass
# `_statically_equal` (a singleton type, so data-carrying arguments never factor), exactly
# as the like-term rule requires of the terms it combines. Each term's coefficient moves
# onto its own unshared argument, `c⟨Au, Bv⟩ = ⟨Au, c Bv⟩`, so coefficients are never
# compared at all. Identical products are left to the like-term rule.
#
# One factored product compiles faster than the products it replaces: the 3D scalar form of
# 27 distinct `innerₕ` terms whose pairs share trial operators three at a time, first
# assemble, 16.4–17.1 s unfactored, 6.6–6.7 s factored (interleaved, 2 threads).
@inline _factor(::LazyOp, ::LazyOp) = nothing
@inline _factor_products(a, b, pa, pb) = nothing
@inline _factor(a::OperatorScale, b::LazyOp) = _factor_products(a, b, a.inner_op, b)
@inline _factor(a::LazyOp, b::OperatorScale) = _factor_products(a, b, a, b.inner_op)
@inline _factor(a::OperatorScale, b::OperatorScale) = _factor_products(a, b, a.inner_op, b.inner_op)
@inline _factor(a::BilinearProduct, b::BilinearProduct) = _factor_products(a, b, a, b)
@inline _factor(a::LinearProduct, b::LinearProduct) = _factor_products(a, b, a, b)

function _factor_products(a, b, pa::BilinearProduct{D, W}, pb::BilinearProduct{D, W}) where {D, W}
    (_component_free(pa) && _component_free(pb)) || return nothing
    _statically_equal(pa, pb) && return nothing
    if _statically_equal(pa.left_op, pb.left_op) && _ast_equal(pa.left_op, pb.left_op)
        r = OperatorAdd(_rescale(a, pa.right_op), _rescale(b, pb.right_op))
        return BilinearProduct{D, W, typeof(pa.left_op), typeof(r)}(pa.left_op, r)
    end
    if _statically_equal(pa.right_op, pb.right_op) && _ast_equal(pa.right_op, pb.right_op)
        l = OperatorAdd(_rescale(a, pa.left_op), _rescale(b, pb.left_op))
        return BilinearProduct{D, W, typeof(l), typeof(pa.right_op)}(l, pa.right_op)
    end
    return nothing
end

# A linear form's left argument is its source, so only the source can be shared.
function _factor_products(a, b, pa::LinearProduct{D, W}, pb::LinearProduct{D, W}) where {D, W}
    (_component_free(pa) && _component_free(pb)) || return nothing
    _statically_equal(pa, pb) && return nothing
    if _statically_equal(pa.left_op, pb.left_op) && _ast_equal(pa.left_op, pb.left_op)
        r = OperatorAdd(_rescale(a, pa.right_op), _rescale(b, pb.right_op))
        return LinearProduct{D, W, typeof(pa.left_op), typeof(r)}(pa.left_op, r)
    end
    return nothing
end

# `s` with `t` factored into the first summand (searched right to left) sharing an argument
# with it, or `nothing` when none does.
@inline _absorb(s::LazyOp, t::LazyOp) = _factor(s, t)
function _absorb(s::OperatorAdd, t::LazyOp)
    r = _absorb(s.right_op, t)
    r === nothing || return OperatorAdd(s.left_op, r)
    l = _absorb(s.left_op, t)
    l === nothing || return OperatorAdd(l, s.right_op)
    return nothing
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

    _is_zero_op(left) && return _zero_of(op)
    _is_zero_op(right) && return _zero_of(op)

    # Distribute over a component-mixing sum on either side: `⟨u, v(i) + v(j)⟩ ->
    # ⟨u, v(i)⟩ + ⟨u, v(j)⟩` for `i ≠ j`. This is the only case that fires -- a
    # same-component (or component-free) sum is left as the single term it already is, so
    # `innerₕ(uₕ, v + 2 * D₋ₓ(v))` keeps assembling as one sweep.
    if right isa OperatorAdd && _mixes_components(right.left_op, right.right_op)
        return simplify_ast(
            OperatorAdd(
            BilinearProduct{D, W, typeof(left), typeof(right.left_op)}(
                left, right.left_op
            ),
            BilinearProduct{D, W, typeof(left), typeof(right.right_op)}(
                left, right.right_op
            )
        ),
        )
    end
    if left isa OperatorAdd && _mixes_components(left.left_op, left.right_op)
        return simplify_ast(
            OperatorAdd(
            BilinearProduct{D, W, typeof(left.left_op), typeof(right)}(
                left.left_op, right
            ),
            BilinearProduct{D, W, typeof(left.right_op), typeof(right)}(
                left.right_op, right
            )
        ),
        )
    end

    # Lift a scalar (static or `Ref`) out of either argument: `⟨c * u, v⟩ -> c * ⟨u, v⟩`,
    # `⟨u, c * v⟩ -> c * ⟨u, v⟩`. Exposes it to the rules above, and to `symmetry.jl`'s
    # `_same_operator_shape`, which only recognises `⟨L(u), L(v)⟩` when nothing else sits
    # between the product and its two arguments.
    cl, al = _scale_parts(left)
    cr, ar = _scale_parts(right)
    inner = BilinearProduct{D, W, typeof(al), typeof(ar)}(al, ar)
    return _lift_scalars(cl, cr, inner)
end

function simplify_ast(op::LinearProduct{D, W}) where {D, W}
    left = simplify_ast(op.left_op)
    right = simplify_ast(op.right_op)

    _is_zero_op(left) && return _zero_of(op)
    _is_zero_op(right) && return _zero_of(op)

    # As `BilinearProduct` above; only the test side can name a component here, since a
    # `LinearProduct`'s left side is always a source (`_is_source_only`).
    if right isa OperatorAdd && _mixes_components(right.left_op, right.right_op)
        return simplify_ast(
            OperatorAdd(
            LinearProduct{D, W, typeof(left), typeof(right.left_op)}(left, right.left_op),
            LinearProduct{D, W, typeof(left), typeof(right.right_op)}(
                left, right.right_op
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
    inner = LinearProduct{D, W, typeof(al), typeof(ar)}(al, ar)
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

@inline function _dirac_piece(::LinearProduct{D, W}, a, ar) where {D, W}
    _any_dirac(a) && !_is_dirac(a) && _throw_mixed_dirac_source()
    return LinearProduct{D, W, typeof(a), typeof(ar)}(a, ar)
end

# --- Stencil shifts: idempotence and additive composition ---------------------------- #

function simplify_ast(op::ShiftNode{D, Dim}) where {D, Dim}
    inner = simplify_ast(op.inner_op)
    op.shift_amount == 0 && return inner  # Shift₀(u) -> u

    if inner isa ShiftNode{D, Dim} && sign(inner.shift_amount) == sign(op.shift_amount)
        # Shift_a(Shift_b(u)) -> Shift_{a+b}(u), along the *same* dimension and in the same
        # sense only. A shift along a different dimension is a different operation. Opposite
        # senses do not cancel at the boundary: a shift reads 0 off the grid, so
        # `S₊ₓ(S₋ₓ(u))` is `u` in the interior but 0 at the last point, where `S₋ₓ(u)`'s
        # value there was never read (gpena/Bramble.jl#352). Two reads in the same sense
        # leave the grid exactly where the merged one does, so that fold is exact.
        total = op.shift_amount + inner.shift_amount
        return ShiftNode{D, Dim, typeof(inner.inner_op)}(total, inner.inner_op)
    end

    return if inner === op.inner_op
        op
    else
        ShiftNode{D, Dim, typeof(inner)}(op.shift_amount, inner)
    end
end
