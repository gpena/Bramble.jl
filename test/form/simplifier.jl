module FormSimplifierTests

using Test
using Bramble
using LinearAlgebra: Diagonal, issymmetric, isposdef
using SparseArrays: nnz
using Random: seed!
using ForwardDiff: ForwardDiff
using ..TestUtils: WITH_SLOW_TESTS, WITH_3D_TESTS
using Bramble:
               D₋ᵧ,
               IdentityOperator,
               ZeroOperator,
               OperatorAdd,
               OperatorScale,
               GridFunctionScale,
               BilinearProduct,
               LinearProduct,
               InterpolationNode,
               DiracSource,
               ShiftNode,
               shift_op,
               source_function,
               dirac,
               simplify_ast,
               resolve_ast,
               resolve_form_ast,
               trial_function,
               test_function,
               form,
               assemble,
               Innerh,
               Innerplus,
               D₋₂,
               D₋ₓ,
               D₊ₓ,
               Dcₓ,
               D₊ᵧ,
               Mₓ,
               jumpₓ,
               inner₊ᵧ,
               inner₊ₓ,
               weights,
               S₊ₓ,
               S₋ₓ,
               forward_shift,
               backward_shift

# `simplify_ast` rewrites only the algebraic layer (`OperatorAdd`, `OperatorScale`,
# `GridFunctionScale`) that `ast.jl`'s `+`/`*`/`/` overloads build: zero terms drop out and
# nested static scales fold. Every check here is against either
# a hand-built reference matrix or a form assembled through a completely different call --
# never against another call to the code under test.

@testset "AST algebraic simplification" begin
    Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 6), (true, true))
    Wₕ = gridspace(Ωₕ)
    A = IdentityOperator(Wₕ)   # a leaf standing in for an arbitrary operator
    B = D₋ₓ(A)                 # a second leaf, structurally different from `A`

    # `S` and `T` are inner products of a trial and a test function, the shape every form
    # built from operators alone has; `A` and `B` carry the grid space in a field.
    S = innerₕ(trial_function(Wₕ), test_function(Wₕ))
    T = inner₊ₓ(D₋ₓ(trial_function(Wₕ)), D₋ₓ(test_function(Wₕ)))

    @testset "Zero and identity" begin
        @test simplify_ast(0 * A) isa ZeroOperator
        @test simplify_ast(1 * A) === A

        # A floating-point coefficient is deliberately left alone, however it compares to
        # `0` or `1`: collapsing on the value would make the node
        # type -- and so the `AST` type parameter of any form built from it -- depend on a
        # number the compiler need not know, which costs `form` its inferred return type
        # for a runtime coefficient and raises `IllegalTypeAnalysisException` under Enzyme.
        @test simplify_ast(0.0 * A) isa OperatorScale
        @test simplify_ast(1.0 * A) isa OperatorScale

        # the zero side of a sum vanishes from the tree entirely, on either side
        @test simplify_ast(A + 0 * B) === A
        @test simplify_ast(0 * B + A) === A

        # `c * 0 == 0` regardless of what scales it, including a dynamic coefficient
        β = Ref(3.0)
        @test simplify_ast(β * (0 * A)) isa ZeroOperator

        # nested static scalars fold: `c1 * (c2 * A) -> (c1 * c2) * A`
        s = simplify_ast(2 * (3 * A))
        @test s isa OperatorScale
        @test s.scalar == 6
        @test s.inner_op === A

        # ... but never across a `RefValue`: its value can change after the form is built
        r = simplify_ast(3 * (β * A))
        @test r isa OperatorScale
        @test r.scalar == 3
        @test r.inner_op isa OperatorScale
        @test r.inner_op.scalar === β
        @test r.inner_op.inner_op === A
    end

    @testset "Int folding throws on overflow" begin
        # gpena/Bramble.jl#704: wrapping `Int` arithmetic turned `2^32 * (2^32 * S)` into a
        # `ZeroOperator`.
        u, v = trial_function(Wₕ), test_function(Wₕ)
        @test_throws OverflowError simplify_ast(2^32 * (2^32 * S))
        @test_throws OverflowError simplify_ast(innerₕ(2^32 * u, 2^32 * v))
        # the largest products that fit still fold exactly
        c = simplify_ast(2^31 * (2^31 * S))
        @test c isa OperatorScale && c.scalar === 2^62 && c.inner_op === S
        c = simplify_ast(innerₕ(2^31 * u, 2^31 * v))
        @test c isa OperatorScale && c.scalar === 2^62 && c.inner_op === S
    end

    @testset "Idempotent and recursive" begin
        vₕ_deep = Rₕ(Wₕ, x -> x[1])

        # simplifying an already-simplified tree is a no-op, here one the pass rebuilt
        t = simplify_ast(2 * (3 * A) + (0 * T + 1 * B))
        @test t isa OperatorAdd && t.right_op === B
        @test simplify_ast(t) === t

        # the rules apply however deep the algebra sits, not only at the root: a zero term
        # drops, a unit scale goes and a nested scale folds, three sums down
        nested = simplify_ast(S + (0 * T + (1 * (2 * (3 * S)))))
        @test nested isa OperatorAdd
        @test nested.left_op === S
        @test nested.right_op isa OperatorScale
        @test nested.right_op.scalar == 6
        @test nested.right_op.inner_op === S
        @test simplify_ast(nested) === nested

        # the zero and identity collapses themselves are leaf-agnostic -- they read the
        # coefficient, not the subtree -- so they still reach a data-carrying leaf
        @test simplify_ast(A + (0 * B + (1 * A))) isa OperatorAdd
        @test simplify_ast(0 * (vₕ_deep * B)) isa ZeroOperator
    end
end

# Full construction -> assembly round trips, checked against matrices built by code that
# shares nothing with the simplifier: either direct linear algebra, or a second `form` call
# whose own AST never touches the rule under test.
@testset "Simplified forms assemble correctly" begin
    S = interval(0.0, 1.0) × interval(0.0, 1.0)
    Ωₕ = mesh(domain(S, :walls => boundary_symbols(S)), (9, 7), (true, true))
    Wₕ = gridspace(Ωₕ)

    H = Matrix(Diagonal(collect(weights(Wₕ, Innerh()))))

    @testset "zero-scaled term: pattern elided" begin
        a_ref = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v))
        a_zero = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + 0 * inner₊ₓ(D₋ₓ(u), D₋ₓ(v)))

        A_ref = assemble(a_ref)
        A_zero = assemble(a_zero)
        @test Matrix(A_zero) ≈ Matrix(A_ref)
        # the zero-scaled `inner₊ₓ(D₋ₓ(u), D₋ₓ(v))` term contributes nothing to the
        # pattern: without elision it would add its own (much wider) stencil's nonzeros.
        @test nnz(A_zero) == nnz(A_ref)
        @test !(resolve_form_ast(a_zero) isa OperatorAdd)

        # The integer `0` above is what buys that, and the restriction is deliberate:
        # a floating-point `0.0` keeps its term, so the matrix
        # still holds the stiffness band as stored zeros. Same numbers, wider pattern.
        a_zero_float = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + 0.0 * inner₊ₓ(D₋ₓ(u), D₋ₓ(v)))
        A_zero_float = assemble(a_zero_float)
        @test Matrix(A_zero_float) ≈ Matrix(A_ref)
        @test nnz(A_zero_float) > nnz(A_ref)
    end

    # Two terms scaled by the same `Ref` stay two terms, and both track its updates
    # (gpena/Bramble.jl#527).
    @testset "RefValue coefficient tracks updates" begin
        β = Ref(1.0)
        a = form(Wₕ, Wₕ, (u, v) -> β * innerₕ(u, v) + β * innerₕ(u, v))
        @test resolve_form_ast(a) isa OperatorAdd

        @test Matrix(assemble(a)) ≈ 2 .* H
        β[] = 3.0
        @test Matrix(assemble(a)) ≈ 6 .* H
    end
end

# A sum is kept as written (gpena/Bramble.jl#771): no two terms are merged, whatever they
# share. The tree keeps every term and coefficient, and each matrix equals the sum of its
# terms assembled one form at a time, on a non-uniform mesh.
_rt_two_term(θ::Float64, W) = form(W, W,
    (u, v) -> θ * innerₕ(D₋ₓ(u), D₊ₓ(v)) + innerₕ(D₋ₓ(u), D₋ᵧ(v)))

@testset "Sums kept as written" begin
    Wₕ = gridspace(mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 8),
        (false, false)))
    A = IdentityOperator(Wₕ)
    B = D₋ₓ(A)
    M(g) = Matrix(assemble(form(Wₕ, Wₕ, g)))

    @testset "the tree" begin
        # one `Ref` on both terms stays on both, by identity, and is never read
        β = Ref(1.5)
        f = simplify_ast(β * A + β * B)
        @test f isa OperatorAdd
        @test f.left_op.scalar === β && f.right_op.scalar === β
        @test f.left_op.inner_op === A && f.right_op.inner_op === B
        # a difference of two equal terms is not collapsed to zero
        d = simplify_ast(A - A)
        @test d isa OperatorAdd && !(d isa ZeroOperator)
    end

    @testset "the matrix" begin
        # the same grid-function scaling in two terms
        gₕ = Rₕ(Wₕ, x -> 1.0 + x[1])
        @test M((u, v) -> 2 * innerₕ(gₕ * u, v) + 3 * innerₕ(gₕ * u, v)) ≈
              5 .* M((u, v) -> innerₕ(gₕ * u, v))
        # four terms, two pairs sharing an argument, one scaled
        @test M((u, v) -> innerₕ(D₋ₓ(u), D₊ₓ(v)) + innerₕ(Dcₓ(u), D₊ᵧ(v)) +
                          innerₕ(D₊ᵧ(u), D₋ᵧ(v)) + 2 * innerₕ(D₋ₓ(u), D₋ᵧ(v))) ≈
              M((u, v) -> innerₕ(D₋ₓ(u), D₊ₓ(v))) + M((u, v) -> innerₕ(Dcₓ(u), D₊ᵧ(v))) +
              M((u, v) -> innerₕ(D₊ᵧ(u), D₋ᵧ(v))) + 2 .* M((u, v) -> innerₕ(D₋ₓ(u), D₋ᵧ(v)))
        # two distinct `Ref`s, one changed after `form`: the next assembly reads it
        a, b = Ref(2.0), Ref(3.0)
        coef = form(Wₕ, Wₕ,
            (u, v) -> innerₕ(a * D₋ₓ(u), D₊ₓ(v)) + innerₕ(b * D₋ₓ(u), D₋ᵧ(v)))
        @test simplify_ast(coef.ast) === coef.ast
        a[] = 5.0
        @test Matrix(assemble(coef)) ≈
              5.0 .* M((u, v) -> innerₕ(D₋ₓ(u), D₊ₓ(v))) +
              3.0 .* M((u, v) -> innerₕ(D₋ₓ(u), D₋ᵧ(v)))
    end

    # a runtime `Float64` coefficient on one of two terms keeps `form` concrete
    @test isconcretetype(only(Base.return_types(_rt_two_term, (Float64, typeof(Wₕ)))))
end

# Scalar lifting out of an inner product, component distribution on a component-mixing sum
# inside one, nested grid-function scales, and shifts kept as built. Each reaches one layer
# deeper than the rules above (into `BilinearProduct`/`LinearProduct`/`ShiftNode`), so every
# one gets its own structural check plus a numeric check against an independent reference.
@testset "Scalar lifting out of inner products" begin
    Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 6), (true, true))
    Wₕ = gridspace(Ωₕ)
    A = IdentityOperator(Wₕ)
    B = D₋ₓ(A)

    @testset "The rewrite itself" begin
        s = simplify_ast(innerₕ(2 * A, B))
        @test s isa OperatorScale
        @test s.scalar == 2
        @test s.inner_op isa BilinearProduct
        @test s.inner_op.left_op === A
        @test s.inner_op.right_op === B

        @test simplify_ast(innerₕ(A, 3 * B)).scalar == 3
        @test simplify_ast(innerₕ(2 * A, 3 * B)).scalar == 6

        # a zero coefficient on either side collapses the whole product, not just its side
        @test simplify_ast(innerₕ(0 * A, B)) isa ZeroOperator
        @test simplify_ast(innerₕ(A, 0 * B)) isa ZeroOperator
    end

    @testset "restores symmetry/SPD detection" begin
        # Before this rule, `innerₕ(2 * D₋ₓ(u), D₋ₓ(v))`'s trial side is an `OperatorScale`
        # and its test side a bare `BackwardDifference` -- different top-level types, so
        # `_same_operator_shape` (symmetry.jl) answered `false` even though `2 * ⟨Lu, Lv⟩`
        # is exactly the symmetric, positive-semidefinite shape it exists to recognise.
        a_hidden = form(Wₕ, Wₕ, (u, v) -> innerₕ(2 * D₋ₓ(u), D₋ₓ(v)))
        @test issymmetric(a_hidden)
        @test isposdef(a_hidden)
        @test issymmetric(Matrix(assemble(a_hidden)))

        # A negative lifted scale is still symmetric, but no longer positive-definite.
        a_neg = form(Wₕ, Wₕ, (u, v) -> innerₕ(-2 * D₋ₓ(u), D₋ₓ(v)))
        @test issymmetric(a_neg)
        @test !isposdef(a_neg)
    end

    # Each term of a sum lifts its own scalar; the two terms stay two.
    @testset "lifted scalar: bilinear sum" begin
        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(2 * D₋ₓ(u), v) + innerₕ(3 * D₋ₓ(u), v))
        ast = resolve_form_ast(a)
        @test ast isa OperatorAdd
        @test ast.left_op isa OperatorScale && ast.left_op.scalar == 2
        @test ast.right_op isa OperatorScale && ast.right_op.scalar == 3
        @test ast.left_op.inner_op isa BilinearProduct
        @test ast.right_op.inner_op isa BilinearProduct
        @test Matrix(assemble(a)) ≈
              5 .* Matrix(assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(D₋ₓ(u), v))))
    end

    @testset "lifting: symbolic source, linear" begin
        # `2 * fₕ` for a plain `VectorElement` `fₕ` is an *eager* numeric scaling (the
        # source side is eager, per the forms tutorial) and never builds an `OperatorScale`
        # at all -- there is nothing for this rule to lift there. A `SourceFunction` (as
        # `πₕ(uₕ)` also builds) is a genuine `LazyOp`, so scaling *that* does.
        sf = source_function(x -> x[1] + 1, Val(2))
        l = form(Wₕ, v -> innerₕ(2 * sf, v) + innerₕ(3 * sf, v))
        ast = resolve_form_ast(l)
        @test ast isa OperatorAdd
        @test ast.left_op isa OperatorScale && ast.left_op.scalar == 2
        @test ast.right_op isa OperatorScale && ast.right_op.scalar == 3
        @test ast.left_op.inner_op isa LinearProduct
        @test assemble(l) ≈ 5 .* assemble(form(Wₕ, v -> innerₕ(sf, v)))
    end

    # Lifting a DiracSource.
    @testset "lifting: DiracSource, linear" begin
        # `dirac(...)` (a `DiracSource`) is a genuine `LazyOp` source exactly like
        # `source_function` above, so it lifts the same way: each coefficient out of its own
        # inner product, the two terms left two.
        d = dirac((0.3, 0.4), 1.0)
        @test d isa DiracSource
        l = form(Wₕ, v -> innerₕ(2 * d, v) + innerₕ(3 * d, v))
        ast = resolve_form_ast(l)
        @test ast isa OperatorAdd
        @test ast.left_op isa OperatorScale
        @test ast.left_op.inner_op isa LinearProduct
        @test ast.right_op isa OperatorScale
        @test assemble(l) ≈ 5 .* assemble(form(Wₕ, v -> innerₕ(d, v)))

        # a zero-scaled DiracSource collapses like any other source
        l_zero = form(Wₕ, v -> innerₕ(0 * d, v))
        @test resolve_form_ast(l_zero) isa ZeroOperator
    end
end

# Nested grid-function scales are not fused into one precomputed array: each keeps reading
# its own coefficient at evaluation time, so an in-place change to either one reaches the next
# `assemble!`. The reference is a freshly built form, and the diagonal
# of the bilinear case is checked against the hand-built product.
@testset "simplifier: nested scales stay live" begin
    for Ωₕ in (mesh(domain(interval(0.0, 1.0)), 9, true),
        mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 6), (false, true)))
        Wₕ = gridspace(Ωₕ)
        uₕ = Rₕ(Wₕ, x -> x[1] + 1.0)
        wₕ = Rₕ(Wₕ, x -> x[end] + 2.0)
        H = Matrix(Diagonal(collect(weights(Wₕ, Innerh()))))

        a = form(Wₕ, Wₕ, (U, V) -> innerₕ(uₕ * (wₕ * U), V))
        ast = resolve_form_ast(a)
        @test ast isa BilinearProduct
        @test ast.left_op isa GridFunctionScale
        @test ast.left_op.inner_op isa GridFunctionScale

        linear = (() -> form(Wₕ, v -> innerₕ(1.0, uₕ * (uₕ * v))),
            () -> form(Wₕ, v -> innerₕ(1.0, uₕ * (wₕ * v))))
        ls = map(mk -> mk(), linear)
        bs = map(assemble, ls)
        A = assemble(a)
        @test Matrix(A) ≈ Diagonal(parent(uₕ) .* parent(wₕ)) * H

        parent(uₕ) .*= 3
        parent(wₕ) .+= 1
        for k in eachindex(ls)
            assemble!(bs[k], ls[k])
            @test bs[k] ≈ assemble(linear[k]())
        end
        assemble!(A, a)
        @test Matrix(A) ≈ Diagonal(parent(uₕ) .* parent(wₕ)) * H
    end
end

# A shift by a runtime amount, along a `Val` direction: `simplify_ast` keeps every shift as
# built, so the node type never depends on the amount and `form` infers concretely
# (gpena/Bramble.jl#530).
_rt_shift(k::Int, W) = form(W, W, (u, v) -> innerₕ(shift_op(u, Val(1), k), v))

@testset "Shift kept as built" begin
    Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 6), (true, true))
    Wₕ = gridspace(Ωₕ)
    A = IdentityOperator(Wₕ)

    # neither `Shift₀` nor nested shifts fold: the amount is a field, not a type parameter
    for op in (shift_op(A, 1, 0), shift_op(shift_op(A, 1, 2), 1, 3),
        shift_op(shift_op(A, 1, 2), 1, -2), shift_op(shift_op(A, 1, 2), 2, 3))
        @test simplify_ast(op) === op
    end

    u = Bramble.TrialFunction{2, 1}()
    for op in (S₊ₓ(u), S₊ₓ(S₊ₓ(u)), S₊ₓ(S₋ₓ(u)), shift_op(u, 1, 0))
        @test isconcretetype(only(Base.return_types(simplify_ast, (typeof(op),))))
    end
    @test isconcretetype(only(Base.return_types(_rt_shift, (Int, typeof(Wₕ)))))

    # values: a zero shift is the identity and nested shifts in one sense are their sum,
    # bitwise, on a non-uniform mesh, bilinear, under a difference, and over a source
    Wn = gridspace(mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (6, 7), (false, false)))
    sf = source_function(x -> 1 + x[1]^2 + x[2], Val(2))
    bil(f) = Matrix(assemble(form(Wn, Wn, (u, v) -> innerₕ(f(u), v))))
    lin(f) = assemble(form(Wn, v -> innerₕ(f(sf), v)))
    @test bil(u -> shift_op(u, 1, 0)) == bil(identity)
    @test Matrix(assemble(_rt_shift(0, Wn))) == bil(identity)
    @test Matrix(assemble(_rt_shift(2, Wn))) == bil(u -> S₊ₓ(S₊ₓ(u)))
    @test lin(f -> shift_op(f, 1, 0)) == lin(identity)
    for (nested, single) in (
        (u -> shift_op(shift_op(u, 1, 1), 1, 2), u -> shift_op(u, 1, 3)),
        (u -> shift_op(shift_op(u, 1, -2), 1, -1), u -> shift_op(u, 1, -3)),
        (u -> S₊ₓ(S₊ₓ(u)), u -> shift_op(u, 1, 2))
    )
        @test bil(nested) == bil(single)
        @test bil(D₋ₓ ∘ nested) == bil(D₋ₓ ∘ single)
        @test bil(D₋ᵧ ∘ nested) == bil(D₋ᵧ ∘ single)
        @test lin(nested) == lin(single)
    end
    @test bil(u -> S₊ₓ(S₊ₓ(u))) != bil(u -> S₊ₓ(u))

    @testset "combined shift: numeric agreement" begin
        Ωₕ1 = mesh(domain(interval(0.0, 1.0)), 8, true)
        Wₕ1 = gridspace(Ωₕ1)
        sf = source_function(x -> x^2 + 1, Val(1))

        b_nested = assemble(form(Wₕ1, v -> innerₕ(shift_op(shift_op(sf, 1, 1), 1, 2), v)))
        b_combined = assemble(form(Wₕ1, v -> innerₕ(shift_op(sf, 1, 3), v)))
        @test b_nested == b_combined
    end

    # the bilinear form of a composed shift against the grid-function computation, on a
    # non-uniform mesh: same-sign shifts compose to their sum, opposite-sign ones keep the
    # boundary zero of the inner read
    @testset "composed shifts: grid function" begin
        W = gridspace(mesh(domain(interval(0.0, 1.0)), 7, false))
        H = Diagonal(collect(weights(W, Innerh())))
        uh = Rₕ(W, x -> 1 + x^2)
        Sp = Matrix(forward_shift(W, Val(1)))
        Sm = Matrix(backward_shift(W, Val(1)))
        for f in (S₊ₓ ∘ S₋ₓ, S₋ₓ ∘ S₊ₓ, S₊ₓ ∘ S₊ₓ, S₋ₓ ∘ S₋ₓ)
            A = assemble(form(W, W, (u, v) -> innerₕ(f(u), v)))
            @test A * parent(uh) ≈ H * parent(f(uh))
        end
        # `shift_op`'s amounts beyond one and zero, against the matrix powers
        for (f, M) in (
            (u -> shift_op(shift_op(u, 1, 2), 1, -2), Sm^2 * Sp^2),
            (u -> shift_op(shift_op(u, 1, 1), 1, 2), Sp^3),
            (u -> shift_op(u, 1, 0), Sp^0)
        )
            @test Matrix(assemble(form(W, W, (u, v) -> innerₕ(f(u), v)))) ≈ H * M
        end
    end
end

# Component distribution on a mixed sum inside one inner product.
@testset "mixed sum: component distribution" begin
    Ωₕ = mesh(domain(interval(0.0, 1.0)), 11, true)
    Wₕ = gridspace(Ωₕ)
    Vₕ = Wₕ^Val(2)
    fₕ = Rₕ(Wₕ, x -> sin(π * x[1]))

    @testset "linear: source to two test equations" begin
        # `innerₕ(fₕ, v(1) + v(2))` names test components 1 and 2 inside one product --
        # unroutable as a single term (`test_component_or_nothing` throws on it,
        # block_extract.jl) before this rule.
        l = form(Vₕ, v -> innerₕ(fₕ, v(1) + v(2)))
        @test resolve_form_ast(l) isa OperatorAdd

        l_ref = form(Vₕ, v -> innerₕ(fₕ, v(1)) + innerₕ(fₕ, v(2)))
        @test assemble(l) ≈ assemble(l_ref)
    end

    # One trial component coupling to two test equations.
    @testset "bilinear: one trial to two tests" begin
        a = form(Vₕ, Vₕ, (u, v) -> innerₕ(u(1), v(1) + v(2)))
        @test resolve_form_ast(a) isa OperatorAdd

        a_ref = form(Vₕ, Vₕ, (u, v) -> innerₕ(u(1), v(1)) + innerₕ(u(1), v(2)))
        @test Matrix(assemble(a)) ≈ Matrix(assemble(a_ref))
    end

    @testset "scaled mixed sum: scale distributes" begin
        # `2 * innerₕ(fₕ, v(1) + v(2))` must not simplify to an `OperatorScale` hiding the
        # distributed `OperatorAdd` from the router -- that would be exactly the
        # unroutable shape this rule exists to avoid, reached through a new path.
        l = form(Vₕ, v -> 2 * innerₕ(fₕ, v(1) + v(2)))
        ast = resolve_form_ast(l)
        @test ast isa OperatorAdd
        @test ast.left_op isa OperatorScale
        @test ast.right_op isa OperatorScale

        l_ref = form(Vₕ, v -> 2 * innerₕ(fₕ, v(1)) + 2 * innerₕ(fₕ, v(2)))
        @test assemble(l) ≈ assemble(l_ref)
    end

    # Distributing it would be a sweep-count regression.
    @testset "same-component sum: not distributed" begin
        # `v(1) + D₋ₓ(v(1))` names the same component on both sides, so it already routes
        # as one term; distributing it anyway would trade that single sweep for two.
        l = form(Vₕ, v -> innerₕ(fₕ, v(1) + D₋ₓ(v(1))))
        @test !(resolve_form_ast(l) isa OperatorAdd)
    end
end

@testset "3+ mixing components: no throw (#235)" begin
    # `_mixes_components` used to reuse `trial_component_or_nothing`/
    # `test_component_or_nothing` (block_extract.jl) directly, which *throw* the moment
    # either side already mixes components -- true of the inner `(A + B)` node on every
    # left-associated three-or-more-term sum, `2.0 * (A + B + C)` parsing as
    # `2.0 * ((A + B) + C)`. Every check below is against a hand-distributed reference
    # (never against another call to the code under test), following bramble-verification.
    Ωₕ = mesh(domain(interval(0.0, 1.0)), 11, true)
    Wₕ = gridspace(Ωₕ)
    V3 = Wₕ^Val(3)
    V4 = Wₕ^Val(4)
    fₕ = Rₕ(Wₕ, x -> sin(π * x[1]))
    cₕ = Rₕ(Wₕ, x -> 1.0 + x[1])

    # Scalar times a sum of 3 and 4 mixing test components.
    @testset "linear: scalar × 3- and 4-term sums" begin
        for (V, n) in ((V3, 3), (V4, 4))
            l = form(V, v -> 2.0 * sum(innerₕ(fₕ, v(i)) for i in 1:n))
            @test resolve_form_ast(l) isa OperatorAdd

            l_dist = form(V, v -> sum(2.0 * innerₕ(fₕ, v(i)) for i in 1:n))
            @test assemble(l) ≈ assemble(l_dist)
        end
    end

    # A grid-function coefficient distributes the same way.
    @testset "linear: grid-function coefficient" begin
        for (V, n) in ((V3, 3), (V4, 4))
            l = form(V, v -> cₕ * sum(innerₕ(fₕ, v(i)) for i in 1:n))
            @test resolve_form_ast(l) isa OperatorAdd

            l_dist = form(V, v -> sum(cₕ * innerₕ(fₕ, v(i)) for i in 1:n))
            @test assemble(l) ≈ assemble(l_dist)
        end
    end

    # One trial component against a sum of 3 and 4 test components.
    @testset "bilinear: one trial, 3- and 4-term sums" begin
        for (V, n) in ((V3, 3), (V4, 4))
            a = form(V, V, (u, v) -> innerₕ(u(1), sum(v(i) for i in 1:n)))
            @test resolve_form_ast(a) isa OperatorAdd

            a_dist = form(V, V, (u, v) -> sum(innerₕ(u(1), v(i)) for i in 1:n))
            @test Matrix(assemble(a)) ≈ Matrix(assemble(a_dist))
        end
    end

    @testset "Association does not change the result" begin
        # Left-associated (`+`'s own default) against an explicitly right-associated tree
        # built the same way `foldr` would.
        l_left = form(V4, v -> innerₕ(fₕ, v(1)) + innerₕ(fₕ, v(2)) + innerₕ(fₕ, v(3)) + innerₕ(fₕ, v(4)))
        l_right = form(
            V4,
            v -> innerₕ(fₕ, v(1)) +
                 (innerₕ(fₕ, v(2)) + (innerₕ(fₕ, v(3)) + innerₕ(fₕ, v(4))))
        )
        @test assemble(l_left) ≈ assemble(l_right)
    end

    # innerₕ(divₕ(u), divₕ(v)): a three-term mixing sum on both sides.
    # 3D only: runs nightly (`slow`).
    WITH_3D_TESTS && @testset "3D innerₕ(divₕ, divₕ): 3-term sums" begin
        # The motivating case: the 3D discrete divergence inner product could not be
        # written at all before this fix (its 2D counterpart, a two-term sum, already
        # worked). Checked against the nine written-out (i, j) single-component products.
        Ω3 = mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (4, 4, 4), (true, true, true))
        W3 = gridspace(Ω3)
        V3d = W3 × W3 × W3
        ops = (D₋ₓ, D₋ᵧ, D₋₂)

        a_div = form(
            V3d, V3d,
            (p, q) -> innerₕ(
                D₋ₓ(p(1)) + D₋ᵧ(p(2)) + D₋₂(p(3)), D₋ₓ(q(1)) + D₋ᵧ(q(2)) + D₋₂(q(3))
            )
        )
        A_div = Matrix(assemble(a_div))

        # `unit` checks one diagonal and one off-diagonal (i, j) product against the matching
        # entries of the sum (the nine products have disjoint supports); `slow` sums all nine.
        A_ij(i, j) = Matrix(assemble(form(V3d, V3d,
            (p, q) -> innerₕ(ops[i](p(i)), ops[j](q(j))))))
        if WITH_SLOW_TESTS
            A_nine = zeros(size(A_div))
            for i in 1:3, j in 1:3
                A_nine .+= A_ij(i, j)
            end
            @test A_div ≈ A_nine
        else
            for (i, j) in ((1, 1), (1, 2))
                B = A_ij(i, j)
                mask = B .!= 0
                @test any(mask)
                @test B[mask] ≈ A_div[mask]
            end
        end
    end
end

# --- Linear wrappers over a component-mixing sum (gpena/Bramble.jl#759) ------------- #
#
# A difference, average, jump, shift or restriction is linear in its operand, so one over a
# sum naming two components distributes over it, as a scale does. Each form is checked
# against the sum of its parts, each part assembled as its own form: the parts land in
# different blocks, so a term routed to the wrong block fails the comparison.

_rt_comp_wrapped(i::Int, V) = form(V, V, (u, v) -> innerₕ(u(i), D₋ₓ(v(i))))
_lit_wrapped_mixed(V) = form(V, V, (u, v) -> innerₕ(u(1), D₋ₓ(v(1) + v(2))))
_lit_mixed_pair(V) = form(V, V, (u, v) -> innerₕ(u(1), (v(1) + v(2)) + (v(2) + v(1))))

@testset "wrapper over mixed sum (#759)" begin
    I = interval(0.0, 1.0)
    Ωd = domain(I × I, :l => :left, :r => :right, :b => :bottom, :t => :top)
    Wₕ = gridspace(mesh(Ωd, (7, 9), (false, false)))   # non-uniform in both directions
    Vₕ = Wₕ × Wₕ
    R = Bramble.restrict_to
    M(f) = Matrix(assemble(form(Vₕ, Vₕ, f)))

    cases = (
        "D₋ₓ, test side" => ((u, v) -> innerₕ(u(1), D₋ₓ(v(1) + v(2))),
            (u, v) -> innerₕ(u(1), D₋ₓ(v(1))), (u, v) -> innerₕ(u(1), D₋ₓ(v(2)))),
        "Mₓ, test side" => ((u, v) -> innerₕ(u(1), Mₓ(v(1) + v(2))),
            (u, v) -> innerₕ(u(1), Mₓ(v(1))), (u, v) -> innerₕ(u(1), Mₓ(v(2)))),
        "D₋ₓ, trial side" => ((u, v) -> innerₕ(D₋ₓ(u(1) + u(2)), v(1)),
            (u, v) -> innerₕ(D₋ₓ(u(1)), v(1)), (u, v) -> innerₕ(D₋ₓ(u(2)), v(1))),
        "jump" => ((u, v) -> innerₕ(u(2), jumpₓ(v(1) + 3.0 * v(2))),
            (u, v) -> innerₕ(u(2), jumpₓ(v(1))), (u, v) -> innerₕ(u(2), jumpₓ(3.0 * v(2)))),
        "shift" => ((u, v) -> innerₕ(u(1), S₊ₓ(v(1) + v(2))),
            (u, v) -> innerₕ(u(1), S₊ₓ(v(1))), (u, v) -> innerₕ(u(1), S₊ₓ(v(2)))),
        "restriction" => ((u, v) -> innerₕ(u(1), R(:l, v(1) + v(2))),
            (u, v) -> innerₕ(u(1), R(:l, v(1))), (u, v) -> innerₕ(u(1), R(:l, v(2)))),
        "D₋ₓ over restrictions" => (
            (u, v) -> innerₕ(u(1), D₋ₓ(R(:l, v(1)) + R(:r, v(2)))),
            (u, v) -> innerₕ(u(1), D₋ₓ(R(:l, v(1)))),
            (u, v) -> innerₕ(u(1), D₋ₓ(R(:r, v(2))))),
        "D₋ₓ ∘ D₊ᵧ, scaled summand" => (
            (u, v) -> innerₕ(u(1), D₋ₓ(D₊ᵧ(v(1) + 2.0 * v(2)))),
            (u, v) -> innerₕ(u(1), D₋ₓ(D₊ᵧ(v(1)))),
            (u, v) -> innerₕ(u(1), D₋ₓ(D₊ᵧ(2.0 * v(2))))),
        "D₋ₓ ∘ Mₓ" => ((u, v) -> innerₕ(u(1), D₋ₓ(Mₓ(v(1) + v(2)))),
            (u, v) -> innerₕ(u(1), D₋ₓ(Mₓ(v(1)))), (u, v) -> innerₕ(u(1), D₋ₓ(Mₓ(v(2))))),
        "R ∘ D₋ₓ" => ((u, v) -> innerₕ(u(1), R(:r, D₋ₓ(v(1) + v(2)))),
            (u, v) -> innerₕ(u(1), R(:r, D₋ₓ(v(1)))), (u, v) -> innerₕ(u(1), R(:r, D₋ₓ(v(2))))),
        "scale inside D₋ₓ" => ((u, v) -> innerₕ(u(1), D₋ₓ(2.0 * (v(1) + v(2)))),
            (u, v) -> innerₕ(u(1), D₋ₓ(2.0 * v(1))), (u, v) -> innerₕ(u(1), D₋ₓ(2.0 * v(2)))),
        "both sides" => ((u, v) -> innerₕ(D₋ₓ(u(1) + u(2)), Mₓ(v(1) + v(2))),
            (u, v) -> innerₕ(D₋ₓ(u(1)), Mₓ(v(1) + v(2))),
            (u, v) -> innerₕ(D₋ₓ(u(2)), Mₓ(v(1) + v(2))))
    )
    @testset "$name" for (name, (whole, p1, p2)) in cases
        a = form(Vₕ, Vₕ, whole)
        @test resolve_form_ast(a) isa OperatorAdd
        P1, P2 = M(p1), M(p2)
        @test any(!iszero, P1) && any(!iszero, P2)
        @test Matrix(assemble(a)) ≈ P1 + P2
    end

    @testset "two mixing summands, bare and under D₋ₓ" begin
        # Both halves already mix, so their classes are the same `Mixed()`: compared alone,
        # they read as agreeing and the sum was left whole.
        @test M((u, v) -> innerₕ(u(1), (v(1) + v(2)) + (v(2) + v(1)))) ≈
              2 .* (M((u, v) -> innerₕ(u(1), v(1))) + M((u, v) -> innerₕ(u(1), v(2))))
        @test M((u, v) -> innerₕ(u(1), D₋ₓ((v(1) + v(2)) + (v(2) + 2.0 * v(1))))) ≈
              M((u, v) -> innerₕ(u(1), D₋ₓ(3.0 * v(1)))) +
              M((u, v) -> innerₕ(u(1), D₋ₓ(2.0 * v(2))))
    end

    @testset "linear form" begin
        L(f) = assemble(form(Vₕ, f))
        whole = L(v -> innerₕ(1.0, D₋ₓ(v(1) + 100.0 * v(2))))
        p1, p2 = L(v -> innerₕ(1.0, D₋ₓ(v(1)))), L(v -> innerₕ(1.0, D₋ₓ(100.0 * v(2))))
        @test any(!iszero, p1) && any(!iszero, p2)
        @test whole ≈ p1 + p2
    end

    @testset "one component or none: returned as built" begin
        # The rewrite reads the operand's component class, a type when it holds no
        # mixing sum, so these come back as the very object given and compile as before.
        u, v = trial_function(Vₕ), test_function(Vₕ)
        uₛ, vₛ = trial_function(Wₕ), test_function(Wₕ)
        for raw in (innerₕ(u(1), D₋ₓ(v(1) + 2.0 * D₊ᵧ(v(1)))),
            innerₕ(u(1), R(:l, D₋ₓ(v(1) + v(1)))),
            innerₕ(D₋ₓ(u(2) + u(2)), Mₓ(v(1) + D₋ₓ(v(1)))),
            innerₕ(Mₓ(uₛ + D₋ₓ(uₛ)), jumpₓ(vₛ + 2 * vₛ)),
            innerₕ(R(:l, uₛ + uₛ), D₋ₓ(R(:r, vₛ) + vₛ)))
            ast = resolve_ast(raw)
            @test simplify_ast(ast) === ast
        end
    end

    @testset "inference" begin
        V = typeof(Vₕ)
        concrete(f, sig) = isconcretetype(only(Base.return_types(f, sig)))
        @test concrete(_rt_comp_wrapped, (Int, V))
        @test concrete(_lit_wrapped_mixed, (V,))
        @test concrete(_lit_mixed_pair, (V,))
    end

    # The routing query is still asked of an undistributed operand, and still refuses it.
    @test_throws ArgumentError Bramble.test_component_or_nothing(
        D₋ₓ(test_function(Vₕ)(1) + test_function(Vₕ)(2)))
end

# A shift is a stencil node like a difference (src/ast/stencil_nodes.jl, #767), so the
# restricted-product split distributes it over a restricted sum: one summand per
# restriction, each a shift of its own, where it was one summand before.
@testset "shift over a restricted sum: split" begin
    I = interval(0.0, 1.0)
    Ωd = domain(I × I, :l => :left, :r => :right, :b => :bottom, :t => :top)
    Wₕ = gridspace(mesh(Ωd, (7, 9), (false, false)))   # non-uniform in both directions
    R = Bramble.restrict_to
    v = test_function(Wₕ)

    for paired in (Val(false), Val(true)),
        op in (S₊ₓ(R(:b, v) + R(:t, v)), S₊ₓ(D₋ᵧ(R(:b, v) + R(:t, v))))

        parts = Bramble._summands_of(op, paired)
        @test length(parts) == 2
        @test all(p -> p isa ShiftNode, parts)
    end
    @test length(Bramble._summands_of(S₊ₓ(v + v), Val(false))) == 1

    # The split changes no matrix: the whole form is the sum of its pieces, each nonzero.
    M(f) = Matrix(assemble(form(Wₕ, Wₕ, f)))
    P1 = M((u, v) -> innerₕ(u, S₊ₓ(R(:b, v))))
    P2 = M((u, v) -> innerₕ(u, S₊ₓ(R(:t, v))))
    @test any(!iszero, P1) && any(!iszero, P2)
    @test M((u, v) -> innerₕ(u, S₊ₓ(R(:b, v) + R(:t, v)))) ≈ P1 + P2
end

# --- Type stability under a runtime coefficient ------------------------------------- #
#
# Every rewrite in `simplify_ast` decides what *node type* to return, so a rule that reads a
# coefficient's value makes `form`'s return type depend on that value: inferred as a `Union`
# of the rewritten and unrewritten trees whenever the compiler cannot fold the comparison.
# That costs every caller a dynamic dispatch into the assembly engine, and it is what Enzyme
# rejects with `IllegalTypeAnalysisException` when differentiating with respect to an
# operator's own coefficient. The value-reading rules are therefore
# restricted to `Integer` coefficients, and these are the checks that pin it: `isconcretetype`
# on the inferred return type, which is exactly the property the gate guarantees.
#
# `Float64` arguments rather than literals on purpose -- a literal coefficient is constant
# -folded by inference and comes out concrete either way, so a literal proves nothing here.
_rt_single(θ::Float64, W) = form(W, W, (u, v) -> θ * innerₕ(u, v))
_rt_sum(θ::Float64, W) = form(W, W, (u, v) -> θ * inner₊ₓ(D₋ₓ(u), D₋ₓ(v)) + (1 - θ) * innerₕ(u, v))
_rt_bare(θ::Float64, W) = form(W, W, (u, v) -> θ * inner₊ₓ(D₋ₓ(u), D₋ₓ(v)) + innerₕ(u, v))
_rt_equal(θ::Float64, W) = form(W, W, (u, v) -> θ * inner₊ₓ(D₋ₓ(u), D₋ₓ(v)) + θ * innerₕ(u, v))
_rt_lifted(θ::Float64, W) = form(W, W, (u, v) -> innerₕ(θ * u, v))
_rt_linear(θ::Float64, W, fₕ) = form(W, v -> θ * innerₕ(fₕ, v))

# Two terms of the *same shape* carrying runtime data. A rule deciding whether the two
# subtrees are equal by comparing them field by field at run time made `form` infer a
# `Union` of the merged and unmerged trees for any such sum. Two distinct diffusion
# coefficients in one bilinear form, or two source vectors in one linear form, are the
# ordinary way to write a model: these pin that both infer concretely.
_rt_two_coeffs(W, g₁, g₂) = form(W, W, (u, v) -> innerₕ(g₁ * u, v) + innerₕ(g₂ * u, v))
_rt_two_sources(W, g₁, g₂) = form(W, v -> innerₕ(g₁, v) + innerₕ(g₂, v))

# A runtime `Integer` coefficient, which is the one shape deliberately left unstable. The
# `0 * A -> 0` and `1 * A -> A` collapses read the coefficient's *value*, and that is free
# only when the value is a literal inference constant-folds. `n::Int` taken from an argument
# is not, so `_wrap_scale`/`simplify_ast(::OperatorScale)` return
# `Union{ZeroOperator, typeof(A), OperatorScale}` and `form`'s own return type follows.
# Removing the value branch would cost `0 * A` and `1 * A` their collapse -- both documented,
# and `1` is what `_scale_parts` reports for every bare node -- so the trade-off stands and the
# caller's fix is `float(n)` or a `Ref`. Pinned here, asserted *false*, so that changing the
# trade-off has to be a deliberate edit to this test rather than a silent drift either way.
_rt_runtime_int(n::Int, W) = form(W, W, (u, v) -> n * innerₕ(u, v))

# A runtime component index (gpena/Bramble.jl#529). Two `2 *` terms on component `i` stay
# two terms and `form` infers concretely. The distribution gates read `component_idx`
# values: whether `v(i) + v(1)` mixes components is known only at run time, so that form's
# type is a `Union` of the distributed and undistributed trees -- kept on purpose, see
# `_mixes_components`.
_rt_comp_factor(i::Int, V) = form(V, V, (u, v) -> 2 * innerₕ(u(i), v(i)) + 2 * innerₕ(D₋ₓ(u(i)), v(i)))
_rt_comp_mixed(i::Int, V) = form(V, V, (u, v) -> innerₕ(u(i), v(i) + v(1)))

# Sums of `Ref`-scaled terms (gpena/Bramble.jl#527). A rule firing when both sides held the
# *same* `Ref` object made `form` infer a `Union`: `cl === cr` on two `RefValue{Float64}`
# field loads is a run-time value. `_rt_ref_two` scales one product twice, `_rt_ref_shared`
# two different products, `_rt_ref_linear` two sources of a linear form.
_rt_ref_two(β, γ, W) = form(W, W, (u, v) -> β * innerₕ(u, v) + γ * innerₕ(u, v))
_rt_ref_shared(β, W) = form(W, W, (u, v) -> β * inner₊ₓ(D₋ₓ(u), D₋ₓ(v)) + β * innerₕ(u, v))
_rt_ref_linear(β, W, f, g) = form(W, v -> β * innerₕ(f, v) + β * innerₕ(g, v))

_infers(f, sig) = isconcretetype(only(Base.return_types(f, sig)))

@testset "runtime coefficient: form type-stable" begin
    Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 6), (true, true))
    Wₕ = gridspace(Ωₕ)
    W = typeof(Wₕ)
    fₕ = Rₕ(Wₕ, x -> x[1])

    @test _infers(_rt_single, (Float64, W))
    @test _infers(_rt_sum, (Float64, W))
    @test _infers(_rt_bare, (Float64, W))
    @test _infers(_rt_equal, (Float64, W))
    @test _infers(_rt_lifted, (Float64, W))
    @test _infers(_rt_linear, (Float64, W, typeof(fₕ)))

    # Two same-shaped terms carrying distinct runtime data, bilinear and linear.
    g₁ = Rₕ(Wₕ, x -> 1.0 + x[1])
    g₂ = Rₕ(Wₕ, x -> 2.0 - x[2])
    @test _infers(_rt_two_coeffs, (W, typeof(g₁), typeof(g₂)))
    @test _infers(_rt_two_sources, (W, typeof(g₁), typeof(g₂)))

    # `Ref` coefficients, shared or not: see `_rt_ref_two`'s comment.
    R = Base.RefValue{Float64}
    @test _infers(_rt_ref_two, (R, R, W))
    @test _infers(_rt_ref_shared, (R, W))
    @test _infers(_rt_ref_linear, (R, W, typeof(g₁), typeof(g₂)))

    # ... and the deliberate hole, asserted as such: see `_rt_runtime_int`'s comment.
    @test !_infers(_rt_runtime_int, (Int, W))

    # Runtime component index: see `_rt_comp_factor`'s comment. The hole is asserted as a
    # two-member `Union`, so a form that infers `Union{}` (a throwing path) cannot pass it.
    Ω₁ = mesh(domain(interval(0.0, 1.0)), 11, true)
    V₂ = gridspace(Ω₁)^Val(2)
    @test _infers(_rt_comp_factor, (Int, typeof(V₂)))
    T_mixed = only(Base.return_types(_rt_comp_mixed, (Int, typeof(V₂))))
    @test T_mixed isa Union && length(Base.uniontypes(T_mixed)) == 2
    # Same-component terms are no longer factored, but assemble to what was written.
    M₂(f) = Matrix(assemble(form(V₂, V₂, f)))
    @test Matrix(assemble(_rt_comp_factor(2, V₂))) ≈
          2 .* M₂((u, v) -> innerₕ(u(2), v(2))) .+ 2 .* M₂((u, v) -> innerₕ(D₋ₓ(u(2)), v(2)))
    @test Matrix(assemble(_rt_comp_mixed(2, V₂))) ≈
          M₂((u, v) -> innerₕ(u(2), v(2))) .+ M₂((u, v) -> innerₕ(u(2), v(1)))

    # A `Float64` coefficient is carried through as written. Assembling it must give the
    # same matrix: this pass may change how a form is routed, never what it computes.
    @test Matrix(assemble(form(Wₕ, Wₕ, (u, v) -> 2.0 * innerₕ(u, v) + 2.0 * inner₊ₓ(D₋ₓ(u), D₋ₓ(v))))) ≈
          2.0 .* Matrix(assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊ₓ(D₋ₓ(u), D₋ₓ(v)))))
end

# The sign `-u` and `a - b` store (`_MinusOne`, src/ast/ast.jl) is decided by its type, so
# `simplify_ast` folds it whatever the operator's values, and it acts as the `Int` `-1` it
# replaced wherever a coefficient meets a number: float constructors, `convert`,
# promotion, a `ForwardDiff.Dual` beside it, and the two forms below differentiated through.
_rt_negated(W) = form(W, W, (u, v) -> innerₕ(u, v) - innerₕ(D₋ₓ(u), v))

@testset "the sign of minus acts as -1" begin
    Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 6), (false, false))
    Wₕ = gridspace(Ωₕ)
    u = trial_function(Wₕ)
    M = (-u).scalar
    @test M isa Bramble._MinusOne
    @test (u - 2.0 * u).right_op.scalar isa Bramble._MinusOne
    @test isconcretetype(only(Base.return_types(simplify_ast, (typeof(-u),))))
    @test isconcretetype(only(Base.return_types(simplify_ast, (typeof(u - 2.0 * u),))))
    @test _infers(_rt_negated, (typeof(Wₕ),))
    @test simplify_ast(-(-u)) === u
    @test string(M) == "-1" && M < 0
    @test Bramble.expression(-u) == "-u"
    @test startswith(Bramble.expression(u - D₋ₓ(u)), "u - ")
    @test (M + M, M - M, M * M) === (-2, 0, 1)
    @test hash(M) == hash(-1) && M == -1 && -1 == M && M != 2 && 2 != M
    @test !Bramble._same_operator_shape(-D₋ₓ(u), 2 * D₋ₓ(test_function(Wₕ)))

    # No `Int` constructor (it invalidates Base on load), so the fixed-width integers and
    # `Rational` only promote.
    Dual = ForwardDiff.Dual{Nothing, Float64, 1}
    for T in (Float32, Float64, BigFloat, ComplexF64, Dual)
        @test T(M) isa T && T(M) == -1
        @test convert(T, M) isa T && convert(T, M) == -1
    end
    for T in (Float32, Float64, BigFloat, Int, Int32, Rational{Int}, ComplexF64, Dual)
        @test promote_type(typeof(M), T) === promote_type(Int, T)
    end
    # A `BigInt` divisor scales by a `BigFloat`, which the sign meets.
    @test Matrix(assemble(form(Wₕ, Wₕ, (u, v) -> -(innerₕ(u, v) / big(2))))) ==
          Matrix(assemble(form(Wₕ, Wₕ, (u, v) -> -0.5 * innerₕ(u, v))))
    @test M * ForwardDiff.Dual{Nothing}(2.0, 1.0) == ForwardDiff.Dual{Nothing}(-2.0, -1.0)

    # d/ds of `s A - A` is `A`, and of `(s - 1) b` is `b`: their sums are the mass's.
    A₁ = assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v)))
    b₁ = assemble(form(Wₕ, v -> innerₕ(1.0, v)))
    dA = ForwardDiff.derivative(
        s -> sum(assemble(form(Wₕ, Wₕ, (u, v) -> s * innerₕ(u, v) - innerₕ(u, v)))), 1.0)
    db = ForwardDiff.derivative(
        s -> sum(assemble(form(Wₕ, v -> innerₕ(s, v) - innerₕ(1.0, v)))), 1.0)
    @test dA ≈ sum(A₁)
    @test db ≈ sum(b₁)
end

# The `same_mesh` rule of gpena/Bramble.jl#771 (`_same_mesh_stencil`,
# operators/interpolation.jl) acts in the stencil, after this pass: `πₕ(u)` stays in the
# stored tree, which is the same tree, of the same type, whether or not the two spaces
# share a mesh. Its matrices are tested in test/form/interpolation_operator.jl.
_rt_interp(W1, W2) = form(W1, W2, (u, v) -> innerₕ(πₕ(u), v) + inner₊ₓ(D₋ₓ(πₕ(u)), D₋ₓ(v)))

@testset "πₕ forms: the stored tree" begin
    seed!(771)
    Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
    W, W2 = gridspace(mesh(Ω, (9, 8), (false, false))), gridspace(mesh(Ω, (7, 6), (false, false)))
    a = @inferred _rt_interp(W, W)
    @test simplify_ast(a.ast) === a.ast
    @test _infers(_rt_interp, (typeof(W), typeof(W)))
    @test typeof(a.ast) === typeof(_rt_interp(W, W2).ast) === typeof(_rt_interp(W2, W).ast)
    @test a.ast.left_op.left_op isa InterpolationNode
    @test a.ast.right_op.left_op.inner_op isa InterpolationNode
end

end # module FormSimplifierTests
