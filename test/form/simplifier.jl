module FormSimplifierTests

using Test
using Bramble
using LinearAlgebra: Diagonal, issymmetric, isposdef
using SparseArrays: nnz
using Bramble:
               IdentityOperator,
               ZeroOperator,
               OperatorAdd,
               OperatorScale,
               GridFunctionScale,
               BilinearProduct,
               LinearProduct,
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
               Innerplus

# `simplify_ast` rewrites only the algebraic layer (`OperatorAdd`, `OperatorScale`,
# `GridFunctionScale`) that `ast.jl`'s `+`/`*`/`/` overloads build, into a tree that routes
# to fewer mesh sweeps (gpena/Bramble.jl#159, rules 1-3). Every check here is against either
# a hand-built reference matrix or a form assembled through a completely different call --
# never against another call to the code under test.

@testset "AST algebraic simplification" begin
    Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 6), (true, true))
    Wₕ = gridspace(Ωₕ)
    A = IdentityOperator(Wₕ)   # a leaf standing in for an arbitrary operator
    B = D₋ₓ(A)                 # a second leaf, structurally different from `A`

    # Like-term combining fires only when the two subtrees' equality is settled by their types
    # alone -- `_statically_equal`, i.e. `Base.issingletontype` (gpena/Bramble.jl#240). `A` and
    # `B` above are *not* singletons: `IdentityOperator` carries the grid space in a field, so
    # two of them are only equal by a run-time comparison. `S` and `T` are the singleton
    # counterparts -- an inner product of a trial and a test function, which is the shape every
    # form built from operators alone actually has -- and they are what the combining rules are
    # checked against below.
    S = innerₕ(trial_function(Wₕ), test_function(Wₕ))
    T = inner₊ₓ(D₋ₓ(trial_function(Wₕ)), D₋ₓ(test_function(Wₕ)))

    @testset "Zero and identity" begin
        @test simplify_ast(0 * A) isa ZeroOperator
        @test simplify_ast(1 * A) === A

        # A floating-point coefficient is deliberately left alone, however it compares to
        # `0` or `1` (gpena/Bramble.jl#240): collapsing on the value would make the node
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

    @testset "Combining like terms" begin
        # `S + S -> 2 * S`
        c = simplify_ast(S + S)
        @test c isa OperatorScale
        @test c.scalar == 2
        @test c.inner_op === S

        # `c1 * S + c2 * S -> (c1 + c2) * S`
        c = simplify_ast(2 * S + 3 * S)
        @test c isa OperatorScale
        @test c.scalar == 5
        @test c.inner_op === S

        # `S - S -> 0`
        @test simplify_ast(S - S) isa ZeroOperator

        # structurally different subtrees are never merged, however they compare numerically
        @test !(simplify_ast(S + T) isa OperatorScale)
        @test !(simplify_ast(A + B) isa OperatorScale)
    end

    @testset "Data-carrying terms are never combined" begin
        # The gate, stated as tests: equality decided by reading a field at run time would make
        # this method's return type depend on that read, and `form`'s return type a `Union` of
        # the combined and uncombined trees -- which is what Enzyme rejects
        # (gpena/Bramble.jl#240). So a node holding data is left as the sum it was written as,
        # *even when both sides hold the identical object*. One extra routed term, same numbers.
        vₕ = Rₕ(Wₕ, x -> x[1])

        same_array = simplify_ast(2 * (vₕ * B) + 3 * (vₕ * B))
        @test same_array isa OperatorAdd
        @test same_array.left_op isa OperatorScale
        @test same_array.right_op isa OperatorScale

        # `IdentityOperator` carries its grid space, so it is in the same boat as the array
        @test simplify_ast(A + A) isa OperatorAdd
        @test !(simplify_ast(A - A) isa ZeroOperator)

        # ... and the numbers are unchanged by not combining, which is the only thing that
        # may never move. Checked through `form`, against the single-term form scaled by hand.
        five = form(Wₕ, Wₕ, (u, v) -> 2 * innerₕ(vₕ * u, v) + 3 * innerₕ(vₕ * u, v))
        one_term = form(Wₕ, Wₕ, (u, v) -> innerₕ(vₕ * u, v))
        @test resolve_form_ast(five) isa OperatorAdd
        @test Matrix(assemble(five)) ≈ 5 .* Matrix(assemble(one_term))

        # two independently-built grid functions were never merged even before the gate --
        # only object identity ever counted -- but the common `2` still factors out
        wₕ = Rₕ(Wₕ, x -> x[1])  # same values, different array
        different = simplify_ast(2 * (vₕ * B) + 2 * (wₕ * B))
        @test different isa OperatorScale
        @test different.inner_op isa OperatorAdd
        @test different.inner_op.left_op isa GridFunctionScale
        @test different.inner_op.right_op isa GridFunctionScale
    end

    @testset "Distributive factoring" begin
        # `c * A + c * B -> c * (A + B)`, `A` and `B` structurally different
        f = simplify_ast(2 * A + 2 * B)
        @test f isa OperatorScale
        @test f.scalar == 2
        @test f.inner_op isa OperatorAdd
        @test f.inner_op.left_op === A
        @test f.inner_op.right_op === B

        # a shared dynamic coefficient factors the same way
        β = Ref(1.5)
        f = simplify_ast(β * A + β * B)
        @test f isa OperatorScale
        @test f.scalar === β
        @test f.inner_op isa OperatorAdd

        # different coefficients never factor
        g = simplify_ast(2 * A + 3 * B)
        @test g isa OperatorAdd
    end

    @testset "Idempotent and recursive" begin
        vₕ_deep = Rₕ(Wₕ, x -> x[1])

        # simplifying an already-simplified tree is a no-op
        t = simplify_ast(2 * A + 2 * B)
        @test simplify_ast(t) === t

        # the rules apply however deep the algebra sits, not only at the root. `S`, not `A`:
        # the innermost step is a like-term combine, which needs a singleton leaf.
        nested = simplify_ast(S + (0 * T + (1 * S)))
        @test nested isa OperatorScale
        @test nested.scalar == 2
        @test nested.inner_op === S

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

    @testset "Zero-scaled term elides from the sparsity pattern" begin
        a_ref = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v))
        a_zero = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + 0 * inner₊ₓ(D₋ₓ(u), D₋ₓ(v)))

        A_ref = assemble(a_ref)
        A_zero = assemble(a_zero)
        @test Matrix(A_zero) ≈ Matrix(A_ref)
        # the zero-scaled `inner₊ₓ(D₋ₓ(u), D₋ₓ(v))` term contributes nothing to the
        # pattern: without elision it would add its own (much wider) stencil's nonzeros.
        @test nnz(A_zero) == nnz(A_ref)
        @test !(resolve_form_ast(a_zero) isa OperatorAdd)

        # The integer `0` above is what buys that, and the restriction is deliberate
        # (gpena/Bramble.jl#240): a floating-point `0.0` keeps its term, so the matrix
        # still holds the stiffness band as stored zeros. Same numbers, wider pattern.
        a_zero_float = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + 0.0 * inner₊ₓ(D₋ₓ(u), D₋ₓ(v)))
        A_zero_float = assemble(a_zero_float)
        @test Matrix(A_zero_float) ≈ Matrix(A_ref)
        @test nnz(A_zero_float) > nnz(A_ref)
    end

    @testset "Combining like terms merges two sweeps into one" begin
        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + innerₕ(u, v))
        @test resolve_form_ast(a) isa OperatorScale
        @test resolve_form_ast(a).scalar == 2
        @test Matrix(assemble(a)) ≈ 2 .* H
    end

    @testset "Distributive factoring, checked against two independent single-term forms" begin
        Ax = Matrix(assemble(form(Wₕ, Wₕ, (u, v) -> inner₊ₓ(D₋ₓ(u), D₋ₓ(v)))))
        Ay = Matrix(assemble(form(Wₕ, Wₕ, (u, v) -> inner₊ᵧ(D₋ᵧ(u), D₋ᵧ(v)))))

        a = form(
            Wₕ, Wₕ, (u, v) -> 2 * inner₊ₓ(D₋ₓ(u), D₋ₓ(v)) + 2 * inner₊ᵧ(D₋ᵧ(u), D₋ᵧ(v))
        )
        ast = resolve_form_ast(a)
        @test ast isa OperatorScale
        @test ast.scalar == 2
        @test ast.inner_op isa OperatorAdd
        @test Matrix(assemble(a)) ≈ 2 .* (Ax + Ay)
    end

    @testset "A RefValue coefficient combined at construction still tracks its updates" begin
        β = Ref(1.0)
        a = form(Wₕ, Wₕ, (u, v) -> β * innerₕ(u, v) + β * innerₕ(u, v))
        @test resolve_form_ast(a) isa OperatorScale

        @test Matrix(assemble(a)) ≈ 2 .* H
        β[] = 3.0
        @test Matrix(assemble(a)) ≈ 6 .* H
    end
end

# Rules 4-7: scalar lifting out of an inner product, component distribution on a
# component-mixing sum inside one, grid-function fusion, and shift idempotence. Each reaches
# one layer deeper than rules 1-3 (into `BilinearProduct`/`LinearProduct`/`ShiftNode`), so
# every one gets its own structural check plus a numeric check against an independent
# reference, exactly as rules 1-3 did above.
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

    @testset "Restores structural symmetry/SPD detection" begin
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

    @testset "A lifted scalar feeds the combine rule (bilinear)" begin
        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(2 * D₋ₓ(u), v) + innerₕ(3 * D₋ₓ(u), v))
        ast = resolve_form_ast(a)
        @test ast isa OperatorScale
        @test ast.scalar == 5
        @test ast.inner_op isa BilinearProduct
        @test Matrix(assemble(a)) ≈
              5 .* Matrix(assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(D₋ₓ(u), v))))
    end

    @testset "Lifting from a symbolic source (linear form)" begin
        # `2 * fₕ` for a plain `VectorElement` `fₕ` is an *eager* numeric scaling (the
        # source side is eager, per the forms tutorial) and never builds an `OperatorScale`
        # at all -- there is nothing for this rule to lift there. A `SourceFunction` (as
        # `πₕ(uₕ)` also builds) is a genuine `LazyOp`, so scaling *that* does.
        sf = source_function(x -> x[1] + 1, Val(2))
        l = form(Wₕ, v -> innerₕ(2 * sf, v) + innerₕ(3 * sf, v))
        ast = resolve_form_ast(l)
        @test ast isa OperatorScale
        @test ast.scalar == 5
        @test ast.inner_op isa LinearProduct
        @test assemble(l) ≈ 5 .* assemble(form(Wₕ, v -> innerₕ(sf, v)))
    end

    @testset "Lifting and combining a DiracSource (linear form)" begin
        # `dirac(...)` (a `DiracSource`) is a genuine `LazyOp` source exactly like
        # `source_function` above, so it lifts and combines the same way (#226).
        d = dirac((0.3, 0.4), 1.0)
        @test d isa DiracSource
        # Each coefficient still lifts out of its own inner product. The two terms are *not*
        # combined into `5 * innerₕ(d, v)`, though: a `DiracSource` carries its points and
        # strengths in fields, so deciding the two sides are the same subtree would mean
        # reading those at run time -- and the rule's two outcomes are different node types,
        # which is what cost `form` its inferred return type (gpena/Bramble.jl#240). What may
        # never change is the vector, and it does not.
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

@testset "Grid function fusion" begin
    Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 6), (true, true))
    Wₕ = gridspace(Ωₕ)
    vₕ = Rₕ(Wₕ, x -> x[1] + 1.0)
    wₕ = Rₕ(Wₕ, x -> x[2] + 2.0)

    a = form(Wₕ, Wₕ, (u, v) -> vₕ * (wₕ * innerₕ(u, v)))
    ast = resolve_form_ast(a)
    @test ast isa GridFunctionScale
    @test ast.grid_function ≈ parent(vₕ) .* parent(wₕ)
    @test ast.inner_op isa BilinearProduct

    H = Matrix(Diagonal(collect(weights(Wₕ, Innerh()))))
    @test Matrix(assemble(a)) ≈ Diagonal(parent(vₕ) .* parent(wₕ)) * H
end

@testset "Shift idempotence" begin
    Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 6), (true, true))
    Wₕ = gridspace(Ωₕ)
    A = IdentityOperator(Wₕ)

    @test simplify_ast(shift_op(A, 1, 0)) === A

    s = simplify_ast(shift_op(shift_op(A, 1, 2), 1, 3))
    @test s isa ShiftNode
    @test s.shift_amount == 5
    @test s.inner_op === A

    # a shift and its exact inverse collapse straight to the unshifted operator
    @test simplify_ast(shift_op(shift_op(A, 1, 2), 1, -2)) === A

    # shifts along different dimensions never combine into one node
    s2 = simplify_ast(shift_op(shift_op(A, 1, 2), 2, 3))
    @test s2 isa ShiftNode
    @test s2.inner_op isa ShiftNode

    @testset "Numeric agreement with the combined shift" begin
        Ωₕ1 = mesh(domain(interval(0.0, 1.0)), 8, true)
        Wₕ1 = gridspace(Ωₕ1)
        sf = source_function(x -> x^2 + 1, Val(1))

        b_nested = assemble(form(Wₕ1, v -> innerₕ(shift_op(shift_op(sf, 1, 1), 1, 2), v)))
        b_combined = assemble(form(Wₕ1, v -> innerₕ(shift_op(sf, 1, 3), v)))
        @test b_nested ≈ b_combined
    end
end

@testset "Component distribution on a mixed sum inside one inner product" begin
    Ωₕ = mesh(domain(interval(0.0, 1.0)), 11, true)
    Wₕ = gridspace(Ωₕ)
    Vₕ = Wₕ^Val(2)
    fₕ = Rₕ(Wₕ, x -> sin(π * x[1]))

    @testset "Linear form: a source coupling to two test equations" begin
        # `innerₕ(fₕ, v(1) + v(2))` names test components 1 and 2 inside one product --
        # unroutable as a single term (`test_component_or_nothing` throws on it,
        # block_extract.jl) before this rule.
        l = form(Vₕ, v -> innerₕ(fₕ, v(1) + v(2)))
        @test resolve_form_ast(l) isa OperatorAdd

        l_ref = form(Vₕ, v -> innerₕ(fₕ, v(1)) + innerₕ(fₕ, v(2)))
        @test assemble(l) ≈ assemble(l_ref)
    end

    @testset "Bilinear form: one trial component coupling to two test equations" begin
        a = form(Vₕ, Vₕ, (u, v) -> innerₕ(u(1), v(1) + v(2)))
        @test resolve_form_ast(a) isa OperatorAdd

        a_ref = form(Vₕ, Vₕ, (u, v) -> innerₕ(u(1), v(1)) + innerₕ(u(1), v(2)))
        @test Matrix(assemble(a)) ≈ Matrix(assemble(a_ref))
    end

    @testset "A scaled mixed sum distributes the scale along with it" begin
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

    @testset "A same-component sum is not distributed (no sweep-count regression)" begin
        # `v(1) + D₋ₓ(v(1))` names the same component on both sides, so it already routes
        # as one term; distributing it anyway would trade that single sweep for two.
        l = form(Vₕ, v -> innerₕ(fₕ, v(1) + D₋ₓ(v(1))))
        @test !(resolve_form_ast(l) isa OperatorAdd)
    end
end

@testset "Sums of three or more mixing components do not throw (#235)" begin
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

    @testset "Linear form: scalar × sum of 3 and 4 mixing test components" begin
        for (V, n) in ((V3, 3), (V4, 4))
            l = form(V, v -> 2.0 * sum(innerₕ(fₕ, v(i)) for i in 1:n))
            @test resolve_form_ast(l) isa OperatorAdd

            l_dist = form(V, v -> sum(2.0 * innerₕ(fₕ, v(i)) for i in 1:n))
            @test assemble(l) ≈ assemble(l_dist)
        end
    end

    @testset "Linear form: a grid-function coefficient distributes the same way" begin
        for (V, n) in ((V3, 3), (V4, 4))
            l = form(V, v -> cₕ * sum(innerₕ(fₕ, v(i)) for i in 1:n))
            @test resolve_form_ast(l) isa OperatorAdd

            l_dist = form(V, v -> sum(cₕ * innerₕ(fₕ, v(i)) for i in 1:n))
            @test assemble(l) ≈ assemble(l_dist)
        end
    end

    @testset "Bilinear form: one trial component against a sum of 3 and 4 test components" begin
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

    @testset "innerₕ(divₕ(u), divₕ(v)) in 3D: a three-term mixing sum on both sides" begin
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

        A_nine = zeros(size(A_div))
        for i in 1:3, j in 1:3

            a_ij = form(V3d, V3d, (p, q) -> innerₕ(ops[i](p(i)), ops[j](q(j))))
            A_nine .+= Matrix(assemble(a_ij))
        end
        @test A_div ≈ A_nine
    end
end

# --- Type stability under a runtime coefficient ------------------------------------- #
#
# Every rewrite in `simplify_ast` decides what *node type* to return, so a rule that reads a
# coefficient's value makes `form`'s return type depend on that value: inferred as a `Union`
# of the rewritten and unrewritten trees whenever the compiler cannot fold the comparison.
# That costs every caller a dynamic dispatch into the assembly engine, and it is what Enzyme
# rejects with `IllegalTypeAnalysisException` when differentiating with respect to an
# operator's own coefficient (gpena/Bramble.jl#240). The value-reading rules are therefore
# restricted to `Integer` coefficients, and these are the checks that pin it: `isconcretetype`
# on the inferred return type, which is exactly the property that was false before.
#
# `Float64` arguments rather than literals on purpose -- a literal coefficient is constant
# -folded by inference and comes out concrete either way, so a literal proves nothing here.
_rt_single(θ::Float64, W) = form(W, W, (u, v) -> θ * innerₕ(u, v))
_rt_sum(θ::Float64, W) = form(W, W, (u, v) -> θ * inner₊ₓ(D₋ₓ(u), D₋ₓ(v)) + (1 - θ) * innerₕ(u, v))
_rt_bare(θ::Float64, W) = form(W, W, (u, v) -> θ * inner₊ₓ(D₋ₓ(u), D₋ₓ(v)) + innerₕ(u, v))
_rt_equal(θ::Float64, W) = form(W, W, (u, v) -> θ * inner₊ₓ(D₋ₓ(u), D₋ₓ(v)) + θ * innerₕ(u, v))
_rt_lifted(θ::Float64, W) = form(W, W, (u, v) -> innerₕ(θ * u, v))
_rt_linear(θ::Float64, W, fₕ) = form(W, v -> θ * innerₕ(fₕ, v))

# Two terms of the *same shape* carrying runtime data. These were the two shapes the `Integer`
# restriction above did not reach: the like-term rule used to be decided by comparing the two
# subtrees field by field at run time, so `form` inferred as `Union{..., OperatorAdd},
# {..., OperatorScale}}` for any sum of same-typed data-carrying terms -- not just the
# duplicate expressions that limitation was first described as. Two distinct diffusion
# coefficients in one bilinear form, or two source vectors in one linear form, are the
# ordinary way to write a model, and both were affected. The rule is now gated on
# `_statically_equal`, so these infer concretely and the sums assemble as written.
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

_infers(f, sig) = isconcretetype(only(Base.return_types(f, sig)))

@testset "A runtime coefficient leaves `form` type-stable" begin
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

    # Two same-shaped terms carrying distinct runtime data, bilinear and linear
    # (gpena/Bramble.jl#240). These were `false` before the like-term rule was gated.
    g₁ = Rₕ(Wₕ, x -> 1.0 + x[1])
    g₂ = Rₕ(Wₕ, x -> 2.0 - x[2])
    @test _infers(_rt_two_coeffs, (W, typeof(g₁), typeof(g₂)))
    @test _infers(_rt_two_sources, (W, typeof(g₁), typeof(g₂)))

    # ... and the deliberate hole, asserted as such: see `_rt_runtime_int`'s comment.
    @test !_infers(_rt_runtime_int, (Int, W))

    # The rewrites themselves are unchanged for the `Integer` coefficients they are written
    # for, which is what makes the restriction affordable -- these are the same trees the
    # rules built before, still built.
    A = IdentityOperator(Wₕ)
    B = D₋ₓ(A)
    S = innerₕ(trial_function(Wₕ), test_function(Wₕ))
    @test simplify_ast(2 * A + 2 * B) isa OperatorScale     # common factor, still factored
    @test simplify_ast(2 * S + 3 * S) isa OperatorScale     # like terms, still combined
    @test simplify_ast(0 * A) isa ZeroOperator              # still elided
    @test simplify_ast(1 * A) === A

    # ... and a `Float64` coefficient is carried through as written instead. Assembling it
    # must still give the same matrix: this pass may change how a form is routed, never what
    # it computes.
    @test simplify_ast(2.0 * A + 2.0 * B) isa OperatorAdd
    @test Matrix(assemble(form(Wₕ, Wₕ, (u, v) -> 2.0 * innerₕ(u, v) + 2.0 * inner₊ₓ(D₋ₓ(u), D₋ₓ(v))))) ≈
          2.0 .* Matrix(assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊ₓ(D₋ₓ(u), D₋ₓ(v)))))

    # A shared `Ref` coefficient still factors: that rule compares object identity, which
    # inference settles from the types, so it never cost stability in the first place.
    β = Ref(1.5)
    @test simplify_ast(β * A + β * B) isa OperatorScale
end

end # module FormSimplifierTests
