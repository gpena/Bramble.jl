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

    @testset "Zero and identity" begin
        @test simplify_ast(0 * A) isa ZeroOperator
        @test simplify_ast(0.0 * A) isa ZeroOperator
        @test simplify_ast(1 * A) === A
        @test simplify_ast(1.0 * A) === A

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
        # `A + A -> 2 * A`
        c = simplify_ast(A + A)
        @test c isa OperatorScale
        @test c.scalar == 2
        @test c.inner_op === A

        # `c1 * A + c2 * A -> (c1 + c2) * A`
        c = simplify_ast(2 * A + 3 * A)
        @test c isa OperatorScale
        @test c.scalar == 5
        @test c.inner_op === A

        # `A - A -> 0`
        @test simplify_ast(A - A) isa ZeroOperator

        # structurally different subtrees are never merged, however they compare numerically
        @test !(simplify_ast(A + B) isa OperatorScale)

        # two independently-built grid functions are never treated as the same operator,
        # even scaling the same inner term identically -- only object identity counts
        vₕ = Rₕ(Wₕ, x -> x[1])
        wₕ = Rₕ(Wₕ, x -> x[1])  # same values, different array: must not be merged
        different = simplify_ast(2 * (vₕ * B) + 2 * (wₕ * B))
        @test different isa OperatorScale  # still factors the common `2`
        @test different.inner_op isa OperatorAdd
        @test different.inner_op.left_op isa GridFunctionScale
        @test different.inner_op.right_op isa GridFunctionScale

        # the same grid function reused on both sides combines the coefficients
        same = simplify_ast(2 * (vₕ * B) + 3 * (vₕ * B))
        @test same isa OperatorScale
        @test same.scalar == 5
        @test same.inner_op isa GridFunctionScale
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
        # simplifying an already-simplified tree is a no-op
        t = simplify_ast(2 * A + 2 * B)
        @test simplify_ast(t) === t

        # the rules apply however deep the algebra sits, not only at the root
        nested = simplify_ast(A + (0 * B + (1 * A)))
        @test nested isa OperatorScale
        @test nested.scalar == 2
        @test nested.inner_op === A
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
        a_zero = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + 0.0 * inner₊ₓ(D₋ₓ(u), D₋ₓ(v)))

        A_ref = assemble(a_ref)
        A_zero = assemble(a_zero)
        @test Matrix(A_zero) ≈ Matrix(A_ref)
        # the zero-scaled `inner₊ₓ(D₋ₓ(u), D₋ₓ(v))` term contributes nothing to the
        # pattern: without elision it would add its own (much wider) stencil's nonzeros.
        @test nnz(A_zero) == nnz(A_ref)
        @test !(resolve_form_ast(a_zero) isa OperatorAdd)
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
        l = form(Wₕ, v -> innerₕ(2 * d, v) + innerₕ(3 * d, v))
        ast = resolve_form_ast(l)
        @test ast isa OperatorScale
        @test ast.scalar == 5
        @test ast.inner_op isa LinearProduct
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

end # module FormSimplifierTests
