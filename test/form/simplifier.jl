using Test
using Bramble
using LinearAlgebra: Diagonal
using SparseArrays: nnz
using Bramble:
    IdentityOperator,
    ZeroOperator,
    OperatorAdd,
    OperatorScale,
    GridFunctionScale,
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
