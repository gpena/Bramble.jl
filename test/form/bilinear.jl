using Test
using Bramble
using LinearAlgebra: Diagonal, I
using SparseArrays: sparse, nnz, nonzeros
using Bramble: BilinearForm, form, assemble, assemble!, assemble_parallel!, trial_space,
               test_space, resolve_form_ast, allocate_system_matrix, ndofs, values,
               Innerh, Innerplus, block_of, trial_component_or_nothing,
               test_component_or_nothing

# Assembling the matrix of a bilinear form.
#
# The convention throughout: `a(u, v) = vᵀ A u`, so a row of `A` is indexed by the test
# function and a column by the trial function. Every check below is written against a matrix
# built by code that shares nothing with `local_stencil` — the operator's own sparse form and
# the inner product's own weights — so the two agree or one of them is wrong.

@testset verbose=true "Bilinear forms" begin
    S = interval(0.0, 1.0) × interval(0.0, 1.0)
    Ωₕ = mesh(domain(S, :walls => get_boundary_symbols(S)), (9, 7), (true, true))
    Wₕ = gridspace(Ωₕ)
    n = ndofs(Wₕ)

    H = Matrix(Diagonal(collect(weights(Wₕ, Innerh()))))
    Hx = Matrix(Diagonal(collect(weights(Wₕ, Innerplus(), 1))))
    Dx = Matrix(D₋ₓ(Wₕ))
    Mx = Matrix(M₋ₓ(Wₕ))
    Idm = Matrix(1.0I, n, n)

    @testset "the assembled matrix is the matrix expression" begin
        @test Matrix(assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v)))) ≈ H
        @test Matrix(assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(D₋ₓ(u), v)))) ≈ H * Dx
        @test Matrix(assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, D₋ₓ(v))))) ≈
              transpose(Dx) * H
        @test Matrix(assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(M₋ₓ(u), v)))) ≈ H * Mx

        # the stiffness matrix, which is the reason the package exists
        @test Matrix(assemble(form(Wₕ, Wₕ, (u, v) -> inner₊ₓ(D₋ₓ(u), D₋ₓ(v))))) ≈
              transpose(Dx) * Hx * Dx

        # a sum of two kinds, and a linear combination inside one argument
        @test Matrix(assemble(form(Wₕ, Wₕ,
            (u, v) -> innerₕ(u, v) + inner₊ₓ(D₋ₓ(u), D₋ₓ(v))))) ≈
              H + transpose(Dx) * Hx * Dx
        @test Matrix(assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v + 2 * D₋ₓ(v))))) ≈
              transpose(Idm + 2 * Dx) * H
    end

    @testset "the entry points agree" begin
        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊ₓ(D₋ₓ(u), D₋ₓ(v)))
        Apar = assemble(a)
        Aser = similar(sparse(Apar))
        assemble!(Aser, a)

        @test Matrix(Aser) ≈ Matrix(Apar)
        @test trial_space(a) === Wₕ
        @test test_space(a) === Wₕ

        # the functor contracts as vᵀ A u
        uₕ = Rₕ(Wₕ, x -> sin(x[1]))
        vₕ = Rₕ(Wₕ, x -> x[2] + 1)
        @test a(uₕ, vₕ) ≈ dot(values(vₕ), Matrix(Apar) * values(uₕ))

        # re-assembly overwrites rather than accumulating
        assemble!(Aser, a)
        @test Matrix(Aser) ≈ Matrix(Apar)

        # and Dirichlet rows are pinned
        Abc = assemble(a; dirichlet_labels = :walls)
        marked = index_in_marker(Ωₕ, :walls)
        for i in 1:n
            marked[i] || continue
            @test Abc[i, i] ≈ 1.0
            @test count(!iszero, Abc[i, :]) == 1
        end
    end

    @testset "blocks of a composite system" begin
        Vₕ = gridspace(Ωₕ, Val(2))
        blk(A, i, j) = Matrix(A)[((i - 1) * n + 1):(i * n), ((j - 1) * n + 1):(j * n)]

        # A term naming neither side is the same integrand on every diagonal block, because
        # Σᵢ innerₕ(uᵢ, vᵢ) is block diagonal and not full.
        Ad = assemble(form(Vₕ, Vₕ, (u, v) -> innerₕ(u, v)))
        @test blk(Ad, 1, 1) ≈ H
        @test blk(Ad, 2, 2) ≈ H
        @test all(iszero, blk(Ad, 1, 2))
        @test all(iszero, blk(Ad, 2, 1))

        # A term naming both sides is one block, off-diagonal included. This is what used to
        # vanish: the pattern held diagonal blocks only and `add_to_sparse!` returns quietly
        # when an entry is missing, so the contribution was dropped without a word.
        Ao = assemble(form(Vₕ, Vₕ, (u, v) -> innerₕ(u(1), v(2))))
        @test blk(Ao, 2, 1) ≈ H          # row from the test component, column from the trial
        @test all(iszero, blk(Ao, 1, 1))
        @test all(iszero, blk(Ao, 2, 2))
        @test all(iszero, blk(Ao, 1, 2))

        # a full 2x2 system
        Af = assemble(form(Vₕ, Vₕ,
            (u, v) -> innerₕ(u(1), v(1)) + innerₕ(u(1), v(2)) +
                      innerₕ(u(2), v(1)) + innerₕ(u(2), v(2))))
        for i in 1:2, j in 1:2

            @test blk(Af, i, j) ≈ H
        end

        # blocks carrying operators, which is what a real coupled system looks like
        Aop = assemble(form(Vₕ, Vₕ,
            (u, v) -> inner₊ₓ(D₋ₓ(u(1)), D₋ₓ(v(1))) + innerₕ(u(2), v(1))))
        @test blk(Aop, 1, 1) ≈ transpose(Dx) * Hx * Dx
        @test blk(Aop, 1, 2) ≈ H
        @test all(iszero, blk(Aop, 2, 2))
    end

    @testset "serial and parallel assembly agree" begin
        # The serial and threaded paths are separate walks over the same terms, and
        # `assemble` reaches only the threaded one — so a break in the serial path is
        # invisible unless it is compared against the other. It was, once.
        Vₕ = gridspace(Ωₕ, Val(2))
        V3 = gridspace(Ωₕ, Val(3))

        for (nm, sp, g) in (
            ("scalar", Wₕ, (u, v) -> innerₕ(u, v)),
            ("scalar with operators", Wₕ, (u, v) -> inner₊ₓ(D₋ₓ(u), D₋ₓ(v))),
            ("composite, diagonal", Vₕ, (u, v) -> innerₕ(u, v)),
            ("composite, off-diagonal", Vₕ, (u, v) -> innerₕ(u(1), v(2))),
            ("composite, mixed spellings", Vₕ,
            (u, v) -> innerₕ(u, v) + innerₕ(u(1), v(2))),
            ("three components, crossed", V3,
            (u, v) -> innerₕ(u(1), v(3)) + innerₕ(u(3), v(1))),
            ("blocks with operators", Vₕ,
            (u, v) -> inner₊ₓ(D₋ₓ(u(1)), D₋ₓ(v(1))) + innerₕ(u(2), v(2))))
            a = form(sp, sp, g)
            Apar = assemble(a)
            Aser = similar(sparse(Apar))
            assemble!(Aser, a)
            @test Matrix(Aser) ≈ Matrix(Apar)
        end
    end

    @testset "nesting is just more leaves" begin
        # A composite of composites needs no separate type and no separate constructor. Its
        # blocks are numbered by leaf, so a two-by-two nesting is four blocks addressed
        # `u(1)` through `u(4)` — the same spelling a flat space uses, and the same one
        # `linear.jl` uses for a right-hand side.
        #
        # There used to be a `CoupledBilinearForm` reached only when a space was
        # hierarchical, taking its expression as nested tuples: `((u, p), (v, q)) -> ...`.
        # It was the only way to reach an off-diagonal block, which is why a flat space
        # could not have one.
        nested = Bramble.CompositeGridSpace((gridspace(Ωₕ, Val(2)), gridspace(Ωₕ, Val(2))))
        @test length(Bramble.leaf_spaces_offsets(nested)) == 4
        @test ndofs(nested) == 4n

        blk(A, i, j) = Matrix(A)[((i - 1) * n + 1):(i * n), ((j - 1) * n + 1):(j * n)]

        A = assemble(form(nested, nested, (u, v) -> innerₕ(u(1), v(3))))
        @test blk(A, 3, 1) ≈ H
        for i in 1:4, j in 1:4

            (i == 3 && j == 1) && continue
            @test all(iszero, blk(A, i, j))
        end

        Ad = assemble(form(nested, nested, (u, v) -> innerₕ(u, v)))
        for i in 1:4
            @test blk(Ad, i, i) ≈ H
        end

        a = form(nested, nested, (u, v) -> innerₕ(u(2), v(4)) + innerₕ(u, v))
        Ap = assemble(a)
        As = similar(sparse(Ap))
        assemble!(As, a)
        @test Matrix(As) ≈ Matrix(Ap)

        # and the range check counts leaves, not top-level components
        @test_throws ArgumentError assemble(form(nested, nested,
            (u, v) -> innerₕ(u(1), v(5))))
    end

    @testset "a term has to name both components or neither" begin
        Vₕ = gridspace(Ωₕ, Val(2))

        # `innerₕ(u(1), v)` is not something written in a variational formulation, and
        # reading it as a whole row or column of blocks would be a guess.
        @test_throws ArgumentError assemble(form(Vₕ, Vₕ, (u, v) -> innerₕ(u(1), v)))
        @test_throws ArgumentError assemble(form(Vₕ, Vₕ, (u, v) -> innerₕ(u, v(2))))

        # and a component the space does not have is an error rather than an empty block
        @test_throws ArgumentError assemble(form(Vₕ, Vₕ, (u, v) -> innerₕ(u(1), v(5))))
        @test_throws ArgumentError assemble(form(Vₕ, Vₕ, (u, v) -> innerₕ(u(0), v(1))))

        # the walks themselves, which is where the decision is made
        u = Bramble.TrialFunction{2}()
        v = Bramble.TestFunction{2}()
        @test trial_component_or_nothing(innerₕ(u(1), v(2))) == 1
        @test test_component_or_nothing(innerₕ(u(1), v(2))) == 2
        @test trial_component_or_nothing(innerₕ(u, v)) === nothing
        @test block_of(innerₕ(u(1), v(2)), 2, 2) == (1, 2)
        @test block_of(innerₕ(u, v), 2, 2) === nothing
        @test_throws ArgumentError block_of(innerₕ(u(1), v), 2, 2)
    end

    @testset "construction is cheap" begin
        # `form` used to evaluate a sample stencil and bin the whole grid into a vector of
        # vectors before anything was assembled — 9,271,600 B at 90,000 degrees of freedom.
        # The colouring is a property of the AST and the grid, so it is derived where it is
        # used.
        function _form_bytes(W)
            form(W, W, (u, v) -> innerₕ(u, v))
            return @allocated form(W, W, (u, v) -> innerₕ(u, v))
        end
        @test _form_bytes(Wₕ) < 8 * n          # far below one vector, let alone the grid

        # and a malformed expression fails here rather than at the first assemble
        @test_throws ArgumentError form(Wₕ, Wₕ, (u, v) -> 42)
    end
end
