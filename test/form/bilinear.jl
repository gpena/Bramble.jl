module FormBilinearTests

using Test
using Bramble
using Bramble: matrix_type, execution_policy
# Internal names: defined and documented, not exported.
import Bramble: M₊ᵧ
using ForwardDiff
using LinearAlgebra: Diagonal, I, diag, dot
import SparseArrays
using SparseArrays: sparse, nnz, nonzeros, rowvals, SparseMatrixCSC
using Random
using Supposition
using ..TestUtils: WITH_SLOW_TESTS, WITH_AD_TESTS, @test_allocs
using ..TestUtils: _nonuniform_points
using Bramble:
               BilinearForm,
               form,
               assemble,
               assemble!,
               assemble_parallel!,
               trial_space,
               test_space,
               resolve_form_ast,
               resolve_ast,
               allocate_system_matrix,
               ndofs,
               restrict_to,
               Innerh,
               Innerplus,
               block_of,
               trial_component_or_nothing,
               test_component_or_nothing,
               Block,
               blocks,
               leaf_spaces_offsets,
               visit_bilinear_stencil,
               PatternSink,
               _sink_entry!,
               _sink_dedups,
               _entry_target,
               _trial_column,
               AbsoluteColumn,
               TrialFunction,
               TestFunction,
               indices,
               Dcₓ,
               D₋ᵧ,
               D₋ₓ,
               Mₓ,
               index_in_marker,
               inner₊ᵧ,
               inner₊ₓ,
               set_points!,
               weights

# Assembling the matrix of a bilinear form.
#
# The convention throughout: `a(u, v) = vᵀ A u`, so a row of `A` is indexed by the test
# function and a column by the trial function. Every check below is written against a matrix
# built by code that shares nothing with `local_stencil`: the operator's own sparse form and
# the inner product's own weights, so the two agree or one of them is wrong.

@testset "Bilinear forms" begin
    S = interval(0.0, 1.0) × interval(0.0, 1.0)
    Ωₕ = mesh(domain(S, :walls => boundary_symbols(S)), (9, 7), (true, true))
    Wₕ = gridspace(Ωₕ)
    n = ndofs(Wₕ)

    H = Matrix(Diagonal(collect(weights(Wₕ, Innerh()))))
    Hx = Matrix(Diagonal(collect(weights(Wₕ, Innerplus(), 1))))
    Dx = Matrix(D₋ₓ(Wₕ))
    Mx = Matrix(Mₓ(Wₕ))
    Idm = Matrix(1.0I, n, n)

    @testset "Matrix expression equivalence" begin
        @test Matrix(assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v)))) ≈ H
        @test Matrix(assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(D₋ₓ(u), v)))) ≈ H * Dx
        @test Matrix(assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, D₋ₓ(v))))) ≈
              transpose(Dx) * H
        @test Matrix(assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(Mₓ(u), v)))) ≈ H * Mx

        # the stiffness matrix, which is the reason the package exists
        @test Matrix(assemble(form(Wₕ, Wₕ, (u, v) -> inner₊ₓ(D₋ₓ(u), D₋ₓ(v))))) ≈
              transpose(Dx) * Hx * Dx

        # a sum of two kinds, and a linear combination inside one argument
        @test Matrix(
            assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊ₓ(D₋ₓ(u), D₋ₓ(v))))
        ) ≈ H + transpose(Dx) * Hx * Dx
        @test Matrix(assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v + 2 * D₋ₓ(v))))) ≈
              transpose(Idm + 2 * Dx) * H

        # unary minus on a whole form, on one operator and on the gradient tuple
        @test Matrix(assemble(form(Wₕ, Wₕ, (u, v) -> -inner₊ₓ(D₋ₓ(u), D₋ₓ(v))))) ≈
              -(transpose(Dx) * Hx * Dx)
        @test Matrix(assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(-D₋ₓ(u), v)))) ≈ -(H * Dx)
        G = Matrix(assemble(form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))))
        @test !iszero(G)
        @test Matrix(assemble(form(Wₕ, Wₕ, (u, v) -> -inner₊(∇ₕ(u), ∇ₕ(v))))) ≈ -G
        @test Matrix(assemble(form(Wₕ, Wₕ, (u, v) -> inner₊(-∇ₕ(u), ∇ₕ(v))))) ≈ -G

        # the `-1` is an Integer literal, so a Float32 space stays Float32
        Ω32 = mesh(domain(interval(0.0f0, 1.0f0)), 9, false; backend = backend(Float32))
        W32 = gridspace(Ω32)
        K32 = assemble(form(W32, W32, (u, v) -> -innerₕ(D₋ₓ(u), D₋ₓ(v))))
        @test eltype(K32) === Float32
        @test K32 ≈ -assemble(form(W32, W32, (u, v) -> innerₕ(D₋ₓ(u), D₋ₓ(v))))
    end

    @testset "Entry point agreement" begin
        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊ₓ(D₋ₓ(u), D₋ₓ(v)))
        Apar = assemble(a)

        @test trial_space(a) === Wₕ
        @test test_space(a) === Wₕ

        # the functor contracts as vᵀ A u
        uₕ = Rₕ(Wₕ, x -> sin(x[1]))
        vₕ = Rₕ(Wₕ, x -> x[2] + 1)
        @test a(uₕ, vₕ) ≈ dot(parent(vₕ), Matrix(Apar) * parent(uₕ))

        # Dirichlet rows are pinned
        Abc = assemble(a; dirichlet = :walls)
        marked = index_in_marker(Ωₕ, :walls)
        for i in 1:n
            marked[i] || continue
            @test Abc[i, i] ≈ 1.0
            @test count(!iszero, Abc[i, :]) == 1
        end
    end

    @testset "Composite blocks" begin
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
        Af = assemble(
            form(
            Vₕ,
            Vₕ,
            (u, v) -> innerₕ(u(1), v(1)) +
                      innerₕ(u(1), v(2)) +
                      innerₕ(u(2), v(1)) +
                      innerₕ(u(2), v(2))
        ),
        )
        for i in 1:2, j in 1:2

            @test blk(Af, i, j) ≈ H
        end

        # blocks carrying operators, which is what a real coupled system looks like
        Aop = assemble(
            form(Vₕ, Vₕ, (u, v) -> inner₊ₓ(D₋ₓ(u(1)), D₋ₓ(v(1))) + innerₕ(u(2), v(1)))
        )
        @test blk(Aop, 1, 1) ≈ transpose(Dx) * Hx * Dx
        @test blk(Aop, 1, 2) ≈ H
        @test all(iszero, blk(Aop, 2, 2))
    end

    @testset "dirichlet_components restriction" begin
        # The motivating case: a Stokes-style system where leaf 1 ("velocity") gets a
        # boundary condition and leaf 2 ("pressure") stays completely free. Before
        # `dirichlet_components` existed, `dirichlet` bound to every leaf sharing the
        # named marker: there was no way to say "this leaf only" through `assemble`/
        # `assemble!` at all.
        Vₕ = gridspace(Ωₕ, Val(2))
        a = form(Vₕ, Vₕ, (u, v) -> innerₕ(u(1), v(1)) + innerₕ(u(2), v(2)))
        marked = index_in_marker(Ωₕ, :walls)

        A = assemble(a; dirichlet = :walls, dirichlet_components = 1)
        blk(i, j) = Matrix(A)[((i - 1) * n + 1):(i * n), ((j - 1) * n + 1):(j * n)]

        for i in 1:n
            if marked[i]
                @test blk(1, 1)[i, i] ≈ 1.0
                @test count(!iszero, blk(1, 1)[i, :]) == 1
            end
        end
        # leaf 2 (pressure) is untouched: still exactly the assembled H, no pinned rows
        @test blk(2, 2) ≈ H

        # assemble! into a pre-allocated matrix follows the same keyword
        A2 = allocate_system_matrix(a)
        assemble!(A2, a; dirichlet = :walls, dirichlet_components = 1)
        @test Matrix(A2) ≈ Matrix(A)

        # without dirichlet_components, the same labels bind to every leaf that has the
        # marker (this is the pre-existing, still-default behaviour, confirmed unchanged).
        Aboth = assemble(a; dirichlet = :walls)
        blk2(i, j) = Matrix(Aboth)[((i - 1) * n + 1):(i * n), ((j - 1) * n + 1):(j * n)]
        for i in 1:n
            if marked[i]
                @test blk2(2, 2)[i, i] ≈ 1.0    # leaf 2 pinned too, unlike above
            end
        end
    end

    # Dirichlet labels pin the test space's rows, not the trial space's.
    @testset "Dirichlet pins TEST-space rows (#48)" begin
        # `apply_dirichlet_labels!` used to call `dirichlet_bc!(A, trial_space(form), ...)`.
        # Rows are indexed by the test function (see the file header), so that pinned the
        # wrong rows whenever trial_space and test_space disagree on leaf layout. On a
        # square, same-space form (trial === test, the overwhelmingly common case, and every
        # other test in this file) the two spaces' leaf offsets coincide and the bug is
        # numerically invisible. Catching it needs trial and test spaces that are genuinely
        # different: built from leaves of different sizes, in reversed order, so pinning
        # against the wrong space's offsets lands on different rows entirely rather than
        # merely fewer or more of the same ones.
        n1, n2 = 5, 7
        W1 = gridspace(mesh(domain(interval(0.0, 1.0)), n1, true))   # leaf size n1
        W2 = gridspace(mesh(domain(interval(0.0, 1.0)), n2, true))   # leaf size n2

        trial = W1 × W2   # leaf 1 has size n1 (offset 0), leaf 2 size n2 (offset n1)
        test = W2 × W1   # leaf 1 has size n2 (offset 0), leaf 2 size n1 (offset n2)

        # Cross terms, so each pairing is same-size (required by `_check_block_meshes`) and
        # lands off the "obvious" diagonal: trial leaf 1 (W1, n1) pairs with test leaf 2
        # (W1, n1); trial leaf 2 (W2, n2) pairs with test leaf 1 (W2, n2).
        a = form(trial, test, (u, v) -> innerₕ(u(1), v(2)) + innerₕ(u(2), v(1)))
        @test trial_space(a) !== test_space(a)

        N = n1 + n2
        @test size(assemble(a)) == (N, N)   # square: total ndofs agree, just laid out differently

        # `:boundary` is reserved and auto-computed on every mesh: index 1 and index n. So
        # each leaf contributes exactly two marked rows/columns, with no domain setup needed.
        Abc = assemble(a; dirichlet = :boundary)

        # The two candidate row sets, computed directly rather than re-derived from the fix:
        # pin using test_space's own offsets (what the interface promises), and, separately,
        # what the pre-fix code actually pinned (trial_space's offsets). If these coincided
        # the test would prove nothing; with n1 ≠ n2 and the leaves in reversed order, they
        # do not.
        A_using_test = Matrix(dirichlet_bc!(assemble(a), test_space(a), :boundary))
        A_using_trial = Matrix(dirichlet_bc!(assemble(a), trial_space(a), :boundary))
        @test A_using_test != A_using_trial   # the two spaces really do disagree

        pinned_rows(A) = [i for i in 1:N if A[i, i] ≈ 1.0 && count(!iszero, A[i, :]) == 1]
        correct_rows = sort!(union(1, n2, n2 + 1, N))          # test leaves: offsets 0, n2
        buggy_rows = sort!(union(1, n1, n1 + 1, N))          # trial leaves: offsets 0, n1
        @test correct_rows != buggy_rows   # n1 ≠ n2 makes the two sets genuinely different

        @test pinned_rows(A_using_test) == correct_rows
        @test pinned_rows(A_using_trial) == buggy_rows

        # The fixed `assemble` must agree with test_space's own pinning, not trial_space's.
        @test Matrix(Abc) ≈ A_using_test
        @test !(Matrix(Abc) ≈ A_using_trial)

        # `assemble!` into a pre-allocated matrix takes the same path and must agree.
        A2 = allocate_system_matrix(a)
        assemble!(A2, a; dirichlet = :boundary)
        @test Matrix(A2) ≈ Matrix(Abc)

        # `dirichlet_components` on an asymmetric form restricts by TEST leaf, since that is
        # what the rows mean: component 1 is test's leaf 1 (W2, offset 0, size n2).
        A1 = Matrix(assemble(a; dirichlet = :boundary, dirichlet_components = 1))
        @test pinned_rows(A1) == [1, n2]              # only test leaf 1's marked rows
        @test !(1 + n2 in pinned_rows(A1))              # test leaf 2 untouched
    end

    @testset "Serial vs parallel agreement" begin
        # The serial and threaded paths are separate walks over the same terms, so a break in
        # the serial path is invisible unless it is compared against the other. It was, once.
        # `assemble_parallel!` always threads regardless of the backend's policy,
        # which is what makes it the right fixed reference here; `assemble!` on `Wₕ`'s
        # default Serial() backend gives the serial answer.
        Vₕ = gridspace(Ωₕ, Val(2))
        V3 = gridspace(Ωₕ, Val(3))

        # Any: each space and closure is its own type
        for (nm, sp, g) in Any[
            ("scalar", Wₕ, (u, v) -> innerₕ(u, v)),
            ("scalar with operators", Wₕ, (u, v) -> inner₊ₓ(D₋ₓ(u), D₋ₓ(v))),
            ("composite, diagonal", Vₕ, (u, v) -> innerₕ(u, v)),
            ("composite, off-diagonal", Vₕ, (u, v) -> innerₕ(u(1), v(2))),
            ("composite, mixed spellings", Vₕ, (u, v) -> innerₕ(u, v) + innerₕ(u(1), v(2))),
            (
                "three components, crossed",
                V3,
                (u, v) -> innerₕ(u(1), v(3)) + innerₕ(u(3), v(1))
            ),
            (
                "blocks with operators",
                Vₕ,
                (u, v) -> inner₊ₓ(D₋ₓ(u(1)), D₋ₓ(v(1))) + innerₕ(u(2), v(2))
            )
        ]
            a = form(sp, sp, g)
            Aser = assemble(a)
            Apar = similar(sparse(Aser))
            assemble_parallel!(Apar, a)
            @test Matrix(Aser) ≈ Matrix(Apar)
        end
    end

    @testset "Determinism under threads" begin
        # `≈` above tolerates float summation reordering; the claim here is stronger --
        # bit-for-bit identical `nzval`, which only a genuine absence of a race across the
        # multi-colour scatter can guarantee run after run. Only meaningful with more than
        # one thread actually available: on one thread the colours
        # never run concurrently, so nothing could race in the first place, and CI already
        # runs with JULIA_NUM_THREADS=auto (see the @warn in test/runtests.jl for a local,
        # single-threaded `Pkg.test()`).
        #
        # A small mesh gives each colour very few points, which is exactly where a race at
        # a boundary phase transition is most likely to surface -- a larger mesh averages
        # rare timing windows away. Repeated 50 times against one fixed serial reference,
        # since an intermittent race need not show on the first run.
        if Threads.nthreads() > 1
            Ω5 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 5), (true, true))
            W5 = gridspace(Ω5)
            a5 = form(W5, W5, (u, v) -> inner₊ₓ(D₋ₓ(u), D₋ₓ(v)) + inner₊ᵧ(D₋ᵧ(u), D₋ᵧ(v)))
            Aser5 = sparse(assemble(a5))
            reference = copy(Aser5.nzval)

            for _ in 1:50
                Apar5 = similar(Aser5)
                assemble_parallel!(Apar5, a5)
                @test Apar5.nzval == reference
            end
        else
            @test_skip "bit-for-bit determinism under threads not exercised: only one thread available"
        end
    end

    @testset "Backend policy" begin
        # assemble!/assemble do not hardcode parallel. Both read form.trial_space's
        # execution_policy, defaulting to Serial() like the vector form.
        @test execution_policy(Wₕ) isa Serial
        Ω_par = mesh(
            domain(S, :walls => boundary_symbols(S)),
            (9, 7),
            (true, true);
            backend = backend(policy = Parallel())
        )
        W_par = gridspace(Ω_par)
        @test execution_policy(W_par) isa Parallel

        a_serial = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v))
        a_parallel = form(W_par, W_par, (u, v) -> innerₕ(u, v))

        A_default = assemble(a_serial)
        A_via_policy = assemble(a_parallel)
        @test Matrix(A_via_policy) ≈ Matrix(A_default)

        # Directly against assemble_parallel!, the lower-level entry point that always
        # threads regardless of the backend's policy: a Parallel()-backend assemble must
        # agree with it exactly.
        A_forced_parallel = similar(sparse(A_via_policy))
        assemble_parallel!(A_forced_parallel, a_parallel)
        @test Matrix(A_via_policy) ≈ Matrix(A_forced_parallel)
    end

    @testset "Nested leaf traversal" begin
        # A composite of composites needs no separate type and no separate constructor. Its
        # blocks are numbered by leaf, so a two-by-two nesting is four blocks addressed
        # `u(1)` through `u(4)`: the same spelling a flat space uses, and the same one
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
        @test_throws ArgumentError assemble(
            form(nested, nested, (u, v) -> innerₕ(u(1), v(5)))
        )
    end

    @testset "Component naming rules" begin
        Vₕ = gridspace(Ωₕ, Val(2))

        # `innerₕ(u(1), v)` is not something written in a variational formulation, and
        # reading it as a whole row or column of blocks would be a guess.
        @test_throws ArgumentError assemble(form(Vₕ, Vₕ, (u, v) -> innerₕ(u(1), v)))
        @test_throws ArgumentError assemble(form(Vₕ, Vₕ, (u, v) -> innerₕ(u, v(2))))

        # and a component the space does not have is an error rather than an empty block
        @test_throws ArgumentError assemble(form(Vₕ, Vₕ, (u, v) -> innerₕ(u(1), v(5))))
        @test_throws ArgumentError assemble(form(Vₕ, Vₕ, (u, v) -> innerₕ(u(0), v(1))))
    end

    @testset "Block resolution (#49)" begin
        # `blocks(term, trial_leaves, test_leaves)` is the one place the trial/test row/
        # column asymmetry is resolved, and is testable directly against `leaf_spaces_offsets`
        # without assembling a matrix; the other way to see a wrong offset is in an assembled
        # matrix's numbers. Asymmetric leaf sizes and reversed order, as in the test
        # above, so a row/column offset mix-up lands on the wrong number rather than the same
        # one by coincidence.
        n1, n2 = 5, 7
        W1 = gridspace(mesh(domain(interval(0.0, 1.0)), n1, true))
        W2 = gridspace(mesh(domain(interval(0.0, 1.0)), n2, true))

        trial = W1 × W2   # leaf 1 is W1 at offset 0, leaf 2 is W2 at offset n1
        test = W2 × W1    # leaf 1 is W2 at offset 0, leaf 2 is W1 at offset n2

        trial_leaves = leaf_spaces_offsets(trial)
        test_leaves = leaf_spaces_offsets(test)

        u1 = Bramble.TrialFunction{1}()
        v1 = Bramble.TestFunction{1}()

        # Row from the test leaf, column from the trial leaf.
        @testset "Named block: test row, trial column" begin
            bs = blocks(innerₕ(u1(1), v1(2)), trial_leaves, test_leaves)
            @test length(bs) == 1
            blk = only(bs)
            @test blk isa Block
            @test blk.trial_leaf === W1        # trial leaf 1
            @test blk.test_leaf === W1         # test leaf 2 is also W1
            @test blk.row_offset == n2          # test leaf 2's own offset, not trial's
            @test blk.col_offset == 0           # trial leaf 1's own offset
        end

        # One Block per diagonal leaf pair.
        @testset "Unrouted: one Block per diagonal pair" begin
            bs = blocks(innerₕ(u1, v1), trial_leaves, test_leaves)
            @test length(bs) == 2
            @test bs[1].trial_leaf === W1 && bs[1].test_leaf === W2
            @test bs[1].row_offset == 0 && bs[1].col_offset == 0
            @test bs[2].trial_leaf === W2 && bs[2].test_leaf === W1
            @test bs[2].row_offset == n2 && bs[2].col_offset == n1
        end

        @testset "Zero allocations" begin
            _named() = blocks(innerₕ(u1(1), v1(2)), trial_leaves, test_leaves)
            _diag() = blocks(innerₕ(u1, v1), trial_leaves, test_leaves)
            _named()
            _diag()
            @test (@allocated _named()) == 0
            @test (@allocated _diag()) == 0
        end
    end

    WITH_AD_TESTS && @testset "Matrix differentiation" begin
        # A coefficient in the integrand: a(u, v) = ∫ c·u·v, so A = H·diag(c) and the
        # derivative of `sum(A)` with respect to `cᵢ` is `Hᵢᵢ`. Checked against that rather
        # than against itself, so a gradient of the wrong thing cannot pass.
        Vₕ = gridspace(Ωₕ, Val(2))
        c1 = fill(1.0, n)
        scalar_form(w) = form(Wₕ, Wₕ, (u, v) -> innerₕ(Bramble.element(Wₕ, w) * u, v))

        @test ForwardDiff.gradient(w -> sum(assemble(scalar_form(w))), c1) ≈ diag(H)

        # the element type follows the data: the matrix has to be able to hold a Dual, and
        # taking it from the space instead is what made this impossible
        wd = ForwardDiff.Dual.(c1, 1.0)
        @test eltype(assemble(scalar_form(wd))) <: ForwardDiff.Dual

        # the serial path as well as the threaded one
        @test ForwardDiff.gradient(c1) do w
            a = scalar_form(w)
            A = allocate_system_matrix(a)
            assemble!(A, a)
            sum(A)
        end ≈ diag(H)

        # and through the block routing, where a wrong component would show up as a
        # derivative in a block that should not have one
        @test ForwardDiff.gradient(c1) do w
            sum(
                assemble(
                form(
                Vₕ,
                Vₕ,
                (u, v) -> innerₕ(Bramble.element(Wₕ, w) * u(1), v(1)) +
                          innerₕ(u(2), v(2))
            ),
            ),
            )
        end ≈ diag(H)
    end

    @testset "In-place reassembly" begin
        # The pattern is the expensive half and does not change between assemblies, so the
        # intended shape of a loop is `assemble` once and `assemble!` after. This pins that
        # the second path costs nothing.
        cₕ = Rₕ(Wₕ, x -> 1.0)
        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(cₕ * u, v))

        A = assemble(a)

        # assemble! uses the pre-resolved ast stored in the form and allocates 0 bytes.
        @test_allocs assemble!(A, a)

        # Coefficients with operators pre-resolve at form construction time and also allocate 0 bytes during assembly
        dcₕ = D₋ₓ(cₕ)
        aop = form(Wₕ, Wₕ, (u, v) -> innerₕ(dcₕ * u, v))
        Aop = assemble(aop)
        @test_allocs assemble!(Aop, aop)

        ainline = form(Wₕ, Wₕ, (u, v) -> innerₕ(D₋ₓ(cₕ) * u, v))
        Ain = assemble(ainline)
        @test_allocs assemble!(Ain, ainline)
        @test Matrix(Ain) ≈ Matrix(Aop)              # and the two agree
    end

    # In-place reassembly.
    @testset "Composite reassembly allocates nothing" begin
        # The checks above only exercise the scalar core. The block-routing core (going through
        # `blocks`) needs its own guard, so a routing change cannot
        # reintroduce an allocation (e.g. from building an intermediate `Block` per term).
        Vₕ = gridspace(Ωₕ, Val(2))

        a_diag = form(Vₕ, Vₕ, (u, v) -> innerₕ(u, v))          # blk === nothing path
        @test_allocs assemble!(assemble(a_diag), a_diag)

        a_off = form(Vₕ, Vₕ, (u, v) -> innerₕ(u(1), v(2)))     # named-block path
        @test_allocs assemble!(assemble(a_off), a_off)

        a_mixed = form(Vₕ, Vₕ, (u, v) -> innerₕ(u, v) + innerₕ(u(1), v(2))) # both, one term each
        @test_allocs assemble!(assemble(a_mixed), a_mixed)
    end

    @testset "Diagonal-segment replay: no allocation" begin
        # The third replay shape, alongside the scalar and composite cores above: a term
        # whose recorded positions come out as a constant per-tap stride, which
        # `_try_diagonal_segment` repackages so `DiagonalReplaySink` can walk the interior
        # by arithmetic instead of reading a position list. The entry loops both replay shapes share
        # (`_visit_entries`/`_visit_entries_unguarded`, form/bilinear_traversal.jl).
        #
        # The assertion on `is_diagonal` is what makes this test about that path rather
        # than a fourth copy of the flat one: a stencil-margin or peeling change that
        # stopped producing diagonal segments would otherwise leave this quietly measuring
        # the flat replay again.
        # A 1D uniform mesh, not this file's 2D `Wₕ`: the repackaging needs a constant
        # per-tap stride across the interior, which a 2D stiffness term does not have (its
        # recorded positions jump by a row's worth at each row boundary) -- `assemble` on
        # `Wₕ` records a flat segment, and asserting otherwise is what the first version of
        # this test got wrong.
        Ω1d = mesh(domain(interval(0.0, 1.0)), 41, true)
        W1d = gridspace(Ω1d)
        a_stiff = form(W1d, W1d, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
        A_stiff = assemble(a_stiff)
        assemble!(A_stiff, a_stiff)
        @test a_stiff.cache.segments[1].is_diagonal

        @test_allocs assemble!(A_stiff, a_stiff)
    end

    @testset "Restricted in-place reassembly allocs" begin
        # A `RegionRestriction`'s stencil is `()` or a full tuple depending on the point's
        # marker. Inside a sum (`v + restrict_to(:boundary, v)` below, what the simplifier
        # folds the two terms into) that `Union` used to be collected by a `map` over the
        # summands into a tuple with a non-concrete element, boxed at every grid point: 864 B
        # on the 1D grid and 6720 B on the 2D one, where the unrestricted form allocated
        # nothing. The refill must also match the two terms assembled apart.
        function _loop_bytes(A, a)
            assemble!(A, a)
            assemble!(A, a)
            return @allocated assemble!(A, a)
        end
        Ω1d = mesh(domain(interval(0.0, 1.0)), 11, true)
        Ω2d = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 9), (false, true))
        for Ω in (Ω1d, Ω2d)
            W = gridspace(Ω)
            a = form(W, W, (u, v) -> innerₕ(u, v) + innerₕ(u, restrict_to(:boundary, v)))
            A = assemble(a)
            @test _loop_bytes(A, a) == 0
            parts = assemble(form(W, W, (u, v) -> innerₕ(u, v))) +
                    assemble(form(W, W, (u, v) -> innerₕ(u, restrict_to(:boundary, v))))
            @test Matrix(A) ≈ Matrix(parts)
        end
    end

    @testset "Cached nzval positions (#26)" begin
        # `assemble!` used to search for every scattered entry's nzval position on every
        # call. It now records that search's result the first time a given matrix is
        # assembled into and replays it thereafter -- these exercise both paths and the
        # points where they meet (a matrix swap, a changed `ast`), not just a single
        # before/after allocation count.

        @testset "Replay matches search, repeated calls" begin
            a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊ₓ(D₋ₓ(u), D₋ₓ(v)))
            A = assemble(a)                 # record, inside assemble's own call
            reference = copy(A.nzval)
            for _ in 1:5                    # several replay calls, not just one
                assemble!(A, a)
                @test A.nzval ≈ reference
            end
        end

        @testset "Live coefficients update under replay" begin
            cₕ = Rₕ(Wₕ, x -> 1.0)
            a = form(Wₕ, Wₕ, (u, v) -> innerₕ(cₕ * u, v))
            A = assemble(a)                 # record
            s1 = sum(A)
            nnz_before = nnz(A)
            for factor in (3.0, -2.0, 5.0)
                Rₕ!(cₕ, x -> factor)
                assemble!(A, a)              # replay, each time with a different live value
                @test sum(A) ≈ factor * s1
                @test nnz(A) == nnz_before   # the pattern is untouched by it
            end
        end

        @testset "Dirichlet applied after cached replay" begin
            a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊ₓ(D₋ₓ(u), D₋ₓ(v)))
            A = assemble(a)                             # record, unconstrained
            assemble!(A, a)                             # replay, unconstrained
            assemble!(A, a; dirichlet = :walls)   # replay core, then Dirichlet applied
            marked = index_in_marker(Ωₕ, :walls)
            for i in 1:n
                marked[i] || continue
                @test A[i, i] ≈ 1.0
                @test count(!iszero, A[i, :]) == 1
            end
        end

        # Rebuilds rather than corrupting the replay.
        @testset "Switching matrices rebuilds the cache" begin
            a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊ₓ(D₋ₓ(u), D₋ₓ(v)))
            A1 = assemble(a)
            A2 = similar(sparse(A1))
            assemble!(A2, a)    # different matrix object: must record again, not reuse A1's cache
            @test A2.nzval ≈ A1.nzval
            assemble!(A1, a)    # back to A1: must record again (cache now points at A2)
            @test A1.nzval ≈ A2.nzval
            assemble!(A1, a)    # each is still independently replayable afterwards
            assemble!(A2, a)
            @test A1.nzval ≈ A2.nzval
        end

        # Diagonal, off-diagonal, mixed and nested blocks all replay correctly.
        @testset "Composite: every block kind replays" begin
            Vₕ = gridspace(Ωₕ, Val(2))
            # Any: each closure is its own type
            for g in Any[
                (u, v) -> innerₕ(u, v),
                (u, v) -> innerₕ(u(1), v(2)),
                (u, v) -> innerₕ(u, v) + innerₕ(u(1), v(2)),
                (u, v) -> inner₊ₓ(D₋ₓ(u(1)), D₋ₓ(v(1))) + innerₕ(u(2), v(1))
            ]
                a = form(Vₕ, Vₕ, g)
                A = assemble(a)
                reference = copy(A.nzval)
                for _ in 1:3
                    assemble!(A, a)
                    @test A.nzval ≈ reference
                end
            end

            # more than two segments to replay, in order (a nesting shape)
            nested = Bramble.CompositeGridSpace((
                gridspace(Ωₕ, Val(2)), gridspace(Ωₕ, Val(2))
            ))
            a = form(nested, nested, (u, v) -> innerₕ(u(2), v(4)) + innerₕ(u, v))
            A = assemble(a)
            reference = copy(A.nzval)
            for _ in 1:3
                assemble!(A, a)
                @test A.nzval ≈ reference
            end
        end

        # Reassembles rather than replaying the stale cache.
        @testset "Different form, same matrix: rebuilds" begin
            # Each form keeps its own `_AssemblyCache` (keyed on the exact matrix object it
            # last assembled into), so assembling a second, same-reach form into `a`'s matrix
            # records fresh under *its own* cache rather than touching `a`'s -- `a`'s own
            # cache is still valid afterwards and replays correctly. Assembling the
            # other form directly demonstrates this without the deprecated `ast` keyword.
            a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v))
            A = assemble(a)                                   # records under a's own cache
            alt = form(Wₕ, Wₕ, (u, v) -> 2.0 * innerₕ(u, v))  # same reach, different form
            assemble!(A, alt)                                  # alt's own cache: fresh record
            @test sum(A) ≈ 2 * sum(H)
            assemble!(A, a)                                    # a's own cache, untouched, still valid
            @test Matrix(A) ≈ H
        end
    end

    # A pattern that cannot hold the form raises.
    @testset "Too-small pattern raises (#50)" begin
        # `add_to_sparse!` used to return quietly when an entry was missing, so a matrix
        # whose pattern was built for a different form assembled to a plausible wrong
        # answer. Both the serial recording pass and the threaded path now say so instead.
        Ω = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (6, 5), (true, true))
        W = gridspace(Ω)

        narrow = form(W, W, (u, v) -> innerₕ(u, v))            # reaches the diagonal only
        wide = form(W, W, (u, v) -> innerₕ(D₋ₓ(u), D₋ₓ(v)))    # reaches a neighbour too

        # Serial: the recording pass searches and reports. `wide` is assembled directly
        # into a matrix built for `narrow`'s (narrower) pattern, rather than overriding
        # `narrow`'s own `ast` keyword -- the two are equivalent, and only the latter is
        # deprecated.
        A = allocate_system_matrix(narrow, resolve_form_ast(narrow))
        @test_throws ArgumentError assemble!(A, wide)

        msg = try
            assemble!(allocate_system_matrix(narrow, resolve_form_ast(narrow)), wide)
        catch e
            sprint(showerror, e)
        end
        @test occursin("outside its preallocated", msg)

        # Threaded: same refusal, reached through `add_to_sparse!` rather than the cache.
        Wp = gridspace(
            mesh(
            domain(interval(0.0, 1.0) × interval(0.0, 1.0)),
            (6, 5),
            (true, true);
            backend = backend(policy = Parallel())
        ),
        )
        np = form(Wp, Wp, (u, v) -> innerₕ(u, v))
        wp = form(Wp, Wp, (u, v) -> innerₕ(D₋ₓ(u), D₋ₓ(v)))
        Ap = allocate_system_matrix(np, resolve_form_ast(np))
        @test_throws Exception assemble_parallel!(Ap, wp)

        # and the matching pattern still assembles, so the check is not simply always on
        Aok = allocate_system_matrix(wide, resolve_form_ast(wide))
        @test assemble!(Aok, wide) === Aok
    end

    @testset "One traversal, pluggable sinks (#50)" begin
        # What the sink split buys: the traversal is now testable on its own, against a sink
        # that only records. Before this, every property below could only be checked through
        # a fully assembled matrix, where a dropped entry looks like a zero.
        struct CollectSink
            seen::Vector{Tuple{Int, Int, Float64}}
        end
        # The sink contract's fifth argument is the replay slot, which only `ReplaySink`
        # reads; a recording sink ignores it.
        Bramble._sink_entry!(s::CollectSink, row::Int, col::Int, w, ::Int) = (
            push!(s.seen, (row, col, Float64(w))); nothing)

        Ω = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 6), (true, true))
        W = gridspace(Ω)
        u, v = TrialFunction{2}(), TestFunction{2}()

        @testset "Every scattered entry is in the pattern" begin
            # Both traversals read the same walk, so the invariant can be asserted
            # directly rather than inferred from a matrix that came out right.
            for ast in (
                resolve_form_ast(form(W, W, (a, b) -> innerₕ(a, b))),
                resolve_form_ast(form(W, W, (a, b) -> innerₕ(D₋ₓ(a), D₋ₓ(b)))),
                resolve_form_ast(form(W, W, (a, b) -> inner₊(∇ₕ(a), ∇ₕ(b)))),
                resolve_form_ast(form(W, W, (a, b) -> innerₕ(Dcₓ(a), M₊ᵧ(b))))
            )
                pat = visit_bilinear_stencil(PatternSink(Int[], Int[]), ast, W, 0, 0)
                pattern = Set(zip(pat.I_vec, pat.J_vec))
                got = visit_bilinear_stencil(CollectSink([]), ast, W, 0, 0)
                @test !isempty(got.seen)
                @test all(((r, c, _),) -> (r, c) in pattern, got.seen)
            end
        end

        # The pattern pass de-duplicates and the value pass does not.
        @testset "Pattern de-duplicates, values do not" begin
            # Two identical terms name every coordinate twice. The pattern wants each once;
            # the values have to accumulate both, or the matrix comes out halved.
            #
            # Built with `resolve_ast` directly rather than `form(...)`: `form` runs
            # `simplify_ast`, which combines two identical terms into
            # one (`innerₕ(a,b) + innerₕ(a,b) -> 2 * innerₕ(a,b)`) precisely to avoid the
            # double sweep this test exists to protect against -- so producing the
            # duplicate-term tree this traversal invariant is about has to bypass it.
            ast = resolve_ast(innerₕ(u, v) + innerₕ(u, v))
            pat = visit_bilinear_stencil(PatternSink(Int[], Int[]), ast, W, 0, 0)
            got = visit_bilinear_stencil(CollectSink([]), ast, W, 0, 0)

            @test length(pat.I_vec) == length(unique(zip(pat.I_vec, pat.J_vec)))
            @test length(got.seen) == 2 * length(pat.I_vec)
            # and the halving it protects against shows up in the assembled matrix
            @test Matrix(assemble(form(W, W, (a, b) -> innerₕ(a, b) + innerₕ(a, b)))) ≈
                  2 .* Matrix(assemble(form(W, W, (a, b) -> innerₕ(a, b))))
        end

        # Only the pattern sink asks for de-duplication.
        @testset "Only the pattern sink de-duplicates" begin
            @test _sink_dedups(PatternSink(Int[], Int[]))
            @test !_sink_dedups(CollectSink([]))
        end

        @testset "Block offsets shift sink entries" begin
            ast = resolve_form_ast(form(W, W, (a, b) -> innerₕ(a, b)))
            base = visit_bilinear_stencil(CollectSink([]), ast, W, 0, 0).seen
            shifted = visit_bilinear_stencil(CollectSink([]), ast, W, 100, 7).seen
            @test length(base) == length(shifted)
            @test all(
                ((b, s),) -> s[1] == b[1] + 100 && s[2] == b[2] + 7, zip(base, shifted)
            )
        end

        @testset "Entry targets: guard, AbsoluteColumn" begin
            lin = LinearIndices(indices(Ω))
            I = CartesianIndex(1, 1)

            # inside: the row follows off_v, the column off_u, both shifted by the block
            @test _entry_target(lin, CartesianIndex(3, 3), (0, 0), (0, 0), 0, 0) ==
                  (lin[3, 3], lin[3, 3])
            @test _entry_target(lin, CartesianIndex(3, 3), (0, 0), (0, 0), 10, 5) ==
                  (lin[3, 3] + 10, lin[3, 3] + 5)

            # a row off the grid drops the entry, and so does a column off the grid
            @test _entry_target(lin, I, (0, 0), (-1, 0), 0, 0) == (0, 0)
            @test _entry_target(lin, I, (-1, 0), (0, 0), 0, 0) == (0, 0)

            # an interpolation entry names its source column outright rather than an offset
            @test _trial_column(lin, I, AbsoluteColumn(42)) == 42
            @test _entry_target(
                lin, CartesianIndex(2, 2), AbsoluteColumn(42), (0, 0), 0, 3
            ) == (lin[2, 2], 45)
        end
    end

    @testset "Interior/boundary peeling (#160)" begin
        # `visit_bilinear_stencil` splits the grid into an interior core (no bounds guard)
        # and a boundary shell (the old guarded path) whenever `_stencil_margin(term)` and
        # the grid size allow it. Every test below exists because the margin can exceed 1 --
        # a composed difference or a multi-cell shift -- and treating a 1-cell rim as always
        # safe for those would silently corrupt the rows nearest the boundary rather than
        # merely running slower.
        using Bramble:
                       _stencil_margin,
                       _peelable,
                       _interior_range,
                       _boundary_shell_slabs,
                       _visit_guarded_region!,
                       shift_op,
                       ShiftNode,
                       markers,
                       mesh

        # Not a hardcoded 1.
        @testset "_stencil_margin reads composed reach" begin
            u, v = TrialFunction{2}(), TestFunction{2}()
            @test _stencil_margin(resolve_ast(innerₕ(u, v))) == 0
            @test _stencil_margin(resolve_ast(innerₕ(D₋ₓ(u), v))) == 1
            @test _stencil_margin(resolve_ast(innerₕ(D₋ₓ(D₋ₓ(u)), v))) == 2
            @test _stencil_margin(resolve_ast(innerₕ(shift_op(u, 1, 3), v))) == 3
            # the reach comes from whichever side -- trial or test -- reaches further
            @test _stencil_margin(resolve_ast(innerₕ(u, D₋ₓ(v)))) == 1
            @test _stencil_margin(resolve_ast(innerₕ(D₋ₓ(u), D₋ₓ(D₋ₓ(v))))) == 2
            # a sum's margin is the widest of its terms
            @test _stencil_margin(resolve_ast(innerₕ(u, v) + innerₕ(D₋ₓ(u), D₋ₓ(v)))) == 1
        end

        # They cover the grid exactly once.
        @testset "Interior + boundary slabs partition" begin
            for D in (1, 2, 3), margin in (0, 1, 2)

                Ωd = domain(reduce(×, ntuple(_ -> interval(0.0, 1.0), D)))
                Ω = mesh(Ωd, ntuple(_ -> 7, D), ntuple(_ -> true, D))
                grid_inds = indices(Ω)
                ax = axes(grid_inds)
                @test _peelable(ax, margin)   # every axis here has 7 >= 2*margin points

                lin = LinearIndices(grid_inds)
                seen = Int[]
                for I in CartesianIndices(map(r -> _interior_range(r, margin), ax))
                    push!(seen, lin[I])
                end
                for slab in _boundary_shell_slabs(ax, margin), I in slab

                    push!(seen, lin[I])
                end
                # every grid point exactly once: no gap, no double-scatter
                @test sort(seen) == 1:length(grid_inds)
            end

            # A margin that cannot fit twice into the shortest axis must refuse to peel,
            # rather than let the low and high rim overlap and double-scatter a point.
            Ω = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (3, 7), (true, true))
            @test !_peelable(axes(indices(Ω)), 2)
        end

        # Agrees with the guarded fallback, entry for entry.
        @testset "Peeled traversal matches fallback" begin
            struct _MarginCollectSink
                seen::Vector{Tuple{Int, Int, Float64}}
            end
            Bramble._sink_entry!(s::_MarginCollectSink, row::Int, col::Int, w, ::Int) = (
                push!(s.seen, (row, col, Float64(w))); nothing)

            for D in (1, 2, 3)
                Ωd = domain(reduce(×, ntuple(_ -> interval(0.0, 1.0), D)))
                Ω = mesh(Ωd, ntuple(_ -> 6, D), ntuple(_ -> true, D))
                W = gridspace(Ω)
                u, v = TrialFunction{D}(), TestFunction{D}()

                terms = Any[resolve_ast(innerₕ(u, v)), resolve_ast(innerₕ(D₋ₓ(u), D₋ₓ(v)))]  # Any: each AST is its own node type
                D >= 2 && push!(terms, resolve_ast(inner₊(∇ₕ(u), ∇ₕ(v))))
                # margin 2: a composed difference and a multi-cell shift, neither a
                # single tap -- exactly the case a hardcoded 1-cell rim would get wrong.
                push!(terms, resolve_ast(innerₕ(D₋ₓ(D₋ₓ(u)), v)))
                push!(terms, resolve_ast(innerₕ(shift_op(u, 1, 2), v)))

                mesh_markers = markers(Ω)
                lin_indices = LinearIndices(indices(Ω))
                for ast in terms
                    peeled = visit_bilinear_stencil(
                        _MarginCollectSink(Tuple{Int, Int, Float64}[]), ast, W, 0, 0
                    ).seen
                    guarded = _MarginCollectSink(Tuple{Int, Int, Float64}[])
                    _visit_guarded_region!(
                        guarded, ast, W, mesh_markers, lin_indices, indices(Ω), 0, 0
                    )
                    @test sort(peeled) == sort(guarded.seen)
                end
            end
        end

        # Nonzero Dirichlet boundary values solve correctly under peeling.
        @testset "Peeling: nonzero Dirichlet values" begin
            # The peeled path only ever changes which points skip the bounds guard; every
            # point still gets visited exactly once (see above). This solves an actual
            # manufactured Poisson problem with *nonzero* boundary data through the
            # production `assemble`/`assemble!` pipeline -- which now always peels where
            # `_stencil_margin` and the grid allow it -- and checks the discrete solution
            # against the exact one, not merely that assembly runs.
            sol(x) = 1 + sin(pi * x[1]) * cos(pi * x[2])
            rhs(x) = 2 * pi^2 * sin(pi * x[1]) * cos(pi * x[2])

            # Non-uniform (`false, false`): mesh1d.jl's `_generate_random_points!` draws from
            # the global RNG, so an unseeded run's point placement differs run to run -- and
            # the norm bound below is a fixed threshold that an unlucky draw could miss.
            # Seeded so the test is reproducible, not just usually passing (see the identical
            # rationale in test/form/type_cached_assemble.jl).
            Random.seed!(20260912)
            Ωd = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
            Ω = mesh(Ωd, (48, 48), (false, false))
            W = gridspace(Ω)
            bcs = dirichlet_constraints(Ωd, :boundary => sol)

            a = form(W, W, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
            A = assemble(a; dirichlet = :boundary)
            fₕ = element(W)
            avgₕ!(fₕ, rhs)
            l = form(W, v -> innerₕ(fₕ, v))
            F = assemble(l; dirichlet = bcs)

            uₕ = element(W)
            uₕ .= A \ F
            # `sol` is nonzero along the whole boundary (`sin(pi*x[1])` vanishes there, but
            # the constant `1` does not), so this exercises exactly the near-boundary rows
            # the boundary shell (not the interior core) assembles.
            @test norm₁ₕ(uₕ .- Rₕ(W, sol)) < 6e-3
        end
    end

    @testset "Form construction" begin
        # `form` used to evaluate a sample stencil and bin the whole grid into a vector of
        # vectors before anything was assembled (9,271,600 B at 90,000 degrees of freedom).
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

    # The matrix-type seam: `allocate_system_matrix`, `assemble`,
    # `assemble!` and `assemble_parallel!` read the matrix type from
    # `matrix_type(backend(test_space(form)))` rather than hardcoding `SparseMatrixCSC`,
    # reaching storage through `_scatter_position`/`_scatter_add!` (bilinear_traversal.jl),
    # `_allocate_from_pattern` (bilinear_pattern.jl) and `_zero_stored!` (bilinear.jl), each
    # with a `SparseMatrixCSC` method and a generic `AbstractMatrix` fallback. A dense
    # `Matrix{Float64}` backend exercises that fallback -- the positive control proving the
    # seam is real -- and must assemble the exact same values as the default CSC backend,
    # with and without Dirichlet and `symmetrize!`, while the CSC path keeps its
    # zero-allocation refill.
    @testset "Dense backend" begin
        Sd = interval(0.0, 1.0) × interval(0.0, 1.0)
        Ωd_domain = domain(Sd, :dir => (x) -> x[1] ≈ 0.0)
        Ωc = mesh(Ωd_domain, (7, 6), (true, true))
        Ωd = mesh(Ωd_domain, (7, 6), (true, true); backend = backend(matrix_type = Matrix{Float64}))
        Wc, Wd = gridspace(Ωc), gridspace(Ωd)

        κ = x -> 1 + x[1] * x[2]
        f = (W) -> begin
            kh = Rₕ(W, κ)
            form(W, W, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(kh * u, v))
        end

        @testset "assemble agrees with CSC" begin
            Ac, Ad = assemble(f(Wc)), assemble(f(Wd))
            @test Ac isa SparseMatrixCSC
            @test Ad isa Matrix{Float64}
            @test isapprox(Matrix(Ac), Ad; atol = 1e-13)
            @test count(!iszero, Ad) == nnz(Ac)
        end

        @testset "Dirichlet and symmetrize agree with CSC" begin
            Acd = assemble(f(Wc); dirichlet = (:dir,))
            Add = assemble(f(Wd); dirichlet = (:dir,))
            @test isapprox(Matrix(Acd), Add; atol = 1e-13)

            Fc, Fd = ones(ndofs(Wc)), ones(ndofs(Wd))
            Bc, Bd = copy(Acd), copy(Add)
            symmetrize!(Bc, Fc, Wc, :dir)
            symmetrize!(Bd, Fd, Wd, :dir)
            @test isapprox(Matrix(Bc), Bd; atol = 1e-13)
            @test isapprox(Fc, Fd)
        end

        @testset "assemble! refill: CSC zero-allocation" begin
            Ac2, Ad2 = allocate_system_matrix(f(Wc)), allocate_system_matrix(f(Wd))
            @test Ac2 isa SparseMatrixCSC
            @test Ad2 isa Matrix{Float64}

            fc, fd = f(Wc), f(Wd)
            assemble!(Ac2, fc)
            assemble!(Ad2, fd)
            @test isapprox(Matrix(Ac2), Ad2; atol = 1e-13)

            assemble!(Ac2, fc)
            @test (@allocated assemble!(Ac2, fc)) == 0
        end
    end
end

# Linearity of assembly.
#
# A bilinear form is linear in each argument, and the assembled matrix inherits that: the
# matrix of a sum of terms is the sum of their matrices, and a scalar in front of a term
# scales that term's matrix alone. Stated on the assembler rather than on matrix-vector
# arithmetic, since `A * (αu + v) == αAu + Av` holds for any matrix at all and would say
# nothing about whether the right matrix was built.
#
# The grids are drawn by Supposition, so the property is checked on non-uniform partitions
# with no relation between the directions -- the case where a term picking up the wrong
# metric weight cannot cancel against another term's.
WITH_SLOW_TESTS && @testset "Assembly linearity (Supposition)" begin
    positive_h = Data.Floats{Float64}(;
        minimum = 0.01, maximum = 10.0, nans = false, infs = false
    )
    scalar = Data.Floats{Float64}(;
        minimum = -5.0, maximum = 5.0, nans = false, infs = false
    )

    # Absolute floor beside the relative one: a drawn partition can make an entry
    # analytically zero land at round-off.
    _agree(A, B) = isapprox(Matrix(A), Matrix(B); atol = 1e-10, rtol = 1e-10)

    @check function check_assembly_is_linear_in_terms(
            h = Data.Vectors(positive_h; min_size = 3, max_size = 10), α = scalar
    )
        pts = _nonuniform_points(h)
        Ωₕ = mesh(domain(interval(0.0, 1.0)), length(pts), false)
        set_points!(Ωₕ, pts)
        Wₕ = gridspace(Ωₕ)

        mass = assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v)))
        stiffness = assemble(form(Wₕ, Wₕ, (u, v) -> inner₊ₓ(D₋ₓ(u), D₋ₓ(v))))
        combined = assemble(
            form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + α * inner₊ₓ(D₋ₓ(u), D₋ₓ(v)))
        )

        return _agree(combined, mass + α * stiffness)
    end

    # The same statement for a linear form, where the source rather than the operator
    # carries the combination: `l` reads `α * u₁ + u₂` through the same evaluation path
    # that a single grid function takes.
    @check function check_linear_form_is_linear_in_source(
            h = Data.Vectors(positive_h; min_size = 3, max_size = 10), α = scalar
    )
        pts = _nonuniform_points(h)
        Ωₕ = mesh(domain(interval(0.0, 1.0)), length(pts), false)
        set_points!(Ωₕ, pts)
        Wₕ = gridspace(Ωₕ)

        u₁ = Rₕ(Wₕ, x -> sin(3x) + 1)
        u₂ = Rₕ(Wₕ, x -> x^2 - 2)
        w = α * u₁ + u₂

        b₁ = assemble(form(Wₕ, v -> innerₕ(u₁, v)))
        b₂ = assemble(form(Wₕ, v -> innerₕ(u₂, v)))
        b = assemble(form(Wₕ, v -> innerₕ(w, v)))

        return isapprox(b, α * b₁ + b₂; atol = 1e-10, rtol = 1e-10)
    end

    # Non-vacuous: mass and stiffness are different matrices, so the sum above is not the
    # same statement twice, and the scalar genuinely changes the result.
    @testset "Non-vacuous" begin
        Random.seed!(20260913)
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 12, false)
        Wₕ = gridspace(Ωₕ)

        mass = assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v)))
        stiffness = assemble(form(Wₕ, Wₕ, (u, v) -> inner₊ₓ(D₋ₓ(u), D₋ₓ(v))))
        @test !isapprox(Matrix(mass), Matrix(stiffness))

        combined = assemble(
            form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + 3.0 * inner₊ₓ(D₋ₓ(u), D₋ₓ(v)))
        )
        @test !isapprox(Matrix(combined), Matrix(mass + stiffness))
    end
end

@testset "bilinear: restricted sum replay width" begin
    # A restricted term fused into a sum writes a varying number of
    # entries per point, which the fixed-width diagonal replay must not accept.
    right = x -> x[1] > 0.8
    function replay_matches(W)
        a = form(W, W, (u, v) -> innerₕ(u, v) + Ref(2.0) * innerₕ(u, Bramble.restrict_to(:right, v)))
        R = assemble(form(W, W, (u, v) -> innerₕ(u, v))) +
            2.0 * assemble(form(W, W, (u, v) -> innerₕ(u, Bramble.restrict_to(:right, v))))
        A = assemble(a)
        ok = Matrix(A) ≈ Matrix(R)
        assemble!(A, a)
        assemble!(A, a)
        ok &= Matrix(A) ≈ Matrix(R)
        return ok, @allocated(assemble!(A, a))
    end
    Random.seed!(1234)
    Ω₁ = domain(interval(0.0, 1.0), :right => right)
    for _ in 1:50
        ok, b = replay_matches(gridspace(mesh(Ω₁, 11, false)))
        @test ok
        @test b == 0
    end
    Ω₂ = domain(interval(0.0, 1.0) × interval(0.0, 1.0), :right => right)
    for _ in 1:20
        ok, b = replay_matches(gridspace(mesh(Ω₂, (7, 9), (false, false))))
        @test ok
        @test b == 0
    end

    # Interior points 2:5 at width 1, except the last, which is one entry wider.
    grid_inds = CartesianIndices((1:6,))
    interior = CartesianIndices((2:5,))
    @test Bramble._uniform_interior_width(grid_inds, interior, [5, 1, 2, 3, 4, 6, 7], 1)
    @test !Bramble._uniform_interior_width(grid_inds, interior, [6, 1, 2, 3, 4, 7, 8], 1)
end

@testset "bilinear: matrix-free form call" begin
    # `a(u, v)` sums `vᵀ A u` over the stencil walk instead of
    # assembling `A`, on non-uniform meshes, with a composite space, a transposed pair, a
    # region restriction and a coefficient.
    Random.seed!(3263)
    S = interval(0.0, 1.0) × interval(0.0, 2.0)
    W = gridspace(mesh(domain(S, :left => x -> x[1] < 0.3), (9, 11), (false, true)))
    κ = Rₕ(W, x -> 1 + x[1]^2 + x[2])
    V = W × W
    cases = (
        form(W, W, (u, v) -> innerₕ(u, v) + inner₊(κ * ∇ₕ(u), ∇ₕ(v))),
        form(W, W, (u, v) -> innerₕ(D₋ₓ(u), D₋ᵧ(v)) + innerₕ(D₋ᵧ(u), D₋ₓ(v))),
        form(W, W, (u, v) -> innerₕ(u, v) + innerₕ(κ * u, restrict_to(:left, v))),
        form(V, V, (u, v) -> innerₕ(u(1), v(1)) + inner₊(∇ₕ(u(2)), ∇ₕ(v(2))) +
                             innerₕ(u(1), v(2)))
    )
    contract(a, u, v) = (a(u, v); @allocated a(u, v))
    for (k, a) in enumerate(cases)
        u = element(trial_space(a), randn(ndofs(trial_space(a))))
        v = element(test_space(a), randn(ndofs(test_space(a))))
        ref = dot(parent(v), assemble(a) * parent(u))
        @test a(u, v) ≈ ref rtol = 1e-12
        @test a(u, v) ≈ ref rtol = 1e-12        # a second call does not accumulate
        @test a(parent(u), parent(v)) ≈ ref rtol = 1e-12
        @test a(u, v) isa Float64
        k == 3 || @test contract(a, u, v) <= 64   # the call's one accumulator cell
    end

    # and that cell is all it allocates, at any grid size
    bytes = map((101, 10001)) do n
        W1 = gridspace(mesh(domain(interval(0.0, 1.0)), n, false))
        a1 = form(W1, W1, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
        contract(a1, element(W1, randn(n)), element(W1, randn(n)))
    end
    @test bytes[1] == bytes[2]

    # element types promote: a `Float32` argument, and a rectangular form
    a = cases[1]
    u32 = element(W, randn(Float32, ndofs(W)))
    v = element(W, randn(ndofs(W)))
    @test a(u32, v) ≈ dot(parent(v), assemble(a) * parent(u32)) rtol = 1e-12
    @test a(u32, element(W, Float32.(parent(v)))) isa Float64
    Wc = gridspace(mesh(domain(interval(0.0, 1.0)), 7, false))
    Wf = gridspace(mesh(domain(interval(0.0, 1.0)), 13, false))
    r = form(Wc, Wf, (u, v) -> innerₕ(πₕ(u), v))
    uc, vf = Rₕ(Wc, x -> sin(3x[1])), Rₕ(Wf, x -> x[1] + 1)
    @test r(uc, vf) ≈ dot(parent(vf), assemble(r) * parent(uc)) rtol = 1e-12

    # views are read as themselves, not as the array behind them
    A = assemble(a)
    nW = ndofs(W)
    wr = @view randn(nW)[end:-1:1]
    wo = view(randn(nW + 3), 2:(nW + 1))
    w = parent(v)
    @test a(w, wr) ≈ dot(wr, A * w) rtol = 1e-12
    @test a(wr, w) ≈ dot(w, A * wr) rtol = 1e-12
    @test a(w, wo) ≈ dot(wo, A * w) rtol = 1e-12
    @test a(wo, wr) ≈ dot(wr, A * wo) rtol = 1e-12

    # the walk indexes from 1, so other axes are refused rather than misread
    @test_throws ArgumentError a(w, view(randn(nW + 1), Base.IdentityUnitRange(2:(nW + 1))))
    @test_throws DimensionMismatch a(zeros(ndofs(W) - 1), v)
    @test_throws DimensionMismatch a(v, zeros(ndofs(W) + 1))

    WITH_AD_TESTS && @testset "Dual arguments (#326)" begin
        u = randn(ndofs(W))
        vv = parent(v)
        g = ForwardDiff.gradient(w -> a(w, vv), u)
        @test g ≈ transpose(assemble(a)) * vv rtol = 1e-12
    end
end

@testset "pair walk, unequal block counts" begin
    # `_pair_plan` pairs summands by type, and a transposed pair resolves to as many blocks as
    # its first term, so the fallback that walks each term alone when the counts differ is
    # reached here directly, with two terms of 2 and 1 blocks. Units are visited in the order
    # the form's own recording stores them: the first term's blocks, then the second's.
    Random.seed!(5150)
    Ω = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 7), (false, false))
    W = gridspace(Ω)
    n = ndofs(W)
    V = gridspace(Ω, Val(2))
    a = form(V, V, (u, v) -> innerₕ(u, v) + 100 * innerₕ(u(1), v(2)))
    t1, t2 = Bramble._summands(a.ast)
    leaves = leaf_spaces_offsets(V)
    @test length(blocks(Bramble._bare_product(t1), leaves, leaves)) == 2
    @test length(blocks(Bramble._bare_product(t2), leaves, leaves)) == 1

    calls = Tuple{Int, Int, Int, Int, Int}[]
    Bramble._foreach_pair_block_unit(
        (_, _, ro, co, dr, dc, half) -> (push!(calls, (ro, co, dr, dc, half)); nothing),
        t1, t2, leaves, leaves)
    # (row_offset, col_offset): the two diagonal blocks, then test leaf 2 against trial leaf 1
    @test calls == [(0, 0, 0, 0, -1), (n, n, 0, 0, -1), (n, 0, 0, 0, -1)]

    A = assemble(a)                   # the serial fill records one segment per unit
    @test length(a.cache.segments) == 3
    fill!(nonzeros(A), 0.0)
    next = Bramble._replay_pair_blocks!(
        Bramble._SerialReplay(), A, t1, t2, leaves, leaves, a.cache.segments, 0, true)
    @test next == 3
    H = Matrix(Diagonal(collect(weights(W, Innerh()))))
    M = Matrix(A)
    blk(i, j) = M[((i - 1) * n + 1):(i * n), ((j - 1) * n + 1):(j * n)]
    @test blk(1, 1) ≈ H
    @test blk(2, 2) ≈ H
    @test blk(2, 1) ≈ 100 * H
    @test iszero(blk(1, 2))
end

@testset "zeroing a non-CSC sparse matrix" begin
    # `assemble!` zeroes the stored values of any `AbstractSparseMatrix` before a refill,
    # never `fill!(A, 0)` over every (i, j); a `FixedSparseCSC` is one such host type.
    A = sparse([1, 2, 3, 1], [1, 2, 3, 3], [1.0, 2.0, 3.0, 4.0])
    F = SparseArrays.FixedSparseCSC(A)
    @test !(F isa SparseMatrixCSC)
    @test Bramble._zero_stored!(F) === F
    @test nnz(F) == 4
    @test all(iszero, nonzeros(F))
    @test rowvals(F) == rowvals(A)
end

@testset "foreign tree via the deprecated keyword" begin
    # `assemble!(A, a; ast = other)` with a tree of another type records and fills, then
    # stores nothing: the cache's AST type is `a.ast`'s, and `a`'s own recording stays valid.
    Random.seed!(105)
    W = gridspace(mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 6), (false, false)))
    H = Matrix(Diagonal(collect(weights(W, Innerh()))))
    a = form(W, W, (u, v) -> innerₕ(u, v))
    alt = form(W, W, (u, v) -> 2.0 * innerₕ(u, v))
    @test typeof(alt.ast) != typeof(a.ast)
    A = assemble(a)
    B = copy(A)
    @test_deprecated r"`ast` keyword" assemble!(B, a; ast = alt.ast)
    @test Matrix(B) ≈ 2 * H
    @test a.cache.valid && a.cache.A_id == objectid(A) && a.cache.ast === a.ast
    fill!(nonzeros(A), NaN)
    assemble!(A, a)
    @test Matrix(A) ≈ H
end

struct _PlainSink end

@testset "bilinear: traversal and policy traits" begin
    Random.seed!(7)
    W = gridspace(mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 4), (false, false)))
    A = assemble(form(W, W, (u, v) -> innerₕ(u, v)))
    lin = LinearIndices(indices(mesh(W)))
    I = CartesianIndex(1, 1)

    # an absolute slot is in bounds wherever it points; an offset slot is checked
    @test Bramble._trial_inbounds(lin, I, AbsoluteColumn(10_000))
    @test Bramble._test_inbounds(lin, I, Bramble.AbsoluteRow(10_000))
    @test !Bramble._trial_inbounds(lin, I, (-1, 0))
    @test !Bramble._test_inbounds(lin, I, (0, -1))

    # replay sinks read positions, not coordinates; any other sink is given coordinates,
    # its slot is 0, and it keeps duplicates
    @test Bramble._sink_needs_coordinates(_PlainSink())
    @test Bramble._sink_point!(_PlainSink(), 3, I) == 0
    @test !_sink_dedups(_PlainSink())
    @test _sink_dedups(PatternSink(Int[], Int[]))
    @test !Bramble._sink_needs_coordinates(Bramble.ReplaySink(A, Int[], Int[], 1.0))
    @test !Bramble._sink_needs_coordinates(
        Bramble._PairReplaySink(A, Int[], Int[], Int[], 1.0, 1.0, 0))
    @test !Bramble._sink_needs_coordinates(
        Bramble.DiagonalReplaySink(A, CartesianIndices((1:0,)), Int[], Int[], 1, 1.0))
    @test !Bramble._sink_needs_coordinates(Bramble._StrideReplaySink(A, Int[], Int[], 0, 1.0))
    # a pattern sink records each coordinate pair it is handed
    ps = PatternSink(Int[], Int[])
    _sink_entry!(ps, 4, 2, 1.0, 0)
    @test (ps.I_vec, ps.J_vec) == ([4], [2])

    # the policy a forced-threaded sweep runs under, and which policies replay
    @test Bramble._coerce_serial_to_threaded(Bramble.GpuKernel()) === Bramble.CpuThreaded()
    @test Bramble._threaded_replay_policy(Bramble.CpuThreaded())
    @test Bramble._threaded_replay_policy(Bramble.CpuSerial()) == false
    # only 1D records diagonal segments
    @test Bramble._diagonal_replay(Val(1))
    @test !Bramble._diagonal_replay(Val(2))
    # a composite on either side assembles block by block
    V = gridspace(mesh(W), Val(2))
    @test !Bramble._is_block_pair(W, W)
    @test Bramble._is_block_pair(V, W)
    @test Bramble._is_block_pair(W, V)
    @test Bramble._is_block_pair(V, V)
end

@testset "symbolic and source-only leaves" begin
    u, v = TrialFunction{2}(), TestFunction{2}()
    ui, vi = Bramble.IndexedTrialFunction{2}(1), Bramble.IndexedTestFunction{2}(2)
    sf = Bramble.source_function(x -> x[1], Val(2))
    sv = Bramble.SourceVector{2, Vector{Float64}}([1.0])
    sc = Bramble.SourceConstant{2, Float64}(2.0)
    δ = Bramble.dirac((0.5, 0.5))
    # every leaf is symbolic; only the sources are source-only
    for (op, src) in ((u, false), (v, false), (ui, false), (vi, false), (sf, true),
        (sv, true), (sc, true), (δ, true))
        @test Bramble.is_symbolic(op)
        @test Bramble._is_source_only(op) == src
    end
    # a product is symbolic and never a bare source, whichever kind it is
    bp = innerₕ(u, v)
    lp = innerₕ(sf, v)
    @test bp isa Bramble.BilinearProduct && lp isa Bramble.LinearProduct
    for p in (bp, lp)
        @test Bramble.is_symbolic(p)
        @test !Bramble._is_source_only(p)
    end
    @test Bramble._is_source_only(sf + sv)
    @test !Bramble._is_source_only(sf + u)
    @test !Bramble._is_source_only(Bramble.IdentityOperator(gridspace(mesh(
        domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (3, 3), (true, true)))))
end

@testset "source stencils, non-uniform mesh" begin
    # A point source's strength is a number, a `Ref` or a thunk, read at each fill; its
    # load goes to the two nodes around the point with the linear interpolation weights.
    Random.seed!(226)
    W = gridspace(mesh(domain(interval(0.0, 1.0)), 11, false))
    x = points(mesh(W))
    x0 = (x[4] + 2 * x[5]) / 3
    expected = zeros(ndofs(W))
    expected[4] = 2.5 * (x[5] - x0) / (x[5] - x[4])
    expected[5] = 2.5 * (x0 - x[4]) / (x[5] - x[4])
    for strength in (2.5, Ref(2.5), () -> 2.5)
        @test assemble(form(W, v -> innerₕ(dirac(x0, strength), v))) ≈ expected
    end
    # several points with strengths of mixed kinds
    x1 = (3 * x[8] + x[9]) / 4
    b = assemble(form(W, v -> innerₕ(dirac([(x0,), (x1,)], Any[Ref(2.5), () -> 4.0]), v)))
    @test b[1:7] ≈ expected[1:7]
    @test b[8] ≈ 4.0 * (x[9] - x1) / (x[9] - x[8])
    @test b[9] ≈ 4.0 * (x1 - x[8]) / (x[9] - x[8])
    @test sum(b) ≈ 6.5

    # a composite linear form contracts summand by summand, each on its own block; the
    # oracle is the nodal values times the inner-product weights
    V = gridspace(mesh(W), Val(2))
    l = form(V, v -> innerₕ(y -> 1 + y[1], v(1)) + innerₕ(y -> 100.0, v(2)))
    w = collect(weights(W, Innerh()))
    vh = Rₕ(V, (y -> sin(3 * y[1]), y -> 1 + y[1]^2))
    v1, v2 = parent(components(vh)[1]), parent(components(vh)[2])
    @test l(vh) ≈ sum(w .* (1 .+ x) .* v1) + sum(w .* 100.0 .* v2)
end

end # module FormBilinearTests
