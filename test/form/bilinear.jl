using Test
using Bramble
using ForwardDiff
using LinearAlgebra: Diagonal, I
using SparseArrays: sparse, nnz, nonzeros
using Bramble: BilinearForm, form, assemble, assemble!, assemble_parallel!, trial_space,
               test_space, resolve_form_ast, allocate_system_matrix, ndofs, values,
               Innerh, Innerplus, block_of, trial_component_or_nothing,
               test_component_or_nothing, Block, blocks, leaf_spaces_offsets

# Assembling the matrix of a bilinear form.
#
# The convention throughout: `a(u, v) = vᵀ A u`, so a row of `A` is indexed by the test
# function and a column by the trial function. Every check below is written against a matrix
# built by code that shares nothing with `local_stencil`: the operator's own sparse form and
# the inner product's own weights, so the two agree or one of them is wrong.

@testset "Bilinear forms" begin
    S = interval(0.0, 1.0) × interval(0.0, 1.0)
    Ωₕ = mesh(domain(S, :walls => get_boundary_symbols(S)), (9, 7), (true, true))
    Wₕ = gridspace(Ωₕ)
    n = ndofs(Wₕ)

    H = Matrix(Diagonal(collect(weights(Wₕ, Innerh()))))
    Hx = Matrix(Diagonal(collect(weights(Wₕ, Innerplus(), 1))))
    Dx = Matrix(D₋ₓ(Wₕ))
    Mx = Matrix(M₋ₓ(Wₕ))
    Idm = Matrix(1.0I, n, n)

    @testset "Matrix expression equivalence" begin
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

    @testset "Entry point agreement" begin
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

    @testset "dirichlet_components restriction" begin
        # The motivating case: a Stokes-style system where leaf 1 ("velocity") gets a
        # boundary condition and leaf 2 ("pressure") stays completely free. Before
        # `dirichlet_components` existed, `dirichlet_labels` bound to every leaf sharing the
        # named marker: there was no way to say "this leaf only" through `assemble`/
        # `assemble!` at all.
        Vₕ = gridspace(Ωₕ, Val(2))
        a = form(Vₕ, Vₕ, (u, v) -> innerₕ(u(1), v(1)) + innerₕ(u(2), v(2)))
        marked = index_in_marker(Ωₕ, :walls)

        A = assemble(a; dirichlet_labels = :walls, dirichlet_components = 1)
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
        assemble!(A2, a; dirichlet_labels = :walls, dirichlet_components = 1)
        @test Matrix(A2) ≈ Matrix(A)

        # without dirichlet_components, the same labels bind to every leaf that has the
        # marker (this is the pre-existing, still-default behaviour, confirmed unchanged).
        Aboth = assemble(a; dirichlet_labels = :walls)
        blk2(i, j) = Matrix(Aboth)[((i - 1) * n + 1):(i * n), ((j - 1) * n + 1):(j * n)]
        for i in 1:n
            if marked[i]
                @test blk2(2, 2)[i, i] ≈ 1.0    # leaf 2 pinned too, unlike above
            end
        end
    end

    @testset "Dirichlet labels pin the TEST space's rows, not the trial space's (#48)" begin
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

        trial = W1 × W2   # leaf 1: size n1 (offset 0), leaf 2: size n2 (offset n1)
        test = W2 × W1   # leaf 1: size n2 (offset 0), leaf 2: size n1 (offset n2)

        # Cross terms, so each pairing is same-size (required by `_check_block_meshes`) and
        # lands off the "obvious" diagonal: trial leaf 1 (W1, n1) pairs with test leaf 2
        # (W1, n1); trial leaf 2 (W2, n2) pairs with test leaf 1 (W2, n2).
        a = form(trial, test, (u, v) -> innerₕ(u(1), v(2)) + innerₕ(u(2), v(1)))
        @test trial_space(a) !== test_space(a)

        N = n1 + n2
        @test size(assemble(a)) == (N, N)   # square: total ndofs agree, just laid out differently

        # `:boundary` is reserved and auto-computed on every mesh: index 1 and index n. So
        # each leaf contributes exactly two marked rows/columns, with no domain setup needed.
        Abc = assemble(a; dirichlet_labels = :boundary)

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
        assemble!(A2, a; dirichlet_labels = :boundary)
        @test Matrix(A2) ≈ Matrix(Abc)

        # `dirichlet_components` on an asymmetric form restricts by TEST leaf, since that is
        # what the rows mean: component 1 is test's leaf 1 (W2, offset 0, size n2).
        A1 = Matrix(assemble(a; dirichlet_labels = :boundary, dirichlet_components = 1))
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
        # one thread actually available (gpena/Bramble.jl#84): on one thread the colours
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
        # assemble!/assemble no longer hardcode parallel (the asymmetry this closed:
        # assemble(a::BilinearForm) used to call assemble_parallel! unconditionally, the
        # opposite default from LinearForm's serial-by-default assemble). Both now read
        # form.trial_space's execution_policy, defaulting to Serial() like the vector form.
        @test execution_policy(Wₕ) isa Serial
        Ω_par = mesh(domain(S, :walls => get_boundary_symbols(S)), (9, 7), (true, true);
            backend = backend(policy = Parallel()))
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
        @test_throws ArgumentError assemble(form(nested, nested,
            (u, v) -> innerₕ(u(1), v(5))))
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

    @testset "Block resolution (#49)" begin
        # `blocks(term, trial_leaves, test_leaves)` is the one place the trial/test row/
        # column asymmetry is resolved, and is testable directly against `leaf_spaces_offsets`
        # without assembling a matrix -- unlike before it existed, when the only way to see
        # a wrong offset was in an assembled matrix's numbers (which is exactly how #48 was a
        # live bug for a while). Asymmetric leaf sizes and reversed order, as in the #48 test
        # above, so a row/column offset mix-up lands on the wrong number rather than the same
        # one by coincidence.
        n1, n2 = 5, 7
        W1 = gridspace(mesh(domain(interval(0.0, 1.0)), n1, true))
        W2 = gridspace(mesh(domain(interval(0.0, 1.0)), n2, true))

        trial = W1 × W2   # leaf 1: W1, offset 0.  leaf 2: W2, offset n1
        test = W2 × W1    # leaf 1: W2, offset 0.  leaf 2: W1, offset n2

        trial_leaves = leaf_spaces_offsets(trial)
        test_leaves = leaf_spaces_offsets(test)

        u1 = Bramble.TrialFunction{1}()
        v1 = Bramble.TestFunction{1}()

        @testset "Named block: row from test leaf, column from trial leaf" begin
            bs = blocks(innerₕ(u1(1), v1(2)), trial_leaves, test_leaves)
            @test length(bs) == 1
            blk = only(bs)
            @test blk isa Block
            @test blk.trial_leaf === W1        # trial leaf 1
            @test blk.test_leaf === W1         # test leaf 2 is also W1
            @test blk.row_offset == n2          # test leaf 2's own offset, not trial's
            @test blk.col_offset == 0           # trial leaf 1's own offset
        end

        @testset "Unrouted term: one Block per diagonal leaf pair" begin
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

    @testset "Matrix differentiation" begin
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
            sum(assemble(form(Vₕ, Vₕ,
                (u, v) -> innerₕ(Bramble.element(Wₕ, w) * u(1), v(1)) + innerₕ(u(2), v(2)))))
        end ≈ diag(H)
    end

    @testset "In-place reassembly" begin
        # The pattern is the expensive half and does not change between assemblies, so the
        # intended shape of a loop is `assemble` once and `assemble!` after. This pins that
        # the second path agrees with the first and costs nothing.
        cₕ = Rₕ(Wₕ, x -> 1.0)
        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(cₕ * u, v))

        A = assemble(a)
        first_sum = sum(A)
        assemble!(A, a)
        @test sum(A) ≈ first_sum                     # idempotent, so it overwrites

        # a coefficient written through is seen, and the pattern is untouched by it
        nnz_before = nnz(A)
        Rₕ!(cₕ, x -> 3.0)
        assemble!(A, a)
        @test sum(A) ≈ 3 * first_sum
        @test nnz(A) == nnz_before

        # assemble! uses the pre-resolved ast stored in the form and allocates 0 bytes.
        function _loop_bytes(A, a)
            assemble!(A, a)
            return @allocated assemble!(A, a)
        end
        @test _loop_bytes(A, a) == 0

        # Coefficients with operators pre-resolve at form construction time and also allocate 0 bytes during assembly
        dcₕ = D₋ₓ(cₕ)
        aop = form(Wₕ, Wₕ, (u, v) -> innerₕ(dcₕ * u, v))
        Aop = assemble(aop)
        @test _loop_bytes(Aop, aop) == 0

        ainline = form(Wₕ, Wₕ, (u, v) -> innerₕ(D₋ₓ(cₕ) * u, v))
        Ain = assemble(ainline)
        @test _loop_bytes(Ain, ainline) == 0
        @test Matrix(Ain) ≈ Matrix(Aop)              # and the two agree
    end

    @testset "Composite in-place reassembly (zero allocations)" begin
        # `_loop_bytes` above only exercises the scalar core. The block-routing core (going
        # through `blocks` -- see #49) had no equivalent guard, so a routing change could
        # reintroduce an allocation (e.g. from building an intermediate `Block` per term)
        # with nothing in the suite to catch it.
        Vₕ = gridspace(Ωₕ, Val(2))
        function _loop_bytes(A, a)
            assemble!(A, a)
            return @allocated assemble!(A, a)
        end

        a_diag = form(Vₕ, Vₕ, (u, v) -> innerₕ(u, v))          # blk === nothing path
        @test _loop_bytes(assemble(a_diag), a_diag) == 0

        a_off = form(Vₕ, Vₕ, (u, v) -> innerₕ(u(1), v(2)))     # named-block path
        @test _loop_bytes(assemble(a_off), a_off) == 0

        a_mixed = form(Vₕ, Vₕ, (u, v) -> innerₕ(u, v) + innerₕ(u(1), v(2))) # both, one term each
        @test _loop_bytes(assemble(a_mixed), a_mixed) == 0
    end

    @testset "Cached nzval positions (#26)" begin
        # `assemble!` used to search for every scattered entry's nzval position on every
        # call. It now records that search's result the first time a given matrix is
        # assembled into and replays it thereafter -- these exercise both paths and the
        # points where they meet (a matrix swap, a changed `ast`), not just a single
        # before/after allocation count.

        @testset "Replay matches search across many repeated calls" begin
            a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊ₓ(D₋ₓ(u), D₋ₓ(v)))
            A = assemble(a)                 # record, inside assemble's own call
            reference = copy(A.nzval)
            for _ in 1:5                    # several replay calls, not just one
                assemble!(A, a)
                @test A.nzval ≈ reference
            end
        end

        @testset "Live coefficients still update under replay" begin
            cₕ = Rₕ(Wₕ, x -> 1.0)
            a = form(Wₕ, Wₕ, (u, v) -> innerₕ(cₕ * u, v))
            A = assemble(a)                 # record
            s1 = sum(A)
            for factor in (3.0, -2.0, 5.0)
                Rₕ!(cₕ, x -> factor)
                assemble!(A, a)              # replay, each time with a different live value
                @test sum(A) ≈ factor * s1
            end
        end

        @testset "Dirichlet labels still applied after a cached replay" begin
            a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊ₓ(D₋ₓ(u), D₋ₓ(v)))
            A = assemble(a)                             # record, unconstrained
            assemble!(A, a)                             # replay, unconstrained
            assemble!(A, a; dirichlet_labels = :walls)   # replay core, then Dirichlet applied
            marked = index_in_marker(Ωₕ, :walls)
            for i in 1:n
                marked[i] || continue
                @test A[i, i] ≈ 1.0
                @test count(!iszero, A[i, :]) == 1
            end
        end

        @testset "Switching matrices rebuilds rather than corrupting" begin
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

        @testset "Composite: diagonal, off-diagonal, mixed, and nested all replay correctly" begin
            Vₕ = gridspace(Ωₕ, Val(2))
            for g in ((u, v) -> innerₕ(u, v),
                (u, v) -> innerₕ(u(1), v(2)),
                (u, v) -> innerₕ(u, v) + innerₕ(u(1), v(2)),
                (u, v) -> inner₊ₓ(D₋ₓ(u(1)), D₋ₓ(v(1))) + innerₕ(u(2), v(1)))
                a = form(Vₕ, Vₕ, g)
                A = assemble(a)
                reference = copy(A.nzval)
                for _ in 1:3
                    assemble!(A, a)
                    @test A.nzval ≈ reference
                end
            end

            # more than two segments to replay, in order (#64's own nesting shape)
            nested = Bramble.CompositeGridSpace((
                gridspace(Ωₕ, Val(2)), gridspace(Ωₕ, Val(2))))
            a = form(nested, nested, (u, v) -> innerₕ(u(2), v(4)) + innerₕ(u, v))
            A = assemble(a)
            reference = copy(A.nzval)
            for _ in 1:3
                assemble!(A, a)
                @test A.nzval ≈ reference
            end
        end

        @testset "A different ast forces a rebuild rather than a stale replay" begin
            a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v))
            A = assemble(a)                                   # records for a's own ast
            alt = form(Wₕ, Wₕ, (u, v) -> 2.0 * innerₕ(u, v))  # same reach, different ast object
            assemble!(A, a; ast = resolve_form_ast(alt))
            @test sum(A) ≈ 2 * sum(H)
            assemble!(A, a)                                   # back to a's own ast: rebuilds again
            @test Matrix(A) ≈ H
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
end
