module FormCrossMeshBlocksTests

using Test
using Bramble
using LinearAlgebra: Diagonal
using ..TestUtils: alloc_ok, alloc_test, WITH_3D_TESTS
using Bramble:
               CompositeGridSpace,
               CpuThreaded,
               backend,
               boundary_indices,
               indices,
               πₕ,
               form,
               assemble,
               assemble!,
               assemble_parallel!,
               allocate_system_matrix,
               D₋ₓ,
               Mₓ,
               inner₊ₓ,
               weights,
               Innerh

# Coupling two leaves of a composite space whose meshes differ in size.
#
# A coupled bilinear term is assembled by walking the *test* leaf's grid and reading the
# trial column out of that same index space, offset into the trial leaf's block:
# `lin_indices[I + off_u] + col_offset`. That only means something when the two leaves share
# an index space, which they always did until heterogeneous composite spaces arrived (a
# space built by repeating one space hands every leaf the same mesh object).
#
# On leaves of different sizes it failed in whichever direction the sizes ran, and one of the
# two failures was silent:
#   innerₕ(u(2), v(1))  small trial, big test  → ArgumentError from inside sparse!, naming
#                                                column indices rather than the real problem
#   innerₕ(u(1), v(2))  big trial, small test  → assembled, in-range but WRONG columns
#
# There is no correspondence between an index on one mesh and an index on the other, so the
# term has no assembly until something says how to map between them, such as an interpolation
# operator `πₕ(u)`. Without that mapping it is refused at every entry point.

@testset "Cross-mesh blocks" begin
    Ωbig = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (6, 6), (true, true))
    # the small leaf is non-uniform, so its weights differ from cell to cell and from the
    # big leaf's: a block on the wrong leaf's weights, or on none, cannot match by accident
    Ωsmall = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (3, 3), (false, false))
    Wbig, Wsmall = gridspace(Ωbig), gridspace(Ωsmall)
    Vh = CompositeGridSpace((Wbig, Wsmall))

    @testset "Entry point refusal" begin
        # the direction that used to throw from inside `sparse!`, and the one that used to
        # assemble silently wrong columns; the second is the reason this is a test and not
        # just a nicer error message
        for g in ((u, v) -> innerₕ(u(2), v(1)), (u, v) -> innerₕ(u(1), v(2)))
            a = form(Vh, Vh, g)
            @test_throws ArgumentError allocate_system_matrix(a)
            @test_throws ArgumentError assemble(a)
        end

        # and through the refill entry points, which skip the pattern step entirely
        a_ok = form(Vh, Vh, (u, v) -> innerₕ(u(1), v(1)) + innerₕ(u(2), v(2)))
        A = assemble(a_ok)
        a_bad = form(Vh, Vh, (u, v) -> innerₕ(u(2), v(1)))
        @test_throws ArgumentError assemble!(A, a_bad)
        @test_throws ArgumentError assemble_parallel!(A, a_bad)
    end

    @testset "Error message precision" begin
        a = form(Vh, Vh, (u, v) -> innerₕ(u(2), v(1)))
        msg = try
            assemble(a)
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("different meshes", msg)
        @test occursin("(6, 6)", msg)          # the two sizes, so the reader can see which
        @test occursin("(3, 3)", msg)
        @test occursin("πₕ", msg)              # and what to do on the linear side instead
    end

    @testset "Operator wrapper check" begin
        for g in (
            (u, v) -> inner₊ₓ(D₋ₓ(u(2)), D₋ₓ(v(1))),
            (u, v) -> innerₕ(Mₓ(u(1)), v(2)),
            (u, v) -> innerₕ(u(1), v(1)) + innerₕ(u(2), v(1))
        )   # one good term, one bad
            @test_throws ArgumentError assemble(form(Vh, Vh, g))
        end
    end

    @testset "Permitted configurations" begin
        # the mass matrix of each leaf, from its own weights: the blocks every term below
        # must reproduce, which a size or a nonzero count cannot tell from a wrong leaf's
        H(W) = Matrix(Diagonal(collect(weights(W, Innerh()))))
        Hdiag = cat(H(Wbig), H(Wsmall); dims = (1, 2))

        # diagonal blocks on the same heterogeneous space: each leaf couples to itself, so
        # the index spaces match by construction
        a_diag = form(Vh, Vh, (u, v) -> innerₕ(u(1), v(1)) + innerₕ(u(2), v(2)))
        A = assemble(a_diag)
        @test size(A) == (ndofs(Vh), ndofs(Vh))
        @test Matrix(A) ≈ Hdiag
        # the refills write in place and reproduce the same values; the threaded one replays
        # through its own mode, so its values are pinned separately
        @test assemble!(A, a_diag) === A
        @test Matrix(A) ≈ Hdiag
        @test assemble_parallel!(A, a_diag) === A
        @test Matrix(A) ≈ Hdiag

        # a term naming no component at all, on a heterogeneous space: it reaches every
        # diagonal block, which is the `blk === nothing` branch of the same routing
        a_broadcast = form(Vh, Vh, (u, v) -> innerₕ(u, v))
        @test Matrix(assemble(a_broadcast)) ≈ Hdiag

        # cross blocks on a HOMOGENEOUS composite space are untouched: every leaf shares one
        # mesh, so the index arithmetic was always well defined there
        Vhom = gridspace(Ωbig, Val(2))
        Ahom = assemble(form(Vhom, Vhom, (u, v) -> innerₕ(u(2), v(1))))
        @test size(Ahom) == (2 * ndofs(Wbig), 2 * ndofs(Wbig))
        @test count(!iszero, Ahom) == ndofs(Wbig)
        # the block really is the off-diagonal one, not the diagonal
        @test all(iszero, Ahom[1:ndofs(Wbig), 1:ndofs(Wbig)])
        @test !all(iszero, Ahom[1:ndofs(Wbig), (ndofs(Wbig) + 1):end])

        # two distinct mesh objects of the same size still couple: the check is on the index
        # space, which is what the offset arithmetic actually needs
        Ωtwin = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (6, 6), (true, true))
        Vtwin = CompositeGridSpace((Wbig, gridspace(Ωtwin)))
        n = ndofs(Wbig)
        At = Matrix(assemble(form(Vtwin, Vtwin, (u, v) -> innerₕ(u(2), v(1)))))
        # rows are the test leaf (1), columns the trial leaf (2); both leaves are uniform
        # 6×6, so the block is the same whichever leaf's weights it is built from
        @test At[1:n, (n + 1):(2n)] ≈ H(Wbig)
        @test count(!iszero, At) == n
    end
end

# A scalar form between two spaces on different meshes whose summands interpolate on
# different sides, `⟨πₕu, v⟩ + ⟨u, πₕv⟩` (gpena/Bramble.jl#398). Each summand walks its own
# leaf: the test leaf for a trial-side `πₕ`, the trial leaf for a test-side one. The sum was
# walked on one leaf for every summand, so the trial-side term wrote rows past the test
# space's count (`ArgumentError` from `sparse!`), though each term assembled alone.
@testset "Trial- and test-side πₕ summed" begin
    f1 = (u, v) -> innerₕ(πₕ(u), v)
    f2 = (u, v) -> innerₕ(u, πₕ(v))
    # a difference operator beside each interpolation, so the sum composes two operators
    g1 = (u, v) -> inner₊ₓ(D₋ₓ(πₕ(u)), D₋ₓ(v))
    g2 = (u, v) -> innerₕ(Mₓ(u), πₕ(v))
    unit = interval(0.0, 1.0)
    cases = Any[
        (domain(unit), 9, 5, true),
        (domain(unit), 13, 8, false),
        (domain(unit × unit), (9, 5), (5, 3), true),
        (domain(unit × unit), (9, 6), (5, 4), false)
    ]
    # 3D only outside the unit group (#650); it runs in `slow`
    WITH_3D_TESTS && push!(cases, (domain(unit × unit × unit), (5, 4, 5), (3, 3, 2), false))

    @testset "$(length(nf))D, uniform = $unif" for (Ω, nf, nc, unif) in cases
        D = length(nf)
        flags = ntuple(_ -> unif, D)
        Wf = gridspace(mesh(Ω, nf, flags))
        Wc = gridspace(mesh(Ω, nc, flags))
        # fine trial and coarse test, then the other way round
        for (Wu, Wv) in ((Wf, Wc), (Wc, Wf)), (a1, a2) in ((f1, f2), (g1, g2))

            A1 = Matrix(assemble(form(Wu, Wv, a1)))
            A2 = Matrix(assemble(form(Wu, Wv, a2)))
            # each term alone is nonzero, so a sum that dropped one cannot match
            @test !iszero(A1) && !iszero(A2)
            a = form(Wu, Wv, (u, v) -> a1(u, v) + a2(u, v))
            A = assemble(a)
            @test size(A) == (ndofs(Wv), ndofs(Wu))
            @test Matrix(A) ≈ A1 + A2
            # the composite path walks a leaf per block: the same matrix, entry for entry
            Vu, Vv = CompositeGridSpace((Wu,)), CompositeGridSpace((Wv,))
            @test Matrix(A) == Matrix(assemble(form(Vu, Vv, (u, v) -> a1(u, v) + a2(u, v))))
            # the refills replay the recording the first fill made
            B = copy(A)
            @test assemble!(A, a) === A
            @test A == B
            @test assemble_parallel!(A, a) === A
            @test Matrix(A) ≈ Matrix(B)
            # one summand per order, so the first unit is walked on either leaf
            A21 = assemble(form(Wu, Wv, (u, v) -> a2(u, v) + a1(u, v)))
            @test Matrix(A21) ≈ A1 + A2
        end
    end

    Ω = domain(unit × unit)
    Wf = gridspace(mesh(Ω, (9, 6), (false, false)))
    Wc = gridspace(mesh(Ω, (5, 4), (false, false)))
    sumf = (u, v) -> f1(u, v) + f2(u, v)

    @testset "Threaded backend" begin
        be = backend(policy = CpuThreaded())
        Tf = gridspace(mesh(Ω, (9, 6), (false, false); backend = be))
        Tc = gridspace(mesh(Ω, (5, 4), (false, false); backend = be))
        ref = Matrix(assemble(form(Wf, Wc, sumf)))
        a = form(Tf, Tc, sumf)
        A = assemble(a)
        @test Matrix(A) ≈ ref
        @test assemble!(A, a) === A
        @test Matrix(A) ≈ ref
    end

    @testset "Dirichlet rows" begin
        # a Dirichlet row is the test space's: rows on the coarse boundary are those the
        # single term's Dirichlet assembly writes, every other row is the plain sum's
        a = form(Wf, Wc, sumf)
        AD = Matrix(assemble(a; dirichlet = :boundary))
        A1D = Matrix(assemble(form(Wf, Wc, f1); dirichlet = :boundary))
        S = Matrix(assemble(a))
        lin = LinearIndices(indices(mesh(Wc)))
        bnd = sort!(unique!([lin[I] for f in boundary_indices(mesh(Wc)) for I in f]))
        inner = setdiff(1:ndofs(Wc), bnd)
        @test !isempty(bnd) && !isempty(inner)
        @test AD[bnd, :] == A1D[bnd, :]
        @test AD[inner, :] == S[inner, :]
        @test AD[bnd, :] != S[bnd, :]
        A = assemble(a; dirichlet = :boundary)
        @test assemble!(A, a; dirichlet = :boundary) === A
        @test Matrix(A) == AD
    end

    @testset "Warm refill allocations" begin
        # the mixed sum, and a sum whose summands share one leaf (both trial-side), refill
        # in place as a single-leaf sum always has
        same = (u, v) -> f1(u, v) + g1(u, v)
        for g in (sumf, same)
            a = form(Wf, Wc, g)
            A = assemble(a)
            @test alloc_ok(alloc_test(assemble!, A, a))
        end
    end
end

end # module FormCrossMeshBlocksTests
