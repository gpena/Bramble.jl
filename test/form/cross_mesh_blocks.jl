using Test
using Bramble
using Bramble: CompositeGridSpace, form, assemble, assemble!, assemble_parallel!,
               allocate_system_matrix

# Coupling two leaves of a composite space whose meshes differ in size (point 69).
#
# A coupled bilinear term is assembled by walking the *test* leaf's grid and reading the
# trial column out of that same index space, offset into the trial leaf's block:
# `lin_indices[I + off_u] + col_offset`. That only means something when the two leaves share
# an index space, which they always did until heterogeneous composite spaces arrived — a
# space built by repeating one space hands every leaf the same mesh object.
#
# On leaves of different sizes it failed in whichever direction the sizes ran, and one of the
# two failures was silent:
#   innerₕ(u(2), v(1))  small trial, big test  → ArgumentError from inside sparse!, naming
#                                                column indices rather than the real problem
#   innerₕ(u(1), v(2))  big trial, small test  → assembled, in-range but WRONG columns
#
# There is no correspondence between an index on one mesh and an index on the other, so the
# term has no assembly until something says how to map between them — a symbolic
# interpolation operator, which is point 61 and is not built. Until then it is refused by
# name, at every entry point.

@testset "cross-mesh bilinear blocks are refused, not guessed at (point 69)" begin
    Ωbig = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (6, 6), (true, true))
    Ωsmall = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (3, 3), (true, true))
    Wbig, Wsmall = gridspace(Ωbig), gridspace(Ωsmall)
    Vh = CompositeGridSpace((Wbig, Wsmall))

    @testset "both directions refuse, through every entry point" begin
        # the direction that used to throw from inside `sparse!`, and the one that used to
        # assemble silently wrong columns — the second is the reason this is a test and not
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

    @testset "the message names the problem, not the symptom" begin
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

    @testset "an operator on either side does not smuggle it past the check" begin
        for g in ((u, v) -> inner₊ₓ(D₋ₓ(u(2)), D₋ₓ(v(1))),
            (u, v) -> innerₕ(M₋ₓ(u(1)), v(2)),
            (u, v) -> innerₕ(u(1), v(1)) + innerₕ(u(2), v(1)))   # one good term, one bad
            @test_throws ArgumentError assemble(form(Vh, Vh, g))
        end
    end

    @testset "what must keep working" begin
        # diagonal blocks on the same heterogeneous space: each leaf couples to itself, so
        # the index spaces match by construction
        a_diag = form(Vh, Vh, (u, v) -> innerₕ(u(1), v(1)) + innerₕ(u(2), v(2)))
        A = assemble(a_diag)
        @test size(A) == (ndofs(Vh), ndofs(Vh))
        @test count(!iszero, A) == ndofs(Vh)
        @test assemble!(A, a_diag) === A
        @test assemble_parallel!(A, a_diag) === A

        # a term naming no component at all, on a heterogeneous space: it reaches every
        # diagonal block, which is the `blk === nothing` branch of the same routing
        a_broadcast = form(Vh, Vh, (u, v) -> innerₕ(u, v))
        @test size(assemble(a_broadcast)) == (ndofs(Vh), ndofs(Vh))

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
        @test size(assemble(form(Vtwin, Vtwin, (u, v) -> innerₕ(u(2), v(1))))) ==
              (ndofs(Vtwin), ndofs(Vtwin))

        # and a plain scalar form, where trial and test are one space
        Wₕ = gridspace(Ωbig)
        @test size(assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v)))) ==
              (ndofs(Wₕ), ndofs(Wₕ))
    end

    @testset "the matrix names its rows by the test space and columns by the trial one" begin
        # equal here, because the check above is what allows the form at all — asserted so
        # that a future cross-mesh operator (point 61) has something to change deliberately
        Wₕ = gridspace(Ωbig)
        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v))
        A = allocate_system_matrix(a)
        @test size(A, 1) == ndofs(Bramble.test_space(a))
        @test size(A, 2) == ndofs(Bramble.trial_space(a))
    end

    @testset "the linear side is unaffected — that is the case πₕ already makes well posed" begin
        # the same cross-mesh coupling in a LINEAR form is legitimate and works: πₕ supplies
        # the mapping the bilinear side lacks
        uv = Rₕ(Vh, (x -> 0.0, x -> x[1] + x[2]))
        b = assemble(form(Vh, v -> innerₕ(πₕ(uv(2)), v(1))))
        @test length(b) == ndofs(Vh)
        @test !all(iszero, b)
    end
end
