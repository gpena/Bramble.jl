using Test
using Bramble
using SparseArrays
using LinearAlgebra: issymmetric

# Symmetrizing the constrained system.
#
# `dirichlet_bc!` zeros the marked *rows* and puts one on the diagonal, which destroys the
# symmetry of an operator that had it. `symmetrize!` restores it by eliminating the marked
# *columns* into the right-hand side. The pair is what lets a symmetric assembled operator
# stay symmetric under constraints, which is what a Cholesky factorization needs — and the
# ordering matters: between the two calls the matrix is not symmetric.
#
# Two things this file pins that were wrong or absent:
#
#   - `symmetrize!` had no `CompositeGridSpace` method at all, while `dirichlet_bc!` did. A
#     coupled system met a MethodError on one and not the other, though a system is rarely
#     constrained by one and not the other.
#   - the dense path called `findall(mask)` to list the marked indices, allocating a vector
#     that grows with the boundary. Both paths now walk the mask's set bits.

# a symmetric, structurally symmetric operator to constrain
_tri(m) = spdiagm(0 => fill(4.0, m), 1 => fill(-1.0, m - 1), -1 => fill(-1.0, m - 1))

@testset "Symmetrizing the constrained system" begin
    Ωₕ = mesh(
        domain(interval(0.0, 1.0) × interval(0.0, 1.0),
            :bottom => :bottom, :top => :top),
        (6, 6), (true, true))
    Wₕ = gridspace(Ωₕ)
    Vₕ = gridspace(Ωₕ, Val(3))
    n = ndofs(Wₕ)

    @testset "it is what restores symmetry, and only after the rows are set" begin
        A = _tri(n)
        F = collect(1.0:n)
        @test issymmetric(A)

        @test dirichlet_bc!(A, Ωₕ, :bottom) === A
        @test !issymmetric(A)              # rows zeroed, columns untouched

        @test symmetrize!(A, F, Ωₕ, :bottom) === nothing
        @test issymmetric(A)
    end

    @testset "the constrained solve returns the boundary values" begin
        A = _tri(n)
        F = collect(1.0:n)
        dirichlet_bc!(A, Ωₕ, :bottom)
        symmetrize!(A, F, Ωₕ, :bottom)

        x = A \ F
        marked = index_in_marker(Ωₕ, :bottom)
        @test any(marked)                  # the test is not vacuous
        for i in 1:n
            marked[i] && @test x[i] ≈ F[i]
        end
    end

    @testset "dense and sparse agree" begin
        As, Fs = _tri(n), collect(1.0:n)
        Ad, Fd = Matrix(_tri(n)), collect(1.0:n)
        for (A, F) in ((As, Fs), (Ad, Fd))
            dirichlet_bc!(A, Ωₕ, :bottom)
            symmetrize!(A, F, Ωₕ, :bottom)
        end
        @test Ad == Matrix(As)
        @test Fd == Fs
    end

    @testset "a composite system is the scalar one, block by block" begin
        Av = blockdiag(_tri(n), _tri(n), _tri(n))
        Fv = repeat(collect(1.0:n), 3)
        dirichlet_bc!(Av, Vₕ, :bottom)
        symmetrize!(Av, Fv, Vₕ, :bottom)

        As, Fs = _tri(n), collect(1.0:n)
        dirichlet_bc!(As, Ωₕ, :bottom)
        symmetrize!(As, Fs, Ωₕ, :bottom)

        for k in 0:2
            r = (k * n + 1):((k + 1) * n)
            @test Av[r, r] == As
            @test Fv[r] == Fs
        end
        @test issymmetric(Av)
        # the leaves are independent: nothing leaked across the block boundary
        @test all(iszero, Av[1:n, (n + 1):(2n)])
        @test all(iszero, Av[(n + 1):(2n), 1:n])
    end

    @testset "the offsets are right under nesting" begin
        # a composite of composites must still see each leaf at its own offset
        inner = gridspace(Ωₕ, Val(2))
        nested = Bramble.CompositeGridSpace((Wₕ, inner, Wₕ))
        flat = gridspace(Ωₕ, Val(4))
        @test ndofs(nested) == ndofs(flat) == 4n

        An = blockdiag(ntuple(_ -> _tri(n), 4)...)
        Af = copy(An)
        Fn, Ff = repeat(collect(1.0:n), 4), repeat(collect(1.0:n), 4)

        dirichlet_bc!(An, nested, :bottom)
        symmetrize!(An, Fn, nested, :bottom)
        dirichlet_bc!(Af, flat, :bottom)
        symmetrize!(Af, Ff, flat, :bottom)
        @test An == Af
        @test Fn == Ff
    end

    @testset "several labels, and none" begin
        A2, F2 = _tri(n), collect(1.0:n)
        dirichlet_bc!(A2, Ωₕ, :bottom, :top)
        symmetrize!(A2, F2, Ωₕ, :bottom, :top)
        @test issymmetric(A2)

        # applying them one at a time is the same as both at once
        A1, F1 = _tri(n), collect(1.0:n)
        dirichlet_bc!(A1, Ωₕ, :bottom)
        dirichlet_bc!(A1, Ωₕ, :top)
        symmetrize!(A1, F1, Ωₕ, :bottom)
        symmetrize!(A1, F1, Ωₕ, :top)
        @test A1 == A2
        @test F1 ≈ F2

        # no labels changes nothing
        A0, F0 = _tri(n), collect(1.0:n)
        Abefore, Fbefore = copy(A0), copy(F0)
        symmetrize!(A0, F0, Ωₕ)
        @test A0 == Abefore
        @test F0 == Fbefore
    end

    @testset "the wrapper does both, in order" begin
        Aw, Fw = _tri(n), collect(1.0:n)
        Bramble.dirichlet_bc_symmetrize!(Aw, Fw, Ωₕ, :bottom)

        Ae, Fe = _tri(n), collect(1.0:n)
        dirichlet_bc!(Ae, Ωₕ, :bottom)
        symmetrize!(Ae, Fe, Ωₕ, :bottom)

        @test Aw == Ae
        @test Fw == Fe
        @test issymmetric(Aw)

        # the stored zeros stay stored: the sparsity pattern is the stencil's, and is not
        # allowed to start depending on the boundary data. This is why the `dropzeros`
        # option was removed rather than defaulted off.
        @test nnz(Aw) == nnz(_tri(n))
        @test count(iszero, nonzeros(Aw)) > 0
    end

    @testset "the diagonal, found in the sweep or put back afterwards" begin
        # The diagonal is written where the sweep finds it rather than through
        # `A[i, i] = one(T)` afterwards, which would binary search the column for an entry
        # the loop has just walked past. Both paths have to end up in the same place.
        A, F = _tri(n), collect(1.0:n)
        dirichlet_bc!(A, Ωₕ, :bottom)          # leaves a stored diagonal
        symmetrize!(A, F, Ωₕ, :bottom)
        marked = index_in_marker(Ωₕ, :bottom)
        @test all(A[i, i] == 1.0 for i in 1:n if marked[i])

        # a matrix that stores no diagonal at all: the fallback has to supply it
        B = spzeros(n, n)
        for j in 1:n
            B[mod1(j + 1, n), j] = 2.0
        end
        @test !any(B[i, i] != 0 for i in 1:n if marked[i])
        symmetrize!(B, collect(1.0:n), Ωₕ, :bottom)
        @test all(B[i, i] == 1.0 for i in 1:n if marked[i])
    end

    @testset "homogeneous conditions take the short path to the same answer" begin
        # A zero boundary value contributes nothing to F, so the elimination is skipped —
        # worth about 12% on the conditions that are most common. The result must not
        # depend on which branch was taken.
        Az, Fz = _tri(n), zeros(n)
        dirichlet_bc!(Az, Ωₕ, :bottom)
        symmetrize!(Az, Fz, Ωₕ, :bottom)
        @test issymmetric(Az)
        @test all(iszero, Fz)

        # the matrix is the same one an inhomogeneous condition produces: only F differs
        An, Fn = _tri(n), collect(1.0:n)
        dirichlet_bc!(An, Ωₕ, :bottom)
        symmetrize!(An, Fn, Ωₕ, :bottom)
        @test Az == An
    end

    @testset "the interface takes a mesh, a scalar space or a composite space" begin
        # All three entry points now accept all three, which they did not: `symmetrize!`
        # was the one that rejected a `ScalarGridSpace`, so `dirichlet_bc!(A, Wₕ, :bottom)`
        # worked while `symmetrize!(A, F, Wₕ, :bottom)` was a MethodError — for two calls
        # that are almost always written together.
        bcs = dirichlet_constraints(set(Ωₕ), :bottom => (x -> 7.0))
        for (holder, nd) in ((Ωₕ, n), (Wₕ, n), (Vₕ, ndofs(Vₕ)))
            A = blockdiag(ntuple(_ -> _tri(n), nd ÷ n)...)
            F = ones(nd)
            dirichlet_bc!(A, holder, :bottom)
            symmetrize!(A, F, holder, :bottom)
            @test issymmetric(A)

            v = zeros(nd)
            dirichlet_bc!(v, holder, bcs, :bottom)
            @test any(==(7.0), v)
        end

        # a scalar space and its mesh give the same answer
        Aw, Fw = _tri(n), ones(n)
        Am, Fm = _tri(n), ones(n)
        symmetrize!(Aw, Fw, Wₕ, :bottom)
        symmetrize!(Am, Fm, Ωₕ, :bottom)
        @test Aw == Am
        @test Fw == Fm
    end

    @testset "constraints and time-evaluated constraints share one method" begin
        # `EvaluatedDomainMarkers` holds the original alongside a timestamp, so it is a
        # distinct type — but it answers `conditions`, `label` and `identifier`
        # identically, and applying a condition never needs to tell the two apart. There
        # used to be two byte-identical methods, one per type.
        tb = dirichlet_constraints(set(Ωₕ), interval(0.0, 1.0),
            :bottom => ((x, t) -> t * x[1] + 1))
        ev = tb(0.5)
        @test ev isa Bramble.EvaluatedDomainMarkers

        @test which(dirichlet_bc!, Tuple{
            Vector{Float64}, typeof(Ωₕ), typeof(tb), Symbol}) ===
              which(dirichlet_bc!, Tuple{Vector{Float64}, typeof(Ωₕ), typeof(ev), Symbol})

        v = zeros(n)
        dirichlet_bc!(v, Ωₕ, ev, :bottom)
        marked = index_in_marker(Ωₕ, :bottom)
        pts = [Bramble.point(Ωₕ, Bramble.indices(Ωₕ)[i]) for i in 1:n if marked[i]]
        vals = v[marked]
        @test !isempty(pts)
        @test all(vals[k] ≈ 0.5 * pts[k][1] + 1 for k in eachindex(pts))
    end

    @testset "allocation free, scalar and composite, dense and sparse" begin
        # It runs once per step of a time loop. The dense path used to allocate a
        # `findall` vector that grew with the boundary; both paths now walk the mask's set
        # bits, which also does work proportional to what is marked rather than to the
        # whole grid. Measured inside a function on concrete locals — read from a
        # non-const global the arguments box at the call boundary.
        function counts(N)
            Ω = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0), :bottom => :bottom),
                (N, N), (true, true))
            W, V = gridspace(Ω), gridspace(Ω, Val(3))
            m, mv = ndofs(W), ndofs(V)
            As, Fs = _tri(m), ones(m)
            Ad, Fd = Matrix(_tri(m)), ones(m)
            Cs, Cf = blockdiag(_tri(m), _tri(m), _tri(m)), ones(mv)
            Cd, Cdf = Matrix(blockdiag(_tri(m), _tri(m), _tri(m))), ones(mv)

            symmetrize!(As, Fs, Ω, :bottom)
            symmetrize!(Ad, Fd, Ω, :bottom)
            symmetrize!(Cs, Cf, V, :bottom)
            symmetrize!(Cd, Cdf, V, :bottom)

            return (sparse_scalar = @allocated(symmetrize!(As, Fs, Ω, :bottom)),
                dense_scalar = @allocated(symmetrize!(Ad, Fd, Ω, :bottom)),
                sparse_composite = @allocated(symmetrize!(Cs, Cf, V, :bottom)),
                dense_composite = @allocated(symmetrize!(Cd, Cdf, V, :bottom)))
        end

        for N in (8, 24)          # 9x the degrees of freedom apart
            c = counts(N)
            @test c.sparse_scalar == 0
            @test c.dense_scalar == 0
            @test c.sparse_composite == 0
            @test c.dense_composite == 0
        end
    end
end
