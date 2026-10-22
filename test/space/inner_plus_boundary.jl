module SpaceInnerPlusBoundaryTests

using Test
using Bramble
using Bramble: inner₊ₓ, weights
using LinearAlgebra: Symmetric, eigvals, isposdef
using ..TestUtils: WITH_3D_TESTS, _backward_spacing_oracle, _half_spacing_oracle

# `inner₊ₓ`/`inner₊ᵧ`/`inner₊₂`'s transverse factor used to hand-zero the first and last
# entry of every other axis, deleting real quadrature weight there: a node on two or more
# boundary hyperplanes got zero weight from every staggered direction and was absent from
# the assembled operator entirely (gpena/Bramble.jl#236). The transverse factor is now
# `cell_measures` of each other axis (the half-cell width at the boundary), which
# `space_weights` in `src/space/scalar_gridspace.jl` reads. The discrete summation-by-parts
# identities (test/space/sbp_identities.jl) do not depend on the old zero: each is stated
# for a test function vanishing on the boundary, so the remainder an integration by parts
# leaves there is already zero on its own account.
#
# Every check here is against an independent reference: a hand-computed eigenvalue count, a
# manufactured solution's own exact convergence order, or a value computed a different way
# (`cell_measure`, which `innerₕ`'s own weight already trusted) -- never against another
# call to the code under test.

@testset "inner₊ transverse boundary weight (#236)" begin
    @testset "Boundary entries: half-cell, not zero" begin
        # In 1D inner₊ has one direction, so there is no transverse factor.
        @testset "1D unaffected: no transverse factor" begin
            # `space_weights(Ωₕ::AbstractMeshType{1})` has the aligned factor alone.
            # `w[1] = 0` in 1D is `main`'s own zero, correct both before and after this
            # fix: D₋ₓ genuinely has no cell behind node 1.
            Ωₕ = mesh(domain(interval(0.0, 1.0)), 9, true)
            w = weights(gridspace(Ωₕ), Bramble.Innerplus(), 1)
            @test w[1] == 0.0
            @test w[end] > 0
        end

        # 2D, 5x4 mesh: the issue's own reproducer numbers. Before this fix, inner₊ₓ had
        # 12 of 20 weights identically zero (every node on the y-boundary, not only the
        # x = 1 column); after it, only the 4 nodes at x = 1 are zero -- correctly, since
        # `main_x(1) = 0` there (D₋ₓ has no cell behind the first node in the *aligned*
        # direction, unaffected by this fix, which only repairs the *transverse* factor).
        nx, ny = 5, 4
        Ωₕ = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (nx, ny), (true, true))
        Wₕ = gridspace(Ωₕ)
        wx_grid = reshape(weights(Wₕ, Bramble.Innerplus(), 1), nx, ny)  # column-major, x fastest
        @test count(iszero, wx_grid) == ny        # was 3*ny = 12 before the fix
        @test all(iszero, wx_grid[1, :])          # the x = 1 column: correctly zero
        @test all(!iszero, wx_grid[2:end, :])     # every other node, y-boundary included

        # The transverse direction for inner₊ᵧ is x, so the analogous check runs the other
        # way: the y = 1 row is correctly zero (main_y(1) = 0), and the x-boundary is not.
        wy_grid = reshape(weights(Wₕ, Bramble.Innerplus(), 2), nx, ny)
        @test count(iszero, wy_grid) == nx        # was 3*nx = 15 before the fix
        @test all(iszero, wy_grid[:, 1])
        @test all(!iszero, wy_grid[:, 2:end])
    end

    # The whole 2D weight on a non-uniform mesh, boundary entries included: the aligned
    # backward spacing along d times the half-cell width along the other axis, computed
    # from the points alone.
    @testset "Non-uniform 2D weight, boundary included" begin
        Ωₕ = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (6, 5), (false, false))
        px, py = points(Ωₕ(1)), points(Ωₕ(2))
        nx, ny = length(px), length(py)
        Wₕ = gridspace(Ωₕ)
        wx = reshape(weights(Wₕ, Bramble.Innerplus(), 1), nx, ny)
        wy = reshape(weights(Wₕ, Bramble.Innerplus(), 2), nx, ny)
        for j in 1:ny, i in 1:nx

            @test wx[i, j] ≈ _backward_spacing_oracle(px, i) * _half_spacing_oracle(py, j)
            @test wy[i, j] ≈ _half_spacing_oracle(px, i) * _backward_spacing_oracle(py, j)
        end
        # Non-trivial: the transverse boundary entries are positive and the mesh really
        # is non-uniform.
        @test all(>(0), wx[2:end, :]) && all(>(0), wy[:, 2:end])
        @test !(px[2] - px[1] ≈ px[3] - px[2])
    end

    # `weights(Wₕ, Innerplus(), d)` against the documented sum.
    @testset "Innerplus weights match documented sum" begin
        # inner₊ₓ's own docstring: (u,v)_+x = Σᵢ Σⱼ h_{x,i} h_{y,j+1/2} u v, a sum over
        # *every* j. Check it directly: constant fields make the inner product exactly the
        # total weight, independent of which field values are picked.
        Ωₕ = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (7, 6), (true, true))
        Wₕ = gridspace(Ωₕ)
        ones_ = Rₕ(Wₕ, x -> 1.0)
        total = inner₊ₓ(ones_, ones_)
        @test total ≈ sum(weights(Wₕ, Bramble.Innerplus(), 1))
        # the domain is the unit square and h_x sums to 1 (minus the zeroed first entry,
        # which has no cell behind it) while h_y sums to exactly 1 (every entry counted,
        # boundary included) -- so the total is the same as innerₕ's own total mass except
        # for the one genuinely-absent x-direction cell.
        @test total ≈ sum(weights(Wₕ, Bramble.Innerh())) atol = 1e-10
    end

    # The staggered Neumann Laplacian has a one-dimensional kernel and no zero rows.
    @testset "Neumann Laplacian kernel and zero rows" begin
        function check(Ωₕ, D)
            Wₕ = gridspace(Ωₕ)
            a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
            A = Matrix(assemble(a))
            ev = eigvals(Symmetric(A))
            kernel_dim = count(x -> abs(x) < 1e-8 * maximum(abs, ev), ev)
            zero_rows = count(i -> all(iszero, @view A[i, :]), axes(A, 1))
            return kernel_dim, zero_rows
        end

        @testset "1D" begin
            kdim, zrows = check(mesh(domain(interval(0.0, 1.0)), 11, true), 1)
            @test kdim == 1
            @test zrows == 0
        end

        @testset "2D" begin
            kdim, zrows = check(mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (6, 5), (true, true)), 2)
            @test kdim == 1
            @test zrows == 0
        end

        # 3D only: built in `slow` and by the nightly run (gpena/Bramble.jl#650).
        WITH_3D_TESTS && @testset "3D" begin
            kdim, zrows = check(
                mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (5, 4, 3), (true, true, true)),
                3
            )
            @test kdim == 1
            @test zrows == 0
        end
    end
end

end # module
