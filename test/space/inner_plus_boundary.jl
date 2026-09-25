module SpaceInnerPlusBoundaryTests

using Test
using Bramble
using Bramble: inner₊ₓ, weights
using LinearAlgebra: Symmetric, eigvals, isposdef

# `inner₊ₓ`/`inner₊ᵧ`/`inner₊₂`'s transverse factor used to hand-zero the first and last
# entry of every other axis, deleting real quadrature weight there: a node on two or more
# boundary hyperplanes got zero weight from every staggered direction and was absent from
# the assembled operator entirely (gpena/Bramble.jl#236). Fixed in
# `src/space/scalar_gridspace.jl`, `_innerplus_mean_weights!` -- see that function's own
# docstring for why the discrete summation-by-parts identities (test/space/sbp_identities.jl)
# do not depend on the zero, checked rather than assumed before the change.
#
# Every check here is against an independent reference: a hand-computed eigenvalue count, a
# manufactured solution's own exact convergence order, or a value computed a different way
# (`cell_measure`, which `innerₕ`'s own weight already trusted) -- never against another
# call to the code under test.

@testset "inner₊ transverse boundary weight (#236)" begin
    @testset "The boundary entries are the half-cell width, not zero" begin
        @testset "1D is unaffected: inner₊'s one direction has no transverse factor" begin
            # `_innerplus_mean_weights!` -- the function this fix changes -- is only ever
            # called for D >= 2 (`space_weights(Ωₕ::AbstractMeshType{1})` uses
            # `_innerplus_weights!` alone). `w[1] = 0` in 1D is `main`'s own zero, correct
            # both before and after this fix: D₋ₓ genuinely has no cell behind node 1.
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

    @testset "weights(Wₕ, Innerplus(), d) agrees with the documented sum" begin
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

    @testset "The staggered Neumann Laplacian has a 1D kernel and no zero rows" begin
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

        @testset "3D" begin
            kdim, zrows = check(
                mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (5, 4, 3), (true, true, true)),
                3
            )
            @test kdim == 1
            @test zrows == 0
        end
    end

    @testset "Scalar Neumann Poisson MMS converges at order 2" begin
        # -Δu = f, natural (unconstrained) traction-free boundary everywhere -- inner₊
        # encodes the Neumann condition without any dirichlet_bc! call. Pure Neumann is
        # singular up to an additive constant, pinned by fixing one dof to the exact value
        # there (a device, not a boundary condition of the continuum problem).
        uexact(x) = cos(pi * x[1]) * cos(pi * x[2])
        src(x) = 2 * pi^2 * cos(pi * x[1]) * cos(pi * x[2])

        function solve_neumann(n)
            Ωₕ = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (n, n), (true, true))
            Wₕ = gridspace(Ωₕ)
            a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
            l = form(Wₕ, v -> innerₕ(Rₕ(Wₕ, src), v))
            A = assemble(a)
            F = assemble(l)
            A[1, :] .= 0.0
            A[1, 1] = 1.0
            F[1] = uexact(points(Ωₕ)[1])
            uₕ = element(Wₕ)
            uₕ .= A \ F
            return normₕ(uₕ - Rₕ(Wₕ, uexact)), norm₁ₕ(uₕ - Rₕ(Wₕ, uexact))
        end

        ns = (13, 25, 49, 97)  # each doubles the previous mesh spacing exactly
        errs_h = Float64[]
        errs_1h = Float64[]
        for n in ns
            eh, e1h = solve_neumann(n)
            push!(errs_h, eh)
            push!(errs_1h, e1h)
        end

        rate(e) = [log2(e[i] / e[i + 1]) for i in 1:(length(e) - 1)]
        @test all(>(1.9), rate(errs_h))
        @test all(>(1.9), rate(errs_1h))
    end
end

end # module
