using Test
using Bramble
using Random
using Supposition
using Bramble: values, components

# The centered difference.
#
#   Dc(uₕ)(i) = (u(x_{i+1}) - u(x_{i-1})) / (h_i + h_{i+1})
#
# The denominator is x_{i+1} - x_{i-1}, so the operator differences over the whole span
# its stencil covers. Two consequences are worth pinning, because they hold on any grid
# rather than only on a uniform one:
#
#   - it reproduces the derivative of an affine function exactly, and
#   - it is skew-symmetric in innerₕ for grid functions vanishing on the boundary.
#
# The second follows from the first: innerₕ weights point i by the cell measure
# (h_i + h_{i+1})/2, which cancels the denominator outright, leaving half the sum of
# u_{i+1} - u_{i-1} against v_i. That cancellation is the reason to divide by this
# denominator rather than by the two spacings separately.

# The operators as matrices, for `test_operator_matrix_equivalence` (test/space/difference.jl).
centered_ops(::Val{1}) = (Dcₓ,)
centered_ops(::Val{2}) = (Dcₓ, Dcᵧ)
centered_ops(::Val{3}) = (Dcₓ, Dcᵧ, Dc₂)

@testset "Centered difference" begin
    @testset "Definition match" begin
        for (lbl, unif) in (("uniform", true), ("random", false))
            @testset "$lbl" begin
                Random.seed!(20260830)
                Ωₕ = mesh(domain(interval(0.0, 1.0)), 11, unif)
                Wₕ = gridspace(Ωₕ)
                n = npoints(Ωₕ)
                uₕ = Rₕ(Wₕ, x -> x^2 + sin(x))
                u = values(uₕ)

                want = [(i == 1 || i == n) ? 0.0 :
                        (u[i + 1] - u[i - 1]) / (spacing(Ωₕ, i) + spacing(Ωₕ, i + 1))
                        for i in 1:n]
                @test values(Dcₓ(uₕ)) ≈ want

                # the denominator is the span the stencil covers
                @test all(values(Dcₓ(uₕ))[i] ≈
                          (u[i + 1] - u[i - 1]) /
                          (points(Ωₕ)[i + 1] - points(Ωₕ)[i - 1]) for i in 2:(n - 1))
            end
        end
    end

    @testset "Truncation at ends" begin
        # Unlike the one-sided families, which lose one slice, this loses two: neither
        # the first nor the last point has a neighbour on both sides.
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (6, 7), (true, false))
        Random.seed!(20260830)
        Wₕ = gridspace(Ωₕ)
        n = npoints(Ωₕ, Tuple)
        uₕ = Rₕ(Wₕ, x -> exp(x[1]) * (x[2] + 1))

        rx = reshape(values(Dcₓ(uₕ)), n)
        @test all(iszero, rx[1, :])
        @test all(iszero, rx[end, :])
        @test !any(iszero, rx[2:(end - 1), :])

        ry = reshape(values(Dcᵧ(uₕ)), n)
        @test all(iszero, ry[:, 1])
        @test all(iszero, ry[:, end])
        @test !any(iszero, ry[:, 2:(end - 1)])
    end

    @testset "Exact on affine" begin
        # The property the denominator buys. It holds for every grid, not only uniform
        # ones, which is what makes the operator worth having in this form.
        for (lbl, unif) in (("uniform", true), ("random", false))
            @testset "$lbl" begin
                Random.seed!(20260830)
                Ωₕ = mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (5, 6, 4),
                    (unif, unif, unif))
                Wₕ = gridspace(Ωₕ)
                n = npoints(Ωₕ, Tuple)

                @test all(iszero, values(Dcₓ(Rₕ(Wₕ, x -> 3.0))))

                for (d, op) in ((1, Dcₓ), (2, Dcᵧ), (3, Dc₂))
                    # constant along d differences to zero along d
                    @test all(iszero, values(op(Rₕ(Wₕ, x -> x[mod1(d + 1, 3)]))))
                    # and 3x + 1 along d differences to exactly 3, away from both
                    # truncated slices
                    r = reshape(values(op(Rₕ(Wₕ, x -> 3x[d] + 1))), n)
                    interior = ntuple(k -> k == d ? (2:(n[k] - 1)) : (1:n[k]), 3)
                    @test all(≈(3.0), r[interior...])
                end
            end
        end
    end

    @testset "Weighted average of D₋ & D₊" begin
        # h_{i+1} D₊ + h_i D₋ telescopes to u_{i+1} - u_{i-1}, so the centered difference
        # is that combination divided by h_i + h_{i+1}. On a uniform grid it reduces to
        # the plain mean of the two, which is the familiar form.
        for (lbl, unif) in (("uniform", true), ("random", false))
            @testset "$lbl" begin
                Random.seed!(20260830)
                Ωₕ = mesh(domain(interval(0.0, 1.0)), 17, unif)
                Wₕ = gridspace(Ωₕ)
                n = npoints(Ωₕ)
                uₕ = Rₕ(Wₕ, x -> sin(3x))
                dc, dm, dp = values(Dcₓ(uₕ)), values(D₋ₓ(uₕ)), values(D₊ₓ(uₕ))

                @test all(dc[i] ≈
                          (spacing(Ωₕ, i + 1) * dp[i] + spacing(Ωₕ, i) * dm[i]) /
                          (spacing(Ωₕ, i) + spacing(Ωₕ, i + 1))
                for i in 2:(n - 1))

                if unif
                    @test all(dc[i] ≈ (dm[i] + dp[i]) / 2 for i in 2:(n - 1))
                end
            end
        end
    end

    @testset "Convergence order" begin
        # Second order on a uniform grid, and first order on a non-uniform one: the
        # centered difference approximates the derivative at the midpoint of its stencil,
        # which coincides with xᵢ only when the two spacings match.
        function orders(unif; steps = 4)
            Random.seed!(20260830)
            Ωₕ = mesh(domain(interval(0.0, 1.0)), 21, unif)
            errs = Float64[]
            for k in 0:steps
                k > 0 && iterative_refinement!(Ωₕ)
                Wₕ = gridspace(Ωₕ)
                e = values(Dcₓ(Rₕ(Wₕ, sin))) .- values(Rₕ(Wₕ, cos))
                push!(errs, maximum(abs, e[2:(end - 1)]))
            end
            return [log2(errs[k] / errs[k + 1]) for k in 1:(length(errs) - 1)]
        end

        ou = orders(true)
        @test all(o -> abs(o - 2.0) < 0.05, ou)

        # Refinement halves every interval, so the non-uniform grids stay nested and the
        # ratio is an order even though the starting grid is random.
        orand = orders(false)
        @test all(>(0.95), orand)
        @test last(orand) < 1.15
    end

    @testset "Directional family" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 6), (true, false))
        Random.seed!(20260830)
        Wₕ = gridspace(Ωₕ)
        Vₕ = gridspace(Ωₕ, Val(2))
        uₕ = Rₕ(Wₕ, x -> x[1] * x[2])

        @test Dcₕ(uₕ) isa NTuple{2, VectorElement}
        @test values(Dcₕ(uₕ)[1]) == values(Dcₓ(uₕ))
        @test values(Dcₕ(uₕ)[2]) == values(Dcᵧ(uₕ))

        # in one dimension the tuple and the grid function coincide
        Ω1 = mesh(domain(interval(0.0, 1.0)), 7, true)
        u1 = Rₕ(gridspace(Ω1), sin)
        @test !(Dcₕ(u1) isa Tuple)
        @test values(Dcₕ(u1)) == values(Dcₓ(u1))

        # composite grid functions apply componentwise, as the other operators do
        fs = (x -> x[1], x -> x[2]^2)
        cₕ = Rₕ(Vₕ, fs)
        scalars = (Rₕ(Wₕ, fs[1]), Rₕ(Wₕ, fs[2]))
        rₕ = Dcₓ(cₕ)
        @test length(values(rₕ)) == length(values(cₕ))
        for k in 1:2
            @test values(components(rₕ)[k]) == values(Dcₓ(scalars[k]))
        end
    end

    @testset "Type stability & allocations" begin
        Ωₕ1 = mesh(domain(interval(0.0, 1.0)), 33, false)
        Ωₕ2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 8), (true, false))
        u1 = Rₕ(gridspace(Ωₕ1), sin)
        u2 = Rₕ(gridspace(Ωₕ2), x -> x[1] * x[2])

        @test @inferred(Dcₓ(u1)) isa VectorElement
        @test @inferred(Dcᵧ(u2)) isa VectorElement
        @test @inferred(Dcₕ(u2)) isa NTuple{2, VectorElement}

        @test alloc_test(Dcₓ, u1) == alloc_test(similar, u1)
        @test alloc_test(Dcᵧ, u2) == alloc_test(similar, u2)
    end

    @testset "Skew-symmetry" begin
        # innerₕ(Dcₓ(uₕ), vₕ) == -innerₕ(uₕ, Dcₓ(vₕ)) when both vanish on the boundary.
        #
        # innerₕ weights point i by (h_i + h_{i+1})/2, which is exactly half the centered
        # denominator, so the weights cancel and the left side is half the sum of
        # (u_{i+1} - u_{i-1}) v_i over the interior. Shifting that sum by one gives the
        # right side, exactly and on any grid.
        skew(u, v) = isapprox(innerₕ(Dcₓ(u), v), -innerₕ(u, Dcₓ(v));
            atol = 1e-12, rtol = 1e-12)

        @testset "1D" begin
            for (lbl, unif) in (("uniform", true), ("random", false)), n in (11, 51, 201)

                Random.seed!(20260830)
                Ωₕ = mesh(domain(interval(0.0, 1.0)), n, unif)
                Wₕ = gridspace(Ωₕ)
                uₕ = Rₕ(Wₕ, x -> sin(pi * x))
                vₕ = Rₕ(Wₕ, x -> sin(2pi * x) * x * (1 - x))
                @test skew(uₕ, vₕ)

                # the cancellation that makes it exact
                u, v = values(uₕ), values(vₕ)
                @test innerₕ(Dcₓ(uₕ), vₕ) ≈
                      sum((u[i + 1] - u[i - 1]) * v[i] for i in 2:(n - 1)) / 2
            end
        end

        @testset "Vanishing boundary" begin
            Random.seed!(20260830)
            Ωₕ = mesh(domain(interval(0.0, 1.0)), 41, false)
            Wₕ = gridspace(Ωₕ)
            zero_bdry = Rₕ(Wₕ, x -> sin(pi * x))
            nonzero = Rₕ(Wₕ, x -> cos(x) + 0.7)

            @test skew(zero_bdry, zero_bdry)
            @test !skew(zero_bdry, nonzero)
            @test !skew(nonzero, zero_bdry)
            @test !skew(nonzero, nonzero)
        end

        @testset "2D & 3D directions" begin
            f2 = x -> sin(pi * x[1]) * sin(pi * x[2])
            g2 = x -> sin(2pi * x[1]) * sin(pi * x[2]) * x[1]
            f3 = x -> sin(pi * x[1]) * sin(pi * x[2]) * sin(pi * x[3])
            g3 = x -> sin(2pi * x[1]) * sin(pi * x[2]) * sin(pi * x[3]) * x[2]

            for unif in (true, false)
                Random.seed!(20260830)
                Ω2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (21, 19),
                    (unif, unif))
                W2 = gridspace(Ω2)
                a, b = Rₕ(W2, f2), Rₕ(W2, g2)
                for op in (Dcₓ, Dcᵧ)
                    @test isapprox(innerₕ(op(a), b), -innerₕ(a, op(b));
                        atol = 1e-12, rtol = 1e-12)
                end

                Ω3 = mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (11, 9, 8),
                    (unif, unif, unif))
                W3 = gridspace(Ω3)
                c, d = Rₕ(W3, f3), Rₕ(W3, g3)
                for op in (Dcₓ, Dcᵧ, Dc₂)
                    @test isapprox(innerₕ(op(c), d), -innerₕ(c, op(d));
                        atol = 1e-12, rtol = 1e-12)
                end
            end
        end

        @testset "Random grids (Supposition)" begin
            positive_h = Data.Floats{Float64}(; minimum = 0.01, maximum = 10.0,
                nans = false, infs = false)
            field_val = Data.Floats{Float64}(; minimum = -100.0, maximum = 100.0,
                nans = false, infs = false)

            # 1D: arbitrary non-uniform mesh with both fields vanishing on the boundary
            @check function check_dc_skew_1d(
                    h = Data.Vectors(positive_h; min_size = 3, max_size = 30),
                    u_raw = Data.Vectors(field_val; min_size = 31, max_size = 31),
                    v_raw = Data.Vectors(field_val; min_size = 31, max_size = 31)
            )
                n = length(h) + 1
                pts = zeros(Float64, n)
                for i in 1:length(h)
                    pts[i + 1] = pts[i] + h[i]
                end
                pts ./= pts[end]

                Ωₕ = mesh(domain(interval(0.0, 1.0)), n, false)
                set_points!(Ωₕ, pts)
                Wₕ = gridspace(Ωₕ)

                u_vals = copy(u_raw[1:n])
                v_vals = copy(v_raw[1:n])
                u_vals[1] = 0.0
                u_vals[end] = 0.0
                v_vals[1] = 0.0
                v_vals[end] = 0.0

                uₕ = element(Wₕ, u_vals)
                vₕ = element(Wₕ, v_vals)

                lhs = innerₕ(Dcₓ(uₕ), vₕ)
                rhs = -innerₕ(uₕ, Dcₓ(vₕ))
                scale = max(abs(lhs), abs(rhs), 1.0)
                isapprox(lhs, rhs; atol = 1e-10 * scale, rtol = 1e-10)
            end

            # 2D: arbitrary non-uniform tensor product mesh across both coordinates
            @check function check_dc_skew_2d(
                    hx = Data.Vectors(positive_h; min_size = 2, max_size = 8),
                    hy = Data.Vectors(positive_h; min_size = 2, max_size = 8),
                    u_raw = Data.Vectors(field_val; min_size = 81, max_size = 81),
                    v_raw = Data.Vectors(field_val; min_size = 81, max_size = 81)
            )
                nx = length(hx) + 1
                ny = length(hy) + 1
                pts_x = zeros(Float64, nx)
                for i in 1:length(hx)
                    pts_x[i + 1] = pts_x[i] + hx[i]
                end
                pts_x ./= pts_x[end]

                pts_y = zeros(Float64, ny)
                for j in 1:length(hy)
                    pts_y[j + 1] = pts_y[j] + hy[j]
                end
                pts_y ./= pts_y[end]

                Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (nx, ny),
                    (false, false))
                set_points!(Ωₕ(1), pts_x)
                set_points!(Ωₕ(2), pts_y)
                Wₕ = gridspace(Ωₕ)

                total = nx * ny
                u_mat = reshape(copy(u_raw[1:total]), nx, ny)
                v_mat = reshape(copy(v_raw[1:total]), nx, ny)

                u_mat[1, :] .= 0.0
                u_mat[end, :] .= 0.0
                u_mat[:, 1] .= 0.0
                u_mat[:, end] .= 0.0
                v_mat[1, :] .= 0.0
                v_mat[end, :] .= 0.0
                v_mat[:, 1] .= 0.0
                v_mat[:, end] .= 0.0

                uₕ = element(Wₕ, vec(u_mat))
                vₕ = element(Wₕ, vec(v_mat))

                lhs_x = innerₕ(Dcₓ(uₕ), vₕ)
                rhs_x = -innerₕ(uₕ, Dcₓ(vₕ))
                scale_x = max(abs(lhs_x), abs(rhs_x), 1.0)
                ok_x = isapprox(lhs_x, rhs_x; atol = 1e-10 * scale_x, rtol = 1e-10)

                lhs_y = innerₕ(Dcᵧ(uₕ), vₕ)
                rhs_y = -innerₕ(uₕ, Dcᵧ(vₕ))
                scale_y = max(abs(lhs_y), abs(rhs_y), 1.0)
                ok_y = isapprox(lhs_y, rhs_y; atol = 1e-10 * scale_y, rtol = 1e-10)

                ok_x && ok_y
            end
        end
    end

    @testset "Matrix agreement" begin
        # `diag(1/(hᵢ + hᵢ₊₁))` times `shift₊₁ - shift₋₁`, which is the stencil skipping
        # its own centre.
        test_operator_matrix_equivalence(centered_ops)

        Ωm = mesh(domain(interval(0.0, 1.0)), 7, false)
        @test Dcₓ(gridspace(Ωm)) == Dcₓ(Ωm)

        # both ends are truncated, so both end rows are empty
        n = npoints(Ωm)
        M = Matrix(Dcₓ(Ωm))
        @test all(iszero, M[1, :])
        @test all(iszero, M[n, :])

        # and a mesh too short for the stencil is refused, as the grid function form is
        @test_throws ArgumentError Dcₓ(mesh(domain(interval(0.0, 1.0)), 2, true))
    end
end
