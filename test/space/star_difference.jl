module SpaceStarDifferenceTests

using Test
using Bramble
# Internal since v3.0 (gpena/Bramble.jl#211): defined and documented, not exported.
import Bramble: D₊ₓ
using Random
using Supposition
using ..TestUtils: WITH_SLOW_TESTS
using Bramble: components, star_spacings, StarSpacings, submeshes
using ..TestUtils: alloc_test, @test_allocs, _nonuniform_points, _zero_boundary!
using ..SpaceDifferenceTests: test_operator_matrix_equivalence

# The starred forward difference and the identity it exists for.
#
#   D̃(uₕ)(i) = (u(x_{i+1}) - u(x_i)) / ((h_i + h_{i+1}) / 2)
#
# It is the forward difference over the averaged spacing rather than over the forward
# spacing, and it is the operator that makes the discrete integration by parts close:
#
#   innerₕ(D̃ₓ(uₕ), vₕ) == -inner₊ₓ(uₕ, D₋ₓ(vₕ))
#
# whenever vₕ vanishes on the boundary.

# The operators as matrices, for `test_operator_matrix_equivalence` (test/space/difference.jl).
star_ops(::Val{1}) = (D̃ₓ,)
star_ops(::Val{2}) = (D̃ₓ, D̃ᵧ)
star_ops(::Val{3}) = (D̃ₓ, D̃ᵧ, D̃₂)

@testset "Starred forward difference" begin
    @testset "Averaged spacing" begin
        for (lbl, unif) in (("uniform", true), ("random", false))
            @testset "$lbl" begin
                Random.seed!(20260830)
                Ωₕ = mesh(domain(interval(0.0, 1.0)), 9, unif)
                n = npoints(Ωₕ)
                hs = star_spacings(Ωₕ)

                @test hs isa StarSpacings
                @test length(hs) == n - 1
                @test all(
                    hs[i] ≈ (spacing(Ωₕ, i) + spacing(Ωₕ, i + 1)) / 2 for i in 1:(n - 1)
                )

                # away from the first point this is the width of the cell around xᵢ
                @test all(hs[i] ≈ half_spacing(Ωₕ, i) for i in 2:(n - 1))
                # at the first point it is not: the cached h₁ repeats the first interval,
                # so this gives x₂ - x₁ where the cell width gives half of it
                @test hs[1] ≈ 2 * half_spacing(Ωₕ, 1)
                @test hs[1] ≈ points(Ωₕ)[2] - points(Ωₕ)[1]
            end
        end
    end

    @testset "Definition match" begin
        for (lbl, unif) in (("uniform", true), ("random", false))
            @testset "$lbl" begin
                Random.seed!(20260830)
                Ωₕ = mesh(domain(interval(0.0, 1.0)), 11, unif)
                Wₕ = gridspace(Ωₕ)
                n = npoints(Ωₕ)
                uₕ = Rₕ(Wₕ, x -> x^2 + sin(x))
                u = parent(uₕ)

                want = [if i == n
                            0.0
                        else
                            (u[i + 1] - u[i]) / ((spacing(Ωₕ, i) + spacing(Ωₕ, i + 1)) / 2)
                        end
                        for i in 1:n]
                @test parent(D̃ₓ(uₕ)) ≈ want

                # the last point has no forward neighbour and is truncated, as in D₊ₓ
                @test parent(D̃ₓ(uₕ))[n] == 0.0
            end
        end
    end

    @testset "Exactness" begin
        # a constant differences to zero, and x differences to one, in every direction
        Ωₕ = mesh(
            domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (5, 6, 4), (true, true, true)
        )
        Wₕ = gridspace(Ωₕ)
        n = npoints(Ωₕ, Tuple)

        @test all(iszero, parent(D̃ₓ(Rₕ(Wₕ, x -> 3.0))))

        for (d, op) in ((1, D̃ₓ), (2, D̃ᵧ), (3, D̃₂))
            # a function constant along d differences to zero along d
            @test all(iszero, parent(op(Rₕ(Wₕ, x -> x[mod1(d + 1, 3)]))))
            # and one linear along d differences to one, away from the truncated slice
            r = reshape(parent(op(Rₕ(Wₕ, x -> x[d]))), n)
            interior = ntuple(k -> k == d ? (1:(n[k] - 1)) : (1:n[k]), 3)
            @test all(≈(1.0), r[interior...])
        end
    end

    @testset "Directional family" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 6), (true, false))
        Wₕ = gridspace(Ωₕ)
        Vₕ = gridspace(Ωₕ, Val(2))
        uₕ = Rₕ(Wₕ, x -> x[1] * x[2])

        @test D̃ₕ(uₕ) isa NTuple{2, VectorElement}
        @test parent(D̃ₕ(uₕ)[1]) == parent(D̃ₓ(uₕ))
        @test parent(D̃ₕ(uₕ)[2]) == parent(D̃ᵧ(uₕ))

        # in one dimension the tuple and the grid function coincide
        Ω1 = mesh(domain(interval(0.0, 1.0)), 7, true)
        u1 = Rₕ(gridspace(Ω1), sin)
        @test !(D̃ₕ(u1) isa Tuple)
        @test parent(D̃ₕ(u1)) == parent(D̃ₓ(u1))

        # composite grid functions apply componentwise, as the other operators do
        fs = (x -> x[1], x -> x[2]^2)
        cₕ = Rₕ(Vₕ, fs)
        scalars = (Rₕ(Wₕ, fs[1]), Rₕ(Wₕ, fs[2]))
        rₕ = D̃ₓ(cₕ)
        @test length(parent(rₕ)) == length(parent(cₕ))
        for k in 1:2
            @test parent(components(rₕ)[k]) == parent(D̃ₓ(scalars[k]))
        end
    end

    @testset "Type stability & allocations" begin
        Ωₕ1 = mesh(domain(interval(0.0, 1.0)), 33, false)
        Ωₕ2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 8), (true, false))
        u1 = Rₕ(gridspace(Ωₕ1), sin)
        u2 = Rₕ(gridspace(Ωₕ2), x -> x[1] * x[2])

        @test @inferred(D̃ₓ(u1)) isa VectorElement
        @test @inferred(D̃ᵧ(u2)) isa VectorElement
        @test @inferred(D̃ₕ(u2)) isa NTuple{2, VectorElement}
        @test @inferred(star_spacings(Ωₕ1)) isa StarSpacings

        # the denominator is a lazy view over the cached spacings, so it costs nothing
        @test_allocs star_spacings(Ωₕ1)
        @test alloc_test(D̃ₓ, u1) == alloc_test(similar, u1)
        @test alloc_test(D̃ᵧ, u2) == alloc_test(similar, u2)
    end

    @testset "Summation by parts" begin
        # innerₕ(D̃(uₕ), vₕ) == -inner₊(uₕ, D₋(vₕ)) when vₕ vanishes on the boundary.
        #
        # Only vₕ has to vanish: the boundary term of the discrete integration by parts
        # is the product of the two, so vₕ being zero there is enough. uₕ below is
        # deliberately non-zero on the boundary to pin that.
        #
        # Compared with an absolute floor as well as a relative one. Where uₕ happens not
        # to vary along the direction being differenced both sides are zero, and a purely
        # relative comparison reports a large error on two values of order 1e-17.
        agree(a, b) = isapprox(a, b; atol = 1e-12, rtol = 1e-12)

        @testset "1D" begin
            for (lbl, unif) in (("uniform", true), ("random", false)), n in (11, 51, 201)

                Random.seed!(20260830)
                Ωₕ = mesh(domain(interval(0.0, 1.0)), n, unif)
                Wₕ = gridspace(Ωₕ)
                uₕ = Rₕ(Wₕ, x -> cos(x) + 0.7)          # not zero at the boundary
                vₕ = Rₕ(Wₕ, x -> sin(pi * x))           # zero at both ends
                @test agree(innerₕ(D̃ₓ(uₕ), vₕ), -inner₊ₓ(uₕ, D₋ₓ(vₕ)))
            end
        end

        @testset "2D & 3D directions" begin
            u2 = x -> cos(x[1]) + 0.7 + 0.3x[2]^2 + 0.2x[1] * x[2]
            v2 = x -> sin(pi * x[1]) * sin(pi * x[2])
            u3 = x -> cos(x[1]) + 0.7 + 0.3x[2]^2 + 0.4x[3] + 0.2x[1] * x[3]
            v3 = x -> sin(pi * x[1]) * sin(pi * x[2]) * sin(pi * x[3])

            for unif in (true, false)
                Random.seed!(20260830)
                Ω2 = mesh(
                    domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (21, 19), (unif, unif)
                )
                W2 = gridspace(Ω2)
                a, b = Rₕ(W2, u2), Rₕ(W2, v2)
                @test agree(innerₕ(D̃ₓ(a), b), -inner₊ₓ(a, D₋ₓ(b)))
                @test agree(innerₕ(D̃ᵧ(a), b), -inner₊ᵧ(a, D₋ᵧ(b)))

                Ω3 = mesh(
                    domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))),
                    (11, 9, 8),
                    (unif, unif, unif)
                )
                W3 = gridspace(Ω3)
                c, d = Rₕ(W3, u3), Rₕ(W3, v3)
                @test agree(innerₕ(D̃ₓ(c), d), -inner₊ₓ(c, D₋ₓ(d)))
                @test agree(innerₕ(D̃ᵧ(c), d), -inner₊ᵧ(c, D₋ᵧ(d)))
                @test agree(innerₕ(D̃₂(c), d), -inner₊₂(c, D₋₂(d)))
            end
        end

        @testset "Boundary vanishing" begin
            Ωₕ = mesh(domain(interval(0.0, 1.0)), 101, true)
            Wₕ = gridspace(Ωₕ)
            zero_bdry = Rₕ(Wₕ, x -> sin(pi * x))
            nonzero = Rₕ(Wₕ, x -> cos(x) + 0.7)

            sbp(uₕ, vₕ) = agree(innerₕ(D̃ₓ(uₕ), vₕ), -inner₊ₓ(uₕ, D₋ₓ(vₕ)))

            @test sbp(zero_bdry, zero_bdry)
            @test sbp(nonzero, zero_bdry)      # uₕ need not vanish
            @test !sbp(zero_bdry, nonzero)     # vₕ must
            @test !sbp(nonzero, nonzero)
        end

        @testset "Vectorial form" begin
            # The same identity for a vector field, written through the tuple-valued
            # operators: the starred divergence as the sum of the directional starred
            # differences of the components, and the right-hand side as one call to the
            # tuple method of inner₊ against ∇ₕ(wₕ), which sums the directional inner
            # products. Only wₕ vanishes on the boundary, as in the componentwise form.
            div_star(vₕ::NTuple{2, VectorElement}) = D̃ₓ(vₕ[1]) + D̃ᵧ(vₕ[2])
            div_star(vₕ::NTuple{3, VectorElement}) = D̃ₓ(vₕ[1]) + D̃ᵧ(vₕ[2]) +
                                                     D̃₂(vₕ[3])

            Random.seed!(20260830)
            Ω1 = mesh(domain(interval(0.0, 1.0)), 41, false)
            W1 = gridspace(Ω1)
            v1 = Rₕ(W1, x -> cos(x) + 0.7)          # not zero at the boundary
            w1 = Rₕ(W1, x -> sin(pi * x))
            # in one dimension ∇ₕ is the grid function D₋ₓ gives, not a tuple
            @test agree(innerₕ(D̃ₕ(v1), w1), -inner₊(v1, ∇ₕ(w1)))

            Ω2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (17, 15), (false, false))
            W2 = gridspace(Ω2)
            # distinct per component, so a mix-up between the two cannot pass
            v2 = (Rₕ(W2, x -> cos(x[1]) + 0.7), Rₕ(W2, x -> x[2]^2 + 0.2x[1]))
            w2 = Rₕ(W2, x -> sin(pi * x[1]) * sin(pi * x[2]))
            @test agree(innerₕ(div_star(v2), w2), -inner₊(v2, ∇ₕ(w2)))

            Ω3 = mesh(
                domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (9, 8, 7), (false, false, false)
            )
            W3 = gridspace(Ω3)
            v3 = (
                Rₕ(W3, x -> cos(x[1]) + 0.7),
                Rₕ(W3, x -> x[2]^2 + 0.2x[1]),
                Rₕ(W3, x -> 0.4x[3] + x[1] * x[2])
            )
            w3 = Rₕ(W3, x -> sin(pi * x[1]) * sin(pi * x[2]) * sin(pi * x[3]))
            @test agree(innerₕ(div_star(v3), w3), -inner₊(v3, ∇ₕ(w3)))

            # and the tuple route gives what summing the directional inner products gives
            @test -inner₊(v3, ∇ₕ(w3)) ≈ -(inner₊ₓ(v3[1], D₋ₓ(w3)) +
                                          inner₊ᵧ(v3[2], D₋ᵧ(w3)) + inner₊₂(v3[3], D₋₂(w3)))
        end

        WITH_SLOW_TESTS && @testset "Random grids (Supposition)" begin
            positive_h = Data.Floats{Float64}(;
                minimum = 0.01, maximum = 10.0, nans = false, infs = false
            )
            field_val = Data.Floats{Float64}(;
                minimum = -100.0, maximum = 100.0, nans = false, infs = false
            )

            # 1D: arbitrary non-uniform mesh and unconstrained u vs boundary-vanishing v
            @check function check_sbp_1d(
                    h = Data.Vectors(positive_h; min_size = 2, max_size = 30),
                    u_raw = Data.Vectors(field_val; min_size = 31, max_size = 31),
                    v_raw = Data.Vectors(field_val; min_size = 31, max_size = 31)
            )
                pts = _nonuniform_points(h)
                n = length(pts)

                Ωₕ = mesh(domain(interval(0.0, 1.0)), n, false)
                set_points!(Ωₕ, pts)
                Wₕ = gridspace(Ωₕ)

                uₕ = element(Wₕ, copy(u_raw[1:n]))                      # unconstrained
                vₕ = element(Wₕ, _zero_boundary!(copy(v_raw[1:n])))

                lhs = innerₕ(D̃ₓ(uₕ), vₕ)
                rhs = -inner₊ₓ(uₕ, D₋ₓ(vₕ))
                scale = max(abs(lhs), abs(rhs), 1.0)
                isapprox(lhs, rhs; atol = 1e-10 * scale, rtol = 1e-10)
            end

            # 2D: arbitrary non-uniform tensor product mesh and fields across coordinates
            @check function check_sbp_2d(
                    hx = Data.Vectors(positive_h; min_size = 2, max_size = 8),
                    hy = Data.Vectors(positive_h; min_size = 2, max_size = 8),
                    u_raw = Data.Vectors(field_val; min_size = 81, max_size = 81),
                    v_raw = Data.Vectors(field_val; min_size = 81, max_size = 81)
            )
                pts_x = _nonuniform_points(hx)
                pts_y = _nonuniform_points(hy)
                nx, ny = length(pts_x), length(pts_y)

                Ωₕ = mesh(
                    domain(interval(0.0, 1.0) × interval(0.0, 1.0)),
                    (nx, ny),
                    (false, false)
                )
                set_points!(Ωₕ(1), pts_x)
                set_points!(Ωₕ(2), pts_y)
                Wₕ = gridspace(Ωₕ)

                total = nx * ny
                grid(raw) = reshape(copy(raw[1:total]), nx, ny)

                uₕ = element(Wₕ, vec(grid(u_raw)))                       # unconstrained
                vₕ = element(Wₕ, vec(_zero_boundary!(grid(v_raw))))

                lhs_x = innerₕ(D̃ₓ(uₕ), vₕ)
                rhs_x = -inner₊ₓ(uₕ, D₋ₓ(vₕ))
                scale_x = max(abs(lhs_x), abs(rhs_x), 1.0)
                ok_x = isapprox(lhs_x, rhs_x; atol = 1e-10 * scale_x, rtol = 1e-10)

                lhs_y = innerₕ(D̃ᵧ(uₕ), vₕ)
                rhs_y = -inner₊ᵧ(uₕ, D₋ᵧ(vₕ))
                scale_y = max(abs(lhs_y), abs(rhs_y), 1.0)
                ok_y = isapprox(lhs_y, rhs_y; atol = 1e-10 * scale_y, rtol = 1e-10)

                ok_x && ok_y
            end

            # 3D: arbitrary non-uniform tensor product mesh and fields across coordinates.
            # Axis sizes kept smaller than the 2D check's (max 5 intervals, not 8) so the
            # total point count (up to 6³ = 216) stays a fast random search.
            @check function check_sbp_3d(
                    hx = Data.Vectors(positive_h; min_size = 2, max_size = 5),
                    hy = Data.Vectors(positive_h; min_size = 2, max_size = 5),
                    hz = Data.Vectors(positive_h; min_size = 2, max_size = 5),
                    u_raw = Data.Vectors(field_val; min_size = 216, max_size = 216),
                    v_raw = Data.Vectors(field_val; min_size = 216, max_size = 216)
            )
                pts_x = _nonuniform_points(hx)
                pts_y = _nonuniform_points(hy)
                pts_z = _nonuniform_points(hz)
                nx, ny, nz = length(pts_x), length(pts_y), length(pts_z)

                Ωₕ = mesh(
                    domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))),
                    (nx, ny, nz),
                    (false, false, false)
                )
                set_points!(Ωₕ(1), pts_x)
                set_points!(Ωₕ(2), pts_y)
                set_points!(Ωₕ(3), pts_z)
                Wₕ = gridspace(Ωₕ)

                total = nx * ny * nz
                grid(raw) = reshape(copy(raw[1:total]), nx, ny, nz)

                uₕ = element(Wₕ, vec(grid(u_raw)))                       # unconstrained
                vₕ = element(Wₕ, vec(_zero_boundary!(grid(v_raw))))

                lhs_x = innerₕ(D̃ₓ(uₕ), vₕ)
                rhs_x = -inner₊ₓ(uₕ, D₋ₓ(vₕ))
                scale_x = max(abs(lhs_x), abs(rhs_x), 1.0)
                ok_x = isapprox(lhs_x, rhs_x; atol = 1e-10 * scale_x, rtol = 1e-10)

                lhs_y = innerₕ(D̃ᵧ(uₕ), vₕ)
                rhs_y = -inner₊ᵧ(uₕ, D₋ᵧ(vₕ))
                scale_y = max(abs(lhs_y), abs(rhs_y), 1.0)
                ok_y = isapprox(lhs_y, rhs_y; atol = 1e-10 * scale_y, rtol = 1e-10)

                lhs_z = innerₕ(D̃₂(uₕ), vₕ)
                rhs_z = -inner₊₂(uₕ, D₋₂(vₕ))
                scale_z = max(abs(lhs_z), abs(rhs_z), 1.0)
                ok_z = isapprox(lhs_z, rhs_z; atol = 1e-10 * scale_z, rtol = 1e-10)

                ok_x && ok_y && ok_z
            end

            # 2D vector field: the vectorial statement of the identity on a random mesh,
            # with the two components drawn independently. Goes through inner₊'s tuple
            # method and ∇ₕ rather than the directional calls the checks above make.
            @check function check_sbp_vectorial_2d(
                    hx = Data.Vectors(positive_h; min_size = 2, max_size = 8),
                    hy = Data.Vectors(positive_h; min_size = 2, max_size = 8),
                    vx_raw = Data.Vectors(field_val; min_size = 81, max_size = 81),
                    vy_raw = Data.Vectors(field_val; min_size = 81, max_size = 81),
                    w_raw = Data.Vectors(field_val; min_size = 81, max_size = 81)
            )
                pts_x = _nonuniform_points(hx)
                pts_y = _nonuniform_points(hy)
                nx, ny = length(pts_x), length(pts_y)

                Ωₕ = mesh(
                    domain(interval(0.0, 1.0) × interval(0.0, 1.0)),
                    (nx, ny),
                    (false, false)
                )
                set_points!(Ωₕ(1), pts_x)
                set_points!(Ωₕ(2), pts_y)
                Wₕ = gridspace(Ωₕ)

                total = nx * ny
                grid(raw) = reshape(copy(raw[1:total]), nx, ny)

                vₕ = (
                    element(Wₕ, vec(grid(vx_raw))),                      # unconstrained
                    element(Wₕ, vec(grid(vy_raw)))
                )
                wₕ = element(Wₕ, vec(_zero_boundary!(grid(w_raw))))

                lhs = innerₕ(D̃ₓ(vₕ[1]) + D̃ᵧ(vₕ[2]), wₕ)
                rhs = -inner₊(vₕ, ∇ₕ(wₕ))
                scale = max(abs(lhs), abs(rhs), 1.0)
                isapprox(lhs, rhs; atol = 1e-10 * scale, rtol = 1e-10)
            end
        end
    end

    @testset "Matrix agreement" begin
        # This family had no matrix form until the three centred ones were given one. It is
        # a diagonal scaling of the unscaled forward difference (specifically `diag(2/(hᵢ + hᵢ₊₁))`
        # times it), so the two routes have to give the same numbers.
        test_operator_matrix_equivalence(star_ops)

        Ωm = mesh(domain(interval(0.0, 1.0)), 7, false)
        @test D̃ₓ(gridspace(Ωm)) == D̃ₓ(Ωm)     # a space answers as its mesh does

        # the truncated point is an empty row, matching the zero the grid function gets
        n = npoints(Ωm)
        @test all(iszero, Matrix(D̃ₓ(Ωm))[n, :])
    end
end

end # module SpaceStarDifferenceTests
