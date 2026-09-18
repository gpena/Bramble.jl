module SpaceSbpIdentitiesTests

using Test
using Bramble
# Internal since v3.0 (gpena/Bramble.jl#211): defined and documented, not exported.
import Bramble: M₊ₓ
using Random
using Supposition
using ..TestUtils: WITH_SLOW_TESTS
using ..TestUtils: _nonuniform_points, _zero_boundary!

# Discrete integration by parts for the centered divergence.
#
#   innerₕ(Dcₓ(uₓ) + Dcᵧ(u_y) + Dc₂(u_z), vₕ)
#       == -(inner₊ₓ(Mₓ(uₓ), D₋ₓ(vₕ)) + inner₊ᵧ(Mᵧ(u_y), D₋ᵧ(vₕ)) + inner₊₂(M₂(u_z), D₋₂(vₕ)))
#
# The centered divergence of a vector field pairs with the backward gradient of a scalar
# through the *backward* average, not the forward one. The reason is indexing: inner₊ₓ
# weights index i by the backward spacing hᵢ, and D₋ₓ(vₕ)(i) = (vᵢ - vᵢ₋₁)/hᵢ reads the
# interval [xᵢ₋₁, xᵢ], so the average that sits on that same interval is
# Mₓ(uₓ)(i) = (uᵢ₋₁ + uᵢ)/2. The literature writes this operator as Mₕ on the dual grid,
# where the same average carries the half index i - 1/2; with Bramble's whole-index
# convention that is M, and pairing M₊ instead shifts one factor by a cell and breaks the
# identity (pinned below in "Forward average does not close it").
#
# Only vₕ has to vanish on the boundary. The identity is stated for uₕ ∈ [V_{H,0}]^D and
# vₕ ∈ V_{H,0}, and that is what the Supposition checks generate, but the boundary term
# that the telescoping leaves behind is a product of the two, so vₕ alone carries it --
# "Boundary vanishing" below pins that sharper version, as star_difference.jl does for its
# own identity.
#
# Componentwise checks of the two identities this one is built out of live with their
# operators: the starred divergence against the backward gradient in star_difference.jl
# ("Summation by parts"), and the skew-symmetry of Dc in centered_difference.jl.

@testset "Centered divergence integration by parts" begin
    # Compared with an absolute floor as well as a relative one: on fields that happen not
    # to vary along a direction both sides are zero up to rounding, where a purely relative
    # comparison reports a large error on two values of order 1e-17.
    agree(a, b) = isapprox(a, b; atol = 1e-12, rtol = 1e-12)

    # The identity, per direction and summed, for a vector field given componentwise.
    divergence_ibp(uₕ::VectorElement, vₕ) = (innerₕ(Dcₓ(uₕ), vₕ), -inner₊ₓ(Mₓ(uₕ), D₋ₓ(vₕ)))

    function divergence_ibp(uₕ::NTuple{2, VectorElement}, vₕ)
        lhs = innerₕ(Dcₓ(uₕ[1]) + Dcᵧ(uₕ[2]), vₕ)
        rhs = -(inner₊ₓ(Mₓ(uₕ[1]), D₋ₓ(vₕ)) + inner₊ᵧ(Mᵧ(uₕ[2]), D₋ᵧ(vₕ)))
        return (lhs, rhs)
    end

    function divergence_ibp(uₕ::NTuple{3, VectorElement}, vₕ)
        lhs = innerₕ(Dcₓ(uₕ[1]) + Dcᵧ(uₕ[2]) + Dc₂(uₕ[3]), vₕ)
        rhs = -(inner₊ₓ(Mₓ(uₕ[1]), D₋ₓ(vₕ)) + inner₊ᵧ(Mᵧ(uₕ[2]), D₋ᵧ(vₕ)) +
                inner₊₂(M₂(uₕ[3]), D₋₂(vₕ)))
        return (lhs, rhs)
    end

    @testset "1D" begin
        for (lbl, unif) in (("uniform", true), ("random", false)), n in (11, 51, 201)

            @testset "$lbl, $n points" begin
                Random.seed!(20260913)
                Ωₕ = mesh(domain(interval(0.0, 1.0)), n, unif)
                Wₕ = gridspace(Ωₕ)
                uₕ = Rₕ(Wₕ, x -> sin(pi * x) * (1 + x))     # zero at both ends
                vₕ = Rₕ(Wₕ, x -> sin(2pi * x) * x * (1 - x))

                @test agree(divergence_ibp(uₕ, vₕ)...)
            end
        end
    end

    @testset "2D & 3D" begin
        for unif in (true, false)
            @testset "$(unif ? "uniform" : "random")" begin
                Random.seed!(20260913)
                Ω2 = mesh(
                    domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (21, 19), (unif, unif)
                )
                W2 = gridspace(Ω2)
                b2 = x -> sin(pi * x[1]) * sin(pi * x[2])
                # distinct per component, so a mix-up between the two cannot pass
                u2 = (Rₕ(W2, x -> b2(x) * (1 + x[1])), Rₕ(W2, x -> b2(x) * (2 - x[2]^2)))
                v2 = Rₕ(W2, x -> b2(x) * sin(2pi * x[1]))

                @test agree(divergence_ibp(u2, v2)...)

                Ω3 = mesh(
                    domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))),
                    (11, 9, 8),
                    (unif, unif, unif)
                )
                W3 = gridspace(Ω3)
                b3 = x -> sin(pi * x[1]) * sin(pi * x[2]) * sin(pi * x[3])
                u3 = (
                    Rₕ(W3, x -> b3(x) * (1 + x[1])),
                    Rₕ(W3, x -> b3(x) * (2 - x[2]^2)),
                    Rₕ(W3, x -> b3(x) * (3 + x[3] * x[1]))
                )
                v3 = Rₕ(W3, x -> b3(x) * sin(2pi * x[2]))

                @test agree(divergence_ibp(u3, v3)...)
            end
        end
    end

    @testset "Vectorial form" begin
        # The same identity written through the tuple-valued operators: ∇ₕ for the
        # backward gradient and the tuple method of inner₊, which sums the directional
        # inner products. The right-hand side is then one call rather than a sum of D of
        # them, and it goes through inner₊'s generated tuple path instead of the scalar one.
        Random.seed!(20260913)
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (17, 15), (false, false))
        Wₕ = gridspace(Ωₕ)
        b = x -> sin(pi * x[1]) * sin(pi * x[2])
        uₕ = (Rₕ(Wₕ, x -> b(x) * (1 + x[1])), Rₕ(Wₕ, x -> b(x) * (2 - x[2]^2)))
        vₕ = Rₕ(Wₕ, x -> b(x) * sin(2pi * x[1]))

        lhs = innerₕ(Dcₓ(uₕ[1]) + Dcᵧ(uₕ[2]), vₕ)
        rhs = -inner₊((Mₓ(uₕ[1]), Mᵧ(uₕ[2])), ∇ₕ(vₕ))

        @test agree(lhs, rhs)
        # and it is the same number the componentwise form gives
        @test rhs ≈ divergence_ibp(uₕ, vₕ)[2]
    end

    @testset "Boundary vanishing" begin
        Random.seed!(20260913)
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 41, false)
        Wₕ = gridspace(Ωₕ)
        zero_bdry = Rₕ(Wₕ, x -> sin(pi * x) * (1 + x))
        nonzero = Rₕ(Wₕ, x -> cos(x) + 0.7)

        ibp(uₕ, vₕ) = agree(divergence_ibp(uₕ, vₕ)...)

        @test ibp(zero_bdry, zero_bdry)
        @test ibp(nonzero, zero_bdry)      # uₕ need not vanish
        @test !ibp(zero_bdry, nonzero)     # vₕ must
        @test !ibp(nonzero, nonzero)
    end

    @testset "Forward average does not close it" begin
        # The control for the M/M₊ convention above: substituting the forward average
        # shifts one factor by a cell, and the identity fails even on fields vanishing on
        # the whole boundary.
        Random.seed!(20260913)
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 41, false)
        Wₕ = gridspace(Ωₕ)
        uₕ = Rₕ(Wₕ, x -> sin(pi * x) * (1 + x))
        vₕ = Rₕ(Wₕ, x -> sin(2pi * x) * x * (1 - x))

        lhs, rhs = divergence_ibp(uₕ, vₕ)
        @test agree(lhs, rhs)
        @test !agree(lhs, -inner₊ₓ(M₊ₓ(uₕ), D₋ₓ(vₕ)))
    end

    WITH_SLOW_TESTS && @testset "Random grids (Supposition)" begin
        positive_h = Data.Floats{Float64}(;
            minimum = 0.01, maximum = 10.0, nans = false, infs = false
        )
        field_val = Data.Floats{Float64}(;
            minimum = -100.0, maximum = 100.0, nans = false, infs = false
        )

        # The comparison the checks use: scale-normalised, because a random grid can put
        # several orders of magnitude between the smallest and the largest cell and the
        # two sides are then sums of terms much larger than their difference.
        scaled_agree(lhs, rhs) = isapprox(
            lhs, rhs; atol = 1e-10 * max(abs(lhs), abs(rhs), 1.0), rtol = 1e-10
        )

        # 1D: arbitrary non-uniform mesh, both fields vanishing at the two ends. The mesh
        # needs at least three points for Dcₓ, hence min_size = 3 on the spacings.
        @check function check_divergence_ibp_1d(
                h = Data.Vectors(positive_h; min_size = 3, max_size = 30),
                u_raw = Data.Vectors(field_val; min_size = 31, max_size = 31),
                v_raw = Data.Vectors(field_val; min_size = 31, max_size = 31)
        )
            pts = _nonuniform_points(h)
            n = length(pts)

            Ωₕ = mesh(domain(interval(0.0, 1.0)), n, false)
            set_points!(Ωₕ, pts)
            Wₕ = gridspace(Ωₕ)

            uₕ = element(Wₕ, _zero_boundary!(copy(u_raw[1:n])))
            vₕ = element(Wₕ, _zero_boundary!(copy(v_raw[1:n])))

            lhs = innerₕ(Dcₓ(uₕ), vₕ)
            rhs = -inner₊ₓ(Mₓ(uₕ), D₋ₓ(vₕ))
            scaled_agree(lhs, rhs)
        end

        # 2D: arbitrary non-uniform tensor product mesh, with the two components of the
        # vector field drawn independently so a mix-up between them cannot pass
        @check function check_divergence_ibp_2d(
                hx = Data.Vectors(positive_h; min_size = 3, max_size = 8),
                hy = Data.Vectors(positive_h; min_size = 3, max_size = 8),
                ux_raw = Data.Vectors(field_val; min_size = 81, max_size = 81),
                uy_raw = Data.Vectors(field_val; min_size = 81, max_size = 81),
                v_raw = Data.Vectors(field_val; min_size = 81, max_size = 81)
        )
            pts_x = _nonuniform_points(hx)
            pts_y = _nonuniform_points(hy)
            nx, ny = length(pts_x), length(pts_y)

            Ωₕ = mesh(
                domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (nx, ny), (false, false)
            )
            set_points!(Ωₕ(1), pts_x)
            set_points!(Ωₕ(2), pts_y)
            Wₕ = gridspace(Ωₕ)

            total = nx * ny
            field(raw) = element(
                Wₕ, vec(_zero_boundary!(reshape(copy(raw[1:total]), nx, ny)))
            )
            uₕ = (field(ux_raw), field(uy_raw))
            vₕ = field(v_raw)

            lhs = innerₕ(Dcₓ(uₕ[1]) + Dcᵧ(uₕ[2]), vₕ)
            rhs = -(inner₊ₓ(Mₓ(uₕ[1]), D₋ₓ(vₕ)) + inner₊ᵧ(Mᵧ(uₕ[2]), D₋ᵧ(vₕ)))
            scaled_agree(lhs, rhs)
        end

        # 3D: axis sizes kept smaller than the 2D check's (max 5 intervals, not 8) so the
        # total point count (up to 6³ = 216) stays a fast random search, as in
        # star_difference.jl's 3D check
        @check function check_divergence_ibp_3d(
                hx = Data.Vectors(positive_h; min_size = 3, max_size = 5),
                hy = Data.Vectors(positive_h; min_size = 3, max_size = 5),
                hz = Data.Vectors(positive_h; min_size = 3, max_size = 5),
                ux_raw = Data.Vectors(field_val; min_size = 216, max_size = 216),
                uy_raw = Data.Vectors(field_val; min_size = 216, max_size = 216),
                uz_raw = Data.Vectors(field_val; min_size = 216, max_size = 216),
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
            field(raw) = element(
                Wₕ, vec(_zero_boundary!(reshape(copy(raw[1:total]), nx, ny, nz)))
            )
            uₕ = (field(ux_raw), field(uy_raw), field(uz_raw))
            vₕ = field(v_raw)

            lhs = innerₕ(Dcₓ(uₕ[1]) + Dcᵧ(uₕ[2]) + Dc₂(uₕ[3]), vₕ)
            rhs = -(inner₊ₓ(Mₓ(uₕ[1]), D₋ₓ(vₕ)) + inner₊ᵧ(Mᵧ(uₕ[2]), D₋ᵧ(vₕ)) +
                    inner₊₂(M₂(uₕ[3]), D₋₂(vₕ)))
            scaled_agree(lhs, rhs)
        end
    end
end

end # module SpaceSbpIdentitiesTests
