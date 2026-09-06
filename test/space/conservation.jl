using Test
using Bramble

# Discrete conservation (Gauss/divergence theorem): for a vector field vanishing on ∂Ω,
# the total discrete divergence over the mesh must be zero.
#
# `cell_measure` is NOT the companion weight for this identity: `inner₊ₓ`'s own internal
# weight is the plain backward spacing (zeroed at the truncated index), which is what
# `D₋ₓ` was divided by in the first place (see the comment above `D₋(u)ᵢ = diff₋(u)ᵢ/hᵢ`
# in difference.jl and the SBP identity in star_difference.jl). So
#
#   Σ inner₊ₓ-weight(i) * D₋ₓ(u)(i) = u(nx) - u(1)     (telescoping, exact)
#
# which vanishes when u is zero at both x-boundaries, and likewise per direction. Summing
# the per-direction telescoped fluxes is the discrete divergence theorem; substituting
# `cell_measure` for `inner₊ₓ`'s weight breaks the telescope (it uses the averaged/star
# spacing, not the plain one `D₋ₓ` divides by) and would not catch the metric/stencil
# mismatch this test exists for.

@testset "Discrete conservation" begin
    agree(a, b) = isapprox(a, b; atol = 1e-12, rtol = 1e-12)

    @testset "2D" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (23, 19), (true, false))
        Wₕ = gridspace(Ωₕ)
        onesₕ = Rₕ(Wₕ, x -> 1.0)

        # u vanishes at x = 0, 1 (any y); v vanishes at y = 0, 1 (any x) -- a "no normal
        # flow" boundary condition, not merely a divergence-free field, since it is the
        # vanishing that makes the flux telescope to zero, not the field being solenoidal.
        u = Rₕ(Wₕ, x -> sin(pi * x[1]) * (1.0 + 2.0 * cos(pi * x[2])))
        v = Rₕ(Wₕ, x -> sin(pi * x[2]) * (1.0 + 2.0 * cos(pi * x[1])))

        flux = inner₊ₓ(D₋ₓ(u), onesₕ) + inner₊ᵧ(D₋ᵧ(v), onesₕ)
        @test agree(flux, 0.0)

        # A field NOT vanishing on ∂Ω must not accidentally pass: this is the control that
        # confirms the identity is testing the boundary condition, not trivially zero.
        u_bad = Rₕ(Wₕ, x -> cos(pi * x[1]))   # nonzero at x = 0, 1
        flux_bad = inner₊ₓ(D₋ₓ(u_bad), onesₕ) + inner₊ᵧ(D₋ᵧ(v), onesₕ)
        @test !isapprox(flux_bad, 0.0; atol = 1e-8)
    end

    @testset "3D" begin
        Ωₕ = mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (11, 9, 8),
            (true, false, true))
        Wₕ = gridspace(Ωₕ)
        onesₕ = Rₕ(Wₕ, x -> 1.0)

        u = Rₕ(Wₕ, x -> sin(pi * x[1]) * (2.0 + cos(pi * x[2]) + 0.5cos(pi * x[3])))
        v = Rₕ(Wₕ, x -> sin(pi * x[2]) * (2.0 + cos(pi * x[1]) + 0.5cos(pi * x[3])))
        w = Rₕ(Wₕ, x -> sin(pi * x[3]) * (2.0 + cos(pi * x[1]) + 0.5cos(pi * x[2])))

        flux = inner₊ₓ(D₋ₓ(u), onesₕ) + inner₊ᵧ(D₋ᵧ(v), onesₕ) + inner₊₂(D₋₂(w), onesₕ)
        @test agree(flux, 0.0)
    end
end
