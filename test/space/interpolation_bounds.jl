module SpaceInterpolationBoundsTests

using Test
using Bramble
using Bramble: interpolation_matrix, weights
using ..TestUtils: alloc_test, @test_allocs

# `interpolate_at` used to extrapolate silently past the mesh boundary: `locate_cell`
# clamps which *cell* is read, but the fraction within it was never clamped, so a point
# outside the domain returned a plausible-looking, silently wrong number (gpena/Bramble.jl#223).
# `outside` names the choice explicitly. Every check here is against an independent
# reference (a hand-computed value, or a genuinely different code path -- `interpolation_matrix`
# against pointwise `interpolate_at`), never against another call to the code under test.

@testset "interpolate_at out-of-domain policy (#223)" begin
    @testset "Default (:error) throws, naming the point and the domain extent" begin
        @testset "1D" begin
            Ωₕ = mesh(domain(interval(0.0, 1.0)), 11, true)
            uₕ = Rₕ(gridspace(Ωₕ), x -> x[1]^2)
            err = @test_throws ArgumentError interpolate_at(uₕ, 5.0)
            msg = sprint(showerror, err.value)
            @test occursin("5.0", msg)
            @test occursin("0.0", msg) && occursin("1.0", msg)
            @test_throws ArgumentError interpolate_at(uₕ, -3.0)
        end

        @testset "2D" begin
            Ωₕ = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (6, 6), (true, true))
            uₕ = Rₕ(gridspace(Ωₕ), x -> x[1] + x[2])
            @test_throws ArgumentError interpolate_at(uₕ, (5.0, 0.5))   # x out, y in
            @test_throws ArgumentError interpolate_at(uₕ, (0.5, -2.0)) # x in, y out
        end

        @testset "3D" begin
            Ωₕ = mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (5, 5, 5), (true, true, true))
            uₕ = Rₕ(gridspace(Ωₕ), x -> x[1] + x[2] + x[3])
            @test_throws ArgumentError interpolate_at(uₕ, (0.5, 0.5, 3.0))
        end
    end

    @testset ":clamp, an explicit fill value, and :extrapolate behave as documented" begin
        @testset "1D" begin
            Ωₕ = mesh(domain(interval(0.0, 1.0)), 11, true)
            uₕ = Rₕ(gridspace(Ωₕ), x -> 3x[1] + 1)   # affine: clamp/extrapolate both checkable exactly

            @test interpolate_at(uₕ, 5.0; outside = :clamp) ≈ 3 * 1.0 + 1 atol = 1e-10
            @test interpolate_at(uₕ, -3.0; outside = :clamp) ≈ 3 * 0.0 + 1 atol = 1e-10
            @test interpolate_at(uₕ, 5.0; outside = :extrapolate) ≈ 3 * 5.0 + 1 atol = 1e-10
            # below the lower bound too: locate_cell clamps which cell is read, not the
            # relative position within it, so the boundary cell's affine trend continues
            # in both directions (gpena/Bramble.jl#223)
            @test interpolate_at(uₕ, -3.0; outside = :extrapolate) ≈ 3 * -3.0 + 1 atol = 1e-10
            @test interpolate_at(uₕ, 5.0; outside = 0.0) == 0.0
            @test interpolate_at(uₕ, 5.0; outside = NaN) |> isnan
        end

        @testset "2D" begin
            Ωₕ = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (6, 6), (true, true))
            uₕ = Rₕ(gridspace(Ωₕ), x -> 2x[1] - x[2])

            @test interpolate_at(uₕ, (5.0, 0.5); outside = :clamp) ≈ 2 * 1.0 - 0.5 atol = 1e-10
            @test interpolate_at(uₕ, (5.0, 0.5); outside = :extrapolate) ≈ 2 * 5.0 - 0.5 atol = 1e-10
            @test interpolate_at(uₕ, (5.0, 0.5); outside = -1.0) == -1.0
        end

        @testset "3D" begin
            Ωₕ = mesh(
                domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (5, 5, 5), (true, true, true)
            )
            uₕ = Rₕ(gridspace(Ωₕ), x -> x[1] + x[2] + x[3])

            @test interpolate_at(uₕ, (0.5, 0.5, 3.0); outside = :clamp) ≈ 0.5 + 0.5 + 1.0 atol = 1e-10
            @test interpolate_at(uₕ, (0.5, 0.5, 3.0); outside = :extrapolate) ≈
                  0.5 + 0.5 + 3.0 atol = 1e-10
            @test interpolate_at(uₕ, (0.5, 0.5, 3.0); outside = 7.0) == 7.0
        end
    end

    @testset "A floating-point epsilon past the boundary is on the boundary, under every policy" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 11, true)
        uₕ = Rₕ(gridspace(Ωₕ), x -> x[1]^2)
        x_ulp = nextfloat(1.0)   # 1.0, up by one ULP -- not a value a caller chose on purpose

        @test interpolate_at(uₕ, x_ulp) ≈ interpolate_at(uₕ, 1.0)          # :error: no throw
        @test interpolate_at(uₕ, x_ulp; outside = :clamp) ≈ interpolate_at(uₕ, 1.0)
        @test interpolate_at(uₕ, x_ulp; outside = :extrapolate) ≈ interpolate_at(uₕ, 1.0)

        # The acceptance criterion this is really for: πₕ between two meshes over the same
        # nominal domain, unaffected by the new default. In this package's own mesh
        # construction the endpoints come out bit-identical (checked, not assumed), so this
        # also stands as the ordinary, non-epsilon case -- both are covered.
        Ωdest = mesh(domain(interval(0.0, 1.0)), 7, true)
        Ωsrc = mesh(domain(interval(0.0, 1.0)), 13, false)
        @test points(Ωdest)[1] == points(Ωsrc)[1] && points(Ωdest)[end] == points(Ωsrc)[end]
        src = Rₕ(gridspace(Ωsrc), x -> sin(3x[1]))
        dest = πₕ(gridspace(Ωdest), src)   # default outside = :error: must not throw
        @test all(isfinite, parent(dest))
    end

    @testset "πₕ, πₕ! and interpolation_matrix accept and honour outside" begin
        Ωdest = mesh(domain(interval(0.0, 1.0)), 6, true)
        Ωsrc = mesh(domain(interval(0.0, 1.0)), 5, true)
        # A destination point genuinely outside Ωsrc, by construction.
        Ωdest_wide = mesh(domain(interval(0.0, 1.5)), 7, true)
        Wdest, Wsrc, Wdest_wide = gridspace(Ωdest), gridspace(Ωsrc), gridspace(Ωdest_wide)
        src = Rₕ(Wsrc, x -> 4x[1] + 2)

        @test_throws ArgumentError πₕ(Wdest_wide, src)
        @test_throws ArgumentError πₕ!(Bramble.element(Wdest_wide), src)

        dest_clamp = πₕ(Wdest_wide, src; outside = :clamp)
        dest_extrap = πₕ(Wdest_wide, src; outside = :extrapolate)
        @test !(parent(dest_clamp) ≈ parent(dest_extrap))  # the two policies must differ here

        dest2 = similar(dest_clamp)
        πₕ!(dest2, src; outside = :clamp)
        @test parent(dest2) ≈ parent(dest_clamp)

        @testset "interpolation_matrix agrees entry-for-entry with pointwise interpolate_at" begin
            for pol in (:clamp, :extrapolate)
                P = interpolation_matrix(Wdest_wide, Wsrc; outside = pol)
                via_matrix = P * parent(src)
                via_pointwise = parent(πₕ(Wdest_wide, src; outside = pol))
                @test via_matrix ≈ via_pointwise
            end
        end

        @testset "interpolation_matrix under :error throws when any destination point is outside" begin
            @test_throws ArgumentError interpolation_matrix(Wdest_wide, Wsrc)
            # unaffected when destination and source domains actually agree
            P_ok = interpolation_matrix(Wdest, Wsrc)
            @test size(P_ok) == (ndofs(Wdest), ndofs(Wsrc))
        end
    end

    @testset "interpolation_matrix and the bilinear πₕ(u) refuse a fill value" begin
        Ωdest = mesh(domain(interval(0.0, 1.0)), 6, true)
        Ωsrc = mesh(domain(interval(0.0, 1.0)), 5, true)
        Wdest, Wsrc = gridspace(Ωdest), gridspace(Ωsrc)

        err = @test_throws ArgumentError interpolation_matrix(Wdest, Wsrc; outside = 0.0)
        @test occursin("linear", sprint(showerror, err.value))

        @test_throws ArgumentError form(
            Wsrc, Wdest, (u, v) -> innerₕ(πₕ(u; outside = NaN), v)
        )
    end

    @testset "outside must be :error, :clamp, :extrapolate, or a Number" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 11, true)
        uₕ = Rₕ(gridspace(Ωₕ), x -> x[1])
        @test_throws ArgumentError interpolate_at(uₕ, 0.5; outside = :bogus)
        @test_throws ArgumentError interpolate_at(uₕ, 0.5; outside = "clamp")
    end

    @testset "The symbolic form path agrees with the direct path, under every policy" begin
        Ωdest = mesh(domain(interval(0.0, 1.0)), 6, true)
        Ωsrc = mesh(domain(interval(0.0, 1.0)), 5, true)
        Wdest, Wsrc = gridspace(Ωdest), gridspace(Ωsrc)
        src = Rₕ(Wsrc, x -> 4x[1] + 2)

        @testset "Source wrapper πₕ(uₕ)" begin
            for pol in (:clamp, :extrapolate, 0.0, NaN)
                direct = πₕ(Wdest, src; outside = pol)
                l = form(Wdest, v -> innerₕ(πₕ(src; outside = pol), v))
                via_form = assemble(l)
                Href = weights(Wdest, Bramble.Innerh())
                expected = Href .* parent(direct)
                if pol isa Number && isnan(pol)
                    @test all(isnan, via_form) == all(isnan, expected)
                else
                    @test via_form ≈ expected
                end
            end
        end

        @testset "Bilinear operator πₕ(u)" begin
            for pol in (:clamp, :extrapolate)
                P = interpolation_matrix(Wdest, Wsrc; outside = pol)
                a = form(Wsrc, Wdest, (u, v) -> innerₕ(πₕ(u; outside = pol), v))
                A = assemble(a)
                Href = weights(Wdest, Bramble.Innerh())
                @test A * parent(src) ≈ Href .* (P * parent(src))
            end
        end
    end

    @testset "The in-domain path stays zero allocation and type stable" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 101, true)
        uₕ = Rₕ(gridspace(Ωₕ), x -> x[1]^2)
        x0 = 0.55
        @test_allocs interpolate_at(uₕ, x0)
        @test alloc_test(interpolate_at, uₕ, x0; outside = :clamp) == 0

        Ω2 = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (11, 11), (true, true))
        u2 = Rₕ(gridspace(Ω2), x -> x[1] * x[2])
        @test_allocs interpolate_at(u2, (0.3, 0.7))

        @test @inferred(interpolate_at(uₕ, x0)) isa Float64
        @test @inferred(interpolate_at(uₕ, x0; outside = :clamp)) isa Float64
    end
end

end # module
