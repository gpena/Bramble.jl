module SpaceInterpolationBoundsTests

using Test
using Bramble
using Bramble: interpolation_matrix, locate_cell, weights
using ..TestUtils: alloc_test, @test_allocs

# `interpolate_at` used to extrapolate silently past the mesh boundary: `locate_cell`
# clamps which *cell* is read, but the fraction within it was never clamped, so a point
# outside the domain returned a plausible-looking, silently wrong number (gpena/Bramble.jl#223).
# `outside` names the choice explicitly. Every check here is against an independent
# reference (a hand-computed value, or a genuinely different code path -- `interpolation_matrix`
# against pointwise `interpolate_at`), never against another call to the code under test.

@testset "interpolate_at outside policy (#223)" begin
    # The default policy throws, naming the point and the domain extent.
    @testset "Default :error names point and extent" begin
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

    # Each behaves as documented.
    @testset ":clamp, fill value, :extrapolate" begin
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
            @test interpolate_at(uₕ, 1e20; outside = :extrapolate) ≈ 3e20 + 1
            @test interpolate_at(uₕ, -1e20; outside = :extrapolate) ≈ -3e20 + 1
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

    # A floating-point epsilon past the boundary is on it, under every policy.
    @testset "Epsilon past the boundary is inside" begin
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

    # All three accept and honour the `outside` keyword.
    @testset "πₕ, πₕ!, interpolation_matrix: outside" begin
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

        # interpolation_matrix, entry for entry.
        @testset "Matrix matches pointwise interpolate_at" begin
            for pol in (:clamp, :extrapolate)
                P = interpolation_matrix(Wdest_wide, Wsrc; outside = pol)
                via_matrix = P * parent(src)
                via_pointwise = parent(πₕ(Wdest_wide, src; outside = pol))
                @test via_matrix ≈ via_pointwise
            end
        end

        # interpolation_matrix throws when any destination point is outside.
        @testset "Matrix under :error throws if outside" begin
            @test_throws ArgumentError interpolation_matrix(Wdest_wide, Wsrc)
            # unaffected when destination and source domains actually agree
            P_ok = interpolation_matrix(Wdest, Wsrc)
            @test size(P_ok) == (ndofs(Wdest), ndofs(Wsrc))
        end
    end

    # interpolation_matrix and the bilinear πₕ(u) refuse a fill value.
    @testset "Matrix and bilinear πₕ refuse fill" begin
        Ωdest = mesh(domain(interval(0.0, 1.0)), 6, true)
        Ωsrc = mesh(domain(interval(0.0, 1.0)), 5, true)
        Wdest, Wsrc = gridspace(Ωdest), gridspace(Ωsrc)

        err = @test_throws ArgumentError interpolation_matrix(Wdest, Wsrc; outside = 0.0)
        @test occursin("linear", sprint(showerror, err.value))

        @test_throws ArgumentError form(
            Wsrc, Wdest, (u, v) -> innerₕ(πₕ(u; outside = NaN), v)
        )
    end

    # It must be one of `:error`, `:clamp`, `:extrapolate`, or a Number.
    @testset "outside rejects unknown values" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 11, true)
        uₕ = Rₕ(gridspace(Ωₕ), x -> x[1])
        @test_throws ArgumentError interpolate_at(uₕ, 0.5; outside = :bogus)
        @test_throws ArgumentError interpolate_at(uₕ, 0.5; outside = "clamp")
    end

    # The symbolic form path against the direct path.
    @testset "Form path matches direct, every policy" begin
        Ωdest = mesh(domain(interval(0.0, 1.5)), 6, true)
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
                    @test isnan.(via_form) == isnan.(expected)
                    @test any(isnan, via_form)
                    keep = .!isnan.(expected)
                    @test via_form[keep] ≈ expected[keep]
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

    # Far outside the mesh the corner weights grow past 1/eps: `1 - t` rounds to `-t` and
    # the weighted sum cancels, so a constant field came back as 0.0 (gpena/Bramble.jl#625).
    # `interpolate_at` now blends differences on an extrapolated axis. The matrix, stencil
    # and Dirac paths keep the weights and are out of scope.
    @testset "Far :extrapolate keeps a constant (#625)" begin
        far(u, x) = interpolate_at(u, x; outside = :extrapolate)
        signs = (-1, 1)

        @testset "1D" begin
            u = Rₕ(gridspace(mesh(domain(interval(0.0, 1.0)), 11, true)), x -> 1.0)
            for s in signs
                @test far(u, s * 1e20) == 1.0
            end
        end

        @testset "2D, uniform and non-uniform" begin
            dom = domain(box((0.0, 0.0), (1.0, 1.0)))
            for unif in ((true, true), (false, false))
                u = Rₕ(gridspace(mesh(dom, (11, 65), unif)), x -> 1.0)
                for s1 in signs, s2 in signs

                    @test far(u, (s1 * 1e20, s2 * 1e20)) == 1.0
                end
                @test far(u, (1e20, 0.5)) == 1.0   # one axis inside
            end
        end

        @testset "3D" begin
            dom = domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
            u = Rₕ(gridspace(mesh(dom, (5, 5, 5), (true, true, true))), x -> 1.0)
            for s1 in signs, s2 in signs, s3 in signs
                @test far(u, (s1 * 1e20, s2 * 1e20, s3 * 1e20)) == 1.0
            end
            # ∏ tᵈ overflows here: expanded weights would give Inf * 0 = NaN
            @test far(u, (1e200, -1e200, 1e200)) == 1.0
        end

        @testset "Float32, moderately far" begin
            dom = domain(box((0.0f0, 0.0f0), (1.0f0, 1.0f0)))
            u = Rₕ(gridspace(mesh(dom, (11, 11), (true, true))), x -> 1.0f0)
            # t ≈ 1e5 per axis: the weights' rounding was ~1e3 times the constant
            @test far(u, (1.0f4, -1.0f4)) === 1.0f0
        end

        # Affine on dyadic data, so every corner difference is exact: the mixed
        # difference is exactly 0 and only rounding of the result itself remains.
        @testset "Affine, dyadic 5x5" begin
            dom = domain(box((0.0, 0.0), (1.0, 1.0)))
            f(x) = 2x[1] - x[2] + 0.5
            u = Rₕ(gridspace(mesh(dom, (5, 5), (true, true))), f)
            pts = (
                (1e6, -1e6), (-1e6, 1e6), (1e12, 1e12), (-3.0, 0.5), (0.25, 7.0),
                (1e20, -1e20), (1e20, 1e20), (1e16, 1e16)
            )
            for p in pts
                @test isapprox(far(u, p), f(p); rtol = 4eps())
            end
        end

        # Two or more axes far out: each axis's slope must survive, not just the first.
        @testset "Affine, dyadic 5x5x5" begin
            dom = domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
            g(x) = 2x[1] - x[2] + 4x[3] + 0.5
            u = Rₕ(gridspace(mesh(dom, (5, 5, 5), (true, true, true))), g)
            for p in ((1e20, -1e20, 1e20), (1e20, 0.5, -1e20), (0.5, 1e20, 1e20))
                @test isapprox(far(u, p), g(p); rtol = 4eps())
            end
        end
    end

    # Inside the mesh the weighted form is kept: the result is the same bits as before
    # #625, checked against that corner-weight sum written out, and exactly the node
    # value on the upper boundary (where `locate_cell` gives t = 1).
    @testset "Inside unchanged bitwise (#625)" begin
        f(x) = sin(3x[1]) + x[2]^2
        for unif in ((true, true), (false, false))
            Ωₕ = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (5, 7), unif)
            u = Rₕ(gridspace(Ωₕ), f)
            v = reshape(parent(u), 5, 7)
            p1, p2 = points(Ωₕ(1)), points(Ωₕ(2))
            for x in ((0.3, 0.6), (0.71, 0.13), (0.5, 0.999))
                i, j = Tuple(locate_cell(Ωₕ, x))
                t1 = (x[1] - p1[i]) / (p1[i + 1] - p1[i])
                t2 = (x[2] - p2[j]) / (p2[j + 1] - p2[j])
                ref = 0.0
                ref += (1 - t1) * (1 - t2) * v[i, j]
                ref += t1 * (1 - t2) * v[i + 1, j]
                ref += (1 - t1) * t2 * v[i, j + 1]
                ref += t1 * t2 * v[i + 1, j + 1]
                for pol in (:error, :clamp, :extrapolate, NaN)
                    @test interpolate_at(u, x; outside = pol) == ref
                end
            end
            for pol in (:error, :clamp, :extrapolate)
                @test interpolate_at(u, (1.0, 1.0); outside = pol) == v[5, 7]
                @test interpolate_at(u, (1.0, p2[3]); outside = pol) == v[5, 3]
                @test interpolate_at(u, (p1[2], 1.0); outside = pol) == v[2, 7]
            end
        end
        Ω1 = mesh(domain(interval(0.0, 1.0)), 11, false)
        u1 = Rₕ(gridspace(Ω1), x -> sin(3x[1]))
        p = points(Ω1)
        i = locate_cell(Ω1, 0.37)
        t = (0.37 - p[i]) / (p[i + 1] - p[i])
        @test interpolate_at(u1, 0.37; outside = :extrapolate) ==
              (1 - t) * parent(u1)[i] + t * parent(u1)[i + 1]
        @test interpolate_at(u1, 1.0; outside = :extrapolate) == parent(u1)[end]
    end

    @testset "In-domain path allocates nothing, infers" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 101, true)
        uₕ = Rₕ(gridspace(Ωₕ), x -> x[1]^2)
        x0 = 0.55
        @test_allocs interpolate_at(uₕ, x0)
        @test alloc_test(interpolate_at, uₕ, x0; outside = :clamp) == 0

        Ω2 = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (11, 11), (true, true))
        u2 = Rₕ(gridspace(Ω2), x -> x[1] * x[2])
        @test_allocs interpolate_at(u2, (0.3, 0.7))
        @test alloc_test(interpolate_at, uₕ, 1e20; outside = :extrapolate) == 0
        @test alloc_test(interpolate_at, u2, (1e20, -1e20); outside = :extrapolate) == 0
        Ω3 = mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (5, 5, 5), (true, true, true))
        u3 = Rₕ(gridspace(Ω3), x -> x[1] * x[2])
        @test alloc_test(interpolate_at, u3, (1e20, 0.5, -1e20); outside = :extrapolate) == 0
        @test @inferred(interpolate_at(u3, (1e20, 0.5, -1e20); outside = :extrapolate)) isa
              Float64

        @test @inferred(interpolate_at(uₕ, x0)) isa Float64
        @test @inferred(interpolate_at(uₕ, x0; outside = :clamp)) isa Float64
    end

    @testset "Float32 point on Float64 mesh (#525)" begin
        # #525: an `:extrapolate` coordinate kept the point's own type, so the per-axis
        # tuple was a Float32/Float64 Union and allocated (80-112 B) inside and outside.
        @test Base.return_types(Bramble._interp_resolve_coord, (Float32, Float64, Float64, Symbol)) ==
              Any[Float64]
        Ω2 = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (7, 6), (false, false))
        u2 = Rₕ(gridspace(Ω2), x -> x[1]^2 + 2x[2])
        Ω3 = mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (5, 6, 4), (false, false, false))
        u3 = Rₕ(gridspace(Ω3), x -> x[1]^2 + 2x[2] - x[3])
        for (u, xin, xout) in ((u2, (0.3f0, 0.4f0), (1.7f0, 0.4f0)),
                               (u3, (0.3f0, 0.4f0, 0.6f0), (1.7f0, 0.4f0, -0.2f0)))
            for pol in (:error, :clamp, :extrapolate, NaN)
                @test alloc_test(interpolate_at, u, xin; outside = pol) == 0
                @test interpolate_at(u, xin; outside = pol) == interpolate_at(u, Float64.(xin))
            end
            for pol in (:clamp, :extrapolate, NaN)
                @test alloc_test(interpolate_at, u, xout; outside = pol) == 0
            end
            @test interpolate_at(u, xout; outside = :extrapolate) ==
                  interpolate_at(u, Float64.(xout); outside = :extrapolate)
        end
    end
end

end # module
