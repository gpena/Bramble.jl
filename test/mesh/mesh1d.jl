module MeshMesh1dTests

# Unit tests for 1D mesh generation, geometric queries, and refinement operations.
# Verifies point coordinate generation, half-point and half-spacing caches,
# marker propagation, and degenerate single-point interval behaviors.

using Test
using Random
using Bramble
import Bramble:
                indices,
                change_points!,
                npoints,
                dim,
                spacing,
                half_spacings,
                generate_indices,
                boundary_symbol_to_dict,
                markers,
                backend,
                set_indices!,
                points,
                set_points!,
                set_markers!,
                point,
                half_point,
                half_spacing,
                iterative_refinement!,
                set,
                is_collapsed,
                is_uniform,
                spacings,
                stepsize
import Bramble:
                cell_measure, cell_measures, hₘₐₓ, hₘᵢₙ, half_points, boundary_indices, interior_indices
import Bramble: DomainMarkers, Mesh1D, Backend, forward_spacing, MeshMarkers
import Base: diff

@testset "One-dimensional meshes" begin
    function create_test_domain(a = 0.0, b = 1.0; markers = nothing)
        I = interval(a, b)

        if markers isa Nothing
            return domain(I)
        else
            return domain(I, markers)
        end
    end

    @testset "Helper utilities" begin
        @testset "generate_indices" begin
            @test generate_indices(5) == CartesianIndices((5,))
            @test generate_indices(1) == CartesianIndices((1,))
        end

        @testset "boundary_symbol_to_dict" begin
            indices = CartesianIndices((10,))
            dict = boundary_symbol_to_dict(indices)
            @test dict[:left] == CartesianIndices((1:1,))
            @test dict[:right] == CartesianIndices((10:10,))

            indices_single = CartesianIndices((1,))
            dict_single = boundary_symbol_to_dict(indices_single)
            @test dict_single[:left] == CartesianIndices((1:1,))
            @test dict_single[:right] == CartesianIndices((1:1,))
        end
    end

    @testset "Construction and properties" begin
        Ω = create_test_domain(0.0, 2.0)
        npts = 5
        Ωₕ_unif = mesh(Ω, npts, true; backend = backend())
        Ωₕ_nonunif = mesh(Ω, npts, false; backend = backend())

        @testset "Uniform mesh" begin
            @test Ωₕ_unif isa Mesh1D
            @test backend(Ωₕ_unif) isa Backend
            @test eltype(Ωₕ_unif) == Float64
            @test dim(Ωₕ_unif) == 1
            @test dim(typeof(Ωₕ_unif)) == 1
            @test npoints(Ωₕ_unif) == npts
            @test npoints(Ωₕ_unif, Tuple) == (npts,)
            @test indices(Ωₕ_unif) == CartesianIndices((npts,))
            @test length(points(Ωₕ_unif)) == npts
            @test points(Ωₕ_unif) ≈ [0.0, 0.5, 1.0, 1.5, 2.0]
            @test point(Ωₕ_unif, 3) ≈ 1.0
            @test point(Ωₕ_unif, CartesianIndex(3)) ≈ 1.0
        end

        @testset "Non-uniform mesh" begin
            @test Ωₕ_nonunif isa Mesh1D
            @test backend(Ωₕ_nonunif) isa Backend
            @test eltype(Ωₕ_nonunif) == Float64
            @test dim(Ωₕ_nonunif) == 1
            @test npoints(Ωₕ_nonunif) == npts
            @test npoints(Ωₕ_nonunif, Tuple) == (npts,)
            @test indices(Ωₕ_nonunif) == CartesianIndices((npts,))
            pts_nonunif = points(Ωₕ_nonunif)
            @test length(pts_nonunif) == npts
            @test pts_nonunif[1] ≈ 0.0
            @test pts_nonunif[end] ≈ 2.0
            @test all(diff(pts_nonunif) .> 0) # Check sorted
            # Cannot test exact values, but check bounds and sorting
            @test all(pts_nonunif .>= 0.0) && all(pts_nonunif .<= 2.0)
            @test point(Ωₕ_nonunif, 1) ≈ 0.0
            @test point(Ωₕ_nonunif, npts) ≈ 2.0

            # Float32 draws collide, and so does the map onto [a, b]: every point must
            # still be distinct (gpena/Bramble.jl#494). spacings(Ωf)[1] is x₂ - x₁.
            for seed in 1:5
                Random.seed!(seed)
                Ωf = mesh(create_test_domain(0.0f0, 1.0f0), 10_000, false;
                    backend = backend(Float32))
                @test eltype(Ωf) == Float32
                @test all(>(0), spacings(Ωf))
            end
            for seed in 1:3
                Random.seed!(seed)
                Ωf = mesh(create_test_domain(1.0f0, 2.0f0), 10_000, false;
                    backend = backend(Float32))
                @test eltype(Ωf) == Float32
                @test all(>(0), spacings(Ωf))
            end
            # The device fill method on a host vector: Float64 interval, Float32 storage.
            for seed in 1:5
                Random.seed!(seed)
                x = Vector{Float32}(undef, 10_000)
                Bramble._points!(x, set(create_test_domain(0.0, 1.0)), false,
                    backend(Float32))
                @test all(>(0), diff(x))
            end
            # 100000 distinct Float32 points do not fit in (1, 1.01): 83886 values do.
            @test_throws ArgumentError mesh(create_test_domain(1.0f0, 1.01f0), 100_000,
                false; backend = backend(Float32))
        end

        @testset "Zero or negative point count" begin
            msg = "a mesh needs at least one point per axis; got"
            for n in (0, -2), unif in (true, false)

                @test_throws ArgumentError mesh(create_test_domain(0.0, 1.0), n, unif)
                @test_throws msg mesh(create_test_domain(0.0, 1.0), n, unif)
            end
            # A collapsed interval is no exception: 0 points is still refused.
            @test_throws ArgumentError mesh(create_test_domain(1.0, 1.0), 0)
        end

        @testset "Spacing overflow" begin
            # Finite ends whose spacing b - a overflows are refused (#867).
            Ω = create_test_domain(-1.0e308, 1.0e308)
            @test_throws ArgumentError mesh(Ω, 2, false)
            @test_throws "spacing overflow Float64" mesh(Ω, 2, false)
        end

        @testset "Storage-eltype collapse" begin
            # The interval is a point in Float32 storage: one point, never five copies of
            # 1f6 with zero spacings.
            Ωw = create_test_domain(1.0e6, 1.0e6 + 0.01)
            for unif in (true, false)
                Ωf = mesh(Ωw, 5, unif; backend = backend(Float32))
                @test npoints(Ωf) == 1
                @test is_collapsed(Ωf)
                @test points(Ωf) == [1.0f6]
            end
            # Float64 storage keeps five distinct points.
            for unif in (true, false)
                Ωd = mesh(Ωw, 5, unif; backend = backend())
                @test npoints(Ωd) == 5
                @test all(>(0), diff(points(Ωd)))
            end
            # Five uniform points do not fit in a 1-ulp interval.
            @test_throws ArgumentError mesh(create_test_domain(1.0, nextfloat(1.0)), 5, true)
            # Two do: the endpoints themselves.
            @test points(mesh(create_test_domain(1.0, nextfloat(1.0)), 2, true)) ==
                  [1.0, nextfloat(1.0)]
            # The uniform fill runs in the wider of the set and storage eltypes. A
            # Float64 set on Float32 storage gets the correctly rounded, symmetric nodes.
            a, b, n = -1.0, 1.0, 11
            p32 = points(mesh(create_test_domain(a, b), n, true; backend = backend(Float32)))
            @test p32 == Float32.(a .+ (0:(n - 1)) .* ((b - a) / (n - 1)))
            @test p32 == -reverse(p32)
            # A Float32 set on Float64 storage gets nine distinct points that Float32
            # arithmetic would tie.
            I32 = create_test_domain(1.0f6, 1.0f6 + 0.25f0)
            Ω64 = mesh(I32, 9, true; backend = backend(Float64))
            @test eltype(points(Ω64)) == Float64
            @test npoints(Ω64) == 9 && all(>(0), diff(points(Ω64)))
            @test points(Ω64)[1] == 1.0e6
            # GMG coarsening builds such a uniform mesh from a non-uniform fine one.
            Ω64n = mesh(I32, 17, false; backend = backend(Float64))
            H = GeometricMeshHierarchy(Ω64n, 2)
            @test npoints(H[1]) == 9 && all(>(0), diff(points(H[1])))
        end

        @testset "set_points! & set_indices!" begin
            Ω = create_test_domain(0.0, 1.0)
            Ωₕ = mesh(Ω, 3, true; backend = backend()) # [0.0, 0.5, 1.0]

            new_pts = [0.0, 0.3, 0.7, 1.0]
            new_indices = CartesianIndices((4,))

            set_points!(Ωₕ, new_pts)
            set_indices!(Ωₕ, new_indices) # Need to update indices if npts changes via set_points!

            @test points(Ωₕ) === new_pts # Check identity for mutable struct
            @test npoints(Ωₕ) == 4
            @test indices(Ωₕ) == new_indices

            # A new point count rebuilds the markers for the new grid (the old ones were
            # sized for 3 points), and the marker words with them.
            @test markers(Ωₕ)[:boundary] == BitVector([1, 0, 0, 1])
            @test markers(Ωₕ)[:interior] == BitVector([0, 1, 1, 0])
            set_points!(Ωₕ, collect(range(0, 1; length = 100)) .^ 2)
            @test findall(markers(Ωₕ)[:boundary]) == [1, 100]
            @test findall(markers(Ωₕ)[:interior]) == 2:99
            @test size(Bramble._marker_words(Ωₕ), 1) == 2

            # A point-count change on a mesh with a custom label is refused, naming the
            # call that keeps it; the mesh is left as it was.
            dm = markers(interval(0, 1), :BC => :left)
            Ωc = mesh(create_test_domain(0.0, 1.0; markers = dm), 3, true; backend = backend())
            msgs = ["set_points!(Ωₕ, pts) was asked to change the point count",
                "custom markers", ":BC", "change_points!(Ωₕ, markers(Ω), pts)"]
            @test_throws msgs set_points!(Ωc, [0.0, 0.3, 0.7, 1.0])
            @test npoints(Ωc) == 3
            # the same count only moves the points, so it needs no domain
            set_points!(Ωc, [0.0, 0.3, 1.0])
            @test points(Ωc) == [0.0, 0.3, 1.0]
        end

        @testset "set_points! refuses invalid points" begin
            # Non-increasing, tied or non-finite points are refused, naming the first bad
            # index, before anything changes: version and spacings included (#698).
            Ωₕ = mesh(create_test_domain(0.0, 1.0), 5, false; backend = backend())
            p0 = copy(points(Ωₕ))
            s0 = copy(spacings(Ωₕ))
            v0 = Bramble._mesh_version(Ωₕ)
            bad = [([0.0, 0.5, 0.25, 0.75, 1.0], "point 3"),
                ([0.0, 0.25, 0.25, 0.75, 1.0], "point 3"),
                ([0.0, 0.25, NaN, 0.75, 1.0], "point 3"),
                ([0.0, 0.25, 0.5, 0.75, Inf], "point 5"),
                ([-Inf, 0.25, 0.5, 0.75, 1.0], "point 1"),
                ([0.0, 0.75, 0.5], "point 3"),
                (Float64[], "at least one point"),
                # finite points whose half-point, spacing or cell measure overflows (#867)
                ([1.0e308, 1.7e308], "half-point at point 2"),
                ([-1.0e308, 1.0e308], "spacing at point 2"),
                ([-1.0e308, 0.0, 1.0e308], "cell measure at point 2")]
            for (pts, msg) in bad
                @test_throws ArgumentError set_points!(Ωₕ, pts)
                @test_throws msg set_points!(Ωₕ, pts)
                @test points(Ωₕ) == p0
                @test spacings(Ωₕ) == s0
                @test npoints(Ωₕ) == 5
                @test Bramble._mesh_version(Ωₕ) == v0
            end
            # change_points! goes through the same check
            @test_throws ArgumentError change_points!(Ωₕ, [0.0, 0.5, 0.25, 0.75, 1.0])
            @test points(Ωₕ) == p0

            # The overflow is judged in the storage eltype: points finite in Float32 whose
            # spacing is not are refused on a Float32 mesh, though fine in Float64.
            Ω32 = mesh(create_test_domain(0.0f0, 1.0f0), 2, false;
                backend = backend(Float32))
            @test_throws "spacing at point 2" set_points!(Ω32, [-3.0e38, 3.0e38])
            @test points(Ω32) == Float32[0, 1]
            @test spacings(Ω32) == Float32[1, 1]
            # a point that overflows only on conversion, even alone
            @test_throws "point 1 is Inf" set_points!(Ω32, [1.0e39])
            @test points(Ω32) == Float32[0, 1]

            # Strict increase is judged on the stored values too: 0.5 and 0.5 + 1e-9 are
            # distinct in Float64 but one Float32, a zero spacing (gpena/Bramble.jl#969).
            # The message shows the stored values, never the raw 0.500000001.
            Ω32₄ = mesh(create_test_domain(0.0f0, 1.0f0), 4, true;
                backend = backend(Float32))
            p32 = copy(points(Ω32₄))
            v32 = Bramble._mesh_version(Ω32₄)
            tie32 = [0.0, 0.5, 0.5 + 1.0e-9, 1.0]
            @test Float32(tie32[3]) == Float32(tie32[2])
            err = try
                set_points!(Ω32₄, tie32)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            msg = sprint(showerror, err)
            @test occursin("strictly increasing once stored as Float32", msg)
            @test occursin("point 3, stored as 0.5", msg)
            @test !occursin("0.500000001", msg)
            @test points(Ω32₄) == p32
            @test Bramble._mesh_version(Ω32₄) == v32
            @test_throws "point 3, stored as 0.5" change_points!(Ω32₄, tie32)
            @test points(Ω32₄) == p32
            # the same list is accepted on a Float64 mesh, where it stays increasing
            Ω64₄ = mesh(create_test_domain(0.0, 1.0), 4, true; backend = backend())
            set_points!(Ω64₄, tie32)
            @test points(Ω64₄) == tie32

            # The cell measure is checked as the mesh computes it, (h₁ + h₂) * 0.5: here
            # x₃ - x₁ is floatmax, but the rounded h₁ + h₂ ties up to Inf.
            x_tie = [-2.0^1023, nextfloat(2.0^970), 2.0^1023 - 2.0^971]
            @test isfinite(x_tie[3] - x_tie[1])
            @test !isfinite((x_tie[2] - x_tie[1]) + (x_tie[3] - x_tie[2]))
            @test_throws "cell measure at point 2" set_points!(Ωₕ, x_tie)
            @test points(Ωₕ) == p0
            @test Bramble._mesh_version(Ωₕ) == v0

            # An empty list is refused on a collapsed mesh too, which keeps its one point.
            Ωc = mesh(create_test_domain(1.0, 1.0), 1; backend = backend())
            @test_throws ArgumentError set_points!(Ωc, Float64[])
            @test npoints(Ωc) == 1
            @test points(Ωc) == [1.0]

            # A valid non-uniform list is accepted unchanged.
            set_points!(Ωₕ, [0.0, 0.1, 0.5, 0.9, 1.0])
            @test points(Ωₕ) == [0.0, 0.1, 0.5, 0.9, 1.0]
            @test spacings(Ωₕ) ≈ [0.1, 0.1, 0.4, 0.4, 0.1]
            @test Bramble.locate_cell(Ωₕ, 0.3) == 2
        end
    end

    @testset "Geometric properties" begin
        npts = 5
        Ω_unif = create_test_domain(0.0, 4.0) # Step = 1.0
        Ωₕ_unif = mesh(Ω_unif, npts, true; backend = backend()) # Pts: 0, 1, 2, 3, 4

        # Create a non-uniform mesh manually for predictable spacing
        Ω_nonunif = create_test_domain(0.0, 5.0)
        Ωₕ_nonunif = mesh(Ω_nonunif, 4, true; backend = backend()) # Start uniform
        nonunif_pts = [0.0, 1.0, 3.0, 5.0] # Spacing: 1.0, 2.0, 2.0
        set_points!(Ωₕ_nonunif, nonunif_pts)
        set_indices!(Ωₕ_nonunif, CartesianIndices((4,)))

        @testset "spacing" begin
            # Uniform
            @test spacing(Ωₕ_unif, 1) ≈ 1.0 # Defined as pts[2]-pts[1]
            @test spacing(Ωₕ_unif, 2) ≈ 1.0
            @test spacing(Ωₕ_unif, 5) ≈ 1.0
            @test collect(spacings(Ωₕ_unif)) ≈ [1.0, 1.0, 1.0, 1.0, 1.0]

            # Non-uniform
            @test spacing(Ωₕ_nonunif, 1) ≈ 1.0 # pts[2]-pts[1]
            @test spacing(Ωₕ_nonunif, 2) ≈ 1.0 # pts[2]-pts[1]
            @test spacing(Ωₕ_nonunif, 3) ≈ 2.0 # pts[3]-pts[2]
            @test spacing(Ωₕ_nonunif, 4) ≈ 2.0 # pts[4]-pts[3]
            # spacings starts from index 1, using the definition for spacing(mesh, i)
            @test collect(spacings(Ωₕ_nonunif)) ≈ [1.0, 1.0, 2.0, 2.0]
        end

        @testset "hₘₐₓ" begin
            @test hₘₐₓ(Ωₕ_unif) ≈ 1.0
            @test hₘₐₓ(Ωₕ_nonunif) ≈ 2.0

            # spacings [1, 1, 2, 2]: the smallest and largest differ, so a min/max mix-up shows
            @test hₘᵢₙ(Ωₕ_unif) == 1.0
            @test hₘᵢₙ(Ωₕ_nonunif) == 1.0

            # hₘᵢₙ is the smallest spacing. On [0, 2, 2.5, 5] (spacings 2, 2, 0.5, 2.5) it
            # sits inside, so it differs from the first spacing and from hₘₐₓ / 2, which
            # both equal it on the mesh above.
            Ωₕ_inner = mesh(Ω_nonunif, 4, true; backend = backend())
            set_points!(Ωₕ_inner, [0.0, 2.0, 2.5, 5.0])
            @test hₘᵢₙ(Ωₕ_inner) == 0.5
            @test hₘₐₓ(Ωₕ_inner) == 2.5

            # On [0, 2, 4, 4.5] (spacings 2, 2, 2, 0.5) the smallest spacing is the last
            # one, so a minimum that skips the last cell shows
            Ωₕ_last = mesh(Ω_nonunif, 4, true; backend = backend())
            set_points!(Ωₕ_last, [0.0, 2.0, 4.0, 4.5])
            @test hₘᵢₙ(Ωₕ_last) == 0.5
        end

        @testset "half_spacing" begin
            # Uniform: h=1.0 => h_half should be 0.5, 1.0, 1.0, 1.0, 0.5
            @test half_spacing(Ωₕ_unif, 1) ≈ 0.5 * spacing(Ωₕ_unif, 1) ≈ 0.5
            @test half_spacing(Ωₕ_unif, 2) ≈
                  0.5 * (spacing(Ωₕ_unif, 2) + spacing(Ωₕ_unif, 3)) ≈
                  1.0
            @test half_spacing(Ωₕ_unif, 4) ≈
                  0.5 * (spacing(Ωₕ_unif, 4) + spacing(Ωₕ_unif, 5)) ≈
                  1.0
            @test half_spacing(Ωₕ_unif, 5) ≈ 0.5 * spacing(Ωₕ_unif, 5) ≈ 0.5
            @test collect(half_spacings(Ωₕ_unif)) ≈ [0.5, 1.0, 1.0, 1.0, 0.5]

            # Non-uniform: h = [1.0, 1.0, 2.0, 2.0] (spacings at indices 1, 2, 3, 4)
            # h_half should be: h1/2, (h1+h2)/2, (h2+h3)/2, h3/2  <- NO! Definition uses i and i+1
            # h_half(1) = spacing(1)/2 = 1.0/2 = 0.5
            # h_half(2) = (spacing(2) + spacing(3))/2 = (1.0 + 2.0)/2 = 1.5
            # h_half(3) = (spacing(3) + spacing(4))/2 = (2.0 + 2.0)/2 = 2.0
            # h_half(4) = spacing(4)/2 = 2.0/2 = 1.0
            @test half_spacing(Ωₕ_nonunif, 1) ≈ 0.5 * spacing(Ωₕ_nonunif, 1) ≈ 0.5
            @test half_spacing(Ωₕ_nonunif, 2) ≈
                  0.5 * (spacing(Ωₕ_nonunif, 2) + spacing(Ωₕ_nonunif, 3)) ≈
                  1.5
            @test half_spacing(Ωₕ_nonunif, 3) ≈
                  0.5 * (spacing(Ωₕ_nonunif, 3) + spacing(Ωₕ_nonunif, 4)) ≈
                  2.0
            @test half_spacing(Ωₕ_nonunif, 4) ≈ 0.5 * spacing(Ωₕ_nonunif, 4) ≈ 1.0
            @test collect(half_spacings(Ωₕ_nonunif)) ≈ [0.5, 1.5, 2.0, 1.0]
        end

        @testset "cell_measure" begin
            # Should be identical to half_spacing
            @test cell_measure(Ωₕ_unif, 1) ≈ half_spacing(Ωₕ_unif, 1)
            @test cell_measure(Ωₕ_unif, 3) ≈ half_spacing(Ωₕ_unif, 3)
            @test cell_measure(Ωₕ_unif, 5) ≈ half_spacing(Ωₕ_unif, 5)
            @test collect(cell_measures(Ωₕ_unif)) ≈ collect(half_spacings(Ωₕ_unif))

            @test cell_measure(Ωₕ_nonunif, 1) ≈ half_spacing(Ωₕ_nonunif, 1)
            @test cell_measure(Ωₕ_nonunif, 2) ≈ half_spacing(Ωₕ_nonunif, 2)
            @test cell_measure(Ωₕ_nonunif, 4) ≈ half_spacing(Ωₕ_nonunif, 4)
            @test collect(cell_measures(Ωₕ_nonunif)) ≈ collect(half_spacings(Ωₕ_nonunif))
        end

        @testset "half_points" begin
            # Uniform: pts = 0, 1, 2, 3, 4; npts=5
            # Indices for half_points go from 1 to npts+1 = 6
            # hp(1) = pts(1) = 0
            # hp(2) = (pts(1)+pts(2))/2 = 0.5
            # hp(3) = (pts(2)+pts(3))/2 = 1.5
            # hp(4) = (pts(3)+pts(4))/2 = 2.5
            # hp(5) = (pts(4)+pts(5))/2 = 3.5
            # hp(6) = pts(5) = 4
            @test half_point(Ωₕ_unif, 1) ≈ 0.0
            @test half_point(Ωₕ_unif, 2) ≈ 0.5
            @test half_point(Ωₕ_unif, 3) ≈ 1.5
            @test half_point(Ωₕ_unif, 5) ≈ 3.5
            @test half_point(Ωₕ_unif, 6) ≈ 4.0
            @test collect(half_points(Ωₕ_unif)) ≈ [0.0, 0.5, 1.5, 2.5, 3.5, 4.0]

            # Non-uniform: pts = 0, 1, 3, 5; npts=4
            # Indices for half_points go from 1 to npts+1 = 5
            # hp(1) = pts(1) = 0
            # hp(2) = (pts(1)+pts(2))/2 = 0.5
            # hp(3) = (pts(2)+pts(3))/2 = 2.0
            # hp(4) = (pts(3)+pts(4))/2 = 4.0
            # hp(5) = pts(4) = 5.0
            @test half_point(Ωₕ_nonunif, 1) ≈ 0.0
            @test half_point(Ωₕ_nonunif, 2) ≈ 0.5
            @test half_point(Ωₕ_nonunif, 3) ≈ 2.0
            @test half_point(Ωₕ_nonunif, 4) ≈ 4.0
            @test half_point(Ωₕ_nonunif, 5) ≈ 5.0
            @test collect(half_points(Ωₕ_nonunif)) ≈ [0.0, 0.5, 2.0, 4.0, 5.0]
        end
    end

    @testset "Index subsets" begin
        npts = 5
        Ω = create_test_domain(0.0, 1.0)
        Ωₕ = mesh(Ω, npts, true; backend = backend())

        @test boundary_indices(Ωₕ) == (CartesianIndices((1:1,)), CartesianIndices((npts:npts,)))
        @test interior_indices(Ωₕ) == CartesianIndices((2:(npts - 1),))

        # Edge cases
        Ωₕ_2 = mesh(Ω, 2, true; backend = backend())
        @test boundary_indices(Ωₕ_2) == (CartesianIndices((1:1,)), CartesianIndices((2:2,)))
        @test isempty(interior_indices(Ωₕ_2)) # Interior is empty range 2:1

        # Test on CartesianIndices directly
        inds = CartesianIndices((10,))
        @test boundary_indices(inds) == (CartesianIndices((1:1,)), CartesianIndices((10:10,)))
        @test interior_indices(inds) == CartesianIndices((2:9,))
    end

    @testset "Marker setting" begin
        I = interval(0, 1)

        # Define markers
        dm = markers(
            I,
            :Dirichlet => :left,
            :Neumann => :right,
            :Mixed => (:left, :right),
            :LowerHalf => x -> x[1] < 0.5,
            :UpperHalf => x -> x[1] > 0.5,
            :PointMarker => x -> isapprox(x[1], 0.75)
        )

        Ω = create_test_domain(0.0, 1.0; markers = dm)

        # Which points carry each marker, not only its name: a mask left all-false, the
        # left and right ends swapped, or a tuple marking only its first symbol fails here.
        # :UpperHalf holds at x = 1, so a predicate skipping the last point fails too.
        function check_masks(Ωₕ; lower = Bool[1, 1, 0, 0, 0], upper = Bool[0, 0, 0, 1, 1],
                point = Bool[0, 0, 0, 1, 0])
            @test markers(Ωₕ)[:Dirichlet] == Bool[1, 0, 0, 0, 0]
            @test markers(Ωₕ)[:Neumann] == Bool[0, 0, 0, 0, 1]
            @test markers(Ωₕ)[:Mixed] == Bool[1, 0, 0, 0, 1]
            @test markers(Ωₕ)[:LowerHalf] == lower
            @test markers(Ωₕ)[:UpperHalf] == upper
            @test markers(Ωₕ)[:PointMarker] == point
        end

        npts = 5 # Points: 0.0, 0.25, 0.5, 0.75, 1.0
        Ωₕ = mesh(Ω, npts, true; backend = backend())

        # Test marker retrieval before explicit setting (should be done by constructor)
        # :boundary/:interior are always present too now (test/mesh/markers.jl covers them).
        @test Set(keys(markers(Ωₕ))) == Set([
            :Dirichlet, :Neumann, :Mixed, :LowerHalf, :UpperHalf, :PointMarker, :boundary,
            :interior
        ])
        check_masks(Ωₕ)

        # Test explicit call to set_markers! (should ideally yield the same)
        set_markers!(Ωₕ, dm) # Recalculate
        # :boundary/:interior are always present too now (test/mesh/markers.jl covers them).
        @test Set(keys(markers(Ωₕ))) == Set([
            :Dirichlet, :Neumann, :Mixed, :LowerHalf, :UpperHalf, :PointMarker, :boundary,
            :interior
        ])
        check_masks(Ωₕ)

        # On non-uniform points the predicates read the stored points: a uniform
        # coordinate rebuilt from the index would put 0.25, 0.5, 0.75 in their place.
        change_points!(Ωₕ, dm, [0.0, 0.1, 0.2, 0.3, 1.0])
        check_masks(Ωₕ; lower = Bool[1, 1, 1, 1, 0], upper = Bool[0, 0, 0, 0, 1],
            point = Bool[0, 0, 0, 0, 0])
    end

    @testset "Mesh modification" begin
        # A label's mask as both stores hold it: the Dict `markers(Ωₕ)` and the word matrix
        # assembly reads (`_is_marked`), point by point.
        function stored_masks(Ωₕ, label)
            words, id = Bramble._marker_words(Ωₕ), Bramble._marker_id(Ωₕ, label)
            word_mask = BitVector(Bramble._is_marked(words, id, i) for i in 1:npoints(Ωₕ))
            return markers(Ωₕ)[label], word_mask
        end

        @testset "iterative_refinement!" begin
            dm = markers(interval(0, 1), :BC => :left, :Center => x -> 0.4 < x[1] < 0.6,
                :Q => x -> 0.2 < x[1] < 0.3)
            Ω = create_test_domain(0.0, 1.0; markers = dm)

            npts_initial = 3 # Pts: 0.0, 0.5, 1.0
            npts_refined2 = 2 * npts_initial - 1 # 5

            # Refining a mesh carrying custom markers (:BC, :Center)
            # without supplying domain markers now refuses outright -- there is no domain
            # here to re-derive them from -- rather than silently dropping them. Left
            # untouched, not partially refined.
            Ωₕ = mesh(Ω, npts_initial, true; backend = backend())
            @test_throws ArgumentError iterative_refinement!(Ωₕ)
            @test_throws ["iterative_refinement!(Ωₕ) was asked to refine",
                "custom markers", ":BC", ":Center",
                "iterative_refinement!(Ωₕ, domain_markers)"] iterative_refinement!(Ωₕ)
            @test npoints(Ωₕ) == npts_initial
            @test points(Ωₕ) == [0.0, 0.5, 1.0]

            # Refine *with* marker update
            Ωₕ2 = mesh(Ω, npts_initial, true; backend = backend()) # Start fresh: 0.0, 0.5, 1.0
            iterative_refinement!(Ωₕ2, dm)
            @test npoints(Ωₕ2) == npts_refined2
            @test indices(Ωₕ2) == CartesianIndices((npts_refined2,))
            @test points(Ωₕ2) == [0.0, 0.25, 0.5, 0.75, 1.0]
            # the markers are re-derived on the refined points, not left at the old three
            @test all(==(Bool[0, 0, 1, 0, 0]), stored_masks(Ωₕ2, :Center))
            @test all(==(Bool[1, 0, 0, 0, 0]), stored_masks(Ωₕ2, :BC))
            @test all(==(Bool[1, 0, 0, 0, 1]), stored_masks(Ωₕ2, :boundary))
            # :Q holds only at the new midpoint 0.25, not at any old point
            @test all(==(Bool[0, 1, 0, 0, 0]), stored_masks(Ωₕ2, :Q))

            # On non-uniform points the masks follow the refined points themselves.
            # [0, 0.9, 1] refines to [0, 0.45, 0.9, 0.95, 1], where the masks of a
            # uniform grid (Center at 0.5, Q at 0.25) would be wrong.
            Ωₕ_nu = mesh(Ω, npts_initial, true; backend = backend())
            change_points!(Ωₕ_nu, dm, [0.0, 0.9, 1.0])
            iterative_refinement!(Ωₕ_nu, dm)
            @test points(Ωₕ_nu) ≈ [0.0, 0.45, 0.9, 0.95, 1.0]
            @test all(==(Bool[0, 1, 0, 0, 0]), stored_masks(Ωₕ_nu, :Center))
            @test all(==(Bool[0, 0, 0, 0, 0]), stored_masks(Ωₕ_nu, :Q))
            @test all(==(Bool[1, 0, 0, 0, 0]), stored_masks(Ωₕ_nu, :BC))

            # On a 1-ulp interval the midpoint of the two endpoints rounds onto one of
            # them, so refinement would tie neighbours (gpena/Bramble.jl#621). Both forms
            # refuse before any mutation: points, indices, markers and version unchanged.
            Ω_ulp = create_test_domain(1.0, nextfloat(1.0))
            dm_ulp = markers(Ω_ulp)
            for refine! in (iterative_refinement!, M -> iterative_refinement!(M, dm_ulp))
                Ωₜ = mesh(Ω_ulp, 2, true; backend = backend())
                markers_before = deepcopy(markers(Ωₜ))
                version_before = Bramble._mesh_version(Ωₜ)
                @test_throws ArgumentError refine!(Ωₜ)
                err = try
                    refine!(Ωₜ)
                catch e
                    e
                end
                @test occursin("midpoint", sprint(showerror, err))
                @test points(Ωₜ) == [1.0, nextfloat(1.0)]
                @test npoints(Ωₜ) == 2
                @test indices(Ωₜ) == CartesianIndices((2,))
                @test markers(Ωₜ) == markers_before
                @test Bramble._mesh_version(Ωₜ) == version_before
            end

            # On Float16 [40000, 60000] the sum `a + b` overflows to Inf, so even the
            # 2-point mesh is refused at construction: its one half-point overflows.
            Ω_big = create_test_domain(Float16(40000), Float16(60000))
            for n in (2, 3)
                err = try
                    mesh(Ω_big, n, true; backend = backend(Float16))
                catch e
                    e
                end
                @test err isa ArgumentError
                msg = sprint(showerror, err)
                @test occursin("half-point overflow", msg)
                @test occursin("Float16", msg)
                @test occursin("$n points", msg)
            end

            # On Float16 [0, 50000] the 3-point mesh is refused at construction: the
            # half-point (25000 + 50000) * 0.5 overflows (gpena/Bramble.jl#636).
            Ω_half = create_test_domain(Float16(0), Float16(50000))
            err = try
                mesh(Ω_half, 3, true; backend = backend(Float16))
            catch e
                e
            end
            @test err isa ArgumentError
            msg = sprint(showerror, err)
            @test occursin("half-point overflow", msg)
            @test occursin("Float16", msg)
            @test occursin("3 points", msg)

            # The 2-point mesh is fine, and refining it adds the finite midpoint 25000
            # whose half-point overflows. Both forms refuse with an overflow message, not
            # the tie message, and leave points, half-points, indices, markers and version
            # unchanged.
            dm_half = markers(Ω_half)
            for refine! in (iterative_refinement!, M -> iterative_refinement!(M, dm_half))
                Ωₒ = mesh(Ω_half, 2, true; backend = backend(Float16))
                @test all(isfinite, Bramble.half_points(Ωₒ))
                half_before = copy(Bramble.half_points(Ωₒ))
                points_before = copy(points(Ωₒ))
                markers_before = deepcopy(markers(Ωₒ))
                version_before = Bramble._mesh_version(Ωₒ)
                @test_throws ArgumentError refine!(Ωₒ)
                err = try
                    refine!(Ωₒ)
                catch e
                    e
                end
                msg = sprint(showerror, err)
                @test occursin("half-point overflow", msg)
                @test occursin("Float16", msg)
                @test !occursin("rounds a midpoint", msg)
                @test points(Ωₒ) == points_before == Float16[0, 50000]
                @test Bramble.half_points(Ωₒ) == half_before
                @test eltype(points(Ωₒ)) == Float16
                @test npoints(Ωₒ) == 2
                @test indices(Ωₒ) == CartesianIndices((2,))
                @test markers(Ωₒ) == markers_before
                @test Bramble._mesh_version(Ωₒ) == version_before
            end

            # The two arities have a genuine (not accidental)
            # asymmetry on a single-point, non-collapsed mesh (nothing to refine either
            # way), but the one-argument form must leave existing markers untouched
            # (no domain to re-derive them from), while the two-argument form must still
            # (re)apply the domain markers it was given, since `set_markers!` needs no
            # interval to do that. Hoisting both to `AbstractMeshType` in `interface.jl`
            # must not collapse this distinction into a single shared guard.
            Ω_one = create_test_domain(2.0, 5.0; markers = dm)
            Ωₕ_one_arg = mesh(Ω_one, 1, true; backend = backend())
            @test !is_collapsed(Ωₕ_one_arg)
            markers_before = deepcopy(markers(Ωₕ_one_arg))
            iterative_refinement!(Ωₕ_one_arg)
            @test npoints(Ωₕ_one_arg) == 1
            @test point(Ωₕ_one_arg, 1) == 2.0
            @test markers(Ωₕ_one_arg) == markers_before

            Ωₕ_one_dm = mesh(Ω_one, 1, true; backend = backend())
            iterative_refinement!(Ωₕ_one_dm, dm)
            @test npoints(Ωₕ_one_dm) == 1
            @test haskey(markers(Ωₕ_one_dm), :BC)
            @test haskey(markers(Ωₕ_one_dm), :Center)
        end

        @testset "change_points!" begin
            dm = markers(interval(0, 1), :Endpoint => :right, :NearStart => x -> x[1] < 0.3,
                :Far => x -> x[1] > 0.3)
            Ω = create_test_domain(0.0, 2.0; markers = dm)
            npts = 5 # Pts: 0.0, 0.5, 1.0, 1.5, 2.0
            Ωₕ = mesh(Ω, npts, true; backend = backend())

            # Original markers
            new_pts_valid = [0.0, 0.1, 0.5, 1.5, 2.0] # Keep endpoints, change interior
            new_pts_invalid_len = [0.0, 1.0, 2.0]
            new_pts_invalid_ends = [0.1, 0.5, 1.0, 1.5, 2.1]

            # The two-argument form refuses a mesh carrying the predicate labels NearStart
            # and Far, before anything changes. Endpoint is a face label.
            Ωₕ_copy1 = deepcopy(Ωₕ)
            @test_throws ["change_points!(Ωₕ, pts)", "(:NearStart, :Far)",
                "change_points!(Ωₕ, markers(Ω), pts)"] change_points!(
                Ωₕ_copy1, new_pts_valid)
            @test points(Ωₕ_copy1) ≈ [0.0, 0.5, 1.0, 1.5, 2.0]
            @test all(==(Bool[1, 0, 0, 0, 0]), stored_masks(Ωₕ_copy1, :NearStart))

            # Test valid change *with* marker update
            Ωₕ_copy2 = deepcopy(Ωₕ)
            change_points!(Ωₕ_copy2, dm, new_pts_valid)
            @test points(Ωₕ_copy2) ≈ new_pts_valid
            # the three-argument form re-derives them on the new points (0.1 < 0.3)
            @test all(==(Bool[1, 1, 0, 0, 0]), stored_masks(Ωₕ_copy2, :NearStart))
            @test all(==(Bool[0, 0, 0, 0, 1]), stored_masks(Ωₕ_copy2, :Endpoint))
            @test all(==(Bool[0, 0, 1, 1, 1]), stored_masks(Ωₕ_copy2, :Far))

            # A different point count is rejected and leaves the points untouched.
            Ωₕ_copy3 = deepcopy(Ωₕ)
            @test_throws DimensionMismatch change_points!(Ωₕ_copy3, dm, new_pts_invalid_len)
            @test points(Ωₕ_copy3) ≈ [0.0, 0.5, 1.0, 1.5, 2.0]
        end
    end

    @testset "Package-local RNG for non-uniform points" begin
        # Oracle: an independent Xoshiro with the same seed, drawn as Float64, sorted and
        # mapped onto [a, b]; the mesh's spacings and half points follow from those points.
        a, b, n, seed = -1.0, 3.0, 9, 2024
        rng = Random.Xoshiro(seed)
        interior = sort!(rand(rng, Float64, n - 2))
        expected = a .+ vcat(0.0, interior, 1.0) .* (b - a)

        Bramble._seed_mesh1d_rng!(seed)
        Ωₕ = try
            mesh(create_test_domain(a, b), n, false; backend = backend())
        finally
            Bramble._unseed_mesh1d_rng!()
        end

        @test points(Ωₕ) ≈ expected
        @test collect(spacings(Ωₕ))[2:end] ≈ diff(expected)
        @test collect(half_points(Ωₕ))[2:n] ≈ (expected[1:(end - 1)] .+ expected[2:end]) ./ 2
        d = diff(expected)
        hs = Bramble.host_half_spacings(Ωₕ)
        @test hs isa Vector{Float64}
        @test hs ≈ vcat(d[1] / 2, (d[1:(end - 1)] .+ d[2:end]) ./ 2, d[end] / 2)

        # Once disarmed, the global RNG drives the interior points again.
        Random.seed!(seed)
        Ωₕ2 = mesh(create_test_domain(a, b), n, false; backend = backend())
        Random.seed!(seed)
        @test points(Ωₕ2) ≈ a .+ vcat(0.0, sort!(rand(n - 2)), 1.0) .* (b - a)

        # An armed Float32 mesh is redrawn past its ties too (gpena/Bramble.jl#494).
        Bramble._seed_mesh1d_rng!(1)
        Ωf = try
            mesh(create_test_domain(0.0f0, 1.0f0), 10_000, false; backend = backend(Float32))
        finally
            Bramble._unseed_mesh1d_rng!()
        end
        @test eltype(Ωf) == Float32
        @test all(>(0), diff(points(Ωf)))
    end

    @testset "Additional methods" begin
        Ω = create_test_domain(0.0, 4.0)
        Ωₕ = mesh(Ω, 5, true; backend = backend()) # [0, 1, 2, 3, 4]

        @testset "Field accessors" begin
            # Test set accessor
            @test set(Ωₕ) == interval(0.0, 4.0)

            # Test is_collapsed
            @test is_collapsed(Ωₕ) == false

            # Collapsed mesh
            Ω_pt = create_test_domain(1.0, 1.0)
            Ωₕ_pt = mesh(Ω_pt, 1, true; backend = backend())
            @test is_collapsed(Ωₕ_pt) == true
            @test spacing(Ωₕ_pt, 1) == 0.0
            @test forward_spacing(Ωₕ_pt, 1) == 0.0
        end

        @testset "Single-point mesh" begin
            # A collapsed interval [c, c] must be meshed at c, not at the origin.
            for c in (1.0, 3.0, -2.5)
                Ωₕ_c = mesh(create_test_domain(c, c), 1, true; backend = backend())
                @test npoints(Ωₕ_c) == 1
                @test points(Ωₕ_c) == [c]
                @test point(Ωₕ_c, 1) == c
            end

            # Requesting more points on a collapsed interval still yields the one point.
            Ωₕ_many = mesh(create_test_domain(3.0, 3.0), 7, true; backend = backend())
            @test npoints(Ωₕ_many) == 1
            @test points(Ωₕ_many) == [3.0]

            # A genuine interval reduced to a single point uses the lower bound,
            # and has zero spacing rather than reading past the end of the vector.
            Ωₕ_one = mesh(create_test_domain(2.0, 5.0), 1, true; backend = backend())
            @test points(Ωₕ_one) == [2.0]
            @test spacing(Ωₕ_one, 1) == 0.0
            @test forward_spacing(Ωₕ_one, 1) == 0.0
            @test hₘₐₓ(Ωₕ_one) == 0.0
        end

        @testset "Spacing edge cases" begin
            # Test forward_spacing at boundaries
            @test forward_spacing(Ωₕ, 1) ≈ 1.0  # pts[2] - pts[1]
            @test forward_spacing(Ωₕ, 5) ≈ 1.0  # pts[5] - pts[4] (backward at end)
            @test forward_spacing(Ωₕ, CartesianIndex(3)) ≈ 1.0
        end

        @testset "Submesh indexing" begin
            # Test that mesh(i) returns itself for 1D
            @test Ωₕ(1) === Ωₕ
            @test Ωₕ(999) === Ωₕ  # Any value returns itself
        end

        @testset "Type stability" begin
            # The value-level `eltype`/`dim` are "Uniform mesh"'s; this is the type-level
            # query, which resolves through a different method.
            @test eltype(typeof(Ωₕ)) == Float64
        end

        @testset "Indexing" begin
            @test Ωₕ[1] ≈ 0.0
            @test Ωₕ[3] ≈ 2.0
            @test Ωₕ[CartesianIndex(5)] ≈ 4.0
        end

        @testset "Pretty printing" begin
            # Detailed is `MIME"text/plain"`, compact is the two-argument `show`
            #.
            buf = IOBuffer()
            show(buf, MIME"text/plain"(), Ωₕ)
            str = String(take!(buf))
            @test occursin("Mesh1D", str)
            @test occursin("5 points", str)
            @test !endswith(str, '\n')

            show(buf, Ωₕ)
            str_c = String(take!(buf))
            @test occursin("Mesh1D{5 pts}", str_c)
            @test !occursin('\n', str_c)
        end

        @testset "Convenience constructors" begin
            Ω_c = create_test_domain(0.0, 1.0)
            Ωₕ_default = mesh(Ω_c, 10)
            @test Ωₕ_default isa Mesh1D
            @test npoints(Ωₕ_default) == 10
            @test is_uniform(Ωₕ_default)
        end

        @testset "is_uniform caching" begin
            Ω_c = create_test_domain(0.0, 1.0)
            Ωₕ = mesh(Ω_c, 10; backend = backend())

            # Uniform mesh: repeated default-tolerance calls hit the cache and agree.
            @test is_uniform(Ωₕ)
            @test is_uniform(Ωₕ)
            @test stepsize(Ωₕ) ≈ 1 / 9

            # Mutating the mesh bumps `version`, invalidating the cached answer.
            nonunif_pts = collect(range(0.0, 1.0; length = 10))
            nonunif_pts[3] += 0.01
            set_points!(Ωₕ, nonunif_pts)
            @test !is_uniform(Ωₕ)
            @test !is_uniform(Ωₕ) # still false, re-cached against the new version
            @test_throws ArgumentError stepsize(Ωₕ)

            # An explicit `tol` always recomputes and never pollutes the default-tol cache:
            # a large enough tolerance calls the same non-uniform mesh uniform...
            @test is_uniform(Ωₕ; tol = 1.0)
            # ...but the default-tolerance answer (and the cache behind it) is unaffected.
            @test !is_uniform(Ωₕ)

            # Mutating back to a uniform layout bumps the version again and the cache
            # correctly picks up the new (true) answer.
            change_points!(Ωₕ, collect(range(0.0, 1.0; length = 10)))
            @test is_uniform(Ωₕ)
            @test stepsize(Ωₕ) ≈ 1 / 9

            # Far from the origin, rounding drift tracks the coordinates' ulp, and the
            # default tolerance scales with it (gpena/Bramble.jl#492); the magnitude comes
            # from the points, not the set `change_points!` leaves stale.
            change_points!(Ωₕ, collect(range(1.0e6, 1.0e6 + 1; length = 10)))
            @test is_uniform(Ωₕ)

            Ω32 = mesh(domain(interval(10.0f0, 11.0f0)), 11)
            @test is_uniform(Ω32)
            @test stepsize(Ω32) ≈ 0.1f0
            Ω64 = mesh(domain(interval(1.0e6, 1.0e6 + 1)), 11)
            @test is_uniform(Ω64)
            @test stepsize(Ω64) ≈ 0.1
            @test is_uniform(mesh(domain(interval(3.0e7, 3.0e7 + 1)), 11))

            # ...while a perturbed mesh at the same magnitudes, or on a tiny domain where
            # the 1e-10 floor governs, stays non-uniform.
            Random.seed!(492)
            @test !is_uniform(mesh(domain(interval(10.0f0, 11.0f0)), 11, false))
            Random.seed!(492)
            @test !is_uniform(mesh(domain(interval(1.0e6, 1.0e6 + 1)), 11, false))
            Random.seed!(492)
            @test !is_uniform(mesh(domain(interval(0.0, 1.0e-8)), 11, false))

            # The widened tolerance never admits a spacing of the opposite sign. The public
            # setters refuse such points (#698), so the geometry is written past them.
            Ω_back = mesh(domain(interval(0.0f0, 1.0f0)), 5)
            back_pts = 1.0f6 .+ Float32[0, 0.25, 0.125, 0.375, 0.5]
            @test_throws ArgumentError change_points!(Ω_back, back_pts)
            Bramble._set_points_geometry!(Ω_back, back_pts)
            @test !is_uniform(Ω_back)
            # ...and still clears the 1-ulp drift when the spacing is only 2 ulps wide.
            @test is_uniform(mesh(domain(interval(1.0f6, 1.0f6 + 400)), 2561))
        end
    end
end

end # module MeshMesh1dTests
