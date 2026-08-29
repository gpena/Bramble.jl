"""
Test coverage for mesh module

Focus on edge cases, marker combinations, and complex interactions
"""

import Bramble: set, markers, CartesianProduct, Mesh1D, MeshnD

@testset "Meshes Modules Coverage" begin
    @testset "Domain Edge Cases" begin
        @testset "1D Domain variations" begin
            I = interval(-1.0, 2.0)

            # Domain without markers
            X1 = domain(I)
            @test set(X1) isa CartesianProduct

            # Domain with single marker
            X2 = domain(I, markers(I, :left => x -> x[1] < -0.5))
            @test !isnothing(markers(X2))

            # Domain with multiple markers
            X3 = domain(I,
                markers(I,
                    :left => x -> x[1] < -0.5,
                    :right => x -> x[1] > 1.5,
                    :center => x -> -0.5 ≤ x[1] ≤ 1.5))
            @test !isnothing(markers(X3))
        end

        @testset "2D Domain variations" begin
            I = interval(0.0, 1.0)
            Ω = I × I

            # Domain without markers
            X1 = domain(Ω)
            @test !isnothing(X1)

            # Domain with boundary markers
            X2 = domain(Ω,
                markers(Ω,
                    :bottom => x -> x[2] < 0.01,
                    :top => x -> x[2] > 0.99,
                    :left => x -> x[1] < 0.01,
                    :right => x -> x[1] > 0.99))
            @test !isnothing(X2)

            # Domain with interior markers
            X3 = domain(Ω, markers(Ω,
                :interior => x -> 0.25 < x[1] < 0.75 && 0.25 < x[2] < 0.75))
            @test !isnothing(X3)
        end

        @testset "3D Domain" begin
            I = interval(0.0, 1.0)
            Ω = I × I × I

            X = domain(Ω, markers(Ω,
                :boundary => x -> any(x .< 0.01) || any(x .> 0.99)))
            @test !isnothing(X)
            @test set(X) isa CartesianProduct
        end
    end

    @testset "Marker Combinations" begin
        I = interval(0.0, 1.0)

        @testset "Disjoint markers" begin
            X = domain(I,
                markers(I,
                    :region1 => x -> x[1] < 0.33,
                    :region2 => x -> 0.33 ≤ x[1] < 0.67,
                    :region3 => x -> x[1] ≥ 0.67))

            Mh = mesh(X, 10, false)
            @test Mh isa Mesh1D
        end

        @testset "Overlapping markers" begin
            X = domain(I,
                markers(I,
                    :left_half => x -> x[1] ≤ 0.6,
                    :right_half => x -> x[1] ≥ 0.4,
                    :center => x -> 0.3 ≤ x[1] ≤ 0.7))

            Mh = mesh(X, 10, false)
            @test Mh isa Mesh1D
        end

        @testset "Nested markers" begin
            Ω = I × I
            X = domain(Ω,
                markers(Ω,
                    :outer => x -> all(0.1 .≤ x .≤ 0.9),
                    :middle => x -> all(0.3 .≤ x .≤ 0.7),
                    :inner => x -> all(0.4 .≤ x .≤ 0.6)))

            Mh = mesh(X, (5, 5), (false, false))
            @test Mh isa MeshnD
        end
    end

    @testset "Marker Evaluation" begin
        I = interval(0.0, 1.0)

        @testset "Boolean marker functions" begin
            # Simple threshold
            m1 = markers(I, :boundary => x -> x[1] < 0.1 || x[1] > 0.9)
            @test !isnothing(m1)

            # Complex logical expression
            m2 = markers(I, :region =>
                x -> (x[1] > 0.2 && x[1] < 0.4) || (x[1] > 0.6 && x[1] < 0.8))
            @test !isnothing(m2)
        end

        @testset "Marker with different predicates" begin
            Ω = I × I

            # Distance-based marker
            center = (0.5, 0.5)
            m1 = markers(Ω, :circle =>
                x -> sqrt((x[1] - center[1])^2 + (x[2] - center[2])^2) < 0.3)
            @test !isnothing(m1)

            # Box marker
            m2 = markers(Ω, :box => x -> all(0.2 .≤ x .≤ 0.8))
            @test !isnothing(m2)

            # Annulus marker
            m3 = markers(Ω, :annulus =>
                x -> begin
                    r = sqrt((x[1] - center[1])^2 + (x[2] - center[2])^2)
                    0.2 < r < 0.4
                end)
            @test !isnothing(m3)
        end
    end

    @testset "Set Operations" begin
        @testset "1D Sets" begin
            I1 = interval(0.0, 1.0)
            I2 = interval(-1.0, 0.5)
            I3 = interval(0.5, 2.0)

            @test I1 isa CartesianProduct
            @test I2 isa CartesianProduct
            @test I3 isa CartesianProduct
        end

        @testset "2D Sets (Cartesian Products)" begin
            I = interval(0.0, 1.0)
            J = interval(-1.0, 1.0)

            Ω1 = I × I  # Square
            @test Ω1 isa CartesianProduct

            Ω2 = I × J  # Rectangle
            @test Ω2 isa CartesianProduct

            Ω3 = J × I  # Different rectangle
            @test Ω3 isa CartesianProduct
        end

        @testset "3D Sets" begin
            I = interval(0.0, 1.0)
            J = interval(-0.5, 0.5)
            K = interval(0.0, 2.0)

            Ω1 = I × I × I  # Cube
            @test Ω1 isa CartesianProduct

            Ω2 = I × J × K  # Box
            @test Ω2 isa CartesianProduct
        end
    end

    @testset "Domain with Mesh Integration" begin
        @testset "1D Domain to Mesh" begin
            I = interval(0.0, π)
            X = domain(I, markers(I,
                :left => x -> x[1] < 0.1,
                :right => x -> x[1] > π - 0.1))

            Mh = mesh(X, 20, false)
            @test Mh isa Mesh1D
            @test npoints(Mh) == 20
            @test haskey(markers(Mh), :left)
            @test haskey(markers(Mh), :right)
        end

        @testset "2D Domain to Mesh" begin
            I = interval(0.0, 1.0)
            Ω = I × I
            X = domain(Ω, markers(Ω,
                :boundary => x -> any(x .< 0.01) || any(x .> 0.99)))

            Mh = mesh(X, (6, 6), (false, false))
            @test Mh isa MeshnD
            @test npoints(Mh) == 36
            @test haskey(markers(Mh), :boundary)
        end

        @testset "3D Domain to Mesh (small)" begin
            I = interval(0.0, 1.0)
            Ω = I × I × I
            X = domain(Ω)

            Mh = mesh(X, (3, 3, 3), (false, false, false))
            @test Mh isa MeshnD
            @test npoints(Mh) == 27
        end
    end

    @testset "Marker Access and Queries" begin
        I = interval(0.0, 1.0)
        Ω = I × I

        X = domain(Ω,
            markers(Ω,
                :left => x -> x[1] < 0.01,
                :right => x -> x[1] > 0.99,
                :bottom => x -> x[2] < 0.01,
                :top => x -> x[2] > 0.99))

        m = markers(X)
        @test !isnothing(m)

        Mh = mesh(X, (5, 5), (true, true))
        @test haskey(markers(Mh), :left)
        @test haskey(markers(Mh), :right)
        @test haskey(markers(Mh), :bottom)
        @test haskey(markers(Mh), :top)
        @test any(markers(Mh)[:left])
        @test any(markers(Mh)[:right])
    end

    @testset "Empty and Trivial Cases" begin
        @testset "Domain without markers" begin
            I = interval(0.0, 1.0)
            X = domain(I)

            @test !isnothing(X)
            # Should work with mesh even without markers
            Mh = mesh(X, 5, false)
            @test Mh isa Mesh1D
        end

        @testset "Marker that includes everything" begin
            I = interval(0.0, 1.0)
            X = domain(I, markers(I, :all => x -> true))

            @test !isnothing(X)
            Mh = mesh(X, 5, false)
            @test Mh isa Mesh1D
        end

        @testset "Marker that includes nothing" begin
            I = interval(0.0, 1.0)
            X = domain(I, markers(I, :none => x -> false))

            @test !isnothing(X)
            Mh = mesh(X, 5, false)
            @test Mh isa Mesh1D
        end
    end

    @testset "Extended Interface Methods" begin
        @testset "1D Mesh Extended Interface" begin
            I = interval(0.0, 1.0)
            M1 = mesh(domain(I), 11)

            # Collection interface
            @test size(M1) == (11,)
            @test size(M1, 1) == 11
            @test length(M1) == 11
            @test axes(M1) == (Base.OneTo(11),)
            @test axes(M1, 1) == Base.OneTo(11)
            @test firstindex(M1) == 1
            @test M1[begin] == 0.0
            @test lastindex(M1) == 11
            @test M1[end] == 1.0
            @test count(_ -> true, M1) == 11

            # Stepsize and metrics
            @test stepsize(M1) ≈ 0.1
            @test stepsize(M1, 1) ≈ 0.1
            @test hₘₐₓ(M1) ≈ 0.1
            @test hₘᵢₙ(M1) ≈ 0.1

            # locate_cell
            @test locate_cell(M1, -0.5) == 1
            @test locate_cell(M1, 0.0) == 1
            @test locate_cell(M1, 0.35) == 4
            @test locate_cell(M1, 1.0) == 10
            @test locate_cell(M1, 1.5) == 10

            # normal_vector
            @test normal_vector(M1, :left) == SVector{1, Float64}(-1.0)
            @test normal_vector(M1, :right) == SVector{1, Float64}(1.0)
            @test_throws ArgumentError normal_vector(M1, :unknown)
        end

        @testset "2D Mesh Extended Interface" begin
            I = interval(0.0, 1.0)
            J = interval(0.0, 2.0)
            M2 = mesh(domain(I × J), (11, 21))

            # Collection interface
            @test size(M2) == (11, 21)
            @test size(M2, 1) == 11
            @test size(M2, 2) == 21
            @test length(M2) == 231
            @test axes(M2) == (Base.OneTo(11), Base.OneTo(21))
            @test axes(M2, 1) == Base.OneTo(11)
            @test axes(M2, 2) == Base.OneTo(21)
            @test firstindex(M2) == CartesianIndex(1, 1)
            @test M2[begin] == (0.0, 0.0)
            @test lastindex(M2) == CartesianIndex(11, 21)
            @test M2[end] == (1.0, 2.0)
            @test count(_ -> true, M2) == 231

            # Stepsize and metrics
            @test stepsize(M2) == (stepsize(M2(1)), stepsize(M2(2)))
            @test stepsize(M2, 1) ≈ 0.1
            @test stepsize(M2, 2) ≈ 0.1
            @test hₘₐₓ(M2) ≈ hypot(0.1, 0.1)
            @test hₘᵢₙ(M2) ≈ 0.1

            # Non-uniform stepsize error assertion
            M2_nu = mesh(domain(I × J), (11, 21), (false, false))
            @test_throws AssertionError stepsize(M2_nu)

            # locate_cell
            @test locate_cell(M2, (0.35, 1.05)) == CartesianIndex(4, 11)
            @test locate_cell(M2, [0.35, 1.05]) == CartesianIndex(4, 11)

            # normal_vector
            @test normal_vector(M2, :left) == SVector{2, Float64}(-1.0, 0.0)
            @test normal_vector(M2, :right) == SVector{2, Float64}(1.0, 0.0)
            @test normal_vector(M2, :bottom) == SVector{2, Float64}(0.0, -1.0)
            @test normal_vector(M2, :top) == SVector{2, Float64}(0.0, 1.0)
            @test_throws ArgumentError normal_vector(M2, :invalid)
        end

        @testset "3D Mesh Extended Interface" begin
            I = interval(0.0, 1.0)
            M3 = mesh(domain(I × I × I), (5, 5, 5))

            @test size(M3) == (5, 5, 5)
            @test length(M3) == 125
            @test stepsize(M3) == (0.25, 0.25, 0.25)
            @test hₘₐₓ(M3) ≈ hypot(0.25, 0.25, 0.25)
            @test hₘᵢₙ(M3) ≈ 0.25

            # Points in 1D submesh are [0.0, 0.25, 0.5, 0.75, 1.0]
            # For coordinate 0.5, the bounding cell index is 3 (interval [0.5, 0.75])
            @test locate_cell(M3, (0.5, 0.5, 0.5)) == CartesianIndex(3, 3, 3)
            @test locate_cell(M3, (0.1, 0.3, 0.8)) == CartesianIndex(1, 2, 4)

            @test normal_vector(M3, :back) == SVector{3, Float64}(-1.0, 0.0, 0.0)
            @test normal_vector(M3, :front) == SVector{3, Float64}(1.0, 0.0, 0.0)
            @test normal_vector(M3, :left) == SVector{3, Float64}(0.0, -1.0, 0.0)
            @test normal_vector(M3, :right) == SVector{3, Float64}(0.0, 1.0, 0.0)
            @test normal_vector(M3, :bottom) == SVector{3, Float64}(0.0, 0.0, -1.0)
            @test normal_vector(M3, :top) == SVector{3, Float64}(0.0, 0.0, 1.0)
        end
    end
end

# A mesh type that implements nothing, used to reach the interface fallbacks.
struct BareMesh <: Bramble.AbstractMeshType{1} end

@testset "Mesh interface coverage" begin
    import Bramble: generate_indices, interior_indices, _extract_linear_index,
                    spacing_for_derivative, forward_spacing_for_derivative,
                    cell_measures, normal_vector
    using StaticArrays

    Ωₕ = mesh(domain(interval(0.0, 1.0)), 5, true)
    Ω2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (4, 3), (true, true))

    @testset "generate_indices accepts an SVector of counts" begin
        @test generate_indices(SVector(4, 3)) == CartesianIndices((4, 3))
        @test generate_indices((4, 3)) == generate_indices(SVector(4, 3))
        @test generate_indices(5) == CartesianIndices((5,))
    end

    @testset "interior_indices leaves a collapsed axis alone" begin
        # an axis with one point cannot lose its boundary, so the range passes through
        Ωc = mesh(domain(interval(0.0, 1.0) × interval(2.0, 2.0)), (5, 1), (true, true))
        ii = interior_indices(Ωc)
        @test size(ii, 2) == 1                 # the collapsed axis is untouched
        @test size(ii, 1) == 3                 # the other axis loses both ends
    end

    @testset "out-of-range indices throw BoundsError" begin
        @test_throws BoundsError point(Ωₕ, 99)
        @test_throws BoundsError point(Ωₕ, 0)
        @test_throws BoundsError point(Ω2, CartesianIndex(99, 1))
        @test_throws BoundsError point(Ω2, CartesianIndex(1, 99))
        @test_throws BoundsError half_point(Ωₕ, 99)

        # the CartesianIndex{1} path on a 1D mesh
        @test point(Ωₕ, CartesianIndex(2)) == point(Ωₕ, 2)
        @test _extract_linear_index(CartesianIndex(3)) == 3
        @test _extract_linear_index(3) == 3
    end

    @testset "interface fallbacks error on a type that implements nothing" begin
        @test_throws ErrorException eltype(BareMesh())
        @test_throws ErrorException eltype(BareMesh)
    end

    @testset "collection interface" begin
        @test firstindex(Ωₕ) == 1
        @test lastindex(Ωₕ) == npoints(Ωₕ)
        @test firstindex(Ω2, 1) == 1
        @test firstindex(Ω2, 2) == 1
        @test lastindex(Ω2, 1) == size(Ω2, 1)
        @test lastindex(Ω2, 2) == size(Ω2, 2)
    end

    @testset "unknown boundary symbols are rejected in every dimension" begin
        box3 = box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        Ω3 = mesh(domain(box3), (3, 3, 3), (true, true, true))
        @test_throws ArgumentError normal_vector(Ωₕ, :nonsense)
        @test_throws ArgumentError normal_vector(Ω2, :nonsense)
        @test_throws ArgumentError normal_vector(Ω3, :nonsense)
        @test normal_vector(Ω3, :front) == SVector(1.0, 0.0, 0.0)
    end

    @testset "cell_measures returns every cell width" begin
        @test cell_measures(Ωₕ) == half_spacings(Ωₕ)
        @test length(cell_measures(Ωₕ)) == npoints(Ωₕ)

        cm = cell_measures(Ω2)
        @test cm isa NTuple{2, Any}
        @test cm[1] == cell_measures(Ω2(1))
        @test cm[2] == cell_measures(Ω2(2))
    end

    @testset "CartesianIndex forms of the spacings" begin
        for i in 1:npoints(Ωₕ)
            @test spacing(Ωₕ, CartesianIndex(i)) == spacing(Ωₕ, i)
            @test half_spacing(Ωₕ, CartesianIndex(i)) == half_spacing(Ωₕ, i)
            @test forward_spacing(Ωₕ, CartesianIndex(i)) == forward_spacing(Ωₕ, i)
        end
    end

    @testset "derivative spacings vanish at the missing neighbour" begin
        N = npoints(Ωₕ)
        # backward difference has no neighbour at the first point
        @test spacing_for_derivative(Ωₕ, 1) == 0
        @test spacing_for_derivative(Ωₕ, 2) == spacing(Ωₕ, 2)
        @test spacing_for_derivative(Ωₕ, CartesianIndex(2)) == spacing(Ωₕ, 2)
        # forward difference has none at the last
        @test forward_spacing_for_derivative(Ωₕ, N) == 0
        @test forward_spacing_for_derivative(Ωₕ, 1) == forward_spacing(Ωₕ, 1)
        @test forward_spacing_for_derivative(Ωₕ, CartesianIndex(1)) ==
              forward_spacing(Ωₕ, 1)
    end

    @testset "copy is deep in the data and shallow in the geometry" begin
        c1 = copy(Ωₕ)
        @test c1 isa Mesh1D
        @test points(c1) == points(Ωₕ)
        @test points(c1) !== points(Ωₕ)          # data copied
        @test set(c1) === set(Ωₕ)                # geometry shared
        @test markers(c1) !== markers(Ωₕ)
        points(c1)[1] = -99.0
        @test points(Ωₕ)[1] != -99.0             # the original is untouched

        c2 = copy(Ω2)
        @test c2 isa MeshnD
        @test npoints(c2, Tuple) == npoints(Ω2, Tuple)
        @test c2(1) !== Ω2(1)                    # submeshes copied
        @test points(c2(1)) == points(Ω2(1))
    end

    @testset "refinement is a no-op where there is nothing to refine" begin
        # collapsed mesh: a single point, so no interval to halve
        Ωpt = mesh(domain(interval(3.0, 3.0)), 1, true)
        @test npoints(Ωpt) == 1
        iterative_refinement!(Ωpt)
        @test npoints(Ωpt) == 1
        iterative_refinement!(Ωpt, markers(domain(interval(3.0, 3.0))))
        @test npoints(Ωpt) == 1

        # single point on a non-degenerate interval: not collapsed, but still
        # has no interval to halve, so refinement must leave it untouched
        Ω1 = mesh(domain(interval(0.0, 1.0)), 1, true)
        @test !Bramble.is_collapsed(Ω1)
        @test npoints(Ω1) == 1
        iterative_refinement!(Ω1)
        @test npoints(Ω1) == 1
        @test point(Ω1, 1) == 0.0

        # a normal mesh does refine
        Ωr = mesh(domain(interval(0.0, 1.0)), 4, true)
        iterative_refinement!(Ωr)
        @test npoints(Ωr) == 2 * 4 - 1
    end
end

@testset "Cached spacings stay consistent with the points" begin
    import Bramble: spacings, spacings!, spacing!, backward_spacings_for_derivative,
                    forward_spacings_for_derivative

    # The invariant the cache has to hold, stated independently of the cache itself.
    backward(pts, i) = i == 1 ? pts[2] - pts[1] : pts[i] - pts[i - 1]

    @testset "agrees with the accessors on construction" begin
        for unif in (true, false), n in (2, 5, 17)

            Ωₕ = mesh(domain(interval(0.0, 1.0)), n, unif)
            pts = points(Ωₕ)
            @test length(spacings(Ωₕ)) == npoints(Ωₕ)
            @test all(spacings(Ωₕ)[i] ≈ backward(pts, i) for i in 1:n)
            @test all(spacing(Ωₕ, i) == spacings(Ωₕ)[i] for i in 1:n)
            # forward_spacing reads the same vector one entry along
            @test all(forward_spacing(Ωₕ, i) == spacings(Ωₕ)[i == n ? n : i + 1]
            for i in 1:n)
        end
    end

    @testset "rebuilt by every path that changes the points" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 5, true)

        iterative_refinement!(Ωₕ)
        @test length(spacings(Ωₕ)) == npoints(Ωₕ) == 9
        @test all(spacings(Ωₕ)[i] ≈ backward(points(Ωₕ), i) for i in 1:9)

        # change_points! keeps the point count, so the replacement has to match.
        new_pts = [0.0, 0.05, 0.1, 0.2, 0.4, 0.5, 0.7, 0.9, 1.0]
        change_points!(Ωₕ, new_pts)
        @test points(Ωₕ) == new_pts
        @test all(spacings(Ωₕ)[i] ≈ backward(points(Ωₕ), i) for i in 1:9)

        # half_spacings are derived from the spacings, so they must agree too
        @test half_spacing(Ωₕ, 1) ≈ spacings(Ωₕ)[1] * 0.5
        @test half_spacing(Ωₕ, 3) ≈ (spacings(Ωₕ)[3] + spacings(Ωₕ)[4]) * 0.5
    end

    @testset "copy is independent" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 5, true)
        c = copy(Ωₕ)
        @test spacings(c) == spacings(Ωₕ)
        @test spacings(c) !== spacings(Ωₕ)
        change_points!(c, [0.0, 0.1, 0.2, 0.7, 1.0])
        @test spacings(c) != spacings(Ωₕ)
        @test all(spacings(Ωₕ)[i] ≈ backward(points(Ωₕ), i) for i in 1:5)
    end

    @testset "collapsed mesh has zero spacing" begin
        Ωc = mesh(domain(interval(3.0, 3.0)), 1, true)
        @test spacings(Ωc) == [0.0]
        @test spacing(Ωc, 1) == 0.0
    end

    @testset "derivative views select the right entries" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 5, false)
        n = npoints(Ωₕ)
        bwd = backward_spacings_for_derivative(Ωₕ)
        fwd = forward_spacings_for_derivative(Ωₕ)
        # Entry 1 of bwd and the last of fwd are not meaningful; the engines never read
        # them, so only the interior stencil is asserted here.
        @test all(bwd[i] == Bramble.spacing_for_derivative(Ωₕ, i) for i in 2:n)
        @test all(fwd[i] == Bramble.forward_spacing_for_derivative(Ωₕ, i)
        for i in 1:(n - 1))
    end
end
