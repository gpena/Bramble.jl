module MeshMeshesTests

# Unit tests for mesh edge cases, multi-dimensional domains, and boundary queries.
# Focuses on domain variations, marker combinations, predicates, and interface fallbacks.

using Test
using Bramble
using Bramble: hₘᵢₙ, normal_vector
import Bramble:
                set, markers, CartesianProduct, Mesh1D, MeshnD, normal_vector, hₘᵢₙ, is_collapsed,
                change_points!, half_point, half_spacing, indices, is_uniform, locate_cell,
                point, stepsize

@testset "Comprehensive mesh test suite" begin
    @testset "Domain edge cases" begin
        @testset "One-dimensional domain variations" begin
            I = interval(-1.0, 2.0)

            # Domain without markers
            X1 = domain(I)
            @test set(X1) isa CartesianProduct

            # `domain(X)` with no markers still declares `:boundary` itself
            @test Set(labels(markers(X1))) == Set((:boundary,))

            # Domain with single marker
            X2 = domain(I, markers(I, :left => x -> x[1] < -0.5))
            @test Set(labels(markers(X2))) == Set((:left,))

            # Domain with multiple markers
            X3 = domain(
                I,
                markers(
                    I,
                    :left => x -> x[1] < -0.5,
                    :right => x -> x[1] > 1.5,
                    :center => x -> -0.5 ≤ x[1] ≤ 1.5
                )
            )
            @test Set(labels(markers(X3))) == Set((:left, :right, :center))
        end

        @testset "Two-dimensional domain variations" begin
            I = interval(0.0, 1.0)
            Ω = I × I

            # Domain without markers
            X1 = domain(Ω)
            @test set(X1) === Ω
            @test Set(labels(markers(X1))) == Set((:boundary,))

            # Domain with boundary markers
            X2 = domain(
                Ω,
                markers(
                    Ω,
                    :bottom => x -> x[2] < 0.01,
                    :top => x -> x[2] > 0.99,
                    :left => x -> x[1] < 0.01,
                    :right => x -> x[1] > 0.99
                )
            )
            @test Set(labels(markers(X2))) == Set((:bottom, :top, :left, :right))

            # Domain with interior markers
            X3 = domain(
                Ω, markers(Ω, :interior => x -> 0.25 < x[1] < 0.75 && 0.25 < x[2] < 0.75)
            )
            @test Set(labels(markers(X3))) == Set((:interior,))
        end

        @testset "Three-dimensional domain" begin
            I = interval(0.0, 1.0)
            Ω = I × I × I

            X = domain(Ω, markers(Ω, :boundary => x -> any(x .< 0.01) || any(x .> 0.99)))
            @test Set(labels(markers(X))) == Set((:boundary,))
            @test set(X) isa CartesianProduct
        end
    end

    @testset "Marker combinations" begin
        I = interval(0.0, 1.0)

        @testset "Disjoint markers" begin
            X = domain(
                I,
                markers(
                    I,
                    :region1 => x -> x[1] < 0.33,
                    :region2 => x -> 0.33 ≤ x[1] < 0.67,
                    :region3 => x -> x[1] ≥ 0.67
                )
            )

            # A uniform mesh: this testset is about marker semantics, not about spacing,
            # and a random grid only makes whether any point lands in a given third a
            # matter of luck rather than a property of the code.
            Mh = mesh(X, 10, true)
            @test Mh isa Mesh1D

            # the three predicates partition the line, so their masks must partition the
            # grid: every point is in exactly one of them
            r1 = Bramble.index_in_marker(Mh, :region1)
            r2 = Bramble.index_in_marker(Mh, :region2)
            r3 = Bramble.index_in_marker(Mh, :region3)
            @test all(r1 .+ r2 .+ r3 .== 1)
            @test count(r1) > 0 && count(r2) > 0 && count(r3) > 0
        end

        @testset "Overlapping markers" begin
            X = domain(
                I,
                markers(
                    I,
                    :left_half => x -> x[1] ≤ 0.6,
                    :right_half => x -> x[1] ≥ 0.4,
                    :center => x -> 0.3 ≤ x[1] ≤ 0.7
                )
            )

            # Uniform again, for the same reason: whether a point falls in the overlap
            # is not something these assertions should be left to draw for.
            Mh = mesh(X, 10, true)
            @test Mh isa Mesh1D

            # these deliberately overlap: 0.4 ≤ x ≤ 0.6 is in both halves, and the whole
            # line is covered by the two of them
            left = Bramble.index_in_marker(Mh, :left_half)
            right = Bramble.index_in_marker(Mh, :right_half)
            @test any(left .& right)
            @test all(left .| right)
            # and :center straddles the seam: it meets both halves rather than sitting
            # inside either one
            center = Bramble.index_in_marker(Mh, :center)
            @test any(center .& left)
            @test any(center .& right)
        end

        @testset "Nested markers" begin
            Ω = I × I
            X = domain(
                Ω,
                markers(
                    Ω,
                    :outer => x -> all(0.1 .≤ x .≤ 0.9),
                    :middle => x -> all(0.3 .≤ x .≤ 0.7),
                    :inner => x -> all(0.4 .≤ x .≤ 0.6)
                )
            )

            # A uniform 5x5 grid puts points at 0, 0.25, 0.5, 0.75 and 1 along each axis,
            # so every one of the three boxes catches at least one of them. On a random
            # grid the innermost box is usually empty, which would make the nesting
            # assertions below pass vacuously.
            Mh = mesh(X, (5, 5), (true, true))
            @test Mh isa MeshnD

            # nested boxes, so the masks must nest too: inner ⊆ middle ⊆ outer
            outer = Bramble.index_in_marker(Mh, :outer)
            middle = Bramble.index_in_marker(Mh, :middle)
            inner = Bramble.index_in_marker(Mh, :inner)
            @test all(inner .<= middle)
            @test all(middle .<= outer)
            # and the nesting above is only meaningful because none of the masks is empty
            @test count(inner) > 0
            @test count(middle) > 0
            @test count(outer) > 0
        end
    end

    @testset "Marker evaluation" begin
        I = interval(0.0, 1.0)

        # Each predicate is bound once and used twice: to declare the marker, and as the
        # oracle the resulting mask is compared against point by point. Comparing against
        # the same function the marker was built from is what makes this independent of
        # where the grid points happen to fall, rather than a hand-counted constant that
        # would have to be rederived whenever the mesh changes.
        @testset "Boolean marker functions" begin
            threshold = x -> x[1] < 0.1 || x[1] > 0.9
            region = x -> (x[1] > 0.2 && x[1] < 0.4) || (x[1] > 0.6 && x[1] < 0.8)

            Ωₕ = mesh(domain(I, markers(I, :boundary => threshold, :region => region)), 11, true)
            mask(pred) = BitVector(pred(point(Ωₕ, i)) for i in indices(Ωₕ))

            @test Bramble.index_in_marker(Ωₕ, :boundary) == mask(threshold)
            @test Bramble.index_in_marker(Ωₕ, :region) == mask(region)

            # and neither is vacuous nor everything: an always-false predicate would
            # satisfy the comparison above against its own equally empty oracle. The
            # counts are deliberately not pinned to a literal -- `range(0, 1, 11)` lands a
            # point at 0.2000000000000000111, which is > 0.2, so :region catches three
            # points rather than the two the interval endpoints suggest.
            @test count(Bramble.index_in_marker(Ωₕ, :boundary)) == 2   # x = 0.0 and x = 1.0
            @test 0 < count(Bramble.index_in_marker(Ωₕ, :region)) < npoints(Ωₕ)
        end

        @testset "Marker with different predicates" begin
            Ω = I × I
            center = (0.5, 0.5)
            radius = x -> sqrt((x[1] - center[1])^2 + (x[2] - center[2])^2)

            circle = x -> radius(x) < 0.3
            box = x -> all(0.2 .≤ x .≤ 0.8)
            annulus = x -> 0.2 < radius(x) < 0.4

            Ωₕ = mesh(
                domain(Ω, markers(Ω, :circle => circle, :box => box, :annulus => annulus)),
                (11, 11), (true, true)
            )
            mask(pred) = BitVector(pred(point(Ωₕ, I2)) for I2 in vec(indices(Ωₕ)))

            @test Bramble.index_in_marker(Ωₕ, :circle) == mask(circle)
            @test Bramble.index_in_marker(Ωₕ, :box) == mask(box)
            @test Bramble.index_in_marker(Ωₕ, :annulus) == mask(annulus)

            # the shapes relate as their definitions say. The disc (r < 0.3) and the
            # annulus (0.2 < r < 0.4) are not disjoint -- they share the band
            # 0.2 < r < 0.3 -- and the annulus also reaches beyond the disc.
            circle_mask = Bramble.index_in_marker(Ωₕ, :circle)
            annulus_mask = Bramble.index_in_marker(Ωₕ, :annulus)
            @test any(circle_mask .& annulus_mask)
            @test any(annulus_mask .& .!circle_mask)
            @test 0 < count(circle_mask) < npoints(Ωₕ)
            @test 0 < count(Bramble.index_in_marker(Ωₕ, :box)) < npoints(Ωₕ)
        end
    end

    @testset "Mesh integration" begin
        @testset "One-dimensional domain to mesh" begin
            I = interval(0.0, π)
            X = domain(
                I, markers(I, :left => x -> x[1] < 0.1, :right => x -> x[1] > π - 0.1)
            )

            Mh = mesh(X, 20, false)
            @test Mh isa Mesh1D
            @test npoints(Mh) == 20
            @test haskey(markers(Mh), :left)
            @test haskey(markers(Mh), :right)
        end

        @testset "Two-dimensional domain to mesh" begin
            I = interval(0.0, 1.0)
            Ω = I × I
            X = domain(Ω, markers(Ω, :boundary => x -> any(x .< 0.01) || any(x .> 0.99)))

            # The custom :boundary above is a coordinate-threshold predicate, which does not
            # track a non-uniform ((false, false)) mesh's actual boundary *indices* exactly —
            # a deliberate mismatch with the geometric definition, not a mistake, so it is
            # silenced explicitly rather than left to print an unrelated warning on every run
            # of this testset (point 18, gpena/Bramble.jl#18).
            Mh = mesh(X, (6, 6), (false, false); warn_marker_mismatch = false)
            @test Mh isa MeshnD
            @test npoints(Mh) == 36
            @test haskey(markers(Mh), :boundary)
        end

        @testset "Three-dimensional domain to mesh" begin
            I = interval(0.0, 1.0)
            Ω = I × I × I
            X = domain(Ω)

            Mh = mesh(X, (3, 3, 3), (false, false, false))
            @test Mh isa MeshnD
            @test npoints(Mh) == 27
        end
    end

    @testset "Marker access & queries" begin
        I = interval(0.0, 1.0)
        Ω = I × I

        X = domain(
            Ω,
            markers(
                Ω,
                :left => x -> x[1] < 0.01,
                :right => x -> x[1] > 0.99,
                :bottom => x -> x[2] < 0.01,
                :top => x -> x[2] > 0.99
            )
        )

        @test Set(labels(markers(X))) == Set((:left, :right, :bottom, :top))

        Mh = mesh(X, (5, 5), (true, true))
        @test haskey(markers(Mh), :left)
        @test haskey(markers(Mh), :right)
        @test haskey(markers(Mh), :bottom)
        @test haskey(markers(Mh), :top)
        @test any(markers(Mh)[:left])
        @test any(markers(Mh)[:right])
    end

    @testset "Empty & trivial cases" begin
        @testset "Domain without markers" begin
            I = interval(0.0, 1.0)
            X = domain(I)

            # Should work with mesh even without markers -- `domain(X)` declares only the
            # `:boundary` one it adds itself
            Mh = mesh(X, 5, false)
            @test Mh isa Mesh1D
            @test count(Bramble.index_in_marker(Mh, :boundary)) == 2   # the two endpoints
        end

        @testset "Marker that includes everything" begin
            I = interval(0.0, 1.0)
            X = domain(I, markers(I, :all => x -> true))

            Mh = mesh(X, 5, false)
            @test Mh isa Mesh1D
            @test all(Bramble.index_in_marker(Mh, :all))
            @test count(Bramble.index_in_marker(Mh, :all)) == npoints(Mh)
        end

        @testset "Marker that includes nothing" begin
            I = interval(0.0, 1.0)
            X = domain(I, markers(I, :none => x -> false))

            Mh = mesh(X, 5, false)
            @test Mh isa Mesh1D
            @test !any(Bramble.index_in_marker(Mh, :none))
        end
    end

    @testset "Extended interface" begin
        @testset "One-dimensional extended interface" begin
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
            @test normal_vector(M1, :xmin) == (-1.0,)
            @test normal_vector(M1, :xmax) == (1.0,)
            @test normal_vector(M1, :left) == (-1.0,)
            @test normal_vector(M1, :right) == (1.0,)
            @test_throws ArgumentError normal_vector(M1, :unknown)
        end

        @testset "Two-dimensional extended interface" begin
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
            # hₘᵢₙ is the diagonal of the smallest cell, matching hₘₐₓ; the smallest
            # extent along one coordinate is that submesh's own hₘᵢₙ
            @test hₘᵢₙ(M2) ≈ hypot(0.1, 0.1)
            @test hₘᵢₙ(M2(1)) ≈ 0.1
            @test hₘᵢₙ(M2(2)) ≈ 0.1

            # Non-uniform stepsize error assertion
            M2_nu = mesh(domain(I × J), (11, 21), (false, false))
            @test_throws ArgumentError stepsize(M2_nu)

            # is_uniform on the D-dim path (`all(ntuple(...))`, gpena/Bramble.jl#332)
            @test is_uniform(M2)
            @test !is_uniform(M2_nu)

            # Uniform along one axis, non-uniform along the other: the D-dim `stepsize`
            # no longer runs its own redundant `is_uniform` scan and relies entirely on
            # each per-axis `stepsize` call to validate and throw.
            M2_mixed = mesh(domain(I × J), (11, 21), (true, false))
            @test !is_uniform(M2_mixed)
            @test_throws ArgumentError stepsize(M2_mixed)

            # locate_cell
            @test locate_cell(M2, (0.35, 1.05)) == CartesianIndex(4, 11)
            @test locate_cell(M2, [0.35, 1.05]) == CartesianIndex(4, 11)

            # normal_vector
            @test normal_vector(M2, :xmin) == normal_vector(M2, :left) == (-1.0, 0.0)
            @test normal_vector(M2, :xmax) == normal_vector(M2, :right) == (1.0, 0.0)
            @test normal_vector(M2, :ymin) == normal_vector(M2, :bottom) == (0.0, -1.0)
            @test normal_vector(M2, :ymax) == normal_vector(M2, :top) == (0.0, 1.0)
            @test_throws ArgumentError normal_vector(M2, :invalid)
        end

        @testset "Three-dimensional extended interface" begin
            I = interval(0.0, 1.0)
            M3 = mesh(domain(I × I × I), (5, 5, 5))

            @test size(M3) == (5, 5, 5)
            @test length(M3) == 125
            @test stepsize(M3) == (0.25, 0.25, 0.25)
            @test hₘₐₓ(M3) ≈ hypot(0.25, 0.25, 0.25)
            @test hₘᵢₙ(M3) ≈ hypot(0.25, 0.25, 0.25)
            @test all(hₘᵢₙ(M3(d)) ≈ 0.25 for d in 1:3)

            # Points in 1D submesh are [0.0, 0.25, 0.5, 0.75, 1.0]
            # For coordinate 0.5, the bounding cell index is 3 (interval [0.5, 0.75])
            @test locate_cell(M3, (0.5, 0.5, 0.5)) == CartesianIndex(3, 3, 3)
            @test locate_cell(M3, (0.1, 0.3, 0.8)) == CartesianIndex(1, 2, 4)

            @test normal_vector(M3, :xmin) == normal_vector(M3, :back) == (-1.0, 0.0, 0.0)
            @test normal_vector(M3, :xmax) == normal_vector(M3, :front) == (1.0, 0.0, 0.0)
            @test normal_vector(M3, :ymin) == normal_vector(M3, :left) == (0.0, -1.0, 0.0)
            @test normal_vector(M3, :ymax) == normal_vector(M3, :right) == (0.0, 1.0, 0.0)
            @test normal_vector(M3, :zmin) == normal_vector(M3, :bottom) == (0.0, 0.0, -1.0)
            @test normal_vector(M3, :zmax) == normal_vector(M3, :top) == (0.0, 0.0, 1.0)
        end
    end
end

# A mesh type that implements nothing, used to reach the interface fallbacks.
struct BareMesh <: Bramble.AbstractMeshType{1} end

@testset "Interface coverage" begin
    import Bramble:
                    generate_indices,
                    interior_indices,
                    _extract_linear_index,
                    spacing_for_derivative,
                    forward_spacing_for_derivative,
                    cell_measures,
                    normal_vector,
                    half_spacings

    Ωₕ = mesh(domain(interval(0.0, 1.0)), 5, true)
    Ω2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (4, 3), (true, true))

    @testset "generate_indices" begin
        @test generate_indices((4, 3)) == CartesianIndices((4, 3))
    end

    @testset "interior_indices with collapsed axis" begin
        # an axis with one point cannot lose its boundary, so the range passes through
        Ωc = mesh(domain(interval(0.0, 1.0) × interval(2.0, 2.0)), (5, 1), (true, true))
        ii = interior_indices(Ωc)
        @test size(ii, 2) == 1                 # the collapsed axis is untouched
        @test size(ii, 1) == 3                 # the other axis loses both ends
    end

    @testset "Out-of-range indexing" begin
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

    @testset "Unimplemented interface fallback" begin
        @test_throws ErrorException eltype(BareMesh())
        @test_throws ErrorException eltype(BareMesh)
    end

    @testset "Collection interface" begin
        # the one-argument forms are asserted with the 1D mesh above; these are the
        # per-dimension ones
        @test firstindex(Ω2, 1) == 1
        @test firstindex(Ω2, 2) == 1
        @test lastindex(Ω2, 1) == size(Ω2, 1)
        @test lastindex(Ω2, 2) == size(Ω2, 2)
    end

    @testset "Unknown boundary symbols" begin
        box3 = box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        Ω3 = mesh(domain(box3), (3, 3, 3), (true, true, true))
        @test_throws ArgumentError normal_vector(Ωₕ, :nonsense)
        @test_throws ArgumentError normal_vector(Ω2, :nonsense)
        @test_throws ArgumentError normal_vector(Ω3, :nonsense)
        @test normal_vector(Ω3, :front) == (1.0, 0.0, 0.0)
    end

    @testset "cell_measures widths" begin
        @test cell_measures(Ωₕ) == half_spacings(Ωₕ)
        @test length(cell_measures(Ωₕ)) == npoints(Ωₕ)

        cm = cell_measures(Ω2)
        @test cm isa NTuple{2, Any}
        @test cm[1] == cell_measures(Ω2(1))
        @test cm[2] == cell_measures(Ω2(2))
    end

    @testset "CartesianIndex spacings" begin
        for i in 1:npoints(Ωₕ)
            @test spacing(Ωₕ, CartesianIndex(i)) == spacing(Ωₕ, i)
            @test half_spacing(Ωₕ, CartesianIndex(i)) == half_spacing(Ωₕ, i)
            @test forward_spacing(Ωₕ, CartesianIndex(i)) == forward_spacing(Ωₕ, i)
        end
    end

    @testset "Missing neighbour spacings" begin
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

    @testset "Deep versus shallow copy" begin
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

    @testset "Trivial refinement" begin
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

@testset "Cached spacings" begin
    import Bramble:
                    spacings,
                    spacings!,
                    spacing!,
                    backward_spacings_for_derivative,
                    forward_spacings_for_derivative

    # The invariant the cache has to hold, stated independently of the cache itself.
    backward(pts, i) = i == 1 ? pts[2] - pts[1] : pts[i] - pts[i - 1]

    @testset "Accessor agreement" begin
        for unif in (true, false), n in (2, 5, 17)

            Ωₕ = mesh(domain(interval(0.0, 1.0)), n, unif)
            pts = points(Ωₕ)
            @test length(spacings(Ωₕ)) == npoints(Ωₕ)
            @test all(spacings(Ωₕ)[i] ≈ backward(pts, i) for i in 1:n)
            @test all(spacing(Ωₕ, i) == spacings(Ωₕ)[i] for i in 1:n)
            # forward_spacing reads the same vector one entry along
            @test all(
                forward_spacing(Ωₕ, i) == spacings(Ωₕ)[i == n ? n : i + 1] for i in 1:n
            )
        end
    end

    @testset "Point change rebuild" begin
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

    @testset "Independent copy" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 5, true)
        c = copy(Ωₕ)
        @test spacings(c) == spacings(Ωₕ)
        @test spacings(c) !== spacings(Ωₕ)
        change_points!(c, [0.0, 0.1, 0.2, 0.7, 1.0])
        @test spacings(c) != spacings(Ωₕ)
        @test all(spacings(Ωₕ)[i] ≈ backward(points(Ωₕ), i) for i in 1:5)
    end

    @testset "Collapsed mesh spacing" begin
        Ωc = mesh(domain(interval(3.0, 3.0)), 1, true)
        @test spacings(Ωc) == [0.0]
        @test spacing(Ωc, 1) == 0.0
    end

    @testset "Derivative views" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 5, false)
        n = npoints(Ωₕ)
        bwd = backward_spacings_for_derivative(Ωₕ)
        fwd = forward_spacings_for_derivative(Ωₕ)
        # Entry 1 of bwd and the last of fwd are not meaningful; the engines never read
        # them, so only the interior stencil is asserted here.
        @test all(bwd[i] == Bramble.spacing_for_derivative(Ωₕ, i) for i in 2:n)
        @test all(
            fwd[i] == Bramble.forward_spacing_for_derivative(Ωₕ, i) for i in 1:(n - 1)
        )
    end
end

end # module MeshMeshesTests
