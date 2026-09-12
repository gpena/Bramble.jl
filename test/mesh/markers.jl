using Test
using Bramble

# `:boundary`/`:interior` are reserved markers every mesh now carries automatically,
# computed from the mesh's own shape (see `_ensure_geometric_markers!`
# in src/mesh/marker.jl). Every case here is checked against a real mesh's marker
# BitVectors, not just that construction did or didn't throw.

@testset "Reserved geometric markers" begin
    S = interval(0.0, 1.0) × interval(0.0, 1.0)

    @testset "Default markers" begin
        Ωₕ = mesh(domain(S), (4, 4), (true, true))
        @test Set(keys(Bramble.markers(Ωₕ))) == Set([:boundary, :interior])
        @test sum(Bramble.markers(Ωₕ)[:boundary]) == 12   # 16 points, 4 strictly interior
        @test sum(Bramble.markers(Ωₕ)[:interior]) == 4
        @test Bramble.markers(Ωₕ)[:interior] == .!Bramble.markers(Ωₕ)[:boundary]
    end

    @testset "Custom label agreement" begin
        Ωₕ = mesh(domain(S, :bottom => :bottom), (4, 4), (true, true))
        @test Set(keys(Bramble.markers(Ωₕ))) == Set([:bottom, :boundary, :interior])

        Ωₕ_default = mesh(domain(S), (4, 4), (true, true))
        @test Bramble.markers(Ωₕ)[:boundary] == Bramble.markers(Ωₕ_default)[:boundary]
        @test Bramble.markers(Ωₕ)[:interior] == Bramble.markers(Ωₕ_default)[:interior]
    end

    @testset "Single-point mesh" begin
        Ωₕ = mesh(domain(interval(1.0, 1.0)), 1, true)
        @test Bramble.markers(Ωₕ)[:boundary] == [true]
        @test Bramble.markers(Ωₕ)[:interior] == [false]
    end

    @testset "Geometric interior" begin
        # The bug this closes: :interior used to mean "not :boundary", and a mesh with no
        # :boundary key silently made that "true everywhere".
        Ωₕ = mesh(domain(S, :bottom => :bottom), (4, 4), (true, true))
        Wₕ = gridspace(Ωₕ)
        u = Rₕ(Wₕ, x -> 1.0)
        v = Rₕ(Wₕ, x -> 1.0)
        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(Bramble.restrict_to(:interior, u), v))
        interior_sum = Bramble.dot(v.data, assemble(a) * u.data)
        full_sum = innerₕ(u, v)
        @test interior_sum < full_sum
        @test interior_sum ≈
            sum(Bramble.weights(Wₕ, Bramble.Innerh())[Bramble.markers(Ωₕ)[:interior]])
    end

    @testset "Reserved symbol match" begin
        Ωₕ = mesh(
            domain(S, :boundary => (:left, :right, :top, :bottom)), (4, 4), (true, true)
        )
        Ωₕ_default = mesh(domain(S), (4, 4), (true, true))
        @test Bramble.markers(Ωₕ)[:boundary] == Bramble.markers(Ωₕ_default)[:boundary]
    end

    @testset "Reserved symbol override" begin
        # `:boundary`/`:interior` were already usable as ordinary custom labels before this
        # existed, so a mismatch warns rather than errors: the custom definition wins, not
        # the geometric one, since erroring would break that pre-existing freedom.
        Ωₕ = @test_logs (:warn, r"boundary.*something other than") mesh(
            domain(S, :boundary => :left), (4, 4), (true, true)
        )
        @test Bramble.markers(Ωₕ)[:boundary] !=
            Bramble.markers(mesh(domain(S), (4, 4), (true, true)))[:boundary]
        @test sum(Bramble.markers(Ωₕ)[:boundary]) == 4   # just the :left face on a 4x4 grid
    end

    @testset "warn_marker_mismatch = false silences a deliberate redefinition" begin
        # gpena/Bramble.jl#18: the warning has no way to tell "a mistake" from "the caller
        # redefined the label on purpose" — this is that opt-out, checked in both directions
        # so it silences the warning without silently dropping the custom marker too.
        Ωₕ = @test_logs mesh(
            domain(S, :boundary => :left), (4, 4), (true, true); warn_marker_mismatch=false
        )
        @test sum(Bramble.markers(Ωₕ)[:boundary]) == 4   # the custom definition still wins

        # The default stays warn-on-mismatch — false is opt-in, not a silent global change.
        @test_logs (:warn, r"boundary.*something other than") mesh(
            domain(S, :boundary => :left), (4, 4), (true, true)
        )
    end

    @testset "Condition markers" begin
        is_geom_boundary(x) = x[1] == 0.0 || x[1] == 1.0 || x[2] == 0.0 || x[2] == 1.0

        # matches geometry exactly -- no warning, no divergence
        Ωₕ = mesh(domain(S, :boundary => is_geom_boundary), (4, 4), (true, true))
        Ωₕ_default = mesh(domain(S), (4, 4), (true, true))
        @test Bramble.markers(Ωₕ)[:boundary] == Bramble.markers(Ωₕ_default)[:boundary]

        Ωₕ2 = mesh(
            domain(S, :interior => (x -> !is_geom_boundary(x))), (4, 4), (true, true)
        )
        @test Bramble.markers(Ωₕ2)[:interior] == Bramble.markers(Ωₕ_default)[:interior]

        # a custom, non-reserved condition marker is untouched by any of this
        Ωₕ3 = mesh(domain(S, :left_half => (x -> x[1] < 0.5)), (4, 4), (true, true))
        @test haskey(Bramble.markers(Ωₕ3), :left_half)
        @test haskey(Bramble.markers(Ωₕ3), :boundary)
        @test haskey(Bramble.markers(Ωₕ3), :interior)

        # a condition meaning something else under a reserved name warns, keeps its own value
        Ωₕ4 = @test_logs (:warn, r"boundary") mesh(
            domain(S, :boundary => (x -> x[1] < 0.5)), (4, 4), (true, true)
        )
        @test sum(Bramble.markers(Ωₕ4)[:boundary]) == 8   # x[1] < 0.5 on a 4x4 grid
    end

    @testset "Empty-selection marker" begin
        # A predicate matching no grid point at all -- `index_in_marker` must come back a
        # clean all-false mask rather than throwing or producing something haskey can't see.
        Ωₕ = mesh(domain(S, :empty => (x -> x[1] > 10.0)), (4, 4), (true, true))
        @test haskey(Bramble.markers(Ωₕ), :empty)
        @test sum(Bramble.markers(Ωₕ)[:empty]) == 0
        @test !any(Bramble.index_in_marker(Ωₕ, :empty))
        @test Bramble.index_in_marker(Ωₕ, :empty) isa BitVector
    end

    @testset "Zero-allocation marker setup (gpena/Bramble.jl#124)" begin
        # `_ensure_geometric_markers!` and `_set_markers_symbols!` used to route every
        # boundary-facet lookup through `boundary_symbol_to_dict`, allocating a fresh
        # `Dict{Symbol, CartesianIndices}` per call just to iterate or index it once. Both
        # now query `boundary_symbol_to_cartesian`'s `NamedTuple` directly, so that lookup
        # itself is zero-allocation (checked below) and `_set_markers_symbols!` measures
        # 832 B -> 0 B against the pre-#124 commit (43df4c0).
        #
        # `_ensure_geometric_markers!` as a whole is NOT claimed zero: it still allocates
        # 192 B for this 8x8 mesh, two `BitVector`s' worth (`falses(npoints(Ωₕ))` for the
        # boundary mask, `.!boundary_set` for its interior complement) -- unconditionally
        # computed every call, whether or not :boundary/:interior turn out to already be
        # registered. Measured directly (not from an isolated empty-dict call, which adds
        # an unrelated ~368 B Dict-growth artifact never reachable through the public API,
        # since `domain(X)` always pre-registers `:boundary`): this 192 B is unchanged by
        # #124 and is unrelated to it -- a one-time mesh-construction cost, not a hot loop.
        # Whether it can be cut further (writing the interior mask directly instead of
        # negating a scratch copy) is a separate question from what #124 fixed.
        Ωₕ = mesh(domain(S), (8, 8), (true, true))
        idxs = Bramble.indices(Ωₕ)

        iterate_boundary_facets(idxs) = (
            c=0;
            for face in values(Bramble.boundary_symbol_to_cartesian(idxs))
                c += length(face)
            end;
            c
        )
        @test_allocs Bramble.boundary_symbol_to_cartesian(idxs)
        @test_allocs iterate_boundary_facets(idxs)

        dm = markers(S, :inlet => :left, :outlet => :right, :walls => (:top, :bottom))
        mesh_markers = Bramble.MeshMarkers()
        Bramble.process_label_for_mesh!(
            Bramble.npoints(Ωₕ), mesh_markers, Bramble.label_symbols(dm)
        )
        @test_allocs Bramble._set_markers_symbols!(mesh_markers, Bramble.symbols(dm), Ωₕ)
    end

    @testset "Corner-only marker" begin
        # A predicate matching exactly the four geometric corners of a 2D box, as its own
        # named region -- distinct from :boundary (every edge point) and from any single
        # face marker.
        is_corner(x) = (x[1] == 0.0 || x[1] == 1.0) && (x[2] == 0.0 || x[2] == 1.0)
        Ωₕ = mesh(domain(S, :corners => is_corner), (4, 4), (true, true))
        @test sum(Bramble.markers(Ωₕ)[:corners]) == 4
        @test all(Bramble.index_in_marker(Ωₕ, :corners) .<= Bramble.markers(Ωₕ)[:boundary])   # every corner is on the boundary
    end
end
