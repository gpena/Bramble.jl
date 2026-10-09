module MeshMarkersTests

using Test
using Bramble
using ..TestUtils: alloc_test, @test_allocs

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
        # The old bug was that `:interior` meant "not `:boundary`", and a mesh with no
        # `:boundary` key silently made that "true everywhere".
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

    @testset "warn_marker_mismatch = false is silent" begin
        # The warning has no way to tell "a mistake" from "the caller
        # redefined the label on purpose". This is that opt-out, checked in both directions
        # so it silences the warning without silently dropping the custom marker too.
        Ωₕ = @test_logs mesh(
            domain(S, :boundary => :left), (4, 4), (true, true); warn_marker_mismatch = false
        )
        @test sum(Bramble.markers(Ωₕ)[:boundary]) == 4   # the custom definition still wins
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

    @testset "Zero-allocation marker setup (#124)" begin
        # `_ensure_geometric_markers!` and `_set_markers_symbols!` used to route every
        # boundary-facet lookup through `boundary_symbol_to_dict`, allocating a fresh
        # `Dict{Symbol, CartesianIndices}` per call just to iterate or index it once. Both
        # now query `boundary_symbol_to_cartesian`'s `NamedTuple` directly, so that lookup
        # itself is zero-allocation (checked below) and `_set_markers_symbols!` measures
        # 0 B.
        #
        # `_ensure_geometric_markers!` as a whole is NOT claimed zero: it still allocates
        # 192 B for this 8x8 mesh, two `BitVector`s' worth (`falses(npoints(Ωₕ))` for the
        # boundary mask and `.!boundary_set` for its interior complement), computed on every
        # call whether or not the markers are already registered. Measured directly (not from an isolated empty-dict call, which adds
        # an unrelated ~368 B Dict-growth artifact never reachable through the public API,
        # since `domain(X)` always pre-registers `:boundary`): this 192 B is unrelated to
        # the marker fill -- a one-time mesh-construction cost, not a hot loop.
        # Whether it can be cut further (writing the interior mask directly instead of
        # negating a scratch copy) is a separate question from the fill itself.
        Ωₕ = mesh(domain(S), (8, 8), (true, true))
        idxs = Bramble.indices(Ωₕ)

        iterate_boundary_facets(idxs) = (
            c = 0;
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

    @testset "Coordinate boundary symbols (#152)" begin
        # 1D Domains
        I1 = interval(0.0, 1.0)
        Ωₕ_1d = mesh(domain(I1, :x_lo => :xmin, :x_hi => :xmax, :l => :left, :r => :right), 5, true)
        @test Bramble.index_in_marker(Ωₕ_1d, :x_lo) == Bramble.index_in_marker(Ωₕ_1d, :l)
        @test Bramble.index_in_marker(Ωₕ_1d, :x_hi) == Bramble.index_in_marker(Ωₕ_1d, :r)

        # 2D Domains
        S2 = interval(0.0, 1.0) × interval(0.0, 2.0)
        Ωₕ_2d = mesh(
            domain(
                S2,
                :xm => :xmin, :xp => :xmax, :ym => :ymin, :yp => :ymax,
                :l => :left, :r => :right, :b => :bottom, :t => :top
            ),
            (4, 5),
            (true, true)
        )
        @test Bramble.index_in_marker(Ωₕ_2d, :xm) == Bramble.index_in_marker(Ωₕ_2d, :l)
        @test Bramble.index_in_marker(Ωₕ_2d, :xp) == Bramble.index_in_marker(Ωₕ_2d, :r)
        @test Bramble.index_in_marker(Ωₕ_2d, :ym) == Bramble.index_in_marker(Ωₕ_2d, :b)
        @test Bramble.index_in_marker(Ωₕ_2d, :yp) == Bramble.index_in_marker(Ωₕ_2d, :t)

        # 3D Domains check axis alignment and resolve the 3D axis transposition ambiguity.
        #
        #     Axis 1 (x): `:xmin` <-> `:back`, `:xmax` <-> `:front`
        #     Axis 2 (y): `:ymin` <-> `:left`, `:ymax` <-> `:right`
        #     Axis 3 (z): `:zmin` <-> `:bottom`, `:zmax` <-> `:top`
        S3 = interval(0.0, 1.0) × interval(0.0, 2.0) × interval(0.0, 3.0)
        Ωₕ_3d = mesh(
            domain(
                S3,
                :xm => :xmin, :xp => :xmax,
                :ym => :ymin, :yp => :ymax,
                :zm => :zmin, :zp => :zmax,
                :bk => :back, :fr => :front,
                :lt => :left, :rt => :right,
                :bm => :bottom, :tp => :top
            ),
            (3, 4, 5),
            (true, true, true)
        )
        @test Bramble.index_in_marker(Ωₕ_3d, :xm) == Bramble.index_in_marker(Ωₕ_3d, :bk)
        @test Bramble.index_in_marker(Ωₕ_3d, :xp) == Bramble.index_in_marker(Ωₕ_3d, :fr)
        @test Bramble.index_in_marker(Ωₕ_3d, :ym) == Bramble.index_in_marker(Ωₕ_3d, :lt)
        @test Bramble.index_in_marker(Ωₕ_3d, :yp) == Bramble.index_in_marker(Ωₕ_3d, :rt)
        @test Bramble.index_in_marker(Ωₕ_3d, :zm) == Bramble.index_in_marker(Ωₕ_3d, :bm)
        @test Bramble.index_in_marker(Ωₕ_3d, :zp) == Bramble.index_in_marker(Ωₕ_3d, :tp)

        # Alias fallback in index_in_marker when only one style was registered
        Ωₕ_alias_test = mesh(domain(S2, :left => :left, :ymax => :ymax), (4, 4), (true, true))
        @test Bramble.index_in_marker(Ωₕ_alias_test, :xmin) === Bramble.index_in_marker(Ωₕ_alias_test, :left)
        @test Bramble.index_in_marker(Ωₕ_alias_test, :top) === Bramble.index_in_marker(Ωₕ_alias_test, :ymax)

        # Zero-allocation verification for boundary symbols and alias helper
        @test_allocs Bramble._boundary_symbol_alias(Val(1), :xmin)
        @test_allocs Bramble._boundary_symbol_alias(Val(2), :ymin)
        @test_allocs Bramble._boundary_symbol_alias(Val(3), :zmax)
        @test_allocs Bramble.index_in_marker(Ωₕ_alias_test, :xmin)
        @test_allocs Bramble.boundary_indices(Bramble.indices(Ωₕ_1d))
        @test_allocs Bramble.boundary_indices(Bramble.indices(Ωₕ_2d))
        @test_allocs Bramble.boundary_indices(Bramble.indices(Ωₕ_3d))
        @test_allocs Bramble.boundary_symbol_to_cartesian(Bramble.indices(Ωₕ_1d))
    end
end

# The predicate probes in `markers(space, pairs...)` (src/geometry/marker.jl): a 1D predicate
# takes the scalar coordinate, and one that does not is refused with a message naming the
# label.
@testset "Marker predicate probes" begin
    S1 = interval(0.0, 1.0)
    S2 = interval(0.0, 1.0) × interval(0.0, 2.0)

    @testset "1D predicate taking neither is refused" begin
        err = try
            Bramble.markers(S1, :broken => x -> error("no"))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        msg = sprint(showerror, err)
        @test occursin("label :broken", msg)
        @test occursin("scalar coordinate", msg)
    end

    @testset "2D predicate, wrong arity, is refused" begin
        err = try
            Bramble.markers(S2, :broken => x -> x[3] > 0.0)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        msg = sprint(showerror, err)
        @test occursin("label :broken", msg)
        @test occursin("2-element coordinate tuple", msg)
    end

    @testset "Time-dependent predicates are probed too" begin
        T = interval(0.0, 1.0)
        dm = Bramble.markers(S2, T, :moving => (x, t) -> x[1] < t, :still => (x, t) -> x[2] < 1.0)
        @test length(Bramble.conditions(dm)) == 2

        # A spatial-only predicate has no t to evaluate at, so the time form refuses it.
        err_arity = try
            Bramble.markers(S2, T, :still => x -> x[2] < 1.0)
            nothing
        catch e
            e
        end
        @test err_arity isa ArgumentError
        @test occursin("marker :still", sprint(showerror, err_arity))
        @test occursin("(x, t)", sprint(showerror, err_arity))

        err_call = try
            Bramble.markers(S2, T, :broken => (x, t) -> x[3] < t)
            nothing
        catch e
            e
        end
        @test err_call isa ArgumentError
        @test occursin("label :broken", sprint(showerror, err_call))

        err_bool = try
            Bramble.markers(S2, T, :num => (x, t) -> x[1] * t)
            nothing
        catch e
            e
        end
        @test err_bool isa ArgumentError
        @test occursin("label :num", sprint(showerror, err_bool))
        @test occursin("Float64", sprint(showerror, err_bool))
    end
end

# `(x, t, p)` conditions fixed at a time and a parameter. The pairs go
# straight to `_create_generic_markers`: `markers(space, pairs...)` would probe a three-argument
# predicate with one argument and refuse it.
@testset "Parameter-evaluated markers" begin
    dm = Bramble._create_generic_markers(
        :l => :xmin, :w => (:xmin, :xmax), :c => (x, t, p) -> x[1] < t * p
    )
    edm = dm(0.5, 3.0)
    @test edm isa Bramble.EvaluatedParametricDomainMarkers

    @test Bramble.symbols(edm) === Bramble.symbols(dm)
    @test Bramble.tuples(edm) === Bramble.tuples(dm)
    @test Bramble.labels(edm) == (:l, :w, :c)
    @test Bramble.label_identifiers(edm) == (:l, :w, :c)
    @test collect(Bramble.label_symbols(edm)) == [:l]
    @test collect(Bramble.label_tuples(edm)) == [:w]
    @test collect(Bramble.label_conditions(edm)) == [:c]
    @test length(edm) == 3 == length(dm)
    @test !isempty(edm)
    @test isempty(Bramble._create_generic_markers()(0.5, 3.0))

    # t * p = 1.5: the condition is `x -> x[1] < 1.5`, not `x[1] < p * t`'s swapped reading
    c = only(Bramble.conditions(edm))
    @test Bramble.identifier(c)((1.4,)) === true
    @test Bramble.identifier(c)((1.6,)) === false
end

# The mesh queries of src/mesh/queries.jl, on non-uniform meshes. Every expected value is
# written out from the mesh's own points (`host_points`), never read back from the function
# under test.
@testset "Mesh queries" begin
    hs(x, i) = i == 1 ? (x[2] - x[1]) / 2 :
               i == length(x) ? (x[end] - x[end - 1]) / 2 : (x[i + 1] - x[i - 1]) / 2

    @testset "1D: first index, iteration, normals" begin
        Ω1 = mesh(domain(interval(0.0, 1.0)), 7, false)
        xs = Bramble.host_points(Ω1)
        @test length(xs) == 7
        @test !all(≈(xs[2] - xs[1]), diff(xs))     # the mesh is not uniform
        @test firstindex(Ω1) == 1
        @test firstindex(Ω1, 1) == 1
        @test collect(Ω1) == xs
        @test Bramble.normal_vector(Ω1, :xmin) == (-1.0,)
        @test Bramble.normal_vector(Ω1, :right) == (1.0,)
        @test_throws ArgumentError Bramble.normal_vector(Ω1, :top)
    end

    @testset "2D: first index, iteration, normals" begin
        Ω2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (5, 4), (false, false))
        xs, ys = Bramble.host_points(Ω2)
        @test firstindex(Ω2) == CartesianIndex(1, 1)
        @test firstindex(Ω2, 1) == 1
        @test firstindex(Ω2, 2) == 1
        @test [Tuple(p) for p in Ω2] == [(xs[i], ys[j]) for j in 1:4 for i in 1:5]
        for (sym, n) in ((:xmin, (-1.0, 0.0)), (:left, (-1.0, 0.0)), (:xmax, (1.0, 0.0)),
            (:right, (1.0, 0.0)), (:ymin, (0.0, -1.0)), (:bottom, (0.0, -1.0)),
            (:ymax, (0.0, 1.0)), (:top, (0.0, 1.0)))
            @test Bramble.normal_vector(Ω2, sym) == n
        end
        @test_throws ArgumentError Bramble.normal_vector(Ω2, :zmin)
    end

    @testset "3D: normals and face symbols" begin
        Ω3 = mesh(
            domain(interval(0.0, 1.0) × interval(0.0, 2.0) × interval(0.0, 3.0)),
            (4, 5, 4), (false, false, false)
        )
        for (sym, n) in ((:xmin, (-1.0, 0.0, 0.0)), (:back, (-1.0, 0.0, 0.0)),
            (:xmax, (1.0, 0.0, 0.0)), (:front, (1.0, 0.0, 0.0)),
            (:ymin, (0.0, -1.0, 0.0)), (:left, (0.0, -1.0, 0.0)),
            (:ymax, (0.0, 1.0, 0.0)), (:right, (0.0, 1.0, 0.0)),
            (:zmin, (0.0, 0.0, -1.0)), (:bottom, (0.0, 0.0, -1.0)),
            (:zmax, (0.0, 0.0, 1.0)), (:top, (0.0, 0.0, 1.0)))
            @test Bramble.normal_vector(Ω3, sym) == n
        end
        @test_throws ArgumentError Bramble.normal_vector(Ω3, :inlet)

        for (sym, face) in ((:xmin, (1, 1)), (:back, (1, 1)), (:xmax, (1, 2)),
            (:front, (1, 2)), (:ymin, (2, 1)), (:left, (2, 1)), (:ymax, (2, 2)),
            (:right, (2, 2)), (:zmin, (3, 1)), (:bottom, (3, 1)), (:zmax, (3, 2)),
            (:top, (3, 2)))
            @test Bramble._face_of_symbol(Val(3), sym) == face
        end
        @test_throws "does not name one in 3D" Bramble._face_of_symbol(Val(3), :inlet)
    end

    @testset "Face symbols in 1D and 2D" begin
        @test Bramble._face_of_symbol(Val(1), :left) == (1, 1)
        @test Bramble._face_of_symbol(Val(1), :xmax) == (1, 2)
        @test_throws "does not name one in 1D" Bramble._face_of_symbol(Val(1), :ymin)
        for (sym, face) in ((:xmin, (1, 1)), (:left, (1, 1)), (:xmax, (1, 2)),
            (:right, (1, 2)), (:ymin, (2, 1)), (:bottom, (2, 1)), (:ymax, (2, 2)),
            (:top, (2, 2)))
            @test Bramble._face_of_symbol(Val(2), sym) == face
        end
        @test_throws "does not name one in 2D" Bramble._face_of_symbol(Val(2), :zmin)
    end

    @testset "Face masks" begin
        @test Bramble._face_mask(Val(2), (:xmin,)) == ((true, false), (false, false))
        @test Bramble._face_mask(Val(2), (:xmin, :top)) == ((true, false), (false, true))
        @test Bramble._face_mask(Val(2), (:left, :xmax)) == ((true, true), (false, false))
        @test Bramble._face_mask(Val(2), (:boundary,)) == ((true, true), (true, true))
        @test Bramble._face_mask(Val(3), (:back, :bottom)) ==
              ((true, false), (false, false), (true, false))
        @test Bramble._face_mask(Val(1), (:xmax,)) == ((false, true),)
        @test_throws ArgumentError Bramble._face_mask(Val(1), (:ymin,))
    end

    @testset "Surface: 3 points on doubly-faced axis" begin
        thin = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (2, 5), (true, true))
        full = Bramble._face_mask(Val(2), (:boundary,))
        @test_throws "not (D-1)-dimensional" Bramble._check_surface_is_thin(thin, full)
        @test Bramble._check_surface_is_thin(thin, Bramble._face_mask(Val(2), (:ymin,))) ===
              nothing
        wide = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (3, 5), (true, true))
        @test Bramble._check_surface_is_thin(wide, full) === nothing
    end

    @testset "Surface weights vs averaged spacings" begin
        # 1D: counting measure, 1 on a face point and 0 elsewhere
        Ω1 = mesh(domain(interval(0.0, 1.0)), 7, false)
        m1 = Bramble._face_mask(Val(1), (:boundary,))
        @test [Bramble._surface_weight(Ω1, m1, CartesianIndex(i)) for i in 1:7] ==
              [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        # 2D: a point on :ymin weighs the half spacing in x; a corner on two faces, the sum
        Ω2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (6, 5), (false, false))
        xs, ys = Bramble.host_points(Ω2)
        mb = Bramble._face_mask(Val(2), (:ymin,))
        for i in 1:6
            @test Bramble._surface_weight(Ω2, mb, CartesianIndex(i, 1)) ≈ hs(xs, i)
            @test Bramble._surface_weight(Ω2, mb, CartesianIndex(i, 3)) == 0.0
        end
        mc = Bramble._face_mask(Val(2), (:xmin, :ymin))
        @test Bramble._surface_weight(Ω2, mc, CartesianIndex(1, 1)) ≈ hs(xs, 1) + hs(ys, 1)
        @test Bramble._surface_weight(Ω2, mc, CartesianIndex(1, 3)) ≈ hs(ys, 3)

        # 3D: on :zmax the weight is the product of the two transverse half spacings
        Ω3 = mesh(
            domain(interval(0.0, 1.0) × interval(0.0, 2.0) × interval(0.0, 3.0)),
            (4, 5, 4), (false, false, false)
        )
        x3, y3, z3 = Bramble.host_points(Ω3)
        mz = Bramble._face_mask(Val(3), (:zmax,))
        for i in 1:4, j in 1:5

            @test Bramble._surface_weight(Ω3, mz, CartesianIndex(i, j, 4)) ≈
                  hs(x3, i) * hs(y3, j)
            @test Bramble._surface_weight(Ω3, mz, CartesianIndex(i, j, 2)) == 0.0
        end
    end
end

@testset "Time domain meshes at its evaluation t" begin
    X = interval(0.0, 1.0) × interval(0.0, 1.0)
    T = interval(0.0, 1.0)
    Ω = domain(X, T, :moving => (x, t) -> x[1] > t, :wall => :left, :sides => (:top, :bottom))

    # 5 × 5 points at x = 0, 0.25, 0.5, 0.75, 1: `x > t` holds for 2 columns at t = 0.5.
    Ωₕ = mesh(Ω(0.5), (5, 5), (true, true))
    @test count(Bramble.markers(Ωₕ)[:moving]) == 10
    @test count(Bramble.markers(Ωₕ)[:wall]) == 5
    @test count(Bramble.markers(Ωₕ)[:sides]) == 10

    # The predicate follows t: 4 columns at t = 0, none at t = 1.
    @test count(Bramble.markers(mesh(Ω(0.0), (5, 5), (true, true)))[:moving]) == 20
    @test count(Bramble.markers(mesh(Ω(1.0), (5, 5), (true, true)))[:moving]) == 0

    # Same marker set as the points-wise evaluation of the predicate.
    xs, ys = Bramble.host_points(Ωₕ)
    expected = vec([xi > 0.5 for xi in xs, _ in ys])
    @test Bramble.markers(Ωₕ)[:moving] == expected

    # 1D mesh through the same path.
    Ω1 = domain(interval(0.0, 1.0), T, :late => (x, t) -> x > t)
    @test count(Bramble.markers(mesh(Ω1(0.5), 5, true))[:late]) == 2

    # A time domain needs (x, t) predicates, at construction.
    @test_throws ArgumentError domain(X, T, :s => x -> x[1] > 0.5)
    @test_throws ArgumentError markers(interval(0.0, 1.0), T, :s => x -> x > 0.5)
end

end # module MeshMarkersTests
