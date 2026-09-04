using Test
using Bramble
using Bramble: Marker, MarkerPair, Domain, DomainMarkers,
               EvaluatedDomainMarkers, dim, set, markers, labels, CartesianProduct
using Bramble: get_boundary_symbols, label, identifier, domain, symbols, tuples, conditions
using Bramble: marker_identifiers, process_identifier, marker_symbols,
               marker_tuples, marker_conditions
using Bramble: label_identifiers, label_symbols, label_tuples, label_conditions, point_type,
               topo_dim, is_collapsed
using StaticArrays

if !@isdefined(alloc_test)
    @inline function alloc_test(f::F, args...; kwargs...) where {F}
        f(args...; kwargs...)
        return @allocated(f(args...; kwargs...))
    end
end

if !@isdefined(var"@test_allocs")
    macro test_allocs(call_expr)
        if Meta.isexpr(call_expr, :call)
            fn = call_expr.args[1]
            args = call_expr.args[2:end]
            quote
                @test alloc_test($(esc(fn)), $(map(esc, args)...)) == 0
            end
        else
            quote
                let
                    $(esc(call_expr))
                    @test (@allocated $(esc(call_expr))) == 0
                end
            end
        end
    end
end

@testset "Computational domains" begin
    # Test data setup: Cartesian products and coordinate indicator predicates.
    I1D = interval(0.0, 1.0)
    I2D = interval(0.0f0, 1.0f0) × interval(2.0f0, 3.0f0) # Float32
    I3D = interval(0.0, 1.0) × interval(2.0, 3.0) × interval(4.0, 5.0)

    func1 = x -> x[1] > 0.5
    func2 = x -> x[2] < 2.5
    func3 = x -> x[1] == 0.0

    # Invariant: Marker instances associate a symbol label with a boundary symbol,
    # a set of symbols, or a raw closure condition. MarkerPair instances provide
    # convenient pair notation (`:label => identifier`).
    @testset "Marker and MarkerPair data structures" begin
        m_sym = Marker(:boundary, :left)
        m_tup = Marker(:corners, Set((:top, :right)))
        # A function-valued marker holds the raw closure directly without wrapping.
        m_fun = Marker(:region, func1)

        @test label(m_sym) === :boundary
        @test identifier(m_sym) === :left

        @test label(m_tup) === :corners
        @test identifier(m_tup) == Set((:top, :right))

        @test label(m_fun) === :region
        @test identifier(m_fun) === func1

        # MarkerPair
        pair_sym = :inlet => :left
        @test label(pair_sym) === :inlet
        @test identifier(pair_sym) === :left
    end

    # Invariant: Boundary symbols for 1D, 2D, and 3D geometries are extractable
    # from either value instances or type signatures of sets and domains.
    @testset "Boundary symbol extraction" begin
        @test get_boundary_symbols(I1D) == (:left, :right)
        @test get_boundary_symbols(I2D) == (:bottom, :top, :left, :right)
        @test get_boundary_symbols(I3D) == (:bottom, :top, :back, :front, :left, :right)

        # Type-level boundary symbols
        @test get_boundary_symbols(typeof(I1D)) == (:left, :right)
        @test get_boundary_symbols(typeof(I2D)) == (:bottom, :top, :left, :right)
        @test get_boundary_symbols(typeof(I3D)) ==
              (:bottom, :top, :back, :front, :left, :right)

        # On Domain
        Ω1 = domain(I1D)
        Ω2 = domain(I2D)
        @test get_boundary_symbols(Ω1) == (:left, :right)
        @test get_boundary_symbols(Ω2) == (:bottom, :top, :left, :right)
        @test get_boundary_symbols(typeof(Ω1)) == (:left, :right)
    end

    # Invariant: `process_identifier` normalizes symbols, tuples, and vectors
    # of symbols into canonical boundary representations. Function-valued pairs
    # are stored directly as closures and are not routed through `process_identifier`.
    @testset "Identifier processing" begin
        @test process_identifier(I1D, :left) === :left
        @test process_identifier(I2D, (:top, :right)) == Set((:top, :right))
        # A vector of symbols normalizes to the same Set as the tuple form.
        @test process_identifier(I2D, [:top, :right]) == Set((:top, :right))
        @test process_identifier(I2D, [:top, :right]) ==
              process_identifier(I2D, (:top, :right))
    end

    # Invariant: `DomainMarkers` correctly partitions and deduplicates boundary
    # symbols, boundary tuples, and functional conditions.
    @testset "Domain marker container creation" begin
        # Empty container instantiation.
        dm_empty = markers(I1D)
        @test dm_empty isa DomainMarkers
        @test isempty(dm_empty.symbols)
        @test isempty(dm_empty.tuples)
        @test isempty(dm_empty.conditions)
        @test isempty(dm_empty)
        @test length(dm_empty) == 0

        # Mixed marker types.
        pairs = (:bnd_left => :left,
            :bnd_right => :right,
            :corners => (:top, :right),
            :all_bnd => (:top, :bottom, :left, :right),
            :region1 => func1,
            :region2 => func2)
        dm_mixed = markers(I2D, pairs...)
        @test dm_mixed isa DomainMarkers
        @test length(dm_mixed) == 6
        @test !isempty(dm_mixed)

        # Duplicate labels with different identifiers are preserved.
        dm_dup_label = markers(I1D, :boundary => :left, :boundary => :right)
        @test length(dm_dup_label.symbols) == 2
        @test Set(label(m) for m in dm_dup_label.symbols) == Set([:boundary])

        # Duplicate markers with identical label and identifier are deduplicated.
        dm_dup_marker = markers(I1D, :boundary => :left, :boundary => :left)
        @test length(dm_dup_marker.symbols) == 1
    end

    # Invariant: Domain constructors preserve geometric traits including
    # spatial dimension, element type, topological dimension, and point type.
    @testset "Domain construction and geometric traits" begin
        # Default constructor from geometric set.
        Ω1_def = domain(I1D)
        @test set(Ω1_def) === I1D
        @test dim(Ω1_def) == 1
        @test dim(typeof(Ω1_def)) == 1
        @test eltype(Ω1_def) === Float64
        @test eltype(typeof(Ω1_def)) === Float64
        @test topo_dim(Ω1_def) == 1
        @test point_type(Ω1_def) === Float64
        @test point_type(typeof(Ω1_def)) === Float64

        Ω2_def = domain(I2D)
        @test set(Ω2_def) === I2D
        @test dim(Ω2_def) == 2
        @test eltype(Ω2_def) === Float32
        @test point_type(Ω2_def) === NTuple{2, Float32}
        @test point_type(typeof(Ω2_def)) === NTuple{2, Float32}

        # Domain constructed with preallocated DomainMarkers.
        markers_premade = markers(I2D, :neumann => :top, :fixed => func1)
        Ω_premade = domain(I2D, markers_premade)
        @test set(Ω_premade) === I2D
        @test markers(Ω_premade) === markers_premade
        @test length(Ω_premade) == 2
        @test !isempty(Ω_premade)

        # Domain constructed with marker pairs directly.
        Ω_pairs = domain(I2D, :neumann => :top, :fixed => func1, :mixed => (:left, :bottom))
        @test set(Ω_pairs) === I2D
        @test length(Ω_pairs) == 3

        # Domain constructed with empty marker set.
        Ω_empty = domain(I1D, markers(I1D))
        @test isempty(Ω_empty)
        @test length(Ω_empty) == 0
    end

    # Invariant: Domain accessor functions expose categorized markers, labels,
    # symbols, tuples, functional conditions, and dimension projections.
    @testset "Domain accessors and marker iterators" begin
        Ω = domain(I2D,
            :bnd_left => :left,
            :corners => (:top, :right),
            :region1 => func1,
            :boundary => :top)

        dm_retrieved = markers(Ω)
        @test dm_retrieved isa DomainMarkers
        @test length(symbols(Ω)) == 2
        @test length(tuples(Ω)) == 1
        @test length(conditions(Ω)) == 1

        lbls = Set(labels(Ω))
        @test lbls == Set([:bnd_left, :corners, :region1, :boundary])

        @test Set(marker_symbols(Ω)) == Set([:left, :top])
        @test Set(marker_tuples(Ω)) == Set([Set((:top, :right))])
        @test length(collect(marker_conditions(Ω))) == 1
        @test length(collect(marker_identifiers(Ω))) == 4

        @test Set(label_symbols(Ω)) == Set([:bnd_left, :boundary])
        @test Set(label_tuples(Ω)) == Set([:corners])
        @test Set(label_conditions(Ω)) == Set([:region1])
        @test Set(label_identifiers(Ω)) == Set([:bnd_left, :corners, :region1, :boundary])

        # Dimension projection onto 1D coordinate intervals.
        proj1 = projection(Ω, 1)
        proj2 = projection(Ω, 2)
        @test proj1 isa CartesianProduct{1, Float32}
        @test proj1.box[1] == (0.0f0, 1.0f0)
        @test proj2 isa CartesianProduct{1, Float32}
        @test proj2.box[1] == (2.0f0, 3.0f0)
    end

    # Invariant: Time-dependent marker conditions can be evaluated at a specific
    # temporal parameter t, yielding an `EvaluatedDomainMarkers` container.
    @testset "Time-dependent domain evaluation" begin
        I_space = interval(0.0, 1.0) × interval(0.0, 1.0)
        I_time = interval(0.0, 2.0)
        func_time = (x, t) -> x[1] > t

        # Create time-dependent DomainMarkers.
        dm_time = markers(I_space, I_time,
            :moving => func_time,
            :fixed_bnd => :left)

        @test length(dm_time) == 2
        @test length(conditions(dm_time)) == 1

        # Temporal evaluation of DomainMarkers.
        dm_eval = dm_time(0.5)
        @test dm_eval isa EvaluatedDomainMarkers
        @test length(dm_eval) == 2
        @test !isempty(dm_eval)
        @test length(collect(symbols(dm_eval))) == 1
        @test length(collect(tuples(dm_eval))) == 0
        @test length(collect(conditions(dm_eval))) == 1
        @test Set(collect(label_identifiers(dm_eval))) == Set([:moving, :fixed_bnd])
        @test Set(collect(label_symbols(dm_eval))) == Set([:fixed_bnd])
        @test isempty(collect(label_tuples(dm_eval)))
        @test Set(collect(label_conditions(dm_eval))) == Set([:moving])

        # Evaluate the instantiated condition closure.
        moving_marker = first(conditions(dm_eval))
        @test label(moving_marker) == :moving
        bf = identifier(moving_marker)
        @test bf((0.8, 0.5)) == true
        @test bf((0.2, 0.5)) == false

        # Direct evaluation on Domain: Ω(t).
        Ω_time = domain(I_space, I_time, :moving => func_time, :fixed_bnd => :left)
        @test dim(Ω_time) == 2
        Ω_eval = Ω_time(0.5)
        @test Ω_eval isa Domain
        @test markers(Ω_eval) isa EvaluatedDomainMarkers
        @test length(Ω_eval) == 2
    end

    # Invariant: Querying geometric traits, projections, and boundary symbols
    # on stack-allocated domains infers cleanly and allocates zero heap memory.
    @testset "Domain type stability and zero allocations" begin
        Ω = domain(I2D, :left => :left, :right => :right)
        @inferred set(Ω)
        @inferred dim(Ω)
        @inferred eltype(Ω)
        @inferred topo_dim(Ω)
        @inferred point_type(Ω)
        @inferred projection(Ω, 1)
        @inferred get_boundary_symbols(Ω)
        @inferred is_collapsed(Ω)
        @inferred is_collapsed(Ω, 1)
        @inferred Base.length(Ω)
        @inferred Base.isempty(Ω)

        @test_allocs set(Ω)
        @test_allocs dim(Ω)
        @test_allocs eltype(Ω)
        @test_allocs topo_dim(Ω)
        @test_allocs is_collapsed(Ω)
        @test_allocs is_collapsed(Ω, 1)
        @test_allocs projection(Ω, 1)
        @test_allocs get_boundary_symbols(Ω)
        @test_allocs Base.length(Ω)
        @test_allocs Base.isempty(Ω)

        # Zero allocations iterating directly over homogeneous marker categories
        Ω_markers = domain(I2D, :left_bnd => :left, :corner => (:top, :right), :sub =>
            (x -> x[1] > 0.5f0))
        iterate_symbols(dom) = (c = 0; for s in label_symbols(dom)
                c += 1
            end; c)
        iterate_tuples(dom) = (c = 0; for t in label_tuples(dom)
                c += 1
            end; c)
        iterate_conditions(dom) = (c = 0; for cond in label_conditions(dom)
                c += 1
            end; c)
        iterate_marker_syms(dom) = (c = 0; for m in marker_symbols(dom)
                c += 1
            end; c)
        iterate_marker_tups(dom) = (c = 0; for m in marker_tuples(dom)
                c += 1
            end; c)
        iterate_marker_conds(dom) = (c = 0; for m in marker_conditions(dom)
                c += 1
            end; c)

        @test_allocs iterate_symbols(Ω_markers)
        @test_allocs iterate_tuples(Ω_markers)
        @test_allocs iterate_conditions(Ω_markers)
        @test_allocs iterate_marker_syms(Ω_markers)
        @test_allocs iterate_marker_tups(Ω_markers)
        @test_allocs iterate_marker_conds(Ω_markers)
    end

    # Invariant: Textual display formatting for markers, marker containers, and
    # domains across dimensions (1D, 2D, 3D, collapsed) produces valid output.
    @testset "Domain string representation" begin
        # Marker formatting.
        m_s = Marker(:left, :left)
        m_t = Marker(:corner, Set([:top, :right]))
        m_f = Marker(:level, x -> x[1] > 0)

        @test occursin("Marker(:left => :left)", repr(m_s))
        @test occursin("Marker(:corner => (", repr(m_t))
        @test occursin("Marker(:level => <function>)", repr(m_f))

        # DomainMarkers detailed display.
        dm = markers(I1D, :left => :left, :right => (:top, :bottom), :fn => func1)
        io = IOBuffer()
        show(io, dm)
        str_dm = String(take!(io))
        @test occursin("DomainMarkers:", str_dm)
        @test occursin("Symbol markers", str_dm)
        @test occursin("Tuple markers", str_dm)
        @test occursin("Function markers", str_dm)

        # DomainMarkers compact and empty display.
        show(IOContext(io, :compact => true), dm)
        @test occursin("DomainMarkers(3 total)", String(take!(io)))

        show(io, markers(I1D))
        @test occursin("(empty)", String(take!(io)))

        # Domain detailed and compact display.
        Ω_1d = domain(I1D)
        show(io, Ω_1d)
        str_d1 = String(take!(io))
        @test occursin("Domain", str_d1)
        @test occursin("Set:", str_d1)
        @test occursin("Markers:", str_d1)

        show(IOContext(io, :compact => true), Ω_1d)
        @test occursin("Domain{1D, Float64}:", String(take!(io)))

        # 2D domain with empty markers.
        Ω_empty = domain(I2D, markers(I2D))
        show(io, Ω_empty)
        str_d_empty = String(take!(io))
        @test occursin("(none)", str_d_empty)

        # 2D domain with markers (covers tuple markers branch).
        Ω_2d = domain(I2D, :wall => (:top, :bottom))
        show(io, Ω_2d)
        str_2d = String(take!(io))
        @test occursin("Domain", str_2d)
        @test occursin("Markers:", str_2d)

        # 3D domain (covers D > 1 coordinate formatting).
        Ω_3d = domain(I3D, :dirichlet => :left)
        show(io, Ω_3d)
        str_3d = String(take!(io))
        @test occursin("z:", str_3d)

        # 3D domain with collapsed dimension (topological dimension < D).
        I3D_c = interval(0.0, 1.0) × interval(0.0, 1.0) × point(0.5)
        Ω_3d_c = domain(I3D_c)
        show(io, Ω_3d_c)
        str_3d_c = String(take!(io))
        @test occursin("Topological dimension", str_3d_c)
    end

    # Invariant: `Domain` forwards geometric property queries (`center`, `in`,
    # `tails`, `is_collapsed`, `projection`, boundary symbols) directly to
    # the underlying geometric set.
    @testset "Geometric property delegation" begin
        Ω_1d = domain(interval(0.0, 4.0))
        Ω_2d = domain(interval(0.0, 2.0) × interval(1.0, 3.0))

        # Geometric centroid.
        @test center(Ω_1d)[1] ≈ 2.0
        @test center(Ω_2d)[1] ≈ 1.0 && center(Ω_2d)[2] ≈ 2.0

        # Point containment.
        @test 2.0 ∈ Ω_1d
        @test 5.0 ∉ Ω_1d
        @test (1.0, 2.0) ∈ Ω_2d
        @test (5.0, 2.0) ∉ Ω_2d

        # Boundary coordinate limits.
        @test tails(Ω_1d) == (0.0, 4.0)
        @test tails(Ω_1d, 1) == (0.0, 4.0)
        @test tails(Ω_2d) == ((0.0, 2.0), (1.0, 3.0))
        @test tails(Ω_2d, 1) == (0.0, 2.0)
        @test tails(Ω_2d, 2) == (1.0, 3.0)

        # Degenerate dimension detection across 1D and nD domains.
        @test !is_collapsed(Ω_1d)
        @test !is_collapsed(Ω_2d)
        @test !is_collapsed(Ω_2d, 1)
        @test !is_collapsed(Ω_2d, 2)
        Ω_pt = domain(point(0.5))
        @test is_collapsed(Ω_pt)
        Ω_2d_c = domain(interval(0.0, 1.0) × point(2.0))
        @test is_collapsed(Ω_2d_c)
        @test !is_collapsed(Ω_2d_c, 1)
        @test is_collapsed(Ω_2d_c, 2)
        @test_throws BoundsError is_collapsed(Ω_2d_c, 0)
        @test_throws BoundsError is_collapsed(Ω_2d_c, 3)

        # Coordinate projection.
        @test projection(Ω_2d, 1) == interval(0.0, 2.0)
        @test projection(Ω_2d, 2) == interval(1.0, 3.0)

        # Coordinate interval indexing.
        @test Ω_2d(1) == (0.0, 2.0)
        @test Ω_2d(2) == (1.0, 3.0)

        # Boundary symbols from domain instance.
        @test get_boundary_symbols(Ω_1d) == (:left, :right)
        @test get_boundary_symbols(Ω_2d) == (:bottom, :top, :left, :right)

        # Boundary symbols from domain type.
        @test get_boundary_symbols(typeof(Ω_1d)) == (:left, :right)
        @test get_boundary_symbols(typeof(Ω_2d)) == (:bottom, :top, :left, :right)
    end

    # Invariant: `EvaluatedDomainMarkers` handles static condition fallbacks
    # when evaluating time-dependent marker collections and supports standard iteration.
    @testset "Evaluated domain marker iteration and traits" begin
        using Bramble: EvaluatedDomainMarkers, label_identifiers, label_symbols,
                       label_tuples, label_conditions

        I_time = interval(0.0, 1.0)
        I_space = interval(0.0, 1.0)

        # Static boolean function taking spatial coordinate only (not applicable to scalar time t).
        staticfunc = x -> x > 0.5
        dm = markers(I_space, :region => staticfunc)
        edm = dm(0.5)
        @test edm isa EvaluatedDomainMarkers

        # Labels query on evaluated marker container.
        lbls = collect(labels(edm))
        @test :region ∈ lbls

        # Categorized label accessors on EvaluatedDomainMarkers.
        dm2 = markers(I_space, :left => :left, :wall => (:top, :bottom), :region =>
            staticfunc)
        edm2 = dm2(0.0)
        @test :left ∈ collect(label_symbols(edm2))
        @test :wall ∈ collect(label_tuples(edm2))
        @test :region ∈ collect(label_conditions(edm2))
        @test length(edm2) == 3
        @test !isempty(edm2)
    end

    # Invariant: `get_boundary_symbols` returns canonical boundary names for
    # dimensions 1, 2, and 3, and raises an error for unsupported dimensions.
    @testset "Default boundary symbol mappings" begin
        @test get_boundary_symbols(1) == (:left, :right)
        @test get_boundary_symbols(2) == (:bottom, :top, :left, :right)
        @test get_boundary_symbols(3) == (:bottom, :top, :back, :front, :left, :right)
        @test_throws ErrorException get_boundary_symbols(4)
    end

    # Invariant: Spatiotemporal domains can be constructed by combining spatial
    # and temporal sets alongside time-dependent boundary conditions.
    @testset "Spatiotemporal domain construction" begin
        I_space = interval(0.0, 1.0) × interval(0.0, 1.0)
        I_time = interval(0.0, 2.0)
        f = (x, t) -> x[1] > t
        Ω = domain(I_space, I_time, :moving => f, :fixed => :left)
        @test set(Ω) === I_space
        @test dim(Ω) == 2
        @test length(Ω) == 2
    end
end

@testset "Higher-dimensional domains and collapsed sets" begin
    # Invariant: Boundary symbols are defined up to 3D; higher dimensions throw errors.
    @testset "Higher-dimensional domains" begin
        # 4D boxes have no canonical boundary face names; queries throw an ErrorException.
        X4 = interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0) ×
             interval(0.0, 1.0)
        @test dim(X4) == 4
        @test_throws ErrorException get_boundary_symbols(4)
        @test_throws ErrorException get_boundary_symbols(typeof(X4))

        # Supported dimensions resolve identically across value and type queries.
        @test get_boundary_symbols(1) == (:left, :right)
        @test get_boundary_symbols(2) == (:bottom, :top, :left, :right)
        @test get_boundary_symbols(3) == (:bottom, :top, :back, :front, :left, :right)

        I = interval(0.0, 1.0)
        @test get_boundary_symbols(I) == get_boundary_symbols(typeof(I))
        @test get_boundary_symbols(typeof(domain(I))) == get_boundary_symbols(typeof(I))

        X2 = I × I
        @test get_boundary_symbols(X2) == get_boundary_symbols(typeof(X2))
        X3 = I × I × I
        @test get_boundary_symbols(X3) == get_boundary_symbols(typeof(X3))
    end

    # Invariant: A domain wrapping a degenerate interval displays as a Point rather than Interval.
    @testset "Collapsed set display formatting" begin
        out = sprint(show, domain(interval(3.0, 3.0)))
        @test occursin("Point", out)
        @test occursin("3.0", out)
        @test !occursin("Interval", out)

        out2 = sprint(show, domain(interval(0.0, 1.0)))
        @test occursin("Interval", out2)
        @test !occursin("Point", out2)
    end
end
