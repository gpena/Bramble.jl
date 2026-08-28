using Test
using Bramble
using Bramble: Marker, MarkerPair, BrambleFunction, Domain, DomainMarkers, EvaluatedDomainMarkers, dim, set, markers, labels
using Bramble: get_boundary_symbols, label, identifier, domain, symbols, tuples, conditions
using Bramble: marker_identifiers, _embed_notime, process_identifier, marker_symbols, marker_tuples, marker_conditions
using Bramble: label_identifiers, label_symbols, label_tuples, label_conditions, point_type, topo_dim
using StaticArrays

@testset "Domain System Tests" begin
	# --- Setup Test Data ---
	I1D = interval(0.0, 1.0)
	I2D = interval(0.0f0, 1.0f0) × interval(2.0f0, 3.0f0) # Float32
	I3D = interval(0.0, 1.0) × interval(2.0, 3.0) × interval(4.0, 5.0)

	func1 = x -> x[1] > 0.5
	func2 = x -> x[2] < 2.5
	func3 = x -> x[1] == 0.0

	@testset "Marker and MarkerPair" begin
		m_sym = Marker(:boundary, :left)
		m_tup = Marker(:corners, Set((:top, :right)))
		bf_func1 = _embed_notime(I1D, func1; CoType = eltype(I1D))
		m_fun = Marker(:region, bf_func1)

		@test label(m_sym) === :boundary
		@test identifier(m_sym) === :left

		@test label(m_tup) === :corners
		@test identifier(m_tup) == Set((:top, :right))

		@test label(m_fun) === :region
		@test identifier(m_fun) isa BrambleFunction

		# MarkerPair
		pair_sym = :inlet => :left
		@test label(pair_sym) === :inlet
		@test identifier(pair_sym) === :left
	end

	@testset "Boundary Symbols" begin
		@test get_boundary_symbols(I1D) == (:left, :right)
		@test get_boundary_symbols(I2D) == (:bottom, :top, :left, :right)
		@test get_boundary_symbols(I3D) == (:bottom, :top, :back, :front, :left, :right)

		# Type-level boundary symbols
		@test get_boundary_symbols(typeof(I1D)) == (:left, :right)
		@test get_boundary_symbols(typeof(I2D)) == (:bottom, :top, :left, :right)
		@test get_boundary_symbols(typeof(I3D)) == (:bottom, :top, :back, :front, :left, :right)

		# On Domain
		Ω1 = domain(I1D)
		Ω2 = domain(I2D)
		@test get_boundary_symbols(Ω1) == (:left, :right)
		@test get_boundary_symbols(Ω2) == (:bottom, :top, :left, :right)
		@test get_boundary_symbols(typeof(Ω1)) == (:left, :right)
	end

	@testset "Process Identifier Function" begin
		@test process_identifier(I1D, :left) === :left
		@test process_identifier(I2D, (:top, :right)) == Set((:top, :right))
		@test process_identifier(I1D, func1) isa BrambleFunction
	end

	@testset "Create Markers" begin
		# Empty call
		dm_empty = markers(I1D)
		@test dm_empty isa DomainMarkers
		@test isempty(dm_empty.symbols)
		@test isempty(dm_empty.tuples)
		@test isempty(dm_empty.conditions)
		@test isempty(dm_empty)
		@test length(dm_empty) == 0

		# Mixed types
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

		# Duplicate labels (different identifiers kept)
		dm_dup_label = markers(I1D, :boundary => :left, :boundary => :right)
		@test length(dm_dup_label.symbols) == 2
		@test Set(label(m) for m in dm_dup_label.symbols) == Set([:boundary])

		# Duplicate markers (same label and identifier -> set deduplication)
		dm_dup_marker = markers(I1D, :boundary => :left, :boundary => :left)
		@test length(dm_dup_marker.symbols) == 1
	end

	@testset "Domain Construction & Traits" begin
		# Default constructor
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
		@test point_type(Ω2_def) === NTuple{2,Float32}
		@test point_type(typeof(Ω2_def)) === NTuple{2,Float32}

		# Domain with premade DomainMarkers
		markers_premade = markers(I2D, :neumann => :top, :fixed => func1)
		Ω_premade = domain(I2D, markers_premade)
		@test set(Ω_premade) === I2D
		@test markers(Ω_premade) === markers_premade
		@test length(Ω_premade) == 2
		@test !isempty(Ω_premade)

		# Domain with pairs
		Ω_pairs = domain(I2D, :neumann => :top, :fixed => func1, :mixed => (:left, :bottom))
		@test set(Ω_pairs) === I2D
		@test length(Ω_pairs) == 3

		# Domain with empty markers
		Ω_empty = domain(I1D, markers(I1D))
		@test isempty(Ω_empty)
		@test length(Ω_empty) == 0
	end

	@testset "Domain Accessors and Iterators" begin
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

		# Projection
		proj1 = projection(Ω, 1)
		proj2 = projection(Ω, 2)
		@test proj1 isa CartesianProduct{1,Float32}
		@test proj1.box[1] == (0.0f0, 1.0f0)
		@test proj2 isa CartesianProduct{1,Float32}
		@test proj2.box[1] == (2.0f0, 3.0f0)
	end

	@testset "Time-Dependent Domain and EvaluatedDomainMarkers" begin
		I_space = interval(0.0, 1.0) × interval(0.0, 1.0)
		I_time = interval(0.0, 2.0)
		func_time = (x, t) -> x[1] > t

		# Create time-dependent DomainMarkers
		dm_time = markers(I_space, I_time,
						  :moving => func_time,
						  :fixed_bnd => :left)

		@test length(dm_time) == 2
		@test length(conditions(dm_time)) == 1

		# Time evaluation of DomainMarkers
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

		# Test evaluated condition function
		moving_marker = first(conditions(dm_eval))
		@test label(moving_marker) == :moving
		bf = identifier(moving_marker)
		@test bf((0.8, 0.5)) == true
		@test bf((0.2, 0.5)) == false

		# Direct evaluation on Domain: Ω(t)
		Ω_time = domain(I_space, I_time, :moving => func_time, :fixed_bnd => :left)
		@test dim(Ω_time) == 2
		Ω_eval = Ω_time(0.5)
		@test Ω_eval isa Domain
		@test markers(Ω_eval) isa EvaluatedDomainMarkers
		@test length(Ω_eval) == 2
	end

	@testset "Type Inference & Performance" begin
		Ω = domain(I2D, :left => :left, :right => :right)
		@inferred set(Ω)
		@inferred dim(Ω)
		@inferred eltype(Ω)
		@inferred topo_dim(Ω)
		@inferred point_type(Ω)
		@inferred projection(Ω, 1)
		@inferred get_boundary_symbols(Ω)
		@inferred Base.length(Ω)
		@inferred Base.isempty(Ω)

		@test_allocs set(Ω)
		@test_allocs dim(Ω)
		@test_allocs eltype(Ω)
		@test_allocs topo_dim(Ω)
		@test_allocs projection(Ω, 1)
		@test_allocs get_boundary_symbols(Ω)
		@test_allocs Base.length(Ω)
		@test_allocs Base.isempty(Ω)
	end

	@testset "Display / Show" begin
		# Marker show
		m_s = Marker(:left, :left)
		m_t = Marker(:corner, Set([:top, :right]))
		m_f = Marker(:level, _embed_notime(I1D, x -> x[1] > 0))

		@test occursin("Marker(:left => :left)", repr(m_s))
		@test occursin("Marker(:corner => (", repr(m_t))
		@test occursin("Marker(:level => <function>)", repr(m_f))

		# DomainMarkers show
		dm = markers(I1D, :left => :left, :right => (:top, :bottom), :fn => func1)
		io = IOBuffer()
		show(io, dm)
		str_dm = String(take!(io))
		@test occursin("DomainMarkers:", str_dm)
		@test occursin("Symbol markers", str_dm)
		@test occursin("Tuple markers", str_dm)
		@test occursin("Function markers", str_dm)

		# DomainMarkers compact and empty
		show(IOContext(io, :compact => true), dm)
		@test occursin("DomainMarkers(3 total)", String(take!(io)))

		show(io, markers(I1D))
		@test occursin("(empty)", String(take!(io)))

		# Domain show (detailed and compact)
		Ω_1d = domain(I1D)
		show(io, Ω_1d)
		str_d1 = String(take!(io))
		@test occursin("Domain", str_d1)
		@test occursin("Set:", str_d1)
		@test occursin("Markers:", str_d1)

		show(IOContext(io, :compact => true), Ω_1d)
		@test occursin("Domain{1D, Float64}:", String(take!(io)))

		# Domain 2D with empty markers
		Ω_empty = domain(I2D, markers(I2D))
		show(io, Ω_empty)
		str_d_empty = String(take!(io))
		@test occursin("(none)", str_d_empty)

		# 2D domain with markers — show (hits n_tup path)
		Ω_2d = domain(I2D, :wall => (:top, :bottom))
		show(io, Ω_2d)
		str_2d = String(take!(io))
		@test occursin("Domain", str_2d)
		@test occursin("Markers:", str_2d)

		# 3D domain — triggers the D>1 branch in Base.show
		Ω_3d = domain(I3D, :dirichlet => :left)
		show(io, Ω_3d)
		str_3d = String(take!(io))
		@test occursin("z:", str_3d)

		# 3D domain with collapsed dimension — triggers topodim < D branch
		I3D_c = interval(0.0, 1.0) × interval(0.0, 1.0) × point(0.5)
		Ω_3d_c = domain(I3D_c)
		show(io, Ω_3d_c)
		str_3d_c = String(take!(io))
		@test occursin("Topological dimension", str_3d_c)
	end

	@testset "Domain delegation (center, tails, is_collapsed, projection, in)" begin
		Ω_1d = domain(interval(0.0, 4.0))
		Ω_2d = domain(interval(0.0, 2.0) × interval(1.0, 3.0))

		# center (line 204)
		@test center(Ω_1d)[1] ≈ 2.0
		@test center(Ω_2d)[1] ≈ 1.0 && center(Ω_2d)[2] ≈ 2.0

		# in (line 211)
		@test 2.0 ∈ Ω_1d
		@test 5.0 ∉ Ω_1d
		@test (1.0, 2.0) ∈ Ω_2d
		@test (5.0, 2.0) ∉ Ω_2d

		# tails (lines 218-219)
		@test tails(Ω_1d) == (0.0, 4.0)
		@test tails(Ω_1d, 1) == (0.0, 4.0)
		@test tails(Ω_2d) == ((0.0, 2.0), (1.0, 3.0))
		@test tails(Ω_2d, 1) == (0.0, 2.0)
		@test tails(Ω_2d, 2) == (1.0, 3.0)

		# is_collapsed on 1D Domain (line 226)
		@test !is_collapsed(Ω_1d)
		Ω_pt = domain(point(0.5))
		@test is_collapsed(Ω_pt)

		# projection (line 235)
		@test projection(Ω_2d, 1) == interval(0.0, 2.0)
		@test projection(Ω_2d, 2) == interval(1.0, 3.0)

		# Domain(i) call — delegates to CartesianProduct(i) returning (min, max) tuple (line 228)
		@test Ω_2d(1) == (0.0, 2.0)
		@test Ω_2d(2) == (1.0, 3.0)

		# get_boundary_symbols on Domain (line 246)
		@test get_boundary_symbols(Ω_1d) == (:left, :right)
		@test get_boundary_symbols(Ω_2d) == (:bottom, :top, :left, :right)

		# get_boundary_symbols on Domain type (line 254)
		@test get_boundary_symbols(typeof(Ω_1d)) == (:left, :right)
		@test get_boundary_symbols(typeof(Ω_2d)) == (:bottom, :top, :left, :right)
	end

	@testset "EvaluatedDomainMarkers labels and _evaluate_marker_at_time fallback" begin
		using Bramble: EvaluatedDomainMarkers, label_identifiers, label_symbols, label_tuples, label_conditions

		I_time = interval(0.0, 1.0)
		I_space = interval(0.0, 1.0)

		# markers with a time-independent condition (non-applicable at time t → fallback line 258)
		# Use a static boolean function — applicable(f, t) will be false since it takes (x::Float64)
		staticfunc = x -> x > 0.5
		dm = markers(I_space, :region => staticfunc)
		edm = dm(0.5)  # evaluates time-dependent markers; condition not applicable to scalar t
		@test edm isa EvaluatedDomainMarkers

		# labels on EvaluatedDomainMarkers (line 263)
		lbls = collect(labels(edm))
		@test :region ∈ lbls

		# label_symbols, label_tuples, label_conditions on EvaluatedDomainMarkers
		dm2 = markers(I_space, :left => :left, :wall => (:top, :bottom), :region => staticfunc)
		edm2 = dm2(0.0)
		@test :left ∈ collect(label_symbols(edm2))
		@test :wall ∈ collect(label_tuples(edm2))
		@test :region ∈ collect(label_conditions(edm2))
		@test length(edm2) == 3
		@test !isempty(edm2)
	end
end