using Bramble: Marker, BrambleFunction, DomainMarkers
using Bramble: get_boundary_symbols, create_markers, label, identifier, domain, set, markers, labels
using Bramble: marker_identifiers, _embed_notime, process_identifier, marker_symbols, marker_tuples, marker_conditions

# --- Test Suite ---

@testset "Domain System Tests" begin

	# --- Setup Test Data ---
	I1D = interval(0.0, 1.0)
	I2D = interval(0.0f0, 1.0f0) × interval(2.0f0, 3.0f0) # Use Float32 for eltype test
	I3D = interval(0.0, 1.0) × interval(2.0, 3.0) × interval(4.0, 5.0)

	# Use distinct functions for testing equality/hashing
	func1 = x -> x[1] > 0.5
	func2 = x -> x[2] < 2.5
	func3 = x -> x[1] == 0.0 # Another function

	# Expected BrambleFunction type parameterization
	BFTypeFloat64 = BrambleFunction{Float64,false,Bool} where F<:Function
	BFTypeFloat32 = BrambleFunction{Float32,false,Bool} where F<:Function

	@testset "Marker Struct" begin
		m_sym = Marker(:boundary, :left)
		m_tup = Marker(:corners, Set((:top, :right))) # Identifier stored as Set
		bf_func1 = _embed_notime(I1D, func1; CoType = eltype(I1D))
		m_fun = Marker(:region, bf_func1)

		@test label(m_sym) === :boundary
		@test identifier(m_sym) === :left

		@test label(m_tup) === :corners
		@test identifier(m_tup) == Set((:top, :right)) # Compare content

		@test label(m_fun) === :region
		@test identifier(m_fun) isa BrambleFunction
	end

	@testset "Boundary Symbols" begin
		@test get_boundary_symbols(I1D) == (:left, :right)
		@test get_boundary_symbols(I2D) == (:bottom, :top, :left, :right)
		@test get_boundary_symbols(I3D) == (:bottom, :top, :back, :front, :left, :right)
		# Test Val dispatch directly
		@test get_boundary_symbols(Val(1)) == (:left, :right)
		@test get_boundary_symbols(Val(2)) == (:bottom, :top, :left, :right)
		@test get_boundary_symbols(Val(3)) == (:bottom, :top, :back, :front, :left, :right)
	end

	@testset "Process Identifier Function" begin
		bf_func1_f64 = _embed_notime(I1D, func1; CoType = Float64)
		bf_func1_f32 = _embed_notime(I2D, func1; CoType = Float32)

		@test process_identifier(I1D, :left) === :left
		@test process_identifier(I2D, (:top, :right)) == Set((:top, :right)) # Test NTuple{Symbol}
		@test process_identifier(I1D, func1) isa BrambleFunction
	end

	@testset "Create Markers Function" begin
		# Test empty call
		dm_empty = create_markers(I1D)
		@test dm_empty isa DomainMarkers
		@test isempty(dm_empty.symbols)
		@test isempty(dm_empty.tuples)
		@test isempty(dm_empty.functions)
		@test eltype(dm_empty.functions) <: Marker{<:BrambleFunction} # Check function marker type

		# Test mixed types (using Float32 domain I2D)
		pairs = (:bnd_left => :left,
				 :bnd_right => :right,                 # Symbol
				 :corners => (:top, :right),           # NTuple{Symbol}
				 :all_bnd => (:top, :bottom, :left, :right), # NTuple{Symbol}
				 :region1 => func1,                    # Function
				 :region2 => func2)
		dm_mixed = create_markers(I2D, pairs...)

		# Test type stability (check the type parameter of DomainMarkers)
		@test dm_mixed isa DomainMarkers

		# Test duplicate labels (different identifiers should be kept)
		dm_dup_label = create_markers(I1D, :boundary => :left, :boundary => :right)
		@test length(dm_dup_label.symbols) == 2
		@test Set(label(m) for m in dm_dup_label.symbols) == Set([:boundary])
		@test Set(identifier(m) for m in dm_dup_label.symbols) == Set([:left, :right])

		# Test duplicate markers (same label, same identifier -> only one kept)
		dm_dup_marker = create_markers(I1D, :boundary => :left, :boundary => :left)
		@test length(dm_dup_marker.symbols) == 1
		@test first(dm_dup_marker.symbols) == Marker(:boundary, :left)

		# Test only function markers
		dm_only_func = create_markers(I1D, :region1 => func1)
		@test isempty(dm_only_func.symbols)
		@test isempty(dm_only_func.tuples)
		@test length(dm_only_func.functions) == 1
		@test first(dm_only_func.functions).label == :region1
	end

	@testset "Domain Construction" begin
		# Test default constructor (1D - Float64)
		Ω1_def = domain(I1D)
		@test set(Ω1_def) === I1D
		expected_markers_1d = create_markers(I1D, :dirichlet => (:left, :right)) # Uses NTuple internally
		@test dim(Ω1_def) == 1
		@test eltype(Ω1_def) == Float64

		# Test default constructor (2D - Float32)
		Ω2_def = domain(I2D)
		@test set(Ω2_def) === I2D
		expected_markers_2d = create_markers(I2D, :dirichlet => (:bottom, :top, :left, :right))
		@test dim(Ω2_def) == 2
		@test eltype(Ω2_def) == Float32

		# Test constructor with DomainMarkers object
		markers_premade = create_markers(I2D, :neumann => :top, :fixed => func1)
		Ω_premade = domain(I2D, markers_premade)
		@test set(Ω_premade) === I2D

		# Test constructor with pairs...
		Ω_pairs = domain(I2D, :neumann => :top, :fixed => func1, :mixed => (:left, :bottom))
		@test set(Ω_pairs) === I2D
		expected_markers_pairs = create_markers(I2D, :neumann => :top, :fixed => func1, :mixed => (:left, :bottom))
	end

	@testset "Domain Accessors and Helpers" begin
		# Use Float32 domain I2D for this test section
		Ω = domain(I2D,
				   :bnd_left => :left,           # Symbol
				   :corners => (:top, :right),   # Tuple
				   :region1 => func1,            # Function
				   :boundary => :top)

		@test set(Ω) === I2D
		@test dim(Ω) == 2
		@test eltype(Ω) == Float32
		@test dim(typeof(Ω)) == 2      # Test type method
		@test eltype(typeof(Ω)) == Float32 # Test type method

		# Test markers() returns the correct DomainMarkers object
		dm_retrieved = markers(Ω)

		@test dm_retrieved isa DomainMarkers
		@test length(dm_retrieved.symbols) == 2
		@test length(dm_retrieved.tuples) == 1
		@test length(dm_retrieved.functions) == 1

		# Test labels (collect generator)
		lbls = Set(labels(Ω))
		@test lbls == Set([:bnd_left, :corners, :region1, :boundary])

		# Test specific identifier types (collect generators)
		@test Set(marker_symbols(Ω)) == Set([:left, :top])
		@test Set(marker_tuples(Ω)) == Set([Set((:top, :right))])

		# Test accessors on domain with only one type of marker
		Ω_sym_only = domain(I1D, :a => :left)
		@test Set(labels(Ω_sym_only)) == Set([:a])
		@test Set(marker_symbols(Ω_sym_only)) == Set([:left])
		@test isempty(collect(marker_tuples(Ω_sym_only)))
		@test isempty(collect(marker_conditions(Ω_sym_only))) # Test empty generator

		# Test projection
		proj1 = projection(Ω, 1)
		proj2 = projection(Ω, 2)
		@test proj1 isa CartesianProduct{1,Float32}
		@test proj1.box == ((0.0f0, 1.0f0),)
		@test proj2 isa CartesianProduct{1,Float32}
		@test proj2.box == ((2.0f0, 3.0f0),)
	end

	@testset "Show Method" begin
		Ω = domain(I2D, :bnd => :top, :region => func1)
		# Capture output
		output = sprint(show, Ω)

		# Basic checks for content - adjust if CartesianProduct show changes
		@test occursin("Type: Float32", output)
		@test occursin("Dim: 2", output)
		@test occursin("Set: [0.0, 1.0] × [2.0, 3.0]", output)
		@test occursin("Markers: ", output)
		# Order isn't guaranteed by Set iteration, so check for presence of labels
		@test occursin("bnd", output)
		@test occursin("region", output)

		# Test show with no markers
		Ω_empty = domain(I1D, create_markers(I1D)) # Manually create empty markers
		output_empty = sprint(show, Ω_empty)
		@test occursin("Type: Float64", output_empty)
		@test occursin("Dim: 1", output_empty)
		@test occursin("Set: [0.0, 1.0]", output_empty)

		# Test show with default dirichlet marker
		Ω_def = domain(I1D)
		output_def = sprint(show, Ω_def)
		@test occursin("Markers: dirichlet", output_def)
	end
end # End Domain System Tests
