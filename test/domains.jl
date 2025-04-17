using Bramble: Marker, BrambleFunction
using Bramble: get_boundary_symbols, create_markers, label, identifier, domain, set, markers, labels, marker_identifiers

# Helper fFunction if needed (using isapprox for floats)  # Test fFunctions
inner_f1 = x -> x[1] - 0.0
inner_f2(x) = sum(x .^ 2) - 0.25 # Example for 2D/3D

# Test.box
M = [-3.0 10.0; 70.0 100.0; -15.0 -1.0]
I1 = interval(M[1, 1], M[1, 2])
I2 = interval(M[2, 1], M[2, 2])
I3 = interval(M[3, 1], M[3, 2])
X1 = I1
X2 = I1 × I2
X3 = I1 × I2 × I3

@testset "Domain Tests" begin
	@testset "get_boundary_symbols" begin
		@test get_boundary_symbols(Val(1)) === (:left, :right)
		@test get_boundary_symbols(Val(2)) === (:bottom, :top, :left, :right) # Check order matches definition
		@test get_boundary_symbols(Val(3)) === (:bottom, :top, :back, :front, :left, :right) # Check order
		@test get_boundary_symbols(X1) === (:left, :right)
		@test get_boundary_symbols(X2) === (:bottom, :top, :left, :right)
		@test get_boundary_symbols(X3) === (:bottom, :top, :back, :front, :left, :right)
	end

	@testset "create_markers" begin
		m1 = create_markers(X1, :func => inner_f1)
		@test length(m1) == 1
		@test m1[1] isa Marker{<:BrambleFunction}
		@test label(m1[1]) === :func
		@test identifier(m1[1]) isa BrambleFunction

		m2 = create_markers(X2, :bnd => :left, :crn => (:top, :right), :fun => inner_f2)
		@test length(m2) == 3
		@test m2[1] isa Marker{Symbol}
		@test identifier(m2[1]) === :left
		@test m2[2] isa Marker{Tuple{Symbol,Symbol}}
		@test identifier(m2[2]) === (:top, :right)
		@test m2[3] isa Marker{<:BrambleFunction}
		@test label(m2[3]) === :fun

		# Test empty call
		m_empty = create_markers(X1)
		@test m_empty isa Tuple{}
		@test length(m_empty) == 0

		# Test error propagation from process_identifier
		@test_throws ErrorException create_markers(X1, :bad => 123)
	end

	@testset "Default Domain Constructor" begin
		d1_def = domain(X1)
		@test set(d1_def) === X1
		@test dim(d1_def) == 1
		@test eltype(d1_def) === Float64
		m_def1 = markers(d1_def)
		@test length(m_def1) == 1
		@test label(m_def1[1]) === :dirichlet
		# Default identifier is tuple of all boundary symbols
		@test identifier(m_def1[1]) === get_boundary_symbols(X1)

		d2_def = domain(X2)
		@test dim(d2_def) == 2
		m_def2 = markers(d2_def)
		@test length(m_def2) == 1
		@test label(m_def2[1]) === :dirichlet
		@test identifier(m_def2[1]) === get_boundary_symbols(X2)
	end

	@testset "Domain Constructors & Accessors" begin
		# Create markers first
		markers_d1 = create_markers(X1, :neumann => :left, :robin => inner_f1)
		markers_d2 = create_markers(X2, :bc1 => :top, :bc2 => inner_f2, :bc3 => (:left, :right))
		markers_d3 = create_markers(X3, :all_bnd => get_boundary_symbols(X3))

		# Test domain(X, markers_tuple)
		d1 = domain(X1, markers_d1)
		d2 = domain(X2, markers_d2)
		d3 = domain(X3, markers_d3)

		# Test dim, eltype, set
		@test dim(d1) === 1
		@test eltype(d1) === Float64
		@test set(d1) === X1
		@test dim(d2) === 2
		@test eltype(d2) === Float64
		@test set(d2) === X2
		@test dim(d3) === 3
		@test eltype(d3) === Float64
		@test set(d3) === X3

		# Test domain(X, pair1, pair2...)
		d1_vp = domain(X1, :neumann => :left, :robin => inner_f1)
		d2_vp = domain(X2, :bc1 => :top, :bc2 => inner_f2, :bc3 => (:left, :right))

		# Test markers(), labels(), marker_identifiers()
		@test markers(d1) === markers_d1
		@test collect(labels(d1)) == [:neumann, :robin]
		ids1 = collect(marker_identifiers(d1))
		@test ids1[1] === :left
		@test ids1[2] isa BrambleFunction

		@test markers(d2) === markers_d2
		@test collect(labels(d2)) == [:bc1, :bc2, :bc3]
		ids2 = collect(marker_identifiers(d2))
		@test ids2[1] === :top
		@test ids2[2] isa BrambleFunction
		@test ids2[3] === (:left, :right)

		@test markers(d3) === markers_d3
		@test collect(labels(d3)) == [:all_bnd]
		ids3 = collect(marker_identifiers(d3))
		@test ids3[1] === get_boundary_symbols(X3)
	end

	@testset "Projection Tests" begin
		d1 = domain(X1) # Use default for simplicity
		d2 = domain(X2)
		d3 = domain(X3)
		domains = (d1, d2, d3)
		sets = (X1, X2, X3)

		for D in 1:3
			local X = domains[D] # Use local to avoid scope issues in loop
			local setX = sets[D]
			for i in 1:D
				Pi = projection(X, i)
				# Check projection is the correct 1D interval
				@test Pi isa CartesianProduct{1}
				# Compare bounds using validate_equal (or isapprox)
				@test validate_equal(Pi.box[1][1], M[i, 1]) # Pi.box is ((low, upp),)
				@test validate_equal(Pi.box[1][2], M[i, 2])
			end
			# Test assertion error for invalid index
			@test_throws AssertionError projection(X, D + 1)
			@test_throws AssertionError projection(X, 0)
		end
	end
end # End Domain Tests testset
