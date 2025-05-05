using Bramble: __prod, mesh, box, domain, interval, dim, backend, npoints, vector, matrix, _innerplus_weights!, spacing, build_innerh_weights!, _innerplus_mean_weights!, __innerplus_weights!, half_spacing, create_space_weights, SpaceWeights,
			   create_space_backward_diff_matrices, vector_type, matrix_type, mesh, weights, has_backward_diff_matrix, has_average_matrix, gridspace, ndofs
using LinearAlgebra: norm

@testset "SingleGridSpace construction" begin
	@testset "Weight Helper Functions" begin
		mesh1d = mesh(domain(interval(0, 1)), 10, true)
		mesh2d = mesh(domain(box((0, 0), (0.5, 0.6))), (5, 6), (true, true))
		D1, D2 = dim(mesh1d), dim(mesh2d)
		VT = Vector{Float64}

		@testset "__prod" begin
			# Test D=1
			v1 = ([1.0, 2.0, 3.0],)
			idx1 = CartesianIndex(2)
			@test __prod(v1, idx1) ≈ 2.0

			# Test D=2
			v2 = ([1.0, 2.0], [3.0, 4.0, 5.0])
			idx2 = CartesianIndex(2, 3)
			@test __prod(v2, idx2) ≈ 2.0 * 5.0 ≈ 10.0
		end

		@testset "build_innerh_weights!" begin
			u = vector(backend(mesh2d), npoints(mesh2d))
			build_innerh_weights!(u, mesh2d)
			expected_norm = 0.05952940449895328
			@test norm(u) .≈ expected_norm
		end

		@testset "_innerplus_weights!" begin
			u = vector(backend(mesh1d), npoints(mesh1d))
			_innerplus_weights!(u, mesh1d, 1)
			@test u[1] == 0.0
			for i ∈ 2:npoints(mesh1d)
				@test u[i] ≈ spacing(mesh1d, i)
			end
		end

		@testset "_innerplus_mean_weights!" begin
			u = vector(backend(mesh1d), npoints(mesh1d))
			N = npoints(mesh1d)
			_innerplus_mean_weights!(u, mesh1d, 1)
			@test u[1] == 0.0
			@test u[N] == 0.0
			for i ∈ 2:(N - 1)
				@test u[i] ≈ half_spacing(mesh1d, i)
			end
		end

		@testset "__innerplus_weights!" begin
			# Use D=2 example
			npts_tup = npoints(mesh2d, Tuple)
			v = zeros(Float64, npts_tup) # Reshaped array target
			comp_weights = (rand(npts_tup[1]), rand(npts_tup[2])) # Example component weights
			__innerplus_weights!(v, comp_weights)
			# Test a specific point
			idx = CartesianIndex(3, 4)
			@test v[idx] ≈ comp_weights[1][idx[1]] * comp_weights[2][idx[2]]
		end
	end

	@testset "Space Creation Functions" begin
		mesh2d = mesh(domain(box((0, 0), (0.5, 0.6))), (5, 6), (true, true))
		D = dim(mesh2d)
		T = eltype(mesh2d)
		b = backend(mesh2d)
		VT = vector_type(b)
		MT = matrix_type(b)

		@testset "create_space_weights" begin
			# Use the optimized version if available
			weights = create_space_weights(mesh2d) # Or create_space_weights
			@test weights isa SpaceWeights{D,VT}
			@test size(weights.innerh) == (npoints(mesh2d),)
			@test length(weights.innerplus) == D
			@test all(size(w) == (npoints(mesh2d),) for w in weights.innerplus)
			# Could add value checks if mocks are deterministic enough
		end

		@testset "create_space_backward_diff_matrices" begin
			mats = create_space_backward_diff_matrices(mesh2d)
			@test mats isa NTuple{D,MT}
			@test length(mats) == D
			@test all(size(m) == (npoints(mesh2d), npoints(mesh2d)) for m in mats)
		end
	end

	@testset "SingleGridSpace Constructor and Accessors" begin
		mesh1d = mesh(domain(box(0, 1)), 3, true)
		mesh3d = mesh(domain(box((0, 0, 0), (0.5, 0.6, 0.7))), (5, 6, 4), (true, true, true))

		# Test default caching
		space1d = gridspace(mesh1d)
		@test mesh(space1d) === mesh1d
		@test weights(space1d) isa SpaceWeights{1,Vector{Float64}}
		@test has_backward_diff_matrix(space1d) == true
		@test has_average_matrix(space1d) == false
		@test length(space1d.backward_diff_matrix) == 1
		@test length(space1d.average_matrix) == 1

		# Test custom caching
		space3d = gridspace(mesh3d; cache_average_matrices = true, cache_backward_diff_matrices = false)
		@test dim(space3d) == 3
		@test ndofs(space3d) == npoints(mesh3d)
		@test eltype(space3d) == Float64
		@test has_backward_diff_matrix(space3d) == false
		@test has_average_matrix(space3d) == true
	end
end