using Test
using Bramble
using Bramble: __prod, _innerplus_weights!, spacing, _innerh_weights!, _innerplus_mean_weights!, __innerplus_weights!, half_spacing, space_weights, SpaceWeights
using Bramble: Innerh, Innerplus
using LinearAlgebra: norm

@testset "Scalar and Vector GridSpaces" begin
	mesh1d = mesh(domain(interval(0, 1)), 10, true)
	mesh2d = mesh(domain(box((0, 0), (0.5, 0.6))), (5, 6), (true, true))
	mesh3d = mesh(domain(box((0, 0, 0), (0.5, 0.6, 0.7))), (4, 4, 4), (true, true, true))

	@testset "Weight Helper Functions" begin
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

		@testset "_innerh_weights!" begin
			# 1D
			u1 = vector(backend(mesh1d), npoints(mesh1d))
			_innerh_weights!(u1, mesh1d)
			@test length(u1) == npoints(mesh1d)
			@test all(u1 .> 0)

			# 2D
			u2 = vector(backend(mesh2d), npoints(mesh2d))
			_innerh_weights!(u2, mesh2d)
			expected_norm = 0.05952940449895328
			@test norm(u2) ≈ expected_norm
		end

		@testset "_innerplus_weights!" begin
			u = vector(backend(mesh1d), npoints(mesh1d))
			_innerplus_weights!(u, mesh1d, 1)
			@test u[1] == 0.0
			for i in 2:npoints(mesh1d)
				@test u[i] ≈ spacing(mesh1d, i)
			end
		end

		@testset "_innerplus_mean_weights!" begin
			u = vector(backend(mesh1d), npoints(mesh1d))
			N = npoints(mesh1d)
			_innerplus_mean_weights!(u, mesh1d, 1)
			@test u[1] == 0.0
			@test u[N] == 0.0
			for i in 2:(N - 1)
				@test u[i] ≈ half_spacing(mesh1d, i)
			end
		end

		@testset "__innerplus_weights!" begin
			npts_tup = npoints(mesh2d, Tuple)
			v = zeros(Float64, npts_tup)
			comp_weights = (rand(npts_tup[1]), rand(npts_tup[2]))
			__innerplus_weights!(v, comp_weights)

			idx = CartesianIndex(3, 4)
			@test v[idx] ≈ comp_weights[1][idx[1]] * comp_weights[2][idx[2]]
		end
	end

	@testset "Space Weights Computation" begin
		D2 = dim(mesh2d)
		b2 = backend(mesh2d)
		VT2 = vector_type(b2)

		sw2 = space_weights(mesh2d)
		@test sw2 isa SpaceWeights{D2,VT2}
		@test length(sw2.innerh) == npoints(mesh2d)
		@test length(sw2.innerplus) == D2
		@test all(length(w) == npoints(mesh2d) for w in sw2.innerplus)
	end

	@testset "ScalarGridSpace Construction and Properties" begin
		W1 = gridspace(mesh1d)
		W2 = gridspace(mesh2d)
		W3 = gridspace(mesh3d)

		@testset "Types and Fields" begin
			@test W1 isa ScalarGridSpace
			@test W2 isa ScalarGridSpace
			@test W3 isa ScalarGridSpace
			@test isimmutable(W1)

			# Verify cached matrices are NOT stored on the struct
			@test propertynames(W1) == (:mesh, :weights, :vector_buffer)
			@test !hasfield(ScalarGridSpace, :backward_difference_matrix)
			@test !hasfield(ScalarGridSpace, :average_matrix)
			@test !hasfield(ScalarGridSpace, :has_backward_difference_matrix)
			@test !hasfield(ScalarGridSpace, :has_average_matrix)
		end

		@testset "Interface Accessors" begin
			@test mesh(W1) === mesh1d
			@test mesh(W2) === mesh2d
			@test mesh_type(W1) === typeof(mesh1d)
			@test mesh_type(typeof(W1)) === typeof(mesh1d)
			@test dim(W1) == 1
			@test dim(W2) == 2
			@test dim(W3) == 3
			@test dim(typeof(W1)) == 1
			@test dim(typeof(W2)) == 2
			@test dim(typeof(W3)) == 3
			@test eltype(W1) === Float64
			@test eltype(typeof(W1)) === Float64
			@test backend(W1) === backend(mesh1d)
			@test ncomponents(W1) == 1
			@test ncomponents(typeof(W1)) == 1
			@test spaces(W1) === (W1,)

			@test ndofs(W1) == 10
			@test ndofs(W1, Tuple) == (10,)
			@test ndofs(W2) == 30
			@test ndofs(W2, Tuple) == (5, 6)
			@test ndofs(W3) == 64
			@test ndofs(W3, Tuple) == (4, 4, 4)

			@test vector_buffer(W1) isa Bramble.GridSpaceBuffer
		end

		@testset "Weights Accessors" begin
			w_h = weights(W2, Innerh())
			@test w_h isa AbstractVector
			@test length(w_h) == 30
			@test weights(W2, Innerh(), 1) === w_h

			w_plus = weights(W2, Innerplus())
			@test w_plus isa Tuple
			@test length(w_plus) == 2
			@test weights(W2, Innerplus(), 1) === w_plus[1]
			@test weights(W2, Innerplus(), 2) === w_plus[2]
		end
	end

	@testset "CompositeGridSpace / VectorGridSpace Construction and Properties" begin
		W = gridspace(mesh2d)

		@testset "Constructors" begin
			# Via Val(N)
			V_val = gridspace(mesh2d, Val(2))
			@test V_val isa CompositeGridSpace{2}
			@test V_val isa VectorGridSpace{2}
			@test isimmutable(V_val)

			# Via integer N
			V_int = gridspace(mesh2d, 2)
			@test V_int isa CompositeGridSpace{2}
			@test length(V_int) == 2

			# Via vector_gridspace
			V_vec = vector_gridspace(mesh2d)
			@test V_vec isa CompositeGridSpace{2}
			@test length(V_vec) == dim(mesh2d)

			V_vec3 = vector_gridspace(mesh2d, 3)
			@test length(V_vec3) == 3

			# Via exponentiation ^
			V_pow = W^2
			@test V_pow isa CompositeGridSpace{2}
			@test V_pow[1] === W
			@test V_pow[2] === W

			V_pow_val = W^Val(3)
			@test V_pow_val isa CompositeGridSpace{3}
			@test V_pow_val[1] === W

			# Via product ×
			V_prod = W × W
			@test V_prod isa CompositeGridSpace{2}

			# Vararg constructor
			V_vararg = CompositeGridSpace(W, W, W)
			@test V_vararg isa CompositeGridSpace{3}
		end

		@testset "Interface Accessors" begin
			V = W^2
			@test mesh(V) === mesh2d
			@test mesh_type(V) === typeof(mesh2d)
			@test dim(V) == 2
			@test eltype(V) === Float64
			@test eltype(typeof(V)) === Float64
			@test backend(V) === backend(mesh2d)
			@test vector_buffer(V) === vector_buffer(W)
			@test ncomponents(V) == 2
			@test ncomponents(typeof(V)) == 2

			@test ndofs(V) == 2 * ndofs(W)
			@test ndofs(V, Tuple) == (ndofs(W), ndofs(W))
			@test spaces(V) === (W, W)

			# Weights forwarding
			@test weights(V) === weights(W)
			@test weights(V, Innerh()) === weights(W, Innerh())
			@test weights(V, Innerplus(), 1) === weights(W, Innerplus(), 1)
		end

		@testset "Collection Interface" begin
			W_a = gridspace(mesh1d)
			W_b = gridspace(mesh1d)
			V = CompositeGridSpace(W_a, W_b)

			@test length(V) == 2
			@test firstindex(V) == 1
			@test lastindex(V) == 2
			@test V[1] === W_a
			@test V[2] === W_b
			@test eachindex(V) == 1:2
			@test keys(V) == 1:2

			# Iteration
			collected = [s for s in V]
			@test length(collected) == 2
			@test collected[1] === W_a
			@test collected[2] === W_b
		end

		@testset "Hierarchical Spaces (for coupled problems like Stokes)" begin
			# Vh (velocity, 2D) and Qh (pressure, 1D)
			Vh = W × W
			Qh = W
			SystemSpace = Vh × Qh
			@test SystemSpace isa CompositeGridSpace{2}
			@test SystemSpace[1] isa CompositeGridSpace{2}
			@test SystemSpace[2] isa ScalarGridSpace
			@test ndofs(SystemSpace) == 3 * ndofs(W)
		end
	end
end