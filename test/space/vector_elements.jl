import Bramble: VectorElement, spacing, points, half_points, space, values, ndofs, values!, _func2array!, ComponentStyle, half_spacings_iterator, half_points_iterator, indices, point
using LinearAlgebra: norm

valid_interior_range(i::Int, dims::NTuple{D}) where D = ntuple(k -> k == i ? (2:dims[k]) : (1:dims[k]), Val(D))

"""
Calculates the exact cell average of `x -> exp(-x)` over an interval.
"""
@inline function cell_avg_exp(v::NTuple{3,T}) where T
	h, x0, x1 = v
	return (exp(-x0) - exp(-x1)) / h
end

"""
Populates array `w` with cell-averaged values of the separable function
`f(x) = exp(-sum(x))` on the given `mesh`.
"""
function compute_exp_cell_averages!(w::Array{T,D}, mesh) where {T,D}
	# Create an iterator for each dimension that yields `(hᵢ, xᵢ, xᵢ₊₁)` for cell interfaces
	cell_data_iterators = ntuple(Val(D)) do i
		mesh_dim = mesh(i)
		zip(half_spacings_iterator(mesh_dim),
			half_points_iterator(mesh_dim),
			Iterators.drop(half_points_iterator(mesh_dim), 1))
	end

	# Iterate over the Cartesian product of the dimensional iterators
	@inbounds for (i, v_tuple) in enumerate(Iterators.product(cell_data_iterators...))
		w[i] = prod(cell_avg_exp, v_tuple)
	end
end

"""
Sets up the test grid, space, and a sample element for a given dimension `D`.
"""
function setup_test_grid(::Val{D}) where D
	# Using tuples indexed by D is a clean way to handle dimension-specific settings
	npts_options = ((4,), (4, 4), (4, 4, 4))
	unif_options = ((false,), (false, false), (false, false, false))

	dims = npts_options[D]
	unif = unif_options[D]

	intervals = ntuple(_ -> interval(-1.0, 4.0), Val(D))
	Ω = domain(reduce(×, intervals))

	Ωₕ = mesh(Ω, dims, unif)
	Wₕ = gridspace(Ωₕ)
	uₕ = element(Wₕ, 1)

	return dims, Wₕ, uₕ
end

@testset "VectorElement Tests" begin
	# Setup a mock space
	W = gridspace(mesh(domain(box(0, 1)), 4, true))

	@testset "Constructors" begin
		u1 = element(W)
		@test u1 isa VectorElement
		@test space(u1) === W
		@test values(u1) isa Vector
		@test length(values(u1)) == ndofs(W)
		@test eltype(values(u1)) == Float64

		u2 = element(W, 5.0)
		@test u2 isa VectorElement
		@test space(u2) === W
		@test all(==(5.0), values(u2))
		@test length(u2) == 4

		v_init = collect(1.0:4.0)
		u3 = element(W, v_init)
		@test u3 isa VectorElement
		@test space(u3) === W
		@test values(u3) == v_init
		@test_throws AssertionError element(W, collect(1.0:5.0))

		u4 = element(W, 3) # Test with Int
		@test u4 isa VectorElement
		@test space(u4) === W
		@test all(==(3.0), values(u4))
		@test eltype(u4) == Float64
	end

	@testset "Getters and Setters" begin
		u = element(W, 1.0:4.0)
		@test space(u) === W
		@test values(u) == collect(1.0:4.0)

		values!(u, fill(2.0, 4))
		@test values(u) == fill(2.0, 4)

		# Test copyto! alias
		copyto!(u, fill(3.0, 4))
		@test values(u) == fill(3.0, 4)
	end

	@testset "Forwarded Methods" begin
		u = element(W, 1.0:4.0)
		@test size(u) == (4,)
		@test length(u) == 4
		@test firstindex(u) == 1
		@test lastindex(u) == 4
		@test eltype(u) == Float64
		@test collect(u) == collect(1.0:4.0)
	end

	@testset "ndims" begin
		@test ndims(VectorElement) == 1
		u = element(W)
		@test ndims(u) == 1 # ndims often works on instances too
	end

	@testset "Indexing" begin
		u = element(W, 1.0:4.0)
		@test u[1] == 1.0
		@test u[4] == 4.0

		u[3] = 99.0
		@test u[3] == 99.0
		@test values(u)[3] == 99.0
	end

	@testset "similar" begin
		u = element(W, 1.0:4.0)
		s = similar(u)
		@test s isa VectorElement
		@test space(s) === space(u)
		@test length(s) == length(u)
		@test eltype(s) == eltype(u)
		# Values are uninitialized, so don't test their content directly
		s[1] = 1.0
		@test s[1] == 1.0
	end

	@testset "copyto!" begin
		u = element(W, 1.0:4.0)
		v = element(W, 11.0:14.0)
		z = element(W) # Uninitialized

		# VectorElement to VectorElement
		copyto!(z, u)
		@test values(z) == values(u)
		@test !(values(z) === values(u)) # Ensure it's a copy

		# AbstractVector to VectorElement
		vec_data = fill(5.5, 4)
		copyto!(z, vec_data)
		@test values(z) == vec_data
	end

	@testset "Broadcasting" begin
		u = element(W, 1.0:4.0)
		v = element(W, fill(2.0, 4))
		w = element(W)
		α = 3.0
		β = 2.0

		# Test similar for broadcast result
		bc = Base.broadcasted(+, u, v)
		s = similar(bc)
		@test s isa VectorElement
		@test space(s) === space(u)
		@test length(s) == length(u)

		# Test copyto! broadcast (u .= v)
		copyto!(u, Base.broadcasted(identity, v))
		@test values(u) == values(v)

		# Test materialize! / fused (w .= u .+ v .* α)
		w .= u .+ v .* α # Uses materialize! implicitly
		expected_w = values(u) .+ values(v) .* α
		@test values(w) ≈ expected_w

		# Test copyto! variant (w .= β .* v)
		w .= β .* v
		expected_w2 = β .* values(v)
		@test values(w) ≈ expected_w2

		# Test scalar assignment via broadcast
		w .= 5.0
		@test all(==(5.0), values(w))
	end

	@testset "Arithmetic Operators" begin
		u_data = collect(1.0:4.0)
		v_data = fill(2.0, 4)
		u = element(W, u_data)
		v = element(W, v_data)
		α = 3.0
		β = 2.0

		# VectorElement + VectorElement
		r3 = u + v
		@test r3 isa VectorElement
		@test space(r3) === space(u)
		@test values(r3) ≈ u_data .+ v_data

		# Scalar * VectorElement
		r4 = α * u
		@test values(r4) ≈ α .* u_data

		# VectorElement * Scalar
		r5 = u * α
		@test values(r5) ≈ u_data .* α

		# VectorElement * VectorElement
		r6 = u .* v
		@test values(r6) ≈ u_data .* v_data

		# Subtraction
		r7 = u - v
		@test values(r7) ≈ u_data .- v_data
		r8 = u .- α
		@test values(r8) ≈ u_data .- α
		r9 = α .- u
		@test values(r9) ≈ α .- u_data

		# Power
		r13 = u .^ β
		@test values(r13) ≈ u_data .^ β

		r15 = u .^ v # Elementwise
		@test values(r15) ≈ u_data .^ v_data
	end
end

@testset "PDE Operators (Rₕ, avgₕ, ∇₋ₕ)" begin
	for D in 1:3
		@testset "$D-Dimensional Tests" begin
			dims, Wₕ, uₕ = setup_test_grid(Val(D))
			@test length(uₕ) == prod(dims)

			@testset "Rₕ! (Projection)" begin
				test_function(x) = exp(-sum(x))
				Rₕ!(uₕ, test_function)

				# Reference calculation
				w = Array{Float64,D}(undef, dims)
				test_function_idx(idx) = test_function(point(mesh(Wₕ), idx))
				_func2array!(ComponentStyle(typeof(Wₕ)), w, test_function_idx, indices(mesh(Wₕ)))

				w_flat = reshape(w, prod(dims))
				@test norm(values(uₕ) - w_flat) < 1e-15
			end

			@testset "avgₕ! (Cell-Average)" begin
				avgₕ!(uₕ, x -> exp(-sum(x)))

				w = Array{Float64,D}(undef, dims)
				compute_exp_cell_averages!(w, mesh(Wₕ))

				u_reshaped = reshape(values(uₕ), dims)
				interior = valid_interior_range(D, dims)
				@test @views norm(u_reshaped[interior...] - w[interior...]) < 1e-4
			end

			@testset "∇₋ₕ (Backward Difference Gradient)" begin
				function get_gradient_component(der, i, dims, D)
					data = D == 1 ? der.data : der[i].data
					return reshape(data, dims)
				end

				@testset "Gradient of a Constant" begin
					uₕ .= 1.0
					der = ∇₋ₕ(uₕ)
					for i in 1:D
						grad_comp = get_gradient_component(der, i, dims, D)
						interior = valid_interior_range(i, dims)
						@test @views norm(grad_comp[interior...]) < 1e-14
					end
				end

				@testset "Gradient of Linear Functions" begin
					for component in 1:D
						# Test with f(x) = x[component]
						Rₕ!(uₕ, x -> x[component])
						der = ∇₋ₕ(uₕ)

						for i in 1:D
							grad_comp = get_gradient_component(der, i, dims, D)
							interior = valid_interior_range(i, dims)
							# The derivative should be 1 if i==component, and 0 otherwise.
							expected = (i == component) ? 1.0 : 0.0
							@test @views norm(grad_comp[interior...] .- expected) < 1e-14
						end
					end
				end
			end
		end
	end
end