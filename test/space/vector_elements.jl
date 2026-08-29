import Bramble: VectorElement, spacing, points, half_points, space, values, ndofs, values!, _func2array!, half_spacings_iterator, half_points_iterator, indices, point
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
		@test_throws DimensionMismatch element(W, collect(1.0:5.0))

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
				_func2array!(w, test_function_idx, indices(mesh(Wₕ)))

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

			# Defer ∇₋ₕ tests until space/operators/difference.jl is enabled
		end
	end

	@testset "Component Indexing & Multi-Component Spaces" begin
		m = mesh(domain(box((0, 0), (1, 1))), (5, 6), (true, true))
		W = gridspace(m)
		V = W^2

		u_vec = element(V)
		@test length(u_vec) == 2 * ndofs(W)
		@test ncomponents(space(u_vec)) == 2

		# Component extraction via functor call u(i) and component(u, i)
		u1 = u_vec(1)
		u2 = u_vec(2)
		@test u1 isa VectorElement
		@test u2 isa VectorElement
		@test space(u1) === W
		@test space(u2) === W
		@test length(u1) == ndofs(W)
		@test length(u2) == ndofs(W)
		@test component(u_vec, 1) === u1 || values(component(u_vec, 1)) == values(u1)

		# Component ranges
		@test component_range(V, 1) == 1:ndofs(W)
		@test component_range(V, 2) == (ndofs(W) + 1):(2 * ndofs(W))
		@test component_ranges(V) == (1:ndofs(W), (ndofs(W) + 1):(2 * ndofs(W)))

		# components() tuple
		comps = components(u_vec)
		@test length(comps) == 2
		@test comps[1] isa VectorElement
		@test comps[2] isa VectorElement

		# Scalar space component indexing
		u_scal = element(W, 3.0)
		@test u_scal(1) === u_scal
		@test component(u_scal, 1) === u_scal
		@test components(u_scal) === (u_scal,)
		@test_throws BoundsError u_scal(2)
		@test_throws BoundsError u_vec(0)
		@test_throws BoundsError u_vec(3)

		# In-place mutation through component views
		u1 .= 10.0
		u2 .= 25.0
		@test all(==(10.0), values(u_vec)[1:ndofs(W)])
		@test all(==(25.0), values(u_vec)[ndofs(W)+1:2*ndofs(W)])

		# to_matrix on multi-component elements
		mats = to_matrix(u_vec)
		@test mats isa Tuple
		@test length(mats) == 2
		@test size(mats[1]) == (5, 6)
		@test size(mats[2]) == (5, 6)
		@test all(==(10.0), mats[1])
		@test all(==(25.0), mats[2])

		# Multi-component Rₕ
		Rₕ!(u_vec, (x -> x[1], x -> x[2]))
		@test mats[1][1, 1] ≈ m[1, 1][1]
		@test mats[2][1, 1] ≈ m[1, 1][2]

		# Multi-component avgₕ
		u_avg = avgₕ(V, (x -> 2.0, x -> 5.0))
		mats_avg = to_matrix(u_avg)
		@test mats_avg[1][2, 2] ≈ 2.0
		@test mats_avg[2][2, 2] ≈ 5.0
	end

@testset "avgₕ quadrature" begin
	import Bramble: _gauss_rule, AVG_QUAD_POINTS, values
	using StaticArrays

	Ωₕ = mesh(domain(interval(0.0, 1.0)), 40, false)   # non-uniform
	W  = gridspace(Ωₕ)
	u  = element(W)
	f(x) = exp(-sum(x))

	# exact cell averages of exp(-x) over [xᵢ₋₁ᐟ₂, xᵢ₊₁ᐟ₂]
	xh = half_points(Ωₕ)
	exact = [(exp(-xh[i]) - exp(-xh[i + 1])) / (xh[i + 1] - xh[i]) for i in 1:npoints(Ωₕ)]

	@testset "converges in the number of points" begin
		errs = map(1:4) do nq
			avgₕ!(u, f; quad_points = nq)
			maximum(abs, values(u) .- exact)
		end
		# The mesh is randomly non-uniform, so assert the trend and generous
		# bounds rather than tight magic constants.
		@test errs[1] > errs[2] > errs[3]
		@test errs[1] < 1e-3             # 1 point is the midpoint rule
		@test errs[2] < 1e-7
		@test errs[4] < 1e-10

		# the shipped default must reach machine precision on this integrand
		avgₕ!(u, f)
		@test maximum(abs, values(u) .- exact) < 1e-11

		@test_throws ArgumentError avgₕ!(u, f; quad_points = 0)
	end

	@testset "rule is exact for polynomials of degree 2N-1" begin
		# with N points the rule must integrate x^(2N-1) exactly
		for nq in 1:4
			deg = 2nq - 1
			g(x) = sum(x)^deg
			avgₕ!(u, g; quad_points = nq)
			ex = [(xh[i + 1]^(deg + 1) - xh[i]^(deg + 1)) / ((deg + 1) * (xh[i + 1] - xh[i]))
				  for i in 1:npoints(Ωₕ)]
			@test maximum(abs, values(u) .- ex) < 1e-12
		end
	end

	@testset "rule construction is free for IEEE floats" begin
		for T in (Float64, Float32)
			nodes, wts = _gauss_rule(Val(3), T)
			@test nodes isa SVector{3,T}
			@test wts isa SVector{3,T}
			@test sum(wts) ≈ one(T)
			# folded to a constant at compile time, so obtaining it allocates nothing
			get_rule() = _gauss_rule(Val(3), T)
			get_rule()
			@test (@allocated get_rule()) == 0
		end

		# BigFloat keeps the run-time path: its precision is a run-time setting.
		nb, wb = _gauss_rule(Val(3), BigFloat)
		@test eltype(nb) === BigFloat
		@test abs(sum(wb) - one(BigFloat)) < 1e-50
	end

	@testset "allocations do not grow with the grid" begin
		# Measured behind a function barrier: at global scope @allocated also
		# counts the boxing of the non-const globals it touches.
		function avg_bytes(n)
			Ω2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (n, n))
			W2 = gridspace(Ω2); u2 = element(W2)
			run!(uu, g) = avgₕ!(uu, g)
			run!(u2, f); run!(u2, f)
			return @allocated run!(u2, f)
		end
		# 4096x the degrees of freedom must not cost more per call.
		@test avg_bytes(1024) <= avg_bytes(16)
	end
end

end

@testset "Composite components and single-pass evaluation" begin
	import Bramble: component_range, component_ranges, components, values, ndofs, spaces

	W5 = gridspace(mesh(domain(interval(0.0, 1.0)), 5, true))
	W9 = gridspace(mesh(domain(interval(0.0, 1.0)), 9, true))

	@testset "Heterogeneous composites are indexed by cumulative size" begin
		# Subspaces of the same *type* can hold different numbers of degrees of
		# freedom, so component ranges must be summed, never inferred from types.
		V = W5 × W9
		@test ndofs(V) == 14
		@test component_range(V, 1) == 1:5
		@test component_range(V, 2) == 6:14
		@test component_ranges(V) == (1:5, 6:14)

		u = element(V, 0.0)
		cs = components(u)
		@test length.(cs) == (5, 9)

		# Writing through one component must not touch the other.
		cs[1] .= 1.0
		@test all(==(1.0), values(u)[1:5])
		@test all(==(0.0), values(u)[6:14])

		@test_throws BoundsError component_range(V, 3)
	end

	@testset "One function per component equals one vector-valued function" begin
		Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (8, 8))
		W = gridspace(Ωₕ)
		for NC in (2, 3, 4)
			V = W^Val(NC)
			fvec = x -> ntuple(k -> sin(k * x[1]) + cos(k * x[2]), Val(NC))
			ftup = ntuple(k -> (x -> sin(k * x[1]) + cos(k * x[2])), Val(NC))

			a = element(V); b = element(V)
			Rₕ!(a, fvec); Rₕ!(b, ftup)
			@test values(a) == values(b)

			c = element(V); d = element(V)
			avgₕ!(c, fvec); avgₕ!(d, ftup)
			@test values(c) == values(d)
		end
	end

	@testset "A one-tuple of functions works on a scalar space" begin
		Ωₕ = mesh(domain(interval(0.0, 1.0)), 8, true)
		W = gridspace(Ωₕ)
		f = x -> 2.0

		u1 = element(W); u2 = element(W)
		Rₕ!(u1, f); Rₕ!(u2, (f,))
		@test values(u1) == values(u2)

		v1 = element(W); v2 = element(W)
		avgₕ!(v1, f); avgₕ!(v2, (f,))
		@test values(v1) == values(v2)
	end

	@testset "In-place operators return nothing" begin
		Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (6, 6))
		W = gridspace(Ωₕ); V = W^Val(2)
		u = element(W); uv = element(V)
		f = x -> 1.0
		ft = (x -> 1.0, x -> 2.0)
		fv = x -> (1.0, 2.0)

		@test Rₕ!(u, f) === nothing
		@test Rₕ!(uv, ft) === nothing
		@test Rₕ!(uv, fv) === nothing
		@test avgₕ!(u, f) === nothing
		@test avgₕ!(uv, ft) === nothing
		@test avgₕ!(uv, fv) === nothing

		# the allocating forms still hand back the element
		@test Rₕ(W, f) isa VectorElement
		@test avgₕ(W, f) isa VectorElement
	end
end

