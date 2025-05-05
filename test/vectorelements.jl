import Bramble: VectorElement, spacing, points, half_points, space, values, ndofs, values!
#=
valid_range(i::Int, ds::NTuple{D,Int}) where D = ntuple(k -> k == i ? (2:ds[1]) : (1:ds[k]), D)

@inline function __exp(v::NTuple{D,T}) where {D,T}
	h, x0, x1 = v
	return -(exp(-x1) - exp(-x0)) / h
end

function __func2array_med(w::Array{T,D}, M::MType) where {MType,D,T}
	Xi = ntuple(i -> zip(half_spacing(M(i), Iterator), half_points(M(i), Iterator), Iterators.drop(half_points(M(i), Iterator), 1)), D)

	@inbounds for (i, v) in enumerate(Iterators.product(Xi...))
		w[i] = prod(ntuple(j -> __exp(v[j]), D))
	end
end

function __init(::Val{D}) where {D}
	npts = ((3,), (3, 3), (3, 3, 3))
	unif = ((false,), (true, false), (false, true, false))
	dims = npts[D]

	intervals = ntuple(j -> interval(-1.0, 4.0), D)
	Ω = domain(reduce(×, intervals))

	Ωₕ = mesh(Ω, dims, unif[D])
	Wₕ = gridspace(Ωₕ)
	uₕ = element(Wₕ, 1)

	return dims, Wₕ, uₕ
end

function vector_element_tests(::Val{D}) where {D}
	dims, Wₕ, u = __init(Val(D))

	v = element(Wₕ, 1.0)

	z = normₕ(v)

	if D == 1
		@test validate_equal(z, sqrt(sum(half_spacing(mesh(Wₕ), Iterator))))
		@test validate_zero(norm₊(D₋ₓ(u)))
	elseif D == 2
		@test validate_equal(z, 5.0)
	else
		@test validate_equal(z, 11.180339887498947)
	end

	@test(length(u)==prod(dims))

	for op in (+, -, *, /)
		res = op(u, v)
		@test(validate_equal(res, map(op, u, v)))
	end

	test_function = @embed(mesh(Wₕ), x->exp(-sum(x)))
	Rₕ!(u, test_function)

	w = Array{Float64,D}(undef, dims)
	Bramble._func2array!(w, test_function, mesh(Wₕ))

	w2 = reshape(w, prod(dims))
	@test(validate_equal(u, w2))

	avgₕ!(u, test_function)
	__func2array_med(w, mesh(Wₕ))

	vv = reshape(u.values, dims)
	@test @views isapprox(vv[valid_range(D, dims)...], w[valid_range(D, dims)...]; atol = 1e-5)

	u .= 1.0
	der = ∇₋ₕ(u)
	for i in 1:D
		dd = D == 1 ? reshape(der.values, dims) : reshape(der[i].values, dims)
		@views ee = dd[valid_range(i, dims)...]
		@test(validate_zero(ee))
	end

	for dimension in 1:D
		func = @embed(mesh(Wₕ), x->x[dimension])
		Rₕ!(u, func)
		der = ∇₋ₕ(u)

		for i in 1:D
			dd = D == 1 ? reshape(der.values, dims) : reshape(der[i].values, dims)
			@views ee = dd[valid_range(i, dims)...]

			@test(validate_equal(ee, (i != dimension ? 0.0 : 1.0)))
		end
	end
end

for i in 1:3
	vector_element_tests(Val(i))
end
=#

@testset "VectorElement Tests" begin
	# Setup a mock space
	W = gridspace(mesh(domain(box(0, 1)), 10, true))

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
		@test length(u2) == 10

		v_init = collect(1.0:10.0)
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
		u = element(W, 1.0:10.0)
		@test space(u) === W
		@test values(u) == collect(1.0:10.0)

		values!(u, fill(2.0, 10))
		@test values(u) == fill(2.0, 10)

		# Test copyto! alias
		copyto!(u, fill(3.0, 10))
		@test values(u) == fill(3.0, 10)
	end

	@testset "Forwarded Methods" begin
		u = element(W, 1.0:10.0)
		@test size(u) == (10,)
		@test length(u) == 10
		@test firstindex(u) == 1
		@test lastindex(u) == 10
		@test eltype(u) == Float64
		@test collect(u) == collect(1.0:10.0)
	end

	@testset "ndims" begin
		@test ndims(VectorElement) == 1
		u = element(W)
		@test ndims(u) == 1 # ndims often works on instances too
	end

	@testset "Indexing" begin
		u = element(W, 1.0:10.0)
		@test u[1] == 1.0
		@test u[5] == 5.0
		@test u[10] == 10.0
		@test_throws BoundsError u[0]
		@test_throws BoundsError u[11]

		u[3] = 99.0
		@test u[3] == 99.0
		@test values(u)[3] == 99.0
		@test_throws BoundsError (u[11]=0.0)
	end

	@testset "similar" begin
		u = element(W, 1.0:10.0)
		s = similar(u)
		@test s isa VectorElement
		@test space(s) === space(u)
		@test length(s) == length(u)
		@test eltype(s) == eltype(u)
		# Values are uninitialized, so don't test their content directly
		s[1] = 1.0 # Check if it's writable
		@test s[1] == 1.0
	end

	@testset "copyto!" begin
		u = element(W, 1.0:10.0)
		v = element(W, 11.0:20.0)
		z = element(W) # Uninitialized

		# VectorElement to VectorElement
		copyto!(z, u)
		@test values(z) == values(u)
		@test !(values(z) === values(u)) # Ensure it's a copy

		# AbstractVector to VectorElement
		vec_data = fill(5.5, 10)
		copyto!(z, vec_data)
		@test values(z) == vec_data
	end

	@testset "Broadcasting" begin
		u = element(W, 1.0:10.0)
		v = element(W, fill(2.0, 10))
		w = element(W)
		α = 3.0
		β = 2.0

		# Test similar for broadcast result
		bc = Base.broadcasted(+, u, v)
		s = similar(bc, Float64)
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
		u_data = collect(1.0:10.0)
		v_data = fill(2.0, 10)
		u = element(W, u_data)
		v = element(W, v_data)
		α = 3.0
		β = 2.0

		# Scalar + VectorElement
		r1 = α + u
		@test r1 isa VectorElement
		@test space(r1) === space(u)
		@test values(r1) ≈ α .+ u_data

		# VectorElement + Scalar
		r2 = u + α
		@test r2 isa VectorElement
		@test space(r2) === space(u)
		@test values(r2) ≈ u_data .+ α

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
		r6 = u * v
		@test values(r6) ≈ u_data .* v_data

		# Subtraction
		r7 = u - v
		@test values(r7) ≈ u_data .- v_data
		r8 = u - α
		@test values(r8) ≈ u_data .- α
		r9 = α - u
		@test values(r9) ≈ α .- u_data

		# Division
		r10 = u / β
		@test values(r10) ≈ u_data ./ β
		r11 = β / u # Elementwise
		@test values(r11) ≈ β ./ u_data
		r12 = u / v # Elementwise
		@test values(r12) ≈ u_data ./ v_data

		# Power
		r13 = u^β
		@test values(r13) ≈ u_data .^ β
		# r14 = β ^ u # Might not be standard, depends on tmap! impl. Check if needed.
		r15 = u^v # Elementwise
		@test values(r15) ≈ u_data .^ v_data
	end
end