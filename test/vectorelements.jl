import Bramble: spacing, points, half_points

valid_range(i::Int, ds::NTuple{D,Int}) where {D} = ntuple(k -> k == i ? (2:ds[1]) : (1:ds[k]), D)

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

@inline __test_function(x) = exp(-sum(x))

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

	u .= v

	@test(length(u)==prod(dims))

	for op in (+, -, *, /)
		res = op(u, v)
		@test(validate_equal(res, map(op, u, v)))
	end

	Rₕ!(u, __test_function)

	w = Array{Float64,D}(undef, dims)
	Bramble._func2array!(w, __test_function, mesh(Wₕ))

	w2 = reshape(w, prod(dims))
	@test(validate_equal(u, w2))

	avgₕ!(u, __test_function)
	__func2array_med(w, mesh(Wₕ))

	vv = reshape(u.values, dims)
	@test @views isapprox(vv[valid_range(D, dims)...], w[valid_range(D, dims)...]; atol = 1e-5)

	u .= 1.0
	der = ∇ₕ(u)
	for i in 1:D
		dd = D == 1 ? reshape(der.values, dims) : reshape(der[i].values, dims)
		@views ee = dd[valid_range(i, dims)...]
		@test(validate_zero(ee))
	end

	wf(x, i) = x[i]
	for dimension in 1:D
		Rₕ!(u, Base.Fix2(wf, dimension))
		der = ∇ₕ(u)

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
