
@inline _inner_product(u, h, v) = transpose(v) * h * u

@inline _inner_product(u::Vector{T}, h::Diagonal{T,Vector{T}}, v::Vector{T}) where T = _dot(u, h.diag, v)

@inline function _dot(u::AbstractVector, v::AbstractVector, w::AbstractVector)
	s = zero(eltype(u))

	@simd for i in eachindex(u, v, w)
		s += u[i] * v[i] * w[i]
	end

	return s
end

##########################
# implementation of innerh 
@inline innerh_weights(S::SpaceType) = S.innerh_weights::Diagonal{eltype(S),Vector{eltype(S)}}
#=
innerₕ(U::VecOrMatElem, V::VecOrMatElem, ::Val{D}) where D = _inner_product(U.values, innerh_weights(space(U)), V.values)
innerₕ(U::VectorElement, V::MatrixElement, ::Val{D}) where {D} = _inner_product(U.values, innerh_weights(space(U)), V.values)::Vector{eltype(U)}

function innerₕ(U::VectorElement, V::VectorElement, ::Val{D}) where D
	T = eltype(U)

	weights = innerh_weights(space(U))::Diagonal{T, Vector{T}}
	return _dot(U.values, weights.diag, V.values)
end

innerₕ(u::VecOrMatElem, v::VecOrMatElem) = innerₕ(u, v, Val(dim(u)))
=#
innerₕ(U::VecOrMatElem, V::VecOrMatElem) = _inner_product(U.values, innerh_weights(space(U)), V.values)

function innerₕ(U::VectorElement, V::VectorElement)
	T = eltype(U)

	weights = innerh_weights(space(U))::Diagonal{T,Vector{T}}
	return _dot(U.values, weights.diag, V.values)
end

normₕ(u::VectorElement) = sqrt(innerₕ(u, u)::eltype(u))

##########################################
# implementation of innerplus and normplus 
@inline innerplus_weights(S::SpaceType, ::Val{i}) where i = S.innerplus_weights[i]
@inline innerplus_weights(S::SpaceType, i::Int) = S.innerplus_weights[i]

e2conttype(::Type{<:VectorElement{S}}) where S = eltype(S)
e2conttype(::Type{<:MatrixElement{S}}) where S = SparseMatrixCSC{eltype(S),Int}

inner₊(U::VecOrMatElem, V::VecOrMatElem) = (@assert dim(mesh(space(U))) == 1 == dim(mesh(space(V))); inner₊(U, V, Val(1)))# : @error "inner₊ not defined for 2D/3D functions"
inner₊(U::VecOrMatElem, V::VecOrMatElem, ::Val{D}) where D = _inner_product(U.values, innerplus_weights(space(U), Val(D)), V.values)::e2conttype(typeof(V))
inner₊(U::VectorElement, V::MatrixElement, ::Val{D}) where D = _inner_product(U.values, innerplus_weights(space(U), Val(D)), V.values)::Vector{eltype(U)}
inner₊(U::VectorElement, V::VectorElement, ::Val{D}) where D = _inner_product(U.values, innerplus_weights(space(U), Val(D)), V.values)::eltype(U)

function inner₊(u::NTuple{D,VecOrMatElem}, v::NTuple{D,VecOrMatElem}) where D
	VType = typeof(u[1])
	TType = e2conttype(VType)
	z = ntuple(i -> inner₊(u[i], v[i], Val(i))::e2conttype(typeof(v[i])), D)::NTuple{D,TType}
	return sum(z)
end

function inner₊(u::NTuple{D,VectorElement}, v::NTuple{D,VectorElement}) where D
	T = eltype(u[1])
	z = ntuple(i -> inner₊(u[i], v[i], Val(i))::T, D)
	return sum(z)
end

inner₊ₓ(U::VecOrMatElem, V::VecOrMatElem) = inner₊(U, V, Val(1))
inner₊ᵧ(U::VecOrMatElem, V::VecOrMatElem) = inner₊(U, V, Val(2))
inner₊₂(U::VecOrMatElem, V::VecOrMatElem) = inner₊(U, V, Val(3))

norm₊(u::VectorElement{SType}) where SType = sqrt(inner₊(u, u)::eltype(SType))::eltype(SType)
norm₊(u::NTuple{D,VectorElement}) where D = sqrt(inner₊(u, u))
norm₊(u::Tuple{VectorElement{SType}}) where SType = sqrt(inner₊(u[1], u[1]))::eltype(SType)

# implementation of snorm₁ₕ and norm₁ₕ

function __snorm_aux(v, u, ::Val{D}) where D
	if D == 1
		_backward_differencex!(v.values, u.values, Base.Fix1(Bramble.hspace, mesh(space(u))(1)), npoints(u, Tuple))
	end

	if D == 2
		_backward_differencey!(v.values, u.values, Base.Fix1(Bramble.hspace, mesh(space(u))(2)), npoints(u, Tuple))
	end

	if D == 3
		_diff1!(v.values, u.values, Base.Fix1(Bramble.hspace, mesh(space(u))(3)), npoints(u, Tuple))
	end
	
	return inner₊(v, v, Val(D))
end

function snorm₁ₕ(u::VectorElement)
	v = similar(u)
	D = dim(mesh(space(u)))

	_backward_differencex!(v.values, u.values, Base.Fix1(Bramble.hspace, mesh(space(u))(1)), npoints(u, Tuple))
	s = inner₊(v, v, Val(1))

	if D >= 2
		_backward_differencey!(v.values, u.values, Base.Fix1(Bramble.hspace, mesh(space(u))(2)), npoints(u, Tuple))
		s += inner₊(v, v, Val(2))

		if D == 3
			_diff1!(v.values, u.values, Base.Fix1(Bramble.hspace, mesh(space(u))(3)), npoints(u, Tuple))
			s += inner₊(v, v, Val(3))
		end
	end
	return sqrt(s)
	#norm₊(∇ₕ(u))
end

@inline norm₁ₕ(u) = sqrt(normₕ(u)^2 + snorm₁ₕ(u)^2)::eltype(u)
