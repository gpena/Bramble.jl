# some helper functions to calculate the inner products in discrete spaces
@inline function _dot(u::AbstractVector, v::AbstractVector, w::AbstractVector)
	s = zero(eltype(u))

	@simd for i in eachindex(u, v, w)
		s += u[i] * v[i] * w[i]
	end

	return s
end

@inline _inner_product(u::Vector{T}, h::Vector{T}, v::Vector{T}) where T = _dot(u, h, v)

@inline function _inner_product(u::Vector{T}, h::Vector{T}, v::SparseMatrixCSC{T,Int}) where T
	z = similar(u)
	#z .= transpose(v) * Diagonal(h) * u
	@simd for i in eachindex(z, u, h)
		z[i] = zero(T)
		for j in v.colptr[i]:(v.colptr[i + 1] - 1)
			z[i] += u[v.rowval[j]] * h[v.rowval[j]] * v.nzval[j]
		end
	end

	return z
end

@inline function _inner_product(u::SparseMatrixCSC{T,Int}, h::Vector{T}, v::Vector{T}) where T
	z = transpose(similar(v))
	#z .= transpose(v) * Diagonal(h) * u
	@simd for i in eachindex(z, v, h)
		z[i] = zero(T)
		for j in u.colptr[i]:(u.colptr[i + 1] - 1)
			z[i] += v[u.rowval[j]] * h[u.rowval[j]] * u.nzval[j]
		end
	end

	return z
end

@inline function _inner_product_add!(z::AbstractVector, u::AbstractVector, v::AbstractVector, w::AbstractVector)
	@simd for i in eachindex(z, u, v, w)
		z[i] += u[i] * v[i] * w[i]
	end
end

@inline function _inner_product_add!(z::Vector{T}, u::Vector{T}, h::Vector{T}, v::SparseMatrixCSC{T,Int}) where T
	#z .= transpose(v) * Diagonal(h) * u
	@simd for i in eachindex(z, u, h)
		for j in v.colptr[i]:(v.colptr[i + 1] - 1)
			z[i] += u[v.rowval[j]] * h[v.rowval[j]] * v.nzval[j]
		end
	end
end

@inline function _inner_product_add!(z::LinearAlgebra.Transpose{T,Vector{T}}, u::SparseMatrixCSC{T,Int}, h::Vector{T}, v::Vector{T}) where T
	#z .= transpose(v) * Diagonal(h) * u
	@simd for i in eachindex(z, v, h)
		for j in u.colptr[i]:(u.colptr[i + 1] - 1)
			z[i] += v[u.rowval[j]] * h[u.rowval[j]] * u.nzval[j]
		end
	end
end

@inline function _inner_product_add!(Z::SparseMatrixCSC{T,Int}, U::SparseMatrixCSC{T,Int}, h::Vector{T}, V::SparseMatrixCSC{T,Int}) where T 
	Z .+= transpose(V) * Diagonal(h) * U
end

@inline _inner_product(u, h, v) = transpose(v) * Diagonal(h) * u

# implementation of innerh 
@inline innerh_weights(Wₕ::SpaceType) = Wₕ.innerh_weights#::Vector{eltype(Wₕ)}
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
@inline innerₕ(U, V) = _inner_product(U.values, innerh_weights(space(U)), V.values)
@inline innerₕ(uₕ::VectorElement, vₕ::VectorElement) = _dot(uₕ.values, innerh_weights(space(uₕ)), vₕ.values)

@inline normₕ(uₕ::VectorElement) = sqrt(innerₕ(uₕ, uₕ))

##########################################
# implementation of innerplus and normplus 
@inline innerplus_weights(Wₕ::SpaceType, ::Val{i}) where i = (@assert 1 <= i <= dim(mesh(Wₕ));
															  Wₕ.innerplus_weights[i])

@inline inner₊(Uₕ::VecOrMatElem, Vₕ::VecOrMatElem) = inner₊(Uₕ, Vₕ, Val(dim(mesh(space(Uₕ)))))
@inline inner₊(Uₕ::VecOrMatElem, Vₕ::VecOrMatElem, ::Val{i}) where i = _inner_product(Uₕ.values, innerplus_weights(space(Uₕ), Val(i)), Vₕ.values)

#@inline inner₊(uₕ::VectorElement, vₕ::VectorElement, i::Int) = _inner_product(uₕ.values, innerplus_weights(space(uₕ), Val(i)), vₕ.values)::eltype(uₕ)
#@inline inner₊(uₕ::VectorElement, Vₕ::MatrixElement, i::Int) = _inner_product(uₕ.values, innerplus_weights(space(uₕ), Val(i)), Vₕ.values)
#@inline inner₊(Uₕ::MatrixElement, vₕ::VectorElement, i::Int) = _inner_product(Uₕ.values, innerplus_weights(space(Uₕ), Val(i)), vₕ.values)

@inline @generated inner₊(u::NTuple{D}, v::NTuple{D}, ::Type{Tuple}) where D = :(Base.Cartesian.@ntuple $D i->inner₊(u[i], v[i], Val(i)))
@inline inner₊(u::NTuple{D,VectorElement}, v::NTuple{D,VectorElement}) where D = sum(inner₊(u, v, Tuple))

@generated function inner₊(u::NTuple{D,MatrixElement{S1,T}}, v::NTuple{D,VectorElement{S2,T}}) where {S1,S2,T,D}
	ex = :()
	push!(ex.args, :(s = similar(transpose(v[1].values))))
	push!(ex.args, :(s .= zero(T));)

	for i in 1:D
		push!(ex.args, :(inner₊add!(s, u[$i], v[$i], Val($i))))
	end

	return ex
end

@inline inner₊add!(z::LinearAlgebra.Transpose{T,Vector{T}}, Uₕ::MatrixElement, vₕ::VectorElement, ::Val{i}) where {T,i} = _inner_product_add!(z, Uₕ.values, innerplus_weights(space(Uₕ), Val(i)), vₕ.values)

@generated function inner₊(u::NTuple{D,VectorElement{S1,T}}, v::NTuple{D,MatrixElement{S2,T}}) where {S1,S2,T,D}
	ex = :()
	push!(ex.args, :(s = similar(u[1].values)))
	push!(ex.args, :(s .= zero(T));)

	for i in 1:D
		push!(ex.args, :(inner₊add!(s, u[$i], v[$i], Val($i))))
	end

	return ex
end

@inline inner₊add!(z::Vector, uₕ::VectorElement, Vₕ::MatrixElement, ::Val{i}) where i = _inner_product_add!(z, uₕ.values, innerplus_weights(space(uₕ), Val(i)), Vₕ.values)

@generated function inner₊(u::NTuple{D,MatrixElement{S1,T}}, v::NTuple{D,MatrixElement{S2,T}}) where {S1,S2,T,D}
	ex = :()
	push!(ex.args, :(s = similar(u[1].values)))
	push!(ex.args, :(s.nzval .= zero(T));)

	for i in 1:D
		push!(ex.args, :(inner₊add!(s, u[$i], v[$i], Val($i))))
	end

	return ex
end

@inline inner₊add!(Z::SparseMatrixCSC{T,Int}, Uₕ::MatrixElement, Vₕ::MatrixElement, ::Val{i}) where {T,i} = _inner_product_add!(Z, Uₕ.values, innerplus_weights(space(Uₕ), Val(i)), Vₕ.values)

inner₊ₓ(U::VecOrMatElem, V::VecOrMatElem) = inner₊(U, V, Val(1))
inner₊ᵧ(U::VecOrMatElem, V::VecOrMatElem) = inner₊(U, V, Val(2))
inner₊₂(U::VecOrMatElem, V::VecOrMatElem) = inner₊(U, V, Val(3))

norm₊(u::VectorElement{SType}) where SType = sqrt(inner₊(u, u))
norm₊(u::NTuple{D,VectorElement}) where D = sqrt(inner₊(u, u))
norm₊(u::Tuple{VectorElement{SType}}) where SType = sqrt(inner₊(u[1], u[1]))

# implementation of snorm₁ₕ and norm₁ₕ
function __snorm_aux(v, u, ::Val{D}) where D
	if D == 1
		_backward_finite_differencex!(v.values, u.values, Base.Fix1(spacing, mesh(space(u))(1)), npoints(u, Tuple))
	end

	if D == 2
		_backward_finite_differencey!(v.values, u.values, Base.Fix1(spacing, mesh(space(u))(2)), npoints(u, Tuple))
	end

	if D == 3
		_diff1!(v.values, u.values, Base.Fix1(spacing, mesh(space(u))(3)), npoints(u, Tuple))
	end

	return inner₊(v, v, Val(D))
end

function snorm₁ₕ(u::VectorElement)
	v = similar(u)
	Ωₕ = mesh(space(u))
	D = dim(Ωₕ)

	_backward_finite_differencex!(v.values, u.values, Base.Fix1(spacing, Ωₕ(1)), npoints(Ωₕ, Tuple))
	s = inner₊(v, v, Val(1))

	if D >= 2
		_backward_finite_differencey!(v.values, u.values, Base.Fix1(spacing, Ωₕ(2)), npoints(Ωₕ, Tuple))
		s += inner₊(v, v, Val(2))

		if D == 3
			_backward_finite_differencez!(v.values, u.values, Base.Fix1(spacing, Ωₕ(3)), npoints(Ωₕ, Tuple))
			s += inner₊(v, v, Val(3))
		end
	end
	return sqrt(s)
end


@inline norm₁ₕ(u) = sqrt(normₕ(u)^2 + snorm₁ₕ(u)^2)