# inner products
## innerₕ
"""
	innerₕ(uₕ::VectorElement, vₕ::VectorElement)

Returns the discrete ``L^2`` inner product of the grid functions `uₕ` and `vₕ`

  - 1D case

```math
\\textrm{inner}_h (u_h, v_h) = \\sum_{i=1}^N h_{i+1/2} u_h(x_i) v_h(x_i)
```

  - 2D case

```math
\\textrm{inner}_h (u_h, v_h) = \\sum_{i=1}^{N_x} \\sum_{j=1}^{N_y} h_{x,i+1/2} h_{y,j+1/2} u_h(x_i,y_j) v_h(x_i,y_j)
```

  - 3D case

```math
\\textrm{inner}_h (u_h, v_h) = \\sum_{i=1}^{N_x} \\sum_{j=1}^{N_y}  \\sum_{l=1}^{N_z}  h_{x,i+1/2} h_{y,j+1/2} h_{z,l+1/2} u_h(x_i,y_j) v_h(x_i,y_j)
```
"""
@inline innerₕ(uₕ::VectorElement, vₕ::VectorElement) = _dot(uₕ.values, innerh_weights(space(uₕ)), vₕ.values)
@inline innerₕ(Uₕ::VecOrMatElem, Vₕ::VecOrMatElem) = _inner_product(Uₕ.values, innerh_weights(space(Uₕ)), Vₕ.values)

"""
	innerh_weights(Wₕ::SpaceType)

Returns the weights to be used in the calculation of [`innerₕ`](@ref).
"""
@inline innerh_weights(Wₕ::SpaceType) = Wₕ.innerh_weights

"""
	normₕ(uₕ::VectorElement)

	Returns the discrete ``L^2`` norm of the grid function `uₕ`, defined as

```math
\\textrm{norm}_h (u_h) = \\sqrt{\\textrm{inner}_h (u_h, u_h)}
```
"""
@inline normₕ(uₕ::VectorElement) = sqrt(innerₕ(uₕ, uₕ))

"""
	inner₊(uₕ::VectorElement, vₕ::VectorElement)

Returns the discrete modified ``L^2`` inner product of the grid functions `uₕ` and `vₕ`

  - 1D case

```math
(u_h, v_h)_+ = \\sum_{i=1}^{N_x} h_{i} u_h(x_i) v_h(x_i)
```

  - 2D case

```math
(u_h, v_h)_+ = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y} \\left( h_{x,i} h_{y,j+1/2} +  h_{x,i} h_{y,j+1/2}  \\right) u_h(x_i,y_j) v_h(x_i,y_j)
```

  - 3D case

```math
(u_h, v_h)_+ = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y}\\sum_{l=1}^{N_z} \\left( h_{x,i} h_{y,j+1/2} h_{z,l+1/2} +  h_{x,i+1/2} h_{y,j} h_{z,l+1/2} +  h_{x,i+1/2} h_{y,j+1/2} h_{z,l}\\right) u_h(x_i,y_j,z_l) v_h(x_i,y_j,z_l).
```
"""
@inline @generated function inner₊(uₕ::VectorElement{SType}, vₕ::VectorElement{SType}) where SType
	D = dim(mesh(SType))
	res = :(_inner_product(uₕ.values, innerplus_weights(space(uₕ), Val(1)), vₕ.values))

	for i in 2:D
		res = :(_inner_product(uₕ.values, innerplus_weights(space(uₕ), Val($i)), vₕ.values) + $res)
	end

	return res
end

@inline @generated function inner₊(uₕ::VecOrMatElem{SType}, vₕ::VecOrMatElem{SType}) where SType
	D = dim(mesh(SType))
	res = :(x = _inner_product(uₕ.values, innerplus_weights(space(uₕ), Val(1)), vₕ.values))

	for i in 2:D
		push!(res.args, :(x .+= _inner_product_add!(x, uₕ.values, innerplus_weights(space(uₕ), Val($i)), vₕ.values)))
	end

	return res
end

"""
	inner₊(uₕ::VecOrMatElem, vₕ::VecOrMatElem, ::Type{Tuple})

Returns a `D`-tuple of the ``\\textrm{inner}_{x_i,+}`` applied to `uₕ` and `vₕ`, where `D` is the topological dimension of the mesh associated with the elements.
"""
@inline @generated function inner₊(uₕ::VecOrMatElem{SType}, vₕ::VecOrMatElem{SType}, ::Type{Tuple}) where SType
	D = dim(mesh(SType))
	return :(Base.Cartesian.@ntuple $D i->_inner_product(uₕ.values, innerplus_weights(space(uₕ), Val(i)), vₕ.values))
end

"""
	inner₊(uₕ::NTuple{D,VecOrMatElem}, vₕ::NTuple{D,VecOrMatElem})

Returns the sum of the inner products ``\\textrm{inner}_+(u_h[i],v_h[i])``

```math
\\sum_{i=1}^D \\textrm{inner}_+(u_h[i],v_h[i])
```

where `D` is the topological dimension of the mesh associated with the elements.
"""
@inline @generated function inner₊(uₕ::NTuple{D,VecOrMatElem}, vₕ::NTuple{D,VecOrMatElem}) where D
	res = :(x = _inner_product(uₕ[1].values, innerplus_weights(space(uₕ[1]), Val(1)), vₕ[1].values))

	for i in 2:D
		push!(res.args, :(x .+= _inner_product(uₕ[$i].values, innerplus_weights(space(uₕ[$i]), Val($i)), vₕ[$i].values)))
	end
	return res
end

"""
	inner₊(uₕ::NTuple{D,VecOrMatElem}, vₕ::NTuple{D,VecOrMatElem})

Returns a tuple with the inner products ``\\textrm{inner}_+(u_h[i],v_h[i])``
"""
@inline @generated function inner₊(uₕ::NTuple{D,VecOrMatElem}, vₕ::NTuple{D,VecOrMatElem}, ::Type{Tuple}) where D
	return :(Base.Cartesian.@ntuple $D i->_inner_product(uₕ[i].values, innerplus_weights(space(uₕ[i]), Val(i)), vₕ[i].values))
end

#@inline inner₊(Uₕ::VecOrMatElem, Vₕ::VecOrMatElem) = _inner_product(Uₕ.values, innerplus_weights(space(Uₕ), Val(dim(mesh(space(Uₕ))))), Vₕ.values)

"""
	innerplus_weights(Wₕ::SpaceType, ::Val{D})

Returns the weights to be used in the calculation of [`inner₊`](@ref).
"""
@inline innerplus_weights(Wₕ::SpaceType, ::Val{D}) where D = (@assert 1 <= D <= dim(mesh(Wₕ));
															  Wₕ.innerplus_weights[D])

"""
	inner₊ₓ(uₕ::VecOrMatElem, vₕ::VecOrMatElem)

Returns the discrete modified ``L^2`` inner product of the grid functions `uₕ` and `vₕ` associated with the first variable

  - 1D case

```math
(u_h, v_h)_+ = \\sum_{i=1}^{N_x} h_{i} u_h(x_i) v_h(x_i)
```

  - 2D case

```math
(u_h, v_h)_{+x} = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y}  h_{x,i} h_{y,j+1/2}  u_h(x_i,y_j) v_h(x_i,y_j)
```

  - 3D case

```math
(u_h, v_h)_{+x} = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y}\\sum_{l=1}^{N_z}   h_{x,i} h_{y,j+1/2} h_{z,l+1/2}  u_h(x_i,y_j,z_l) v_h(x_i,y_j,z_l).
```
"""
@inline inner₊ₓ(uₕ::VecOrMatElem, vₕ::VecOrMatElem) = _inner_product(uₕ.values, innerplus_weights(space(uₕ), Val(1)), vₕ.values)#inner₊(uₕ, vₕ, Val(1))

"""
	inner₊ᵧ(uₕ::VectorElement, vₕ::VectorElement)

Returns the discrete modified ``L^2`` inner product of the grid functions `uₕ` and `vₕ` associated with the second variable

  - 2D case

```math
(u_h, v_h)_{x+} = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y}   h_{x,i} h_{y,j+1/2}   u_h(x_i,y_j) v_h(x_i,y_j)
```

  - 3D case

```math
(u_h, v_h)_{y+} = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y}\\sum_{l=1}^{N_z}   h_{x,i+1/2} h_{y,j} h_{z,l+1/2} u_h(x_i,y_j,z_l) v_h(x_i,y_j,z_l).
```
"""
@inline inner₊ᵧ(uₕ::VecOrMatElem, vₕ::VecOrMatElem) = _inner_product(uₕ.values, innerplus_weights(space(uₕ), Val(2)), vₕ.values)#inner₊(uₕ, vₕ, Val(2))

"""
	inner₊₂(uₕ::VectorElement, vₕ::VectorElement)

Returns the discrete modified ``L^2`` inner product of the grid functions `uₕ` and `vₕ` associated with the `z` variable

```math
(u_h, v_h)_{z+} = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y}\\sum_{l=1}^{N_z}  h_{x,i+1/2} h_{y,j+1/2} h_{z,l} u_h(x_i,y_j,z_l) v_h(x_i,y_j,z_l).
```
"""
@inline inner₊₂(uₕ::VecOrMatElem, vₕ::VecOrMatElem) = _inner_product(uₕ.values, innerplus_weights(space(uₕ), Val(3)), vₕ.values)#inner₊(uₕ, vₕ, Val(3))

"""
	norm₊(uₕ::VectorElement)

	Returns the discrete modified ``L^2`` norm of the grid function `uₕ`, defined as

```math
\\textrm{norm}_+ (u_h) = \\sqrt{\\textrm{inner}_+ (u_h,u_h)}.
```
"""
@inline norm₊(uₕ::VectorElement) = sqrt(inner₊(uₕ, uₕ))

"""
	norm₊(uₕ::NTuple{D,VectorElement})

	Returns the discrete modified ``L^2`` norm of a tuple of grid functions `uₕ`, defined as

```math
\\textrm{norm}_+ (u_h) = \\sqrt{ \\sum_{i=1}^D \\textrm{norm}_+ (u_h[i],u_h[i])}.
```
"""
@inline norm₊(uₕ::NTuple{D,VectorElement}) where D = sqrt(inner₊(uₕ, uₕ))

"""
	norm₁ₕ(uₕ::VectorElement)

Returns the discrete version of the standard ``H^1`` norm

```math
\\textrm{norm}_{1h}(u_h) = \\sqrt{ \\Vert u_h \\Vert_h^2 +  \\Vert \\nabla_h u_h \\Vert_h^2   }
```
"""
@inline norm₁ₕ(uₕ::VectorElement) = sqrt(normₕ(uₕ)^2 + snorm₁ₕ(uₕ)^2)

"""
	norm₁ₕ(uₕ::VectorElement)

Returns the discrete version of the standard ``H^1`` seminorm
```math
\\textrm{snorm}_{1h}(u_h) = \\Vert \\nabla_h u_h \\Vert_h^2
```
"""
function snorm₁ₕ(uₕ::VectorElement)
	vₕ = similar(uₕ)
	Ωₕ = mesh(space(uₕ))
	D = dim(Ωₕ)
	npts = npoints(Ωₕ, Tuple)
	weights(i) = Base.Fix1(spacing, Ωₕ(i))

	u = uₕ.values
	v = vₕ.values

	_backward_finite_differencex!(v, u, weights(1), npts)
	s = inner₊ₓ(vₕ, vₕ)

	if D >= 2
		_backward_finite_differencey!(v, u, weights(2), npts)
		s += inner₊ᵧ(vₕ, vₕ)

		if D == 3
			_backward_finite_differencez!(v, u, weights(3), npts)
			s += inner₊₂(vₕ, vₕ)
		end
	end
	return sqrt(s)
end

# some helper functions to calculate the inner products in discrete spaces
@inline function _dot(u::Vector{T}, v::Vector{T}, w::Vector{T}) where T
	s = zero(T)

	@simd for i in eachindex(u, v, w)
		s += u[i] * v[i] * w[i]
	end

	return s
end

@inline _inner_product(u::Vector{T}, h::Vector{T}, v::Vector{T}) where T = _dot(u, h, v)

@inline function _inner_product_add!(z::Vector{T}, u::Vector{T}, v::Vector{T}, w::Vector{T}) where T
	@simd for i in eachindex(z, u, v, w)
		z[i] += u[i] * v[i] * w[i]
	end
end

@inline function _inner_product(u::Vector{T}, h::Vector{T}, A::SparseMatrixCSC{T,Int}) where T
	z = zeros(T, size(u))
	_inner_product_add!(z, u, h, A)

	return z
end

@inline function _inner_product_add!(z, u::Vector{T}, h::Vector{T}, A::SparseMatrixCSC{T,Int}) where T
	#z = transpose(A) * Diagonal(h) * u
	@simd for j in eachindex(h, u)
		for idx in A.colptr[j]:(A.colptr[j + 1] - 1)
			i = A.rowval[idx]
			val = A.nzval[idx]

			z[j] += val * h[i] * u[i]
		end
	end
end

@inline function _inner_product(A::SparseMatrixCSC{T,Int}, h::Vector{T}, v::Vector{T}) where T
	z = transpose(zeros(T, size(v)))
	_inner_product_add!(z, A, h, v)

	return z
end

@inline function _inner_product_add!(z, A::SparseMatrixCSC{T,Int}, h::Vector{T}, v::Vector{T}) where T
	#z = transpose(v) * Diagonal(h) * A
	@simd for j in eachindex(h, v)
		for idx in A.colptr[j]:(A.colptr[j + 1] - 1)
			i = A.rowval[idx]
			val = A.nzval[idx]

			z[j] += h[i] * v[i] * val
		end
	end
end

@inline function _inner_product_add!(Z::SparseMatrixCSC{T,Int}, U::SparseMatrixCSC{T,Int}, h::Vector{T}, V::SparseMatrixCSC{T,Int}) where T
	Z .+= transpose(V) * Diagonal(h) * U
end

@inline _inner_product(u, h, v) = transpose(v) * Diagonal(h) * u

# implementation of innerh 
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

##########################################
# implementation of innerplus and normplus 

#@inline inner₊(uₕ::VectorElement, vₕ::VectorElement, i::Int) = _inner_product(uₕ.values, innerplus_weights(space(uₕ), Val(i)), vₕ.values)::eltype(uₕ)
#@inline inner₊(uₕ::VectorElement, Vₕ::MatrixElement, i::Int) = _inner_product(uₕ.values, innerplus_weights(space(uₕ), Val(i)), Vₕ.values)
#@inline inner₊(Uₕ::MatrixElement, vₕ::VectorElement, i::Int) = _inner_product(Uₕ.values, innerplus_weights(space(Uₕ), Val(i)), vₕ.values)

#=
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
=#

# implementation of snorm₁ₕ and norm₁ₕ
#=
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
=#