# inner products
## innerₕ
"""
	innerₕ(uₕ::VectorElement, vₕ::VectorElement)
	innerₕ(Uₕ::VecOrMatElem, Vₕ::VecOrMatElem)

Returns the discrete ``L^2`` inner product of the grid functions `uₕ` and `vₕ`

  - 1D case

```math
(\\textrm{u}_h, \\textrm{v}_h)_h \\vcentcolon = \\sum_{i=1}^N |\\square_{i}| \\textrm{u}_h(x_i) \\textrm{v}_h(x_i)
```

  - 2D case

```math
(\\textrm{u}_h, \\textrm{v}_h)_h \\vcentcolon = \\sum_{i=1}^{N_x} \\sum_{j=1}^{N_y} |\\square_{i,j}| \\textrm{u}_h(x_i,y_j) \\textrm{v}_h(x_i,y_j)
```

  - 3D case

```math
(\\textrm{u}_h, \\textrm{v}_h)_h \\vcentcolon = \\sum_{i=1}^{N_x} \\sum_{j=1}^{N_y}  \\sum_{l=1}^{N_z}  |\\square_{i,j,l}| \\textrm{u}_h(x_i,y_j) \\textrm{v}_h(x_i,y_j)
```
"""
@inline innerₕ(uₕ::VectorElement, vₕ::VectorElement) = _dot(uₕ.values, innerh_weights(space(uₕ)), vₕ.values)
@inline innerₕ(Uₕ::VecOrMatElem, Vₕ::VecOrMatElem) = _inner_product(Uₕ.values, innerh_weights(space(Uₕ)), Vₕ.values)

"""
	innerh_weights(Wₕ::SpaceType)

Returns the weights to be used in the calculation of [innerₕ](@ref innerₕ(uₕ::VectorElement, vₕ::VectorElement)).
"""
@inline innerh_weights(Wₕ::SpaceType) = Wₕ.innerh_weights

"""
	normₕ(uₕ::VectorElement)

Returns the discrete ``L^2`` norm of the grid function `uₕ`, defined as

```math
\\Vert \\textrm{u}_h \\Vert_h \\vcentcolon = \\sqrt{(\\textrm{u}_h, \\textrm{u}_h)_h}
```
"""
@inline normₕ(uₕ::VectorElement) = sqrt(innerₕ(uₕ, uₕ))

"""
	inner₊(uₕ::VecOrMatElem, vₕ::VecOrMatElem)
	inner₊(uₕ::VecOrMatElem, vₕ::VecOrMatElem, Tuple)
	inner₊(uₕ::NTuple, vₕ::NTuple)

Returns the discrete modified ``L^2`` inner product of the grid functions `uₕ` and `vₕ`. It accepts arguments of type [VectorElement](@ref) or [MatrixElement](@ref), in any order.

If the `Tuple` argument is given, it returns `D`-tuple of all ``\\textrm{inner}_{x_i,+}`` applied to its input arguments, where `D` is the topological dimension of the mesh associated with the elements.

If `NTuple`s of [VectorElement](@ref) or [MatrixElement](@ref) are passed as input arguments, it returns the sum of all inner products ``(\\textrm{u}_h[i],\\textrm{v}_h[i])_{+x_i}``.

For [VectorElement](@ref)s, the definition is given by

  - 1D case

```math
(\\textrm{u}_h, \\textrm{v}_h)_+ \\vcentcolon = \\sum_{i=1}^{N_x} h_{i} \\textrm{u}_h(x_i) \\textrm{v}_h(x_i)
```

  - 2D case

```math
(\\textrm{u}_h, \\textrm{v}_h)_+ \\vcentcolon = (\\textrm{u}_h, \\textrm{v}_h)_{+x} + (\\textrm{u}_h, \\textrm{v}_h)_{+y}
```

  - 3D case

```math
(\\textrm{u}_h, \\textrm{v}_h)_+ \\vcentcolon = (\\textrm{u}_h, \\textrm{v}_h)_{+x} + (\\textrm{u}_h, \\textrm{v}_h)_{+y} + (\\textrm{u}_h, \\textrm{v}_h)_{+z}.
```

See the definitions of [inner₊ₓ](@ref inner₊ₓ(uₕ::VecOrMatElem, vₕ::VecOrMatElem)), [inner₊ᵧ](@ref inner₊ᵧ(uₕ::VecOrMatElem, vₕ::VecOrMatElem)) and [inner₊₂](@ref inner₊₂(uₕ::VecOrMatElem, vₕ::VecOrMatElem)) for more details.
"""
@inline @generated function inner₊(uₕ::VectorElement{SType}, vₕ::VectorElement{SType}) where SType
	D = dim(mesh(SType))
	res = :(_inner_product(uₕ.values, innerplus_weights(space(uₕ), Val(1)), vₕ.values))

	for i in 2:D
		res = :(_inner_product(uₕ.values, innerplus_weights(space(uₕ), Val($i)), vₕ.values) + $res)
	end

	return res
end

@inline @generated function inner₊(Uₕ::MatrixElement{SType}, vₕ::VectorElement{SType}) where SType
	D = dim(mesh(SType))
	res = :(x = _inner_product(Uₕ.values, innerplus_weights(space(Uₕ), Val(1)), vₕ.values))

	for i in 2:D
		push!(res.args, :(x .+= _inner_product_add!(x, Uₕ.values, innerplus_weights(space(Uₕ), Val($i)), vₕ.values)))
	end

	return res
end

@inline @generated function inner₊(uₕ::VectorElement{SType}, Vₕ::MatrixElement{SType}) where SType
	D = dim(mesh(SType))
	res = :(x = _inner_product(uₕ.values, innerplus_weights(space(uₕ), Val(1)), vₕ.values))

	for i in 2:D
		push!(res.args, :(x .+= _inner_product_add!(x, uₕ.values, innerplus_weights(space(uₕ), Val($i)), Vₕ.values)))
	end

	return res
end

@inline @generated function inner₊(uₕ::MatrixElement{SType}, vₕ::MatrixElement{SType}) where SType
	D = dim(mesh(SType))
	res = :(_inner_product(uₕ.values, innerplus_weights(space(uₕ), Val(1)), vₕ.values))

	for i in 2:D
		res = :($res + _inner_product_add!(x, uₕ.values, innerplus_weights(space(uₕ), Val($i)), vₕ.values))
	end

	return res
end

@inline @generated function inner₊(uₕ::VecOrMatElem{SType}, vₕ::VecOrMatElem{SType}, ::Type{Tuple}) where SType
	D = dim(mesh(SType))
	return :(Base.Cartesian.@ntuple $D i->_inner_product(uₕ.values, innerplus_weights(space(uₕ), Val(i)), vₕ.values))
end

@inline @generated inner₊(uₕ::NTuple{D,VecOrMatElem}, vₕ::NTuple{D,VecOrMatElem}) where D = :(sum(inner₊(uₕ, vₕ, Tuple)))

@inline @generated function inner₊(uₕ::NTuple{D,VecOrMatElem}, vₕ::NTuple{D,VecOrMatElem}, ::Type{Tuple}) where D
	return :(Base.Cartesian.@ntuple $D i->_inner_product(uₕ[i].values, innerplus_weights(space(uₕ[i]), Val(i)), vₕ[i].values))
end

"""
	innerplus_weights(Wₕ::SpaceType, ::Val{D})

Returns the weights to be used in the calculation of [`inner₊`](@ref).
"""
@inline innerplus_weights(Wₕ::SpaceType, ::Val{D}) where D = (@assert 1 <= D <= dim(mesh(Wₕ));
															  Wₕ.innerplus_weights[D])

"""
	inner₊ₓ(uₕ::VecOrMatElem, vₕ::VecOrMatElem)

Returns the discrete modified ``L^2`` inner product of the grid functions `uₕ` and `vₕ` associated with the first variable. It accepts arguments of type [VectorElement](@ref) or [MatrixElement](@ref), in any order.

For [VectorElement](@ref)s, it is defined as

  - 1D case

```math
(\\textrm{u}_h, \\textrm{v}_h)_+ \\vcentcolon = \\sum_{i=1}^{N_x} h_{i} \\textrm{u}_h(x_i) \\textrm{v}_h(x_i)
```

  - 2D case

```math
(\\textrm{u}_h, \\textrm{v}_h)_{+x} \\vcentcolon = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y}  h_{x,i} h_{y,j+1/2}  \\textrm{u}_h(x_i,y_j) \\textrm{v}_h(x_i,y_j)
```

  - 3D case

```math
(\\textrm{u}_h, \\textrm{v}_h)_{+x} \\vcentcolon = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y}\\sum_{l=1}^{N_z}   h_{x,i} h_{y,j+1/2} h_{z,l+1/2}  \\textrm{u}_h(x_i,y_j,z_l) \\textrm{v}_h(x_i,y_j,z_l).
```
"""
@inline inner₊ₓ(uₕ::VecOrMatElem, vₕ::VecOrMatElem) = _inner_product(uₕ.values, innerplus_weights(space(uₕ), Val(1)), vₕ.values)#inner₊(uₕ, vₕ, Val(1))

"""
	inner₊ᵧ(uₕ::VecOrMatElem, vₕ::VecOrMatElem)

Returns the discrete modified ``L^2`` inner product of the grid functions `uₕ` and `vₕ` associated with the second variable. It accepts

  - 2D case

```math
(\\textrm{u}_h, \\textrm{v}_h)_{+y} \\vcentcolon = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y}   h_{x,i} h_{y,j+1/2}   \\textrm{u}_h(x_i,y_j) \\textrm{v}_h(x_i,y_j)
```

  - 3D case

```math
(\\textrm{u}_h, \\textrm{v}_h)_{+y} \\vcentcolon = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y}\\sum_{l=1}^{N_z}   h_{x,i+1/2} h_{y,j} h_{z,l+1/2} \\textrm{u}_h(x_i,y_j,z_l) \\textrm{v}_h(x_i,y_j,z_l).
```
"""
@inline inner₊ᵧ(uₕ::VecOrMatElem, vₕ::VecOrMatElem) = _inner_product(uₕ.values, innerplus_weights(space(uₕ), Val(2)), vₕ.values)#inner₊(uₕ, vₕ, Val(2))

"""
	inner₊₂(uₕ::VecOrMatElem, vₕ::VecOrMatElem)

Returns the discrete modified ``L^2`` inner product of the grid functions `uₕ` and `vₕ` associated with the `z` variable

```math
(\\textrm{u}_h, \\textrm{v}_h)_{+z} \\vcentcolon = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y}\\sum_{l=1}^{N_z}  h_{x,i+1/2} h_{y,j+1/2} h_{z,l} \\textrm{u}_h(x_i,y_j,z_l) \\textrm{v}_h(x_i,y_j,z_l).
```
"""
@inline inner₊₂(uₕ::VecOrMatElem, vₕ::VecOrMatElem) = _inner_product(uₕ.values, innerplus_weights(space(uₕ), Val(3)), vₕ.values)#inner₊(uₕ, vₕ, Val(3))

"""
	norm₊(uₕ::VectorElement)
	norm₊(uₕ::NTuple{D,VectorElement})

Returns the discrete modified ``L^2`` norm of the grid function `uₕ`. It also accepts a `NTuple` of [VectorElement](@ref)s.

For [VectorElement](@ref)s `uₕ`, it is defined as

```math
\\Vert \\textrm{u}_h \\Vert_+ = \\sqrt{(\\textrm{u}_h,\\textrm{u}_h)_+}.
```

and for `NTuple`s of [VectorElement](@ref)s it returns

```math
\\Vert \\textrm{u}_h \\Vert_+ \\vcentcolon = \\sqrt{ \\sum_{i=1}^D(\\textrm{u}_h[i],\\textrm{u}_h[i])_{+,x_i}}.
```
"""
@inline norm₊(uₕ::VectorElement) = sqrt(inner₊(uₕ, uₕ))
@inline norm₊(uₕ::NTuple{D,VectorElement}) where D = sqrt(inner₊(uₕ, uₕ))

"""
	norm₁ₕ(uₕ::VectorElement)

Returns the discrete version of the standard ``H^1`` norm of [VectorElement](@ref) `uₕ`.

```math
\\Vert \\textrm{u}_h \\Vert_{1h} \\vcentcolon = \\sqrt{ \\Vert \\textrm{u}_h \\Vert_h^2 +  \\Vert \\nabla_h \\textrm{u}_h \\Vert_h^2   }
```
"""
@inline norm₁ₕ(uₕ::VectorElement) = sqrt(normₕ(uₕ)^2 + snorm₁ₕ(uₕ)^2)

"""
	snorm₁ₕ(uₕ::VectorElement)

Returns the discrete version of the standard ``H^1`` seminorm of [VectorElement](@ref) `uₕ`.

```math
|\\textrm{u}_h|_{1h} \\vcentcolon = \\Vert \\nabla_h \\textrm{u}_h \\Vert_h
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