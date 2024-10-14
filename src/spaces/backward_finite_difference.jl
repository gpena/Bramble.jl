##############################################################################
#                                                                            #
#       Implementation of the backward finite difference operators           #
#                                                                            #
##############################################################################


@inline function _create_backward_diff_matrix(Wₕ::SpaceType, ::Val{1}; vector = _create_vector(Wₕ))
	A = _create_D₋ₓ(mesh(Wₕ), vector = vector)
	return elements(Wₕ, A)
end

@inline function _create_backward_diff_matrix(Wₕ::SpaceType, ::Val{2}; vector = _create_vector(Wₕ))
	A = _create_D₋ᵧ(mesh(Wₕ), vector = vector)
	return elements(Wₕ, A)
end

@inline function _create_backward_diff_matrix(Wₕ::SpaceType, ::Val{3}; vector = _create_vector(Wₕ))
	A = _create_D₋₂(mesh(Wₕ), vector = vector)
	return elements(Wₕ, A)
end

@inline function create_backward_diff_matrices(Wₕ::SpaceType; vector = _create_vector(Wₕ))
	return ntuple(i -> _create_backward_diff_matrix(Wₕ, Val(i), vector = vector), dim(mesh(Wₕ)))
end

"""
	D₋ₓ(Wₕ::SpaceType)
	D₋ₓ(Ωₕ::MeshType)

Returns a [MatrixElement](@ref) implementing the backward finite difference matrix for the mesh grid of `Wₕ`, in the `x` direction. It is defined as being the (sparse) matrix representation of the linear operator defined by [D₋ₓ](@ref D₋ₓ(uₕ::VectorElement)). It can also accept a mesh as an argument.
"""
@inline D₋ₓ(Wₕ::SpaceType) = Wₕ.diff_matrix_cache[1]
@inline D₋ₓ(Ωₕ::MeshType) = _create_D₋ₓ(Ωₕ)

"""
	D₋ᵧ(Wₕ::SpaceType)
	D₋ᵧ(Ωₕ::MeshType)

Returns a [MatrixElement](@ref) implementing the backward finite difference matrix for the mesh grid of `Wₕ`, in the `y` direction. It is defined as being the (sparse) matrix representation of the linear operator defined by [D₋ᵧ](@ref D₋ᵧ(uₕ::VectorElement)). It can also accept a mesh as an argument.
"""
@inline D₋ᵧ(Wₕ::SpaceType) = Wₕ.diff_matrix_cache[2]
@inline D₋ᵧ(Ωₕ::MeshType) = _create_D₋ₓ(Ωₕ)

"""
	D₋₂(Wₕ::SpaceType)
	D₋₂(Ωₕ::MeshType)

Returns a [MatrixElement](@ref) implementing the backward finite difference matrix for the mesh grid of `Wₕ`, in the `z` direction. It is defined as being the (sparse) matrix representation of the linear operator defined by [D₋₂](@ref D₋₂(uₕ::VectorElement)). It can also accept a mesh as an argument.
"""
@inline D₋₂(Wₕ::SpaceType) = Wₕ.diff_matrix_cache[3]
@inline D₋₂(Ωₕ::MeshType) = _create_D₋ₓ(Ωₕ)

@inline D₋ᵢ(Wₕ::SpaceType, ::Val{i}) where i = (@assert 1 <= i <= dim(mesh(Wₕ));
												Wₕ.diff_matrix_cache[i])

"""
	∇₋ₕ(Wₕ::SpaceType)

Returns a tuple of [MatrixElement](@ref)s implementing the backward finite difference operators in the `x`, `y`, and `z` directions. If the problem is `1`-dimensional, it returns a single [MatrixElement](@ref).
"""
@inline ∇₋ₕ(Wₕ::SpaceType) = ∇₋ₕ(Wₕ, Val(dim(mesh(Wₕ))))
@inline ∇₋ₕ(Wₕ::SpaceType, ::Val{1}) = D₋ₓ(Wₕ)
@inline ∇₋ₕ(Wₕ::SpaceType, ::Val{D}) where D = ntuple(i -> D₋ᵢ(Wₕ, Val(i)), D)

# Implementation of the backward finite difference operators for vector elements
"""
	D₋ₓ(uₕ::VectorElement)

Returns the backward finite difference, in the `x` direction, of the element `uₕ`.

  - 1D case

```math
\\textrm{D}_{-x} \\textrm{u}_h (x_i) \\vcentcolon = \\frac{\\textrm{u}_h(x_i) - \\textrm{u}_h(x_{i-1})}{h_i}
```

  - 2D and 3D case

```math
\\textrm{D}_{-x} \\textrm{u}_h (x_i, \\dots) \\vcentcolon = \\frac{\\textrm{u}_h(x_i, \\dots)-\\textrm{u}_h(x_{i-1}, \\dots)}{h_{x,i}}
```
"""
@inline D₋ₓ(uₕ::VectorElement) = D₋ᵢ(uₕ, Val(1)) 
@inline D₋ₓ!(x, uₕ::VectorElement) = D₋ₓ!(x, uₕ.values, space(uₕ))

Base.@propagate_inbounds function D₋ₓ!(x, u, S::SpaceType)
	@assert length(x) == length(u)

	h = Base.Fix1(spacing, mesh(S)(1))
	dims = npoints(mesh(S), Tuple)

	_backward_finite_differencex!(x, u, h, dims)
end

"""
	D₋ᵧ(uₕ::VectorElement)

Returns the backward finite difference, in the `y` direction, of the element `uₕ`.

```math
\\textrm{D}_{-y} \\textrm{u}_h(x_i, y_j, \\dots) \\vcentcolon = \\frac{\\textrm{u}_h(x_i, y_j, \\dots)-\\textrm{u}_h(x_i, y_{j-1}, \\dots)}{h_{y,j}}
```
"""
@inline D₋ᵧ(uₕ::VectorElement) = D₋ᵢ(uₕ, Val(2)) 

Base.@propagate_inbounds function D₋ᵧ!(x, u, S::SpaceType)
	@assert length(x) == length(u)

	Ωₕ = mesh(S)
	hy = Base.Fix1(spacing, Ωₕ(2))
	dims = npoints(Ωₕ, Tuple)

	_backward_finite_differencey!(x, u, hy, dims)
end

@inline D₋ᵧ!(x, uₕ::VectorElement) = D₋ᵧ!(x, uₕ.values, space(uₕ))

"""
	D₋₂(uₕ::VectorElement)

Returns the backward finite difference, in the `z` direction, of the element `uₕ`.

```math
\\textrm{D}_{-z} \\textrm{u}_h(x_i, y_j, z_l) \\vcentcolon = \\frac{\\textrm{u}_h(x_i, y_j, z_l)-\\textrm{u}_h(x_i, y_j, z_)}{h_{z,l}}
```
"""
@inline D₋₂(uₕ::VectorElement) = D₋ᵢ(uₕ, Val(3)) 

Base.@propagate_inbounds function D₋₂!(x, u, S::SpaceType)
	@assert length(x) == length(u)

	Ωₕ = mesh(S)
	D = dim(Ωₕ)

	if D === 1 || D === 2
		@error "No backward finite difference in variable z in 1D ou 2D"
	end

	hz = Base.Fix1(spacing, Ωₕ(3))
	dims = npoints(Ωₕ, Tuple)

	_backward_finite_differencez!(x, u, hz, dims)
end

@inline D₋₂!(x::AbstractVector, uₕ::VectorElement) = D₋₂!(x, uₕ.values, space(uₕ))
	
@inline function D₋ᵢ(uₕ::VectorElement, ::Val{1}) 
	v = similar(uₕ)
	D₋ₓ!(v.values, uₕ)
	return v
end

@inline function D₋ᵢ(uₕ::VectorElement, ::Val{2}) 
	v = similar(uₕ)
	D₋ᵧ!(v.values, uₕ)
	return v
end

@inline function D₋ᵢ(uₕ::VectorElement, ::Val{3}) 
	v = similar(uₕ)
	D₋₂!(v.values, uₕ)
	return v
end

@inline D₋ᵢ!(x, uₕ::VectorElement, ::Val{1}) = D₋ₓ!(x,uₕ)
@inline D₋ᵢ!(x, uₕ::VectorElement, ::Val{2}) = D₋ᵧ!(x,uₕ)
@inline D₋ᵢ!(x, uₕ::VectorElement, ::Val{3}) = D₋₂!(x,uₕ)
@inline D₋ᵢ!(x, u, S, ::Val{1}) = D₋ₓ!(x,u,S)
@inline D₋ᵢ!(x, u, S, ::Val{2}) = D₋ᵧ!(x,u,S)
@inline D₋ᵢ!(x, u, S, ::Val{3}) = D₋₂!(x,u,S)

"""
	∇₋ₕ(uₕ::VectorElement)

Returns a tuple of [VectorElement](@ref)s implementing the backward finite difference operators in the `x`, `y`, and `z` directions applied to `uₕ`. If the problem is `1`-dimensional, it returns a single [VectorElement](@ref).
"""
@inline ∇₋ₕ(uₕ::VectorElement) = ∇₋ₕ(uₕ, Val(dim(mesh(space(uₕ)))))
@inline ∇₋ₕ(uₕ::VectorElement, ::Val{1}) = D₋ₓ(uₕ)
@inline ∇₋ₕ(uₕ::VectorElement, ::Val{D}) where D = ntuple(i -> D₋ᵢ(uₕ, Val(i)), D)

# Backward finite difference operatos applied to matrices
"""
	D₋ₓ(Uₕ::MatrixElement)

Returns a [MatrixElement](@ref) resulting of the multiplication of the backward finite difference matrix [`D₋ₓ`](@ref) with the [MatrixElement](@ref) `Uₕ`.
"""
@inline D₋ₓ(Uₕ::MatrixElement) = D₋ₓ(space(Uₕ)) * Uₕ

"""
	D₋ᵧ(Uₕ::MatrixElement)

Returns a [MatrixElement](@ref) resulting of the multiplication of the backward finite difference matrix [`D₋ᵧ`](@ref) with the [MatrixElement](@ref) `Uₕ`.
"""
@inline D₋ᵧ(Uₕ::MatrixElement) = D₋ᵧ(space(Uₕ)) * Uₕ

"""
	D₋₂(Uₕ::MatrixElement)

Returns a [MatrixElement](@ref) resulting of the multiplication of the backward finite difference matrix [`D₋₂`](@ref) with the [MatrixElement](@ref) `Uₕ`.
"""
@inline D₋₂(Uₕ::MatrixElement) = D₋₂(space(Uₕ)) * Uₕ

@inline D₋ᵢ(Uₕ::MatrixElement, ::Val{1}) = D₋ₓ(Uₕ)
@inline D₋ᵢ(Uₕ::MatrixElement, ::Val{2}) = D₋ᵧ(Uₕ)
@inline D₋ᵢ(Uₕ::MatrixElement, ::Val{3}) = D₋₂(Uₕ)

"""
	∇₋ₕ(Uₕ::MatrixElement)

Returns a tuple of [MatrixElement](@ref)s implementing the backward finite difference operators in the `x`, `y`, and `z` directions applied to `Uₕ`. If the problem is `1`-dimensional, it returns a single [MatrixElement](@ref).
"""
@inline ∇₋ₕ(Uₕ::MatrixElement) = ∇₋ₕ(Uₕ, Val(dim(mesh(space(Uₕ)))))
@inline ∇₋ₕ(Uₕ::MatrixElement, ::Val{1}) = D₋ₓ(Uₕ)
@inline ∇₋ₕ(Uₕ::MatrixElement, ::Val{D}) where D = ntuple(i -> D₋ᵢ(Uₕ, Val(i)), D)