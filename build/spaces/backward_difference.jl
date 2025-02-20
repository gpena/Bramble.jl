##############################################################################
#                                                                            #
#           Implementation of the backward difference operators              #
#                                                                            #
##############################################################################

# Implementation of the backward difference operators as matrices
"""
	diff₋ₓ(Wₕ::SpaceType)
	diff₋ₓ(Ωₕ::MeshType)

Returns a [MatrixElement](@ref) implementing the backward difference matrix for the mesh grid of `Wₕ`, in the `x` direction. It is defined as being the (sparse) matrix representation of the linear operator defined by [diff₋ₓ](@ref diff₋ₓ(uₕ::VectorElement)). It can also be calculated passing a mesh as argument.
"""
@inline diff₋ₓ(Wₕ::SpaceType) = elements(Wₕ, diff₋ₓ(mesh(Wₕ)))
@inline diff₋ₓ(Ωₕ::MeshType) = shiftₓ(Ωₕ, Val(dim(Ωₕ)), Val(0)) - shiftₓ(Ωₕ, Val(dim(Ωₕ)), Val(-1))

"""
	diff₋ᵧ(Wₕ::SpaceType)
	diff₋ᵧ(Ωₕ::MeshType)

Returns a [MatrixElement](@ref) implementing the backward difference matrix for the mesh grid of `Wₕ`, in the `y` direction. It is defined as being the (sparse) matrix representation of the linear operator defined by [diff₋ᵧ](@ref diff₋ᵧ(uₕ::VectorElement)). It can also be calculated passing a mesh as argument.
"""
@inline diff₋ᵧ(Wₕ::SpaceType) = elements(Wₕ, diff₋ᵧ(mesh(Wₕ)))
@inline diff₋ᵧ(Ωₕ::MeshType) = shiftᵧ(Ωₕ, Val(dim(Ωₕ)), Val(0)) - shiftᵧ(Ωₕ, Val(dim(Ωₕ)), Val(-1))

"""
	diff₋₂(Wₕ::SpaceType)
	diff₋₂(Ωₕ::MeshType)

Returns a [MatrixElement](@ref) implementing the backward difference matrix for the mesh grid of `Wₕ`, in the `z` direction. It is defined as being the (sparse) matrix representation of the linear operator defined by [diff₋₂](@ref diff₋₂(uₕ::VectorElement)). It can also be calculated passing a mesh as argument.
"""
@inline diff₋₂(Wₕ::SpaceType) = elements(Wₕ, diff₋₂(mesh(Wₕ)))
@inline diff₋₂(Ωₕ::MeshType) = shift₂(Ωₕ, Val(dim(Ωₕ)), Val(0)) - shift₂(Ωₕ, Val(dim(Ωₕ)), Val(-1))

@inline diff₋ᵢ(Wₕ::SpaceType, ::Val{1}) = diff₋ₓ(Wₕ)
@inline diff₋ᵢ(Wₕ::SpaceType, ::Val{2}) = diff₋ᵧ(Wₕ)
@inline diff₋ᵢ(Wₕ::SpaceType, ::Val{3}) = diff₋₂(Wₕ)

"""
	diff₋ₕ(Wₕ::SpaceType)

Returns a tuple of [MatrixElement](@ref)s implementing the backward difference operators in the `x`, `y`, and `z` directions. If the problem is `1`-dimensikonal, it returns a single [MatrixElement](@ref).
"""
@inline diff₋ₕ(Wₕ::SpaceType) = diff₋ₕ(Wₕ, Val(dim(mesh(Wₕ))))
@inline diff₋ₕ(Wₕ::SpaceType, ::Val{1}) = diff₋ₓ(Wₕ)
@inline diff₋ₕ(Wₕ::SpaceType, ::Val{D}) where D = ntuple(i -> diff₋ᵢ(Wₕ, Val(i)), D)

# Implementation of the backward difference operators acting on vectors
"""
	diff₋ₓ(uₕ::VectorElement)

Returns the backward difference, in the `x` direction, of the element `uₕ`.

  - 1D case

```math
\\textrm{diff}_{-x} \\textrm{u}_h(x_i) \\vcentcolon = \\textrm{u}_h(x_i) - \\textrm{u}_h(x_{i-1})
```

  - 2D and 3D case

```math
\\textrm{diff}_{-x} \\textrm{u}_h(x_i, \\dots) \\vcentcolon = \\textrm{u}_h(x_i, \\dots)-\\textrm{u}_h(x_{i-1}, \\dots)
```
"""
Base.@propagate_inbounds function diff₋ₓ(uₕ::VectorElement)
	vₕ = similar(uₕ)
	dims = npoints(mesh(space(uₕ)), Tuple)

	_backward_differencex!(vₕ.values, uₕ.values, dims)
	return vₕ
end

"""
	diff₋ᵧ(uₕ::VectorElement)

Returns the backward difference, in the `y` direction, of the element `uₕ`.

  - 2D and 3D case

```math
\\textrm{diff}_{-y} \\textrm{u}_h(x_i, y_j,\\dots) \\vcentcolon = \\textrm{u}_h(x_i, y_j,\\dots)-\\textrm{u}_h(x_i, y_{j-1}, \\dots)
```
"""
Base.@propagate_inbounds function diff₋ᵧ(uₕ::VectorElement)
	Ωₕ = mesh(space((uₕ)))
	D = dim(Ωₕ)
	vₕ = similar(uₕ)
	dims = npoints(Ωₕ, Tuple)

	if D === 1
		@error "no backward difference on y variable in 1D"
	end

	_backward_differencey!(vₕ.values, uₕ.values, dims)
	return vₕ
end

"""
	diff₋₂(uₕ::VectorElement)

Returns the backward difference, in the `z` direction, of the element `uₕ`.

```math
\\textrm{diff}_{-z} \\textrm{u}_h(x_i, y_j,z_l) \\vcentcolon = \\textrm{u}_h(x_i, y_j,z_l)-\\textrm{u}_h(x_i, y_j, z_{l-1})
```
"""
Base.@propagate_inbounds function diff₋₂(uₕ::VectorElement)
	Ωₕ = mesh(space(uₕ))
	D = dim(Ωₕ)
	vₕ = similar(uₕ)
	dims = npoints(Ωₕ, Tuple)

	if D === 1 || D === 2
		@error "No backward difference in variable z in 1D ou 2D"
	end

	_backward_differencez!(vₕ.values, uₕ.values, dims)

	return vₕ
end

@inline diff₋ᵢ(uₕ::VectorElement, ::Val{1}) = diff₋ₓ(uₕ)
@inline diff₋ᵢ(uₕ::VectorElement, ::Val{2}) = diff₋ᵧ(uₕ)
@inline diff₋ᵢ(uₕ::VectorElement, ::Val{3}) = diff₋₂(uₕ)

"""
	diff₋ₕ(uₕ::VectorElement)

Returns a tuple of [VectorElement](@ref)s implementing the backward difference operators in the `x`, `y`, and `z` directions applied to `uₕ`. If the problem is `1`-dimensional, it returns a single [VectorElement](@ref).
"""
@inline diff₋ₕ(uₕ::VectorElement) = diff₋ₕ(uₕ, Val(dim(mesh(space(uₕ)))))
@inline diff₋ₕ(uₕ::VectorElement, ::Val{1}) = diff₋ₓ(uₕ)
@inline diff₋ₕ(uₕ::VectorElement, ::Val{D}) where D = ntuple(i -> diff₋ᵢ(uₕ, Val(i)), D)

# Implementation of the backward difference operators for matrix elements
"""
	diff₋ₓ(Uₕ::MatrixElement)

Returns a [MatrixElement](@ref) resulting of the multiplication of the backward difference matrix, in the `x` direction, by `Uₕ`.
"""
@inline diff₋ₓ(Uₕ::MatrixElement) = diff₋ₓ(space(Uₕ)) * Uₕ

"""
	diff₋ᵧ(Uₕ::MatrixElement)

Returns a [MatrixElement](@ref) resulting of the multiplication of the backward difference matrix, in the `y` direction, by `Uₕ`.
"""
@inline diff₋ᵧ(Uₕ::MatrixElement) = diff₋ᵧ(space(Uₕ)) * Uₕ

"""
	diff₋₂(Uₕ::MatrixElement)

Returns a [MatrixElement](@ref) resulting of the multiplication of the backward difference matrix, in the `z` direction, by `Uₕ`.
"""
@inline diff₋₂(Uₕ::MatrixElement) = diff₋₂(space(Uₕ)) * Uₕ

@inline diff₋ᵢ(Uₕ::MatrixElement, ::Val{1}) = diff₋ₓ(Uₕ)
@inline diff₋ᵢ(Uₕ::MatrixElement, ::Val{2}) = diff₋ᵧ(Uₕ)
@inline diff₋ᵢ(Uₕ::MatrixElement, ::Val{3}) = diff₋₂(Uₕ)

"""
	diff₋ₕ(Uₕ::MatrixElement)

Returns a tuple of [MatrixElement](@ref)s implementing the forward difference operators in the `x`, `y`, and `z` directions applied to `Uₕ`. If the problem is `1`-dimensional, it returns a single [MatrixElement](@ref).
"""
@inline diff₋ₕ(Uₕ::MatrixElement) = diff₋ₕ(Uₕ, Val(dim(mesh(space(Uₕ)))))
@inline diff₋ₕ(Uₕ::MatrixElement, ::Val{1}) = diff₋ₓ(Uₕ)
@inline diff₋ₕ(Uₕ::MatrixElement, ::Val{D}) where D = ntuple(i -> diff₋ᵢ(Uₕ, Val(i)), D)