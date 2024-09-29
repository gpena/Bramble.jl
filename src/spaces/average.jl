###############################################################
#                                                             #
# Implementation of the average operators for vector elements #
#                                                             #
###############################################################

# Average operator as a matrix
"""
	Mₕₓ(Wₕ::SpaceType)
	Mₕₓ(Ωₕ::MeshType)

Returns a [MatrixElement](@ref) implementing the average matrix for the mesh grid of `Wₕ`, in the `x` direction. It is defined as being the (sparse) matrix representation of the linear operator defined by [Mₕₓ](@ref Mₕₓ(uₕ::VectorElement)). It also accepts a mesh as argument.
"""
@inline Mₕₓ(Wₕ::SpaceType) = elements(Wₕ, Mₕₓ(mesh(Wₕ)))
@inline Mₕₓ(Ωₕ::MeshType) = (shiftₓ(Ωₕ, Val(dim(Ωₕ)), Val(0)) + shiftₓ(Ωₕ, Val(dim(Ωₕ)), Val(-1))) * convert(eltype(Ωₕ), 0.5)::eltype(Ωₕ)

"""
	Mₕᵧ(Wₕ::SpaceType)
	Mₕᵧ(Ωₕ::MeshType)

Returns a [MatrixElement](@ref) implementing the average matrix for the mesh grid of `Wₕ`, in the `y` direction. It is defined as being the (sparse) matrix representation of the linear operator defined by [Mₕᵧ](@ref Mₕᵧ(uₕ::VectorElement)). It also accepts a mesh as argument.
"""
@inline Mₕᵧ(Wₕ::SpaceType) = elements(Wₕ, Mₕᵧ(mesh(Wₕ)))
@inline Mₕᵧ(Ωₕ::MeshType) = (shiftᵧ(Ωₕ, Val(dim(Ωₕ)), Val(0)) + shiftᵧ(Ωₕ, Val(dim(Ωₕ)), Val(-1))) * convert(eltype(Ωₕ), 0.5)::eltype(Ωₕ)

"""
	Mₕ₂(Wₕ::SpaceType)
	Mₕ₂(Ωₕ::MeshType)

Returns a [MatrixElement](@ref) implementing the average matrix for the mesh grid of `Wₕ`, in the `z` direction. It is defined as being the (sparse) matrix representation of the linear operator defined by [Mₕ₂](@ref Mₕ₂(uₕ::VectorElement)). It also accepts a mesh as argument.
"""
@inline Mₕ₂(Wₕ::SpaceType) = elements(Wₕ, Mₕ₂(mesh(Wₕ)))
@inline Mₕ₂(Ωₕ::MeshType) = (shift₂(Ωₕ, Val(dim(Ωₕ)), Val(0)) + shift₂(Ωₕ, Val(dim(Ωₕ)), Val(-1))) * convert(eltype(Ωₕ), 0.5)::eltype(Ωₕ)

@inline Mₕᵢ(Wₕ::SpaceType, ::Val{1}) = Mₕₓ(Wₕ)
@inline Mₕᵢ(Wₕ::SpaceType, ::Val{2}) = Mₕᵧ(Wₕ)
@inline Mₕᵢ(Wₕ::SpaceType, ::Val{3}) = Mₕ₂(Wₕ)

"""
	Mₕ(Wₕ::SpaceType)

Returns a tuple of [MatrixElement](@ref)s implementing the average operators in the `x`, `y`, and `z` directions.
"""
@inline Mₕ(Wₕ::SpaceType) = ntuple(i -> Mₕᵢ(Wₕ, Val(i)), dim(mesh(Wₕ)))

# Average operator applied to vectors
"""
	Mₕₓ(uₕ::VectorElement)

Returns the average, in the `x` direction, of the element `uₕ`.

  - 1D case

```math
\\textrm{M}_{hx} \\textrm{u}_h(x_i) \\vcentcolon = \\frac{\\textrm{u}_h(x_i) + \\textrm{u}_h(x_{i-1})}{2}
```

  - 2D and 3D case

```math
\\textrm{M}_{hx} \\textrm{u}_h(x_i, \\dots) \\vcentcolon = \\frac{\\textrm{u}_h(x_i, \\dots)+\\textrm{u}_h(x_{i-1}, \\dots)}{2}
```
"""
@inline Mₕₓ(uₕ::VectorElement) = element(space(uₕ), Mₕₓ(space(uₕ)).values * uₕ.values)

"""
	Mₕᵧ(uₕ::VectorElement)

Returns the Mₕ, in the `y` direction, of the element `uₕ`.

```math
\\textrm{Mₕ}_{hy} \\textrm{u}_h(x_i, y_j,\\dots) \\vcentcolon = \\textrm{u}_h(x_i, y_j\\dots)-\\textrm{u}_h(x_i, y_{j+1}, \\dots)
```
"""
@inline Mₕᵧ(uₕ::VectorElement) = element(space(uₕ), Mₕᵧ(space(uₕ)).values * uₕ.values)

"""
	Mₕ₂(uₕ::VectorElement)

Returns the average, in the `z` direction, of the element `uₕ`.

```math
\\textrm{Mₕ}_{hz} \\textrm{u}_h(x_i, y_j,z_l) \\vcentcolon = \\frac{\\textrm{u}_h(x_i, y_j, z_l)+\\textrm{u}_h(x_i, y_j, z_{l-1})}{2}
```
"""
@inline Mₕ₂(uₕ::VectorElement) = element(space(uₕ), Mₕ₂(space(uₕ)).values * uₕ.values)

@inline Mₕᵢ(uₕ::VectorElement, ::Val{1}) = Mₕₓ(uₕ)
@inline Mₕᵢ(uₕ::VectorElement, ::Val{2}) = Mₕᵧ(uₕ)
@inline Mₕᵢ(uₕ::VectorElement, ::Val{3}) = Mₕ₂(uₕ)

"""
	Mₕ(uₕ::VectorElement)

Returns a tuple of [VectorElement](@ref)s implementing the average operators in the `x`, `y`, and `z` directions applied to `uₕ`.
"""
@inline Mₕ(uₕ::VectorElement) = ntuple(i -> element(space(uₕ), Mₕᵢ(space(uₕ), Val(i)).values * uₕ.values), dim(mesh(space(uₕ))))

# Average operator applied to matrices
"""
	Mₕₓ(Uₕ::MatrixElement)

Returns a [MatrixElement](@ref) resulting of the multiplication of the average matrix [Mₕₓ(Wₕ::SpaceType)](@ref) with the [MatrixElement](@ref) `Uₕ`.
"""
@inline Mₕₓ(Uₕ::MatrixElement) = elements(space(Uₕ), Mₕₓ(space(Uₕ)).values * Uₕ.values)

"""
	Mₕᵧ(Uₕ::MatrixElement)

Returns a [MatrixElement](@ref) resulting of the multiplication of the average matrix [Mₕᵧ(Wₕ::SpaceType)](@ref) with the [MatrixElement](@ref) `Uₕ`.
"""
@inline Mₕᵧ(Uₕ::MatrixElement) = elements(space(Uₕ), Mₕᵧ(space(Uₕ)).values * Uₕ.values)

"""
	Mₕ₂(Uₕ::MatrixElement)

Returns a [MatrixElement](@ref) resulting of the multiplication of the average matrix [Mₕ₂(Wₕ::SpaceType)](@ref) with the [MatrixElement](@ref) `Uₕ`.
"""
@inline Mₕ₂(Uₕ::MatrixElement) = elements(space(Uₕ), Mₕ₂(space(Uₕ)).values * Uₕ.values)

@inline Mₕᵢ(Uₕ::MatrixElement, ::Val{1}) = Mₕₓ(Uₕ)
@inline Mₕᵢ(Uₕ::MatrixElement, ::Val{2}) = Mₕᵧ(Uₕ)
@inline Mₕᵢ(Uₕ::MatrixElement, ::Val{3}) = Mₕ₂(Uₕ)

"""
	Mₕ(Uₕ::MatrixElement)

Returns a tuple of [MatrixElement](@ref)s implementing the Mₕ operators in the `x`, `y`, and `z` directions applied to `Uₕ`. In the `1`-dimensional case, it returns just the average.
"""
@inline Mₕ(Uₕ::MatrixElement) = Mₕ(Uₕ, Val(dim(mesh(space(Uₕ)))))
@inline Mₕ(Uₕ::MatrixElement, ::Val{1}) = Mₕₓ(Uₕ)
@inline Mₕ(Uₕ::MatrixElement, ::Val{D}) where D = ntuple(i -> Mₕᵢ(Uₕ, Val(i)) * Uₕ, dim(mesh(space(Uₕ))))