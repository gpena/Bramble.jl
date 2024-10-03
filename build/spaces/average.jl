###############################################################
#                                                             #
# Implementation of the average operators for vector elements #
#                                                             #
###############################################################

# Average operator as a matrix
"""
	M₋ₕₓ(Wₕ::SpaceType)
	M₋ₕₓ(Ωₕ::MeshType)

Returns a [MatrixElement](@ref) implementing the average matrix for the mesh grid of `Wₕ`, in the `x` direction. It is defined as being the (sparse) matrix representation of the linear operator defined by [M₋ₕₓ](@ref M₋ₕₓ(uₕ::VectorElement)). It also accepts a mesh as argument.
"""
@inline M₋ₕₓ(Wₕ::SpaceType) = elements(Wₕ, M₋ₕₓ(mesh(Wₕ)))
@inline M₋ₕₓ(Ωₕ::MeshType) = (shiftₓ(Ωₕ, Val(dim(Ωₕ)), Val(0)) + shiftₓ(Ωₕ, Val(dim(Ωₕ)), Val(-1))) * convert(eltype(Ωₕ), 0.5)::eltype(Ωₕ)

"""
	M₋ₕᵧ(Wₕ::SpaceType)
	M₋ₕᵧ(Ωₕ::MeshType)

Returns a [MatrixElement](@ref) implementing the average matrix for the mesh grid of `Wₕ`, in the `y` direction. It is defined as being the (sparse) matrix representation of the linear operator defined by [M₋ₕᵧ](@ref M₋ₕᵧ(uₕ::VectorElement)). It also accepts a mesh as argument.
"""
@inline M₋ₕᵧ(Wₕ::SpaceType) = elements(Wₕ, M₋ₕᵧ(mesh(Wₕ)))
@inline M₋ₕᵧ(Ωₕ::MeshType) = (shiftᵧ(Ωₕ, Val(dim(Ωₕ)), Val(0)) + shiftᵧ(Ωₕ, Val(dim(Ωₕ)), Val(-1))) * convert(eltype(Ωₕ), 0.5)::eltype(Ωₕ)

"""
	M₋ₕ₂(Wₕ::SpaceType)
	M₋ₕ₂(Ωₕ::MeshType)

Returns a [MatrixElement](@ref) implementing the average matrix for the mesh grid of `Wₕ`, in the `z` direction. It is defined as being the (sparse) matrix representation of the linear operator defined by [M₋ₕ₂](@ref M₋ₕ₂(uₕ::VectorElement)). It also accepts a mesh as argument.
"""
@inline M₋ₕ₂(Wₕ::SpaceType) = elements(Wₕ, M₋ₕ₂(mesh(Wₕ)))
@inline M₋ₕ₂(Ωₕ::MeshType) = (shift₂(Ωₕ, Val(dim(Ωₕ)), Val(0)) + shift₂(Ωₕ, Val(dim(Ωₕ)), Val(-1))) * convert(eltype(Ωₕ), 0.5)::eltype(Ωₕ)

@inline M₋ₕᵢ(Wₕ::SpaceType, ::Val{1}) = M₋ₕₓ(Wₕ)
@inline M₋ₕᵢ(Wₕ::SpaceType, ::Val{2}) = M₋ₕᵧ(Wₕ)
@inline M₋ₕᵢ(Wₕ::SpaceType, ::Val{3}) = M₋ₕ₂(Wₕ)

"""
	M₋ₕ(Wₕ::SpaceType)

Returns a tuple of [MatrixElement](@ref)s implementing the average operators in the `x`, `y`, and `z` directions.
"""
@inline M₋ₕ(Wₕ::SpaceType) = ntuple(i -> M₋ₕᵢ(Wₕ, Val(i)), dim(mesh(Wₕ)))

# Average operator applied to vectors
"""
	M₋ₕₓ(uₕ::VectorElement)

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
@inline M₋ₕₓ(uₕ::VectorElement) = element(space(uₕ), M₋ₕₓ(space(uₕ)).values * uₕ.values)

"""
	M₋ₕᵧ(uₕ::VectorElement)

Returns the average, in the `y` direction, of the element `uₕ`.

```math
\\textrm{M}_{hy} \\textrm{u}_h(x_i, y_j,\\dots) \\vcentcolon = \\textrm{u}_h(x_i, y_j\\dots)-\\textrm{u}_h(x_i, y_{j+1}, \\dots)
```
"""
@inline M₋ₕᵧ(uₕ::VectorElement) = element(space(uₕ), M₋ₕᵧ(space(uₕ)).values * uₕ.values)

"""
	M₋ₕ₂(uₕ::VectorElement)

Returns the average, in the `z` direction, of the element `uₕ`.

```math
\\textrm{M}_{hz} \\textrm{u}_h(x_i, y_j,z_l) \\vcentcolon = \\frac{\\textrm{u}_h(x_i, y_j, z_l)+\\textrm{u}_h(x_i, y_j, z_{l-1})}{2}
```
"""
@inline M₋ₕ₂(uₕ::VectorElement) = element(space(uₕ), M₋ₕ₂(space(uₕ)).values * uₕ.values)

@inline M₋ₕᵢ(uₕ::VectorElement, ::Val{1}) = M₋ₕₓ(uₕ)
@inline M₋ₕᵢ(uₕ::VectorElement, ::Val{2}) = M₋ₕᵧ(uₕ)
@inline M₋ₕᵢ(uₕ::VectorElement, ::Val{3}) = M₋ₕ₂(uₕ)

"""
	M₋ₕ(uₕ::VectorElement)

Returns a tuple of [VectorElement](@ref)s implementing the average operators in the `x`, `y`, and `z` directions applied to `uₕ`. If `uₕ` is `1`-dimensional, it returns just the average.
"""
@inline M₋ₕ(uₕ::VectorElement) = M₋ₕ(uₕ, Val(dim(mesh(space(uₕ)))))
@inline M₋ₕ(uₕ::VectorElement, ::Val{1}) = element(space(uₕ), M₋ₕᵢ(space(uₕ), Val(1)).values * uₕ.values)
@inline M₋ₕ(uₕ::VectorElement, ::Val{D}) where D = ntuple(i -> element(space(uₕ), M₋ₕᵢ(space(uₕ), Val(i)).values * uₕ.values), D)

# Average operator applied to matrices
"""
	M₋ₕₓ(Uₕ::MatrixElement)

Returns a [MatrixElement](@ref) resulting of the multiplication of the average matrix [M₋ₕₓ](@ref M₋ₕₓ(Wₕ::SpaceType)) with the [MatrixElement](@ref) `Uₕ`.
"""
@inline M₋ₕₓ(Uₕ::MatrixElement) = elements(space(Uₕ), M₋ₕₓ(space(Uₕ)).values * Uₕ.values)

"""
	M₋ₕᵧ(Uₕ::MatrixElement)

Returns a [MatrixElement](@ref) resulting of the multiplication of the average matrix [M₋ₕᵧ](@ref M₋ₕᵧ(Wₕ::SpaceType)) with the [MatrixElement](@ref) `Uₕ`.
"""
@inline M₋ₕᵧ(Uₕ::MatrixElement) = elements(space(Uₕ), M₋ₕᵧ(space(Uₕ)).values * Uₕ.values)

"""
	M₋ₕ₂(Uₕ::MatrixElement)

Returns a [MatrixElement](@ref) resulting of the multiplication of the average matrix [M₋ₕ₂](@ref M₋ₕ₂(Wₕ::SpaceType)) with the [MatrixElement](@ref) `Uₕ`.
"""
@inline M₋ₕ₂(Uₕ::MatrixElement) = elements(space(Uₕ), M₋ₕ₂(space(Uₕ)).values * Uₕ.values)

@inline M₋ₕᵢ(Uₕ::MatrixElement, ::Val{1}) = M₋ₕₓ(Uₕ)
@inline M₋ₕᵢ(Uₕ::MatrixElement, ::Val{2}) = M₋ₕᵧ(Uₕ)
@inline M₋ₕᵢ(Uₕ::MatrixElement, ::Val{3}) = M₋ₕ₂(Uₕ)

"""
	M₋ₕ(Uₕ::MatrixElement)

Returns a tuple of [MatrixElement](@ref)s implementing the M₋ₕ operators in the `x`, `y`, and `z` directions applied to `Uₕ`. In the `1`-dimensional case, it returns just the average.
"""
@inline M₋ₕ(Uₕ::MatrixElement) = M₋ₕ(Uₕ, Val(dim(mesh(space(Uₕ)))))
@inline M₋ₕ(Uₕ::MatrixElement, ::Val{1}) = M₋ₕₓ(Uₕ)
@inline M₋ₕ(Uₕ::MatrixElement, ::Val{D}) where D = ntuple(i -> M₋ₕᵢ(Uₕ, Val(i)) * Uₕ, dim(mesh(space(Uₕ))))