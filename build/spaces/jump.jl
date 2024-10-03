##############################################################################
#                                                                            #
#                   Implementation of the jump operators                     #
#                                                                            #
##############################################################################

# Jump operator as a matrix
"""
	jumpₓ(Wₕ::SpaceType)
	jumpₓ(Ωₕ::MeshType)

Returns a [MatrixElement](@ref) implementing the jump matrix for the mesh grid of `Wₕ`, in the `x` direction. It is defined as being the (sparse) matrix representation of the linear operator defined by [jumpₓ](@ref jumpₓ(uₕ::VectorElement)). It also accepts a mesh as an argument.
"""
@inline jumpₓ(Wₕ::SpaceType) = elements(Wₕ, jumpₓ(mesh(Wₕ)))
@inline jumpₓ(Ωₕ::MeshType) = shiftₓ(Ωₕ, Val(dim(Ωₕ)), Val(0)) - shiftₓ(Ωₕ, Val(dim(Ωₕ)), Val(1))

"""
	jumpᵧ(Wₕ::SpaceType)
	jumpᵧ(Ωₕ::MeshType)

Returns a [MatrixElement](@ref) implementing the jump matrix for the mesh grid of `Wₕ`, in the `y` direction. It is defined as being the (sparse) matrix representation of the linear operator defined by [jumpᵧ](@ref jumpᵧ(uₕ::VectorElement)). It also accepts a mesh as an argument.
"""
@inline jumpᵧ(Wₕ::SpaceType) = elements(Wₕ, jumpᵧ(mesh(Wₕ)))
@inline jumpᵧ(Ωₕ::MeshType) = shiftᵧ(Ωₕ, Val(dim(Ωₕ)), Val(0)) - shiftᵧ(Ωₕ, Val(dim(Ωₕ)), Val(1))

"""
	jump₂(Wₕ::SpaceType)
	jump₂(Ωₕ::MeshType)

Returns a [MatrixElement](@ref) implementing the jump matrix for the mesh grid of `Wₕ`, in the `z` direction. It is defined as being the (sparse) matrix representation of the linear operator defined by [jump₂](@ref jump₂(uₕ::VectorElement)). It also accepts a mesh as an argument.
"""
@inline jump₂(Wₕ::SpaceType) = elements(Wₕ, jump₂(mesh(Wₕ)))
@inline jump₂(Ωₕ::MeshType) = shift₂(Ωₕ, Val(dim(Ωₕ)), Val(1)) - shift₂(Ωₕ, Val(dim(Ωₕ)), Val(0))

@inline jumpᵢ(Wₕ::SpaceType, ::Val{1}) = jumpₓ(Wₕ)
@inline jumpᵢ(Wₕ::SpaceType, ::Val{2}) = jumpᵧ(Wₕ)
@inline jumpᵢ(Wₕ::SpaceType, ::Val{3}) = jump₂(Wₕ)

"""
	jumpₕ(Wₕ::SpaceType)

Returns a tuple of [MatrixElement](@ref)s implementing the jump operators in the `x`, `y`, and `z` directions. If the problem is `1`-dimensional, it returns a single [MatrixElement](@ref).
"""
@inline jumpₕ(Wₕ::SpaceType) = jumpₕ(Wₕ, Val(dim(mesh(Wₕ))))
@inline jumpₕ(Wₕ::SpaceType, ::Val{1}) = jumpₓ(Wₕ)
@inline jumpₕ(Wₕ::SpaceType, ::Val{D}) where D = ntuple(i -> jumpᵢ(Wₕ, Val(i)), D)

# Jump operator applied to vectors
"""
	jumpₓ(uₕ::VectorElement)

Returns the jump, in the `x` direction, of the element `uₕ`.

  - 1D case

```math
\\textrm{jump}_{x} \\textrm{u}_h(x_i) \\vcentcolon = \\textrm{u}_h(x_i) - \\textrm{u}_h(x_{i+1})
```

  - 2D and 3D case

```math
\\textrm{jump}_{x} \\textrm{u}_h(x_i, \\dots) \\vcentcolon = \\textrm{u}_h(x_i, \\dots)-\\textrm{u}_h(x_{i+1}, \\dots)
```
"""
@inline jumpₓ(uₕ::VectorElement) = element(space(uₕ), jumpₓ(space(uₕ)).values * uₕ.values)

"""
	jumpᵧ(uₕ::VectorElement)

Returns the jump, in the `y` direction, of the element `uₕ`.

```math
\\textrm{jump}_{y} \\textrm{u}_h(x_i, y_j,\\dots) \\vcentcolon = \\textrm{u}_h(x_i, y_j\\dots)-\\textrm{u}_h(x_i, y_{j+1}, \\dots)
```
"""
@inline jumpᵧ(uₕ::VectorElement) = element(space(uₕ), jumpᵧ(space(uₕ)).values * uₕ.values)

"""
	jump₂(uₕ::VectorElement)

Returns the jump, in the `z` direction, of the element `uₕ`.

```math
\\textrm{jump}_{z} \\textrm{u}_h(x_i, y_j,z_l) \\vcentcolon = \\textrm{u}_h(x_i, y_j, z_l)-\\textrm{u}_h(x_i, y_j, z_{l+1})
```
"""
@inline jump₂(uₕ::VectorElement) = element(space(uₕ), jump₂(space(uₕ)).values * uₕ.values)

@inline jumpᵢ(uₕ::VectorElement, ::Val{1}) = jumpₓ(uₕ)
@inline jumpᵢ(uₕ::VectorElement, ::Val{2}) = jumpᵧ(uₕ)
@inline jumpᵢ(uₕ::VectorElement, ::Val{3}) = jump₂(uₕ)

"""
	jumpₕ(uₕ::VectorElement)

Returns a tuple of [VectorElement](@ref)s implementing the jump operators in the `x`, `y`, and `z` directions applied to `uₕ`. If the problem is `1`-dimensional, it returns a single [VectorElement](@ref).
"""
@inline jumpₕ(uₕ::VectorElement) = jumpₕ(uₕ, Val(dim(mesh(space(uₕ)))))
@inline jumpₕ(uₕ::VectorElement, ::Val{1}) = jumpₓ(uₕ)
@inline jumpₕ(uₕ::VectorElement, ::Val{D}) where D = ntuple(i -> element(space(uₕ), jumpᵢ(space(uₕ), Val(i)).values * uₕ.values), D)

# Jump operator applied to matrices
"""
	jumpₓ(Uₕ::MatrixElement)

Returns a [MatrixElement](@ref) resulting of the multiplication of the jump matrix [jumpₓ](@ref jumpₓ(Wₕ::SpaceType)) with the [MatrixElement](@ref) `Uₕ`.
"""
@inline jumpₓ(Uₕ::MatrixElement) = elements(space(Uₕ), jumpₓ(space(Uₕ)).values * Uₕ.values)

"""
	jumpᵧ(Uₕ::MatrixElement)

Returns a [MatrixElement](@ref) resulting of the multiplication of the jump matrix [jumpᵧ](@ref jumpᵧ(Wₕ::SpaceType)) with the [MatrixElement](@ref) `Uₕ`.
"""
@inline jumpᵧ(Uₕ::MatrixElement) = elements(space(Uₕ), jumpᵧ(space(Uₕ)).values * Uₕ.values)

"""
	jump₂(Uₕ::MatrixElement)

Returns a [MatrixElement](@ref) resulting of the multiplication of the jump matrix [jump₂](@ref jump₂(Wₕ::SpaceType)) with the [MatrixElement](@ref) `Uₕ`.
"""
@inline jump₂(Uₕ::MatrixElement) = elements(space(Uₕ), jump₂(space(Uₕ)).values * Uₕ.values)

@inline jumpᵢ(Uₕ::MatrixElement, ::Val{1}) = jumpₓ(Uₕ)
@inline jumpᵢ(Uₕ::MatrixElement, ::Val{2}) = jumpᵧ(Uₕ)
@inline jumpᵢ(Uₕ::MatrixElement, ::Val{3}) = jump₂(Uₕ)

"""
	jumpₕ(Uₕ::MatrixElement)

Returns a tuple of [MatrixElement](@ref)s implementing the jump operators in the `x`, `y`, and `z` directions applied to `Uₕ`. If the problem is `1`-dimensional, it returns a single [MatrixElement](@ref).
"""
@inline jumpₕ(Uₕ::MatrixElement) = jumpₕ(Uₕ, Val(dim(mesh(space(Uₕ)))))
@inline jumpₕ(Uₕ::MatrixElement, ::Val{1}) = jumpₓ(Uₕ)
@inline jumpₕ(Uₕ::MatrixElement, ::Val{D}) where D = ntuple(i -> elements(space(Uₕ), jumpᵢ(space(Uₕ), Val(i)).values * Uₕ.values), D)