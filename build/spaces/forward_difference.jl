##############################################################################
#                                                                            #
#             Implementation of the forward difference operators             #
#                                                                            #
##############################################################################

# Forward difference operators as a matrix
"""
	diffₓ(Wₕ::SpaceType)
	diffₓ(Ωₕ::MeshType)

Returns a [MatrixElement](@ref) implementing the forward difference matrix for the mesh grid of `Wₕ`, in the `x` direction. It is defined as being the (sparse) matrix representation of the linear operator defined by [diffₓ](@ref diffₓ(uₕ::VectorElement)). It can also accept a mesh as an argument.
"""
@inline diffₓ(Wₕ::SpaceType) = elements(Wₕ, diffₓ(mesh(Wₕ)))
@inline diffₓ(Ωₕ::MeshType) = shiftₓ(Ωₕ, Val(dim(Ωₕ)), Val(1)) - shiftₓ(Ωₕ, Val(dim(Ωₕ)), Val(0))

"""
	diffᵧ(Wₕ::SpaceType)
	diffᵧ(Ωₕ::MeshType)

Returns a [MatrixElement](@ref) implementing the forward difference matrix for the mesh grid of `Wₕ`, in the `y` direction. It is defined as being the (sparse) matrix representation of the linear operator defined by [diffᵧ](@ref diffᵧ(uₕ::VectorElement)). It can also accept a mesh as an argument.
"""
@inline diffᵧ(Wₕ::SpaceType) = elements(Wₕ, diffᵧ(mesh(Wₕ)))
@inline diffᵧ(Ωₕ::MeshType) = shiftᵧ(Ωₕ, Val(dim(Ωₕ)), Val(1)) - shiftᵧ(Ωₕ, Val(dim(Ωₕ)), Val(0))

"""
	diff₂(Wₕ::SpaceType)
	diff₂(Ωₕ::MeshType)

Returns a [MatrixElement](@ref) implementing the forward difference matrix for the mesh grid of `Wₕ`, in the `z` direction. It is defined as being the (sparse) matrix representation of the linear operator defined by [diff₂](@ref diff₂(uₕ::VectorElement)). It can also accept a mesh as an argument.
"""
@inline diff₂(Wₕ::SpaceType) = elements(Wₕ, diff₂(mesh(Wₕ)))
@inline diff₂(Ωₕ::MeshType) = shift₂(Ωₕ, Val(dim(Ωₕ)), Val(1)) - shift₂(Ωₕ, Val(dim(Ωₕ)), Val(0))

@inline diffᵢ(Wₕ::SpaceType, ::Val{1}) = diffₓ(Wₕ)
@inline diffᵢ(Wₕ::SpaceType, ::Val{2}) = diffᵧ(Wₕ)
@inline diffᵢ(Wₕ::SpaceType, ::Val{3}) = diff₂(Wₕ)

"""
	diffₕ(Wₕ::SpaceType)

Returns a tuple of [MatrixElement](@ref)s implementing the forward difference operators in the   `x`, `y` and `z` directions. If the problem is `1`-dimensional, it returns a single [MatrixElement](@ref).
"""
@inline diffₕ(Wₕ::SpaceType) = diffₕ(Wₕ, Val(dim(mesh(Wₕ))))
@inline diffₕ(Wₕ::SpaceType, ::Val{1}) = diffₓ(Wₕ)
@inline diffₕ(Wₕ::SpaceType, ::Val{D}) where D = ntuple(i -> diffᵢ(Wₕ, Val(i)), D)

# Implementation of the forward difference operators acting on vectors
"""
	diffₓ(uₕ::VectorElement)

Returns the forward difference, in the `x` direction, of the element `uₕ`.

  - 1D case

```math
\\textrm{diff}_{x} \\textrm{u}_h(x_i) \\vcentcolon = \\textrm{u}_h(x_{i+1}) - \\textrm{u}_h(x_i)
```

  - 2D and 3D case

```math
\\textrm{diff}_{x} \\textrm{u}_h(x_i, \\dots) \\vcentcolon = \\textrm{u}_h(x_{i+1}, \\dots)-\\textrm{u}_h(x_i, \\dots)
```
"""
Base.@propagate_inbounds function diffₓ(uₕ::VectorElement)
	vₕ = similar(uₕ)
	# implement 
	vₕ.values .= diffₓ(space(uₕ)) * uₕ.values

	return vₕ
end

"""
	diffᵧ(uₕ::VectorElement)

Returns the forward difference, in the `y` direction, of the element `uₕ`.

  - 2D and 3D case

```math
\\textrm{diff}_{y} \\textrm{u}_h(x_i, y_j,\\dots) \\vcentcolon = \\textrm{u}_h(x_i, y_{j+1},\\dots)-\\textrm{u}_h(x_i, y_j, \\dots)
```
"""
Base.@propagate_inbounds function diffᵧ(uₕ::VectorElement)
	vₕ = similar(uₕ)
	vₕ.values .= diffᵧ(space(uₕ)) * uₕ.values

	return vₕ
end

"""
	diff₂(uₕ::VectorElement)

Returns the forward difference, in the `z` direction, of the element `uₕ`.

```math
\\textrm{diff}_{-z} \\textrm{u}_h(x_i, y_j,z_l) \\vcentcolon = \\textrm{u}_h(x_i, y_j,z_l)-\\textrm{u}_h(x_i, y_j, z_{l-1})
```
"""
Base.@propagate_inbounds function diff₂(uₕ::VectorElement)
	vₕ = similar(uₕ)
	vₕ.values .= diff₂(space(uₕ)) * uₕ.values
	return vₕ
end

@inline diffᵢ(uₕ::VectorElement, ::Val{1}) = diffₓ(uₕ)
@inline diffᵢ(uₕ::VectorElement, ::Val{2}) = diffᵧ(uₕ)
@inline diffᵢ(uₕ::VectorElement, ::Val{3}) = diff₂(uₕ)

"""
	diffₕ(uₕ::VectorElement)

Returns a tuple of [VectorElement](@ref)s implementing the forward difference operators in the `x`, `y` and `z` directions applied to `uₕ`. If the problem is `1`-dimensional, it returns a single [VectorElement](@ref).
"""
@inline diffₕ(uₕ::VectorElement) = diffₕ(uₕ, Val(dim(mesh(space(uₕ)))))
@inline diffₕ(uₕ::VectorElement, ::Val{1}) = diffₓ(uₕ)
@inline diffₕ(uₕ::VectorElement, ::Val{D}) where D = ntuple(i -> diffᵢ(uₕ, Val(i)), D)

# Implementation of the forward difference operators for matrix elements
"""
	diffₓ(Uₕ::MatrixElement)

Returns a [MatrixElement](@ref) resulting of the multiplication of the forward difference matrix, in the `x` direction, by `Uₕ`.
"""
@inline diffₓ(Uₕ::MatrixElement) = diffₓ(space(Uₕ)) * Uₕ

"""
	diffᵧ(Uₕ::MatrixElement)

Returns a [MatrixElement](@ref) resulting of the multiplication of the forward difference matrix, in the `y` direction, by `Uₕ`.
"""
@inline diffᵧ(Uₕ::MatrixElement) = diffᵧ(space(Uₕ)) * Uₕ

"""
	diff₂(Uₕ::MatrixElement)

Returns a [MatrixElement](@ref) resulting of the multiplication of the forward difference matrix, in the `z` direction, by `Uₕ`.
"""
@inline diff₂(Uₕ::MatrixElement) = diff₂(space(Uₕ)) * Uₕ

@inline diffᵢ(Uₕ::MatrixElement, ::Val{1}) = diffₓ(Uₕ)
@inline diffᵢ(Uₕ::MatrixElement, ::Val{2}) = diffᵧ(Uₕ)
@inline diffᵢ(Uₕ::MatrixElement, ::Val{3}) = diff₂(Uₕ)

"""
	diffₕ(Uₕ::MatrixElement)

Returns a tuple of [MatrixElement](@ref)s implementing the forward difference operators in the   `x`, `y` and `z` directions applied to `Uₕ`. If the problem is `1`-dimensional, it returns a single [MatrixElement](@ref).
"""
@inline diffₕ(Uₕ::MatrixElement) = diffₕ(Uₕ, Val(dim(mesh(space(Uₕ)))))
@inline diffₕ(Uₕ::MatrixElement, ::Val{1}) = diffₓ(Uₕ)
@inline diffₕ(Uₕ::MatrixElement, ::Val{D}) where D = ntuple(i -> diffᵢ(Uₕ, Val(i)), D)