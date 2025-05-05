"""
	SpaceType

Abstract type for grid spaces defined on meshes of type [MeshType](@ref).
"""
abstract type AbstractSpaceType <: BrambleType end

"""
	struct VectorElement{S, VectorType, T} <: AbstractVector{T}
		space::S
		values::VectorType
	end

Vector element of `space` with coefficients stored in `values`.
"""
struct VectorElement{S,VectorType,T} <: AbstractVector{T}
	space::S
	values::VectorType
end

"""
	MatrixElement{S, MatrixType, T} <: AbstractMatrix{T}

A `MatrixElement` is a container with a matrix of type `MatrixType`. The container also has a space to retain the information to which this special element belongs to. Its purpose is to represent discretization matrices from finite difference methods.
"""
struct MatrixElement{S,MatrixType,T} <: AbstractMatrix{T}
	space::S
	values::MatrixType
end

struct SpaceWeights{D,VT} <: BrambleType
	innerh::VT
	innerplus::NTuple{D,VT}
end

"""
	struct SingleGridSpace{MType,D,BT,VT,MT} <: SpaceType{MType}
		mesh::MType
		weights::SpaceWeights{D,VT}
		diff_matrix::NTuple{D,MT}
		vector_buffer::GridSpaceBuffer{BT,VT}
	end

Structure for a gridspace defined on a mesh.

The dictionary `weights` has the weights for several the standard discrete ``L^2``
inner product on the space of grid functions.

They are defined as follows:

  - weights[:innerₕ]

	  + 1D case

```math
(u_h, v_h)_h = \\sum_{i=1}^{N_x} |\\square_{i}| u_h(x_i) v_h(x_i)
```

	+ 2D case

```math
(u_h, v_h)_h = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y} |\\square_{i,j}| u_h(x_i,y_j) v_h(x_i,y_j)
```

	+ 3D case

```math
(u_h, v_h)_h = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y}\\sum_{l=1}^{N_z} |\\square_{i,j,l}| u_h(x_i,y_j,z_l) v_h(x_i,y_j,z_l).
```

Here, ``|\\cdot|`` denotes the measure of the set and all details on the definition of ``\\square_{i}``, ``\\square_{i,j}`` and ``\\square_{i,j,l}`` can be found in functions [cell_measure](@ref cell_measure(Ωₕ::Mesh1D, i)) (for the `1`-dimensional case) and [cell_measure](@ref cell_measure(Ωₕ::MeshnD, i)) (for the `n`-dimensional cases).

  - weights[:inner₊], weights[:inner₊ₓ], weights[:inner₊ᵧ], weights[:inner₊₂]
	These are the weights for the modified discrete ``L^2`` inner product on the space of grid functions, for each component (x, y, z).

		+ 1D case

```math
(u_h, v_h)_+ = \\sum_{i=1}^{N_x} h_{i} u_h(x_i) v_h(x_i)
```

	+ 2D case

```math
(u_h, v_h)_{+x} = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y} h_{x,i} h_{y,j+1/2} u_h(x_i,y_j) v_h(x_i,y_j)
```

```math
(u_h, v_h)_{+y} = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y} h_{x,i} h_{y,j+1/2} u_h(x_i,y_j) v_h(x_i,y_j)
```

	+ 3D case

```math
(u_h, v_h)_{+x} = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y}\\sum_{l=1}^{N_z} h_{x,i} h_{y,j+1/2} h_{z,l+1/2} u_h(x_i,y_j,z_l) v_h(x_i,y_j,z_l)
```

```math
(u_h, v_h)_{+y} = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y}\\sum_{l=1}^{N_z} h_{x,i+1/2} h_{y,j} h_{z,l+1/2} u_h(x_i,y_j,z_l) v_h(x_i,y_j,z_l)
```

```math
(u_h, v_h)_{+z} = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y}\\sum_{l=1}^{N_z} h_{x,i+1/2} h_{y,j+1/2} h_{z,l} u_h(x_i,y_j,z_l) v_h(x_i,y_j,z_l).
```
"""
mutable struct SingleGridSpace{MType,D,VT,MT,SpaceBufferType} <: AbstractSpaceType
	const mesh::MType
	const weights::SpaceWeights{D,VT}
	const backward_diff_matrix::NTuple{D,MT}
	has_backward_diff_matrix::Bool
	const average_matrix::NTuple{D,MT}
	has_average_matrix::Bool
	const vector_buffer::SpaceBufferType
end

struct CompositeGridSpace{S1,S2} <: AbstractSpaceType
	space1::S1
	space2::S2
end

# a tuple storing the symbols used for the different coordinate directions
const _BRAMBLE_var2symbol = ("ₓ", "ᵧ", "₂")
