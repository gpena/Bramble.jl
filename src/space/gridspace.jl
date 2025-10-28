# A constant tuple storing subscript symbols for x, y, z coordinates.
# Used for creating nicely formatted output, like labels in plots or printed variable names (e.g., Dₓ).
const _BRAMBLE_var2symbol = ("ₓ", "ᵧ", "₂")
const _BRAMBLE_var2label = ("x", "y", "z")
"""
	AbstractSpaceType{N}

Abstract supertype for all function spaces defined on a mesh.

This is the top-level abstraction for a grid-based function space. The parameter `N` represents the number of components of the field (e.g., `N=1` for a scalar field, `N=3` for a 3D vector field).
"""
abstract type AbstractSpaceType{N} end

"""
	$(TYPEDEF)

Represents a **grid function** (a vector) that belongs to a specific function space.

This is a wrapper that bundles the raw numerical data (the vector `data`) with its parent `space`. The `space` provides the essential context, such as the underlying mesh and associated operators. By subtyping `AbstractVector`, a [VectorElement](@ref) can be used just like a regular Julia vector in most operations.

# Fields

$(FIELDS)
"""
struct VectorElement{S,T,VT<:AbstractVector{T}} <: AbstractVector{T}
	"the raw vector data containing the degrees of freedom."
	data::VT
	"the parent function space to which this vector belongs."
	space::S
end

"""
	$(TYPEDEF)

Represents a **discrete linear operator** (a matrix) that acts on a function space.

Similar to , this container bundles a raw matrix (`data`) with its parent `space`. This is used to represent discretization matrices from methods like finite differences (e.g., a differentiation or averaging matrix). Subtyping `AbstractMatrix` allows it to be used like a standard Julia matrix.

# Fields

$(FIELDS)
"""
struct MatrixElement{S,T,MT<:AbstractMatrix{T}} <: AbstractMatrix{T}
	"the matrix data representing the linear operator."
	data::MT
	"the parent function space to which this vector belongs."
	space::S
end

#=
The `ComponentStyle` and `InnerProductType` hierarchies are a common Julia pattern
for **compile-time dispatch**. Instead of using `if/else` checks, functions can
specialize their behavior by dispatching on these types, leading to more efficient
and extensible code. ✨
=#
abstract type ComponentStyle end
struct SingleComponent <: ComponentStyle end
struct MultiComponent{D} <: ComponentStyle end

# --- Type System for Dispatch ---
abstract type InnerProductType end
struct Innerplus <: InnerProductType end
struct Innerh <: InnerProductType end

#=
The following functions define the **mandatory interface** for any concrete subtype
of `AbstractSpaceType`. Any struct that subtypes `AbstractSpaceType` must implement
these methods to be considered a valid function space in this framework. The default
implementations throw an error, guiding developers to provide the correct implementation.
=#

"""
	space(Wₕ::AbstractSpaceType)

Returns the function space `Wₕ` itself.
"""
@inline space(Wₕ::AbstractSpaceType) = return Wₕ

"""
	mesh(Wₕ::AbstractSpaceType)

Returns the underlying mesh object associated with the function space `Wₕ`.
"""
@inline function mesh(Wₕ::AbstractSpaceType)
	error("Interface function 'mesh' not implemented for $(typeof(Wₕ))")
end

"""
	mesh_type(Wₕ::AbstractSpaceType)

Returns the type of the mesh associated with the function space `Wₕ`. Also works if the argument is the type of the space.
"""
@inline function mesh_type(Wₕ::AbstractSpaceType)
	error("Interface function 'mesh_type' not implemented for $(typeof(Wₕ))")
end
@inline function mesh_type(::Type{<:AbstractSpaceType})
	error("Interface function 'mesh_type' not implemented for this space type.")
end

"""
	backward_difference_matrix(Wₕ::AbstractSpaceType, i)

Returns the backward difference matrix for the `i`-th dimension of the space `Wₕ`.
"""
@inline function backward_difference_matrix(Wₕ::AbstractSpaceType, i)
	error("Interface function 'backward_difference_matrix' not implemented for $(typeof(Wₕ))")
end

"""
	average_matrix(Wₕ::AbstractSpaceType, i)

Returns the averaging matrix for the `i`-th dimension of the space `Wₕ`.
"""
@inline function average_matrix(Wₕ::AbstractSpaceType, i)
	error("Interface function 'average_matrix' not implemented for $(typeof(Wₕ))")
end

"""
	vector_buffer(Wₕ::AbstractSpaceType)

Returns the [GridSpaceBuffer](@ref) used for efficient memory management in the space `Wₕ`.
"""
@inline function vector_buffer(Wₕ::AbstractSpaceType)
	error("Interface function 'vector_buffer' not implemented for $(typeof(Wₕ))")
end

"""
	has_backward_difference_matrix(Wₕ::AbstractSpaceType)

Checks if the backward difference matrices have been computed and stored for `Wₕ`.
"""
@inline function has_backward_difference_matrix(Wₕ::AbstractSpaceType)
	error("Interface function 'has_backward_difference_matrix' not implemented for $(typeof(Wₕ))")
end

"""
	has_average_matrix(Wₕ::AbstractSpaceType)

Checks if the averaging matrices have been computed and stored for `Wₕ`.
"""
@inline function has_average_matrix(Wₕ::AbstractSpaceType)
	error("Interface function 'has_average_matrix' not implemented for $(typeof(Wₕ))")
end

"""
	$(TYPEDSIGNATURES)

Returns the computational backend associated with the space `Wₕ`.
"""
@inline function backend(Wₕ::AbstractSpaceType)
	error("Interface function 'backend' not implemented for $(typeof(Wₕ))")
end

"""
	dim(Wₕ::AbstractSpaceType)

Returns the spatial dimension of the mesh associated with the functionpace `Wₕ`.
"""
@inline function dim(Wₕ::AbstractSpaceType)
	error("Interface function 'dim' not implemented for $(typeof(Wₕ))")
end

"""
	ndofs(Wₕ::AbstractSpaceType, [::Type{Tuple}])

Returns the total number of degrees of freedom (DOFs) in the function space `Wₕ`.
If `Tuple` is passed, it returns a tuple with the number of DOFs in each dimension.
"""
@inline function ndofs(Wₕ::AbstractSpaceType)
	error("Interface function 'ndofs' not implemented for $(typeof(Wₕ))")
end
@inline function ndofs(Wₕ::AbstractSpaceType, ::Type{Tuple})
	error("Interface function 'ndofs' not implemented for $(typeof(Wₕ))")
end

"""
	eltype(Wₕ::AbstractSpaceType)

Returns the element type (e.g., `Float64`) of the data in the function space `Wₕ`. It also works if the argument is the type of the space.
"""
@inline function eltype(Wₕ::AbstractSpaceType)
	error("Interface function 'eltype' not implemented for $(typeof(Wₕ))")
end
@inline function eltype(::Type{W}) where W<:AbstractSpaceType
	error("Interface function 'eltype' not implemented for this space type.")
end
