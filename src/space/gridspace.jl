# a tuple storing the symbols used for the different coordinate directions
const _BRAMBLE_var2symbol = ("ₓ", "ᵧ", "₂")

"""
	AbstractSpaceType{N}

Abstract type for N grid spaces defined on meshes of type [MeshType](@ref).
"""
abstract type AbstractSpaceType{N} <: BrambleType end

"""
	struct VectorElement{S,T,VT<:AbstractVector{T}} <: AbstractVector{T}
		data::VT
		space::S
	end

Vector element of `space` with coefficients stored in `values`.
"""
struct VectorElement{S,T,VT<:AbstractVector{T}} <: AbstractVector{T}
	data::VT
	space::S
end

"""
	struct MatrixElement{S,T,MT<:AbstractMatrix{T}} <: AbstractMatrix{T}
		data::MT
		space::S
	end

A `MatrixElement` is a container with a matrix of type `MatrixType`. The container also has a space to retain the information to which this special element belongs to. Its purpose is to represent discretization matrices from finite difference methods.
"""
struct MatrixElement{S,T,MT<:AbstractMatrix{T}} <: AbstractMatrix{T}
	data::MT
	space::S
end

# Accessors for GridSpaces
abstract type ComponentStyle <: BrambleType end
struct SingleComponent <: ComponentStyle end
struct MultiComponent{D} <: ComponentStyle end

# --- Type System for Dispatch ---
abstract type InnerProductType end
struct Innerplus <: InnerProductType end
struct Innerh <: InnerProductType end

@inline space(Wₕ::AbstractSpaceType) = return Wₕ

@inline function mesh(Wₕ::AbstractSpaceType)
	error("mesh not implemented for $(typeof(Wₕ))")
end

@inline function mesh_type(Wₕ::AbstractSpaceType)
	error("mesh_type not implemented for $(typeof(Wₕ))")
end

@inline function mesh_type(::Type{<:AbstractSpaceType})
	error("mesh_type not implemented for $(typeof(Wₕ))")
end

@inline function backward_difference_matrix(Wₕ::AbstractSpaceType, i)
	error("backward_difference_matrix not implemented for $(typeof(Wₕ))")
end

@inline function average_matrix(Wₕ::AbstractSpaceType, i)
	error("average_matrix not implemented for $(typeof(Wₕ))")
end

@inline function vector_buffer(Wₕ::AbstractSpaceType)
	error("vector_buffer not implemented for $(typeof(Wₕ))")
end

@inline function has_backward_difference_matrix(Wₕ::AbstractSpaceType)
	error("has_backward_difference_matrix not implemented for $(typeof(Wₕ))")
end

@inline function has_average_matrix(Wₕ::AbstractSpaceType)
	error("has_average_matrix not implemented for $(typeof(Wₕ))")
end

@inline function backend(Wₕ::AbstractSpaceType)
	error("backend not implemented for $(typeof(Wₕ))")
end

@inline function dim(Wₕ::AbstractSpaceType)
	error("dim not implemented for $(typeof(Wₕ))")
end

"""
	ndofs(Wₕ::AbstractSpaceType)
	ndofs(Wₕ::AbstractSpaceType, [::Type{Tuple}])

Returns the total number of degrees of freedom (or a tuple with the degrees of freedom per direction) of the [GridSpace](@ref) `Wₕ`.
"""
@inline function ndofs(Wₕ::AbstractSpaceType)
	error("ndofs not implemented for $(typeof(Wₕ))")
end

@inline function ndofs(Wₕ::AbstractSpaceType, ::Type{Tuple})
	error("ndofs not implemented for $(typeof(Wₕ))")
end

"""
	eltype(Wₕ::AbstractSpaceType)
	eltype(::Type{<:AbstractSpaceType})

Returns the element type of the mesh associated with [GridSpace](@ref) `Wₕ`. If the input argument is a type derived from [AbstractSpaceType](@ref) then the function returns the element type of the [AbstractMeshType](@ref) associated with it.
"""
@inline function eltype(Wₕ::AbstractSpaceType)
	error("eltype not implemented for $(typeof(Wₕ))")
end

@inline function eltype(::Type{W}) where W<:AbstractSpaceType
	error("eltype not implemented for $(typeof(Wₕ))")
end