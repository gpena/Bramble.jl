"""
# indices.jl

Cartesian index generation and boundary/interior facet predicates shared by every mesh
type in Bramble.

- `generate_indices`: build the `CartesianIndices` of a mesh from its point counts.
- `is_boundary_index`, `boundary_indices`, `interior_indices`: facet and interior
  predicates over `CartesianIndices` directly, or over an `AbstractMeshType` by
  delegating to its own indices.

See also: [`Mesh1D`](@ref), [`MeshnD`](@ref)
"""

#------------------------------------------------------------------------------------------#
# Cartesian Index Generation & Boundary Queries
#------------------------------------------------------------------------------------------#

"""
    generate_indices(pts::Int) -> CartesianIndices{1}
    generate_indices(pts::NTuple{D, Int}) -> CartesianIndices{D}

Return the `CartesianIndices` of a mesh with `pts[i]` points in each direction.

For scalar input (`Int`), returns 1D `CartesianIndices`. For tuple input, returns
multi-dimensional `CartesianIndices`.
"""
@inline generate_indices(pts::Int) = CartesianIndices((pts,))
@inline generate_indices(pts::NTuple{D,Int}) where {D} = CartesianIndices(pts)

"""
    is_boundary_index(idxs::CartesianIndices{D}, idx) -> Bool
    is_boundary_index(Ωₕ::AbstractMeshType, idx) -> Bool

Determine whether index `idx` lies on the boundary of `idxs` or mesh `Ωₕ`.
"""
function is_boundary_index(idxs::CartesianIndices{D}, idx) where {D}
    _idx = CartesianIndex(idx)
    @inbounds for i in 1:D
        axis = idxs.indices[i]
        if length(axis) > 1 && (_idx[i] == first(axis) || _idx[i] == last(axis))
            return true
        end
    end
    return false
end

"""
    boundary_indices(idxs::CartesianIndices{D}) -> NTuple{2D, CartesianIndices{D}}
    boundary_indices(Ωₕ::AbstractMeshType{D}) -> NTuple{2D, CartesianIndices{D}}

Return all boundary facets of a `CartesianIndices` domain or mesh `Ωₕ` as a tuple of `CartesianIndices`.
"""
@inline boundary_indices(idxs::CartesianIndices) = Tuple(boundary_symbol_to_cartesian(idxs))

"""
    interior_indices(indices::CartesianIndices{D}) -> CartesianIndices{D}
    interior_indices(Ωₕ::AbstractMeshType{D}) -> CartesianIndices{D}

Compute the `CartesianIndices` representing the interior of a domain or mesh, excluding
all boundary points. Dimensions with a length of one or less remain unchanged.
"""
@inline function interior_indices(indices::CartesianIndices{D}) where {D}
    original_ranges = indices.indices

    interior_ranges_tuple = ntuple(Val(D)) do i
        @inbounds r = original_ranges[i]
        if length(r) <= 1
            return r
        else
            (first(r) + 1):(last(r) - 1)
        end
    end

    return CartesianIndices(interior_ranges_tuple)
end

#------------------------------------------------------------------------------------------#
# Mesh-Level Delegation
#------------------------------------------------------------------------------------------#

@inline is_boundary_index(Ωₕ::AbstractMeshType, idx) = is_boundary_index(indices(Ωₕ), idx)
@inline boundary_indices(Ωₕ::AbstractMeshType) = boundary_indices(indices(Ωₕ))
@inline interior_indices(Ωₕ::AbstractMeshType) = interior_indices(indices(Ωₕ))
