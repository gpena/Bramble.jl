# normal.jl
# The outward unit normal as a discrete grid function (gpena/Bramble.jl#213).
#
# `normal_vector(Ωₕ, :xmax)` (mesh/queries.jl) answers the geometric question: the constant
# vector a named face points along. This answers the discrete one: the same vector *sampled
# on the grid*, zero away from the face, so it can multiply a grid function or be exported
# beside one. The two are the same number; only one of them has a degree of freedom per
# point to put it in.

"""
    normal_vector(Wₕ::ScalarGridSpace{D}, marker::Symbol) -> NTuple{D, VectorElement}

Returns the outward unit normal of the boundary face `marker` names, as one grid function per
coordinate: component `d` holds `normal_vector(mesh(Wₕ), marker)[d]` at every point of that
face and zero everywhere else.

`marker` names a whole coordinate face -- `:xmin`…`:zmax` or a viewpoint alias -- for the
same reason [`inner_Γ`](@ref) does: a normal is a property of a face, and a marked set that
is not one has no single normal to sample.

The components are zero off the face rather than the face's normal everywhere, so a product
against one restricts to the face without a second mask.

# Examples

```jldoctest
Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 5), (true, true))
Wₕ = gridspace(Ωₕ)
nₕ = normal_vector(Wₕ, :ymin)
(sum(parent(nₕ[1])), sum(parent(nₕ[2])))

# output

(0.0, -5.0)
```

See also: [`normal_vector`](@ref), [`inner_Γ`](@ref)
"""
function normal_vector(Wₕ::ScalarGridSpace{D}, marker::Symbol) where {D}
    Ωₕ = mesh(Wₕ)
    ν = normal_vector(Ωₕ, marker)  # throws the ArgumentError for an unknown marker
    facet = boundary_symbol_to_cartesian(indices(Ωₕ))[marker]
    lin = LinearIndices(indices(Ωₕ))

    # `element(Wₕ, zero(eltype(Wₕ)))` fills through the `VectorElement` wrapper's generic
    # `setindex!`, which scalar-writes a device array once per point (a gap in
    # `element(::AbstractSpaceType, ::Number)`, src/space/vectorelement.jl, outside this
    # file's ownership). Building the element uninitialized and zeroing its `parent` directly
    # reaches the array's own `fill!` instead -- correct and no slower on a host array either.
    T = eltype(Wₕ)
    nₕ = ntuple(_ -> element(Wₕ, T), Val(D))
    for d in 1:D
        fill!(parent(nₕ[d]), zero(T))
    end
    _fill_normal!(locality(backend(Wₕ)), nₕ, facet, ν, lin, Val(D))
    return nₕ
end

# Host fill: visits only the facet slice (gpena/Bramble.jl#333), never the whole volume.
@inline function _fill_normal!(::HostLocality, nₕ, facet, ν, lin, ::Val{D}) where {D}
    @inbounds for I in facet
        k = lin[I]
        for d in 1:D
            parent(nₕ[d])[k] = ν[d]
        end
    end
    return nothing
end

# Device counterpart (gpena/Bramble.jl#311, #333): the components are already zeroed on the
# device, so only the facet's linear indices -- O(facet), not O(volume) -- are built on the
# host, moved to the device once, and each component is written there by indexed broadcast.
@noinline function _fill_normal!(::DeviceLocality, nₕ, facet, ν, lin, ::Val{D}) where {D}
    T = eltype(parent(nₕ[1]))
    host_idx = Int[lin[I] for I in facet]
    idx = similar(parent(nₕ[1]), Int, length(host_idx))
    copyto!(idx, host_idx)
    for d in 1:D
        view(parent(nₕ[d]), idx) .= T(ν[d])
    end
    return nothing
end
