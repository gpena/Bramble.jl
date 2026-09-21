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
    ν = normal_vector(Ωₕ, marker)
    mask = _face_mask(Val(D), (marker,))
    np = npoints(Ωₕ, Tuple)
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
    _fill_normal!(locality(backend(Wₕ)), nₕ, Ωₕ, mask, ν, np, lin, Val(D))
    return nₕ
end

# Host fill: the per-point loop `normal_vector` always ran, moved behind a locality-dispatched
# helper (gpena/Bramble.jl#311) so a device-backed space can take the bulk-copy path below
# instead, without duplicating the mask/index setup above in a second method.
@inline function _fill_normal!(::HostLocality, nₕ, Ωₕ, mask, ν, np, lin, ::Val{D}) where {D}
    @inbounds for I in indices(Ωₕ)
        _on_face(mask, I, np, Val(D)) || continue
        k = lin[I]
        for d in 1:D
            parent(nₕ[d])[k] = ν[d]
        end
    end
    return nothing
end

# Device counterpart: the loop above would scalar-write a device array once per boundary
# point. Fill one host buffer per component with the same loop, then `copyto!` it onto the
# device array once -- one transfer per component instead of one write per boundary point.
@noinline function _fill_normal!(::DeviceLocality, nₕ, Ωₕ, mask, ν, np, lin, ::Val{D}) where {D}
    T = eltype(parent(nₕ[1]))
    host = ntuple(_ -> zeros(T, length(lin)), Val(D))
    @inbounds for I in indices(Ωₕ)
        _on_face(mask, I, np, Val(D)) || continue
        k = lin[I]
        for d in 1:D
            host[d][k] = ν[d]
        end
    end
    for d in 1:D
        copyto!(parent(nₕ[d]), host[d])
    end
    return nothing
end

# Whether `I` lies on any face of the mask. `_surface_weight` asks the same question per
# direction; this one only needs the disjunction, and is not on an assembly hot path.
@inline function _on_face(mask, I, np, ::Val{D}) where {D}
    for d in 1:D
        (mask[d][1] && I[d] == 1) && return true
        (mask[d][2] && I[d] == np[d]) && return true
    end
    return false
end
