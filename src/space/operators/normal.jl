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

    nₕ = ntuple(_ -> element(Wₕ, zero(eltype(Wₕ))), Val(D))
    @inbounds for I in indices(Ωₕ)
        _on_face(mask, I, np, Val(D)) || continue
        for d in 1:D
            parent(nₕ[d])[lin[I]] = ν[d]
        end
    end
    return nₕ
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
