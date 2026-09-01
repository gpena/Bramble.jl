module BrambleVTKExt

using Bramble: Bramble, AbstractMeshType, VectorElement, CompositeGridSpace,
               points, to_matrix, components

using WriteVTK: WriteVTK, vtk_grid, vtk_save

# `vtk_grid` for a rectilinear grid wants at least two coordinate vectors, `z` defaulting to
# a single point when omitted. A 1D mesh gets a degenerate second axis for the same reason,
# built by hand since there is only one axis to pad.
_vtk_axes(Ωₕ::AbstractMeshType{1}) = (points(Ωₕ), [zero(eltype(Ωₕ))])
_vtk_axes(Ωₕ::AbstractMeshType) = points(Ωₕ)

# What `vtk[name] = ...` wants for one field. A scalar space gives an array shaped like the
# grid — `to_matrix` already reshapes a `VectorElement`'s flat storage that way, in the same
# column-major order `points(Ωₕ)`'s axes imply, so no permutation is needed. A composite
# space gives a `Tuple` of them: WriteVTK reads `length(data)` off a `Tuple` as the number of
# vector components, one array per component.
_vtk_data(uₕ::VectorElement{<:CompositeGridSpace}) = Tuple(to_matrix.(components(uₕ)))
_vtk_data(uₕ::VectorElement) = to_matrix(uₕ)
_vtk_data(a::AbstractArray) = a

function Bramble._export_vtk(
        filename::AbstractString, Ωₕ::AbstractMeshType, fields::Pair...)
    vtk = vtk_grid(filename, _vtk_axes(Ωₕ)...)
    for (name, data) in fields
        vtk[name] = _vtk_data(data)
    end
    return vtk_save(vtk)
end

end
