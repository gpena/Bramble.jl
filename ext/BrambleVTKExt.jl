module BrambleVTKExt

using Bramble:
               Bramble,
               AbstractMeshType,
               VectorElement,
               CompositeGridSpace,
               points,
               components,
               export_vtk,
               domain,
               interval,
               ×,
               mesh,
               gridspace,
               Rₕ

using WriteVTK: WriteVTK, vtk_grid, vtk_save
using PrecompileTools: @setup_workload, @compile_workload

# `vtk_grid` for a rectilinear grid wants at least two coordinate vectors, `z` defaulting to
# a single point when omitted. A 1D mesh gets a degenerate second axis for the same reason,
# built by hand since there is only one axis to pad.
_vtk_axes(Ωₕ::AbstractMeshType{1}) = (points(Ωₕ), [zero(eltype(Ωₕ))])
_vtk_axes(Ωₕ::AbstractMeshType) = points(Ωₕ)

# What `vtk[name] = ...` wants for one field. A scalar space gives an array shaped like the
# grid: `reshape(uₕ)` already reshapes a `VectorElement`'s flat storage that way, in the same
# column-major order `points(Ωₕ)`'s axes imply, so no permutation is needed. A composite
# space gives a `Tuple` of them: WriteVTK reads `length(data)` off a `Tuple` as the number of
# vector components, one array per component.
_vtk_data(uₕ::VectorElement{<:CompositeGridSpace}) = Tuple(reshape.(components(uₕ)))
_vtk_data(uₕ::VectorElement) = reshape(uₕ)
_vtk_data(a::AbstractArray) = a

function Bramble._export_vtk(
        filename::AbstractString, Ωₕ::AbstractMeshType, fields::Pair...
)
    vtk = vtk_grid(filename, _vtk_axes(Ωₕ)...)
    for (name, data) in fields
        vtk[name] = _vtk_data(data)
    end
    return vtk_save(vtk)
end

# Warms `export_vtk` (the public entry point calling `_export_vtk` above) for a 1D and a 2D
# mesh, writing to a scratch directory removed once precompilation finishes. Only reachable
# once `WriteVTK` is loaded, so only this extension's own precompile pass reaches it.
if Bramble.PRECOMPILE_WORKLOAD
    @setup_workload begin
        Ωₕ1 = mesh(domain(interval(0.0, 1.0)), 5, true)
        Ωₕ2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (4, 4), (true, true))
        W1 = gridspace(Ωₕ1)
        W2 = gridspace(Ωₕ2)
        u1 = Rₕ(W1, x -> x[1])
        u2 = Rₕ(W2, x -> x[1] * x[2])

        @compile_workload begin
            mktempdir() do dir
                export_vtk(joinpath(dir, "pc1"), Ωₕ1, "u" => u1)
                export_vtk(joinpath(dir, "pc2"), u2)
            end
        end
    end
end

end
