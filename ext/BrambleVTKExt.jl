module BrambleVTKExt

using Bramble:
               Bramble,
               AbstractMeshType,
               export_vtk,
               domain,
               interval,
               ×,
               mesh,
               gridspace,
               Rₕ,
               _vtk_axes,
               _vtk_data

using WriteVTK: WriteVTK, vtk_grid, vtk_save, paraview_collection
using PrecompileTools: @setup_workload, @compile_workload

function Bramble._export_vtk(
        filename::AbstractString, Ωₕ::AbstractMeshType, fields::Pair...
)
    vtk = vtk_grid(filename, _vtk_axes(Ωₕ)...)
    for (name, data) in fields
        vtk[name] = _vtk_data(data)
    end
    return vtk_save(vtk)
end

# Wraps the `WriteVTK.CollectionFile` (a type WriteVTK does not make public) so that
# `pvd[t] = (filename, Ωₕ, fields...)` can be a `Base.setindex!` method without committing
# type piracy: neither `CollectionFile` nor `Tuple` belongs to Bramble, so the method needs a
# Bramble-owned first argument to be legal. Left untyped rather than importing the
# non-public `CollectionFile` name just to annotate a field nothing dispatches on. Not
# exported -- the user only ever sees it as the `pvd` argument of their own `do`-block.
struct _VTKCollection
    pvd
end

function Base.setindex!(coll::_VTKCollection, entry::Tuple, t::Real)
    filename, Ωₕ, fields... = entry
    vtk = vtk_grid(filename, _vtk_axes(Ωₕ)...)
    for (name, data) in fields
        vtk[name] = _vtk_data(data)
    end
    coll.pvd[t] = vtk
    return coll
end

# `paraview_collection(filename; append) do pvd ... end` already closes over `try`/`finally`
# and calls `vtk_save` on both a normal return and an exception (verified by hand: an error
# raised mid-loop still leaves a `.pvd` with every step assigned before it, and one raised
# before any assignment still leaves a valid empty `<Collection/>`) -- nothing further to
# add here beyond wrapping the raw `CollectionFile` `f` receives.
function Bramble._export_vtk_collection(f::Function, filename::AbstractString; append::Bool = false)
    return paraview_collection(filename; append = append) do pvd
        f(_VTKCollection(pvd))
    end
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

                export_vtk(joinpath(dir, "pc3")) do pvd
                    for (i, t) in enumerate((0.0, 0.5, 1.0))
                        uₜ = Rₕ(W2, x -> t * x[1] * x[2])
                        pvd[t] = (joinpath(dir, "pc3_$i"), Ωₕ2, "u" => uₜ)
                    end
                end
            end
        end
    end
end

end
