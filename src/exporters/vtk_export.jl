"""
	export_vtk(filename::AbstractString, Ωₕ::AbstractMeshType, fields::Pair...)
	export_vtk(filename::AbstractString, uₕ::VectorElement, name::AbstractString = "u")

Write `Ωₕ`, and any number of named fields over it, to a VTK rectilinear grid file
(`.vtr`).

Each entry in `fields` is `name => data`, where `data` is a [`VectorElement`](@ref) over a
grid space on `Ωₕ` — scalar or composite — or a plain array already shaped like the grid.
The second method is a shorthand for a single field, named `"u"` unless told otherwise.

A 1D mesh gets a degenerate second axis rather than being refused: VTK has no dedicated 1D
grid type, but a rectilinear grid one point deep in `y` opens and renders correctly.

Requires [WriteVTK.jl](https://github.com/JuliaVTK/WriteVTK.jl) — `using WriteVTK` before
calling this.

# Examples

```julia
using Bramble, WriteVTK

Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (20, 20), (true, true))
Wₕ = gridspace(Ωₕ)
uₕ = Rₕ(Wₕ, x -> sin(x[1]) * x[2])

export_vtk("solution", Ωₕ, "u" => uₕ)   # writes solution.vtr
export_vtk("solution", uₕ)              # the same field, named "u"
```
"""
function export_vtk(filename::AbstractString, Ωₕ::AbstractMeshType, fields::Pair...)
    return _export_vtk(filename, Ωₕ, fields...)
end

function export_vtk(
        filename::AbstractString, uₕ::VectorElement, name::AbstractString = "u")
    return export_vtk(filename, mesh(uₕ), name => uₕ)
end

# Errors by default, same idiom as `metal_backend`/`_metal_backend`: a helpful message
# rather than a bare `MethodError` when the weak dependency has not been loaded. The
# `BrambleVTKExt` extension overrides this with the real implementation.
#
# `Ωₕ` is untyped here on purpose. `export_vtk` above already restricts it to
# `AbstractMeshType`, so nothing is given up by loosening it in this internal fallback —
# but the extension's method has to be a strict *specialization* of this one rather than an
# identical signature, or loading it overwrites a method during precompilation, which Julia
# refuses.
function _export_vtk(::AbstractString, ::Any, ::Pair...)
    error("export_vtk requires WriteVTK.jl. Add `using WriteVTK` before calling this " *
          "function.")
end
