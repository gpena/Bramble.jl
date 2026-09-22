"""
    export_vtk(filename::AbstractString, Ωₕ::AbstractMeshType, fields::Pair...) -> Vector{String}
    export_vtk(filename::AbstractString, uₕ::VectorElement, name::AbstractString = "u") -> Vector{String}

Write `Ωₕ`, and any number of named fields over it, to a VTK rectilinear grid file
(`.vtr`).

Each entry in `fields` is `name => data`, where `data` is a [`VectorElement`](@ref) over a
grid space on `Ωₕ` (scalar or composite) or a plain array already shaped like the grid.
The second method is a shorthand for a single field, named `"u"` unless told otherwise.

A 1D mesh gets a degenerate second axis rather than being refused: VTK has no dedicated 1D
grid type, but a rectilinear grid one point deep in `y` opens and renders correctly.

Requires [WriteVTK.jl](https://github.com/JuliaVTK/WriteVTK.jl); call `using WriteVTK` before
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

See also `export_vtk(f::Function, filename)` below, for a time series of these.
"""
function export_vtk(filename::AbstractString, Ωₕ::AbstractMeshType, fields::Pair...)
    return _export_vtk(filename, Ωₕ, fields...)
end

function export_vtk(filename::AbstractString, uₕ::VectorElement, name::AbstractString = "u")
    return export_vtk(filename, mesh(uₕ), name => uₕ)
end

# Errors by default, same idiom as `metal_backend`/`_metal_backend`: a helpful message
# rather than a bare `MethodError` when the weak dependency has not been loaded. The
# `BrambleVTKExt` extension overrides this with the real implementation.
#
# `Ωₕ` is untyped here on purpose. `export_vtk` above already restricts it to
# `AbstractMeshType`, so nothing is given up by loosening it in this internal fallback,
# but the extension's method has to be a strict *specialization* of this one rather than an
# identical signature, or loading it overwrites a method during precompilation, which Julia
# refuses.
function _export_vtk(::AbstractString, ::Any, ::Pair...)
    return error(
        "export_vtk requires WriteVTK.jl. Add `using WriteVTK` before calling this " *
        "function.",
    )
end

# What `vtk_grid` for a rectilinear grid wants for its coordinate axes and what `vtk[name] =
# ...` wants for one field's data. Neither touches a WriteVTK type -- both only reshape what
# Bramble already has -- so they live here rather than in `BrambleVTKExt`, letting the
# `BrambleVTKSciMLExt` combined extension (WriteVTK + SciMLBase) share them without either
# extension depending on the other.
#
# `vtk_grid` wants at least two coordinate vectors, `z` defaulting to a single point when
# omitted. A 1D mesh gets a degenerate second axis for the same reason, built by hand since
# there is only one axis to pad.
_vtk_axes(Ωₕ::AbstractMeshType{1}) = (points(Ωₕ), [zero(eltype(Ωₕ))])
_vtk_axes(Ωₕ::AbstractMeshType) = points(Ωₕ)

# A scalar space gives an array shaped like the grid: `reshape(uₕ)` already reshapes a
# `VectorElement`'s flat storage that way, in the same column-major order `points(Ωₕ)`'s
# axes imply, so no permutation is needed. A composite space gives a `Tuple` of them:
# WriteVTK reads `length(data)` off a `Tuple` as the number of vector components, one array
# per component.
_vtk_data(uₕ::VectorElement{<:CompositeGridSpace}) = map(reshape, components(uₕ))
_vtk_data(uₕ::VectorElement) = reshape(uₕ)
_vtk_data(a::AbstractArray) = a

"""
    export_vtk(f::Function, filename::AbstractString; append::Bool = false) -> Vector{String}

Write a ParaView time-series collection (`.pvd`) alongside the `.vtr` file each step of `f`
writes, closing the collection when `f` returns *and* when it throws -- an interrupted run
leaves a valid, readable partial animation rather than truncated XML.

`f` is called with a collection handle `pvd`; assign into it with `pvd[t] = (step_filename,
Ωₕ, fields...)`, the same `(filename, Ωₕ, name => data, ...)` shape [`export_vtk`](@ref)
itself takes, once per time value `t`. Steps do not need a uniform spacing: `t` is recorded
exactly as given, which is what an adaptive `OrdinaryDiffEq` integrator's own saved times
already are.

# Keywords
- `append`: `true` to append new steps to an existing `.pvd` at `filename` rather than
  overwrite it (default `false`).

Requires [WriteVTK.jl](https://github.com/JuliaVTK/WriteVTK.jl); call `using WriteVTK` before
calling this.

# Examples

```julia
using Bramble, WriteVTK

Ωₕ = mesh(domain(interval(0.0, 1.0)), 41, true)
Wₕ = gridspace(Ωₕ)

export_vtk(joinpath(mktempdir(), "wave")) do pvd
    for (i, t) in enumerate((0.0, 0.3, 1.0))
        uₕ = Rₕ(Wₕ, x -> sin(x[1] - t))
        pvd[t] = ("step_\$(lpad(i, 3, '0'))", Ωₕ, "u" => uₕ)
    end
end
```

See also `export_vtk(filename, Ωₕ, fields...)` above, for a single snapshot, and
`export_vtk(filename, Wₕ, sol)` below, for a one-call `SciMLBase` solution export.
"""
function export_vtk(f::Function, filename::AbstractString; append::Bool = false)
    return _export_vtk_collection(f, filename; append = append)
end

# Same fallback idiom as `_export_vtk` above: the extension specializes the second,
# `::Any`-typed argument down to `AbstractString`, a strict specialization rather than an
# identical signature.
function _export_vtk_collection(::Function, ::Any; kwargs...)
    return error(
        "export_vtk requires WriteVTK.jl. Add `using WriteVTK` before calling this " *
        "function.",
    )
end

"""
    export_vtk(filename::AbstractString, Wₕ::AbstractSpaceType, sol;
               name::AbstractString = "u", times = nothing) -> Vector{String}

Write a `SciMLBase` ODE solution `sol` -- the result of solving an [`ode_problem`](@ref) --
as a ParaView time-series collection (`.pvd`), one `.vtr` step per saved (or interpolated)
time.

`sol.u[i]` is a plain coefficient vector with no shape of its own; `Wₕ`, the space the
problem was built over, is what turns it back into a properly reshaped field, the same way
[`element`](@ref)`(Wₕ, ::AbstractVector)` does anywhere else.

# Keywords
- `name`: the field name written at every step (default `"u"`). One call writes one field;
  for several named fields per step (solution, coefficient, error, ...), use
  `export_vtk(f::Function, filename)` directly.
- `times`: `nothing` (default) writes exactly `sol`'s own saved steps, at their true,
  possibly non-uniform values -- the common case for an adaptive integrator. Given an
  iterable of times instead, each is sampled from `sol`'s continuous interpolation
  (`sol(t)`).

Requires [WriteVTK.jl](https://github.com/JuliaVTK/WriteVTK.jl) and
[SciMLBase.jl](https://github.com/SciML/SciMLBase.jl) (pulled in by any solver package, e.g.
`OrdinaryDiffEq`); call `using WriteVTK` before calling this.

# Examples

```julia
using Bramble, WriteVTK, OrdinaryDiffEqBDF

prob = ode_problem(sd, u₀, I)
sol = solve(prob, FBDF())

export_vtk(joinpath(mktempdir(), "solution"), Wₕ, sol)                       # every saved step
export_vtk(joinpath(mktempdir(), "solution"), Wₕ, sol; times = 0:0.1:1.0)     # interpolated
```

See also [`ode_problem`](@ref), [`element`](@ref).
"""
function export_vtk(
        filename::AbstractString, Wₕ::AbstractSpaceType, sol;
        name::AbstractString = "u", times = nothing
)
    return _export_vtk_solution(filename, Wₕ, sol; name = name, times = times)
end

# Same fallback idiom again: the extension specializes the third, `::Any`-typed argument
# down to `SciMLBase.AbstractODESolution`.
function _export_vtk_solution(::AbstractString, ::AbstractSpaceType, ::Any; kwargs...)
    return error(
        "export_vtk for a solution object requires WriteVTK.jl and SciMLBase.jl. Add " *
        "`using WriteVTK` and a solver package (e.g. `OrdinaryDiffEq`) before calling " *
        "this function.",
    )
end
