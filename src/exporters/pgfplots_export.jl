# Extension of the coordinate for a filename with none, or one not already recognisable as
# plain-text tabular data. `pgfplots` itself does not care about the extension: it reads
# whatever `\addplot table {...}` is pointed at, and this is only for a sensible default.
function _pgf_filename(f::AbstractString)
    any(
        e -> endswith(f, e), (".dat", ".txt", ".tsv", ".csv")) ? f : f * ".dat"
end

# Expands one `name => data` pair into the column(s) it contributes: one for a scalar field,
# one per component (named `name_1`, `name_2`, ...) for a composite one, since a table's
# columns have no way to group themselves the way a VTK field's `NumberOfComponents` does.
function _pgf_columns(name, uₕ::VectorElement{<:CompositeGridSpace})
    [(string(name, "_", i), to_matrix(c)) for (i, c) in enumerate(components(uₕ))]
end
_pgf_columns(name, uₕ::VectorElement) = [(string(name), to_matrix(uₕ))]
_pgf_columns(name, a::AbstractVector) = [(string(name), a)]

# The single scalar field a 2D `surf`/`mesh` table can carry, reshaped to the grid if it is
# not already. A composite element has no reading as one z-value per (x, y) point, so it is
# refused here rather than silently taking one component.
function _pgf_error_composite(name)
    throw(ArgumentError(
        "export_pgfplots writes one scalar field per 2D file: pgfplots' surf/mesh format " *
        "has no way to encode more than one z-value per (x, y) point. \"$name\" is a " *
        "composite element; export each of its components(...) separately."))
end
_pgf_grid(name, uₕ::VectorElement{<:CompositeGridSpace}, dims) = _pgf_error_composite(name)
_pgf_grid(name, uₕ::VectorElement, dims) = to_matrix(uₕ)
function _pgf_grid(name, a::AbstractMatrix, dims)
    size(a) == dims || throw(ArgumentError(
        "\"$name\" has size $(size(a)), but the mesh has $dims points"))
    return a
end
_pgf_grid(name, a::AbstractVector, dims) = reshape(_pgf_grid_len(name, a, prod(dims)), dims)
function _pgf_grid_len(name, a, n)
    length(a) == n || throw(ArgumentError(
        "\"$name\" has length $(length(a)), but the mesh has $n points"))
    return a
end

"""
    export_pgfplots(filename::AbstractString, Ωₕ::AbstractMeshType{1}, fields::Pair...) -> String
    export_pgfplots(filename::AbstractString, Ωₕ::AbstractMeshType{2}, field::Pair) -> String
    export_pgfplots(filename::AbstractString, uₕ::VectorElement, name::AbstractString = "u") -> String

Write grid data to a plain-text table, laid out the way `pgfplots` reads it directly: no
external package needed, since the format is just whitespace-separated numbers.

On a 1D mesh, any number of named fields become columns: `x name₁ name₂ ...`, one row per
grid point, readable with `\\addplot table {file.dat}` or `\\addplot table[y=name] {file.dat}`
to pick one column by name. A composite [`VectorElement`](@ref) expands into one column per
component, named `name_1`, `name_2`, and so on.

On a 2D mesh, exactly one field is written as `x y z` triples (pgfplots' `surf`/`mesh`
format has no way to encode more than one value per point) with a blank line after each
run of constant `x`. That blank line is what tells `\\addplot3[surf] table {file.dat}` where
one row of the grid ends and the next begins; without it the same numbers plot as a
shredded zigzag instead of a surface. A composite element is refused with a message saying
so, rather than silently writing one of its components.

`data` in a field pair can be a [`VectorElement`](@ref), or a plain array already shaped
like the grid (a vector in 1D, a vector or a `(nx, ny)` matrix in 2D).

A 3D mesh is refused: `\\addplot3[surf]` plots a height field over a 2D domain, not a true
3D volume, so a 3D mesh has no faithful representation in this format. See
[`export_vtk`](@ref).

# Examples

```julia
using Bramble

Ωₕ = mesh(domain(interval(0.0, 1.0)), 33, true)
Wₕ = gridspace(Ωₕ)
uₕ = Rₕ(Wₕ, sin)
export_pgfplots("curve", Ωₕ, "u" => uₕ)   # writes curve.dat

Ω2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (20, 20), (true, true))
W2 = gridspace(Ω2)
v2 = Rₕ(W2, x -> sin(x[1]) * x[2])
export_pgfplots("surface", Ω2, "u" => v2)   # writes surface.dat, for \\addplot3[surf]
```
"""
function export_pgfplots(filename::AbstractString, Ωₕ::AbstractMeshType{1}, fields::Pair...)
    x = points(Ωₕ)
    n = length(x)

    names = String[]
    cols = Vector[]
    for (name, data) in fields
        for (nm, col) in _pgf_columns(name, data)
            length(col) == n || throw(ArgumentError(
                "\"$nm\" has length $(length(col)), but the mesh has $n points"))
            push!(names, nm)
            push!(cols, col)
        end
    end

    out = _pgf_filename(filename)
    open(out, "w") do io
        println(io, join(("x", names...), ' '))
        for i in 1:n
            println(io, join((x[i], (c[i] for c in cols)...), ' '))
        end
    end
    return out
end

function export_pgfplots(filename::AbstractString, Ωₕ::AbstractMeshType{2}, fields::Pair...)
    length(fields) == 1 || throw(ArgumentError(
        "export_pgfplots writes one scalar field per 2D file: pgfplots' surf/mesh format " *
        "has no way to encode more than one z-value per (x, y) point. Got " *
        "$(length(fields)) fields; call export_pgfplots once per field."))
    name, data = only(fields)

    x, y = points(Ωₕ)
    Z = _pgf_grid(name, data, (length(x), length(y)))

    out = _pgf_filename(filename)
    open(out, "w") do io
        println(io, "% x y ", name)
        for i in eachindex(x)
            for j in eachindex(y)
                println(io, x[i], ' ', y[j], ' ', Z[i, j])
            end
            i == lastindex(x) || println(io)
        end
    end
    return out
end

function export_pgfplots(
        ::AbstractString, ::AbstractMeshType{D}, ::Pair...) where {D}
    throw(ArgumentError(
        "export_pgfplots supports 1D and 2D meshes only. pgfplots' \\addplot3[surf] plots " *
        "a height field over a 2D domain, not a true 3D volume, so a $(D)D mesh has no " *
        "faithful representation in this format; use export_vtk instead."))
end

function export_pgfplots(
        filename::AbstractString, uₕ::VectorElement, name::AbstractString = "u")
    return export_pgfplots(filename, mesh(uₕ), name => uₕ)
end
