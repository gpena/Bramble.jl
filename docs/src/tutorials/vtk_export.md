# Writing VTK files

Once a solution exists — the result of the [forms tutorial](form.md), or any grid function
— the last step is usually getting it into a viewer. `export_vtk` writes a mesh and any
number of named fields to a `.vtr` file, readable by ParaView or any other VTK-aware tool.
This tutorial covers:

1. Writing a mesh with a named field.
2. The shorthand for a single field.
3. A composite element as one vector field, not several scalar ones.
4. The 1D case.

`export_vtk` needs [WriteVTK.jl](https://github.com/JuliaVTK/WriteVTK.jl), which is a weak
dependency: `using WriteVTK` before calling it, or the call errors with a message that says
so rather than a bare `MethodError`.

Every code block below was run before being written down, and each produces the files it claims to.

## 1. A mesh and a named field

`export_vtk` takes a filename, a mesh, and any number of `name => data` pairs:

```@example vtk
using Bramble, WriteVTK

Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (20, 20), (true, true))
Wₕ = gridspace(Ωₕ)
uₕ = Rₕ(Wₕ, x -> sin(x[1]) * x[2])

files = export_vtk(joinpath(mktempdir(), "solution"), Ωₕ, "u" => uₕ)
```

`data` can be a [`VectorElement`](@ref) — which is reshaped to match the grid, the same way
[`to_matrix`](@ref) does — or a plain array already shaped that way. Passing more than one
pair writes more than one field into the same file:

```@example vtk
vₕ = Rₕ(Wₕ, x -> x[1] + x[2])
export_vtk(joinpath(mktempdir(), "two_fields"), Ωₕ, "u" => uₕ, "v" => vₕ)
nothing # hide
```

## 2. One field, without naming the mesh

A lone [`VectorElement`](@ref) already carries its mesh, so the field can be written
directly. The field is named `"u"` unless told otherwise:

```@example vtk
export_vtk(joinpath(mktempdir(), "shorthand"), uₕ)
nothing # hide
```

## 3. A composite element is one vector field

An element over a composite space — `Wₕ^Val(2)` and the rest — writes as a single field with
one component per block, rather than as separate scalar fields per component. This is the
shape of a Stokes solve's output: a vector velocity next to a scalar pressure, on the same
mesh, in one file.

```@example vtk
Vₕ = Wₕ^Val(2)
velocity = Rₕ(Vₕ, x -> (sin(π * x[1]) * cos(π * x[2]), -cos(π * x[1]) * sin(π * x[2])))
pressure = Rₕ(Wₕ, x -> cos(2π * x[1]) * cos(2π * x[2]))

export_vtk(joinpath(mktempdir(), "stokes"), Ωₕ, "velocity" => velocity, "pressure" => pressure)
nothing # hide
```

`velocity` is the classical divergence-free field
``(\sin(\pi x)\cos(\pi y),\, -\cos(\pi x)\sin(\pi y))``, not the result of solving
anything — a stand-in to check that `export_vtk` gives a viewer one two-component
`velocity` vector alongside a one-component `pressure` scalar, which is what a coupled
solve's fields look like once assembled. Solving the system that produces them is the
[forms tutorial](form.md)'s subject; this one is only about writing the result out once you
have it.

## 4. One dimension

VTK has no dedicated 1D grid type, so a 1D mesh gets a rectilinear grid one point deep in
`y` — which opens and renders correctly, rather than being refused:

```@example vtk
Ω1 = mesh(domain(interval(0.0, 1.0)), 33, true)
W1 = gridspace(Ω1)
f1 = Rₕ(W1, sin)

export_vtk(joinpath(mktempdir(), "curve"), f1)
nothing # hide
```

## Where to go next

A full VTK file is usually more than a single plot going into a LaTeX document needs. For
that, the [PGFPlots export tutorial](pgfplots_export.md) writes the plain text table
`pgfplots` reads directly, with no external package at all.
