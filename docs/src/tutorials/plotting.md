```@meta
CurrentModule = Bramble
```

# Plotting directly

**What you will learn.** How to plot a grid function straight inside a Julia session, as a curve in 1D or a colour map in 2D, with Makie or Plots.jl.

**What you need first.** The [mesh tutorial](@ref tutorial_mesh) and the [space tutorial](@ref tutorial_space), for the mesh and grid space the plotted element lives on.

**Where next.** The [VTK export tutorial](vtk_export.md) writes the same fields to a file for ParaView.

[`export_vtk`](@ref) and [`export_pgfplots`](@ref) write a file for another tool to open. Sometimes a plot straight inside the current Julia session is what you want instead. Two package extensions provide it, and you need no code beyond loading a plotting package.

The code on this page is not executed as part of the documentation build: plotting
backends are heavy dependencies, and building the documentation should not need to install
one. Every call shown here was verified directly against a real backend before being
written down.

## Makie

Loading any Makie backend (`CairoMakie`, `GLMakie`, `WGLMakie`) makes `lines`, `scatter`,
`heatmap` and `contour` work directly on a [`VectorElement`](@ref):

```julia
using Bramble, CairoMakie

Ωₕ = mesh(domain(interval(0.0, 1.0)), 33, true)
Wₕ = gridspace(Ωₕ)
uₕ = Rₕ(Wₕ, sin)

lines(uₕ)      # a curve
scatter(uₕ)    # the same points, unconnected
```

`lines(uₕ)` draws the values of `uₕ` at the mesh points, so the plot is the discrete function itself, not a smooth curve fitted through it.

!!! tip "Try this"
    Replace `33` by `9` and plot again. The curve turns into a visible polyline through the nine mesh points, which is exactly what the grid function stores.

```julia
Ω2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (30, 30), (true, true))
W2 = gridspace(Ω2)
u2 = Rₕ(W2, x -> sin(x[1]) * x[2])

heatmap(u2)    # a flat colour map
contour(u2)    # contour lines
```

A composite element has no single reading as one curve or one grid: plot each of its
components separately:

```julia
Vₕ = Wₕ^Val(2)
vₕ = Rₕ(Vₕ, x -> (sin(π * x), cos(π * x)))

lines(components(vₕ)[1])
lines(components(vₕ)[2])
```

## Plots.jl

`RecipesBase` covers `Plots.jl` and anything else built on it, the same way:

```julia
using Bramble, Plots

plot(uₕ)     # a line by default, the same field as above
heatmap(u2)  # same 2D field as above
```

## What is not covered

Both extensions are scoped to 1D and 2D. Makie genuinely can render a true 3D volume
(`Makie.volume`), unlike PGFPlots, which cannot represent one at all, but that path is not
wired up here; a 3D [`VectorElement`](@ref) raises an `ArgumentError` naming
[`export_vtk`](@ref) as the tool for it, in both extensions.
