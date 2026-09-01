# Plotting directly

`export_vtk` and `export_pgfplots` write a file for another tool to open. Sometimes a plot
straight inside the current Julia session is what is wanted instead — `Bramble.jl` supports
that too, through two package extensions that need no code beyond loading a plotting
package.

The code on this page is not executed as part of the documentation build — plotting
backends are heavy dependencies, and building the documentation should not need to install
one. Every call shown here was verified directly against a real backend before being
written down.

## Makie

Loading any Makie backend — `CairoMakie`, `GLMakie`, `WGLMakie` — makes `lines`, `scatter`,
`heatmap` and `contour` work directly on a [`VectorElement`](@ref):

```julia
using Bramble, CairoMakie

Ωₕ = mesh(domain(interval(0.0, 1.0)), 33, true)
Wₕ = gridspace(Ωₕ)
uₕ = Rₕ(Wₕ, sin)

lines(uₕ)      # a curve
scatter(uₕ)    # the same points, unconnected
```

```julia
Ω2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (30, 30), (true, true))
W2 = gridspace(Ω2)
u2 = Rₕ(W2, x -> sin(x[1]) * x[2])

heatmap(u2)    # a flat colour map
contour(u2)    # contour lines
```

A composite element has no single reading as one curve or one grid — plot each of its
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

plot(uₕ)     # a line by default, matching Rₕ! above
heatmap(u2)  # same 2D field as above
```

## What is not covered

Both extensions are scoped to 1D and 2D. Makie genuinely can render a true 3D volume
(`Makie.volume`) — unlike PGFPlots, which cannot represent one at all — but that path is not
wired up here; a 3D [`VectorElement`](@ref) raises an `ArgumentError` naming
[`export_vtk`](@ref) as the tool for it, in both extensions.
