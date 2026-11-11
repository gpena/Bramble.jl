```@meta
CurrentModule = Bramble
```

# [Plot or export a grid function](@id howto_plotting_export)

**Goal.** Look at a grid function in the Julia session, or write it to a file for ParaView
or a LaTeX figure.

**You need.** A mesh and a grid function on it ([space tutorial](@ref tutorial_space)).

## Plot in the session with Makie

Loading a Makie backend (`CairoMakie`, `GLMakie` or `WGLMakie`) makes `lines`, `scatter`,
`heatmap`, `contour` and `surface` accept a [`VectorElement`](@ref). `lines` and `scatter`
take a 1D element and plot its values at the mesh points, so a coarse mesh shows as a
polyline. `heatmap`, `contour` and `surface` take a 2D element. Makie also exports `mesh`
and `lift`, so name Bramble's `mesh` before calling it:

```julia
using Bramble, CairoMakie
using Bramble: mesh

Ωₕ = mesh(domain(interval(0.0, 1.0)), 33, true)
uₕ = Rₕ(gridspace(Ωₕ), sin)

lines(uₕ)
scatter(uₕ)

Ω2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (30, 30), (true, true))
u2 = Rₕ(gridspace(Ω2), x -> sin(x[1]) * x[2])

heatmap(u2)
contour(u2)
```

A composite element has no single curve or grid: plot `components(vₕ)[1]` and
`components(vₕ)[2]` separately. A 3D element is not wired up. Both are refused with an
`ArgumentError`, which `lines` shows wrapped in a Makie `ResolveException`. The 3D message
names [`export_vtk`](@ref).

## Write a VTK file for ParaView

[`export_vtk`](@ref) writes a mesh and named fields to a `.vtr` file and returns the file
names. It needs [WriteVTK.jl](https://github.com/JuliaVTK/WriteVTK.jl): run `using WriteVTK`
first, or the call fails with a message that says so.

```@example plotting_export
using Bramble, WriteVTK

Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (20, 20), (true, true))
Wₕ = gridspace(Ωₕ)
uₕ = Rₕ(Wₕ, x -> sin(x[1]) * x[2])
vₕ = Rₕ(Wₕ, x -> x[1] + x[2])

files = export_vtk(joinpath(mktempdir(), "two_fields"), Ωₕ, "u" => uₕ, "v" => vₕ)
xml = read(only(files), String)
@assert endswith(only(files), ".vtr")
@assert occursin("Name=\"u\"", xml) && occursin("Name=\"v\"", xml)
@assert !occursin("Name=\"w\"", xml)
files
```

A lone element carries its mesh, so `export_vtk(filename, uₕ)` writes it as the field
`"u"`. Pass the name as a third argument to change it. An element over a composite space
such as `Wₕ^Val(2)` is written as one field with one component per block, so a velocity
and a pressure share a file:

```@example plotting_export
velocity = Rₕ(Wₕ^Val(2), x -> (sin(π * x[1]) * cos(π * x[2]), -cos(π * x[1]) * sin(π * x[2])))
pressure = Rₕ(Wₕ, x -> cos(2π * x[1]) * cos(2π * x[2]))

f = only(export_vtk(joinpath(mktempdir(), "fields"), Ωₕ, "velocity" => velocity, "pressure" => pressure))
xml = read(f, String)
@assert occursin("Name=\"velocity\"", xml) && occursin("NumberOfComponents=\"2\"", xml)
@assert occursin("Name=\"pressure\"", xml)
nothing # hide
```

VTK has no 1D grid type, so a 1D mesh is written as a rectilinear grid one point deep in `y`.

## Write a time series

Open a collection with `export_vtk(f, filename)`. `f` receives a handle `pvd`; assign
`pvd[t] = (step_filename, Ωₕ, name => data, ...)` once per time `t`. The collection is
closed when `f` throws too, so an interrupted run leaves a readable `.pvd`. Times are
recorded as given, uniform or not. `append = true` adds steps to an existing collection.

```@example plotting_export
Ω1 = mesh(domain(interval(0.0, 1.0)), 41, true)
W1 = gridspace(Ω1)
dir = mktempdir()
times = (0.0, 0.1, 0.3, 1.0)

export_vtk(joinpath(dir, "wave")) do pvd
    for (i, t) in enumerate(times)
        pvd[t] = (joinpath(dir, "step_$i"), Ω1, "u" => Rₕ(W1, x -> sin(x - t)))
    end
end

pvd_xml = read(joinpath(dir, "wave.pvd"), String)
@assert count("<DataSet ", pvd_xml) == length(times)
@assert all(occursin("timestep=\"$t\"", pvd_xml) for t in times)
@assert length(filter(endswith(".vtr"), readdir(dir))) == length(times)
nothing # hide
```

In ParaView, open `wave.pvd`, not the `.vtr` steps, to get the time slider.

The solution of an [`ode_problem`](@ref) has a one-call form. Pass the space the problem
was built over, which turns each stored vector back into a grid function. The default
writes the saved times, and `times = ...` samples the solution's interpolation instead:

```@example plotting_export
using OrdinaryDiffEqBDF

Ωt = domain(interval(0.0, 1.0), :left => :left, :right => :right)
Wt = gridspace(mesh(Ωt, 41))
a = form(Wt, Wt, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
l = form(Wt, v -> innerₕ(element(Wt, 0.0), v))
bcs = dirichlet_constraints(mesh(Wt), interval(0.0, 1.0), :boundary => (x, t) -> 0.0)
sd = semidiscretize(a, l; dirichlet = bcs)
sol = solve(ode_problem(sd, Rₕ(Wt, x -> sinpi(x[1])), interval(0.0, 1.0)), FBDF())

out = mktempdir()
export_vtk(joinpath(out, "saved"), Wt, sol)
export_vtk(joinpath(out, "sampled"), Wt, sol; times = 0:0.1:1.0)
@assert count("<DataSet ", read(joinpath(out, "saved.pvd"), String)) == length(sol.t)
@assert count("<DataSet ", read(joinpath(out, "sampled.pvd"), String)) == 11
nothing # hide
```

## Write a PGFPlots table for LaTeX

[`export_pgfplots`](@ref) writes a whitespace-separated text table that `pgfplots` reads
with `\addplot table`. It needs no package. On a 1D mesh the first line names the
columns, `x` and then each field:

```@example plotting_export
Ωc = mesh(domain(interval(0.0, 1.0)), 33, true)
Wc = gridspace(Ωc)
curve = Rₕ(Wc, sin)

f = export_pgfplots(joinpath(mktempdir(), "curve"), Ωc, "u" => curve, "v" => Rₕ(Wc, x -> x^2))
rows = readlines(f)
@assert split(rows[1]) == ["x", "u", "v"]
table = [parse(Float64, c) for r in rows[2:end] for c in split(r)]
cols = reshape(table, 3, :)
@assert size(cols, 2) == 33
@assert cols[2, :] ≈ sin.(cols[1, :]) && cols[3, :] ≈ cols[1, :] .^ 2
first(rows, 2)
```

Use `\addplot table[x=x, y=u] {curve.dat};` to pick a column. A composite field expands to
one column per component, named `u_1`, `u_2`, and so on.

On a 2D mesh the file holds `x y z` triples for `\addplot3[surf] table`. Points are
grouped in scan lines of constant `x`, and a blank line ends each one: it tells
`pgfplots` where a grid row ends.

```@example plotting_export
Ω2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 4), (true, true))
W2 = gridspace(Ω2)
u2 = Rₕ(W2, x -> sin(x[1]) * x[2])

f2 = export_pgfplots(joinpath(mktempdir(), "surf"), Ω2, "u" => u2)
text = read(f2, String)
@assert count("\n\n", text) == 4          # one blank line between the 5 scan lines
nothing # hide
```

A 2D file takes one scalar field. A second field or a composite field throws an
`ArgumentError`, and so does a 3D mesh: `pgfplots` draws a height field over a plane, not
a volume. Use [`export_vtk`](@ref) for those.

```@example plotting_export
Ω3 = mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (3, 3, 3), (true, true, true))
u3 = Rₕ(gridspace(Ω3), x -> x[1])
path = joinpath(mktempdir(), "refused")

for call in (() -> export_pgfplots(path, Ω2, "u" => u2, "v" => u2),
             () -> export_pgfplots(path, Ω3, "u" => u3))
    err = try
        call()
    catch e
        e
    end
    @assert err isa ArgumentError
end
nothing # hide
```
