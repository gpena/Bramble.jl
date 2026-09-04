# Writing PGFPlots data files

`export_vtk` writes a full grid for a viewer like ParaView. For a plot going straight into a
LaTeX document, that is usually more than is wanted: `export_pgfplots` writes the plain
whitespace-separated table `pgfplots` reads directly with `\addplot table {...}`, needing no
package beyond `Base`. This tutorial covers:

1. A 1D curve, and more than one field in the same file.
2. A composite field, which becomes one column per component.
3. A 2D surface, and the blank lines that make it one.
4. What is refused, and why.

Every code block below was run before being written down, and each produces the file it
claims to.

## 1. A 1D curve

`export_pgfplots` takes a filename, a mesh, and any number of `name => data` pairs (each
pair becomes a column):

```@example pgf
using Bramble

Ωₕ = mesh(domain(interval(0.0, 1.0)), 33, true)
Wₕ = gridspace(Ωₕ)
uₕ = Rₕ(Wₕ, sin)

f = export_pgfplots(joinpath(mktempdir(), "curve"), Ωₕ, "u" => uₕ)
read(f, String) |> s -> s[1:60] * "..."
```

The first line is a header naming the columns, `x` and then each field in order; read it
with `\addplot table {curve.dat};`, or pick a column explicitly with
`\addplot table[x=x, y=u] {curve.dat};`, which matters once a file has more than one field:

```@example pgf
vₕ = Rₕ(Wₕ, x -> x^2)
export_pgfplots(joinpath(mktempdir(), "two_fields"), Ωₕ, "u" => uₕ, "v" => vₕ)
nothing # hide
```

A lone [`VectorElement`](@ref) can skip naming the mesh, the same shorthand `export_vtk` has:

```@example pgf
export_pgfplots(joinpath(mktempdir(), "shorthand"), uₕ)
nothing # hide
```

## 2. A composite field

A table's columns have no way to group themselves the way a VTK field's component count
does, so a composite element expands into one column per component, named `name_1`,
`name_2`, and so on:

```@example pgf
Vₕ = Wₕ^Val(2)
cₕ = Rₕ(Vₕ, x -> (sin(π * x), cos(π * x)))

f2 = export_pgfplots(joinpath(mktempdir(), "comp"), Ωₕ, "u" => cₕ)
readlines(f2)[1]
```

## 3. A 2D surface

On a 2D mesh, `export_pgfplots` writes `x y z` triples for `\addplot3[surf] table {...}` or
`[mesh]`. Points are grouped into scan lines (one run of constant `x`) separated by a
**blank line**, which is what tells `pgfplots` where one row of the grid ends and the next
begins:

```@example pgf
Ω2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 4), (true, true))
W2 = gridspace(Ω2)
u2 = Rₕ(W2, x -> sin(x[1]) * x[2])

f3 = export_pgfplots(joinpath(mktempdir(), "surf"), Ω2, "u" => u2)
read(f3, String)
```

Miss those blank lines (writing every point as one long unbroken list) and the same numbers
plot as a shredded zigzag instead of a surface, because `pgfplots` has no way to tell where
one row stops and the next starts.

## 4. What is refused

PGFPlots' surface format has no way to encode more than one value per `(x, y)` point, so a
second field in one call, or a composite field, both raise an `ArgumentError` naming the
reason rather than silently writing one of several fields:

```@example pgf
try
    export_pgfplots(joinpath(mktempdir(), "bad"), Ω2, "u" => u2, "v" => u2)
catch e
    println(e)
end
```

And `pgfplots`' own `\addplot3[surf]` plots a height field over a 2D domain, not a true 3D
volume, so a 3D mesh is refused outright, pointing at [`export_vtk`](@ref) instead:

```@example pgf
Ω3 = mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (3, 3, 3), (true, true, true))
W3 = gridspace(Ω3)
u3 = Rₕ(W3, x -> x[1])
try
    export_pgfplots(joinpath(mktempdir(), "bad3d"), Ω3, "u" => u3)
catch e
    println(e)
end
```
