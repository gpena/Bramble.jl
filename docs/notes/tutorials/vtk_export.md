```@meta
CurrentModule = Bramble
```

# Writing VTK files

**What you will learn.** How to write a grid function, or a whole time series, to a file that ParaView opens.

**What you need first.** The [mesh tutorial](@ref tutorial_mesh) and the [space tutorial](@ref tutorial_space), for meshes and grid functions, and the [form tutorial](@ref tutorial_form) if you want to export a solved problem.

**Where next.** The [PGFPlots export tutorial](pgfplots_export.md) writes a plain table for a LaTeX figure instead.

The last step after a solve is usually getting the result into a viewer. [`export_vtk`](@ref) writes a mesh and any number of named fields to a `.vtr` file, which ParaView and other VTK tools open.

It needs [WriteVTK.jl](https://github.com/JuliaVTK/WriteVTK.jl), a weak dependency. Run `using WriteVTK` before calling it; without it the call fails with a message that says so, not a bare `MethodError`. Every block below runs when this page is built and writes the files it claims to.

## 1. A mesh and a named field

`export_vtk` takes a filename, a mesh, and any number of `name => data` pairs:

```@example vtk
using Bramble, WriteVTK

Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (20, 20), (true, true))
Wₕ = gridspace(Ωₕ)
uₕ = Rₕ(Wₕ, x -> sin(x[1]) * x[2])

files = export_vtk(joinpath(mktempdir(), "solution"), Ωₕ, "u" => uₕ)
```

!!! tip "Try this"
    Change `(20, 20)` to `(6, 6)` and write the file again. Open it in ParaView and the same field shows as coarse cells, so you see the mesh behind the values.

`data` can be a [`VectorElement`](@ref), which is reshaped to match the grid the same way
`reshape` does, or a plain array already shaped that way. Passing more than one
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

An element over a composite space (`Wₕ^Val(2)` and the rest) writes as a single field with
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
anything: a stand-in to check that `export_vtk` gives a viewer one two-component
`velocity` vector alongside a one-component `pressure` scalar, which is what a coupled
solve's fields look like once assembled. Solving the system that produces them is the
[form tutorial](@ref tutorial_form)'s subject; this one is only about writing the result out once you
have it.

## 4. One dimension

VTK has no dedicated 1D grid type, so a 1D mesh gets a rectilinear grid one point deep in
`y`, which opens and renders correctly, rather than being refused:

```@example vtk
Ω1 = mesh(domain(interval(0.0, 1.0)), 33, true)
W1 = gridspace(Ω1)
f1 = Rₕ(W1, sin)

export_vtk(joinpath(mktempdir(), "curve"), f1)
nothing # hide
```

## 5. Time series for ParaView

A transient solve produces many snapshots, not one. Writing each with `export_vtk` above
gives ParaView a pile of unrelated `.vtr` files: no time slider, no animation, no correct
time axis if the steps are non-uniform. A ParaView collection (`.pvd`) ties them together.

`export_vtk(f, filename)` opens the collection, calls `f` with a handle `pvd`, and closes it
whether `f` returns normally or throws -- an interrupted run still leaves a valid, readable
partial animation rather than truncated XML. Assign into `pvd` with the same
`(filename, Ωₕ, name => data, ...)` shape `export_vtk` itself takes, once per time value:

```@example vtk
Ω1 = mesh(domain(interval(0.0, 1.0)), 41, true)
W1 = gridspace(Ω1)
dir = mktempdir()

export_vtk(joinpath(dir, "wave")) do pvd
    for (i, t) in enumerate((0.0, 0.1, 0.3, 1.0))  # non-uniform, on purpose
        uₕ = Rₕ(W1, x -> sin(x[1] - t))
        pvd[t] = (joinpath(dir, "step_$i"), Ω1, "u" => uₕ)
    end
end
nothing # hide
```

Opening `wave.pvd` in ParaView (`File > Open`, not the individual `.vtr` files) loads all
four steps as one dataset with a working time slider, each at its own recorded `t` --
`0.1` and `0.3` a step apart, `0.3` and `1.0` much further, exactly as given. Passing
`append = true` adds further steps to an existing collection instead of overwriting it.

A `SciMLBase` solution -- what solving an [`ode_problem`](@ref) with `OrdinaryDiffEq` hands
back -- has a one-call shorthand instead of a hand-written loop:

```@example vtk
using OrdinaryDiffEqBDF

Ωt = domain(interval(0.0, 1.0), :left => :left, :right => :right)
Ωₕt = mesh(Ωt, 41)
Wₕt = gridspace(Ωₕt)
fₕ = element(Wₕt, 0.0)

a = form(Wₕt, Wₕt, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
l = form(Wₕt, v -> innerₕ(fₕ, v))
bcs = dirichlet_constraints(Ωₕt, interval(0.0, 1.0), :boundary => (x, t) -> 0.0)
sd = semidiscretize(a, l; dirichlet = bcs)

prob = ode_problem(sd, Rₕ(Wₕt, x -> sinpi(x[1])), interval(0.0, 1.0))
sol = solve(prob, FBDF())

export_vtk(joinpath(mktempdir(), "solution"), Wₕt, sol)                    # every saved step
export_vtk(joinpath(mktempdir(), "solution"), Wₕt, sol; times = 0:0.1:1.0) # interpolated
nothing # hide
```

`Wₕt` (not just the mesh) is what turns each raw solution vector back into a properly shaped
field, the same way [`element`](@ref)`(Wₕ, ::AbstractVector)` does anywhere else. The
default writes exactly `sol`'s own saved times, non-uniform steps included; passing `times`
samples `sol`'s continuous interpolation at those points instead. See the
[heat equation example](../examples/heat_equation.md) for this run against a real solve.

## Where to go next

A full VTK file is usually more than a single plot going into a LaTeX document needs. For
that, the [PGFPlots export tutorial](pgfplots_export.md) writes the plain text table
`pgfplots` reads directly, with no external package at all. And for a plot inside the
current Julia session, with no file at all, see [Plotting directly](plotting.md).
