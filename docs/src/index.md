# Bramble.jl

This documentation is for `Bramble.jl`, a Julia library implementing discretization methods to solve partial differential equations using finite differences on nonuniform grids.

For more information on the types of discretizations encompassed by `Bramble.jl`, please consult the papers
* J. A. Ferreira and R. D. Grigorieff, [On the supraconvergence of elliptic finite difference schemes](https://doi.org/10.1016/S0168-9274(98)00048-8), Applied Numerical Mathematics 28 (1998), pp. 275-292

* S. Barbeiro, J. A. Ferreira and R. D. Grigorieff, [Supraconvergence of a finite difference scheme for solutions in ``H^s(0,L)``](https://doi.org/10.1093/imanum/dri018), IMA Journal of Numerical Analysis 25.4 (2005), pp. 797–811

* J. A. Ferreira and R. D. Grigorieff, [Supraconvergence and Supercloseness of a Scheme for Elliptic Equations on Nonuniform Grids](https://doi.org/10.1080/01630560600796485), Numerical Functional Analysis and Optimization 27.5-6 (2006), pp. 539–564

## Precompilation

`Bramble.jl` ships a precompilation workload that exercises 1D, 2D and 3D meshes on
load. It costs a few seconds when the package is first built and cuts the time to
first result by roughly a factor of four.

While working on the package itself you may prefer faster rebuilds over faster first
use. To skip the workload:

```julia
using Preferences, Bramble
set_preferences!(Bramble, "precompile_workload" => false)
```

Julia tracks preferences in the precompilation cache, so the change takes effect on
the next `using Bramble` with no manual cache clearing. Restore the default with

```julia
delete_preferences!(Bramble, "precompile_workload"; force = true)
```

The setting is written to `LocalPreferences.toml` next to your active project.