---
name: bramble
description: "Write, debug and speed up code that uses Bramble.jl, a finite-difference library for PDEs on structured (often non-uniform) Cartesian meshes: domains, meshes, grid spaces, discrete operators, variational forms, assembly, boundary conditions, time integration, autodiff and GPU backends. Use when the user's Julia code calls Bramble or they ask how to discretise a PDE with it."
---

# Writing code with Bramble.jl

Bramble discretises a PDE in six steps; each has one entry point:

| Step | Call | Detail |
| --- | --- | --- |
| geometry | `domain(interval(0.0, 1.0) × interval(0.0, 1.0), :inlet => :left)` | `reference/meshes-spaces.md` |
| points | `mesh(Ω, (33, 33), (true, false))` (uniform per axis, or not) | `reference/meshes-spaces.md` |
| unknowns | `gridspace(Ωₕ)`, `gridspace(Ωₕ)^Val(2)` for a vector field | `reference/meshes-spaces.md` |
| equation | `form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))` | `reference/operators.md`, `reference/forms-assembly.md` |
| matrix | `A, F = assemble(a, l; dirichlet = bcs)` | `reference/forms-assembly.md` |
| solution | `uₕ .= A \ F`, or `semidiscretize` and an ODE solver | `reference/backends-solvers.md` |

Worked programs, each ending in assertions against an exact solution or a conservation law.
Start from the closest one:

| Problem | File | Shows |
| --- | --- | --- |
| linear Poisson, 2D | `examples/poisson.jl` | random non-uniform mesh, refinement, convergence order |
| nonlinear Poisson, 1D | `examples/poisson_nonlinear.jl` | Picard with in-place reassembly; Newton with a sparse AD Jacobian |
| convection-diffusion, 2D | `examples/convection_diffusion.jl` | a first-order term, `Mₕ(u)` paired with `∇ₕ(v)` |
| heat equation, 1D | `examples/heat.jl` | `semidiscretize`, a time-dependent source, OrdinaryDiffEq |
| wave equation, 2D | `examples/wave.jl` | second-order semidiscretisation, energy conservation |
| linear elasticity, 2D | `examples/elasticity.jl` | a vector field in `Wₕ^Val(2)`, `εₕ`, `divₕ` |
| coupled reaction-diffusion | `examples/coupled_reaction_diffusion.jl` | two unknowns in one composite space, Newton |
| point sources | `examples/point_source.jl` | `dirac`, boundary flux with `Bramble.reaction` |

## Rules that prevent wrong or slow code

1. **Exported versus `public`.** `using Bramble` brings in exported names only. A `public`
   name (`CpuThreaded`, `D₊`, `mass_matrix`, `allocate_system_matrix`, `half_points`, ...)
   needs `Bramble.name` or `using Bramble: name`. The reference files mark them. `Parallel()`
   and `Serial()` are the exported policy names.
2. **Non-uniform meshes are the general case.** `mesh(Ω, n, false)` draws random points:
   call `Random.seed!(k)` before building it, never only before the assertion.
3. **Compare with an absolute floor.** Spacings on random meshes reach `1e-17`, so a
   relative-only check reports huge errors: `isapprox(a, b; atol = 1e-12, rtol = 1e-12)`.
4. **Say where an order of accuracy is measured.** On a non-uniform mesh `D₋ₓ` is first
   order at the nodes and second order at cell midpoints; the Poisson solve with
   `inner₊(∇ₕ(u), ∇ₕ(v))` converges at second order in `norm₁ₕ`.
5. **Integer coefficients in a form must be literals.** `2 * innerₕ(u, v)` is fine. A
   runtime `n::Int` makes the form's type depend on `n`'s value (zero and one fold away):
   pass `float(n)`, or a `Ref` to change it between assemblies.
6. **Build forms once, assemble many times.** `form` stores the expression; `assemble!` on a
   matrix from `Bramble.allocate_system_matrix(a)` refills it without allocating. A
   coefficient that changes in time is a grid function the form captures: overwrite its
   values (`Rₕ!(fₕ, f)`, `fill!(parent(αₕ), α)`), never rebuild the form in the loop.
7. **Composite spaces need distinguishable test data.** Give components values like 1, 100
   and 10000, never equal ones, and check each block, not the whole vector.
8. **Measure allocations inside a warmed function**, never with `@allocated` at top level
   over globals (it reports bytes that a function call does not allocate). See
   `reference/performance.md`.
9. **A backend's policy and storage must agree.** Host arrays take a CPU policy, device
   arrays (`metal_backend`) a GPU one, or construction throws. See `reference/gpu.md`.

## Notation

| Entity | Julia | Rule |
| --- | --- | --- |
| domain, mesh | `Ω`, `Ωₕ` | subscript `ₕ` marks discrete objects |
| grid spaces | `Wₕ` scalar, `Vₕ` vector | capital plus `ₕ` |
| grid functions | `uₕ`, `vₕ` | lowercase plus `ₕ`; components `uₕ(1)` or `uₓ, uᵧ = components(uₕ)` |
| trial and test functions in a form | `u`, `v` | no subscript |
| spacing | `hₘₐₓ(Ωₕ)`, `Bramble.hₘᵢₙ(Ωₕ)` | |

Directions are subscripts: `ₓ` (U+2093), `ᵧ` (U+1D67), `₂` (U+2082), and `ₕ` for all of them
(`D₋ₓ`, `inner₊ᵧ`, `∇ₕ`). A runtime direction is an argument: `∇ₕ[d]`, or
`Bramble.D₋(uₕ, d)` with `d` an `Int` or `Symbol`. Inside a form write `Val(d)`. Functions
ending in `!` write into their first argument and return it.

## Reference files

Read the one the task needs; each is self-contained.

- `reference/meshes-spaces.md`: domains and markers, meshes and their metric, refinement,
  grid spaces, elements, components.
- `reference/operators.md`: restriction `Rₕ`, cell averages `avgₕ`, interpolation `πₕ`,
  differences, jumps and averages, vector calculus, inner products and norms.
- `reference/forms-assembly.md`: bilinear and linear forms, `assemble` and its in-place
  variants, Dirichlet constraints, symmetrisation, point sources, the Kronecker path.
- `reference/backends-solvers.md`: execution policies and backends, exporters,
  `semidiscretize` and SciML problems, sparse factorisations and preconditioners.
- `reference/performance.md`: type stability, allocation measurement, in-place routines,
  scratch buffers, threads.
- `reference/autodiff.md`: which AD backends work through Bramble and how to set them up.
- `reference/gpu.md`: Metal and other device backends, and their traps.

The documentation's API pages are authoritative when this skill and the installed version
disagree: check with `?name` in the REPL.
