```@meta
CurrentModule = Bramble
CollapsedDocStrings = false
```

# Automatic differentiation

Grid functions, operators and form assemblies differentiate end to end through Julia's automatic differentiation (AD) ecosystem.

## Separation of geometry and coefficients

A grid function's coefficients and the coordinates of the underlying mesh are kept distinct.
- **Geometry** (mesh coordinates, element spacings and quadrature nodes) stays in its native floating-point type, typically `Float64`.
- **Coefficients** live in grid functions, which store values in whatever scalar type a computation requires, including `ForwardDiff.Dual` numbers or tracked values from reverse-mode packages. The [space tutorial](@ref space_composite) introduces vector elements.

Because differentiation typically targets parameter sensitivities or solution fields rather than grid coordinates, the mesh geometry remains undifferentiated. Arithmetic operations, difference operators, and discrete inner products propagate sensitivities via standard Julia multiple dispatch.

## Container and allocation rules

Propagating derivatives through discrete operators requires that allocations do not discard sensitivity types or revert to space defaults:

1. **Output allocation with `similar`**: Discrete operators allocate destination buffers using `similar(uₕ)` on their operand. This preserves the operand's coefficient type (such as `ForwardDiff.Dual`) rather than reverting to the grid space's default float type.
2. **Explicit value types with `element`**: When allocating a grid function for in-place operations during an AD sweep, the element type can be specified directly:
   ```julia
   uₕ = element(Wₕ, typeof(dual_val))
   ```
3. **Quadrature evaluation in `avgₕ`**: Cell averaging builds Gauss-Legendre quadrature rules in the coordinate type of the mesh, while evaluating the integrand at those real nodes. Sensitivities present in the integrand propagate into the destination element without requiring dual-valued quadrature tables.

## AD backends

Bramble's runtime kernels mutate destination buffers using `setindex!`, so a backend has to support array mutation. Two backends do, both reached through [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl). The [Automatic differentiation tutorial](../tutorials/autodiff.md) has the setup, runnable examples and the choice between them.

| Backend | `CpuSerial` | `CpuThreaded` | `CpuPolyester` |
| :--- | :--- | :--- | :--- |
| ForwardDiff | works, with limits | works, with limits | works, with limits |
| Enzyme | works, with limits | works, with limits | works, with limits |

[`ad_coverage.md`](ad_coverage.md) lists, per feature and policy, what each backend differentiates, what it excludes, and the limits.

ForwardDiff propagates dual numbers through coefficients, difference operators, weak forms and boundary conditions. Enzyme differentiates `Float64` code, so no buffer widens to `Dual`, and `BrambleEnzymeExt` supplies the `pde_solve` adjoint.

Other AD backends are not offered. The measurements and the full reasoning are in [`ad_backend_evaluation.md`](ad_backend_evaluation.md).

## Calling AD libraries

Calls that differentiate Bramble go through [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl) (DI), with three named exceptions. The evidence, the call census and the cost measurements are in [`ad_di_evaluation.md`](ad_di_evaluation.md).

The exceptions stay as they are:

- `test/ext/chainrules_enzyme_ext.jl` calls Enzyme directly. DI differentiates one argument, and the file needs `A` and `F` active together and calls Enzyme's own entry points so that the `pde_solve` rule fires.
- Hand-built `ForwardDiff.Dual`, `value` and `partials` sites stay. They are not differentiation calls, and ForwardDiff stays a test dependency.
- The closed-form calculus helpers in `docs/src/examples/elasticity_3d.jl` and `elasticity_traction_2d.jl` stay as they are, because they differentiate no Bramble code.

The extensions keep their backend-native rules (`ChainRulesCore.rrule` and the Enzyme rule), which DI picks up. `src/` takes no DI dependency.

The Enzyme failures belong to the backend, since direct calls fail the same way, so they are not an exception to this policy.

## Differentiating forms and linear systems

Sensitivities can appear in the differential operator, the source term, or the boundary data:

### Dual matrix systems
Bilinear forms assembled with `assemble(a)` or refilled in-place with `assemble!(A, a)` produce sparse matrices whose entries match the form's coefficient type. With ForwardDiff loaded, `pde_solve(A, F)` on a `Dual` matrix or load is solved by `BrambleForwardDiffExt`. The extension factors the `Float64` values `A₀ = value.(A)` once, by the route the `Float64` solve of the same keywords takes, then solves every partial against that factor: the value is `x₀ = A₀⁻¹b₀` and partial `k` is `xₖ = A₀⁻¹(bₖ − Aₖx₀)`. Nested `Dual` levels reuse the same factor. A non-square `Dual` matrix throws an `ArgumentError`. The bordered solve (`mean_zero`, `gauge`) takes the same route on the bordered matrix `[A C; Cᵀ 0]`: the primal bordered system of the values is factored once, and each partial solves `[Fₖ − Aₖu₀; 0]` against that factor. `w` and `Z` are constants, and a `Dual` one throws an `ArgumentError`, as does a `RankOneUpdate` over a `Dual` matrix, whose `σ` takes the matrix's element type. `pde_solve(fact, F)` keeps its route.

### Dirichlet boundary conditions
Boundary conditions constructed with `dirichlet_constraints` store raw closures without type erasure. When boundary data depends on a differentiated parameter:
```@example internals_ad
using Bramble

Ωₕ = mesh(domain(interval(0.0, 1.0)), 11)
a = 2.0
bcs = dirichlet_constraints(Ωₕ, :boundary => (x -> a * sin(x[1])))
```
`dirichlet_bc!` writes sensitivity values directly into the target vector. When eliminating Dirichlet rows in `symmetrize!`, zero checks use `iszero(d)`, which tests both the nominal value and all partial derivatives. A boundary perturbation that evaluates nominally to zero with non-zero sensitivity is therefore correctly retained rather than skipped.

## Sparse Jacobians for nonlinear equations

For nonlinear PDEs, residuals $R(u) = A(u)u - F$ inherit the local stencil pattern of the discrete differential operators. The Jacobian $\partial R / \partial u$ has the same compact sparsity structure as the system matrix.

Using `DifferentiationInterface.jl` with `SparseConnectivityTracer.jl` and `SparseMatrixColorings.jl`, the sparsity pattern and column coloring are determined once. Each Newton step then evaluates the sparse Jacobian with a minimal number of directional derivatives. See the [Nonlinear Poisson example](../examples/poisson_nonlinear.md#Newton's-method) for a complete implementation.
