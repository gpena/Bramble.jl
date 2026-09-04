```@meta
CollapsedDocStrings = false
```

# Automatic differentiation

`Bramble.jl` is designed so that grid functions, operators, and form assemblies can be differentiated end-to-end using Julia's automatic differentiation (AD) ecosystem.

## Separation of geometry and coefficients

A grid function's coefficients and the coordinates of the underlying mesh are kept distinct:
- **Geometry**: Mesh coordinates, element spacings, and quadrature nodes remain in their native floating-point type (typically `Float64`).
- **Coefficients**: Grid functions ([`VectorElement`](@ref)) store values in whatever scalar type a computation requires, including `ForwardDiff.Dual` numbers or tracked values from reverse-mode packages.

Because differentiation typically targets parameter sensitivities or solution fields rather than grid coordinates, the mesh geometry remains undifferentiated. Arithmetic operations, difference operators, and discrete inner products propagate sensitivities via standard Julia multiple dispatch.

## Container and allocation rules

Propagating derivatives through discrete operators requires that allocations do not discard sensitivity types or revert to space defaults:

1. **Output allocation with `similar`**: Discrete operators allocate destination buffers using `similar(uₕ)` on their operand. This preserves the operand's coefficient type (such as `ForwardDiff.Dual`) rather than reverting to the grid space's default float type.
2. **Explicit value types with `element`**: When allocating a grid function for in-place operations during an AD sweep, the element type can be specified directly:
   ```julia
   uₕ = element(Wₕ, typeof(dual_val))
   ```
3. **Quadrature evaluation in `avgₕ`**: Cell averaging builds Gauss-Legendre quadrature rules in the coordinate type of the mesh, while evaluating the integrand at those real nodes. Sensitivities present in the integrand propagate into the destination element without requiring dual-valued quadrature tables.

## Supported automatic differentiation backends

Bramble's runtime kernels mutate destination buffers using `setindex!`. AD backends that support array mutation work out of the box:

- **ForwardDiff**: Dual numbers propagate through coefficients, difference operators, weak forms, and boundary conditions.
- **ReverseDiff**: Tracked arrays and scalars propagate through space operators and form assembly.
- **Mooncake & Enzyme**: Differentiate operator evaluations and space methods.
- **Zygote**: Not supported directly because it disallows array mutation (`setindex!`).

In practice, using [DifferentiationInterface.jl](https://github.com/JuliaDiff/DifferentiationInterface.jl) allows swapping between forward and reverse backends through a unified API. For setup instructions and runnable examples across all 5 supported backends, see the [Automatic differentiation tutorial](../tutorials/autodiff.md).

## Differentiating forms and linear systems

Sensitivities can appear in the differential operator, the source term, or the boundary data:

### Dual matrix systems
Bilinear forms assembled with `assemble(a)` or refilled in-place with `assemble!(A, a)` produce sparse matrices whose entries match the form's coefficient type. When solving `A \ F`, standard linear algebra solvers propagate sensitivities into the solution vector.

### Dirichlet boundary conditions
Boundary conditions constructed with `dirichlet_constraints` store raw closures without type erasure. When boundary data depends on a differentiated parameter:
```julia
bcs = dirichlet_constraints(set(Ωₕ), :boundary => (x -> a * sin(x[1])))
```
`dirichlet_bc!` writes sensitivity values directly into the target vector. When eliminating Dirichlet rows in `symmetrize!`, zero checks use `iszero(d)`, which tests both the nominal value and all partial derivatives. A boundary perturbation that evaluates nominally to zero with non-zero sensitivity is therefore correctly retained rather than skipped.

## Sparse Jacobians for nonlinear equations

For nonlinear PDEs, residuals $R(u) = A(u)u - F$ inherit the local stencil pattern of the discrete differential operators. The Jacobian $\partial R / \partial u$ has the same compact sparsity structure as the system matrix.

Using `DifferentiationInterface.jl` with `SparseConnectivityTracer.jl` and `SparseMatrixColorings.jl`, the sparsity pattern and column coloring are determined once. Each Newton step then evaluates the sparse Jacobian with a minimal number of directional derivatives. See the [Nonlinear Poisson example](../examples/poisson_nonlinear.md#Newton's-method) for a complete implementation.
