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

Differentiation typically targets parameter sensitivities or solution fields rather than grid coordinates, so most of the geometry is a constant. Arithmetic operations, difference operators, and discrete inner products propagate sensitivities via standard Julia multiple dispatch. The geometric inputs that were measured, each one a cell of `test/space/ad_coverage.jl`:

| Input | ForwardDiff | Enzyme |
| :--- | :--- | :--- |
| mesh endpoints (`mesh(…, [0, b] × [0, 1])`) | works | works |
| strength of `dirac(x, strength)` | works | works |
| location of `dirac(x, strength)` | `MethodError`: `dirac` converts the point to `Float64` | works, measured under `CpuSerial` |
| datum and coefficients of the `polygon_spaces` of a polygon mesh | works | reverse works under the three policies, forward throws `MethodError`, with the mesh built outside the function ([gpena/Bramble.jl#1212](https://github.com/gpena/Bramble.jl/issues/1212)) |
| vertices of `polygon(V)` | `MethodError`: the exact `Rational{BigInt}` vertices cannot hold a `Dual` | throws `IllegalTypeAnalysisException` |
| parameter of `φ` in `levelset(φ)` | named `ArgumentError` ([gpena/Bramble.jl#1172](https://github.com/gpena/Bramble.jl/issues/1172)) | not run: no reliable verdict ([gpena/Bramble.jl#1211](https://github.com/gpena/Bramble.jl/issues/1211)) |

A polygon or level-set mesh is built outside a function Enzyme differentiates. Built inside, the exact `Rational` and `BigFloat` arithmetic of the polygon's grid makes Enzyme throw `IllegalTypeAnalysisException`, and the level-set mesh has no reliable verdict. ForwardDiff differentiates a polygon mesh built inside the function. [`ad_coverage.md`](ad_coverage.md) has the policies, the figures and the measurements (`ad_reductions/1171_geometric_inputs.md`).

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
Bilinear forms assembled with `assemble(a)` or refilled in-place with `assemble!(A, a)` produce sparse matrices whose entries match the form's coefficient type. With ForwardDiff loaded, `pde_solve(A, F)` on a `Dual` matrix or load is solved by `BrambleForwardDiffExt`. The extension factors the `Float64` values `A₀ = value.(A)` once, by the route the `Float64` solve of the same keywords takes, then solves every partial against that factor: the value is `x₀ = A₀⁻¹b₀` and partial `k` is `xₖ = A₀⁻¹(bₖ − Aₖx₀)`. Nested `Dual` levels reuse the same factor. A non-square `Dual` matrix throws an `ArgumentError`. The bordered solve (`mean_zero`, `gauge`) takes the same route on the bordered matrix `[A C; Cᵀ 0]`: the primal bordered system of the values is factored once, and each partial solves `[Fₖ − Aₖu₀; 0]` against that factor. `w` and `Z` are constants, and a `Dual` one throws an `ArgumentError`, as does a `RankOneUpdate` over a `Dual` matrix, whose `σ` takes the matrix's element type. A kept factorization has its own route, in the next section.

### Solver calls under each backend
The solver routes of v4.2.4, each a cell of `test/space/ad_coverage.jl`; [`ad_coverage.md`](ad_coverage.md) has the figures and the policies. The Enzyme column is `CpuSerial`. Under `CpuThreaded` and `CpuPolyester` the same cells work except `gmg_solve` forward mode, which aborts the process when batched ([gpena/Bramble.jl#1232](https://github.com/gpena/Bramble.jl/issues/1232)), and the Chebyshev row, which works there; [`ad_coverage.md`](ad_coverage.md) has the figures and the limit. "Throws" means a named error. Where a cell returns a wrong gradient, the page says so.

| Call | ForwardDiff | Enzyme |
| :--- | :--- | :--- |
| `pde_solve(A, F)` | works | works |
| `pde_solve(fact, F)` and `ldiv!(uₕ, fact, F)`, with `fact = sparse_factorize(A)` built outside the differentiated function and `F` active | works | works, both modes |
| `pde_solve(sparse_factorize(A), F)` with `A` active, the factorization built inside | works | throws `IllegalTypeAnalysisException` |
| `pde_solve(RankOneUpdate(A, σ, w, w), F)` with `A` and `F` active | throws `ArgumentError`, naming `RankOneUpdate` | works, both modes |
| `fdm_solve(a, F)` with `F` active | works | works, both modes |
| `fdm_solve(a, F)` with a coefficient of `a` active | throws `ArgumentError` | throws `ArgumentError` |
| `gmg_solve(build, Ωₕ, b)` with `b` active | works | works, both modes |
| `gmg_solve` with a scalar or a grid-function coefficient of `build` active | works | reverse works, forward throws `ArgumentError` |
| `chebyshev_preconditioner(a) \ x` with a coefficient of `a` active | works | throws `IllegalTypeAnalysisException` |

Where a call throws, this is what to call instead:

- A kept factorization with the matrix active: `pde_solve(A, F)`. Enzyme refuses the factorization by name, and ForwardDiff differentiates it.
- A `RankOneUpdate` over a `Dual` matrix under ForwardDiff: `pde_solve(A, F; mean_zero = w)`.
- `fdm_solve` with an active coefficient or mesh: `pde_solve(assemble(a, l)..., F)`.
- `gmg_solve` with an active coefficient under Enzyme forward mode: Enzyme reverse mode, ForwardDiff, or `pde_solve(assemble(build(gridspace(Ωₕ))), b)`.
- `fact \ F` is not routed, because each backend defines its own `\`: call `pde_solve(fact, F)`. `ldiv!(uₕ, factorize(A), F)` over a `Dual` matrix overflows inside the standard library's `factorize`, upstream of Bramble.

The limits behind those rows, each with the call to use:

- **Enzyme does not carry `Ā` through a kept factorization.** A factorization object has no shadow that can hold it, so a rule would need the factorization to keep `A` and a second store for its cotangent across two rules. `pde_solve(fact, F)` and `ldiv!(uₕ, fact, F)` differentiate in `F` only. A factorization with a shadow is refused by name, and so is a standard-library dense `lu` with an all-zero shadow. Differentiate `pde_solve(A, F)` with `A` active.
- **A factorization built inside the differentiated function is unsupported under Enzyme.** `sparse_factorize` infers a `Union` (across backends once AppleAccelerate loads), and Enzyme stops with `IllegalTypeAnalysisException`. Each fix measured, one concrete SuiteSparse type or an inference barrier, passes reverse mode and aborts the process in forward mode on an LLVM assertion. Build the factorization outside the function, or differentiate `pde_solve(A, F)`. Reverse mode also needs the factorization's adjoint solve, and a backend without one is refused by name.
- **`pde_solve(::RankOneUpdate, F)` under Enzyme** needs `K.A` to be a `SparseMatrixCSC` (any other type is refused by name) and refuses batched reverse mode.
- **`fdm_solve` under Enzyme** differentiates in `F` only:
  - forward mode has width 1;
  - reverse mode refuses a form with a non-symmetric term (advection), whose solve takes the generalised Schur route and has no adjoint here;
  - a `KroneckerLinearOperator` operand has no rule;
  - only a 2D or 3D scalar `BilinearForm` gets a rule, so ghost, 1D and composite forms have none, and Enzyme reports `fdm_solve`'s own `ArgumentError` for them.
- **`gmg_solve` under Enzyme** differentiates through an implicit rule on the finest matrix. For any form these limits refuse, differentiate `pde_solve(assemble(build(gridspace(Ωₕ))), b)`.
  - Reverse mode refuses a non-Hermitian form by name, and a form whose operator changed between the solve and the reverse pass (a `Ref` the form holds, set after the solve).
  - An active mesh is refused by name, so build the mesh outside the differentiated function. A mesh built inside a reverse loss is refused the same way.
  - A composite-space `build` stops in LLVM (`broken gc calling conv fix`) and raises no `ArgumentError`. `build` is an opaque closure type, so no rule signature can exclude it.
  - A ghost-space `build` with active coefficients is refused in both modes. A constant ghost-space `build` that captures a scalar is refused in reverse mode even with only `b` active.
  - Under `CpuThreaded` and `CpuPolyester`, a batched forward call through `gmg_solve` aborts the process on an LLVM assertion (`invertedReturn`, [gpena/Bramble.jl#1232](https://github.com/gpena/Bramble.jl/issues/1232)), before any rule runs. DI's gradient is batched at the number of parameters; width 1 works under `CpuThreaded` (a single parameter, or `DI.pushforward` with one tangent, which was not measured), and `CpuSerial` works at any width. It predates the coefficient rule.
  - Forward mode refuses any `build` whose shadow carries data. Under runtime activity Enzyme can pass a constant closure's primal, captured values included, as its shadow, so no rule can tell a constant `build` from an active one whose tangent equals its value. Three forward designs gave silent wrong tangents. Forward mode with only `b` active still works.
  - Forward mode over-refuses a constant number captured by value from a non-`const` global, and `gmg_mixed(0.3, <untyped global Ref>)`. Use a `const` global, a `Ref`, or a `Const` or `DI.Constant` argument. `DI.derivative` in a coefficient is refused too.
  - An active value stored into a shared mutable that `build` captures (a global `Ref`, a `const` global vector, a `Const` vector argument) gets 0 from Enzyme in both modes, with no Bramble rule involved. Close over the value instead.
- **Plain `Enzyme.Reverse` needs runtime activity** (`set_runtime_activity`). Without it `ldiv!` throws `EnzymeRuntimeActivityError`, and DI's plain `AutoEnzyme` throws `EnzymeMutabilityException` (see [`ad_coverage.md`](ad_coverage.md)).
- **Open follow-ups:** `gmg_solve` with `Dual`s of different tags ([gpena/Bramble.jl#1227](https://github.com/gpena/Bramble.jl/issues/1227)), and a CHOLMOD factorization into a `VectorElement` load, which fails with a `MethodError` before any AD is involved ([gpena/Bramble.jl#1228](https://github.com/gpena/Bramble.jl/issues/1228)).

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
