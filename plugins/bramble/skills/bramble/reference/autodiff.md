# Automatic differentiation through Bramble

Bramble's operators, forms and assembly accept any element type, so derivatives flow through
them when the data carries the tracked type. Write code against DifferentiationInterface and
pick the backend:

| Backend | Status |
| --- | --- |
| ForwardDiff | works; with `AutoSparse` it gives sparse Jacobians |
| ReverseDiff, Mooncake | work |
| Enzyme | works, with the two settings below |
| Zygote | does not work: the operators mutate arrays (`setindex!`) |

## Rules

- **Allocate from the input's element type**, not from the space:
  `uₕ = element(Wₕ, T)` where `T = eltype(u_vec)`. A `Float64` buffer rejects
  `ForwardDiff.Dual`.
- **Assemble inside the function being differentiated** when the matrix depends on the
  unknown (a nonlinear coefficient). See `examples/poisson_nonlinear.jl` and
  `examples/coupled_reaction_diffusion.jl` for a Newton solve with a sparse Jacobian:

```julia
using DifferentiationInterface, ForwardDiff
import SparseConnectivityTracer, SparseMatrixColorings
ad = AutoSparse(AutoForwardDiff();
    sparsity_detector = SparseConnectivityTracer.TracerSparsityDetector(),
    coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm())
prep = prepare_jacobian(residual, ad, u)
J = DifferentiationInterface.jacobian(residual, prep, ad, u)       # structure once
DifferentiationInterface.jacobian!(residual, J, prep, ad, u)       # refill per iteration
```

- **Enzyme** needs runtime activity, and closures that capture a mesh or space must be
  marked constant, or it raises `EnzymeRuntimeActivityError` or `EnzymeMutabilityException`:

```julia
mode = Enzyme.set_runtime_activity(Enzyme.Reverse)
backend = AutoEnzyme(; mode = mode, function_annotation = Enzyme.Const)
```

  Passing the mesh or space as an argument instead of capturing it also avoids the second
  error.
- **Integer coefficients must be literals** inside a differentiated form, or Enzyme's type
  analysis fails (`IllegalTypeAnalysisException`): use `float(n)` or a `Ref`.

`pde_solve(A, F)` is `A \ F` with a reverse rule, for gradients of a quantity that depends on
a linear solve. Check any new gradient against finite differences once before relying on it.
