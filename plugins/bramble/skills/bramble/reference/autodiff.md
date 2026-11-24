# Automatic differentiation through Bramble

Bramble's operators, forms and assembly accept any element type, so derivatives flow through
them when the data carries the tracked type. Write code against DifferentiationInterface and
pick one of the two supported backends:

| Backend | Status |
| --- | --- |
| ForwardDiff | forward; with `AutoSparse` it gives sparse Jacobians |
| Enzyme | reverse and forward, with the two settings below |

No other backend is supported. The stencil kernels mutate preallocated arrays, which rules
out Zygote; the ReverseDiff extension was removed in v4.2.0.

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

  When the matrix comes from a `BilinearForm`, `Bramble.ast_sparsity_detector(a, U -> U)`
  reads the pattern off the form and replaces the tracer.

- **Enzyme** needs runtime activity, and closures that capture a mesh or space must be
  marked constant, or it raises `EnzymeRuntimeActivityError` or `EnzymeMutabilityException`:

```julia
mode = Enzyme.set_runtime_activity(Enzyme.Reverse)   # or Enzyme.Forward
backend = AutoEnzyme(; mode = mode, function_annotation = Enzyme.Const)
```

  Passing the mesh or space as an argument instead of capturing it also avoids the second
  error.
- **Integer coefficients must be literals** inside a differentiated form, or Enzyme's type
  analysis fails (`IllegalTypeAnalysisException`): use `float(n)` or a `Ref`.
- **Threaded policies differentiate.** Both backends work under `CpuSerial`, `CpuThreaded`
  and `CpuPolyester`; Enzyme under `CpuPolyester` needs `using Polyester`.

## Linear solves

`pde_solve(A, F)` carries an Enzyme reverse and forward rule and a ChainRulesCore reverse
rule. With ForwardDiff, a `Dual` load is solved against one factorization of the `Float64`
matrix, and a `Dual` matrix goes to `pde_solve` as it is. Limits:

- Enzyme's reverse rule takes one direction at a time: a batched call (DI's reverse
  `jacobian` of a vector output) throws an `ArgumentError`. Differentiate a scalar with
  `gradient`.
- `pde_solve(sparse_factorize(A), F)` with `A` active fails under Enzyme; pass `A` directly.
- A non-square `Dual` matrix throws an `ArgumentError`.
- Enzyme's first call compiles for a long time (about 100 s for the `pde_solve` adjoint).

## Exact Jacobian without AD in assembly

`Bramble.jacobian_form` builds the Jacobian of `R(u) = A(u) u - F` as one form `a + ∂a` from
the AST. Declare each state-dependent coefficient as `values => (U -> L(U), f)`, with
`values == f.(L(u))`:

```julia
using ForwardDiff                      # differentiates each f
αx = element(Wₕ)
a = form(Wₕ, Wₕ, (U, V) -> inner₊(αx * ∇ₕ(U), ∇ₕ(V)))
jf = Bramble.jacobian_form(a, αx => (U -> Mₕ(U), α); dirichlet = :boundary)
J = Bramble.jacobian_prototype(jf)     # pattern of a + ∂a, can be wider than A's
Bramble.jacobian!(J, jf, u)            # J = ∂R/∂u at u, no Dual through assembly
```

A coefficient under a node with no derivative rule throws `Bramble.MissingJacobianRule`.
`semidiscretize(a, l; coefficients = (αx => (U -> Mₕ(U), α),))` does the same for the method
of lines (scalar grid spaces only); its residual then refuses a `Dual` state, so take the
Jacobian from `Bramble.jacobian!(J, sd, u, p, t)` instead of AD.

Check any new gradient against finite differences once before relying on it.
