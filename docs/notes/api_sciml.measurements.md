# Measurements moved out of `docs/src/api_sciml.md`

## Apple Accelerate: speedup

**Speedup.** Against `A \ F` -- what `:default` did before Accelerate existed -- a
symmetric Poisson-plus-mass system is a 1.2-1.3x win (0.78-0.83x the runtime, measured at
`n = 80` and `n = 120`) and an unsymmetric convection-diffusion system is now exactly
1.00x, because `:default` no longer routes it to Accelerate at all
(gpena/Bramble.jl#246, integrator re-measurement; one run of `benchmark/accelerate_solvers.jl`
on one machine, not a tracked baseline). Earlier, broader factorisation-level numbers, also a
single unrecorded run, against `:suitesparse` (not `A \ F`) put sparse SPD Cholesky at 0.75-0.85x and
sparse symmetric LDLᵀ at 0.26-0.78x across `n = 40, 80, 160`, while unsymmetric LUTPP was
1.66-4.07x **slower** -- the reason `:default` never reaches Accelerate for an unsymmetric
system.

## JuliaSparse ecosystem evaluation (intro, as it stood)

## JuliaSparse ecosystem evaluation

[gpena/Bramble.jl#244](https://github.com/gpena/Bramble.jl/issues/244) asked whether other
packages in the [JuliaSparse](https://github.com/JuliaSparse) organization and its
neighbours are worth adopting for assembly, direct solves, or iterative preconditioning.
Each candidate below was installed on Julia 1.12 and measured directly against Bramble's
own functions -- never a synthetic microbenchmark standing in for them (see
`bramble-verification`) -- so a "no" here is a measured "no", not a guess. Numbers are a
single run on one machine, not a tracked baseline; treat them as directional.

## JuliaSparse ecosystem evaluation (details)

### Assembly & storage formats

**`SparseMatricesCOO.jl` is not adopted.** Bramble's own assembly skips the triplet stage
after the first pass: `allocate_system_matrix` walks the stencil once (`_CoordSink`, documented
in [Forms](internals/form.md)) to build the pattern with `sparse!`, and every later `assemble!`
into that matrix writes straight into `nzval`, never building `(I, J, V)` again. `assemble`
itself calls `allocate_system_matrix`, so it rebuilds the pattern on every call. Measured on
a 2D 60×60 Poisson system (`n = 3600`, `nnz = 17760`):

| Path | Time |
|:--- |:--- |
| Bramble `assemble` (first call, builds the pattern) | 0.37 ms |
| Bramble `assemble` (repeat call, rebuilds the pattern) | 0.37 ms |
| `Base.sparse(I, J, V)` on the identical triplets | 0.06 ms |
| `SparseMatricesCOO.jl` COO→CSC on the identical triplets | 203 **seconds** |

`SparseMatricesCOO.jl` defines no specialised `SparseMatrixCSC(::SparseMatrixCOO)`
constructor, so the conversion falls through to Julia's generic dense-iteration
`AbstractMatrix` fallback -- an `O(m \cdot n \cdot \mathrm{nnz})` scan through every
`getindex`, itself an `O(\mathrm{nnz})` linear search of the triplet arrays (confirmed by
reading `SparseMatricesCOO.jl`'s source, not assumed from the number alone). The package
is designed by [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers) as an
NLP-solver interop format (handing Jacobian/Hessian triplets to IPOPT-style solvers that
want COO directly), not as a fast intermediate for building a `SparseMatrixCSC` -- the
wrong tool for what this issue asked it to do here. Bramble's first assembly is already
about as fast as its thousandth, which is the actual bar a triplet library would need to
clear.

**`SymRCM.jl`**: evaluated under reordering, below -- not for assembly.

### Tensor-compiler assembly

**Finch.jl: not adopted.** [gpena/Bramble.jl#217](https://github.com/gpena/Bramble.jl/issues/217)
asked whether a `@finch`-compiled loop nest -- [Finch.jl](https://github.com/finch-tensor/Finch.jl)'s
domain-specific compiler for structured and sparse tensors -- beats Bramble's own
record/replay sink assembly by enough to justify a dedicated backend. The issue set its
own threshold: greater than 1.5x speedup or greater than 50% memory reduction on the resulting
object, at N = 1e6.

`benchmark/finch_assembly.jl` measured 1D, 2D and 3D Poisson and convection-diffusion forms, N
from 1e2 to 1e6 on non-uniform (seeded `rand!`) meshes: Bramble's first `assemble` (pattern
discovery plus fill) against Finch's first `@finch` build, Bramble's `assemble!` refill against
Finch's refill, `Base.summarysize` of the resulting matrix/tensor, and time to first execution
(TTFX). In that run (its output is not saved in the repository; rerun the script to reproduce it),
every one of the 18 (dimension, form, size) cases produced a Finch tensor identical to
Bramble's matrix to `1e-12` -- both are built from the same non-uniform-mesh `(row, col, value)`
triplets, `findnz` on the CSC matrix Bramble already assembled -- which is what makes the timing
comparison meaningful rather than a comparison between two different answers. The six N = 1e6
cases, the ones the adoption rule is evaluated against:

| Dim | Form | Bramble refill (ms) | Bramble size (MiB) | Finch refill (ms) | Finch size (MiB) | Refill speedup | Size Δ% | Match |
|:--- |:--- |:--- |:--- |:--- |:--- |:--- |:--- |:--- |
| 1 | Poisson | 4.372 | 91.553 | 12.383 | 71.63 | 0.35 | 21.8 | true |
| 1 | Convection-diffusion | 6.101 | 129.7 | 10.373 | 71.63 | 0.59 | 44.8 | true |
| 2 | Poisson | 7.504 | 175.43 | 13.34 | 135.63 | 0.56 | 22.7 | true |
| 2 | Convection-diffusion | 12.711 | 251.709 | 12.239 | 135.63 | 1.04 | 46.1 | true |
| 3 | Poisson | 15.008 | 258.713 | 16.19 | 135.63 | 0.93 | 47.6 | true |
| 3 | Convection-diffusion | 53.266 | 372.925 | 18.312 | 135.63 | 2.91 | 63.6 | true |

Only one of the six clears the bar, 3D convection-diffusion, with a 2.91x refill speedup and a 63.6%
smaller resident tensor. The rule needs at least two qualifying cases out of six, so the table
alone already falls short. Finch's own compilation cost settles it further: time to first
execution -- the very first `@finch` call in the process, before any warm-up -- was about 39
seconds against Bramble's 0.1 millisecond. A package whose users open a REPL, run one assembly
and look at the result cannot pay a 39-second tax on the first call, even for a backend that
eventually wins on refills. Nothing in `ext/` was written: `finch_backend` and
`BrambleFinchExt.jl` from the issue's proposed architecture do not exist.

Two things bound how far this "no" reaches. First, a methodology departure recorded in
`benchmark/finch_assembly.jl`'s own header: the script hands Finch the `(row, col, value)`
triplets Bramble's assembly already computed and times only how fast a `@finch` loop nest copies
them into a `Tensor(Dense(SparseList(Element(0.0))))` -- the insertion half of assembly, not the
fused stencil-evaluation-and-insertion Finch's compiler actually promises and the issue's own
Problem Statement names as the point. That measures an upper bound favouring Finch, and Finch
still lost under it. Second, both the Bramble and the Finch runs were made on battery power
under heavy concurrent load, so the ratios in the table, not the absolute millisecond figures,
carry this decision.

What would change the answer: a fused evaluate-and-insert extension, where Finch compiles the
stencil evaluation itself from Bramble's own AST rather than consuming triplets Bramble already
produced, together with a way to amortise the roughly 39-second TTFX across precompilation
rather than a user's first call.

### Direct sparse solvers

**Sparspak.jl: done, not re-evaluated here.** Built in
[gpena/Bramble.jl#247](https://github.com/gpena/Bramble.jl/issues/247); see
[Sparspak sparse direct solver (pure Julia)](@ref) above.

**`Pardiso.jl`: not adopted, for a licensing reason rather than a technical one.**
`Pardiso.jl` bridges to one of two backends, and neither is available without something
Bramble cannot bundle:
- Intel MKL PARDISO needs a separately installed MKL; at the time of the
  [gpena/Bramble.jl#244](https://github.com/gpena/Bramble.jl/issues/244) evaluation,
  `Pardiso.mkl_is_available()` was `false` on a plain Julia 1.12 environment, and constructing an `MKLPardisoSolver` threw
  `"MKL is not available"`.
- Panua (formerly the free academic) PARDISO needs a separately downloaded, licensed
  shared library; constructing a `PardisoSolver` threw `"Panua pardiso library was not
  loaded"`.

Both were reproduced directly (not assumed) on a fresh Julia 1.12 environment. This is the
same shape of blocker that closed
[gpena/Bramble.jl#245](https://github.com/gpena/Bramble.jl/issues/245) (`ThreadedSparseCSR.jl`)
as won't-fix: a real, verified dependency the package cannot satisfy on behalf of a user,
rather than missing integration work. A user who already holds an MKL or Panua license and
wants to use it can still call `Pardiso.jl` directly against `A`/`F` from
[`assemble`](@ref) -- nothing in Bramble stands in the way of that -- it is just not
something this package can wire up as a first-class `solver` option for everyone.

### Iterative solvers & preconditioners

Krylov methods are already reachable through `solve` with `solver =
KrylovJL_GMRES()` etc. (`BrambleSciMLExt`), and [`amg_preconditioner`](@ref) already covers
algebraic multigrid preconditioning. What #244 asked to evaluate is whether `ILUZero.jl` /
`IncompleteLU.jl` add anything beyond that. Measured on an unsymmetric 2D convection-diffusion
system (90×90 grid, `n = 8100`, diffusion `1\mathrm{e}{-2}` against unit advection in both
directions -- the convection-dominated regime the issue named), unrestarted GMRES to
`atol = rtol = 1\mathrm{e}{-10}`:

| Preconditioner | Time | Iterations | Converged |
|:--- |:--- |:--- |:--- |
| none | 51.4 ms | 179 | yes |
| AMG (`ruge_stuben`) | 6475.9 ms | 2000 (capped) | **no** |
| `IncompleteLU.jl` (τ = 0.01) | 19.6 ms | 95 | yes |
| `ILUZero.jl` (ILU(0)) | 4.5 ms | 18 | yes |

Classical algebraic multigrid assumes something close to an M-matrix and does not fail
gracefully once advection dominates diffusion this strongly -- it neither converges nor
finishes quickly here, which is a known limitation of `ruge_stuben`-style coarsening on
non-symmetric, convection-dominated operators, not a bug in `AlgebraicMultigrid.jl`.
`ILUZero.jl`'s zero-fill ILU(0), reusing `A`'s own sparsity pattern, is the clear winner:
about 11× fewer iterations and 11× less wall time than no preconditioner, and 4× less than
`IncompleteLU.jl`'s drop-tolerance variant, at a fraction of the setup cost either of the
others carries. Built as [`ilu_preconditioner`](@ref) in
[gpena/Bramble.jl#255](https://github.com/gpena/Bramble.jl/issues/255), mirroring
[`amg_preconditioner`](@ref)'s shape -- see "Zero-fill ILU preconditioning for convection-dominated
systems" above.

`Metis.jl`'s graph partitioning was evaluated under reordering, not as a preconditioner,
below.

### Fill-reducing reordering

Measured on a 3D 24×24×24 Poisson system (`n = 13824`, `nnz = 93312`), CHOLMOD Cholesky
factorization with three orderings:

| Ordering | Factor time | `nnz(L)` |
|:--- |:--- |:--- |
| CHOLMOD default (built-in AMD) | 25.0 ms | 2,147,132 |
| `Metis.jl` (nested dissection) | 20.1 ms | 1,654,868 |
| `SymRCM.jl` (Cuthill-McKee) | 53.3 ms | 4,768,508 |

`Metis.jl`'s nested-dissection ordering measurably beats CHOLMOD's own default AMD here --
about 20% less factorization time and 23% less fill on this 3D system (one run; the
ordering script is not in the repository). `SymRCM.jl` is worse on both counts: Cuthill-McKee minimises bandwidth, not fill,
and 3D discretizations are exactly where that distinction costs the most. Both orderings
reach `suitesparse_factorize`/`sparse_factorize` **today, with no new extension needed** --
[gpena/Bramble.jl#248](https://github.com/gpena/Bramble.jl/issues/248) already forwards a
`perm` keyword straight to CHOLMOD:

```julia
using Metis
import Bramble: suitesparse_factorize
perm, _ = Metis.permutation(A)
fact = suitesparse_factorize(A; sym = :spd, perm = Int.(perm))
```

`Metis.jl` is worth naming explicitly in the ordering documentation rather than building
anything further for it.

