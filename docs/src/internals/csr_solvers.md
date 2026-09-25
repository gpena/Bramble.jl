```@meta
CurrentModule = Bramble
```

# `SparseMatrixCSR` direct solve

`Bramble.jl`'s sparse direct solvers -- `sparse_factorize`, `pde_solve`, `refactor!`/
`sparse_refactor!` -- accept a `SparseMatrixCSR` (from
[`SparseMatricesCSR.jl`](https://github.com/gridap/SparseMatricesCSR.jl), a weak
dependency also used for CSR-backed [`gridspace`](@ref)s, `csr_backend()`) in addition to
the `SparseMatrixCSC` every backend is written against. This page records what that support
actually is, and, honestly, what it is not: gpena/Bramble.jl#275 named eight
solver/preconditioner ecosystems for "CSR support"; only one new code path was built, and
the rest are recorded below as evaluated-and-excluded, with reasons, rather than attempted.

## What's implemented

**A CSC-conversion fallback, nothing more.** Neither `SparseMatricesCSR.jl` itself nor
SuiteSparse, MUMPS, Sparspak, or Apple Accelerate define a native `ldiv!`, `factorize`, or
`\` that takes a `SparseMatrixCSR` -- confirmed directly by grepping the installed
`SparseMatricesCSR.jl` package source (zero such methods). There is no native CSR solve
path to integrate anywhere in this dependency stack. What `sparse_factorize(A::SparseMatrixCSR)`,
`pde_solve(A::SparseMatrixCSR, F)`, and `refactor!(fact, A::SparseMatrixCSR)` do instead is
convert `A` to a `SparseMatrixCSC` and delegate to the existing, unmodified `SparseMatrixCSC`
methods -- every solver (`:default`/`:suitesparse`, `:accelerate`, `:mumps`, `:sparspak`),
every symmetry hint, and every keyword `sparse_factorize`/`pde_solve` already support for
`SparseMatrixCSC` therefore works for `SparseMatrixCSR` for free, with no bespoke CSR-native
factorization code anywhere.

The conversion (`Bramble._csr_to_csc`, `src/solvers/sparse_solvers.jl`) is an O(nnz)
row-major -> column-major triplet re-layout through `SparseArrays.sparse`, not an O(n²)
`Matrix(A)` densification -- the same technique `benchmark/backends.jl`'s `_to_csc` already
uses for its own correctness oracle. It reads `A.rowptr`, `A.colval`, `A.nzval` directly
(duck-typed field access, not a static `SparseMatrixCSR` import) and rebuilds a fresh
`SparseMatrixCSC` from those three arrays as a plain `(I, J, V)` triplet.

### Why the dispatch is reflection-based, not a typed method

`SparseMatricesCSR.jl` is a weak dependency (`Project.toml`'s `[weakdeps]`/`[extensions]`
entry `BrambleSparseMatricesCSRExt = "SparseMatricesCSR"`), so `SparseMatrixCSR` cannot be
named as a compile-time type anywhere under `src/` -- Bramble itself never depends on the
package that defines it, and a source file that tried would fail to load whenever the
weak dependency isn't present. `pde_solve`'s own `:default` route already reaches an
optional backend the same way (`Base.get_extension(Bramble, :BrambleAppleAccelerateExt)`),
so the CSR fallback follows that established pattern instead of adding a new one:

```julia
_csr_extension() = Base.get_extension(Bramble, :BrambleSparseMatricesCSRExt)

function _is_csr(A)
    ext = _csr_extension()
    return ext !== nothing && A isa ext.SparseMatrixCSR
end
```

When `ext/BrambleSparseMatricesCSRExt.jl` is loaded (i.e. the caller has done
`using SparseMatricesCSR`), that extension module's own
`using SparseMatricesCSR: SparseMatricesCSR, SparseMatrixCSR, sparsecsr` leaves
`SparseMatrixCSR` bound as a name on the extension module object itself, reachable by
reflection (`ext.SparseMatrixCSR`) without Bramble ever importing it. This keeps the new
dispatch entirely inside the three files the CSR-solver work owns
(`src/solvers/sparse_solvers.jl`, `src/solvers/pde_solve.jl`,
`test/ext/sparse_csr_ext.jl`) rather than requiring a change to
`ext/BrambleSparseMatricesCSRExt.jl`, which today only covers construction/assembly, not
solving.

`sparse_factorize`'s existing catch-all (`A::AbstractMatrix`) already threw `MethodError`
for anything that isn't `SparseMatrixCSC`, deliberately, to keep
`test/form/sparse_solvers.jl`'s "Type safety: sparse_factorize only accepts SparseMatrixCSC"
guarantee for genuinely unsupported types (a dense `Matrix`, say). The CSR fallback
narrows that catch-all rather than replacing it: `_is_csr(A)` intercepts only a matrix the
loaded CSR extension itself recognizes, and everything else still falls straight through
to the same `MethodError`. `pde_solve` gained the equivalent new `AbstractMatrix` method
(it previously had none beyond `SparseMatrixCSC`) with the same shape, and `refactor!`'s
existing `Factorization, A::Any` fallback (which threw `ArgumentError` naming
`SparseMatrixCSC` explicitly) gained the same one-line interception before its throw.

### Entry points covered

- `sparse_factorize(A::SparseMatrixCSR; solver = :default, sym = :auto, kwargs...)` --
  converts, then dispatches through the ordinary `:default`/`:suitesparse`/`:accelerate`/
  `:mumps`/`:sparspak` branches.
- `pde_solve(A::SparseMatrixCSR, F::AbstractVector; kwargs...)` -- converts, then reuses
  the `SparseMatrixCSC` method's own `:default` narrowing (Accelerate on macOS for a
  symmetric system, `A \ F` otherwise) and every explicit `solver` choice.
- `refactor!(fact::Factorization, A::SparseMatrixCSR)` / `sparse_refactor!` (its alias) --
  converts, then reuses whichever backend-specific `refactor!` method matches `fact`'s
  concrete type.

Covered by `test/ext/sparse_csr_ext.jl`'s "sparse_factorize/pde_solve accept SparseMatrixCSR"
and "Factorization reuse and refactoring" testsets, the latter running
`test/ext/SolverContracts.jl`'s shared `refactor_contract` (`size`, `ldiv!` in both the
out-of-place and in-place forms, the `VectorElement` destination overload, a complex
right-hand side, every route to a numeric refactorization, and rejection of a dense
matrix) against a CSR-built Poisson system -- the same contract every other solver
extension (SuiteSparse, MUMPS, Apple Accelerate, Sparspak) is tested against. A further
testset confirms the catch-all's guarantee for a genuinely unsupported type (dense
`Matrix`) is unweakened: loading the CSR extension only ever widens dispatch for a
`SparseMatrixCSR`, never for anything else.

`poisson_solve_contract` (the other shared contract) is not reused directly: its `solve`/
`solve_form` closures are exercised against the systems `SolverContracts.poisson_system`
builds internally, which always use the default `SparseMatrixCSC` backend -- there is no
parameter to make that fixture CSR-backed. The 1D/2D/3D unified-dispatcher checks it would
otherwise provide are therefore written out by hand in `sparse_csr_ext.jl` against
CSR-built systems instead.

## Performance: CSC vs. CSR for this fallback path

The commit `7c901266` figure measured a bare `A \ F` on a raw, unconverted `SparseMatrixCSR`
(falling through to a generic, non-CSC-optimized `\` implementation) -- a different code
path from the one built here, so it is not reused below. This is a fresh measurement of the
actual conversion-then-delegate fallback.

**Method**: 2D unit-square Poisson (`-Δu = f`, homogeneous Dirichlet, symmetrized),
assembled once via `Bramble.backend()` (CSC) and once via `csr_backend()` (CSR), both
solved through `pde_solve`/`sparse_factorize` with `:default` (SuiteSparse). Median of 11
samples per call, both paths warmed (JIT-compiled) before timing, load-gated
(`bramble-verification` §9: 1-minute load average confirmed under half `Sys.CPU_THREADS`
== 4, both immediately before each run) and run under `caffeinate -i` to prevent idle sleep
mid-measurement.

**Date**: 2026-09-22. **Power**: battery (user-approved for this run; the figure would
otherwise be re-measured on AC).

| ndofs | nnz (CSC) | `pde_solve` CSC | `pde_solve` CSR (fallback) | ratio | bare conversion |
|---|---|---|---|---|---|
| 3,600 | 17,760 | 2.270 ms | 2.383 ms | **1.05x** | 0.119 ms |
| 22,500 | 111,900 | 13.503 ms | 13.964 ms | **1.03x** | 1.053 ms |

`sparse_factorize` alone (no solve) shows the same shape: 1.06x and 1.08x respectively.
The O(nnz) conversion itself is a small, roughly constant fraction of the total (about
5-8% of the solve time at both sizes measured) -- the fallback is not a "convert to dense
and hope" tax, it costs close to what the CSC solve costs plus one cheap triplet re-layout.
This is consistent with there being no algorithmic reason for CSR to be slower once the
matrix reaches an actual solver: SuiteSparse only ever sees a `SparseMatrixCSC` either way.

## Evaluated and excluded

gpena/Bramble.jl#275 named eight ecosystems. Only the CSC-conversion fallback above was
built; the rest, with reasons:

- **Pardiso.jl**: requires a proprietary Intel MKL license and is entirely absent from this
  repository -- no `ext/` file, no `[weakdeps]` entry, no `Manifest.toml` entry. Untestable
  here without a license this project does not have.
- **MUMPS.jl, Apple Accelerate (`AppleAccelerate.jl`), Sparspak.jl**: each is internally
  `SparseMatrixCSC`-only (confirmed: `mumps_solver.jl`, `accelerate_solver.jl`,
  `sparspak_solver.jl` all pin their factorize/refactor methods to `A::SparseMatrixCSC`).
  A "CSR path" for any of them can only ever be convert-then-delegate -- exactly the
  fallback built here, which already covers all three (and SuiteSparse) uniformly through
  the `solver` keyword. There is no separate per-backend integration to do.
- **AlgebraicMultigrid.jl, ILUZero.jl**: both already take a generic `AbstractMatrix` (AMG)
  or `SparseMatrixCSC` with no CSR-specific fast path to add, and neither is a *direct*
  solver (this issue's scope) -- they precondition an iterative solve, which is a separate
  concern from `sparse_factorize`/`pde_solve`. No CSR-specific work applies.
- **Krylov.jl**: reachable today only indirectly, via `LinearSolve.jl`'s `KrylovJL_CG`/
  `KrylovJL_GMRES` wrappers -- not a direct dependency of Bramble, and, like AMG/ILUZero,
  an iterative rather than direct solver. Resolvable in principle, out of scope for a
  *direct*-solver issue.
- **ThreadedSparseCSR.jl**: zero trace in this repository (no `ext/`, no `[weakdeps]`
  entry, no `Manifest.toml` entry). Resolvable in principle (a genuinely CSR-native
  matrix-vector product package, which could matter for an iterative solver's `mul!`), but
  out of scope for a *direct*-solver issue for the same reason as Krylov.jl -- and,
  circularly, only useful once an iterative CSR path exists to plug it into.

None of these are attempted by this change. A future direct or iterative CSR-specific
solver path would start from whichever of these is no longer out of scope for its own
issue, not from this fallback.

## Refilling a `VectorElement` from a factorization

`ldiv!(::VectorElement, ::Factorization, ::AbstractVector)` extends `LinearAlgebra.ldiv!`,
so `Modules = [Bramble]` autodocs cannot find its docstring; it is private and documented
explicitly here.

```@docs
ldiv!(::VectorElement, ::Factorization, ::AbstractVector)
```
