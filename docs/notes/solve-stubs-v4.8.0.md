# First solve of the three extension backends, with and without the stub invalidation (v4.8.0, S4.2)

Answers #1274: does the invalidation that loading an extension causes in the core's own solve
callers cost a measurable first call? The extension methods `_sparspak_solve(::SparseMatrixCSC,
::AbstractVector)`, `_suitesparse_solve` and `_accelerate_solve` are more specific than the core
stubs `_*_solve(A, ::Any)` those callers were compiled against, so loading the extension
supersedes the stub and invalidates the callers' instances. This note holds the counts, the
prototype that removes them, and the first-call timing from `benchmark/ttfx.jl`.

## Method and conditions

Trees. `before` is the branch tree `~/worktrees/Bramble/v4-8-0` (head `3fda8d39`). `after` is a
detached worktree of the same head, `~/bramble-scratch/v480-stubs`, with the diff below and
nothing else. It is never committed to the branch.

Invalidation counts. `~/bramble-scratch/v480-stubs-snoop.jl <tree> <backend>` loads the backend
package first, then runs `@snoop_invalidations using Bramble`, so the extension loads inside
the snoop. It follows `test/quality/invalidations_sciml_snoop.jl` and runs in a fresh
`julia --threads=1` per (tree, backend). SnoopCompile 3.2.10 and SnoopCompileCore 3.1.4 come from
a temp environment per tree (`~/bramble-scratch/v480-env-before`, `v480-env-after`: the tree
developed, the three backends and the two Snoop packages), not from `test/Project.toml`. A tree
counts as a solve tree when its inserting method belongs to the backend's extension and is named
`_suitesparse_solve`, `_sparspak_solve`, `_accelerate_solve` (before) or `_backend_solve`
(after). "Instances" is `SnoopCompile.countchildren` over those trees. The script is S4.3's
starting point for the gate; it prints `SOLVE_TREES`, `SOLVE_INVALIDATED`, `OWNED_TREES`,
`OWNED_INVALIDATED`, then each solve tree.

Correctness of the prototype. `suitesparse_solve`, `sparspak_solve` and `accelerate_solve` on
the `benchmark/ttfx.jl` fixture (8 x 8 non-uniform mesh, `a_spd`, `l`) on both trees:
relative residual `norm(A*x - F) / norm(F)` is between 4e-14 and 2e-13 in every row, the
solution matches `A \ F` to 4e-14 or better, `pde_solve(A, F; solver = ...)` matches, the
`sym`/`kind` keywords still reach the extension, an unknown keyword is still a `MethodError`,
a dense `A` is still the same `ArgumentError`, and without the extension the three stubs still
raise their "requires ... Add `using ...`" errors. Julia 1.13.4, 1 thread.

## Invalidation counts (measured)

Trees rooted in the extension's solve method, and the instances they invalidate, per backend:

| Backend | before: trees | before: instances | after: trees | after: instances |
|---|---|---|---|---|
| SuiteSparse | 1 | 11 | 0 | 0 |
| Sparspak | 1 | 8 | 0 | 0 |
| AppleAccelerate | 1 | 11 | 0 | 0 |

In the `before` tree each of these is the only tree rooted in a Bramble method, so
`OWNED_TREES` and `OWNED_INVALIDATED` equal the table's figures; in `after` both are 0.
The 11, 8 and 11 match the review's 8 to 11. The `before` trees, from the script:

```
SuiteSparse: inserting kwcall(::NamedTuple, ::typeof(Bramble._suitesparse_solve), A::SparseMatrixCSC, F::AbstractVector) @ BrambleSuiteSparseExt
   backedges: 1: superseding kwcall(::NamedTuple, ::typeof(Bramble._suitesparse_solve), A, ::Any) @ Bramble with MethodInstance for Core.kwcall(::@NamedTuple{sym::Symbol}, ::typeof(Bramble._suitesparse_solve), ::SparseMatrixCSC{Float64, Int64}, ::Vector{Float64}) (11 children)
Sparspak: inserting _sparspak_solve(A::SparseMatrixCSC, F::AbstractVector) @ BrambleSparspakExt
   backedges: 1: superseding _sparspak_solve(A, ::Any) @ Bramble with MethodInstance for Bramble._sparspak_solve(::SparseMatrixCSC{Float64, Int64}, ::Vector{Float64}) (8 children)
AppleAccelerate: inserting kwcall(::NamedTuple, ::typeof(Bramble._accelerate_solve), A::SparseMatrixCSC, F::AbstractVector) @ BrambleAppleAccelerateExt
   backedges: 1: superseding kwcall(::NamedTuple, ::typeof(Bramble._accelerate_solve), A, ::Any) @ Bramble with MethodInstance for Core.kwcall(::@NamedTuple{sym::Symbol}, ::typeof(Bramble._accelerate_solve), ::SparseMatrixCSC{Float64, Int64}, ::Vector{Float64}) (11 children)
```

The instance at the root is the one the precompile workload cached for the first call of each
backend (`sym::Symbol` keyword form for the two that take keywords), and the callers above it
are its children.

## The prototype

The extension adds no method to a function that has a compiled caller in the core. The core
keeps one method per stub and reaches a new function `_backend_solve` through
`Base.invokelatest`, dispatching on a backend tag type. A first attempt called `_backend_solve`
directly. It left the same invalidation: the stub's instance records a backedge on the missing
method, and adding the extension's method supersedes it (SuiteSparse: 1 tree, 12 instances, as
`mt_backedges`). Only the `invokelatest` call leaves no edge, hence the 0 above.

Cost of the shape, for S4.3 to weigh: `invokelatest` makes the stubs' return type `Any`, so the
callers above them (`suitesparse_solve(A, F)` and so on) infer `Any` where they inferred the
extension's type before the extension loaded and after the old invalidation recompiled them.
The stub also tests `A isa SparseMatrixCSC && F isa AbstractVector` itself, to keep the
`ArgumentError` the old fallback raised for a matrix of another type. Each call pays one
dynamic dispatch, negligible next to a factorization.

Full diff (`git diff -- src ext` in the scratch tree):

```diff
diff --git a/ext/BrambleAppleAccelerateExt.jl b/ext/BrambleAppleAccelerateExt.jl
index 323fb670..058a62b7 100644
--- a/ext/BrambleAppleAccelerateExt.jl
+++ b/ext/BrambleAppleAccelerateExt.jl
@@ -583,7 +583,9 @@ else
     end
 end
 
-function Bramble._accelerate_solve(A::SparseMatrixCSC, F::AbstractVector; sym = :auto, kind = :auto)
+function Bramble._backend_solve(
+        ::Bramble.AccelerateBackend, A::SparseMatrixCSC, F::AbstractVector; sym = :auto, kind = :auto
+    )
     fact = Bramble._accelerate_factorize(A; sym = sym, kind = kind)
     return fact \ F
 end
diff --git a/ext/BrambleSparspakExt.jl b/ext/BrambleSparspakExt.jl
index 8cd71409..2cb8cb19 100644
--- a/ext/BrambleSparspakExt.jl
+++ b/ext/BrambleSparspakExt.jl
@@ -301,7 +301,7 @@ function Bramble.sparspak_refactor!(fact::ConcreteSparspakFactorization, A::Spar
     return fact
 end
 
-function Bramble._sparspak_solve(A::SparseMatrixCSC, F::AbstractVector)
+function Bramble._backend_solve(::Bramble.SparspakBackend, A::SparseMatrixCSC, F::AbstractVector)
     Bramble._check_tag_order(eltype(A), "matrix")
     Bramble._check_tag_order(eltype(F))
     T, S = eltype(A), promote_type(eltype(A), eltype(F))
diff --git a/ext/BrambleSuiteSparseExt.jl b/ext/BrambleSuiteSparseExt.jl
index b033d9b7..423c522b 100644
--- a/ext/BrambleSuiteSparseExt.jl
+++ b/ext/BrambleSuiteSparseExt.jl
@@ -223,7 +223,9 @@ function Bramble.suitesparse_refactor!(fact::ConcreteSuiteSparseFactorization, A
     return fact
 end
 
-function Bramble._suitesparse_solve(A::SparseMatrixCSC, F::AbstractVector; sym = :auto, kwargs...)
+function Bramble._backend_solve(
+        ::Bramble.SuiteSparseBackend, A::SparseMatrixCSC, F::AbstractVector; sym = :auto, kwargs...
+    )
     fact = Bramble._suitesparse_factorize(A; sym = sym, kwargs...)
     return fact \ F
 end
diff --git a/src/solvers/accelerate_solver.jl b/src/solvers/accelerate_solver.jl
index 97a01cf9..dfb19d0a 100644
--- a/src/solvers/accelerate_solver.jl
+++ b/src/solvers/accelerate_solver.jl
@@ -226,11 +226,12 @@ function _accelerate_factorize(A; kwargs...)
     )
 end
 
-function _accelerate_solve(A, ::Any; kwargs...)
-    _ext_loaded(:BrambleAppleAccelerateExt) && _throw_unsupported_matrix("accelerate_solve", A)
-    return error(
+function _accelerate_solve(A, F; kwargs...)
+    _ext_loaded(:BrambleAppleAccelerateExt) || error(
         "accelerate_solve requires AppleAccelerate.jl. Add `using AppleAccelerate` before calling this function.",
     )
+    A isa SparseMatrixCSC && F isa AbstractVector || _throw_unsupported_matrix("accelerate_solve", A)
+    return Base.invokelatest(_backend_solve, AccelerateBackend(), A, F; kwargs...)
 end
 
 function accelerate_refactor!(::AccelerateFactorization, A)
diff --git a/src/solvers/sparse_solvers.jl b/src/solvers/sparse_solvers.jl
index 1fbdc26e..a6850781 100644
--- a/src/solvers/sparse_solvers.jl
+++ b/src/solvers/sparse_solvers.jl
@@ -162,6 +162,16 @@ function _throw_unsupported_matrix(name::String, A, expected::String = "a Sparse
     throw(ArgumentError("$name takes $expected; got a $(typeof(A))."))
 end
 
+# One-shot solves of the extension backends. The core stubs `_*_solve(A, F)` stay the only
+# methods of their functions. An extension adds its method to `_backend_solve`, which has none
+# here, and the stubs reach it through `invokelatest`: a plain call leaves a backedge on the
+# missing method, and adding it invalidates the stub's instance.
+struct SuiteSparseBackend end
+struct SparspakBackend end
+struct AccelerateBackend end
+
+function _backend_solve end
+
 # Only a `SparseMatrixCSC` is supported (test/solvers/sparse_solvers.jl keeps the `MethodError`).
 # This method states that, and converts a CSR matrix and delegates back, so every
 # `AbstractMatrix` argument ends at a `SparseMatrixCSC` method or a `MethodError`.
diff --git a/src/solvers/sparspak_solver.jl b/src/solvers/sparspak_solver.jl
index a4792fc9..820d7fde 100644
--- a/src/solvers/sparspak_solver.jl
+++ b/src/solvers/sparspak_solver.jl
@@ -146,11 +146,12 @@ function _sparspak_factorize(A)
     )
 end
 
-function _sparspak_solve(A, ::Any)
-    _ext_loaded(:BrambleSparspakExt) && _throw_unsupported_matrix("sparspak_solve", A)
-    return error(
+function _sparspak_solve(A, F)
+    _ext_loaded(:BrambleSparspakExt) || error(
         "sparspak_solve requires Sparspak.jl. Add `using Sparspak` before calling this function.",
     )
+    A isa SparseMatrixCSC && F isa AbstractVector || _throw_unsupported_matrix("sparspak_solve", A)
+    return Base.invokelatest(_backend_solve, SparspakBackend(), A, F)
 end
 
 function sparspak_refactor!(::SparspakFactorization, A)
diff --git a/src/solvers/suitesparse_solver.jl b/src/solvers/suitesparse_solver.jl
index fd7bac76..327a715e 100644
--- a/src/solvers/suitesparse_solver.jl
+++ b/src/solvers/suitesparse_solver.jl
@@ -130,11 +130,12 @@ function _suitesparse_factorize(A; kwargs...)
     )
 end
 
-function _suitesparse_solve(A, ::Any; kwargs...)
-    _ext_loaded(:BrambleSuiteSparseExt) && _throw_unsupported_matrix("suitesparse_solve", A)
-    return error(
+function _suitesparse_solve(A, F; kwargs...)
+    _ext_loaded(:BrambleSuiteSparseExt) || error(
         "suitesparse_solve requires SuiteSparse.jl. Add `using SuiteSparse` before calling this function.",
     )
+    A isa SparseMatrixCSC && F isa AbstractVector || _throw_unsupported_matrix("suitesparse_solve", A)
+    return Base.invokelatest(_backend_solve, SuiteSparseBackend(), A, F; kwargs...)
 end
 
 function suitesparse_refactor!(::SuiteSparseFactorization, A)
```

Tests that name these methods and that S4.3 must keep green, found by grep, not run here:
`test/ext/sparse_csr_ext.jl:505-513` (the three stubs on a CSR matrix),
`test/ext/forwarddiff_ext.jl:547` (`_sparspak_solve` over `Dual`s) and the name lists in
`test/quality/explicit_imports.jl:339-628`.

## First-call timing

Run in the user's terminal, from the branch tree, with the four slots:

```
BRAMBLE_JULIA_SLOTS=4 .claude/scripts/julia_slot.sh 4 julia --startup-file=no benchmark/ttfx.jl \
    --before ~/worktrees/Bramble/v4-8-0 --after ~/bramble-scratch/v480-stubs --runs 3 --threads 4
```

The harness times every case; read the three solve rows. Each launch is a fresh `julia`
under a fresh depot. The spread is the largest minus the smallest of the three raw samples the
harness prints under "Raw samples (seconds)". The `after` tree needs the root `Manifest.toml`
beside it (copied from the branch tree; the test project is a workspace member).

| Row | before median (ms) | before samples (ms) | before spread (ms) | after median (ms) | after samples (ms) | after spread (ms) | before minus after (ms) | clears the spread? |
|---|---|---|---|---|---|---|---|---|
| `Bramble.suitesparse_solve(A, F)` | 1.9 | 1.86, 1.85, 1.90 | 0.05 | 6.1 | 6.15, 6.20, 5.92 | 0.28 | -4.3 | yes, slower after |
| `Bramble.sparspak_solve(A, F)` | 21.4 | 22.12, 20.53, 21.39 | 1.59 | 24.6 | 24.30, 24.64, 25.03 | 0.73 | -3.2 | yes, slower after |
| `Bramble.accelerate_solve(A, F)` | 396.9 | 376.29, 396.94, 411.78 | 35.49 | 402.9 | 378.91, 403.77, 402.85 | 24.86 | -5.9 | no |

Run on 2026-11-20 from the branch tree at `3fda8d39`, Julia 1.13.4, 4 threads, 3 runs, on AC
power, with the whole Julia slot pool held. The harness's preflight flagged the 1-min load at
2.25 (its gate is 2) and could not confirm OneDrive syncing was paused; both trees live outside
OneDrive, and `before` and `after` run back to back in each case, so the comparison holds even
if the absolute figures carry some load. Core precompile: 61.4 s before, 61.5 s after. The full
log was `/tmp/v480_s42_ttfx.log`.

"Clears the spread" holds when `before minus after` is larger than the larger of the two
spreads. The `accelerate_solve` extension is not in the harness's extension precompile, so its
own precompile runs inside the case's first launch before the timer starts, not in the timed
call.

## Decision for S4.3

- `suitesparse_solve`: clears the spread, in the wrong direction. The first call is 4.3 ms
  slower without the invalidation.
- `sparspak_solve`: clears the spread, in the wrong direction. The first call is 3.2 ms
  slower.
- `accelerate_solve`: inside the spread (a 5.9 ms difference against a 35 ms spread).

Removing the 8 to 11 invalidated instances buys nothing on the first call, and the
`invokelatest` the prototype needs to get there costs a few milliseconds, along with the `Any`
return type its callers then infer. The stubs stay as they are and S4.3 does not run. #1274
closes with this pull request. The prototype diff above and
`~/bramble-scratch/v480-stubs-snoop.jl` stay as the record, in case a later change makes the
invalidation matter.
