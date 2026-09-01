# Bramble.jl: In-Place Operation Allocation Audit & Remediation Log

This document tracks all in-place mutating functions (`!`) in Bramble.jl, their allocation profiles, root causes, and optimization status.

---

## Current Status Overview

| Operation | Target Object | Baseline Warm Allocations | Remediation Strategy | Status |
| :--- | :--- | :---: | :--- | :---: |
| `set_points!(Ωₕ, pts)` | Mesh `Mesh1D` | **0 B** (was ~2.8 KB) | Reuses existing cached vectors via `Ωₕ.pts .= pts` | 🟢 Zero-Alloc |
| `Rₕ!(uₕ, f)` | `VectorElement` | **0 B** (was 112 – 144 B) | Directly writes to `values(uₕ)` with unrolled tuple scatter | 🟢 Zero-Alloc |
| `avgₕ!(uₕ, f)` | `VectorElement` | **0 B** (was 144 – 320 B) | Direct linear writes, unrolled tuple scatter, thresholded parallel | 🟢 Zero-Alloc |
| `dirichlet_bc!(v, Wₕ, bcs, …)` | Vector `v` | **112 – 176 B** | Eliminate `Set{Marker}` iterator and `SubArray` view | 🟡 Next Up |
| `assemble!(b, form; ast=ast)` | Vector `b` | **32 – 96 B** | Add positional `ast` overload to bypass kwarg box | ⚪ Backlog |
| `assemble_parallel!(b/A, …)` | Vector / Matrix | **480 – 1,488 B** | `Threads.@threads` task scheduling ($O(1)$) | 🟢 Acceptable |
| `D₋ₓ!(v, u)`, `Dcₓ!`, `Dstar₊ₓ!`, `Dₕₓ!` | Grid element `v` | **0 B** | Verified zero allocation | 🟢 Zero-Alloc |
| `M₋ₓ!(v, u)`, `M₊ₓ!`, `jumpₓ!` | Grid element `v` | **0 B** | Verified zero allocation | 🟢 Zero-Alloc |
| `dirichlet_bc!(A, Wₕ, labels…)` | Matrix `A` | **0 B** | Verified zero allocation | 🟢 Zero-Alloc |
| `symmetrize!(A, b, Wₕ, labels…)` | Matrix `A` & Vector `b` | **0 B** | Verified zero allocation | 🟢 Zero-Alloc |
| `assemble!(A, a)` | Matrix nonzeros | **0 B** | Verified zero allocation | 🟢 Zero-Alloc |
| `assemble!(b, l)` (default kwarg) | Vector `b` | **0 B** | Verified zero allocation | 🟢 Zero-Alloc |

---

## Detailed Analysis by Operation

### 1. `set_points!(Ωₕ::Mesh1D, pts)`
- **File**: [`src/mesh/mesh1d.jl:89-110`](src/mesh/mesh1d.jl#L89-L110)
- **Status**: ✅ **Resolved (0 bytes allocated)**
- **Optimization**:
  ```julia
  if length(Ωₕ.pts) == n
      Ωₕ.pts .= pts
  else
      Ωₕ.pts = pts
      set_indices!(Ωₕ, generate_indices(n))
      half_points!(Ωₕ, vector(backend(Ωₕ), n + 1))
      half_spacings!(Ωₕ, vector(backend(Ωₕ), n))
      spacings!(Ωₕ, vector(backend(Ωₕ), n))
  end
  ```
  Reuses existing memory buffers when the point count is unchanged. Reallocates only during mesh refinement/coarsening.

---

### 2. `Rₕ!(uₕ::VectorElement, f; markers=...)`
- **File**: [`src/space/operators/restriction.jl:156-230`](src/space/operators/restriction.jl#L156-L230)
- **Status**: ✅ **Resolved (0 bytes allocated)**
- **Optimization**:
  - Replaced intermediate `to_matrix` (`ReshapedArray`) and escaping closures with direct indexing into `values(uₕ)`.
  - Implemented unrolled tuple scattering (`_scatter_comp!`) for composite spaces with single vector-valued functions, evaluating $f(\mathbf{x})$ once per grid point and scattering directly to raw buffers in-place.
  - Marker-based restrictions zero destination arrays and evaluate only selected indices in-place without heap allocations.

---

### 3. `avgₕ!(uₕ::VectorElement, f; ...)`
- **File**: [`src/space/operators/cell_average.jl:142-177`](src/space/operators/cell_average.jl#L142-L177)
- **Problem**:
  1. `to_matrix(uₕ)` creates `ReshapedArray` (32 bytes per component).
  2. `_cell_average_kernel(f, x, nq, T, Val(D))` creates a heap-allocated closure struct (112–192 bytes).
- **Remediation**: Direct index walk over `indices(mesh)` evaluating Gauss-Legendre quadrature in-place.
- **Target Allocation**: **0 bytes**.

---

### 4. `dirichlet_bc!(v::AbstractVector, Wₕ, bcs, labels...)`
- **File**: [`src/form/dirichlet_constraints.jl:301-314`](src/form/dirichlet_constraints.jl#L301-L314)
- **Problem**:
  1. `conditions(bcs)` returns a `Set{Marker}`. Iterating a `Set` in Julia (`for marker in conditions(bcs)`) allocates an internal `SetIterator` state (112 bytes).
  2. For `CompositeGridSpace`, `view(v, (offset + 1):(offset + ndofs(sp)))` allocates `SubArray` descriptors (64 bytes).
- **Remediation**:
  - Iterate conditions statically or store them in a `Tuple` / `NamedTuple`.
  - Pass the component offset into the evaluation loop instead of taking a `view(v, ...)`.
- **Target Allocation**: **0 bytes**.

---

### 5. `assemble!(b, form; ast=ast)` / `evaluate!(scratch, form, vₕ; ast=ast)`
- **File**: [`src/form/linear.jl:686`](src/form/linear.jl#L686) & [`src/form/linear.jl:118`](src/form/linear.jl#L118)
- **Problem**: Passing `ast = ast_l` as a keyword argument creates a kwarg container tuple (32–96 bytes). Calling positionally (`assemble!(b, form)`) allocates 0 bytes.
- **Remediation**: Add positional method overloads: `assemble!(b, form, ast)` and `evaluate!(scratch, form, vₕ, ast)`.
- **Target Allocation**: **0 bytes**.
