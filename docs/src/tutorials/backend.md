```@meta
CurrentModule = Bramble
```

# [Backend tutorial](@id tutorial_backend)

Every mesh, grid space and form in `Bramble.jl` carries a **backend**: a compile-time
configuration saying which vector and matrix types to allocate, and whether
threading-capable operations run serially or in parallel. Chosen once, when a mesh is
built, and inherited by everything constructed from it afterwards.

---

## 1. What a backend is

A [`Backend`](@ref) is `Backend{VT, MT, EP}`: a vector type, a matrix type, and an
[`ExecutionPolicy`](@ref), carried as type parameters on a struct with no fields at all.
Building one costs nothing at runtime: the types alone tell every downstream `vector`/
`matrix` allocation and every threading-capable loop what to construct and how to run.

```@example backend
using Bramble

be = backend()
```

```@example backend
vector_type(be), matrix_type(be), execution_policy(be)
```

The default is dense `Vector{Float64}`, sparse `SparseMatrixCSC{Float64,Int}`, and the
[`Serial`](@ref) policy: a plain loop, unconditionally, for every operation that could
otherwise thread.

## 2. Choosing element, vector and matrix types

[`backend`](@ref) takes `vector_type`/`matrix_type` keywords directly, or a single
element type as its first argument:

```@example backend
using SparseArrays

f32 = backend(Float32)                                    # Vector{Float32}, sparse Float32 matrix
dense_backend = backend(vector_type = Vector{Float64}, matrix_type = Matrix{Float64})
matrix_type(dense_backend)
```

`backend(T)` is what [`mesh`](@ref) uses when no backend is given at all: `mesh`
defaults to `backend(eltype(Ω))`, so a mesh built over `Float32` points already gets a
`Float32` backend without asking for one explicitly.

## 3. Attaching a backend to a mesh

```@example backend
Ω = domain(interval(0.0, 1.0))
Ωₕ = mesh(Ω, 21; backend = backend(Float64))
backend(Ωₕ) === backend(Float64)
```

Everything built from `Ωₕ` afterwards ([`gridspace`](@ref), its `VectorElement`s, and
any [`form`](@ref) over it) carries the same backend, so choosing one at mesh
construction is choosing it for the whole computation downstream:

```@example backend
Wₕ = gridspace(Ωₕ)
backend(Wₕ) === backend(Ωₕ)
```

## 4. `Serial()` or `Parallel()`, chosen once

`Rₕ!`, `avgₕ!`, a grid space's own quadrature-weight construction, and form assembly
(`assemble!`/`assemble`, for both a `LinearForm` and a `BilinearForm`) all read
[`execution_policy`](@ref) off the mesh or space they are given, and take the serial or
threaded branch accordingly; nothing else about how you call them changes:

```@example backend
Ωₕ_par = mesh(Ω, 100_000; backend = backend(policy = Parallel()))
Wₕ_par = gridspace(Ωₕ_par)
execution_policy(Wₕ_par)
```

There is **no automatic size threshold**. A `Parallel()` backend threads every eligible
call, however small, however many times a loop repeats it. That is deliberate: a
threshold would have to be tuned per operation (`Rₕ!`'s crossover is not `avgₕ!`'s, and
neither is assembly's), tuning the caller cannot see or override. Asking for `Parallel()`
and getting it, every time, is the point.

So the choice belongs to you, not a heuristic:
- **`Serial()`** (the default) for small, frequently repeated calls: a per-step
  operation inside a time loop, where the cost of spawning threads would dwarf the work
  itself.
- **`Parallel()`** once a single call is expensive enough on its own that spawning tasks
  pays for itself: a one-shot restriction or assembly over a large mesh.

The [benchmarks page](../benchmarks.md) carries the actual measurements across sizes;
that is what should decide, not a guess.

### Measured crossovers

Where `CpuThreaded` and [`CpuBatch`](@ref) (§7 below) start beating `CpuSerial` was
measured per workload, not assumed, on an Apple M2 host, `--threads=4`, on AC power
(`benchmark/polyester_crossover.jl`, commit `4b76d62b`): the smallest grid size at which
each policy beats `CpuSerial` twice running.

| Workload | `CpuThreaded` | `CpuBatch` |
|---|---|---|
| `Rₕ!` unmasked | 64-96 points/axis | 8-24 |
| `Rₕ!` masked | 256 | 16 |
| `avgₕ!` (nq=3) | 24-32 | 8 |
| `innerₕ`/`_dot` | -- | 1,000 elements |

The blank `CpuThreaded` cell for `innerₕ`/`_dot` is not a missing measurement: `_dot(::CpuThreaded, ...)`
(`src/utils/linear_algebra.jl`) forwards to the identical serial reduction, so an inner
product does not thread under this policy at all
([#112](https://github.com/gpena/Bramble.jl/issues/112)) -- it runs exactly as fast, or
slow, as `CpuSerial` regardless of size, and there is no crossover to report.

The crossover differs by an order of magnitude between workloads, so a figure measured
for one does not transfer to another -- that is why four rows are published here rather
than a single number. Where both policies have an entry, `CpuBatch` beats `CpuThreaded`
at every crossover measured, falling one to two orders of magnitude below it.

These are one machine's numbers, taken under one power state, not a portable constant:
re-measure before leaning on them for a different host.

### The policy hierarchy

`Serial` and `Parallel` are the names above, and they are aliases: `Serial === CpuSerial`
and `Parallel === CpuThreaded`. The types they alias sit in a hierarchy that splits by
which processor a strategy targets, not only how much of the work runs at once:

```
ExecutionPolicy
├── CpuPolicy
│   ├── CpuSerial      (Serial)    -- one CPU thread
│   ├── CpuThreaded    (Parallel)  -- Base.Threads.@threads
│   └── CpuBatch                   -- Polyester.jl's @batch
└── GpuPolicy
    └── GpuAsync                   -- launched on the device
```

The split exists because "serial or threaded" had no way to say which processor
([#191](https://github.com/gpena/Bramble.jl/issues/191)). A GPU backend was constructed
with `Serial()` -- a policy meaning one CPU thread walks the array element by element,
which is the one thing a device array refuses. `metal_backend()` now carries `GpuAsync()`
by default, and pairing device storage with a `CpuPolicy` (or host storage with a
`GpuPolicy`) is rejected outright rather than left to fail on scalar indexing several
frames deeper -- the next subsection covers why, and what the rejection looks like.

Both spellings work everywhere; use whichever reads better. [`CpuBatch`](@ref) is a third
`CpuPolicy`, a Polyester-backed sibling of `CpuThreaded`
([#190](https://github.com/gpena/Bramble.jl/issues/190)) -- §7 below covers what it needs
before its first sweep.

### Locality and strategy

The hierarchy above groups policies by which processor they target, but *where* a
backend's work can legally run is not chosen through the policy at all -- it is decided
by the array types the backend was built with. That is [`locality`](@ref):
[`HostLocality`](@ref) for storage a CPU loop can index element by element,
[`DeviceLocality`](@ref) for storage an accelerator schedules against instead.
`locality`, `Locality`, `HostLocality` and `DeviceLocality` are `public`, not exported, so
they are called qualified:

```@example backend
Bramble.locality(Vector{Float64}), Bramble.locality(CpuSerial()), Bramble.locality(GpuAsync())
```

Locality is derived from the array type, never declared. A [`CpuPolicy`](@ref) claims
[`HostLocality`](@ref) and a [`GpuPolicy`](@ref) claims [`DeviceLocality`](@ref) -- that is
what lets a backend check the two against each other. An array type this package has not
been taught otherwise about counts as host memory (the fallback method is
`Bramble.locality(::Type{<:AbstractArray}) = Bramble.HostLocality()`), so a custom wrapper
array works as a `vector_type`/`matrix_type` without registering anything, as long as it
genuinely is host-indexable. A GPU package extension adds one method for its own array
type to answer [`DeviceLocality`](@ref) instead: `BrambleMetalExt` does this for
`MtlVector` and `MtlMatrix` (and both Metal sparse matrix types).

Strategy is the genuine choice, and it only exists *within* one locality: host storage
admits [`CpuSerial`](@ref), [`CpuThreaded`](@ref) and [`CpuBatch`](@ref); device storage
admits the device kernel, [`GpuAsync`](@ref). A policy whose locality does not match the
storage it is paired with cannot execute anything -- a CPU loop cannot scalar-index device
memory, and a host array has no device to schedule against -- so [`Backend`](@ref)'s inner
constructor checks that the vector type and the policy agree on locality (and, since a
backend must be wholly host or wholly one GPU's, that the matrix type agrees with the
vector type too) and rejects the combination before it exists:

```julia
using Bramble, Metal

metal_backend(; policy = CpuSerial())
# ERROR: ArgumentError: Backend vector type MtlVector{Float32} has locality
# Bramble.DeviceLocality(), but execution policy CpuSerial has locality
# Bramble.HostLocality(): a Backend's storage and its execution policy must agree on
# locality, or the combination cannot execute anything.
```

## 5. One interface, governed by the backend

Call `assemble!`/`assemble`, `Rₕ!`/`avgₕ!` the same way regardless of which policy the
backend carries; that is the entire reason to choose a policy once, on the backend,
rather than per call:

```@example backend
f = Rₕ(Wₕ_par, sin)
l = form(Wₕ_par, v -> innerₕ(f, v))
b = assemble(l)      # threads, because Wₕ_par's backend says Parallel()
nothing # hide
```

Do not reach for `assemble_parallel!` (or a hand-picked threaded variant) to get
parallel behaviour; that bypasses whatever the backend says and threads regardless,
which defeats the point of choosing a policy at all. It still exists, but only as a
deliberate, explicit override: a one-off forced comparison, or a benchmark that wants
the threaded path irrespective of the ambient backend. Build the backend you want and
call the ordinary entry point; see the [forms tutorial](form.md) for the full picture
of how assembly uses it.

## 6. Backend storage: CSC or CSR

Every backend above keeps `SparseMatrixCSC{Float64,Int}`, the default `matrix_type`.
[`csr_backend`](@ref) swaps that for `SparseMatricesCSR.jl`'s `SparseMatrixCSR{1,T,Int}`
instead (the one-based variant, matching Bramble's own indexing), built the same way as
any other backend:

```julia
using Bramble, SparseMatricesCSR

be = csr_backend()                 # Float64, Serial()
matrix_type(be)                    # SparseMatrixCSR{1, Float64, Int}
```

[`csr_backend`](@ref) requires `SparseMatricesCSR.jl` loaded alongside `Bramble.jl`;
without it, it throws, the same way [`metal_backend`](@ref) does without `using Metal`.

### Which one to reach for

A finite-difference stencil is assembled row by row, which CSR storage reaches without
the column scatter a `SparseMatrixCSC` assembly needs
([#214](https://github.com/gpena/Bramble.jl/issues/214)). Measured on one machine (AC
power, 1-minute load 2.13, 4 threads):

- **Assembly is a wash between CSC and CSR**: first assembly and `assemble!` refill both
  land within about 0.94-1.05x of each other across 1D, 2D and 3D, and neither backend
  allocates in `assemble!`.
- **CSR stores the same matrix in roughly half the memory**: 5.3 MiB against CSC's
  10.7 MiB in 1D, 24.4 against 59.1 MiB in 3D.
- **CSR loses the direct solve**, 2.4x to 4.2x slower than CSC across the sizes measured
  -- `A \ F` on a `SparseMatrixCSR` routes through a transpose wrapper around a sparse LU
  rather than reaching UMFPACK directly the way CSC's `\` does.

So: reach for [`csr_backend`](@ref) when memory is the constraint and assembly, not a
direct solve, dominates; keep the default CSC backend when `A \ F` is on the critical
path. These are one run on one machine, not a guarantee -- re-measure before leaning on
them for a specific problem size.

## 7. A Polyester-batched CPU policy

[`CpuBatch`](@ref) is a third [`CpuPolicy`](@ref), alongside [`CpuSerial`](@ref) and
[`CpuThreaded`](@ref): it directs grid operations and form assembly through
`Polyester.jl`'s `@batch` instead of `Base.Threads.@threads`, a primitive whose lower
per-call overhead can pay off on grids where `CpuThreaded`'s threading does not
([#190](https://github.com/gpena/Bramble.jl/issues/190)).

```julia
using Bramble, Polyester

be = backend(policy = CpuBatch())
Ωₕ = mesh(domain(interval(0.0, 1.0)), 100_000; backend = be)
Wₕ = gridspace(Ωₕ)
execution_policy(Wₕ)   # CpuBatch()
```

The policy type ships with `Bramble.jl` itself, but the sweeps it selects live in the
`BramblePolyesterExt` package extension: `using Polyester` must be loaded before a
`CpuBatch` backend runs its first sweep, or the call errors, naming `Polyester.jl` -- the
same precedent as [`csr_backend`](@ref) and [`metal_backend`](@ref) without their own
package. Call `assemble!`/`assemble`, `Rₕ!`/`avgₕ!` exactly as with any other backend;
choosing `CpuBatch()` on the backend is the only thing that changes.

## 8. A GPU backend (Metal), or letting `gpu_backend` pick it

```julia
using Bramble, Metal

gpu = metal_backend()                    # Float32, GpuAsync()
```

`Float64` is not supported on Apple Silicon GPUs; use `Float32` or `Float16`.
[`metal_backend`](@ref) requires `Metal.jl` loaded alongside `Bramble.jl`; without it,
it throws. A [`CpuPolicy`](@ref) is rejected too, at construction: `MtlVector`/
`MtlMatrix` storage has device locality, and a CPU loop cannot scalar-index it -- see
"Locality and strategy" in §4 above for the mismatch and what it looks like.

[`gpu_backend`](@ref) is the entry point to reach for when you do not want to name a
device backend by hand:

```julia
using Bramble, Metal

gpu = gpu_backend()                      # resolves to metal_backend(): Float32, GpuAsync()
```

It checks which GPU package extension is loaded, with `Base.get_extension`, and
forwards to that backend's own constructor -- currently only [`metal_backend`](@ref),
under `using Metal`. A CUDA or AMDGPU extension joins the same dispatch once one exists
(gpena/Bramble.jl#11, v3.5.0).

The check requires both: the extension *loaded* and its device *functional*
(`Metal.functional()`). `using Metal` succeeds on any platform, degrading gracefully
rather than erroring, so loading it alone does not prove a working GPU is present --
`gpu_backend()` on a host where `Metal` is loaded but `Metal.functional()` is false
refuses immediately, rather than handing back a `metal_backend()` that would only fail
once actually used.

Two failures give two different diagnostics, deliberately not sharing a message:

- **No GPU extension loaded at all**: `gpu_backend` throws an error naming the package
  to load, chosen from the host architecture: `using Metal` on Apple Silicon, `using
  CUDA` on Linux or Windows, and a generic message naming no supported hardware
  otherwise. Fixed by an import.
- **An extension is loaded but its device is not functional**: `gpu_backend` throws a
  separate error saying so. This points at a driver, hardware or virtualisation problem
  -- not something an import can fix, which is why it reads differently from the first
  case.

## 9. Introspection

- [`vector_type`](@ref)`(be)`, [`matrix_type`](@ref)`(be)`: the two type parameters.
- [`backend_types`](@ref)`(be)`: `(eltype, VT, MT, typeof(be))`, all four at once.
- [`execution_policy`](@ref)`(x)`: works on a `Backend`, a mesh, or a grid space, and
  always answers the same question: what would a threading-capable call on this do.

```@example backend
backend_types(be)
```

## 10. The precompilation workload

`Bramble.jl` ships a precompilation workload that exercises 1D, 2D and 3D meshes on
load. It costs a few seconds when the package is first built and cuts the time to
first result by roughly a factor of four.

While working on the package itself you may prefer faster rebuilds over faster first
use. To skip the workload:

```julia
using Preferences, Bramble
set_preferences!(Bramble, "precompile_workload" => false)
```

Julia tracks preferences in the precompilation cache, so the change takes effect on
the next `using Bramble` with no manual cache clearing. Restore the default with

```julia
delete_preferences!(Bramble, "precompile_workload"; force = true)
```

The setting is written to `LocalPreferences.toml` next to your active project.
