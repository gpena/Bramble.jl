```@meta
CurrentModule = Bramble
```

# [Backends and execution policies](@id tutorial_backend)

**What you will learn.** What a backend is, how to choose between serial and threaded execution, and how large a problem must be before threading pays off.

**What you need first.** The [mesh tutorial](@ref tutorial_mesh) and the [space tutorial](@ref tutorial_space), for meshes and grid spaces, and the [form tutorial](@ref tutorial_form), for assembling a system.

**Where next.** The [solvers tutorial](solvers.md) shows what to do with the linear system once it is assembled.

Every block below runs when this page is built, except the ones that need a GPU or an optional package, which are marked as plain code.

---

## What a backend carries

Every mesh, grid space and form carries a **backend**: a compile-time choice of which vector and matrix types to allocate, and whether the operations that can thread do so. It is chosen once, when a mesh is built, and inherited by everything made from that mesh. Ask for the default one:

```@example backend
using Bramble
using Bramble: vector_type, matrix_type, backend_types, execution_policy

be = backend()
vector_type(be), matrix_type(be), execution_policy(be)
```

A [`Backend`](@ref) is only three type parameters: a vector type, a matrix type and an [`ExecutionPolicy`](@ref). It has no fields, so making one costs nothing. The default stores dense `Vector{Float64}` and sparse `SparseMatrixCSC{Float64,Int}`, and runs every operation as a plain loop under the [`Serial`](@ref) policy.

[`backend`](@ref) takes a single element type, or the two storage types as keywords:

```@example backend
f32 = backend(Float32)
dense = backend(vector_type = Vector{Float64}, matrix_type = Matrix{Float64})
backend_types(f32), matrix_type(dense)
```

[`mesh`](@ref) calls `backend(eltype(Ω))` when you give it no backend, so a mesh over `Float32` points already gets a `Float32` backend.

## Attach a backend to a mesh

Pass the backend when the mesh is built, and every object made from the mesh carries it:

```@example backend
Ω = domain(interval(0.0, 1.0))
Ωₕ = mesh(Ω, 21; backend = backend(Float64))
Wₕ = gridspace(Ωₕ)

backend(Ωₕ) === backend(Float64), backend(Wₕ) === backend(Ωₕ)
```

---

## [Choose a policy](@id backend_policies)

An **execution policy** says whether the operations that can thread, namely `Rₕ!`, `avgₕ!`, the quadrature weights of a grid space and form assembly, run in a plain loop or on several threads. The two names you will use are `Serial()` and `Parallel()`. The same code runs under both; only the backend differs. Build one mesh of each kind:

```@example backend
Ω_big = domain(interval(0.0, 1.0))
Wₛ = gridspace(mesh(Ω_big, 100_000; backend = backend(policy = Serial())))
Wₚ = gridspace(mesh(Ω_big, 100_000; backend = backend(policy = Parallel())))

execution_policy(Wₛ), execution_policy(Wₚ)
```

Now restrict the same function to both and compare:

```@example backend
uₛ = Rₕ(Wₛ, sin)
uₚ = Rₕ(Wₚ, sin)

parent(uₛ) == parent(uₚ)
```

The two policies give identical values. Only the schedule of the work differs, and you never change the call. Assembly reads the policy the same way:

```@example backend
l = form(Wₚ, v -> innerₕ(uₚ, v))
b = assemble(l)      # threads, because Wₚ's backend says Parallel()
nothing # hide
```

!!! tip "Try this"
    Build the same two spaces with 100 points instead of 100,000. The values still agree, but `Parallel()` now starts threads for work that is too small to repay them. The next section puts numbers on that.

There is **no automatic size threshold**. A `Parallel()` backend threads every eligible call, however small and however many times a loop repeats it. A threshold would have to be tuned per operation, because `Rₕ!`'s crossover is not `avgₕ!`'s and neither is assembly's, and the caller could neither see nor override it. So the choice is yours:

- Use `Serial()`, the default, for small calls that repeat, such as an operator inside a time loop, where starting threads would cost more than the work.
- Use `Parallel()` once one call is expensive enough to repay the threads, such as a one-shot restriction or assembly over a large mesh.

Do not call `assemble_parallel!` to get parallel behaviour. It threads regardless of the backend, so it is only for a deliberate comparison. Build the backend you want and call the ordinary `assemble`, as the [form tutorial](@ref tutorial_form) does.

### Measured crossovers

A **crossover** is the smallest problem size at which a parallel policy beats `CpuSerial`
twice running. Below it, starting threads costs more than the work saved. Each crossover
below was measured per operation, not assumed, on an Apple M2, four threads (`--threads=4`),
on AC power, from `benchmark/polyester_crossover.jl` (commit `4b76d62b`). The sizes are
points per axis on a two-dimensional grid, or elements for the vector reductions.

| Operation | `CpuThreaded` | `CpuPolyester` |
|---|---|---|
| `Rₕ!` unmasked | 64-96 points per axis | 8-24 points per axis |
| `Rₕ!` masked | 256 points per axis | 16 points per axis |
| `avgₕ!` (nq=3) | 24-32 points per axis | 8 points per axis |
| `innerₕ`/`_dot` | 100,000-300,000 elements | 1,000 elements |

`_dot`/`_dot_masked(::CpuThreaded, ...)` (`src/utils/linear_algebra.jl`) are a real threaded
reduction (static chunks, dependency-free; masked reductions walk set bits per word;
`SeparableWeights` banded along the last axis), and their crossover against `CpuSerial` was
measured the same way as the rows above: 100,000-300,000 elements across four runs (commit
`560e8394`) -- the exact crossing point moved within that range from one run to the next, the
same jitter the other workloads show near their own crossing point.

The crossover differs by an order of magnitude between workloads, so a figure measured
for one does not transfer to another. That is why the table has four rows rather than a
single number. Where both policies have an entry, `CpuPolyester` beats `CpuThreaded`
at every crossover measured, falling one to two orders of magnitude below it.

These are one machine's numbers, taken under one power state, not a portable constant:
re-measure before leaning on them for a different host. The [benchmarks page](../benchmarks.md) carries the measurements across sizes.

### Measure your own crossovers

To get these figures for your own machine, at the thread count you plan to use, run the crossover script from a checkout of the repository:

```bash
julia --threads=N --project=benchmark benchmark/policy_crossover.jl
```

It sweeps every workload above, and a few more, in 1D, 2D and 3D, and prints one `CROSSOVER | ...` line per workload and dimension. Each line gives the smallest size at which `CpuThreaded` and `CpuPolyester` start beating `CpuSerial`. Those lines count total degrees of freedom per dimension, so they are not directly comparable with the table above, which counts points per axis on a 2D grid. Pass `--smoke` for a quick check with a few tiny sizes, `--max-dofs N` to raise the default cap of 1e6 degrees of freedom, or `--out file.md` to write the tables to a Markdown file.

### The policy hierarchy

`Serial` and `Parallel` are aliases: `Serial === CpuSerial` and `Parallel === CpuThreaded`. They sit in a hierarchy that splits by which processor a strategy targets:

```
ExecutionPolicy
├── CpuPolicy
│   ├── CpuSerial      (Serial)    -- one CPU thread
│   ├── CpuThreaded    (Parallel)  -- Base.Threads.@threads
│   └── CpuPolyester                -- Polyester.jl's @batch
└── GpuPolicy
    └── GpuKernel                   -- launched on the device
```

Both spellings work everywhere. [`CpuPolyester`](@ref) is a threaded policy with a lower per-call overhead, and the table above shows it crossing over earlier. It needs `using Polyester` before its first sweep, or the call errors and names `Polyester.jl`. [`GpuKernel`](@ref) runs on a device and needs a GPU backend, described in the reference below. The older names `CpuBatch` and `GpuAsync` remain as deprecated aliases of `CpuPolyester` and `GpuKernel`.

---

## Reference

### Locality and strategy

*Where* a backend's work can run is decided by its array types, not by its policy. That is [`locality`](@ref): [`HostLocality`](@ref) for storage a CPU loop can index element by element, [`DeviceLocality`](@ref) for storage an accelerator schedules against. The names are `public` but not exported, so call them qualified:

```@example backend
import Bramble: CpuSerial, GpuKernel
Bramble.locality(Vector{Float64}), Bramble.locality(CpuSerial()), Bramble.locality(GpuKernel())
```

A [`CpuPolicy`](@ref) claims host locality and a [`GpuPolicy`](@ref) claims device locality. Any array type the package knows nothing about counts as host memory, so a custom wrapper array works as a `vector_type` without registering anything, as long as it can be indexed on the host. A GPU extension answers device locality for its own array types. The strategy is the genuine choice, and it exists only within one locality: host storage admits [`CpuSerial`](@ref), [`CpuThreaded`](@ref) and [`CpuPolyester`](@ref), and device storage admits [`GpuKernel`](@ref). The [`Backend`](@ref) constructor rejects any other pairing, and a matrix type that disagrees with the vector type, before the backend exists:

```julia
using Bramble, Metal

metal_backend(; policy = CpuSerial())
# ERROR: ArgumentError: Backend vector type MtlVector{Float32} has locality
# Bramble.DeviceLocality(), but execution policy CpuSerial has locality
# Bramble.HostLocality(): a Backend's storage and its execution policy must agree on
# locality, or the combination cannot execute anything.
```

### Introspection

[`vector_type`](@ref)`(be)` and [`matrix_type`](@ref)`(be)` give the two storage types, and [`backend_types`](@ref)`(be)` gives `(eltype, VT, MT, typeof(be))` at once. [`execution_policy`](@ref)`(x)` works on a backend, a mesh or a grid space, and answers what a threading-capable call on it would do.

### CSC or CSR storage

[`csr_backend`](@ref) swaps the default `SparseMatrixCSC{Float64,Int}` for the one-based `SparseMatrixCSR{1,T,Int}` of `SparseMatricesCSR.jl`. It needs that package loaded and throws without it.

```julia
using Bramble, SparseMatricesCSR

be = csr_backend()                 # Float64, Serial()
matrix_type(be)                    # SparseMatrixCSR{1, Float64, Int}
```

A finite-difference stencil is assembled row by row, which CSR storage reaches without the column scatter a CSC assembly needs ([#214](https://github.com/gpena/Bramble.jl/issues/214)). Measured on one machine (AC power, 1-minute load 2.13, 4 threads):

- **Assembly is a wash**: first assembly and `assemble!` refill land within about 0.94-1.05x of each other across 1D, 2D and 3D, and neither allocates in `assemble!`.
- **CSR stores the same matrix in roughly half the memory**: 5.3 MiB against 10.7 MiB in 1D, 24.4 against 59.1 MiB in 3D.
- **CSR loses the direct solve**, 2.4x to 4.2x slower across the sizes measured, because `A \ F` on a `SparseMatrixCSR` goes through a transpose wrapper around a sparse LU instead of reaching UMFPACK directly.

Reach for CSR when memory is the constraint and assembly dominates. Keep CSC when `A \ F` is on the critical path. Re-measure before relying on these figures for a specific size.

### A Polyester policy

```julia
using Bramble, Polyester

be = backend(policy = CpuPolyester())
Ωₕ = mesh(domain(interval(0.0, 1.0)), 100_000; backend = be)
execution_policy(gridspace(Ωₕ))   # CpuPolyester()
```

The policy type ships with `Bramble.jl`, but its sweeps live in the `BramblePolyesterExt` extension, which loads with `using Polyester`. After that you call `assemble`, `Rₕ!` and `avgₕ!` exactly as before.

### A GPU backend

```julia
using Bramble, Metal

gpu = metal_backend()                    # Float32, GpuKernel()
gpu = gpu_backend()                      # picks whichever GPU extension is loaded
```

`Float64` is not supported on Apple Silicon GPUs, so use `Float32` or `Float16`. [`metal_backend`](@ref) needs `Metal.jl` loaded and throws without it. [`gpu_backend`](@ref) checks which GPU extension is loaded and forwards to its constructor, which today means only `metal_backend`. It needs the extension loaded and its device functional (`Metal.functional()`), and the two failures give different errors. With no extension loaded, it names the package to import: `Metal` on Apple Silicon, `CUDA` on Linux or Windows. With an extension loaded but no working device, it reports a driver, hardware or virtualisation problem that an import cannot fix.

A kernel under [`GpuKernel`](@ref) only enqueues work on the device's queue and returns, so a chain of device operators pipelines without a host round trip between steps. Synchronisation happens at a few real boundaries:

- converting a device array to a host one, with `Array(...)` or `host_points`;
- a host-side reduction, since `innerₕ`, `normₕ` and `norm₁ₕ` return a host scalar and so wait for the queue;
- an explicit [`ka_synchronize`](@ref), for timing a device computation alone.

```julia
using Bramble, Metal
import Bramble: D₋ₓ, D₋ᵧ

Ωₕ = mesh(domain(interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0)), (1025, 1025), (true, true);
    backend = metal_backend())
Wₕ = gridspace(Ωₕ)
uₕ = Rₕ(Wₕ, x -> sin(x[1]) * cos(x[2]))

vₕ = D₋ₓ(uₕ) + D₋ᵧ(uₕ)   # two kernel launches, neither blocks on the other
s  = normₕ(vₕ)           # the actual wait happens here
```

Every allocating operator is `similar(uₕ)` plus a kernel launch, and on a GPU that allocation is a device allocation (issue #302 measured the cost). Inside a loop, prefer the in-place forms `D₋ₓ!` and `D₋ᵧ!` with buffers made once outside it. The [internals page on the GPU substrate](../internals/gpu.md) covers the kernel side.

### Offloading only the restriction

[`GpuOffload`](@ref) is a [`CpuPolicy`](@ref) that keeps host storage and sends only the fill step of `Rₕ!` and `avgₕ!` to a device backend. Every other operation runs under the inner policy it wraps:

```julia
using Bramble, Metal

be = backend(Float32; policy = Bramble.GpuOffload(metal_backend(), Bramble.CpuThreaded()))
Ωₕ = mesh(domain(interval(0.0f0, 1.0f0)), 10_000_001; backend = be)
Wₕ = gridspace(Ωₕ)
uₕ = Rₕ(Wₕ, sin)          # fills through Metal, copies the result back to a host Vector
Bramble.execution_policy(Wₕ)     # Bramble.CpuThreaded()
```

Nothing is cached on the device between calls, so each call uploads its mesh axis and copies the result back. Measured against plain `CpuThreaded()` (`benchmark/gpu_offload.jl`, commit `2c7217e6`, Apple M2, AC power, `--threads=4`): `avgₕ!` wins clearly (3.27x at 1D n=10,000,000, 10.86x at 2D 3000x3000), `Rₕ!` wins at 2D 3000x3000 (1.79x), and `Rₕ!` at 1D n=10,000,000 loses, at 0.86x. Measure the specific call and grid size before choosing it over the inner policy alone. The internals page has the full table.

### The precompilation workload

`Bramble.jl` precompiles a workload over 1D, 2D and 3D meshes. It costs a few seconds at build time and cuts the time to first result by roughly a factor of four. To skip it while developing the package itself:

```julia
using Preferences, Bramble
set_preferences!(Bramble, "precompile_workload" => false)
```

Julia tracks preferences in the precompilation cache, so the change applies on the next `using Bramble`. Restore the default with `delete_preferences!(Bramble, "precompile_workload"; force = true)`. The setting is written to `LocalPreferences.toml` next to your active project.
