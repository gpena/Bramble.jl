# Sections moved out of `docs/src/tutorials/backend.md`

As they stood. The headings and `@docs` blocks stay on the published page.

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

Nothing is cached on the device between calls, so each call uploads its mesh axis and copies the result back. Whether that transfer pays depends on the call and the grid size: `avgₕ!`, which does more work per point, gains more than `Rₕ!`. Measure the specific call and grid size with `benchmark/gpu_offload.jl` before choosing it over the inner policy alone.


# GPU passages removed from the remaining sections

## Intro

except the ones that need a GPU or an optional package, which are marked as plain code.

## Policy hierarchy

They sit in a hierarchy that splits by which processor a strategy targets:

```
ExecutionPolicy
├── CpuPolicy
│   ├── CpuSerial      (Serial)    -- one CPU thread
│   ├── CpuThreaded    (Parallel)  -- Base.Threads.@threads
│   └── CpuPolyester                -- Polyester.jl's @batch
└── GpuPolicy
    └── GpuKernel                   -- launched on the device
```

## Policy hierarchy: GpuKernel

[`GpuKernel`](@ref) runs on a device and needs a GPU backend, described in the reference below. The older names `CpuBatch` and `GpuAsync` remain as deprecated aliases of `CpuPolyester` and `GpuKernel`.

## Locality and strategy

### Locality and strategy

*Where* a backend's work can run is decided by its array types, not by its policy. That is [`locality`](@ref): [`HostLocality`](@ref) for storage a CPU loop can index element by element, [`DeviceLocality`](@ref) for storage an accelerator schedules against. The names are `public` but not exported, so call them qualified:

```@example backend
import Bramble: CpuSerial, GpuKernel
Bramble.locality(Vector{Float64}), Bramble.locality(CpuSerial()), Bramble.locality(GpuKernel())
```

A [`CpuPolicy`](@ref) claims host locality and a [`GpuPolicy`](@ref) claims device locality. Any array type the package knows nothing about counts as host memory, so a custom wrapper array needs no locality registration to work as a `vector_type`. It must still be a `DenseVector`, be indexable on the host, and be constructible as `T(undef, n)` (or `T(n)`, see `supports_undef_construction`). A GPU extension answers device locality for its own array types. The strategy is the genuine choice, and it exists only within one locality: host storage admits [`CpuSerial`](@ref), [`CpuThreaded`](@ref), [`CpuPolyester`](@ref) and `GpuOffload` (below), and device storage admits [`GpuKernel`](@ref). The [`Backend`](@ref) constructor rejects any other pairing, and a matrix type that disagrees with the vector type, before the backend exists:

```julia
using Bramble, Metal

metal_backend(; policy = CpuSerial())
# ERROR: ArgumentError: Backend vector type MtlVector{Float32} has locality
# Bramble.DeviceLocality(), but execution policy CpuSerial has locality
# Bramble.HostLocality(): a Backend's storage and its execution policy must agree on
# locality, or the combination cannot execute anything.
```

