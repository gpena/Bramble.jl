# GPU backends

`using Metal` enables `metal_backend()` on Apple GPUs. `gpu_backend()` picks whichever GPU
extension is loaded and has a working device (today only Metal), and throws a message naming
the package to load otherwise. Both default to `Float32` and the `GpuKernel` policy.

```julia
using Bramble, Metal
b = metal_backend(Float32)                       # GpuKernel policy, device storage
Ωₕ = mesh(Ω, (513, 513), (false, false); backend = b)
Wₕ = gridspace(Ωₕ)
```

## Traps

- **Float32 only on Metal.** There is no Float64 on Apple GPUs. Tolerances written for
  Float64 fail: compare against `eps(Float32)` (about `1.2e-7`). In device code a `Float64`
  literal (`0.5`) forces double arithmetic; divide by an integer (`/ 2`) instead.
- **Policy and storage must agree.** Device arrays need a GPU policy and host arrays a CPU
  one; `backend(Float32; policy = Parallel())` is a CPU backend and cannot use Metal arrays.
  A mismatch throws when the backend is built.
- **No scalar indexing.** A host loop over a device array throws
  `ScalarIndexingDisallowed`. Move data in one transfer (`Array(parent(uₕ))`) and work on
  the host copy, or use whole-array operations.
- **Masked restriction runs on the host only.** `Rₕ!` and `avgₕ!` with non-empty `markers`
  have no device kernel and raise under a GPU policy; masked reductions (`innerₕ`, `normₕ`
  with markers) do run on the device.
- **Kernel launches advance the global random number generator**, so seeding cannot make a
  random device mesh match a random host mesh. Build one, read its points to the host and
  apply them to the other with `Bramble.change_points!`.
- **Asynchronous execution hides races.** A small single run can pass while a large or
  repeated one fails: test device code at scale and repeat it.

## Measuring

Use `Metal.@profile` or `Metal.@bprofile` for device time, not wall-clock time around one
call. Compare variants inside one warmed process: separate processes differ by tens of
percent on millisecond-scale operations.
