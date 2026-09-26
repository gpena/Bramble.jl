# src/utils/device_kernels.jl: the seam between a Bramble `Backend` and a
# `KernelAbstractions.jl` device (gpena/Bramble.jl#174, S0.1 of
# .agents/plans/metal-and-apple-silicon-acceleration.md). No `KernelAbstractions` dependency
# here -- only the generic fallback, same idiom as `metal_backend`/`_metal_backend`
# (`src/utils/backend.jl:403-409`) and `export_vtk`/`_export_vtk`
# (`src/exporters/vtk_export.jl:41-52`): a helpful error naming the packages to load, not a
# bare `MethodError`.
#
# Every GPU backend extension (today: `BrambleMetalExt`, `v3.5.0`: a CUDA/ROCm/oneAPI
# extension) adds its own `ka_device` method for its own array type. That is the whole
# contract: once a backend answers `ka_device`, every `@kernel` written in
# `BrambleKernelAbstractionsExt` against `KernelAbstractions.Backend` runs on it, with no
# further wiring.

"""
    ka_device(be) -> KernelAbstractions.Backend

Return the `KernelAbstractions.jl` device backend that runs kernels for the Bramble
[`Backend`](@ref) `be`.

This is the extension contract every device backend implements: `BrambleMetalExt` adds a
method for a `Backend` built from `MtlVector`/`MtlMatrix`, returning `Metal.MetalBackend()`,
and a future GPU extension does the same for its own array types. Every `@kernel` declared
in `BrambleKernelAbstractionsExt` is written against `KernelAbstractions.Backend` alone, so
it becomes available on a new device the moment that device supplies this one method.

# Throws
- `ErrorException`: if no loaded package defines `ka_device` for `be`. Requires
  `using KernelAbstractions` and the package that supplies `be`'s device (for example
  `using Metal` for a Metal backend).
"""
function ka_device(be)
    return _throw_no_ka_device(be)
end

@noinline function _throw_no_ka_device(be)
    return error(
        "ka_device has no method for $(typeof(be)). Add `using KernelAbstractions` and " *
        "the package providing this backend's device (e.g. `using Metal`) before calling " *
        "this function.",
    )
end

"""
    ka_synchronize(x) -> Nothing

Block the calling thread until every `KernelAbstractions.jl` kernel and device transfer
already queued against `x`'s device backend has completed.

Through gpena/Bramble.jl#94's S4.2, every `@kernel` launch in `BrambleKernelAbstractionsExt`
called this right after launching, unconditionally. gpena/Bramble.jl#302/#306 (S11) removed
that: under [`GpuKernel`](@ref) -- the only [`GpuPolicy`](@ref) there is -- a kernel launch
now only enqueues onto the device's own command queue and returns, so a chain of operators
(`D₋ₓ` into `D₋ᵧ`, say) pipelines instead of paying a host round-trip after each step.
Kernels enqueued on the same queue still run in that queue's order, so this is *not* needed
between chained device calls, only at a genuine host boundary:

  - converting a device array to a host one (`Array(...)`, `host_points`, ...), or any other
    host-side read, reduction or assertion -- though a `GPUArrays` reduction that returns a
    plain host scalar (`sum`, which `_dot`'s device method in `src/utils/linear_algebra.jl`
    uses, and so `innerₕ`/`normₕ` too) already synchronises by fetching that scalar, with no
    call to this function needed;
  - a write that reaches device memory some way other than a `@kernel` launch on that same
    queue -- a plain `copyto!`, which queues asynchronously exactly like a kernel launch
    does, but is not one of the launches above. `_flush_device_scatter!`
    (`src/assembly/bilinear_traversal.jl`) is this function's first caller for exactly that
    reason: it ends a device-resident matrix's assembly with
    `copyto!(A.nzVal, mirror.nzval)`, and calling this right after is what keeps
    `assemble`/`assemble!` from returning before that write lands (S4.2's race, found at
    `n = 513` over repeated assemblies, invisible at a small `CHECK` size);
  - an explicit call from user code that needs a hard barrier before doing something this
    package cannot see, such as timing a device computation in isolation.

Same idiom as [`ka_device`](@ref): a helpful error naming the packages to load, not a bare
`MethodError`, and the real method comes from `BrambleKernelAbstractionsExt`
(`synchronize(get_backend(x))`) -- written against `KernelAbstractions` alone, so a future
GPU backend inherits it for free.

# Throws
- `ErrorException`: if no loaded package defines `ka_synchronize` for `x`. Requires
  `using KernelAbstractions`.
"""
function ka_synchronize(x)
    return _throw_no_ka_synchronize(x)
end

@noinline function _throw_no_ka_synchronize(x)
    return error(
        "ka_synchronize has no method for $(typeof(x)). Add `using KernelAbstractions` " *
        "before calling this function.",
    )
end
