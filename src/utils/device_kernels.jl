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

The extension contract every `@kernel` launch in `BrambleKernelAbstractionsExt` already
follows itself (each one calls `synchronize(get_backend(...))` right after launching), and
that any other device write must follow too: a plain `copyto!` into a device array queues
the transfer and returns immediately, exactly like a kernel launch does, so a caller reading
that array right after gets whatever has landed by then, not what the write intended
(gpena/Bramble.jl#94, S4.2 -- `_flush_device_scatter!` in `src/form/bilinear_traversal.jl`
is this function's first caller, added after that race surfaced at `n = 513` in a full-stack
test the milestone's own 33-point `CHECK` was too small to catch).

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
