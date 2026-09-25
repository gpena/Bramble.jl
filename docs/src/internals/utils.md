```@meta
CollapsedDocStrings = false
CurrentModule = Bramble
```

# Utilities

## Backend

```@autodocs
Modules = [Bramble]
Public = false
Pages = ["utils/backend.jl", ]
```

## Device dispatch

`ka_device` (`utils/device_kernels.jl`) is private. The [GPU internals page](gpu.md)'s own
`@autodocs` block for that file filters to `ka_synchronize` only, so `ka_device` is
documented here instead.

```@docs
ka_device
```

## Linear algebra

```@autodocs
Modules = [Bramble]
Public = false
Filter = x -> x ∉ (
    Bramble._batch_for!, Bramble._batch_axis_for!, Bramble._batch_scatter_for!,
    Bramble._batch_dot, Bramble._batch_dot_masked, Bramble._gpu_for!, Bramble._gpu_scatter_for!
)
Pages = ["utils/linear_algebra.jl", ]
```

## The `CpuPolyester` sweep hooks

Private since gpena/Bramble.jl#339, but still an extension contract rather than an internal:
`BramblePolyesterExt` implements them by their qualified `Bramble.` names. `src/` carries
error-only stubs that name Polyester. The "Linear algebra" block above filters all five out so
they are listed here.

```@autodocs
Modules = [Bramble]
Public = false
Filter = x -> x in (
    Bramble._batch_for!, Bramble._batch_axis_for!, Bramble._batch_scatter_for!,
    Bramble._batch_dot, Bramble._batch_dot_masked
)
Pages = ["utils/linear_algebra.jl", ]
```

## The `GpuPolicy` sweep hooks

Private since gpena/Bramble.jl#339, but still an extension contract rather than an internal:
`BrambleKernelAbstractionsExt` implements them by their qualified `Bramble.` names with
`KernelAbstractions.@kernel`s. `src/utils/linear_algebra.jl` carries error-only stubs that
throw when no device extension is loaded. The "Linear algebra" block above filters both out so
they are listed here.

```@autodocs
Modules = [Bramble]
Public = false
Filter = x -> x in (Bramble._gpu_for!, Bramble._gpu_scatter_for!)
Pages = ["utils/linear_algebra.jl", ]
```

## Macros

```@autodocs
Modules = [Bramble]
Public = false
Pages = ["utils/macros.jl", ]
```
