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

```@docs
ka_device
ka_synchronize
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
