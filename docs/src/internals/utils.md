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
