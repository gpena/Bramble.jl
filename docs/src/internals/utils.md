```@meta
CollapsedDocStrings = false
```

# Utilities

## Backend

```@autodocs
Modules = [Bramble]
Public = false
Pages = ["utils/backend.jl", ]
```

## Linear algebra

```@autodocs
Modules = [Bramble]
Public = false
Pages = ["utils/linear_algebra.jl", ]
```

## The `CpuBatch` sweep hooks

Declared `public` in `src/Bramble.jl` rather than exported: `BramblePolyesterExt` implements
them by name, so they are an extension contract, not an internal. `src/` carries error-only
stubs that name Polyester. The "Linear algebra" block above filters to private names and would
drop all five.

```@autodocs
Modules = [Bramble]
Public = true
Private = false
Filter = x -> x in (
    Bramble._batch_for!, Bramble._batch_axis_for!, Bramble._batch_scatter_for!,
    Bramble._batch_dot, Bramble._batch_dot_masked
)
Pages = ["utils/linear_algebra.jl", ]
```

## Macros

```@autodocs
Modules = [Bramble]
Public = false
Pages = ["utils/macros.jl", ]
```
