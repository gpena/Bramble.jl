```@meta
CollapsedDocStrings = false
CurrentModule = Bramble
```

# Forms

## The extension contract

```@autodocs
Modules = [Bramble]
Public = false
Filter = x -> x in (
    Bramble._allocate_from_pattern, Bramble._scatter_position, Bramble._scatter_add!,
    Bramble._zero_stored!, Bramble._batch_bilinear_colour_sweep!,
    Bramble._batch_bilinear_band_sweep!, Bramble._batch_linear_colour_sweep!,
    Bramble._batch_linear_band_sweep!
)
Pages = [
    "assembly/linear.jl",
    "assembly/bilinear.jl",
    "assembly/bilinear_traversal.jl",
    "assembly/bilinear_pattern.jl",
    "assembly/bilinear_execution.jl"
]
```

```@autodocs
Modules = [Bramble]
Public = false
Filter = x -> x ∉ (
    Bramble.DiracSource, Bramble.issymmetric, Bramble.isposdef,
    Bramble._allocate_from_pattern, Bramble._scatter_position, Bramble._scatter_add!,
    Bramble._zero_stored!, Bramble._batch_bilinear_colour_sweep!,
    Bramble._batch_bilinear_band_sweep!, Bramble._batch_linear_colour_sweep!,
    Bramble._batch_linear_band_sweep!
)
Pages = [
    "ast/ast.jl",
    "ast/common.jl",
    "assembly/stencil_eval.jl",
    "ast/simplifier.jl",
    "ast/component.jl",
    "assembly/block_extract.jl",
    "ast/stencil_pattern.jl",
    "assembly/symmetry.jl",
    "operators/inner.jl",
    "operators/region_restriction.jl",
    "assembly/dirichlet_constraints.jl",
    "assembly/linear.jl",
    "assembly/bilinear.jl",
    "assembly/bilinear_traversal.jl",
    "assembly/bilinear_pattern.jl",
    "assembly/bilinear_execution.jl"
]
```

### Point sources, bandwidth analysis and structural properties

```@docs
DiracSource
issymmetric(::BilinearForm)
isposdef(::BilinearForm)
```
