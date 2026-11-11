```@meta
CollapsedDocStrings = false
CurrentModule = Bramble
```

# Inner products and norms

Three families of weights, each with its own inner product:

| Inner product | Weight at a point | Use |
|:--|:--|:--|
| [`innerₕ`](@ref) | the cell measure, the product of the per-axis cell widths | the discrete ``L^2`` product, `normₕ` |
| [`inner₊`](@ref) | the backward spacing along the chosen direction (zero at the first node), the cell width across the others | pairs with the backward differences for summation by parts, `norm₊` |
| [`inner_Γ`](@ref) | the product of the transverse half-spacings, so the sum is a ``(D-1)``-dimensional boundary integral | surface integrals over marked faces |

The weight vectors come from [`weights`](@ref), named by `Bramble.Innerh` and `Bramble.Innerplus`.

```@docs
innerₕ
inner_Γ
η
skew_symmetric
inner₊
inner₊ₓ
inner₊ᵧ
inner₊₂
normₕ
norm₁ₕ
snorm₁ₕ
norm₊
norminf
norm(::VectorElement, ::AbstractString)
```
