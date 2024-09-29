```@meta
CollapsedDocStrings = false
CurrentModule = Bramble
```

# API

Documentation for `Bramble.jl`'s public API.

## Geometries

```@docs
interval(x, y)
×
embed(X::CartesianProduct, f)
↪(X::CartesianProduct, f)
domain(X::CartesianProduct)
create_markers
markers(Ω::Domain)
labels(Ω::Domain)
embed(Ω::Domain, f)
↪(Ω::Domain, f)
```

## Meshes

```@docs
mesh(Ω::Domain{CartesianProduct{1,T},MarkersType}, npts::Int, unif::Bool) where {T,MarkersType}
mesh(Ω::Domain, npts::NTuple{D,Int}, unif::NTuple{D,Bool}) where D
points
hₘₐₓ
embed(Ωₕ::MeshType, f)
↪(Ωₕ::MeshType, f)
```

## Spaces

```@docs
gridspace
element
mesh(Wₕ::SpaceType{MType}) where MType
avgₕ
avgₕ!
Rₕ
Rₕ!
diff₋ₓ
diff₋ᵧ
diff₋₂
diff₋
diffₓ
diffᵧ
diff₂
diff(Wₕ::SpaceType)
D₋ₓ
D₋ᵧ
D₋₂
∇ₕ
jumpₓ
jumpᵧ
jump₂
jump
Mₕₓ
Mₕᵧ
Mₕ₂
Mₕ
innerₕ
inner₊ₓ
inner₊ᵧ
inner₊₂
inner₊
normₕ
norm₊
snorm₁ₕ
norm₁ₕ
```
