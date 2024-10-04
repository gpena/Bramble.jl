```@meta
CollapsedDocStrings = false
CurrentModule = Bramble
```

# API

Documentation for `Bramble.jl`'s public API.

## Geometries and meshes

```@docs
interval(x, y)
×
dim
domain(X::CartesianProduct)
create_markers
markers(Ω::Domain)
labels(Ω::Domain)
@embed
mesh(Ω::Domain, npts::NTuple{D,Int}, unif::NTuple{D,Bool}) where D
points
hₘₐₓ
npoints
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
diff₋ₕ
diffₓ
diffᵧ
diff₂
diffₕ(Wₕ::SpaceType)
D₋ₓ
D₋ᵧ
D₋₂
∇₋ₕ(Wₕ::SpaceType)
jumpₓ
jumpᵧ
jumpₕ(Wₕ::SpaceType)
jump₂
M₋ₕₓ
M₋ₕᵧ
M₋ₕ₂
M₋ₕ(Wₕ::SpaceType)
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

## Linear and bilinear forms

```@docs
form
assemble
assemble!
constraints
symmetrize!
```