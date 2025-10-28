```@meta
CollapsedDocStrings = false
CurrentModule = Bramble
```

# API

Documentation for `Bramble.jl`'s public API.

## Geometry and mesh

```@docs
box
interval
×
dim
topo_dim
markers
domain
labels
mesh
points
hₘₐₓ
npoints
change_points!
iterative_refinement!
```

## Space

```@docs
gridspace
element
mesh(Wₕ::AbstractSpaceType)
ndofs
```

### Interpolation operators

```@docs
avgₕ
avgₕ!
Rₕ
Rₕ!
```

### Differential operators

```@docs
diff₋ₓ
diff₋ᵧ
diff₋₂
diff₋ₕ
diff₊ₓ
diff₊ᵧ
diff₊₂
diff₊ₕ
D₋ₓ
D₋ᵧ
D₋₂
∇₋ₕ
D₊ₓ
D₊ᵧ
D₊₂
∇₊ₕ
jump₋ₓ
jump₋ᵧ
jump₋₂
jump₋ₕ
jump₊ₓ
jump₊ᵧ
jump₊₂
jump₊ₕ
```

### Average operators

```@docs
M₋ₓ
M₋ᵧ
M₋₂
M₋ₕ
M₊ₓ
M₊ᵧ
M₊₂
M₊ₕ
```

### Inner products and norms

```@docs
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

## Form

```@docs
dirichlet_constraints
form
assemble
assemble!
symmetrize!
```
