```@meta
CollapsedDocStrings = false
```

# Forms

## Lock-free parallel assembly

Threaded assembly needs no locks, and no per-thread buffers to reduce afterwards. It
partitions the grid by *stride*, so that two points written at the same time cannot touch the
same entry.

`_colour_strides` reads the offsets an operator's stencil reaches and returns, per dimension,
`hi - lo + 1`: the width of the footprint one grid point writes. Two points of the same colour
differ by a multiple of that stride in some dimension, so by at least `span + 1` there, while
each writes a footprint `span` wide about itself. More than a width apart, the footprints
cannot overlap, so no two points in a colour ever target the same row and the sweep needs no
coordination of any kind.

The number of colours is `prod(strides)`, and the common case is one:

| form | offsets reached | strides | colours |
|:--- |:--- |:--- |:--- |
| `innerₕ(fₕ, v)` | `(0, 0)` | `(1, 1)` | 1 |
| `innerₕ(fₕ, D₋ₓ(v))` | `(-1, 0)`, `(0, 0)` | `(2, 1)` | 2 |
| `inner₊(∇₋ₕ(fₕ), ∇₋ₕ(v))` | `(-1, 0)`, `(0, -1)`, `(0, 0)` | `(2, 2)` | 4 |

Any form whose test argument carries no difference strides by 1 in every dimension, and is
swept as a single flat parallel loop with no phases at all — both `_sweep_parallel!` and
`_sweep_bilinear!` check `prod(strides) == 1` and take that path directly.

A colour is a strided sub-grid, not a materialised list of indices:

```julia
_colour_subgrid(grid_inds, c, strides) =
    CartesianIndices(ntuple(d -> c[d]:strides[d]:last(axes(grid_inds, d)), D))
```

so a colour costs nothing to build, and the writes within one still run in ascending order.
The implementation this replaced binned every index into a vector of vectors.

### Matrix assembly colours on the test side alone

A bilinear stencil writes to `(I + off_v, I + off_u)`, so two points collide on an entry only
if their *row* footprints overlap: rows disjoint implies entries disjoint whatever the columns
do. `_bilinear_colour_strides` therefore takes the span of the test-side offsets only, which
is the same quantity `_colour_strides` computes for a vector assembly.

It reads them from an evaluated sample stencil rather than from `stencil_offsets`, which
refuses a `BilinearProduct` on purpose: its offsets are pairs, and only one side is wanted
here.

The colouring is what makes the matrix sweep correct rather than merely fast. `add_to_sparse!`
searches a column and updates the entry in place, so two threads landing on the same entry
would race on the value, not just on the structure.

```@raw html
<figure>
<svg viewBox="0 0 740 290" width="100%" style="max-width:740px;height:auto;font-family:system-ui,-apple-system,'Segoe UI',sans-serif"
     xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="A grid partitioned into four colours by a stride of two in each dimension, for lock-free parallel form assembly.">
  <!-- Grid of 6x6 colored squares -->
  <g transform="translate(50, 40)">
    <text x="105" y="-15" font-size="13" font-weight="bold" fill="currentColor" text-anchor="middle">Four colours of a (2, 2) stride</text>
    <!-- 6x6 grid cells of size 35x35 -->
    <!-- Row 1 -->
    <rect x="0"   y="0" width="35" height="35" fill="#ef4444" fill-opacity="0.7" stroke="currentColor" stroke-width="0.8"/>
    <rect x="35"  y="0" width="35" height="35" fill="#3b82f6" fill-opacity="0.7" stroke="currentColor" stroke-width="0.8"/>
    <rect x="70"  y="0" width="35" height="35" fill="#ef4444" fill-opacity="0.7" stroke="currentColor" stroke-width="0.8"/>
    <rect x="105" y="0" width="35" height="35" fill="#3b82f6" fill-opacity="0.7" stroke="currentColor" stroke-width="0.8"/>
    <rect x="140" y="0" width="35" height="35" fill="#ef4444" fill-opacity="0.7" stroke="currentColor" stroke-width="0.8"/>
    <rect x="175" y="0" width="35" height="35" fill="#3b82f6" fill-opacity="0.7" stroke="currentColor" stroke-width="0.8"/>

    <!-- Row 2 -->
    <rect x="0"   y="35" width="35" height="35" fill="#10b981" fill-opacity="0.7" stroke="currentColor" stroke-width="0.8"/>
    <rect x="35"  y="35" width="35" height="35" fill="#f59e0b" fill-opacity="0.7" stroke="currentColor" stroke-width="0.8"/>
    <rect x="70"  y="35" width="35" height="35" fill="#10b981" fill-opacity="0.7" stroke="currentColor" stroke-width="0.8"/>
    <rect x="105" y="35" width="35" height="35" fill="#f59e0b" fill-opacity="0.7" stroke="currentColor" stroke-width="0.8"/>
    <rect x="140" y="35" width="35" height="35" fill="#10b981" fill-opacity="0.7" stroke="currentColor" stroke-width="0.8"/>
    <rect x="175" y="35" width="35" height="35" fill="#f59e0b" fill-opacity="0.7" stroke="currentColor" stroke-width="0.8"/>

    <!-- Row 3 -->
    <rect x="0"   y="70" width="35" height="35" fill="#ef4444" fill-opacity="0.7" stroke="currentColor" stroke-width="0.8"/>
    <rect x="35"  y="70" width="35" height="35" fill="#3b82f6" fill-opacity="0.7" stroke="currentColor" stroke-width="0.8"/>
    <rect x="70"  y="70" width="35" height="35" fill="#ef4444" fill-opacity="0.7" stroke="currentColor" stroke-width="0.8"/>
    <rect x="105" y="70" width="35" height="35" fill="#3b82f6" fill-opacity="0.7" stroke="currentColor" stroke-width="0.8"/>
    <rect x="140" y="70" width="35" height="35" fill="#ef4444" fill-opacity="0.7" stroke="currentColor" stroke-width="0.8"/>
    <rect x="175" y="70" width="35" height="35" fill="#3b82f6" fill-opacity="0.7" stroke="currentColor" stroke-width="0.8"/>

    <!-- Row 4 -->
    <rect x="0"   y="105" width="35" height="35" fill="#10b981" fill-opacity="0.7" stroke="currentColor" stroke-width="0.8"/>
    <rect x="35"  y="105" width="35" height="35" fill="#f59e0b" fill-opacity="0.7" stroke="currentColor" stroke-width="0.8"/>
    <rect x="70"  y="105" width="35" height="35" fill="#10b981" fill-opacity="0.7" stroke="currentColor" stroke-width="0.8"/>
    <rect x="105" y="105" width="35" height="35" fill="#f59e0b" fill-opacity="0.7" stroke="currentColor" stroke-width="0.8"/>
    <rect x="140" y="105" width="35" height="35" fill="#10b981" fill-opacity="0.7" stroke="currentColor" stroke-width="0.8"/>
    <rect x="175" y="105" width="35" height="35" fill="#f59e0b" fill-opacity="0.7" stroke="currentColor" stroke-width="0.8"/>

    <!-- Row 5 -->
    <rect x="0"   y="140" width="35" height="35" fill="#ef4444" fill-opacity="0.7" stroke="currentColor" stroke-width="0.8"/>
    <rect x="35"  y="140" width="35" height="35" fill="#3b82f6" fill-opacity="0.7" stroke="currentColor" stroke-width="0.8"/>
    <rect x="70"  y="140" width="35" height="35" fill="#ef4444" fill-opacity="0.7" stroke="currentColor" stroke-width="0.8"/>
    <rect x="105" y="140" width="35" height="35" fill="#3b82f6" fill-opacity="0.7" stroke="currentColor" stroke-width="0.8"/>
    <rect x="140" y="140" width="35" height="35" fill="#ef4444" fill-opacity="0.7" stroke="currentColor" stroke-width="0.8"/>
    <rect x="175" y="140" width="35" height="35" fill="#3b82f6" fill-opacity="0.7" stroke="currentColor" stroke-width="0.8"/>

    <!-- Row 6 -->
    <rect x="0"   y="175" width="35" height="35" fill="#10b981" fill-opacity="0.7" stroke="currentColor" stroke-width="0.8"/>
    <rect x="35"  y="175" width="35" height="35" fill="#f59e0b" fill-opacity="0.7" stroke="currentColor" stroke-width="0.8"/>
    <rect x="70"  y="175" width="35" height="35" fill="#10b981" fill-opacity="0.7" stroke="currentColor" stroke-width="0.8"/>
    <rect x="105" y="175" width="35" height="35" fill="#f59e0b" fill-opacity="0.7" stroke="currentColor" stroke-width="0.8"/>
    <rect x="140" y="175" width="35" height="35" fill="#10b981" fill-opacity="0.7" stroke="currentColor" stroke-width="0.8"/>
    <rect x="175" y="175" width="35" height="35" fill="#f59e0b" fill-opacity="0.7" stroke="currentColor" stroke-width="0.8"/>
  </g>

  <!-- Explanation Panel -->
  <g transform="translate(320, 30)">
    <rect x="0" y="0" width="380" height="230" rx="6" fill="none" stroke="currentColor" stroke-opacity="0.2" stroke-width="1"/>
    <text x="190" y="25" font-size="14" font-weight="bold" fill="currentColor" text-anchor="middle">One colour at a time, in parallel</text>

    <!-- Step 1: Color sync -->
    <circle cx="25" cy="55" r="10" fill="#ef4444"/>
    <text x="25" y="59" font-size="11" font-weight="bold" fill="#ffffff" text-anchor="middle">1</text>
    <text x="45" y="54" font-size="12" font-weight="bold" fill="currentColor">Color phase 1 (Red cells)</text>
    <text x="45" y="70" font-size="11" fill="currentColor" opacity="0.8">Every red point is swept in parallel across threads.</text>
    <text x="45" y="84" font-size="11" fill="currentColor" opacity="0.8">Footprints cannot overlap, so there is nothing to coordinate.</text>

    <!-- Step 2 -->
    <circle cx="25" cy="110" r="10" fill="#3b82f6"/>
    <text x="25" y="114" font-size="11" font-weight="bold" fill="#ffffff" text-anchor="middle">2</text>
    <text x="45" y="109" font-size="12" font-weight="bold" fill="currentColor">Color phase 2 (Blue cells)</text>
    <text x="45" y="125" font-size="11" fill="currentColor" opacity="0.8">The threaded loop joins, then the next colour proceeds.</text>

    <!-- Step 3 & 4 -->
    <circle cx="25" cy="165" r="10" fill="#10b981"/>
    <text x="25" y="169" font-size="11" font-weight="bold" fill="#ffffff" text-anchor="middle">3</text>
    <text x="45" y="164" font-size="12" font-weight="bold" fill="currentColor">Color phases 3 &amp; 4 (Green &amp; Amber)</text>
    <text x="45" y="180" font-size="11" fill="currentColor" opacity="0.8">Four colours complete the grid, for this stencil.</text>

    <text x="190" y="215" font-size="11" font-weight="bold" fill="#10b981" text-anchor="middle">No locks, and each colour is a range rather than a list</text>
  </g>
</svg>
</figure>
```

```@autodocs
Modules = [Bramble]
Public = false
Pages = [
    "form/common.jl",
    "form/component.jl",
    "form/block_extract.jl",
    "form/stencil_pattern.jl",
    "form/operators/average.jl",
    "form/operators/difference.jl",
    "form/operators/inner.jl",
    "form/operators/interpolation.jl",
    "form/operators/jump.jl",
    "form/operators/restriction.jl",
    "form/dirichlet_constraints.jl",
    "form/linear.jl",
    "form/bilinear.jl"
]
```
