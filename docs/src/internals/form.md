```@meta
CollapsedDocStrings = false
```

# Forms

## Lock-free parallel assembly

Multi-threaded assembly in `Bramble.jl` uses a lock-free multi-coloring partition. Given the stencil offsets of a bilinear form, `compute_safe_strides` calculates the minimum stride between mesh coordinates that guarantees zero overlap between simultaneous matrix writes.

`partition_grid_by_colors` then sorts the Cartesian grid indices into independent color phases:

```@raw html
<figure>
<svg viewBox="0 0 740 290" width="100%" style="max-width:740px;height:auto;font-family:system-ui,-apple-system,'Segoe UI',sans-serif"
     xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="Multi-coloring partition of a 2D computational grid for lock-free parallel form assembly.">
  <!-- Grid of 6x6 colored squares -->
  <g transform="translate(50, 40)">
    <text x="105" y="-15" font-size="13" font-weight="bold" fill="currentColor" text-anchor="middle">Grid partitioned by independent colors</text>
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
    <text x="190" y="25" font-size="14" font-weight="bold" fill="currentColor" text-anchor="middle">Lock-free parallel assembly</text>

    <!-- Step 1: Color sync -->
    <circle cx="25" cy="55" r="10" fill="#ef4444"/>
    <text x="25" y="59" font-size="11" font-weight="bold" fill="#ffffff" text-anchor="middle">1</text>
    <text x="45" y="54" font-size="12" font-weight="bold" fill="currentColor">Color phase 1 (Red cells)</text>
    <text x="45" y="70" font-size="11" fill="currentColor" opacity="0.8">All red cells are assembled in parallel across threads.</text>
    <text x="45" y="84" font-size="11" fill="currentColor" opacity="0.8">No stencil overlap = zero race conditions.</text>

    <!-- Step 2 -->
    <circle cx="25" cy="110" r="10" fill="#3b82f6"/>
    <text x="25" y="114" font-size="11" font-weight="bold" fill="#ffffff" text-anchor="middle">2</text>
    <text x="45" y="109" font-size="12" font-weight="bold" fill="currentColor">Color phase 2 (Blue cells)</text>
    <text x="45" y="125" font-size="11" fill="currentColor" opacity="0.8">Thread synchronization barrier; next color phase proceeds.</text>

    <!-- Step 3 & 4 -->
    <circle cx="25" cy="165" r="10" fill="#10b981"/>
    <text x="25" y="169" font-size="11" font-weight="bold" fill="#ffffff" text-anchor="middle">3</text>
    <text x="45" y="164" font-size="12" font-weight="bold" fill="currentColor">Color phases 3 &amp; 4 (Green &amp; Amber)</text>
    <text x="45" y="180" font-size="11" fill="currentColor" opacity="0.8">Completes the full global sparse matrix assembly.</text>

    <text x="190" y="215" font-size="11" font-weight="bold" fill="#10b981" text-anchor="middle">Strictly zero mutex locks, zero memory allocation at runtime</text>
  </g>
</svg>
</figure>
```

```@autodocs
Modules = [Bramble]
Public = false
Pages = ["form/dirichlet_constraints.jl", "form/linear.jl", "form/bilinear.jl"]
```
