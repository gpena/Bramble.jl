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
swept as a single flat parallel loop with no phases at all: both `_sweep_parallel!` and
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
do. Matrix assembly colours from `_colour_strides(stencil_offsets(ast))` too, the same
function and the same quantity a vector assembly colours from
([gpena/Bramble.jl#54](https://github.com/gpena/Bramble.jl/issues/54)): `stencil_offsets`
reduces a `BilinearProduct` to its test factor's reach, since that is the only side
colouring ever needs, so there is one static answer to "what does this reach" rather than
a second one re-derived from a sample stencil evaluation.

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

## Algebraic simplification of the `+`/`*` layer

`form(Wₕ, Vₕ, f)`/`form(Wₕ, f)` call [`Bramble.simplify_ast`](@ref) on the resolved expression before
storing it (`form/simplifier.jl`, [gpena/Bramble.jl#159](https://github.com/gpena/Bramble.jl/issues/159)).
Most of it rewrites three node types: `OperatorAdd`, `OperatorScale` and `GridFunctionScale`
— exactly what `ast.jl`'s `+`, `*` and `/` overloads build. Every other node — differences,
averages, jumps, restrictions, interpolation, and every leaf — is semantic rather than
algebraic, and is left as it is. Two exceptions reach one layer deeper, into
`BilinearProduct`/`LinearProduct` (what `innerₕ`/`inner₊`/... build) and into `ShiftNode`:
leaving them untouched would mean either a correctness gap (§"Component distribution" below)
or a documented dead end (a hidden scalar defeating `symmetry.jl`'s structural shape check).

The rules matter here rather than only in the tutorial because of where the router splits
work: `_visit_operator_add*` (`stencil_eval.jl`) recurses on `OperatorAdd` alone, so every
other node is one routed term and one mesh sweep, however large the subtree underneath it.
Fewer top-level `OperatorAdd` nodes is therefore not a cosmetic rewrite of the tree but a
smaller number of sweeps for the same matrix or vector:

| Input | Simplifies to | Effect on routing |
|:--- |:--- |:--- |
| `0 * A` | a `ZeroOperator` | a one-point pattern instead of `A`'s full stencil |
| `A + 0`, `0 + A` | `A` | the zero term is not a term at all |
| `1 * A` | `A` | no wrapper node to route through |
| `c1 * (c2 * A)`, both static | `(c1 * c2) * A` | unchanged term count, one multiply instead of two |
| `A + A`, `A` a singleton node | `2 * A` | two routed terms become one |
| `c1 * A + c2 * A`, same singleton `A` | `(c1 + c2) * A` | two routed terms become one |
| `c * A + c * B`, same `c` | `c * (A + B)` | two routed terms become one |

`ZeroOperator{D,Nothing}(nothing)` is synthesized for the zero case rather than reusing a
concrete space, because a `LazyOp{D}` subtree in general carries no space to read back —
`space(op)` is only ever implemented for `IdentityOperator`/`ZeroOperator` themselves. Every
consumer of `ZeroOperator` (`local_stencil`, `stencil_offsets`, `component`) reads only its
`D` type parameter; the one exception, `symmetry.jl`'s `_same_operator_shape` comparing
`a.space === b.space`, settles `nothing === nothing` the same way two zero operators over
the same space would.

"Same `A`" is two predicates, not one. `_ast_equal` (`form/simplifier.jl`) is the *definition*:
a structural equality over `LazyOp` subtrees — the same concrete node type, and every field
equal, recursively for a field that is itself a `LazyOp`, by `===` otherwise. `===` rather than `==` for a leaf field (a grid
function, a closure, a component index) is deliberate: two arrays holding equal values right
now are not the same operator once one of them is mutated in place and the other is not, and
two independently built closures are never "the same" scaling function merely because they
compute the same thing. Missing an equal-but-distinct pair only forgoes a rewrite; treating
two different subtrees as equal would change what an assembled form computes, silently, which
none of these rules may ever do — every rewrite here is an algebraic identity, so the
assembled matrix or vector is unaffected up to floating-point association (folding `2 * x +
3 * x` into `5 * x` can move the last bit; not folding it does too, in the other direction).

`_statically_equal` is the *gate*, and it is what the like-term rule actually branches on:
whether `_ast_equal`'s answer is settled by the two types alone, which is
`Base.issingletontype`. The rule has to be gated this way because its two outcomes return
different node types — an `OperatorScale` when it fires, an `OperatorAdd` when it does not — so
an answer inference cannot fold makes `form`'s return type a `Union` of both, costing every
caller a dynamic dispatch into the assembly engine and drawing
`IllegalTypeAnalysisException` from Enzyme (gpena/Bramble.jl#240). That bit every sum of two
same-shaped terms carrying runtime data, not just the duplicate expressions it was first
thought to: `innerₕ(g₁ * u, v) + innerₕ(g₂ * u, v)` for distinct grid functions `g₁ ≠ g₂`, or
two source vectors in a linear form.

In practice the gate is generous, because the trees `form` builds out of pure operators *are*
singletons: a `BilinearProduct` over `TrialFunction`/`TestFunction`, wrapped in any stack of
difference or average nodes, has singleton fields all the way down. What it excludes is a node
carrying data — a `GridFunctionScale` holding an array, a `SourceVector`, a `DiracSource`, an
`IndexedTrialFunction` whose `component_idx` is a field rather than a type parameter. Those
sums assemble as the two terms they were written as: one extra routed term, and the same
numbers.

A `Base.RefValue` coefficient (§2's dynamic scalar coefficients) is never dereferenced by the
pass and never combined with a static number, or with a different `Ref`, only recognized as
the same coefficient when it is the same `Ref` object on both sides — the whole point of a
`Ref` coefficient is that its value can change after the form is built, so folding its
current value into a static number would bake in a snapshot the rest of the design goes out
of its way to avoid.

### Reaching one layer deeper: inner products and shifts

| Input | Simplifies to | Why |
|:--- |:--- |:--- |
| `⟨c * u, v⟩`, `⟨u, c * v⟩` | `c * ⟨u, v⟩` | exposes `c` to the rules above, and to `symmetry.jl` |
| `⟨u, v(i) + v(j)⟩`, `i ≠ j` (or the trial-side mirror) | `⟨u, v(i)⟩ + ⟨u, v(j)⟩` | the combined shape has no valid single-term routing at all |
| `u_h * (v_h * A)` | `(u_h .* v_h) * A` | one elementwise multiply at construction, not two scalings per point per assembly |
| `Shift₀(u)` | `u` | a zero shift is the identity |
| `Shift_a(Shift_b(u))`, same dimension | `Shift_{a+b}(u)` | additive, so `Shift_k(Shift_{-k}(u))` collapses to `u` via the rule above |

Scalar lifting matters beyond routing: `_same_operator_shape` (`symmetry.jl`) recognises
`⟨L(u), L(v)⟩` — the same operator chain on both sides — structurally, by comparing the
concrete node types down both arguments. `innerₕ(2 * D₋ₓ(u), D₋ₓ(v))`'s trial side used to be
an `OperatorScale` and its test side a bare `BackwardDifference` — different types, so the
check answered `false` even though `2 * ⟨Lu, Lv⟩` is exactly the symmetric,
positive-semidefinite shape it exists to recognise. Lifting the `2` out removes the mismatch.

Component distribution exists because a term naming two different components inside one
product has no other way to assemble: `test_component_or_nothing`/
`trial_component_or_nothing` (`block_extract.jl`) *throw* when the two sides of a sum they
walk into name different components, since the router needs exactly one block (or none) per
routed term and a mixed sum inside one product answers neither. Distributing it into two
clean products, each naming one component, is the only routing-safe shape — so unlike every
other rule here, this one can turn a single sweep back into two. It is guarded accordingly: a
same-component sum (`v(1) + D₋ₓ(v(1))`, or no component at all) is left as the single term it
already is, and only fires when the two sides actually disagree.

That guard — call it `_mixes_components(a, b)` — has to be checked again wherever an
`OperatorAdd` could end up hidden inside an `OperatorScale`/`GridFunctionScale` wrapper,
because hiding one there reintroduces exactly the unroutable shape: `2 * (A + B)` for `A`/`B`
naming different components would throw at assembly the same way the un-lifted `innerₕ(fₕ, v(1)
+ v(2))` above did. So rule 2's factoring step (`c * A + c * B -> c * (A + B)`) refuses to
fire when `A`/`B` mix components, and `simplify_ast(::OperatorScale)`/
`simplify_ast(::GridFunctionScale)` distribute their own coefficient over an inner sum that
mixes, rather than wrapping it, whenever `BilinearProduct`'s/`LinearProduct`'s own
distribution produces one and something still wraps it from outside.

```@autodocs
Modules = [Bramble]
Public = false
Filter = x -> x !== Bramble.DiracSource
Pages = [
    "form/ast.jl",
    "form/common.jl",
    "form/stencil_eval.jl",
    "form/simplifier.jl",
    "form/component.jl",
    "form/block_extract.jl",
    "form/stencil_pattern.jl",
    "form/symmetry.jl",
    "form/operators/average.jl",
    "form/operators/difference.jl",
    "form/operators/inner.jl",
    "form/operators/interpolation.jl",
    "form/operators/jump.jl",
    "form/operators/restriction.jl",
    "form/dirichlet_constraints.jl",
    "form/linear.jl",
    "form/bilinear.jl",
    "form/bilinear_traversal.jl",
    "form/bilinear_pattern.jl",
    "form/bilinear_execution.jl"
]
```
