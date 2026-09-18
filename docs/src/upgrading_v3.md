# Upgrading to v3.0

Bramble 3.0 renames most of the discrete operator surface, takes 28 names off the
`using Bramble` namespace, gives every operator family a dimensional entry point, drops
`πₕ`'s source-space argument and splits the execution policies into CPU and GPU branches.

There is no deprecation machinery in this package and no CHANGELOG: this page is the
migration path. Everything below is a compile-time or load-time error after upgrading, not
a silent behaviour change — with one exception, `∇ₕ`, which is called out where it appears.

## 1. Renamed operators

Four renames, applied across the whole surface:

| v2 | v3 | What it is |
|:--|:--|:--|
| `∇₋ₕ` | `∇ₕ` | the backward gradient |
| `∇ₕ` | `Dₕ` | the cross-weighted gradient |
| `Dstar₊` | `D̽` | the star difference (`D̽ₓ`, `D̽ᵧ`, `D̽₂`) |
| `M₋` | `M` | the backward average (`Mₓ`, `Mᵧ`, `M₂`) |

!!! warning "`∇ₕ` changed meaning, not only spelling"
    In v2, `∇ₕ` was the cross-weighted gradient; in v3 it is the *backward* gradient, and
    the cross-weighted one is `Dₕ`. Code that says `∇ₕ` keeps compiling and computes
    something different. This is the one rename a compiler cannot catch for you — grep for
    it before anything else.

The rename is a text substitution, and the script that performed it in this repository is
committed as `dev/rename_v3.jl`. It is driven by a table and takes explicit paths, so it
runs over a downstream project too:

```julia
julia dev/rename_v3.jl --check path/to/your/src   # dry run, reports what would change
julia dev/rename_v3.jl path/to/your/src
```

It refuses to run twice (it aborts as soon as it finds `D̽` anywhere in the file set) and
checks that occurrence counts add up per file afterwards.

## 2. Names no longer exported

28 names left the `using Bramble` surface. Nothing was deleted: every one is still defined,
documented and reachable as `Bramble.name`, and the ones worth calling directly are declared
`public`. What changed is that they no longer land in your namespace.

The forward and unscaled families are the bulk of it — `D₊`'s directional aliases are
`public`, and `diff₋`/`diff₊` are neither exported nor public. Each family's dimensional
entry point inherits its own family's status: `D₋`, `D̽`, `Dc` and `jump` are exported, `D₊`
is `public`, `diff₋`/`diff₊` are internal.

If a name stops resolving, prefix it:

```julia
u = Bramble.diff₋ₓ(uₕ)
```

## 3. Dimensional entry points

Every family now has one entry point that takes the direction as an argument, alongside the
per-coordinate aliases it always had:

```julia
D₋(uₕ, 1)        # same method D₋ₓ(uₕ) reaches
D₋(uₕ, :x)       # and so does this
D₋(uₕ, Val(1))   # and this
```

All three are type-stable: the dimensional form branches on the mesh dimension first and
then over that many literal `Val`s, so no `Val` is ever built from a runtime value, and an
out-of-range direction reports the mesh's own bound.

Two things to know:

- **There is no bare `M` or `M₊`.** `M` is the most common local name for a mass matrix in
  finite-element code, and exporting it would make a caller's own `M = assemble(a, Wₕ)`
  fail with "cannot assign a value to imported variable M". The averages put the direction
  on `Mₕ`/`M₊ₕ` instead: `Mₕ(uₕ)` is the tuple over every coordinate, `Mₕ(uₕ, 2)` the `y`
  average.
- **`Dₕ` carries both arities.** `Dₕ(uₕ)` is the cross-weighted gradient (a tuple over every
  coordinate); `Dₕ(uₕ, 2)` is its `y` component.

At the form layer the entry points take a `Val` alone — `Dim` is a type parameter of the AST
node, so an `Int` direction would make the return type a `Union`.

The 24 in-place `!` variants were left as they are: they are reached directly, never through
a form, so they have no dimensional entry point of their own.

## 4. `πₕ` over a trial function takes one argument

The bilinear interpolation operator used to take the space it interpolates from:

```julia
# v2
a = form(Wsrc, Wtest, (u, v) -> innerₕ(πₕ(Wsrc, u), v))

# v3
a = form(Wsrc, Wtest, (u, v) -> innerₕ(πₕ(u), v))
```

That argument was always required to be the trial function's own space — it was checked, and
naming any other was refused. Assembly now supplies it, once the concrete trial leaf is
known, which is also what makes it work on a composite space:

```julia
a = form(Vₕ, Vₕ, (u, v) -> innerₕ(πₕ(u(2)), v(1)))
```

Unchanged: the numeric `πₕ(Wₕ, uₕ)`/`πₕ!(dest, src)` pair, which interpolates a grid
function's *values* and still names the destination space; and the symbolic source `πₕ(uₕ)`,
which wraps a known grid function for the source side of a linear form. All three are
methods of the one name, told apart by what they are given.

An operator written *inside* the interpolation is still refused, and interpolating the test
function is still refused.

## 5. Execution policies split into CPU and GPU

```
ExecutionPolicy
├── CpuPolicy
│   ├── CpuSerial      (const Serial   = CpuSerial)
│   └── CpuThreaded    (const Parallel = CpuThreaded)
└── GpuPolicy
    └── GpuAsync
```

`Serial` and `Parallel` are aliases and keep working everywhere, so no existing call site
needs touching. Two behaviour changes:

- `metal_backend()` now carries `GpuAsync()` instead of `Serial()`. A GPU is massively
  parallel and cannot execute serially; the old default was a false statement about the
  hardware rather than a conservative choice.
- The CPU sweeps refuse a `GpuPolicy` with a message naming it, instead of failing on
  scalar indexing several frames deeper.

There is no `CpuBatch`: a Polyester-backed policy arrives with the extension that implements
it, not before.

## 6. Custom backend array types declare how they construct

A backend's vector and matrix types used to be allocated by calling `VT(undef, n)` and
catching the failure to decide whether to try `VT(n)`. That is now a trait, read from the
type:

```julia
Bramble.supports_undef_construction(::Type{<:MySizedArray}) = false
```

The default is `true`, which is what every array type in `Base` and in the GPU packages
answers, so most custom types need nothing. A type built from its dimensions alone must
declare it — otherwise the allocation fails with a message telling you exactly that, rather
than falling back silently.

`Vector`, `DenseMatrix` and `SparseMatrixCSC` each have their own specialised method and
never reach the trait, so nothing about a default backend changes.

## Checklist

1. Grep for `∇ₕ` and decide, per occurrence, whether it means `∇ₕ` or `Dₕ`.
2. Run `dev/rename_v3.jl --check` over your sources, then without `--check`.
3. Fix whatever stops resolving by prefixing `Bramble.` or switching to the family's entry
   point.
4. Drop the space argument from every `πₕ` applied to a trial function.
5. If you implement a custom backend array type, declare
   `supports_undef_construction` if it does not take `undef`.
