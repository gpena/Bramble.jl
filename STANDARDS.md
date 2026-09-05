# Standards

Where each standard is defined, so there is one authority per rule. Skills live in
`.agents/skills/<name>/SKILL.md`. For etiquette and commands see [CLAUDE.md](CLAUDE.md).

## Naming

| Rule | Defined in |
|---|---|
| Mathematical entities (`Ωₕ`, `Wₕ`, `uₕ`) | `bramble-naming` §1 |
| Component and coordinate indexing | `bramble-naming` §2 |
| Discretisation parameters (`h`, `hₘₐₓ`) | `bramble-naming` §3 |
| Directional subscripts (`D₋ₓ`, `M₊ᵧ`, `∇₋ₕ`) | `bramble-naming` §4 |
| `!` on mutating functions | `bramble-naming` §5 |

## Documentation

| Rule | Defined in |
|---|---|
| Docstring template | `bramble-documentation` §2.1 |
| Explicit signatures and return types | `bramble-documentation` §2.2 |
| Concise, mathematical register | `bramble-documentation` §2.3 |
| `[`…`](@ref)` must resolve | `bramble-documentation` §2.4 |
| Examples and doctests | `bramble-documentation` §2.5 |
| Comments and maintainer notes | `bramble-documentation` §2.6 |
| Human-written style contract | `bramble-documentation` §2.7 |
| Heading conventions | `bramble-documentation` §1 |
| Mesh diagrams as inline SVG | `bramble-documentation` §3 |

An example is verified by what it computes, not by whether it runs. A `@ref` only resolves
for documented names, so a broken one can survive the docs build.

## Types and dispatch

Abstract types head the file owning the concept (`AbstractMeshType{D}`,
`AbstractSpaceType{N}`, `OperatorType`, `ExecutionPolicy`, `GridDirection`); concrete types
subtype them in the same subsystem. Traits carry what is not a subtype relation:
`ExecutionPolicy` (`Serial()`/`Parallel()`) and `StencilShiftTrait`.

| Rule | Defined in |
|---|---|
| Dispatch on `Val`, never branch on a runtime value | `julia-performance` §2.1 |
| Never build a type from a runtime integer | `julia-performance` §2.2 |
| Promote across *all* arguments | `julia-performance` §2.3 |
| `adjoint`, not `transpose`, for inner products | `julia-performance` §2.4 |
| Component extraction is a zero-copy view | `julia-performance` §2.5 |
| The `Core.Box` closure trap | `julia-performance` §2.6 |
| The include-order rule | `julia-performance` §3 |
| `@inbounds` and where bounds checks belong | `julia-performance` §4 |
| A mutating function returns what it mutated | `julia-performance` §5.1 |
| **Element types come from the data, never the space** | `bramble-verification` §4 |

The last is load-bearing: `ForwardDiff.Dual` must flow through assembly, so a type read off
the space breaks AD. It has gone wrong four times.

## Tests

`test/` mirrors `src/` folder for folder; `runtests.jl` includes each file into a named
`@testset`. A new source file needs a test file in the matching folder and a line in
`runtests.jl` — nothing is picked up automatically.

- Shared helpers go in `runtests.jl`, never per file: every file is included into `Main`, so
  a second definition silently overwrites the first (`_fd`, `_tri`, `_matches_fd`).
- AD tests compare against a finite difference, not against not-throwing (`_matches_fd`).
- Allocation assertions go behind a function barrier — at `@testset` scope they measure the
  closure. `bramble-verification` §1, `julia-performance` §1.1 and §1.8.
- Composite-space tests need distinct values per component and per-block assertions, so a
  wrong answer cannot pass. `bramble-verification` §5.
- Seed non-uniform meshes; compare with both absolute and relative tolerance.
  `julia-performance` §8.
- `BRAMBLE_TEST_GROUP`: `all` (default), `unit`, `quality`, `ad`, `full`. Mooncake and
  Enzyme sit behind `ad`/`full` and run weekly.

## Gated, not advisory

- `ALLOCATION_BOUNDS` in `benchmark/benchmarks.jl` are exact, not slack. `Serial()`
  guarantees zero unconditionally.
- Timings are read by a person, never gated. Before calling one a regression read
  `bramble-verification` §2 — the same-run ratio is the statistic that survives.
- Benchmarks need mains power and an idle machine; the harness refuses to save on battery.

## Before a change is done

`bramble-code-review` §Gates. In short: full suite green with the broken count unchanged,
JuliaFormatter clean over `src` and `test` (nothing checks this on push —
`bramble-formatting` §4), and for anything that renders, a build you have looked at.
