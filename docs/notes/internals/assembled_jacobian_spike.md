# Assembling a nonlinear form's exact Jacobian from its AST: design spike (gpena/Bramble.jl#1118)

> **Status, 2026-11-08:** a historical record of the #1118 spike, not the current rule set.
> Some rules it marks refused have since shipped in v4.2.1 (#1141, #1142, #1144); each such
> Prototype cell says so. The current truth is the docstring of
> [`jacobian_form`](https://github.com/gpena/Bramble.jl/blob/main/src/assembly/assembled_jacobian.jl).
> The Jacobian-bound timings are in [`assembled_jacobian_workload.md`](assembled_jacobian_workload.md)
> (#1145).

A prototype and its measurements, so the maintainer can decide go or no-go. No
implementation follows from this page alone. The prototype is
[`benchmark/assembled_jacobian_spike.jl`](https://github.com/gpena/Bramble.jl/blob/main/benchmark/assembled_jacobian_spike.jl)
(`module AssembledJacobianSpike`). `benchmark/*` is gitignored, so the file is tracked only
by `git add -f`, as its siblings are.

## Method

The residual is `R(u) = A(u) u - F`, where `a` assembles `A` and reads coefficients the
caller computes from the state, `cᵢ = fᵢ.(Lᵢ uₕ)`, with `Lᵢ` a stencil operator (`Mₕ(U)[1]`)
and `fᵢ` scalar. A form is linear in its trial function, so the state reaches `A` only
through `GridFunctionScale` nodes. For each such node `cᵢ * T(U)` the product rule gives

```
δ(cᵢ ⊙ T(u)) = cᵢ ⊙ T(δu) + βᵢ ⊙ Lᵢ(δu),     βᵢ = fᵢ′.(Lᵢ u) ⊙ T(u).
```

The first term is `A(u)` itself. The second is the same tree with that node replaced by
`βᵢ * Lᵢ(U)`: every node above the coefficient is linear in its operand and is rebuilt around
the new child. So the exact Jacobian is one form, `a + ∂a`, and one `assemble!`:

- `∂a` is built once, by a walk over `a`'s AST (`_dterm`, one method per node kind).
- At each state, `βᵢ` is refilled pointwise: `Lᵢ u` and `T(u)` are read with the stencils the
  assembly itself uses (`local_stencil`), at the point the coefficient is read, and `fᵢ′` is
  one scalar `ForwardDiff.derivative`. No `Dual` passes through assembly.
- Dirichlet rows are replaced by the identity in `a + ∂a`, as in the residual's `A`.

The caller names each coefficient as `Coefficient(values, U -> Lᵢ(U), fᵢ)`: the dependency
`jacobian_pattern` already takes, plus `fᵢ`, which it does not. A node on a path from the
root to a registered coefficient with no rule throws `MissingJacobianRule`, naming the node.
A subtree with no coefficient contributes nothing and needs no rule, because the walk checks
every field of every node for a registered coefficient before it concludes that. A registered
coefficient the walk never reaches throws `ArgumentError`. Coefficients the AST cannot see
(closures, `update_coefficients!` hooks) are out of scope, as #1118 states, and keep going
through AD.

## Error figure

`AssembledJacobianSpike.max_jacobian_error()` compares the assembled Jacobian with
`AutoSparse(AutoForwardDiff())` seeded by `ast_sparsity_detector`, at a state that is not a
solution. It reports the largest relative max-abs difference,
`maximum(abs, J_ast - J_fd) / maximum(abs, J_fd)`, over:

- two seeded non-uniform 2D meshes, 9 × 12 and 24 × 27 nodes;
- `CpuSerial` and `CpuThreaded`;
- two forms:
  - `:flux`, the #1118 form `-∇·(α(u)∇u)`, written
    `inner₊((αx * ∇ₕ(U)[1], αy * ∇ₕ(U)[2]), ∇ₕ(V))` with `αx = α.(Mₕ(uₕ)[1])`, …;
  - `:strong`, the same with an operator above the coefficient and a reaction term:
    `-innerₕ(D₊ₓ(αx * ∇ₕ(U)[1]), V) - innerₕ(D₊ᵧ(αy * ∇ₕ(U)[2]), V) + innerₕ(κ(u) * U, V)`.

Here `α(u) = 3 + 1/(1 + u²)` and `κ(u) = 1 + u²/4`.

| | value |
|---|---|
| `max_jacobian_error()` | **2.05e-16** |
| per case, `CpuSerial` (`:flux` 9, 24; `:strong` 9, 24) | 1.08e-16, 2.05e-16, 1.57e-16, 1.79e-16 |
| `CpuThreaded`, `:flux` 24 | 2.05e-16 (the same bits as serial) |
| control: `A(u)` (the Picard matrix) against `J_fd`, 24 × 27 | 2.4e-2 (`:flux`), 1.8e-2 (`:strong`) |

The control is there to show that the probe can fail. A Jacobian that dropped the
coefficient's derivative would be off by about 2 %, not 1e-16. The timing runs repeat the
check at every size, up to 1.0e6 dofs (table below).

## Timings

### How they were taken

- **Machine and code:** Apple M2, Julia 1.13.3, commit `8b2eb6c2` (v4-2-1's tip), default `-O`.
- **Processes:** every case ran in a fresh `--threads=4` child, started by
  `benchmark/assembled_jacobian_spike.jl --time` under
  `BRAMBLE_JULIA_SLOTS=4 julia_slot.sh 4`, alone on the machine.
- **Preflight:** `check_power_load.sh` and `gate_preflight.sh --timing` passed immediately
  before each driver run (1-minute load 1.93 and 1.98 on 8 cores, on AC, no other Julia).
  The load average rose to about 3 during the runs. That is the runs' own 4 threads.
- **Repeats:** three back-to-back processes per row. Each figure is the median of the three
  per-process minima. The spread in brackets is (max − min) / median across the processes.
- **Problem:** the `:flux` form on an `m × (m+3)` seeded non-uniform mesh. Each process
  checks the two Jacobians against each other (at most 1.9e-16). It also asserts that both
  Newton solves take the same number of iterations and agree to 1e-10.

The two routes:

- **Sparse FD** is `AutoSparse(AutoForwardDiff())`, seeded by
  `ast_sparsity_detector(a, U -> Mₕ(U))` with `GreedyColoringAlgorithm`. Its residual uses
  `type_cached_assemble!`, the fastest documented route
  (`docs/src/examples/poisson_nonlinear.jl`).
- **Assembled** refills the β buffers, then runs `assemble!` on `a + ∂a`.

Both routes share the same in-place `Float64` residual. The quantities timed:

- **Jacobian:** one warm Jacobian evaluation at a fixed state.
- **Newton step:** residual, Jacobian, `lu` and solve.
- **Solve:** Newton from `u = 0` to `‖R‖₂ ≤ 1e-10 ‖R(0)‖₂`.
- **Setup:** construction plus the first Jacobian. *Cold* is the first construction in the
  process, compilation included; the shared `Float64` residual is compiled beforehand, and
  sparse FD runs first. *Warm* is the minimum of three later constructions.

The 1 003 000-dof row comes from an earlier driver run with the same code and the same
preflight. Its `CpuThreaded` case was killed by `julia_slot.sh` at an 11 GB footprint, over
the 10 GB cap. That process held both routes and the LU factors, and where its memory went
was not measured. The completed run therefore stops at 492 100 dofs.

### Per step and per solve (milliseconds)

| dofs | policy | Jacobian: assembled | Jacobian: sparse FD | FD / assembled | Newton step: assembled | Newton step: sparse FD | solve: assembled | solve: sparse FD | iterations |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,300 | `CpuSerial` | 0.256 (2 %) | 0.959 (2 %) | 3.8× | 12.1 (0 %) | 14.7 (1 %) | 36.2 (0 %) | 43 (1 %) | 3 |
| 10,300 | `CpuThreaded` | 0.551 (7 %) | 1.02 (3 %) | 1.8× | 12.8 (15 %) | 15.3 (3 %) | 39.1 (33 %) | 46.6 (31 %) | 3 |
| 90,900 | `CpuSerial` | 2.26 (1 %) | 8.86 (2 %) | 3.9× | 171 (0 %) | 175 (3 %) | 515 (1 %) | 532 (1 %) | 3 |
| 90,900 | `CpuThreaded` | 1.64 (6 %) | 8.22 (1 %) | 5.0× | 172 (0 %) | 179 (2 %) | 517 (1 %) | 535 (1 %) | 3 |
| 492,100 | `CpuSerial` | 12.7 (1 %) | 48 (1 %) | 3.8× | 1,461 (6 %) | 1,390 (3 %) | 4,677 (3 %) | 4,423 (4 %) | 3 |
| 492,100 | `CpuThreaded` | 7.48 (25 %) | 44.6 (1 %) | 6.0× | 1,488 (2 %) | 1,380 (2 %) | 4,670 (5 %) | 4,439 (2 %) | 3 |
| 1,003,000 | `CpuSerial` | 25.5 (1 %) | 96.7 (3 %) | 3.8× | 3,538 (7 %) | 4,008 (5 %) | 11,210 (6 %) | 12,460 (5 %) | 3 |

### Setup

| dofs | policy | setup cold: assembled (s) | setup cold: sparse FD (s) | setup warm: assembled (ms) | setup warm: sparse FD (ms) | FD / assembled, warm | FD, exact pattern (ms) | process max RSS (MB) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 10,300 | `CpuSerial` | 0.72 (1 %) | 1.72 (1 %) | 1.59 (1 %) | 5.68 (4 %) | 3.6× | 0.809 (1 %) | 1047 |
| 10,300 | `CpuThreaded` | 1.09 (2 %) | 2.32 (1 %) | 2.3 (3 %) | 7.08 (8 %) | 3.1× | 0.931 (10 %) | 1118 |
| 90,900 | `CpuSerial` | 0.739 (1 %) | 1.88 (1 %) | 15.4 (3 %) | 97.1 (5 %) | 6.3× | 10.4 (1 %) | 3739 |
| 90,900 | `CpuThreaded` | 1.12 (1 %) | 2.5 (3 %) | 17.8 (5 %) | 99.9 (44 %) | 5.6× | 7.32 (1 %) | 3834 |
| 492,100 | `CpuSerial` | 0.861 (9 %) | 2.3 (1 %) | 82.4 (0 %) | 377 (5 %) | 4.6× | 59.1 (0 %) | 7601 |
| 492,100 | `CpuThreaded` | 1.21 (1 %) | 2.88 (0 %) | 94.4 (5 %) | 347 (12 %) | 3.7× | 40.8 (1 %) | 7555 |
| 1,003,000 | `CpuSerial` | 0.963 (8 %) | 2.7 (10 %) | 172 (3 %) | 711 (22 %) | 4.1× | 120 (0 %) | — |

"FD, exact pattern" is sparse FD seeded with `KnownJacobianSparsityDetector` on the assembled
Jacobian's own pattern. That pattern needs 7 colours where `ast_sparsity_detector`'s needs 9.
It does not close the gap: it is slower than the loose pattern under `CpuSerial` from 90 900
dofs on. So the 3.8× is the cost of the `Dual` sweep itself, not of the superset pattern.

### Reading the figures

- **The Jacobian costs 3.8–3.9× less under `CpuSerial` at every size, and 5–6× less under
  `CpuThreaded` from 90 900 dofs.** Under `CpuThreaded` the FD Jacobian barely beats its own
  serial time (44.6 against 48.0 ms at 492 100 dofs), while the assembled one scales
  (7.5 against 12.7 ms). At 10 300 dofs, threading costs the assembled route more than it
  saves: 0.55 ms threaded against 0.26 ms serial.
- **Setup costs 3.1–6.3× less warm and 2.1–2.8× less cold.** Cold is 0.72–1.2 s against
  1.7–2.9 s. The assembled route compiles one more `Float64` form and no `Dual` path.
  - In the table, sparse FD always ran first. A check in the other order:
    3 fresh processes per policy at 10 300 dofs, assembled route first, preflight load 1.56.
  - Under `CpuSerial` the assembled route took 0.87–0.95 s against 2.36–2.38 s for sparse FD.
  - Under `CpuThreaded` it took 1.25 s against 2.95–3.01 s.
  - Either order gives 2.1–2.8×.
- **The "Jacobian" column leaves out the `Float64` coefficient refill.** The assembled route
  reads `αx`, `αy` as `residual!` left them; the FD sweep refills its `Dual` coefficients
  itself. In Newton the residual always comes first, which is why the step and solve columns
  include it on both sides. Those columns are the like-for-like ones.
- **A Newton step and a Newton solve are dominated by the LU.** The gain is 16–18 % at
  10 300 dofs, 2–4 % at 90 900 dofs, −5 % to −8 % at 492 100 dofs and +12 % at 1 003 000
  dofs serial. The sign at the large sizes comes from the factorisation, not the Jacobian.
  - In a separate check (`m = 700`, `CpuSerial`, taken at load 2.55, so labelled as under
    load), `lu` of the assembled 5-point matrix took 1.43 s, and `lu` of the FD matrix took
    1.34 s. The FD matrix carries the 9-point superset pattern, with stored zeros.
  - The *assembled values* stored on the FD pattern took 1.29 s.
  - UMFPACK orders the two patterns differently. Either route can hand the solver either
    pattern.
- **Iterations are the same (3) for both routes, as they must be for two exact Jacobians.**
  Picard on the same problem converges linearly
  (`docs/src/examples/poisson_nonlinear.jl`).
- **Allocation.** Measured with a warm `@allocated` behind a function barrier, one process,
  `--threads=2`:

  | dofs | assembled, `CpuSerial` | assembled, `CpuThreaded` | sparse FD, `CpuSerial` | sparse FD, `CpuThreaded` |
  |---:|---:|---:|---:|---:|
  | 10 300 | 0 B | 19 360 B | 2.51 MB | 2.52 MB |
  | 90 900 | 0 B | 19 360 B | 21.8 MB | 21.8 MB |
  | 492 100 | 0 B | 19 360 B | 118 MB | 118 MB |

  - The assembled route's 19 360 B under `CpuThreaded` is the `@threads` refill, the same at
    every size.
  - Sparse FD allocates 240 B per dof on every Jacobian: `element(W, Dual)`, `A * u .- F` in
    `Dual`, and DI's buffers.

## Rule surface

The table lists every `LazyOp` struct in `src/` (found by grepping `struct … <: LazyOp` and
`<: StencilNode`, and checked against the `UnaryWrapper` and `TappedNode` unions in
`src/assembly/stencil_eval.jl`). It also lists the paths that are not nodes: composite
spaces, ghost forms, assembly routes and Dirichlet constraints. A node needs a rule only
where it can sit **above** a coefficient. Below the coefficient, `T(u)` is read with the
node's existing `local_stencil`, so stencils that already exist cover every node in that
position. "Prototype" says what the spike implements. Each effort figure is a **guess**, in
working days for one worker, tests included (non-uniform meshes, serial against threaded,
composite per block):

- XS: under ½ day.
- S: 1–2 days.
- M: 3–5 days.
- L: over a week.

| AST node or path | Rule for `∂a` | Prototype | Effort (guess) |
|---|---|---|---|
| `GridFunctionScale` (`src/ast/ast.jl`) | product rule: `c * dT + β * L(U)`, `β = f′(L u) ⊙ T(u)`, refilled per state; an unregistered scale is a constant | implemented, tested | S (productise, nested scales) |
| `OperatorScale` | rebuild `s * dT` | implemented | XS |
| `OperatorAdd` | sum of the summands' derivatives, as written (no combining, `bramble-form` §1) | implemented | XS |
| `BilinearProduct`, trial side, any weight (`InnerH`, `InnerPlus{Dim}`, `InnerPlusSet`) | rebuild with `dT` on the left, the test side unchanged | implemented, `InnerH` and `InnerPlus` tested | XS |
| `BilinearProduct` with `InnerGamma`, `InnerGammaNormal`, `InnerGammaFacet` (boundary terms, `src/operators/inner.jl`, `normal.jl`) | the same rule; `T(u)` read at facet points | generic method; boundary terms **not tested** in the spike; shipped in v4.2.1 (#1142), all three tested in `test/form/assembled_jacobian.jl` | S |
| `BilinearProduct`, coefficient on the test side | if no operator sits above the coefficient on the test side, move it to the trial side (the weight is pointwise); otherwise none: the derivative would multiply trial and test functions under one operator | **refused by name** in the spike. Partly shipped in v4.2.1 (#1142): pointwise only; an operator above the test-side coefficient is still refused | S (pointwise case); otherwise stays refused |
| Differences: `BackwardDifference`, `ForwardDifference`, `CenteredDifference`, `StarDifference`, `CrossWeightedDifference`, `SecondDifference` | linear: rebuild around `dT` (`_rebuilt_around`) | implemented (`TappedNode`); `D₊` above the coefficient tested (`:strong`) | XS |
| Averages: `BackwardAverage`, `ForwardAverage`, `CenteredAverage` | linear: rebuild | implemented (`TappedNode`) | XS |
| `JumpNode` | linear: rebuild | implemented (`TappedNode`), not tested | XS |
| `ShiftNode` | linear: rebuild. The value read off the grid must match what assembly reads for `c` there (a shift reads 0 off the grid, #352) | shipped in v4.2.1 (#1141); the spike's prototype did not cover it | XS |
| `RegionRestriction` (`markers`) | linear: rebuild; the β refill walks `_bind_walk`'s bound term | refused (generic) in the spike; shipped in v4.2.1 (#1141) | S |
| `InterpolationNode` (`πₕ`), above or below the coefficient | above: rebuild. Below: `T(u)` and `L u` are read on the source mesh through `_bind_interp_spaces`; the dependency goes through `πₕ`, as `jacobian_pattern` already takes it (`U -> Mₕ(πₕ(U))`) | refused (generic) | M |
| `LinearProduct` (a load that reads the state, `innerₕ(f(u), v)`) | becomes a `BilinearProduct`, `-innerₕ(f′(L u) * L(U), V)`, in the Jacobian form. This changes the shape: the linear form contributes to the bilinear `∂a` | refused (generic) | M |
| `SourceVector`, `SourceFunction`, `SourceConstant`, `DiracSource` | none: a leaf the AST cannot see into. A `SourceVector` registered as a coefficient falls under `LinearProduct` above | not needed | XS |
| `TrialFunction`, `TestFunction`, `IdentityOperator`, `ZeroOperator` | none: no state below them | not needed | none |
| `IndexedTrialFunction`, `IndexedTestFunction`, `_ComponentSum` (composite spaces) | the dependency names another component's leaf (`U -> U(2)`), so `β * L(U(2))` lands in an off-diagonal block. `T(u)` and `L u` read their components' slices of `u`; block routing as in `jacobian_pattern`'s `_jacobian_pattern_blocks` | refused (generic) in the spike; shipped in v4.2.1 (#1144), routed per block | M–L (`bramble-form` §4 traps) |
| `_FacetExtrapolated`, ghost forms (`GhostBilinearForm`, `GhostCompositeBilinearForm`: a padded form plus relations) | the rule applies to the padded AST; the relations do not depend on the state, so they carry over. A Robin coefficient that reads the state is a `GridFunctionScale` under `InnerGamma*` | not attempted | M |
| `RankOneTerm` (`src/assembly/rank_one.jl`) | a coefficient inside a rank-one factor: the product rule on an outer product, not attempted | not attempted | L, or refuse |
| Dirichlet constraints | rows replaced by the identity in `a + ∂a`, as in `A`. Exact while the boundary data does not read the state | implemented, tested | none |
| Assembly routes: CSC, CSR backend, matrix-free, Kronecker | `a + ∂a` is an ordinary `BilinearForm`, so each route that takes one applies (a matrix-free Jacobian-vector product for Krylov comes with it) | CSC only | S (checks per route) |
| Coefficient shapes | `f(L u)`, one scalar argument: done. `f(L₁u, L₂u, …)`, e.g. `α(\|∇u\|²)`: one β per argument, from the partials of `f`. Closures and hooks are out of scope (#1118) and keep using AD | `f(L u)` only | S–M |

Two more facts, from the timing runs:

- The coefficient-dependent part of the `:flux` Jacobian has the trial pattern of `A` (the
  5-point stencil): `nnz(J_ast) = nnz(A)`.
- `ast_sparsity_detector` returns a superset: 71 290 against 51 094 nonzeros at 10 300 dofs,
  with 9 colours where the exact pattern needs 7.

## The named refusal

`AssembledJacobianSpike.refusals()` builds two forms that put a registered coefficient under a
node with no rule. It returns the messages the walk throws:

```
MissingJacobianRule: no Jacobian rule for `ShiftNode` above a coefficient that reads the state (not in the prototype's rule set); refusing rather than dropping its term
MissingJacobianRule: no Jacobian rule for `BilinearProduct` above a coefficient that reads the state (coefficient on the test side); refusing rather than dropping its term
```

The forms are `inner₊(S₊ₓ(c * ∇ₕ(U)[1]), ∇ₕ(V)[1])` and `inner₊(∇ₕ(U)[1], c * ∇ₕ(V)[1])`.

How the refusal works:

- The fallback `_dterm(op::LazyOp, w)` walks every field of `op` (`_reads_state`).
- It returns "no contribution" only when no registered coefficient lies below `op`.
- Otherwise it throws, naming `nameof(typeof(op))`.

So a node type added later is refused until it gets a rule, never skipped. In the full rule
set, `ShiftNode` gets a one-line rule. The test-side coefficient under an operator is the
refusal that stays. Shipped in v4.2.1 (#1141), `jacobian_form` no longer refuses `ShiftNode`,
so the first message above no longer applies; it still refuses the test-side coefficient
under an operator.

## Recommendation

Recommendation: go, scoped. Adopt the assembled Jacobian for scalar grid spaces first, as a
complement to sparse ForwardDiff, not a replacement.

Reasons:

1. **It is exact.** It agrees with sparse ForwardDiff to at most 2.05e-16 relative on
   non-uniform 2D meshes:
   - `:flux` and `:strong` (an operator above the coefficient, plus a reaction term) on
     9 × 12 and 24 × 27 meshes, under both policies;
   - `:flux` at every timed size, to 1.9e-16: up to 492 100 dofs under both policies, and
     1 003 000 dofs under `CpuSerial`.
2. **The Jacobian costs 3.8–3.9× less serial and 5–6× less threaded** (from 90 900 dofs).
   - Setup costs 3–6× less warm and 2.1–2.8× less cold, whichever route runs first.
   - No `Dual` reaches assembly, so this route cannot hit the bug class of #1105 and
     #1110–#1115.
   - A warm call allocates 0 B on `CpuSerial`, where sparse FD allocates 240 B per dof.
3. **The core rule set is small.** The prototype's rule-set section is about 160 lines of
   code, docstrings included: a product rule at `GridFunctionScale`, plus
   rebuild-around-the-child for every other linear node. Those rules cover the `TappedNode`
   stencils, scales, sums and products. Everything below a coefficient reuses
   `local_stencil`.
4. **It gives `Semidiscretization`'s `jacobian!` an exact Jacobian for about 2.4× the cost
   of the Picard matrix it writes today.** At 40 600 dofs, `CpuSerial`, a single process,
   not repeated:
   - `assemble!(A)` takes 0.41 ms;
   - `a + ∂a` takes 0.64 ms, plus 0.17 ms for each of the two coefficient refills, 0.98 ms
     in all.

   Sparse FD's Jacobian costs 3.8–3.9× the assembled one serial (timing table).

What it does not buy:

- **Newton with a fresh direct LU per step gains little.** Per-solve time is within −6 % to
  +16 % of sparse FD here, because the factorisation dominates. The pattern handed to UMFPACK
  moves that figure more than the Jacobian does.
- **The payoff is in Jacobian-bound workloads:** implicit time stepping that reuses a
  factorisation, Krylov or matrix-free Newton, and repeated Jacobians in sensitivity loops.
  None of those was timed here.

Not covered by the scoped go:

- composite spaces;
- `πₕ`;
- loads that read the state (`LinearProduct`);
- ghost forms;
- rank-one terms;
- multi-argument coefficients.

Each is an M-sized follow-up, or stays refused by name (rule-surface table). The API needs
one thing `jacobian_pattern` does not take today: the scalar function `f` of each
coefficient, besides its dependency `U -> L(U)`.

Side finding, independent of the go:

- `ast_sparsity_detector`'s pattern is a strict superset of the exact one for this form:
  9-point against 5-point, 40 % more stored entries, 9 colours against 7.
- A tighter pattern does not make sparse FD faster here.
- It does change the LU's ordering and its time, by ±5–8 %.

## If go: the milestone

Five subplans, scalar core first. The full subplans, with OWNS, CHECK, EXPECT, WORKER and
WHY, are in the report on #1118.

1. **A1:** `jacobian_form(a, coefficients...)` builds `a + ∂a` on scalar grid spaces, and a
   node with no rule throws `MissingJacobianRule`. Rules: `GridFunctionScale`,
   `OperatorScale`, `OperatorAdd`, `BilinearProduct` (trial side), the `TappedNode` stencils,
   `ShiftNode` and `RegionRestriction`.
2. **A2:** boundary-term weights (`InnerGamma*`) are tested, and a coefficient sitting
   pointwise on the test side moves to the trial side.
3. **A3:** `semidiscretize(...; coefficients = ...)` makes `jacobian!(J, sd, u, p, t)` exact,
   and `nonlinear_problem` takes the same declaration.
4. **A4:** composite spaces: indexed leaves and block routing, as in `jacobian_pattern`.
5. **A5:** docs (`poisson_nonlinear` gains the assembled route), plus one measurement on a
   Jacobian-bound workload, which this spike did not time.

Later, or refused by name: `πₕ`, `LinearProduct`, ghost forms, `RankOneTerm`, and
multi-argument coefficients.
