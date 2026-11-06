# A Jacobian-bound workload: Newton-GMRES, assembled Jacobian against sparse ForwardDiff (gpena/Bramble.jl#1145)

The #1118 spike ([`assembled_jacobian_spike.md`](assembled_jacobian_spike.md)) showed the
assembled Jacobian 3.8-6× cheaper than sparse ForwardDiff, but its one Newton figure used a
direct LU, which dominated the step. This page times a Newton-Krylov step: the residual,
the Jacobian, an ILU(0) of that Jacobian and GMRES to a fixed tolerance, with no LU.

**Result: the workload is not Jacobian-bound.** The Jacobian is 0.1-1% of a step on the
assembled route and 0.7-3% on sparse ForwardDiff, at both sizes and both policies. GMRES
dominates the step. The Jacobian is still cheaper on the assembled route (4.5-5× serial at
90 900 dofs), but that saves under 1% of a step here.

The driver is [`benchmark/assembled_jacobian_workload.jl`](https://github.com/gpena/Bramble.jl/blob/main/benchmark/assembled_jacobian_workload.jl)
(`module AssembledJacobianWorkload`). `benchmark/*` is gitignored, so the file is tracked
only by `git add -f`.

## Problem and routes

- **Problem:** `-∇·(α(u) ∇u) = g` on an `m × (m+3)` seeded non-uniform 2D mesh, with
  `α(s) = 3 + 1/(1 + s²)`, `g = 1 + x y`, and `u = exp(x + y)` on the boundary (Dirichlet
  rows). The state is `u = 0.3 + 0.7 sin(0.37 i)` for the Jacobian figures. Newton starts
  from `u = 0`.
- **Residual:** `R(u) = A(u) u - F`, with `A` assembled in place from `αₓ = α(Mₓu)` and
  `αᵧ = α(Mᵧu)`. Both routes share this `Float64` residual.
- **Assembled Jacobian:** `jacobian_form(a, αₓ => (U -> Mₕ(U)[1], α), αᵧ => (U -> Mₕ(U)[2], α);
  dirichlet = :boundary)`, evaluated by `jacobian!(J, jf, u)` into `jacobian_prototype(jf)`.
- **Sparse ForwardDiff Jacobian:** `AutoSparse(AutoForwardDiff())` seeded by
  `ast_sparsity_detector(a, U -> Mₕ(U))` with `GreedyColoringAlgorithm`, its residual built by
  `type_cached_assemble!`, as in `docs/src/examples/poisson_nonlinear.jl`.
- **Linear solve:** one ILU(0) of the route's own Jacobian (`ILUZero.ilu0`), rebuilt at every
  Newton step, then GMRES (`LinearSolve`'s `KrylovJL_GMRES`, default restart) with relative
  tolerance `1e-8` and absolute tolerance `0`. A GMRES that stops short is an error.
- **Newton stopping:** `‖R‖₂ ≤ 1e-8 ‖R(0)‖₂`.

Each route factors its own Jacobian, so the two ILU(0) preconditioners differ. The sparse
ForwardDiff matrix carries the 9-point superset pattern (`nnz` 71 290 at 10 300 dofs and
633 890 at 90 900), against the assembled 5-point pattern (51 094 and 453 294). The ILU(0)
keeps the pattern it is given, so the two preconditioners differ. GMRES needs more
iterations on the assembled route: 439 against 265 in one Newton solve at 10 300 dofs, and
1 301 against 741 at 90 900. The step and solve columns therefore include a
preconditioner effect as well as the Jacobian cost. The Jacobian columns and the Jacobian
share are the like-for-like figures.

## Correctness (`smoke()`, and every timed process)

- `smoke()` runs `m = 9` (9 × 12) under `CpuSerial` and `CpuThreaded`. The two Jacobians
  agree to 3.4e-16, and both Newton runs take 3 iterations and agree to 5e-14.
- Every timed process rechecks the Jacobians at the state: max relative error 1.9e-16 at
  10 300 dofs and 2.1e-16 at 90 900 dofs.
- Both routes take the same Newton iteration count (3) in every process.

## Machine and runs

- **Machine:** Apple M2, 8 logical cores, macOS 27.0.1, on AC power. Julia 1.13.3, default
  `-O`.
- **Commit:** `f1d45d1d` (the driver). The timed run used that commit, plus a later commit
  that changes only a banner comment.
- **Preflight:** `.claude/scripts/check_power_load.sh`, then `.claude/scripts/gate_preflight.sh --timing`
  passed at one-minute load 1.79 on 8 cores, on AC power, with no other Julia and nothing above
  50% CPU. Earlier attempts failed on load (2.0-2.5) from other Claude sessions on the
  machine, so the run waited for a quiet window.
- **Command:** `BRAMBLE_JULIA_SLOTS=4 .claude/scripts/julia_slot.sh 4 julia --startup-file=no --threads=4 --project=test benchmark/assembled_jacobian_workload.jl --time 100 300`.
  One fresh process per (size, policy, run), three runs back to back. The run ran alone.
- **Memory:** the largest child reached 2.2 GB resident, under the 10 GB cap.
- **Sizes:** `m = 100` (10 300 dofs) and `m = 300` (90 900 dofs). `m = 700` was not timed.
  GMRES iterations grew from 139 per step at 10 300 dofs to 382 at 90 900. A 492 100-dof run
  would take far longer and would not change the finding.

## Figures

Each row is the median over three fresh processes of the per-process minimum over repeats,
in milliseconds. The spread in brackets is `(max − min) / median` across the three processes.

| dofs | policy | Jacobian: assembled | Jacobian: sparse AD | Newton step: assembled | Newton step: sparse AD | Newton solve: assembled | Newton solve: sparse AD |
|---:|---|---:|---:|---:|---:|---:|---:|
| 10 300 | `CpuSerial` | 0.206 [6.3 %] | 0.932 [3.3 %] | 61.3 [0.4 %] | 35.5 [0.4 %] | 199 [17 %] | 113 [1.2 %] |
| 10 300 | `CpuThreaded` | 0.659 [5.7 %] | 1.05 [23 %] | 62.1 [0.8 %] | 36.4 [17 %] | 200 [0.7 %] | 117 [1.1 %] |
| 90 900 | `CpuSerial` | 1.83 [1.7 %] | 8.90 [0.3 %] | 2 556 [1.7 %] | 1 217 [2.8 %] | 9 812 [1.1 %] | 4 296 [2.5 %] |
| 90 900 | `CpuThreaded` | 1.65 [11 %] | 8.27 [0.9 %] | 2 676 [7.2 %] | 1 258 [7.8 %] | 10 530 [5.2 %] | 4 715 [5.7 %] |

Jacobian share of a Newton step (`Jacobian / step`, median over the three processes, spread
in brackets):

| dofs | policy | assembled | sparse AD |
|---:|---|---:|---:|
| 10 300 | `CpuSerial` | 0.34 % [6.7 %] | 2.6 % [3.5 %] |
| 10 300 | `CpuThreaded` | 1.1 % [4.9 %] | 2.9 % [37 %] |
| 90 900 | `CpuSerial` | 0.072 % [3.0 %] | 0.73 % [2.5 %] |
| 90 900 | `CpuThreaded` | 0.062 % [18 %] | 0.65 % [7.8 %] |

Krylov iterations in one Newton solve (three steps): 439 assembled and 265 sparse AD at
10 300 dofs; 1 301 and 741 at 90 900 dofs. Newton takes 3 steps in every row.

## Reading the figures

- **The Jacobian is not a large share of the step.** On both routes it is below 3%, so
  making the Jacobian several times cheaper cannot make this step several times cheaper.
- **GMRES dominates.** At 90 900 dofs one GMRES solve takes about 2.5 s against 2 ms for
  the assembled Jacobian. The iteration count grows with the mesh (139, then 382 per step),
  as expected for ILU(0) on a 2D grid.
- **The assembled Jacobian is still the cheaper one.** Serial, it takes 0.21 ms against
  0.93 ms at 10 300 dofs (4.5×) and 1.83 ms against 8.90 ms at 90 900 dofs (4.9×). Threaded,
  the ratio is 1.6× at 10 300 dofs and 5.0× at 90 900. These agree with the spike's 3.8-6×.
- **Solve figures are noisy.** The 10 300-dof serial assembled solve has a 17% spread, and the
  sparse AD step at 10 300 dofs threaded has a 17% spread. The conclusion does not depend on them.
- **Not tested here.** The GMRES tolerance and restart were fixed at LinearSolve's default
  restart and `1e-8`. A looser tolerance, a restart length change, or a stronger
  preconditioner would move the Jacobian share. None was measured, so this page does not say
  how far it would move.

## What this means for the payoff

The assembled Jacobian is exact and cheaper to build, and that part holds. The payoff the
spike hoped for needs a workload where the Jacobian is a large share of the step. A
Newton-GMRES step with ILU(0) on this mesh is not one. Workloads where the Jacobian
dominates are ones that reuse a factorisation across several Jacobian evaluations, or that
need the Jacobian more often than the linear solve. This page did not time one.
