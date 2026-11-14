# #1174: the 2D composite ghost `Δₕ` cell under Enzyme

Measurement of gpena/Bramble.jl#1174: Enzyme on the composite space `C^2` of a 2D ghost space, with `Δₕ` on both components and a coupling term, assembled under `CpuSerial` (cell `comp` of the v4-2-4 critic's script). On 9faf6345 the reverse was still compiling after 26 minutes at 2.9 GB, and it had never been run at the base. The reproducer is `1174_composite_ghost.jl`, next to this note: one cell and one mode per process, checked against ForwardDiff on the same loss, with a 20 minute limit the script enforces itself. Its command is in the last section. The base is 501128cb, the v4.2.3 merge, run from a detached scratch worktree. The tip is 70e39a58 on `v4-2-4-rest`, which holds S4.2's composite relation fix (b43f3c83, e24ded59) and S5.2's flat tail loop.

Enzyme: 0.13.219, 2026-11-12, macbook (Apple M2, 16 GB, macOS arm64, Julia 1.13.3, default `-O2`, `--startup-file=no`), 1 thread, the same version at the base and at the tip

Each run was one process under `julia_slot.sh 1` and `/usr/bin/time -l`. The columns are cell, commit, mode, status, relative error against ForwardDiff, the seconds of the first Enzyme `gradient` call (the compile included, loading and the reference excluded), and the peak memory footprint from `/usr/bin/time -l` in GB (10⁹ bytes). Other sessions ran Julia at the same time (load average about 3), so treat the seconds as rough. Both commits give the same ForwardDiff reference, `[93531.24063274513, 118171.56203520531, 72404.07333773715]`.

Cell: comp | 501128cb | rev | timeout (still compiling, killed at the 1200 s limit) | NaN | > 1190 | 7.70

Cell: comp | 501128cb | fwd | fails, `IllegalTypeAnalysisException` in `simplify_ast` (src/ast/simplifier.jl line 360) | NaN | 455 | 4.08

Cell: comp | tip | rev | fails, `IllegalTypeAnalysisException` in `simplify_ast` (src/ast/simplifier.jl line 360), at 70e39a58 | NaN | 982 | 5.70

Cell: comp | tip | fwd | fails, `IllegalTypeAnalysisException` in `simplify_ast` (src/ast/simplifier.jl line 360), at 70e39a58 | NaN | 592 | 4.08

Verdict: fix site: src/ast/simplifier.jl:360

The controls ran at the tip only:

Step: one (the scalar ghost space `C`, same mesh and form) | rev | ok 1.6e-16 | 100 | 2.13

Step: diag (`C^2`, `Δₕ` on both components, no coupling term) | rev | fails, `IllegalTypeAnalysisException` in `simplify_ast` (src/ast/simplifier.jl line 360) | 927 | 5.33

## What the runs say

- The cell is not a regression. At the base the reverse does not finish in 20 minutes and the forward fails. At the tip both modes fail with the same exception at the same frame. The reverse now gets far enough to fail, after 982 s.
- Neither mode meets the issue's first criterion (under 10 minutes and 10 GB, within 1e-8 of ForwardDiff). The tip's forward stops at 592 s, just under 10 minutes, and its reverse at 982 s. Peak memory stays under 10 GB in every run, 7.70 GB at most (the base's reverse).
- The scalar control on the same ghost space compiles and differentiates in 100 s. Without the coupling term (`diag`) the composite still fails at the same frame, after 927 s. What the composite adds is the component indices of `u(1)`, `u(2)`, `v(1)` and `v(2)`, not the coupling.
- The forward fails at the same frame at both commits. The base's reverse was still compiling when the limit killed it, and the tip's reached the frame after 982 s. The forward took longer at the tip (592 s against 455 s), so a gap of this size is within this machine's spread under load, and the runs do not say that any commit shortened the compile. The flat tail loop of 70e39a58 does not touch it: on a 2D grid the tail of a region is one range, and the failure happens in `form`, before any walk. The two tip runs used 70e39a58 itself, and no run separates it from its parent.

## Where the compile fails

Enzyme names `simplify_ast(::OperatorAdd{…})`, the generated method at src/ast/simplifier.jl line 360, which simplifies a sum term by term. Enzyme's message points at a `Union` its type analysis cannot handle with strict aliasing on.

Mode `infer` of the script, which makes no Enzyme call, prints what type inference gives `simplify_ast` on each cell's resolved sum when the method is inferred alone, as Enzyme compiles it, and what it gives the whole loss:

- `comp`: `simplify_ast` returns `OperatorAdd{2, LeftType, …} where LeftType<:OperatorAdd{…}`, an abstract type. The loss infers as `Float64`.
- `comp1` (`Δₕ` on component 1 only): a `Union` of two `OperatorAdd` trees. The loss infers as `Float64`.
- `diag` (both `Δₕ`, no coupling term): `OperatorAdd`, abstract. The loss infers as `Float64`.
- `one` (the scalar control): a concrete `OperatorAdd{2, …}`.

So on a composite space the simplified tree is not concrete unless inference knows the component indices. The distribution gates of the pass (`_mixes_components`, line 246, called from the scale and product methods) compare the two `component_idx::Int` fields of `IndexedTrialFunction` and `IndexedTestFunction` (src/ast/common.jl lines 225 and 236). The tree they return has one type if the indices are equal and another if they differ. The comment above `_mixes_components` documents this choice (gpena/Bramble.jl#529, #804). In the primal, `_bilinear_form` (src/assembly/bilinear.jl line 366, `@constprop :aggressive`) propagates the literal indices of `u(1)` into the pass, so `form` and the loss infer concretely, and the primal and ForwardDiff run. Enzyme compiles `simplify_ast(::OperatorAdd)` as a function of its own, which is not inlined (the generated distribution bodies are marked `_inlined`, line 410; the generated sum is not), so the indices are runtime values there and the return type is a `Union`.

This is a different failure from #1153's. There, `_relation_rows` held one copy of the relation forms per component. Here the failure comes before any relation or assembly code: `form(V, V, …)` is inside the differentiated function and its AST pass does not infer without constants. Two samples of the tip's reverse, at about 8 and 12 minutes, before it failed (`sample <pid>`) showed Enzyme's type analysis merging type trees (`TypeAnalyzer::visitAddrSpaceCastInst`, `visitGEPOperator`, `GenericTypeTree::orIn`) inside `CreateAugmentedPrimal`, and `set_module_types!` building type trees of large Julia types, consistent with a tree whose type inference left open.

## For the fix

- The site is src/ast/simplifier.jl line 360, `simplify_ast(::OperatorAdd)`, with the value gates under it (`_mixes_components`, line 246). Either the gates answer by type (component indices in the type of `IndexedTrialFunction` and `IndexedTestFunction`, as `_ComponentSum` already does for the builders), or the pass runs outside what Enzyme compiles (an inactive rule on the AST pass, whose output holds the parameter's scalars and so is not inactive as a whole).
- Check with `comp rev` and `comp fwd` of the script. The `infer` mode should then print a concrete type for `simplify_ast` on `comp`, `comp1` and `diag`.
- The site names where the compile fails, not where its time goes: before the exception, 982 s (reverse) and 592 s (forward) went into Enzyme's type analysis of functions the samples did not name. A concrete `simplify_ast` removes the exception, but nothing here shows the cell then compiles in under 10 minutes and 10 GB, so the new issue keeps that criterion open.
- This closes #1174 on its second criterion ("a reduction names its fix site"). The fix is a new issue.

## Commands run

Each run is one process, in order, through `.claude/scripts/julia_slot.sh 1` with `/usr/bin/time -l`, by a driver in `/tmp/s53` (outside the repository) that writes one log per run. From the repository root:

```text
S1174_LIMIT=1200 .claude/scripts/julia_slot.sh 1 /usr/bin/time -l \
    julia --startup-file=no --project=<checkout> --threads=1 \
    docs/notes/internals/ad_reductions/1174_composite_ghost.jl <cell> <rev|fwd|ref|infer>
```

`<checkout>` is `.` at the tip, and for the base a worktree made by `git worktree add --detach ~/bramble-scratch/s53-501128cb 501128cb` with the tip's `Manifest.toml` copied in (the two `Project.toml` files are identical), removed after the runs. `<cell>` is `comp`, `comp1`, `diag` or `one`. The script prints one `RESULT` line and exits 0 for `ok`. At the limit its watchdog prints `WATCHDOG timeout` to stderr and kills the process, with no `RESULT` line. A first run with `S1174_LIMIT=60` checked that the watchdog fires during an Enzyme compile (60.3 s). The runs above used a watchdog of one `sleep`, stopped at exit, which left that `sleep` behind. The script's watchdog now checks the pid every second and exits with it, checked by a timeout at 20.4 s with `S1174_LIMIT=20` and by a `one ref` run that left no process behind.
