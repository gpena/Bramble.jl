# #1153: Enzyme through `constrained` with parameter-dependent relation data

Reduction of gpena/Bramble.jl#1153: `constrained(W, relations)` alone, no form of the user and no assembly, with a datum or a coefficient of a relation set to the parameter `p[1]`. These are `data_rev` (the Neumann datum of :xmax) and `robin_rev` (the Robin coefficient of :xmax) of `1127_ghost.jl`, which were killed at 47 and 30 minutes. The reproducer is `1153_constrained.jl`, next to this note. It runs one step of the construction, one case and one mode per process, against a central difference of the same loss. Its commands are in the last section.

The construction is `constrained` (src/assembly/ghost_relations.jl, line 110) into `_constrained_space` (line 833), which calls `_relation_rows` (line 691), `_dirichlet_values` (line 990), `_kept_points` (line 1075) and `_relation_closure` (line 1113, with `_block_solve`, line 1185), then `_closure_matrix` and `_closure_data` (src/space/ghost_gridspace.jl, lines 346 and 363). Each step ran as a function of the parameter. The inputs of a step that the parameter does not touch were computed outside; the ones it touches were rebuilt inside as `a0 + p[1] * a1` (exact, the data and the coefficient enter linearly), and the control `inputs` shows that rebuild itself differentiates. The steps that build the relations from the parameter (`_facet_relations`, `_assemble_relation`, `_relation_rows`, `_constrained_space`) parse them inside.

Step: _facet_relations: fwd: ok 2.9e-12, 6 s (data, robin, dirichlet)

Step: _facet_relations: rev: ok 2.9e-12, 7 s (data)

Step: _assemble_relation, one facet (:xmax, line 791): fwd: ok 3.6e-11, 38 s (data and robin)

Step: _assemble_relation, one facet (:xmax, line 791): rev: ok 3.6e-11, 732 s and 3.7 GB (data); ok 3.6e-11, 744 s and 3.2 GB (robin)

Step: _assemble_relation, the load form alone (line 799, data): fwd: ok 1.9e-12, 13 s

Step: _assemble_relation, the load form alone (line 799, data): rev: ok 1.9e-12, 14 s

Step: _assemble_relation, the matrix form alone (line 798, robin): fwd: ok 3.6e-11, 27 s

Step: _assemble_relation, the matrix form alone (line 798, robin): rev: ok 3.6e-11, 177 s and 3.2 GB

Step: _assemble_relation, `sparse(transpose(R))` alone (line 800): rev: ok 1.5e-11, 8 s (robin)

Step: _relation_rows, the three ghost relations (line 691): fwd: ok 3.6e-11, 54 s (data); ok 1.5e-10, 54 s (robin)

Step: _relation_rows, the three ghost relations (line 691): rev: > 15 min (data and robin; 8.7 GB footprint when stopped)

Step: _dirichlet_values: fwd: ok 4.5e-12, 7 s (dirichlet)

Step: _dirichlet_values: rev: ok 4.5e-12, 9 s (dirichlet)

Step: _kept_points: fwd: ok 7.4e-12, 5 s (data; integers only, the parameter scales the sum)

Step: _kept_points: rev: ok 7.4e-12, 5 s (data)

Step: union-find and `Dict` of `_block_solve` (lines 1187 to 1201, a copy): fwd: ok 5.1e-12, 6 s (data)

Step: union-find and `Dict` of `_block_solve` (lines 1187 to 1201, a copy): rev: ok 5.1e-12, 6 s (data)

Step: dense kernel of `_block_solve` (`lu`, `F \ Bb`, `F \ b`, lines 1218 to 1231, on a 2 by 2 block with active entries): fwd: fails EnzymeNoDerivativeError "No forward mode derivative found for ejlstr$dgetrf_64_", 6 s

Step: dense kernel of `_block_solve` (`lu`, `F \ Bb`, `F \ b`, lines 1218 to 1231, on a 2 by 2 block with active entries): rev: fails EnzymeNoDerivativeError "No augmented forward pass found for ejlstr$dgetrf_64_", 6 s

Step: _block_solve (line 1185): fwd: ok 4.9e-10, 9 s (data); ok 4.0e-10, 9 s (robin); fails EnzymeNoDerivativeError dgetrf, 9 s (conormal)

Step: _block_solve (line 1185): rev: ok 4.9e-10, 14 s (data); fails EnzymeNoDerivativeError "No augmented forward pass found" dgetrf, 13 s (conormal)

Step: _relation_closure (line 1113): fwd: ok 6.4e-11, 11 s (data); ok 1.1e-10, 10 s (robin); ok 9.1e-12, 10 s (dirichlet); fails dgetrf, 10 s (conormal)

Step: _relation_closure (line 1113): rev: ok 6.4e-11, 18 s (data); ok 1.1e-10, 18 s (robin); ok 9.1e-12, 17 s (dirichlet); fails dgetrf, 17 s (conormal)

Step: _closure_matrix and _closure_data: fwd: ok 3.4e-11, 6 s (data); ok 2.4e-11, 6 s (robin); ok 3.0e-11, 6 s (dirichlet)

Step: _closure_matrix and _closure_data: rev: ok 3.4e-11, 6 s (data)

Step: _constrained_space (line 833): fwd: ok 2.8e-10, 61 s (data); ok 6.9e-10, 61 s (robin); ok 7.0e-12, 61 s (dirichlet)

Step: _constrained_space (line 833): rev: stopped at 310 s by the 10 GB footprint cap of `julia_slot.sh`, 12.4 GB (data)

Step: constrained (line 110): fwd: ok 2.8e-10, 62 s and 2.0 GB (data); ok 6.9e-10, 61 s and 1.9 GB (robin); ok 7.0e-12, 60 s and 1.9 GB (dirichlet); fails dgetrf, 54 s (conormal)

Step: constrained (line 110): rev: stopped at 311 s by the 10 GB footprint cap, 11.6 GB (data); stopped at 310 s by the cap, 12.7 GB (robin)

Step: constrained with constant data, then a form and its assembly (case const, downstream): fwd: ok 2.5e-11, 63 s

Step: constrained with constant data, then a form and its assembly (case const, downstream): rev: ok 2.5e-11, 95 s

Fix site: src/assembly/ghost_relations.jl, `_assemble_relation` (line 791), called by `_relation_rows` at line 702 (`blocks = map(r -> _assemble_relation(P, r), rels)`), variants `data` and `robin` in reverse. Nothing in the construction fails for them in forward mode, and in reverse every step after the rows (`_relation_closure`, `_closure_matrix`, `_closure_data`) and the ones before it (`_facet_relations`, `_dirichlet_values`, `_kept_points`, the union-find and `Dict`) differentiate in under 20 s. The first step that does not finish is `_relation_rows`: one facet of it (`_assemble_relation`) takes 732 s and 3.7 GB, the rows call it for three facets (:xmin, :xmax, :ymax; the facet is in the type, so presumably one compile each, not probed) and do not finish in 15 minutes, and `_constrained_space` passes 10 GB at about five minutes. It is none of the three kinds the plan lists, and not an error: it is the reverse-mode compile cost of the form assembly inside `_assemble_relation`, the matrix form at line 798 (177 s alone) and the rest of the function (the other 540 s; the load form at line 799 is 14 s alone, and the transposition at line 800 is 8 s). The relation's matrix and load are linear in the datum and in the Robin coefficient, so the fix is to keep the form assembly out of the differentiated code (assemble the unit-datum pieces outside and scale them by the parameter), or to give the assembly a rule. The 1127 note's `plain_rev` and `mass_rev` (361 to 419 s per assembled form) are the same kind of cost; it is not a missing derivative. This note did not probe why the function costs more than its parts. The page's earlier `Illegal calling convention fixup` did not appear at any step in this tree.

Fix site: src/assembly/ghost_relations.jl, `_block_solve`, lines 1217 to 1231 (`lu(Cb; check = false)` at line 1218, `F \ Bb` at line 1226, `F \ b[idx]` at line 1231), kind (a), reached by relations that couple the ghosts of a node: variant `conormal` (`∂ₙ(u; K)` with K = [2 1; 1 2] on the four facets, the test of test/form/ghost_rank_one.jl) fails in forward with `No forward mode derivative found for ejlstr$dgetrf_64_` and in reverse with `No augmented forward pass found for ejlstr$dgetrf_64_`, at `_block_solve`, `_relation_closure` and `constrained` alike. The `data`, `robin` and `dirichlet` relations never take that branch (every block holds one ghost, `length(idx) == 1`, line 1207), which is why they differentiate through `_block_solve` with the call still in its compiled code: the error appears only in the runs whose blocks hold two ghosts. The plan's fix for (a) applies as written: a forward and a reverse rule for the block solve in `ext/BrambleEnzymeExt` (`X = C⁻¹ B`, `dX = C⁻¹ (dB − dC X)`, the adjoint by a transposed solve), or a generic LU for these small blocks. It does not fix the `data` and `robin` cells, and it is the page's earlier forward `dgetrf` measurement, whose relations were not these.

Constant data: `constrained` of the same relations with constant data (`2.0 * u + ∂ₙ(u) => 1.0`, no parameter) still differentiates the downstream assembly, measured here (S3.1 and S3.2 show the same for the `ghost_assemble` cell). It is `constrained` built outside and `form(C, C, (U, V) -> p[1] * innerₕ(-Δₕ(U), V) + p[1]^2 * innerₕ(U, V))` with the load form `assemble(a, l)` of 1127's `loss_assemble`: forward 63 s and reverse 95 s, both against the central difference at 2.5e-11. So the parameter in the relation data, and not `constrained` or the ghost assembly, is what costs: the same `_relation_rows` that runs once outside with constant data (its work is not differentiated) is what the data variants put under Enzyme.

Compile: time runs from the reference to the end of the Enzyme call, compile included (Enzyme compiles inside the first call of `measure`, before its `enzyme start` step). Memory is the peak memory footprint of the process (`/usr/bin/time -l`). A run `julia_slot.sh` stopped at 10 GB is marked so; a run stopped by the driver is at the 15 minute bound. Figures: reverse `_assemble_relation` 732 s and 3.7 GB (data), 744 s and 3.2 GB (robin); reverse `_relation_rows` over 15 minutes and 8.7 GB; reverse `constrained` and `_constrained_space` stopped at five minutes over 10 GB (11.6 to 12.7 GB); forward `constrained` 60 to 62 s and 1.9 to 2.0 GB, the whole construction. No forward step takes over a minute.

Enzyme: 0.13.219 (Julia 1.13.3, macOS arm64, 1 thread, `--optimize=1`, `--check-bounds=auto`, `--startup-file=no`), loaded from a temporary environment holding Enzyme alone, `Enzyme.set_runtime_activity` in both modes. The 1127 note measured 0.13.218.

## What the steps say

- Forward mode differentiates all of `constrained` for the three parameter cases, in about a minute, so the 1127 note's `dgetrf` error in forward is not these relations'. It belongs to relations that couple ghosts (`conormal`).
- Reverse mode fails for `data` and `robin` for cost, at the assembly of the relation matrix and load inside `_assemble_relation`, multiplied by the three facets. Index bookkeeping (the union-find, the `Dict`, the `push!`ed triplets, the dense solve of one-ghost blocks), `_dirichlet_values`, `_kept_points`, `_closure_matrix` and `_closure_data` all differentiate in seconds, so kind (b) does not apply to the cells' own relations, and no Bramble-free Enzyme defect was found, so kind (c) does not either.
- The dense `lu` has no Enzyme derivative in either mode, and a relation set that couples the ghosts of a node reaches it. No cell of the plan does yet.
- The pin for the first finding is S6's `ghost_data` and `ghost_robin` cells; the second needs a cell of its own (`conormal`).

## Commands run

Each run is one process through `.claude/scripts/julia_slot.sh 1` with `/usr/bin/time -l`, by a driver in `/tmp/bramble-1153` (outside the repository) that stops it at 15 minutes by its PID. One Enzyme process at a time. The commands run from the repository root:

```text
.claude/scripts/julia_slot.sh 1 julia --startup-file=no --project=. --threads=1 --optimize=1 \
    --check-bounds=auto docs/notes/internals/ad_reductions/1153_constrained.jl <step> <case> <fwd|rev>
```

`<step>` is inputs, parse, relation, relation_load, relation_matrix, relation_transpose, rows, dirichlet, kept, bookkeeping, lu, block_solve, closure, closure_data, space, constrained or downstream; `<case>` is data, robin, dirichlet, conormal or const (downstream takes const alone). The script exits 0 for `ok` and 1 otherwise and prints one `RESULT` line.

Reproduces: .claude/scripts/julia_slot.sh 1 julia --startup-file=no --project=. --threads=1 --optimize=1 --check-bounds=auto docs/notes/internals/ad_reductions/1153_constrained.jl block_solve conormal fwd   # exit 1, EnzymeNoDerivativeError "No forward mode derivative found for ejlstr$dgetrf_64_", about 15 s

Reproduces: .claude/scripts/julia_slot.sh 1 julia --startup-file=no --project=. --threads=1 --optimize=1 --check-bounds=auto docs/notes/internals/ad_reductions/1153_constrained.jl relation data rev   # exit 0 after about 12 minutes (732 s, 3.7 GB): one facet of the rows; `rows data rev` in place of `relation` does not finish in 15 minutes, `constrained data rev` passes 10 GB at about five minutes and `julia_slot.sh` stops it (exit 143)

Reproduces: .claude/scripts/julia_slot.sh 1 julia --startup-file=no --project=. --threads=1 --optimize=1 --check-bounds=auto docs/notes/internals/ad_reductions/1153_constrained.jl constrained data fwd   # exit 0, about 62 s: the control, forward passes
