# Test closure reuse (#888)

Every anonymous coefficient closure in a test (`x -> 1.0`) is a new type, and form
construction specializes on it, so an identical closure in a second file is inferred again.
Issue #888 replaces the byte-identical closures in 23 unit-test files with the shared
functions in `test/TestUtils.jl` (S8.2). This page records how many Bramble method instances
(MIs) each group of those files makes the compiler infer before the replacement (S8.3.1 to
S8.3.5). S8.4 adds the after rows. The tool is `test/compile_reuse.jl`; the figures are in
`test-closure-reuse.tsv` next to this page, which `CompileReuse.baseline(group)` reads.

Every figure below is measured cold, one Julia process per group (`julia --startup-file=no
--project=test test/compile_reuse.jl S8.3.1`, default optimisation, `BRAMBLE_TEST_GROUP`
unset so the test group is `all`). Inside a process the files run in the order of the group
(a file that imports another test module is preceded by it, see below), so the per-file
`mis` of a later file leaves out MIs an earlier file already inferred; the group row is the
sum over its files and is the figure the rewrite is judged on.

- Commit at baseline: `6c7f6505` (branch `perf/v4.1.4-followups`), which holds S6.1's
  erasure of source closures in `form` (#917's verdict) and S8.2's shared functions, and
  none of the 23 rewritten files. Closures that S6.1 erases no longer key form-time MIs, so
  `mis_coef` is lower here than in the first baseline of this tool (commit `4fdfdb3f`,
  before S6.1): see the comparison below.
- Julia 1.13.2, aarch64 macOS, one thread per run.
- Machine load (`uptime` 1-, 5- and 15-minute averages): 2.12, 2.52, 2.54 after the last
  group of the recorded run; about 3.0 at the start of it. Other sessions shared the
  machine, one slot each.
- Drift: `mis` is a count, not a time. The first baseline of this tool ran S3.4 cold three
  times (S8.3.4 here) and got `mis` 6499 each time, with the same other columns. This
  baseline ran each group once, so it has no repeat of its own; S8.3.4 gave 6530 (the old
  figure at the old head `4fdfdb3f`, 6499; 31 more MIs after the commits in between).
  A `measure` call inside a process that already loaded `Bramble` and `TestUtils` gave
  6490 on the same untouched files at `4fdfdb3f`, so a probe's `mis` sits a few counts
  below the baseline's even with no file changed. `t_infer` does drift with load, by a few
  percent between runs, and is never gated.

## What the columns count

- `mis`: inference frames for a method instance of a method of module `Bramble`. It depends
  on what ran earlier in the process.
- `mis_own`: the subset whose argument types mention, at any depth of type parameters, a
  type defined in a module the measured files created. It does not depend on earlier work.
- `mis_coef`: the subset of `mis_own` keyed on a one-argument closure whose argument is
  named `x` or `p`: a coefficient, not a form lambda. One-argument operator lambdas named
  `x` count here too, before and after alike.
- `mis_coef_shared`: the subset of `mis_coef` whose closure sits on a source line holding a
  body byte-identical to one of S8.2's shared bodies, argument renamed to `x` (a line with
  several closures counts all of them if one matches, so the figure is an upper bound).
- `shared`: `mis_coef_shared / mis_coef`, the figure the plan's abandonment rule reads.
- `passes`, `broken`: test counts, which a rewrite must leave unchanged.
- `t_infer`: exclusive inference seconds of the Bramble frames. Reported, never gated, as
  it is noisy.

Two files import a test module that another file defines, so the tool includes that file
first: `test/space/vector_elements.jl` before `test/space/inner_product.jl` (both are in the
group, so both are counted) and `test/space/difference.jl` before
`test/space/star_difference.jl` (outside the group: it runs, its figures are not counted,
and the MIs it inferred are not counted again for `star_difference.jl`).

## Groups

| group | files | mis | mis_own | mis_coef | mis_coef_shared | shared | passes | broken | t_infer (s) |
|---|---|---|---|---|---|---|---|---|---|
| S8.3.1 | 4 | 87989 | 26877 | 23390 | 6045 | 0.258 | 2148 | 5 | 138.3 |
| S8.3.2 | 6 | 135911 | 14667 | 11324 | 5518 | 0.487 | 3638 | 1 | 301.2 |
| S8.3.3 | 6 | 11861 | 7110 | 6815 | 2237 | 0.328 | 2188 | 0 | 27.1 |
| S8.3.4 | 4 | 6530 | 2811 | 2472 | 777 | 0.314 | 1209 | 0 | 12.4 |
| S8.3.5 | 3 | 55674 | 9278 | 8790 | 4183 | 0.476 | 935 | 0 | 161.4 |

## Against the first baseline of the tool

The first run of this tool (commit `4fdfdb3f`, before S6.1) gave these `mis_coef` and
`mis_coef_shared`; the `shared` column is their ratio, and the figures above are this
baseline's. `mis_coef` fell most in the groups with the most `restrict_to` and form-time
work (S8.3.1's `linear_restriction.jl`: 9164 to 4896, shared 5822 to 1505).

| group | mis_coef then | mis_coef now | shared then | shared now |
|---|---|---|---|---|
| group S8.3.1 | 27016 | 23390 | 0.398 | 0.258 |
| group S8.3.2 | 15891 | 11324 | 0.431 | 0.487 |
| group S8.3.3 | 6815 | 6815 | 0.328 | 0.328 |
| group S8.3.4 | 2452 | 2472 | 0.309 | 0.314 |
| group S8.3.5 | 10829 | 8790 | 0.559 | 0.476 |

`test/form/source_operators.jl` holds 13038 of S8.3.1's `mis_coef` and only 478 shared
(0.037): its closures are mostly not byte-identical to a shared body.

## Files

| file | group | mis | mis_own | mis_coef | mis_coef_shared | shared | passes | broken | t_infer (s) |
|---|---|---|---|---|---|---|---|---|---|
| file test/form/linear.jl | S8.3.1 | 8869 | 2807 | 2632 | 1926 | 0.732 | 381 | 5 | 17.6 |
| file test/form/linear_restriction.jl | S8.3.1 | 54297 | 5641 | 4896 | 1505 | 0.307 | 734 | 0 | 95.2 |
| file test/form/rank_one.jl | S8.3.1 | 7578 | 2976 | 2824 | 2136 | 0.756 | 533 | 0 | 10.8 |
| file test/form/source_operators.jl | S8.3.1 | 17245 | 15453 | 13038 | 478 | 0.037 | 500 | 0 | 14.6 |
| file test/form/bilinear.jl | S8.3.2 | 85461 | 2327 | 947 | 498 | 0.526 | 1622 | 1 | 204.9 |
| file test/form/dirichlet_constraints.jl | S8.3.2 | 1469 | 882 | 549 | 445 | 0.811 | 994 | 0 | 3.6 |
| file test/form/ghost_rank_one.jl | S8.3.2 | 20062 | 3126 | 2876 | 2148 | 0.747 | 263 | 0 | 56.8 |
| file test/form/inner_products.jl | S8.3.2 | 5390 | 1011 | 932 | 284 | 0.305 | 207 | 0 | 7.2 |
| file test/form/normal.jl | S8.3.2 | 19559 | 5422 | 5160 | 1374 | 0.266 | 383 | 0 | 22.4 |
| file test/form/semidiscrete.jl | S8.3.2 | 3970 | 1899 | 860 | 769 | 0.894 | 169 | 0 | 6.3 |
| file test/space/inner_product.jl | S8.3.3 | 4388 | 2500 | 2455 | 927 | 0.378 | 654 | 0 | 11.6 |
| file test/space/interpolation_bounds.jl | S8.3.3 | 777 | 388 | 384 | 160 | 0.417 | 158 | 0 | 0.9 |
| file test/space/operators.jl | S8.3.3 | 2299 | 724 | 492 | 159 | 0.323 | 599 | 0 | 6.0 |
| file test/space/sbp_identities.jl | S8.3.3 | 256 | 256 | 256 | 0 | 0.0 | 18 | 0 | 0.3 |
| file test/space/star_difference.jl | S8.3.3 | 421 | 409 | 409 | 89 | 0.218 | 65 | 0 | 0.5 |
| file test/space/vector_elements.jl | S8.3.3 | 3720 | 2833 | 2819 | 902 | 0.32 | 694 | 0 | 7.7 |
| file test/mesh/levelset_class.jl | S8.3.4 | 352 | 309 | 309 | 30 | 0.097 | 37 | 0 | 1.5 |
| file test/mesh/markers.jl | S8.3.4 | 1590 | 1141 | 857 | 389 | 0.454 | 252 | 0 | 2.4 |
| file test/mesh/meshes.jl | S8.3.4 | 806 | 680 | 678 | 0 | 0.0 | 383 | 0 | 1.4 |
| file test/mesh/meshnd.jl | S8.3.4 | 3782 | 681 | 628 | 358 | 0.57 | 537 | 0 | 7.1 |
| file test/ghost/composite.jl | S8.3.5 | 10006 | 2482 | 2196 | 1006 | 0.458 | 198 | 0 | 33.7 |
| file test/ghost/cross_component.jl | S8.3.5 | 41911 | 5514 | 5360 | 2160 | 0.403 | 542 | 0 | 116.5 |
| file test/ghost/polygon_spaces.jl | S8.3.5 | 3757 | 1282 | 1234 | 1017 | 0.824 | 195 | 0 | 11.3 |
