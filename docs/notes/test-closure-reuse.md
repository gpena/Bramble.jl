# Test closure reuse (#888)

Every anonymous coefficient closure in a test (`x -> 1.0`) is a new type, and form
construction specializes on it, so an identical closure in a second file is inferred again.
Issue #888 replaces the byte-identical closures in 23 unit-test files with the shared
functions in `test/TestUtils.jl`. This page records how many Bramble method instances (MIs)
each group of those files makes the compiler infer before the replacement. S4 adds the
after rows. The tool is `test/compile_reuse.jl`; the figures are in
`test-closure-reuse.tsv` next to this page, which `CompileReuse.baseline(group)` reads.

Every figure below is measured cold, one Julia process per group (`julia --startup-file=no
--project=test test/compile_reuse.jl S3.1`, default optimisation, `BRAMBLE_TEST_GROUP`
unset so the test group is `all`). Inside a process the files run in the order of the group
(a file that imports another test module is preceded by it, see below), so the per-file
`mis` of a later file leaves out MIs an earlier file already inferred; the group row is the
sum over its files and is the figure the rewrite is judged on.

- Commit at baseline: `4fdfdb3f` (branch `perf/test-compile-reuse`, which holds the shared
  functions of S2 and none of the 23 rewritten files).
- Julia 1.13.2, aarch64 macOS, one thread per run.
- Machine load (`uptime` 1-, 5- and 15-minute averages): 1.85, 2.07, 2.26 after the last
  group of the recorded run. Other sessions shared the machine.
- Drift: `mis` is a count, not a time. The S3.4 group, run cold three times (the recorded
  run and two more that did not write the TSV), gave `mis` 6499, 6499 and 6499 (and the
  same `mis_own`, `mis_coef`, `mis_coef_shared`, passes and broken each time). Cold against
  cold there was no drift. A `measure` call inside a process that already loaded `Bramble`
  and `TestUtils` gave 6490 on the same untouched files, so a probe's `mis` sits a few
  counts below the baseline's even with no file changed (9 of 6499 here). `t_infer` does
  drift with load, by a few percent between runs, and is never gated.

## What the columns count

- `mis`: inference frames for a method instance of a method of module `Bramble`. It depends
  on what ran earlier in the process.
- `mis_own`: the subset whose argument types mention, at any depth of type parameters, a
  type defined in a module the measured files created. It does not depend on earlier work.
- `mis_coef`: the subset of `mis_own` keyed on a one-argument closure whose argument is
  named `x` or `p`: a coefficient, not a form lambda. One-argument operator lambdas named
  `x` count here too, before and after alike.
- `mis_coef_shared`: the subset of `mis_coef` whose closure sits on a source line holding a
  body byte-identical to one of S2's shared bodies, argument renamed to `x` (a line with
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
| S3.1 | 4 | 85765 | 30439 | 27016 | 10743 | 0.398 | 2095 | 5 | 132.1 |
| S3.2 | 6 | 137259 | 19257 | 15891 | 6852 | 0.431 | 3638 | 1 | 295.7 |
| S3.3 | 6 | 11862 | 7110 | 6815 | 2237 | 0.328 | 2188 | 0 | 27.2 |
| S3.4 | 4 | 6499 | 2791 | 2452 | 757 | 0.309 | 1209 | 0 | 11.6 |
| S3.5 | 3 | 56610 | 11325 | 10829 | 6055 | 0.559 | 935 | 0 | 162.5 |

No group is under 0.15. S3.1 to S3.4 are under 0.5, so under the plan's rule their probe
factor becomes `1 - 0.8 * shared`; S3.5 keeps 0.6.

## Files

| file | group | mis | mis_own | mis_coef | mis_coef_shared | shared | passes | broken | t_infer (s) |
|---|---|---|---|---|---|---|---|---|---|
| file test/form/linear.jl | S3.1 | 8724 | 2795 | 2620 | 1957 | 0.747 | 377 | 5 | 16.9 |
| file test/form/source_operators.jl | S3.1 | 15522 | 14081 | 11730 | 475 | 0.04 | 451 | 0 | 12.6 |
| file test/form/rank_one.jl | S3.1 | 8081 | 3654 | 3502 | 2489 | 0.711 | 533 | 0 | 10.9 |
| file test/form/linear_restriction.jl | S3.1 | 53438 | 9909 | 9164 | 5822 | 0.635 | 734 | 0 | 91.7 |
| file test/form/semidiscrete.jl | S3.2 | 3929 | 1876 | 837 | 749 | 0.895 | 169 | 0 | 6.0 |
| file test/form/normal.jl | S3.2 | 19806 | 7631 | 7369 | 1632 | 0.221 | 383 | 0 | 21.9 |
| file test/form/bilinear.jl | S3.2 | 85304 | 2422 | 1019 | 498 | 0.489 | 1622 | 1 | 200.1 |
| file test/form/dirichlet_constraints.jl | S3.2 | 1468 | 882 | 549 | 445 | 0.811 | 994 | 0 | 3.8 |
| file test/form/inner_products.jl | S3.2 | 5441 | 1363 | 1284 | 270 | 0.21 | 207 | 0 | 7.4 |
| file test/form/ghost_rank_one.jl | S3.2 | 21311 | 5083 | 4833 | 3258 | 0.674 | 263 | 0 | 56.4 |
| file test/space/vector_elements.jl | S3.3 | 3720 | 2833 | 2819 | 902 | 0.32 | 694 | 0 | 7.8 |
| file test/space/inner_product.jl | S3.3 | 4391 | 2500 | 2455 | 927 | 0.378 | 654 | 0 | 11.8 |
| file test/space/operators.jl | S3.3 | 2303 | 724 | 492 | 159 | 0.323 | 599 | 0 | 5.8 |
| file test/space/star_difference.jl | S3.3 | 421 | 409 | 409 | 89 | 0.218 | 65 | 0 | 0.5 |
| file test/space/interpolation_bounds.jl | S3.3 | 771 | 388 | 384 | 160 | 0.417 | 158 | 0 | 1.0 |
| file test/space/sbp_identities.jl | S3.3 | 256 | 256 | 256 | 0 | 0.0 | 18 | 0 | 0.3 |
| file test/mesh/meshes.jl | S3.4 | 806 | 680 | 678 | 0 | 0.0 | 383 | 0 | 1.3 |
| file test/mesh/markers.jl | S3.4 | 1578 | 1129 | 845 | 377 | 0.446 | 252 | 0 | 2.2 |
| file test/mesh/meshnd.jl | S3.4 | 3763 | 673 | 620 | 350 | 0.565 | 537 | 0 | 6.7 |
| file test/mesh/levelset_class.jl | S3.4 | 352 | 309 | 309 | 30 | 0.097 | 37 | 0 | 1.3 |
| file test/ghost/cross_component.jl | S3.5 | 42423 | 6844 | 6690 | 3553 | 0.531 | 542 | 0 | 117.1 |
| file test/ghost/polygon_spaces.jl | S3.5 | 3720 | 1274 | 1226 | 959 | 0.782 | 195 | 0 | 11.5 |
| file test/ghost/composite.jl | S3.5 | 10467 | 3207 | 2913 | 1543 | 0.53 | 198 | 0 | 33.9 |
