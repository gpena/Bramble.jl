# Precompile workload: the 3D elasticity form (#763, step 4)

Question: does the 3D elasticity form join the default precompile workload? The rule, fixed
before measuring (gate question 7): it joins only if it adds at most 3.0 s of precompile time
and at most 8 MiB of precompile cache. The 3D Laplacian workload paid 2.7 s and 5.3 MiB.
Otherwise nothing is added, and there is no opt-in switch.

## Method

The form is `2μ inner₊(εₕ(u), εₕ(v)) + λ inner₊(divₘₕ(u), divₘₕ(v))` on
`gridspace(Ωₕ)^Val(3)`, as in `benchmark/multiterm_ttfx.jl`. The variant adds `form` and
`assemble` of it in `src/precompile.jl` straight after the 3D Laplacian block, on the same
non-uniform 3D mesh. The baseline is `src/precompile.jl` at f6ee5255.

Both ran from one project copy in `/tmp` (so the cache slug is the same) against a throwaway
depot with the dependencies already precompiled. Before each run the Bramble cache directory
was deleted, then `Pkg.precompile("Bramble")` was timed with 4 threads (Julia 1.13.2). The
size is the `.ji` plus the `.dylib` pkgimage of Bramble alone. Three runs per variant,
interleaved (baseline, variant, baseline, ...).

All six runs were made under load (1-min load 3.1 to 4.3 on 8 cores, desktop app), on AC power,
with no other Julia process.

## Result

| | precompile seconds, median (min to max) | cache MiB, median (min to max) |
|:--- |:--- |:--- |
| baseline | 86.4 (85.7 to 86.4) | 191.75 (191.66 to 191.77) |
| with elasticity | 99.0 (98.3 to 99.5) | 210.90 (210.88 to 210.91) |
| delta | 12.6 | 19.2 |

In plain text: baseline precompile 86.4 s and 191.75 MiB; with elasticity precompile 99.0 s and
210.90 MiB. The elasticity form adds 12.6 s and 19.2 MiB, more than four times the 3.0 s limit
and more than twice the 8 MiB limit.

Decision: none
