# Enzyme under `Threads.@threads`: version bisect (#369, S2.1)

Plan v4-2-0-followup, subplan S2.1. Planning found that Enzyme cannot differentiate a
plain `Threads.@threads` loop on Julia 1.13.3 with Enzyme 0.13.216 (`AssertionError:
mi2 !== nothing`), while #369 recorded `CpuThreaded` as `ok` on Julia 1.13.0. This note
asks whether a version of Enzyme or of Julia explains the change. Every status below is
read from `benchmark/results/enzyme_threads_bisect.toml`, written by
`benchmark/enzyme_threads_bisect.jl`. The TOML's `script_sha` ties the table to that script.

## Reproducer

The reproducer has no Bramble in it. It runs under three schedules: `serial` (the control),
`dynamic` (`Threads.@threads`, the planning failure) and `static` (`Threads.@threads
:static`, the schedule of Bramble's `_static_*` helpers).

```julia
function sq_dynamic!(y, x)
    Threads.@threads for i in eachindex(y)
        y[i] = x[i]^2
    end
    return nothing
end

x = collect(1.0:100.0); dx = zeros(100); y = zeros(100); dy = ones(100)
Enzyme.autodiff(Reverse, sq_dynamic!, Const, Duplicated(y, dy), Duplicated(x, dx))
# dx == 2x expected; on Julia 1.13.3 with Enzyme 0.13.216 it throws instead
```

The driver runs each Enzyme version in a fresh environment (`Pkg.add(; version)`) and each
schedule in its own child process with 2 threads, so a crash or a load failure is recorded
and not inherited. A child that exits cleanly with a result is `ok`, `wrong` or `throws`.
`unresolvable` means Pkg could not install the version, or Enzyme did not load.

## Version table

Julia 1.13.3 (the only 1.13 installed; 1.13.0 is not, see below). Resolved for each row:
Enzyme_jll, GPUCompiler, LLVM 9.13.2, EnzymeCore 0.8.26 (the TOML has all five).

| Enzyme | Julia | serial | dynamic | static |
|:--|:--|:--|:--|:--|
| 0.13.0 | 1.13.3 | unresolvable (GPUCompiler compat excludes 1.13.3) | unresolvable | unresolvable |
| 0.13.100 | 1.13.3 | throws, `Taking the type of an opaque pointer is illegal` | throws (same) | throws (same) |
| 0.13.150 | 1.13.3 | ok | throws, `VERSION < v"1.13"` | throws, `VERSION < v"1.13"` |
| 0.13.180 | 1.13.3 | ok | throws, `VERSION < v"1.13"` | throws, `VERSION < v"1.13"` |
| 0.13.190 | 1.13.3 | ok | throws, `VERSION < v"1.13"` | throws, `VERSION < v"1.13"` |
| 0.13.198 | 1.13.3 | ok | throws, `VERSION < v"1.13"` | throws, `VERSION < v"1.13"` |
| 0.13.204 (#369) | 1.13.3 | ok | throws, `AssertionError: mi2 !== nothing` | throws, same |
| 0.13.216 (root Manifest) | 1.13.3 | ok | throws, `AssertionError: mi2 !== nothing` | throws, same |
| 0.13.204 (#369) | 1.12.7 | ok | ok, `max abs(dx - 2x) = 0` | ok |

Not run: 0.13.205 to 0.13.215 (both endpoints failed, so the bisect did not reach them), and
Julia 1.13.0 (`juliaup status` lists 1.12.7, 1.13.3, 1.14.0-alpha2 and nightly only; nothing
was installed for this note).

The `VERSION < v"1.13"` throws come from Enzyme itself: releases before 0.13.204 refuse
threaded reverse differentiation on Julia 1.13 by assertion. Their `serial` rows pass, so the
threaded failure there is a guard, not a miscompilation. The 0.13.100 row fails even on
`serial`, so its threaded rows say nothing and it is not a working version.

## Bisect path

The driver probed the endpoints first, 0.13.204 (#369's version) and 0.13.216 (current). Both
fail on `dynamic`, so no boundary lies between them. It then sampled older versions, newest
first (198, 190, 180, 150, 100, 0), and none passed the `dynamic` schedule. Only versions
where `serial` also passes count as working, and none does on 1.13.3.

The #369 version is 0.13.204. The session transcripts of 2026-09-14 to 09-25 (Julia 1.13.0)
name 0.13.204 and no other 0.13 release. Enzyme 0.13.205 first appears on 2026-09-26, the
day #369 was filed. The worktree's Manifest is untracked, so the transcripts are the only
record of that version.

## Verdict

Route for S2.2: **rules**. No Enzyme version tested works on Julia 1.13.3 under a threaded
schedule, so no compat change (`Enzyme = "0.13"` in `Project.toml`, Enzyme 0.13.216 in the
root Manifest) avoids the failure. The version bisect cannot isolate an Enzyme release.

The version that #369 ran, 0.13.204, passes on Julia 1.12.7 and fails on 1.13.3. So the
Julia 1.13.x side carries the break, at least for 0.13.204. The releases between 0.13.204
and 0.13.216 were not run to confirm it for every version.
The 1.13.0 to 1.13.3 step is inferred from #369's 1.13.0 result, which this machine cannot
re-measure. Bramble requires Julia 1.13 or later, so pinning Julia is not a route.

What S2.2 must therefore do is keep Enzyme out of `@threads` and `@batch` bodies on 1.13.3,
with custom `EnzymeRules` around each whole `_batch_*` hook as the plan describes. Two limits
of this evidence: the 0.13.205 to 0.13.215 releases were not run, and only one derivative
shape (the bare loop) was measured, not Bramble's own bands.

## Upstream issue (drafted, not filed)

Title: Reverse-mode `Enzyme.autodiff` of a `Threads.@threads` loop throws `AssertionError:
mi2 !== nothing` on Julia 1.13.3

Body: the reproducer above, with Enzyme 0.13.204 and 0.13.216, Julia 1.13.3, 2 threads.
Both throw on the `dynamic` and `static` schedules. The same Enzyme 0.13.204 gives the right
gradient on Julia 1.12.7 with the same reproducer. Julia 1.13.0 with 0.13.204 gave the
right gradient in an earlier run (not re-measured here). Versions 0.13.150 to 0.13.198 refuse
threaded reverse on 1.13 with `VERSION < v"1.13"`. Ask whether the 1.13.3 threading
change broke Enzyme's handling of the loop body, and what the intended behaviour is.
