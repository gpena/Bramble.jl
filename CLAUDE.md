# Working on Bramble.jl

A Julia package for finite-difference discretisations of PDEs: geometry and domains,
meshes, grid spaces, vector elements, difference operators, and linear/bilinear form
assembly. Julia 1.12 or newer.

This file is an index and a set of working agreements. It deliberately does not repeat
what the skills below already say — go to the skill.

## Where the knowledge lives

Read the relevant skill *before* starting, not after something breaks. Each one exists
because the lesson in it was learned the hard way on this package.

| Doing this | Read |
|---|---|
| Anything touching the public API | `bramble` |
| Naming a function, type, or variable | `bramble-naming` |
| Writing or editing Julia source | `bramble-formatting` |
| Writing docstrings or documentation | `bramble-documentation` |
| Reviewing a diff, or auditing a folder | `bramble-code-review` |
| Claiming a speedup, regression, or coverage figure | `bramble-verification` |
| Optimising, chasing allocations, parallelising | `julia-performance` |
| Running benchmarks or saving a baseline | `bramble-benchmarks` |
| Tagging or publishing a version | `bramble-release` |

`bramble-verification` is the one people skip and should not. It is about how to know a
result is real, and every entry in it is a mistake that was actually made here — three of
them were reported as findings before being withdrawn.

## Working agreements

- **Commit freely; push only when asked.** Committing is cheap and reversible. Pushing is
  not, and this repository's `main` is what the docs and releases build from.
- **Stage named files, never `git add -A` or `git add .`.** The tree often has concurrent
  work in it that isn't yours to commit.
- **Anything visible to other people needs asking first**: pushing, opening or closing
  issues, commenting on issues or PRs, creating releases, triggering workflows. Doing the
  work is welcome; announcing it to others is the user's call.
- **Preview before committing anything that renders.** Docs pages, charts, generated
  markdown — build it, look at it, and show the user. "The build passed" is not evidence
  that the page is right.
- **Never update the Zenodo DOI on your own.** Whether a release gets a new DOI at all is
  a deliberate decision, not a step. See `bramble-release`.
- **Examples in documentation must be checked for what they compute**, not merely that
  they run. A snippet that executes cleanly and prints the wrong number is worse than one
  that errors.

## Commands

```bash
# Full test suite — the gate before any release, and after any merge
julia --project=. -e 'using Pkg; Pkg.test("Bramble")'

# Formatting (SciML style; JuliaFormatter is deliberately not a dependency)
julia -e 'using Pkg; Pkg.activate(mktempdir()); Pkg.add("JuliaFormatter");
          using JuliaFormatter; format(["src", "test"])'

# Docs, including the generated benchmarks page
julia --startup-file=no --project=docs docs/make.jl
```

A healthy suite is **~6000 passing, 7 broken, Quality 12/12**. The 7 broken are a known
baseline, not a to-do list — do not silently "fix" them, and do treat a change in that
count as a real signal.

## Things that have actually caused trouble

- **Push-triggered CI runs the tests only.** Formatting, Aqua, JET and the docs build live
  in `nightly.yml` (06:00 UTC) — so a push can be green and still be unformatted or
  failing quality gates. Run the formatter and the full suite yourself; don't wait for CI
  to tell you.
- **A push does not redeploy the docs.** The live site updates on the nightly schedule, or
  when `docs-on-demand.yml` is triggered manually. A docs fix on `main` is not live yet.
- **`docs/src/benchmarks.md` is generated** by `docs/generate_benchmarks.jl`. Edit the
  generator, never the markdown — the next docs build will overwrite anything you hand-edit.
- **`benchmark/baselines/*.json` are settled measurements**, indexed by commit and committed
  to git. Add to them; do not re-run the suite to overwrite numbers already recorded for a
  past release. If a benchmark is renamed, rename the key in the old baselines too, or its
  trend line silently splits in two.
- **Benchmarks need mains power and an otherwise idle machine.** On battery the harness
  refuses to save. Concurrent work of your own on the same machine is the usual cause of a
  measurement that looks like a regression and isn't.
- **Element types come from the data, not the space.** `ForwardDiff.Dual` values have to
  flow through assembly, so reading a type off the space breaks automatic differentiation.
  `bramble-verification` §4 lists the four places this went wrong.
