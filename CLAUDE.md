# Working on Bramble.jl

Finite-difference discretisations of PDEs: domains, meshes, grid spaces, vector elements,
difference operators, form assembly. Julia 1.12+.

An index, not a copy — go to the skill. How code must be written is in
[STANDARDS.md](STANDARDS.md).

## Which skill

| Doing this | Read |
|---|---|
| Touching the public API | `bramble` |
| Writing or editing Julia source | `bramble-formatting`, `bramble-naming` |
| Docstrings or documentation | `bramble-documentation` |
| Reviewing a diff, auditing a folder | `bramble-code-review` |
| Claiming a speedup, regression, or coverage figure | `bramble-verification` |
| Optimising, chasing allocations, parallelising | `julia-performance` |
| Running benchmarks, saving a baseline | `bramble-benchmarks` |
| Tagging or publishing a version | `bramble-release` |

`bramble-verification` is the one people skip and should not: every entry is a mistake
actually made here, three of them reported as findings before being withdrawn.

## Working agreements

- **Commit freely; push only when asked.**
- **Stage named files, never `git add -A`** — the tree often holds concurrent work.
- **Ask before anything others can see**: pushing, opening or closing issues, commenting,
  releases, triggering workflows.
- **Preview anything that renders** before committing. A green build is not evidence the
  page is right.
- **Never update the Zenodo DOI unprompted** — `bramble-release` §8.

## Commands

```bash
# Full suite — healthy is ~6000 passing, 7 broken, Quality 12/12.
# The 7 broken are a known baseline, not a to-do list; a change in that count is a signal.
julia --project=. -e 'using Pkg; Pkg.test("Bramble")'

# Formatting (SciML; JuliaFormatter is deliberately not a dependency)
julia -e 'using Pkg; Pkg.activate(mktempdir()); Pkg.add("JuliaFormatter");
          using JuliaFormatter; format(["src", "test"])'

# Docs, including the generated benchmarks page
julia --startup-file=no --project=docs docs/make.jl
```

## Traps this repo has actually sprung

- **Push-triggered CI runs tests only.** Format, Aqua, JET and docs live in `nightly.yml`
  (06:00 UTC), so a push can be green and still unformatted. Check locally.
- **A push does not redeploy the docs** — nightly, or `docs-on-demand.yml` manually.
- **`docs/src/benchmarks.md` is generated** by `docs/generate_benchmarks.jl`. Edit the
  generator; the next build overwrites the markdown.
- **`benchmark/baselines/*.json` are settled measurements.** Add to them; never re-run to
  overwrite a past release's numbers. Rename a benchmark and you must rename its key in the
  old baselines too, or the trend line silently splits in two.
