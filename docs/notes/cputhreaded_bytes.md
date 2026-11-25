# What a warm `CpuThreaded` call allocates, path by path (four threads, main at 5ee2b6bc)

Answers the first ask of #436: bytes per call under `CpuThreaded` for the in-place paths,
and what the bytes are. It decides nothing about a fix or a bound (S5.3 does).

## Method and conditions

`benchmark/cputhreaded_bytes.jl` builds each path once on a graded 2D mesh of 65 × 65 and of
513 × 513 points (axis `d` has nodes `t^(1 + d/4)`), calls it three times, then 11 more under
`@allocated`. Every row allocated the same bytes on all 11 calls. `@allocated` counts every thread, so the bytes of a spawned task
are in it.

The split comes from `Profile.Allocs` at sample rate 1 on one more warm call (median of 5):

- Task: bytes of `Task` objects.
- Closure: bytes of objects whose type is a `Function` subtype, the closures
  `Threads.@threads` and `Threads.@spawn` build per call.
- Rest: bytes minus Task and closure, so the three columns add up. It holds the
  wait machinery (see below), `Ref`s and tuples. The profiler misses 80 B per
  `@threads` region and 16 B per `@spawn`, and those bytes are in Rest.

The two `calibration` rows run nothing inside the region: they price the machinery alone.
The paths are:

- difference (`D₋ₓ!`, `D₋ᵧ!`, `D₊ₓ!`, `Dcₓ!`, `D̃ₓ!`) and average (`Mₓ!`, `M₊ₓ!`, `Mcₓ!`);
- vector calculus (`∇cₕ!`, `Δₕ!`, `divₕ!`, `divcₕ!`, `diṽₕ!`, `curlₕ!`, `curlcₕ!`, `εₕ!`) on
  a grid space and its two-component composite;
- broadcast of a fused expression and of a sum into a grid function;
- `assemble!` of the linear forms `innerₕ(f, v)`, plain, with `markers = (:dir,)` and with
  `restrict_to(:dir, v)`;
- `assemble!` into a built matrix (the replay of the stored stencil) of
  `inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(u, v)`, plain, with one more term masked to `:dir`, and with
  `dirichlet = :dir`; `jacobian!` of a form with one grid-function coefficient;
- matrix-free `mul!` of that bilinear form, plain and with the masked term;
- one `v_cycle!` and one multigrid `ldiv!` on a variable-diffusion form. The coarsest level
  keeps 17 points per axis, so the 65 × 65 mesh has 3 levels and the 513 × 513 mesh has 6.

Julia 1.13.4, default optimisation, Apple M2, `--threads=4`, `benchmark` environment. Not
measured: 3D, composite bilinear forms, ghost spaces, Kronecker products, and the
Enzyme and ForwardDiff paths.

## Table

| Group | Path | Mesh | Bytes | Task | Closure | Rest |
|---|---|---|---:|---:|---:|---:|
| calibration | `empty Threads.@threads :static region` | none | 1504 | 896 | 128 | 480 |
| calibration | `Threads.@spawn and wait` | none | 320 | 224 | 0 | 96 |
| difference | `D₋ₓ!` | 65 × 65 | 1584 | 896 | 208 | 480 |
| difference | `D₋ₓ!` | 513 × 513 | 1584 | 896 | 208 | 480 |
| difference | `D₋ᵧ!` | 65 × 65 | 1584 | 896 | 208 | 480 |
| difference | `D₋ᵧ!` | 513 × 513 | 1584 | 896 | 208 | 480 |
| difference | `D₊ₓ!` | 65 × 65 | 1616 | 896 | 240 | 480 |
| difference | `D₊ₓ!` | 513 × 513 | 1616 | 896 | 240 | 480 |
| difference | `Dcₓ!` | 65 × 65 | 1584 | 896 | 208 | 480 |
| difference | `Dcₓ!` | 513 × 513 | 1584 | 896 | 208 | 480 |
| difference | `D̃ₓ!` | 65 × 65 | 1584 | 896 | 208 | 480 |
| difference | `D̃ₓ!` | 513 × 513 | 1584 | 896 | 208 | 480 |
| average | `Mₓ!` | 65 × 65 | 1568 | 896 | 192 | 480 |
| average | `Mₓ!` | 513 × 513 | 1568 | 896 | 192 | 480 |
| average | `M₊ₓ!` | 65 × 65 | 1568 | 896 | 192 | 480 |
| average | `M₊ₓ!` | 513 × 513 | 1568 | 896 | 192 | 480 |
| average | `Mcₓ!` | 65 × 65 | 1568 | 896 | 192 | 480 |
| average | `Mcₓ!` | 513 × 513 | 1568 | 896 | 192 | 480 |
| calculus | `∇cₕ!` | 65 × 65 | 3168 | 1792 | 416 | 960 |
| calculus | `∇cₕ!` | 513 × 513 | 3168 | 1792 | 416 | 960 |
| calculus | `Δₕ!` | 65 × 65 | 0 | 0 | 0 | 0 |
| calculus | `Δₕ!` | 513 × 513 | 0 | 0 | 0 | 0 |
| calculus | `divₕ!` | 65 × 65 | 3232 | 1792 | 480 | 960 |
| calculus | `divₕ!` | 513 × 513 | 3232 | 1792 | 480 | 960 |
| calculus | `divcₕ!` | 65 × 65 | 3232 | 1792 | 480 | 960 |
| calculus | `divcₕ!` | 513 × 513 | 3232 | 1792 | 480 | 960 |
| calculus | `diṽₕ!` | 65 × 65 | 3232 | 1792 | 480 | 960 |
| calculus | `diṽₕ!` | 513 × 513 | 3232 | 1792 | 480 | 960 |
| calculus | `curlₕ!` | 65 × 65 | 3232 | 1792 | 480 | 960 |
| calculus | `curlₕ!` | 513 × 513 | 3232 | 1792 | 480 | 960 |
| calculus | `curlcₕ!` | 65 × 65 | 3232 | 1792 | 480 | 960 |
| calculus | `curlcₕ!` | 513 × 513 | 3232 | 1792 | 480 | 960 |
| calculus | `εₕ!` | 65 × 65 | 12864 | 7168 | 1728 | 3968 |
| calculus | `εₕ!` | 513 × 513 | 12864 | 7168 | 1728 | 3968 |
| broadcast | `v .= 2u + u*u` | 65 × 65 | 1664 | 896 | 192 | 576 |
| broadcast | `v .= 2u + u*u` | 513 × 513 | 1664 | 896 | 192 | 576 |
| broadcast | `v .= u + w` | 65 × 65 | 1632 | 896 | 192 | 544 |
| broadcast | `v .= u + w` | 513 × 513 | 1632 | 896 | 192 | 544 |
| innerₕ | `assemble! innerₕ(f, v)` | 65 × 65 | 1648 | 896 | 272 | 480 |
| innerₕ | `assemble! innerₕ(f, v)` | 513 × 513 | 1648 | 896 | 272 | 480 |
| innerₕ | `assemble! innerₕ(f, v; markers)` | 65 × 65 | 1664 | 896 | 288 | 480 |
| innerₕ | `assemble! innerₕ(f, v; markers)` | 513 × 513 | 1664 | 896 | 288 | 480 |
| innerₕ | `assemble! innerₕ(f, restrict_to(v))` | 65 × 65 | 1664 | 896 | 288 | 480 |
| innerₕ | `assemble! innerₕ(f, restrict_to(v))` | 513 × 513 | 1664 | 896 | 288 | 480 |
| assemble! | `assemble! bilinear` | 65 × 65 | 8240 | 4480 | 1360 | 2400 |
| assemble! | `assemble! bilinear` | 513 × 513 | 8240 | 4480 | 1360 | 2400 |
| assemble! | `assemble! bilinear, masked term` | 65 × 65 | 9936 | 5376 | 1680 | 2880 |
| assemble! | `assemble! bilinear, masked term` | 513 × 513 | 9936 | 5376 | 1680 | 2880 |
| assemble! | `assemble! bilinear, dirichlet` | 65 × 65 | 8240 | 4480 | 1360 | 2400 |
| assemble! | `assemble! bilinear, dirichlet` | 513 × 513 | 8240 | 4480 | 1360 | 2400 |
| assemble! | `jacobian!` | 65 × 65 | 14064 | 7168 | 2288 | 4608 |
| assemble! | `jacobian!` | 513 × 513 | 14064 | 7168 | 2288 | 4608 |
| matrix-free | `mul! plain` | 65 × 65 | 1584 | 672 | 96 | 816 |
| matrix-free | `mul! plain` | 513 × 513 | 1584 | 672 | 96 | 816 |
| matrix-free | `mul! masked term` | 65 × 65 | 1616 | 672 | 96 | 848 |
| matrix-free | `mul! masked term` | 513 × 513 | 1616 | 672 | 96 | 848 |
| multigrid | `v_cycle!` | 65 × 65 | 41440 | 19264 | 3360 | 18816 |
| multigrid | `v_cycle!` | 513 × 513 | 103600 | 48160 | 8400 | 47040 |
| multigrid | `ldiv! multigrid` | 65 × 65 | 41440 | 19264 | 3360 | 18816 |
| multigrid | `ldiv! multigrid` | 513 × 513 | 103600 | 48160 | 8400 | 47040 |

## Reading the table

- No path allocates in proportion to the mesh. Every row has the same bytes at 65 × 65
  and at 513 × 513, except the multigrid cycle, which has twice the levels on the large
  mesh and so 2.5 times the bytes. The bytes follow the number of threaded regions.
- Tasks are the largest part. A `Task` is 224 B. A `Threads.@threads :static` region
  makes one per thread (896 B at four threads); `mul!` spawns three bands and runs the
  fourth on the calling task (672 B). Task bytes divided by 224 give the tasks a call makes:
  4 for a stencil, 8 for `∇cₕ!`, `divₕ!` and `curlₕ!`, 20 for `assemble!` into a built
  matrix, 24 with a masked term, 32 for `εₕ!` and `jacobian!`, 86 and 215 for the two
  `v_cycle!` meshes.
- An empty region costs 1504 B and an empty `@spawn` and `wait` 320 B. The profile of
  the empty region shows, per task, a lock (16 B), a condition (32 B) and a list node
  (32 B); per region a task vector (32 B), its `Memory` (48 B) and the closure of
  `threading_run` (128 B). A stencil is that region plus 80 B (`D₋ₓ!`: 1584 B), so the
  band's own work adds almost nothing.
- Threads multiply it. At planning, two threads, a 40 × 37 mesh gave 848 to 896 B for a
  scalar stencil and 1792 B for a vector one. At four threads the same paths are 1568 to
  1616 B and 3232 B: the bytes per task grow with the thread count.
- `Δₕ!` allocates 0 B at both sizes; this run does not say why.
- A masked term adds one region of tasks to `assemble!` (+1696 B, 6 against 5) and
  16 B to the linear forms. `dirichlet = :dir` adds none.
- `jacobian!` is 14064 B for a form with one grid-function coefficient. S3.14's
  measurement at two threads on similar forms was 8304 to 17280 B, so it is the largest
  single call here after the multigrid cycle.

## Command

From the worktree root, with the four-thread slot pool held:

```
BRAMBLE_JULIA_SLOTS=4 .claude/scripts/julia_slot.sh 4 julia --startup-file=no --threads=4 \
    --project=benchmark benchmark/cputhreaded_bytes.jl > table.md
```

The table above is that output. A `SPREAD` line on stderr names a row whose 11 calls
differed; none appeared.
