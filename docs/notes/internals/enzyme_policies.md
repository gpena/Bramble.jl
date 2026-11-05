# Enzyme under the `CpuPolyester` hooks: a census (#369, S1.1)

Plan v4-2-0-followup, subplan S1.1. `CpuPolyester` runs its loops through 17 `_batch_*`
hooks that `ext/BramblePolyesterExt.jl` fills. This note records, for each hook, a smallest
public call that reaches it and what Enzyme does with that call. Every status below is read
from `benchmark/results/enzyme_policy_census.toml`, written by
`benchmark/enzyme_policy_census.jl` from the cases of `test/ext/enzyme_policy_cases.jl`. The
TOML's `script_sha` ties the table to that script. The run is one pass on an Apple M2,
Julia 1.13.3, Enzyme 0.13.216, Polyester 0.7.19, 2 threads, a seeded non-uniform
41 x 41 mesh.

Later subplans call `EnzymePolicyCases.check(hook)`: it throws unless Enzyme reverse (direct
`Enzyme.gradient`, and through DifferentiationInterface) and DifferentiationInterface forward
match `ForwardDiff.gradient` to 1e-10 relative. Nine more entries, `EnzymePolicyCases.THREADED`,
hold the `CpuThreaded` engines for S2.2. The census does not run those.

## How it was measured

Every measurement is a child process (`julia --threads=2`) loading Enzyme from a temporary
environment whose `Manifest.toml` is a copy of the root one, so a crash is recorded and not
fatal. Per hook:

1. A reach pass. One direct reverse gradient of the case runs under `CpuPolyester`, with the
   extension's rule counters read before and after it: each hook's rule counts its entries
   (`_entered!` in `ext/BrambleEnzymePolyesterExt/`, read through the harness's
   `_rule_entries`). `hits` is the per-hook delta, and `reaches` is true when the case's own
   hook's delta is positive. A case whose own rule was not entered is measured again by a
   primal call (no Enzyme), counting the hook entries through `Bramble._late` (the route of
   every `src/` call into a hook); `reach_by` says `primal` then. No stub rules are defined.
   An earlier census put counting stub rules on the hook types. With the extension's rules
   loaded those stubs were never entered (every counter read 0 while the gradients were
   right), and the stubs were not meant to coexist with the real rules, so the census counts
   through the rules' own counters instead. Why the stubs were never entered was not isolated.
2. A status pass per mode (`reverse`, `di_reverse`, `forward`) under `CpuPolyester`, so Enzyme
   meets the real hook rules and bodies.
3. A serial control: the direct reverse gradient under `CpuSerial`.

## Census

All 17 hooks are reached by their case. All 17 throw in every Enzyme mode under
`CpuPolyester` and are `ok` under `CpuSerial`, so what fails is the threaded body, not the
call. The three modes of a hook fail with the same message, so one column carries it.

| hook | smallest call | reaches (by) | reverse | DI reverse | forward | serial reverse | first failure |
|:--|:--|:--|:--|:--|:--|:--|:--|
| `_batch_dot` | `innerₕ(u, w)` | true (rule) | throws | throws | throws | ok | IllegalTypeAnalysisException, in `_batch_dot` |
| `_batch_dot_masked` | `Bramble._dot_masked(policy, u, weights, w, mask)`, positional | true (rule) | throws | throws | throws | ok | IllegalTypeAnalysisException, in `_batch_dot_masked` |
| `_batch_for!` | `Rₕ(W, f)` with a closure over the parameters | true (rule) | throws | throws | throws | ok | IllegalTypeAnalysisException, in `_slabs_run!` |
| `_batch_axis_for!` | `Bramble._sweep_for!` on a `CartesianIndices` | true (rule) | throws | throws | throws | ok | IllegalTypeAnalysisException, in `_slabs_run!` |
| `_batch_scatter_for!` | `Rₕ` on a composite space | true (rule) | throws | throws | throws | ok | IllegalTypeAnalysisException, in `_scatter_cover_run!` |
| `_batch_bilinear_band_replay!` | `assemble` of a diffusion form, 41 x 41 | true (rule) | throws | throws | throws | ok | IllegalTypeAnalysisException, in `_batch_average_engine!` |
| `_batch_bilinear_colour_replay!` | `assemble` of the same form, 41 x 3 | true (rule) | throws | throws | throws | ok | IllegalTypeAnalysisException, in `_batch_average_engine!` |
| `_batch_linear_colour_sweep!` | `assemble` of `innerₕ(uₕ, D₋ᵧ(v))`, 41 x 3 | true (rule) | throws | throws | throws | ok | IllegalTypeAnalysisException, in `_batch_bands!` |
| `_batch_linear_band_sweep!` | `assemble` of `innerₕ(uₕ, v)`, 41 x 41 | true (rule) | throws | throws | throws | ok | IllegalTypeAnalysisException, in `_batch_bands!` |
| `_batch_difference_engine!` | `D₋ₓ(u)` | true (rule) | throws | throws | throws | ok | IllegalTypeAnalysisException, in `_batch_difference_engine!` |
| `_batch_average_engine!` | `Mₓ(u)` | true (rule) | throws | throws | throws | ok | IllegalTypeAnalysisException, in `_batch_average_engine!` |
| `_batch_centered_average_engine!` | `Mcₓ(u)` | true (rule) | throws | throws | throws | ok | IllegalTypeAnalysisException, in `_batch_centered_average_engine!` |
| `_batch_run_bands!` | `S₊(u, 1)` | true (rule) | throws | throws | throws | ok | IllegalTypeAnalysisException, in `_batch_run_bands!` |
| `_batch_mf_bands!` | `mul!` with a `matrix_free_operator` | true (rule) | throws | throws | throws | ok | IllegalTypeAnalysisException, in `_batch_mf_bands!` |
| `_batch_broadcast!` | `u .= v .+ 1.0` | true (rule) | throws | throws | throws | ok | IllegalTypeAnalysisException, in `_batch_broadcast!` |
| `_batch_kron_lines!` | `mul!` with a `kronecker_operator` | true (rule) | throws | throws | throws | ok | IllegalTypeAnalysisException, in `_batch_kron_lines!` |
| `_batch_csr_spmv!` | `semidiscretize_rhs(sd)(du, u, nothing, 0.0)` | true (rule) | throws | throws | throws | ok | IllegalTypeAnalysisException, in `_batch_csr_spmv!` |

`reaches` is a measured count, not an assertion. Two details of the table:

- A case's first failure is not always in its own hook. The bilinear cases build the
  diffusion coefficient with `Mₕ(uₕ)`, which enters `_batch_average_engine!` and
  `_batch_broadcast!` before the replay (`also_reaches` in the TOML), and Enzyme reports the
  first failure it compiles. Likewise `_batch_for!` and `_batch_axis_for!` fail in
  `_slabs_run!`, and the linear sweeps in `_batch_bands!`, the ext helpers those hooks call.
  The cases still reach their own hook, which is what the census claims.
- The failure has one shape everywhere: `IllegalTypeAnalysisException`, first frame
  `+ @ ./pointer.jl:314` under `ManualMemory.store!` in Polyester's `setup_batch!`. The plain
  `Polyester.@batch` loop of the first reproducer below throws the same exception type with
  a different message (an "Illegal updateAnalysis" on the `BatchClosure` store), so the two
  are neighbours, not shown to be one bug.

### Departures from the builders the plan listed

- `_batch_dot_masked` is reached through the positional `Bramble._dot_masked`, with the mask
  and weights built outside the differentiated function. Every public caller
  (`innerₕ(...; markers = ...)`) aborts Enzyme even under `CpuSerial`, so the positional route
  is the only one with a serial `ok` (relative error 8.9e-17). The abort is the third
  reproducer below.
- `_batch_bilinear_colour_replay!` and `_batch_linear_colour_sweep!` use a thin 41 x 3 mesh:
  colour replay runs only when `_band_count(len, span, nthreads) == 0`
  (`src/utils/linear_algebra.jl:408`), which 41 x 41 never reaches. The linear colour case
  also takes a y-difference of the test function, giving the term a colouring stride of 2; a
  plain load's stride of 1 always bands, at 41 points or at 3.
- `_batch_axis_for!` has no public caller. Its case calls `Bramble._sweep_for!` on a
  `CartesianIndices`.

### `_batch_mf_bands!` and `_batch_broadcast!`: rules need a different argument convention

In the earlier stub census both hooks stopped with `CallingConventionMismatchError` at the hook
call, before the stub ran, and their reach came from a primal pass (`reach_by` is `primal`
in the TOML): the hook entries counted through `Bramble._late` (the route of every `src/`
call into a hook) on a plain call of the case, without Enzyme. The error is not the rule's
signature. Enzyme's full message for `_batch_broadcast!` names the cause:

```text
Custom rule for method argument 2 of type Base.Broadcast.Broadcasted{...} has mismatch
between roots_activep (DFT_DUP_ARG) and activep (DFT_CONSTANT).
  - Non-rooted types: (DefaultArrayStyle{1}, typeof(+), Float64, Bool, Int64, Int64)
  - Rooted types: (Vector{Float64},)
```

A plain-Enzyme probe, without Bramble, shows what a custom rule accepts in Enzyme 0.13.216 on
Julia 1.13.3. The hook is `h(v, x, n)`, `x` built from the differentiated parameters, and a
counting rule (the stub of the earlier census) sits on `h`:

| argument `x` of the hook | result |
|:--|:--|
| a `Vector` | rule runs |
| a struct holding only a `Vector` | rule runs |
| a tuple holding only a `Vector` | rule runs |
| `Ref(vector)` | rule runs |
| `Ref(struct with a Vector and a Float64)` | rule runs |
| the vector and the `Float64` as two separate arguments | rule runs |
| a struct with a `Vector` and a `Float64` | `CallingConventionMismatchError` |
| a tuple `(vector, 3.0)` | `CallingConventionMismatchError` |
| a closure capturing a `Vector` and a `Float64` | `CallingConventionMismatchError` |

The convention that works: no by-value struct, tuple or closure argument that mixes an array
(a rooted type) with isbits fields. Pass the pieces as separate arguments, or behind a `Ref`.
`Broadcasted` mixes them (its leaf `Extruded{Vector}` next to `Float64` and the `+`
function), and so do the `ActionSink` and `_MFFusedPlan` that `_batch_mf_bands!` takes (arrays
next to `Float64`, `Int` and `nothing` fields). The other 15 hooks take arrays and scalars
as separate arguments, or closures whose captures are arrays only, so they compile.

For `_batch_mf_bands!` Enzyme's text differs: `Julia compiled the custom rule MethodInstance
... without a specialized entry point`, naming `augmented_primal` with no arguments. Its
arguments have the same mixed shape, but the census did not prove that this is the same cause.
Registering a method with the four arguments spelled out (`s, plan, nbands, walk`) made Enzyme
skip the rule and differentiate the hook body, where it met `_batch_bands!` and the
`IllegalTypeAnalysisException`. S1.3.x owns the rules for these two hooks and has to pass their
arguments in a convention from the table.

## Upstream reproducers

The first two have no Bramble in them and are drafted as issues for Enzyme.jl. Nothing is
filed from here. The TOML records each one's status and the Enzyme and Polyester versions.

### Enzyme.jl: `Polyester.@batch` loop

Enzyme 0.13.216, Polyester 0.7.19, Julia 1.13.3. Throws
`Enzyme.Compiler.IllegalTypeAnalysisException`, with an "Illegal updateAnalysis" message on
the `BatchClosure` store.

```julia
using Polyester, Enzyme
function sq!(y, x)
    Polyester.@batch for i in eachindex(y)
        y[i] = x[i]^2
    end
    return nothing
end
x = collect(1.0:100.0); dx = zeros(100); y = zeros(100); dy = ones(100)
Enzyme.autodiff(Reverse, sq!, Const, Duplicated(y, dy), Duplicated(x, dx))
dx ≈ 2 .* x
```

### Enzyme.jl: `Threads.@threads` loop

Enzyme 0.13.216, Julia 1.13.3, run with `--threads=2`. Throws `AssertionError: mi2 !== nothing`.
The same reproducer ran under several Enzyme versions in `docs/notes/internals/enzyme_threaded.md`.

```julia
using Enzyme
function sq!(y, x)
    Threads.@threads for i in eachindex(y)
        y[i] = x[i]^2
    end
    return nothing
end
x = collect(1.0:100.0); dx = zeros(100); y = zeros(100); dy = ones(100)
Enzyme.autodiff(Reverse, sq!, Const, Duplicated(y, dy), Duplicated(x, dx))
dx ≈ 2 .* x
```

### Bramble: a keyword call to `innerₕ` aborts Enzyme under `CpuSerial`

Not an Enzyme issue as such, and not drafted for Enzyme.jl: Bramble's `innerₕ(u, w; markers =
...)` lowers to a keyword call (`kwcall` of `inner` with `__markers__`). Enzyme dies in the
LLVM pass, with no Julia exception, even with the default serial backend and no threads in
the loop. Measured on Julia 1.13.3 with Enzyme 0.13.216:

```text
Assertion failed: (false && "caching potentially capturing/offset of allocation"), function needsCacheWholeAllocation, file /workspace/srcdir/Enzyme/enzyme/Enzyme/GradientUtils.cpp, line 10433.
signal 6: Abort trap: 6
```

```julia
using Bramble, Enzyme, Random

W = gridspace(mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (41, 41), (false, false)))
B = randn(MersenneTwister(1), ndofs(W), 4)
w = element(W)
parent(w) .= cos.(1:ndofs(W))

function load(p)
    u = element(W, eltype(p))
    parent(u) .= B * p
    return u
end

f(p) = innerₕ(load(p), w; markers = (:boundary,))
Enzyme.gradient(set_runtime_activity(Reverse), Const(f), [0.5, 0.6, 0.7, 0.8])
```

The census avoids it through the positional `_dot_masked`. S1.3.2 owns the rule for
`_batch_dot_masked` and has to make the keyword route differentiate as well.

### Enzyme.jl: a custom rule on a hook with a struct argument mixing an array and an isbits field

Enzyme 0.13.216, Julia 1.13.3. Throws `CallingConventionMismatchError`, with the cause
"mismatch between roots_activep (DFT_DUP_ARG) and activep (DFT_CONSTANT)". Replacing the
struct by `Box1` (the array only), or passing the `Float64` as its own argument, makes the
rule run.

```julia
using Enzyme
using Enzyme.EnzymeRules: EnzymeRules, RevConfig, AugmentedReturn
struct Box{A}
    a::A
    k::Float64
end
@noinline function hook(v, b::Box, n)
    for i in 1:n
        v[i] = b.a[i] * b.k
    end
    return nothing
end
function EnzymeRules.augmented_primal(config::RevConfig, func::Const{typeof(hook)},
        ::Type{<:Const}, args...)
    func.val(map(a -> a.val, args)...)
    return AugmentedReturn(nothing, nothing, nothing)
end
EnzymeRules.reverse(config::RevConfig, func::Const{typeof(hook)}, ::Type{<:Const}, tape,
    args...) = map(_ -> nothing, args)
function f(p)
    x = p .* 2
    y = zeros(length(p))
    hook(y, Box(x, 3.0), length(p))
    return sum(abs2, y)
end
Enzyme.gradient(Reverse, Const(f), [1.0, 2.0, 3.0])
```
