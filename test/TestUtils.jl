# Shared test helpers, in one place rather than duplicated per file.
#
# Every subsystem test file is now its own module (test/<dir>/<name>.jl ->
# module <Dir><Name>Tests ... end), so a name defined here is reached explicitly via
# `using ..TestUtils: name1, name2` -- never implicitly, matching this package's own
# explicit-import convention (STANDARDS.md, bramble-code-review).
#
# These used to live at the top of runtests.jl, back when every test file was included
# into bare Main and so already shared one namespace with them. Two of them, `_fd` and
# `_tri`, used to also be duplicated verbatim in individual files (space/autodiff.jl and
# form/autodiff.jl; form/symmetrize.jl and form/autodiff.jl) before that; hoisting them
# once here is what fixed the silent-overwrite hazard the first time.
module TestUtils

using Test
using SparseArrays: SparseMatrixCSC, nonzeros, spdiagm
using ForwardDiff
using Printf: @sprintf
using Random: Random, Xoshiro
import Bramble
using Bramble: ×, Rₕ, Serial, VectorElement, backend, box, change_points!, components, domain, element, form,
               gridspace, inner₊, innerₕ, interval, iterative_refinement!, mesh, npoints, space, ∇ₕ
using Bramble: indices, point

# The test group, read once here rather than in each entry point, so that a subsystem's own
# `runtests.jl` -- which is a standalone entry point and does not go through
# test/runtests.jl -- decides the same way the full suite does. test/runtests.jl reads these
# too; see its own comments for what each group means.
const TEST_GROUP = get(ENV, "BRAMBLE_TEST_GROUP", "all")

# `BRAMBLE_TEST_SHARD` narrows the `unit` block to one of four parts, so CI can run each on
# its own runner: `forms-1` and `forms-2` are the two halves of test/form/, `ghost` is
# test/ghost/, and `rest` is the remainder of the unit block. It also narrows the package
# extensions block (the `ext` and `backends` groups) to one of two parts, which Weekly.yml
# runs on separate runners: `ext-solvers` is the plotting, solver and Kronecker files and
# `ext-polyester` the Polyester files. Unset runs everything, which is what every local run
# does; the quality, ad, gpu, examples and grids blocks ignore it. The gate sits at the
# include sites in test/runtests.jl and test/form/runtests.jl, through `in_shard`.
const TEST_SHARD = get(ENV, "BRAMBLE_TEST_SHARD", "")

const TEST_SHARDS = ("forms-1", "forms-2", "ghost", "rest", "ext-solvers", "ext-polyester")

"""
    in_shard(name)

True when no shard is selected or the selected shard is `name`. Throws an `ArgumentError`
for a `name`, or a `BRAMBLE_TEST_SHARD`, outside `TEST_SHARDS`, since a
misspelt shard would select nothing and pass green.
"""
function in_shard(name::AbstractString)
    unknown(shard) = ArgumentError("unknown test shard $(repr(shard)); expected one of $TEST_SHARDS")
    name in TEST_SHARDS || throw(unknown(name))
    isempty(TEST_SHARD) || TEST_SHARD in TEST_SHARDS || throw(unknown(TEST_SHARD))
    return isempty(TEST_SHARD) || TEST_SHARD == name
end

# `slow` is the every-push gate's overflow: the unit suite plus the files whose cost is out
# of proportion to what a *push* learns from them. CI.yml (macOS, per push) runs `unit` and
# so skips them; nightly.yml runs `slow` on both platforms once a day, and Weekly.yml
# runs `slow` as the suite half of its matrix. A run with no group set gets `all`, which includes them --
# so `.claude/scripts/test.sh` and a bare `Pkg.test` are unaffected.
#
# What is behind it, and why, with the measured cost of each on Julia 1.13, macOS, 4
# threads, against a 6m06s `unit` run:
#
# (The worked-example pages, 1m03.8s, used to sit here; they are the `examples` group now,
# which the documentation workflow runs and no other group does.)
#
#   form/jacobian_pattern.jl      ~46s   the AST-derived sparsity pattern checked against
#                                        SparseConnectivityTracer's AD-traced pattern as
#                                        ground truth in 1D/2D/3D, plus Newton solves that
#                                        use it. The expensive half is a cross-check against
#                                        another package, and it moves when that package or
#                                        the simplifier moves, not when an operator does.
#                                        Also behind WITH_AD_TESTS (below), so it is off
#                                        in every group until that switch comes back on.
#   form/nested_operators.jl             the full operator-pair grid (81/135 pairs, ~309s
#                                        locally). The cyclic cover of the same pairs
#                                        stays in `unit`.
#   form/vector_calculus.jl     ~159s   composite ∇ₕ/εₕ/divₕ checked against a hand-expanded
#                                        form (S6.5). The cost is 100% compile: the
#                                        hand-expanded helpers branch on `i == j` to return
#                                        structurally different `LazyOp` subtrees, so Julia
#                                        infers a `Union`, then sums D² (9 in 3D) of them
#                                        through `innerₕ`/`+`/`assemble`. A cost of how the
#                                        test is written, not of the feature -- see its own
#                                        comment in form/runtests.jl.
#   form/coefficient_shift.jl   ~11.7s  a coefficient inside a shifting node, swept over
#                                        seven operator kinds via `for (nm, op) in ((...))`.
#                                        `op` is a `Union` of all seven, so each iteration's
#                                        `form()`/`assemble()` infers over the whole Union --
#                                        417ms/test against a ~65-95ms/test median elsewhere
#                                        in this subsystem. Same class of cost as
#                                        vector_calculus.jl above, smaller in absolute terms.
#   the 15 Supposition testsets   ~20s   named `... (Supposition)`, plus meshnd.jl's
#                                        "Refinement invariants", gridspaces.jl's "Partition
#                                        of unity" and jump.jl's "Leibniz product rule",
#                                        which are `@check` blocks under a plainer name.
#                                        719 of the suite's assertions, and they are a
#                                        search: running them more often beats running them
#                                        sooner, because each run draws different inputs.
#                                        The deterministic tests each one sits beside stay
#                                        on the gate, so a property moving here never leaves
#                                        its operator uncovered.
#
# Each Supposition testset is gated at its own site rather than centrally, with
# `WITH_SLOW_TESTS && @testset ...` so the block keeps its indentation. The `Non-vacuous`
# guards nested inside two of them travel with their property, which is right: a guard that
# a property is not vacuously true has no job in a run where the property does not execute.
const WITH_SLOW_TESTS = TEST_GROUP in ("all", "slow", "full", "grids")

# Combinatorial grids (gpena/Bramble.jl#789). A test that checks every combination of a few
# lists (operator pairs, taps over operands, form cases) compiles a form per combination, so
# its cost grows with the product of the lists. `_grid_sample` runs such a grid at three
# sizes:
#
#   unit    the cover alone: a deterministic subset, chosen by the test, that puts every
#           item through on each side. No randomness on the push gate.
#   slow    the cover plus `budget` more, drawn at random from the rest (nightly.yml).
#   grids   every combination (Weekly.yml; `.claude/scripts/test.sh grids` locally). The
#           group runs only the files that hold such grids, listed in test/runtests.jl.
#
# The draw is seeded by `GRID_SEED` mixed with the grid's name, so each grid draws
# independently of the others and of the order they run in. `GRID_SEED` is
# `BRAMBLE_TEST_GRID_SEED` when set, else drawn afresh per run, so successive nightlies
# reach different parts of each grid. A draw is rerun exactly, on the same Julia version,
# with `BRAMBLE_TEST_GRID_SEED=<seed> .claude/scripts/test.sh slow`.
const WITH_FULL_GRIDS = TEST_GROUP == "grids"
const GRID_SEED = let s = get(ENV, "BRAMBLE_TEST_GRID_SEED", "")
    isempty(s) ? rand(Random.RandomDevice(), UInt32) : parse(UInt32, s)
end

"""
    _grid_sample(name, items, incover, budget)

The combinations of `items` to run in this group: those satisfying `incover` under `unit`,
those plus `budget` drawn from the rest under `slow`, and all of them under `grids`. Order
within `items` is kept.
"""
function _grid_sample(name::AbstractString, items, incover, budget::Integer)
    all_items = collect(items)
    WITH_FULL_GRIDS && return all_items
    WITH_SLOW_TESTS || return filter(incover, all_items)
    rest = findall(!incover, all_items)
    rng = Xoshiro(hash(name, UInt64(GRID_SEED)))
    drawn = rest[Random.randperm(rng, length(rest))[1:min(budget, length(rest))]]
    keep = map(incover, all_items)
    keep[drawn] .= true
    return all_items[keep]
end

# Whether a 3D mesh may be built: every group but `unit` (gpena/Bramble.jl#650). It is keyed
# on `unit` itself and not on `WITH_SLOW_TESTS`, which is also false in `ext` and `backends`,
# where the Polyester and Metal files still run their 3D cases. `TEST_DIMS` is the dimension
# loop for a test that covers each dimension: `for D in TEST_DIMS`, or
# `filter(>=(2), TEST_DIMS)` for one that starts at 2D. A case that is 3D only takes
# `WITH_3D_TESTS && @testset ...`, with a comment that it runs nightly.
const WITH_3D_TESTS = TEST_GROUP != "unit"
const TEST_DIMS = WITH_3D_TESTS ? (1, 2, 3) : (1, 2)

# The guard: under `unit`, building a 3D mesh is an error that names the gate to use, so a
# 3D case added to a unit file later fails on its first run and not at the next timing
# measurement. It is a more specific method of the constructor `mesh` ends in (every 3D mesh
# goes through it), and it does not exist in the other groups.
if !WITH_3D_TESTS
    function Bramble.MeshnD(
            ::Bramble.CartesianProduct{3}, ::Bramble.MeshMarkers, ::CartesianIndices{3},
            ::Bramble.Backend, ::Tuple
    )
        error("a 3D mesh is built under the `unit` group (gpena/Bramble.jl#650): loop over ",
            "`TEST_DIMS` or gate the case with `WITH_3D_TESTS && @testset`; it runs in `slow`")
    end
end

# Automatic differentiation and GPU tests are switched off in every group until the
# milestones that own them: AD until v4.3.0, which decides which backends Bramble keeps, and
# GPU until v4.4.0, which brings the device path to parity with the CPU. The files stay in
# the tree, and their packages are not in test/Project.toml. With AD off, one ForwardDiff
# smoke test (form/forwarddiff_smoke.jl) still runs in `unit`, so assembly with dual numbers
# cannot silently regress.
#
# Switched off until v4.3.0 (AD). To turn back on, add Enzyme, Mooncake,
# SparseConnectivityTracer, SparseMatrixColorings, DifferentiationInterface,
# SciMLSensitivity and ChainRulesCore (whichever the file loads) to test/Project.toml and set
# `BRAMBLE_TEST_AD=true`, then remove this switch once v4.3.0 settles the backends.
#   whole files, `unit` (and slow/full/all)
#     space/autodiff.jl, space/autodiff_backends.jl   test/runtests.jl
#     form/autodiff.jl                                test/form/runtests.jl
#     form/jacobian_pattern.jl (also needs `slow`)    test/form/runtests.jl
#   whole files, groups `ad`, `full`, `backends` ("AD backends (expensive)" in runtests.jl)
#     space/autodiff_heavy.jl, space/autodiff_policies.jl, ext/chainrules_enzyme_ext.jl,
#     examples/inverse_diffusion.jl, ext/sparse_ad_ext.jl, ext/ad_backend_verification.jl,
#     ext/sciml_sensitivity_ext.jl, examples/transient_inverse_problem.jl
#   not gated: ext/chainrules_ext.jl runs in the `ext` group (it needs only ChainRulesCore)
#   inline blocks, gated with `WITH_AD_TESTS` at the site
#     form/bilinear.jl "Matrix differentiation", "Dual arguments";
#     form/linear.jl "Parallel differentiation", the nonlinear residual Jacobian `if`,
#     "Assembled residual differentiation"; form/dirac.jl three `if`s (ForwardDiff through
#     Dirac strengths); form/kronecker.jl `if` (Duals through the Kronecker scratch);
#     form/interpolation_operator.jl "Differentiation"; form/type_cached_assemble.jl
#     "matches direct, Float64 and Dual", "structural: pattern matches assemble(a)",
#     "Newton solve matches direct"; form/semidiscrete.jl `if` (Dual `t` rebuild);
#
#     form/source_operators.jl "Source differentiation", "lowered source: Dual
#     propagates"; form/reaction_flux.jl "reaction: Dual load vector";
#     ext/sparspak_ext.jl `if` (generic-eltype AD through Sparspak).
#
# Switched off until v4.4.0 (GPU). To turn back on, add Metal, GPUArrays and
# KernelAbstractions to test/Project.toml, set `BRAMBLE_TEST_GPU=true` and run the `gpu`
# group on an Apple Silicon Mac (outside CI, or with `CI` unset, where `_run_gpu_tests()` below
# skips device kernels on a CI runner), then remove this switch once v4.4.0 lands.
#   whole files, group `gpu`, ext/metal_ext.jl (which now also holds the "Metal GPU
#     backend" testset that used to sit inline in utils/backends.jl),
#     ext/metal_fullstack.jl, ext/metal_assembly_replay.jl, ext/metal_form_assembly.jl.
const WITH_AD_TESTS = get(ENV, "BRAMBLE_TEST_AD", "false") == "true"
const WITH_GPU_TESTS = get(ENV, "BRAMBLE_TEST_GPU", "false") == "true"

# Per-file trace, active only under CI (which sets `CI=true`) or `BRAMBLE_TEST_TRACE=1`,
# silent otherwise -- a maintainer's local run stays quiet by default. Exists because macOS
# CI's unit job was once SIGKILLed by a single test file's compile blowing past the runner's
# memory budget (a 27-term 3D form under `--code-coverage`): the job's log ended mid-file with no
# indication which one was running, and no memory figure to compare against. A start line
# printed (and flushed) before each file names the file that was running if the job dies
# partway through; an end line's `maxrss` shows which file actually holds the peak memory,
# since `Sys.maxrss()` is cumulative for the process and so only distinguishes files when
# read at each file's own boundary.
const TRACE_TESTS = get(ENV, "CI", "false") == "true" ||
                    get(ENV, "BRAMBLE_TEST_TRACE", "0") == "1"

# The trace also reports the JIT compile time spent inside each file, read from Julia's
# process-wide counter (summed over threads), so the counter has to be on for the trace.
TRACE_TESTS && Base.cumulative_compile_timing(true)

# Turns the trace above into a budget: unset (the default) is today's behaviour, no check at
# all. Set, `Sys.maxrss()` is the process peak (see the comment on `TRACE_TESTS`), so the
# first file whose *end-of-file* peak crosses the budget is the one that fails -- not
# necessarily the file that allocated the most, only the one that tipped the running total
# over. Parsed once here rather than in every call to `traced_include`.
#
# The budget is a fraction of the machine's memory, not a fixed figure: the peak follows the
# runner's size, since the GC lets the heap grow further before collecting where there is
# more memory. The same forms-1 shard peaked at 2.70 GB on a 7 GB macOS runner and 6.13 GB on
# a 14 GB one (nightly runs 39943360340 and 40009578770), and a fixed budget low enough to
# fail by name on the first is too low for the second.
const MAXRSS_BUDGET_GB = let v = get(ENV, "BRAMBLE_TEST_MAXRSS_FRACTION", "")
    isempty(v) ? nothing : round(parse(Float64, v) * Sys.total_memory() / 1024^3; digits = 2)
end

# The machine's memory and the `--heap-size-hint` the test process got, once at the top of
# the trace, so a high run can be checked against the runner it got. A hint of 0 means none
# was passed.
if TRACE_TESTS
    let total_gb = Sys.total_memory() / 1024^3
        hint_gb = Base.JLOptions().heap_size_hint / 1024^3
        budget = MAXRSS_BUDGET_GB === nothing ? "none" : "$(MAXRSS_BUDGET_GB) GB"
        println(@sprintf("memory: total %.2f GB, heap-size-hint %.2f GB", total_gb, hint_gb),
            ", maxrss budget ", budget)
    end
    flush(stdout)
end

# Prints the start/end trace lines around one `Base.include(Main, path)` call, when active.
# Broken out so the `include` override below (installed once, outside this module) is a
# one-line forward into it.
#
# Format: `▸ path` before the file, then `✓ path  time  peak (+growth)` after it, with the
# columns aligned and nested includes indented two spaces per level. The growth is how far
# the process peak rose during this file, which is what tells files apart. The compile time
# is the JIT time spent inside the file, nested files included, summed over threads.
# `test/timing.jl` parses these lines.
const TRACE_DEPTH = Ref(0)

# The files being included, innermost last, each with the testset that was current when it
# began: a testset recorded into that testset is a top-level testset of the file, whatever
# wrapper testsets enclose the include. `test/timing.jl` reads it. Only kept while tracing.
const INCLUDE_STACK = Tuple{String, Any}[]

# `path` as `include` resolves it (against the including file's directory), shown relative to
# the test directory, so `form/matrix_free.jl` and `ghost/matrix_free.jl` are told apart.
function trace_path(path)
    full = normpath(isabspath(path) ? path : joinpath(Base.source_dir(), path))
    root = @__DIR__
    return startswith(full, root * Base.Filesystem.path_separator) ? relpath(full, root) :
           full
end

# The run's progress, `[k/N]` on each test file's start line: `N` counts the leaf files (those
# that include no other) the suite will reach, found once by walking runtests.jl's syntax and
# evaluating its `if`, `&&` and `||` conditions in `Main`; a condition that cannot be
# evaluated there counts as true. Only the shards and groups decide the count, so it is exact
# unless a file is included from a loop or a computed path.
const TRACE_DONE = Ref(0)
const TRACE_LEAVES = Ref{Union{Nothing, Set{String}}}(nothing)

function _trace_cond(ex)
    return try
        Core.eval(Main, ex) === true
    catch
        true
    end
end

function _trace_walk!(leaves, file)
    ex = try
        Meta.parseall(read(file, String); filename = file)
    catch
        return false
    end
    found = Ref(false)
    function walk(e)
        e isa Expr || return
        if e.head === :call && e.args[1] === :include && length(e.args) == 2 &&
           e.args[2] isa String
            found[] = true
            child = normpath(joinpath(dirname(file), e.args[2]))
            isfile(child) && !_trace_walk!(leaves, child) && push!(leaves, child)
        elseif e.head === :if || e.head === :elseif
            _trace_cond(e.args[1]) ? walk(e.args[2]) : length(e.args) > 2 && walk(e.args[3])
        elseif e.head === :&&
            _trace_cond(e.args[1]) && walk(e.args[2])
        elseif e.head === :||
            _trace_cond(e.args[1]) || walk(e.args[2])
        elseif e.head !== :function && e.head !== :macro
            foreach(walk, e.args)
        end
    end
    walk(ex)
    return found[]
end

function _trace_leaves()
    if TRACE_LEAVES[] === nothing
        leaves = Set{String}()
        _trace_walk!(leaves, joinpath(@__DIR__, "runtests.jl"))
        TRACE_LEAVES[] = leaves
    end
    return TRACE_LEAVES[]
end

function traced_include(real_include::F, path) where {F}
    TRACE_TESTS || return real_include(Main, path)
    indent = "  "^TRACE_DEPTH[]
    rel = trace_path(path)
    leaves = _trace_leaves()
    full = normpath(isabspath(path) ? path : joinpath(Base.source_dir(), path))
    progress = full in leaves ? "  [$(TRACE_DONE[] += 1)/$(length(leaves))]" : ""
    println("▸ ", indent, rel, progress)
    flush(stdout)
    rss0_gb = Sys.maxrss() / 1024^3
    compile0 = Base.cumulative_compile_time_ns()[1]
    t0 = time()
    TRACE_DEPTH[] += 1
    push!(INCLUDE_STACK, (rel, Test.get_testset()))
    result = try
        real_include(Main, path)
    finally
        pop!(INCLUDE_STACK)
        TRACE_DEPTH[] -= 1
    end
    elapsed = time() - t0
    compile_s = (Base.cumulative_compile_time_ns()[1] - compile0) / 1e9
    maxrss_gb = Sys.maxrss() / 1024^3
    println(
        "✓ ", rpad(indent * rel, 52),
        @sprintf("%8.2f s %6.2f GB (+%.2f) %7.2f s compile",
            elapsed, maxrss_gb, maxrss_gb - rss0_gb, compile_s)
    )
    flush(stdout)
    # A named, failing `@test` beats a SIGKILL with no indication which file was running --
    # the whole reason the trace above exists (see its comment). A `@testset` whose
    # description carries the message is the only way to attach one to a `@test` failure:
    # the `@test` macro itself takes no message argument.
    if MAXRSS_BUDGET_GB !== nothing && maxrss_gb > MAXRSS_BUDGET_GB
        @testset "maxrss $(round(maxrss_gb; digits = 2)) GB > $(MAXRSS_BUDGET_GB) GB: $path" begin
            @test false
        end
    end
    return result
end

@inline function alloc_test(f::F, args...; kwargs...) where {F}
    f(args...; kwargs...) # warm up
    return @allocated(f(args...; kwargs...))
end

# The allocation and bitwise guarantees are stated at `--optimize=1`. At `-O0` nothing is
# inlined, so a wrapper that is free at `-O1` boxes a few hundred bytes (a Kronecker `mul!`
# measured 528 B), and two kernels that round alike from `-O1` up can differ in the last bit
# (`normₕ(u)` against `sqrt(innerₕ(u, u))`: no fused multiply-add, no SIMD reduction).
# `@test alloc_ok(bytes)` and `@test bits_eq(a, b)` hold at `-O0` up to that and demand 0 B
# and bitwise equality from `-O1` up, so the suite can run at `-O0`, about twice as fast as
# at `-O1` (the run is compile-bound).
const OPTIMIZED = Base.JLOptions().opt_level >= 1
alloc_ok(bytes::Integer) = !OPTIMIZED || bytes == 0
bits_eq(a::T, b::T) where {T <: AbstractFloat} = OPTIMIZED ? a === b : isapprox(a, b; rtol = 4eps(T))

# Allocation test helper: uses a function barrier to avoid @testset closure boxing.
#
# These run under code coverage as well. They used to skip there, on the assumption that
# the instrumentation would perturb the counts, and that assumption cost the suite its
# allocation guarantees exactly where they were most useful: CI runs with coverage, so
# every one of them was skipped there and only ever checked by hand locally.
#
# Measured on Julia 1.12 with --code-coverage=user, the counts are identical either way,
# and the whole suite passes with 0 failures. If a future Julia does perturb them, these
# will fail rather than quietly not run, which is the outcome to prefer: the boxing
# regressions this suite exists to catch are invisible to both JET's optimisation analysis
# and AllocCheck (a reproduction of the original bug allocating 23,824 B against 0 B for
# the fix draws no report from either), so a runtime count is the only thing that sees
# them.
macro test_allocs(call_expr)
    if Meta.isexpr(call_expr, :call)
        fn = call_expr.args[1]
        args = call_expr.args[2:end]
        quote
            @test alloc_test($(esc(fn)), $(map(esc, args)...)) == 0
        end
    elseif Meta.isexpr(call_expr, :ref)
        target = call_expr.args[1]
        indices = call_expr.args[2:end]
        quote
            @test alloc_test(getindex, $(esc(target)), $(map(esc, indices)...)) == 0
        end
    else
        quote
            let
                $(esc(call_expr))
                @test (@allocated $(esc(call_expr))) == 0
            end
        end
    end
end

# Two comparison helpers, shared by the files that need them rather than defined in each.

# Central difference of a scalar functional, to compare an AD derivative against. Every
# AD test checks the derivative against this rather than merely checking that it ran.
_fd(f, a; h = 1e-6) = (f(a + h) - f(a - h)) / (2h)

# Is a package resolvable from this environment? The files behind the `ad` and `ext`
# groups use it to `@test_skip` rather than error on a backend the environment does not
# have. It was defined five times over -- once per such file, plus once in
# space/autodiff_backends.jl, which autodiff_heavy.jl then imported by name, coupling the
# two files' include order to a one-line predicate.
_have(mod::Symbol) = Base.identify_package(String(mod)) !== nothing

# Should a GPU-backed testset actually run its device kernels? test/ext/metal_ext.jl and
# metal_fullstack.jl used to gate solely on `Metal.functional()`, on the assumption that a
# CI runner has no working device -- true for a nested VM, but not for GitHub's hosted
# macOS runners, which are real Apple Silicon hardware and expose a functional Metal device
# for headless compute. Weekly.yml's `backends` group therefore risks actually executing GPU
# kernels, unattended, on a shared CI runner. `_run_gpu_tests()` adds an explicit opt-out on
# top of `Metal.functional()`: `BRAMBLE_SKIP_GPU_TESTS=true` (set by intent) or `CI=true`
# (GitHub Actions sets this on every runner) forces a skip regardless of what the host
# reports, so the GPU path only ever runs where a maintainer runs it by hand.
_run_gpu_tests() = get(ENV, "BRAMBLE_SKIP_GPU_TESTS", "false") != "true" &&
                   get(ENV, "CI", "false") != "true"

# Refines a manufactured problem and checks it converges at second order. `errfn(n)`
# returns `(error, spacing)` for an n-point grid; the observed order between consecutive
# refinements must clear `order`, and the errors themselves must fall monotonically -- a
# rate alone can look right while the errors sit on a plateau. Returns the observed orders
# so a caller can pin the finest one further.
#
# This sweep was written out five times (four in ext/sciml_ext.jl, once in
# form/semidiscrete.jl), differing only in the closure it measured.
function _check_eoc(errfn, ns; order = 1.9)
    errors = Float64[]
    spacings = Float64[]
    for n in ns
        e, h = errfn(n)
        push!(errors, e)
        push!(spacings, h)
    end
    eoc = [log(errors[i] / errors[i + 1]) / log(spacings[i] / spacings[i + 1])
           for i in 1:(length(errors) - 1)]
    @test all(>(order), eoc)
    @test issorted(errors; rev = true)
    return eoc
end

# The printed tables of observed orders: `_order_header` once per study, then `_order_row`
# once per case, so a nightly log reads as aligned columns. `cells` label the case, one per
# entry of `widths`; `gate` is the order the case must clear, or `nothing` for a case that is
# only printed; the errors print as `%.2e`, one column per level.
function _order_header(title, labels, widths)
    println("\n", title)
    println("  ", join(rpad.(labels, widths)), rpad("order", 7), rpad("gate", 9), "errors")
    return nothing
end

function _order_row(cells, order, errs, widths; gate = nothing)
    mark = gate === nothing ? "—" : (order > gate ? "> $gate ✓" : "> $gate ✗")
    println("  ", join(rpad.(string.(cells), widths)), rpad(@sprintf("%.3f", order), 7),
        rpad(mark, 9), join((@sprintf("%.2e", e) for e in errs), "  "))
    return nothing
end

# The heat equation `uₜ = uₓₓ + f` on (0, 1) with the manufactured solution
# `exp(-t) sin(πx)` and zero Dirichlet data. form/semidiscrete.jl and ext/sciml_ext.jl check
# the semidiscretisation on it; test/convergence/heat_1d.jl measures its orders.
_heat_uex(x, t) = exp(-t) * sinpi(x[1])
_heat_src(x, t) = (pi^2 - 1) * exp(-t) * sinpi(x[1])

# Its forms on an `n`-point uniform mesh, the source a live element `fₕ`.
function _sd_problem(n)
    Ωₕ = mesh(domain(interval(0.0, 1.0)), n)
    Wₕ = gridspace(Ωₕ)
    fₕ = element(Wₕ, 0.0)
    a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
    l = form(Wₕ, v -> innerₕ(fₕ, v))
    return Ωₕ, Wₕ, fₕ, a, l
end

# The same, semidiscretised on [0, T], `fₕ` refilled from `t` before each assembly.
function _sciml_setup(n; T = 1.0)
    Ωₕ, Wₕ, fₕ, a, l = _sd_problem(n)
    I = interval(0.0, T)
    bcs = Bramble.dirichlet_constraints(Ωₕ, I, :boundary => (x, t) -> 0.0)
    sd = Bramble.semidiscretize(
        a, l; dirichlet = bcs, (update_coefficients!) = t -> Bramble.Rₕ!(fₕ, x -> _heat_src(x, t))
    )
    return Ωₕ, Wₕ, I, a, l, sd
end

# Error of `op` against the exact derivative `df`, over the points whose stencil is not
# truncated. The max norm is used so the result does not depend on the quadrature weights.
function _interior_error(Ωₕ, op, f, df, drop)
    Wₕ = gridspace(Ωₕ)
    e = parent(op(Rₕ(Wₕ, f))) .- parent(Rₕ(Wₕ, df))
    dims = npoints(Ωₕ, Tuple)
    return maximum(abs, drop(reshape(e, dims)))
end

# Successive halvings of the mesh give log2 of the error ratio as the observed order. The
# caller owns the mesh, the seed and the number of `steps`; `drop` selects the interior the
# error is measured on. Returns the per-step ratios alongside the raw errors, so a caller
# can also fit a slope across every level rather than only reading the last pair. Mutates
# `Ωₕ`, which ends refined `steps` times.
function _observed_orders(Ωₕ, op, f, df, drop; steps = 4)
    errs = Float64[]
    for k in 0:steps
        k > 0 && iterative_refinement!(Ωₕ)
        push!(errs, _interior_error(Ωₕ, op, f, df, drop))
    end
    ords = [log2(errs[k] / errs[k + 1]) for k in 1:(length(errs) - 1)]
    return ords, errs
end

# The marker-mask oracle: the predicate `pred` evaluated at every grid point of `Ωₕ`, in
# index order. `index_in_marker(Ωₕ, label)` is compared against it, with `pred` the function
# the marker was declared with, so the check does not depend on where the points fall.
function _marker_mask(Ωₕ, pred)
    return BitVector(pred(point(Ωₕ, I)) for I in vec(indices(Ωₕ)))
end

# A symmetric, structurally symmetric operator to constrain.
_tri(m) = spdiagm(0 => fill(4.0, m), 1 => fill(-1.0, m - 1), -1 => fill(-1.0, m - 1))

# `assemble` and `allocate_system_matrix` infer a union that includes a dense `Matrix`, which
# has no `nonzeros`; the matrices filled here are always `SparseMatrixCSC`, so the assertion
# narrows the type for JET.
function _fillnz!(A, v)
    @assert A isa SparseMatrixCSC
    return fill!(nonzeros(A), v)
end

# A grid function with standard normal entries, from the global RNG.
_random_element(Wₕ) = (uₕ = element(Wₕ); parent(uₕ) .= Random.randn(length(parent(uₕ))); uₕ)

# A box mesh of `n` points on [0, 1]^D on `backend`: uniform, then moved by `change_points!`
# to `t^(1 + d/4)` along axis `d` (as benchmark/operator_routes.jl builds one), so no two
# axes share their nodes.
function _graded_mesh(n::NTuple{D, Int}; backend = backend()) where {D}
    Ωₕ = mesh(domain(reduce(×, ntuple(_ -> interval(0.0, 1.0), D))), n, ntuple(_ -> true, D);
        backend = backend)
    change_points!(Ωₕ, ntuple(d -> range(0.0, 1.0; length = n[d]) .^ (1 + 0.25d), D))
    return Ωₕ
end

# The Poisson fixture the ext solver files share: the unit `D`-cube, its sine source and a
# grid of `n` points per axis on `backend`.
_unit_cube(::Val{D}) where {D} = reduce(×, ntuple(_ -> interval(0.0, 1.0), Val(D)))
_sine_source(::Val{1}) = x -> sin(π * x)
_sine_source(::Val{D}) where {D} = x -> prod(sin(π * xᵢ) for xᵢ in x)

_grid(::Val{1}, Ωd, n; backend = backend()) = mesh(Ωd, n, true; backend = backend)
function _grid(::Val{D}, Ωd, n; backend = backend()) where {D}
    return mesh(
        Ωd, ntuple(_ -> n, Val(D)), ntuple(_ -> true, Val(D)); backend = backend
    )
end

# The points of an arbitrary non-uniform partition of [0, 1], from a vector of positive
# step sizes: cumulative sums, normalised by the last one. Every Supposition check of a
# discrete integration-by-parts identity builds its mesh this way
# (space/star_difference.jl, space/centered_difference.jl, space/sbp_identities.jl), so
# the construction lives here rather than once per file.
function _nonuniform_points(h::AbstractVector{<:Real})
    pts = zeros(Float64, length(h) + 1)
    for (i, hᵢ) in enumerate(h)
        pts[i + 1] = pts[i] + hᵢ
    end
    pts ./= pts[end]
    return pts
end

# Zeroes the boundary planes of a field laid out on the mesh's point grid, along every
# direction, which is what puts it in V_{H,0} (homogeneous Dirichlet). Works in 1D, 2D and
# 3D through `selectdim`, so the identity checks do not spell the slices out per dimension.
function _zero_boundary!(a::AbstractArray{<:Real, N}) where {N}
    for d in 1:N
        selectdim(a, d, 1) .= 0
        selectdim(a, d, size(a, d)) .= 0
    end
    return a
end

# A field on the unit domain vanishing on every boundary plane, from a raw draw: the first
# `prod(dims)` entries of `raw`, laid out on the `dims` point grid with the boundary zeroed.
function _boundary_vanishing(Wₕ, raw, dims)
    a = reshape(copy(raw[1:prod(dims)]), dims)
    return element(Wₕ, vec(_zero_boundary!(a)))
end

# Half-cell width at node `i` of the points `x`, from the points alone: the oracle against
# which half spacings and weights on non-uniform meshes are checked, so it must not call the
# code under test.
function _half_spacing_oracle(x, i)
    i == 1 ? (x[2] - x[1]) / 2 :
    i == length(x) ? (x[end] - x[end - 1]) / 2 : (x[i + 1] - x[i - 1]) / 2
end

# Backward spacing at node `i` of the points `x`, zero at the first node (the weight of a
# staggered axis).
_backward_spacing_oracle(x, i) = i == 1 ? 0.0 : x[i] - x[i - 1]

# Random non-uniform meshes on the unit cube in 1D, 2D and 3D, and one smooth field per
# spatial dimension on each. Returns the mesh, its grid space, the fields and the point
# counts. The mesh draws from the global RNG, so the caller seeds it.
function _fixture(D)
    dom = D == 1 ? domain(interval(0.0, 1.0)) :
          D == 2 ? domain(interval(0.0, 1.0) × interval(0.0, 1.0)) :
          domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    n = (11, 9, 7)
    Ωₕ = D == 1 ? mesh(dom, n[1], false) : mesh(dom, n[1:D], ntuple(_ -> false, D))
    Wₕ = gridspace(Ωₕ)
    fs = (x -> sin(3x[1]) + (D > 1 ? x[2]^2 : 0.0) + (D > 2 ? x[3] : 0.0),
        x -> cos(2x[D]) + x[1]^3,
        x -> x[1] * x[D] + sin(x[1]))
    u = ntuple(k -> Rₕ(Wₕ, D == 1 ? (x -> fs[k]((x,))) : fs[k]), D)
    return Ωₕ, Wₕ, u, npoints(Ωₕ, Tuple)
end

# The same field, spelled as a `D`-leaf composite grid function.
function _composite(Ωₕ, u, D)
    uc = element(gridspace(Ωₕ, Val(D)), 0.0)
    for d in 1:D
        parent(components(uc)[d]) .= parent(u[d])
    end
    return uc
end

# `u` with its boundary planes zeroed, on the `dims` point grid.
function _bubble(u, dims)
    w = copy(u)
    _zero_boundary!(reshape(parent(w), dims))
    return w
end

_field(u, D) = D == 1 ? u[1] : u

# `f` must be a scalar functional of one parameter, evaluated through the library. Checks
# that the AD derivative is right, not merely that it ran. Was `_matches_finite_difference`
# in space/autodiff.jl and `_matches_fd` in form/autodiff.jl (same body, two names, so no
# overwrite warning pointed at it).
function _matches_fd(f, a = 1.3; rtol = 1e-5)
    return isapprox(ForwardDiff.derivative(f, a), _fd(f, a); rtol = rtol)
end

# --- Thread spy ------------------------------------------------------------------------ #
#
# A storage vector recording which threads read it: proves a banded or batched path ran on
# several threads, rather than trusting the dispatch. It was defined six times (the threaded
# stencil, vector-calculus, broadcast and nested-threading tests, and twice in
# ext/polyester_ext.jl), one per file, each with its own global counter. A test calls
# `_reset_seen!()` right before the call it measures and reads `_threads_seen()` after it:
# the counter is shared, so the reset has to stay at every site.
const _SEEN = Threads.Atomic{UInt64}(0)
struct _Spy{T} <: AbstractVector{T}
    x::Vector{T}
end
Base.size(s::_Spy) = size(s.x)
Base.IndexStyle(::Type{<:_Spy}) = IndexLinear()
Base.@propagate_inbounds function Base.getindex(s::_Spy, i::Int)
    Threads.atomic_or!(_SEEN, UInt64(1) << ((Threads.threadid() - 1) % 64))
    return s.x[i]
end

# A grid function whose storage is a spy over a copy of `u`'s.
_spy(u) = VectorElement(_Spy(copy(parent(u))), space(u))

_reset_seen!() = (_SEEN[] = 0)

# How many distinct threads have read a spy since the last reset.
_threads_seen() = count_ones(_SEEN[])

# --- Device-array mock ------------------------------------------------------------------ #

# Minimal DenseArray mock simulating a vendor GPU array type (MtlArray, CuArray) to check
# generic backend dispatch with no GPU hardware or optional dependency. It answers
# `DeviceLocality()` although the storage underneath is a plain host `Array`: that is what
# makes the `Backend` constructor's locality rejection, the offloaded projection path
# (`GpuOffload`) and the sweep guard testable on a host. It claims device locality the same
# way a real vendor array would, so pairing it with a CpuPolicy or a host matrix type must be
# refused exactly as it would be for MtlVector/MtlMatrix. Defined once for every file that
# needs one, so the `Bramble.locality` method is added once too.
struct MockDeviceArray{T, N} <: DenseArray{T, N}
    data::Array{T, N}
end
function MockDeviceArray{T, N}(::UndefInitializer, dims::Vararg{Integer, N}) where {T, N}
    return MockDeviceArray(Array{T, N}(undef, dims...))
end
function MockDeviceArray{T, N}(::UndefInitializer, dims::NTuple{N, Integer}) where {T, N}
    return MockDeviceArray(Array{T, N}(undef, dims))
end
Base.size(A::MockDeviceArray) = size(A.data)
Base.getindex(A::MockDeviceArray, i::Int...) = getindex(A.data, i...)
Base.setindex!(A::MockDeviceArray, v, i::Int...) = setindex!(A.data, v, i...)
Base.IndexStyle(::Type{<:MockDeviceArray}) = IndexLinear()
Base.fill!(A::MockDeviceArray{T}, v) where {T} = (fill!(A.data, v); A)
Bramble.locality(::Type{<:MockDeviceArray}) = Bramble.DeviceLocality()

const MockDeviceVector{T} = MockDeviceArray{T, 1}
const MockDeviceMatrix{T} = MockDeviceArray{T, 2}

# --- Zero-based vector ------------------------------------------------------------------ #

# A `Float64` vector indexed from 0, standing in for an `OffsetVector`: the transfers,
# smoothers, preconditioners and the Kronecker products must refuse offset axes, and this is
# the smallest array that has them. `IdentityUnitRange` keeps the axes a valid index set for
# `Base.require_one_based_indexing` to reject.
struct ZeroBasedVector <: AbstractVector{Float64}
    p::Vector{Float64}
end
Base.size(z::ZeroBasedVector) = size(z.p)
Base.axes(z::ZeroBasedVector) = (Base.IdentityUnitRange(0:(length(z.p) - 1)),)
Base.getindex(z::ZeroBasedVector, i::Int) = z.p[i + 1]
Base.setindex!(z::ZeroBasedVector, v, i::Int) = (z.p[i + 1] = v)

# --- Multigrid fixtures ----------------------------------------------------------------- #
#
# Shared by test/solvers/multigrid.jl and the Polyester multigrid testset of
# test/ext/polyester_ext.jl, which runs the same meshes and the same form under CpuPolyester
# and compares with Serial. Non-uniform throughout: on a uniform mesh rebuilding each level
# from the domain would nest too, and hide a hierarchy that does not take every other point.
const MG_SEED = 3291

# Non-uniform meshes in 1D, 2D and 3D, and two with a collapsed axis (which also cover
# `interpolation_matrix` on collapsed axes), with a level count each. Without
# `WITH_3D_TESTS` the two 3D entries are not built, so the tuple is (1D, 2D, collapsed 2D):
# a caller that indexes it past the second entry gates itself on `WITH_3D_TESTS`.
function _mg_transfer_meshes(bk = backend())
    Random.seed!(MG_SEED)
    unit(a = 0.0, b = 1.0) = interval(a, b)
    return (
        (mesh(domain(unit()), 33, false; backend = bk), 4),
        (mesh(domain(unit() × unit(0.0, 2.0)), (17, 9), false; backend = bk), 3),
        (WITH_3D_TESTS ?
         ((mesh(domain(unit() × unit(-1.0, 1.0) × unit(0.0, 2.0)), (9, 5, 9), false; backend = bk), 3),) :
         ())...,
        (mesh(domain(unit() × unit(0.5, 0.5)), (17, 4), false; backend = bk), 3),
        (WITH_3D_TESTS ?
         ((mesh(domain(unit() × unit(0.5, 0.5) × unit()), (9, 4, 5), false; backend = bk), 2),) :
         ())...
    )
end

# Uniform points jittered by up to ±0.3h along each axis: non-uniform everywhere, with
# bounded cell aspect ratio.
function _mg_jitter_mesh(D, n; bk = backend(), seed = MG_SEED)
    rng = Xoshiro(seed)
    Ω = mesh(domain(reduce(×, ntuple(_ -> interval(0.0, 1.0), D))), ntuple(_ -> n, D),
        ntuple(_ -> true, D); backend = bk)
    h = 1 / (n - 1)
    function pts()
        x = collect(range(0.0, 1.0; length = n)) .+ 0.3h .* (2 .* rand(rng, n) .- 1)
        x[1], x[end] = 0.0, 1.0
        return sort!(x)
    end
    change_points!(Ω, ntuple(_ -> pts(), D))
    return Ω
end

# Mass plus variable diffusion, κ = 1 + |x|²: symmetric positive definite with natural
# boundary conditions.
_mg_spd(W) = (κ = Rₕ(W, x -> 1 + sum(abs2, x)); form(W, W, (u, v) -> innerₕ(u, v) + inner₊(κ * ∇ₕ(u), ∇ₕ(v))))

# --- Threaded stencil fixtures ----------------------------------------------------------- #
#
# Under a threaded policy every CPU stencil engine (the one-sided and centered difference
# engines and both average engines) runs banded along the grid's last axis, one band per
# thread or `@batch` task. Every point is still computed by the very loop body the serial
# engine runs, so the answer must equal the `Serial()` one exactly, not merely to a
# tolerance. The meshes are non-uniform: on a uniform mesh a band that picked up the wrong
# spacing index would still give the right number. test/space/threaded_stencils.jl runs this
# under `Parallel()` and test/ext/polyester_ext.jl under `CpuPolyester()`.

# Every family reaching `_apply_stencil!` or `_apply_averaged!`, spelled from the operator's
# base name so no Unicode is retyped here.
const _STENCIL_FAMILIES = (:D₋, :D₊, :diff₋, :diff₊, :jump, :Dc, :D̃, :D̽, :M, :M₊, :Mc)
# What `unit` runs of them: one per engine (one-sided difference, jump, centered difference,
# average, centered average); `slow` runs them all.
const _STENCIL_UNIT_FAMILIES = (:D₋, :diff₊, :jump, :Dc, :M₊, :Mc)
const _STENCIL_CENTERED = (:Dc, :D̽, :Mc)   # need three points along their direction
const _STENCIL_SUFFIXES = ("ₓ", "ᵧ", "₂")

_stencil_op(fam, d) = getproperty(Bramble, Symbol(fam, _STENCIL_SUFFIXES[d]))
_stencil_op!(fam, d) = getproperty(Bramble, Symbol(fam, _STENCIL_SUFFIXES[d], :!))

function _stencil_domain(D)
    D == 1 ? domain(interval(0.0, 1.0)) :
    D == 2 ? domain(interval(0.0, 1.0) × interval(0.0, 1.0)) :
    domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
end

# The same non-uniform mesh twice, once per policy: the seed fixes the random points.
function _stencil_mesh_pair(n::NTuple{D, Int}, policy; seed = 356) where {D}
    dom = _stencil_domain(D)
    npts = D == 1 ? n[1] : n
    unif = D == 1 ? false : ntuple(_ -> false, D)
    Random.seed!(seed)
    Ωs = mesh(dom, npts, unif; backend = backend(policy = Serial()))
    Random.seed!(seed)
    Ωp = mesh(dom, npts, unif; backend = backend(policy = policy))
    return Ωs, Ωp
end

const _STENCIL_F = (x -> sin(3x) + x^2, x -> sin(3x[1] + 2x[2]) + x[1] * x[2],
    x -> sin(3x[1] + 2x[2] - x[3]) + x[1] * x[3])
const _STENCIL_G = (x -> cos(2x), x -> exp(x[1]) * x[2], x -> x[1] + x[2]^2 * x[3])

# Compare every applicable family and direction, in place and allocating, scalar and
# composite, between `Serial()` and `policy`. `full = false` keeps one family per engine and
# leaves the 3D composite out, the costliest to compile; it is what `unit` runs.
function _check_stencils(n::NTuple{D, Int}, policy; full::Bool = WITH_SLOW_TESTS) where {D}
    Ωs, Ωp = _stencil_mesh_pair(n, policy)
    Ws, Wp = gridspace(Ωs), gridspace(Ωp)
    Vs, Vp = gridspace(Ωs, Val(2)), gridspace(Ωp, Val(2))
    us, up = Rₕ(Ws, _STENCIL_F[D]), Rₕ(Wp, _STENCIL_F[D])
    vs, vp = Rₕ(Vs, (_STENCIL_F[D], _STENCIL_G[D])), Rₕ(Vp, (_STENCIL_F[D], _STENCIL_G[D]))
    @test parent(us) == parent(up)
    for d in 1:D, fam in (full ? _STENCIL_FAMILIES : _STENCIL_UNIT_FAMILIES)

        fam in _STENCIL_CENTERED && n[d] < 3 && continue
        f, f! = _stencil_op(fam, d), _stencil_op!(fam, d)
        @testset "$(fam)$(_STENCIL_SUFFIXES[d]) n=$n" begin
            ws, wp = similar(us), similar(up)
            parent(wp) .= NaN               # every point must be written
            f!(ws, us)
            @test f!(wp, up) === wp
            @test parent(wp) == parent(ws)
            @test parent(f(up)) == parent(f(us))

            # The composite dispatches the same banded engines whatever the dimension.
            if full || D < 3
                ws2, wp2 = similar(vs), similar(vp)
                f!(ws2, vs)
                f!(wp2, vp)
                @test parent(wp2) == parent(ws2)
                @test parent(f(vp)) == parent(f(vs))
            end
        end
    end
end

# --- Threaded broadcast fixtures --------------------------------------------------------- #
#
# `dest .= expr` into a `VectorElement` runs in static bands of the destination's storage,
# one per thread or `@batch` task. Every point runs the very loop body the serial broadcast
# runs, so the answer must equal the `Serial()` one exactly, including when `dest` itself
# appears on the right-hand side. The meshes are non-uniform, so the operands differ from
# point to point in a way a uniform mesh would not show. test/space/threaded_broadcast.jl
# runs this under `Parallel()` and test/ext/polyester_ext.jl under `CpuPolyester()`.

function _broadcast_domain(D)
    D == 1 ? domain(interval(0.0, 1.0)) :
    D == 2 ? domain(interval(0.0, 1.0) × interval(0.0, 2.0)) :
    domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
end

# The same non-uniform mesh under the given policy: the seed fixes the random points.
function _broadcast_space(n::NTuple{D, Int}, policy; seed = 357) where {D}
    Random.seed!(seed)
    npts = D == 1 ? n[1] : n
    unif = D == 1 ? false : ntuple(_ -> false, D)
    return gridspace(mesh(_broadcast_domain(D), npts, unif; backend = backend(policy = policy)))
end

const _BROADCAST_SIZES = (
    (1,), (2,), (7,), (1001,), (5, 3), (40, 37), (WITH_3D_TESTS ? ((4, 3, 5), (13, 11, 9)) : ())...)

# Every broadcast shape, each writing into a fresh `NaN` destination (or
# updating a copy in place): VectorElements only, a plain vector, literal scalars, a `Ref`
# and a runtime `Float64`, and `dest` on its own right-hand side.
function _broadcast_results(n, policy)
    Wₕ = _broadcast_space(n, policy)
    uₕ = Rₕ(Wₕ, x -> sin(3sum(x)) + prod(x))
    wₕ = Rₕ(Wₕ, x -> exp(first(x)) * last(x))
    plain = [cos(0.3i) for i in eachindex(parent(uₕ))]
    r = Ref(0.25)
    α = 1.5
    fresh() = (v = similar(uₕ); parent(v) .= NaN; v)
    out = Dict{String, Vector{Float64}}()

    v = fresh()
    v .= 2.0 .* uₕ .+ wₕ
    out["axpy"] = copy(parent(v))
    v = fresh()
    v .= uₕ .* plain .- r[] .* wₕ .+ 1
    out["mixed"] = copy(parent(v))
    v = fresh()
    v .= α .* sin.(uₕ) ./ (1 .+ wₕ .^ 2)
    out["nested"] = copy(parent(v))
    v = fresh()
    v .= r
    out["fill"] = copy(parent(v))
    v = fresh()
    v .= uₕ
    out["copy"] = copy(parent(v))
    a = copy(uₕ)
    a .= a .+ 0.5 .* wₕ
    out["self"] = copy(parent(a))
    a = copy(uₕ)
    a .= wₕ .- a .* a
    out["self twice"] = copy(parent(a))
    a = copy(uₕ)
    a .*= α
    out["scale"] = copy(parent(a))
    return out
end

function _check_broadcast_equal(n, policy)
    s, p = _broadcast_results(n, Serial()), _broadcast_results(n, policy)
    for key in keys(s)
        @test p[key] == s[key]
    end
end

# Runs one worked-example page. The pages under docs/src/examples/ are Literate scripts:
# docs/make.jl renders each to the markdown Documenter publishes, and the suite runs the same
# file, so the numbers a reader sees are the numbers asserted here. Their `#src` lines are
# those assertions -- stripped on the way to the page, executed from here.
#
# Each page gets its own module rather than sharing `Main`: the pages write the bare
# `domain`/`mesh`/`element` a reader would type, they define names like `sol` and `residual`
# that would collide across pages, and the ext group shares one `Main` with Meshes.jl, which
# exports three of those names itself. Anchored to `Main` explicitly rather than the calling
# module, since the caller is now its own per-file test module rather than `Main` itself.
function _run_example_page(name::Symbol)
    path = joinpath(@__DIR__, "..", "docs", "src", "examples", string(name, ".jl"))
    @test isfile(path)
    @eval Main module $(Symbol(:Page_, name))
    using Test
    include($path)
    end
    return nothing
end

# --- One problem, a plain and a ghost space ---------------------------------------------- #

# A system the shared bodies of test/form/two_spaces.jl run over: the forms `a`, `l`, how
# the Dirichlet data enter (`route`), and where the Dirichlet nodes sit among the nodes in
# mesh order (`fixed`, with `gfixed` their data; `free` the rest).
struct SpaceCase{S, A, L, T <: Tuple, P, N <: Tuple}
    name::String
    route::Symbol            # :rows (`dirichlet` keyword) or :relations (`u => g`)
    space::S
    a::A
    l::L
    labels::T                # the `dirichlet` labels of the :rows route
    pairs::P                 # `label => g`, for the `dirichlet` keyword of `assemble!`
    free::Vector{Int}
    fixed::Vector{Int}
    gfixed::Vector{Float64}
    dims::N
end

# Nodes of `dims` (mesh order, first axis fastest) lying on one of the named facets.
function _facet_node_mask(dims::NTuple{D, Int}, facets) where {D}
    mask = falses(prod(dims))
    for (k, I) in enumerate(CartesianIndices(dims)), F in facets

        s = String(F)
        d = findfirst(==(s[1]), "xyz")
        I[d] == (endswith(s, "min") ? 1 : dims[d]) && (mask[k] = true)
    end
    return mask
end

"""
    plain_and_ghost_spaces(Ωₕ, data) -> (SpaceCase, SpaceCase)

The one problem `-Δu = f` with `u = g` on the facets `data.dirichlet`, on a plain and on a
ghost space, for the shared bodies to run over.

`data` is a `NamedTuple` with `f`, `g` and `h(facet, p)` (functions of the point `p`),
`dirichlet` (facet names such as `:xmin`) and optionally `robin`, a tuple of
`(facet, α, β, r)` for `α u + β ∂ₙ(u) = r` on the remaining facets (`α = 0` is Neumann).

  - Without `robin` the first case is `gridspace(Ωₕ)` with row Dirichlet on `:boundary`.
    With it, it is the ghost space of the Robin facets only, the Dirichlet rows through the
    `dirichlet` keyword.
  - The second case ghosts every facet: `u => g` and `∂ₙₙ(u) => h` on the Dirichlet facets,
    the Robin relations on the others, so its unknowns are the free nodes only.

Where the two should agree is `A[free, free]` and `F[free]` of the first, symmetrized, with
the second's system (departure 8 of the v4.0.0-rest plan).
"""
function plain_and_ghost_spaces(Ωₕ, data)
    dims = npoints(Ωₕ, Tuple)
    D = length(dims)
    ax = D == 1 ? (Bramble.points(Ωₕ),) : Bramble.points(Ωₕ)
    pts = [ntuple(d -> ax[d][I[d]], D) for I in CartesianIndices(dims)]
    mask = _facet_node_mask(dims, data.dirichlet)
    free, fixed = findall(!, mask), findall(mask)
    gfixed = Float64[data.g(pts[k]) for k in fixed]
    robin = get(data, :robin, ())
    lap(u, v) = innerₕ(-1.0 * Bramble.Δₕ(u), v)
    rob(u, (F, α, β, r)) = F => (iszero(α) ? β * Bramble.∂ₙ(u) : α * u + β * Bramble.∂ₙ(u)) => r
    lin(W) = form(W, v -> innerₕ(data.f, v))
    if isempty(robin)
        W = gridspace(Ωₕ)
        labels = (:boundary,)
    else
        Wr = gridspace(Ωₕ; ghosts = Tuple(r[1] for r in robin))
        W = Bramble.constrained(Wr, u -> map(r -> rob(u, r), robin))
        labels = Tuple(data.dirichlet)
    end
    pairs = map(L -> L => data.g, labels)
    plain = SpaceCase("plain space, row Dirichlet", :rows, W, form(W, W, lap), lin(W),
        labels, pairs, free, fixed, gfixed, dims)
    Wg = gridspace(Ωₕ; ghosts = :boundary)
    C = Bramble.constrained(Wg,
        u -> (map(F -> F => u => data.g, data.dirichlet)...,
            map(F -> F => Bramble.∂ₙₙ(u) => (p -> data.h(F, p)), data.dirichlet)...,
            map(r -> rob(u, r), robin)...))
    ghost = SpaceCase("ghost space, u => g and ∂ₙₙ(u) => h", :relations, C,
        form(C, C, lap), lin(C), labels, pairs, free, fixed, gfixed, dims)
    return plain, ghost
end

# --- Meshes and spaces the ghost test files share ------------------------------------ #

"""
    random_mesh(ns; seed = 11, lengths = ones, level = 0)

A seeded random non-uniform mesh of `ns` points per axis on the box `[0, lengths[d]]ᴰ`
(`ns` a tuple, one entry for 1D), halved `level` times by `iterative_refinement!`.
"""
function random_mesh(ns::NTuple{D, Int}; seed = 11, lengths = ntuple(_ -> 1.0, D),
        level = 0) where {D}
    Ωd = domain(foldl(×, ntuple(d -> interval(0.0, Float64(lengths[d])), D)))
    Bramble._seed_mesh1d_rng!(seed)
    Ω = try
        D == 1 ? mesh(Ωd, ns[1], false) : mesh(Ωd, ns, ntuple(_ -> false, D))
    finally
        Bramble._unseed_mesh1d_rng!()
    end
    foreach(_ -> iterative_refinement!(Ω), 1:level)
    return Ω
end

"""
    ghost_mesh(policy, ns; T = Float64, seed = 11)

A seeded random non-uniform mesh on `[0, 1] × [0, 2] × [0, 3]` (its first `length(ns)` axes)
of element type `T`, under the threading `policy`.
"""
function ghost_mesh(pol, ns::NTuple{D, Int}; T = Float64, seed = 11) where {D}
    Bramble._seed_mesh1d_rng!(seed)
    be = backend(vector_type = Vector{T}, matrix_type = SparseMatrixCSC{T, Int},
        policy = pol)
    Ωd = domain(foldl(×, ntuple(d -> interval(T(0), T(d)), D)))
    Ω = mesh(Ωd, D == 1 ? only(ns) : ns, D == 1 ? false : ntuple(_ -> false, D);
        backend = be)
    Bramble._unseed_mesh1d_rng!()
    return Ω
end

# The padded space of `gridspace(Ωₕ; ghosts)`: the ghost nodes as plain unknowns.
padded(Ωₕ; ghosts = :boundary) = Bramble.padded_space(gridspace(Ωₕ; ghosts = ghosts))

# An oracle's per-facet Robin data as `facet => α u + β ∂ₙ(u) => g` pairs.
robin_pairs(u, rels) = map(keys(rels)) do F
    r = rels[F]
    F => r.α * u + r.β * Bramble.∂ₙ(u) => r.g
end

end # module TestUtils

# Installs the per-file trace by overriding Main's own `include`, once, when this file loads
# (guarded, like TestUtils itself, by the `isdefined(Main, :TestUtils) || include(...)` line
# at the top of test/runtests.jl and every test/*/runtests.jl) -- so every already-existing
# `include(path)` call in those files traces automatically, with no call site changed.
#
# `include` in Main is a `const` binding to a callable that forwards to `Base.include(Main,
# path)`; it cannot be redefined with `function include(...)` or a plain assignment (both
# error, the latter suggesting `const`), only reassigned via `const`. Calling
# `Base.include(Main, path)` directly, as the replacement does, resolves `path` exactly like
# a bare `include(path)` would -- relative to whichever file is *currently* being included --
# including through a nested `include` chain (test/runtests.jl -> mesh/runtests.jl ->
# constructors.jl), since that resolution is tracked per-task, not by which callable name
# was used to trigger it.
const include = function (path)
    TestUtils.traced_include(Base.include, path)
end
