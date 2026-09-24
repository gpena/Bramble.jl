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
using SparseArrays: spdiagm
using ForwardDiff

# The test group, read once here rather than in each entry point, so that a subsystem's own
# `runtests.jl` -- which is a standalone entry point and does not go through
# test/runtests.jl -- decides the same way the full suite does. test/runtests.jl reads these
# too; see its own comments for what each group means.
const TEST_GROUP = get(ENV, "BRAMBLE_TEST_GROUP", "all")

# `slow` is the every-push gate's overflow: the unit suite plus the files whose cost is out
# of proportion to what a *push* learns from them. CI.yml (macOS, per push) runs `unit` and
# so skips them; nightly.yml runs `slow` on both platforms once a day, and Weekly.yml's
# `full` includes them as well. A run with no group set gets `all`, which includes them --
# so `.claude/scripts/test.sh` and a bare `Pkg.test` are unaffected.
#
# What is behind it, and why, with the measured cost of each on Julia 1.13, macOS, 4
# threads, against a 6m06s `unit` run:
#
#   examples/pages.jl           1m03.8s  the six worked-example pages, run for the `#src`
#                                        assertions that pin the numbers the docs render.
#                                        Nothing here is a code path that drivers/ and
#                                        form/ do not already cover; what it uniquely
#                                        catches is a *published number* going stale, which
#                                        is a same-day concern, not a same-push one.
#   form/jacobian_pattern.jl      ~46s   the AST-derived sparsity pattern checked against
#                                        SparseConnectivityTracer's AD-traced pattern as
#                                        ground truth in 1D/2D/3D, plus Newton solves that
#                                        use it. The expensive half is a cross-check against
#                                        another package, and it moves when that package or
#                                        the simplifier moves, not when an operator does.
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
const WITH_SLOW_TESTS = TEST_GROUP in ("all", "slow", "full")

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

# Prints the start/end trace lines around one `Base.include(Main, path)` call, when active.
# Broken out so the `include` override below (installed once, outside this module) is a
# one-line forward into it.
function traced_include(real_include::F, path) where {F}
    TRACE_TESTS || return real_include(Main, path)
    println("[trace] start ", path)
    flush(stdout)
    t0 = time()
    result = real_include(Main, path)
    elapsed = time() - t0
    maxrss_gb = Sys.maxrss() / 1024^3
    println(
        "[trace] end ", path,
        " elapsed=", round(elapsed; digits = 2), "s",
        " maxrss=", round(maxrss_gb; digits = 2), "GB"
    )
    flush(stdout)
    return result
end

@inline function alloc_test(f::F, args...; kwargs...) where {F}
    f(args...; kwargs...) # warm up
    return @allocated(f(args...; kwargs...))
end

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
# for headless compute. Weekly.yml's `full` group therefore risks actually executing GPU
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

# A symmetric, structurally symmetric operator to constrain.
_tri(m) = spdiagm(0 => fill(4.0, m), 1 => fill(-1.0, m - 1), -1 => fill(-1.0, m - 1))

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

# `f` must be a scalar functional of one parameter, evaluated through the library. Checks
# that the AD derivative is right, not merely that it ran. Was `_matches_finite_difference`
# in space/autodiff.jl and `_matches_fd` in form/autodiff.jl (same body, two names, so no
# overwrite warning pointed at it).
function _matches_fd(f, a = 1.3; rtol = 1e-5)
    return isapprox(ForwardDiff.derivative(f, a), _fd(f, a); rtol = rtol)
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
