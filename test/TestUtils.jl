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

# `BRAMBLE_TEST_SHARD` narrows the `unit` block to one of three parts, so CI can run each on
# its own runner: `forms-1` and `forms-2` are the two halves of test/form/, `rest` is the
# remainder of the unit block. Unset runs everything, which is what every local run and
# every other group does; the quality, ext, ad, gpu and examples blocks ignore it. The gate
# sits at the include sites in test/runtests.jl and test/form/runtests.jl, through
# `in_shard`.
const TEST_SHARD = get(ENV, "BRAMBLE_TEST_SHARD", "")

const TEST_SHARDS = ("forms-1", "forms-2", "rest")

"""
    in_shard(name)

True when no shard is selected or the selected shard is `name`. Throws an `ArgumentError`
for a `name`, or a `BRAMBLE_TEST_SHARD`, outside `("forms-1", "forms-2", "rest")`, since a
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
const WITH_SLOW_TESTS = TEST_GROUP in ("all", "slow", "full")

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
#     ext/sciml_sensitivity_ext.jl, examples/transient_inverse_problem.jl,
#     ext/chainrules_ext.jl
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

# Turns the trace above into a budget: unset (the default) is today's behaviour, no check at
# all. Set, `Sys.maxrss()` is the process peak (see the comment on `TRACE_TESTS`), so the
# first file whose *end-of-file* peak crosses the budget is the one that fails -- not
# necessarily the file that allocated the most, only the one that tipped the running total
# over. Parsed once here rather than in every call to `traced_include`.
const MAXRSS_BUDGET_GB = let v = get(ENV, "BRAMBLE_TEST_MAXRSS_GB", "")
    isempty(v) ? nothing : parse(Float64, v)
end

# Prints the start/end trace lines around one `Base.include(Main, path)` call, when active.
# Broken out so the `include` override below (installed once, outside this module) is a
# one-line forward into it.
#
# Format: `▸ path` before the file, then `✓ path  time  peak (+growth)` after it, with the
# columns aligned and nested includes indented two spaces per level. The growth is how far
# the process peak rose during this file, which is what tells files apart. `test/timing.jl`
# parses these lines.
const TRACE_DEPTH = Ref(0)

function traced_include(real_include::F, path) where {F}
    TRACE_TESTS || return real_include(Main, path)
    indent = "  "^TRACE_DEPTH[]
    println("▸ ", indent, path)
    flush(stdout)
    rss0_gb = Sys.maxrss() / 1024^3
    t0 = time()
    TRACE_DEPTH[] += 1
    result = try
        real_include(Main, path)
    finally
        TRACE_DEPTH[] -= 1
    end
    elapsed = time() - t0
    maxrss_gb = Sys.maxrss() / 1024^3
    println(
        "✓ ", rpad(indent * path, 52),
        @sprintf("%8.2f s %6.2f GB (+%.2f)", elapsed, maxrss_gb, maxrss_gb - rss0_gb)
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
_half_spacing_oracle(x, i) = i == 1 ? (x[2] - x[1]) / 2 :
                             i == length(x) ? (x[end] - x[end - 1]) / 2 : (x[i + 1] - x[i - 1]) / 2

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
# `interpolation_matrix` on collapsed axes), with a level count each.
function _mg_transfer_meshes(bk = backend())
    Random.seed!(MG_SEED)
    unit(a = 0.0, b = 1.0) = interval(a, b)
    return (
        (mesh(domain(unit()), 33, false; backend = bk), 4),
        (mesh(domain(unit() × unit(0.0, 2.0)), (17, 9), false; backend = bk), 3),
        (mesh(domain(unit() × unit(-1.0, 1.0) × unit(0.0, 2.0)), (9, 5, 9), false; backend = bk), 3),
        (mesh(domain(unit() × unit(0.5, 0.5)), (17, 4), false; backend = bk), 3),
        (mesh(domain(unit() × unit(0.5, 0.5) × unit()), (9, 4, 5), false; backend = bk), 2)
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

const _BROADCAST_SIZES = ((1,), (2,), (7,), (1001,), (5, 3), (40, 37), (4, 3, 5), (13, 11, 9))

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
