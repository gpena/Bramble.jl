if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    test_dir = @__DIR__
    bramble_dir = abspath(joinpath(test_dir, "../"))
    Pkg.activate(joinpath(test_dir, "."))
    Pkg.develop(; path = bramble_dir)
    Pkg.instantiate()
end

using Test
using Bramble

# The 8 shared test helpers (alloc_test, @test_allocs, _fd, _tri, _nonuniform_points,
# _zero_boundary!, _matches_fd, _run_example_page) used to live here, back when every test
# file was `include`d into bare `Main` and so already shared this namespace with them. Now
# that every test file is its own module, they live in their own module instead, reached
# explicitly via `using ..TestUtils: ...` from whichever file needs them.
include("TestUtils.jl")

# Read from TestUtils rather than from `ENV` directly, so that a subsystem's own
# `runtests.jl` -- a standalone entry point that never reaches this file -- resolves the
# group the same way this does.
const __bramble_test_group = TestUtils.TEST_GROUP

# `full` deliberately does *not* imply `quality`. No workflow runs `full`: Weekly.yml splits
# it into its `slow` and `backends` halves, and a local `.claude/scripts/test.sh full` is its
# only caller. Aqua/JET/explicit-imports/exports/invalidations already run daily in
# `nightly.yml`'s own `quality` job, so repeating them under `full` would learn nothing the
# daily run had not already reported. `full` now means "everything the daily
# workflows do not already cover": the unit suite across both Julia versions, the expensive
# AD backends, the package extensions and the manual snippets.
#
# A local `.claude/scripts/test.sh` run with no argument gets `unit` (its own default,
# `.claude/scripts/test.sh`'s `ARG="${1:-${BRAMBLE_TEST_GROUP:-unit}}"`) -- deliberately the
# fast, every-push subset, not `quality` or `slow`. `unit` is only the *script's* default:
# `TestUtils.TEST_GROUP` itself falls back to `all` (see below) for any invocation that does
# not go through the script, e.g. a bare `Pkg.test()` or a subsystem's own standalone
# `runtests.jl` with no `BRAMBLE_TEST_GROUP` set.
const __bramble_with_quality = __bramble_test_group in ("all", "quality")
const __bramble_with_unit_tests = __bramble_test_group in ("all", "unit", "full", "slow")

# `slow` is `unit` plus the blocks whose cost is out of proportion to what a *push*
# learns from them -- CI.yml runs `unit` and skips them, nightly.yml runs `slow` on both
# platforms daily. TestUtils holds the definition (WITH_SLOW_TESTS, read by the subsystem
# runtests.jl files and test files), and the list, with measured costs.

# AD and GPU tests are off in every group until v4.3.0 and v4.4.0; TestUtils holds the
# switches and says how to turn them back on.
const __bramble_with_ad_tests = TestUtils.WITH_AD_TESTS
const __bramble_with_gpu_tests = TestUtils.WITH_GPU_TESTS

# The expensive differentiation backends (Enzyme, Mooncake, and the policy crossings), when
# AD tests are switched on. Their packages are not in test/Project.toml; each file skips a
# backend absent from the environment, so the group reports a skip rather than an error.
#
# The AD extension files (sparse AD, the AD backend verification, SciMLSensitivity, the
# pde_solve rrule and the transient-inverse-problem page) run here too, not under `ext`, so
# that `ext` is CPU extensions only.
#
# `backends` is `ad` plus `ext` with no unit suite: Weekly.yml runs it as one half of the
# full suite and `slow` as the other, since the Julia 1.12 legs outgrew one 90-minute job.
const __bramble_with_ad_backends = __bramble_with_ad_tests &&
                                   __bramble_test_group in ("ad", "full", "backends")

# The Makie/Meshes/RecipesBase weak deps: `test/Project.toml` lists them (so
# `Pkg.instantiate()` always resolves and can precompile them), but they are only ever
# `using`-d, and so only ever pay their compile cost, behind this group -- every push
# otherwise gets none of that weight. CPU extensions only: the AD extension files run in
# `ad`, the Metal files in `gpu` and the extension-dependent example pages in `examples`.
const __bramble_with_ext_backends = __bramble_test_group in ("ext", "full", "backends")

# The four Metal files, in their own group and only with GPU tests switched on (TestUtils
# says how). Each also skips its device kernels on a CI runner (`_run_gpu_tests()`).
const __bramble_with_gpu_group = __bramble_with_gpu_tests && __bramble_test_group == "gpu"

# The worked-example pages themselves, run rather than mirrored: each is a Literate script
# whose `#src` assertions pin the numbers it renders. What they uniquely catch is a
# number the documentation *publishes* going stale, which is the documentation build's
# concern, so they form their own group and no other group runs them: `pages.jl` (the
# pages that need only the test environment) and `ext_pages.jl` (the pages that load a
# stiff solver, `NonlinearSolve`, `LinearSolve` or `AlgebraicMultigrid`).
const __bramble_with_examples = __bramble_test_group == "examples"

# `manual/test_snippets.jl` checks that the code in the 12-chapter PDF manual still
# compiles and gives the answers it claims. Kept out of the every-push groups because it
# assembles several real PDE systems (seconds, not milliseconds) and, unlike the operator
# tests, doesn't catch anything a change to the manual's own prose wouldn't also need a
# human to re-read for.
#
# `manual/` is entirely gitignored (the manual is written and built outside version
# control, by request), so this file does not exist on a fresh checkout -- including CI's.
# `isfile` below makes this a local-only check: it runs when a maintainer has the manual
# checked out and asks for the `full` group, and skips with a clear `@info` (not silently)
# everywhere else, rather than erroring on a file that was never going to be there.
const __bramble_manual_snippets_path = joinpath(
    @__DIR__, "..", "manual", "test_snippets.jl"
)
const __bramble_with_manual_snippets = __bramble_test_group == "full" && isfile(__bramble_manual_snippets_path)
if __bramble_test_group == "full" && !__bramble_with_manual_snippets
    @info "Skipping manual snippets: manual/test_snippets.jl not found (manual/ is gitignored, so this is expected outside a machine that has it checked out locally)."
end

# `assemble_parallel!`/`Parallel()`-backend correctness is only genuinely exercised when
# more than one thread is actually available -- on one thread the multi-colour sweep never
# runs concurrently, so nothing can race, and the tests that check it degrade to
# `@test_skip` (test/form/linear.jl's "Parallel vs serial agreement", test/form/bilinear.jl's
# "Determinism under threads"). CI sets `JULIA_NUM_THREADS=auto`, so this only fires for a
# local `Pkg.test()`, which defaults to one thread -- without this, that run reads as "all
# green" with five-plus concurrency assertions quietly never having run at all.
if __bramble_with_unit_tests && Threads.nthreads() == 1
    @warn "Running on a single thread ($(Threads.nthreads())): concurrency assertions for assemble_parallel!/Parallel() are skipped (@test_skip), not passed. Run `julia --threads=auto` (or pass `julia_args = \`--threads=auto\`` to `Pkg.test`) to actually exercise them."
end

if __bramble_with_unit_tests
    @testset verbose=true "Core library" begin
        include("utils/runtests.jl")
        include("geometry/runtests.jl")
        include("mesh/runtests.jl")

        @testset "Grid spaces" begin
            include("space/gridspaces.jl")
            include("space/weights_staleness.jl")
            include("space/vector_elements.jl")
            include("space/backend_profile.jl")
        end

        @testset "Operators" begin
            include("space/difference.jl")
            include("space/star_difference.jl")
            include("space/star_vector_calculus.jl")
            include("space/centered_difference.jl")
            include("space/centered_vector_calculus.jl")
            include("space/cross_weighted_difference.jl")
            include("space/sbp_identities.jl")
            include("space/sobolev_inequalities.jl")
            include("space/discrete_calculus_identities.jl")
            include("space/commutation.jl")
            include("space/jump.jl")
            include("space/shift.jl")
            include("space/average.jl")
            include("space/centered_average.jl")
            include("space/dimensional_dispatch.jl")
            include("space/inplace_operators.jl")
            include("space/threaded_stencils.jl")
            include("space/threaded_vector_calculus.jl")
            include("space/threaded_broadcast.jl")
            include("space/operators.jl")
            include("space/operator_docstrings.jl")
            include("space/inner_product.jl")
            include("space/inner_plus_boundary.jl")
            include("space/conservation.jl")
            include("space/composite_operators.jl")
            include("space/interpolation.jl")
            include("space/interpolation_bounds.jl")
            include("space/inference_allocation.jl")
            include("convergence/runtests.jl")
            include("space/element_type.jl")
            if __bramble_with_ad_tests
                include("space/autodiff.jl")
                include("space/autodiff_backends.jl")
            end
        end

        include("form/runtests.jl")
        include("solvers/runtests.jl")
        include("exporters/runtests.jl")

        # Independent full-pipeline tests (mesh -> space -> assemble -> solve) for a path no
        # docs page reaches, as opposed to the `examples` group below, which runs the pages.
        include("drivers/runtests.jl")

        # Static allocation verification. Lives under `quality/` because that is what
        # it is, but runs with the unit group because it is the one quality gate cheap enough
        # to pay on every push (3 s, against minutes for JET), and an allocation regression
        # is exactly the kind of thing that should not wait for the nightly to surface it.
        # The `quality` group picks it up too, below, when the unit group is not running.
        @testset "Static allocations" begin
            include("quality/alloccheck.jl")
        end
    end
end

if __bramble_with_quality
    @testset verbose=true "\nQuality" begin
        include("quality/aqua.jl")
        include("quality/exports.jl")
        include("quality/public_docs.jl")
        include("quality/explicit_imports.jl")
        include("quality/source_imports.jl")
        include("quality/testset_names.jl")
        include("quality/jet.jl")
        # These three check the test suite itself, not the package. They stay out of `unit`
        # (JET over every test file takes minutes, like the package JET above) and run
        # with the rest of this group in nightly.yml.
        include("quality/type_stability.jl")
        include("quality/test_env.jl")
        include("quality/test_jet.jl")
        include("quality/invalidations.jl")
        # Decoupled from docs/make.jl (doctest = false there) so a doctest regression is
        # caught here, in parallel with the rest of this group, instead of during every docs
        # build.
        include("quality/doctests.jl")
        include("quality/claude_plugin.jl")
        # Already run above with the unit group; included here so a `quality`-only run
        # (nightly's second job) still covers it, without running it twice for `all`.
        __bramble_with_unit_tests || include("quality/alloccheck.jl")
    end
end

if __bramble_with_ad_backends
    @testset verbose=true "AD backends (expensive)" begin
        # autodiff_backends.jl first: it defines `check_backend` and `_ad_problems`, which
        # this reuses so both files check every backend the same way. (`_have` used to come
        # from here too and is now TestUtils'.)
        __bramble_with_unit_tests || include("space/autodiff_backends.jl")
        include("space/autodiff_heavy.jl")
        # The same backends crossed with the Parallel() and CpuPolyester() policies, dense
        # and sparse; it reuses nothing from the two files above.
        include("space/autodiff_policies.jl")
        # pde_solve's rrule (ext/chainrules_ext.jl, "Package extensions" below) composed with
        # a real reverse-mode backend -- Enzyme; Mooncake pinned as currently unsupported.
        # Self-contained, independent of that file's own run.
        include("ext/chainrules_enzyme_ext.jl")
        # The boundary-condition-recovery worked example needs Enzyme, unlike every other
        # example page -- run here rather than in "Worked examples"/"Package extensions".
        include("examples/inverse_diffusion.jl")
        # The AD extension files, moved here from "Package extensions" so `ext` is CPU only.
        include("ext/sparse_ad_ext.jl")
        include("ext/ad_backend_verification.jl")
        include("ext/sciml_sensitivity_ext.jl")
        # Runs the transient-inverse-problem page itself, whose #src assertions need
        # `SciMLSensitivity`, same as `inverse_diffusion.jl` needing `Enzyme` above.
        include("examples/transient_inverse_problem.jl")
        # BrambleChainRulesExt: the pde_solve rrule's own math, checked against finite
        # differences and by hand -- needs only ChainRulesCore. Enzyme/Mooncake composition
        # is chainrules_enzyme_ext.jl above.
        include("ext/chainrules_ext.jl")
    end
end

if __bramble_with_manual_snippets
    @testset verbose=true "Manual snippets" begin
        include(__bramble_manual_snippets_path)
    end
end

if __bramble_with_ext_backends
    @testset verbose=true "Package extensions" begin
        # The contract the four direct-solver backend files share, loaded before them and
        # reached as `using ..ExtSolverContracts: ...`. The guard lives here rather than
        # inside those modules: an `include` executed inside one of them would define
        # `TestSuiteSparseExt.ExtSolverContracts`, which `using ..ExtSolverContracts` would
        # not then resolve to. Same shape as the `TestUtils` guards in the subsystem
        # runtests.jl files.
        isdefined(Main, :ExtSolverContracts) || include("ext/SolverContracts.jl")

        include("ext/plots_ext.jl")
        include("ext/makie_ext.jl")
        include("ext/meshes_ext.jl")
        include("ext/sciml_ext.jl")
        include("ext/algebraicmultigrid_ext.jl")
        include("ext/iluzero_ext.jl")
        include("ext/suitesparse_ext.jl")
        include("ext/appleaccelerate_ext.jl")
        include("ext/mumps_ext.jl")
        include("ext/sparspak_ext.jl")
        # The memory-scaling milestone's own backend/operator extensions (v3.3.0 plan S3.1,
        # S5.2, S7.2): SparseMatrixCSR assembly, the Kronecker.jl fast-diagonalisation solve,
        # and the Polyester-backed CpuPolyester sweeps. Grouped with the other package-extension
        # files above rather than the every-push suite because each needs its own weak
        # dependency loaded.
        include("ext/sparse_csr_ext.jl")
        include("ext/kronecker_ext.jl")
        include("ext/polyester_ext.jl")
    end
end

if __bramble_with_gpu_group
    @testset verbose=true "GPU (Metal)" begin
        include("ext/metal_ext.jl")
        include("ext/metal_fullstack.jl")
        include("ext/metal_assembly_replay.jl")
        include("ext/metal_form_assembly.jl")
    end
end

if __bramble_with_examples
    @testset verbose=true "Worked examples" begin
        include("examples/pages.jl")
        include("examples/ext_pages.jl")
        include("examples/claude_plugin.jl")
    end
end
