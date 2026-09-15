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

const __bramble_test_group = get(ENV, "BRAMBLE_TEST_GROUP", "all")
const __bramble_with_quality = __bramble_test_group in ("all", "quality", "full")
const __bramble_with_unit_tests = __bramble_test_group in ("all", "unit", "full")

# The differentiation backend survey is split by what it costs, measured per backend:
#
#   ForwardDiff 0.3 s   ReverseDiff 0.5 s   PolyesterForwardDiff 0.6 s
#   Mooncake   25.1 s   Enzyme     33.2 s
#
# So the three cheap ones run with the unit tests (3.3 s between them, load included)
# and the two that spend almost a minute compiling on first call live behind this group.
# What they establish changes when a *backend* changes rather than when Bramble does, so
# paying that per push would be paying it for nothing almost every time. The weekly
# workflow runs `full`, which is this plus everything else.
#
# Either file skips a backend absent from the environment, so running a group without one
# installed reports a skip rather than an error.
const __bramble_with_ad_backends = __bramble_test_group in ("ad", "full")

# The Makie/Meshes/RecipesBase/Metal weak deps: `test/Project.toml` lists them (so
# `Pkg.instantiate()` always resolves and can precompile them, the same tradeoff already
# made for the AD backends above), but they are only ever `using`-d, and so only ever pay
# their compile cost, behind this group -- every push otherwise gets none of that weight.
# Metal within this group further gates on `Metal.functional()`, since installing and
# precompiling it succeeds on any platform (it degrades gracefully, the same convention
# CUDA.jl uses) while only a real Apple Silicon device can actually run anything on it.
const __bramble_with_ext_backends = __bramble_test_group in ("ext", "full")

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
        end

        @testset "Operators" begin
            include("space/difference.jl")
            include("space/star_difference.jl")
            include("space/centered_difference.jl")
            include("space/cross_weighted_difference.jl")
            include("space/sbp_identities.jl")
            include("space/commutation.jl")
            include("space/jump.jl")
            include("space/average.jl")
            include("space/inplace_operators.jl")
            include("space/operators.jl")
            include("space/inner_product.jl")
            include("space/inner_plus_boundary.jl")
            include("space/conservation.jl")
            include("space/composite_operators.jl")
            include("space/interpolation.jl")
            include("space/interpolation_bounds.jl")
            include("space/inference_allocation.jl")
            include("convergence/runtests.jl")
            include("space/element_type.jl")
            include("space/autodiff.jl")
            include("space/autodiff_backends.jl")
        end

        include("form/runtests.jl")
        include("exporters/runtests.jl")

        # The worked-example pages themselves, run rather than mirrored: each is a
        # Literate script whose `#src` assertions pin the numbers it renders (#117). Plus
        # the variable-coefficient case no page covers. ~1m together -- cheap enough to run
        # on every push rather than sit behind a group, and it covers assemble, the
        # Dirichlet path, and the sparse-AD Newton loop as pipelines rather than operator by
        # operator.
        @testset "Worked examples" begin
            include("examples/pages.jl")
        end

        # Bug reproducers that aren't naturally part of one subsystem file's coverage
        # (STANDARDS.md ties this to a closed GitHub issue). Tests that extend an existing
        # subsystem file's own coverage stay there, tagged `(#N)` in the testset title.
        include("issues/runtests.jl")

        # Independent full-pipeline tests (mesh -> space -> assemble -> solve) for a path no
        # docs page reaches, as opposed to "Worked examples" above, which mirrors a page.
        include("drivers/runtests.jl")

        # Static allocation verification (#118). Lives under `quality/` because that is what
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
        include("quality/explicit_imports.jl")
        include("quality/jet.jl")
        include("quality/invalidations.jl")
        # Already run above with the unit group; included here so a `quality`-only run
        # (nightly's second job) still covers it, without running it twice for `all`.
        __bramble_with_unit_tests || include("quality/alloccheck.jl")
    end
end

if __bramble_with_ad_backends
    @testset verbose=true "AD backends (expensive)" begin
        # autodiff_backends.jl first: it defines `check_backend` and `_have`, which this
        # reuses so both files check every backend the same way.
        __bramble_with_unit_tests || include("space/autodiff_backends.jl")
        include("space/autodiff_heavy.jl")
        # pde_solve's rrule (ext/chainrules_ext.jl, "Package extensions" below) composed with
        # a real reverse-mode backend -- Enzyme; Mooncake pinned as currently unsupported.
        # Self-contained, independent of that file's own run.
        include("ext/chainrules_enzyme_ext.jl")
        # The boundary-condition-recovery worked example needs Enzyme, unlike every other
        # example page -- run here rather than in "Worked examples"/"Package extensions".
        include("examples/inverse_diffusion.jl")
    end
end

if __bramble_with_manual_snippets
    @testset verbose=true "Manual snippets" begin
        include(__bramble_manual_snippets_path)
    end
end

if __bramble_with_ext_backends
    @testset verbose=true "Package extensions" begin
        include("ext/plots_ext.jl")
        include("ext/makie_ext.jl")
        include("ext/meshes_ext.jl")
        include("ext/metal_ext.jl")
        include("ext/sparse_ad_ext.jl")
        include("ext/ad_backend_verification.jl")
        include("ext/sciml_ext.jl")
        include("ext/algebraicmultigrid_ext.jl")
        include("ext/suitesparse_ext.jl")
        include("ext/appleaccelerate_ext.jl")
        include("ext/mumps_ext.jl")
        # BrambleChainRulesExt: the pde_solve rrule's own math, checked against finite
        # differences and by hand -- needs only ChainRulesCore, not Enzyme/Mooncake, so it
        # belongs here rather than behind the "ad" group. Enzyme/Mooncake composition is
        # chainrules_enzyme_ext.jl instead, alongside autodiff_heavy.jl below.
        include("ext/chainrules_ext.jl")
        # Runs the worked heat-equation page itself, whose assertions need a stiff solver
        # for a differential-algebraic system -- so it belongs where OrdinaryDiffEq is
        # already loaded rather than in the every-push "Worked examples" group.
        include("examples/heat_equation.jl")
        # Same reasoning: the nonlinear Poisson page's NonlinearSolve.jl comparison needs
        # `NonlinearSolve` loaded, a cost the push path does not otherwise pay.
        include("examples/poisson_nonlinear.jl")
        # Same reasoning again: the coupled reaction-diffusion page's nonlinear_problem
        # section needs `NonlinearSolve` too, once it grew one (#119).
        include("examples/coupled_reaction_diffusion.jl")
        # Same reasoning again: the AMG preconditioning page's LU/CG/AMG-CG comparison needs
        # `LinearSolve` and `AlgebraicMultigrid` loaded.
        include("examples/amg_preconditioning.jl")
    end
end
