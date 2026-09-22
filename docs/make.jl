using Bramble
using Documenter
using Literate
# Loaded here, not only inside `transient_inverse_problem.jl`'s own `@example` block: the
# `@docs Bramble.adjoint_sensitivities` block on `api_sciml.md` needs `BrambleSciMLSensitivityExt`
# already loaded to pick up that method's own (richer) docstring alongside the core stub's --
# `Base.Docs.doc` merges both once both are loaded, but only if `SciMLSensitivity` is loaded
# before Documenter processes that `@docs` block, not merely before this script exits. Page
# processing order is not something to rely on for that (measured: loading it only inside the
# example page left `api_sciml.md`'s block showing the stub alone).
using SciMLSensitivity

include("generate_benchmarks.jl")
generate_benchmarks_markdown()

# Worked-example pages (gpena/Bramble.jl#117). Each one is written as a runnable script and the
# markdown Documenter renders is generated from them here, so the page a reader sees and the
# file the suite runs are the same file. Lines marked `#src` -- the assertions that make the
# rendered numbers load-bearing -- are stripped on the way to markdown and kept when
# test/examples/pages.jl runs them. The generated `.md` files are gitignored.
const LITERATE_EXAMPLES = [
    "poisson_linear.jl",
    "poisson_nonlinear.jl",
    "convection_diffusion_linear.jl",
    "coupled_reaction_diffusion.jl",
    "elasticity_3d.jl",
    "heat_equation.jl",
    "amg_preconditioning.jl",
    "inverse_diffusion.jl",
    "transient_inverse_problem.jl",
    "wave_equation_2d.jl",
    "point_sources_flux.jl",
    "transient_inplace.jl",
    "boundary_layer_graded.jl",
    "memory_scaling.jl"
]

if Threads.nthreads() == 1
    @info "docs/make.jl is running single-threaded — pass `--threads=auto` for a faster build" *
          " (the worked examples' own `@example` blocks below are the slow part, not this)."
end

# `asyncmap` rather than a serial loop: harmless either way since `Literate.markdown` with
# `documenter = true` only rewrites `.jl` syntax into `@example`-tagged markdown here, it
# does not execute any of it (that happens later, inside `makedocs`, one page at a time) —
# measured at ~1.7s total for all 8 files serially, so this is not where a slow build's time
# goes (gpena/Bramble.jl#251), but there is no reason to keep it serial either.
let dir = joinpath(@__DIR__, "src", "examples")
    asyncmap(LITERATE_EXAMPLES) do file
        Literate.markdown(
            joinpath(dir, file), dir;
            documenter = true,
            credit = false,
            repo_root_url = "https://github.com/gpena/Bramble.jl/blob/main"
        )
    end
end

home = "Home" => "index.md"
getting_started = "Getting started" => "getting_started.md"

# Grouped by where a page sits in the workflow rather than as one flat "Tutorials" list:
# a reader meets geometry, meshes, spaces and operators before forms, and the solver,
# AD and backend pages after. `tutorials/backend.md` appears here and nowhere else --
# listing a page twice is a fatal Documenter error, as is leaving one out.
foundations = "Discrete foundations" => [
    "tutorials/geometry.md",
    "tutorials/mesh.md",
    "tutorials/space.md",
    "tutorials/operators.md"
]
forms = "Forms and assembly" => [
    "tutorials/form.md"
]
scientific = "Solvers and scientific computing" => [
    "tutorials/solvers.md",
    "tutorials/autodiff.md",
    "tutorials/backend.md"
]
visualization = "Visualization and export" => [
    "tutorials/plotting.md",
    "tutorials/vtk_export.md",
    "tutorials/pgfplots_export.md"
]
examples = "Examples" => [
    "examples/poisson_linear.md",
    "examples/poisson_nonlinear.md",
    "examples/convection_diffusion_linear.md",
    "examples/coupled_reaction_diffusion.md",
    "examples/elasticity_3d.md",
    "examples/heat_equation.md",
    "examples/amg_preconditioning.md",
    "examples/inverse_diffusion.md",
    "examples/transient_inverse_problem.md",
    "examples/wave_equation_2d.md",
    "examples/point_sources_flux.md",
    "examples/transient_inplace.md",
    "examples/boundary_layer_graded.md",
    "examples/memory_scaling.md"
]
benchmarks = "Benchmarks" => "benchmarks.md"
internals = "Internals" => [
    "internals/utils.md",
    "internals/geometry.md",
    "internals/mesh.md",
    "internals/space.md",
    "internals/form.md",
    "internals/autodiff.md",
    "internals/exporters.md",
    "internals/gpu.md",
    "internals/csr_solvers.md"
]
documentation = "Documentation" => ["api.md", "api_sciml.md", internals]

allpages = [home, getting_started, foundations, forms, scientific,
    visualization, examples, benchmarks, documentation]

makedocs(;
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true",
        # The API reference is one page listing every exported name's docstring, so it grows
        # with the package and crossed 400 KiB when the surface integral and the normal were
        # added (gpena/Bramble.jl#157, #213). Raised rather than split: one searchable page
        # is the point of it, and the threshold exists to catch a page that grew by accident.
        size_threshold = 600 * 1024,
        size_threshold_warn = 450 * 1024,
        # "Signal" theme (#132): retokenizes Documenter's own sidebar/content/breadcrumb
        # shell in place, so search/doctest/@ref keep working unmodified.
        assets = [
            "assets/favicon.ico",
            Documenter.asset(
                "https://fonts.googleapis.com/css2?family=Manrope:wght@700;800&family=Source+Sans+3:wght@400;600&display=swap";
                class = :css
            ),
            "assets/custom.css"
        ]
    ),
    sitename = "Bramble.jl",
    pages = allpages,
    authors = "Gonçalo Pena",
    # `missing_docs` stays a warning: Documenter reports every internal helper it cannot
    # find a page for, so making it an error would mean adding `@docs` stubs to silence it
    # rather than because they help. The rule that matters (every *exported* name has a
    # docstring) is enforced in test/quality/exports.jl instead, where it has no false
    # positives. A broken `@ref` is always a real mistake, so that one is an error.
    warnonly = [:missing_docs],
    # Decoupled (gpena/Bramble.jl#251): checked instead by test/quality/doctests.jl, in
    # parallel with the rest of that group, rather than on every docs build. This only skips
    # Documenter's own separate "Doctest" pipeline stage (the handful of `@jldoctest` blocks
    # in `src/`) — it does *not* skip executing the worked examples' `@example` blocks, which
    # "ExpandTemplates" always runs regardless of this setting and is where a slow build's
    # time actually goes.
    doctest = false
)

# Unversioned, deliberately: one build at the root of `gh-pages`, always the current one.
#
# Versioning was switched on briefly at v3.0.0 and switched back off. It works -- `stable`,
# per-minor directories, a selector, an outdated-version banner -- but it turns one site into
# a tree of them, and the cost lands on every reader and every link: an extra path segment in
# every URL, a root that only redirects, and old versions that have to be built and kept.
# For a package with one supported line at a time, that is machinery without a reader.
#
# A v2 reference still exists: the `v2.17.0` tag.
#
# `push_preview` is not set: it enables PR preview deploys, and no workflow builds docs on a
# pull request.
deploydocs(;
    repo = "github.com/gpena/Bramble.jl.git",
    devbranch = "main",
    branch = "gh-pages",
    versions = nothing
)
