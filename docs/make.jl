using Bramble
using Documenter
using Literate

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
    "inverse_diffusion.jl"
]

let dir = joinpath(@__DIR__, "src", "examples")
    for file in LITERATE_EXAMPLES
        Literate.markdown(
            joinpath(dir, file), dir;
            documenter = true,
            credit = false,
            repo_root_url = "https://github.com/gpena/Bramble.jl/blob/main"
        )
    end
end

home = "Home" => "index.md"
tutorials = "Tutorials" => [
    "tutorials/geometry.md",
    "tutorials/mesh.md",
    "tutorials/backend.md",
    "tutorials/space.md",
    "tutorials/operators.md",
    "tutorials/form.md",
    "tutorials/autodiff.md",
    "tutorials/vtk_export.md",
    "tutorials/pgfplots_export.md",
    "tutorials/plotting.md"
]
examples = "Examples" => [
    "examples/poisson_linear.md",
    "examples/poisson_nonlinear.md",
    "examples/convection_diffusion_linear.md",
    "examples/coupled_reaction_diffusion.md",
    "examples/elasticity_3d.md",
    "examples/heat_equation.md",
    "examples/amg_preconditioning.md",
    "examples/inverse_diffusion.md"
]
benchmarks = "Benchmarks" => "benchmarks.md"
internals = "Internals" => [
    "internals/utils.md",
    "internals/geometry.md",
    "internals/mesh.md",
    "internals/space.md",
    "internals/form.md",
    "internals/autodiff.md",
    "internals/exporters.md"
]
documentation = "Documentation" => ["api.md", "api_sciml.md", internals]

allpages = [home, tutorials, examples, benchmarks, documentation]

makedocs(;
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true",
        size_threshold = 400 * 1024,
        size_threshold_warn = 250 * 1024
    ),
    sitename = "Bramble.jl",
    pages = allpages,
    authors = "Gonçalo Pena and Gemini",
    # `missing_docs` stays a warning: Documenter reports every internal helper it cannot
    # find a page for, so making it an error would mean adding `@docs` stubs to silence it
    # rather than because they help. The rule that matters (every *exported* name has a
    # docstring) is enforced in test/quality/exports.jl instead, where it has no false
    # positives. A broken `@ref` is always a real mistake, so that one is an error.
    warnonly = [:missing_docs]
)

deploydocs(;
    repo = "github.com/gpena/Bramble.jl.git",
    devbranch = "main",
    branch = "gh-pages",
    versions = nothing,
    push_preview = true
)
