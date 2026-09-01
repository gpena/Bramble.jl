using Bramble
using Documenter

include("generate_benchmarks.jl")
generate_benchmarks_markdown()

home = "Home" => "index.md"
tutorials = "Tutorials" =>
    ["tutorials/geometry.md", "tutorials/mesh.md", "tutorials/space.md",
        "tutorials/operators.md"]
# examples = "Examples" => ["examples/poisson_linear.md", "examples/poisson_nonlinear.md"]
benchmarks = "Benchmarks" => "benchmarks.md"
internals = "Internals" => ["internals/utils.md", "internals/geometry.md",
    "internals/mesh.md", "internals/space.md", "internals/form.md"]
documentation = "Documentation" => ["api.md", internals]

allpages = [
    home,
    tutorials,
    # examples,
    benchmarks,
    documentation
]

makedocs(;
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true",
        size_threshold = 400 * 1024,
        size_threshold_warn = 250 * 1024),
    sitename = "Bramble.jl",
    pages = allpages,
    authors = "Gonçalo Pena and Gemini",
    # `missing_docs` stays a warning: Documenter reports every internal helper it cannot
    # find a page for, so making it an error would mean adding `@docs` stubs to silence it
    # rather than because they help. The rule that matters — every *exported* name has a
    # docstring — is enforced in test/quality/docstrings.jl instead, where it has no false
    # positives. A broken `@ref` is always a real mistake, so that one is an error.
    warnonly = [:missing_docs])

deploydocs(;
    repo = "github.com/gpena/Bramble.jl.git",
    devbranch = "main",
    branch = "gh-pages",
    versions = nothing,
    push_preview = true)
