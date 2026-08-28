using Bramble
using Documenter

home = "Home" => "index.md"
tutorials = "Tutorials" => ["tutorials/geometry.md"]
# examples = "Examples" => ["examples/poisson_linear.md", "examples/poisson_nonlinear.md"]
internals = "Internals" => ["internals/utils.md", "internals/geometry.md"]
documentation = "Documentation" => ["api.md", internals]

allpages = [
	home,
	tutorials,
	# examples,
	documentation,
]

makedocs(; format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
		 sitename = "Bramble.jl",
		 pages = allpages,
		 authors = "Gonçalo Pena and Gemini",
		 warnonly = [:cross_references, :missing_docs])

deploydocs(;
		   repo = "github.com/gpena/Bramble.jl.git",
		   devbranch = "main",
		   branch = "gh-pages",
		   versions = nothing,
		   push_preview = true)