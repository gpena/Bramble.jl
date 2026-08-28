using Bramble
using Documenter

home = "Home" => "index.md"
tutorials = "Tutorials" => ["tutorials/geometry.md"]
internals = "Internals" => ["internals/utils.md", "internals/geometry.md"]
documentation = "Documentation" => ["api.md", internals]

allpages = [
	home,
	tutorials,
	documentation,
]

makedocs(; format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
		 sitename = "Bramble.jl",
		 pages = allpages,
		 authors = "Gonçalo Pena",
		 warnonly = [:cross_references, :missing_docs])

deploydocs(;
		   repo = "github.com/gpena/Bramble.jl.git",
		   devbranch = "main",
		   branch = "gh-pages",
		   push_preview = true)