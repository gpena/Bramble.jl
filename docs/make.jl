using Documenter, Bramble

include("pages.jl")

makedocs(sitename = "Bramble.jl",
		 authors = "Gonçalo Pena",
		 modules = [Bramble],
		 clean = true, doctest = false, linkcheck = true,
		 warnonly = [:docs_block, :missing_docs, :cross_references, :linkcheck],
		 format = Documenter.HTML(prettyurls=false),
		 pages = pages)

deploydocs(repo = "github.com/gpena/Bramble.jl.git")