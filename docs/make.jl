push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Documenter, Bramble

internals = "Internals" => ["internals/geometry.md", "internals/mesh.md", "internals/space.md"]

makedocs(#format = Documenter.LaTeX(platform = "none"),
		 format = Documenter.HTML(),
		 sitename = "Bramble.jl",
		 pages = ["Home" => "index.md",
			 "Examples" => ["examples.md"],
			 "Documentation" => ["api.md",
				 internals]],
		 authors = "Gonçalo Pena")

deploydocs(;
		   repo = "github.com/gpena/Bramble.jl",
		   versions = nothing,
		   branch = "gh-pages")