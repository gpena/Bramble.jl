push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Documenter, Bramble

makedocs(sitename = "Bramble.jl",
		 pages = ["Home" => "index.md",
			 "Examples" => ["examples.md"],
			 "Documentation" => ["api.md",
				 "internals.md"]],
		 authors = "Gonçalo Pena")

deploydocs(;
		   repo = "github.com/gpena/Bramble.jl",
		   versions = ["stable" => "v^", "v#.#", "v#.#.#"],
		   branch = "gh-pages")
