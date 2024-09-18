push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Documenter
using Bramble

include("pages.jl")

makedocs(;
		 modules = [Bramble],
		 format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
		 sitename = "Bramble.jl",
		 pages = ["Home" => "index.md",
			 "internals.md"],
		 authors = "Gonçalo Pena")

deploydocs(; repo = "github.com/gpena/Bramble.jl")
