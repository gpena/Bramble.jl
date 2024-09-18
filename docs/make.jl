push!(LOAD_PATH,joinpath(@__DIR__,".."))

using Documenter
using Bramble

include("pages.jl")

makedocs(
    sitename = "Bramble.jl",
		 authors = "Gonçalo Pena",
		 modules = [Bramble]
)

deploydocs(repo = "https://github.com/gpena/gpena.github.io.git")