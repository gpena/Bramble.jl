import Pkg
Pkg.add("Documenter")
using Documenter
push!(LOAD_PATH,"../")
push!(LOAD_PATH,"../src/")
using Bramble

include("pages.jl")

makedocs(
    sitename = "Bramble.jl",
		 authors = "Gonçalo Pena",
		 modules = [Bramble]
)

deploydocs(
    repo = "github.com/gpena/gpena.github.io.git",
    target = "build",
    branch = "gh-pages",
)
