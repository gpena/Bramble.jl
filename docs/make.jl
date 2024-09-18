push!(LOAD_PATH,joinpath(@__DIR__,".."))

using Documenter
using Bramble

include("pages.jl")

makedocs(; sitename="Bramble.jl")

deploydocs(repo = "github.com/gpena/gpena.github.io.git")
