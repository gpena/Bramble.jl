push!(LOAD_PATH,joinpath(@__DIR__,".."))

using Documenter
using Bramble

include("pages.jl")

makedocs()

deploydocs(repo = "github.com/gpena/Bramble.jl.git")