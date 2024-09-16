push!(LOAD_PATH,"../src/")

using Pkg
Pkg.add(url="https://github.com/gpena/Bramble")
using Bramble
using Documenter

makedocs(sitename="Bramble",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    )
)

deploydocs(
    repo = "github.com/gpena/bramble.github.io.git"
)