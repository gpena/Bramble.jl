push!(LOAD_PATH,"../src/")

using Documenter, Bramble

makedocs(sitename="Bramble",
    format = Documenter.HTML(
        prettyurls = get(ENV, "ci", nothing) == "true"
    )
)

deploydocs(
    repo = "github.com/gpena/Bramble.git"
)