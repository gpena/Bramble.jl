push!(LOAD_PATH,"../src/")

using Documenter, Bramble

makedocs(sitename="Bramble",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "false"
    )
)