using Pkg
Pkg.activate(".")
using Bramble
using Documenter
#using LiveServer; servedocs()

home = "Home" => "index.md"
internals = "Internals" => ["internals/utils.md", "internals/geometry.md", "internals/mesh.md", "internals/space.md", "internals/form.md"]
examples = "Examples" => ["examples/poisson_linear.md", "examples/poisson_nonlinear.md"]
documentation = "Documentation" => ["api.md", internals]
allpages = [home,
	examples,
	documentation]

makedocs(; format = Documenter.HTML(),
		 sitename = "Bramble.jl",
		 pages = allpages,
		 authors = "Gonçalo Pena")

deploydocs(;
		   repo = "github.com/gpena/Bramble.jl",
		   versions = nothing,
		   branch = "gh-pages")