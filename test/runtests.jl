if abspath(PROGRAM_FILE) == @__FILE__
	using Pkg
	Pkg.activate(@__DIR__)
	Pkg.develop(PackageSpec(path = joinpath(@__DIR__, "..")))
	Pkg.instantiate()
end

using Test
using LinearAlgebra: norm, \

const __with_examples = true

validate_zero(u) = norm(u, Inf) < 1e-9
validate_equal(u, v) = validate_zero(u .- v)

function leastsquares(x::AbstractVector, y::AbstractVector)
	A = [sum(x .* x) sum(x);
		 sum(x) length(x)]

	b = [sum(x .* y); sum(y)]

	res = A \ b
	return res[1], res[2]
end

function main()
	println("")

	@testset verbose=true "Core library" begin
		@testset "Sets and Domains" begin
			include("sets.jl")
			include("domains.jl")
		end

		@testset "Meshes" begin
			include("mesh1d.jl")
			include("meshnd.jl")
		end

		@testset "Grid spaces" begin
			include("gridspaces.jl")
			include("vectorelements.jl")
			include("matrixelements.jl")
			include("operators.jl")
		end

		@testset "Forms" begin
			include("bilinearforms.jl")
		end
	end

	if __with_examples
		include("examples.jl")
	end
end

main()