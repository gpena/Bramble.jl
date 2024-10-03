using Test

using LinearAlgebra: norm, \

using Bramble

#Aqua.test_all(Bramble)
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

println("")

@time @testset "Sets and Domains" begin
	include("sets.jl")
	include("domains.jl")
end

println("")

@time @testset "Meshes" begin
	include("mesh1d.jl")
	include("meshnd.jl")
end

println("")

@time @testset "Grid spaces" begin
	include("gridspaces.jl")
	include("vectorelements.jl")
	include("matrixelements.jl")
end

println("")

@time @testset "Forms" begin
	include("bilinearforms.jl")
end

if __with_examples
	sep = "--------------"
	println("\n\n$sep Examples batch $sep\n")
	@time @testset "Linear Poisson equation" begin
		include("../docs/examples/poisson_linear.jl")
		#include("problems/1d/advection.jl")
		#include("problems/1d/advectionstab.jl")
		#include("problems/1d/wave.jl")
	end

	println("")

	@time @testset "Nonlinear Poisson equation" begin
		include("../docs/examples/poisson_nonlinear.jl")
	end
end