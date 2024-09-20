using Test

using LinearAlgebra: norm, \

using Bramble

#Aqua.test_all(Bramble)

validate_zero(u) = norm(u, Inf) < 1e-9
validate_equal(u, v) = validate_zero(u .- v)

function leastsquares(x::AbstractVector, y::AbstractVector)
	A = [sum(x .* x) sum(x);
		 sum(x) length(x)]

	b = [sum(x .* y); sum(y)]

	res = A \ b
	return res[1], res[2]
end

sep = "######"

@time @testset "\n$sep Sets and Domains $sep" begin
	include("sets.jl")
	include("domains.jl")
end

@time @testset "\n$sep Meshes $sep" begin
	include("mesh1d.jl")
	include("meshnd.jl")
end

@time @testset "\n$sep Grid spaces $sep" begin
	include("gridspaces.jl")
	include("vectorelements.jl")
	include("matrixelements.jl")
end
