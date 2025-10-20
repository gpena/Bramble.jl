include(joinpath(@__DIR__, "../docs/examples/poisson_linear.jl"))
include(joinpath(@__DIR__, "../docs/examples/poisson_nonlinear.jl"))
include(joinpath(@__DIR__, "../docs/examples/convection_diffusion_linear.jl"))

function least_squares_fit(x, y)
	A = hcat(ones(length(x)), log.(x))
	c = A \ log.(y)
	return c[2], exp(c[1])
end

println("")
@testset verbose=true "Examples" begin
	@testset "Linear Poisson equation" begin
		test_poisson(poisson(1), 10, (i -> 2^i + 1,), ntuple(i -> true, 1))
		test_poisson(poisson(1), 100, (i -> 20 * i,), (false,))

		test_poisson(poisson(2), 4, (i -> 2^i + 1, i -> 2^i + 2), ntuple(i -> true, 2))
		test_poisson(poisson(2), 7, (i -> 2^i + 1, i -> 2^i + 2), ntuple(i -> false, 2))

		test_poisson(poisson(3), 5, (i -> 2^i + 1, i -> 2^i + 2, i -> 2^i + 3), ntuple(i -> true, 3))
	end

	@testset "Nonlinear Poisson equation" begin
		test_poisson_nl(poisson_nl(1), 10, (i -> 2^i + 1,), ntuple(i -> true, 1))
		test_poisson_nl(poisson_nl(1), 10, (i -> 2^i + 1,), ntuple(i -> false, 1))

		test_poisson_nl(poisson_nl(2), 5, (i -> 2^i + 1, i -> 2^i + 2), ntuple(i -> true, 2))

		test_poisson_nl(poisson_nl(3), 5, (i -> 2^i + 1, i -> 2^i + 2, i -> 2^i + 1), ntuple(i -> true, 3))
	end

	@testset "Linear convection-diffusion equation" begin
		test_conv_diff(convection_diffusion(1), 10, (i -> 2^i + 1,), ntuple(i -> true, 1))
		test_conv_diff(convection_diffusion(1), 100, (i -> 20 * i,), (false,))

		test_conv_diff(convection_diffusion(2), 5, (i -> 2^i + 1, i -> 2^i + 2), ntuple(i -> true, 2))
		test_conv_diff(convection_diffusion(2), 8, (i -> 2^i + 1, i -> 2^i + 2), ntuple(i -> false, 2))

		test_conv_diff(convection_diffusion(3), 5, (i -> 2^i + 1, i -> 2^i + 2, i -> 2^i + 3), ntuple(i -> true, 3))
		#test_conv_diff(convection_diffusion(3), 6, (i->2^i+1, i->2^i+2, i->2^i+1), ntuple(i->false, 3)) # the linear solver takes a while to solve
	end
end
