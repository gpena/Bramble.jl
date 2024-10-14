include(joinpath(@__DIR__, "../docs/examples/poisson_linear.jl"))
include(joinpath(@__DIR__, "../docs/examples/poisson_nonlinear.jl"))
include(joinpath(@__DIR__, "../docs/examples/convection_diffusion_linear.jl"))

println("")
@testset verbose=true "Examples" begin
	@testset "Linear Poisson equation" begin
		for strat in (DefaultAssembly(), AutoDetect())
			test_poisson(poisson(1), 10, (i -> 2^i + 1,), ntuple(i -> true, 1), strat)
			test_poisson(poisson(1), 100, (i -> 20 * i,), (false,), strat)

			test_poisson(poisson(2), 4, (i -> 2^i + 1, i -> 2^i + 2), ntuple(i -> true, 2), strat)
			test_poisson(poisson(2), 7, (i -> 2^i + 1, i -> 2^i + 2), ntuple(i -> false, 2), strat)

			test_poisson(poisson(3), 5, (i -> 2^i + 1, i -> 2^i + 2, i -> 2^i + 3), ntuple(i -> true, 3), strat)
		end
	end

	@testset "Nonlinear Poisson equation" begin
		for strat in (AutoDetect(), DefaultAssembly())
			test_poisson_nl(poisson_nl(1), 10, (i -> 2^i + 1,), ntuple(i -> true, 1), strat)
			test_poisson_nl(poisson_nl(1), 10, (i -> 2^i + 1,), ntuple(i -> false, 1), strat)

			test_poisson_nl(poisson_nl(2), 5, (i -> 2^i + 1, i -> 2^i + 2), ntuple(i -> true, 2), strat)
			#test_poisson_nl(poisson_nl(2), 60, (i -> 2*i+1, i -> 3*i), (true, false))

			test_poisson_nl(poisson_nl(3), 5, (i -> 2^i + 1, i -> 2^i + 2, i -> 2^i + 1), ntuple(i -> true, 3), strat)
			#test_poisson_nl(poisson_nl(3), 5, (i -> 2^i + 1, i -> 2^i + 2, i -> 2^i + 1), ntuple(i -> false, 3))
		end
	end

	@testset "Linear convection-diffusion equation" begin
		for strat in (DefaultAssembly(), AutoDetect())
			test_conv_diff(convection_diffusion(1), 10, (i -> 2^i + 1,), ntuple(i -> true, 1), strat)
			test_conv_diff(convection_diffusion(1), 100, (i -> 20 * i,), (false,), strat)

			test_conv_diff(convection_diffusion(2), 4, (i -> 2^i + 1, i -> 2^i + 2), ntuple(i -> true, 2), strat)
			test_conv_diff(convection_diffusion(2), 7, (i -> 2^i + 1, i -> 2^i + 2), ntuple(i -> false, 2), strat)

			test_conv_diff(convection_diffusion(3), 5, (i -> 2^i + 1, i -> 2^i + 2, i -> 2^i + 3), ntuple(i -> true, 3), strat)
			#test_conv_diff(convection_diffusion(3), 6, (i->2^i+1, i->2^i+2, i->2^i+1), ntuple(i->false, 3)) # the linear solver takes a while to solve
		end
	end
end
