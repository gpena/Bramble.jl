include(joinpath(@__DIR__, "../docs/examples/poisson_linear.jl"))
include(joinpath(@__DIR__, "../docs/examples/poisson_nonlinear.jl"))
include(joinpath(@__DIR__, "../docs/examples/convection_diffusion_linear.jl"))
include(joinpath(@__DIR__, "../docs/examples/coupled_reaction_diffusion.jl"))

function least_squares_fit(x, y)
    A = hcat(ones(length(x)), log.(x))
    c = A \ log.(y)
    return c[2], exp(c[1])
end

println("")
@testset "Examples" begin
    @testset "Linear Poisson" begin
        test_poisson(poisson(1), 10, (i -> 2^i + 1,), ntuple(i -> true, 1))
        test_poisson(poisson(1), 100, (i -> 20 * i,), (false,))

        test_poisson(poisson(2), 4, (i -> 2^i + 1, i -> 2^i + 2), ntuple(i -> true, 2))
        test_poisson(poisson(2), 7, (i -> 2^i + 1, i -> 2^i + 2), ntuple(i -> false, 2))

        test_poisson(poisson(3), 5, (i -> 2^i + 1, i -> 2^i + 2, i -> 2^i + 3), ntuple(i -> true, 3))
    end
    @testset "Nonlinear Poisson" begin
        test_poisson_nl(poisson_nl(1), 10, (i -> 2^i + 1,), ntuple(i -> true, 1))
        test_poisson_nl(poisson_nl(1), 10, (i -> 2^i + 1,), ntuple(i -> false, 1))

        test_poisson_nl(poisson_nl(2), 5, (i -> 2^i + 1, i -> 2^i + 2), ntuple(i -> true, 2))

        test_poisson_nl(poisson_nl(3), 5, (i -> 2^i + 1, i -> 2^i + 2, i -> 2^i + 1), ntuple(i -> true, 3))
    end
    @testset "Convection-diffusion" begin
        test_conv_diff(convection_diffusion(1), 10, (i -> 2^i + 1,), ntuple(i -> true, 1))
        test_conv_diff(convection_diffusion(1), 100, (i -> 20 * i,), (false,))

        test_conv_diff(convection_diffusion(2), 5, (i -> 2^i + 1, i -> 2^i + 2), ntuple(i -> true, 2))
        test_conv_diff(convection_diffusion(2), 8, (i -> 2^i + 1, i -> 2^i + 2), ntuple(i -> false, 2))

        test_conv_diff(convection_diffusion(3), 5,
            (i -> 2^i + 1, i -> 2^i + 2, i -> 2^i + 3), ntuple(i -> true, 3))
        #test_conv_diff(convection_diffusion(3), 6, (i->2^i+1, i->2^i+2, i->2^i+1), ntuple(i->false, 3)) # the linear solver takes a while to solve
    end
    @testset "Reaction-diffusion" begin
        grids = [8, 16, 32]
        u1_errs = Float64[]
        u2_errs = Float64[]
        u3_errs = Float64[]
        hs = Float64[]
        for N in grids
            h, e1, e2, e3 = solve_reaction_diffusion(N)
            push!(hs, h)
            push!(u1_errs, e1)
            push!(u2_errs, e2)
            push!(u3_errs, e3)
        end

        order_u1, _ = least_squares_fit(hs, u1_errs)
        order_u2, _ = least_squares_fit(hs, u2_errs)
        order_u3, _ = least_squares_fit(hs, u3_errs)

        @test isapprox(order_u1, 2.0, atol = 0.2)
        @test isapprox(order_u2, 2.0, atol = 0.2)
        @test isapprox(order_u3, 2.0, atol = 0.2)
    end
end
