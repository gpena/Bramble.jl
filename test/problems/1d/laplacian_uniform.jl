using Bramble

# test for -u'' = f, with homogeneous Dirichlet bc
# with u(x) = sin(π*x) as solution in [0,1]

const laplacian1d_sinpi_problem = Bramble.Laplacian(Domain(Interval(0.0, 1.0)), x->sin(π * x), x-> π^2 * sin(π * x))

function lap_1d_uniform()
    nTests = 10
    error = zeros(nTests)
    h = zeros(nTests)

    for i = 1:nTests
        h[i], error[i] = Bramble.solveLaplacian(laplacian1d_sinpi_problem, (2^(i + 2),), (true,))
    end

    order, _ = leastsquares(log.(h), log.(error))
    @test(abs(order - 2.0) < 0.3)
end

lap_1d_uniform()
