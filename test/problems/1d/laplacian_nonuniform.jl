using Bramble

# test for -u'' = f, with homogeneous Dirichlet bc
# with u(x) = sin(π*x) as solution in [0,1]

function lap_nonunif()
    nTests = 100
    error = zeros(nTests)
    h = zeros(nTests)

    for i = 1:nTests
        h[i], error[i] = Bramble.solveLaplacian(laplacian1d_sinpi_problem, (i*20,), (false,))
    end

    # some least squares fitting
    order, _ = leastsquares(log.(h), log.(error))
    @test(abs(order - 2.0) < 0.3)
end

lap_nonunif()