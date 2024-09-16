using Bramble

# test for -u_xx - u_yy = f, with homogeneous Dirichlet bc
# with u(x,y) = sin(π*x)*cos(πy) as solution in [0,1]×[0,1]

function lap_2d_uniform()
    I = Interval(0.0, 1.0)

    Ω = Domain(I×I)

    sol(x) = sin(π * x[1])*cos(π*x[2])
    g(x) = 2.0*π^2 * sol(x)

    problem = Bramble.Laplacian(Ω, sol, g)

    nTests = 6
    error = zeros(nTests)
    hmax = zeros(nTests)

    for i = 1:nTests
        hmax[i], error[i] = Bramble.solveLaplacian(problem, (2^(i+1), 2^(i+2)), (false, false))
    end

    # some least squares fitting
    order, _ = leastsquares(log.(hmax), log.(error))
    @test(abs(order - 2.0) < 0.4)
end

lap_2d_uniform()