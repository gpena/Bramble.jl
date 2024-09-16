using Bramble

# test for -u_xx - u_yy - u_zz = f, with homogeneous Dirichlet bc
# with u(x,y,z) = exp(x+y+z) as solution in [0,1]×[0,1]×[0,1]

function lap_3d_uniform()
    I = Interval(0.0, 1.0)

    Ω = Domain(I×I×I)

    sol(x) = exp(sum(x))
    g(x) = -3.0*sol(x)

    problem = Bramble.Laplacian(Ω, sol, g)

    nTests = 6
    error = zeros(nTests)
    hmax = zeros(nTests)

    for i = 1:nTests
        hmax[i], error[i] = Bramble.solveLaplacian(problem, (2^i +2, 2^i +1, 2^i+3), (false, false, false))
    end

    mask = (!isnan).(error) 
    err2 = error[mask]
    hmax2 = hmax[mask]

    # some least squares fitting
    order, _ = leastsquares(log.(hmax2[3:end]), log.(err2[3:end]))
    @test(abs(order - 2.0) < 0.4)
end

lap_3d_uniform()
