import Bramble: dim, indices, projection, ndofs

function meshnd_tests()
    I = interval(-1.0, 1.0)
    I2 = interval(-2.0, 2.0)
    I3 = interval(-3.0, 3.0)

    set = (I, I × I2, I × I2 × I3)
    npts = (3, (3, 4), (3, 4, 5))
    N = prod.(npts)
    unif = (true, (true, false), (true, false, true))

    for D in 1:3
        Ω = domain(set[D])
        Ωₕ = mesh(Ω, npts[D], unif[D])
        @test(length(indices(Ωₕ)) == N[D])

        @test(dim(Ωₕ) == D)
        @test(ndofs(Ωₕ) == N[D])

        for i = 1:D
            @test(ndofs(Ωₕ(i)) == npts[D][i])
        end
    end
end

meshnd_tests()