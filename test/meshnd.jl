import Bramble: dim, indices, projection, ndofs

function meshnd_tests()
    I = Interval(-1.0, 1.0)
    I2 = Interval(-2.0, 2.0)
    I3 = Interval(-3.0, 3.0)

    set = (I, I × I2, I × I2 × I3)
    npts = (3, (3, 4), (3, 4, 5))
    N = prod.(npts)
    unif = (true, (true, false), (true, false, true))

    for D in 1:3
        X = Domain(set[D])
        mesh = Mesh(X, npts[D], unif[D])
        @test(length(indices(mesh)) == N[D])

        @test(dim(mesh) == D)
        @test(ndofs(mesh) == N[D])

        for i = 1:D
            @test(ndofs(mesh(i)) == npts[D][i])
        end
    end
end

meshnd_tests()