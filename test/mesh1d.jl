import Bramble: indices, npoints, dim, hspaceit, hmeanit, ndofs

function mesh1d_tests()
    I = Interval(-1.0, 4.0)

    N = 4
    X = Domain(I)
    mesh = Mesh(X, N, true)

    @test validate_equal(length(indices(mesh)), N)
    @test validate_equal(ndofs(mesh), N)
    @test dim(mesh) == 1

    h = collect(hspaceit(mesh))
    @test @views validate_equal(diff(h[2:N]), 0.0)

    hmed = collect(hmeanit(mesh))
    @test @views validate_equal(diff(hmed[2:N-1]), 0.0)
end

mesh1d_tests()