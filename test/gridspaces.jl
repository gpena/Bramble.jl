function gridspaces_tests()
    I = Interval(-1.0, 4.0)

    m = markers("Dirichlet" => x->x[1]-1.0)
    X = Domain(I, m)
    mesh = Mesh(X, 4, false)

    Wh = GridSpace(mesh)
    vv = Bramble.Elements(Wh)
    @test validate_equal(length(vv.values), 4 * 4)
    @test validate_equal(length(vv), 4 * 4)
end

gridspaces_tests()