using Bramble

function bilinearforms_test()
    N = 3
    I = interval(-1.0, 4.0)

    X = domain(I, create_markers( :bc => ↪(I, x -> x[1]-4)))
    Mh = mesh(X, N, false)

    Wh = gridspace(Mh)

    a(U, V) = innerₕ(U, V)
    bform = BilinearForm(a, Wh, Wh)
    A = assemble(bform)

    bform2 = BilinearForm((U, V) -> inner₊(D₋ₓ(U), D₋ₓ(V)), Wh, Wh)
    assemble(bform2)

    bform3 = BilinearForm((U, V) -> inner₊(Mₕ(U), D₋ₓ(V)), Wh, Wh)
    assemble(bform3)
    
    gh = element(Wh)
    Rₕ!(gh, x->sin(x[1]))

    l(V) = innerₕ(gh, V)
    lform = LinearForm(l, Wh)
    F = assemble(lform)

    u = A\F

    @test validate_equal( u, sin.(points(Mh)))
    
end

bilinearforms_test()