using Bramble

function bilinearforms_test()
    N = 3
    I = interval(-1.0, 4.0)

    X = domain(I, create_markers( :bc => @embed(I, x -> x[1]-4)))
    Mh = mesh(X, N, false)

    Wh = gridspace(Mh)

    a(U, V) = innerₕ(U, V)
    bform = form(Wh, Wh, a)
    A = assemble(bform)

    bform2 = form(Wh, Wh, (U, V) -> inner₊(D₋ₓ(U), D₋ₓ(V)))
    assemble(bform2)

    bform3 = form(Wh, Wh, (U, V) -> inner₊(M₋ₕ(U), D₋ₓ(V)))
    assemble(bform3)
    
    gh = element(Wh)
    Rₕ!(gh, x->sin(x[1]))

    l(V) = innerₕ(gh, V)
    lform = form(Wh, l)
    F = assemble(lform)

    u = A\F

    @test validate_equal( u, sin.(points(Mh)))
    
end

bilinearforms_test()