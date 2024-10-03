using Bramble

function advection_1d(::Val{1})
    I = Interval(0.0, 1.0)
    X = Domain(I)

    sol(x) = sin(π * x)
    solp(x) = π * cos(π * x)
    solpp(x) = -π^2 * sol(x)

    # linear advection problem -(3u)' = g
    f(u) = 3.0 * u
    fp(u) = 3.0

    g(x) = -fp(sol(x)) * solp(x)
    bc = dirichletbcs(sol)

    nTests = 10
    error = zeros(nTests)
    _hmax = zeros(nTests)

    for i = 1:nTests
        N = 2^(i + 2)

        mesh = Mesh(X, N, true)
        _hmax[i] = hₘₐₓ(mesh)

        Wh = GridSpace(mesh)
        u = Element(Wh)

        gh = avgₕ(Wh, g)

        bform = BilinearForm((U,V) -> advection(U, V; scaling = 3.0), Wh, Wh)
        mat = assemble(bform, bc)
        l(V) = mass(gh, V)

        lform = LinearForm(l, Wh)
        F = assemble(lform, bc)

        solve!(u, mat, F)
        Rₕ!(gh, sol)
        u.values .-= gh.values

        error[i] = norm₁ₕ(u)
    end

    order, _ = leastsquares(log.(_hmax), log.(error))
    @test(abs(order - 2.0) < 0.4)
end

#########################################################################################
function advection_1d(::Val{2})
    I = Interval(0.0, 1.0)
    X = Domain(I)

    sol(x) = sin(π * x)
    solp(x) = π * cos(π * x)
    solpp(x) = -π^2 * sol(x)

    # linear advection problem -(3u)' = g
    f(u) = 3.0 * u
    fp(u) = 3.0

    g(x) = -fp(sol(x)) * solp(x)
    bc = dirichletbcs(sol)

    nTests = 10
    error = zeros(nTests)
    _hmax = zeros(nTests)
    a(U, V) = inner₊(M₋ₕ(f(U)), D₋ₓ(V))

    for i = 1:nTests
        N = 2^(i + 2)

        mesh = Mesh(X, N, true)
        _hmax[i] = hₘₐₓ(mesh)

        Wh = GridSpace(mesh)
        u = Element(Wh)

        gh = avgₕ(Wh, g)

        bform = BilinearForm(a, Wh, Wh)
        mat = assemble(bform, bc)
        l(V) = innerₕ(gh, V)

        lform = LinearForm(l, Wh)
        F = assemble(lform, bc)

        solve!(u, mat, F)
        Rₕ!(gh, sol)
        u.values .-= gh.values

        error[i] = norm₁ₕ(u)
    end

    order, _ = leastsquares(log.(_hmax), log.(error))
    @test(abs(order - 2.0) < 0.4)
end





#########################################################################################
# advection-diffusion problem -((1+x)u')' -((x+5)u)' = g
function advection_1d(::Val{3})
    I = Interval(0.0, 1.0)
    X = Domain(I)

    sol(x) = sin(π * x[1])
    solp(x) = π * cos(π * x[1])
    solpp(x) = -π^2 * sol(x)

    # linear advection problem -(3u)' = g
    #f(u) = 3.0 * u
    #fp(u) = 3.0

    A(x) = 1.0 + x[1]
    b(x) = x[1] + 5.0

    @. g(x) = -solp(x) - A(x) * solpp(x) - sol(x) - b(x) * solp(x)
    bc = dirichletbcs(sol)

    nTests = 10
    error = zeros(nTests)
    _hmax = zeros(nTests)

    for i = 1:nTests
        N = 2^(i + 5)

        mesh = Mesh(X, N, false)
        _hmax[i] = hₘₐₓ(mesh)

        Wh = GridSpace(mesh)
        u = Rₕ(Wh,A)
        gh = Rₕ(Wh,b)

        a(U,V) = inner₊(M₋ₕ(u)*D₋ₓ(U) + M₋ₕ(gh)*M₋ₕ(U),D₋ₓ(V))
        bform = BilinearForm(a, Wh, Wh)
        mat = assemble(bform, bc)

        avgₕ!(gh, g)

        l(V) = innerₕ(gh, V)
        lform = LinearForm(l, Wh)
        F = assemble(lform, bc)

        solve!(u, mat, F)

        Rₕ!(gh, sol)
        u.values .-= gh.values

        error[i] = norm₁ₕ(u)
    end

    order, _ = leastsquares(log.(_hmax), log.(error))
    @test(abs(order - 2.0) < 0.3)
end


#########################################################################################
function advection_1d(::Val{4})
    I = Interval(0.0, 1.0)
    X = Domain(I)

    sol(x) = sin(π * x)
    solp(x) = π * cos(π * x)
    solpp(x) = -π^2 * sol(x)

    # linear advection problem -(3u)' = g
    f(u) = 3.0 * u
    fp(u) = 3.0

    A(x) = 1.0 + x[1]
    b(x) = x[1] + 5.0

    g(x) = -solp(x) - A(x) * solpp(x) - sol(x) - b(x) * solp(x)
    bc = dirichletbcs(sol)

    nTests = 10
    error = zeros(nTests)
    _hmax = zeros(nTests)

    for i = 1:nTests
        N = 2^(i + 2)

        mesh = Mesh(X, N, false)
        _hmax[i] = hₘₐₓ(mesh)

        Wh = GridSpace(mesh)
        a(U, V) = inner₊(M₋ₕ(Rₕ(Wh, A)) * D₋ₓ(U), D₋ₓ(V)) + inner₊(M₋ₕ(Rₕ(Wh, b) * U), D₋ₓ(V))

        gh = avgₕ(Wh, g)

        bform = BilinearForm(a, Wh, Wh)
        mat = assemble(bform, bc)

        l(V) = innerₕ(gh, V)
        lform = LinearForm(l, Wh)
        F = assemble(lform, bc)

        u = Element(Wh)
        solve!(u, mat, F)

        Rₕ!(gh, sol)
        u.values .-= gh.values
        error[i] = norm₁ₕ(u)
    end

    order, _ = leastsquares(log.(_hmax), log.(error))
    @test(abs(order - 2.0) < 0.4)
end


for i in 1:4
    advection_1d(Val(i))
end