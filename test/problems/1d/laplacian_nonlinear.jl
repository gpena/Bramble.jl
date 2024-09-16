using Bramble

function fixed_point!(mat, F, bform, bc, uold, u, A)
    for _ in 1:200
         uold.values .= A.(Mₕ(u).values)
        update!(bform, uold)
        assemble!(mat, bform, bc)

        copyto!(uold, u)
        solve!(u, mat, F)
        uold.values .-= u.values
        if norm₁ₕ(uold) < 1e-6
            break
        end
    end
end

function laplacian_nl(i, bc, A, p)
    N = 2^(i + 2)

    mesh = Mesh(p.X, N, true)

    Wh = GridSpace(mesh)
    u = Element(Wh, 0.0)

    uold = similar(u)
    avgₕ!(uold, p.f)

    #l(V) = innerₕ(uold, V)
    lform = Mass(Wh)#LinearForm(l, Wh)
    update!(lform, uold)
    F = assemble(lform, bc)

    #a(U, V) = inner₊(A(Mₕ(u)) * D₋ₓ(U), D₋ₓ(V))
    #bform = BilinearForm(a, Wh, Wh)
    bform = Diff(Wh, Wh)
    update!(bform, A(Mₕ(u)))
    mat = assemble(bform)

    fixed_point!(mat, F, bform, bc, uold, u, A)

    Rₕ!(uold, p.u)
    u .-= uold

    return hₘₐₓ(mesh), norm₁ₕ(u)
end

function lap_1d_nonlinear()
    I = Interval(0.0, 1.0)
    X = Domain(I)

    sol(x) = sin(π * x)
    solp(x) = π * cos(π * x)
    solpp(x) = -π^2 * sol(x)

    A(u) = 3.0 + 1.0 / (1.0 + u^2.0)
    Ap(u) = -2.0 * u / (1.0 + u^2.0)^2.0

    g(x) = -Ap(sol(x)) * (solp(x))^2 - A(sol(x)) * solpp(x)
    bc = dirichletbcs(sol)

    problem = Bramble.Laplacian(X, sol, g)

    nTests = 7
    error = zeros(nTests)
    _hmax = zeros(nTests)

    for i = 1:nTests
        _hmax[i], error[i] = laplacian_nl(i, bc, A, problem)
    end

    order, _ = leastsquares(log.(_hmax), log.(error))
    @test(abs(order - 2.0) < 0.4)
end

lap_1d_nonlinear()
