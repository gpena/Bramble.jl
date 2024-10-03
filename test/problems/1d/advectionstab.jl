using Bramble

function advection_1d_stab(::Val{1})
    I = Interval(0.0, 1.0)
    X = Domain(I)

    ϵ = 1.0 / 50.0
    b = 1.0

    sol(x) = (exp(b * x[1] / ϵ) - 1.0) / (exp(b / ϵ) - 1.0)

    bc = dirichletbcs(sol)

    nTests = 7
    error = zeros(nTests)
    _hmax = zeros(nTests)

    for i in 1:nTests
        N = 2^(i + 6)

        mesh = Mesh(X, N, true)
        _hmax[i] = hₘₐₓ(mesh)

        Wh = GridSpace(mesh)
        u = Element(Wh)

        gh = Element(Wh, 0.0)
        l(V) = mass(gh, V)

        lform = LinearForm(l, Wh)
        F = assemble(lform, bc)

        gh.values .= collect(Bramble.hspaceit(mesh)) .* b

        # stabilization by jump of the gradient: expected order h^2 for ||_h and h^1.5 for ||D_x||_+
        a(U, V) = inner₊(ϵ * D₋ₓ(U) - b * M₋ₕ(U), D₋ₓ(V)) + innerₕ(gh * jumpₕ(D₋ₓ(U)), jumpₕ(D₋ₓ(V)))
        bform = BilinearForm(a, Wh, Wh)
        mat = assemble(bform, bc)
        
        solve!(u, mat, F)
        Rₕ!(gh, sol)
        gh.values .-= u.values
        error[i] = norm₁ₕ(gh)
    end

    order, _ = leastsquares(log.(_hmax), log.(error))
    @test(abs(order - 1.5) < 0.4)
end


function advection_1d_stab(::Val{2})
    I = Interval(0.0, 1.0)
    X = Domain(I)

    ϵ = 1.0 / 50.0
    b = 1.0

    __sol((b, ϵ),x) = (exp(b * x[1] / ϵ) - 1.0) / (exp(b / ϵ) - 1.0)
    sol = Base.Fix1(__sol, (b,ϵ))

    bc = dirichletbcs(sol)

    nTests = 7
    error = zeros(nTests)
    _hmax = zeros(nTests)

    for i in 1:nTests
        N = 2^(i + 6)

        mesh = Mesh(X, N, true)
        _hmax[i] = hₘₐₓ(mesh)

        Wh = GridSpace(mesh)
        u = Element(Wh)
        gh = Element(Wh, 0.0)

        u.values .= collect(Bramble.hspaceit(mesh)) .* b .+ ϵ

        # stabilization by artificial diffusion: expected order h
        a(U, V) = inner₊(-b * M₋ₕ(U) + u * D₋ₓ(U), D₋ₓ(V))
        bform = BilinearForm(a, Wh, Wh)
        mat = assemble(bform, bc)
        l(V) = mass(gh, V)

        lform = LinearForm(l, Wh)
        F = assemble(lform, bc)

        solve!(u, mat, F)
        Rₕ!(gh, sol)

        gh.values .-= u.values
        error[i] = norm₁ₕ(gh)
    end

    order, _ = leastsquares(log.(_hmax), log.(error))
    @test(abs(order - 1.0) < 0.35)
end

for i in 1:2
    advection_1d_stab(Val(i))
end