using Bramble
using BenchmarkTools
using LinearSolve

#=
    Test for ∂^2 u \ ∂t^2 = c^2 ∂^2 u \ ∂x^2 in (0,T]

    with homogeneous Dirichlet boundary conditions
    and initial data 

        u(x,0) = ϕ(x)
        u_t(x,0) = ψ(x)

=#


struct Wave1D{D<:Domain,T<:AbstractFloat,F<:Function}
    intX::D
    solution::F
    c::T
    intT::Tuple{T,T}
end

function init_wave1d(p::Wave1D, nPoints::Int, unif)
    mesh = Mesh(p.intX, nPoints, unif)
    Wh = GridSpace(mesh)
    _dt = hₘₐₓ(mesh)
    T = p.intT[2]
    nTimeSteps = Int(round(T / _dt))
    dt = T / nTimeSteps
    coeff = 0.5 * (dt^2) * (p.c^2.0)
    bcs(t) =  dirichletbcs(p.solution(t))
    return Wh, dt, _dt, nTimeSteps, coeff, bcs
end


function solveWave(p::Wave1D, nPoints::Int, unif)
    Wh, dt, h, nTimeSteps, coeff, bcs = init_wave1d(p, nPoints, unif)

    a(U,V) = innerₕ(U, V) + coeff * inner₊(D₋ₓ(U), D₋ₓ(V))
    bform = BilinearForm(a, Wh, Wh)
    A = assemble(bform, bcs(2.0*dt))
    #da = factorize(A)

    uh = Rₕ(Wh, p.solution(dt))
    un = Rₕ(Wh, p.solution(0.0))

    rhu = Element(Wh)

    error = 0.0

    #l =  v -> mass(2.0*uh - un, v) - coeff*stiffness(un, v)#
    l(v) = innerₕ( 2.0 * uh - un, v) - coeff * inner₊(D₋ₓ(un), D₋ₓ(v))
    lform = LinearForm(l, Wh)
    F = assemble(lform)

    prob = LinearProblem(A, F)
    linsolve = init(prob)
    linsolve.b = F

    for n in 2:nTimeSteps
        f = p.solution(n*dt)
        assemble!(F, lform, bcs(n*dt))
        

        un.values .= uh.values
        uh.values .= LinearSolve.solve!(linsolve)

        Rₕ!(rhu, f)
        rhu.values .-= uh.values
        error += norm₁ₕ(rhu)^2
    end

    error = sqrt(dt * error)
    return h, error
end

function wave_1d()
    Ω = Domain(Interval(0.0, 1.0))

    c = 1.0
    T = 1.0

    sol(t) = x -> sin(π * x[1]) * cos(π * c * t) + (x[1] + 1) * t

    problem = Wave1D(Ω, sol, c, (0.0, T))

    nTests = 8
    hmax = zeros(nTests)
    error = zeros(nTests)

    for i in 1:nTests
        hmax[i], error[i] = solveWave(problem, 2^(i + 3), true)
    end
    
    order, _ = leastsquares(log.(hmax), log.(error))
    @test(abs(order - 2.0) < 0.3)
end

wave_1d()
#@btime wave_1d()