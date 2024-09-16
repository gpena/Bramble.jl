using Bramble
using LinearAlgebra: \, factorize
using BenchmarkTools
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
    bcs(t) = dirichletbcs(p.solution(t))
    return Wh, dt, _dt, nTimeSteps, coeff, bcs
end


function solveWave(p::Wave1D, nPoints::Int, unif::Bool)
    Wh, dt, h, nTimeSteps, coeff, bcs = init_wave1d(p, nPoints, unif)

    a =  (U,V) -> innerₕ(U, V) + coeff * inner₊(D₋ₓ(U), D₋ₓ(V))
    bform = BilinearForm(a, Wh, Wh)
    A = assemble(bform, bcs(2.0*dt))
    da = factorize(A)

    uh = Rₕ(Wh, p.solution(dt))
    un = Rₕ(Wh, p.solution(0.0))
    aux = Element(Wh, 0.0)
    @.. aux.values = 2 * uh.values - un.values
    rhu = Element(Wh)

    error = 0.0
    l = v -> innerₕ(aux, v) - coeff * inner₊(D₋ₓ(un), D₋ₓ(v))
    lform = LinearForm(l, Wh)
    F = assemble(lform)
    
    for n in 2:nTimeSteps
        @.. aux.values = 2 * uh.values - un.values
        f = p.solution(n*dt)
        assemble!(F, lform, bcs(n*dt))

        copyto!(un, uh)
        copyto!(uh, da \ F)
        Rₕ!(rhu, f)
        @.. rhu.values -= uh.values
        error += norm₁ₕ(rhu; buffer = aux)^2
    end

    error = sqrt(dt * error)
    return h, error
end

function wave_1d()
    Ω = Domain(Interval(0.0, 1.0))

    c = 1.0
    T = 1.0

    sol(t) = x -> sin(π * x[1]) * cos(π * c * t) + (x[1] + 1.0) * t

    problem = Wave1D(Ω, sol, c, (0.0, T))

    nTests = 8
    hmax = zeros(nTests)
    error = zeros(nTests)
    #i=1
    for i in 1:nTests
        hmax[i], error[i] = solveWave(problem, 2^(i + 3), true)
    end
    
    #order, _ = leastsquares(log.(hmax), log.(error))
    #@test(abs(order - 2.0) < 0.3)
end

wave_1d()
#@code_warntype wave_1d()
@benchmark wave_1d()





Ω = Domain(Interval(0.0, 1.0))
c = 1.0
T = 1.0
sol(t) = x -> sin(π * x[1]) * cos(π * c * t) + (x[1] + 1.0) * t
problem = Wave1D(Ω, sol, c, (0.0, T))
mesh = Mesh(problem.intX, 100, true)
Wh = GridSpace(mesh)
u = Element(Wh)
v = Element(Wh)
F = similar(u.values)
lform = LinearForm(v -> (innerₕ(u, v) - 2.0 * inner₊(D₋ₓ(u), D₋ₓ(v))), Wh)
bcs(t) = dirichletbcs(problem.solution(t))
assemble!(F, lform, bcs(1.0))
@code_warntype assemble!(F, lform)
@benchmark assemble!($F, $lform, $(bcs(1.0)))


@code_warntype snorm₁ₕ(u, buffer = v)
snorm₁ₕ(u, buffer = v)
@benchmark snorm₁ₕ($u, buffer = $v)


#@code_warntype Wave1D(Ω, sol, c, (0.0, T))
solveWave(problem, 100, true)
@benchmark solveWave($problem, $100, $true)