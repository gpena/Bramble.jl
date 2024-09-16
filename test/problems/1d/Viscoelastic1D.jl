module Viscoelastic1D

using Test
using Bramble.Geometry
using Bramble.Meshes
using Bramble.Meshes: hmax, points
using Bramble.Spaces
using Bramble.Forms

using Bramble.Tools: leastsquares

using LinearAlgebra: \, factorize
using Plots

#=
    Test for ∂^2 u \ ∂t^2 = α_1 ∂^2 u \ ∂x^2 + α_2 ∂^3 u \ ∂x^2∂t + f(x,t) in (0,T]

    with homogeneous Dirichlet boundary conditions
    and initial data 

        u(x,0) = ϕ(x)
        u_t(x,0) = ψ(x)

=#


struct ViscoelasticWave1D{D,T<:AbstractFloat}
    intX::D
    solution::Function
    Rₕsf::Function
    coeff1::T
    coeff2::T
    intT::Tuple{T,T}
    phi::Function
    psi::Function
end

a = 0.0
b = 1.0

I = Interval(a, b)
markers = add_subdomains(x -> x - a, x -> x - b)
Ω = Domain(I, markers)

const α_1 = 2.0
const α_2 = 1.0
const T = 1.0

@. sol(x, t) = exp(t) * ((2.0 * x - 1.0)^4 - 1.0)
@. d2u_dt2(x, t) = sol(x, t)
@. d2u_dx2(x, t) = exp(t) * (48.0 * (2.0 * x - 1.0)^2)
@. d3u_dx2dt(x, t) = d2u_dx2(x, t)
@. g(x, t) = d2u_dt2(x, t) - α_1 * d2u_dx2(x, t) - α_2 * d3u_dx2dt(x, t)
ϕ(x) = sol(x, 0.0)
ψ(x) = sol(x, 0.0)

problem = ViscoelasticWave1D(Ω, sol, g, α_1, α_2, (0.0, T), ϕ, ψ)

function solveWave(p::ViscoelasticWave1D, nPoints::Int)
    mesh = Mesh(p.intX; nPoints = nPoints, uniform = true)
    Wh = GridSpace(mesh)
    dt = 0.1 * hmax(mesh)#^2

    T = p.intT[2]
    nTimeSteps = Int(round(T / dt) + 1)
    dt = T / nTimeSteps

    tnm1 = 2.0 * dt
    f(x) = p.solution(x, tnm1)
    bc = add_dirichlet_bc(f)

    bform = BilinearForm(
        (U, V) ->
            mass(U, V) + stiffness(U, V; scaling = (dt^2 * p.coeff1 + dt * p.coeff2 / 2.0)),
        Wh,
        Wh,
    )
    A = assemble(bform, bc)
    da = factorize(A)
    uh = Rₕ(Wh, x -> p.solution(x, dt))
    un = Rₕ(Wh, x -> p.solution(x, 0.0))

    Rₕu = Rₕ(Wh, f)
    error = 0.0


    gh = avgₕ(Wh, x -> p.Rₕsf(x, tnm1))
    #lform = LinearForm( v -> mass(2.0*uh-un + dt^2*gh,v) + (p.coeff2*dt/2.0)*stiffness(un,v), Wh)#innerplus(D_x(un),D_x(v))
    lform = LinearForm(
        v ->
            mass(2.0 * uh - un + dt^2 * gh, v) + (p.coeff2 * dt / 2.0) * stiffness(un, v),
        Wh,
    )#innerplus(D_x(un),D_x(v))
    F = assemble(lform)
    tnm1 = dt

    for n = 2:nTimeSteps
        tnm1 += dt
        avgₕ!(gh, x -> p.Rₕsf(x, tnm1))

        assemble!(F, lform, bc)
        copyto!(un, uh)
        copyto!(uh, da \ F)
        Rₕ!(Rₕu, f)

        error = max(error, norm1h(uh - Rₕu))
    end

    return hmax(mesh), error, points(mesh), uh.values
end

h, err, x, y = solveWave(problem, 50)


nTests = 7
_hmax = zeros(nTests)
_error = zeros(nTests)

for i = 1:nTests
    _hmax[i], _error[i] = solveWave(problem, 2^(i + 2))
end

order, _ = leastsquares(log.(_hmax), log.(_error))
@test(abs(order - 2.0) < 0.3)


_, _, x, y = solveWave(problem, 10)

using Plots
scatter(x, y, label = "approx")
plot!(x, sol.(x, T), label = "exact")

scatter(log.(_hmax), log.(_error))
end
