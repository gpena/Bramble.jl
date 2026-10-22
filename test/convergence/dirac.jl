module ConvergenceDiracTests

# Second order of the point source `dirac`: its contraction against a smooth grid function,
# and the Poisson problem it drives against the closed-form Green's function. Moved here
# from test/form/dirac.jl.

using Test
using Bramble
using Bramble: dirac, interpolate_at, points, stepsize

@testset "1D contraction" begin
    strength = 2.5
    phi = x -> sin(2π * x[1]) + exp(x[1])
    # O(h²) convergence of Dirac functional contraction under mesh refinement
    errors = Float64[]
    for N in (21, 41, 81)
        Ω_ref = mesh(domain(interval(0.0, 1.0)), N, true)
        W_ref = gridspace(Ω_ref)
        # Sample point at cell center (t = 0.5) for clean asymptotic scaling
        x_mid = points(Ω_ref)[length(points(Ω_ref)) ÷ 3] + 0.5 * stepsize(Ω_ref)
        l_r = form(W_ref, v -> innerₕ(dirac(x_mid, strength), v))
        phi_r = Rₕ(W_ref, phi)
        push!(errors, abs(l_r(phi_r) - strength * phi((x_mid,))))
    end
    rate = log(errors[1] / errors[end]) / log(4.0)
    @test rate > 1.9 # 2nd order convergence
end

# Poisson problem with a Dirac delta source.
@testset "Poisson: 2D Green function" begin
    # -Δu = S δ(x - x0, y - y0) on (0,1)², u = 0 on ∂Ω. Separation of variables gives a
    # sine series in x with a closed-form (sinh) Green's function in y for each mode --
    # exponentially convergent in the number of terms, unlike a raw double sine series:
    #
    #   G(x, y; x0, y0) = Σₙ 2 sin(nπx) sin(nπx0) sinh(nπ y<) sinh(nπ (1 - y>)) /
    #                     (nπ sinh(nπ))
    #
    # with y< = min(y, y0), y> = max(y, y0). Standard eigenfunction-expansion result for
    # the Dirichlet Laplacian Green's function on a rectangle.
    function green2d(x, y, x0, y0; n_terms = 60)
        total = 0.0
        ylo, yhi = min(y, y0), max(y, y0)
        for n in 1:n_terms
            k = n * pi
            total += 2 * sin(k * x) * sin(k * x0) * sinh(k * ylo) * sinh(k * (1 - yhi)) /
                     (k * sinh(k))
        end
        return total
    end

    x0, y0 = 0.35, 0.65
    S = 3.0
    eval_pts = ((0.15, 0.15), (0.8, 0.2), (0.5, 0.9))  # away from the singularity

    errors = [Float64[] for _ in eval_pts]
    for N in (21, 41, 81)
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (N, N), (true, true))
        Wₕ = gridspace(Ωₕ)

        a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
        l = form(Wₕ, v -> innerₕ(dirac((x0, y0), S), v))
        A, F = assemble(a, l; dirichlet = :boundary => x -> 0.0)

        uₕ = element(Wₕ)
        uₕ .= A \ F

        for (i, p) in enumerate(eval_pts)
            u_exact = S * green2d(p[1], p[2], x0, y0)
            push!(errors[i], abs(interpolate_at(uₕ, p) - u_exact))
        end
    end

    for e in errors
        rate = log(e[1] / e[end]) / log(4.0)
        @test rate > 1.8  # 2nd order convergence away from the singularity
    end
end

end # module ConvergenceDiracTests
