module ConvergenceReactionFluxTests

# Second order of the boundary flux `reaction` extracts, against the outward flux of
# `-Δu = f`, `u = sin(πx)` (1D) or `sin(πx)sin(πy)` (2D), worked out by hand. Moved here from
# test/form/reaction_flux.jl, which holds the convention and the conservation identities.

using Test
using Random
using Bramble
using Bramble: reaction

# Flux at each end, on uniform and non-uniform meshes.
@testset "1D flux: each end" begin
    # u = sin(πx), f = π² sin(πx), u'(x) = π cos(πx).
    # At x=0 the outward normal is -1, so ∂u/∂n = -u'(0) = -π and q·n = -∂u/∂n = π.
    # At x=1 the outward normal is +1, so ∂u/∂n = u'(1) = -π and q·n = π.
    # Heat generated in the interior leaves through both ends, by symmetry.
    sol(x) = sin(pi * x[1])
    src(x) = pi^2 * sin(pi * x[1])
    S = interval(0.0, 1.0)

    I = domain(S, :left => :xmin, :right => :xmax)

    # Uniform: the boundary flux is expected to converge at the same clean 2nd order
    # the interior discretization does (matches test/form/dirac.jl's own 1D rate check,
    # also uniform-only).
    errors_left = Float64[]
    errors_right = Float64[]
    for N in (21, 41, 81)
        Ωₕ = mesh(I, N, true)
        Wₕ = gridspace(Ωₕ)

        a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
        l = form(Wₕ, v -> innerₕ(Rₕ(Wₕ, src), v))
        A, F = assemble(a, l; dirichlet = :boundary => sol)

        uₕ = element(Wₕ)
        uₕ .= A \ F

        push!(errors_left, abs(reaction(a, l, uₕ; marker = :left) - pi))
        push!(errors_right, abs(reaction(a, l, uₕ; marker = :right) - pi))
    end
    @test log(errors_left[1] / errors_left[end]) / log(4.0) > 1.8
    @test log(errors_right[1] / errors_right[end]) / log(4.0) > 1.8

    # Non-uniform: random point placement can knock the boundary-flux extraction (a
    # different discrete quantity from the primary discretization) below a clean 2nd
    # order at any single seed, so only monotone convergence to a small error is
    # asserted here, not a specific rate. Seeded for reproducibility.
    Random.seed!(20260915)
    errors_left_nu = Float64[]
    errors_right_nu = Float64[]
    for N in (21, 41, 81)
        Ωₕ = mesh(I, N, false)
        Wₕ = gridspace(Ωₕ)

        a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
        l = form(Wₕ, v -> innerₕ(Rₕ(Wₕ, src), v))
        A, F = assemble(a, l; dirichlet = :boundary => sol)

        uₕ = element(Wₕ)
        uₕ .= A \ F

        push!(errors_left_nu, abs(reaction(a, l, uₕ; marker = :left) - pi))
        push!(errors_right_nu, abs(reaction(a, l, uₕ; marker = :right) - pi))
    end
    @test issorted(errors_left_nu; rev = true)
    @test issorted(errors_right_nu; rev = true)
    @test errors_left_nu[end] < 0.15 * errors_left_nu[1]
    @test errors_right_nu[end] < 0.15 * errors_right_nu[1]
end

# Every side carries the outward flux ∫₀¹ π sin(πy) dy = 2.
@testset "2D flux: each side" begin
    sol(x) = sin(pi * x[1]) * sin(pi * x[2])
    src(x) = 2 * pi^2 * sin(pi * x[1]) * sin(pi * x[2])
    S = interval(0.0, 1.0) × interval(0.0, 1.0)
    Ω = domain(S, :xmin => :xmin, :xmax => :xmax, :ymin => :ymin, :ymax => :ymax)

    errors = [Float64[] for _ in 1:4]
    sides = (:xmin, :xmax, :ymin, :ymax)
    for N in (11, 21, 41)
        Ωₕ = mesh(Ω, (N, N), (true, true))
        Wₕ = gridspace(Ωₕ)

        a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
        l = form(Wₕ, v -> innerₕ(Rₕ(Wₕ, src), v))
        A, F = assemble(a, l; dirichlet = :boundary => sol)

        uₕ = element(Wₕ)
        uₕ .= A \ F

        for (i, side) in enumerate(sides)
            push!(errors[i], abs(reaction(a, l, uₕ; marker = side) - 2.0))
        end
    end
    for e in errors
        rate = log(e[1] / e[end]) / log(4.0)
        @test rate > 1.8
    end
end

end # module ConvergenceReactionFluxTests
