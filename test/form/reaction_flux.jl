module FormReactionFluxTests

using Test
using Random
using Bramble
using Bramble: reaction, reaction_density, weights

# `reaction` (gpena/Bramble.jl#227) extracts the boundary flux a Dirichlet constraint had
# to supply, from the *unconstrained* operator/load and the already-solved uₕ: r = A uₕ - F
# is ≈ 0 on interior rows and, on a constrained row, exactly the discrete flux there.
#
# Sign/scaling convention (see src/postprocessing/reaction.jl): `reaction` returns `-r` summed over
# the marker -- positive is flux leaving the domain along the outward normal, so summing
# over every boundary marker recovers the net source `∫_Ω f` directly, no rescaling.
#
# Manufactured problem used throughout: -Δu = f, u = 0 on ∂Ω, u = sin(πx) (1D) or
# sin(πx)sin(πy) (2D) / sin(πx)sin(πy)sin(πz) (3D). Physical outward flux q·n = -∂u/∂n is
# worked out by hand for each side below, not merely asserted.

@testset "Reaction / boundary flux (#227)" begin
    @testset "1D: flux at each end, uniform and non-uniform" begin
        # u = sin(πx), f = π² sin(πx). u'(x) = π cos(πx).
        # At x=0 (outward normal -1): ∂u/∂n = -u'(0) = -π, so q·n = -∂u/∂n = π.
        # At x=1 (outward normal +1): ∂u/∂n = u'(1) = -π, so q·n = π.
        # Both ends: heat generated in the interior leaves through both ends, by symmetry.
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

    @testset "2D: flux on each side, conservation to round-off" begin
        # u = sin(πx)sin(πy), f = 2π² sin(πx)sin(πy). By symmetry every side carries the
        # same outward flux: ∫₀¹ π sin(πy) dy = 2, so each of the 4 sides gives 2, and the
        # full boundary sums to 8 = ∫∫ f = 2π² · (2/π) · (2/π).
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

        # Global conservation is an exact discrete identity (interior and constrained rows
        # of the unconstrained/constrained systems coincide away from the boundary, so this
        # holds to round-off at *any* single resolution, not merely in the refinement
        # limit -- unlike the per-side convergence above).
        Ωₕ = mesh(Ω, (21, 21), (true, true))
        Wₕ = gridspace(Ωₕ)
        a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
        l = form(Wₕ, v -> innerₕ(Rₕ(Wₕ, src), v))
        A, F = assemble(a, l; dirichlet = :boundary => sol)
        uₕ = element(Wₕ)
        uₕ .= A \ F

        total_src = sum(assemble(l))
        @test reaction(a, l, uₕ; marker = :boundary) ≈ total_src atol = 1e-9
        @test sum(reaction(a, l, uₕ; marker = s) for s in sides) ≈ total_src atol = 1e-9
    end

    @testset "3D: net flux exists and matches a manufactured solution" begin
        sol(x) = sin(pi * x[1]) * sin(pi * x[2]) * sin(pi * x[3])
        src(x) = 3 * pi^2 * sol(x)
        S = interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0)
        Ω = domain(S, :xmin => :xmin)

        Ωₕ = mesh(Ω, (15, 15, 15), (true, true, true))
        Wₕ = gridspace(Ωₕ)

        a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
        l = form(Wₕ, v -> innerₕ(Rₕ(Wₕ, src), v))
        A, F = assemble(a, l; dirichlet = :boundary => sol)

        uₕ = element(Wₕ)
        uₕ .= A \ F

        # ∫∫ π sin(πy)sin(πz) dy dz over one face = π (2/π)² = 4/π; not asserted to tight
        # tolerance on a coarse 3D mesh, only that it is the right order of magnitude and
        # sums (with the other 5 faces, by symmetry each equal) to the net source.
        r_face = reaction(a, l, uₕ; marker = :xmin)
        @test r_face ≈ 4 / pi rtol = 0.05

        total_src = sum(assemble(l))
        @test reaction(a, l, uₕ; marker = :boundary) ≈ total_src atol = 1e-8
    end

    @testset "Overlapping markers counted once" begin
        # Synthetic A/F rather than a solved PDE: a manufactured solution's own corner
        # residual can happen to be near zero (as `sin(πx)sin(πy)` does at the origin),
        # which would make a double-counting bug undetectable by coincidence. Here every
        # marked point gets a distinct, known value, so inclusion-exclusion is checked by
        # direct arithmetic instead.
        S = interval(0.0, 1.0) × interval(0.0, 1.0)
        Ω = domain(S, :xmin => :xmin, :ymin => :ymin)
        Ωₕ = mesh(Ω, (5, 5), (true, true))
        Wₕ = gridspace(Ωₕ)
        n = ndofs(Wₕ)

        # `points(Ωₕ)` on an ND mesh returns one coordinate vector per axis, not a flat
        # vector of grid points -- the shared corner's own DOF index is found from the
        # mask intersection instead, mesh-representation agnostic.
        mask_corner = Bramble.index_in_marker(Ωₕ, :xmin) .& Bramble.index_in_marker(Ωₕ, :ymin)
        idx = findfirst(mask_corner)
        @test idx !== nothing  # the two markers do share exactly one point

        left_idxs = findall(Bramble.index_in_marker(Ωₕ, :xmin))
        bottom_idxs = findall(Bramble.index_in_marker(Ωₕ, :ymin))

        F = zeros(n)
        F[left_idxs] .= 1.0
        F[bottom_idxs] .= 2.0
        F[idx] = 10.0  # the corner's own distinct value, set last so it is not overwritten

        A = zeros(n, n)         # uₕ = 0, so A's entries never contribute: r = A*0 - F = -F
        uₕ = element(Wₕ, 0.0)

        r_left = reaction(A, F, uₕ; marker = :xmin)
        r_bottom = reaction(A, F, uₕ; marker = :ymin)
        r_combined = reaction(A, F, uₕ; marker = (:xmin, :ymin))

        @test r_left ≈ 10.0 + 1.0 * (length(left_idxs) - 1)
        @test r_bottom ≈ 10.0 + 2.0 * (length(bottom_idxs) - 1)
        # If the corner were double-counted, `r_combined` would equal `r_left + r_bottom`
        # exactly; counted once, it is short by the corner's own contribution.
        @test r_combined ≈ r_left + r_bottom - 10.0
        @test !(r_combined ≈ r_left + r_bottom)
    end

    @testset "Composite space: per-component reactions" begin
        # Two independent copies of the 1D problem stacked as a composite space; each
        # component solves its own manufactured problem, so a component-restricted
        # reaction on component 1 must ignore component 2 entirely, and vice versa.
        sol1(x) = sin(pi * x[1])
        src1(x) = pi^2 * sin(pi * x[1])
        sol2(x) = x[1] * (1 - x[1])   # -u'' = 2, u(0)=u(1)=0
        src2(x) = 2.0

        I = domain(interval(0.0, 1.0), :left => :xmin, :right => :xmax)
        Ωₕ = mesh(I, 41, true)
        Wₕ = gridspace(Ωₕ)
        W = Wₕ × Wₕ

        a = form(W, W, (u, v) -> inner₊(∇ₕ(u(1)), ∇ₕ(v(1))) + inner₊(∇ₕ(u(2)), ∇ₕ(v(2))))
        l = form(
            W,
            v -> innerₕ(Rₕ(Wₕ, src1), v(1)) + innerₕ(Rₕ(Wₕ, src2), v(2))
        )
        # `sol1`/`sol2` both vanish at x=0,1, so one homogeneous condition binds both
        # components identically -- no per-component Dirichlet *value* is needed here.
        A, F = assemble(a, l; dirichlet = :boundary => x -> 0.0)

        uₕ = element(W)
        uₕ .= A \ F

        # Component 1: same flux as the standalone 1D case above, π at each end.
        @test reaction(a, l, uₕ; marker = :left, dirichlet_components = 1) ≈ pi atol = 1e-2
        @test reaction(a, l, uₕ; marker = :right, dirichlet_components = 1) ≈ pi atol = 1e-2

        # Component 2: u = x(1-x), u'(x) = 1-2x. At x=0, outward normal -1: ∂u/∂n = -1,
        # q·n = 1. At x=1, outward normal +1: ∂u/∂n = -1, q·n = 1. Sum = 2 = ∫₀¹ 2 dx.
        @test reaction(a, l, uₕ; marker = :left, dirichlet_components = 2) ≈ 1.0 atol = 1e-9
        @test reaction(a, l, uₕ; marker = :right, dirichlet_components = 2) ≈ 1.0 atol = 1e-9

        # Restricting to the wrong component must not silently include the other one.
        @test !(
            reaction(a, l, uₕ; marker = :left, dirichlet_components = 2) ≈
            reaction(a, l, uₕ; marker = :left, dirichlet_components = 1)
        )
    end

    @testset "reaction_density: pointwise, exported quantity" begin
        sol(x) = sin(pi * x[1])
        src(x) = pi^2 * sin(pi * x[1])
        I = domain(interval(0.0, 1.0), :left => :xmin, :right => :xmax)
        Ωₕ = mesh(I, 21, true)
        Wₕ = gridspace(Ωₕ)

        a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
        l = form(Wₕ, v -> innerₕ(Rₕ(Wₕ, src), v))
        A, F = assemble(a, l; dirichlet = :boundary => sol)

        uₕ = element(Wₕ)
        uₕ .= A \ F

        dens = reaction_density(a, l, uₕ; marker = :left)
        @test dens isa Bramble.VectorElement
        @test parent(dens)[1] ≈ reaction(a, l, uₕ; marker = :left) / weights(Wₕ, Bramble.Innerh())[1]
        # zero away from the marker
        @test all(iszero, parent(dens)[2:end])
    end
end

end # module
