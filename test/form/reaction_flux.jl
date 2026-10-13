module FormReactionFluxTests

using Test
using Random
using ForwardDiff
using Bramble
using ..TestUtils: WITH_AD_TESTS, WITH_3D_TESTS
using Bramble: reaction, reaction!, reaction_density, reaction_density!

# `reaction` extracts the boundary flux a Dirichlet constraint had
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

    # The two ends together carry the discrete net source `(f_h, 1)_h`, to round-off, on a
    # non-uniform mesh. That sum is not the integral: it reaches `2π` only at second order.
    @testset "1D: both ends sum to the net source" begin
        sol(x) = sin(pi * x[1])
        src(x) = pi^2 * sin(pi * x[1])
        I = domain(interval(0.0, 1.0), :left => :xmin, :right => :xmax)
        Random.seed!(20260915)
        Ωₕ = mesh(I, 33, false)
        Wₕ = gridspace(Ωₕ)

        a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
        l = form(Wₕ, v -> innerₕ(Rₕ(Wₕ, src), v))
        A, F = assemble(a, l; dirichlet = :boundary => sol)

        uₕ = element(Wₕ)
        uₕ .= A \ F

        both = reaction(a, l, uₕ; marker = :left) + reaction(a, l, uₕ; marker = :right)
        @test both ≈ sum(assemble(l)) atol = 1e-10
    end

    # Flux on each side; conservation holds to round-off.
    @testset "2D flux: each side, conservation" begin
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

    # The net flux exists and matches a manufactured solution. 3D only: runs nightly (`slow`).
    WITH_3D_TESTS && @testset "3D flux: manufactured solution" begin
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

        # Component 2 has u = x(1-x) and u'(x) = 1-2x. At x=0 the outward normal is -1, so
        # ∂u/∂n = -1 and q·n = 1. At x=1 the outward normal is +1, so ∂u/∂n = -1 and q·n = 1.
        # The sum is 2 = ∫₀¹ 2 dx.
        @test reaction(a, l, uₕ; marker = :left, dirichlet_components = 2) ≈ 1.0 atol = 1e-9
        @test reaction(a, l, uₕ; marker = :right, dirichlet_components = 2) ≈ 1.0 atol = 1e-9

        # Restricting to the wrong component must not silently include the other one.
        @test !(
            reaction(a, l, uₕ; marker = :left, dirichlet_components = 2) ≈
            reaction(a, l, uₕ; marker = :left, dirichlet_components = 1)
        )
    end

    # reaction_density is pointwise and an exported quantity.
    @testset "reaction_density: pointwise" begin
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
        # a 1D face is a point of measure 1, so the density at the end is the reaction itself
        @test parent(dens)[1] ≈ reaction(a, l, uₕ; marker = :left)
        @test parent(dens)[1] ≈ pi rtol = 1.0e-2
        # zero away from the marker
        @test all(iszero, parent(dens)[2:end])
    end

    # In 2D the density divides by the surface weight on the marker's faces, not the cell
    # measure: the cell measure carries a normal half-spacing, so -r/w grows like 1/h
    # (gpena/Bramble.jl#546). u = sin(πx)sin(πy) has q·n = π sin(πy) on x = 0.
    @testset "reaction_density: converges to q·n" begin
        S = interval(0.0, 1.0) × interval(0.0, 1.0)
        sol(x) = sin(pi * x[1]) * sin(pi * x[2])
        function solved(Ωₕ)
            Wₕ = gridspace(Ωₕ)
            a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
            l = form(Wₕ, v -> innerₕ(Rₕ(Wₕ, x -> 2pi^2 * sol(x)), v))
            A, F = assemble(a, l; dirichlet = :boundary => sol)
            uₕ = element(Wₕ)
            uₕ .= A \ F
            return Wₕ, a, l, uₕ
        end
        B = boundary_symbols(S)
        Ω = domain(S, :xmin => :xmin, :boundary => B)

        # uniform, so the midpoint of x = 0 is a node: q·n = π there
        errs = map((21, 41)) do N
            _, a, l, uₕ = solved(mesh(Ω, (N, N), (true, true)))
            dens = reshape(parent(reaction_density(a, l, uₕ; marker = :xmin)), N, N)
            return abs(dens[1, (N + 1) ÷ 2] - pi)
        end
        @test errs[2] < errs[1] / 3
        @test errs[2] < 0.01

        # non-uniform: integrating the density over the faces recovers the reaction
        Ωₕ = mesh(Ω, (13, 11), (false, false))
        Wₕ, a, l, uₕ = solved(Ωₕ)
        one_ = element(Wₕ, 1.0)
        d = reaction_density(a, l, uₕ; marker = :xmin)
        @test inner_Γ(d, one_, :xmin) ≈ reaction(a, l, uₕ; marker = :xmin)
        db = reaction_density(a, l, uₕ; marker = :boundary)
        @test inner_Γ(db, one_, :boundary) ≈ reaction(a, l, uₕ; marker = :boundary)

        # by hand: -r over the surface weight, a mid-face point and a corner on two faces
        r = reshape(assemble(a) * parent(uₕ) - assemble(l), 13, 11)
        x, y = points(Ωₕ(1)), points(Ωₕ(2))
        @test reshape(parent(d), 13, 11)[1, 5] ≈ -r[1, 5] / ((y[6] - y[4]) / 2)
        @test reshape(parent(db), 13, 11)[1, 1] ≈ -r[1, 1] / ((y[2] - y[1]) / 2 + (x[2] - x[1]) / 2)

        # the faces come from the marked points, not the label's name: a label defined as
        # another face never divides by zero
        Ωm = domain(S, :left => :ymin, :boundary => B)
        Wm, am, lm, um = solved(mesh(Ωm, (13, 11), (false, false)))
        dm = try
            parent(reaction_density(am, lm, um; marker = :left))
        catch e
            e isa ArgumentError || rethrow()
            nothing
        end
        @test dm === nothing || all(isfinite, dm)
        dm === nothing || @test inner_Γ(element(Wm, dm), element(Wm, 1.0), :ymin) ≈
                                reaction(am, lm, um; marker = :left)

        # a user-named whole face
        Ωi = domain(S, :inflow => :xmin, :boundary => B)
        Wi, ai, li, ui = solved(mesh(Ωi, (13, 11), (false, false)))
        @test inner_Γ(reaction_density(ai, li, ui; marker = :inflow), element(Wi, 1.0), :xmin) ≈
              reaction(ai, li, ui; marker = :inflow)

        # a face with its corners left out still selects that face
        Ωo = domain(S, :open => x -> x[1] == 1 && 0 < x[2] < 1, :boundary => B)
        Wo, ao, lo, uo = solved(mesh(Ωo, (13, 11), (false, false)))
        do_ = reaction_density(ao, lo, uo; marker = :open)
        @test all(isfinite, parent(do_))
        @test inner_Γ(do_, element(Wo, 1.0), :xmax) ≈ reaction(ao, lo, uo; marker = :open)

        # an interior marker has no surface weight
        Ωc = domain(S, :core => x -> 0.2 < x[1] < 0.8 && 0.2 < x[2] < 0.8, :boundary => B)
        Wc, ac, lc, uc = solved(mesh(Ωc, (13, 11), (false, false)))
        @test_throws ArgumentError reaction_density(ac, lc, uc; marker = :core)
        @test_throws ArgumentError reaction_density!(
            zeros(ndofs(Wc)), assemble(ac), assemble(lc), uc; marker = :core)
        @test isfinite(reaction(ac, lc, uc; marker = :core))

        # part of a face selects no face, alone or beside a whole one
        Ωp = domain(S, :half => x -> x[1] == 0 && x[2] <= 0.5,
            :c => x -> x[1] == 0 || (x[2] == 0 && x[1] <= 0.5), :boundary => B)
        Wp, ap, lp, up = solved(mesh(Ωp, (13, 11), (false, false)))
        Ap, Fp = assemble(ap), assemble(lp)
        for m in (:half, :c)
            @test_throws ArgumentError reaction_density(ap, lp, up; marker = m)
            @test_throws ArgumentError reaction_density!(
                zeros(ndofs(Wp)), Ap, Fp, up; marker = m)
        end
    end

    # The in-place variants, the dense fallbacks, several markers and the space validation,
    # all against a residual assembled by hand: dense `Matrix(A) * u - F` summed over the
    # rows the marker masks select (not the code under test), on non-uniform meshes.
    @testset "In-place, dense fallback, hand residual" begin
        Random.seed!(20261002)
        I = domain(interval(0.0, 1.0), :left => :xmin, :right => :xmax)
        Ω₁ = mesh(I, 9, false)
        Ω₂ = mesh(I, 7, false)
        W₁ = gridspace(Ω₁)
        W₂ = gridspace(Ω₂)
        n₁ = ndofs(W₁)
        n₂ = ndofs(W₂)

        # (marker, leaf, expected rows of the composite space) by hand from the masks
        rows(Ωₕ, m, off) = off .+ findall(Bramble.index_in_marker(Ωₕ, m))

        @testset "scalar space, one and several markers" begin
            a = form(W₁, W₁, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(u, v))
            l = form(W₁, v -> innerₕ(Rₕ(W₁, x -> 1 + x[1]^2), v))
            A = assemble(a)
            F = assemble(l)
            uₕ = element(W₁)
            uₕ .= randn(n₁)
            r = Matrix(A) * parent(uₕ) - F
            # surface weight of a 1D face: a point, of measure 1 (not the cell measure)
            ω = ones(n₁)

            for marker in (:left, :right, (:left, :right), (:left, :left))
                ms = marker isa Symbol ? (marker,) : marker
                idx = sort(unique(vcat((rows(Ω₁, m, 0) for m in ms)...)))
                expected = -sum(r[idx])
                scratch = fill(NaN, n₁)
                @test reaction!(scratch, A, F, uₕ; marker = marker) ≈ expected
                # only the marked entries are written
                @test all(isnan, scratch[setdiff(1:n₁, idx)])
                @test scratch[idx] ≈ r[idx]
                # dense matrix: same flux through the full-matvec fallback
                scratch_d = fill(NaN, n₁)
                @test reaction!(scratch_d, Matrix(A), F, uₕ; marker = marker) ≈ expected
                @test all(isnan, scratch_d[setdiff(1:n₁, idx)])
                @test scratch_d[idx] ≈ r[idx]
                @test reaction(Matrix(A), F, uₕ; marker = marker) ≈ expected

                dens_expected = zeros(n₁)
                dens_expected[idx] .= -r[idx] ./ ω[idx]
                dens = zeros(n₁)
                @test reaction_density!(dens, A, F, uₕ; marker = marker) === dens
                @test dens ≈ dens_expected
                dens_d = zeros(n₁)
                reaction_density!(dens_d, Matrix(A), F, uₕ; marker = marker)
                @test dens_d ≈ dens_expected
                @test parent(reaction_density(A, F, uₕ; marker = marker)) ≈ dens_expected
            end
        end

        @testset "composite space, components, per-block" begin
            W = W₁ × W₂
            a = form(
                W, W,
                (u, v) -> inner₊(∇ₕ(u(1)), ∇ₕ(v(1))) + inner₊(∇ₕ(u(2)), ∇ₕ(v(2))) +
                          innerₕ(u(1), v(1)) + 2 * innerₕ(u(2), v(2))
            )
            l = form(
                W,
                v -> innerₕ(Rₕ(W₁, x -> 1 + x[1]), v(1)) + innerₕ(Rₕ(W₂, x -> 2 - x[1]), v(2))
            )
            A = assemble(a)
            F = assemble(l)
            uₕ = element(W)
            uₕ .= randn(n₁ + n₂)
            r = Matrix(A) * parent(uₕ) - F
            ωs = (ones(n₁), ones(n₂))   # 1D surface weights, as above
            offs = (0, n₁)
            Ωs = (Ω₁, Ω₂)

            for components in (nothing, 1, 2, (1, 2)), marker in (:left, (:left, :right))

                ms = marker isa Symbol ? (marker,) : marker
                sel = components === nothing || components == (1, 2) ? (1, 2) : (components,)
                idx = Int[]
                dens_expected = zeros(n₁ + n₂)
                for k in sel, m in ms

                    ik = rows(Ωs[k], m, offs[k])
                    append!(idx, ik)
                end
                idx = sort(unique(idx))
                for k in sel
                    for m in ms, j in rows(Ωs[k], m, offs[k])

                        dens_expected[j] = -r[j] / ωs[k][j - offs[k]]
                    end
                end
                expected = -sum(r[idx])

                @test reaction(A, F, uₕ; marker = marker, components = components) ≈ expected
                scratch = fill(NaN, n₁ + n₂)
                @test reaction!(
                    scratch, A, F, uₕ; marker = marker, components = components) ≈ expected
                @test all(isnan, scratch[setdiff(1:(n₁ + n₂), idx)])
                scratch_d = fill(NaN, n₁ + n₂)
                @test reaction!(
                    scratch_d, Matrix(A), F, uₕ; marker = marker, components = components) ≈
                      expected
                dens = zeros(n₁ + n₂)
                reaction_density!(dens, A, F, uₕ; marker = marker, components = components)
                @test dens ≈ dens_expected
                dens_d = zeros(n₁ + n₂)
                reaction_density!(
                    dens_d, Matrix(A), F, uₕ; marker = marker, components = components)
                @test dens_d ≈ dens_expected
                @test parent(reaction_density(
                    A, F, uₕ; marker = marker, components = components)) ≈ dens_expected
            end
            # per-block: each leaf's own flux separately, and they add up
            f1 = reaction(A, F, uₕ; marker = :left, components = 1)
            f2 = reaction(A, F, uₕ; marker = :left, components = 2)
            @test f1 ≈ -r[rows(Ω₁, :left, 0)[1]]
            @test f2 ≈ -r[rows(Ω₂, :left, n₁)[1]]
            @test reaction(A, F, uₕ; marker = :left) ≈ f1 + f2
            # a bad component index is rejected by the in-place variants too
            @test_throws ArgumentError reaction!(
                zeros(n₁ + n₂), A, F, uₕ; marker = :left, components = 3)
            @test_throws ArgumentError reaction_density!(
                zeros(n₁ + n₂), A, F, uₕ; marker = :left, components = 3)
        end

        @testset "in-place variants allocate nothing" begin
            a = form(W₁, W₁, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
            A = assemble(a)
            F = ones(n₁)
            uₕ = element(W₁, 1.0)
            scratch = zeros(n₁)
            for marker in (:left, (:left, :right))
                reaction!(scratch, A, F, uₕ; marker = marker)
                reaction_density!(scratch, A, F, uₕ; marker = marker)
                @test (@allocated reaction!(scratch, A, F, uₕ; marker = marker)) == 0
                @test (@allocated reaction_density!(scratch, A, F, uₕ; marker = marker)) == 0
            end
        end

        # A complex load against a Float64 matrix: the dense in-place path must promote, not
        # round `F` into a Float64 buffer (InexactError), and agree with the sparse path.
        @testset "in-place, complex load vector" begin
            A = assemble(form(W₁, W₁, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v))))
            uₕ = Rₕ(W₁, x -> x[1]^2)
            Fc = complex.(ones(n₁), 1.0)
            marker = (:left, :right)
            nmarked = length(union(rows(Ω₁, :left, 0), rows(Ω₁, :right, 0)))
            expected = reaction(A, Fc, uₕ; marker = marker)
            @test imag(expected) ≈ nmarked
            @test reaction!(zeros(ComplexF64, n₁), Matrix(A), Fc, uₕ; marker = marker) ≈
                  expected
            @test sum(reaction_density!(
                zeros(ComplexF64, n₁), Matrix(A), Fc, uₕ; marker = marker)) ≈
                  sum(reaction_density!(zeros(ComplexF64, n₁), A, Fc, uₕ; marker = marker))
        end

        # The internal walk and residual helpers, called through `invokelatest` so each runs as
        # its own compiled method (not inlined into a caller), against hand-derived values.
        @testset "internal helpers vs hand values" begin
            a = form(W₁, W₁, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(u, v))
            l = form(W₁, v -> innerₕ(Rₕ(W₁, x -> 1 + x[1]), v))
            A = assemble(a)
            F = assemble(l)
            u = randn(n₁)
            r = Matrix(A) * u - F
            idx = sort(vcat(rows(Ω₁, :left, 0), rows(Ω₁, :right, 0)))
            leaves = Bramble.leaf_spaces_offsets(W₁)

            # the lazy union walk, shifted by the leaf offset
            m = Bramble._reaction_marked(Ω₁, (:left, :right), 3)
            @test collect(m) == idx .+ 3
            @test Base.IteratorSize(typeof(m)) isa Base.SizeUnknown
            @test eltype(typeof(m)) === Int
            @test Base.invokelatest(iterate, m) == (idx[1] + 3, Base.invokelatest(iterate, m)[2])
            st = Base.invokelatest(iterate, m)[2]
            @test Base.invokelatest(iterate, m, st)[1] == idx[2] + 3

            # per-leaf entries: the union mask, offset, size and selection of each leaf
            entries = Base.invokelatest(
                Bramble._reaction_leaf_entries, leaves, (:left, :right), nothing)
            @test length(entries) == 1
            @test Base.invokelatest(
                Bramble._reaction_leaf_entries_impl, leaves, (:left, :right), nothing, 1) ==
                  entries
            mask, off, nn, active = entries[1]
            @test findall(mask) == idx
            @test (off, nn, active) == (0, n₁, true)
            entries_b = Base.invokelatest(
                Bramble._reaction_leaf_entries!, leaves, (:left, :right), nothing)
            @test findall(entries_b[1][1][1]) == rows(Ω₁, :left, 0)
            @test findall(entries_b[1][1][2]) == rows(Ω₁, :right, 0)
            for row in 1:n₁
                @test Base.invokelatest(Bramble._reaction_row_marked, entries_b, row) ==
                      (row in idx)
            end
            @test !Base.invokelatest(Bramble._reaction_row_marked, (), 1)

            # the restricted residual answers r[j] at the marked rows
            rr = Base.invokelatest(Bramble._reaction_residual, A, F, u, entries)
            for j in idx
                @test Base.invokelatest(getindex, rr, j) ≈ r[j]
            end
            @test eltype(rr) === Float64
        end

        @testset "space validation" begin
            a₁ = form(W₁, W₁, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
            l₁ = form(W₁, v -> innerₕ(Rₕ(W₁, x -> 1.0), v))
            a₂ = form(W₂, W₂, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
            l₂ = form(W₂, v -> innerₕ(Rₕ(W₂, x -> 1.0), v))
            u₁ = element(W₁, 1.0)
            u₂ = element(W₂, 1.0)
            # `a` and `l` on different spaces
            @test_throws ArgumentError reaction(a₁, l₂, u₁; marker = :left)
            @test_throws ArgumentError reaction_density(a₁, l₂, u₁; marker = :left)
            # `uₕ` on a different space from `a`/`l`
            @test_throws ArgumentError reaction(a₁, l₁, u₂; marker = :left)
            @test_throws ArgumentError reaction_density(a₂, l₂, u₁; marker = :left)

            # the matrix methods: every size checked before the `@inbounds` loops run
            A = assemble(a₁)
            F = assemble(l₁)
            for M in (A, Matrix(A))
                # positive control: the right sizes go through
                @test isfinite(reaction!(zeros(n₁), M, F, u₁; marker = :right))
                @test length(reaction_density!(zeros(n₁), M, F, u₁; marker = :right)) == n₁
                # short and long output buffers
                for m in (n₁ - 1, n₁ + 1)
                    @test_throws DimensionMismatch reaction!(zeros(m), M, F, u₁; marker = :right)
                    @test_throws DimensionMismatch reaction_density!(zeros(m), M, F, u₁; marker = :right)
                end
                # `uₕ` from another mesh
                @test_throws DimensionMismatch reaction(M, F, u₂; marker = :left)
                @test_throws DimensionMismatch reaction_density(M, F, u₂; marker = :left)
                @test_throws DimensionMismatch reaction!(zeros(n₂), M, F, u₂; marker = :left)
                @test_throws DimensionMismatch reaction_density!(zeros(n₂), M, F, u₂; marker = :left)
                # a short `F` and a non-square `A`
                @test_throws DimensionMismatch reaction(M, F[1:(end - 1)], u₁; marker = :right)
                @test_throws DimensionMismatch reaction_density(M, F[1:(end - 1)], u₁; marker = :right)
                @test_throws DimensionMismatch reaction(M[1:(end - 1), :], F, u₁; marker = :right)
                @test_throws DimensionMismatch reaction_density(M[:, 1:(end - 1)], F, u₁; marker = :right)
            end
        end
    end

    WITH_AD_TESTS && @testset "reaction: Dual load vector" begin
        # A Dual load against a Float64 matrix must not be rounded to Float64.
        Wₕ = gridspace(mesh(domain(interval(0.0, 1.0)), 7, false))
        A = assemble(form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v))))
        uₕ = Rₕ(Wₕ, x -> x[1]^2)
        g(s) = reaction(A, s .* ones(ndofs(Wₕ)), uₕ; marker = :boundary)
        h(s) = sum(parent(reaction_density(A, s .* ones(ndofs(Wₕ)), uₕ; marker = :boundary)))
        @test ForwardDiff.derivative(g, 2.0) ≈ g(3.0) - g(2.0)
        @test ForwardDiff.derivative(h, 2.0) ≈ h(3.0) - h(2.0)
        gd(s) = reaction(Matrix(A), s .* ones(ndofs(Wₕ)), uₕ; marker = :boundary)
        @test ForwardDiff.derivative(gd, 2.0) ≈ ForwardDiff.derivative(g, 2.0)
        # the dense in-place variants: d reaction/ds = 2 (the two marked ends)
        n = ndofs(Wₕ)
        gi(s) = (F = s .* ones(n); reaction!(similar(F), Matrix(A), F, uₕ; marker = :boundary))
        @test ForwardDiff.derivative(gi, 2.0) ≈ 2.0
        hd(s) = (F = s .* ones(n);
            sum(reaction_density!(zero(F), Matrix(A), F, uₕ; marker = :boundary)))
        hs(s) = (F = s .* ones(n); sum(reaction_density!(zero(F), A, F, uₕ; marker = :boundary)))
        @test ForwardDiff.derivative(hd, 2.0) ≈ ForwardDiff.derivative(hs, 2.0)
        @test ForwardDiff.derivative(hd, 2.0) > 0
        @test reaction(A, Matrix(A) * parent(uₕ), uₕ; marker = :boundary) isa Float64
        @test reaction(Matrix(A), ones(ndofs(Wₕ)), uₕ; marker = :boundary) ≈
              reaction(A, ones(ndofs(Wₕ)), uₕ; marker = :boundary)
    end
end

end # module
