module FormReactionFluxTests

using Test
using Random
using ForwardDiff
using Bramble
using ..TestUtils: WITH_AD_TESTS
using Bramble: reaction, reaction!, reaction_density, reaction_density!, weights

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

    # The net flux exists and matches a manufactured solution.
    @testset "3D flux: manufactured solution" begin
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
        @test parent(dens)[1] ≈ reaction(a, l, uₕ; marker = :left) / weights(Wₕ, Bramble.Innerh())[1]
        # zero away from the marker
        @test all(iszero, parent(dens)[2:end])
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
            w = weights(W₁, Bramble.Innerh())

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
                dens_expected[idx] .= -r[idx] ./ w[idx]
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
            ws = (weights(W₁, Bramble.Innerh()), weights(W₂, Bramble.Innerh()))
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

                        dens_expected[j] = -r[j] / ws[k][j - offs[k]]
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
        @test reaction(A, Matrix(A) * parent(uₕ), uₕ; marker = :boundary) isa Float64
        @test reaction(Matrix(A), ones(ndofs(Wₕ)), uₕ; marker = :boundary) ≈
              reaction(A, ones(ndofs(Wₕ)), uₕ; marker = :boundary)
    end
end

end # module
