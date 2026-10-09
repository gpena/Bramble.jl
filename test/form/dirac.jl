module FormDiracSourceTests

using Test
using LinearAlgebra
using Random
using ForwardDiff
using Bramble
using ..TestUtils: WITH_AD_TESTS
using Bramble:
               DiracSource,
               form,
               assemble,
               assemble!,
               test_space,
               mesh,
               domain,
               interval,
               gridspace,
               Rₕ,
               innerₕ,
               inner₊,
               inner₊ₓ,
               ∇ₕ,
               dirac,
               ndofs,
               locate_cell,
               points,
               Innerh,
               weights,
               VectorGridSpace,
               stepsize,
               backend,
               Mₓ,
               πₕ

@testset "Dirac Point Sources (#226)" begin
    @testset "1D point source (uniform, non-uniform)" begin
        # 1. On-grid point source (uniform grid)
        Ω_unif = mesh(domain(interval(0.0, 1.0)), 21, true)
        W_unif = gridspace(Ω_unif)
        pts_unif = points(Ω_unif)

        x_on = pts_unif[4] # exact grid vertex
        strength = 2.5
        l_on = form(W_unif, v -> innerₕ(dirac(x_on, strength), v))
        b_on = assemble(l_on)

        @test sum(b_on) ≈ strength
        @test b_on[4] ≈ strength
        @test count(!iszero, b_on) == 1

        # 2. Off-grid point source on non-uniform grid (seeded for determinism)
        Random.seed!(20260914)
        Ω_nonunif = mesh(domain(interval(0.0, 1.0)), 21, false)
        W_nonunif = gridspace(Ω_nonunif)
        pts_nonunif = points(Ω_nonunif)

        x_off = 0.347
        l_off = form(W_nonunif, v -> innerₕ(dirac(x_off, strength), v))
        b_off = assemble(l_off)

        i_cell = locate_cell(Ω_nonunif, x_off)
        x_lo, x_hi = pts_nonunif[i_cell], pts_nonunif[i_cell + 1]
        t = (x_off - x_lo) / (x_hi - x_lo)
        # the two weights are pinned by bilinear.jl's non-uniform source-stencil testset;
        # what stays here is that no other entry is touched
        @test count(>(1e-12), b_off) == 2

        # 3. Contraction against smooth grid function on non-uniform grid
        phi = x -> sin(2π * x[1]) + exp(x[1])
        phi_h = Rₕ(W_nonunif, phi)
        val = l_off(phi_h)
        val_expected = strength * ((1 - t) * phi((x_lo,)) + t * phi((x_hi,)))
        @test val ≈ val_expected
        @test val ≈ strength * phi((x_off,)) atol = 0.05

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

        # 4. In-place live coefficient update via Ref on non-uniform grid
        s_ref = Ref(1.0)
        l_ref = form(W_nonunif, v -> innerₕ(dirac(x_off, s_ref), v))
        b_buf = zeros(ndofs(W_nonunif))
        assemble!(b_buf, l_ref)
        @test sum(b_buf) ≈ 1.0

        s_ref[] = 4.2
        assemble!(b_buf, l_ref)
        @test sum(b_buf) ≈ 4.2

        # Zero allocation verification on replay
        allocs = @allocated assemble!(b_buf, l_ref)
        @test allocs == 0
    end

    # On a non-uniform mesh.
    @testset "2D single and multiple sources" begin
        Random.seed!(20260914)
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (21, 21), (false, false))
        Wₕ = gridspace(Ωₕ)

        # 1. Single 2D point source on non-uniform mesh
        p0 = (0.35, 0.65)
        S = 3.0
        l_2d = form(Wₕ, v -> innerₕ(dirac(p0, S), v))
        b_2d = assemble(l_2d)

        @test sum(b_2d) ≈ S

        # Verify bilinear weights on 4 cell corners
        phi = x -> cos(π * x[1]) * sin(π * x[2])
        phi_h = Rₕ(Wₕ, phi)
        @test l_2d(phi_h) ≈ dot(b_2d, parent(phi_h))
        @test l_2d(phi_h) ≈ S * phi(p0) atol = 0.05

        # 2. Vector of multiple point sources
        pts = [(0.2, 0.3), (0.7, 0.8)]
        strengths = [1.5, 2.5]
        l_multi = form(Wₕ, v -> innerₕ(dirac(pts, strengths), v))
        b_multi = assemble(l_multi)

        @test sum(b_multi) ≈ sum(strengths)

        # Superposition equivalence
        l1 = form(Wₕ, v -> innerₕ(dirac(pts[1], strengths[1]), v))
        l2 = form(Wₕ, v -> innerₕ(dirac(pts[2], strengths[2]), v))
        @test b_multi ≈ assemble(l1) + assemble(l2)

        # A sum of sources in one inner product: no cell measure on the Dirac terms (#512)
        δa = dirac(pts[1], strengths[1])
        δb = dirac(pts[2], strengths[2])
        l_sum = form(Wₕ, v -> innerₕ(dirac(pts[1], strengths[1]) +
                                     dirac(pts[2], strengths[2]), v))
        b_sum = assemble(l_sum)
        @test sum(b_sum) ≈ sum(strengths)
        @test b_sum ≈ assemble(l1) + assemble(l2)

        fₕ = Rₕ(Wₕ, x -> 1 + x[1])
        @test assemble(form(Wₕ, v -> innerₕ(πₕ(fₕ) + δa, v))) ≈
              assemble(form(Wₕ, v -> innerₕ(πₕ(fₕ), v))) + assemble(l1)
        @test assemble(form(Wₕ, v -> innerₕ(fₕ * (δa + δb), v))) ≈
              assemble(form(Wₕ, v -> innerₕ(fₕ * δa, v) + innerₕ(fₕ * δb, v)))
        @test assemble(form(Wₕ, v -> innerₕ(δa - δb, v))) ≈ assemble(l1) - assemble(l2)

        # The split reads types alone, so `form` stays concrete
        δs = ntuple(i -> dirac((0.05i, 0.07i), 1.0i), 10)
        k_sub(W, a, b) = form(W, v -> innerₕ(a - b, v))
        k_fold(W, δs) = form(W, v -> innerₕ(foldl(+, δs), v))
        k_mix(W, f, a, b) = form(W, v -> innerₕ(πₕ(f) + a + πₕ(f) + b, v))
        @test isconcretetype(only(Base.return_types(k_sub, typeof.((Wₕ, δa, δb)))))
        @test isconcretetype(only(Base.return_types(k_fold, typeof.((Wₕ, δs)))))
        @test isconcretetype(only(Base.return_types(k_mix, typeof.((Wₕ, fₕ, δa, δb)))))
        @test sum(assemble(k_fold(Wₕ, δs))) ≈ sum(1.0:10.0)

        # A mixed sum under a wrapper would scale the Dirac term: rejected, not wrong
        @test_throws ArgumentError form(Wₕ, v -> innerₕ(Mₓ(πₕ(fₕ) + δa), v))
        @test_throws "dirac" form(Wₕ, v -> innerₕ(Mₓ(πₕ(fₕ) + δa), v))

        buf = similar(b_sum)
        assemble!(buf, l_sum)
        @test (@allocated assemble!(buf, l_sum)) == 0
    end

    @testset "3D Point Source (Non-Uniform)" begin
        Random.seed!(20260914)
        Ωₕ = mesh(
            domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0)),
            (7, 7, 7),
            (false, false, false)
        )
        Wₕ = gridspace(Ωₕ)

        p0 = (0.25, 0.55, 0.75)
        S = 5.0
        l_3d = form(Wₕ, v -> innerₕ(dirac(p0, S), v))
        b_3d = assemble(l_3d)

        @test sum(b_3d) ≈ S
        phi = x -> x[1] + 2 * x[2] + 3 * x[3]
        phi_h = Rₕ(Wₕ, phi)
        # Multilinear interpolation of a linear function is exact on any grid
        @test l_3d(phi_h) ≈ S * phi(p0)
    end

    # Poisson problem with a Dirac delta source.
    @testset "Poisson: 1D Green function" begin
        # -u''(x) = S * δ(x - x0), u(0) = u(1) = 0
        # Exact solution is Green's function:
        # G(x, x0) = S * (1 - x0) * x  for x <= x0
        # G(x, x0) = S * x0 * (1 - x)  for x > x0
        Random.seed!(20260914)
        N = 51
        Ωₕ = mesh(domain(interval(0.0, 1.0)), N, false)
        Wₕ = gridspace(Ωₕ)
        pts_grid = points(Ωₕ)

        # Place source at exact node: index 21
        x0 = pts_grid[21]
        S = 2.0

        a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
        l = form(Wₕ, v -> innerₕ(dirac(x0, S), v))

        A, F = assemble(a, l; dirichlet = :boundary => x -> 0.0)
        u_num = A \ F

        u_exact = [x <= x0 ? S * (1 - x0) * x : S * x0 * (1 - x) for x in pts_grid]
        @test u_num ≈ u_exact atol = 1e-12
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

    @testset "Time-dependent strength, semidiscretize" begin
        # A pulsed/moving source: strength read from a live `Ref`, refilled from `t` by
        # `update_coefficients!` before each assembly -- the same discipline
        # test/form/semidiscrete.jl exercises for a scalar coefficient, applied here to a
        # `dirac` strength so a moving or time-modulated point source works under
        # `semidiscretize`.
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 21, false)
        Wₕ = gridspace(Ωₕ)
        n = ndofs(Wₕ)
        x0 = 0.5

        s_ref = Ref(0.0)
        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v))
        l = form(Wₕ, v -> innerₕ(dirac(x0, s_ref), v))

        strength_at(t) = 2.0 + sin(t)
        sd = semidiscretize(a, l; (update_coefficients!) = t -> (s_ref[] = strength_at(t)))

        du = zeros(n)
        sd(du, zeros(n), nothing, 0.7)
        @test sum(du) ≈ strength_at(0.7)

        du2 = zeros(n)
        sd(du2, zeros(n), nothing, 2.1)
        @test sum(du2) ≈ strength_at(2.1)
        @test !(du2 ≈ du)
    end

    @testset "Composite Space Routing (Non-Uniform)" begin
        Random.seed!(20260914)
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 11, false)
        Wₕ = gridspace(Ωₕ)
        W_comp = Wₕ × Wₕ

        # Source placed in block 1 only
        x0 = 0.5
        S = 3.0
        l_comp = form(W_comp, v -> innerₕ(dirac(x0, S), v(1)))
        b_comp = assemble(l_comp)

        n1 = ndofs(Wₕ)
        b1 = b_comp[1:n1]
        b2 = b_comp[(n1 + 1):end]

        @test sum(b1) ≈ S
        @test all(iszero, b2)
    end

    # On a non-uniform mesh.
    @testset "inner₊ and directional inner with Dirac" begin
        Random.seed!(20260914)
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 11, false)
        Wₕ = gridspace(Ωₕ)

        x0 = 0.35
        S = 2.0
        l_innerh = form(Wₕ, v -> innerₕ(dirac(x0, S), v))
        l_innerplusx = form(Wₕ, v -> inner₊ₓ(dirac(x0, S), v))

        b_h = assemble(l_innerh)
        b_plusx = assemble(l_innerplusx)

        @test b_h ≈ b_plusx
        @test sum(b_plusx) ≈ S

        # 2D directional inner product on non-uniform grid
        Ω2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (11, 11), (false, false))
        W2 = gridspace(Ω2)
        p0 = (0.25, 0.75)
        l_2d_h = form(W2, v -> innerₕ(dirac(p0, S), v))
        l_2d_plusx = form(W2, v -> inner₊ₓ(dirac(p0, S), v))
        @test assemble(l_2d_h) ≈ assemble(l_2d_plusx)
    end

    # The weight's type is the mesh's element type promoted with the strength's
    # A Float32 space with a Float32 strength gives a Float32 vector. A Float64
    # strength (the default `1.0`) on a Float32 space still promotes, as `innerₕ(1.0, v)`
    # does; an integer strength takes the space's type. The location keeps Float64 precision.
    @testset "Weight eltype follows the space (#361)" begin
        Random.seed!(20260927)
        for D in (1, 2)
            Ω = domain(D == 1 ? interval(0.0f0, 1.0f0) : interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0))
            n, unif = D == 1 ? (11, false) : ((11, 9), (false, false))
            W32 = gridspace(mesh(Ω, n, unif; backend = backend(Float32)))
            p1 = D == 1 ? 0.37f0 : (0.37f0, 0.61f0)
            pts = D == 1 ? [(0.2f0,), (0.55f0,)] : [(0.2f0, 0.3f0), (0.55f0, 0.7f0)]
            @testset "$(D)D" begin
                for (label, src, T) in (("Float32 strength", dirac(p1, 2.0f0), Float32),
                    ("points", dirac(pts, [1.5f0, -0.5f0]), Float32),
                    ("Int strength", dirac(p1, 2), Float32),
                    ("default Float64 strength", dirac(p1), Float64),
                    ("Float64 points strength", dirac(pts, [1.5, -0.5]), Float64))
                    l = form(W32, v -> innerₕ(src, v))
                    b = @inferred assemble(l)
                    @test eltype(b) === T
                    total = src.strengths isa AbstractVector ? sum(src.strengths) : src.strengths
                    @test sum(b) ≈ total rtol = 1.0f-5
                    assemble!(b, l)
                    @test (@allocated assemble!(b, l)) == 0
                end
                # Float32 weights round the Float64 ones, nothing more.
                b32 = assemble(form(W32, v -> innerₕ(dirac(p1, 2.0f0), v)))
                b64 = assemble(form(W32, v -> innerₕ(dirac(p1, 2.0), v)))
                @test b32 == Float32.(b64)
            end
        end
        # A Float64 space is unchanged: Float64 vector, and AD through the strength works.
        Ω64 = domain(interval(0.0, 1.0))
        W64 = gridspace(mesh(Ω64, 11, false))
        b64 = @inferred assemble(form(W64, v -> innerₕ(dirac(0.37), v)))
        @test eltype(b64) === Float64
        @test eltype(assemble(form(W64, v -> innerₕ(dirac(0.37, 2.0f0), v)))) === Float64
        if WITH_AD_TESTS
            for src in (s -> dirac(0.37, s), s -> dirac([(0.2,), (0.55,)], [s, 2s]))
                g = ForwardDiff.derivative(s -> assemble(form(W64, v -> innerₕ(src(s), v))), 2.0)
                @test g ≈ assemble(form(W64, v -> innerₕ(src(1.0), v))) rtol = 1e-12
            end
        end
        # A strengths vector with an abstract eltype: each product keeps its own strength's
        # type, so a later `Dual` or a wider float is not narrowed to the first one's.
        pts2 = [(0.2,), (0.45,)]
        if WITH_AD_TESTS
            for strengths in (s -> [() -> 1.0, () -> s], s -> Any[1.0, s])  # Any: the abstract-eltype strengths vector is the input under test
                g = ForwardDiff.derivative(s -> assemble(form(W64, v -> innerₕ(dirac(pts2, strengths(s)), v))), 2.0)
                @test g ≈ assemble(form(W64, v -> innerₕ(dirac(pts2[2:2], 1.0), v))) rtol = 1e-12
            end
        end
        # On a Float32 space, a Float64 strength after a Float32 one keeps Float64 precision.
        # The two points sit far apart, so the second's entries are its alone.
        W32 = gridspace(mesh(domain(interval(0.0f0, 1.0f0)), 11, false; backend = backend(Float32)))
        pts3 = [(0.05,), (0.9,)]
        alone = assemble(form(W32, v -> innerₕ(dirac(pts3[2:2], 0.1), v)))
        nz = findall(!iszero, alone)
        for strengths in ([() -> 1.0f0, () -> 0.1], Real[1.0f0, 0.1], [Ref(1.0f0), Ref(0.1)])
            mixed = assemble(form(W32, v -> innerₕ(dirac(pts3, strengths), v)))
            @test eltype(mixed) === Float64
            @test mixed[nz] == alone[nz]
        end
    end

    # The vector's type is folded from every strength, not read off one node's stencil
    # A plain term first: the flattened sum's stencil must not decide the type by its first
    # entry alone, or the Dirac's `Dual`, or its Float64 strength on a Float32 space, goes unseen. The points sit away from the grid's middle node.
    @testset "dirac: mixed strengths off-probe" begin
        for D in (1, 2)
            Ω = domain(D == 1 ? interval(0.0, 1.0) : interval(0.0, 1.0) × interval(0.0, 1.0))
            n, unif = D == 1 ? (11, false) : ((7, 9), (false, true))
            Wₕ = gridspace(mesh(Ω, n, unif))
            pts = D == 1 ? [(0.07,), (0.93,)] : [(0.07, 0.11), (0.93, 0.88)]
            src = s -> dirac(pts, [() -> 1.0, () -> s])
            l = s -> assemble(form(Wₕ, v -> innerₕ(1.0, v) + innerₕ(src(s), v)))
            alone = assemble(form(Wₕ, v -> innerₕ(dirac(pts[2:2], 1.0), v)))
            if WITH_AD_TESTS
                @test ForwardDiff.derivative(l, 2.0) ≈ alone rtol = 1e-12
                @test sum(ForwardDiff.derivative(l, 2.0)) ≈ 1.0 rtol = 1e-12
            end

            Ω32 = domain(D == 1 ? interval(0.0f0, 1.0f0) : interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0))
            W32 = gridspace(mesh(Ω32, n, unif; backend = backend(Float32)))
            for strengths in ([() -> 1.0f0, () -> 0.1], Real[1.0f0, 0.1], [Ref(1.0f0), Ref(0.1)])
                b = assemble(form(W32, v -> innerₕ(1.0f0, v) + innerₕ(dirac(pts, strengths), v)))
                @test eltype(b) === Float64
                @test sum(b) ≈ sum(assemble(form(W32, v -> innerₕ(1.0f0, v)))) + 1.1 rtol = 1e-6
            end
        end
    end

    # Tuples are the inferable spelling; a vector point normalises to the same tuple, and an
    # empty point or one whose length or kind disagrees with the first is a named error (#531)
    @testset "dirac: point spellings and checks (#531)" begin
        @test @inferred(dirac((0.3, 0.4), 2.0)) === DiracSource{2}((0.3, 0.4), 2.0)
        many = @inferred dirac([(0.3, 0.4), (0.7, 0.2)], [1.0, -1.0])
        @test many isa DiracSource{2, Vector{NTuple{2, Float64}}, Vector{Float64}}
        @test many.points == [(0.3, 0.4), (0.7, 0.2)]
        @test dirac([0.2, 0.7]).points === (0.2, 0.7)

        Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
        Wₕ = gridspace(mesh(Ω, (6, 6), (false, true)))
        b_vec = assemble(form(Wₕ, v -> innerₕ(dirac([0.3, 0.4], 2.0), v)))
        b_tup = assemble(form(Wₕ, v -> innerₕ(dirac((0.3, 0.4), 2.0), v)))
        @test b_vec == b_tup
        @test count(!iszero, b_tup) > 1

        # an untyped list (a `Vector{Any}`) of tuples and vectors is a list of points; one of
        # reals only is one point, like a flat real vector
        mixed = dirac([(0.1, 0.2), [0.3, 0.4]])
        @test mixed isa DiracSource{2, Vector{NTuple{2, Float64}}, Vector{Float64}}
        @test mixed.points == [(0.1, 0.2), (0.3, 0.4)]
        @test dirac(Any[0.2, 0.7]).points === (0.2, 0.7)

        for empty in (Float64[], (), [Float64[]])
            @test_throws ArgumentError dirac(empty)
            @test_throws "empty" dirac(empty)
        end
        @test_throws ArgumentError dirac(Any[])
        @test_throws "at least one point location" dirac(Any[])
        bad = ([[0.1, 0.2], [0.3, 0.4, 0.5]], [[0.1, 0.2, 0.3], [0.4, 0.5]], [(0.1, 0.2), (0.3, 0.4, 0.5)],
               [(0.1, 0.2), [0.3, 0.4, 0.5]], [(0.1, 0.2), 0.3], [0.1, [0.2, 0.3]])
        for pts in bad
            @test_throws ArgumentError dirac(pts)
            @test_throws "dirac point 2" dirac(pts)
        end
    end
end

end # module
