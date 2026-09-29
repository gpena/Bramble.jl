module SpaceWeightsStalenessTests

using Test
using ForwardDiff
using Bramble
using Bramble: norm₊
using Bramble: change_points!, set_points!, weights
using ..TestUtils: alloc_test, @test_allocs

# `ScalarGridSpace` precomputes its inner-product weights once from the mesh at
# `gridspace(Ωₕ)` time (gpena/Bramble.jl#221). The mesh is mutable (`set_points!`,
# `change_points!`, `iterative_refinement!` all rewrite it in place), and nothing used to
# stop a space built before such a mutation from silently computing against weights for a
# mesh that no longer exists. Every check here is against an independent reference -- a
# freshly built `gridspace(Ωₕ)` after the mutation, or a hand-computed exact integral --
# never against another call to the code under test.

@testset "Grid space weights staleness (#221)" begin
    # change_points! gives the exact numbers from the issue.
    @testset "Reproducer: issue's exact numbers" begin
        Ω = domain(interval(0.0, 1.0))
        Ωₕ = mesh(Ω, 5, true)
        Wₕ = gridspace(Ωₕ)

        change_points!(Ωₕ, markers(Ω), [0.0, 0.1, 0.2, 0.3, 1.0])

        # that this throws at all is "All three mutators trigger staleness"'s; what is
        # reproduced here is the numbers a fresh space gives instead.
        Wₕ_fresh = gridspace(Ωₕ)
        @test weights(Wₕ_fresh).innerh ≈ [0.05, 0.1, 0.1, 0.4, 0.35]

        u = Rₕ(Wₕ_fresh, x -> x[1]^2)
        v = Rₕ(Wₕ_fresh, x -> 1.0)
        # Hand-computed against the weights [0.05, 0.1, 0.1, 0.4, 0.35] and point values
        # [0, 0.01, 0.04, 0.09, 1.0] checked above -- not a quadrature-accuracy claim (5
        # points on a grid this skewed, [0.3, 1.0] is one cell, is coarse by design, to
        # keep the reproducer's own grid), only that innerₕ on the rebuilt space computes
        # the discrete sum it is defined to, independent of `innerₕ`/`weights` themselves.
        @test innerₕ(u, v) ≈
              0.05 * 0.0 + 0.1 * 0.01 + 0.1 * 0.04 + 0.4 * 0.09 + 0.35 * 1.0
    end

    @testset "All three mutators trigger staleness" begin
        Ω = domain(interval(0.0, 1.0))

        @testset "change_points!" begin
            Ωₕ = mesh(Ω, 11, true)
            Wₕ = gridspace(Ωₕ)
            change_points!(Ωₕ, markers(Ω), collect(range(0.0, 1.0; length = 11)))
            @test_throws ArgumentError weights(Wₕ)
        end

        @testset "set_points!" begin
            Ωₕ = mesh(Ω, 11, true)
            Wₕ = gridspace(Ωₕ)
            set_points!(Ωₕ, collect(range(0.0, 1.0; length = 11)))
            @test_throws ArgumentError weights(Wₕ)
        end

        @testset "iterative_refinement!" begin
            Ωₕ = mesh(Ω, 5, true)
            Wₕ = gridspace(Ωₕ)
            iterative_refinement!(Ωₕ)
            @test_throws ArgumentError weights(Wₕ)
        end
    end

    # Not only weights() itself.
    @testset "Every guarded entry point throws" begin
        Ω = domain(interval(0.0, 1.0))
        Ωₕ = mesh(Ω, 11, true)
        Wₕ = gridspace(Ωₕ)
        u = Rₕ(Wₕ, x -> x[1])
        v = Rₕ(Wₕ, x -> 1.0)

        change_points!(Ωₕ, markers(Ω), collect(range(0.0, 1.0; length = 11)))

        @test_throws ArgumentError innerₕ(u, v)
        @test_throws ArgumentError inner₊(u, v)
        @test_throws ArgumentError normₕ(u)
        @test_throws ArgumentError norm₊(u)
        @test_throws ArgumentError norm₁ₕ(u)
        @test_throws ArgumentError snorm₁ₕ(u)
    end

    # 2D: mutating one submesh stales the whole space.
    @testset "MeshnD: one submesh stales the space" begin
        S = interval(0.0, 1.0) × interval(0.0, 1.0)
        Ωₕ = mesh(domain(S), (7, 7), (true, true))
        Wₕ = gridspace(Ωₕ)
        u = Rₕ(Wₕ, x -> 1.0)

        # Mutating only the x-axis submesh, directly -- not through the parent's own
        # change_points! -- still invalidates the composite mesh version, since it is
        # derived from the submeshes' own (`_mesh_version(Ωₕ::MeshnD) = sum(...)`), not
        # tracked separately.
        change_points!(Ωₕ(1), markers(interval(0.0, 1.0)), collect(range(0.0, 1.0; length = 7)))

        @test_throws ArgumentError weights(Wₕ)
        @test_throws ArgumentError innerₕ(u, u)

        Wₕ_fresh = gridspace(Ωₕ)
        @test innerₕ(Rₕ(Wₕ_fresh, x -> 1.0), Rₕ(Wₕ_fresh, x -> 1.0)) ≈ 1.0
    end

    @testset "3D (MeshnD)" begin
        S = interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0)
        Ωₕ = mesh(domain(S), (5, 5, 5), (true, true, true))
        Wₕ = gridspace(Ωₕ)

        change_points!(Ωₕ(3), markers(interval(0.0, 1.0)), collect(range(0.0, 1.0; length = 5)))

        @test_throws ArgumentError weights(Wₕ)
    end

    # CompositeGridSpace: staleness in one leaf is caught, never silently ignored.
    @testset "Composite: stale leaf is caught" begin
        Ω = domain(interval(0.0, 1.0))
        Ωₕ = mesh(Ω, 9, true)
        Wₕ = gridspace(Ωₕ)
        W_comp = Wₕ × Wₕ

        u = Rₕ(W_comp, (x -> 1.0, x -> 1.0))

        change_points!(Ωₕ, markers(Ω), collect(range(0.0, 1.0; length = 9)))

        # Both leaves share `Ωₕ`, so both are stale; innerₕ recurses leaf by leaf and the
        # first one it reaches already throws.
        @test_throws ArgumentError innerₕ(u, u)
    end

    # No throw.
    @testset "Rebuilt space works after mutation" begin
        Ω = domain(interval(0.0, 1.0))
        Ωₕ = mesh(Ω, 21, true)
        Wₕ = gridspace(Ωₕ)
        n_before = ndofs(Wₕ)   # captured before mutating: mesh(Wₕ) is Ωₕ itself, not a copy
        iterative_refinement!(Ωₕ)

        Wₕ_fresh = gridspace(Ωₕ)
        u = Rₕ(Wₕ_fresh, x -> x[1])
        v = Rₕ(Wₕ_fresh, x -> 1.0)
        @test innerₕ(u, v) ≈ 0.5 atol = 1.0e-3
        @test ndofs(Wₕ_fresh) == 2 * n_before - 1
    end

    @testset "Fresh success path: zero allocation" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 1001, true)
        Wₕ = gridspace(Ωₕ)
        u = Rₕ(Wₕ, x -> x[1]^2)
        v = Rₕ(Wₕ, x -> 1.0)

        @test_allocs innerₕ(u, v)
        @test_allocs normₕ(u)
        @test_allocs weights(Wₕ)
    end

    # An interpolation's source leaf is read through `weights` when the form binds it, so a
    # host refill whose source mesh has moved throws instead of returning numbers for the
    # old mesh (gpena/Bramble.jl#367). The moved mesh is only ever an interpolation source:
    # no term walks it natively, since a walked stale leaf already throws on its own.
    @testset "Interpolation source mesh moved" begin
        m1(n) = mesh(domain(interval(0.0, 1.0)), n, true)
        m2(n) = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (n, n + 1),
            (true, false))
        function stretch!(M)
            dim(M) == 1 &&
                return change_points!(M, collect(range(0.0, 1.0; length = npoints(M))) .^ 2)
            n1, n2 = npoints(M, Tuple)
            return change_points!(M, (collect(range(0.0, 1.0; length = n1)) .^ 2,
                collect(range(0.0, 1.0; length = n2))))
        end
        cases = (
            ("trial πₕ scalar", (Ws, Wt) -> (Ws, Wt), (u, v) -> innerₕ(πₕ(u), v)),
            ("test πₕ scalar", (Ws, Wt) -> (Wt, Ws), (u, v) -> innerₕ(u, πₕ(v))),
            ("trial πₕ composite", (Ws, Wt) -> (Ws × Wt, Wt × Wt),
                (U, V) -> innerₕ(πₕ(U(1)), V(1)) + innerₕ(U(2), V(2))),
            ("test πₕ composite", (Ws, Wt) -> (Wt × Wt, Ws × Wt),
                (U, V) -> innerₕ(U(1), πₕ(V(1))) + innerₕ(U(2), V(2)))
        )
        for (mk, ns, nt) in ((m1, 7, 11), (m2, 4, 6)), (name, spaces, f) in cases

            @testset "$name, $(dim(mk(3)))D" begin
                Ms, Mt = mk(ns), mk(nt)
                F = form(spaces(gridspace(Ms), gridspace(Mt))..., f)
                A = assemble(F)
                @test alloc_test(assemble!, A, F) == 0
                stretch!(Ms)
                @test_throws ArgumentError assemble!(A, F)
            end
        end
    end

    # A linear form's `πₕ(uₕ)` source is read at every fill, not sampled once when the form
    # is built (gpena/Bramble.jl#408): a refill after new values in `uₕ` matches a fresh
    # form, and one after `uₕ`'s mesh has moved throws instead of interpolating on it, as do
    # `interpolate_at` and `πₕ!` themselves. The source mesh is only ever read through
    # `interpolate_at`; the walked mesh is never moved.
    @testset "Interpolated source mesh moved" begin
        m1(n) = mesh(domain(interval(0.0, 1.0)), n, true)
        m2(n) = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (n, n + 1),
            (true, false))
        function stretch!(M)
            dim(M) == 1 &&
                return change_points!(M, collect(range(0.0, 1.0; length = npoints(M))) .^ 2)
            n1, n2 = npoints(M, Tuple)
            return change_points!(M, (collect(range(0.0, 1.0; length = n1)) .^ 2,
                collect(range(0.0, 1.0; length = n2))))
        end
        g(x) = sum(abs2, x) + 1
        h(x) = 100 * first(x) + 3
        for (mk, x) in ((m1, 0.3), (m2, (0.3, 0.4))), kind in (:scalar, :component)

            @testset "$kind, $(dim(mk(3)))D" begin
                Ms, Mt = mk(5), mk(8)
                Ws, Wt = gridspace(Ms), gridspace(Mt)
                U = kind === :scalar ? Rₕ(Ws, g) : Rₕ(Ws × Ws, (h, g))
                uₕ = kind === :scalar ? U : components(U)[2]
                L = form(Wt, v -> innerₕ(πₕ(uₕ), v))
                b = assemble(L)
                @test alloc_test(assemble!, b, L) == 0

                parent(U) .*= 2
                assemble!(b, L)
                @test b ≈ assemble(form(Wt, v -> innerₕ(πₕ(uₕ), v)))
                @test b ≈ assemble(form(Wt, v -> innerₕ(πₕ(Wt, uₕ), v)))

                stretch!(Ms)
                @test_throws ArgumentError assemble!(b, L)
                @test_throws ArgumentError assemble(L)
                @test_throws ArgumentError interpolate_at(uₕ, x)
                @test_throws ArgumentError πₕ!(element(Wt), uₕ)
                @test_throws ArgumentError πₕ(Wt, uₕ)
            end
        end
    end

    # Unsampled, the interpolant's element type is read off `uₕ`, the walked mesh and the
    # fill, never probed at a point: the walked mesh's middle point lies outside `uₕ`'s mesh
    # here, where a probe would answer with the fill's type.
    @testset "Interpolated source element type" begin
        Ms = mesh(domain(interval(0.0, 0.4)), 11, true)
        Ws = gridspace(Ms)
        Wₕ = gridspace(mesh(domain(interval(0.0, 1.0)), 21, true))
        g = s -> begin
            uₛ = element(Ws, typeof(s))
            parent(uₛ) .= [s * x^2 for x in points(Ms)]
            assemble(form(Wₕ, v -> innerₕ(πₕ(uₛ; outside = 0.0), v)))
        end
        @test !iszero(g(1.0))
        @test ForwardDiff.derivative(g, 2.0) ≈ g(1.0) rtol = 1.0e-12

        W32 = gridspace(mesh(domain(interval(0.0f0, 1.0f0)), 21, true))
        u64 = Rₕ(Ws, x -> 1 / 3 + x[1])
        @test eltype(assemble(form(W32, v -> innerₕ(πₕ(u64; outside = 0.0f0), v)))) ===
              Float64
    end

    # A fill of another type than the values (`outside = 0` against `Float64`) is converted
    # to the blend's type, so the refill evaluating it at every point stays at 0 bytes.
    @testset "Interpolated source with an integer fill" begin
        for (Ma, Mb, f) in (
            (mesh(domain(interval(0.0, 1.0)), 9, true),
                mesh(domain(interval(0.0, 1.0)), 65, true), x -> sin(x[1])),
            (mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (6, 7), (true, true)),
                mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (33, 33),
                    (true, true)), x -> x[1] * x[2]))

            Wa, Wb = gridspace(Ma), gridspace(Mb)
            ua = Rₕ(Wa, f)
            L = form(Wb, v -> innerₕ(πₕ(ua; outside = 0), v))
            b = assemble(L)
            @test eltype(b) === Float64
            @test b ≈ assemble(form(Wb, v -> innerₕ(πₕ(ua; outside = 0.0), v)))
            @test alloc_test(assemble!, b, L) == 0
        end
    end
end

end # module
