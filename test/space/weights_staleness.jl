module SpaceWeightsStalenessTests

using Test
using Bramble
using ..TestUtils: alloc_test, @test_allocs

# `ScalarGridSpace` precomputes its inner-product weights once from the mesh at
# `gridspace(Ωₕ)` time (gpena/Bramble.jl#221). The mesh is mutable (`set_points!`,
# `change_points!`, `iterative_refinement!` all rewrite it in place), and nothing used to
# stop a space built before such a mutation from silently computing against weights for a
# mesh that no longer exists. Every check here is against an independent reference -- a
# freshly built `gridspace(Ωₕ)` after the mutation, or a hand-computed exact integral --
# never against another call to the code under test.

@testset "Grid space weights staleness (#221)" begin
    @testset "Reproducer: change_points! gives the exact numbers from the issue" begin
        Ω = domain(interval(0.0, 1.0))
        Ωₕ = mesh(Ω, 5, true)
        Wₕ = gridspace(Ωₕ)

        change_points!(Ωₕ, markers(Ω), [0.0, 0.1, 0.2, 0.3, 1.0])

        @test_throws ArgumentError weights(Wₕ)

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

    @testset "Every guarded entry point throws, not only weights() itself" begin
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

    @testset "2D (MeshnD): mutating one submesh stales the whole space" begin
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

    @testset "CompositeGridSpace: staleness in one leaf is caught, not only the mutated one silently ignored" begin
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

    @testset "A rebuilt space works correctly after mutation, no throw" begin
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

    @testset "Zero allocation on the untouched (fresh) success path" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 1001, true)
        Wₕ = gridspace(Ωₕ)
        u = Rₕ(Wₕ, x -> x[1]^2)
        v = Rₕ(Wₕ, x -> 1.0)

        @test_allocs innerₕ(u, v)
        @test_allocs normₕ(u)
        @test_allocs weights(Wₕ)
    end
end

end # module
