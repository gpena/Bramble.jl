module MeshLevelSetClassTests

using Test
using LinearAlgebra
using Bramble
using ..TestUtils: _Const

# `Bramble._levelset_class(φ, box)` sorts a curve φ = 0 into the closure path (the crossing
# map T = σₕ∘σᵥ has finite order), the orbit path (T conjugate to an irrational rotation)
# or a named `ArgumentError` (#834). Rotation numbers are checked against values derived here:
# an ellipse is an affine image of the circle, so T is conjugate to the rotation by twice the
# angle β between S e₁ and S e₂ (S the square root of the quadratic form): ρ ∈ {β/π, 1 − β/π}.

# The ellipse ((u/a)² + (v/b)² = 1 in axes rotated by θ, centred at z) and its exact ρ pair.
function ellipse(a, b, θ, z)
    c, s = cos(θ), sin(θ)
    φ = p -> ((c * (p[1] - z[1]) + s * (p[2] - z[2])) / a)^2 +
             ((-s * (p[1] - z[1]) + c * (p[2] - z[2])) / b)^2 - 1
    R = [c s; -s c]
    S = sqrt(Symmetric(R' * Diagonal([1 / a^2, 1 / b^2]) * R))
    β = acos(dot(S[:, 1], S[:, 2]) / (norm(S[:, 1]) * norm(S[:, 2])))
    return φ, (β / π, 1 - β / π)
end

function egg(θ, k)
    c, s = cos(θ), sin(θ)
    return function (p)
        x, y = p[1] - 0.1, p[2] + 0.05
        u, v = c * x + s * y, -s * x + c * y
        return u^2 + v^2 * (1 + k * u) / 0.36 - 1
    end
end

const EB = ((-1.2, 1.3), (-1.25, 1.15))

function message(φ, box)
    try
        Bramble._levelset_class(φ, box)
    catch e
        e isa ArgumentError && return sprint(showerror, e)
        rethrow()
    end
    return ""
end

@testset "Level-set curve classification (#834)" begin
    @testset "Symmetric curves take the closure path" begin
        # One grid-aligned symmetry axis gives T² = id, so ρ = 1/2.
        disk = p -> hypot(p[1], p[2]) - 1
        @test Bramble._levelset_class(disk, ((-1.25, 1.25), (-1.25, 1.25))) == (path = :closure, ρ = 0.5)
        φ, _ = ellipse(1.0, 0.6, 0.0, (0.2, -0.1))
        @test Bramble._levelset_class(φ, ((-1.0, 1.4), (-0.9, 0.7))) == (path = :closure, ρ = 0.5)
        @test Bramble._levelset_class(egg(0.0, 0.4), EB) == (path = :closure, ρ = 0.5)
    end

    @testset "Box element types are accepted" begin
        disk = p -> hypot(p[1], p[2]) - 1
        @test Bramble._levelset_class(disk, ((-2, 2), (-1.5f0, 1.5f0))) == (path = :closure, ρ = 0.5)
    end

    @testset "Rotated ellipses take the orbit path" begin
        for (a, b, θ, z, box) in ((1.0, 0.6, π / 6, (0.0, 0.0), ((-1.25, 1.25), (-1.25, 1.25))),
            (1.0, 0.5, deg2rad(50), (0.1, 0.2), ((-1.1, 1.3), (-1.0, 1.4))))
            φ, ρs = ellipse(a, b, θ, z)
            r = Bramble._levelset_class(φ, box)
            @test r.path === :orbit
            @test minimum(abs.(r.ρ .- ρs)) < 1e-12
        end
    end

    @testset "A 5-fold ellipse takes the closure path" begin
        # Semi-axes 1 and b = √(5 + 2√5) at 45° give cos β = (b² − 1)/(b² + 1) = cos(π/5), so
        # ρ = 1/5 and T⁵ = id: the q ∈ 5:60 return of `_levelset_class`.
        φ, ρs = ellipse(1.0, sqrt(5 + 2sqrt(5)), π / 4, (0.0, 0.0))
        @test ρs[1] ≈ 1 / 5 atol = 1e-12
        r = Bramble._levelset_class(φ, ((-2.6, 2.6), (-2.6, 2.6)))
        @test r.path === :closure
        @test minimum(abs.(r.ρ .- ρs)) < 1e-12
    end

    @testset "A tilted egg takes the orbit path" begin
        # Prototype value (architect report §2.3, weighted Birkhoff at 32000 steps).
        r = Bramble._levelset_class(egg(deg2rad(35), 0.4), EB)
        @test r.path === :orbit
        @test abs(r.ρ - 0.35287719427212) < 1e-11
    end

    @testset "A star is not orthogonally convex" begin
        star = p -> hypot(p[1], p[2]) - (0.6 + 0.2 * sin(5 * atan(p[2], p[1])))
        m = message(star, ((-1.0, 1.0), (-1.0, 1.0)))
        @test occursin(r"grid line [xy] = \S+ meets φ = 0 in [46] points", m)
        @test occursin("orthogonally convex", m)
    end

    @testset "A tilted clover is phase-locked" begin
        clover = p -> hypot(p[1], p[2]) - (1 + 0.15 * cos(4 * (atan(p[2], p[1]) - deg2rad(10))))
        m = message(clover, ((-1.4, 1.4), (-1.4, 1.4)))
        @test occursin("phase-locked at rotation number 1/2", m)
        @test !occursin("nearly", m)
    end

    @testset "A nearly symmetric egg is nearly locked" begin
        m = message(egg(deg2rad(0.2), 0.8), EB)
        @test occursin("nearly phase-locked (rotation number ≈ 0.4999998", m)
    end

    @testset "A curved band is not star-shaped" begin
        band = p -> ((p[2] - p[1]^3) / 0.075)^2 + p[1]^8 - 1
        @test occursin("must be star-shaped", message(band, ((-1.3, 1.3), (-1.3, 1.3))))
    end

    @testset "A swapped box throws" begin
        disk = p -> hypot(p[1], p[2]) - 1
        @test occursin("must have x₀ < x₁ and y₀ < y₁", message(disk, ((1.25, -1.25), (-1.25, 1.25))))
        @test occursin("must have x₀ < x₁ and y₀ < y₁", message(disk, ((-1.25, 1.25), (1.25, -1.25))))
    end

    @testset "A disk far from the origin closes" begin
        # T² = id holds to the coordinates' ulps, coarser than 1e-10 of the box at x = 1e6.
        far = p -> hypot(p[1] - 1e6, p[2]) - 1
        @test Bramble._levelset_class(far, ((1e6 - 1.25, 1e6 + 1.25), (-1.25, 1.25))) == (path = :closure, ρ = 0.5)
    end

    @testset "The result type is inferred" begin
        disk = p -> hypot(p[1], p[2]) - 1
        r = @inferred Bramble._levelset_class(disk, ((-1.25, 1.25), (-1.25, 1.25)))
        @test r isa @NamedTuple{path::Symbol, ρ::Float64}
    end

    @testset "Curves reaching the box throw" begin
        disk = p -> hypot(p[1], p[2]) - 1
        @test occursin("reaches the side x = -1.0 of the box ((-1.0, 1.0), (-1.5, 1.5))",
            message(disk, ((-1.0, 1.0), (-1.5, 1.5))))
        @test occursin("reaches the side", message(disk, ((-0.8, 1.5), (-1.5, 1.5))))
        @test occursin("reaches the side", message(disk, ((-0.5, 0.5), (-0.5, 0.5))))
        # Tangent to a side between two of its samples.
        off = p -> hypot(p[1], p[2] - 0.1234567891) - 1
        @test occursin("reaches the side", message(off, ((-1.5, 1.0), (-1.5, 1.5))))
        offy = p -> hypot(p[1] - 0.1234567891, p[2]) - 1
        @test occursin("reaches the side y = 1.0", message(offy, ((-1.5, 1.5), (-1.5, 1.0))))
        @test occursin("region φ < 0 is empty", message(p -> hypot(p[1], p[2]) + 1, ((-1.0, 1.0), (-1.0, 1.0))))
    end

    @testset "A nonnegative φ throws the empty error" begin
        B = ((0.0, 1.0), (0.0, 1.0))
        @test occursin("nonnegative at every sampled point of the box ((0.0, 1.0), (0.0, 1.0))",
            message(_Const(1.0), B))
        @test_throws ArgumentError Bramble._levelset_class(_Const(1.0), B)
    end

    @testset "The box-side test ignores φ's scale" begin
        B = ((-1.25, 1.25), (-1.25, 1.25))
        # φ grows by e^66 from the curve to the sides, and overflows to Inf on them.
        @test Bramble._levelset_class(p -> exp(50 * (p[1]^2 + p[2]^2)) - exp(12.5), B).path === :closure
        @test Bramble._levelset_class(p -> exp(1000 * (p[1]^2 + p[2]^2)) - exp(250.0), B).path === :closure
        # NaN on part of a side the disk crosses.
        nan = p -> p[1] > 1.2 ? NaN : hypot(p[1], p[2]) - 1
        @test occursin("reaches the side", message(nan, ((-1.25, 1.25), (-0.9, 1.25))))
        # NaN beside a tangency must not pull the golden-section bracket off it.
        Y0 = 0.1234567891
        for δ in (1e-4, 1e-9)
            half = p -> (p[1] >= 1.0 && p[2] > Y0 + δ) ? NaN : hypot(p[1], p[2] - Y0) - 1
            @test occursin("reaches the side", message(half, ((-1.5, 1.0), (-1.5, 1.5))))
        end
        # A side 1e-3 from the disk is clear.
        disk = p -> hypot(p[1], p[2]) - 1
        @test Bramble._levelset_class(disk, ((-1.001, 1.25), (-1.25, 1.25))).path === :closure
    end
end

end # module
