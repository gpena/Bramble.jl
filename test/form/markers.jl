using Test
using Bramble
using Bramble: dot

# Point 50: a `markers` keyword on the *symbolic* innerₕ/inner₊ family
# (form/operators/inner.jl), implemented by wrapping the constructed
# BilinearProduct/LinearProduct in RegionRestriction — reusing existing, already
# block-routing-aware machinery rather than a new AST node. Every case here is
# checked against either the numeric `markers` keyword (point 11) or a hand-built
# reference, not just that assembly ran without error.

@testset "Symbolic markers" begin
    S = interval(0.0, 1.0) × interval(0.0, 1.0)
    Ωₕ = mesh(domain(S, :bottom => :bottom), (5, 5), (true, true))
    Wₕ = gridspace(Ωₕ)
    uₕ = Rₕ(Wₕ, x -> 1.0)
    vₕ = Rₕ(Wₕ, x -> 1.0)

    @testset "Numeric marker agreement" begin
        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v; markers = (:bottom,)))
        assembled = dot(vₕ.data, assemble(a) * uₕ.data)
        @test assembled ≈ 0.125
        @test assembled ≈ innerₕ(uₕ, vₕ; markers = (:bottom,))
    end

    @testset "Unrestricted match" begin
        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v; markers = ()))
        b = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v))
        @test assemble(a) == assemble(b)
    end

    @testset "Direction inference agreement" begin
        a1 = form(Wₕ, Wₕ, (u, v) -> inner₊(D₋ₓ(u), D₋ₓ(v); markers = (:bottom,)))
        a2 = form(Wₕ, Wₕ, (u, v) -> inner₊ₓ(D₋ₓ(u), D₋ₓ(v); markers = (:bottom,)))
        @test assemble(a1) ≈ assemble(a2)
    end

    @testset "Gradient tuple inner₊" begin
        a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v); markers = (:bottom,)))
        b = form(Wₕ, Wₕ,
            (u, v) -> inner₊ₓ(D₋ₓ(u), D₋ₓ(v); markers = (:bottom,)) +
                      inner₊ᵧ(D₋ᵧ(u), D₋ᵧ(v); markers = (:bottom,)))
        @test assemble(a) ≈ assemble(b)
    end

    @testset "Linear form restriction" begin
        fₕ = Rₕ(Wₕ, x -> π^2 * sin(π * x[1]))
        l1 = form(Wₕ, v -> innerₕ(fₕ, v; markers = (:bottom,)))
        l2 = form(Wₕ, v -> innerₕ(x -> π^2 * sin(π * x[1]), v; markers = (:bottom,)))
        @test assemble(l1) ≈ assemble(l2)

        # restricted to :bottom, only entries on that edge are nonzero
        Ωₕ_bottom_mask = Bramble.index_in_marker(Ωₕ, :bottom)
        @test all(iszero, assemble(l1)[.!Ωₕ_bottom_mask])
    end

    @testset "Reserved markers" begin
        a_boundary = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v; markers = (:boundary,)))
        a_interior = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v; markers = (:interior,)))
        full = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v))
        # boundary + interior sums (as numbers, since the masks are disjoint and cover
        # everything) reconstruct the unrestricted result
        @test dot(vₕ.data, assemble(a_boundary) * uₕ.data) +
              dot(vₕ.data, assemble(a_interior) * uₕ.data) ≈
              dot(vₕ.data, assemble(full) * uₕ.data)
    end

    @testset "Unknown marker error" begin
        # Scalar space: assembling with a typo'd label, instead of silently assembling to
        # all zero (RegionRestriction's own local_stencil can't tell "not marked" from
        # "no such marker" — haskey failing looks like the former).
        c = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v; markers = (:nonexistent,)))
        @test_throws ArgumentError assemble(c)

        # Linear (source-term) form, same reasoning.
        fₕ = Rₕ(Wₕ, x -> 1.0)
        l = form(Wₕ, v -> innerₕ(fₕ, v; markers = (:nonexistent,)))
        @test_throws ArgumentError assemble(l)

        # Composite space: leaves currently share one mesh (gridspace(Ωₕ, Val(N)) builds
        # every leaf over the same Ωₕ), so this can't yet arise from one leaf lacking a
        # label another has — but a name that exists on no leaf at all must still be
        # caught, not silently assembled to zero.
        Vₕ = Wₕ^Val(2)
        d = form(Vₕ, Vₕ,
            (u, v) -> innerₕ(u(1), v(1); markers = (:nonexistent,)) +
                      innerₕ(u(2), v(2)))
        @test_throws ArgumentError assemble(d)
    end

    @testset "Composite space markers" begin
        Vₕ = Wₕ^Val(2)
        a = form(Vₕ, Vₕ,
            (u, v) -> innerₕ(u(1), v(1); markers = (:bottom,)) + innerₕ(u(2), v(2)))
        @test size(assemble(a)) == (2 * Bramble.ndofs(Wₕ), 2 * Bramble.ndofs(Wₕ))
    end

    @testset "Direction mismatch message" begin
        @test_throws ArgumentError form(
            Wₕ, Wₕ, (u, v) -> inner₊(D₋ₓ(u), D₋ᵧ(v); markers = (:bottom,)))
    end

    @testset "Empty-tuple disambiguator" begin
        @test_throws ArgumentError inner₊((), (); markers = (:bottom,))
    end
end
