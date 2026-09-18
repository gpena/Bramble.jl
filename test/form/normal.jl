module FormNormalTests

using Test
using Bramble
using LinearAlgebra: dot
using ..TestUtils: @test_allocs

# The outward normal, as a discrete grid function and as a symbol inside a form
# (gpena/Bramble.jl#213).
#
# Every assertion here is against an analytically known flux -- an edge length, a sign, the
# divergence theorem -- rather than against a second call to the same code. The sign
# convention is what most of it is about: a normal term that is right up to sign integrates
# to zero over a closed boundary and looks plausible until someone solves a Neumann problem
# with it.

@testset "The outward normal" begin
    Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))

    @testset "normal_vector on a grid space samples the face" begin
        Ωₕ = mesh(Ω, (9, 8), (true, true))
        Wₕ = gridspace(Ωₕ)

        for (marker, ν) in (
            (:xmin, (-1.0, 0.0)), (:xmax, (1.0, 0.0)),
            (:ymin, (0.0, -1.0)), (:ymax, (0.0, 1.0))
        )
            nₕ = normal_vector(Wₕ, marker)
            @test normal_vector(Ωₕ, marker) == ν
            # the face carries the constant vector and nothing else does, so each component
            # sums to its value times the number of points on the face
            npts = marker in (:xmin, :xmax) ? 8 : 9
            @test sum(parent(nₕ[1])) ≈ ν[1] * npts
            @test sum(parent(nₕ[2])) ≈ ν[2] * npts
            @test count(!iszero, parent(nₕ[1])) == (iszero(ν[1]) ? 0 : npts)
        end

        # a legacy alias names the same face as its canonical spelling
        @test parent(normal_vector(Wₕ, :bottom)[2]) == parent(normal_vector(Wₕ, :ymin)[2])
    end

    @testset "dot(F, n) carries the outward sign" begin
        Ωₕ = mesh(Ω, (9, 8), (true, true))
        Wₕ = gridspace(Ωₕ)
        # F = (1, 0): the flux is +1 through :xmax, -1 through :xmin, and 0 through the two y
        # faces, each times the length of the edge, which is 1 here
        F = (x -> 1.0, x -> 0.0)
        ones_ = Rₕ(Wₕ, x -> 1.0)
        flux(markers) = dot(
            assemble(form(Wₕ, v -> inner_Γ(dot(F, n), v; markers = markers))), parent(ones_)
        )

        @test flux((:xmax,)) ≈ 1.0
        @test flux((:xmin,)) ≈ -1.0
        @test flux((:ymin,)) ≈ 0.0 atol=1e-14
        @test flux((:ymax,)) ≈ 0.0 atol=1e-14
        # the two opposite faces cancel, which is the sign convention stated as an identity
        @test flux((:xmin, :xmax)) ≈ 0.0 atol=1e-12
    end

    @testset "The divergence theorem for a known flux" begin
        # ∮ F·n ds = ∫ div F dx. With F = (x, y) that is 2·area = 2 on the unit square, and
        # the lumped weights make it exact rather than convergent: F is linear on each face
        # and the trapezoidal weight integrates a linear function exactly.
        F = (x -> x[1], x -> x[2])
        for n_pts in ((9, 8), (17, 21), (33, 12))
            Wₕ = gridspace(mesh(Ω, n_pts, (true, true)))
            ones_ = Rₕ(Wₕ, x -> 1.0)
            l = form(Wₕ, v -> inner_Γ(dot(F, n), v; markers = (:boundary,)))
            @test dot(assemble(l), parent(ones_)) ≈ 2.0
        end

        # and on a non-uniform mesh, where a weight that silently assumed uniform spacing
        # would show up
        Wₕ = gridspace(mesh(Ω, (13, 11), (false, false)))
        ones_ = Rₕ(Wₕ, x -> 1.0)
        l = form(Wₕ, v -> inner_Γ(dot(F, n), v; markers = (:boundary,)))
        @test dot(assemble(l), parent(ones_)) ≈ 2.0
    end

    @testset "dot(F, n) with a known field" begin
        Wₕ = gridspace(mesh(Ω, (9, 8), (true, true)))
        ones_ = Rₕ(Wₕ, x -> 1.0)

        # F = (x, 0): F·n is +x on :xmax, so the integral over that edge is 1
        F = (x -> x[1], x -> 0.0)
        l = form(Wₕ, v -> inner_Γ(dot(F, n), v; markers = (:xmax,)))
        @test dot(assemble(l), parent(ones_)) ≈ 1.0
        # and -x on :xmin, where x = 0
        @test dot(assemble(form(Wₕ, v -> inner_Γ(dot(F, n), v; markers = (:xmin,)))),
            parent(ones_)) ≈ 0.0 atol=1e-14

        # grid functions work the same way, and `dot(n, F)` is the same term
        Fₕ = (Rₕ(Wₕ, x -> x[1]), Rₕ(Wₕ, x -> 0.0))
        @test dot(assemble(form(Wₕ, v -> inner_Γ(dot(Fₕ, n), v; markers = (:xmax,)))),
            parent(ones_)) ≈ 1.0
        @test assemble(form(Wₕ, v -> inner_Γ(dot(n, F), v; markers = (:xmax,)))) ≈
              assemble(l)
    end

    @testset "It refills in place at zero allocations" begin
        Wₕ = gridspace(mesh(Ω, (9, 9), (true, true)))
        a = form(Wₕ, Wₕ,
            (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + inner_Γ(u, v; markers = (:ymax,)))
        A = assemble(a)
        B = copy(A)
        assemble!(B, a)
        @test A ≈ B
        @test_allocs assemble!(B, a)
    end

    @testset "Refusals" begin
        Wₕ = gridspace(mesh(Ω, (7, 7), (true, true)))
        v = Bramble.TestFunction{2}()
        F = (x -> 1.0, x -> 0.0)
        @test_throws ArgumentError inner_Γ(dot(F, n), v; markers = (:inlet,))
        @test_throws ArgumentError inner_Γ(dot(F, n), v)
        @test_throws ArgumentError normal_vector(Wₕ, :inlet)
    end
end

end # module FormNormalTests
