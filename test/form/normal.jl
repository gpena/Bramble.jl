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

    @testset "Only the facet slice is set, 1D to 3D, non-uniform (#333)" begin
        # Independent reference: the face is found from the geometric normal alone.
        function reference(Ωₕ, marker, D)
            ν = normal_vector(Ωₕ, marker)
            np = npoints(Ωₕ, Tuple)
            axis = findfirst(!iszero, ν)
            side = ν[axis] < 0 ? 1 : np[axis]
            out = ntuple(_ -> zeros(prod(np)), D)
            for (k, I) in enumerate(CartesianIndices(np))
                I[axis] == side || continue
                for d in 1:D
                    out[d][k] = ν[d]
                end
            end
            return out
        end
        cases = (
            (domain(interval(0.0, 1.0)), 9, false, (:xmin, :xmax, :left, :right)),
            (domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (7, 6), (false, false),
                (:xmin, :xmax, :ymin, :ymax, :left, :right, :bottom, :top)),
            (domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 3.0)),
                (5, 6, 4), (false, false, false),
                (:xmin, :xmax, :ymin, :ymax, :zmin, :zmax,
                    :back, :front, :left, :right, :bottom, :top))
        )
        for (D, (S, sz, unif, labels)) in enumerate(cases)
            Ωₕ = mesh(S, sz, unif)
            Wₕ = gridspace(Ωₕ)
            for m in labels
                nₕ = normal_vector(Wₕ, m)
                ref = reference(Ωₕ, m, D)
                @test all(parent(nₕ[d]) == ref[d] for d in 1:D)
            end
        end
    end

    @testset "Components of n inside inner_Γ, 2D and 3D, non-uniform (#341)" begin
        # Independent reference: g is affine, so the lumped face weights integrate it exactly,
        # and its integral over a face is the face's area times g at the face's centroid.
        # The component n[d] is ±1 on the two faces normal to axis d and 0 on the others.
        labels = ((:xmin, :xmax), (:ymin, :ymax), (:zmin, :zmax))
        function face_flux(L, axis, side, d, g)
            axis == d || return 0.0
            area = prod(L[e] for e in eachindex(L) if e != axis)
            c = ntuple(e -> e == axis ? (side == 1 ? 0.0 : L[e]) : L[e] / 2, length(L))
            return (side == 1 ? -1.0 : 1.0) * area * g(c)
        end
        cases = (
            (domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (1.0, 2.0), (9, 7)),
            (domain(interval(0.0, 1.0) × interval(0.0, 2.0) × interval(0.0, 3.0)),
                (1.0, 2.0, 3.0), (5, 6, 4))
        )
        for (S, L, sz) in cases
            D = length(L)
            Wₕ = gridspace(mesh(S, sz, ntuple(_ -> false, D)))
            ones_ = parent(Rₕ(Wₕ, x -> 1.0))
            g = x -> 1.0 + x[1] + 2x[2] + (D == 3 ? 3x[3] : 0.0)
            gₕ = parent(Rₕ(Wₕ, g))
            lin(h, m) = assemble(form(Wₕ, v -> inner_Γ(h, v; markers = m)))
            bil(h, m) = assemble(form(Wₕ, Wₕ, (u, v) -> inner_Γ(h(u), v; markers = m)))

            # destructuring, integer and symbol indexing name the same singletons
            comps = Tuple(n)[1:D]
            @test length(n) == 3
            @test comps === ntuple(d -> n[d], D)
            @test n[:x] === n[1] && n[:y] === n[2] && n[:z] === n[3]
            @test n[end] === n[3] && n[begin] === n[1] && n[Int32(2)] === n[2]
            @test Base.issingletontype(typeof(n[1]))

            for axis in 1:D, side in 1:2, d in 1:D
                m = (labels[axis][side],)
                ref = face_flux(L, axis, side, d, g)
                c = comps[d]
                # a coefficient function, from either side
                @test dot(lin(g * c, m), ones_) ≈ ref atol=1e-12
                @test lin(c * g, m) == lin(g * c, m)
                # a grid function
                @test dot(lin(Rₕ(Wₕ, g) * c, m), ones_) ≈ ref atol=1e-12
                # a trial function, from either side: ones' A gₕ is the same surface integral
                A = bil(u -> u * c, m)
                @test dot(ones_, A * gₕ) ≈ ref atol=1e-12
                @test bil(u -> c * u, m) == A
                # an AST expression: a scaled trial function
                @test bil(u -> (2.0 * u) * c, m) ≈ 2.0 * A
            end

            # Σ_d F_d n[d] == dot(F, n), for a field that is not affine
            F = ntuple(d -> (x -> sin(d + x[1]) * x[2] + d * x[end]^2), D)
            m = (:boundary,)
            @test sum(lin(F[d] * comps[d], m) for d in 1:D) ≈ lin(dot(F, n), m)
            @test sum(bil(u -> u * comps[d], m) for d in 1:D) ≈
                  bil(u -> dot(ntuple(_ -> u, D), n), m)

            # the same sum written as one linear combination inside inner_Γ
            combo = foldl(+, ntuple(d -> F[d] * comps[d], D))
            @test lin(combo, m) ≈ lin(dot(F, n), m)
            @test bil(u -> foldl(+, ntuple(d -> (d * u) * comps[d], D)), m) ≈
                  bil(u -> dot(ntuple(d -> d * u, D), n), m)
            # scalar multiples, negation and differences of component terms
            @test lin(2 * (F[1] * comps[1]), m) ≈ 2 * lin(F[1] * comps[1], m)
            @test lin((F[1] * comps[1]) * 0.5, m) ≈ 0.5 * lin(F[1] * comps[1], m)
            @test lin(3.0 * combo, m) ≈ 3.0 * lin(dot(F, n), m)
            @test lin(-combo, m) ≈ -lin(dot(F, n), m)
            @test bil(u -> -(u * comps[D]), m) ≈ -bil(u -> u * comps[D], m)
            @test lin(F[1] * comps[1] - F[2] * comps[2], m) ≈
                  lin(F[1] * comps[1], m) - lin(F[2] * comps[2], m)
            # a bare component is the component times one, alone or in a sum
            for d in 1:D
                @test dot(lin(comps[d], (labels[d][2],)), ones_) ≈
                      face_flux(L, d, 2, d, x -> 1.0) atol=1e-12
            end
            @test lin(comps[1] + g * comps[2], m) ≈ lin(1.0 * comps[1], m) + lin(g * comps[2], m)
        end
    end

    @testset "A component refills in place at zero allocations" begin
        Wₕ = gridspace(mesh(Ω, (9, 8), (false, false)))
        nx, ny = n
        a = form(Wₕ, Wₕ,
            (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) +
                      inner_Γ(u * nx - 2.0u * ny, v; markers = (:xmax, :ymin)))
        A = assemble(a)
        B = copy(A)
        assemble!(B, a)
        @test A ≈ B
        @test_allocs assemble!(B, a)
        second(V) = V[2]
        @test only(Base.return_types(second, (typeof(n),))) === typeof(ny)
    end

    @testset "Refusals" begin
        Wₕ = gridspace(mesh(Ω, (7, 7), (true, true)))
        v = Bramble.TestFunction{2}()
        F = (x -> 1.0, x -> 0.0)
        @test_throws ArgumentError inner_Γ(dot(F, n), v; markers = (:inlet,))
        @test_throws ArgumentError inner_Γ(dot(F, n), v)
        @test_throws ArgumentError normal_vector(Wₕ, :inlet)
        # a component of n is refused outside inner_Γ, by name, and past the form's dimension
        f = x -> 1.0
        nx = n[1]
        @test_throws ArgumentError innerₕ(f * nx, v)
        @test_throws ArgumentError innerₕ(v, f * nx)
        @test_throws ArgumentError inner₊(f * nx, v)
        @test_throws ArgumentError inner_Γ(f * n[3], v; markers = (:boundary,))
        @test_throws ArgumentError inner_Γ(f * nx, v)
        @test_throws BoundsError n[4]
        @test_throws BoundsError n[0]
        # products of two components, on either side and inside inner_Γ
        ny = n[2]
        @test_throws ArgumentError nx * ny
        @test_throws ArgumentError f * nx * ny
        @test_throws ArgumentError (f * nx) * (f * ny)
        @test_throws ArgumentError inner_Γ(f * nx, f * ny; markers = (:boundary,))
        # a bare component, or a sum of them, outside inner_Γ
        @test_throws ArgumentError innerₕ(nx, v)
        @test_throws ArgumentError innerₕ(f * nx, f * ny)
        @test_throws ArgumentError inner₊(v, f * nx + f * ny)
        @test_throws ArgumentError f * nx + v
        @test_throws ArgumentError v - f * nx
        # the component belongs with the flux, on the left
        @test_throws ArgumentError inner_Γ(v, f * nx; markers = (:boundary,))
        # the dimension message names one component in 1D, not "1 components"
        v1 = Bramble.TestFunction{1}()
        msg = try
            inner_Γ(f * ny, v1; markers = (:boundary,))
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("has 1 component here", msg)
        @test_throws ArgumentError n[:w]
    end
end

end # module FormNormalTests
