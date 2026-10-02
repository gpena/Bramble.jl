module SpaceInterpolationTests

using Test
using Bramble
using Bramble: D₋ₓ, Mₓ, interpolation_matrix
using SparseArrays: sparse
using LinearAlgebra: Diagonal, issymmetric
using Bramble: form, assemble, weights, Innerh, CompositeGridSpace, TrialFunction,
               TestFunction, IndexedTrialFunction, IndexedTestFunction, InterpolationNode,
               SourceVector, SourceConstant, component, stencil_offsets, expression,
               host_points, _interp_triplet_frac, _is_source_only, _same_operator_shape,
               _all_trial_interpolated, _all_test_interpolated, _has_trial_interp,
               _has_test_interp, _bind_interp_spaces

# `interpolate_at` is the piecewise (multi)linear interpolant of a grid function, evaluable
# at any physical point, not only at its own mesh's points. `πₕ!`/`πₕ` are the numeric
# interpolation operator, exactly `Rₕ!`/`Rₕ` applied to `x -> interpolate_at(src, x)`, named
# after `Rₕ`/`Rₕ!`'s own convention, sharing the name `πₕ` with the one-argument symbolic
# wrapper (operators/interpolation.jl), told apart by arity. The checks below verify
# the interpolant's own correctness (exact on affine data, correct on a non-uniform mesh,
# clamped rather than extrapolated past the boundary) and transfers between distinct meshes
# (moving a grid function between two leaves of a heterogeneous composite space).

@testset "Interpolation" begin
    @testset "1D exact on affine" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 9, false)   # non-uniform
        uₕ = Rₕ(gridspace(Ωₕ), x -> 2x + 3)

        for x in (0.0, 0.37, 0.6321, 1.0)
            @test interpolate_at(uₕ, x) ≈ 2x + 3 atol=1e-12
        end

        # a grid point itself is returned exactly, not approximated by its neighbours
        pt = points(Ωₕ)[5]
        @test interpolate_at(uₕ, pt) ≈ 2pt + 3 atol=1e-12
    end

    @testset "2D exact on affine" begin
        Ωₕ = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (6, 7), (true, true))
        uₕ = Rₕ(gridspace(Ωₕ), x -> 2x[1] - 3x[2] + 1)

        for xt in ((0.0, 0.0), (0.42, 0.61), (1.0, 1.0), (0.99, 0.01))
            @test interpolate_at(uₕ, xt) ≈ 2xt[1] - 3xt[2] + 1 atol=1e-10
        end
    end

    @testset "Cross-mesh interpolation" begin
        Ωbig = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (10, 10), (true, true))
        Ωsmall = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (4, 4), (true, true))
        Wbig, Wsmall = gridspace(Ωbig), gridspace(Ωsmall)

        src = Rₕ(Wsmall, x -> x[1])   # affine, so the interpolant reproduces it exactly
        exact = Rₕ(Wbig, x -> x[1])

        dest = πₕ(Wbig, src)
        @test dest isa Bramble.VectorElement
        @test space(dest) === Wbig
        @test parent(dest) ≈ parent(exact) atol=1e-10

        # the in-place form agrees with the out-of-place one
        dest2 = similar(dest)
        returned = πₕ!(dest2, src)
        @test returned === dest2
        @test parent(dest2) ≈ parent(dest)
    end

    @testset "Matrix agreement" begin
        # P * parent(src) is exactly the same computation πₕ performs pointwise:
        # same corner-weight arithmetic, just emitted as triplets instead of accumulated,
        # so the two must agree to the last bit, not merely approximately.
        Ω1dest = mesh(domain(interval(0.0, 1.0)), 9, false)
        Ω1src = mesh(domain(interval(0.0, 1.0)), 5, true)
        W1dest, W1src = gridspace(Ω1dest), gridspace(Ω1src)
        src1 = Rₕ(W1src, x -> sin(3x) + x^2)
        P1 = interpolation_matrix(W1dest, W1src)
        @test size(P1) == (ndofs(W1dest), ndofs(W1src))
        @test P1 * parent(src1) ≈ parent(πₕ(W1dest, src1))

        Ω2dest = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (11, 9), (true, true))
        Ω2src = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (4, 6), (true, true))
        W2dest, W2src = gridspace(Ω2dest), gridspace(Ω2src)
        src2 = Rₕ(W2src, x -> x[1] * x[2] + x[1])
        P2 = interpolation_matrix(W2dest, W2src)
        @test size(P2) == (ndofs(W2dest), ndofs(W2src))
        @test P2 * parent(src2) ≈ parent(πₕ(W2dest, src2))

        # exact for affine data, same as interpolate_at itself
        exact = Rₕ(W2dest, x -> x[1] * 2 - x[2])
        srcaffine = Rₕ(W2src, x -> x[1] * 2 - x[2])
        @test interpolation_matrix(W2dest, W2src) * parent(srcaffine) ≈ parent(exact) atol=1e-10

        # at most 2^D = 4 nonzeros per row, and every row sums to 1 (a partition of unity,
        # since the corner weights of any cell always sum to 1)
        nnz_per_row = vec(sum(!iszero, P2, dims = 2))
        @test all(<=(4), nnz_per_row)
        @test all(≈(1), vec(sum(P2, dims = 2)))
    end

    @testset "Collapsed axis: per-axis Kronecker" begin
        # The pointwise and matrix paths match the per-axis Kronecker product.
        # A collapsed axis (a single point, from a zero-length interval) has no cell to
        # interpolate across, so it must contribute a 1×1 identity factor. Each pair is
        # nested by 2: the fine mesh is the coarse one refined once, which leaves the
        # collapsed axis at one point.
        axis(d, c) = d == c ? interval(0.5, 0.5) : interval(0.0, 1.0)
        per_axis(Ωf, Ωc, d) = npoints(Ωf(d)) == 1 ? sparse(ones(1, 1)) :
                              interpolation_matrix(gridspace(Ωf(d)), gridspace(Ωc(d)))

        for D in (2, 3), c in 1:D, unif in (true, false)
            Ωc = mesh(domain(reduce(×, ntuple(d -> axis(d, c), D))),
                ntuple(d -> d == c ? 1 : 3 + d, D), ntuple(_ -> unif, D))
            Ωf = deepcopy(Ωc)
            iterative_refinement!(Ωf)
            @test npoints(Ωf, Tuple) == ntuple(d -> d == c ? 1 : 2 * (3 + d) - 1, D)

            P = interpolation_matrix(gridspace(Ωf), gridspace(Ωc))
            K = reduce(kron, reverse(ntuple(d -> per_axis(Ωf, Ωc, d), D)))
            @test size(P) == size(K)
            @test isapprox(Matrix(P), Matrix(K); rtol = 1e-14, atol = 1e-14)

            # The pointwise interpolant agrees with the matrix on the same pair.
            src = Rₕ(gridspace(Ωc), x -> sum(x) + prod(x))
            @test P * parent(src) ≈ parent(πₕ(gridspace(Ωf), src))
        end
    end

    # The matrix is precomputed.
    @testset "πₕ! with an interpolation_matrix (#14)" begin
        # The whole point of building P once: this must agree with the pointwise path
        # (which re-locates every destination point's cell on every call) to the last bit,
        # not merely approximately -- same reasoning as "Matrix agreement" above, one level
        # up (the in-place operator, not the raw matrix product).
        Ω1dest = mesh(domain(interval(0.0, 1.0)), 9, false)
        Ω1src = mesh(domain(interval(0.0, 1.0)), 5, true)
        W1dest, W1src = gridspace(Ω1dest), gridspace(Ω1src)
        src1 = Rₕ(W1src, x -> sin(3x) + x^2)
        P1 = interpolation_matrix(W1dest, W1src)

        dest1 = similar(πₕ(W1dest, src1))
        returned = πₕ!(dest1, P1, src1)
        @test returned === dest1
        @test parent(dest1) ≈ parent(πₕ(W1dest, src1))

        Ω2dest = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (11, 9), (true, true))
        Ω2src = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (4, 6), (true, true))
        W2dest, W2src = gridspace(Ω2dest), gridspace(Ω2src)
        src2 = Rₕ(W2src, x -> x[1] * x[2] + x[1])
        P2 = interpolation_matrix(W2dest, W2src)

        dest2 = similar(πₕ(W2dest, src2))
        πₕ!(dest2, P2, src2)
        @test parent(dest2) ≈ parent(πₕ(W2dest, src2))

        # Across repeated calls.
        @testset "Tracks a live-updated src" begin
            for factor in (1.0, 2.5, -1.0)
                Rₕ!(src2, x -> factor * (x[1] * x[2] + x[1]))
                πₕ!(dest2, P2, src2)
                @test parent(dest2) ≈ parent(πₕ(W2dest, src2))
            end
        end

        @testset "Zero allocations" begin
            function _bytes(dest, P, src)
                πₕ!(dest, P, src)
                return @allocated πₕ!(dest, P, src)
            end
            @test _bytes(dest2, P2, src2) == 0
        end

        @testset "A mismatched P throws DimensionMismatch" begin
            Wwrong = gridspace(
                mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (3, 3), (true, true))
            )
            @test_throws DimensionMismatch πₕ!(Bramble.element(Wwrong), P2, src2)
        end
    end

    @testset "Operator composition" begin
        # once πₕ returns an ordinary VectorElement, every existing numeric
        # operator just works on it with no separate mechanism needed.
        Ωbig = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (8, 8), (true, true))
        Ωsmall = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (3, 3), (true, true))
        Wbig, Wsmall = gridspace(Ωbig), gridspace(Ωsmall)
        src = Rₕ(Wsmall, x -> x[1]^2 + x[2])

        dest = πₕ(Wbig, src)
        dx = D₋ₓ(dest)
        mx = Mₓ(dest)
        @test space(dx) === Wbig
        @test space(mx) === Wbig
        @test all(isfinite, parent(dx))
        @test all(isfinite, parent(mx))
    end

    # The `outside` policies (gpena/Bramble.jl#223) on non-uniform meshes. The oracle is the
    # affine function itself: the interpolant reproduces it exactly inside the domain, and
    # the boundary cell's own slope is the function's slope, so `:extrapolate` reproduces it
    # outside too, while `:clamp` reads it at the nearest boundary point.
    @testset "Out-of-domain policies" begin
        g1(x) = 2x + 3
        u1 = Rₕ(gridspace(mesh(domain(interval(0.0, 1.0)), 9, false)), g1)
        @test_throws ArgumentError interpolate_at(u1, 1.5)
        @test interpolate_at(u1, 1.5; outside = :clamp) ≈ g1(1.0)
        @test interpolate_at(u1, -0.25; outside = :clamp) ≈ g1(0.0)
        @test interpolate_at(u1, 1.5; outside = :extrapolate) ≈ g1(1.5)
        @test interpolate_at(u1, 1.5; outside = 0.0) === 0.0
        @test isnan(interpolate_at(u1, -0.25; outside = NaN))
        # a point within the endpoint tolerance is on the boundary under every policy
        @test interpolate_at(u1, 1.0 + eps(1.0)) ≈ g1(1.0)
        @test interpolate_at(u1, 1.0 + eps(1.0); outside = 0.0) ≈ g1(1.0)
        @test_throws ArgumentError interpolate_at(u1, 0.5; outside = :wrap)
        @test_throws ArgumentError interpolate_at(u1, 0.5; outside = "clamp")

        g2(x) = 2x[1] - 3x[2] + 1
        Ω2 = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (6, 5), (false, false))
        u2 = Rₕ(gridspace(Ω2), g2)
        # a fill value with the point inside: the blend, not the fill
        @test interpolate_at(u2, (0.37, 0.61); outside = 0.0) ≈ g2((0.37, 0.61))
        # outside along either axis alone returns the fill outright
        @test interpolate_at(u2, (1.5, 0.5); outside = 0.0) === 0.0
        @test interpolate_at(u2, (0.5, -0.3); outside = -7.0) === -7.0
        @test_throws ArgumentError interpolate_at(u2, (1.5, 0.5))
        @test interpolate_at(u2, (1.5, 0.5); outside = :clamp) ≈ g2((1.0, 0.5))
        @test interpolate_at(u2, (1.5, -0.3); outside = :extrapolate) ≈ g2((1.5, -0.3))

        # the matrix: a destination reaching past the source domain, on both sides
        Wsrc = gridspace(mesh(domain(interval(0.0, 1.0)), 7, false))
        Wdest = gridspace(mesh(domain(interval(-0.5, 2.0)), 11, false))
        src = Rₕ(Wsrc, g1)
        xd = points(mesh(Wdest))
        @test_throws ArgumentError interpolation_matrix(Wdest, Wsrc)
        @test_throws ArgumentError interpolation_matrix(Wdest, Wsrc; outside = 0.0)
        @test interpolation_matrix(Wdest, Wsrc; outside = :extrapolate) * parent(src) ≈ g1.(xd)
        @test interpolation_matrix(Wdest, Wsrc; outside = :clamp) * parent(src) ≈
              g1.(clamp.(xd, 0.0, 1.0))
        # the pointwise operator under a fill value: zero exactly where the point is outside
        @test parent(πₕ(Wdest, src; outside = 0.0)) ≈
              [0.0 <= x <= 1.0 ? g1(x) : 0.0 for x in xd]
    end

    # `_interp_triplet_frac` for a 1D source given its points as a 1-tuple: the method that
    # resolves Aqua's ambiguity between the flat-vector and the `NTuple{D}` methods.
    # `host_points(::Mesh1D)` never returns a 1-tuple, so it is called directly. Oracle: the
    # returned cell brackets `x` and its fraction reproduces `x` as a convex combination of
    # the cell's two (non-uniformly spaced) end points.
    @testset "1D triplet fraction over a 1-tuple of points" begin
        Ωs = mesh(domain(interval(0.0, 1.0)), 8, false)
        pts = collect(points(Ωs))
        for x in (0.0, 0.2345, 0.71, 1.0)
            i, t = _interp_triplet_frac(Ωs, (host_points(Ωs),), x, :error)
            @test pts[i] <= x <= pts[i + 1]
            @test (1 - t) * pts[i] + t * pts[i + 1] ≈ x atol=1e-14
            @test (i, t) == _interp_triplet_frac(Ωs, host_points(Ωs), x, :error)
        end
        @test _interp_triplet_frac(Ωs, (host_points(Ωs),), 1.5, :clamp) ==
              (length(pts) - 1, 1.0)
        @test_throws ArgumentError _interp_triplet_frac(Ωs, (host_points(Ωs),), 1.5, :error)
    end

    # The symbolic source `πₕ(uₕ)` in a linear form, read through its `GridInterpolant` at
    # every point of a test mesh reaching past the source domain. Oracle: the quadrature
    # weight times the affine function where the point is inside, times the fill outside.
    @testset "Symbolic source with a fill value" begin
        g(x) = 2x + 3
        Ws = gridspace(mesh(domain(interval(0.0, 1.0)), 7, false))
        Wt = gridspace(mesh(domain(interval(0.0, 2.0)), 13, false))
        uₛ = Rₕ(Ws, g)
        w = collect(weights(Wt, Innerh()))
        xt = points(mesh(Wt))
        b = assemble(form(Wt, v -> innerₕ(πₕ(uₛ; outside = 0.0), v)))
        @test b ≈ w .* [x <= 1.0 ? g(x) : 0.0 for x in xt]
        @test expression(πₕ(uₛ)) == "f"
    end

    # The bilinear `πₕ(u)` on non-uniform meshes: a 2D trial-side block, the 1D test-side
    # block, a component taken outside the interpolation on a composite space, and a sum
    # carrying an interpolation on both sides. Oracles act on affine data, which the
    # interpolant reproduces exactly, so they never pass through `interpolation_matrix`.
    @testset "Bilinear interpolation operator" begin
        Hh(W) = Diagonal(collect(weights(W, Innerh())))

        Ω2 = domain(box((0.0, 0.0), (1.0, 1.0)))
        Wt2 = gridspace(mesh(Ω2, (7, 6), (false, false)))
        Ws2 = gridspace(mesh(Ω2, (4, 5), (false, false)))
        g2(x) = 1 + 2x[1] - x[2]
        A2 = assemble(form(Ws2, Wt2, (u, v) -> innerₕ(πₕ(u), v)))
        @test size(A2) == (ndofs(Wt2), ndofs(Ws2))
        @test A2 * parent(Rₕ(Ws2, g2)) ≈ Hh(Wt2) * parent(Rₕ(Wt2, g2))

        # test side: vᵀ A 1 = Σ over the trial grid of w(x) · (πₕ v)(x)
        Ω1 = domain(interval(0.0, 1.0))
        Wu = gridspace(mesh(Ω1, 9, false))
        Wv = gridspace(mesh(Ω1, 6, false))
        g1(x) = 3x - 1
        At = assemble(form(Wu, Wv, (u, v) -> innerₕ(u, πₕ(v))))
        @test size(At) == (ndofs(Wv), ndofs(Wu))
        @test transpose(parent(Rₕ(Wv, g1))) * At * ones(ndofs(Wu)) ≈
              sum(collect(weights(Wu, Innerh())) .* g1.(points(mesh(Wu))))

        # a component taken outside the interpolation is the interpolation of the component
        Wbig = gridspace(mesh(Ω1, 9, false))
        Wsml = gridspace(mesh(Ω1, 5, false))
        Vh = CompositeGridSpace((Wbig, Wsml))
        nb, ns = ndofs(Wbig), ndofs(Wsml)
        u, v = TrialFunction{1}(), TestFunction{1}()
        @test component(πₕ(u), 2) === πₕ(u(2))
        @test component(πₕ(v; outside = :clamp), 1) === πₕ(v(1); outside = :clamp)
        Ac = assemble(form(Vh, Vh, (u, v) -> innerₕ(πₕ(u)(2), v(1))))
        @test Ac[1:nb, (nb + 1):(nb + ns)] * parent(Rₕ(Wsml, g1)) ≈
              Hh(Wbig) * parent(Rₕ(Wbig, g1))
        @test iszero(Ac[1:nb, 1:nb])
        @test iszero(Ac[(nb + 1):(nb + ns), :])

        # a sum interpolating on the trial side, against an interpolated test side: refused
        @test_throws ArgumentError innerₕ(πₕ(u) + u, πₕ(v))
        @test_throws ArgumentError innerₕ(u + πₕ(u), πₕ(v))

        # interpolations on opposite sides are never the same shape: on one mesh each term
        # assembles to H (P is the identity), but the symmetry predicate stays conservative
        a = form(Wu, Wu, (u, v) -> innerₕ(πₕ(u), v) + innerₕ(u, πₕ(v)))
        @test assemble(a) ≈ 2 * Hh(Wu)
        @test !issymmetric(a)
        @test !_same_operator_shape(πₕ(u), πₕ(v))
    end

    # The type-level traits that tell assembly which side of a term interpolates. Each one is
    # a constant method, so the oracle is the rule stated beside it in
    # operators/interpolation.jl: a node contributing no trial column (no test row) answers
    # `true` vacuously, an uninterpolated leaf answers `false`.
    @testset "Interpolation traits" begin
        W = gridspace(mesh(domain(interval(0.0, 1.0)), 6, false))
        u, v = TrialFunction{1}(), TestFunction{1}()
        uᵢ, vᵢ = IndexedTrialFunction{1}(2), IndexedTestFunction{1}(2)
        sf = πₕ(Rₕ(W, x -> x))
        sv = SourceVector{1, Vector{Float64}}(ones(ndofs(W)))
        sc = SourceConstant{1, Float64}(2.0)
        δ = dirac(0.5)
        lp = innerₕ(sf, v)

        @test _is_source_only(πₕ(u)) === false
        @test stencil_offsets(πₕ(u)) == stencil_offsets(u)
        @test stencil_offsets(πₕ(v)) == stencil_offsets(v)

        @test _all_trial_interpolated(u) === false
        @test all(op -> _all_trial_interpolated(op) === true, (sf, sv, sc, δ, v, vᵢ, lp))
        @test _all_test_interpolated(v) === false
        @test all(op -> _all_test_interpolated(op) === true, (sf, sv, sc, δ, u, uᵢ, lp))

        @test _has_trial_interp(u) === false
        @test _has_trial_interp(lp) === false
        @test _has_trial_interp(πₕ(u) + u) === true
        @test _has_trial_interp(u + u) === false
        @test _has_test_interp(v) === false
        @test _has_test_interp(lp) === false

        # binding walks a tuple of terms element by element, leaving uninterpolated ones as
        # they are
        Wv = gridspace(mesh(domain(interval(0.0, 1.0)), 4, false))
        bt = _bind_interp_spaces((πₕ(u), D₋ₓ(πₕ(u)), πₕ(v), u), W, Wv)
        @test bt isa NTuple{4, Any}
        @test bt[1].src_space === W
        @test bt[2].inner_op.src_space === W
        @test bt[3].src_space === Wv
        @test bt[4] === u
    end
end

end # module SpaceInterpolationTests
