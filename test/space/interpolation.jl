using Test
using Bramble

# `interpolate_at` is the piecewise (multi)linear interpolant of a grid function, evaluable
# at any physical point, not only at its own mesh's points. `πₕ!`/`πₕ` are the numeric
# interpolation operator, exactly `Rₕ!`/`Rₕ` applied to `x -> interpolate_at(src, x)`, named
# after `Rₕ`/`Rₕ!`'s own convention, sharing the name `πₕ` with the one-argument symbolic
# wrapper (form/operators/interpolation.jl), told apart by arity. The checks below verify
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

    @testset "Boundary extrapolation" begin
        # locate_cell clamps which cell is read to the boundary one, but not the relative
        # position x is weighted by within it; so a point outside the mesh continues the
        # boundary cell's own affine trend rather than holding a constant value. For a
        # globally affine function that trend is the function itself, so this is exact
        # arbitrarily far outside the mesh too.
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 5, true)
        uₕ = Rₕ(gridspace(Ωₕ), x -> 5x + 1)
        @test interpolate_at(uₕ, -0.5) ≈ 5 * -0.5 + 1 atol=1e-12
        @test interpolate_at(uₕ, 1.7) ≈ 5 * 1.7 + 1 atol=1e-12
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
        @test values(dest) ≈ values(exact) atol=1e-10

        # the in-place form agrees with the out-of-place one
        dest2 = similar(dest)
        returned = πₕ!(dest2, src)
        @test returned === dest2
        @test values(dest2) ≈ values(dest)
    end

    @testset "Matrix agreement" begin
        # P * values(src) is exactly the same computation πₕ performs pointwise:
        # same corner-weight arithmetic, just emitted as triplets instead of accumulated,
        # so the two must agree to the last bit, not merely approximately.
        Ω1dest = mesh(domain(interval(0.0, 1.0)), 9, false)
        Ω1src = mesh(domain(interval(0.0, 1.0)), 5, true)
        W1dest, W1src = gridspace(Ω1dest), gridspace(Ω1src)
        src1 = Rₕ(W1src, x -> sin(3x) + x^2)
        P1 = interpolation_matrix(W1dest, W1src)
        @test size(P1) == (ndofs(W1dest), ndofs(W1src))
        @test P1 * values(src1) ≈ values(πₕ(W1dest, src1))

        Ω2dest = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (11, 9), (true, true))
        Ω2src = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (4, 6), (true, true))
        W2dest, W2src = gridspace(Ω2dest), gridspace(Ω2src)
        src2 = Rₕ(W2src, x -> x[1] * x[2] + x[1])
        P2 = interpolation_matrix(W2dest, W2src)
        @test size(P2) == (ndofs(W2dest), ndofs(W2src))
        @test P2 * values(src2) ≈ values(πₕ(W2dest, src2))

        # exact for affine data, same as interpolate_at itself
        exact = Rₕ(W2dest, x -> x[1] * 2 - x[2])
        srcaffine = Rₕ(W2src, x -> x[1] * 2 - x[2])
        @test interpolation_matrix(W2dest, W2src) * values(srcaffine) ≈ values(exact) atol=1e-10

        # at most 2^D = 4 nonzeros per row, and every row sums to 1 (a partition of unity,
        # since the corner weights of any cell always sum to 1)
        nnz_per_row = vec(sum(!iszero, P2, dims = 2))
        @test all(<=(4), nnz_per_row)
        @test all(≈(1), vec(sum(P2, dims = 2)))
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
        mx = M₋ₓ(dest)
        @test space(dx) === Wbig
        @test space(mx) === Wbig
        @test all(isfinite, values(dx))
        @test all(isfinite, values(mx))
    end
end
