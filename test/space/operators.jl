module SpaceOperatorsTests

using Test
using Bramble
using Bramble: divₕ!, curlₕ!, Δₕ!
using Bramble: D₋
using Bramble: Dcᵧ, Dc₂, Dcₓ, D̃ᵧ, D̃₂, D̃ₓ, D̽ᵧ, D̽₂, D̽ₓ, D₋ᵧ, D₋₂, D₋ₓ, Mᵧ, M₂, Mₓ
using Bramble: index_in_marker, jumpᵧ, jump₂, jumpₓ
using SparseArrays: SparseMatrixCSC, nnz
using LinearAlgebra: Diagonal
using ..TestUtils: MockDeviceArray, TEST_DIMS, WITH_3D_TESTS
using ..TestUtils: _Const, _x1, _x1_plus_x2, _sin_x1_times_x2, _x1_times_x2
using Bramble: components, restrict_to
# Internal: defined and documented, not exported.
import Bramble: D₊ₓ, D₊ᵧ, D₊₂, D₊, div₊ₕ, curl₊ₕ, forward_star_difference
# `M₊*` is `public`, not `export`ed (average.jl's own note on why); `kronecker_operator_matrix`
# is neither, the oracle `stencil_matrix` is checked against.
import Bramble: M₊ₓ, M₊ᵧ, M₊₂, kronecker_operator_matrix
using Bramble:
               IdentityOperator,
               ZeroOperator,
               OperatorScale,
               GridFunctionScale,
               OperatorAdd,
               is_symbolic,
               space

@testset "Linear operators" begin
    for D in TEST_DIMS
        @testset "$(D)D" begin
            I = interval(0.0, 1.0)
            X = domain(reduce(×, ntuple(_ -> I, Val(D))))
            M = mesh(X, ntuple(_ -> 4, Val(D)), ntuple(_ -> false, Val(D)))
            W = gridspace(M)
            u = element(W)
            Rₕ!(u, _Const(1.0))

            x0 = IdentityOperator(W)
            @test space(x0) === W
            @test !is_symbolic(x0)

            z0 = ZeroOperator(W)
            @test fieldcount(typeof(z0)) == 0
            @test !is_symbolic(z0)

            # Scalar scaling
            x1 = 2 * x0
            @test x1 isa OperatorScale
            @test x1.scalar == 2
            @test x1.inner_op === x0
            @test !is_symbolic(x1)

            x1_div = x0 / 2
            @test x1_div isa OperatorScale
            @test x1_div.scalar == 0.5

            # Grid function scaling
            x2 = u * x0
            @test x2 isa GridFunctionScale
            @test x2.grid_function === u
            @test x2.inner_op === x0
            @test !is_symbolic(x2)

            # Operator addition and subtraction
            sum_op = x0 + x0
            @test sum_op isa OperatorAdd
            @test sum_op.left_op === x0
            @test sum_op.right_op === x0
            @test !is_symbolic(sum_op)

            diff_op = x0 - x0
            @test diff_op isa OperatorAdd
            @test diff_op.left_op === x0

            # String representation
            buf = IOBuffer()
            show(buf, x0)
            @test String(take!(buf)) == "I"
            show(buf, z0)
            @test String(take!(buf)) == "0"
        end
    end
end

# The discrete vector calculus operators.
#
# Each is a contraction of the directional differences, evaluated in one traversal rather
# than as nested operator calls, so the property that matters is that the fused form is the
# composition -- checked entry for entry against the operators it is built from, which are
# themselves pinned by `test/convergence/operators.jl`. Values alone would not catch a wrong
# boundary convention, since a truncated slice is self-consistently wrong.
@testset "Vector calculus operators (#158)" begin
    Ω1 = domain(interval(0.0, 1.0))
    Ω2 = domain(interval(0.0, 1.0) × interval(0.0, 2.0))
    Ω3 = domain(interval(0.0, 1.0) × interval(0.0, 2.0) × interval(0.0, 1.5))

    @testset "Δₕ is D̃(D₋(·)) summed over directions" begin
        for (Ω, n, unif) in (
            (Ω1, 21, true), (Ω1, 17, false),
            (Ω2, (9, 8), (true, true)), (Ω2, (11, 9), (false, false)),
            (WITH_3D_TESTS ? ((Ω3, (7, 6, 5), (true, true, true)),) : ())...
        )
            Ωₕ = mesh(Ω, n, unif)
            Wₕ = gridspace(Ωₕ)
            D = dim(Ωₕ)
            uₕ = Rₕ(Wₕ,
                x -> sin(1.3 * x[1]) + (D > 1 ? cos(0.9 * x[2]) : 0.0) +
                     (D > 2 ? x[3]^2 : 0.0))

            ref = zeros(ndofs(Wₕ))
            for d in 1:D
                ref .+= parent(forward_star_difference(D₋(uₕ, Val(d)), Val(d)))
            end
            # the accumulation order differs from the composition's, so 3D agrees to
            # round-off rather than to the bit
            @test parent(Δₕ(uₕ)) ≈ ref atol=1e-14
        end
    end

    @testset "Δₕ! matches Δₕ and refuses to alias" begin
        Wₕ = gridspace(mesh(Ω2, (9, 8), (true, true)))
        uₕ = Rₕ(Wₕ, _sin_x1_times_x2)
        vₕ = element(Wₕ)
        @test parent(Δₕ!(vₕ, uₕ)) == parent(Δₕ(uₕ))
        @test Δₕ!(vₕ, uₕ) === vₕ
        @test_throws ArgumentError Δₕ!(uₕ, uₕ)
    end

    @testset "The summation-by-parts identity holds" begin
        # innerₕ(Δₕ u, v) = -inner₊(∇ₕ u, ∇ₕ v) for grid functions vanishing on the
        # boundary. This is what makes this composition *the* discrete Laplacian rather than
        # one of several plausible five-point stencils, and it fails for a stencil that is
        # off by a spacing anywhere.
        for (n, unif) in (((11, 9), (true, true)), ((13, 11), (false, false)))
            Wₕ = gridspace(mesh(Ω2, n, unif))
            uₕ = Rₕ(Wₕ, x -> sin(π * x[1]) * sin(π * x[2] / 2))
            # deliberately not a mode orthogonal to `uₕ`: with two orthogonal modes both
            # sides are zero and the identity holds vacuously
            vₕ = Rₕ(Wₕ,
                x -> x[1] * (1 - x[1]) * x[2] * (2 - x[2]) * (1 + 0.5x[1] + 0.3x[2]))
            on_boundary = index_in_marker(mesh(Wₕ), :boundary)
            parent(uₕ)[on_boundary] .= 0.0
            parent(vₕ)[on_boundary] .= 0.0
            lhs = innerₕ(Δₕ(uₕ), vₕ)
            @test abs(lhs) > 1e-3
            @test lhs ≈ -inner₊(∇ₕ(uₕ), ∇ₕ(vₕ)) rtol=1e-12
        end
    end

    # The fused loops read their own spacing vectors (`_accumulate_backward!`,
    # `_accumulate_forward!`), so the meshes are non-uniform: on a uniform one an off-by-one
    # spacing index, or forward spacings where backward ones belong, gives the same numbers.
    @testset "divₕ sums the directional differences" begin
        Ωₕ = mesh(Ω2, (9, 8), (false, false))
        Wₕ = gridspace(Ωₕ)
        u1 = Rₕ(Wₕ, x -> x[1]^2 + x[2])
        u2 = Rₕ(Wₕ, _sin_x1_times_x2)

        @test parent(divₕ((u1, u2))) == parent(D₋ₓ(u1)) .+ parent(D₋ᵧ(u2))
        @test parent(div₊ₕ((u1, u2))) == parent(D₊ₓ(u1)) .+ parent(D₊ᵧ(u2))

        # a composite grid function is the same field, spelled the other way
        Vₕ = gridspace(Ωₕ, Val(2))
        cₕ = Rₕ(Vₕ, (x -> x[1]^2 + x[2], _sin_x1_times_x2))
        @test parent(divₕ(cₕ)) == parent(divₕ((u1, u2)))

        # and the in-place form writes the same thing
        wₕ = element(Wₕ)
        @test parent(divₕ!(wₕ, (u1, u2))) == parent(divₕ((u1, u2)))

        # one component per dimension, or it is an error rather than a silent answer
        @test_throws DimensionMismatch divₕ((u1,))
        @test_throws DimensionMismatch divₕ((u1, u2, u1))
    end

    @testset "curlₕ in 2D" begin
        Ωₕ = mesh(Ω2, (9, 8), (false, false))
        Wₕ = gridspace(Ωₕ)
        u1 = Rₕ(Wₕ, x -> x[1]^2 + x[2])
        u2 = Rₕ(Wₕ, _sin_x1_times_x2)
        @test parent(curlₕ((u1, u2))) == parent(D₋ₓ(u2)) .- parent(D₋ᵧ(u1))
        @test parent(curl₊ₕ((u1, u2))) == parent(D₊ₓ(u2)) .- parent(D₊ᵧ(u1))

        # curl of a gradient vanishes where both stencils are untruncated -- a discrete
        # identity, not an approximation, since both differences are backward. Checked on a
        # uniform mesh: on a random one, two near-coincident points leave round-off up to
        # ~1e-11, so no fixed bound is both safe and tight.
        w = Rₕ(gridspace(mesh(Ω2, (9, 8), (true, true))), x -> sin(x[1]) * cos(x[2]))
        interior = reshape(parent(curlₕ(∇ₕ(w))), (9, 8))[3:end, 3:end]
        @test maximum(abs, interior) < 1e-14

        # there is no 1D curl
        W1 = gridspace(mesh(Ω1, 9, true))
        @test_throws ArgumentError curlₕ((Rₕ(W1, sin),))
    end

    # The 3D curl builds 3D meshes, so it runs nightly (gpena/Bramble.jl#650).
    WITH_3D_TESTS && @testset "curlₕ in 3D" begin
        # three components, each the right combination
        Ω3ₕ = mesh(Ω3, (8, 7, 6), (false, false, false))
        W3 = gridspace(Ω3ₕ)
        f = (Rₕ(W3, _sin_x1_times_x2), Rₕ(W3, x -> x[3]^2),
            Rₕ(W3, x -> cos(x[2]) * x[1]))
        c = curlₕ(f)
        @test length(c) == 3
        @test parent(c[1]) == parent(D₋ᵧ(f[3])) .- parent(D₋₂(f[2]))
        @test parent(c[2]) == parent(D₋₂(f[1])) .- parent(D₋ₓ(f[3]))
        @test parent(c[3]) == parent(D₋ₓ(f[2])) .- parent(D₋ᵧ(f[1]))
        # and the forward curl, whose 3D fused loop nothing else here reaches
        c₊ = curl₊ₕ(f)
        @test parent(c₊[1]) == parent(D₊ᵧ(f[3])) .- parent(D₊₂(f[2]))
        @test parent(c₊[2]) == parent(D₊₂(f[1])) .- parent(D₊ₓ(f[3]))
        @test parent(c₊[3]) == parent(D₊ₓ(f[2])) .- parent(D₊ᵧ(f[1]))

        # the in-place 3D form writes into a 3-tuple
        dest = ntuple(_ -> element(W3), Val(3))
        curlₕ!(dest, f)
        @test all(k -> parent(dest[k]) == parent(c[k]), 1:3)
    end
end

# `stencil_matrix`: every family's public per-axis alias routes through the single-pass
# builder in `src/operators/stencil.jl`. The Kronecker products of shift matrices that
# `kronecker_operator_matrix` builds (`src/operators/shift.jl`) are the oracle. Checked entrywise,
# `nnz` included, on non-uniform meshes in 1D/2D/3D so a boundary weight that would only
# coincidentally match on a uniform grid cannot hide a mistake.
@testset "stencil_matrix vs Kronecker (#185)" begin
    meshes = (
        ("", mesh(domain(interval(0.0, 1.0)), 11, false)),
        ("", mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (9, 7), false)),
        (" uniform", mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (9, 7), true)),
        (WITH_3D_TESTS ?
         (("", mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (6, 5, 4), false)),) :
         ())...
    )
    # Each family's stencil taps along its axis: the stored pattern is one entry per tap
    # that stays on the grid, whatever its weight (#662).
    families = (
        (:D₋, (D₋ₓ, D₋ᵧ, D₋₂), (0, -1)),
        (:D₊, (D₊ₓ, D₊ᵧ, D₊₂), (1, 0)),
        (:D̃, (D̃ₓ, D̃ᵧ, D̃₂), (1, 0)),
        (:Dc, (Dcₓ, Dcᵧ, Dc₂), (1, -1)),
        (:D̽, (D̽ₓ, D̽ᵧ, D̽₂), (1, 0, -1)),
        (:jump, (jumpₓ, jumpᵧ, jump₂), (1, 0)),
        (:M, (Mₓ, Mᵧ, M₂), (0, -1)),
        (:M₊, (M₊ₓ, M₊ᵧ, M₊₂), (1, 0))
    )
    for (tag, Ωₕ) in meshes, (name, ops, taps) in families, d in 1:dim(Ωₕ)
        op = ops[d]
        @testset "$name axis $d, $(dim(Ωₕ))D$tag" begin
            A = op(Ωₕ)
            B = kronecker_operator_matrix(Ωₕ, op)
            N = npoints(Ωₕ)
            n = npoints(Ωₕ, Tuple)[d]
            @test A isa SparseMatrixCSC
            @test A == B
            @test nnz(A) == nnz(B)
            @test nnz(A) == sum(N ÷ n * (n - abs(t)) for t in taps)
        end
    end
end

# A zero weight (a truncated boundary row, or `D̽`'s diagonal on a uniform mesh) is stored
# as an explicit zero: the pattern is the stencil's, so it is the same on a uniform and a
# graded mesh, and both constructions keep it. Values match the dense backend bit for bit.
@testset "stored pattern is the stencil's (#662)" begin
    fams = (((D₋ₓ, D₋ᵧ), (0, -1)), ((D₊ₓ, D₊ᵧ), (1, 0)), ((D̃ₓ, D̃ᵧ), (1, 0)),
        ((Dcₓ, Dcᵧ), (1, -1)), ((D̽ₓ, D̽ᵧ), (1, 0, -1)), ((Mₓ, Mᵧ), (0, -1)),
        ((M₊ₓ, M₊ᵧ), (1, 0)), ((Bramble.Mcₓ, Bramble.Mcᵧ), (-1, 0, 1)),
        ((jumpₓ, jumpᵧ), (1, 0)))
    function mk(graded; kw...)
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 7), (true, true); kw...)
        graded && Bramble.change_points!(Ωₕ,
            (range(0.0, 1.0; length = 9) .^ 1.25, range(0.0, 1.0; length = 7) .^ 1.5))
        return Ωₕ
    end
    for graded in (false, true)
        Ωₕ = mk(graded)
        Ωd = mk(graded; backend = backend(matrix_type = Matrix{Float64}))
        N = npoints(Ωₕ)
        for (ops, taps) in fams, d in 1:2

            A = ops[d](Ωₕ)
            B = kronecker_operator_matrix(Ωₕ, ops[d])
            n = npoints(Ωₕ, Tuple)[d]
            @test all(Matrix(A) .=== ops[d](Ωd))
            @test A == B
            @test nnz(A) == sum(N ÷ n * (n - abs(t)) for t in taps)
            @test nnz(B) == nnz(A)
        end
    end
end

# The vectorial aliases (`∇ₕ`, `Mₕ`, `jumpₕ`, ...) destructure and index into the
# per-coordinate aliases they already apply through: `dx, dy = ∇ₕ` and `∇ₕ[1]`,
# `∇ₕ[:x]` are the exact same function object as `D₋ₓ`, not a copy, and the protocol is
# generated once in `@operator_family` for every family with a `vectorial_alias`. This file
# reaches the coordinate names through `Bramble.` rather than relying on the bare name
# staying exported.
# They index too.
@testset "Vectorial aliases destructure (#340)" begin
    Dₓ, Dᵧ, D₂ = Bramble.D₋ₓ, Bramble.D₋ᵧ, Bramble.D₋₂

    # Indexing too.
    @testset "destructuring matches named aliases" begin
        dx, dy = ∇ₕ
        @test dx === Dₓ && dy === Dᵧ

        @test ∇ₕ[1] === Dₓ && ∇ₕ[2] === Dᵧ && ∇ₕ[3] === D₂
        @test firstindex(∇ₕ) == 1 && lastindex(∇ₕ) == 3
        @test eltype(∇ₕ) === Function
        @test collect(∇ₕ) == [Dₓ, Dᵧ, D₂]

        @test_throws BoundsError ∇ₕ[0]
        @test_throws BoundsError ∇ₕ[4]
        @test_throws ArgumentError ∇ₕ[:w]
    end

    # Folds at compile time.
    @testset "indexing folds, zero allocation" begin
        second(V) = V[2]
        @test only(Base.return_types(second, (typeof(∇ₕ),))) === typeof(Dᵧ)
        @test @inferred(second(∇ₕ)) === Dᵧ
        second(∇ₕ) # warm up first
        @test (@allocated second(∇ₕ)) == 0
    end

    # Each supports the protocol.
    @testset "every vectorial_alias family" begin
        families = (
            (∇ₕ, (Bramble.D₋ₓ, Bramble.D₋ᵧ, Bramble.D₋₂)),
            (∇cₕ, (Bramble.Dcₓ, Bramble.Dcᵧ, Bramble.Dc₂)),
            (∇̽ₕ, (Bramble.D̽ₓ, Bramble.D̽ᵧ, Bramble.D̽₂)),
            (∇̃ₕ, (Bramble.D̃ₓ, Bramble.D̃ᵧ, Bramble.D̃₂)),
            (Mₕ, (Bramble.Mₓ, Bramble.Mᵧ, Bramble.M₂)),
            (Mcₕ, (Bramble.Mcₓ, Bramble.Mcᵧ, Bramble.Mc₂)),
            (jumpₕ, (Bramble.jumpₓ, Bramble.jumpᵧ, Bramble.jump₂)),
            (Bramble.∇₊ₕ, (D₊ₓ, D₊ᵧ, D₊₂)),
            (Bramble.M₊ₕ, (M₊ₓ, M₊ᵧ, M₊₂))
        )
        for (V, ops) in families
            a, b, c = V
            @test (a, b, c) === ops
            @test (V[1], V[2], V[3]) === ops
            @test (V[:x], V[:y], V[:z]) === ops
            @test length(V) == 3
        end
    end
end

# `MockDeviceArray` (test/TestUtils.jl) is a stand-in for a vendor GPU array: host storage that
# answers `DeviceLocality()`, so the offloaded projection path (`GpuOffload`) can be driven
# with no GPU.
# `Rₕ!`/`avgₕ!` share one driver, `project!` (src/operators/projection.jl). These cover the
# branches the rest of the suite does not reach on a host: the offloaded path, a marked
# region with no points, the per-leaf rule tuple on a scalar space, the quadrature options
# and the scattered cell average on a composite space.
@testset "Projection paths (project!)" begin
    # Two points along x, so every point is on the boundary and `:interior` is empty.
    Ω = domain(interval(0.0, 1.0) × interval(0.0, 2.0))

    @testset "an empty marked region writes zeros" begin
        Ωₕ = mesh(Ω, (2, 5), (false, false))
        @test !any(index_in_marker(Ωₕ, :interior))
        Wₕ = gridspace(Ωₕ)
        # the probe finds no marked point, so it samples `f` at the first grid point
        r = Rₕ(Wₕ, x -> 1.0 + x[1] * x[2]; markers = (:interior,))
        @test eltype(parent(r)) === Float64
        @test length(parent(r)) == ndofs(Wₕ)
        @test all(iszero, parent(r))
        Vₕ = gridspace(Ωₕ, Val(2))
        c = Rₕ(Vₕ, x -> (x[1], x[2]); markers = (:interior,))
        @test all(k -> all(iszero, parent(components(c)[k])), 1:2)
    end

    @testset "GpuOffload: host dest, device buffer" begin
        dev = backend(
            vector_type = MockDeviceArray{Float32, 1}, matrix_type = MockDeviceArray{Float32, 2},
            policy = Bramble.GpuKernel()
        )
        be = backend(Float32; policy = Bramble.GpuOffload(dev))
        Ω32 = domain(interval(0.0f0, 1.0f0) × interval(0.0f0, 2.0f0))
        Ωₕ = mesh(Ω32, (2, 5), (false, false); backend = be)
        Wₕ = gridspace(Ωₕ)
        Vₕ = gridspace(Ωₕ, Val(2))
        @test !any(index_in_marker(Ωₕ, :interior))

        # Masked onto the empty region, the device buffer is zeroed, nothing is launched, and
        # the zeros are copied back over whatever the destination held.
        u = element(Wₕ)
        parent(u) .= 1
        @test Rₕ!(u, _x1_plus_x2; markers = (:interior,)) === u
        @test parent(u) == zeros(Float32, ndofs(Wₕ))
        # the composite's leaves are views into one vector: each is copied back on its own
        c = element(Vₕ)
        parent(c) .= 1
        Rₕ!(c, x -> (x[1], x[2]); markers = (:interior,))
        @test parent(components(c)[1]) == zeros(Float32, ndofs(Wₕ))
        @test parent(components(c)[2]) == zeros(Float32, ndofs(Wₕ))

        # A Float64 destination is not the device's element type, so it stays on the host
        # sweep and gets the exact values; checked against the points read off the mesh.
        Ω4ₕ = mesh(Ω32, (4, 5), (false, false); backend = be)
        f = x -> 1.0 + x[1] + 3.0 * x[2]
        r = Rₕ(gridspace(Ω4ₕ), f; markers = (:interior,))
        @test eltype(parent(r)) === Float64
        xs = Bramble.points(Ω4ₕ)
        mask = index_in_marker(Ω4ₕ, :interior)
        @test 0 < count(mask) < length(mask)
        expected = vec([mask[k] ? f((xs[1][I[1]], xs[2][I[2]])) : 0.0
                        for (k, I) in enumerate(CartesianIndices(npoints(Ω4ₕ, Tuple)))])
        @test parent(r) == expected

        # An unmasked fill needs the device kernel, which only the KernelAbstractions
        # extension provides: refused by name, not by a scalar-indexing failure.
        msg = try
            Rₕ!(element(Wₕ), _x1)
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("KernelAbstractions", msg)
    end

    @testset "avgₕ!: quadrature options and leaves" begin
        Ωₕ = mesh(Ω, (6, 5), (false, false))
        Wₕ = gridspace(Ωₕ)
        x1, x2 = Bramble.half_points(Ωₕ)
        # Exact cell averages of a quadratic and a bilinear function over [a₁, b₁] × [a₂, b₂],
        # the cell between consecutive half points; two Gauss points integrate both exactly.
        f = x -> x[1]^2 + 3.0 * x[2]
        g = _x1_times_x2
        cells = CartesianIndices((6, 5))
        ref_f = [(x1[I[1] + 1]^3 - x1[I[1]]^3) / (3 * (x1[I[1] + 1] - x1[I[1]])) +
                 1.5 * (x2[I[2]] + x2[I[2] + 1]) for I in cells][:]
        ref_g = [(x1[I[1]] + x1[I[1] + 1]) / 2 * (x2[I[2]] + x2[I[2] + 1]) / 2 for I in cells][:]

        u = element(Wₕ)
        @test parent(avgₕ!(u, f; quad_points = 2)) ≈ ref_f rtol=1e-13
        @test parent(avgₕ!(u, f; quad_points = Val(2))) ≈ ref_f rtol=1e-13
        @test_throws ArgumentError avgₕ!(u, f; quad_points = 0)
        @test_throws ArgumentError avgₕ!(u, f; quad_points = Val(0))
        # a one-tuple of functions on a scalar space is the function itself
        fill!(parent(u), 0.0)
        @test parent(avgₕ!(u, (f,), Val(2))) ≈ ref_f rtol=1e-13

        # one function returning both leaves: averaged once per cell, scattered per leaf
        c = avgₕ(gridspace(Ωₕ, Val(2)), x -> (f(x), g(x)); quad_points = Val(2))
        @test parent(components(c)[1]) ≈ ref_f rtol=1e-13
        @test parent(components(c)[2]) ≈ ref_g rtol=1e-13
    end

    @testset "avgₕ!: quad_points^D calls per cell" begin
        # gpena/Bramble.jl#693: a seed evaluated f once more per cell. Every method of
        # `_cell_average` (1D, 2D, 3D and the generic one, reached here at D = 4) must call
        # f exactly NQ^D times, and a constant 1 must average to 1, which a node counted
        # twice would not.
        calls = Ref(0)
        one_f = x -> (calls[] += 1; 1.0)
        xs = [0.0, 0.3, 1.0]
        for NQ in 1:3
            nodes, wts = Bramble._gauss_rule(Val(NQ), Float64)
            for D in 1:4
                x = D == 1 ? xs : ntuple(_ -> xs, D)
                idx = D == 1 ? 2 : CartesianIndex(ntuple(k -> isodd(k) ? 1 : 2, D))
                calls[] = 0
                @test Bramble._cell_average(one_f, x, idx, nodes, wts) ≈ 1.0 rtol=1e-14
                @test calls[] == NQ^D
            end
        end

        # through avgₕ! on a mesh: the step from quad_points = 1 to 2 costs 2² - 1 = 3 calls
        # per cell, so it isolates the per-cell count from any one-off type probe call
        Wₕ = gridspace(mesh(Ω, (5, 4), (false, false)))
        g = x -> (calls[] += 1; sin(x[1]) * x[2])
        count_calls(q) = (calls[] = 0; avgₕ!(element(Wₕ), g; quad_points = q); calls[])
        c1, c2 = count_calls(1), count_calls(2)
        ncells = (c2 - c1) ÷ 3
        @test ncells == 20
        @test c1 - ncells in (0, 1)
    end

    @testset "Gauss rule at run-time precision" begin
        # BigFloat is not isbits, so the rule is built per call at the current precision.
        # Gauss-Legendre on [0, 1]: two points at 1/2 ∓ √3/6 with weight 1/2 each, three at
        # 1/2 ∓ √15/10 and 1/2 with weights 5/18, 8/18, 5/18.
        nodes2, wts2 = Bramble._gauss_rule(Val(2), BigFloat)
        @test nodes2 isa NTuple{2, BigFloat}
        @test all(isapprox.(nodes2, (0.5 - sqrt(big(3)) / 6, 0.5 + sqrt(big(3)) / 6); atol = 1e-60))
        @test all(isapprox.(wts2, (big(1) / 2, big(1) / 2); atol = 1e-60))
        nodes3, wts3 = Bramble._gauss_rule(Val(3), BigFloat)
        @test all(isapprox.(nodes3, (0.5 - sqrt(big(15)) / 10, big(1) / 2, 0.5 + sqrt(big(15)) / 10);
            atol = 1e-60))
        @test all(isapprox.(wts3, (big(5) / 18, big(8) / 18, big(5) / 18); atol = 1e-60))
    end
end

# `restrict_to` (src/operators/region_restriction.jl) on a non-uniform mesh, against the
# matrices it should equal: the mass matrix with the columns outside the region zeroed, and
# a difference applied after that zeroing.
@testset "Region restriction" begin
    Ω = domain(interval(0.0, 1.0) × interval(0.0, 2.0), :bottom => :bottom, :left => :left)
    Ωₕ = mesh(Ω, (7, 6), (false, false))
    Wₕ = gridspace(Ωₕ)
    M = assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v)))
    keep(mask) = Diagonal(Float64.(mask))

    # a tuple of regions is their union
    union_mask = index_in_marker(Ωₕ, :bottom) .| index_in_marker(Ωₕ, :left)
    @test 0 < count(union_mask) < ndofs(Wₕ)
    A = assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(restrict_to((:bottom, :left), u), v)))
    @test A == M * keep(union_mask)

    # under a difference, each tap reads the region at its own point
    interior = index_in_marker(Ωₕ, :interior)
    B = assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(D₋ₓ(restrict_to(:interior, u)), v)))
    @test B ≈ M * D₋ₓ(Ωₕ) * keep(interior) rtol=1e-14
    @test B != M * D₋ₓ(Ωₕ)
end

end # module SpaceOperatorsTests
