module SpaceInnerProductTests

using Test
using Bramble
using LinearAlgebra: norm
using Supposition
using ..TestUtils: WITH_SLOW_TESTS, @test_allocs
using ..SpaceVectorElementsTests: setup_test_grid, valid_interior_range

@testset "Inner products & norms" begin
    for D in 1:3
        dims, Wh, u = setup_test_grid(Val(D))

        # normₕ of the all-ones element is `sqrt(<domain measure>)` in every dimension --
        # asserted against the named measure in the per-dimension testsets below.
        if D == 1
            @test norm₊(D₋ₓ(u)) ≈ 0.0
        end

        u .= 1.0
        der = ∇ₕ(u)

        if D == 1
            @test norm(der[valid_interior_range(1, dims)...]) ≈ 0.0
        else
            for i in 1:D
                dd = reshape(der[i], dims)
                @views ee = dd[valid_interior_range(i, dims)...]
                @test norm(ee) ≈ 0.0
            end
        end

        wf(x, i) = x[i]
        for dimension in 1:D
            Rₕ!(u, Base.Fix2(wf, dimension))
            der = ∇ₕ(u)

            if D == 1
                @views ee = der[valid_interior_range(1, dims)...]
                @test norm(ee .- 1.0) ≈ 0.0
            else
                for i in 1:D
                    dd = reshape(der[i].data, dims)
                    @views ee = dd[valid_interior_range(i, dims)...]
                    expected_value = i != dimension ? 0.0 : 1.0
                    @test norm(ee .- expected_value) ≈ 0.0
                end
            end
        end
    end

    @testset "1D" begin
        dims_1d, Wₕ_1d, u1 = setup_test_grid(Val(1))
        domain_length = 5.0 # Domain is [-1, 4]

        u2 = u1 * 2.0
        u3 = similar(u1)
        Rₕ!(u3, x->x)

        @testset "innerₕ" begin
            # (1, 2) = ∫ 1*2 dx = 2 * length = 2 * 5 = 10
            @test innerₕ(u1, u2) ≈ 2.0 * domain_length

            # ||1||² = ∫ 1*1 dx = length = 5
            @test innerₕ(u1, u1) ≈ domain_length
            @test normₕ(u1) ≈ sqrt(domain_length)
        end

        @testset "inner₊" begin
            # In 1D, inner₊ should equal inner₊ₓ
            @test inner₊(u1, u2) ≈ inner₊ₓ(u1, u2)

            # The test grid is nonuniform, but for constant functions, the integral should still yield the exact measure.
            @test inner₊(u1, u2) ≈ 2.0 * domain_length

            res_tuple = inner₊(u1, u3, Tuple)
            @test res_tuple isa NTuple{1, Float64}
            @test res_tuple[1] ≈ inner₊ₓ(u1, u3)

            @test norm₊(u1)^2 ≈ inner₊(u1, u1)
        end

        @testset "norm₁ₕ" begin
            # For u(x) = 2x, u'(x) = 2.
            Rₕ!(u1, x->2x)
            # |u|²_1h = ||∇u||²₊ ≈ ∫ (2)^2 dx = 4 * length = 4 * 5 = 20
            @test snorm₁ₕ(u1)^2 ≈ 4.0 * domain_length
        end
    end

    @testset "2D" begin
        dims_2d, Wₕ_2d, u1 = setup_test_grid(Val(2))
        domain_area = 25.0 # Domain is [-1, 4] x [-1, 4]

        u2 = u1 * 2.0
        ux = similar(u1)
        uy = similar(u1)
        Rₕ!(ux, x->x[1])
        Rₕ!(uy, x->x[2])

        @testset "L² and modified L²" begin
            # (1, 2) = ∫∫ 1*2 dx dy = 2 * area = 50
            @test innerₕ(u1, u2) ≈ 2.0 * domain_area
            @test normₕ(u1) ≈ sqrt(domain_area)

            # Test sum of directional components
            @test inner₊(ux, uy) ≈ inner₊ₓ(ux, uy) + inner₊ᵧ(ux, uy)
        end

        @testset "Tuple methods" begin
            res_tuple = inner₊(ux, uy, Tuple)
            @test res_tuple isa NTuple{2, Float64}
            @test res_tuple[1] ≈ inner₊ₓ(ux, uy)
            @test res_tuple[2] ≈ inner₊ᵧ(ux, uy)

            U = (ux, uy)
            V = (u1, u2)
            expected = inner₊ₓ(ux, u1) + inner₊ᵧ(uy, u2)
            @test inner₊(U, V) ≈ expected
        end

        @testset "norm₁ₕ" begin
            # For u(x,y) = x + 2y, ∇u = (1, 2)
            Rₕ!(u1, x -> x[1] + 2*x[2])

            expected_value_snorm = sum(
                i^2 * sum(Bramble.weights(Wₕ_2d, Bramble.Innerplus(), i)) for i in 1:2
            )
            @test snorm₁ₕ(u1)^2 ≈ expected_value_snorm
        end
    end

    @testset "3D" begin
        dims_3d, Wₕ_3d, u1 = setup_test_grid(Val(3))
        domain_volume = 125.0 # Domain is [-1, 4]³

        u2 = u1 * 2.0
        uz = similar(u1)
        Rₕ!(uz, x -> x[3])

        @testset "L² and modified L²" begin
            @test innerₕ(u1, u2) ≈ 2.0 * domain_volume
            @test normₕ(u1) ≈ sqrt(domain_volume)
            @test inner₊(u1, uz) ≈ inner₊ₓ(u1, uz) + inner₊ᵧ(u1, uz) + inner₊₂(u1, uz)
        end

        @testset "norm₁ₕ" begin
            # For u(x,y,z) = x+2y+3z, ∇u = (1, 2, 3)
            Rₕ!(u1, x -> x[1] + 2x[2] + 3x[3])
            expected_value_snorm = sum(
                i^2 * sum(Bramble.weights(Wₕ_3d, Bramble.Innerplus(), i)) for i in 1:3
            )

            @test snorm₁ₕ(u1)^2 ≈ expected_value_snorm
        end
    end
end

# Runtime `inner₊(uₕ, vₕ, Val(S))` for every staggered set `S ⊆ 1:D` (gpena/Bramble.jl#115,
# #234): the existing four (`innerₕ`, `inner₊ₓ`, `inner₊ᵧ`, `inner₊₂`) are the `S = ()` and
# singleton cases, kept as aliases sharing this implementation; every other `S` -- a pair, or
# the full `1:D` set -- is new with this milestone and reduces against the lazy
# `SeparableWeights` `weights(Wₕ, Val(S))` returns for it.
@testset "inner₊(u, v, Val(S)) for every staggered set (#234)" begin
    # A weight for `S`, built directly from `spacing`/`cell_measure` rather than from
    # `weights`/`SpaceWeights`: entry `I` is the product, over every axis `d`, of the
    # backward spacing at `I[d]` (zeroed at `I[d] == 1`, where a backward difference has no
    # stencil) when `d ∈ S`, and the cell-measure factor otherwise -- the same formula
    # `weights(Wₕ, Val(S))`'s own docstring states, re-derived here rather than trusted.
    function hand_weights(Ωₕ, S, ::Val{D}) where {D}
        dims = npoints(Ωₕ, Tuple)
        w = Vector{Float64}(undef, prod(dims))
        li = LinearIndices(dims)
        for I in CartesianIndices(dims)
            p = 1.0
            for d in 1:D
                p *= if d in S
                    I[d] == 1 ? 0.0 : spacing(Ωₕ(d), I[d])
                else
                    cell_measure(Ωₕ(d), I[d])
                end
            end
            w[li[I]] = p
        end
        return w
    end

    all_subsets(D) = Tuple(
        Tuple(d for d in 1:D if ((m >> (d - 1)) & 1) == 1) for m in 0:(2 ^ D - 1)
    )

    @testset "3D: all 8 sets" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0) × interval(0.0, 3.0)),
            (6, 5, 4), (true, false, true))
        Wₕ = gridspace(Ωₕ)
        u = Rₕ(Wₕ, x -> x[1] + 2x[2] - x[3])
        v = Rₕ(Wₕ, x -> 1.0 + x[1] * x[3])

        for S in all_subsets(3)
            w = hand_weights(Ωₕ, S, Val(3))
            expected = sum(parent(u) .* w .* parent(v))
            @test inner₊(u, v, Val(S)) ≈ expected
        end
    end

    @testset "2D: all 4 sets" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.5)), (7, 6), (false, true))
        Wₕ = gridspace(Ωₕ)
        u = Rₕ(Wₕ, x -> sin(x[1]) + x[2])
        v = Rₕ(Wₕ, x -> cos(x[2]))

        for S in all_subsets(2)
            w = hand_weights(Ωₕ, S, Val(2))
            expected = sum(parent(u) .* w .* parent(v))
            @test inner₊(u, v, Val(S)) ≈ expected
        end
    end

    @testset "Alias identities" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0)),
            (5, 4, 6), (true, true, false))
        Wₕ = gridspace(Ωₕ)
        u = Rₕ(Wₕ, x -> x[1] * x[2] + x[3])
        v = Rₕ(Wₕ, x -> 1.0)

        @test inner₊(u, v, Val(())) === innerₕ(u, v)
        @test inner₊(u, v, Val((1,))) === inner₊ₓ(u, v)
        @test inner₊(u, v, Val((2,))) === inner₊ᵧ(u, v)
        @test inner₊(u, v, Val((3,))) === inner₊₂(u, v)
    end

    @testset "markers masking" begin
        S = interval(0.0, 1.0) × interval(0.0, 1.0)
        Ωₕ = mesh(domain(S, :bottom => :bottom, :left => :left), (6, 6), (true, true))
        Wₕ = gridspace(Ωₕ)
        u = Rₕ(Wₕ, x -> 1.0)
        v = Rₕ(Wₕ, x -> 1.0)

        w12 = weights(Wₕ, Val((1, 2)))
        mask = Bramble.index_in_marker(Ωₕ, :bottom)
        byhand = sum(w12[i] for i in eachindex(w12) if mask[i])
        @test inner₊(u, v, Val((1, 2)); markers = (:bottom,)) ≈ byhand

        mask2 = Bramble.index_in_marker(Ωₕ, :bottom) .| Bramble.index_in_marker(Ωₕ, :left)
        byhand2 = sum(w12[i] for i in eachindex(w12) if mask2[i])
        @test inner₊(u, v, Val((1, 2)); markers = (:bottom, :left)) ≈ byhand2

        @test inner₊(u, v, Val((1, 2)); markers = ()) == inner₊(u, v, Val((1, 2)))
    end

    @testset "0-byte allocation, empty set and singletons" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0)),
            (6, 5, 4), (true, true, true))
        Wₕ = gridspace(Ωₕ)
        u = Rₕ(Wₕ, x -> x[1])
        v = Rₕ(Wₕ, x -> 1.0)

        @test_allocs inner₊(u, v, Val(()))
        @test_allocs inner₊(u, v, Val((1,)))
        @test_allocs inner₊(u, v, Val((2,)))
        @test_allocs inner₊(u, v, Val((3,)))
    end

    @testset "CpuBatch reaches the Polyester hook, never the Cartesian loop (#190)" begin
        # `inner₊(uₕ, vₕ, Val(S))` passes `execution_policy(space(uₕ))` through to the
        # policy-dispatched `_dot`/`_dot_masked` (S7.1, `src/utils/linear_algebra.jl`):
        # `CpuSerial`/`CpuThreaded` fall through to the plain methods (positive control
        # below); `CpuBatch` must reach S7.1's `_batch_dot`/`_batch_dot_masked` hook and
        # its "Polyester not loaded" error, for a dense weight and for a `SeparableWeights`
        # alike, without ever running this file's Cartesian loop. A `CpuBatch` grid space
        # cannot be built at all without Polyester (`space_weights` itself needs the
        # policy-dispatched sweep), so this calls `_dot`/`_dot_masked` directly rather than
        # constructing one.
        u = [1.0, 2.0, 3.0, 4.0]
        v = [2.0, 2.0, 2.0, 2.0]
        w_dense = [1.0, 0.5, 0.25, 2.0]
        w_sep = Bramble.SeparableWeights{2, Float64, Vector{Float64}}(
            ([1.0, 2.0], [0.5, 4.0]), (2, 2)
        )
        mask = BitVector([true, false, true, false])

        for w in (w_dense, w_sep)
            expected = Bramble._dot(u, w, v)
            @test Bramble._dot(Bramble.CpuSerial(), u, w, v) == expected
            @test Bramble._dot(Bramble.CpuThreaded(), u, w, v) == expected
            err = try
                Bramble._dot(Bramble.CpuBatch(), u, w, v)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("Polyester", err.msg)

            expected_m = Bramble._dot_masked(u, w, v, mask)
            @test Bramble._dot_masked(Bramble.CpuSerial(), u, w, v, mask) == expected_m
            @test Bramble._dot_masked(Bramble.CpuThreaded(), u, w, v, mask) == expected_m
            err_m = try
                Bramble._dot_masked(Bramble.CpuBatch(), u, w, v, mask)
                nothing
            catch e
                e
            end
            @test err_m isa ArgumentError
            @test occursin("Polyester", err_m.msg)
        end
    end
end

@testset "inner₊ dimension" begin
    import Bramble: get_dimension_from_type, _get_h_val

    W1 = gridspace(mesh(domain(interval(0.0, 1.0)), 5, true))
    W2 = gridspace(
        mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 5), (true, true))
    )
    u1 = Rₕ(W1, x -> x)
    u2 = Rₕ(W2, x -> x[1])
    v2 = Rₕ(W2, x -> x[2])

    @testset "get_dimension_from_type" begin
        @test get_dimension_from_type(typeof((u2, u2))) == 2
        @test get_dimension_from_type(typeof(u2)) == 2
        # Anything else carries no dimension.
        @test get_dimension_from_type(Vector{Float64}) === nothing
        @test get_dimension_from_type(Float64) === nothing
    end

    @testset "Tuple arity precedence" begin
        # A tuple on the left is already covered elsewhere; this is the branch
        # where only the right argument is a tuple.
        @test inner₊(u2, (v2, v2)) ≈ inner₊((u2, u2), v2)
        @test inner₊(u2, (v2, v2)) ≈ sum(inner₊(u2, v2, Tuple))
    end

    @testset "Single-sided dimension" begin
        # The dimension is taken from whichever argument has one; the call then
        # fails on the element type rather than on dimension resolution.
        @test_throws MethodError inner₊(u2, [1.0, 2.0])
        @test_throws MethodError inner₊([1.0, 2.0], u2)
    end

    @testset "Dimension mismatches" begin
        # Both of these used to raise UndefVarError: the message was interpolated
        # inside the quoted expression, so it was evaluated at run time where the
        # generator's locals no longer exist.
        @test_throws ArgumentError inner₊(1.0, 2.0)
        @test_throws DimensionMismatch inner₊(u1, u2)

        err = try
            inner₊(u1, u2)
        catch e
            e
        end
        @test occursin("1", err.msg) && occursin("2", err.msg)
    end

    @testset "_get_h_val" begin
        h = [0.5, 0.25, 0.125]
        @test _get_h_val(h, 1) == 0.5
        @test _get_h_val(h, 3) == 0.125
        # The callable form is what the engines actually pass.
        @test _get_h_val(Base.Fix1(getindex, h), 2) == 0.25
    end
end

@testset "H¹ seminorm gradient" begin
    # snorm₁ₕ(uₕ) == norm₊(∇ₕ(uₕ)) is the definition of the discrete H¹ seminorm, and
    # snorm₁ₕ computes it without materialising the gradient. The two routes must agree
    # in every dimension, on uniform and non-uniform grids.
    meshes = (
        ("1D uniform", mesh(domain(interval(0.0, 1.0)), 21, true)),
        ("1D non-uniform", mesh(domain(interval(0.0, 1.0)), 21, false)),
        (
            "2D",
            mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 9), (true, false))
        ),
        (
            "3D",
            mesh(
                domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))),
                (4, 5, 6),
                (true, false, true)
            )
        )
    )

    for (lbl, Ωₕ) in meshes
        @testset "$lbl" begin
            Wₕ = gridspace(Ωₕ)
            f = Ωₕ isa Bramble.Mesh1D ? (x -> sin(3x) + x) : (x -> sin(3x[1]) + x[end]^2)
            uₕ = Rₕ(Wₕ, f)

            @test snorm₁ₕ(uₕ) ≈ norm₊(∇ₕ(uₕ))
            # the H¹ norm is built from the two of them
            @test norm₁ₕ(uₕ)^2 ≈ normₕ(uₕ)^2 + snorm₁ₕ(uₕ)^2
            # a constant has zero gradient, so zero seminorm
            @test snorm₁ₕ(Rₕ(Wₕ, x -> 1.0)) ≈ 0.0 atol = 1e-14
        end
    end

    WITH_SLOW_TESTS && @testset "Random grids (Supposition)" begin
        positive_h = Data.Floats{Float64}(;
            minimum = 0.01, maximum = 10.0, nans = false, infs = false
        )
        field_val = Data.Floats{Float64}(;
            minimum = -100.0, maximum = 100.0, nans = false, infs = false
        )

        @check function check_sobolev_identities_2d(
                hx = Data.Vectors(positive_h; min_size = 3, max_size = 8),
                hy = Data.Vectors(positive_h; min_size = 3, max_size = 8),
                u_raw = Data.Vectors(field_val; min_size = 81, max_size = 81)
        )
            nx = length(hx) + 1
            ny = length(hy) + 1
            pts_x = zeros(Float64, nx)
            for i in 1:length(hx)
                pts_x[i + 1] = pts_x[i] + hx[i]
            end
            pts_x ./= pts_x[end]

            pts_y = zeros(Float64, ny)
            for j in 1:length(hy)
                pts_y[j + 1] = pts_y[j] + hy[j]
            end
            pts_y ./= pts_y[end]

            Ωₕ = mesh(
                domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (nx, ny), (false, false)
            )
            set_points!(Ωₕ(1), pts_x)
            set_points!(Ωₕ(2), pts_y)
            Wₕ = gridspace(Ωₕ)

            total = nx * ny
            u_mat = reshape(copy(u_raw[1:total]), nx, ny)
            uₕ = element(Wₕ, vec(u_mat))

            # normₕ² == innerₕ(u, u)
            n_sq = normₕ(uₕ)^2
            inn = innerₕ(uₕ, uₕ)
            ok1 = isapprox(n_sq, inn; atol = 1e-10 * max(n_sq, 1.0), rtol = 1e-10)

            # snorm₁ₕ(uₕ) == norm₊(∇ₕ(uₕ))
            sn = snorm₁ₕ(uₕ)
            grad_norm = norm₊(∇ₕ(uₕ))
            ok2 = isapprox(sn, grad_norm; atol = 1e-10 * max(sn, 1.0), rtol = 1e-10)

            # norm₁ₕ(uₕ)² == normₕ(uₕ)² + snorm₁ₕ(uₕ)²
            h1_sq = norm₁ₕ(uₕ)^2
            sum_sq = n_sq + sn^2
            ok3 = isapprox(h1_sq, sum_sq; atol = 1e-10 * max(h1_sq, 1.0), rtol = 1e-10)

            ok1 && ok2 && ok3
        end
    end
end

@testset "One-element tuple inner₊" begin
    # In 1D the one-element tuple and the bare grid function denote the same thing, and
    # inner₊ accepts both. It used to accept only the second: the generated body read the
    # element type off `u_type.parameters[2]`, which does not exist for `Tuple{V}`, so a
    # 1-tuple raised a BoundsError from inside code generation rather than returning a
    # number.
    Ωₕ = mesh(domain(interval(0.0, 1.0)), 9, true)
    Wₕ = gridspace(Ωₕ)
    uₕ = Rₕ(Wₕ, sin)
    vₕ = Rₕ(Wₕ, cos)

    @test inner₊((uₕ,), (vₕ,)) ≈ inner₊(uₕ, vₕ)
    @test inner₊((uₕ,), (uₕ,)) ≈ inner₊(uₕ, uₕ)
    @test norm₊((uₕ,)) ≈ norm₊(uₕ)
    @test @inferred(inner₊((uₕ,), (vₕ,))) isa Float64

    # the tuple arity still wins over the mesh dimension for genuine mixed terms, which
    # is what the element-type lookup is there to keep separate
    @test inner₊((uₕ, vₕ), (uₕ, vₕ)) ≈ inner₊ₓ(uₕ, uₕ) + inner₊ₓ(vₕ, vₕ)

    @testset "Composite space inner product" begin
        # The inner product of a product space is the sum of the components', the only
        # natural definition across product components.
        Ωc = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (12, 12), (true, true))
        Vc = gridspace(Ωc, Val(3))
        uv = Rₕ(Vc, (x -> x[1], x -> x[2], x -> 1.0))
        vv = Rₕ(Vc, (x -> 1.0, x -> 1.0, x -> 1.0))

        byhand = sum(
            innerₕ(Bramble.components(uv)[c], Bramble.components(vv)[c]) for c in 1:3
        )
        @test innerₕ(uv, vv) ≈ byhand
        @test innerₕ(uv, vv) ≈ 0.5 + 0.5 + 1.0        # ∫x + ∫y + ∫1 over the unit square

        # the norm follows from it
        @test normₕ(uv) ≈ sqrt(innerₕ(uv, uv))
        @test normₕ(uv) ≈ sqrt(sum(normₕ(Bramble.components(uv)[c])^2 for c in 1:3))

        # a mismatch in the number of components is an error rather than a silent answer
        V2 = gridspace(Ωc, Val(2))
        @test_throws DimensionMismatch innerₕ(uv, Rₕ(V2, (x -> 1.0, x -> 1.0)))

        # and the scalar case is untouched
        Wc = gridspace(Ωc)
        u1 = Rₕ(Wc, x -> x[1])
        @test innerₕ(u1, Rₕ(Wc, x -> 1.0)) ≈ 0.5
    end
end

# The maximum norm carries no quadrature weight, so every check below is against a value
# computed independently of the grid: an analytic extremum attained at a grid point, or a
# hand-placed spike whose magnitude is known.
@testset "Discrete maximum norm (#186)" begin
    @testset "Scalar spaces, 1D/2D/3D" begin
        # x ↦ x on a uniform mesh of [0,1] attains its maximum at the last grid point, so
        # the answer is exactly 1.0 and does not depend on the number of points.
        Ω1 = mesh(domain(interval(0.0, 1.0)), 11, true)
        W1 = gridspace(Ω1)
        @test norminf_h(Rₕ(W1, x -> x[1])) ≈ 1.0
        @test norm∞ₕ(Rₕ(W1, x -> x[1])) ≈ 1.0

        # the absolute value is taken before the maximum: a field that is everywhere
        # negative has a positive norm
        @test norminf_h(Rₕ(W1, x -> -2.0 - x[1])) ≈ 3.0

        # the zero element is the only one with zero norm
        @test norminf_h(Rₕ(W1, x -> 0.0)) == 0.0

        Ω2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (7, 9), (true, true))
        W2 = gridspace(Ω2)
        @test norminf_h(Rₕ(W2, x -> x[1] + x[2])) ≈ 3.0        # attained at (1, 2)
        @test norminf_h(Rₕ(W2, x -> -x[2])) ≈ 2.0

        Ω3 = mesh(
            domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 3.0)),
            (5, 5, 6), (true, true, true))
        W3 = gridspace(Ω3)
        @test norminf_h(Rₕ(W3, x -> x[3])) ≈ 3.0

        # a single spike dominates everything else, wherever it sits
        uₕ = Rₕ(W3, x -> 0.1)
        parent(uₕ)[7] = -42.0
        @test norminf_h(uₕ) ≈ 42.0
    end

    @testset "Non-uniform meshes do not change it" begin
        # unlike normₕ, no weight enters, so refining or grading the mesh leaves the norm
        # of a restricted function fixed once the extremum sits on a grid point
        Ωa = mesh(domain(interval(0.0, 1.0)), 9, true)
        Ωb = mesh(domain(interval(0.0, 1.0)), 33, false)
        @test norminf_h(Rₕ(gridspace(Ωa), x -> x[1])) ≈
              norminf_h(Rₕ(gridspace(Ωb), x -> x[1]))
    end

    @testset "Composite spaces take the maximum across components" begin
        # distinct values per component, so a wrong component cannot pass
        Ωc = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (12, 12), (true, true))
        Vc = gridspace(Ωc, Val(3))
        uv = Rₕ(Vc, (x -> 2.0 * x[1], x -> -5.0 * x[2], x -> 0.25))

        @test norminf_h(uv) ≈ 5.0
        @test norminf_h(uv) ≈ maximum(norminf_h, Bramble.components(uv))
        # per component, so the 5.0 above is demonstrably the second one's
        comps = Bramble.components(uv)
        @test norminf_h(comps[1]) ≈ 2.0
        @test norminf_h(comps[2]) ≈ 5.0
        @test norminf_h(comps[3]) ≈ 0.25
    end

    @testset "Tuples of grid functions" begin
        # what the vectorial aliases return: ∇ₕ(uₕ) is an NTuple{D, VectorElement} in 2D
        Ω2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 9), (true, true))
        W2 = gridspace(Ω2)
        aₕ = Rₕ(W2, x -> 1.5)
        bₕ = Rₕ(W2, x -> -4.0)
        @test norminf_h((aₕ, bₕ)) ≈ 4.0
        @test norminf_h((aₕ,)) ≈ norminf_h(aₕ)
    end

    @testset "The element type comes from the data" begin
        # bramble-verification §4. The Float32 half of this lives in
        # test/space/element_type.jl, where the backend actually carries that element type.
        Ω1 = mesh(domain(interval(0.0, 1.0)), 7, true)
        uₕ = Rₕ(gridspace(Ω1), x -> x[1])
        @test norminf_h(uₕ) isa Float64
        @test @inferred(norminf_h(uₕ)) isa Float64
    end
end

# A masked sum of existing cell measures, restricted by `markers`, is a volumetric
# masked sum rather than a codimension-1 surface integral. Every test here maintains
# that distinction explicitly.
@testset "Masked inner products" begin
    S = interval(0.0, 1.0) × interval(0.0, 1.0)
    Ωₕ = mesh(domain(S, :bottom => :bottom, :left => :left), (5, 5), (true, true))
    Wₕ = gridspace(Ωₕ)
    uₕ = Rₕ(Wₕ, x -> 1.0)
    vₕ = Rₕ(Wₕ, x -> 1.0)

    @testset "Figure match" begin
        # Masked sum on a 5×5 mesh restricted to :bottom evaluates to 0.125, distinguished
        # from a codimension-1 boundary integral over the same region.
        @test innerₕ(uₕ, vₕ; markers = (:bottom,)) ≈ 0.125
    end

    @testset "Empty default agreement" begin
        @test innerₕ(uₕ, vₕ; markers = ()) == innerₕ(uₕ, vₕ)
        @test inner₊ₓ(uₕ, vₕ; markers = ()) == inner₊ₓ(uₕ, vₕ)
    end

    @testset "Marker union" begin
        mask = Bramble.index_in_marker(Ωₕ, :bottom) .| Bramble.index_in_marker(Ωₕ, :left)
        w = Bramble.weights(Wₕ, Bramble.Innerh())
        byhand = sum(w[i] for i in eachindex(w) if mask[i])
        @test innerₕ(uₕ, vₕ; markers = (:bottom, :left)) ≈ byhand
        # a mesh point on both :bottom and :left (the corner) must count once
        @test innerₕ(uₕ, vₕ; markers = (:bottom, :left)) <
              innerₕ(uₕ, vₕ; markers = (:bottom,)) + innerₕ(uₕ, vₕ; markers = (:left,))
    end

    @testset "Composite threading" begin
        Vₕ = Wₕ^Val(2)
        Uc = Rₕ(Vₕ, x -> (1.0, 2.0))
        Vc = Rₕ(Vₕ, x -> (1.0, 2.0))
        c1, c2 = Bramble.components(Uc)
        byhand = innerₕ(c1, c1; markers = (:bottom,)) + innerₕ(c2, c2; markers = (:bottom,))
        @test innerₕ(Uc, Vc; markers = (:bottom,)) ≈ byhand
    end

    @testset "Directional masking" begin
        wx = Bramble.weights(Wₕ, Bramble.Innerplus(), 1)
        mask = Bramble.index_in_marker(Ωₕ, :left)
        byhand = sum(wx[i] for i in eachindex(wx) if mask[i])
        @test inner₊ₓ(uₕ, vₕ; markers = (:left,)) ≈ byhand
    end

    @testset "Refinement scaling" begin
        # The whole point of point 11's decision: this quantity is O(h) times the boundary
        # integral it is easily mistaken for, and a decreasing sequence under refinement is
        # what tells the two apart, not the single 0.125 figure alone.
        vals = map((5, 10, 20, 40)) do n
            Ω = mesh(domain(S, :bottom => :bottom), (n, n), (true, true))
            W = gridspace(Ω)
            w = Rₕ(W, x -> 1.0)
            innerₕ(w, w; markers = (:bottom,))
        end
        @test issorted(vals; rev = true)
        @test vals[end] < vals[1] / 4
    end

    # The genuine surface integral, against which the masked sums above are the *other*
    # quantity: mesh-independent where those scale like h (gpena/Bramble.jl#157). Every
    # figure here is a closed form -- an edge length, a perimeter, a surface area, a
    # hand-computed corner weight -- never a second call to the code under test.
    @testset "inner_Γ (#157)" begin
        @testset "2D: edge lengths and the perimeter, on every mesh" begin
            Ω2 = domain(interval(0.0, 2.0) × interval(0.0, 3.0))
            for n in ((5, 5), (9, 9), (17, 16))
                W = gridspace(mesh(Ω2, n, (true, true)))
                one_h = Rₕ(W, x -> 1.0)
                @test inner_Γ(one_h, one_h, :ymin) ≈ 2.0          # the bottom edge
                @test inner_Γ(one_h, one_h, :xmax) ≈ 3.0          # the right edge
                @test inner_Γ(one_h, one_h, :boundary) ≈ 10.0     # 2(2 + 3)
            end
        end

        @testset "Marker unions add, corners included" begin
            # ω(:ymin) + ω(:xmin) == ω(:ymin, :xmin) pointwise: at the shared corner the two
            # contributions are h₁/2 and k₁/2 and the union's is their sum, so no point is
            # counted twice and none is missed.
            W = gridspace(mesh(domain(interval(0.0, 2.0) × interval(0.0, 3.0)), (7, 6),
                (true, true)))
            one_h = Rₕ(W, x -> 1.0)
            @test inner_Γ(one_h, one_h, :ymin) + inner_Γ(one_h, one_h, :xmin) ≈
                  inner_Γ(one_h, one_h, :ymin, :xmin)
            @test inner_Γ(one_h, one_h, :left) ≈ inner_Γ(one_h, one_h, :xmin)
        end

        @testset "The pointwise weight matches the closed 2D form" begin
            # a non-uniform mesh, so a wrong weight cannot hide behind a uniform spacing
            Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (6, 5), (false, false))
            W = gridspace(Ωₕ)
            hx = Bramble.half_spacings(Ωₕ(1))
            hy = Bramble.half_spacings(Ωₕ(2))
            n1, n2 = npoints(Ωₕ, Tuple)

            # a bottom-edge interior point weighs the transverse half-spacing in x
            probe(i, j) = (e = element(W, 0.0);
                parent(e)[LinearIndices(indices(Ωₕ))[i, j]] = 1.0; e)
            for i in 2:(n1 - 1)
                @test inner_Γ(probe(i, 1), probe(i, 1), :ymin) ≈ hx[i]
            end
            # and a corner on two requested faces weighs the sum of the two halves
            @test inner_Γ(probe(1, 1), probe(1, 1), :xmin, :ymin) ≈ hx[1] + hy[1]
            # off the surface it is zero
            @test inner_Γ(probe(3, 3), probe(3, 3), :boundary) == 0.0
        end

        @testset "1D is counting measure" begin
            # A (D-1)-face is a point, of measure 1: the empty product, not a limit of the 2D
            # formula. Any spacing factor here would break a 1D Neumann problem's order.
            for n in (5, 9, 33), uniform in (true, false)

                W = gridspace(mesh(domain(interval(0.0, 1.0)), n, uniform))
                one_h = Rₕ(W, x -> 1.0)
                @test inner_Γ(one_h, one_h, :boundary) ≈ 2.0
                @test inner_Γ(one_h, one_h, :xmin) ≈ 1.0
            end
        end

        @testset "3D: the surface area of a box" begin
            Ωₕ = mesh(
                domain(interval(0.0, 1.0) × interval(0.0, 2.0) × interval(0.0, 3.0)),
                (5, 6, 7), (true, true, true))
            W = gridspace(Ωₕ)
            one_h = Rₕ(W, x -> 1.0)
            @test inner_Γ(one_h, one_h, :boundary) ≈ 2 * (1 * 2 + 1 * 3 + 2 * 3)
            @test inner_Γ(one_h, one_h, :zmax) ≈ 1 * 2
        end

        @testset "It is bilinear in its two arguments" begin
            W = gridspace(mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 8),
                (true, true)))
            a = Rₕ(W, x -> sin(x[1]) + x[2])
            b = Rₕ(W, x -> cos(x[2]) * x[1])
            @test inner_Γ(a, b, :ymin) ≈ inner_Γ(b, a, :ymin)
            @test inner_Γ(2.0 * a, b, :ymin) ≈ 2.0 * inner_Γ(a, b, :ymin)
        end

        @testset "Refusals" begin
            W = gridspace(mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 7),
                (true, true)))
            one_h = Rₕ(W, x -> 1.0)
            # a marker that does not name a whole coordinate face
            @test_throws ArgumentError inner_Γ(one_h, one_h, :inlet)
            @test_throws ArgumentError inner_Γ(one_h, one_h, :interior)
            # and no labels at all
            @test_throws ArgumentError inner_Γ(one_h, one_h)

            # a face set that is not (D-1)-dimensional on this mesh: with two points on an
            # axis, both of its faces together cover every grid point
            Wc = gridspace(mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (2, 2),
                (true, true)))
            oc = Rₕ(Wc, x -> 1.0)
            @test_throws ArgumentError inner_Γ(oc, oc, :boundary)
            # one face of that axis alone is still a surface
            @test inner_Γ(oc, oc, :ymin) ≈ 1.0
        end
    end
end

end # module SpaceInnerProductTests
