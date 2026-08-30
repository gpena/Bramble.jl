import Bramble: half_spacings_iterator
using Supposition

@testset "Inner Products and Norms" begin
    for D in 1:3
        dims, Wh, u = setup_test_grid(Val(D))

        v = element(Wh, 1.0)
        z = normₕ(v)

        if D == 1
            @test z ≈ sqrt(sum(half_spacings_iterator(mesh(Wh))))
            @test norm₊(D₋ₓ(u)) ≈ 0.0
        elseif D == 2
            @test z ≈ 5.0
        else
            @test z ≈ 11.180339887498947
        end

        u .= 1.0
        der = ∇₋ₕ(u)

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
            der = ∇₋ₕ(u)

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

    @testset "1D Tests" begin
        dims_1d, Wₕ_1d, u1 = setup_test_grid(Val(1))
        domain_length = 5.0 # Domain is [-1, 4]

        u2 = u1 * 2.0
        u3 = similar(u1)
        Rₕ!(u3, x->x)

        @testset "L² inner product (innerₕ)" begin
            # (1, 2) = ∫ 1*2 dx = 2 * length = 2 * 5 = 10
            @test innerₕ(u1, u2) ≈ 2.0 * domain_length

            # ||1||² = ∫ 1*1 dx = length = 5
            @test innerₕ(u1, u1) ≈ domain_length
            @test normₕ(u1) ≈ sqrt(domain_length)
        end

        @testset "Modified L² inner product (inner₊)" begin
            # In 1D, inner₊ should equal inner₊ₓ
            @test inner₊(u1, u2) ≈ inner₊ₓ(u1, u2)

            # The test grid is nonuniform, but for constant functions, the integral should still yield the exact measure.
            @test inner₊(u1, u2) ≈ 2.0 * domain_length

            res_tuple = inner₊(u1, u3, Tuple)
            @test res_tuple isa NTuple{1, Float64}
            @test res_tuple[1] ≈ inner₊ₓ(u1, u3)

            @test norm₊(u1)^2 ≈ inner₊(u1, u1)
        end

        @testset "H¹ Norms (norm₁ₕ)" begin
            # For u(x) = 2x, u'(x) = 2.
            Rₕ!(u1, x->2x)
            # |u|²_1h = ||∇u||²₊ ≈ ∫ (2)^2 dx = 4 * length = 4 * 5 = 20
            @test snorm₁ₕ(u1)^2 ≈ 4.0 * domain_length

            # Test full H¹ norm identity
            @test norm₁ₕ(u1)^2 ≈ normₕ(u1)^2 + snorm₁ₕ(u1)^2
        end
    end

    @testset "2D Tests" begin
        dims_2d, Wₕ_2d, u1 = setup_test_grid(Val(2))
        domain_area = 25.0 # Domain is [-1, 4] x [-1, 4]

        u2 = u1 * 2.0
        ux = similar(u1)
        uy = similar(u1)
        Rₕ!(ux, x->x[1])
        Rₕ!(uy, x->x[2])

        @testset "L² and Modified L² products" begin
            # (1, 2) = ∫∫ 1*2 dx dy = 2 * area = 50
            @test innerₕ(u1, u2) ≈ 2.0 * domain_area
            @test normₕ(u1) ≈ sqrt(domain_area)

            # Test sum of directional components
            @test inner₊(ux, uy) ≈ inner₊ₓ(ux, uy) + inner₊ᵧ(ux, uy)
        end

        @testset "Tuple and NTuple methods" begin
            res_tuple = inner₊(ux, uy, Tuple)
            @test res_tuple isa NTuple{2, Float64}
            @test res_tuple[1] ≈ inner₊ₓ(ux, uy)
            @test res_tuple[2] ≈ inner₊ᵧ(ux, uy)

            U = (ux, uy)
            V = (u1, u2)
            expected = inner₊ₓ(ux, u1) + inner₊ᵧ(uy, u2)
            @test inner₊(U, V) ≈ expected
        end

        @testset "H¹ Norms (norm₁ₕ)" begin
            # For u(x,y) = x + 2y, ∇u = (1, 2)
            Rₕ!(u1, x -> x[1] + 2*x[2])

            expected_value_snorm = sum(i^2 *
                                       sum(Bramble.weights(Wₕ_2d, Bramble.Innerplus(), i))
            for i in 1:2)
            @test snorm₁ₕ(u1)^2 ≈ expected_value_snorm
            @test norm₁ₕ(u1)^2 ≈ normₕ(u1)^2 + snorm₁ₕ(u1)^2
        end
    end

    @testset "3D Tests" begin
        dims_3d, Wₕ_3d, u1 = setup_test_grid(Val(3))
        domain_volume = 125.0 # Domain is [-1, 4]³

        u2 = u1 * 2.0
        uz = similar(u1)
        Rₕ!(uz, x -> x[3])

        @testset "L² and Modified L² products" begin
            @test innerₕ(u1, u2) ≈ 2.0 * domain_volume
            @test normₕ(u1) ≈ sqrt(domain_volume)
            @test inner₊(u1, uz) ≈ inner₊ₓ(u1, uz) + inner₊ᵧ(u1, uz) + inner₊₂(u1, uz)
        end

        @testset "H¹ Norms (norm₁ₕ)" begin
            # For u(x,y,z) = x+2y+3z, ∇u = (1, 2, 3)
            Rₕ!(u1, x -> x[1] + 2x[2] + 3x[3])
            expected_value_snorm = sum(i^2 *
                                       sum(Bramble.weights(Wₕ_3d, Bramble.Innerplus(), i))
            for i in 1:3)

            @test snorm₁ₕ(u1)^2 ≈ expected_value_snorm
            @test norm₁ₕ(u1)^2 ≈ normₕ(u1)^2 + snorm₁ₕ(u1)^2
        end
    end
end

@testset "Dimension resolution in inner₊" begin
    import Bramble: get_dimension_from_type, _get_h_val

    W1 = gridspace(mesh(domain(interval(0.0, 1.0)), 5, true))
    W2 = gridspace(mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 5), (
        true, true)))
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

    @testset "tuple arity wins on either side" begin
        # A tuple on the left is already covered elsewhere; this is the branch
        # where only the right argument is a tuple.
        @test inner₊(u2, (v2, v2)) ≈ inner₊((u2, u2), v2)
        @test inner₊(u2, (v2, v2)) ≈ sum(inner₊(u2, v2, Tuple))
    end

    @testset "one side carries no dimension" begin
        # The dimension is taken from whichever argument has one; the call then
        # fails on the element type rather than on dimension resolution.
        @test_throws MethodError inner₊(u2, [1.0, 2.0])
        @test_throws MethodError inner₊([1.0, 2.0], u2)
    end

    @testset "unresolvable and mismatched dimensions report properly" begin
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

    @testset "_get_h_val accepts a raw spacing vector" begin
        h = [0.5, 0.25, 0.125]
        @test _get_h_val(h, 1) == 0.5
        @test _get_h_val(h, 3) == 0.125
        # The callable form is what the engines actually pass.
        @test _get_h_val(Base.Fix1(getindex, h), 2) == 0.25
    end
end

@testset "The H¹ seminorm is the ₊ norm of the gradient" begin
    # snorm₁ₕ(uₕ) == norm₊(∇₋ₕ(uₕ)) is the definition of the discrete H¹ seminorm, and
    # snorm₁ₕ computes it without materialising the gradient. The two routes must agree
    # in every dimension, on uniform and non-uniform grids.
    meshes = (("1D uniform", mesh(domain(interval(0.0, 1.0)), 21, true)),
        ("1D non-uniform", mesh(domain(interval(0.0, 1.0)), 21, false)),
        ("2D",
            mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 9), (true, false))),
        ("3D",
            mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (4, 5, 6),
                (true, false, true))))

    for (lbl, Ωₕ) in meshes
        @testset "$lbl" begin
            Wₕ = gridspace(Ωₕ)
            f = Ωₕ isa Bramble.Mesh1D ? (x -> sin(3x) + x) : (x -> sin(3x[1]) + x[end]^2)
            uₕ = Rₕ(Wₕ, f)

            @test snorm₁ₕ(uₕ) ≈ norm₊(∇₋ₕ(uₕ))
            # the H¹ norm is built from the two of them
            @test norm₁ₕ(uₕ)^2 ≈ normₕ(uₕ)^2 + snorm₁ₕ(uₕ)^2
            # a constant has zero gradient, so zero seminorm
            @test snorm₁ₕ(Rₕ(Wₕ, x -> 1.0)) ≈ 0.0 atol = 1e-14
        end
    end

    @testset "arbitrary random grids and fields (Supposition)" begin
        positive_h = Data.Floats{Float64}(; minimum = 0.01, maximum = 10.0,
            nans = false, infs = false)
        field_val = Data.Floats{Float64}(; minimum = -100.0, maximum = 100.0,
            nans = false, infs = false)

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

            Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (nx, ny),
                (false, false))
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

            # snorm₁ₕ(uₕ) == norm₊(∇₋ₕ(uₕ))
            sn = snorm₁ₕ(uₕ)
            grad_norm = norm₊(∇₋ₕ(uₕ))
            ok2 = isapprox(sn, grad_norm; atol = 1e-10 * max(sn, 1.0), rtol = 1e-10)

            # norm₁ₕ(uₕ)² == normₕ(uₕ)² + snorm₁ₕ(uₕ)²
            h1_sq = norm₁ₕ(uₕ)^2
            sum_sq = n_sq + sn^2
            ok3 = isapprox(h1_sq, sum_sq; atol = 1e-10 * max(h1_sq, 1.0), rtol = 1e-10)

            ok1 && ok2 && ok3
        end
    end
end

@testset "inner₊ accepts a one-element tuple" begin
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
end
