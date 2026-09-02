using Test
using Bramble
using Bramble: __prod, _innerplus_weights!, spacing, _innerh_weights!,
               _innerplus_mean_weights!, __innerplus_weights!, half_spacing, space_weights,
               SpaceWeights
using Bramble: Innerh, Innerplus
using LinearAlgebra: norm
using Supposition

@testset "Scalar and Vector GridSpaces" begin
    mesh1d = mesh(domain(interval(0, 1)), 10, true)
    mesh2d = mesh(domain(box((0, 0), (0.5, 0.6))), (5, 6), (true, true))
    mesh3d = mesh(domain(box((0, 0, 0), (0.5, 0.6, 0.7))), (4, 4, 4), (true, true, true))

    @testset "Weight Helper Functions" begin
        @testset "__prod" begin
            # Test D=1
            v1 = ([1.0, 2.0, 3.0],)
            idx1 = CartesianIndex(2)
            @test __prod(v1, idx1) ≈ 2.0

            # Test D=2
            v2 = ([1.0, 2.0], [3.0, 4.0, 5.0])
            idx2 = CartesianIndex(2, 3)
            @test __prod(v2, idx2) ≈ 2.0 * 5.0 ≈ 10.0
        end

        @testset "_innerh_weights!" begin
            # 1D
            u1 = vector(backend(mesh1d), npoints(mesh1d))
            _innerh_weights!(u1, mesh1d)
            @test length(u1) == npoints(mesh1d)
            @test all(u1 .> 0)

            # 2D
            u2 = vector(backend(mesh2d), npoints(mesh2d))
            _innerh_weights!(u2, mesh2d)
            expected_norm = 0.05952940449895328
            @test norm(u2) ≈ expected_norm
        end

        @testset "_innerplus_weights!" begin
            u = vector(backend(mesh1d), npoints(mesh1d))
            _innerplus_weights!(u, mesh1d, 1)
            @test u[1] == 0.0
            for i in 2:npoints(mesh1d)
                @test u[i] ≈ spacing(mesh1d, i)
            end
        end

        @testset "_innerplus_mean_weights!" begin
            u = vector(backend(mesh1d), npoints(mesh1d))
            N = npoints(mesh1d)
            _innerplus_mean_weights!(u, mesh1d, 1)
            @test u[1] == 0.0
            @test u[N] == 0.0
            for i in 2:(N - 1)
                @test u[i] ≈ half_spacing(mesh1d, i)
            end
        end

        @testset "__innerplus_weights!" begin
            npts_tup = npoints(mesh2d, Tuple)
            v = zeros(Float64, npts_tup)
            comp_weights = (rand(npts_tup[1]), rand(npts_tup[2]))
            __innerplus_weights!(Serial(), v, comp_weights)

            idx = CartesianIndex(3, 4)
            @test v[idx] ≈ comp_weights[1][idx[1]] * comp_weights[2][idx[2]]
        end
    end

    @testset "Space Weights Computation" begin
        D2 = dim(mesh2d)
        b2 = backend(mesh2d)
        VT2 = vector_type(b2)

        sw2 = space_weights(mesh2d)
        @test sw2 isa SpaceWeights{D2, VT2}
        @test length(sw2.innerh) == npoints(mesh2d)
        @test length(sw2.innerplus) == D2
        @test all(length(w) == npoints(mesh2d) for w in sw2.innerplus)
    end

    @testset "ScalarGridSpace Construction and Properties" begin
        W1 = gridspace(mesh1d)
        W2 = gridspace(mesh2d)
        W3 = gridspace(mesh3d)

        @testset "Types and Fields" begin
            @test W1 isa ScalarGridSpace
            @test W2 isa ScalarGridSpace
            @test W3 isa ScalarGridSpace
            @test isimmutable(W1)

            # Verify cached matrices are NOT stored on the struct
            @test propertynames(W1) == (:mesh, :weights)
            @test !hasfield(ScalarGridSpace, :backward_difference_matrix)
            @test !hasfield(ScalarGridSpace, :average_matrix)
            @test !hasfield(ScalarGridSpace, :has_backward_difference_matrix)
            @test !hasfield(ScalarGridSpace, :has_average_matrix)
        end

        @testset "Interface Accessors" begin
            @test mesh(W1) === mesh1d
            @test mesh(W2) === mesh2d
            @test mesh_type(W1) === typeof(mesh1d)
            @test mesh_type(typeof(W1)) === typeof(mesh1d)
            @test dim(W1) == 1
            @test dim(W2) == 2
            @test dim(W3) == 3
            @test dim(typeof(W1)) == 1
            @test dim(typeof(W2)) == 2
            @test dim(typeof(W3)) == 3
            @test eltype(W1) === Float64
            @test eltype(typeof(W1)) === Float64
            @test backend(W1) === backend(mesh1d)
            @test ncomponents(W1) == 1
            @test ncomponents(typeof(W1)) == 1
            @test spaces(W1) === (W1,)

            @test ndofs(W1) == 10
            @test ndofs(W1, Tuple) == (10,)
            @test ndofs(W2) == 30
            @test ndofs(W2, Tuple) == (5, 6)
            @test ndofs(W3) == 64
            @test ndofs(W3, Tuple) == (4, 4, 4)
        end

        @testset "Weights Accessors" begin
            w_h = weights(W2, Innerh())
            @test w_h isa AbstractVector
            @test length(w_h) == 30
            @test weights(W2, Innerh(), 1) === w_h

            w_plus = weights(W2, Innerplus())
            @test w_plus isa Tuple
            @test length(w_plus) == 2
            @test weights(W2, Innerplus(), 1) === w_plus[1]
            @test weights(W2, Innerplus(), 2) === w_plus[2]
        end
    end

    @testset "CompositeGridSpace / VectorGridSpace Construction and Properties" begin
        W = gridspace(mesh2d)

        @testset "Constructors" begin
            # Via Val(N)
            V_val = gridspace(mesh2d, Val(2))
            @test V_val isa CompositeGridSpace{2}
            @test V_val isa VectorGridSpace{2}
            @test isimmutable(V_val)

            # Via integer N
            V_int = gridspace(mesh2d, 2)
            @test V_int isa CompositeGridSpace{2}
            @test length(V_int) == 2

            # Via vector_gridspace
            V_vec = vector_gridspace(mesh2d)
            @test V_vec isa CompositeGridSpace{2}
            @test length(V_vec) == dim(mesh2d)

            V_vec3 = vector_gridspace(mesh2d, 3)
            @test length(V_vec3) == 3

            # Via exponentiation ^
            V_pow = W^2
            @test V_pow isa CompositeGridSpace{2}
            @test V_pow[1] === W
            @test V_pow[2] === W

            V_pow_val = W^Val(3)
            @test V_pow_val isa CompositeGridSpace{3}
            @test V_pow_val[1] === W

            # Via product ×
            V_prod = W × W
            @test V_prod isa CompositeGridSpace{2}

            # Vararg constructor
            V_vararg = CompositeGridSpace(W, W, W)
            @test V_vararg isa CompositeGridSpace{3}
        end

        @testset "Interface Accessors" begin
            V = W^2
            @test mesh(V) === mesh2d
            @test mesh_type(V) === typeof(mesh2d)
            @test dim(V) == 2
            @test eltype(V) === Float64
            @test eltype(typeof(V)) === Float64
            @test backend(V) === backend(mesh2d)
            @test ncomponents(V) == 2
            @test ncomponents(typeof(V)) == 2

            @test ndofs(V) == 2 * ndofs(W)
            @test ndofs(V, Tuple) == (ndofs(W), ndofs(W))
            @test spaces(V) === (W, W)

            # Weights forwarding
            @test weights(V) === weights(W)
            @test weights(V, Innerh()) === weights(W, Innerh())
            @test weights(V, Innerplus(), 1) === weights(W, Innerplus(), 1)
        end

        @testset "Collection Interface" begin
            W_a = gridspace(mesh1d)
            W_b = gridspace(mesh1d)
            V = CompositeGridSpace(W_a, W_b)

            @test length(V) == 2
            @test firstindex(V) == 1
            @test lastindex(V) == 2
            @test V[1] === W_a
            @test V[2] === W_b
            @test eachindex(V) == 1:2
            @test keys(V) == 1:2

            # Iteration
            collected = [s for s in V]
            @test length(collected) == 2
            @test collected[1] === W_a
            @test collected[2] === W_b
        end

        @testset "Hierarchical Spaces (for coupled problems like Stokes)" begin
            # Vh (velocity, 2D) and Qh (pressure, 1D)
            Vh = W × W
            Qh = W
            SystemSpace = Vh × Qh
            @test SystemSpace isa CompositeGridSpace{2}
            @test SystemSpace[1] isa CompositeGridSpace{2}
            @test SystemSpace[2] isa ScalarGridSpace
            @test ndofs(SystemSpace) == 3 * ndofs(W)
        end
    end

    @testset "Int and Val component counts agree" begin
        W = gridspace(mesh2d)

        # The two spellings must produce the same type for every N.
        for n in 1:4
            @test typeof(gridspace(mesh2d, n)) === typeof(gridspace(mesh2d, Val(n)))
            @test typeof(W^n) === typeof(W^Val(n))
        end

        # N == 1 collapses to the scalar space rather than a one-component composite.
        @test gridspace(mesh2d, 1) isa ScalarGridSpace
        @test gridspace(mesh2d, Val(1)) isa ScalarGridSpace
        @test (W^1) === W
        @test (W^Val(1)) === W

        # The generic element interface still works on that scalar result.
        u = element(W^1)
        @test u(1) === u
        @test components(u) === (u,)

        @test_throws ArgumentError gridspace(mesh2d, 0)
        @test_throws ArgumentError W^0

        # A literal component count must stay type stable; a runtime one need not.
        lit2(Ω) = gridspace(Ω, 2)
        lit5(Ω) = gridspace(Ω, 5)
        pow2(Wx) = Wx^2
        @test isconcretetype(Base.return_types(lit2, (typeof(mesh2d),))[1])
        @test isconcretetype(Base.return_types(lit5, (typeof(mesh2d),))[1])
        @test isconcretetype(Base.return_types(pow2, (typeof(W),))[1])
        @test isconcretetype(Base.return_types(gridspace, (typeof(mesh2d), Val{3}))[1])

        # Components share one scalar space, so weights are computed once.
        V = gridspace(mesh2d, Val(3))
        @test all(sp === spaces(V)[1] for sp in spaces(V))
        @test ndofs(V) == 3 * ndofs(W)
    end

    @testset "Interface fallbacks on AbstractSpaceType" begin
        # These are @inline one-liners that get fully inlined, so line coverage
        # never sees them; they still need exercising.
        import Bramble: space, spaces, ncomponents, AbstractSpaceType

        W = gridspace(mesh2d)
        @test space(W) === W
        @test spaces(W) === (W,)
        @test ncomponents(W) == 1
        @test ncomponents(typeof(W)) == 1

        V = W^Val(3)
        @test ncomponents(V) == 3
        @test ncomponents(typeof(V)) == 3
        @test length(spaces(V)) == 3
        @test all(sp === W for sp in spaces(V))

        # a composite is itself an AbstractSpaceType
        @test V isa AbstractSpaceType
        @test space(V) === V
    end

    @testset "Composite constructors and collection interface" begin
        import Bramble: CompositeGridSpace, vector_gridspace, spaces

        W = gridspace(mesh2d)

        # the {N}-only inner constructor
        V = CompositeGridSpace{2}((W, W))
        @test V isa CompositeGridSpace{2}
        @test length(V) == 2
        @test spaces(V) === (W, W)

        # vector_gridspace with an explicit Val
        Vv = vector_gridspace(mesh2d, Val(3))
        @test Vv isa CompositeGridSpace{3}
        @test all(sp === spaces(Vv)[1] for sp in spaces(Vv))

        # firstindex / lastindex / eachindex / keys
        # Note: firstindex is an @inline method whose body is the literal 1, so
        # Julia emits no coverage point for it and it reads as uncovered however
        # it is called. It is exercised here regardless.
        @test firstindex(Vv) == 1
        @test lastindex(Vv) == 3
        @test eachindex(Vv) == 1:3
        @test keys(Vv) == 1:3
        @test Vv[firstindex(Vv)] === Vv[1]
        @test Vv[lastindex(Vv)] === Vv[3]
    end
end

@testset "1D inner-product weights on awkward grids" begin
    # space_weights has a one-dimensional method, because two of the four full-length
    # vectors the general method builds are dead there: the transverse factor is never
    # selected when there is no transverse direction, and the product over a single
    # factor is a copy.
    #
    # These assert the weights against the mesh rather than against the general method,
    # so they hold whichever method produces them. The grids are chosen rather than drawn
    # so the spacing ratio is genuinely extreme; `mesh(Ω, n, false)` gives a random grid,
    # which is not.
    import Bramble: Innerh, Innerplus, set_points!

    grids = Dict(
        "uniform" => n -> collect(range(0.0, 1.0, length = n)),
        "graded t^4" => n -> [(k / (n - 1))^4 for k in 0:(n - 1)],
        "clustered ends" => n -> [0.5 * (1 - cos(pi * k / (n - 1))) for k in 0:(n - 1)],
        "one tiny cell" =>
            n -> (v = collect(range(0.0, 1.0, length = n));
                v[2] = v[1] + 1e-9;
                v))

    for n in (3, 17, 64), lbl in sort(collect(keys(grids)))

        @testset "n=$n $lbl" begin
            Ωₕ = mesh(domain(interval(0.0, 1.0)), n, true)
            set_points!(Ωₕ, grids[lbl](n))
            Wₕ = gridspace(Ωₕ)

            wh = weights(Wₕ, Innerh())
            wp = weights(Wₕ, Innerplus(), 1)

            @test length(wh) == n
            @test length(wp) == n

            # the cell-measure weights are the cell measures
            @test all(wh[i] == cell_measure(Ωₕ, i) for i in 1:n)

            # the staggered weights are the backward spacings, truncated to zero at the
            # first point where the backward stencil has none
            @test wp[1] == 0.0
            @test all(wp[i] == spacing(Ωₕ, i) for i in 2:n)

            # and they are what the inner products actually use
            uₕ = Rₕ(Wₕ, x -> x^2 + 1)
            @test innerₕ(uₕ, uₕ) ≈ sum(wh[i] * Bramble.values(uₕ)[i]^2 for i in 1:n)
            @test inner₊(uₕ, uₕ) ≈ sum(wp[i] * Bramble.values(uₕ)[i]^2 for i in 1:n)
        end
    end

    @testset "domain measure and partition of unity (Supposition)" begin
        positive_float = Data.Floats{Float64}(; minimum = 0.1, maximum = 10.0,
            nans = false, infs = false)
        coord_float = Data.Floats{Float64}(; minimum = -10.0, maximum = 10.0,
            nans = false, infs = false)

        @check function check_domain_measure_2d(
                a1 = coord_float,
                len1 = positive_float,
                a2 = coord_float,
                len2 = positive_float,
                nx = Data.Integers(3, 10),
                ny = Data.Integers(3, 10)
        )
            b1 = a1 + len1
            b2 = a2 + len2
            vol = len1 * len2

            Ωₕ = mesh(
                domain(interval(a1, b1) × interval(a2, b2)), (nx, ny), (false, false))
            Wₕ = gridspace(Ωₕ)

            wh = weights(Wₕ, Innerh())

            # 1. Sum of cell measures equals total domain volume
            sum_wh = sum(wh)
            ok_vol = isapprox(sum_wh, vol; atol = 1e-11 * vol, rtol = 1e-11)

            # 2. Each cell measure is strictly positive
            ok_pos = all(wh .> 0)

            # 3. L² norm of constant function 1 equals sqrt(volume)
            u_one = element(Wₕ, 1.0)
            norm_one = normₕ(u_one)
            ok_norm = isapprox(
                norm_one, sqrt(vol); atol = 1e-11 * sqrt(vol), rtol = 1e-11)

            ok_vol && ok_pos && ok_norm
        end
    end
end
