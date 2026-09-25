module SpaceGridspacesTests

using Test
using Bramble
using Bramble: D₋ₓ, Mₓ, ScalarGridSpace, cell_measure, jumpₓ, weights
using Bramble:
               __prod,
               _innerplus_weights!,
               spacing,
               _innerh_weights!,
               _innerplus_mean_weights!,
               __innerplus_weights!,
               half_spacing,
               space_weights,
               SpaceWeights
using Bramble: Innerh, Innerplus
using Bramble: VectorGridSpace, mesh_type
using Bramble: vector
using LinearAlgebra: norm
using Random
using Supposition
using ..TestUtils: WITH_SLOW_TESTS
using ..TestUtils: alloc_test, @test_allocs, _nonuniform_points

@testset "Grid spaces" begin
    mesh1d = mesh(domain(interval(0, 1)), 10, true)
    mesh2d = mesh(domain(box((0, 0), (0.5, 0.6))), (5, 6), (true, true))
    mesh3d = mesh(domain(box((0, 0, 0), (0.5, 0.6, 0.7))), (4, 4, 4), (true, true, true))

    @testset "Weight helpers" begin
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
            # The transverse factor: every entry, boundary included, is the mesh's own
            # half_spacing there (gpena/Bramble.jl#236) -- unlike _innerplus_weights!
            # above, the *aligned* factor, whose first entry is correctly zero (no cell
            # behind node 1 along the direction being differenced). Hand-zeroing the two
            # boundary entries here used to delete real quadrature weight instead: see
            # _innerplus_mean_weights!'s own docstring for why that was wrong and how it
            # was checked against the discrete summation-by-parts identities before fixing.
            u = vector(backend(mesh1d), npoints(mesh1d))
            N = npoints(mesh1d)
            _innerplus_mean_weights!(u, mesh1d, 1)
            for i in 1:N
                @test u[i] ≈ half_spacing(mesh1d, i)
            end
            @test u[1] > 0 && u[N] > 0
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

    @testset "Weights computation" begin
        D2 = dim(mesh2d)
        b2 = backend(mesh2d)
        VT2 = vector_type(b2)

        sw2 = space_weights(mesh2d)
        @test sw2 isa SpaceWeights{D2, eltype(VT2), VT2}
        @test length(sw2.innerh) == npoints(mesh2d)
        @test length(sw2.innerplus) == D2
        @test all(length(w) == npoints(mesh2d) for w in sw2.innerplus)
    end

    @testset "ScalarGridSpace" begin
        W1 = gridspace(mesh1d)
        W2 = gridspace(mesh2d)
        W3 = gridspace(mesh3d)

        @testset "Types & fields" begin
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

        @testset "Accessors" begin
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

        @testset "Weight accessors" begin
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

        # weights(Wₕ, Val(S)) for every staggered set S ⊆ 1:D (gpena/Bramble.jl#115, #234).
        # Checked against the mesh's own spacing/half_spacing directly, hand-multiplied per
        # axis -- independent of the per-axis factors `SpaceWeights` stores internally.
        @testset "weights(Wₕ, Val(S))" begin
            Ω3 = domain(box((0.0, 0.0, 0.0), (0.5, 0.6, 0.7)))
            Ωₕ3 = mesh(Ω3, (4, 3, 5), (false, false, false))
            W3v = gridspace(Ωₕ3)
            n = npoints(Ωₕ3, Tuple)

            aligned(d, i) = i == 1 ? 0.0 : spacing(Ωₕ3(d), i)
            cellfac(d, i) = half_spacing(Ωₕ3(d), i)

            subsets = ((), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3))
            for S in subsets
                w = weights(W3v, Val(S))
                @test length(w) == prod(n)
                for I in CartesianIndices(n)
                    expected = prod(d -> (d in S ? aligned(d, I[d]) : cellfac(d, I[d])), 1:3)
                    @test w[LinearIndices(n)[I]] ≈ expected
                    # Every S, including the two existing families (S = () and
                    # singletons), returns a `SeparableWeights` (gpena/Bramble.jl#115,
                    # S6.8), which answers a `CartesianIndex` directly (the access pattern
                    # an assembly loop already has for free -- see its own docstring) as
                    # well as a linear index.
                    @test w[I] ≈ expected
                end
            end

            # The four existing families are the same objects, not recomputed copies.
            @test weights(W3v, Val(())) === weights(W3v, Innerh())
            for d in 1:3
                @test weights(W3v, Val((d,))) === weights(W3v, Innerplus(), d)
            end
        end
    end

    @testset "CompositeGridSpace" begin
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

        @testset "Accessors" begin
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

            # gpena/Bramble.jl#67: `weights` used to forward to the first leaf, which
            # silently answered with the wrong vector on a composite whose leaves have
            # different meshes. Deleted in favor of the same "reject at dispatch" contract
            # `normₕ`/`norm₊` already use for composites — checked here rather than just
            # asserted, since `V`'s two leaves happen to share one mesh and so cannot tell
            # a correct forward from a wrong one.
            @test_throws MethodError weights(V)
            @test_throws MethodError weights(V, Innerh())
            @test_throws MethodError weights(V, Innerplus(), 1)
        end

        @testset "Collection interface" begin
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

        @testset "Hierarchical spaces" begin
            # Explicit constructor builds nested / hierarchical composite spaces
            Vh = W × W
            Qh = W
            SystemSpace = CompositeGridSpace((Vh, Qh))
            @test SystemSpace isa CompositeGridSpace{2}
            @test SystemSpace[1] isa CompositeGridSpace{2}
            @test SystemSpace[2] isa ScalarGridSpace
            @test ndofs(SystemSpace) == 3 * ndofs(W)
        end

        @testset "Space product operator (×) and associative flattening (#154)" begin
            using LinearAlgebra: LinearAlgebra
            W1 = gridspace(mesh1d)
            W2 = gridspace(mesh1d)
            W3 = gridspace(mesh1d)
            W4 = gridspace(mesh1d)

            _chain2(a, b) = a × b
            _chain3_l(a, b, c) = a × b × c
            _chain3_r(a, b, c) = a × (b × c)
            _chain4(a, b, c, d) = (a × b) × (c × d)

            # Pairwise
            V2 = W1 × W2
            @test V2 isa CompositeGridSpace{2}
            @test V2[1] === W1
            @test V2[2] === W2
            @test @inferred(W1 × W2) isa CompositeGridSpace{2}
            @test_allocs _chain2(W1, W2)

            # Left-chaining: (W1 × W2) × W3 flattens to CompositeGridSpace{3}
            V3 = W1 × W2 × W3
            @test V3 isa CompositeGridSpace{3}
            @test V3[1] === W1
            @test V3[2] === W2
            @test V3[3] === W3
            @test @inferred(W1 × W2 × W3) isa CompositeGridSpace{3}
            @test_allocs _chain3_l(W1, W2, W3)

            # Right-chaining: W1 × (W2 × W3) flattens to CompositeGridSpace{3}
            V3_r = W1 × (W2 × W3)
            @test V3_r isa CompositeGridSpace{3}
            @test V3_r[1] === W1
            @test V3_r[2] === W2
            @test V3_r[3] === W3
            @test @inferred(W1 × (W2 × W3)) isa CompositeGridSpace{3}
            @test_allocs _chain3_r(W1, W2, W3)

            # Composite × Composite: (W1 × W2) × (W3 × W4) flattens to CompositeGridSpace{4}
            V4 = (W1 × W2) × (W3 × W4)
            @test V4 isa CompositeGridSpace{4}
            @test V4[1] === W1
            @test V4[2] === W2
            @test V4[3] === W3
            @test V4[4] === W4
            @test @inferred((W1 × W2) × (W3 × W4)) isa CompositeGridSpace{4}
            @test_allocs _chain4(W1, W2, W3, W4)

            # Heterogeneous spaces (different meshes or dimensions)
            W_2d = gridspace(mesh2d)
            V_het = W1 × W_2d × W1
            @test V_het isa CompositeGridSpace{3}
            @test V_het[1] === W1
            @test V_het[2] === W_2d
            @test V_het[3] === W1
            @test ndofs(V_het) == ndofs(W1) + ndofs(W_2d) + ndofs(W1)
            @test @inferred(W1 × W_2d × W1) isa CompositeGridSpace{3}
            @test_allocs _chain3_l(W1, W_2d, W1)

            # LinearAlgebra coexistence: vector cross product and space product share ×
            u_vec = [1.0, 0.0, 0.0]
            v_vec = [0.0, 1.0, 0.0]
            @test (u_vec × v_vec) == [0.0, 0.0, 1.0]
            @test (W1 × W2) isa CompositeGridSpace{2}
        end
    end

    @testset "Component count agreement" begin
        W = gridspace(mesh2d)

        # The two spellings must produce the same type for every N.
        for n in 1:4
            @test typeof(gridspace(mesh2d, n)) === typeof(gridspace(mesh2d, Val(n)))
        end
        # `^`'s Int spelling is capped at 3 (gpena/Bramble.jl#147); see below.
        for n in 1:3
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

        # gpena/Bramble.jl#147: `^`'s Int spelling only accepts 1 <= N <= 3. Beyond
        # that it throws rather than silently falling back to Val(N), because that
        # fallback is exactly what reintroduces the union-return instability below.
        @test_throws ArgumentError W^4
        @test_throws ArgumentError W^100
        @test (W^Val(4)) isa CompositeGridSpace{4}  # uncapped: the Val spelling still works

        # A literal component count must stay type stable; a runtime one need not,
        # *except* for `^`, whose Int spelling is deliberately capped at N ∈ {1,2,3}
        # so that a dynamic N (one the compiler cannot constant-fold, e.g. threaded
        # through a generic argument) still resolves to a union of concrete leaves.
        # Julia's return-type inference only union-splits up to 3 concrete types; one
        # more branch (even N ∈ {1,2,3,4}) collapses the whole union back down to the
        # abstract `Union{ScalarGridSpace, CompositeGridSpace}` — confirmed directly
        # against `Base.return_types` before picking 3 as the cap, not assumed.
        lit2(Ω) = gridspace(Ω, 2)
        lit5(Ω) = gridspace(Ω, 5)
        pow2(Wx) = Wx^2
        pow_dynamic(Wx, N::Int) = Wx^N
        @test isconcretetype(Base.return_types(lit2, (typeof(mesh2d),))[1])
        @test isconcretetype(Base.return_types(lit5, (typeof(mesh2d),))[1])
        @test isconcretetype(Base.return_types(pow2, (typeof(W),))[1])
        @test isconcretetype(Base.return_types(gridspace, (typeof(mesh2d), Val{3}))[1])

        pow_dynamic_rt = Base.return_types(pow_dynamic, (typeof(W), Int))[1]
        @test pow_dynamic_rt isa Union
        @test all(isconcretetype, Base.uniontypes(pow_dynamic_rt))

        # Components share one scalar space, so weights are computed once.
        V = gridspace(mesh2d, Val(3))
        @test all(sp === spaces(V)[1] for sp in spaces(V))
        @test ndofs(V) == 3 * ndofs(W)
    end

    @testset "Interface fallbacks" begin
        # These are @inline one-liners that get fully inlined, so line coverage
        # never sees them; they still need exercising.
        import Bramble: space, spaces, ncomponents, AbstractSpaceType

        W = gridspace(mesh2d)
        @test space(W) === W

        V = W^Val(3)
        @test ncomponents(V) == 3
        @test ncomponents(typeof(V)) == 3
        @test length(spaces(V)) == 3
        @test all(sp === W for sp in spaces(V))

        # a composite is itself an AbstractSpaceType
        @test V isa AbstractSpaceType
        @test space(V) === V
    end

    @testset "Composite collections" begin
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

        # firstindex / lastindex / eachindex / keys are generic in the component count
        # and asserted by "Collection interface"; indexing through them is not.
        # Note: firstindex is an @inline method whose body is the literal 1, so
        # Julia emits no coverage point for it and it reads as uncovered however
        # it is called. It is exercised here regardless.
        @test Vv[firstindex(Vv)] === Vv[1]
        @test Vv[lastindex(Vv)] === Vv[3]
    end
end

@testset "Awkward grid weights" begin
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
        "one tiny cell" => n -> (v = collect(range(0.0, 1.0, length = n)); v[2] = v[1] + 1e-9; v)
    )

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
            @test innerₕ(uₕ, uₕ) ≈ sum(wh[i] * Bramble.parent(uₕ)[i]^2 for i in 1:n)
            @test inner₊(uₕ, uₕ) ≈ sum(wp[i] * Bramble.parent(uₕ)[i]^2 for i in 1:n)
        end
    end

    WITH_SLOW_TESTS && @testset "Partition of unity" begin
        positive_float = Data.Floats{Float64}(;
            minimum = 0.1, maximum = 10.0, nans = false, infs = false
        )
        coord_float = Data.Floats{Float64}(;
            minimum = -10.0, maximum = 10.0, nans = false, infs = false
        )

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

            Ωₕ = mesh(domain(interval(a1, b1) × interval(a2, b2)), (nx, ny), (false, false))
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
            ok_norm = isapprox(norm_one, sqrt(vol); atol = 1e-11 * sqrt(vol), rtol = 1e-11)

            ok_vol && ok_pos && ok_norm
        end
    end
end

# Invariants tested (gpena/Bramble.jl#17, #45):
# 1. Neither grid space had a `show` or `summary` of its own, so both fell through to the
#    default: `summary(gridspace(...))` was 563 characters of nested type parameters, and
#    displaying one dumped every weight vector with it.
# 2. Two-argument `show` is the embeddable one-liner; `MIME"text/plain"` is the detailed
#    block, and neither may end with a newline.
# 3. A composite whose leaves are all identical collapses to one `N × …` line; a
#    heterogeneous one enumerates its leaves, since that is when per-leaf detail informs.
@testset "Display" begin
    Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (3, 3), (true, true))
    Wₕ = gridspace(Ωₕ)
    Vₕ = Wₕ × Wₕ
    W4 = gridspace(
        mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (4, 4), (true, true))
    )
    Het = Wₕ × W4

    @testset "Scalar space" begin
        compact = sprint(show, Wₕ)
        @test compact == "ScalarGridSpace{2D, Float64, 9 dofs}"
        @test !occursin('\n', compact)
        @test summary(Wₕ) == compact
        # The whole point: the default `summary` was 563 characters.
        @test length(summary(Wₕ)) < 60

        detailed = sprint(show, MIME"text/plain"(), Wₕ)
        @test occursin("ScalarGridSpace", detailed)
        @test occursin("Mesh", detailed)
        @test occursin("MeshnD{2D, 9 pts}", detailed)
        @test occursin("Dofs", detailed)
        @test occursin("9", detailed)
        @test !endswith(detailed, '\n')
        # No weight vector dumped into the display.
        @test !occursin("SpaceWeights", detailed)
    end

    @testset "Composite space, identical leaves collapse" begin
        compact = sprint(show, Vₕ)
        @test compact == "CompositeGridSpace{2 components, 18 dofs}"
        @test !occursin('\n', compact)

        detailed = sprint(show, MIME"text/plain"(), Vₕ)
        @test occursin("CompositeGridSpace", detailed)
        @test occursin("2 components", detailed)
        @test occursin("18 (9 per component)", detailed)
        @test occursin("2 × ScalarGridSpace{2D, Float64, 9 dofs}", detailed)
        @test !endswith(detailed, '\n')
    end

    @testset "Composite space, differing leaves enumerate" begin
        detailed = sprint(show, MIME"text/plain"(), Het)
        @test occursin("1: ScalarGridSpace{2D, Float64, 9 dofs}", detailed)
        @test occursin("2: ScalarGridSpace{2D, Float64, 16 dofs}", detailed)
        # Not collapsed, since the leaves genuinely differ.
        @test !occursin("2 × ", detailed)
        @test !endswith(detailed, '\n')
    end

    @testset "Grid function" begin
        uₕ = Rₕ(Wₕ, x -> x[1])
        compact = sprint(show, uₕ)
        @test compact == "VectorElement{2D, Float64, 9 dofs}"
        @test !occursin('\n', compact)
        @test summary(uₕ) == compact
        @test length(summary(uₕ)) < 60

        detailed = sprint(show, MIME"text/plain"(), uₕ)
        @test occursin("VectorElement", detailed)
        @test occursin("Space", detailed)
        @test occursin("ScalarGridSpace{2D, Float64, 9 dofs}", detailed)
        @test occursin("Values", detailed)
        @test !endswith(detailed, '\n')
    end

    @testset "points/half_points forward to the mesh" begin
        @testset "1D" begin
            Ωₕ1 = mesh(domain(interval(0.0, 1.0)), 9, false)
            Wₕ1 = gridspace(Ωₕ1)
            @test points(Wₕ1) == points(Ωₕ1)
            @test Bramble.half_points(Wₕ1) == Bramble.half_points(Ωₕ1)
        end

        @testset "2D" begin
            Ωₕ2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (7, 6), (false, false))
            Wₕ2 = gridspace(Ωₕ2)
            @test points(Wₕ2) == points(Ωₕ2)
            @test Bramble.half_points(Wₕ2) == Bramble.half_points(Ωₕ2)

            x, y = points(Wₕ2)
            @test x == points(Ωₕ2)[1] && y == points(Ωₕ2)[2]
        end
    end
end

# Composite grid space invariants (gpena/Bramble.jl#120).
#
# Component count, grid size and partition are all drawn, so nothing here rests on the
# particular `Val(2)`/`Val(3)` spaces the deterministic tests use. The components are given
# values that differ by four orders of magnitude between blocks, for the reason recorded in
# the verification notes: a routing bug that sends block `j`'s data to block `k` is
# invisible when every component carries the same number, and `innerₕ` on a composite space
# once returned `[0.5, 0.5]` where the answer was `[0.5, 50.0]` and passed every test there
# was.
WITH_SLOW_TESTS && @testset "Composite space properties (Supposition)" begin
    positive_h = Data.Floats{Float64}(;
        minimum = 0.01, maximum = 10.0, nans = false, infs = false
    )

    # The k-th component is `10^(2(k-1))` times a shape that is not constant, so a block
    # that receives the wrong source shows it in the number rather than agreeing by accident.
    _component(k) = x -> 10.0^(2 * (k - 1)) * (sin(3x) + 2)

    function _setup(h, ncomp)
        pts = _nonuniform_points(h)
        Ωₕ = mesh(domain(interval(0.0, 1.0)), length(pts), false)
        set_points!(Ωₕ, pts)
        Wₕ = gridspace(Ωₕ)
        Vₕ = gridspace(Ωₕ, Val(ncomp))
        return Wₕ, Vₕ
    end

    @check function check_component_extraction_and_block_layout(
            h = Data.Vectors(positive_h; min_size = 3, max_size = 8),
            ncomp = Data.Integers(2, 4)
    )
        Wₕ, Vₕ = _setup(h, ncomp)
        n = ndofs(Wₕ)
        ndofs(Vₕ) == ncomp * n || return false

        fs = ntuple(_component, ncomp)
        uₕ = Rₕ(Vₕ, fs)

        for k in 1:ncomp
            scalar = parent(Rₕ(Wₕ, fs[k]))
            # the component accessor and the raw block of the underlying vector are the
            # same numbers, in the same order
            parent(uₕ(k)) == scalar || return false
            parent(uₕ)[((k - 1) * n + 1):(k * n)] == scalar || return false
        end
        return true
    end

    @check function check_innerh_splits_over_blocks(
            h = Data.Vectors(positive_h; min_size = 3, max_size = 8),
            ncomp = Data.Integers(2, 4)
    )
        Wₕ, Vₕ = _setup(h, ncomp)
        fs = ntuple(_component, ncomp)
        uₕ = Rₕ(Vₕ, fs)
        vₕ = Rₕ(Vₕ, ntuple(k -> (x -> _component(k)(x) + x^2), ncomp))

        total = innerₕ(uₕ, vₕ)
        blockwise = sum(innerₕ(uₕ(k), vₕ(k)) for k in 1:ncomp)
        scale = max(abs(total), abs(blockwise), 1.0)
        isapprox(total, blockwise; atol = 1e-11 * scale, rtol = 1e-11) || return false

        # Blocks are orthogonal: a function living in one component only contributes
        # nothing against a function living in another.
        ncomp < 2 && return true
        eₕ1, eₕ2 = element(Vₕ), element(Vₕ)
        parent(eₕ1) .= 0.0
        parent(eₕ2) .= 0.0
        parent(eₕ1(1)) .= 1.0
        parent(eₕ2(2)) .= 1.0
        return abs(innerₕ(eₕ1, eₕ2)) <= 1e-12
    end

    @check function check_operators_act_blockwise(
            h = Data.Vectors(positive_h; min_size = 3, max_size = 8),
            ncomp = Data.Integers(2, 4)
    )
        Wₕ, Vₕ = _setup(h, ncomp)
        fs = ntuple(_component, ncomp)
        uₕ = Rₕ(Vₕ, fs)

        for op in (D₋ₓ, Mₓ, jumpₓ)
            composite = op(uₕ)
            for k in 1:ncomp
                scalar = parent(op(Rₕ(Wₕ, fs[k])))
                parent(composite(k)) == scalar || return false
            end
        end
        return true
    end

    # Non-vacuous: the blocks really do carry different numbers, so the equalities above are
    # not comparing a value with itself.
    @testset "Non-vacuous" begin
        Random.seed!(20260913)
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 9, false)
        Vₕ = gridspace(Ωₕ, Val(3))
        uₕ = Rₕ(Vₕ, ntuple(_component, 3))

        @test !isapprox(parent(uₕ(1)), parent(uₕ(2)))
        @test !isapprox(parent(uₕ(2)), parent(uₕ(3)))
        @test innerₕ(uₕ(3), uₕ(3)) > 1e3 * innerₕ(uₕ(1), uₕ(1))
    end
end

end # module SpaceGridspacesTests
