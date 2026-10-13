module MeshConstructorsTests

using Test
using Random
using Bramble
using Bramble: _expand_uniform, Backend, Serial, stepsize, Mesh1D, MeshnD, is_uniform
using ..TestUtils: alloc_test, WITH_3D_TESTS, @test_allocs

@testset "Mesh constructors" begin
    @testset "Internal _expand_uniform helper" begin
        @test @inferred(_expand_uniform(true, Val(1))) === (true,)
        @test @inferred(_expand_uniform(false, Val(1))) === (false,)
        @test @inferred(_expand_uniform(true, Val(2))) === (true, true)
        @test @inferred(_expand_uniform(false, Val(2))) === (false, false)
        @test @inferred(_expand_uniform(true, Val(3))) === (true, true, true)
        @test @inferred(_expand_uniform(false, Val(3))) === (false, false, false)

        @test @inferred(_expand_uniform((true,), Val(1))) === (true,)
        @test @inferred(_expand_uniform((true, false), Val(2))) === (true, false)
        @test @inferred(_expand_uniform((true, false, true), Val(3))) === (true, false, true)
    end

    @testset "mesh docstring" begin
        # The docstring must attach to `mesh`, not to a helper defined next to it.
        doc = string(@doc mesh)
        @test occursin("isotropic", doc)
        @test occursin("warn_marker_mismatch", doc)
    end

    @testset "Direct CartesianProduct input" begin
        I = interval(0.0, 1.0)
        X2 = interval(0.0, 1.0) × interval(0.0, 2.0)
        X3 = box((0.0, 0.0, 0.0), (1.0, 2.0, 3.0))

        @testset "1D CartesianProduct" begin
            M1 = @inferred mesh(I, 11)
            @test M1 isa Mesh1D
            @test size(M1) == (11,)
            @test is_uniform(M1)
            @test length(M1) == 11
            @test M1[begin] == 0.0
            @test M1[end] == 1.0
            @test :boundary in keys(markers(M1))
            @test :interior in keys(markers(M1))
            @test markers(M1)[:boundary] == markers(mesh(domain(I), 11))[:boundary]
            @test markers(M1)[:interior] == markers(mesh(domain(I), 11))[:interior]

            M1_tup = @inferred mesh(I, (11,))
            @test size(M1_tup) == (11,)
            @test is_uniform(M1_tup)
            @test points(M1_tup) == points(M1)

            M1_nu = @inferred mesh(I, 11, false)
            @test M1_nu isa Mesh1D
            @test size(M1_nu) == (11,)
            @test !all(diff(points(M1_nu)::AbstractVector) .≈ stepsize(M1))
            @test !is_uniform(M1_nu)

            M1_kw = @inferred mesh(I, 11; uniform = false)
            @test size(M1_kw) == (11,)
            @test !is_uniform(M1_kw)

            M1_kw_u = @inferred mesh(I, 11; uniform = true)
            @test is_uniform(M1_kw_u)
        end

        @testset "2D CartesianProduct" begin
            M2 = @inferred mesh(X2, (11, 21))
            @test M2 isa MeshnD{2}
            @test size(M2) == (11, 21)
            @test is_uniform(M2(1)) && is_uniform(M2(2))
            @test length(M2) == 231
            @test M2[begin] == (0.0, 0.0)
            @test M2[end] == (1.0, 2.0)
            @test :boundary in keys(markers(M2))
            @test :interior in keys(markers(M2))
            @test markers(M2)[:boundary] == markers(mesh(domain(X2), (11, 21)))[:boundary]
            @test markers(M2)[:interior] == markers(mesh(domain(X2), (11, 21)))[:interior]

            # Isotropic single integer resolution
            M2_iso = @inferred mesh(X2, 15)
            @test M2_iso isa MeshnD{2}
            @test size(M2_iso) == (15, 15)
            @test is_uniform(M2_iso(1)) && is_uniform(M2_iso(2))
            @test points(M2_iso) == points(mesh(domain(X2), (15, 15)))
            @test markers(M2_iso)[:boundary] == markers(mesh(domain(X2), (15, 15)))[:boundary]
            @test markers(M2_iso)[:interior] == markers(mesh(domain(X2), (15, 15)))[:interior]

            # Positional non-uniform and mixed uniformity
            M2_nu = @inferred mesh(X2, 10, false)
            @test size(M2_nu) == (10, 10)
            @test !is_uniform(M2_nu(1)) && !is_uniform(M2_nu(2))

            M2_u = @inferred mesh(X2, (10, 15), true)
            @test is_uniform(M2_u(1)) && is_uniform(M2_u(2))

            M2_pnu = @inferred mesh(X2, (10, 15), false)
            @test !is_uniform(M2_pnu(1)) && !is_uniform(M2_pnu(2))

            M2_mixed = @inferred mesh(X2, (10, 15), (true, false))
            @test size(M2_mixed) == (10, 15)
            @test is_uniform(M2_mixed(1)) && !is_uniform(M2_mixed(2))

            M2_mixed_rev = @inferred mesh(X2, (10, 15), (false, true))
            @test !is_uniform(M2_mixed_rev(1)) && is_uniform(M2_mixed_rev(2))

            M2_iso_mixed = @inferred mesh(X2, 12, (true, false))
            @test size(M2_iso_mixed) == (12, 12)
            @test is_uniform(M2_iso_mixed(1)) && !is_uniform(M2_iso_mixed(2))

            # Keyword uniformity
            M2_kw_bool = @inferred mesh(X2, (10, 12); uniform = false)
            @test size(M2_kw_bool) == (10, 12)
            @test !is_uniform(M2_kw_bool(1)) && !is_uniform(M2_kw_bool(2))

            M2_kw_rev = @inferred mesh(X2, (10, 12); uniform = (false, true))
            @test !is_uniform(M2_kw_rev(1)) && is_uniform(M2_kw_rev(2))

            M2_kw_tup = @inferred mesh(X2, 14; uniform = (true, false))
            @test size(M2_kw_tup) == (14, 14)
            @test is_uniform(M2_kw_tup(1)) && !is_uniform(M2_kw_tup(2))
        end

        # 3D only: built in `slow` and by the nightly run (gpena/Bramble.jl#650).
        WITH_3D_TESTS && @testset "3D CartesianProduct" begin
            M3 = @inferred mesh(X3, (5, 6, 7))
            @test M3 isa MeshnD{3}
            @test size(M3) == (5, 6, 7)
            @test :boundary in keys(markers(M3))
            @test :interior in keys(markers(M3))
            @test markers(M3)[:boundary] == markers(mesh(domain(X3), (5, 6, 7)))[:boundary]

            M3_iso = @inferred mesh(X3, 8)
            @test M3_iso isa MeshnD{3}
            @test size(M3_iso) == (8, 8, 8)
            @test points(M3_iso) == points(mesh(domain(X3), (8, 8, 8)))
            @test all(i -> is_uniform(M3(i)) && is_uniform(M3_iso(i)), 1:3)

            M3_mixed = @inferred mesh(X3, (5, 6, 7), (true, false, true))
            @test size(M3_mixed) == (5, 6, 7)
            @test is_uniform(M3_mixed(1)) && !is_uniform(M3_mixed(2)) && is_uniform(M3_mixed(3))

            M3_iso_mixed = @inferred mesh(X3, 6, (false, true, false))
            @test size(M3_iso_mixed) == (6, 6, 6)
            @test !is_uniform(M3_iso_mixed(1)) && is_uniform(M3_iso_mixed(2)) &&
                  !is_uniform(M3_iso_mixed(3))
        end
    end

    @testset "Isotropic integer resolution on Domain" begin
        I = interval(0.0, 1.0)
        X2 = interval(0.0, 1.0) × interval(0.0, 2.0)
        X3 = box((0.0, 0.0, 0.0), (1.0, 2.0, 3.0))

        Ω1 = domain(I)
        Ω2 = domain(X2)
        Ω3 = domain(X3)

        # 1D equivalence
        m1_int = @inferred mesh(Ω1, 25)
        m1_tup = @inferred mesh(Ω1, (25,))
        @test points(m1_int) == points(m1_tup)
        @test markers(m1_int)[:boundary] == markers(m1_tup)[:boundary]
        @test markers(m1_int)[:interior] == markers(m1_tup)[:interior]

        # 2D equivalence
        m2_int = @inferred mesh(Ω2, 20)
        m2_tup = @inferred mesh(Ω2, (20, 20))
        @test size(m2_int) == (20, 20)
        @test is_uniform(m2_int(1)) && is_uniform(m2_int(2))
        @test points(m2_int) == points(m2_tup)
        @test stepsize(m2_int) == stepsize(m2_tup)
        @test markers(m2_int)[:boundary] == markers(m2_tup)[:boundary]
        @test markers(m2_int)[:interior] == markers(m2_tup)[:interior]

        # 2D non-uniform isotropic
        m2_int_nu = @inferred mesh(Ω2, 15, false)
        @test size(m2_int_nu) == (15, 15)
        @test !is_uniform(m2_int_nu(1)) && !is_uniform(m2_int_nu(2))

        m2_int_u = @inferred mesh(Ω2, 15, true)
        @test is_uniform(m2_int_u(1)) && is_uniform(m2_int_u(2))

        # 2D mixed uniformity with isotropic npts
        m2_int_mixed = @inferred mesh(Ω2, 15, (false, true))
        @test size(m2_int_mixed) == (15, 15)
        @test !is_uniform(m2_int_mixed(1)) && is_uniform(m2_int_mixed(2))

        # 3D equivalence, built in `slow` and by the nightly run (gpena/Bramble.jl#650)
        if WITH_3D_TESTS
            m3_int = @inferred mesh(Ω3, 6)
            m3_tup = @inferred mesh(Ω3, (6, 6, 6))
            @test size(m3_int) == (6, 6, 6)
            @test points(m3_int) == points(m3_tup)
            @test markers(m3_int)[:boundary] == markers(m3_tup)[:boundary]
            @test markers(m3_int)[:interior] == markers(m3_tup)[:interior]
        end

        # Preserving custom markers
        Ω_custom = domain(X2, :left_edge => :left, :bottom_edge => :bottom)
        m_custom = @inferred mesh(Ω_custom, 12)
        @test :left_edge in keys(markers(m_custom))
        @test :bottom_edge in keys(markers(m_custom))
        @test :boundary in keys(markers(m_custom))
        @test :interior in keys(markers(m_custom))
        @test markers(m_custom)[:left_edge] == markers(mesh(Ω_custom, (12, 12)))[:left_edge]
    end

    @testset "Keyword option forwarding" begin
        X = interval(0.0, 1.0) × interval(0.0, 1.0)
        Ω = domain(X)

        # Backend forwarding
        be_f32 = backend(Float32)
        m_f32_set = @inferred mesh(X, 10; backend = be_f32)
        @test eltype(m_f32_set) === Float32

        m_f32_dom = @inferred mesh(Ω, 10; backend = be_f32)
        @test eltype(m_f32_dom) === Float32

        m_f32_tup = @inferred mesh(X, (10, 10); backend = be_f32)
        @test eltype(m_f32_tup) === Float32

        # warn_marker_mismatch forwarding
        Ω_disagree = domain(X, :boundary => :left)
        @test_logs (:warn, r"boundary") mesh(Ω_disagree, 8; warn_marker_mismatch = true)
        @test_logs mesh(Ω_disagree, 8; warn_marker_mismatch = false)
        @test_logs (:warn, r"boundary") mesh(Ω_disagree, (8, 8); warn_marker_mismatch = true)
        @test_logs mesh(Ω_disagree, (8, 8); warn_marker_mismatch = false)
    end

    @testset "Endpoints pinned to a and b" begin
        # `a + (n - 1) * h` ends at 0.9999999999999999 here (gpena/Bramble.jl#695)
        m50 = mesh(domain(interval(0.0, 1.0)), 50)
        @test last(points(m50)) === 1.0
        @test is_uniform(m50)
        @test last(points(mesh(domain(interval(0.0f0, 1.0f0)), 50))) === 1.0f0

        # a seeded sweep of uniform and non-uniform intervals, both ends pinned
        rng = Random.Xoshiro(7)
        missed = Tuple{Float64, Float64, Int, Bool}[]
        for _ in 1:300
            a = rand(rng) - 0.5
            b = a + 0.1 + rand(rng)
            n = rand(rng, 3:60)
            for u in (true, false)
                p = points(mesh(domain(interval(a, b)), n, u))
                (first(p) == a && last(p) == b) || push!(missed, (a, b, n, u))
            end
        end
        @test isempty(missed)

        # an equality marker on the last node marks a whole side of a 50 x 50 square
        I = interval(0.0, 1.0)
        Ω = domain(I × I, :r => x -> x[1] == 1.0)
        @test count(Bramble.index_in_marker(mesh(Ω, (50, 50), (true, true)), :r)) == 50
    end

    @testset "Zero allocations in _expand_uniform" begin
        @test_allocs _expand_uniform(true, Val(1))
        @test_allocs _expand_uniform(false, Val(2))
        @test_allocs _expand_uniform((true, false), Val(2))
        @test_allocs _expand_uniform(true, Val(3))

        # The mesh accessors themselves are measured in mesh/inference_allocation.jl's
        # "Zero allocations"; what belongs here is the constructor helper above.
    end
end

end # module MeshConstructorsTests
