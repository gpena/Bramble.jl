module MeshConstructorsTests

using Test
using Bramble
using Bramble: _expand_uniform, Backend, Serial, stepsize, Mesh1D, MeshnD
using ..TestUtils: alloc_test, @test_allocs

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

    @testset "Direct CartesianProduct input" begin
        I = interval(0.0, 1.0)
        X2 = interval(0.0, 1.0) × interval(0.0, 2.0)
        X3 = box((0.0, 0.0, 0.0), (1.0, 2.0, 3.0))

        @testset "1D CartesianProduct" begin
            M1 = @inferred mesh(I, 11)
            @test M1 isa Mesh1D
            @test size(M1) == (11,)
            @test length(M1) == 11
            @test M1[begin] == 0.0
            @test M1[end] == 1.0
            @test :boundary in keys(markers(M1))
            @test :interior in keys(markers(M1))
            @test markers(M1)[:boundary] == markers(mesh(domain(I), 11))[:boundary]
            @test markers(M1)[:interior] == markers(mesh(domain(I), 11))[:interior]

            M1_tup = @inferred mesh(I, (11,))
            @test size(M1_tup) == (11,)
            @test points(M1_tup) == points(M1)

            M1_nu = @inferred mesh(I, 11, false)
            @test M1_nu isa Mesh1D
            @test size(M1_nu) == (11,)
            @test !all(diff(points(M1_nu)) .≈ stepsize(M1))

            M1_kw = @inferred mesh(I, 11; uniform = false)
            @test size(M1_kw) == (11,)
        end

        @testset "2D CartesianProduct" begin
            M2 = @inferred mesh(X2, (11, 21))
            @test M2 isa MeshnD{2}
            @test size(M2) == (11, 21)
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
            @test points(M2_iso) == points(mesh(domain(X2), (15, 15)))
            @test markers(M2_iso)[:boundary] == markers(mesh(domain(X2), (15, 15)))[:boundary]
            @test markers(M2_iso)[:interior] == markers(mesh(domain(X2), (15, 15)))[:interior]

            # Positional non-uniform and mixed uniformity
            M2_nu = @inferred mesh(X2, 10, false)
            @test size(M2_nu) == (10, 10)

            M2_mixed = @inferred mesh(X2, (10, 15), (true, false))
            @test size(M2_mixed) == (10, 15)

            M2_iso_mixed = @inferred mesh(X2, 12, (true, false))
            @test size(M2_iso_mixed) == (12, 12)

            # Keyword uniformity
            M2_kw_bool = @inferred mesh(X2, (10, 12); uniform = false)
            @test size(M2_kw_bool) == (10, 12)

            M2_kw_tup = @inferred mesh(X2, 14; uniform = (true, false))
            @test size(M2_kw_tup) == (14, 14)
        end

        @testset "3D CartesianProduct" begin
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
        @test points(m2_int) == points(m2_tup)
        @test stepsize(m2_int) == stepsize(m2_tup)
        @test markers(m2_int)[:boundary] == markers(m2_tup)[:boundary]
        @test markers(m2_int)[:interior] == markers(m2_tup)[:interior]

        # 2D non-uniform isotropic
        m2_int_nu = @inferred mesh(Ω2, 15, false)
        @test size(m2_int_nu) == (15, 15)

        # 2D mixed uniformity with isotropic npts
        m2_int_mixed = @inferred mesh(Ω2, 15, (false, true))
        @test size(m2_int_mixed) == (15, 15)

        # 3D equivalence
        m3_int = @inferred mesh(Ω3, 6)
        m3_tup = @inferred mesh(Ω3, (6, 6, 6))
        @test size(m3_int) == (6, 6, 6)
        @test points(m3_int) == points(m3_tup)
        @test markers(m3_int)[:boundary] == markers(m3_tup)[:boundary]
        @test markers(m3_int)[:interior] == markers(m3_tup)[:interior]

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
