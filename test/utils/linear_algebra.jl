using Test
using Bramble
using Bramble: _dot, _dot_masked,
               _cpu_threaded_for!, _serial_for!, _cpu_threaded_scatter_for!,
               _write_components!, Serial, Parallel
using LinearAlgebra: dot
using StaticArrays

@testset "Linear Algebra Utilities" begin
    @testset "_dot function" begin
        # Test basic dot product
        u = [1.0, 2.0, 3.0]
        v = [4.0, 5.0, 6.0]
        w = [2.0, 2.0, 2.0]

        result = _dot(u, v, w)
        expected = (1.0 * 4.0 * 2.0) + (2.0 * 5.0 * 2.0) + (3.0 * 6.0 * 2.0)
        @test result ≈ expected
        @test result ≈ 64.0

        # Test with zeros
        u_zero = [0.0, 0.0, 0.0]
        @test _dot(u_zero, v, w) ≈ 0.0
        @test _dot(u, u_zero, w) ≈ 0.0
        @test _dot(u, v, u_zero) ≈ 0.0

        # Test with ones
        ones_vec = [1.0, 1.0, 1.0, 1.0]
        @test _dot(ones_vec, ones_vec, ones_vec) ≈ 4.0

        # Test Float32
        u_f32 = Float32[1.0, 2.0, 3.0]
        v_f32 = Float32[4.0, 5.0, 6.0]
        w_f32 = Float32[2.0, 2.0, 2.0]
        result_f32 = _dot(u_f32, v_f32, w_f32)
        @test result_f32 isa Float32
        @test result_f32 ≈ 64.0f0

        # Test mixed types (Float32 and Float64)
        result_mixed = _dot(u_f32, v, w_f32)
        @test result_mixed isa Float64
        @test result_mixed ≈ 64.0

        # Test single element
        @test _dot([2.0], [3.0], [4.0]) ≈ 24.0

        # Test dimension mismatch error
        @test_throws DimensionMismatch _dot([1.0, 2.0], [1.0, 2.0, 3.0], [1.0, 2.0])

        # Test SVector (zero allocations)
        sv_u = SVector(1.0, 2.0, 3.0)
        sv_v = SVector(4.0, 5.0, 6.0)
        sv_w = SVector(2.0, 2.0, 2.0)
        @test _dot(sv_u, sv_v, sv_w) ≈ 64.0
        @test_allocs _dot(sv_u, sv_v, sv_w)

        # Test larger vectors
        n = 100
        u_large = collect(1.0:n)
        v_large = ones(n)
        w_large = fill(2.0, n)
        result_large = _dot(u_large, v_large, w_large)
        expected_large = 2.0 * sum(1:n)
        @test result_large ≈ expected_large
    end

    @testset "_serial_for!" begin
        n = 10
        v = zeros(n)
        idxs = 1:n

        # Test simple assignment
        f = i -> Float64(i^2)
        _serial_for!(v, idxs, f)
        @test v == [Float64(i^2) for i in 1:n]

        # Test with partial indices
        v2 = ones(n)
        idxs_partial = 3:7
        f2 = i -> Float64(i * 10)
        _serial_for!(v2, idxs_partial, f2)
        @test v2[1:2] == [1.0, 1.0]
        @test v2[3:7] == [30.0, 40.0, 50.0, 60.0, 70.0]
        @test v2[8:10] == [1.0, 1.0, 1.0]

        # Test with CartesianIndices (2D array)
        A = zeros(3, 4)
        cart_idxs = CartesianIndices(A)
        f3 = idx -> Float64(idx[1] + idx[2])
        _serial_for!(A, cart_idxs, f3)
        for i in 1:3, j in 1:4

            @test A[i, j] ≈ Float64(i + j)
        end
    end

    @testset "_cpu_threaded_for!" begin
        # Dispatched on policy now, not gated by size: both Serial() and Parallel() are
        # exercised directly at every size below, rather than needing a grid large enough to
        # cross a threshold (there is none left to cross).
        for policy in (Serial(), Parallel())
            n = 100
            v = zeros(n)
            idxs = 1:n

            f = i -> Float64(i^2)
            _cpu_threaded_for!(policy, v, idxs, f)
            @test v == [Float64(i^2) for i in 1:n]

            # Test with partial indices
            v2 = ones(n)
            idxs_partial = 10:50
            f2 = i -> Float64(i * 2)
            _cpu_threaded_for!(policy, v2, idxs_partial, f2)
            @test v2[1:9] == ones(9)
            @test v2[10:50] == [Float64(i * 2) for i in 10:50]
            @test v2[51:100] == ones(50)

            # Test with CartesianIndices
            B = zeros(10, 10)
            cart_idxs = CartesianIndices(B)
            f3 = idx -> Float64(idx[1] * idx[2])
            _cpu_threaded_for!(policy, B, cart_idxs, f3)
            for i in 1:10, j in 1:10

                @test B[i, j] ≈ Float64(i * j)
            end
        end

        # Verify Parallel() gives the same result as Serial()
        n = 100
        idxs = 1:n
        v_serial = zeros(n)
        v_parallel = zeros(n)
        f_test = i -> sin(Float64(i)) + cos(Float64(i))
        _cpu_threaded_for!(Serial(), v_serial, idxs, f_test)
        _cpu_threaded_for!(Parallel(), v_parallel, idxs, f_test)
        @test v_serial ≈ v_parallel
    end

    @testset "_dot_masked" begin
        u = [1.0, 2.0, 3.0, 4.0]
        v = [2.0, 3.0, 4.0, 5.0]
        w = [0.5, 1.0, 1.5, 2.0]

        # Mask selecting elements 2 and 4
        mask = BitVector([false, true, false, true])
        expected = (2.0 * 3.0 * 1.0) + (4.0 * 5.0 * 2.0) # 6 + 40 = 46
        @test _dot_masked(u, v, w, mask) ≈ expected

        # All-true mask matches unmasked _dot
        mask_all = trues(4)
        @test _dot_masked(u, v, w, mask_all) ≈ _dot(u, v, w)

        # All-false mask produces zero
        mask_none = falses(4)
        @test _dot_masked(u, v, w, mask_none) == 0.0

        # Dimension mismatch on vectors
        @test_throws DimensionMismatch _dot_masked([1.0], [1.0, 2.0], [1.0], mask_all)

        # Dimension mismatch on mask length
        @test_throws DimensionMismatch _dot_masked(u, v, w, BitVector([true, false]))

        # StaticArrays allocation test
        sv_u = SVector(1.0, 2.0, 3.0, 4.0)
        sv_v = SVector(2.0, 3.0, 4.0, 5.0)
        sv_w = SVector(0.5, 1.0, 1.5, 2.0)
        @test _dot_masked(sv_u, sv_v, sv_w, mask) ≈ expected
    end

    @testset "_write_components! and _cpu_threaded_scatter_for!" begin
        # Direct _write_components! recursion
        a = zeros(3)
        b = zeros(3)
        c = zeros(3)
        _write_components!((a, b, c), (10.0, 20.0, 30.0), 2)
        @test a[2] == 10.0
        @test b[2] == 20.0
        @test c[2] == 30.0
        @test _write_components!((), (), 1) === nothing

        # _cpu_threaded_scatter_for! with Serial and Parallel
        for policy in (Serial(), Parallel())
            n = 50
            m1 = zeros(n)
            m2 = zeros(n)
            g = i -> (Float64(i), Float64(2i))
            _cpu_threaded_scatter_for!(policy, (m1, m2), 1:n, g)
            @test m1 == [Float64(i) for i in 1:n]
            @test m2 == [Float64(2i) for i in 1:n]
        end

        # Serial and Parallel match
        n = 64
        m1_s, m2_s = zeros(n), zeros(n)
        m1_p, m2_p = zeros(n), zeros(n)
        g_fn = i -> (sin(Float64(i)), cos(Float64(i)))
        _cpu_threaded_scatter_for!(Serial(), (m1_s, m2_s), 1:n, g_fn)
        _cpu_threaded_scatter_for!(Parallel(), (m1_p, m2_p), 1:n, g_fn)
        @test m1_s ≈ m1_p
        @test m2_s ≈ m2_p
    end

    @testset "Type Stability" begin
        u = [1.0, 2.0, 3.0]
        v = [4.0, 5.0, 6.0]
        w = [2.0, 2.0, 2.0]
        mask = BitVector([true, false, true])
        @inferred _dot(u, v, w)
        @inferred _dot_masked(u, v, w, mask)
    end
end
