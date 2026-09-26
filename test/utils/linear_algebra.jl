module UtilsLinearAlgebraTests

using Test
using Bramble
using Bramble:
               _dot,
               _dot_masked,
               MarkedIndices,
               _sweep_for!,
               _serial_for!,
               _sweep_scatter_for!,
               _write_components!,
               Serial,
               Parallel,
               CpuSerial,
               CpuThreaded,
               GpuKernel
using LinearAlgebra: dot
using StaticArrays
using ..TestUtils: alloc_test, @test_allocs

@testset "Linear algebra utilities" begin
    # Invariants tested (gpena/Bramble.jl#191, #298 defect 2):
    # 1. A host destination under a GpuPolicy is refused (message 1), naming both the
    #    destination's locality and the policy's.
    # 2. A device-locality destination under a CpuPolicy is refused too (message 2), naming
    #    both localities, and never advising a CpuPolicy backend rebuild -- that combination
    #    is itself rejected at construction (gpena/Bramble.jl#296). No GPU is needed to
    #    exercise this: the seam reads locality from the destination array's own type.
    # 3. Locality-agreeing pairs (a host destination under CpuSerial/CpuThreaded) still
    #    sweep and produce correct values -- this testset is not throw-only.
    # 4. The alias spellings still select the same two CPU methods they always did.
    @testset "Sweep guard refuses locality mismatches" begin
        v = zeros(4)
        @test_throws ArgumentError _sweep_for!(GpuKernel(), v, 1:4, identity)
        @test_throws ArgumentError _sweep_scatter_for!(
            GpuKernel(), (v,), 1:4, i -> (float(i),)
        )
        err = try
            _sweep_for!(GpuKernel(), v, 1:4, identity)
        catch e
            e
        end
        msg = sprint(showerror, err)
        @test occursin("destination array has host locality", msg)
        @test occursin("claims device locality", msg)
        @test occursin("GpuKernel", msg)

        _sweep_for!(Serial(), v, 1:4, i -> 2.0 * i)
        @test v == [2.0, 4.0, 6.0, 8.0]
        fill!(v, 0.0)
        _sweep_for!(CpuSerial(), v, 1:4, i -> 3.0 * i)
        @test v == [3.0, 6.0, 9.0, 12.0]
        fill!(v, 0.0)
        _sweep_for!(CpuThreaded(), v, 1:4, i -> 4.0 * i)
        @test v == [4.0, 8.0, 12.0, 16.0]

        # Reverse direction (gpena/Bramble.jl#298 defect 2): a device-locality destination
        # under a CpuPolicy, with no GPU or KernelAbstractions involved -- a small host-backed
        # array type that claims DeviceLocality() through the trait is enough.
        struct _FakeDeviceVector{T} <: DenseVector{T}
            data::Vector{T}
        end
        Base.size(v::_FakeDeviceVector) = size(v.data)
        Base.getindex(v::_FakeDeviceVector, i::Int) = getindex(v.data, i)
        Base.setindex!(v::_FakeDeviceVector, val, i::Int) = setindex!(v.data, val, i)
        Base.IndexStyle(::Type{<:_FakeDeviceVector}) = IndexLinear()

        Bramble.locality(::Type{<:_FakeDeviceVector}) = Bramble.DeviceLocality()

        v_dev = _FakeDeviceVector(zeros(4))
        @test_throws ArgumentError _sweep_for!(CpuSerial(), v_dev, 1:4, identity)
        err2 = try
            _sweep_for!(CpuSerial(), v_dev, 1:4, identity)
        catch e
            e
        end
        msg2 = sprint(showerror, err2)
        @test occursin("destination array has device locality", msg2)
        @test occursin("claims host locality", msg2)
        @test occursin("CpuSerial", msg2)
        @test !occursin("build this backend with a CpuPolicy", msg2)
        @test !occursin("build the backend with a CpuPolicy", msg2)
    end

    # Invariants tested:
    # 1. Trilinear form evaluation: ∑ u_i * v_i * w_i matches hand-calculated expected values.
    # 2. Annihilation: any zero vector argument produces a zero result.
    # 3. Precision preservation: Float32 inputs produce Float32 outputs; mixed types promote correctly.
    # 4. Dimension checking: mismatched vector lengths throw DimensionMismatch.
    # 5. Zero-allocation guarantee: static arrays (SVector) execute with zero heap allocations.
    @testset "Weighted trilinear dot product" begin
        u = [1.0, 2.0, 3.0]
        v = [4.0, 5.0, 6.0]
        w = [2.0, 2.0, 2.0]

        result = _dot(u, v, w)
        expected = (1.0 * 4.0 * 2.0) + (2.0 * 5.0 * 2.0) + (3.0 * 6.0 * 2.0)
        @test result ≈ expected
        @test result ≈ 64.0

        # Annihilation with zero vectors
        u_zero = [0.0, 0.0, 0.0]
        @test _dot(u_zero, v, w) ≈ 0.0
        @test _dot(u, u_zero, w) ≈ 0.0
        @test _dot(u, v, u_zero) ≈ 0.0

        ones_vec = [1.0, 1.0, 1.0, 1.0]
        @test _dot(ones_vec, ones_vec, ones_vec) ≈ 4.0

        # Precision preservation and type promotion
        u_f32 = Float32[1.0, 2.0, 3.0]
        v_f32 = Float32[4.0, 5.0, 6.0]
        w_f32 = Float32[2.0, 2.0, 2.0]
        result_f32 = _dot(u_f32, v_f32, w_f32)
        @test result_f32 isa Float32
        @test result_f32 ≈ 64.0f0

        result_mixed = _dot(u_f32, v, w_f32)
        @test result_mixed isa Float64
        @test result_mixed ≈ 64.0

        @test _dot([2.0], [3.0], [4.0]) ≈ 24.0

        # Dimension mismatch detection
        @test_throws DimensionMismatch _dot([1.0, 2.0], [1.0, 2.0, 3.0], [1.0, 2.0])

        # Zero allocations on static arrays
        sv_u = SVector(1.0, 2.0, 3.0)
        sv_v = SVector(4.0, 5.0, 6.0)
        sv_w = SVector(2.0, 2.0, 2.0)
        @test _dot(sv_u, sv_v, sv_w) ≈ 64.0
        @test_allocs _dot(sv_u, sv_v, sv_w)

        n = 100
        u_large = collect(1.0:n)
        v_large = ones(n)
        w_large = fill(2.0, n)
        result_large = _dot(u_large, v_large, w_large)
        expected_large = 2.0 * sum(1:n)
        @test result_large ≈ expected_large
    end

    # Invariants tested:
    # 1. Sequential mutation across 1D linear index ranges and Cartesian index collections.
    # 2. Sub-range execution leaves untouched elements unmodified.
    @testset "Serial in-place iteration" begin
        n = 10
        v = zeros(n)
        idxs = 1:n

        f = i -> Float64(i^2)
        _serial_for!(v, idxs, f)
        @test v == [Float64(i^2) for i in 1:n]

        # Partial range iteration
        v2 = ones(n)
        idxs_partial = 3:7
        f2 = i -> Float64(i * 10)
        _serial_for!(v2, idxs_partial, f2)
        @test v2[1:2] == [1.0, 1.0]
        @test v2[3:7] == [30.0, 40.0, 50.0, 60.0, 70.0]
        @test v2[8:10] == [1.0, 1.0, 1.0]

        # Cartesian index iteration
        A = zeros(3, 4)
        cart_idxs = CartesianIndices(A)
        f3 = idx -> Float64(idx[1] + idx[2])
        _serial_for!(A, cart_idxs, f3)
        for i in 1:3, j in 1:4

            @test A[i, j] ≈ Float64(i + j)
        end

        # Zero allocations during serial iteration on preallocated buffer
        v_alloc = zeros(10)
        @test_allocs _serial_for!(v_alloc, 1:10, f)
    end

    # Invariants tested:
    # 1. Serial() and Parallel() policies produce mathematically identical array mutations.
    # 2. Multidimensional CartesianIndices work consistently under both policies.
    @testset "Threaded in-place iteration" begin
        for policy in (Serial(), Parallel())
            n = 100
            v = zeros(n)
            idxs = 1:n

            f = i -> Float64(i^2)
            _sweep_for!(policy, v, idxs, f)
            @test v == [Float64(i^2) for i in 1:n]

            v2 = ones(n)
            idxs_partial = 10:50
            f2 = i -> Float64(i * 2)
            _sweep_for!(policy, v2, idxs_partial, f2)
            @test v2[1:9] == ones(9)
            @test v2[10:50] == [Float64(i * 2) for i in 10:50]
            @test v2[51:100] == ones(50)

            B = zeros(10, 10)
            cart_idxs = CartesianIndices(B)
            f3 = idx -> Float64(idx[1] * idx[2])
            _sweep_for!(policy, B, cart_idxs, f3)
            for i in 1:10, j in 1:10

                @test B[i, j] ≈ Float64(i * j)
            end
        end

        # Direct equivalence test between Serial() and Parallel()
        n = 100
        idxs = 1:n
        v_serial = zeros(n)
        v_parallel = zeros(n)
        f_test = i -> sin(Float64(i)) + cos(Float64(i))
        _sweep_for!(Serial(), v_serial, idxs, f_test)
        _sweep_for!(Parallel(), v_parallel, idxs, f_test)
        @test v_serial ≈ v_parallel

        # Zero allocations during Serial() policy execution
        v_serial_alloc = zeros(100)
        f_alloc = i -> Float64(i^2)
        @test_allocs _sweep_for!(Serial(), v_serial_alloc, 1:100, f_alloc)
    end

    # Invariants tested:
    # 1. Trilinear form restricted to indices in support of BitVector mask.
    # 2. 64-bit chunk bit-scanning accurately identifies active bit indices.
    # 3. All-true mask matches unmasked _dot.
    # 4. All-false mask produces exact zero.
    # 5. Length mismatches between vectors or between vector and mask raise DimensionMismatch.
    # 6. Allocation-free evaluation on static arrays.
    @testset "Masked dot product" begin
        u = [1.0, 2.0, 3.0, 4.0]
        v = [2.0, 3.0, 4.0, 5.0]
        w = [0.5, 1.0, 1.5, 2.0]

        # Mask selecting indices 2 and 4
        mask = BitVector([false, true, false, true])
        expected = (2.0 * 3.0 * 1.0) + (4.0 * 5.0 * 2.0)
        @test _dot_masked(u, v, w, mask) ≈ expected

        # Boundary cases: all-true and all-false masks
        mask_all = trues(4)
        @test _dot_masked(u, v, w, mask_all) ≈ _dot(u, v, w)

        mask_none = falses(4)
        @test _dot_masked(u, v, w, mask_none) == 0.0

        # Dimension validation
        @test_throws DimensionMismatch _dot_masked([1.0], [1.0, 2.0], [1.0], mask_all)
        @test_throws DimensionMismatch _dot_masked(u, v, w, BitVector([true, false]))

        # Static array evaluation and zero-allocation guarantee
        sv_u = SVector(1.0, 2.0, 3.0, 4.0)
        sv_v = SVector(2.0, 3.0, 4.0, 5.0)
        sv_w = SVector(0.5, 1.0, 1.5, 2.0)
        @test _dot_masked(sv_u, sv_v, sv_w, mask) ≈ expected
        @test_allocs _dot_masked(sv_u, sv_v, sv_w, mask)
    end

    # gpena/Bramble.jl#71: `MarkedIndices` is the one bit-walk `_dot_masked` above and
    # `_each_marked` (form/dirichlet_constraints.jl) both call, rather than each keeping its
    # own copy. Checked directly here, past a single 64-bit chunk, since the masks above are
    # all short enough to never exercise the chunk-skipping loop or the chunk-boundary
    # bit-index arithmetic at all.
    @testset "MarkedIndices" begin
        @test collect(MarkedIndices(falses(200))) == Int[]

        mask = falses(200)
        set_bits = [1, 5, 63, 64, 65, 127, 128, 129, 199, 200]
        mask[set_bits] .= true
        @test collect(MarkedIndices(mask)) == set_bits

        # A nonzero offset shifts every yielded index, as consulting a composite leaf's own
        # mask at its position in the global vector needs.
        @test collect(MarkedIndices(mask, 1000)) == set_bits .+ 1000

        # A mask whose length isn't a multiple of 64: the padding bits of the final chunk
        # must be zero (a `BitVector` invariant), so the walk must not yield past `length(mask)`.
        odd_mask = falses(70)
        odd_mask[[3, 70]] .= true
        @test collect(MarkedIndices(odd_mask)) == [3, 70]

        # Allocation-free: the whole point of walking chunks instead of `findall`.
        count_bits(m) = (
            n = 0;
            for _ in MarkedIndices(m)
                n += 1
            end;
            n
        )
        @test_allocs count_bits(mask)
    end

    # Invariants tested:
    # 1. _write_components! unrolls via recursion on tuple types without allocations.
    # 2. _sweep_scatter_for! scatters multi-component kernel evaluations in a single pass.
    # 3. Serial() and Parallel() execution policies yield identical results.
    @testset "Component scattering" begin
        a = zeros(3)
        b = zeros(3)
        c = zeros(3)
        _write_components!((a, b, c), (10.0, 20.0, 30.0), 2)
        @test a[2] == 10.0
        @test b[2] == 20.0
        @test c[2] == 30.0
        @test _write_components!((), (), 1) === nothing

        # Zero allocations during component unpacking
        targets = (zeros(3), zeros(3), zeros(3))
        vals = (10.0, 20.0, 30.0)
        @test_allocs _write_components!(targets, vals, 2)

        for policy in (Serial(), Parallel())
            n = 50
            m1 = zeros(n)
            m2 = zeros(n)
            g = i -> (Float64(i), Float64(2i))
            _sweep_scatter_for!(policy, (m1, m2), 1:n, g)
            @test m1 == [Float64(i) for i in 1:n]
            @test m2 == [Float64(2i) for i in 1:n]
        end

        # Policy equivalence check
        n = 64
        m1_s, m2_s = zeros(n), zeros(n)
        m1_p, m2_p = zeros(n), zeros(n)
        g_fn = i -> (sin(Float64(i)), cos(Float64(i)))
        _sweep_scatter_for!(Serial(), (m1_s, m2_s), 1:n, g_fn)
        _sweep_scatter_for!(Parallel(), (m1_p, m2_p), 1:n, g_fn)
        @test m1_s ≈ m1_p
        @test m2_s ≈ m2_p

        # Zero allocations during Serial() component scattering
        scatter_targets = (zeros(50), zeros(50))
        g_scatter = i -> (Float64(i), Float64(2i))
        @test_allocs _sweep_scatter_for!(Serial(), scatter_targets, 1:50, g_scatter)
    end

    # Invariants tested:
    # 1. Return types are completely inferred by compiler for _dot and _dot_masked.
    @testset "Type stability" begin
        u = [1.0, 2.0, 3.0]
        v = [4.0, 5.0, 6.0]
        w = [2.0, 2.0, 2.0]
        mask = BitVector([true, false, true])
        @inferred _dot(u, v, w)
        @inferred _dot_masked(u, v, w, mask)
    end
end

end # module UtilsLinearAlgebraTests
