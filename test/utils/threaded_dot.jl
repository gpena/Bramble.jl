module UtilsThreadedDotTests

# S2.1 (gpena/Bramble.jl#301): `_dot`/`_dot_masked(::CpuThreaded, ...)` used to fall through
# to the plain serial kernel -- there was no separate threaded reduction to check against.
# Now they chunk `1:length(u)` into `Threads.nthreads()` static bands, each task landing a
# partial sum in a fixed-size buffer summed serially at the end. This tests the new
# `_threaded_dot`/`_threaded_dot_masked` kernels directly, plus the `_dot`/`_dot_masked`
# `CpuThreaded()` policy dispatch that now reaches them, against the untouched serial
# kernels as ground truth -- dense, single-mask (`BitVector`) and multi-marker
# (`MarkedIndicesUnion`) alike.
using Test
using Bramble
using Bramble: _dot, _dot_masked, _threaded_dot, _threaded_dot_masked, MarkedIndicesUnion, CpuThreaded

@testset "Threaded _dot/_dot_masked (CpuThreaded chunked reduction)" begin
    # Invariants tested:
    # 1. Dense `_threaded_dot` matches the serial `_dot` at rtol=1e-12, across a length that
    #    is empty, length 1, shorter than the thread count, and long enough to give every
    #    thread real work.
    # 2. `_dot(CpuThreaded(), u, v, w)` (the policy seam) agrees with `_threaded_dot` called
    #    directly -- the dispatch change didn't just add a dead method.
    # 3. Same two checks for a `BitVector` mask and for a two-marker `MarkedIndicesUnion`.
    # 4. Allocation is independent of vector length: two very different lengths allocate
    #    exactly the same number of bytes (the buffer's size is `Threads.nthreads()`, not
    #    `length(u)`).
    nthreads = Threads.nthreads()

    # Deterministic, non-uniform values (no `range(...; length=0/1)` edge case to dodge).
    _u(n) = Float64[0.3 + 0.11 * i for i in 1:n]
    _v(n) = Float64[-0.6 + 0.07 * i for i in 1:n]
    _w(n) = Float64[1.0 + 0.05 * i for i in 1:n]

    @testset "Dense, length $n" for n in (0, 1, max(1, nthreads - 1), 37, 5000)
        u, v, w = _u(n), _v(n), _w(n)

        expected = _dot(u, v, w)
        got = _threaded_dot(u, v, w)
        @test isapprox(got, expected; rtol = 1.0e-12, atol = 1.0e-12)
        @test isapprox(_dot(CpuThreaded(), u, v, w), got; rtol = 1.0e-12, atol = 1.0e-12)
    end

    @test_throws DimensionMismatch _threaded_dot([1.0, 2.0], [1.0, 2.0, 3.0], [1.0, 2.0])

    @testset "BitVector mask, length $n" for n in (0, 1, max(1, nthreads - 1), 37, 5000)
        u, v, w = _u(n), _v(n), _w(n)
        mask = falses(n)
        n > 0 && (mask[1:2:n] .= true) # every other index set, exercises both branches

        expected = _dot_masked(u, v, w, mask)
        got = _threaded_dot_masked(u, v, w, mask)
        @test isapprox(got, expected; rtol = 1.0e-12, atol = 1.0e-12)
        @test isapprox(
            _dot_masked(CpuThreaded(), u, v, w, mask), got; rtol = 1.0e-12, atol = 1.0e-12
        )

        # An all-false mask is the zero vector, an all-true mask matches the dense dot.
        @test _threaded_dot_masked(u, v, w, falses(n)) == 0.0
        @test isapprox(
            _threaded_dot_masked(u, v, w, trues(n)), _threaded_dot(u, v, w);
            rtol = 1.0e-12, atol = 1.0e-12
        )
    end

    @test_throws DimensionMismatch _threaded_dot_masked(
        [1.0], [1.0, 2.0], [1.0], BitVector([true, false])
    )

    @testset "MarkedIndicesUnion (two markers), length $n" for n in (
        0, 1, max(1, nthreads - 1), 37, 5000,
    )
        u, v, w = _u(n), _v(n), _w(n)
        mask1 = falses(n)
        mask2 = falses(n)
        n > 0 && (mask1[1:3:n] .= true) # disjoint-ish, overlapping on some indices
        n > 0 && (mask2[2:3:n] .= true)
        union_mask = MarkedIndicesUnion((mask1, mask2))
        bit_union = mask1 .| mask2

        expected = _dot_masked(u, v, w, bit_union)
        got = _threaded_dot_masked(u, v, w, union_mask)
        @test isapprox(got, expected; rtol = 1.0e-12, atol = 1.0e-12)
        @test isapprox(
            _dot_masked(CpuThreaded(), u, v, w, union_mask), got;
            rtol = 1.0e-12, atol = 1.0e-12
        )
    end

    @test_throws DimensionMismatch _threaded_dot_masked(
        [1.0], [1.0, 2.0], [1.0], MarkedIndicesUnion((BitVector([true]), BitVector([false])))
    )

    # Real masks are typically a mesh marker such as `:boundary`: O(perimeter) set bits
    # inside an O(n^2) grid, nothing like the dense/every-other masks above. The word-chunked
    # walk (word range partitioned across threads, whole zero words skipped, set bits of a
    # nonzero word walked via `trailing_zeros`) must still land on the same value the serial
    # kernel's own set-bit walk does.
    @testset "Sparse mesh markers (:boundary, and a two-marker union) match serial" begin
        S = interval(0.0, 1.0) × interval(0.0, 1.0)
        Ωd = domain(S, :bottom => :bottom, :left => :left)
        n = 201
        Ωₕ = mesh(Ωd, (n, n), (true, true))
        boundary = Bramble.markers(Ωₕ)[:boundary]
        m = length(boundary)
        u, v, w = _u(m), _v(m), _w(m)

        expected = _dot_masked(u, v, w, boundary)
        got = _threaded_dot_masked(u, v, w, boundary)
        @test isapprox(got, expected; rtol = 1.0e-12, atol = 1.0e-12)
        @test isapprox(
            _dot_masked(CpuThreaded(), u, v, w, boundary), got; rtol = 1.0e-12, atol = 1.0e-12
        )

        bottom = Bramble.markers(Ωₕ)[:bottom]
        left = Bramble.markers(Ωₕ)[:left]
        union_mask = MarkedIndicesUnion((bottom, left))
        bit_union = bottom .| left

        expected_u = _dot_masked(u, v, w, bit_union)
        got_u = _threaded_dot_masked(u, v, w, union_mask)
        @test isapprox(got_u, expected_u; rtol = 1.0e-12, atol = 1.0e-12)
        @test isapprox(
            _dot_masked(CpuThreaded(), u, v, w, union_mask), got_u;
            rtol = 1.0e-12, atol = 1.0e-12
        )
    end

    @testset "Allocation independent of vector length" begin
        # Not asserted equal outright: `Threads.@threads` task bookkeeping can jitter by a
        # couple hundred bytes call to call regardless of `n` (observed 1024 vs 1136 B here),
        # the same reason test/ext/polyester_ext.jl's own allocation testset bounds rather
        # than equates. What must hold, and what a per-element-proportional allocation would
        # break, is that going from 64 to 200_000 elements (a 3000x length increase) does not
        # move the byte count by anything like that factor.
        function _dense_allocs(n)
            u, v, w = fill(1.0, n), fill(2.0, n), fill(3.0, n)
            _threaded_dot(u, v, w) # warm up
            return @allocated _threaded_dot(u, v, w)
        end
        small_bytes = _dense_allocs(64)
        large_bytes = _dense_allocs(200_000)
        @test large_bytes <= 4 * small_bytes + 1024
        @test large_bytes < 100_000 # proportional to length would be MBs

        function _masked_allocs(n)
            u, v, w = fill(1.0, n), fill(2.0, n), fill(3.0, n)
            mask = trues(n)
            _threaded_dot_masked(u, v, w, mask) # warm up
            return @allocated _threaded_dot_masked(u, v, w, mask)
        end
        small_masked = _masked_allocs(64)
        large_masked = _masked_allocs(200_000)
        @test large_masked <= 4 * small_masked + 1024
        @test large_masked < 100_000
    end
end

end # module
