module UtilsThreadedDotTests

# S2.1 (gpena/Bramble.jl#301): `_dot`/`_dot_masked(::CpuThreaded, ...)` used to fall through
# to the plain serial kernel -- there was no separate threaded reduction to check against.
# Now they chunk `1:length(u)` into `Threads.nthreads()` static bands, each task landing a
# partial sum in a fixed-size buffer summed serially at the end. This tests the new
# `_threaded_dot`/`_threaded_dot_masked` kernels directly, plus the `_dot`/`_dot_masked`
# `CpuThreaded()` policy dispatch that now reaches them, against the untouched serial
# kernels as ground truth -- dense, single-mask (`BitVector`) and multi-marker
# (`MarkedIndicesUnion`) alike.
#
# S2.2 (gpena/Bramble.jl#301, #288) adds the `SeparableWeights` specializations of
# `_threaded_dot`/`_threaded_dot_masked` in `src/space/inner_product.jl`: without them a
# `SeparableWeights` weight under `CpuThreaded` fell through to the dense kernels above,
# reading the weight through its linear `Int` `getindex` (an `O(D)` divrem per point) instead
# of the serial kernel's line-walk. The final testset here checks those specializations,
# dense/`BitVector`/`MarkedIndicesUnion` alike, against the untouched serial `SeparableWeights`
# kernels across 1D/2D/3D non-uniform grids.
using Test
using Bramble
using Bramble: _dot, _dot_masked, _threaded_dot, _threaded_dot_masked, MarkedIndicesUnion, CpuThreaded, weights,
               SeparableWeights
using ..TestUtils: WITH_3D_TESTS

@testset "CpuThreaded _dot/_dot_masked" begin
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

    @testset "Two-marker union, length $n" for n in (
        0, 1, max(1, nthreads - 1), 37, 5000
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
    # The :boundary marker, and a two-marker union.
    @testset "Sparse mesh markers match serial" begin
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

@testset "SeparableWeights, CpuThreaded (#288)" begin
    # S2.2: `_dot`/`_dot_masked(CpuThreaded(), u, w, v[, mask])` must reach the
    # `SeparableWeights` specialization of `_threaded_dot` and the `SeparableWeights` term of
    # `_threaded_dot_masked` (`src/space/inner_product.jl`), not the dense ones -- which would read `w`
    # through its linear `Int` `getindex` (an `O(D)` divrem per point, gpena/Bramble.jl#288)
    # instead of the line-walk the serial `SeparableWeights` kernel uses. Checked against the
    # untouched serial kernels at rtol=1e-12, for dense, `BitVector`-masked and two-marker
    # `MarkedIndicesUnion`-masked sums, across D=1/2/3 non-uniform grids -- including one grid
    # per dimension whose *last* axis is shorter than `Threads.nthreads()`, the axis the
    # threaded dense kernel bands (`_last_axis_chunks`).
    nthreads = Threads.nthreads()

    _D1 = domain(interval(0.0, 1.0))
    _D2 = domain(interval(0.0, 1.0) × interval(0.0, 2.0))
    _D3 = domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0))
    _nonuniform_mesh(D, dims) = mesh(
        D == 1 ? _D1 : D == 2 ? _D2 : _D3, dims, ntuple(_ -> false, D))

    _separable_weights(D, dims) = weights(gridspace(_nonuniform_mesh(D, dims)), Val(()))

    grids = (
        (1, (37,)),
        (1, (max(1, nthreads - 1),)),  # the only axis shorter than nthreads
        (2, (23, 17)),
        (2, (23, max(1, nthreads - 1))),  # last axis shorter than nthreads
        # The 3D grids are built from `slow` on; the second has its last axis shorter than nthreads.
        (WITH_3D_TESTS ? ((3, (9, 7, 6)), (3, (9, 7, max(1, nthreads - 1)))) : ())...
    )

    # The values alone cannot tell: the dense kernels read `w` through its linear
    # `getindex` and return the same sum. So each call is also checked to dispatch to a
    # method whose weight slot is a `SeparableWeights` (read through the type name, since
    # the dense specialisation's `SeparableWeights{D, <:Any, VT}` leaves `VT` unbounded and
    # is not `<: SeparableWeights`), and to the same method a call on any `AbstractVector`
    # `u`/`v`, any `SeparableWeights` and any mask of kind `M` reaches, so that a method
    # narrowed on any argument and sending these weights to the dense kernel fails too. Such
    # a method can make the second `which` ambiguous, and `which` throws on that.
    function _reaches_separable(f, args, M...)
        m = which(f, typeof(args))
        slot = Base.typename(Base.unwrap_unionall(m.sig).parameters[3])
        any_weights = Tuple{AbstractVector, SeparableWeights, AbstractVector, M...}
        m_any = try
            which(f, any_weights)
        catch e
            e isa ErrorException || rethrow()
            nothing
        end
        return slot === Base.typename(SeparableWeights) && m === m_any
    end

    # The masked reduction is one driver and one band for every weight (`_threaded_dot_masked`,
    # `_dot_masked_band`), which read `w` through `_masked_cart(w)`: `nothing` for a vector
    # and the `CartesianIndices` of a `SeparableWeights`. So the masked calls are checked to
    # reach the term that reads `w[cart[i]]`, whose `cart` slot is a `CartesianIndices`
    # (through the type name, as `cart` is a `UnionAll` there), while a plain vector reaches
    # the `Nothing` term, the linear `getindex` read.
    function _masked_term_slot(u, w, v)
        cart = Bramble._masked_cart(w)
        m = which(Bramble._masked_term,
            Tuple{eltype(u), typeof(u), typeof(w), typeof(v), Int, Type{eltype(u)}, typeof(cart)})
        return Base.typename(Base.unwrap_unionall(m.sig).parameters[end])
    end
    _masked_reaches_separable(u, w, v) = Bramble._masked_cart(w) isa CartesianIndices &&
                                         _masked_term_slot(u, w, v) === Base.typename(CartesianIndices)

    @testset "D=$D, dims=$dims" for (D, dims) in grids
        w = _separable_weights(D, dims)
        @test w isa SeparableWeights
        n = length(w)
        u = Float64[0.3 + 0.11 * i for i in 1:n]
        v = Float64[-0.6 + 0.07 * i for i in 1:n]

        @test _reaches_separable(_threaded_dot, (u, w, v))
        expected = _dot(u, w, v)
        got = _threaded_dot(u, w, v)
        @test isapprox(got, expected; rtol = 1.0e-12, atol = 1.0e-12)
        @test isapprox(_dot(CpuThreaded(), u, w, v), got; rtol = 1.0e-12, atol = 1.0e-12)

        mask = falses(n)
        n > 0 && (mask[1:2:n] .= true) # every other index set, exercises both branches
        @test _masked_reaches_separable(u, w, v)
        expected_m = _dot_masked(u, w, v, mask)
        got_m = _threaded_dot_masked(u, w, v, mask)
        @test isapprox(got_m, expected_m; rtol = 1.0e-12, atol = 1.0e-12)
        @test isapprox(
            _dot_masked(CpuThreaded(), u, w, v, mask), got_m; rtol = 1.0e-12, atol = 1.0e-12
        )

        mask1 = falses(n)
        mask2 = falses(n)
        n > 0 && (mask1[1:3:n] .= true) # disjoint-ish, overlapping on some indices
        n > 0 && (mask2[2:3:n] .= true)
        union_mask = MarkedIndicesUnion((mask1, mask2))
        bit_union = mask1 .| mask2
        @test _masked_reaches_separable(u, w, v)
        expected_u = _dot_masked(u, w, v, bit_union)
        got_u = _threaded_dot_masked(u, w, v, union_mask)
        @test isapprox(got_u, expected_u; rtol = 1.0e-12, atol = 1.0e-12)
        @test isapprox(
            _dot_masked(CpuThreaded(), u, w, v, union_mask), got_u;
            rtol = 1.0e-12, atol = 1.0e-12
        )
    end

    @testset "Plain weights read linearly" begin
        # The negative control of `_masked_reaches_separable`: a plain vector weight has no
        # `CartesianIndices` and reaches the `Nothing` term.
        u = Float64[0.3 + 0.11 * i for i in 1:37]
        w = Float64[0.5 + 0.01 * i for i in 1:37]
        v = Float64[-0.6 + 0.07 * i for i in 1:37]
        @test Bramble._masked_cart(w) === nothing
        @test !_masked_reaches_separable(u, w, v)
        @test _masked_term_slot(u, w, v) === Base.typename(Nothing)
    end
end

@testset "Complex u conjugated (#644)" begin
    # `_threaded_dot`/`_threaded_dot_masked` conjugate `u` as the serial kernels do, dense
    # and `SeparableWeights` alike. Checked against an explicit `conj(u) v w` sum, not only
    # against the serial kernel, which an unconjugated pair of kernels would also match.
    _cu(n) = ComplexF64[complex(0.3 + 0.11 * i, 0.9 - 0.13 * i) for i in 1:n]
    _cv(n) = ComplexF64[complex(-0.6 + 0.07 * i, 0.2 + 0.05 * i) for i in 1:n]
    _oracle(u, v, w, idx) = sum(conj(u[i]) * v[i] * w[i] for i in idx; init = 0.0im)
    _close(a, b) = isapprox(a, b; rtol = 1.0e-12, atol = 1.0e-12)

    @testset "Dense, length $n" for n in (0, 1, 37, 5000)
        u, v, w = _cu(n), _cv(n), Float64[1.0 + 0.05 * i for i in 1:n]
        mask = falses(n)
        n > 0 && (mask[1:2:n] .= true)
        mask2 = falses(n)
        n > 0 && (mask2[2:3:n] .= true)
        union_mask = MarkedIndicesUnion((mask, mask2))

        expected = _oracle(u, v, w, 1:n)
        @test _close(_dot(u, v, w), expected)
        @test _close(_threaded_dot(u, v, w), expected)
        expected_m = _oracle(u, v, w, findall(mask))
        @test _close(_dot_masked(u, v, w, mask), expected_m)
        @test _close(_threaded_dot_masked(u, v, w, mask), expected_m)
        expected_u = _oracle(u, v, w, findall(mask .| mask2))
        @test _close(_dot_masked(u, v, w, union_mask), expected_u)
        @test _close(_threaded_dot_masked(u, v, w, union_mask), expected_u)
    end

    @testset "SeparableWeights, $(D)D" for (D, dims) in ((1, (37,)), (2, (23, 17)))
        Ω = D == 1 ? domain(interval(0.0, 1.0)) :
            domain(interval(0.0, 1.0) × interval(0.0, 2.0))
        w = weights(gridspace(mesh(Ω, dims, ntuple(_ -> false, D))), Val(()))
        n = length(w)
        u, v, wd = _cu(n), _cv(n), collect(w)
        mask = falses(n)
        mask[1:2:n] .= true

        @test _close(_threaded_dot(u, w, v), _oracle(u, v, wd, 1:n))
        @test _close(_threaded_dot_masked(u, w, v, mask), _oracle(u, v, wd, findall(mask)))
    end
end

end # module
