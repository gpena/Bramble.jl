# test/ext/polyester_parity.jl: `CpuPolyester` sweeps over a descending index collection
# visit the same indices as `CpuSerial` (ext/BramblePolyesterExt.jl, `_ascending`).
#
# Gated like every other ext/*.jl file (test/runtests.jl only reaches this group under
# BRAMBLE_TEST_GROUP=ext or full). Standalone:
#
#   julia --project=test -t 2 -e 'using Bramble, Test; include("test/TestUtils.jl");
#     include("test/ext/polyester_parity.jl")'
#
# `@batch` splits its range assuming a positive step: on one thread it walks `9:-1:1`
# correctly, so these pass at any thread count and exercise the defect at 2 or more.
#
# A `CpuPolyester` reduction returns the same value nested in another `CpuPolyester` sweep
# as at top level, and `CpuThreaded`'s value (gpena/Bramble.jl#473). At one thread a nested
# and a top-level `@batch` both run as one chunk, so the nesting legs bite at 2 or more.
module TestPolyesterParity

using Test
using Bramble
using Bramble: CpuPolyester, CpuSerial, CpuThreaded, MarkedIndicesUnion, VectorElement,
               _sweep_for!, _sweep_scatter_for!, __prod, _dot, _dot_masked
using Polyester
using Random

const _PP_RANGES = (9:-1:1, 9:-2:1, 1:-1:2, UInt(9):-1:UInt(1), UInt(9):-2:UInt(1))

@testset "Polyester descending-range parity" begin
    @testset "_sweep_for! on $r" for r in _PP_RANGES
        c = [3.0]
        for k in (i -> c[1] * i, Base.Fix1(__prod, (collect(1.0:9.0),)))
            v, w = fill(-1.0, 9), fill(-1.0, 9)
            _sweep_for!(CpuPolyester(), v, r, k)
            _sweep_for!(CpuSerial(), w, r, k)
            @test v == w
            @test count(!=(-1.0), w) == length(r)
        end
    end

    # A `CartesianIndices` takes the slab paths, whose slabs each task walks in order.
    @testset "_sweep_for! on a descending axis" begin
        c = [3.0]
        ci = CartesianIndices((1:3, 3:-1:1))
        for k in (I -> c[1] * I[1] + I[2], I -> 3.0 * I[1] + I[2])
            v, w = fill(-1.0, 3, 3), fill(-1.0, 3, 3)
            _sweep_for!(CpuPolyester(), v, ci, k)
            _sweep_for!(CpuSerial(), w, ci, k)
            @test v == w
            @test !any(==(-1.0), w)
        end
    end

    # A closure over a `Vector` crosses `@batch` whole; the `CellAverage` kernel is split.
    @testset "_sweep_scatter_for! on $r" for r in _PP_RANGES
        c = [3.0]
        sp = gridspace(mesh(domain(interval(0.0, 1.0)), 9, true))
        split = Bramble._rule_scatter_kernel(
            Bramble.CellAverage(x -> (x^2, 2x), Val(2)), sp, Val(2))
        for g in (i -> (c[1] * i, 2.0 * i), split)
            m = (fill(-1.0, 9), fill(-1.0, 9))
            s = (fill(-1.0, 9), fill(-1.0, 9))
            _sweep_scatter_for!(CpuPolyester(), m, r, g)
            _sweep_scatter_for!(CpuSerial(), s, r, g)
            @test m == s
            @test count(!=(-1.0), s[1]) == length(r)
        end
    end

    # Index sets that are no range go through the per-index loop, as the serial sweep does.
    @testset "scatter over Vector and CartesianIndices" begin
        c = [3.0]
        g = i -> (c[1] * i, 2.0 * i)
        m, s = (fill(-1.0, 9), fill(-1.0, 9)), (fill(-1.0, 9), fill(-1.0, 9))
        _sweep_scatter_for!(CpuPolyester(), m, [3, 1, 7], g)
        _sweep_scatter_for!(CpuSerial(), s, [3, 1, 7], g)
        @test m == s
        @test count(!=(-1.0), s[1]) == 3
        h = I -> (c[1] * I[1] + I[2],)
        for ci in (CartesianIndices((3, 3)), CartesianIndices((1:3, 3:-1:1)))
            A, B = (fill(-1.0, 3, 3),), (fill(-1.0, 3, 3),)
            _sweep_scatter_for!(CpuPolyester(), A, ci, h)
            _sweep_scatter_for!(CpuSerial(), B, ci, h)
            @test A == B
            @test !any(==(-1.0), B[1])
        end
    end

    @testset "_ascending" begin
        E = Base.get_extension(Bramble, :BramblePolyesterExt)
        ci = CartesianIndices((1:3, 3:-1:1))
        a = E._ascending(ci)
        @test all(s -> s > 0, map(step, a.indices))
        @test sort(vec(collect(a))) == sort(vec(collect(ci)))
        @test isbits(a)
        @test E._ascending(9:-2:1) === 1:2:9
        @test E._ascending(1:-1:2) == 1:0
        @test isempty(E._ascending(1:-1:2))
        @test E._ascending(Base.OneTo(4)) === Base.OneTo(4)
        @test isbits(E._ascending(9:-1:1))
        v = [3, 1, 2]
        @test E._ascending(v) === v
    end
end

# The reductions under test, each a function of one grid function: innerₕ, normₕ and
# inner₊ₓ unmasked and masked by one marker (a `BitVector`) and two (a
# `MarkedIndicesUnion`). All of them reach a `SeparableWeights` hook.
const _RED_REDUCTIONS = (
    ("innerₕ", u -> innerₕ(u, u)),
    ("normₕ", u -> normₕ(u)),
    ("innerₕ :a", u -> innerₕ(u, u; markers = (:a,))),
    ("innerₕ :a :b", u -> innerₕ(u, u; markers = (:a, :b))),
    ("inner₊ₓ", u -> inner₊(u, u, 1)),
    ("inner₊ₓ :a :b", u -> inner₊(u, u, 1; markers = (:a, :b)))
)

_red_box(::Val{1}) = interval(0.0, 1.0)
_red_box(::Val{2}) = interval(0.0, 1.0) × interval(0.0, 2.0)
_red_box(::Val{3}) = interval(0.0, 1.0) × interval(0.0, 2.0) × interval(0.0, 1.0)
function _red_space(n::NTuple{D, Int}, uniform::Bool, policy) where {D}
    X = _red_box(Val(D))
    Ω = domain(X, markers(X, :a => x -> x[1] < 0.6, :b => x -> x[end] > 0.3))
    Random.seed!(473)
    return gridspace(mesh(Ω, n, uniform; backend = backend(policy = policy)))
end
_red_f(x) = x[1] + sum(x) / 3 + x[1] * x[end] / 7

# `r(u)` evaluated inside a CpuPolyester `Rₕ!` (every point of the sweep reduces again, on
# a busy worker), to compare with the top-level value bitwise, point by point.
_red_nested(W, r, u) = parent(Rₕ!(element(W), x -> r(u) * x[1]))
_red_expected(W, s) = parent(Rₕ!(element(W), x -> s * x[1]))

@noinline function _red_bytes(f::F, args...) where {F}
    f(args...)
    f(args...)
    return @allocated f(args...)
end

# No leg nests a CpuPolyester reduction in a CpuThreaded sweep: the two policies are never
# nested in one another (a `@batch` started from a `Threads.@threads` task waits for busy
# workers).
@testset "reductions nested = top level (#473)" verbose = true begin
    # (3, 2) has fewer last-axis blocks than threads at 3 or more threads only.
    grids = (((1001,), false), ((9, 7), true), ((129, 129), true), ((100, 77), false),
        ((13, 11, 7), false), ((3, 2), false))
    @testset "$name on $n" for (n, uniform) in grids, (name, r) in _RED_REDUCTIONS

        Wp = _red_space(n, uniform, CpuPolyester())
        up = Rₕ(Wp, _red_f)
        s = r(Rₕ(_red_space(n, uniform, CpuSerial()), _red_f))
        t = r(Rₕ(_red_space(n, uniform, CpuThreaded()), _red_f))
        p = r(up)
        @test _red_nested(Wp, r, up) == _red_expected(Wp, p)
        @test p == t
        @test isapprox(p, s; rtol = 1e-12, atol = 1e-14)
        Threads.nthreads() == 1 && @test p == s
    end

    # Tasks reducing at once, each holding its own partial sums, whatever the free workers.
    @testset "concurrent tasks" begin
        Wl = _red_space((100, 77), false, CpuPolyester())
        ul = Rₕ(Wl, _red_f)
        for (name, r) in _RED_REDUCTIONS
            q = r(ul)
            @test all(==(q), fetch.([Threads.@spawn r(ul) for _ in 1:(2 * Threads.nthreads())]))
        end
    end

    # The dense hooks, reached with a plain weight vector.
    @testset "dense weights" begin
        Wp = _red_space((100, 77), false, CpuPolyester())
        rng = Xoshiro(473)
        a, b, c = rand(rng, 7700), rand(rng, 7700), rand(rng, 7700)
        m1, m2 = rand(rng, 7700) .< 0.3, rand(rng, 7700) .< 0.2
        U = MarkedIndicesUnion((m1.chunks, m2.chunks), 7700)
        for g in (p -> _dot(p, a, b, c), p -> _dot_masked(p, a, b, c, m1),
            p -> _dot_masked(p, a, b, c, U))
            p = g(CpuPolyester())
            @test parent(Rₕ!(element(Wp), x -> g(CpuPolyester()) * x[1])) ==
                  _red_expected(Wp, p)
            @test p == g(CpuThreaded())
            @test isapprox(p, g(CpuSerial()); rtol = 1e-12)
            Threads.nthreads() == 1 && @test p == g(CpuSerial())
        end
        @test _red_bytes((x, y, z) -> _dot(CpuPolyester(), x, y, z), a, b, c) == 0
        @test _red_bytes((x, y, z, m) -> _dot_masked(CpuPolyester(), x, y, z, m), a, b, c, m1) == 0
        @test _red_bytes((x, y, z, m) -> _dot_masked(CpuPolyester(), x, y, z, m), a, b, c, U) == 0
    end

    # Strided storage: `@batch` turns a strided view into a strided `PtrArray` at top level
    # but not on a busy worker, so the hooks hand it over unconverted (`_Opaque`).
    @testset "strided views" begin
        n = (40, 31)
        N = prod(n)
        Wp = _red_space(n, false, CpuPolyester())
        Wt = _red_space(n, false, CpuThreaded())
        x = rand(Xoshiro(1), 2N)
        up = VectorElement(view(x, 1:2:(2N)), Wp)
        ut = VectorElement(view(x, 1:2:(2N)), Wt)
        for (name, r) in _RED_REDUCTIONS
            p = r(up)
            @test _red_nested(Wp, r, up) == _red_expected(Wp, p)
            @test p == r(ut)
        end
        rng = Xoshiro(473)
        a, b, c = rand(rng, 2N), rand(rng, N), rand(rng, 2N)
        m = rand(rng, N) .< 0.3
        va, vc = view(a, 1:2:(2N)), view(c, 2:2:(2N))
        for g in (p -> _dot(p, va, b, vc), p -> _dot_masked(p, va, b, vc, m))
            p = g(CpuPolyester())
            @test parent(Rₕ!(element(Wp), x -> g(CpuPolyester()) * x[1])) ==
                  _red_expected(Wp, p)
            @test p == g(CpuThreaded())
        end
    end

    # Partial sums are kept per element type: a Float32 grid reduces in Float32.
    @testset "Float32" begin
        X = interval(0.0f0, 1.0f0) × interval(0.0f0, 2.0f0)
        Random.seed!(473)
        Wp = gridspace(mesh(domain(X), (40, 31), false;
            backend = backend(Float32; policy = CpuPolyester())))
        up = Rₕ(Wp, _red_f)
        p = innerₕ(up, up)
        @test p isa Float32
        @test _red_nested(Wp, u -> innerₕ(u, u), up) == _red_expected(Wp, p)
        @test _red_bytes(u -> innerₕ(u, u), up) == 0
    end
end

end # module TestPolyesterParity
