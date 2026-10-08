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
module TestPolyesterParity

using Test
using Bramble
using Bramble: CpuPolyester, CpuSerial, _sweep_for!, _sweep_scatter_for!, __prod
using Polyester

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

end # module TestPolyesterParity
