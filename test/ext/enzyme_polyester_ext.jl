module ExtEnzymePolyesterExtTests

# BrambleEnzymePolyesterExt: Enzyme rules for the `CpuPolyester` hooks
# (gpena/Bramble.jl#369).
# Enzyme is not in test/Project.toml: load it first, through a temporary environment
# (test/ext/enzyme_policy_cases.jl says how), with at least 2 threads. Without it the first
# testset still runs, since the transposed band kernels it checks are plain Julia in `src/`.
#
# Every derivative is checked against ForwardDiff through the `CpuSerial` path, on seeded
# non-uniform spacings. The hook-level cases reuse one output buffer within the function, so
# a rule that leaves the overwritten output's cotangent in place gives a wrong gradient.

using Test
using Bramble
using Bramble: CpuPolyester, CpuSerial, StarSpacings, Forward, Backward, Centered,
               CrossWeighted, _difference_band!, _difference_band_adjoint!,
               _difference_spacing_adjoint!, _difference_band_spacing_tangent!,
               _star_spacings_adjoint!, _difference_engine!, _sweep_for!
using ForwardDiff
using LinearAlgebra: dot
using Polyester
using Random
using ..TestUtils: _have

const RUN_ENZYME = _have(:Enzyme) && Threads.nthreads() >= 2
RUN_ENZYME && @eval using Enzyme
RUN_ENZYME && include(joinpath(@__DIR__, "enzyme_policy_cases.jl"))

const TOL = 1e-10
const STENCILS = ((Backward(), Val(:none)), (Backward(), Val(:vector)),
    (Forward(), Val(:none)), (Forward(), Val(:view)), (Forward(), Val(:star)),
    (Centered(), Val(:star)), (CrossWeighted(), Val(:vector)))

# The spacing a family passes the engine, built from the spacings `s` of one axis.
_spacing(::Val{:none}, s) = nothing
_spacing(::Val{:vector}, s) = s
_spacing(::Val{:view}, s) = view(s, 2:length(s))
_spacing(::Val{:star}, s) = StarSpacings(s)

# The cotangent of `s` from that of the spacing values `d_hv` (indexed as the spacing is).
_fold(::Val{:vector}, d_hv, s) = copy(d_hv)
_fold(::Val{:view}, d_hv, s) = [0.0; d_hv]
function _fold(::Val{:star}, d_hv, s)
    d_s = zeros(length(s))
    _star_spacings_adjoint!(d_s, d_hv, 1, 1)
    return d_s
end

function _apply(x, h, dims, dir, dim_val, nbands)
    out = zeros(promote_type(eltype(x), eltype(something(h, Float64[]))), prod(dims))
    for b in 1:nbands
        _difference_band!(out, x, h, dims, dir, dim_val, nbands, b)
    end
    return out
end

@testset "Transposed difference bands" begin
    rng = MersenneTwister(369)
    for dims in ((9,), (7, 6), (5, 4, 6)), D in 1:length(dims), (dir, kind) in STENCILS
        n, len = dims[D], prod(dims)
        s = 0.1 .+ rand(rng, n)
        h = _spacing(kind, s)
        x, w = randn(rng, len), randn(rng, len)
        # Two bands, and more bands than the last axis has points (empty bands).
        for nbands in (2, last(dims) + 2)
            d_in = zeros(len)
            for b in 1:nbands
                _difference_band_adjoint!(d_in, w, h, dims, dir, Val(D), nbands, b)
            end
            @test dot(w, _apply(x, h, dims, dir, Val(D), nbands)) ≈ dot(d_in, x) rtol=1e-12
        end
        h === nothing && continue
        # The spacing cotangent and tangent against ForwardDiff in the spacings.
        g(t) = dot(w, _apply(x, _spacing(kind, t), dims, dir, Val(D), 1))
        d_hv = zeros(length(h))
        for b in 1:3
            _difference_spacing_adjoint!(d_hv, x, w, h, dims, dir, Val(D), 3, b)
        end
        @test _fold(kind, d_hv, s) ≈ ForwardDiff.gradient(g, s) rtol=1e-12
        δ = randn(rng, n)
        tangent = zeros(len)
        for b in 1:3
            _difference_band_spacing_tangent!(tangent, x, h, _spacing(kind, δ), dims, dir,
                Val(D), 3, b)
        end
        ref = ForwardDiff.derivative(t -> _apply(x, _spacing(kind, s .+ t .* δ), dims, dir,
                Val(D), 1), 0.0)
        @test tangent ≈ ref rtol=1e-12
    end
end

# The relative distance between a gradient and its reference.
_relerr(g, ref) = maximum(abs, g .- ref) / max(maximum(abs, ref), floatmin(Float64))

# The worst relative error of Enzyme under `CpuPolyester` against ForwardDiff under
# `CpuSerial`, per mode: reverse, forward, and the two width-3 batch modes of
# enzyme_policy_cases.jl. `Enzyme.Forward`, since `Forward` is Bramble's stencil direction.
function _gradients(f, p; batch_modes = EnzymePolicyCases.BATCH_MODES)
    ref = ForwardDiff.gradient(q -> f(q, CpuSerial()), p)
    fp = q -> f(q, CpuPolyester())
    rev = Enzyme.gradient(set_runtime_activity(Enzyme.Reverse), Const(fp), p)[1]
    fwd = Enzyme.gradient(set_runtime_activity(Enzyme.Forward), Const(fp), p)[1]
    batch = map(batch_modes) do mode
        got, want = EnzymePolicyCases._batch(fp, p, ref, mode)
        return maximum(_relerr(g, w) for (g, w) in zip(got, want))
    end
    return (_relerr(rev, ref), _relerr(fwd, ref), batch...)
end

const DIMS = (11, 9)

# Seeded data for `_engine_case` on axis `D`: base spacings, two spacing directions, base
# field, a field direction and the weights of the second sum.
function _engine_data(D::Int)
    rng = MersenneTwister(1)
    n, len = DIMS[D], prod(DIMS)
    return (0.1 .+ rand(rng, n), randn(rng, n), randn(rng, n), randn(rng, len),
        randn(rng, len), randn(rng, len))
end

# Two differences into one reused buffer, the field and the spacings both from `p`: `p[1]`
# and `p[2]` move the spacings of axis `D`, `p[3]` and `p[4]` the field.
function _engine_case(p, policy, dir, kind, ::Val{D}, data) where {D}
    s0, s1, s2, x0, x1, c = data
    s = s0 .* (1 .+ 0.1 .* (p[1] .* s1 .+ p[2] .* s2))
    h = _spacing(kind, s)
    x = x0 .+ p[3] .* x1 .+ p[4]^2 .* c
    out = zeros(eltype(p), length(x0))
    _difference_engine!(policy, out, x, h, DIMS, dir, Val(D))
    first_sum = sum(abs2, out)
    _difference_engine!(policy, out, x .* x, h, DIMS, dir, Val(D))
    return first_sum + sum(out .* c)
end

# `v[i] = f(i)` into one reused vector, over a stepped range, an index vector and the whole
# vector. The last two closures read `v` itself, nonlinearly and times a parameter, so the
# reverse pass must see `v` as it was before each call.
function _for_case(p, policy)
    v = zeros(eltype(p), 23)
    _sweep_for!(policy, v, 1:2:23, i -> p[1] * sin(i) + p[2]^2 * i)
    first_sum = sum(abs2, v)
    _sweep_for!(policy, v, [5, 1, 9, 2, 17, 4], i -> p[3] * cos(i) * p[1])
    _sweep_for!(policy, v, 1:2:23, i -> p[1] * v[i]^2)
    _sweep_for!(policy, v, 1:23, i -> sin(v[i]) * v[i] + p[4] * i)
    return first_sum + sum(abs2, v)
end

# A user closure capturing an array and a `Float64`: one taken from the parameters, one
# constant, and the constant behind a `Ref`.
function _closure_case(p, policy, ::Val{:active})
    a = p[1]^2
    v = zeros(eltype(p), 17)
    _sweep_for!(policy, v, 1:17, i -> a * p[mod1(i, 4)] * i)
    return sum(abs2, v)
end

function _closure_case(p, policy, ::Val{:constant})
    a = 2.0
    v = zeros(eltype(p), 17)
    _sweep_for!(policy, v, 1:17, i -> a * p[mod1(i, 4)] * i)
    return sum(abs2, v)
end

function _closure_case(p, policy, ::Val{:ref})
    r = Ref(2.0)
    v = zeros(eltype(p), 17)
    _sweep_for!(policy, v, 1:17, i -> r[] * p[mod1(i, 4)] * i)
    return sum(abs2, v)
end

# A closure that reaches the vector `_sweep_for!` writes by another route than a capture: a
# `Dict`, the vector's `Memory`, a `Dict` holding a tuple, or a global `Ref`.
const _GV = Ref{Any}(nothing)
_reach_v(p) = [p[mod1(i, 4)] * i / 10 for i in 1:13]
function _reach_case(p, policy, ::Val{:dict})
    v = _reach_v(p)
    d = Dict(:v => v)
    _sweep_for!(policy, v, 1:13, i -> d[:v][i]^2 * p[1])
    return sum(abs2, v)
end
function _reach_case(p, policy, ::Val{:memory})
    v = _reach_v(p)
    m = v.ref.mem
    _sweep_for!(policy, v, 1:13, i -> m[i]^2)
    return sum(abs2, v)
end
function _reach_case(p, policy, ::Val{:dict_tuple})
    v = _reach_v(p)
    d = Dict(:k => (v,))
    _sweep_for!(policy, v, 1:13, i -> sin(d[:k][1][i]) * p[2])
    return sum(abs2, v)
end
function _reach_case(p, policy, ::Val{:global})
    v = _reach_v(p)
    _GV[] = v
    _sweep_for!(policy, v, 1:13, i -> (_GV[]::Vector{eltype(p)})[i]^2)
    return sum(abs2, v)
end

# A closure holding a 200_000-node linked list: copying or reducing its shadow must not
# recurse once per node.
mutable struct _Node
    val::Float64
    next::Union{_Node, Nothing}
end
function _chain(n::Int)
    head = _Node(1.0, nothing)
    node = head
    for _ in 2:n
        node.next = _Node(1.0, nothing)
        node = node.next
    end
    return head
end
function _list_case(p, policy, list)
    v = _reach_v(p)
    _sweep_for!(policy, v, 1:13, i -> list.val * p[mod1(i, 4)]^2)
    return sum(abs2, v)
end

@testset "Enzyme rules under CpuPolyester" begin
    if RUN_ENZYME
        @testset "Policy cases" begin
            modes = (EnzymePolicyCases.MODES..., EnzymePolicyCases.BATCH_MODES...)
            for hook in ("_batch_difference_engine!", "_batch_for!",
                "_batch_for_const_capture", "_batch_for_mixed_capture", "_batch_for_reads_v")
                for mode in modes
                    c = EnzymePolicyCases.case(hook)
                    r = EnzymePolicyCases.evaluate(c; policy = CpuPolyester(), mode)
                    @test r.status == "ok"
                end
            end
        end

        @testset "Difference engine, reused output" begin
            p = [0.3, -0.4, 0.7, 0.5]
            for (dir, kind) in STENCILS, D in 1:2

                data = _engine_data(D)
                case = (q, pol) -> _engine_case(q, pol, dir, kind, Val(D), data)
                @test all(<=(TOL), _gradients(case, p))
            end
        end

        @testset "Closure sweep reading its output" begin
            @test all(<=(TOL), _gradients(_for_case, [0.3, -0.4, 0.7, 0.5]))
        end

        # A closure with a `Float64` beside an array, constant or from the parameters,
        # failed Enzyme's calling convention by value; `_sweep_for!` now passes the kernel
        # behind a `Ref`, so every shape works in every mode.
        @testset "Closure capture shapes" begin
            p = [0.5, 0.7, 1.1, 1.3]
            for shape in (Val(:constant), Val(:active), Val(:ref))
                case = (q, pol) -> _closure_case(q, pol, shape)
                @test all(<=(TOL), _gradients(case, p))
            end
        end

        @testset "Closure reaching its output" begin
            p = [0.5, 0.7, 1.1, 1.3]
            for route in (Val(:dict), Val(:memory), Val(:dict_tuple))
                case = (q, pol) -> _reach_case(q, pol, route)
                @test all(<=(TOL), _gradients(case, p))
            end
            # A global reached only inside the closure: the per-slab derivative is its own
            # Enzyme compilation, which does not see the shadow the caller stored there.
            case = (q, pol) -> _reach_case(q, pol, Val(:global))
            @test_broken all(<=(TOL), _gradients(case, p))
            list = _chain(200_000)
            @test all(<=(TOL), _gradients((q, pol) -> _list_case(q, pol, list), p))
        end
    else
        @test_skip RUN_ENZYME
    end
end

end
