module ExtEnzymePolyesterExtTests

# BrambleEnzymePolyesterExt: Enzyme rules for the `CpuPolyester` hooks
# (gpena/Bramble.jl#369).
# Enzyme is not in test/Project.toml: load it first, through a temporary environment
# (test/ext/enzyme_policy_cases.jl says how), with at least 2 threads. The transposed band
# kernels these rules call are plain Julia in `src/`, checked in test/space/threaded_stencils.jl.
#
# Every derivative is checked against ForwardDiff through the `CpuSerial` path, on seeded
# non-uniform spacings. The hook-level cases reuse one output buffer within the function, so
# a rule that leaves the overwritten output's cotangent in place gives a wrong gradient.

using Test
using Bramble
using Bramble: CpuPolyester, CpuSerial, D₋ₓ, StarSpacings, Forward, Backward, Centered,
               CrossWeighted, _difference_engine!, _sweep_for!, _sweep_scatter_for!
using ForwardDiff
using LinearAlgebra: mul!
using Polyester
using Random
using Sparspak
import DifferentiationInterface as DI
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

# Complex sweeps: the closure writes complex values and a complex-weighted real part reads
# them, so the reverse rule must pull back with `conj` of the output's cotangent.
const _CC = randn(MersenneTwister(5), ComplexF64, 37)
const _WC = randn(MersenneTwister(6), ComplexF64, 37)
_cdot(w, v) = real(sum(conj(w[i]) * v[i] for i in eachindex(v)))
function _complex_for_case(p, policy)
    v = zeros(Complex{eltype(p)}, 37)
    _sweep_for!(policy, v, 1:37, i -> p[1] * _CC[i] + p[2]^2 * _CC[i]^2)
    return _cdot(_WC, v)
end
function _complex_scatter_case(p, policy)
    a = zeros(Complex{eltype(p)}, 37)
    b = zeros(Complex{eltype(p)}, 37)
    _sweep_scatter_for!(policy, (a, b), 1:37, i -> (p[1] * _CC[i], p[2]^2 * conj(_CC[i])))
    return _cdot(_WC, a) + _cdot(reverse(_WC), b)
end
# One real and one complex destination, in both orders: the slab's dot must be complex
# whichever comes first.
const _XB = randn(MersenneTwister(7), 37)
function _mixed_scatter_case(p, policy, ::Val{real_first}) where {real_first}
    a = zeros(eltype(p), 37)
    b = zeros(Complex{eltype(p)}, 37)
    g = i -> (p[1] * _XB[i], p[2]^2 * _CC[i] + p[1])
    if real_first
        _sweep_scatter_for!(policy, (a, b), 1:37, g)
    else
        _sweep_scatter_for!(policy, (b, a), 1:37, i -> reverse(g(i)))
    end
    return sum(abs2, a) + _cdot(_WC, b) + sum(abs2, b)
end

# The relative error of reverse Enzyme under `CpuPolyester` against ForwardDiff under
# `CpuSerial`, and whether the rule of `hook` ran.
function _reverse_error(f, p, hook)
    ref = ForwardDiff.gradient(q -> f(q, CpuSerial()), p)
    n0 = EnzymePolicyCases._rule_entries(hook)
    got = Enzyme.gradient(set_runtime_activity(Enzyme.Reverse),
        Const(q -> f(q, CpuPolyester())), p)[1]
    return _relerr(got, ref), EnzymePolicyCases._rule_entries(hook) > n0
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

# `Rₕ!` on a `Float32` space, its function capturing a parameter. The kernel holds the
# closure, the mesh's points and its `CartesianIndices`: behind the sweep's `Ref`, Enzyme
# lost the captured `Float32`'s derivative, in the slab thunk and, for the scatter and
# `CartesianIndices` sweeps, in the non-inlined method that built the `Ref`.
function _f32_space(policy)
    Bramble._seed_mesh1d_rng!(7)
    try
        Ωₕ = mesh(domain(interval(0.0f0, 1.0f0) × interval(0.0f0, 2.0f0)), (9, 7),
            (false, false); backend = backend(Float32; policy = policy))
        return gridspace(Ωₕ)
    finally
        Bramble._unseed_mesh1d_rng!()
    end
end
function _f32_restriction_case(p, policy, ::Val{:scalar})
    u = element(_f32_space(policy), eltype(p))
    c = p[1]
    Rₕ!(u, x -> c * x[1] + x[2])
    return sum(abs2, parent(u))
end
function _f32_restriction_case(p, policy, ::Val{:composite})
    W = _f32_space(policy)
    u = element(W × W, eltype(p))
    c = p[1]
    Rₕ!(u, x -> (c * x[1], c * x[2] + x[1]))
    return sum(abs2, parent(u))
end
# The same kernel over a `CartesianIndices`, through the axis sweep.
function _f32_restriction_case(p, policy, ::Val{:axis})
    T = eltype(p)
    v = zeros(T, 9, 7)
    c = p[1]
    x = (T.(_XB[1:9]), T.(_XB[11:17]))
    k = Bramble._RₕKernel(y -> c * y[1] + y[2], x, CartesianIndices(v))
    _sweep_for!(policy, v, CartesianIndices(v), k)
    return sum(abs2, v)
end

# A closure returning `Float64` into `Float32` destinations: the slab's dot added `Float64`
# terms to a `Float32` seed, a `Union` that stopped Enzyme (`!sret_union`). The `CpuSerial`
# reference keeps the parameters' type in its destinations.
function _wide_value_case(p, policy)
    E = policy isa CpuSerial ? eltype(p) : Float32
    a = zeros(E, 37)
    b = zeros(E, 37)
    _sweep_scatter_for!(policy, (a, b), 1:37, i -> (p[1] * _XB[i], p[2] * _XB[i]))
    return sum(abs2, a) + sum(b)
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

# A band kernel taking an `Active` value `c` by value, beside arrays and constants as the
# ghost operators' facet band takes its `_FacetFix` tuple: a scalar weight, an isbits tuple
# of floats, or a tuple of structs with integer and `Bool` fields beside the floats. The
# rule's return type was `Union{Nothing, T}` in every slot, which Enzyme rejects; the struct
# tuple was the one that hit it here, at width 2 (gpena/Bramble.jl#1127).
struct _Fix{T}
    i::Int
    on::Bool
    h::NTuple{2, T}
end
_coef(w::Number, i::Int) = (w, w)
_coef(c::NTuple{2, Number}, i::Int) = c
function _coef(c::NTuple{2, _Fix}, i::Int)
    f = c[mod1(i, 2)]
    return (f.on ? f.h[1] : f.h[2] + f.i, f.h[2])
end
function _wide_band!(out, x, y, ax, dims::Tuple, c, n::Int, b::Int)
    for i in Bramble._band_range(ax, n, b)
        s, t = _coef(c, i)
        out[i] = s * x[i]^2 + t * y[i] * dims[1] / dims[2]
    end
    return nothing
end

_active_arg(p, ::Val{:weight}) = p[3] * p[4]
_active_arg(p, ::Val{:tuple}) = (p[3]^2, p[3] * p[4])
function _active_arg(p, ::Val{:fix})
    return (_Fix(1, true, (p[3]^2, p[4])), _Fix(2, false, (p[2], p[3] * p[4])))
end

# `_run_bands!` with the fields and the `Active` value all from `p`, on 37 points so the
# bands are uneven.
function _bands_case(p, policy, kind::Val)
    x = [p[1] + 0.1k * p[2] for k in 1:37]
    y = [p[3] - 0.05k * p[1] for k in 1:37]
    out = zeros(eltype(p), 37)
    Bramble._run_bands!(policy, _wide_band!, out, x, y, Base.OneTo(37), (37, 1),
        _active_arg(p, kind))
    return sum(abs2, out)
end

# A band kernel whose derivative reads `out`: `out[i] *= x[i]`. At width 2 the reverse
# replays each lane in turn, so lane 2 must replay on `out` as it was before the call, not
# on lane 1's writes.
function _scale_band!(out, x, n::Int, b::Int)
    for i in Bramble._band_range(Base.OneTo(length(out)), n, b)
        out[i] = out[i] * x[i]
    end
    return nothing
end
function _scale_case(p, policy)
    x = [p[1] + 0.1k * p[2] for k in 1:29]
    out = [1 + k * p[3] / 29 for k in 1:29]
    Bramble._run_bands!(policy, _scale_band!, out, x)
    return sum(abs2, out)
end

# A band kernel that writes two arguments besides `out`: `y` in place and `z` over values
# taken from the parameters, as `jacobian!` writes its coefficient values. Reverse gave each
# task a zeroed copy of their shadows, so their cotangents never reached `x` and `z`'s
# stayed on its old values. `z` is a view, which takes the generic merge, not the array one.
function _write_band!(out, y, z, x, n::Int, b::Int)
    for i in Bramble._band_range(Base.OneTo(length(out)), n, b)
        y[i] *= x[i]
        z[i] = x[i]^2
        out[i] = sin(x[i]) * y[i]
    end
    return nothing
end
function _write_case(p, policy)
    x = [p[1] + 0.1k * p[2] for k in 1:37]
    y = [1 + k * p[3] / 37 for k in 1:37]
    zb = fill(p[2] * p[3], 39)
    out = zeros(eltype(p), 37)
    Bramble._run_bands!(policy, _write_band!, out, y, view(zb, 2:38), x)
    return sum(abs2, out) + sum(abs2, y) + sum(abs2, zb)
end

# A `Float32` beside an array in `_run_bands!`'s kernel, and in a closure argument. Reverse
# gave the `Float32` a zero derivative: in the kernel, when the band thunks took it behind
# its `Ref`, and in the closure, through a non-inlined `_run_bands!`. `:wrapped` passes the
# closure through a caller that does not inline: batched reverse stopped Enzyme
# (`n_shadow_roots`), and a `Float32` capture still gets a zero derivative, since Enzyme
# drops it from a struct holding a `Float32`, padding and an array once it sits in a `Ref`
# that a callee reads.
struct _F32Band{T, X}
    c::T
    x::X
end
function (k::_F32Band)(out, y, n::Int, b::Int)
    for i in Bramble._band_range(Base.OneTo(37), n, b)
        out[i] = k.c * y[i] + k.x[i]
    end
    return nothing
end
function _apply_band!(out, x, g, n::Int, b::Int)
    for i in Bramble._band_range(Base.OneTo(37), n, b)
        out[i] = g(x[i])
    end
    return nothing
end
function _f32_bands_case(p, policy, ::Val{:kernel})
    T = eltype(p)
    y = [p[2] + T(0.1) * k for k in 1:37]
    out = zeros(T, 37)
    Bramble._run_bands!(policy, _F32Band(p[1], T.(_XB)), out, y)
    return sum(abs2, out)
end
function _f32_bands_case(p, policy, ::Val{:closure})
    T = eltype(p)
    c = p[1]
    w = T.(_XB)
    x = [p[2] + T(0.1) * k for k in 1:37]
    out = zeros(T, 37)
    Bramble._run_bands!(policy, _apply_band!, out, x, y -> c * y + w[1])
    return sum(abs2, out)
end
@noinline function _wrapped_bands!(policy, out, x, g)
    return Bramble._run_bands!(policy, _apply_band!, out, x, g)
end
function _f32_bands_case(p, policy, ::Val{:wrapped})
    T = eltype(p)
    c = p[1]
    w = T.(_XB)
    x = [p[2] + T(0.1) * k for k in 1:37]
    out = zeros(T, 37)
    _wrapped_bands!(policy, out, x, y -> c * y + w[1])
    return sum(abs2, out)
end

# A caller's own `Ref` of a closure, which `_run_bands!` must pass as it is, and an isbits
# argument holding a `Float32` beside an `Int`. Reverse gives the latter's `Float32` a zero
# derivative: Enzyme drops it across the padding after it.
function _ref_band!(out, x, r, n::Int, b::Int)
    for i in Bramble._band_range(Base.OneTo(37), n, b)
        out[i] = r[](x[i])
    end
    return nothing
end
function _f32_bands_case(p, policy, ::Val{:ref})
    T = eltype(p)
    c = p[1]
    w = T.(_XB)
    x = [p[2] + T(0.1) * k for k in 1:37]
    out = zeros(T, 37)
    Bramble._run_bands!(policy, _ref_band!, out, x, Ref(y -> c * y + w[1]))
    return sum(abs2, out)
end
struct _PadArg{T}
    c::T
    m::Int
end
function _pad_band!(out, x, a, n::Int, b::Int)
    for i in Bramble._band_range(Base.OneTo(37), n, b)
        out[i] = a.c * a.m * x[i]
    end
    return nothing
end
function _f32_bands_case(p, policy, ::Val{:isbits})
    T = eltype(p)
    x = [p[2] + T(0.1) * k for k in 1:37]
    out = zeros(T, 37)
    Bramble._run_bands!(policy, _pad_band!, out, x, _PadArg(p[1], 3))
    return sum(abs2, out)
end

# The relative errors of reverse at width 1, reverse at width 2 (`BatchDuplicated`, lane
# `k` seeded `cots[k]`) and forward under `CpuPolyester`, against ForwardDiff under
# `CpuSerial`.
function _bands_errors(case, p)
    ref = ForwardDiff.gradient(q -> case(q, CpuSerial()), p)
    fp = q -> case(q, CpuPolyester())
    rev = Enzyme.gradient(set_runtime_activity(Enzyme.Reverse), Const(fp), p)[1]
    g!(y, q) = (y[1] = fp(q); nothing)
    cots = (1.0, -0.5)
    dxs = (zero(p), zero(p))
    Enzyme.autodiff(set_runtime_activity(Enzyme.Reverse), Const(g!), Const,
        BatchDuplicated(zeros(1), ([cots[1]], [cots[2]])), BatchDuplicated(copy(p), dxs))
    fwd = Enzyme.gradient(set_runtime_activity(Enzyme.Forward), Const(fp), p)[1]
    return (_relerr(rev, ref), _relerr(dxs[1], cots[1] .* ref),
        _relerr(dxs[2], cots[2] .* ref), _relerr(fwd, ref))
end

# Data a hook reads behind a fresh `Ref`, zeroed after the call, which `overwritten` misses.
# Each read is nonlinear, so the reverse pass gives a wrong gradient unless the rule kept
# the data from before the primal: a coefficient of an assembled form (`κ₁ * (κ₂ * U)`),
# a closure's capture in a sweep, and a band kernel's capture or argument.
const _CS = 1 .+ rand(MersenneTwister(7), 37)
function _overwritten_case(p, policy, ::Val{:assembly})
    W = gridspace(EnzymePolicyCases._seeded_mesh((21, 17), policy))
    n = ndofs(W)
    κ₁ = element(W, eltype(p))
    κ₂ = element(W, eltype(p))
    parent(κ₁) .= 1e4 .* (1 .+ p[1] .* sin.(1:n) .^ 2)
    parent(κ₂) .= 1 .+ p[2] .* cos.(1:n) .^ 2
    A = assemble(form(W, W, (U, V) -> innerₕ(κ₁ * (κ₂ * U), V) + inner₊(∇ₕ(U), ∇ₕ(V))))
    parent(κ₁) .= 0
    return sum(abs2, A * cos.(1:n))
end
function _overwritten_case(p, policy, ::Val{:sweep})
    a = copy(p)
    v = zeros(eltype(p), 37)
    _sweep_for!(policy, v, 1:37, i -> a[1]^2 * _CS[i] + a[2] * _CS[i]^2)
    s = sum(abs2, v)
    fill!(a, 0)
    return s
end
function _overwritten_case(p, policy, ::Val{:bands})
    a = copy(p)
    x = [p[1] + 0.1k * p[2] for k in 1:37]
    out = zeros(eltype(p), 37)
    Bramble._run_bands!(policy, (out, x, n, b) -> begin
            for i in Bramble._band_range(Base.OneTo(37), n, b)
                out[i] = a[1]^2 * x[i] + a[2] * x[i]^2
            end
        end, out, x)
    s = sum(abs2, out)
    fill!(a, 0)
    return s
end
# A band kernel's argument rather than its capture: a closure, and a `StarSpacings` whose
# spacings are overwritten.
_closure_band!(out, x, g, n::Int, b::Int) = foreach(
    i -> out[i] = g(x[i]), Bramble._band_range(Base.OneTo(37), n, b))
_star_band!(out, x, h, n::Int, b::Int) = foreach(
    i -> out[i] = x[i] / h[i], Bramble._band_range(Base.OneTo(36), n, b))
function _overwritten_case(p, policy, ::Val{:band_closure})
    a = copy(p)
    x = [p[1] + 0.1k * p[2] for k in 1:37]
    out = zeros(eltype(p), 37)
    Bramble._run_bands!(policy, _closure_band!, out, x, y -> a[1]^2 * y + a[2] * y^2)
    s = sum(abs2, out)
    fill!(a, 0)
    return s
end
function _overwritten_case(p, policy, ::Val{:band_star})
    h = [1 + p[1]^2 * k / 37 + p[2] * sin(k) / 9 for k in 1:37]
    x = [p[1] + 0.1k * p[2] for k in 1:37]
    out = zeros(eltype(p), 36)
    Bramble._run_bands!(policy, _star_band!, out, x, StarSpacings(h))
    s = sum(abs2, out)
    fill!(h, 1)
    return s
end
# Outputs that are not `Array`s: a matrix-free `mul!` into a column view and into a column
# view of a reshape, and a linear assembly into a column view. The reverse pairs the output's
# shadow with a scratch, which must have its type; `similar` does not give it. `Y[:, 2]`
# enters the loss with `p[1]`, so a reverse that writes outside the view is caught.
function _view_output_case(p, policy, kind::Val)
    W = gridspace(EnzymePolicyCases._seeded_mesh((21, 17), policy))
    n = ndofs(W)
    κ = element(W, eltype(p))
    parent(κ) .= 1 .+ p[1] .* sin.(1:n) .^ 2
    Y = _view_matrix(kind, eltype(p), n)
    Y[:, 2] .= p[1] .* cos.(1:n)
    _view_apply!(kind, view(Y, :, 1), W, κ, p[2] .* cos.(1:n))
    return sum(abs2, Y) + sum(Y[:, 2] .* Y[:, 1])
end
_view_matrix(::Val, T, n) = zeros(T, n, 2)
_view_matrix(::Val{:mul_reshape}, T, n) = reshape(view(zeros(T, 2n + 1), 1:2n), n, 2)
_view_apply!(::Val{:assemble}, y, W, κ, x) = assemble!(y, form(W, v -> innerₕ(κ, v)))
_view_apply!(::Val, y, W, κ, x) = mul!(y,
    matrix_free_operator(form(W, W, (U, V) -> inner₊(Mₕ(κ) * ∇ₕ(U), ∇ₕ(V)))), x)
const _VIEW_OUTPUTS = ((Val(:mul), "_batch_mf_bands!"),
    (Val(:mul_reshape), "_batch_mf_bands!"), (Val(:assemble), "_batch_linear_band_sweep!"))

const _OVERWRITTEN_HOOKS = ((Val(:assembly), "_batch_bilinear_band_replay!"),
    (Val(:sweep), "_batch_for!"), (Val(:bands), "_batch_run_bands!"),
    (Val(:band_closure), "_batch_run_bands!"), (Val(:band_star), "_batch_run_bands!"))

# Workload 1: a functional of the nodal field of `κₕ` (its derivative, its norm and an
# assembled load) on the seeded 41 x 41 mesh of `EnzymePolicyCases`, under `policy`.
function _dense_functional(policy)
    W = gridspace(EnzymePolicyCases._mesh(policy))
    function f(x)
        κₕ = element(W, eltype(x))
        κₕ .= x
        l = form(W, v -> innerₕ(κₕ, v))
        return sum(abs2, parent(D₋ₓ(κₕ))) + normₕ(κₕ)^2 + sum(abs2, assemble(l))
    end
    return f, [1.0 + 0.5 * sin(0.37k)^2 for k in 1:ndofs(W)]
end

# Workload 2: the misfit of `pde_solve` of the diffusion system against data, in the 16
# nodal values `idx` of the coefficient. `fd` is the same misfit solved by Sparspak, the oracle's
# route, since ForwardDiff does not go through the default solver.
function _pde_inverse(policy)
    W = gridspace(EnzymePolicyCases._mesh(policy))
    n = ndofs(W)
    gₕ = element(W)
    avgₕ!(gₕ, x -> exp(x[1] * x[2]))
    F = assemble(form(W, v -> innerₕ(gₕ, v)); dirichlet = :boundary => x -> 0.0)
    diffusion = κₕ -> form(W, W, (U, V) -> inner₊(Mₕ(κₕ) * ∇ₕ(U), ∇ₕ(V)))
    function sys(κ)
        κₕ = element(W, eltype(κ))
        κₕ .= κ
        return assemble(diffusion(κₕ); dirichlet = :boundary)
    end
    κ0 = [1.0 + 0.5 * sin(0.37k)^2 for k in 1:n]
    interior = [i + (j - 1) * 41 for j in 2:40 for i in 2:40]
    idx = interior[round.(Int, range(1, 39^2; length = 16))]
    kap = p -> (κ = zeros(eltype(p), n); copyto!(κ, κ0); κ[idx] .= p; κ)
    d = pde_solve(sys(κ0 .* [1 + 0.3 * cos(0.11k) for k in 1:n]), F)
    f = p -> sum(abs2, pde_solve(sys(kap(p)), F) .- d)
    fd = p -> (A = sys(kap(p)); sum(abs2, pde_solve(A, convert(Vector{eltype(A)}, F);
        solver = :sparspak) .- d))
    return f, fd, κ0[idx]
end

# A DI `AutoEnzyme` in `mode` with runtime activity, the function annotated `ann`.
function _auto(mode, ann)
    return DI.AutoEnzyme(; mode = Enzyme.set_runtime_activity(mode),
        function_annotation = ann)
end

# The relative errors of workload 1 against ForwardDiff under `CpuSerial`, in three groups:
# DI's reverse and forward gradients, a closure writing its input into a captured buffer
# (`Duplicated`), and the plain `AutoEnzyme` gradients, which may throw (a throw leaves the
# group empty: a gradient that comes back must be right).
function _dense_errors()
    f, x = _dense_functional(CpuPolyester())
    f0, _ = _dense_functional(CpuSerial())
    ref = ForwardDiff.gradient(f0, x)
    modes = (Enzyme.Reverse, Enzyme.Forward)
    di = [DI.gradient(f, _auto(m, Enzyme.Const), x) for m in modes]
    buf = zeros(length(x))
    fc = y -> (buf .= y; f(buf))
    dup = [DI.gradient(fc, _auto(m, Enzyme.Duplicated), x) for m in modes]
    plain = Float64[]
    for m in modes
        g = try
            DI.gradient(f, DI.AutoEnzyme(; mode = m), x)
        catch err
            err isa InterruptException && rethrow()
            nothing
        end
        g === nothing || push!(plain, _relerr(g, ref))
    end
    return (di = [_relerr(g, ref) for g in di], dup = [_relerr(g, ref) for g in dup],
        plain = plain)
end

# The relative error of workload 2's reverse gradient against the Sparspak misfit's.
function _pde_errors()
    f, _, x = _pde_inverse(CpuPolyester())
    _, fd, _ = _pde_inverse(CpuSerial())
    ref = ForwardDiff.gradient(fd, x)
    return _relerr(DI.gradient(f, _auto(Enzyme.Reverse, Enzyme.Const), x), ref)
end

@testset "Enzyme rules under CpuPolyester" begin
    if RUN_ENZYME
        # Each evaluation leaves its garbage in the process footprint until a collection, so the
        # testsets that evaluate many cases collect after each one (25 Policy evaluations peak
        # at 3.6 GB without it and 2.0 GB with it; julia_slot caps a process at 10 GB).
        @testset "Policy cases" begin
            modes = (EnzymePolicyCases.MODES..., EnzymePolicyCases.BATCH_MODES...)
            for hook in ("_batch_difference_engine!", "_batch_for!",
                "_batch_for_const_capture", "_batch_for_mixed_capture", "_batch_for_reads_v")
                for mode in modes
                    c = EnzymePolicyCases.case(hook)
                    r = EnzymePolicyCases.evaluate(c; policy = CpuPolyester(), mode)
                    @test r.status == "ok"
                    GC.gc(true)
                end
            end
        end

        @testset "Workload: dense functional" begin
            e = _dense_errors()
            @test all(<=(TOL), e.di)
            @test all(<=(TOL), e.dup)
            @test all(<=(TOL), e.plain)
            GC.gc(true)
        end

        @testset "Workload: pde_solve inverse problem" begin
            @test _pde_errors() <= 1e-8
            GC.gc(true)
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

        @testset "Complex closure sweep, reverse" begin
            p = [0.6, -0.9]
            for (case, hook) in ((_complex_for_case, "_batch_for!"),
                (_complex_scatter_case, "_batch_scatter_for!"),
                ((q, pol) -> _mixed_scatter_case(q, pol, Val(true)), "_batch_scatter_for!"),
                ((q, pol) -> _mixed_scatter_case(q, pol, Val(false)), "_batch_scatter_for!"))
                err, ran = _reverse_error(case, p, hook)
                @test ran
                @test err <= TOL
            end
        end

        @testset "Float32 Rₕ! kernel, reverse" begin
            for (kind, hook) in ((Val(:scalar), "_batch_for!"),
                (Val(:composite), "_batch_scatter_for!"), (Val(:axis), "_batch_axis_for!"))
                case = (q, pol) -> _f32_restriction_case(q, pol, kind)
                err, ran = _reverse_error(case, Float32[0.7], hook)
                @test ran
                @test err <= 1e-4
            end
        end

        @testset "Float64 values, Float32 output" begin
            err, ran = _reverse_error(_wide_value_case, [0.6, -0.9], "_batch_scatter_for!")
            @test ran
            @test err <= 1e-6
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

        @testset "Band engine, Active by-value argument" begin
            p = [0.3, -0.4, 0.7, 0.5]
            for kind in (Val(:weight), Val(:tuple), Val(:fix))
                n0 = EnzymePolicyCases._rule_entries("_batch_run_bands!")
                @test all(<=(TOL), _bands_errors((q, pol) -> _bands_case(q, pol, kind), p))
                @test EnzymePolicyCases._rule_entries("_batch_run_bands!") > n0
            end
        end

        @testset "Band engine reading its output" begin
            n0 = EnzymePolicyCases._rule_entries("_batch_run_bands!")
            @test all(<=(TOL), _bands_errors(_scale_case, [0.8, 0.3, 1.2]))
            @test EnzymePolicyCases._rule_entries("_batch_run_bands!") > n0
        end

        @testset "Band engine writing its arguments" begin
            n0 = EnzymePolicyCases._rule_entries("_batch_run_bands!")
            @test all(<=(TOL), _bands_errors(_write_case, [0.8, 0.3, 1.2]))
            @test EnzymePolicyCases._rule_entries("_batch_run_bands!") > n0
        end

        @testset "Float32 band kernel and closure" begin
            p = Float32[0.7, -0.4]
            closure = (q, pol) -> _f32_bands_case(q, pol, Val(:closure))
            err, ran = _reverse_error(closure, p, "_batch_run_bands!")
            @test ran
            @test err <= 1e-4
            kernel = (q, pol) -> _f32_bands_case(q, pol, Val(:kernel))
            n0 = EnzymePolicyCases._rule_entries("_batch_run_bands!")
            @test all(<=(1e-4), _bands_errors(kernel, p))
            @test EnzymePolicyCases._rule_entries("_batch_run_bands!") > n0
            wrapped = (q, pol) -> _f32_bands_case(q, pol, Val(:wrapped))
            @test all(<=(TOL), _bands_errors(wrapped, [0.7, -0.4]))
            err, ran = _reverse_error(wrapped, p, "_batch_run_bands!")
            @test ran
            @test_broken err <= 1e-4
            byref = (q, pol) -> _f32_bands_case(q, pol, Val(:ref))
            @test byref(p, CpuPolyester()) ≈ byref(p, CpuSerial())
            for q in (p, [0.7, -0.4])
                err, ran = _reverse_error(byref, q, "_batch_run_bands!")
                @test ran
                @test err <= (eltype(q) === Float32 ? 1e-4 : TOL)
            end
            padded = (q, pol) -> _f32_bands_case(q, pol, Val(:isbits))
            err, ran = _reverse_error(padded, p, "_batch_run_bands!")
            @test ran
            @test_broken err <= 1e-4
        end

        @testset "Data overwritten after the call" begin
            p = [0.6, -0.9]
            for (kind, hook) in _OVERWRITTEN_HOOKS
                case = (q, pol) -> _overwritten_case(q, pol, kind)
                err, ran = _reverse_error(case, p, hook)
                @test ran
                @test err <= TOL
            end
            GC.gc(true)
        end

        @testset "Output into a view" begin
            for (kind, hook) in _VIEW_OUTPUTS
                case = (q, pol) -> _view_output_case(q, pol, kind)
                err, ran = _reverse_error(case, [0.3, 0.8], hook)
                @test ran
                @test err <= TOL
                GC.gc(true)
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
