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
               CrossWeighted, _difference_engine!, _sweep_for!
using ForwardDiff
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
