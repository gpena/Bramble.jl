module ExtEnzymeThreadedExtTests

# BrambleEnzymeExt's rules for the `CpuThreaded` engine entries (gpena/Bramble.jl#369).
# Enzyme is not in test/Project.toml: load it first, through a temporary environment
# (test/ext/enzyme_policy_cases.jl says how), with at least 2 threads. Without it the file
# runs nothing.
#
# Every derivative is checked against ForwardDiff, on seeded non-uniform spacings, and the
# entry's rule must have run: Enzyme cannot differentiate the `Threads.@threads` loop itself
# on Julia 1.13.3, so a route around the rule would throw. The small cases run here, against
# ForwardDiff through the `CpuSerial` path; the cases of test/ext/enzyme_policy_cases.jl run
# in child processes (`run_isolated`), against ForwardDiff under `CpuThreaded`.

using Test
using Bramble
using Bramble: CpuSerial, CpuThreaded, StarSpacings, Forward, Backward, Centered,
               CrossWeighted, _difference_engine!, _sweep_for!, _sweep_scatter_for!
using ForwardDiff
using Random
using ..TestUtils: _have

const RUN_ENZYME = _have(:Enzyme) && Threads.nthreads() >= 2
RUN_ENZYME && @eval using Enzyme
RUN_ENZYME && include(joinpath(@__DIR__, "enzyme_policy_cases.jl"))

const TOL = 1e-10
const DIMS = (11, 9)

_relerr(g, ref) = maximum(abs, g .- ref) / max(maximum(abs, ref), floatmin(Float64))

# How many times the rule for `hook` has run.
_entries(hook::Symbol) = Base.get_extension(Bramble, :BrambleEnzymeExt)._rule_entries(hook)

# The worst relative error of Enzyme under `CpuThreaded` against ForwardDiff under
# `CpuSerial`, per mode: reverse, forward, and the two width-3 batch modes of
# enzyme_policy_cases.jl. `Enzyme.Forward`, since `Forward` is Bramble's stencil direction.
function _gradients(f, p)
    ref = ForwardDiff.gradient(q -> f(q, CpuSerial()), p)
    fp = q -> f(q, CpuThreaded())
    rev = Enzyme.gradient(set_runtime_activity(Enzyme.Reverse), Const(fp), p)[1]
    fwd = Enzyme.gradient(set_runtime_activity(Enzyme.Forward), Const(fp), p)[1]
    batch = map(EnzymePolicyCases.BATCH_MODES) do mode
        got, want = EnzymePolicyCases._batch(fp, p, ref, mode)
        return maximum(_relerr(g, w) for (g, w) in zip(got, want))
    end
    return (_relerr(rev, ref), _relerr(fwd, ref), batch...)
end

const STENCILS = ((Backward(), Val(:none)), (Backward(), Val(:vector)),
    (Forward(), Val(:star)), (Centered(), Val(:star)), (CrossWeighted(), Val(:vector)))

_spacing(::Val{:none}, s) = nothing
_spacing(::Val{:vector}, s) = s
_spacing(::Val{:star}, s) = StarSpacings(s)

function _engine_data(D::Int)
    rng = MersenneTwister(1)
    n, len = DIMS[D], prod(DIMS)
    return (0.1 .+ rand(rng, n), randn(rng, n), randn(rng, n), randn(rng, len),
        randn(rng, len), randn(rng, len))
end

# Two differences into one reused buffer, the field and the spacings both from `p`: `p[1]`
# and `p[2]` move the spacings of axis `D`, `p[3]` and `p[4]` the field. A rule that leaves
# the overwritten output's cotangent in place gives a wrong gradient.
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

# The same three captures in a tuple-valued kernel scattered into two vectors.
_capture(p, ::Val{:active}) = (a = p[1]^2; i -> a * p[mod1(i, 4)] * i)
_capture(p, ::Val{:constant}) = (a = 2.0; i -> a * p[mod1(i, 4)] * i)
_capture(p, ::Val{:ref}) = (r = Ref(2.0); i -> r[] * p[mod1(i, 4)] * i)
function _scatter_case(p, policy, kind)
    k = _capture(p, kind)
    v, w = zeros(eltype(p), 17), zeros(eltype(p), 17)
    _sweep_scatter_for!(policy, (v, w), 1:17, i -> (k(i), sin(i) * k(i)^2))
    return sum(abs2, v) + sum(w)
end

# A banded broadcast whose tree mixes an array with a `Float64` (`2.0 .* u`), into a
# `VectorElement` and through `2.0 * Mₕ(u)`, the shape `inner₊(2.0 * Mₕ(κ) * ∇ₕ(U), ∇ₕ(V))`
# builds. By value, such a `Broadcasted` fails Enzyme's calling convention at the
# `_static_bands!` rule, so the call site passes it behind a `Ref`. `spaces` holds a
# non-uniform grid space per policy, `B` a fixed basis.
function _broadcast_case(p, policy, spaces, B)
    W = policy isa CpuThreaded ? spaces[2] : spaces[1]
    u = element(W, eltype(p))
    parent(u) .= 1.0 .+ B * p
    v = element(W, eltype(p))
    v .= 2.0 .* u .+ u .^ 2
    m = 2.0 * Mₕ(u)
    return sum(abs2, parent(v)) + sum(c -> sum(abs2, parent(c)), m)
end

# One mesh per policy, each from the same seed, so both have the same nodes.
function _broadcast_spaces()
    Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
    return map((Serial(), Parallel())) do policy
        Bramble._seed_mesh1d_rng!(1)
        try
            gridspace(mesh(Ω, DIMS, (false, false); backend = backend(; policy)))
        finally
            Bramble._unseed_mesh1d_rng!()
        end
    end
end

# Runs `hook`'s case of enzyme_policy_cases.jl under `policy` in child processes
# (`run_isolated`) and checks every mode: the child exited cleanly, reported the mode,
# matched ForwardDiff and entered the rule the case names.
function _check_children(hook, modes; separate = (), policy = :CpuThreaded)
    codes, results = EnzymePolicyCases.run_isolated(hook, modes; policy, separate)
    @test all(==(0), codes)
    for m in modes
        @test haskey(results, m)
        haskey(results, m) || continue
        @test results[m].status == :ok
        @test results[m].entered
    end
    return nothing
end

@testset "Enzyme rules under CpuThreaded" begin
    if RUN_ENZYME
        @testset "Difference engine, reused output" begin
            p = [0.3, -0.2, 0.5, 0.7]
            for D in 1:2
                data = _engine_data(D)
                for (dir, kind) in STENCILS
                    n = _entries(:_static_bands!)
                    case = (q, pol) -> _engine_case(q, pol, dir, kind, Val(D), data)
                    @test all(<=(TOL), _gradients(case, p))
                    @test _entries(:_static_bands!) > n
                end
            end
        end

        @testset "Closure sweep reading its output" begin
            p = [0.3, -0.2, 0.5, 0.7]
            n = _entries(:_static_for!)
            @test all(<=(TOL), _gradients(_for_case, p))
            @test _entries(:_static_for!) > n
        end

        @testset "Broadcast mixing an array and a scalar" begin
            p = [0.3, -0.2, 0.5, 0.7]
            spaces = _broadcast_spaces()
            B = randn(MersenneTwister(2), prod(DIMS), length(p))
            n = _entries(:_static_bands!)
            @test all(<=(TOL), _gradients((q, pol) -> _broadcast_case(q, pol, spaces, B), p))
            @test _entries(:_static_bands!) > n
        end

        @testset "Closure capture shapes" begin
            p = [0.3, -0.2, 0.5, 0.7]
            for kind in (Val(:active), Val(:constant), Val(:ref))
                n = _entries(:_static_for!)
                @test all(<=(TOL), _gradients((q, pol) -> _closure_case(q, pol, kind), p))
                @test _entries(:_static_for!) > n
                n = _entries(:_static_scatter_for!)
                @test all(<=(TOL), _gradients((q, pol) -> _scatter_case(q, pol, kind), p))
                @test _entries(:_static_scatter_for!) > n
            end
        end

        # The reproducers from the rules' review, and the `THREADED` engine entries, each in
        # a child process of its own: one case's Enzyme compile can peak near 7 GB, and
        # compiled code stays resident, so they cannot share this process under julia_slot's
        # 10 GB cap.
        @testset "Reproducers in child processes" begin
            for c in EnzymePolicyCases.THREADED_CASES
                # The two nested cases differentiate inside a user's `Threads.@threads` loop,
                # where the rules above `_static_or_serial` run their tasks in order.
                modes = startswith(c.hook, "_threaded_nested_") ?
                        EnzymePolicyCases.NESTED_MODES :
                        (:reverse, :forward, EnzymePolicyCases.BATCH_MODES...)
                @testset "$(c.hook)" begin
                    _check_children(c.hook, modes)
                end
            end
        end

        @testset "Policy cases" begin
            for c in EnzymePolicyCases.THREADED
                # Its case's rectangle never takes the cut walk (`_RowRestricted`), so the
                # batch modes would report `bypassed`; the PROBE skips it as well.
                c.hook == "_static_cut_band_colour!" && continue
                # Reverse and forward for every entry, the batch modes on two of them. The
                # diffusion entries peak at 7 GB with every mode in one child, so each of their
                # modes runs in a child of its own.
                modes = c.hook in ("_static_bands!", "_static_for!") ?
                        (:reverse, :forward, EnzymePolicyCases.BATCH_MODES...) :
                        (:reverse, :forward)
                @testset "$(c.hook)" begin
                    _check_children(c.hook, modes;
                        separate = ("_static_band_colour!", "_static_colour!"))
                end
            end
        end
    end
end

# Reverse through an assembly of three summands under both threaded policies, in child
# processes: each must match ForwardDiff and enter its policy's bilinear rule.
@testset "Three-summand assembly reverse" begin
    if RUN_ENZYME
        for c in EnzymePolicyCases.ASSEMBLY_REVERSE_CASES
            @testset "$(c.hook)" begin
                _check_children(c.hook, (:reverse, :di_reverse); policy = c.policy,
                    separate = (c.hook,))
            end
        end
    end
end

end # module
