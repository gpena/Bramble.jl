module ExtEnzymeForwardAssemblyTests

# Enzyme forward mode through `assemble` (gpena/Bramble.jl#1124). On the diffusion form
# `inner₊(2.0 * Mₕ(κ) * ∇ₕ(u), ∇ₕ(v)) + innerₕ(κ * u, 0.5 * D₋ᵧ(v))`, whose `inner₊` expands
# to two products, forward mode used to add the constant-scaled third summand's own primal
# to the tangent, one seed-independent offset in every gradient entry, under every policy.
# Reverse mode and the two-summand forms were exact. Spaces held their weights inline, which
# put the third summand's scale at byte 592 of the `BilinearForm`, past Enzyme's 512-byte
# type-analysis offset; raising that bound to 1024 made the gradient exact. Holding
# `SpaceWeights` by reference (3aefc79f, #1154) moved the scale to byte 80 and fixed it.
# The note docs/notes/internals/ad_reductions/1124_forward_offset.md has the bisect.
#
# `m + m + 0.25m` is not pinned. It was wrong when written as a literal closure, and exact
# when passed as a function, as below: the miscompile follows how the closure compiles.
#
# Every gradient is checked against ForwardDiff on the same mesh object (seeded non-uniform
# spacings), through DI's `AutoEnzyme` with runtime activity. Each policy runs in a child
# Julia process of its own, since the three together pass julia_slot's 10 GB cap. A child
# prints one `RESULT_1124` line per case. Enzyme is not in test/Project.toml: load it first,
# through a temporary environment (test/ext/enzyme_policy_cases.jl says how), with at least
# 2 threads. Without it the file runs nothing; with Enzyme and one thread it runs only the
# level-set testset.

using Test
using Bramble
using Bramble: CpuSerial, CpuThreaded, CpuPolyester, D₋ᵧ
using ForwardDiff
using Polyester
using Random
using SparseArrays: nonzeros
import DifferentiationInterface as DI
using ..TestUtils: _have

const HAVE_ENZYME = _have(:Enzyme)
const RUN_ENZYME = HAVE_ENZYME && Threads.nthreads() >= 2
HAVE_ENZYME && @eval using Enzyme

const TOL = 1e-12
const NP = 4
const X0 = [0.5 + 0.3 * sin(k) for k in 1:NP]
# Set in a child to the policy it runs.
const POLICY_ENV = "BRAMBLE_1124_POLICY"
const ENZYME_TOML = "[deps]\nEnzyme = \"7da242da-08ed-463a-9acd-ee780be4f1d9\"\n"

_relerr(g, ref) = maximum(abs, g .- ref) / max(maximum(abs, ref), floatmin(Float64))

_auto(mode) = DI.AutoEnzyme(; mode = Enzyme.set_runtime_activity(mode),
    function_annotation = Enzyme.Const)

# The form #1124 broke: a constant-scaled summand last of three.
_diffusion(κ, u, v) = inner₊(2.0 * Mₕ(κ) * ∇ₕ(u), ∇ₕ(v)) + innerₕ(κ * u, 0.5 * D₋ᵧ(v))
# Two summands, the scaled one last: exact.
_mass2(κ, u, v) = innerₕ(κ * u, v) + 0.25 * innerₕ(κ * u, v)
_react2(κ, u, v) = innerₕ(κ * u, v) + innerₕ(κ * u, 0.5 * D₋ᵧ(v))

# `p -> sum(abs2, nonzeros(assemble(a(p))))` for the form `terms` on a 17x17 non-uniform
# mesh under `policy`, its coefficient `κ = B * p` for a fixed random basis `B`.
function _functional(terms, policy)
    Bramble._seed_mesh1d_rng!(1124)
    Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
    M = mesh(Ω, (17, 17), (false, false); backend = backend(policy = policy))
    Bramble._unseed_mesh1d_rng!()
    W = gridspace(M)
    B = randn(MersenneTwister(7), ndofs(W), NP)
    function f(p::AbstractVector{T}) where {T}
        κ = element(W, T)
        parent(κ) .= B * p
        return sum(abs2, nonzeros(assemble(form(W, W, (u, v) -> terms(κ, u, v)))))
    end
    return f
end

# The mass matrix of a level-set mesh fitted to the circle of radius `r`, scaled by `c`.
# `φ` captures the float `r`: an immutable closure, whose `===` compares its bits.
function _levelset_mass(r, c)
    φ = x -> x[1]^2 + x[2]^2 - r^2
    Bramble._seed_mesh1d_rng!(1237)
    Ωₕ = try
        Ω = domain(interval(-2.0, 2.0) × interval(-2.0, 2.0), :D => levelset(φ))
        mesh(Ω, (12, 12), (false, false))
    finally
        Bramble._unseed_mesh1d_rng!()
    end
    W = gridspace(Ωₕ)
    return c * sum(abs2, nonzeros(assemble(form(W, W, (U, V) -> innerₕ(U, V)))))
end

# Forward mode's derivative of `f` at `[x]`, seed 1.0.
function _levelset_forward(f, x)
    mode = Enzyme.set_runtime_activity(Enzyme.Forward)
    return only(Enzyme.autodiff(mode, Enzyme.Const(f), Enzyme.Duplicated,
        Enzyme.Duplicated([x], [1.0])))
end

# A relative error, `Inf` when the derivative throws, so a throw fails its `@test`.
function _report(io, key, thunk)
    r = try
        thunk()
    catch e
        @warn "$key threw" exception = (e, catch_backtrace())
        Inf
    end
    println(io, "RESULT_1124 ", key, " ", r)
    flush(io)
    return nothing
end

# A child's work: every case under `policy`, one `RESULT_1124` line each. The width-2 batch
# (two tangents in one call, Enzyme's `BatchDuplicated` under DI) runs under `CpuSerial`.
function _child(policy)
    f = _functional(_diffusion, policy)
    ref = ForwardDiff.gradient(f, X0)
    _report(stdout, :diffusion_forward,
        () -> _relerr(DI.gradient(f, _auto(Enzyme.Forward), X0), ref))
    _report(stdout, :diffusion_reverse,
        () -> _relerr(DI.gradient(f, _auto(Enzyme.Reverse), X0), ref))
    if policy isa CpuSerial
        seeds = (Float64.(1:NP .== 1), Float64.(1:NP .== 2))
        _report(stdout, :diffusion_batch2,
            () -> _relerr(collect(DI.pushforward(f, _auto(Enzyme.Forward), X0, seeds)),
                ref[1:2]))
    end
    for (key, terms) in ((:mass2_forward, _mass2), (:react2_forward, _react2))
        g = _functional(terms, policy)
        r = ForwardDiff.gradient(g, X0)
        _report(stdout, key, () -> _relerr(DI.gradient(g, _auto(Enzyme.Forward), X0), r))
    end
    return nothing
end

# Runs `_child(policy)` in a Julia process of its own, with this one's project, threads and
# `--optimize`, and returns its exit code and its results by key.
function _run_child(policy)
    src = """
        _e = mktempdir()
        write(joinpath(_e, "Project.toml"), $(repr(ENZYME_TOML)))
        push!(LOAD_PATH, _e)
        using Test
        Base.include(Main, $(repr(joinpath(@__DIR__, "..", "TestUtils.jl"))))
        Base.include(Main, $(repr(@__FILE__)))
        """
    opt = max(Base.JLOptions().opt_level, 0)
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) --threads=$(Threads.nthreads()) --optimize=$opt -e $src`
    out = IOBuffer()
    env = POLICY_ENV => string(nameof(typeof(policy)))
    p = run(pipeline(ignorestatus(addenv(cmd, env)); stdout = out, stderr = out))
    text = String(take!(out))
    results = Dict{Symbol, Float64}()
    for line in split(text, '\n')
        startswith(line, "RESULT_1124 ") || continue
        _, key, value = split(line)
        results[Symbol(key)] = parse(Float64, value)
    end
    # A derivative that threw reports `Inf`, its warning in `text`: shown here.
    (p.exitcode == 0 && all(isfinite, values(results))) ||
        @warn "child for $policy: exit $(p.exitcode), results $results" tail = last(text, 4000)
    return p.exitcode, results
end

if RUN_ENZYME && haskey(ENV, POLICY_ENV)
    _child(getfield(Bramble, Symbol(ENV[POLICY_ENV]))())
else
    @testset "Enzyme forward assembly (#1124)" begin
        if RUN_ENZYME
            for policy in (CpuSerial(), CpuThreaded(), CpuPolyester())
                @testset "$(nameof(typeof(policy)))" begin
                    code, r = _run_child(policy)
                    @test code == 0
                    @test get(r, :diffusion_forward, Inf) < TOL
                    @test get(r, :diffusion_reverse, Inf) < TOL
                    @test get(r, :mass2_forward, Inf) < TOL
                    @test get(r, :react2_forward, Inf) < TOL
                    policy isa CpuSerial &&
                        @test get(r, :diffusion_batch2, Inf) < TOL
                end
            end
        end
    end
    # Forward mode refuses a `φ` over the parameter by name. At p = 1.0 the seed 1.0 equals
    # the captured radius, so the closure's shadow is `===` its primal: a rule that tested
    # `===` would take the grid as data and return 0.0, where central differences give
    # 0.115. A constant `φ` built inside the function keeps its derivative. One thread is
    # enough here.
    @testset "Enzyme forward, level set over p" begin
        if HAVE_ENZYME
            fparam(p) = _levelset_mass(p[1], 1.0)
            fconst(p) = _levelset_mass(0.9, p[1])
            for x in (0.7, 1.0)
                err = @test_throws ArgumentError _levelset_forward(fparam, x)
                msg = err isa Test.Pass ? sprint(showerror, err.value) : ""
                @test occursin("parameter of its level-set function", msg)
            end
            @test isapprox(_levelset_forward(fconst, 1.3), fconst([1.0]); rtol = 1e-12)
        end
    end
end

end
