module ExtEnzymePolyesterHooksTests

# The Enzyme rules of the `CpuPolyester` hooks that the census does not run (gpena/Bramble.jl#369):
# the `CAPTURES` of test/ext/enzyme_policy_cases.jl, each called through the closure route that
# `_sweep_for!` and the matrix-free replay take. Each capture runs in a child Julia process of
# its own (`EnzymePolicyCases.run_isolated`), because one capture's Enzyme compile alone peaks
# near 6 GB and compiled code stays resident, so the captures cannot share a process under
# julia_slot's 10 GB cap. A child prints one `CASE_RESULT` line per mode, and this file asserts
# each. Enzyme is not in test/Project.toml: the children load it through a temporary environment
# (test/ext/enzyme_policy_cases.jl says how), and the parent needs at least 2 threads.

using Test
using Bramble
using ..TestUtils: _have

const RUN_ENZYME = _have(:Enzyme) && Threads.nthreads() >= 2
RUN_ENZYME && @eval using Enzyme
RUN_ENZYME && include(joinpath(@__DIR__, "enzyme_policy_cases.jl"))

# The captures whose modes, run in one child, peak above 7 GB: `_batch_mf_bands_coef` (reverse
# and forward, 5.96 GB for reverse alone) and `_batch_mf_overwrite_coef` (all five modes, 7.6 GB).
# Each of their modes runs in a child of its own.
const SEPARATE_MODES = ("_batch_mf_bands_coef", "_batch_mf_overwrite_coef")

@testset "Captured closures under CpuPolyester" begin
    if RUN_ENZYME
        # One testset per `CAPTURES` hook: reverse and forward, and all five modes for
        # `_batch_mf_overwrite_coef`, the fused rule's own case. Each mode is one line of the
        # child's output, and a missing line fails the mode.
        @testset "Hook rules" begin
            for c in EnzymePolicyCases.CAPTURES
                @testset "$(c.hook)" begin
                    modes = (:reverse, :forward)
                    c.hook == "_batch_mf_overwrite_coef" &&
                        (modes = (EnzymePolicyCases.MODES..., EnzymePolicyCases.BATCH_MODES...))
                    codes, results = EnzymePolicyCases.run_isolated(c.hook, modes;
                        policy = :CpuPolyester, separate = SEPARATE_MODES)
                    @test all(==(0), codes)
                    for m in modes
                        @test haskey(results, m)
                        haskey(results, m) && @test results[m].status == :ok
                    end
                end
            end
        end
    else
        @test_skip RUN_ENZYME
    end
end

# An inner product whose result Enzyme marks inactive (a value used only in a branch) reaches the
# reverse rule as `Const`: no seed, nothing to pull back. The child differentiates a function
# that reads `innerₕ(u, u)` only in `s > 0` and prints the gradient's error against ForwardDiff
# under `CpuSerial`, and whether the `_batch_dot` rule ran. Without Enzyme the file still loads:
# `EnzymePolicyCases` exists only when `RUN_ENZYME` holds.
const INACTIVE_DOT_SOURCE = !RUN_ENZYME ? "" : """
    _e = mktempdir()
    write(joinpath(_e, "Project.toml"), $(repr(EnzymePolicyCases.ENZYME_TOML)))
    push!(LOAD_PATH, _e)
    using Bramble, Enzyme, Polyester, ForwardDiff
    using Bramble: CpuPolyester, CpuSerial
    function smesh(pol)
        Bramble._seed_mesh1d_rng!(1234)
        try
            return mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (21, 17),
                (false, false); backend = backend(policy = pol))
        finally
            Bramble._unseed_mesh1d_rng!()
        end
    end
    function fdot(p, pol)
        W = gridspace(smesh(pol))
        u = element(W, eltype(p))
        parent(u) .= p[1] .* sin.(1:ndofs(W)) .+ p[2] .* cos.(1:ndofs(W))
        s = innerₕ(u, u)
        return s > 0 ? sum(abs2, parent(u)) : zero(eltype(p))
    end
    entries = Base.get_extension(Bramble, :BrambleEnzymePolyesterExt)._rule_entries
    p = [0.4, -0.7]
    ref = ForwardDiff.gradient(q -> fdot(q, CpuSerial()), p)
    n0 = entries(:_batch_dot)
    got = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse),
        Enzyme.Const(q -> fdot(q, CpuPolyester())), p)[1]
    println("INACTIVE_DOT ", maximum(abs, got .- ref) / maximum(abs, ref), " ",
        entries(:_batch_dot) > n0)
    """

@testset "Reverse dot with an inactive result" begin
    if RUN_ENZYME
        opt = max(Base.JLOptions().opt_level, 0)
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) --threads=$(Threads.nthreads()) --optimize=$opt -e $INACTIVE_DOT_SOURCE`
        out = IOBuffer()
        p = run(pipeline(ignorestatus(cmd); stdout = out, stderr = out))
        text = String(take!(out))
        p.exitcode == 0 || @warn "child exited with $(p.exitcode)" tail = last(text, 2000)
        @test p.exitcode == 0
        line = findfirst(l -> startswith(l, "INACTIVE_DOT "), split(text, '\n'))
        @test line !== nothing
        if line !== nothing
            _, relerr, entered = split(split(text, '\n')[line])
            @test parse(Float64, relerr) <= 1e-10
            @test parse(Bool, entered)
        end
    else
        @test_skip RUN_ENZYME
    end
end

end
