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

end
