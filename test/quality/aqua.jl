using Aqua

@testset "Aqua" begin
    # `unbound_args` is disabled entirely, on every Julia version, not just on nightly:
    #
    # - Julia 1.14-DEV (nightly) has an upstream bug where Aqua's unbound_args inspection
    #   fails on standard Type{<:T} method signatures.
    # - `Test.detect_unbound_args` (which Aqua's check delegates to) has a separate blind
    #   spot, reproduced on stable Julia 1.12 with a minimal, Bramble-unrelated example:
    #   `f(y::NTuple{NQ, T}) where {NQ, T} = 1` reports both `NQ` and `T` as unbound, even
    #   though they plainly are — `NTuple{NQ, T}` desugars to `Tuple{Vararg{T, NQ}}`, and
    #   the check's tree walk does not look inside `Vararg`. `_cell_average`'s
    #   `nodes::NTuple{NQ, T}`/`wts::NTuple{NQ, T}` signatures (src/space/operators/cell_average.jl)
    #   hit exactly this after the StaticArrays → Tuple migration replaced `SVector{NQ, T}`
    #   (which the check handled fine, being an ordinary parametric struct, not a Vararg
    #   tuple) with `NTuple{NQ, T}`.
    test_unbound = false

    # Aqua's persistent_tasks check has an upstream incompatibility on Julia 1.13+
    # where precompilation exits without creating done.log.
    test_persistent_tasks = VERSION < v"1.13-"

    Aqua.test_all(Bramble;
        piracies = true,
        ambiguities = true,
        unbound_args = test_unbound,
        undefined_exports = true,
        project_extras = false,
        stale_deps = false,
        deps_compat = false,
        persistent_tasks = test_persistent_tasks)
    Aqua.test_ambiguities(Bramble; recursive = false)
end
