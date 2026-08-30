using Aqua

@testset "Aqua analysis" begin
    # Julia 1.14-DEV (nightly) has an upstream bug where Aqua's unbound_args inspection
    # fails on standard Type{<:T} method signatures. Run unbound_args on stable releases.
    test_unbound = isempty(VERSION.prerelease)

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
