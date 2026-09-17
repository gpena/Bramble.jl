module QualityAquaTests

using Test
using Bramble
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

    Aqua.test_all(
        Bramble;
        piracies = true,
        ambiguities = true,
        unbound_args = test_unbound,
        undefined_exports = true,
        project_extras = true,
        stale_deps = true,
        deps_compat = true,
        # This check ran on every version until ea501d8 guarded it off Julia 1.13, where
        # the wrapper precompilation was exiting without writing Aqua's done.log. That was
        # measured against a 1.13 prerelease a month before the release and does not
        # reproduce on 1.13 itself: it passes with the same Aqua 0.8.16, both on a warm
        # depot and on a cold one that had to compile Bramble from source. The guard is
        # gone rather than widened to 1.14, since it would otherwise silence the check on
        # the version CI now gates on.
        #
        # `tmax = 30` rather than 0.8.16's default of 10. That budget covers only the
        # wrapper process exiting *after* Bramble has finished loading, and 10 s of it is
        # thin enough on a loaded runner that a run of downstream packages have papered
        # over spurious failures with retries (JuliaTesting/Aqua.jl#315). Upstream raised
        # the default to 30 in Aqua.jl#389, merged but unreleased as of 0.8.16 — drop this
        # argument once a release carries it.
        persistent_tasks = (tmax = 30,)
    )
    Aqua.test_ambiguities(Bramble; recursive = false)

    # `Aqua.test_all`/`test_ambiguities` above inspect the methods of `Bramble` only, and a
    # package extension is a module of its own -- so a method defined there that clashes with
    # one in `Bramble` is invisible to that check. This is not hypothetical: every sparse
    # direct solver extension defines
    # `ldiv!(::AbstractVector, ::Concrete...Factorization, ::AbstractVector)`, which was
    # ambiguous with Bramble's `ldiv!(::VectorElement, ::Factorization, ::AbstractVector)`
    # for a `VectorElement` destination until each extension added the disambiguating method,
    # and each `\(::Concrete...Factorization, ::AbstractVector)` was ambiguous with
    # `LinearAlgebra`'s complex-right-hand-side `\` on a `Factorization`.
    #
    # `Test.detect_ambiguities` rather than `Aqua.test_ambiguities` for this part: Aqua runs
    # its detection in a subprocess that `import`s each module by `PkgId`, and an extension
    # module cannot be loaded that way -- every one of them fails there with
    # `ConcurrencyViolationError("deadlock detected in loading ...")`, so its methods are
    # never inspected and the check passes without having looked at anything. Run in this
    # process, against the extensions the `using` below has already triggered, it does look.
    #
    # The list covers the extensions that define `ldiv!` or `\` on a `Factorization`
    # subtype, which is where this class of clash lives. `BrambleAlgebraicMultigridExt` is
    # deliberately absent: loading `AlgebraicMultigrid` also loads `LHLFactorization`, whose
    # `ldiv!(::AbstractVector, ::SparseLHLFactorization, ::AbstractVector)` is ambiguous with
    # Bramble's `VectorElement` method in exactly the same way -- and Bramble cannot add the
    # disambiguating method for a package it has no (weak) dependency on.
    #
    # Only extensions whose trigger package is loadable are checked; the rest are reported
    # and skipped, so this file stays runnable wherever one of them is missing.
    ext_triggers = [
        :SuiteSparse => :BrambleSuiteSparseExt,
        :Sparspak => :BrambleSparspakExt,
        :MUMPS => :BrambleMUMPSExt,
        :AppleAccelerate => :BrambleAppleAccelerateExt,
        :ILUZero => :BrambleILUZeroExt
    ]

    loaded_exts = Module[]
    for (trigger, extname) in ext_triggers
        try
            @eval using $trigger
        catch err
            @info "Skipping ambiguity check for $extname: trigger package $trigger not loadable." exception = err
            continue
        end
        ext = Base.get_extension(Bramble, extname)
        if ext === nothing
            @info "Skipping ambiguity check for $extname: extension not loaded."
        else
            push!(loaded_exts, ext)
        end
    end

    @testset "Extension method ambiguity" begin
        ext_ambiguities = Test.detect_ambiguities(Bramble, loaded_exts...; recursive = false)
        if !isempty(ext_ambiguities)
            for (m1, m2) in ext_ambiguities
                @error "Ambiguous method pair" m1 m2
            end
        end
        @test isempty(ext_ambiguities)
    end
end

end # module QualityAquaTests
