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
    # absent because it defines neither -- it only wraps `AlgebraicMultigrid`'s multigrid
    # hierarchy as a preconditioner, so adding it to `loaded_exts` would inspect a module
    # with nothing relevant in it. That absence does not, however, keep the ambiguity out of
    # sight: loading `AlgebraicMultigrid` also loads `LHLFactorization`, whose
    # `ldiv!(::AbstractVector, ::SparseLHLFactorization, ::AbstractVector)` is ambiguous with
    # Bramble's `VectorElement` method below regardless of what `loaded_exts` contains --
    # `Test.detect_ambiguities(Bramble, loaded_exts...)` pairs Bramble's own methods against
    # every method already loaded in the process, not only against `loaded_exts`'s. See the
    # explicit exclusion list below for this pair and two more of the same shape.
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

    # Three ambiguity pairs are excluded below because Bramble has no way to resolve them
    # itself: the clashing method belongs to a package Bramble has no (weak) dependency on,
    # loaded only transitively through a package it does depend on (weakly) or through a test
    # dependency, so there is nowhere in this codebase to hang a disambiguating method. Each
    # entry names the generic function and the external module defining the clashing method;
    # a pair is dropped only when one of its two methods is Bramble's own and the other
    # matches one of these (function, module) pairs exactly, so a genuinely new ambiguity
    # involving `ldiv!` or `mul!` but a different module -- Bramble's own or a third party's
    # -- still fails the test below.
    #
    # `LHLFactorizationSparseExt` (loaded as part of `AlgebraicMultigrid`, see the comment
    # above) defines `ldiv!(::AbstractVector, ::SparseLHLFactorization, ::AbstractVector)`,
    # ambiguous with Bramble's `ldiv!(::VectorElement, ::Factorization, ::AbstractVector)`
    # (src/space/vectorelement.jl) in exactly the shape the sparse direct solver extensions
    # each resolve with their own disambiguating method -- except this one belongs to
    # `LHLFactorization`, not to a package Bramble extends. Dropped until `LHLFactorization`
    # adds the disambiguating method upstream, the way every solver extension Bramble does
    # depend on already has.
    #
    # `PureKLUForwardDiffExt` (`PureKLU`'s own extension, loaded once `ForwardDiff` is also
    # in the process -- both already test dependencies, for the AD-backend and sparse-solver
    # tests respectively) defines `ldiv!(::AbstractArray{<:ForwardDiff.Dual}, ::
    # KLUFactorization, ::AbstractArray{<:ForwardDiff.Dual})`, ambiguous with the same
    # Bramble method for the same reason. `PureKLU` is a dependency of `LinearSolve`'s
    # default sparse factorization, not one Bramble names directly. Dropped for the same
    # reason and until the same kind of upstream fix.
    #
    # `ReverseDiff` (a test dependency, used to check that `pde_solve`'s AD rules compose
    # with third-party backends) defines
    # `mul!(::TrackedArray, ::AbstractMatrix, ::TrackedArray{V, D, 1})` directly, not in an
    # extension, ambiguous with `KroneckerLinearOperator`'s own
    # `mul!(::AbstractVector, ::KroneckerLinearOperator, ::AbstractVector)`
    # (src/form/kronecker.jl) because `KroneckerLinearOperator <: AbstractMatrix` satisfies
    # ReverseDiff's unconstrained middle argument. See the comment beside that `mul!` for why
    # neither narrowing it nor deleting it resolves this. Dropped until Bramble gains a (weak)
    # dependency on `ReverseDiff` to define the disambiguating method, or `ReverseDiff`
    # narrows its own signature away from bare `AbstractMatrix`.
    known_unfixable_ambiguities = (
        (:ldiv!, :LHLFactorizationSparseExt),
        (:ldiv!, :PureKLUForwardDiffExt),
        (:mul!, :ReverseDiff)
    )

    function _is_known_unfixable(m1::Method, m2::Method)
        bramble_method, other_method = m1.module === Bramble ? (m1, m2) : (m2, m1)
        bramble_method.module === Bramble || return false
        return any(known_unfixable_ambiguities) do (fname, modname)
            other_method.name === fname && nameof(other_method.module) === modname
        end
    end

    @testset "Extension method ambiguity" begin
        ext_ambiguities = Test.detect_ambiguities(Bramble, loaded_exts...; recursive = false)
        ext_ambiguities = filter(((m1, m2),) -> !_is_known_unfixable(m1, m2), ext_ambiguities)
        if !isempty(ext_ambiguities)
            for (m1, m2) in ext_ambiguities
                @error "Ambiguous method pair" m1 m2
            end
        end
        @test isempty(ext_ambiguities)
    end
end

end # module QualityAquaTests
