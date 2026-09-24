module QualityExplicitImportsTests

using Test
using Bramble
using ExplicitImports

# What Bramble and its extensions take from other modules, and how.
#
# The package already writes every import as `using X: a, b` rather than bare `using X`, and
# already has none unused -- so this file is a ratchet, not a cleanup: it fails when a new
# import drifts from that, not today. Coverage depends on which extensions are loaded when
# this runs: `test/utils/backends.jl` loads Metal incidentally on Apple Silicon even in the
# unit group, so `BrambleMetalExt` is usually checked too; the rest -- now including
# `BrambleSciMLSensitivityExt` -- are reached only when the "ext"/"full" group has loaded
# their triggers earlier in the same process. Every ignore entry below was verified against
# all of them loaded together, so the check is exact under "full" and a (harmless) subset of
# it otherwise -- an ignored name never encountered is not an error.
#
# The two checks worth having most are the cheap ones. `check_no_implicit_imports` keeps a
# bare `using X` from pulling in names nobody can see at the call site. `check_no_stale_
# explicit_imports` catches an import left behind after the code using it moved or went --
# the mistake a reviewer skims straight past, since an unused import breaks nothing and shows
# up in no test. Both are unconditional: nothing here is a deliberately-kept exception.

@testset "Explicit imports" begin
    @testset "No implicit or stale imports" begin
        @test check_no_implicit_imports(Bramble) === nothing
        @test check_no_stale_explicit_imports(Bramble) === nothing
    end

    # `BrownFullBasicInit` (BrambleSciMLSensitivityExt): `SciMLSensitivity`'s own `using
    # OrdinaryDiffEqCore: ..., BrownFullBasicInit, ...` brings it in without re-exporting it,
    # so `Base.which` traces the true owner past `SciMLSensitivity` -- to `OrdinaryDiffEqCore`
    # in some resolutions, further still to `DiffEqBase` in others, since `OrdinaryDiffEqCore`
    # itself only re-exports it too. `SciMLSensitivity` is the intended, documented way to
    # reach it (the same "reached through a re-export" shape as `MPI`/`Init`/`Initialized`
    # below, from MUMPS's own re-export of its MPI submodule); depending on either deeper
    # package directly, just to import one name past what actually uses it, would be worse.
    @testset "Imports come from the module that owns them" begin
        @test check_all_explicit_imports_via_owners(
            Bramble; ignore = (:BrownFullBasicInit,)
        ) === nothing
    end

    # Every name below is a deliberate reach into another module's internals -- spelling it
    # here means a *new* one fails this test instead of passing unremarked.
    #
    # - `sparse!`: `SparseArrays`' in-place `sparse`, never marked public there, and the
    #   whole point of `allocate_system_matrix`/`jacobian_pattern`: building a
    #   `SparseMatrixCSC` from `I`/`J`/`V` without the intermediate copies `sparse` makes.
    # - `Backend`, `_backend_eye`, `_backend_zeros` (BrambleMetalExt): Bramble's own backend
    #   type and the two hooks a new backend implements, reached from its own extension.
    # - `BilinearForm` (BrambleSparseADExt), `LinearForm` (BrambleSciMLExt),
    #   `CartesianProduct` (BrambleMeshesExt, BrambleSciMLExt): internal Bramble types named
    #   in a field or method signature, none of them exported.
    # - `trial_space` (BrambleSciMLExt): read back off a `BilinearForm` to unwrap a
    #   `LinearSolution` into a `VectorElement` over the right space.
    # - `_vtk_axes`, `_vtk_data` (BrambleVTKExt, BrambleVTKSciMLExt): reshape a mesh/field
    #   into what `vtk_grid`/`vtk[name] = ...` want. Neither touches a WriteVTK type, so both
    #   live in core `Bramble` (src/exporters/vtk_export.jl) rather than in either extension,
    #   letting the two share them without one depending on the other.
    # - `AbstractSpaceType` (BrambleVTKSciMLExt): narrows `export_vtk`'s solution-export
    #   method to a real discretisation space, the same reason `BrambleSciMLExt` reaches for
    #   it below.
    # - `CHOLMOD`, `UMFPACK` (BrambleSuiteSparseExt): SuiteSparse's own Cholesky/LU submodules,
    #   reached for the factorization backends.
    # - `AAFactorization`, `factor!`, `refactor!`, `SparseFactorizationCholesky`,
    #   `SparseFactorizationLDLT`, `SparseFactorizationLUTPP`, `SparseFactorizationQR`
    #   (BrambleAppleAccelerateExt): AppleAccelerate's own factorization type and the
    #   kind/hook names its `factor!`/`refactor!` calls need, none of them public there.
    # - `finalize!`, `get_sol!`, `set_cntl!`, `set_icntl!`, `suppress_display!`
    #   (BrambleMUMPSExt): MUMPS's own solver-control API, none of it public there.
    # - `BrownFullBasicInit` (BrambleSciMLSensitivityExt): brought into `SciMLSensitivity`'s
    #   own namespace from `OrdinaryDiffEqCore` without being re-exported (the "owners" check
    #   above has the full chain) -- reached the same way `MPI`/`Init`/`Initialized` below
    #   reach MUMPS's un-exported MPI submodule.
    @testset "Non-public imports are the declared ones" begin
        @test check_all_explicit_imports_are_public(
            Bramble;
            ignore = (
                # BramblePolyesterExt (gpena/Bramble.jl#190) reimplements the `CpuThreaded`
                # sweeps with `Polyester.@batch`, so it needs the same internals those sweeps
                # are built from: the colour/band geometry, the scatter primitives and the
                # masked-index iterator. None is public, and none should be.
                :MarkedIndicesUnion,
                :_band_range,
                :_reduce_or_chunk,
                :_scatter_linear_point!,
                :_scatter_point!,
                :_throw_dot_dim_error,
                :_write_components!,
                # `SeparableWeights` (commit d8c34602): the Bramble internal `weights(Wₕ,
                # Val(S))` returns for `length(S) >= 2` (src/space/scalar_gridspace.jl) --
                # `_batch_dot`/`_batch_dot_masked` are specialised on it
                # (ext/BramblePolyesterExt.jl:25) the same way the `CpuSerial`/`CpuThreaded`
                # methods in `space/inner_product.jl` already are, so a `CpuBatch` inner
                # product avoids the same per-point `CartesianIndex` conversion cost. Not
                # exported or public.
                :SeparableWeights,
                :sparse!,
                # `TrackedArray` (BrambleReverseDiffExt, commit c5ae771f): the argument type of the
                # `mul!` method that resolves the ambiguity with `KroneckerLinearOperator`
                # (gpena/Bramble.jl#295). ReverseDiff exports no public name for it.
                :TrackedArray,
                :Backend,
                :_backend_eye,
                :_backend_zeros,
                :BilinearForm,
                :LinearForm,
                :CartesianProduct,
                # `_DeviceSparseMirror` (BrambleMetalExt, gpena/Bramble.jl#313): the host
                # staging buffer type named in `MetalSparseMatrixCSR`'s own `mirror` field,
                # the same "internal type in a field signature" shape as `BilinearForm` above.
                :_DeviceSparseMirror,
                :trial_space,
                :_vtk_axes,
                :_vtk_data,
                :AbstractSpaceType,
                :BrownFullBasicInit,
                :CHOLMOD,
                :UMFPACK,
                :AAFactorization,
                :factor!,
                :refactor!,
                :SparseFactorizationCholesky,
                :SparseFactorizationLDLT,
                :SparseFactorizationLUTPP,
                :SparseFactorizationQR,
                :finalize!,
                :get_sol!,
                :set_cntl!,
                :set_icntl!,
                :suppress_display!
            )
        ) === nothing
    end

    # `VectorElement` is an `AbstractVector` and `ScalarGridSpace` hands out reshaped views,
    # so the broadcasting and array internals below have to be reached by name -- Base
    # exposes no public spelling for any of them. `eval` is `Core.eval`, reached by the
    # `@forward` macro.
    #
    # The rest are extension methods added to a function that is not itself exported by the
    # module supplying it, the same idiom as `Base.show`/`Base.size` above:
    #
    # - `expand_dimensions` (BrambleMakieExt): a Makie plotting-pipeline internal, overridden
    #   for `VectorElement` -- confirmed by rendering, see the file's own note.
    # - `_metal_backend`, `Metal.fill!` (BrambleMetalExt): the backend constructor hook and
    #   Metal's own `fill!` on an `MtlArray`.
    # - `EnzymeRules.augmented_primal`/`reverse`/`width`/`needs_primal`/`needs_shadow`
    #   (BrambleEnzymeExt): the custom-rule interface `EnzymeRules` documents extending this
    #   way, none of it exported (an exported `reverse` would shadow `Base`'s own for anyone
    #   who `using`s `EnzymeRules` directly). Which of the five this check actually flags
    #   depends on exactly what else is loaded alongside `Enzyme` when it runs -- `reverse`
    #   alone with `SciMLSensitivity` loaded but not `Enzyme`/`ChainRulesCore` directly
    #   (`EnzymeCore` only transitively), `augmented_primal` too once `Enzyme` itself joins
    #   -- so all five are listed together rather than chased one at a time as the loaded
    #   set shifts. Nothing about `BrambleEnzymeExt` changed; only what else was loaded
    #   alongside it did, once `SciMLSensitivity` (BrambleSciMLSensitivityExt) joined the
    #   "ext" group's set.
    # - `adjoint_sensitivities` (BrambleSciMLSensitivityExt): the one entry point not
    #   underscored -- `Bramble.adjoint_sensitivities` is meant to be called, just not
    #   exported, since `SciMLSensitivity` exports a function of the exact same name and
    #   `using Bramble, SciMLSensitivity` together would collide on the bare name regardless
    #   of what Bramble does (the stub's own docstring, `form/semidiscrete_problems.jl`, has the
    #   reasoning). Its core fallback still gives the same helpful error the underscored ones
    #   below do.
    # - `_ast_sparsity_detector` (BrambleSparseADExt), `_ode_function`/`_ode_problem`/
    #   `_linear_problem`/`_nonlinear_problem`/`_second_order_ode_function`/
    #   `_second_order_ode_problem` (BrambleSciMLExt), `_export_vtk` (BrambleVTKExt): the
    #   underscored-fallback idiom every weak-dependency entry point uses (`ast_sparsity_
    #   detector`, `ode_function`, `export_vtk`, ...): a helpful error by default in
    #   `Bramble`, overridden by a strict specialisation in the extension so loading it never
    #   tries to replace a method during precompilation.
    # - `AbstractSpaceType` (BrambleSciMLExt): narrows the `element`/`VectorElement` unwrap
    #   methods for a `LinearSolution`, the same reason the doctring next to them gives.
    # - `_amg_operator` (BrambleAlgebraicMultigridExt), `_ilu_operator` (BrambleILUZeroExt):
    #   the preconditioner-building hooks `solve`'s `preconditioner = :amg`/`:ilu0` keywords
    #   reach.
    # - `_amg_preconditioner` (BrambleAlgebraicMultigridExt), `_ilu_preconditioner`
    #   (BrambleILUZeroExt): the public `amg_preconditioner`/`ilu_preconditioner` functions'
    #   own underscored fallbacks.
    # - `_export_vtk_collection` (BrambleVTKExt), `_export_vtk_solution`
    #   (BrambleVTKSciMLExt): the same underscored-fallback idiom as `_export_vtk` above, one
    #   entry point per `export_vtk` method that needs a weak dependency.
    # - `apply_recipe` (BramblePlotsExt): the function `@recipe` generates methods on;
    #   warmed by name in the precompile workload rather than through a plotting call, which
    #   this coverage-dependent check had simply not caught loaded alongside the others
    #   before.
    # - `_sparspak_factorize`, `_sparspak_refactor!`, `_sparspak_solve` (BrambleSparspakExt):
    #   the preconditioner-building hooks its `factorize`/`refactor!`/`\` methods reach,
    #   the same underscored-fallback idiom as `_amg_operator` above.
    # - `_accelerate_factorize`, `_accelerate_refactor!`, `_accelerate_solve`
    #   (BrambleAppleAccelerateExt): the same underscored-fallback idiom, one entry point
    #   per AppleAccelerate-backed `factorize`/`refactor!`/`\` method.
    # - `_mumps_factorize`, `_mumps_refactor!`, `_mumps_solve` (BrambleMUMPSExt): the same
    #   underscored-fallback idiom again, one entry point per MUMPS-backed
    #   `factorize`/`refactor!`/`\` method.
    # - `MPI`, `Init`, `Initialized` (BrambleMUMPSExt): MUMPS's own re-export of its MPI
    #   submodule, reached once to initialise MPI lazily on first use.
    # - `FACTOR`, `SOLVE`, `invoke_mumps!` (BrambleMUMPSExt): MUMPS's job-type constants and
    #   the low-level driver call its `ldiv!`/`refactor!` methods issue directly.
    # - `_suitesparse_factorize`, `_suitesparse_refactor!`, `_suitesparse_solve`
    #   (BrambleSuiteSparseExt): the same underscored-fallback idiom again, one entry point
    #   per SuiteSparse-backed `factorize`/`refactor!`/`\` method.
    @testset "Non-public qualified accesses are the declared ones" begin
        @test check_all_qualified_accesses_are_public(
            Bramble;
            ignore = (
                # Internals this milestone's extensions reach into, the same way
                # `_metal_backend` below already is. `_csr_backend` is the stub
                # BrambleSparseMatricesCSRExt fills (gpena/Bramble.jl#214), exactly
                # `_metal_backend`'s shape; `_backend_eye`/`_backend_zeros`,
                # `_dirichlet_bc_rows!`/`_dirichlet_bc_indices!` and `_each_marked` are the
                # allocation and constraint internals a storage backend has to specialise;
                # `_kron_coeff` (BrambleKroneckerExt, gpena/Bramble.jl#259) is the scale a
                # separable term carries, read when the extension rebuilds that sum as a
                # `Kronecker.jl` object. None is something a user calls.
                :_csr_backend,
                :_backend_eye,
                :_backend_zeros,
                :_dirichlet_bc_rows!,
                :_dirichlet_bc_indices!,
                :_each_marked,
                :_kron_coeff,
                # `BrambleKernelAbstractionsExt` (gpena/Bramble.jl#94, #174): the stencil and
                # component helpers its `@kernel`s call so the device answer is computed by
                # the very same quadrature/stencil arithmetic the CPU sweep uses, rather than
                # a second implementation kept in sync by hand. None is a launch hook an
                # extension implements (those are the `public _launch_*`/`_gpu_*` contract in
                # `src/Bramble.jl`) -- these are plain internals the kernels reach into.
                :_cell_average,
                :_compute_average,
                :_compute_difference,
                :_neighbour,
                :_stencil_step,
                :_stencil_boundary_dim,
                :_write_components!,
                # The direction/stencil-kind dispatch types the fused vector-calculus and
                # difference kernels are parametrised over (src/space/operators/stencil.jl,
                # src/space/operators/difference.jl) -- named in the `@kernel`s' own method
                # signatures, the same way `CartesianProduct` below is, and none exported.
                :GridDirection,
                :Forward,
                :Backward,
                :Centered,
                :CrossWeighted,
                :ArrayStyle,
                :BroadcastStyle,
                :Broadcasted,
                :RefValue,
                :ReshapedArray,
                :SizeUnknown,
                :eval,
                :mightalias,
                :show,
                :expand_dimensions,
                :_metal_backend,
                Symbol("fill!"),
                :augmented_primal,
                :reverse,
                :width,
                :needs_primal,
                :needs_shadow,
                :adjoint_sensitivities,
                :_ast_sparsity_detector,
                :CartesianProduct,
                :_linear_problem,
                :_ode_function,
                :_ode_problem,
                :_nonlinear_problem,
                :_second_order_ode_function,
                :_second_order_ode_problem,
                :_export_vtk,
                :AbstractSpaceType,
                :_amg_operator,
                :_amg_preconditioner,
                :_ilu_operator,
                :_ilu_preconditioner,
                :_export_vtk_collection,
                :_export_vtk_solution,
                :apply_recipe,
                :_sparspak_factorize,
                :_sparspak_refactor!,
                :_sparspak_solve,
                :_accelerate_factorize,
                :_accelerate_refactor!,
                :_accelerate_solve,
                :_mumps_factorize,
                :_mumps_refactor!,
                :_mumps_solve,
                :MPI,
                :Init,
                :Initialized,
                :FACTOR,
                :SOLVE,
                :invoke_mumps!,
                :_suitesparse_factorize,
                :_suitesparse_refactor!,
                :_suitesparse_solve,
                # BrambleMetalExt reaching into Bramble's own internals (gpena/Bramble.jl#192,
                # #250): `_gpu_functional` is the loaded-and-functional predicate `gpu_backend`
                # dispatches on by `Val`, more specific than the stub in `src/utils/backend.jl`
                # and not itself public. `_gpu_functional_override` (commit 9698755b) is the
                # `Ref` test hook `_gpu_functional` reads to fake device (un)availability
                # without redefining the method; the extension reads the same `Ref` so a test
                # can force GPU-unavailable behaviour through it too. Neither is public.
                # `metal_sparse_csr`/`metal_sparse_csc` carry docstrings
                # on their `src/utils/backend.jl` stubs, but neither is exported nor declared
                # `public` in `src/Bramble.jl`, nor documented in `docs/src/api.md` -- so today
                # they are unqualified internals too, the same as the extension's other entry
                # points above. `SparseArrays.sparse!` (gpena/Bramble.jl#94) is that package's
                # in-place `sparse`, never marked public there, and how `_allocate_from_pattern`
                # builds the host-side CSR arrays it hands to `metal_sparse_csr` -- the same
                # combiner `SparseMatrixCSC`'s own method already reaches for by the same name.
                :_gpu_functional,
                :_gpu_functional_override,
                :metal_sparse_csr,
                :metal_sparse_csc,
                Symbol("sparse!"),
                # BrambleMetalExt's device sparse placeholder types (gpena/Bramble.jl#250)
                # subtype `Metal.GPUArrays`'s own `AbstractGPUSparseMatrixCSR`/
                # `AbstractGPUSparseMatrixCSC` until tagged Metal.jl ships the real
                # `MtlSparseMatrixCSR`/`MtlSparseMatrixCSC` -- upstream internals that neither
                # `Metal` nor `GPUArrays` declares public, with no public alternative to reach
                # them by. Adapting those placeholders to the device
                # (`Metal.Adapt.adapt_structure`/`adapt`) reaches `Adapt` and `GPUArrays`
                # themselves the same way, through `Metal`'s own non-public re-export of each.
                :AbstractGPUSparseMatrixCSR,
                :AbstractGPUSparseMatrixCSC,
                :Adapt,
                :GPUArrays,
                # `_KronDeviceDiagonal`, `_KronDeviceSparse` (BrambleKroneckerExt, commit
                # ece71258): the device-resident factor types `mul!` dispatches on, named in
                # the extension's method signatures. Plain internals, not a launch hook.
                :_KronDeviceDiagonal,
                :_KronDeviceSparse,
                # `SparseArrays.getcolptr` (src/form/kronecker.jl, commit ece71258): copies a
                # factor's column pointers to the device; no public accessor exists.
                :getcolptr,
                # `Base.inferencebarrier` (src/form/bilinear_execution.jl): the fallback for a
                # transposed pair whose two block tuples differ in length, a case the types
                # already rule out, so the barrier keeps it from being inferred at all.
                :inferencebarrier,
                # `ReverseDiff.record_mul!` (BrambleReverseDiffExt, #295): records the tape entry
                # for that `mul!`; ReverseDiff has no public equivalent.
                :record_mul!,
                # `Core.kwcall` (gpena/Bramble.jl#283): named in `precompile(Core.kwcall,
                # (...))` directives in `src/precompile/solver_sessions.jl` and
                # `src/precompile/form_sessions.jl`, caching the keyword-call entry
                # signature a REPL call dispatches to -- otherwise inlined into the
                # workload's static calls and never cached standalone. The documented
                # lowering target of a keyword call, but not declared public in `Core`.
                :kwcall
            )
        ) === nothing
    end

    # `@forward VectorElement.data (Base.size, Bramble.show)` and
    # `@forward VectorElement.space (Bramble.mesh,)` name the function they extend in full,
    # which is what the macro takes; unqualified would be a different binding.
    @testset "Self-qualified accesses are the declared ones" begin
        @test check_no_self_qualified_accesses(Bramble; ignore = (:mesh, :show)) === nothing
    end
end

end # module QualityExplicitImportsTests
