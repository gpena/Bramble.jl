module QualityExplicitImportsTests

using Test
using Bramble
using ExplicitImports
using Polyester: Polyester
using Kronecker: Kronecker

# What Bramble and its extensions take from other modules, and how.
#
# The package already writes every import as `using X: a, b` rather than bare `using X`, and
# already has none unused -- so this file is a ratchet, not a cleanup: it fails when a new
# import drifts from that, not today. Coverage depends on which extensions are loaded when
# this runs. `BramblePolyesterExt` and `BrambleKroneckerExt` are always checked: this file
# loads `Polyester` and `Kronecker` itself, then asserts both extensions are there.
# `test/utils/backends.jl` loads Metal incidentally on Apple Silicon even in the unit group,
# so `BrambleMetalExt` is usually checked too; the rest -- now including
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
    # The two lines above load the triggers; this fails the file if either extension is still
    # not there, so the checks below can never pass vacuously for them.
    @testset "Polyester and Kronecker are loaded" begin
        @test Base.get_extension(Bramble, :BramblePolyesterExt) !== nothing
        @test Base.get_extension(Bramble, :BrambleKroneckerExt) !== nothing
    end

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
    #
    # `libblastrampoline` (BrambleKroneckerExt): `LinearAlgebra.BLAS` re-exports the library
    # name `ccall` wants from `libblastrampoline_jll`, which Bramble does not depend on and
    # should not just to spell one `ccall` target; `LinearAlgebra.BLAS` is where the manual
    # documents it.
    @testset "Imports come from the owning module" begin
        @test check_all_explicit_imports_via_owners(
            Bramble; ignore = (:BrownFullBasicInit, :libblastrampoline)
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
                # BramblePolyesterExt reimplements the `CpuThreaded`
                # sweeps with `Polyester.@batch`, so it needs the same internals those sweeps
                # are built from: the colour/band geometry, the scatter primitives and the
                # masked-index iterator. None is public, and none should be.
                :MarkedIndicesUnion,
                :_band_range,
                :_reduce_or_chunk,
                # `_ReplayTarget`, `_replay_point!`: the warmed-refill
                # replay's target union and per-point step, called from
                # `_batch_bilinear_band_replay!`/`_batch_bilinear_colour_replay!` above the
                # same way the searching sweep calls `_scatter_point!`. Neither is public.
                :_ReplayTarget,
                # `_ActionTarget`: the matrix-free product's sink union,
                # passed through the same two Polyester replay hooks as `_ReplayTarget`. Not
                # public.
                :_ActionTarget,
                :_replay_point!,
                :_scatter_linear_point!,
                :_scatter_point!,
                :_throw_dot_dim_error,
                :_write_components!,
                # `SeparableWeights` (commit d8c34602): the Bramble internal `weights(Wₕ,
                # Val(S))` returns for `length(S) >= 2` (src/space/scalar_gridspace.jl) --
                # `_batch_dot`/`_batch_dot_masked` are specialised on it
                # (ext/BramblePolyesterExt.jl:25) the same way the `CpuSerial`/`CpuThreaded`
                # methods in `space/inner_product.jl` already are, so a `CpuPolyester` inner
                # product avoids the same per-point `CartesianIndex` conversion cost. Not
                # exported or public.
                :SeparableWeights,
                :sparse!,
                # `_difference_band!`, `_average_band!`, `_centered_average_band!`
                #: the per-band loop bodies
                # `_batch_difference_engine!`/`_batch_average_engine!`/
                # `_batch_centered_average_engine!` below run under `@batch`, the same bodies
                # `_threaded_difference_engine!`/`_threaded_average_engine!`/
                # `_threaded_centered_average_engine!` already run under `Threads.@threads`.
                # Neither exported nor public.
                :_difference_band!,
                :_average_band!,
                :_centered_average_band!,
                # `_broadcast_band!`: the
                # per-band broadcast loop body `_batch_broadcast!` below runs under `@batch`,
                # the same body `_threaded_broadcast!` already runs under `Threads.@threads`
                # (src/space/vectorelement.jl). Neither exported nor public.
                :_broadcast_band!,
                # `ReplaySink`, `_PairReplaySink`, `_DiagonalReplayTarget`, `ActionSink`,
                # `_PairActionSink`, `_ScatterCSC` (BramblePolyesterExt): the sink and target
                # types the warmed-refill replay and the matrix-free product pass between
                # Bramble's sweeps and the extension's `@batch` bodies, named in its method
                # signatures and in `_batch_split`'s skeletons. None is exported or public.
                :ReplaySink,
                :_PairReplaySink,
                :_DiagonalReplayTarget,
                :ActionSink,
                :_PairActionSink,
                :_ScatterCSC,
                # `_AvgKernel`, `_AvgScatterKernel`, `_MaskedKernel`, `_RₕKernel`, `__prod`:
                # the per-point kernels of the cell-average, projection and restriction
                # operators, run by the extension's `@batch` bodies as the `CpuThreaded` sweeps
                # run them, and the diagonal product the separable weights share.
                :_AvgKernel,
                :_AvgScatterKernel,
                :_MaskedKernel,
                :_RₕKernel,
                :__prod,
                # `_batch_split`, `_batch_rebuild`, `_batch_splittable` (src/utils/batch_split.jl):
                # `@batch` cannot capture a struct that holds GC references, so the extension
                # splits one into a bits-only skeleton plus its arrays and rebuilds it inside
                # the task. Neither these nor the host-side splitters below are public.
                :_batch_split,
                :_batch_rebuild,
                :_batch_splittable,
                # `_bc_host_raw`, `_kron_host_raw`, `_kron_host_rebuild`, `_bc_host_rebuild`:
                # the broadcast and Kronecker operands' `@batch`-crossing forms (the same
                # skeleton idea as above, spelled per operand type). `_kron_line_init!`,
                # `_kron_line_terms!`: the per-line bodies of the Kronecker product that
                # `_batch_kron_lines!` runs under `@batch`.
                :_bc_host_raw,
                :_bc_host_rebuild,
                :_kron_host_raw,
                :_kron_host_rebuild,
                :_kron_line_init!,
                :_kron_line_terms!,
                # `_MFFusedPlan`, `_MFPass`, `_MF_BAND`, `_MF_NO_COLLECT`, `_mf_apply_parts!`,
                # `_mf_host_ast` (src/assembly/matrix_free.jl): the matrix-free product's fused
                # plan, its pass kinds and its host-side AST copy, which `_batch_mf_bands!`
                # runs under `@batch`; `_mf_apply_parts!` is the method it extends.
                :_MFFusedPlan,
                :_MFPass,
                :_MF_BAND,
                :_MF_NO_COLLECT,
                :_mf_apply_parts!,
                :_mf_host_ast,
                # `_dot_band`, `_last_axis_chunks`, `_separable_line_band`,
                # `_separable_block_band` (src/utils/linear_algebra.jl,
                # src/space/inner_product.jl): the per-band bodies and the last-axis chunking of
                # the dot product and the separable inner product, shared with the
                # `CpuThreaded` methods so a `CpuPolyester` reduction cuts and sums identically.
                :_dot_band,
                :_last_axis_chunks,
                :_separable_line_band,
                :_separable_block_band,
                # `BlasInt`, `@blasfunc`, `libblastrampoline` (BrambleKroneckerExt): the `ccall`
                # vocabulary of its direct LAPACK bindings (`sygvd`, `potrf`, `gges3`, which
                # `LinearAlgebra.LAPACK` does not wrap for the shapes the fast diagonalisation
                # solve needs). `LinearAlgebra` documents none of the three as public.
                :BlasInt,
                Symbol("@blasfunc"),
                :libblastrampoline,
                # `TrackedArray` (BrambleReverseDiffExt, commit c5ae771f): the argument type of the
                # `mul!` method that resolves the ambiguity with `KroneckerLinearOperator`
                #. ReverseDiff exports no public name for it.
                :TrackedArray,
                :Backend,
                :_backend_eye,
                :_backend_zeros,
                :BilinearForm,
                :LinearForm,
                :CartesianProduct,
                # `_DeviceSparseMirror`: the host
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
                :suppress_display!,
                # v3.12.0 narrowed the export/public surface; the names below are
                # Bramble's own extension hooks and internals, private since that release,
                # reached only from the extension that implements or specialises them.
                :AbstractMeshType,
                :ka_device,
                :ka_synchronize,
                :_launch_uniform_mesh1d_init!,
                :_launch_half_points!,
                :_launch_spacing!,
                :_launch_half_spacing!,
                :_launch_nonuniform_mesh1d_metrics!,
                :_launch_refine_indices!,
                :_gpu_for!,
                :_gpu_scatter_for!,
                :_launch_restriction!,
                :_launch_restriction_scatter!,
                :_launch_restriction_nd!,
                :_launch_restriction_scatter_nd!,
                :_launch_cell_average!,
                :_launch_cell_average_scatter!,
                :_launch_cell_average_nd!,
                :_launch_cell_average_scatter_nd!,
                :_launch_difference_onesided!,
                :_launch_difference_centered!,
                :_launch_average_engine!,
                :_launch_spmv_csr!,
                :_launch_spmm_csr!,
                :_launch_dirichlet_rows_csr!,
                :_launch_kron_fused!,
                :_launch_fused_divergence!,
                :_launch_fused_curl2d!,
                :_launch_fused_curl3d!,
                :_launch_fused_laplacian!,
                :_launch_fused_strain_offdiag!,
                :suitesparse_solve,
                :suitesparse_refactor!,
                :sparspak_factorize,
                :sparspak_solve,
                :sparspak_refactor!
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
    #   `_`-prefixed -- `Bramble.adjoint_sensitivities` is meant to be called, just not
    #   exported, since `SciMLSensitivity` exports a function of the exact same name and
    #   `using Bramble, SciMLSensitivity` together would collide on the bare name regardless
    #   of what Bramble does (the stub's own docstring, `form/semidiscrete_problems.jl`, has the
    #   reasoning). Its core fallback still gives the same helpful error the `_`-prefixed ones
    #   below do.
    # - `_ast_sparsity_detector` (BrambleSparseADExt), `_ode_function`/`_ode_problem`/
    #   `_linear_problem`/`_nonlinear_problem`/`_second_order_ode_function`/
    #   `_second_order_ode_problem` (BrambleSciMLExt), `_export_vtk` (BrambleVTKExt): the
    #   `_`-prefixed fallback idiom every weak-dependency entry point uses (`ast_sparsity_
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
    #   own `_`-prefixed fallbacks.
    # - `_export_vtk_collection` (BrambleVTKExt), `_export_vtk_solution`
    #   (BrambleVTKSciMLExt): the same `_`-prefixed fallback idiom as `_export_vtk` above, one
    #   entry point per `export_vtk` method that needs a weak dependency.
    # - `apply_recipe` (BramblePlotsExt): the function `@recipe` generates methods on;
    #   warmed by name in the precompile workload rather than through a plotting call, which
    #   this coverage-dependent check had simply not caught loaded alongside the others
    #   before.
    # - `_sparspak_factorize`, `_sparspak_refactor!`, `_sparspak_solve` (BrambleSparspakExt):
    #   the preconditioner-building hooks its `factorize`/`refactor!`/`\` methods reach,
    #   the same `_`-prefixed fallback idiom as `_amg_operator` above.
    # - `_accelerate_factorize`, `_accelerate_refactor!`, `_accelerate_solve`
    #   (BrambleAppleAccelerateExt): the same `_`-prefixed fallback idiom, one entry point
    #   per AppleAccelerate-backed `factorize`/`refactor!`/`\` method.
    # - `_mumps_factorize`, `_mumps_refactor!`, `_mumps_solve` (BrambleMUMPSExt): the same
    #   `_`-prefixed fallback idiom again, one entry point per MUMPS-backed
    #   `factorize`/`refactor!`/`\` method.
    # - `MPI`, `Init`, `Initialized` (BrambleMUMPSExt): MUMPS's own re-export of its MPI
    #   submodule, reached once to initialise MPI lazily on first use.
    # - `FACTOR`, `SOLVE`, `invoke_mumps!` (BrambleMUMPSExt): MUMPS's job-type constants and
    #   the low-level driver call its `ldiv!`/`refactor!` methods issue directly.
    # - `_suitesparse_factorize`, `_suitesparse_refactor!`, `_suitesparse_solve`
    #   (BrambleSuiteSparseExt): the same `_`-prefixed fallback idiom again, one entry point
    #   per SuiteSparse-backed `factorize`/`refactor!`/`\` method.
    @testset "Non-public qualified accesses declared" begin
        @test check_all_qualified_accesses_are_public(
            Bramble;
            ignore = (
                # Internals this milestone's extensions reach into, the same way
                # `_metal_backend` below already is. `_csr_backend` is the stub
                # BrambleSparseMatricesCSRExt fills, exactly
                # `_metal_backend`'s shape; `_backend_eye`/`_backend_zeros`,
                # `_dirichlet_bc_rows!`/`_dirichlet_bc_indices!` and `_each_marked` are the
                # allocation and constraint internals a storage backend has to specialise;
                # `_kron_coeff` is the scale a
                # separable term carries, read when the extension rebuilds that sum as a
                # `Kronecker.jl` object, and `_kron_scalar` converts it to the operator's
                # element type there. None is something a user calls.
                :_csr_backend,
                :_backend_eye,
                :_backend_zeros,
                :_dirichlet_bc_rows!,
                :_dirichlet_bc_indices!,
                :_each_marked,
                :_kron_coeff,
                :_kron_scalar,
                # `_allocate_from_pattern` (BrambleMetalExt, BrambleSparseMatricesCSRExt): the
                # system-matrix allocation hook a storage backend specialises on its own sparse
                # type, same shape as `_csr_backend` above.
                :_allocate_from_pattern,
                # `ka_device`, `_launch_spmv_csr!`, `_launch_spmm_csr!` (BrambleMetalExt): the
                # device-kernel substrate seam and the device SpMV/SpMM launch hooks it
                # specialises, each also reached as `Bramble.name(...)` alongside the
                # `import Bramble: ...` above -- both forms need declaring.
                :ka_device,
                :_launch_spmv_csr!,
                :_launch_spmm_csr!,
                # `_has_device_csr_mirror`: the trait
                # that opts a device CSR matrix into the Dirichlet row kernel.
                :_has_device_csr_mirror,
                # `_scatter_position`, `_scatter_add!`, `_zero_stored!`
                # (BrambleSparseMatricesCSRExt): the row-major CSR counterparts of the CSC
                # scatter/zero primitives `bilinear_traversal.jl`/`bilinear.jl` already reach.
                :_scatter_position,
                :_scatter_add!,
                :_zero_stored!,
                # `_batch_for!`, `_batch_axis_for!`, `_batch_scatter_for!`, `_batch_dot`,
                # `_batch_dot_masked`, `_batch_bilinear_colour_sweep!`,
                # `_batch_bilinear_band_sweep!`, `_batch_linear_colour_sweep!`,
                # `_batch_linear_band_sweep!`: the
                # `Polyester.@batch` counterparts of the `CpuThreaded` sweeps and reductions in
                # `src/utils/linear_algebra.jl`, `src/assembly/bilinear_execution.jl` and
                # `src/assembly/linear.jl`, extended here rather than called.
                :_batch_for!,
                :_batch_axis_for!,
                :_batch_scatter_for!,
                :_batch_dot,
                :_batch_dot_masked,
                :_batch_bilinear_colour_sweep!,
                :_batch_bilinear_band_sweep!,
                :_batch_linear_colour_sweep!,
                :_batch_linear_band_sweep!,
                # `_threaded_replay_policy`, `_batch_bilinear_band_replay!`,
                # `_batch_bilinear_colour_replay!`:
                # opt `CpuPolyester` into the warmed-refill replay and its `Polyester.@batch`
                # counterparts of `_batch_bilinear_band_sweep!`/`_batch_bilinear_colour_sweep!`
                # above, reached instead of them once a unit's leaf can replay.
                :_threaded_replay_policy,
                :_batch_bilinear_band_replay!,
                :_batch_bilinear_colour_replay!,
                # `_batch_difference_engine!`, `_batch_average_engine!`,
                # `_batch_centered_average_engine!` (BramblePolyesterExt): the `Polyester.@batch` counterparts of the `CpuThreaded` stencil
                # engines in `src/operators/difference.jl` and
                # `src/operators/average.jl`, extended here rather than called.
                :_batch_difference_engine!,
                :_batch_average_engine!,
                :_batch_centered_average_engine!,
                # `_batch_run_bands!`: the
                # `Polyester.@batch` counterpart of `_run_bands!`'s `CpuThreaded` arm in
                # `src/operators/vector_calculus.jl`, reached by the divergence, curl
                # and strain-average engines. Unlike the three S7.2 hooks above it stays
                # generic over the band function `f` instead of naming one, extended here
                # rather than called.
                :_batch_run_bands!,
                # `_batch_broadcast!`: the
                # `Polyester.@batch` counterpart of `_threaded_broadcast!`'s `CpuThreaded` arm
                # in `src/space/vectorelement.jl`, reached by `_polyester_broadcast!`,
                # extended here rather than called.
                :_batch_broadcast!,
                # `_batch_csr_spmv!` (src/problems/semidiscrete_rhs.jl), `_batch_kron_lines!`
                # (src/assembly/kronecker.jl), `_batch_mf_bands!` (src/assembly/matrix_free.jl):
                # the same `Polyester.@batch` counterparts, one per CSR product, Kronecker line
                # sweep and matrix-free band sweep, extended here rather than called.
                :_batch_csr_spmv!,
                :_batch_kron_lines!,
                :_batch_mf_bands!,
                # `semidiscretize_rhs` (the precompile workload): called as
                # `Bramble.semidiscretize_rhs` to warm the right-hand-side closure; not public.
                :semidiscretize_rhs,
                # `Base.FastContiguousSubArray` (`_PtrAlike`, `_Rawable`): the
                # contiguous-view type a pointer-based reduction accepts and that
                # crosses the type-erased `@batch` loop as a raw pointer view; Base has
                # no public spelling for it.
                :FastContiguousSubArray,
                # `KroneckerBlockOperator`, `_kron_check_fresh`, `_kron_check_spaces`,
                # `_kron_is_fresh`, `_kron_leaves`, `_kron_reads_coef`, `resolve_form_ast`,
                # `test_space`, `trial_space` (BrambleKroneckerExt): the Bramble internals it
                # reads to turn a separable form into a Kronecker object and to refuse a
                # stale or mismatched one -- the block-operator type, the freshness and space
                # checks, the leaf and coefficient walkers over the form's AST. Not public.
                :KroneckerBlockOperator,
                :_kron_check_fresh,
                :_kron_check_spaces,
                :_kron_is_fresh,
                :_kron_leaves,
                :_kron_reads_coef,
                :resolve_form_ast,
                :test_space,
                :trial_space,
                # `LinearAlgebra.BLAS.get_config`, `LinearAlgebra.LAPACK.chklapackerror`,
                # `chkargsok`, `Base.Libc.Libdl` (BrambleKroneckerExt): the extension finds
                # the loaded LAPACK by name and looks up `zgges3_`/`cgges3_` in it, reporting
                # an error through LAPACK's own checker. None is public.
                :get_config,
                :chklapackerror,
                :chkargsok,
                :Libdl,
                # `BrambleKernelAbstractionsExt`: the stencil and
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
                # difference kernels are parametrised over (src/operators/stencil.jl,
                # src/operators/difference.jl) -- named in the `@kernel`s' own method
                # signatures, the same way `CartesianProduct` below is, and none exported.
                :GridDirection,
                :Forward,
                :Backward,
                :Centered,
                :CrossWeighted,
                :ArrayStyle,
                :BroadcastStyle,
                :Broadcasted,
                :Extruded,
                :RefValue,
                # `Base.ReshapedArray` (src/space/vectorelement.jl; `_Rawable` in
                # BramblePolyesterExt): a reshape of an `Array` crosses the type-erased
                # `@batch` loop as a raw pointer view.
                :ReshapedArray,
                :SizeUnknown,
                :eval,
                :mightalias,
                # `broadcasted` (src/space/vectorelement.jl:510): the customization hook for
                # `copyto!(dest::VectorElement, src::VectorElement)`.
                :broadcasted,
                # `instantiate`, `preprocess`, `throwdm` (src/space/vectorelement.jl): Base's
                # own pre-loop steps the threaded broadcast copyto! repeats before banding
                #.
                :instantiate,
                :preprocess,
                :throwdm,
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
                # BrambleMetalExt reaching into Bramble's own internals:
                # `_gpu_functional` is the loaded-and-functional predicate `gpu_backend`
                # dispatches on by `Val`, more specific than the stub in `src/utils/backend.jl`
                # and not itself public. `_gpu_functional_override` is the
                # `Ref` test hook `_gpu_functional` reads to fake device (un)availability
                # without redefining the method; the extension reads the same `Ref` so a test
                # can force GPU-unavailable behaviour through it too. Neither is public.
                # `metal_sparse_csr`/`metal_sparse_csc` carry docstrings
                # on their `src/utils/backend.jl` stubs, but neither is exported nor declared
                # `public` in `src/Bramble.jl`, nor documented in `docs/src/api/` -- so today
                # they are unqualified internals too, the same as the extension's other entry
                # points above. `SparseArrays.sparse!` is that package's
                # in-place `sparse`, never marked public there, and how `_allocate_from_pattern`
                # builds the host-side CSR arrays it hands to `metal_sparse_csr` -- the same
                # combiner `SparseMatrixCSC`'s own method already reaches for by the same name.
                :_gpu_functional,
                :_gpu_functional_override,
                :metal_sparse_csr,
                :metal_sparse_csc,
                Symbol("sparse!"),
                # BrambleMetalExt's device sparse placeholder types
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
                # `SparseArrays.getcolptr` (src/assembly/kronecker.jl, commit ece71258): copies a
                # factor's column pointers to the device; no public accessor exists.
                :getcolptr,
                # `Base.inferencebarrier` (src/assembly/bilinear_execution.jl): the fallback for a
                # transposed pair whose two block tuples differ in length, a case the types
                # already rule out, so the barrier keeps it from being inferred at all. In
                # BramblePolyesterExt, `_rerun_bands!`/`_rerun_scatter!` sit behind it, so a
                # split sweep's caller does not infer the host rerun over the user function.
                :inferencebarrier,
                # `ReverseDiff.record_mul!`: records the tape entry
                # for that `mul!`; ReverseDiff has no public equivalent.
                :record_mul!,
                # `Core.kwcall`: named in `precompile(Core.kwcall,
                # (...))` directives in `src/precompile/solver_sessions.jl` and
                # `src/precompile/form_sessions.jl`, caching the keyword-call entry
                # signature a REPL call dispatches to -- otherwise inlined into the
                # workload's static calls and never cached standalone. The documented
                # lowering target of a keyword call, but not declared public in `Core`.
                :kwcall,
                # `Base.deepcopy_internal` (src/mesh/mesh1d.jl): the method `deepcopy`
                # documents for a type that needs its own deep copy; a deep-copied mesh takes
                # a fresh identity there.
                :deepcopy_internal,
                # `Base.typename`, `Base.unwrap_unionall` (src/assembly/block_extract.jl): a
                # generated function rebuilds an operator wrapper around a new operand from
                # the wrapper's own type; Base has no public accessor for a type's wrapper.
                :typename,
                :unwrap_unionall
            )
        ) === nothing
    end

    # `@forward VectorElement.data (Base.size, Bramble.show)` and
    # `@forward VectorElement.space (Bramble.mesh,)` name the function they extend in full,
    # which is what the macro takes; unqualified would be a different binding.
    @testset "Self-qualified accesses declared" begin
        @test check_no_self_qualified_accesses(Bramble; ignore = (:mesh, :show)) === nothing
    end
end

end # module QualityExplicitImportsTests
