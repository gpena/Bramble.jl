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
# unit group, so `BrambleMetalExt` is usually checked too; the other five extensions are
# reached only when the "ext"/"full" group has loaded their triggers earlier in the same
# process. Every ignore entry below was verified against all six loaded together, so the
# check is exact under "full" and a (harmless) subset of it otherwise -- an ignored name
# never encountered is not an error.
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

    @testset "Imports come from the module that owns them" begin
        @test check_all_explicit_imports_via_owners(Bramble) === nothing
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
    @testset "Non-public imports are the declared ones" begin
        @test check_all_explicit_imports_are_public(
            Bramble;
            ignore = (
                :sparse!,
                :Backend,
                :_backend_eye,
                :_backend_zeros,
                :BilinearForm,
                :LinearForm,
                :CartesianProduct,
                :trial_space,
                :_vtk_axes,
                :_vtk_data,
                :AbstractSpaceType,
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
                :_suitesparse_solve
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
