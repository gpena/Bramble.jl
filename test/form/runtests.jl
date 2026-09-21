# Standalone entry point: `include("test/form/runtests.jl")` alone in a fresh session
# runs this subsystem's tests without going through the full test/runtests.jl.
isdefined(Main, :TestUtils) || include(joinpath(@__DIR__, "..", "TestUtils.jl"))

@testset "Forms" begin
    include("dirichlet_constraints.jl")
    include("reaction_flux.jl")
    include("difference_ast.jl")
    include("operators.jl")
    include("inner_products.jl")
    include("linear.jl")
    include("dirac.jl")
    include("source_operators.jl")
    # Behind `slow`: `for (nm, op) in (("D₋ₓ", D₋ₓ), ("D₊ₓ", D₊ₓ), ...)` binds `op` to a
    # `Union` of the seven operator types being swept, and each iteration runs a full
    # `form()`/`assemble()` on it -- 417ms/test, 11.7s total, against a ~65-95ms/test median
    # elsewhere in this subsystem. Same class of bug as vector_calculus.jl below: a runtime
    # loop over a heterogeneous tuple of operators forces the Union through the whole
    # assembly pipeline. A cost of how the test is written, not of the feature.
    TestUtils.WITH_SLOW_TESTS && include("coefficient_shift.jl")
    include("interpolation.jl")
    include("bilinear.jl")
    include("assemble_add.jl")
    include("cross_mesh_blocks.jl")
    include("interpolation_operator.jl")
    include("normal.jl")
    include("skew.jl")
    include("symmetry.jl")
    include("markers.jl")
    include("extended_operators.jl")
    include("symmetrize.jl")
    include("autodiff.jl")
    include("common.jl")
    include("simplifier.jl")
    include("block_extract.jl")
    include("component.jl")
    include("stencil_pattern.jl")
    # Behind the `slow` group (TestUtils.WITH_SLOW_TESTS): at ~46s it is the most expensive
    # file in this subsystem, and most of that is the structural half -- the AST-derived
    # pattern checked against SparseConnectivityTracer's AD-traced one as ground truth, in
    # 1D, 2D and 3D. That is a cross-check against another package, so it moves when that
    # package or the simplifier moves rather than when an operator does. Daily on both
    # platforms, and in any run with no group set, including this file's own standalone one.
    TestUtils.WITH_SLOW_TESTS && include("jacobian_pattern.jl")
    include("type_cached_assemble.jl")
    include("semidiscrete.jl")
    include("sparse_solvers.jl")
    # v3.3.0 plan (memory scaling): `bandwidths`/`blockbandwidths` read from the AST alone
    # (S4.1) and the dependency-free Kronecker operator (S5.1). Neither needs a weak
    # dependency, so both run with the rest of this subsystem rather than behind the `ext`
    # group.
    include("bandwidth.jl")
    include("kronecker.jl")
    # Composite trial/test functions through the symbolic `∇ₕ`/`εₕ`/`divₕ` builders (S6.5).
    # Behind `slow`: its hand-expanded comparison functions (`hand_strain`, in particular)
    # branch on `i == j` to return structurally different `LazyOp` subtrees, so Julia infers
    # their result as a `Union`; summing D² (9 in 3D) of those unioned subtrees through
    # `innerₕ`/`+`/`assemble` is what costs this file ~159s of its ~160s total -- 100%
    # compile, on meshes too small to cost anything at runtime. That is a cost of how the
    # *test* is written, not of the feature, so it is parked behind `slow` (daily on both
    # platforms) rather than paid on every push until the hand-expanded helpers are
    # rewritten to keep each term's type concrete (gpena/Bramble.jl -- compile-time issue).
    TestUtils.WITH_SLOW_TESTS && include("vector_calculus.jl")
end
