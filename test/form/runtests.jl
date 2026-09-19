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
    include("coefficient_shift.jl")
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
    # (S4.1), the dependency-free Kronecker operator (S5.1), and composite trial/test
    # functions through the symbolic `∇ₕ`/`εₕ`/`divₕ` builders (S6.5). None needs a weak
    # dependency, so all three run with the rest of this subsystem rather than behind the
    # `ext` group.
    include("bandwidth.jl")
    include("kronecker.jl")
    include("vector_calculus.jl")
end
