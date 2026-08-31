if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    test_dir = @__DIR__
    bramble_dir = abspath(joinpath(test_dir, "../"))
    Pkg.activate(joinpath(test_dir, "."))
    Pkg.develop(path = bramble_dir)
    Pkg.instantiate()
end

using Test
using Bramble

@inline function alloc_test(f::F, args...) where {F}
    f(args...) # warm up
    return @allocated(f(args...))
end

# Allocation test helper: uses a function barrier to avoid @testset closure boxing.
#
# These run under code coverage as well. They used to skip there, on the assumption that
# the instrumentation would perturb the counts, and that assumption cost the suite its
# allocation guarantees exactly where they were most useful: CI runs with coverage, so
# every one of them was skipped there and only ever checked by hand locally.
#
# Measured on Julia 1.12 with --code-coverage=user, the counts are identical either way,
# and the whole suite passes with 0 failures. If a future Julia does perturb them, these
# will fail rather than quietly not run, which is the outcome to prefer: the boxing
# regressions this suite exists to catch are invisible to both JET's optimisation analysis
# and AllocCheck — a reproduction of the original bug allocating 23,824 B against 0 B for
# the fix draws no report from either — so a runtime count is the only thing that sees
# them.
macro test_allocs(call_expr)
    if Meta.isexpr(call_expr, :call)
        fn = call_expr.args[1]
        args = call_expr.args[2:end]
        quote
            @test alloc_test($(esc(fn)), $(map(esc, args)...)) == 0
        end
    else
        quote
            let
                $(esc(call_expr))
                @test (@allocated $(esc(call_expr))) == 0
            end
        end
    end
end

const __bramble_with_examples = false
const __bramble_test_group = get(ENV, "BRAMBLE_TEST_GROUP", "all")
const __bramble_with_quality = __bramble_test_group in ("all", "quality", "full")
const __bramble_with_unit_tests = __bramble_test_group in ("all", "unit", "full")

# The differentiation backend survey is split by what it costs, measured per backend:
#
#   ForwardDiff 0.3 s   ReverseDiff 0.5 s   PolyesterForwardDiff 0.6 s
#   Mooncake   25.1 s   Enzyme     33.2 s
#
# So the three cheap ones run with the unit tests — 3.3 s between them, load included —
# and the two that spend almost a minute compiling on first call live behind this group.
# What they establish changes when a *backend* changes rather than when Bramble does, so
# paying that per push would be paying it for nothing almost every time. The weekly
# workflow runs `full`, which is this plus everything else.
#
# Either file skips a backend absent from the environment, so running a group without one
# installed reports a skip rather than an error.
const __bramble_with_ad_backends = __bramble_test_group in ("ad", "full")

if __bramble_with_unit_tests
    @testset verbose=true "Core library" begin
        @testset verbose=true "Utilities" begin
            include("utils/macros.jl")
            include("utils/backends.jl")
            include("utils/linear_algebra.jl")
            include("utils/bramble_functions.jl")
        end

        @testset verbose=true "Sets and Domains" begin
            include("geometry/sets.jl")
            include("geometry/domains.jl")
        end

        @testset verbose=true "Meshes" begin
            include("mesh/mesh1d.jl")
            include("mesh/meshnd.jl")
            include("mesh/meshes.jl")
            include("mesh/inference_allocation.jl")
        end

        @testset verbose=true "Grid spaces" begin
            include("space/gridspaces.jl")
            include("space/vector_elements.jl")
        end

        @testset verbose=true "Operators" begin
            include("space/difference.jl")
            include("space/star_difference.jl")
            include("space/centered_difference.jl")
            include("space/cross_weighted_difference.jl")
            include("space/commutation.jl")
            include("space/jump.jl")
            include("space/average.jl")
            include("space/inplace_operators.jl")
            include("space/inner_product.jl")
            include("space/composite_operators.jl")
            include("space/inference_allocation.jl")
            include("space/convergence.jl")
            include("space/element_type.jl")
            include("space/autodiff.jl")
            include("space/autodiff_backends.jl")
        end

        @testset verbose=true "Forms" begin
            include("form/dirichlet_constraints.jl")
            include("form/difference_ast.jl")
            include("form/operators.jl")
            include("form/inner_products.jl")
            include("form/linear.jl")
            include("form/extended_operators.jl")
            include("form/symmetrize.jl")
            include("form/autodiff.jl")
            include("form/common.jl")
            include("form/block_extract.jl")
            include("form/stencil_pattern.jl")
        end

        #=
		@testset "Forms" begin
			include("form/dirichlet_constraints.jl")
			include("form/grid_coloring.jl")
			include("form/forms.jl")
			include("form/linear_forms.jl")
			include("form/bilinear_forms.jl")
			include("form/composite_forms.jl")
		end
=#
        #=@testset "Exporters" begin
			include("exporters/exporter_coverage.jl")
		end=#
    end
end

if __bramble_with_examples
    include("examples.jl")
end

if __bramble_with_quality
    @testset verbose=true "\nQuality" begin
        include("quality/aqua.jl")
        include("quality/exports.jl")
        include("quality/jet.jl")
    end
end

if __bramble_with_ad_backends
    @testset verbose=true "AD backends (expensive)" begin
        # autodiff_backends.jl first: it defines `check_backend` and `_have`, which this
        # reuses so both files check every backend the same way.
        __bramble_with_unit_tests || include("space/autodiff_backends.jl")
        include("space/autodiff_heavy.jl")
    end
end
