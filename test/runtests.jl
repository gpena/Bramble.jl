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
const __bramble_with_quality = __bramble_test_group in ("all", "quality")
const __bramble_with_unit_tests = __bramble_test_group in ("all", "unit")

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
            include("space/buffers.jl")
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
            include("space/inner_product.jl")
            include("space/composite_operators.jl")
            include("space/inference_allocation.jl")
            include("space/convergence.jl")
            include("space/element_type.jl")
            include("space/autodiff.jl")
        end

        @testset verbose=true "Forms" begin
            include("form/dirichlet_constraints.jl")
            include("form/difference_ast.jl")
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
