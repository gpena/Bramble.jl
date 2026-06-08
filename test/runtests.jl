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

const __bramble_with_examples = true
const __bramble_with_quality = false
const __bramble_with_unit_tests = true

if __bramble_with_unit_tests
	@testset verbose=true "Core library" begin
		@testset verbose=true "Backends and BrambleFunctions" begin
			include("utils/bramble_functions.jl")
			include("utils/backends.jl")
		end

		@testset verbose=true "Sets and Domains" begin
			include("geometry/sets.jl")
			include("geometry/domains.jl")
		end

		@testset verbose=true "Meshes" begin
			include("mesh/mesh1d.jl")
			include("mesh/meshnd.jl")
			include("mesh/meshes.jl")
		end

		@testset verbose=true "Grid spaces" begin
			include("space/buffers.jl")
			include("space/gridspaces.jl")
			include("space/vector_elements.jl")
			include("space/matrix_elements.jl")
			include("space/difference.jl")
			include("space/jump.jl")
			include("space/average.jl")
			include("space/inner_product.jl")
			include("space/linear_operators.jl")
		end

		@testset "Forms" begin
			include("form/dirichlet_constraints.jl")
			include("form/grid_coloring.jl")
			include("form/forms.jl")
			include("form/linear_forms.jl")
			include("form/bilinear_forms.jl")
			include("form/composite_forms.jl")
		end

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
		include("quality/jet.jl")
	end
end