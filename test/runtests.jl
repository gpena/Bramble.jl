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

const __bramble_with_examples = false
const __bramble_with_quality = false
const __bramble_with_unit_tests = true

if __bramble_with_unit_tests
	@testset verbose=true "Core library" begin
		@testset "Backends and BrambleFunctions" begin
			include("bramblefunctions.jl")
			include("backends.jl")
		end

		@testset "Sets and Domains" begin
			include("sets.jl")
			include("domains.jl")
		end

		@testset "Meshes" begin
			include("mesh1d.jl")
			include("meshnd.jl")
		end
		#=
						@testset "Grid spaces" begin
							include("gridspaces.jl")
							include("vectorelements.jl")
							include("matrixelements.jl")
							include("operators.jl")
						end

						@testset "Forms" begin
							include("bilinearforms.jl")
						end=#
	end
end

if __bramble_with_examples
	include("examples.jl")
end

if __bramble_with_quality
	@testset verbose=true "\nQuality" begin
		include("aqua.jl")
		include("jet.jl")
	end
end
