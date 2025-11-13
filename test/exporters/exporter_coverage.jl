"""
Test coverage for exporters (Priority 3)

Tests for:
- exporter_vtk.jl (0% → 60% target)

Focus on basic VTK export functionality for 2D and 3D meshes
(Note: 1D export is not supported by ExporterVTK)
"""

@testset "VTK Exporter Coverage" begin
	@testset "2D VTK Export - Basic" begin
		I = interval(0.0, 1.0)
		Ω = I × I
		X = domain(Ω)
		Mh = mesh(X, (5, 5), (false, false))
		Wh = gridspace(Mh)

		temp_dir = mktempdir()
		temp_file = "test_2d_basic"

		try
			# Create exporter
			exporter = ExporterVTK(Wh, temp_file, temp_dir * "/")

			# Add a dataset
			uₕ = element(Wh)
			Rₕ!(uₕ, x -> x[1] * x[2])
			addScalarDataset!(exporter, "solution", uₕ)

			# Save to file
			save2file(exporter)

			# Check that file was created
			@test isfile(joinpath(temp_dir, temp_file * ".vtu"))
		finally
			# Cleanup
			isdir(temp_dir) && rm(temp_dir, recursive = true)
		end
	end

	@testset "2D VTK Export - Multiple Datasets" begin
		I = interval(0.0, 1.0)
		Ω = I × I
		X = domain(Ω)
		Mh = mesh(X, (4, 4), (false, false))
		Wh = gridspace(Mh)

		temp_dir = mktempdir()
		temp_file = "test_2d_multiple"

		try
			exporter = ExporterVTK(Wh, temp_file, temp_dir * "/")

			# Add multiple datasets
			u1 = element(Wh)
			Rₕ!(u1, x -> x[1])
			addScalarDataset!(exporter, "field1", u1)

			u2 = element(Wh)
			Rₕ!(u2, x -> x[2])
			addScalarDataset!(exporter, "field2", u2)

			save2file(exporter)

			@test isfile(joinpath(temp_dir, temp_file * ".vtu"))
		finally
			isdir(temp_dir) && rm(temp_dir, recursive = true)
		end
	end

	@testset "2D VTK Export - Zero Solution" begin
		I = interval(0.0, 1.0)
		Ω = I × I
		X = domain(Ω)
		Mh = mesh(X, (3, 3), (false, false))
		Wh = gridspace(Mh)

		temp_dir = mktempdir()
		temp_file = "test_2d_zero"

		try
			exporter = ExporterVTK(Wh, temp_file, temp_dir * "/")

			u = element(Wh, 0.0)
			addScalarDataset!(exporter, "zeros", u)

			save2file(exporter)

			@test isfile(joinpath(temp_dir, temp_file * ".vtu"))
		finally
			isdir(temp_dir) && rm(temp_dir, recursive = true)
		end
	end

	@testset "3D VTK Export - Small Mesh" begin
		I = interval(0.0, 1.0)
		Ω = I × I × I
		X = domain(Ω)
		Mh = mesh(X, (2, 2, 2), (false, false, false))
		Wh = gridspace(Mh)

		temp_dir = mktempdir()
		temp_file = "test_3d_small"

		try
			exporter = ExporterVTK(Wh, temp_file, temp_dir * "/")

			u = element(Wh)
			Rₕ!(u, x -> x[1] * x[2] * x[3])
			addScalarDataset!(exporter, "solution3d", u)

			save2file(exporter)

			@test isfile(joinpath(temp_dir, temp_file * ".vtu"))
		finally
			isdir(temp_dir) && rm(temp_dir, recursive = true)
		end
	end

	@testset "VTK Exporter Helper Functions" begin
		I = interval(0.0, 1.0)
		Ω = I × I
		X = domain(Ω)
		Mh = mesh(X, (3, 3), (false, false))
		Wh = gridspace(Mh)

		temp_dir = mktempdir()

		try
			exporter = ExporterVTK(Wh, "test", temp_dir * "/")

			# Test accessor functions
			@test space(exporter) === Wh
			@test mesh(exporter) === Mh
			@test filename(exporter) == "test"
			@test dir(exporter) == temp_dir * "/"
			@test !isTimeDependent(exporter)
			@test timeLevel(exporter) == 0
		finally
			isdir(temp_dir) && rm(temp_dir, recursive = true)
		end
	end
end
