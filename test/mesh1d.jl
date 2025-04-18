import Bramble: indices, npoints, dim, spacing, half_spacing, Iterator
import Base: diff

function mesh1d_tests()
	I = interval(-1.0, 4.0)

	N = 4
	Ω = domain(I)
	Ωₕ = mesh(Ω, N, true)

	@test validate_equal(length(indices(Ωₕ)), N)
	@test validate_equal(npoints(Ωₕ), N)
	@test dim(Ωₕ) == 1

	h = collect(spacing(Ωₕ, Iterator))
	@test @views validate_equal(diff(h[2:N]), 0.0)

	hmed = collect(half_spacing(Ωₕ, Iterator))
	@test @views validate_equal(diff(hmed[2:(N - 1)]), 0.0)
end

mesh1d_tests()

#=
using Test
using Bramble
using Bramble: MeshMarkers, interval, domain, Backend, vector, mesh, vector_type, matrix_type, MarkerIndices, merge_consecutive_indices!
using Dictionaries

@testset "Mesh1D Tests" begin
	@testset "Mesh Creation and Properties" begin
		I = interval(0, 1)
		Ω = domain(I)
		Ωₕ = mesh(Ω, 10, true)

		@test dim(Ωₕ) == 1
		@test npoints(Ωₕ) == 10
		@test eltype(Ωₕ) == Float64
		@test eltype(typeof(Ωₕ)) == Float64
		@test npoints(Ωₕ, Tuple) == (10,)
		@test typeof(Ωₕ.markers) <: MeshMarkers{1}
		@test typeof(Ωₕ.indices) <: CartesianIndices{1}

		Ωₕ_non_uniform = mesh(Ω, 10, false)
		@test npoints(Ωₕ_non_uniform) == 10

		Ωₕ_backend = mesh(Ω, 10, true; backend = Backend(:Threads))
		@test typeof(Ωₕ_backend.backend) == Backend{(:Threads,)}
	end

	@testset "Points, Spacing, and Half-Points" begin
		I = interval(0, 1)
		Ω = domain(I)
		Ωₕ = mesh(Ω, 5, true)

		pts = points(Ωₕ)
		@test length(pts) == 5
		@test pts[1] == 0.0
		@test pts[end] == 1.0
		@test points(Ωₕ, 1) == 0.0
		@test points(Ωₕ, 5) == 1.0

		# Test Iterator interface
		pts_iter = collect(points(Ωₕ, Iterator))
		@test pts_iter == pts

		sp = spacing(Ωₕ, 2)
		@test sp == 0.25
		sp = spacing(Ωₕ, 1)
		@test sp == 0.25

		sp_iter = collect(spacing(Ωₕ, Iterator))
		@test length(sp_iter) == 5
		@test all(sp_iter .== 0.25)
		@test hₘₐₓ(Ωₕ) == 0.25

		half_sp_iter = collect(half_spacing(Ωₕ, Iterator))
		@test length(half_sp_iter) == 5
		@test half_sp_iter[1] == 0.125
		@test all(half_sp_iter[2:4] .== 0.25)
		@test half_sp_iter[5] == 0.125

		half_pts_iter = collect(half_points(Ωₕ, Iterator))
		@test length(half_pts_iter) == 6
		@test half_pts_iter[1] == 0.0
		@test half_pts_iter[2] ≈ 0.125
		@test half_pts_iter[3] ≈ 0.375
		@test half_pts_iter[4] ≈ 0.625
		@test half_pts_iter[5] ≈ 0.875
		@test half_pts_iter[6] ≈ 1.0
	end

	@testset "Cell Measure" begin
		I = interval(0, 1)
		Ω = domain(I)
		Ωₕ = mesh(Ω, 5, true)

		cell_measures = collect(cell_measure(Ωₕ, Iterator))
		@test length(cell_measures) == 5
		@test cell_measures[1] == 0.125
		@test all(cell_measures[2:4] .== 0.25)
		@test cell_measures[end] == 0.125

		for i in eachindex(indices(Ωₕ))
			idx = CartesianIndex(i)
			@test cell_measure(Ωₕ,idx) == cell_measures[i]
		end

	end

	@testset "Boundary and Interior Indices" begin
		I = interval(0, 1)
		Ω = domain(I)
		Ωₕ = mesh(Ω, 10, true)

		boundary_inds = boundary_indices(Ωₕ)
		@test boundary_inds == (CartesianIndex(1), CartesianIndex(10))

		interior_inds = interior_indices(Ωₕ)
		@test interior_inds == CartesianIndices((2:9,))
	end

	@testset "Marker Handling" begin
		I = interval(0, 1)
		Ω = domain(I, markers = ["Dirichlet"])
		Ωₕ = mesh(Ω, 10, true)

		@test length(Ωₕ.markers) == 1
		@test haskey(Ωₕ.markers, :Dirichlet)

		set_markers!(Ωₕ, Ω)
		#TODO: test that the markers have the correct indexes

	end

	 @testset "merge_consecutive_indices! (1D)" begin
		@testset "Empty Set" begin
			index_set = Set{CartesianIndex{1}}()
			indices_set = Set{CartesianIndices{1}}()
			marker_data = MarkerIndices{1}(index_set, indices_set)
			merge_consecutive_indices!(marker_data)
			@test isempty(marker_data.c_index)
			@test isempty(marker_data.c_indices)
		end

		@testset "Single Index" begin
			index_set = Set([CartesianIndex(5)])
			indices_set = Set{CartesianIndices{1}}()
			marker_data = MarkerIndices{1}(index_set, indices_set)
			merge_consecutive_indices!(marker_data)
			@test marker_data.c_index == Set([CartesianIndex(5)])
			@test isempty(marker_data.c_indices)
		end

		@testset "Consecutive Indices" begin
			index_set = Set([CartesianIndex(1), CartesianIndex(2), CartesianIndex(3)])
			indices_set = Set{CartesianIndices{1}}()
			marker_data = MarkerIndices{1}(index_set, indices_set)
			merge_consecutive_indices!(marker_data)
			@test isempty(marker_data.c_index)
			@test marker_data.c_indices == Set([CartesianIndices((1:3,))])
		end

		@testset "Mixed Indices" begin
			index_set = Set([CartesianIndex(1), CartesianIndex(2), CartesianIndex(3), CartesianIndex(5), CartesianIndex(7), CartesianIndex(8)])
			indices_set = Set{CartesianIndices{1}}()
			marker_data = MarkerIndices{1}(index_set, indices_set)
			merge_consecutive_indices!(marker_data)
			@test marker_data.c_index == Set([CartesianIndex(5)])
			@test marker_data.c_indices == Set([CartesianIndices((1:3,)), CartesianIndices((7:8,))])
		end
		 @testset "Mixed Indices and duplicated ranges" begin
			index_set = Set([CartesianIndex(1), CartesianIndex(2), CartesianIndex(3), CartesianIndex(5), CartesianIndex(7), CartesianIndex(8), CartesianIndex(7), CartesianIndex(8)])
			indices_set = Set{CartesianIndices{1}}()
			marker_data = MarkerIndices{1}(index_set, indices_set)
			merge_consecutive_indices!(marker_data)
			@test marker_data.c_index == Set([CartesianIndex(5)])
			@test marker_data.c_indices == Set([CartesianIndices((1:3,)), CartesianIndices((7:8,))])
		end

		@testset "Only Non-Consecutive Indices" begin
			index_set = Set([CartesianIndex(1), CartesianIndex(4), CartesianIndex(7)])
			indices_set = Set{CartesianIndices{1}}()
			marker_data = MarkerIndices{1}(index_set, indices_set)
			merge_consecutive_indices!(marker_data)
			@test marker_data.c_index == Set([CartesianIndex(1), CartesianIndex(4), CartesianIndex(7)])
			@test isempty(marker_data.c_indices)
		end

		@testset "only consecutive with two indexes" begin
			index_set = Set([CartesianIndex(1), CartesianIndex(2)])
			indices_set = Set{CartesianIndices{1}}()
			marker_data = MarkerIndices{1}(index_set, indices_set)
			merge_consecutive_indices!(marker_data)
			@test isempty(marker_data.c_index)
			@test marker_data.c_indices == Set([CartesianIndices((1:2,))])
		end
	end
	 @testset "merge_consecutive_indices! (nD)" begin
		 @testset "empty" begin
			marker_data = MarkerIndices{2}(Set(), Set())
			merge_consecutive_indices!(marker_data)
			@test isempty(marker_data.c_index)
			@test isempty(marker_data.c_indices)

			 marker_data = MarkerIndices{3}(Set(), Set())
			merge_consecutive_indices!(marker_data)
			@test isempty(marker_data.c_index)
			@test isempty(marker_data.c_indices)

		 end
		@testset "2D - Horizontal Sequence" begin
			index_set = Set([CartesianIndex(1, 1), CartesianIndex(1, 2), CartesianIndex(1, 3)])
			indices_set = Set{CartesianIndices{2}}()
			marker_data = MarkerIndices{2}(index_set, indices_set)
			merge_consecutive_indices!(marker_data)
			@test isempty(marker_data.c_index)
			@test marker_data.c_indices == Set([CartesianIndices((1:1, 1:3))])
		end

		@testset "2D - Vertical Sequence" begin
			index_set = Set([CartesianIndex(1, 1), CartesianIndex(2, 1), CartesianIndex(3, 1)])
			indices_set = Set{CartesianIndices{2}}()
			marker_data = MarkerIndices{2}(index_set, indices_set)
			merge_consecutive_indices!(marker_data)
			@test isempty(marker_data.c_index)
			@test marker_data.c_indices == Set([CartesianIndices((1:3, 1:1))])
		end

		 @testset "2D - Both Horizontal and Vertical Sequences" begin
			index_set = Set([
				CartesianIndex(1, 1), CartesianIndex(1, 2), CartesianIndex(1, 3),
				CartesianIndex(2, 1), CartesianIndex(3, 1), CartesianIndex(4, 1),
				CartesianIndex(3, 3)
			])
			indices_set = Set{CartesianIndices{2}}()
			marker_data = MarkerIndices{2}(index_set, indices_set)
			merge_consecutive_indices!(marker_data)
			@test marker_data.c_index == Set([CartesianIndex(3,3)])
			@test marker_data.c_indices == Set([CartesianIndices((1:1, 1:3)), CartesianIndices((2:4, 1:1))])
		end

		@testset "3D - Along X" begin
			index_set = Set([CartesianIndex(1, 1, 1), CartesianIndex(2, 1, 1), CartesianIndex(3, 1, 1)])
			indices_set = Set{CartesianIndices{3}}()
			marker_data = MarkerIndices{3}(index_set, indices_set)
			merge_consecutive_indices!(marker_data)
			@test isempty(marker_data.c_index)
			@test marker_data.c_indices == Set([CartesianIndices((1:3, 1:1, 1:1))])
		end

		 @testset "3D - Along Y" begin
			index_set = Set([CartesianIndex(1, 1, 1), CartesianIndex(1, 2, 1), CartesianIndex(1, 3, 1)])
			indices_set = Set{CartesianIndices{3}}()
			marker_data = MarkerIndices{3}(index_set, indices_set)
			merge_consecutive_indices!(marker_data)
			@test isempty(marker_data.c_index)
			@test marker_data.c_indices == Set([CartesianIndices((1:1, 1:3, 1:1))])
		end
		@testset "3D - Along Z" begin
			index_set = Set([CartesianIndex(1, 1, 1), CartesianIndex(1, 1, 2), CartesianIndex(1, 1, 3)])
			indices_set = Set{CartesianIndices{3}}()
			marker_data = MarkerIndices{3}(index_set, indices_set)
			merge_consecutive_indices!(marker_data)
			@test isempty(marker_data.c_index)
			@test marker_data.c_indices == Set([CartesianIndices((1:1, 1:1, 1:3))])
		end
	 end
end

=#
