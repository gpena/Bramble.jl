using Test
using Bramble
using Bramble: spacings

# Type stability and allocation of the mesh interface.
#
# These pin properties the rest of the library depends on but no test asserted: the
# accessors return a concrete type, and reading geometry off a mesh allocates nothing.
# Both were true and both were unguarded, so a regression in either was invisible until
# it showed up as a slowdown somewhere else.
#
# @test_allocs is skipped under coverage, since the instrumentation allocates.

@testset "Mesh inference and allocation" begin
    Ωₕ1 = mesh(domain(interval(0.0, 1.0)), 64, false)
    Ωₕ2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (8, 9), (true, false))
    Ωₕ3 = mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (4, 5, 6),
        (true, false, true))
    Ωu = mesh(domain(interval(0.0, 1.0)), 32, true)     # uniform, for stepsize

    @testset "geometry accessors are type stable" begin
        for (lbl, Ωₕ) in (("1D", Ωₕ1), ("2D", Ωₕ2), ("3D", Ωₕ3))
            @testset "$lbl" begin
                @test @inferred(npoints(Ωₕ)) isa Int
                @test @inferred(npoints(Ωₕ, Tuple)) isa Tuple
                @test @inferred(hₘₐₓ(Ωₕ)) isa Float64
                @test @inferred(hₘᵢₙ(Ωₕ)) isa Float64
                @test @inferred(eltype(Ωₕ)) === Float64
                @test @inferred(dim(Ωₕ)) isa Int
            end
        end

        # per-point geometry, indexed the way each dimension expects
        @test @inferred(point(Ωₕ1, 3)) isa Float64
        @test @inferred(spacing(Ωₕ1, 3)) isa Float64
        @test @inferred(forward_spacing(Ωₕ1, 3)) isa Float64
        @test @inferred(half_spacing(Ωₕ1, 3)) isa Float64
        @test @inferred(half_point(Ωₕ1, 3)) isa Float64
        @test @inferred(cell_measure(Ωₕ1, 3)) isa Float64
        @test @inferred(spacings(Ωₕ1)) isa AbstractVector

        # the per-axis vectors, which every nD mesh answers as an NTuple of vectors
        for (Ωₕ, D) in ((Ωₕ2, 2), (Ωₕ3, 3))
            @test @inferred(spacings(Ωₕ)) isa NTuple{D, Vector{Float64}}
            @test @inferred(half_spacings(Ωₕ)) isa NTuple{D, Vector{Float64}}
            @test @inferred(cell_measures(Ωₕ)) isa NTuple{D, Vector{Float64}}
            @test @inferred(points(Ωₕ)) isa NTuple{D, Vector{Float64}}
            # the same shape as its siblings, and the entries `spacing` reports
            @test map(length, spacings(Ωₕ)) == map(length, half_spacings(Ωₕ))
            @test all(spacings(Ωₕ)[d][i] == spacing(Ωₕ(d), i)
            for d in 1:D for i in 1:npoints(Ωₕ(d)))
        end

        @test @inferred(point(Ωₕ2, CartesianIndex(2, 3))) isa NTuple{2, Float64}
        @test @inferred(cell_measure(Ωₕ2, CartesianIndex(2, 3))) isa Float64
        @test @inferred(point(Ωₕ3, CartesianIndex(2, 3, 4))) isa NTuple{3, Float64}
        @test @inferred(cell_measure(Ωₕ3, CartesianIndex(2, 3, 4))) isa Float64

        @test @inferred(locate_cell(Ωₕ1, 0.5)) isa Int
        @test @inferred(normal_vector(Ωₕ1, :left)) isa AbstractVector
        @test @inferred(stepsize(Ωu)) isa Float64
    end

    @testset "geometry accessors do not allocate" begin
        # Reading geometry is on the inner loop of every operator, so none of it may
        # allocate. All of these measured 0 B when the tests were written.
        @test_allocs point(Ωₕ1, 3)
        @test_allocs spacing(Ωₕ1, 3)
        @test_allocs forward_spacing(Ωₕ1, 3)
        @test_allocs half_spacing(Ωₕ1, 3)
        @test_allocs half_point(Ωₕ1, 3)
        @test_allocs cell_measure(Ωₕ1, 3)
        @test_allocs spacings(Ωₕ1)
        @test_allocs spacings(Ωₕ2)
        @test_allocs spacings(Ωₕ3)
        @test_allocs npoints(Ωₕ1)
        @test_allocs hₘₐₓ(Ωₕ1)
        @test_allocs hₘᵢₙ(Ωₕ1)
        @test_allocs locate_cell(Ωₕ1, 0.5)
        @test_allocs normal_vector(Ωₕ1, :left)
        @test_allocs stepsize(Ωu)

        @test_allocs point(Ωₕ2, CartesianIndex(2, 3))
        @test_allocs cell_measure(Ωₕ2, CartesianIndex(2, 3))
        @test_allocs npoints(Ωₕ2, Tuple)
        @test_allocs point(Ωₕ3, CartesianIndex(2, 3, 4))
        @test_allocs cell_measure(Ωₕ3, CartesianIndex(2, 3, 4))
    end
end
