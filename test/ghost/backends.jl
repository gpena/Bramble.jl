module BackendsTests

# A ghost space keeps the physical mesh's backend and stores what that backend chooses:
# the ghost mesh, the closure `(E, d)`, the padded matrix, the reduction map and the load
# vector are all of the backend's vector and matrix types, on every CPU policy, in 1D to
# 3D, for `Float64` and `Float32`, with every facet or named ones, and the backend survives
# `constrained`, `change_points!` and `iterative_refinement!`.

using Test
using Bramble
import Bramble
using Polyester
using SparseArrays: SparseMatrixCSC

using ..TestUtils: TEST_DIMS

const POLICIES = (Bramble.CpuSerial(), Bramble.CpuThreaded(), Bramble.CpuPolyester())

# Non-uniform meshes of 7, 6 and 5 points per axis, on [0, 1], [0, 2] and [0, 3].
function backend_mesh(be, ::Type{T}, D) where {T}
    ns = (7, 6, 5)[1:D]
    Ωd = domain(foldl(×, ntuple(d -> interval(T(0), T(d)), D)))
    Ω = mesh(Ωd, ns, ntuple(_ -> false, D); backend = be)
    move_points!(Ω)
    return Ω
end

# Squares the points on every axis, `x -> x² / x_end`, so the mesh stays non-uniform. The
# points go in as a vector on a `Mesh1D` and as a tuple of them on a `MeshnD`.
function move_points!(Ω)
    D = dim(Ω)
    xs = ntuple(d -> collect(points(Ω(d))), D)
    ys = ntuple(d -> xs[d] .^ 2 ./ xs[d][end], D)
    Bramble.change_points!(Ω, D == 1 ? only(ys) : ys)
    return ys
end

make_backend(::Type{T}, pol) where {T} = backend(
    vector_type = Vector{T}, matrix_type = SparseMatrixCSC{T, Int}, policy = pol)

# A relation on every facet of the named set, a Neumann one on each.
function neumann_relations(names)
    return u -> map(n -> n => ∂ₙ(u) => 1.0, names)
end

@testset "Ghost spaces keep the backend" begin
    for pol in POLICIES, T in (Float64, Float32), D in TEST_DIMS
        be = make_backend(T, pol)
        Ω = backend_mesh(be, T, D)
        @testset "$(nameof(typeof(pol))) $T $(D)D" begin
            for ghosts in (:boundary, (:xmin, :xmax)[1:min(D, 2)])
                W = gridspace(Ω; ghosts = ghosts)
                @test backend(mesh(W)) == be
                @test backend(W) == be
                @test backend(Bramble.padded_space(W)) == be
                @test backend(Bramble.physical_mesh(mesh(W))) == be
                E, d = Bramble.closure(W)
                @test E isa SparseMatrixCSC{T, Int}
                @test d isa Vector{T}
                @test size(E) == (npoints(mesh(W)), ndofs(W))
                @test all(iszero, d)
            end
        end
    end
end

@testset "Ghost forms store the backend's types" begin
    for pol in POLICIES, T in (Float64, Float32), D in TEST_DIMS
        be = make_backend(T, pol)
        Ω = backend_mesh(be, T, D)
        @testset "$(nameof(typeof(pol))) $T $(D)D" begin
            W = gridspace(Ω; ghosts = :boundary)
            a = form(W, W, (u, v) -> innerₕ(u, v))
            l = form(W, v -> innerₕ(x -> one(T), v))
            @test a.A_pad isa SparseMatrixCSC{T, Int}
            @test a.src isa Vector{Int}
            @test a.dst isa Vector{Int}
            @test a.colptr isa Vector{Int}
            @test a.rowval isa Vector{Int}
            @test a.coef isa Vector{T}
            @test l.F_pad isa Vector{T}
            # Assembling a ghost form on `CpuPolyester` waits for the batch state of the
            # ghost mesh (the ghost-dynamics plan, P2).
            if !(pol isa Bramble.CpuPolyester)
                A, F = assemble(a, l)
                @test A isa SparseMatrixCSC{T, Int}
                @test F isa Vector{T}
                # Only the physical nodes weigh: the load sums to the measure of the domain.
                @test sum(F) ≈ prod(ntuple(d -> T(d), D)) rtol = 100 * eps(T)
            end
        end
    end
end

@testset "constrained keeps the backend" begin
    # `constrained` on a `CpuPolyester` mesh is checked with the Polyester relation rows.
    for pol in (Bramble.CpuSerial(), Bramble.CpuThreaded()), T in (Float64, Float32),
        D in TEST_DIMS

        # 3D relation rows are slow to compile: one element type covers them.
        D == 3 && T === Float32 && continue
        be = make_backend(T, pol)
        Ω = backend_mesh(be, T, D)
        names = (:xmin, :xmax, :ymin, :ymax, :zmin, :zmax)
        for ghosts in (:boundary, names[1:2:(2D)])
            D == 3 && ghosts !== :boundary && continue
            W = gridspace(Ω; ghosts = ghosts)
            facets = ghosts === :boundary ? names[1:(2D)] : ghosts
            C = constrained(W, neumann_relations(facets))
            @test backend(C) == be
            @test backend(mesh(C)) == be
            E, d = Bramble.closure(C)
            @test E isa SparseMatrixCSC{T, Int}
            @test d isa Vector{T}
            a = form(C, C, (u, v) -> innerₕ(u, v))
            @test a.A_pad isa SparseMatrixCSC{T, Int}
            @test a.coef isa Vector{T}
            @test assemble(a) isa SparseMatrixCSC{T, Int}
        end
    end
end

@testset "Mesh edits keep the backend" begin
    for pol in POLICIES, T in (Float64, Float32), D in (1, 2)
        be = make_backend(T, pol)
        Ω = backend_mesh(be, T, D)
        W = gridspace(Ω; ghosts = :boundary)
        G = mesh(W)
        x₁ = collect(points(G(1)))
        ys = move_points!(Ω)
        # The ghost mesh re-pads from the physical points, in the backend's vector type.
        @test backend(mesh(W)) == be
        @test points(mesh(W)(1)) isa Vector{T}
        @test collect(points(mesh(W)(1))) != x₁
        y = first(ys)
        @test first(points(mesh(W)(1))) ≈ 2y[1] - y[2] rtol = 100 * eps(T)
        Bramble.iterative_refinement!(Ω)
        @test_throws ArgumentError points(mesh(W)(1))
        W2 = gridspace(Ω; ghosts = :boundary)
        @test backend(W2) == be
        @test backend(mesh(W2)) == be
        E, d = Bramble.closure(W2)
        @test E isa SparseMatrixCSC{T, Int}
        @test d isa Vector{T}
    end
end

@testset "Ghost constructors are inferable" begin
    for pol in POLICIES, T in (Float64, Float32), D in TEST_DIMS
        be = make_backend(T, pol)
        Ω = backend_mesh(be, T, D)
        @test @inferred(Bramble.GhostMesh(Ω; ghosts = :boundary)) isa Bramble.GhostMesh
        W = @inferred gridspace(Ω; ghosts = :boundary)
        @test W isa Bramble.GhostGridSpace
        @test @inferred(gridspace(Ω; ghosts = (:xmin, :xmax))) isa Bramble.GhostGridSpace
        @test @inferred(form(W, W, (u, v) -> innerₕ(u, v))) isa Bramble.GhostBilinearForm
        @test @inferred(form(W, v -> innerₕ(x -> one(T), v))) isa Bramble.GhostLinearForm
    end
end

end
