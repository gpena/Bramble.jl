module GhostKernelTests

# `Bramble.ghost_kernel(A, Wₕ)` against the dense spectrum of `A`: the reference nullity
# counts the singular values below 1e-14 σ₁ (the default `rank` tolerance miscounts on 3D
# meshes), and a basis of the kernel must have that many independent columns with
# `A Z = 0`, zero at every physical degree of freedom, each column on the ghosts of a
# single node. Random and graded meshes only.

using Test
using Bramble
import Bramble
using LinearAlgebra: norm, rank, svdvals
using SparseArrays: SparseMatrixCSC, nnz, rowvals, nzrange
using ..TestUtils: WITH_3D_TESTS

const Δform = (u, v) -> innerₕ(Δₕ(u), Δₕ(v)) + innerₕ(u, v)

function reference_nullity(A)
    s = svdvals(Matrix(A))
    return count(<(1e-14 * s[1]), s)
end

function random_mesh(seed, ranges, n)
    Bramble._seed_mesh1d_rng!(seed)
    try
        return mesh(domain(foldl(×, ranges)), n, ntuple(_ -> false, length(n)))
    finally
        Bramble._unseed_mesh1d_rng!()
    end
end

# The points `x^1.5` on [0, 1] and their double on [0, 2].
function graded_mesh_2d(n)
    Ω = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), n, (true, true))
    Bramble.change_points!(Ω, (collect(range(0.0, 1.0; length = n[1])) .^ 1.5,
        2 .* collect(range(0.0, 1.0; length = n[2])) .^ 1.5))
    return Ω
end

# `Z` is a basis of the whole kernel of `A`, ghost-only (zero on the `np` physical nodes).
function is_ghost_kernel_basis(A, Z, np, k)
    Z isa SparseMatrixCSC || return false
    size(Z) == (size(A, 2), k) || return false
    k == 0 && return true
    return norm(A * Z, Inf) <= 1e-12 * norm(A, Inf) * norm(Z, Inf) &&
           rank(Matrix(Z)) == k && iszero(Z[1:np, :]) &&
           all(j -> norm(Z[:, j]) ≈ 1, 1:k)
end

# On an unconstrained ghost space DOF `np + g` is ghost `g`: every column of `Z` sits on
# the ghosts of one physical node.
function columns_at_one_node(Z, Wₕ, np)
    G = mesh(Wₕ)
    return all(1:size(Z, 2)) do j
        nodes = [first(Bramble.boundary_node(G, rowvals(Z)[q] - np)) for q in nzrange(Z, j)]
        allequal(nodes)
    end
end

@testset "Ghost kernel: 2D corners" begin
    for Ω in (random_mesh(11, (interval(0.0, 1.0), interval(0.0, 2.0)), (6, 7)),
        graded_mesh_2d((9, 11)))
        W = gridspace(Ω; ghosts = :boundary)
        A = assemble(form(W, W, Δform))
        k = reference_nullity(A)
        @test k == 4
        np = npoints(Ω)
        Z = Bramble.ghost_kernel(A, W)
        @test is_ghost_kernel_basis(A, Z, np, k)
        @test nnz(Z) == 2k
        @test columns_at_one_node(Z, W, np)
        @test Bramble.ghost_kernel(A, W) == Z
        # the first nonzero entry of each column is positive
        @test all(j -> Z.nzval[first(nzrange(Z, j))] > 0, 1:k)
        # a dense `A` gives the same basis
        @test Bramble.ghost_kernel(Matrix(A), W) ≈ Z
    end
end

@testset "Ghost kernel: one facet or 1D, none" begin
    Ω = random_mesh(11, (interval(0.0, 1.0), interval(0.0, 2.0)), (6, 7))
    for ghosts in ((:xmin,), (:ymax,))
        W = gridspace(Ω; ghosts = ghosts)
        A = assemble(form(W, W, Δform))
        @test reference_nullity(A) == 0
        Z = Bramble.ghost_kernel(A, W)
        @test Z isa SparseMatrixCSC && size(Z) == (size(A, 2), 0)
    end
    Ω1 = random_mesh(5, (interval(0.0, 1.0),), (9,))
    W1 = gridspace(Ω1; ghosts = :boundary)
    A1 = assemble(form(W1, W1, Δform))
    @test reference_nullity(A1) == 0
    @test size(Bramble.ghost_kernel(A1, W1)) == (11, 0)
end

@testset "Ghost kernel: two facets, one corner" begin
    Ω = graded_mesh_2d((8, 9))
    W = gridspace(Ω; ghosts = (:xmin, :ymax))
    A = assemble(form(W, W, Δform))
    k = reference_nullity(A)
    @test k == 1
    Z = Bramble.ghost_kernel(A, W)
    np = npoints(Ω)
    @test is_ghost_kernel_basis(A, Z, np, k)
    @test columns_at_one_node(Z, W, np)
    @test first(Bramble.boundary_node(mesh(W), rowvals(Z)[1] - np)) == CartesianIndex(1, 9)
end

# A form that never reads the ghosts leaves every ghost in the kernel, one column each.
@testset "Ghost kernel: ghosts never read" begin
    Ω = random_mesh(3, (interval(0.0, 1.0), interval(0.0, 2.0)), (5, 6))
    W = gridspace(Ω; ghosts = :boundary)
    A = assemble(form(W, W, (u, v) -> innerₕ(u, v)))
    np, ng = npoints(Ω), Bramble.nghosts(mesh(W))
    @test reference_nullity(A) == ng
    Z = Bramble.ghost_kernel(A, W)
    @test is_ghost_kernel_basis(A, Z, np, ng)
    @test nnz(Z) == ng
end

@testset "Ghost kernel: constrained spaces" begin
    Ω = random_mesh(17, (interval(0.0, 1.0), interval(0.0, 2.0)), (7, 6))
    W = gridspace(Ω; ghosts = :boundary)
    # :ymax keeps its ghosts as unknowns; its corners pair a kept ghost with an eliminated one
    C1 = constrained(W, u -> (:xmin => ∂ₙ(u) => 0.0, :xmax => ∂ₙ(u) => 1.0,
        :ymin => u + ∂ₙ(u) => 0.0))
    A1 = assemble(form(C1, C1, Δform))
    @test reference_nullity(A1) == 0
    Z1 = Bramble.ghost_kernel(A1, C1)
    @test size(Z1) == (Bramble.ndofs(C1), 0)
    # :xmin and :ymax keep theirs, and the corner they share carries the kernel
    C2 = constrained(W, u -> (:xmax => ∂ₙ(u) => 0.0, :ymin => ∂ₙ(u) => 0.0))
    A2 = assemble(form(C2, C2, Δform))
    k2 = reference_nullity(A2)
    @test k2 == 1
    np = npoints(Ω)
    Z2 = Bramble.ghost_kernel(A2, C2)
    @test is_ghost_kernel_basis(A2, Z2, np, k2)
    @test nnz(Z2) == 2
    # Dirichlet nodes leave the physical unknowns; the kernel stays on the kept ghosts
    C3 = constrained(W, u -> (:xmax => u => 0.0, :xmax => ∂ₙ(u) => 0.0,
        :ymin => ∂ₙ(u) => 0.0))
    A3 = assemble(form(C3, C3, Δform))
    k3 = reference_nullity(A3)
    @test k3 == 1
    Z3 = Bramble.ghost_kernel(A3, C3)
    @test is_ghost_kernel_basis(A3, Z3, np - 6, k3)
    # relations on every facet remove the kernel
    C4 = constrained(W, u -> (:xmin => ∂ₙ(u) => 0.0, :xmax => ∂ₙ(u) => 0.0,
        :ymin => ∂ₙ(u) => 0.0, :ymax => ∂ₙ(u) => 0.0))
    A4 = assemble(form(C4, C4, Δform))
    @test reference_nullity(A4) == 0
    @test size(Bramble.ghost_kernel(A4, C4)) == (np, 0)
end

@testset "Ghost kernel: refusals" begin
    Ω = random_mesh(11, (interval(0.0, 1.0), interval(0.0, 2.0)), (6, 7))
    W = gridspace(Ω; ghosts = :boundary)
    A = assemble(form(W, W, Δform))
    @test_throws ArgumentError Bramble.ghost_kernel(A[1:(end - 1), 1:(end - 1)], W)
    @test_throws ArgumentError Bramble.ghost_kernel(A[:, 1:(end - 1)], W)
    V = gridspace(Ω)
    B = assemble(form(V, V, (u, v) -> innerₕ(u, v)))
    @test_throws ArgumentError Bramble.ghost_kernel(B, V)
    @test_throws ArgumentError Bramble.ghost_kernel(B, gridspace(Ω; ghosts = ()))
end

# 3D runs nightly: the edges carry one column per interior node, the corners two each.
WITH_3D_TESTS && @testset "Ghost kernel: 3D edges and corners" begin
    Ω = random_mesh(7, (interval(0.0, 1.0), interval(0.0, 2.0), interval(0.0, 1.5)), (4, 5, 6))
    W = gridspace(Ω; ghosts = :boundary)
    A = assemble(form(W, W, Δform))
    n = (4, 5, 6)
    k = reference_nullity(A)
    @test k == 4 * sum(n .- 2) + 16
    np = npoints(Ω)
    Z = Bramble.ghost_kernel(A, W)
    @test is_ghost_kernel_basis(A, Z, np, k)
    @test columns_at_one_node(Z, W, np)
    @test nnz(Z) <= 3k
end

end
