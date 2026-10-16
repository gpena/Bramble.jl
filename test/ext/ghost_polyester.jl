# test/ext/ghost_polyester.jl: ghost forms assemble on a `CpuPolyester` mesh, to the serial
# matrix and load (gpena/Bramble.jl#785).
#
# Gated like every other ext/*.jl file (test/runtests.jl only reaches this group under
# BRAMBLE_TEST_GROUP=ext or full). Standalone:
#
#   julia --project=test -t 2 -e 'using Bramble, Test; include("test/TestUtils.jl");
#     include("test/ext/ghost_polyester.jl")'
#
# A `@batch` sweep passes the padded space as `_batch_split` gives it, and each task
# rebuilds a `GhostMesh` over its meshes' walk states (src/utils/batch_split.jl), so the
# ghost methods (the corner rule, the physical facets) still apply there. The mixed and
# facet forms below change value where those methods are lost. At one thread `@batch` runs
# one chunk inline, so the split bites at 2 or more.
module GhostPolyesterTests

using Test
using Bramble
using Bramble: CpuPolyester, CpuSerial, Dcₓ, Dcᵧ, D̃ₓ, D₋ₓ, D̃ᵧ, D₋ᵧ
using LinearAlgebra: norm
using SparseArrays: nonzeros
using Polyester

using ..TestUtils: WITH_3D_TESTS

const POLICIES = (CpuSerial(), CpuPolyester())
const K = [2.0 0.5; 0.5 1.0]

# A seeded non-uniform mesh on [0, 1] × [0, 2] × [0, 3] (its first `D` axes) under `pol`.
function ghost_mesh(pol, ns::NTuple{D, Int}; seed = 11) where {D}
    Bramble._seed_mesh1d_rng!(seed)
    Ωd = domain(foldl(×, ntuple(d -> interval(0.0, Float64(d)), D)))
    Ω = mesh(Ωd, D == 1 ? only(ns) : ns, D == 1 ? false : ntuple(_ -> false, D);
        backend = backend(policy = pol))
    Bramble._unseed_mesh1d_rng!()
    return Ω
end

# `assemble(a, l)` on the space `space(Ω)` under each policy, serial first.
function pair_by_policy(ns, space, f, g; kwargs...)
    return map(POLICIES) do pol
        C = space(ghost_mesh(pol, ns))
        assemble(form(C, C, f), form(C, g); kwargs...)
    end
end

relerr(X, Y) = norm(X - Y, Inf) / norm(Y, Inf)

laplacian(u, v) = innerₕ(-Δₕ(u), v)
source(v) = innerₕ(x -> 1.0 + x[1], v)

@testset "Ghost forms on CpuPolyester (#785)" begin
    @test Threads.nthreads() >= 2

    @testset "Rebuilt mesh keeps the ghost type" begin
        W = gridspace(ghost_mesh(CpuPolyester(), (6, 7)); ghosts = (:xmin, :ymax))
        P = Bramble.padded_space(W)
        G = mesh(P)
        @test Bramble._batch_splittable(typeof(P))
        skeleton, arrays = Bramble._batch_split(P)
        @test isbits(skeleton)
        R = mesh(Bramble._batch_rebuild(skeleton, arrays))
        @test R isa Bramble.GhostMesh
        @test (R.n, R.lo, R.active, R.offset) == (G.n, G.lo, G.active, G.offset)
        @test Bramble.nghosts(R) == Bramble.nghosts(G)
        @test Bramble.physical_indices(R) == Bramble.physical_indices(G)
        @test all(d -> collect(points(R(d))) == collect(points(G(d))), 1:2)
    end

    @testset "2D Laplacian, Dirichlet relation" begin
        rel = u -> (:xmin => ∂ₙ(u) => 1.0, :xmax => u + ∂ₙ(u) => x -> x[2],
            :ymin => u => x -> x[1], :ymax => ∂ₙ(u) => 1.0)
        space = Ω -> constrained(gridspace(Ω; ghosts = (:xmin, :xmax, :ymax)), rel)
        (A0, F0), (A1, F1) = pair_by_policy((9, 11), space, laplacian, source)
        @test any(!iszero, last(Bramble.closure(space(ghost_mesh(CpuSerial(), (9, 11))))))
        @test relerr(A1, A0) <= 1e-13
        @test relerr(F1, F0) <= 1e-13
    end

    @testset "2D conormal corners" begin
        rel = u -> map(F -> F => ∂ₙ(u; K) => 1.0, (:xmin, :xmax, :ymin, :ymax))
        space = Ω -> constrained(gridspace(Ω; ghosts = :boundary), rel)
        mixed = (u, v) -> innerₕ(K[1, 1] * D̃ₓ(D₋ₓ(u)) + 2 * K[1, 2] * Dcₓ(Dcᵧ(u)) +
                                 K[2, 2] * D̃ᵧ(D₋ᵧ(u)), v)
        facet = (u, v) -> innerₕ(-Δₕ(u), v) + inner_Γ(∂ₙ(u), v)
        for f in (mixed, facet)
            (A0, F0), (A1, F1) = pair_by_policy((9, 11), space, f, source)
            @test relerr(A1, A0) <= 1e-13
            @test relerr(F1, F0) <= 1e-13
        end
    end

    @testset "Dirichlet by row replacement" begin
        rel = u -> map(F -> F => ∂ₙ(u) => 0.5, (:xmin, :xmax, :ymin, :ymax))
        space = Ω -> constrained(gridspace(Ω; ghosts = :boundary), rel)
        for symmetrize in (false, true)
            (A0, F0), (A1, F1) = pair_by_policy((9, 11), space, laplacian, source;
                dirichlet = :xmin => (x -> x[2]), symmetrize)
            @test relerr(A1, A0) <= 1e-13
            @test relerr(F1, F0) <= 1e-13
        end
    end

    @testset "assemble! refills" begin
        # Homogeneous data, so `d = 0` and the load refills on its own (`assemble!(b, l)`).
        rel = u -> (:xmin => ∂ₙ(u) => 0.0, :xmax => u + ∂ₙ(u) => 0.0,
            :ymin => ∂ₙ(u) => 0.0, :ymax => 2.0 * u + ∂ₙ(u) => 0.0)
        out = map(POLICIES) do pol
            C = constrained(gridspace(ghost_mesh(pol, (9, 11)); ghosts = :boundary), rel)
            a, l = form(C, C, laplacian), form(C, source)
            A, F = assemble(a), assemble(l)
            A0, F0 = copy(A), copy(F)
            fill!(nonzeros(A), 0.0)
            fill!(F, 0.0)
            assemble!(A, a)
            assemble!(F, l)
            return A, F, A0, F0
        end
        (A0, F0, Af0, Ff0), (A1, F1, Af1, Ff1) = out
        @test A1 == Af1
        @test F1 == Ff1
        @test relerr(A1, A0) <= 1e-13
        @test relerr(F1, F0) <= 1e-13
        @test norm(F0, Inf) > 0
    end

    @testset "1D Robin and Dirichlet relations" begin
        rel = u -> (:xmin => u + ∂ₙ(u) => 2.0, :xmax => u => 1.0)
        space = Ω -> constrained(gridspace(Ω; ghosts = :xmin), rel)
        (A0, F0), (A1, F1) = pair_by_policy((17,), space, laplacian, source)
        @test relerr(A1, A0) <= 1e-13
        @test relerr(F1, F0) <= 1e-13
    end

    WITH_3D_TESTS && @testset "3D Laplacian, Dirichlet relation" begin
        rel = u -> (:xmin => ∂ₙ(u) => 1.0, :xmax => u + ∂ₙ(u) => x -> x[3],
            :ymin => ∂ₙ(u) => 0.0, :ymax => ∂ₙ(u) => 0.0, :zmin => u => x -> x[1],
            :zmax => ∂ₙ(u) => 1.0)
        ghosts = (:xmin, :xmax, :ymin, :ymax, :zmax)
        space = Ω -> constrained(gridspace(Ω; ghosts), rel)
        (A0, F0), (A1, F1) = pair_by_policy((5, 6, 4), space, laplacian, source)
        @test relerr(A1, A0) <= 1e-13
        @test relerr(F1, F0) <= 1e-13
    end
end

end
