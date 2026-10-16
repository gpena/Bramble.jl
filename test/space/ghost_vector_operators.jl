module GhostVectorOperatorsTests

# The runtime operators on a `GhostVector` (gpena/Bramble.jl#776): each gives the values
# of the same operator on the padded grid function at the physical nodes.
#
# The reference is the operator on `lift(Wₕ, dofs)`, the padded `VectorElement`, read at
# the physical nodes through `physical_indices`: the plain engines over the padded grid,
# which share no slab code with the path under test. `CpuPolyester` is in
# test/ext/ghost_vector_polyester.jl, where Polyester is loaded.

using Test
using Bramble
import Bramble
using Bramble: CpuSerial, CpuThreaded
using Bramble: D₋ₓ, D₋ₓ!, D₊ₓ, D₊ᵧ, D₋ᵧ, Mₓ, M₊ᵧ, Dcₓ, Dcᵧ, D̃ᵧ, D̽ₓ, D̽ᵧ, Δₕ!
using LinearAlgebra: norm
using Random: MersenneTwister
using SparseArrays: SparseMatrixCSC

using ..TestUtils: WITH_3D_TESTS

const K = [2.0 1.0; 1.0 2.0]

# The seven one-axis families, allocating and in place: D₋, D₊, M, M₊, Dc, D̃, D̽.
const FAMILIES = (Bramble.backward_finite_difference, Bramble.forward_finite_difference,
    Bramble.backward_average, Bramble.forward_average, Bramble.centered_difference,
    Bramble.forward_star_difference, Bramble.cross_weighted_difference)
const BANGS = (Bramble.backward_finite_difference!, Bramble.forward_finite_difference!,
    Bramble.backward_average!, Bramble.forward_average!, Bramble.centered_difference!,
    Bramble.forward_star_difference!, Bramble.cross_weighted_difference!)

# The bytes of the third call of `f`, the first two compiling and warming it.
alloc(f::F) where {F} = (f(); f(); @allocated f())

relerr(a, b) = norm(a - b, Inf) / max(norm(b, Inf), 1)

# A seeded non-uniform mesh on [0, 1] × [0, 2] × [0, 3] (its first `D` axes) under `pol`.
function ghost_mesh(pol, ns::NTuple{D, Int}; T = Float64, seed = 11) where {D}
    Bramble._seed_mesh1d_rng!(seed)
    be = backend(vector_type = Vector{T}, matrix_type = SparseMatrixCSC{T, Int},
        policy = pol)
    Ωd = domain(foldl(×, ntuple(d -> interval(T(0), T(d)), D)))
    Ω = mesh(Ωd, D == 1 ? only(ns) : ns, D == 1 ? false : ntuple(_ -> false, D);
        backend = be)
    Bramble._unseed_mesh1d_rng!()
    return Ω
end

function conormal2(u)
    (:xmin => ∂ₙ(u; K) => 1.0, :xmax => u + ∂ₙ(u; K) => x -> x[2],
        :ymin => ∂ₙ(u; K) => 0.0, :ymax => 2.0 * u + ∂ₙ(u; K) => 1.0)
end
function robin2(u)
    (:xmin => u + ∂ₙ(u) => 1.0, :xmax => 3.0 * u + ∂ₙ(u) => x -> x[2],
        :ymin => 0.5 * u + ∂ₙ(u) => 0.0, :ymax => 2.0 * u + ∂ₙ(u) => 1.0)
end
# `:ymin` has no ghosts, its Dirichlet datum fixes its nodes.
function dirichlet2(u)
    (:xmin => ∂ₙ(u) => 1.0, :xmax => ∂ₙₙ(u) => x -> x[1],
        :ymin => u => x -> 1.0 + x[1], :ymax => ∂ₙ(u) => 1.0)
end
robin1(u) = (:xmin => u + ∂ₙ(u) => 2.0, :xmax => ∂ₙ(u) => x -> x[1])
function robin3(u)
    (:xmin => ∂ₙ(u) => 1.0, :xmax => u + ∂ₙ(u) => x -> x[3],
        :ymin => ∂ₙ(u) => 0.0, :ymax => ∂ₙₙ(u) => 0.0, :zmin => u => x -> x[1],
        :zmax => ∂ₙ(u) => 1.0)
end

# The physical positions in the padded grid of `G`, in the physical linear order.
function physical_positions(G)
    pl = LinearIndices(npoints(G, Tuple))
    return vec([pl[J] for J in Bramble.physical_indices(G)])
end

# A `GhostVector` on the constrained space `C` holding random degrees of freedom, its ghosts
# filled by `update_ghosts!`, and the padded grid function `lift(C, u)` they come from.
function constrained_operand(C)
    u = rand(MersenneTwister(1234), eltype(C), Bramble.ndofs(C))
    U = Bramble.lift(C, u)
    ip = physical_positions(mesh(C))
    v = update_ghosts!(GhostVector(C, parent(U)[ip]), C)
    return v, U, ip
end

# A `GhostVector` on the unconstrained ghost space `W` with random physical and ghost
# values, and the padded grid function of those degrees of freedom.
function free_operand(W)
    G = mesh(W)
    rng = MersenneTwister(4321)
    v = GhostVector(W, rand(rng, eltype(W), length(Bramble.physical_indices(G))))
    Bramble.ghost_values(v) .= rand(rng, eltype(W), Bramble.nghosts(G))
    return v, Bramble.lift(W, collect(v)), physical_positions(G)
end

# One family along the axis `dv`, behind a function barrier so the warm bytes are the call's.
function check_axis(f::F, f!::G, v, U, ip, out, dv::Val, tol) where {F, G}
    ref = vec(parent(f(U, dv)))[ip]
    r = @inferred f(v, dv)
    @test r isa Vector{eltype(v)}
    @test relerr(r, ref) <= tol
    @test (@inferred f!(out, v, dv)) === out
    @test out == r
    @test alloc(() -> f!(out, v, dv)) == 0
    return nothing
end

# Every operator on `v` against the same operator on the padded `U` at the physical nodes
# `ip`: the allocating form, the in-place form (bitwise the allocating one), `@inferred`,
# and 0 B warm in place.
function check_operators(v, U, ip; tol = 1e-14)
    D = Bramble.dim(v.mesh)
    T = eltype(v)
    out = zeros(T, length(ip))
    for d in 1:D, (f, f!) in zip(FAMILIES, BANGS)

        check_axis(f, f!, v, U, ip, out, Val(d), tol)
    end
    ref = vec(parent(Δₕ(U)))[ip]
    r = @inferred Δₕ(v)
    @test r isa Vector{T}
    @test relerr(r, ref) <= tol
    @test (@inferred Δₕ!(out, v)) === out
    @test out == r
    @test alloc(() -> Δₕ!(out, v)) == 0
    g, gr = (@inferred ∇ₕ(v)), ∇ₕ(U)
    if D == 1
        @test relerr(g, vec(parent(gr))[ip]) <= tol
    else
        @test all(k -> relerr(g[k], vec(parent(gr[k]))[ip]) <= tol, 1:D)
    end
    return nothing
end

constrained_on(pol, ns, ghosts, rel; kwargs...) = constrained(
    gridspace(ghost_mesh(pol, ns; kwargs...); ghosts), rel)

@testset "operators on a GhostVector (#776)" begin
    pol = CpuSerial()
    @testset "2D Neumann, Robin, conormal" begin
        check_operators(constrained_operand(constrained_on(pol, (6, 7), :boundary,
            u -> (:xmin => ∂ₙ(u) => 1.0, :xmax => u + ∂ₙ(u) => x -> x[2],
                :ymin => ∂ₙ(u) => 0.0, :ymax => 2.0 * u + ∂ₙ(u) => 1.0)))...)
        check_operators(constrained_operand(constrained_on(pol, (6, 7), :boundary,
            robin2))...)
        for ns in ((6, 7), (13, 9))
            check_operators(constrained_operand(constrained_on(pol, ns, :boundary,
                conormal2))...)
        end
    end

    # Off the facet slabs the values are the plain operator's on the physical values,
    # bitwise; on a ghosted facet they read the ghosts and differ from its truncation.
    @testset "the ghosts reach the facet slabs" begin
        C = constrained_on(pol, (6, 7), :boundary, robin2)
        v, _, _ = constrained_operand(C)
        uₕ = element(gridspace(Bramble.physical_mesh(mesh(C))), Bramble.physical(v))
        on_x = vec([I[1] in (1, 6) for I in CartesianIndices((6, 7))])
        on_y = vec([I[2] in (1, 7) for I in CartesianIndices((6, 7))])
        for (op, slab) in ((D₋ₓ, on_x), (Mₓ, on_x), (D₊ᵧ, on_y), (M₊ᵧ, on_y),
            (Dcₓ, on_x), (D̃ᵧ, on_y), (D̽ₓ, on_x), (Δₕ, on_x .| on_y))
            r, plain = op(v), parent(op(uₕ))
            @test r[.!slab] == plain[.!slab]
            @test r[slab] != plain[slab]
        end
    end

    @testset "named facets only, the rest truncate" begin
        W = gridspace(ghost_mesh(pol, (6, 7)); ghosts = (:xmin, :ymax))
        check_operators(free_operand(W)...)
        v, _, _ = free_operand(W)
        uₕ = element(gridspace(Bramble.physical_mesh(mesh(W))), Bramble.physical(v))
        # `xmax` and `ymin` carry no ghosts, so the forward x and backward y slabs truncate
        @test D₊ₓ(v) == parent(D₊ₓ(uₕ))
        @test D₋ᵧ(v) == parent(D₋ᵧ(uₕ))
        @test Dcₓ(v)[6:6:42] == zeros(7)
    end

    @testset "Dirichlet u => g, ∂ₙₙ" begin
        check_operators(constrained_operand(constrained_on(pol, (6, 7),
            (:xmin, :xmax, :ymax), dirichlet2))...)
    end

    @testset "an axis of two points" begin
        W = gridspace(ghost_mesh(pol, (6, 2)); ghosts = :boundary)
        check_operators(free_operand(W)...)
        Wx = gridspace(ghost_mesh(pol, (6, 2)); ghosts = (:xmin, :xmax))
        v, _, _ = free_operand(Wx)
        @test_throws ArgumentError Dcᵧ(v)
        @test_throws ArgumentError D̽ᵧ(v)
    end

    @testset "1D Robin" begin
        check_operators(constrained_operand(constrained_on(pol, (17,), :boundary,
            robin1))...)
        check_operators(free_operand(gridspace(ghost_mesh(pol, (9,)); ghosts = (:xmax,)))...)
    end

    @testset "Float32" begin
        C = constrained_on(pol, (6, 7), :boundary, conormal2; T = Float32)
        check_operators(constrained_operand(C)...; tol = 1e-5)
    end

    # nightly only: 3D meshes are left out of the unit group (#650)
    WITH_3D_TESTS && @testset "3D Robin, ∂ₙₙ, Dirichlet" begin
        check_operators(constrained_operand(constrained_on(pol, (5, 6, 4),
            (:xmin, :xmax, :ymin, :ymax, :zmax), robin3))...)
        check_operators(free_operand(gridspace(ghost_mesh(pol, (4, 5, 6));
            ghosts = (:xmax, :ymin, :zmin)))...)
    end
end

@testset "GhostVector operator refusals (#776)" begin
    Ω = ghost_mesh(CpuSerial(), (6, 7))
    C = constrained(gridspace(Ω; ghosts = :boundary), conormal2)
    v, _, _ = constrained_operand(C)
    r = D₋ₓ(v)
    # a destination on the same mesh: a GhostVector's physical part, or a grid function
    w = GhostVector(C, zeros(42))
    @test D₋ₓ!(w, v) === w
    @test Bramble.physical(w) == r
    uₕ = element(gridspace(Ω))
    @test D₋ₓ!(uₕ, v) === uₕ
    @test parent(uₕ) == r
    # another mesh with the same sizes
    C2 = constrained(gridspace(ghost_mesh(CpuSerial(), (6, 7)); ghosts = :boundary),
        conormal2)
    @test_throws ArgumentError D₋ₓ!(GhostVector(C2, zeros(42)), v)
    @test_throws ArgumentError Δₕ!(GhostVector(C2, zeros(42)), v)
    @test_throws ArgumentError D₋ₓ!(element(gridspace(ghost_mesh(CpuSerial(), (6, 7)))), v)
    # a destination of another length, or sharing memory with v
    @test_throws ArgumentError D₋ₓ!(zeros(43), v)
    @test_throws ArgumentError Δₕ!(zeros(41), v)
    @test_throws ArgumentError D₋ₓ!(Bramble.physical(v), v)
    @test_throws ArgumentError Δₕ!(v, v)
    # a part resized after the vector was built
    z = GhostVector(C, rand(42))
    push!(Bramble.ghost_values(z), 1.0)
    @test_throws ArgumentError D₋ₓ(z)
    @test_throws ArgumentError Δₕ(z)
    # a direction past the mesh's dimension
    @test_throws ArgumentError Bramble.backward_finite_difference(v, Val(3))
end

# The warm bytes of each operator on a `GhostVector` and of its plain counterpart on
# `gridspace(Ωₕ)`, on a `CpuThreaded` mesh.
function threaded_figures(ns)
    Ω = ghost_mesh(CpuThreaded(), ns)
    C = constrained(gridspace(Ω; ghosts = :boundary), conormal2)
    v, _, _ = constrained_operand(C)
    np = prod(ns)
    out = zeros(np)
    W = gridspace(Ω)
    uₕ, outₕ = element(W, rand(np)), element(W)
    ghost = map(f! -> alloc(() -> f!(out, v, Val(2))), BANGS)
    plain = map(f! -> alloc(() -> f!(outₕ, uₕ, Val(2))), BANGS)
    return (; ghost, plain, Δ = alloc(() -> Δₕ!(out, v)),
        Δplain = alloc(() -> Δₕ!(outₕ, uₕ)), spawn = alloc(() -> D₋ₓ!(outₕ, uₕ)))
end

@testset "GhostVector operators, CpuThreaded" begin
    @test Threads.nthreads() >= 2
    for (ns, ghosts, rel) in (((9, 11), :boundary, conormal2),
        ((9, 11), (:xmin, :xmax, :ymax), dirichlet2), ((17,), :boundary, robin1))
        vs, _, _ = constrained_operand(constrained_on(CpuSerial(), ns, ghosts, rel))
        Ct = constrained_on(CpuThreaded(), ns, ghosts, rel)
        vt = update_ghosts!(GhostVector(Ct, copy(Bramble.physical(vs))), Ct)
        for d in 1:length(ns), f in FAMILIES

            @test f(vt, Val(d)) == f(vs, Val(d))
        end
        @test Δₕ(vt) == Δₕ(vs)
    end

    # A one-axis operator is one band pass, as its plain counterpart is; `Δₕ!` is the plain
    # serial engine and one band pass for its facets. Each pass spawns its tasks once, its
    # closure capturing the ghost buffer and the facet data besides; nothing grows with the
    # grid.
    @testset "CpuThreaded bound" begin
        f1 = threaded_figures((9, 11))
        f2 = threaded_figures((30, 33))
        @test f1 == f2
        @test all(f1.ghost .<= f1.plain .+ 256)
        @test f1.Δ <= f1.Δplain + f1.spawn + 256
    end
end

end
