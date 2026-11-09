module GhostPolygonCompositeTests

# Composites of `polygon_spaces` (#1162): `U1 × U2`, `U^Val(n)` and `U^N` of the trial
# spaces, tested by the same power of the test space, on the jittered Figure 1, the V-notch
# (a conflict node) and the level-set disk and dented egg. The reduced system is checked
# block by block against an oracle sharing no closure code: the block `(t, c)` of the plain
# composite's matrix (`gridspace(Ωₕ) × gridspace(Ωₕ)`, or `^Val(3)`) keeps the interior
# rows and columns, moves its boundary columns to the load of component `t` through
# component `c`'s own `g`, and replaces each exterior column of a tested row by that row's
# own (3.2) row in the columns of component `c`. The cross blocks read diagonal exterior
# nodes (`Dcₓ(Dcᵧ(u(1)))` tested on `v(2)`), so a conflict move written into the columns
# of the test component, not of the trial one, breaks the V-notch rows.

using Test
using Random
using LinearAlgebra: dot
using Bramble
import Bramble
using Bramble: lift, polygon, polygon_spaces, Dcₓ, Dcᵧ

const FIG1 = [(0.0, 2.0), (2.0, 0.0), (4.0, 0.0), (6.0, 2.0), (6.0, 3.0), (4.0, 3.0),
    (4.0, 5.0), (2.0, 5.0), (0.0, 3.0)]
const VNOTCH = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0), (0.0, 3.0), (1.0, 2.0),
    (0.0, 1.0)]
const BOX_DISK = interval(-1.25, 1.25) × interval(-1.25, 1.25)
const BOX_EGG = interval(-1.35, 1.35) × interval(-1.35, 1.35)

disk(p) = hypot(p[1], p[2]) - 1

# The dented egg of levelset_spaces.jl: non-convex, orthogonally convex, rotated by 10°.
function dented_egg(p)
    θ = atan(p[2], p[1]) - deg2rad(10)
    Δ = mod(θ - 3π / 4 + π, 2π) - π
    return hypot(p[1], p[2]) - (1 + 0.12 * cos(θ) + 0.05 * cos(2θ) - 0.06 * exp(-(Δ / 0.25)^2))
end

function polygon_mesh(V, box, n, unif)
    Random.seed!(2)
    return mesh(domain(box, :P => polygon(V)), n, unif)
end

function levelset_mesh(φ, box, n, seed)
    Random.seed!(seed)
    return mesh(domain(box, :D => levelset(φ)), (n, n), (false, false))
end

# (name, mesh, marker, whether a conflict node is read)
const MESHES = (
    ("Figure 1", polygon_mesh(FIG1, interval(0.0, 6.0) × interval(0.0, 5.0), (10, 9),
            (false, false)), :P, false),
    ("V-notch", polygon_mesh(VNOTCH, interval(0.0, 4.0) × interval(0.0, 4.0), (5, 5),
            (true, true)), :P, true),
    ("disk", levelset_mesh(disk, BOX_DISK, 20, 2), :D, false),
    ("dented egg", levelset_mesh(dented_egg, BOX_EGG, 40, 3), :D, false))

g1(x) = 1 + x[1] - 2x[2] + x[1] * x[2]
g2(x) = x[1]^2 - x[2]
bvar(x) = 0.3 + 0.1 * x[1] * x[2]
mixed(w) = Dcₓ(Dcᵧ(w)) + Dcᵧ(Dcₓ(w))

# Variable coefficients `b`, and cross blocks reading the diagonal exterior nodes: (2, 1)
# of `coupled2`, (1, 3) and (3, 1) of `coupled3`.
function coupled2(b)
    a11(w) = -1.0 * Δₕ(w) - Dcₓ(b * Dcᵧ(w)) - Dcᵧ(b * Dcₓ(w))
    (u, v) -> innerₕ(a11(u(1)), v(1)) + innerₕ(b * Dcₓ(u(2)), v(1)) +
              innerₕ(Dcₓ(Dcᵧ(u(1))), v(2)) +
              innerₕ(-1.0 * Δₕ(u(2)) - 0.5 * mixed(u(2)), v(2))
end

function coupled3(b)
    (u, v) -> innerₕ(-1.0 * Δₕ(u(1)) - Dcₓ(b * Dcᵧ(u(1))), v(1)) +
              innerₕ(Dcᵧ(Dcₓ(u(3))), v(1)) + innerₕ(-1.0 * Δₕ(u(2)) + u(2), v(2)) +
              innerₕ(b * Dcₓ(Dcᵧ(u(1))), v(3)) +
              innerₕ(-1.0 * Δₕ(u(3)) - 0.5 * mixed(u(3)), v(3))
end
load2 = v -> innerₕ(x -> sin(x[1]) + x[2], v(1)) + innerₕ(x -> x[1] * x[2], v(2))
load3 = v -> load2(v) + innerₕ(x -> cos(x[2]), v(3))

labels(Ωₕ, L) = (Bramble.index_in_marker(Ωₕ, L),
    Bramble.index_in_marker(Ωₕ, Symbol(L, :_boundary)))

# A block of the plain matrix, each exterior column `p = i + (sx, sy)` of a tested row `i`
# moved as `+1` on `i + (sx, 0)` and `i + (0, sy)` and `−1` on `i`, columns of the same
# block; any other exterior read is an error.
function substitute(Ωₕ, L, Bp)
    I, Γ = labels(Ωₕ, L)
    car = CartesianIndices(Bramble.npoints(Ωₕ, Tuple))
    lin = LinearIndices(car)
    B = copy(Bp)
    for i in findall(I), p in findall(.!(I .| Γ))

        a = Bp[i, p]
        iszero(a) && continue
        sx, sy = Tuple(car[p] - car[i])
        @assert abs(sx) == 1 && abs(sy) == 1
        B[i, p] = 0
        B[i, lin[car[i] + CartesianIndex(sx, 0)]] += a
        B[i, lin[car[i] + CartesianIndex(0, sy)]] += a
        B[i, i] -= a
    end
    return B
end

# The reduced system from the plain composite `P`'s matrix and load, block by block: the
# substituted block `(t, c)` on the interior nodes, and `F_t − Σ_c B_tc[I, Γ] g_c[Γ]`. Also
# the plain blocks with the exterior columns dropped, the positive control.
function oracle(Ωₕ, L, P, f_a, f_l, gs)
    Ap = assemble(form(P, P, f_a))
    Fp = assemble(form(P, f_l))
    I, Γ = labels(Ωₕ, L)
    car = CartesianIndices(Bramble.npoints(Ωₕ, Tuple))
    N, m = length(car), length(gs)
    gv = [[Γ[k] ? g(Bramble.point(Ωₕ, car[k])) : 0.0 for k in 1:N] for g in gs]
    blk(t) = ((t - 1) * N + 1):(t * N)
    Bp = [Matrix(Ap[blk(t), blk(c)]) for t in 1:m, c in 1:m]
    B = [substitute(Ωₕ, L, Bp[t, c]) for t in 1:m, c in 1:m]
    A = reduce(vcat, [reduce(hcat, [B[t, c][I, I] for c in 1:m]) for t in 1:m])
    Z = reduce(vcat, [reduce(hcat, [Bp[t, c][I, I] for c in 1:m]) for t in 1:m])
    F = reduce(vcat,
        [Fp[blk(t)][I] .- sum(B[t, c][I, Γ] * gv[c][Γ] for c in 1:m) for t in 1:m])
    return A, F, Z
end

relerr(A, B) = maximum(abs, Matrix(A) .- B) / maximum(abs, B)

# A function barrier, as the testset's closure would box the arguments.
function warm_bytes(A, F, a, l)
    assemble!(A, F, a, l)
    return @allocated assemble!(A, F, a, l)
end

@testset "Polygon composite spaces (#1162)" begin
    @testset "rows match the plain-matrix oracle" begin
        for (name, Ωₕ, L, conflict) in MESHES
            U1, W = polygon_spaces(Ωₕ, L, g1)
            U2, _ = polygon_spaces(Ωₕ, L, g2)
            n = Bramble.ndofs(U1)
            Wp = gridspace(Ωₕ)
            b, bp = Rₕ(Bramble.padded_space(U1), bvar), Rₕ(Wp, bvar)
            @test any(U1.rows.conflict) == conflict
            for (V, T, P, f, l, gs) in (
                (U1 × U2, W × W, Wp × Wp, coupled2, load2, (g1, g2)),
                (U1^Val(2), W^Val(2), Wp × Wp, coupled2, load2, (g1, g1)),
                (U1^2, W^2, Wp × Wp, coupled2, load2, (g1, g1)),
                (U1^Val(3), W^Val(3), Wp^Val(3), coupled3, load3, (g1, g1, g1)))
                A = assemble(form(V, T, f(b)))
                Ao, _, Z = oracle(Ωₕ, L, P, f(bp), l, gs)
                @test relerr(A, Ao) < 1e-12
                # The substitution is exercised in the cross block (2, 1) or (3, 1).
                m = length(gs)
                @test Ao[((m - 1) * n + 1):(m * n), 1:n] != Z[((m - 1) * n + 1):(m * n), 1:n]
            end
        end
    end

    @testset "loads and lift per component" begin
        for (name, Ωₕ, L, _) in MESHES
            U1, W = polygon_spaces(Ωₕ, L, g1)
            U2, _ = polygon_spaces(Ωₕ, L, g2)
            n = Bramble.ndofs(U1)
            Wp = gridspace(Ωₕ)
            b, bp = Rₕ(Bramble.padded_space(U1), bvar), Rₕ(Wp, bvar)
            _, F = assemble(form(U1 × U2, W × W, coupled2(b)), form(W × W, load2))
            _, Fo, _ = oracle(Ωₕ, L, Wp × Wp, coupled2(bp), load2, (g1, g2))
            @test maximum(abs, F .- Fo) < 1e-12 * maximum(abs, Fo)
            # Each component enters with its own `g`: the swapped pair gives another load.
            _, Fs, _ = oracle(Ωₕ, L, Wp × Wp, coupled2(bp), load2, (g2, g1))
            @test maximum(abs, Fs .- Fo) > 1e-3 * maximum(abs, Fo)
            z = rand(Xoshiro(7), 2n)
            u = lift(U1 × U2, z).data
            @test u == vcat(lift(U1, z[1:n]).data, lift(U2, z[(n + 1):end]).data)
            _, Γ = labels(Ωₕ, L)
            N = length(Γ)
            @test u[(N + 1):end][Γ] == Rₕ(Wp, g2).data[Γ]
            @test u[1:N][Γ] == Rₕ(Wp, g1).data[Γ]
        end
    end

    @testset "coupled solve exact for quadratics" begin
        # Separable-additive quadratics: (3.2) and the scheme are exact, and the mixed
        # differences vanish; −Δu₁ = −4, −Δu₂ = −2.
        u1(x) = x[1]^2 + x[2]^2 + x[1] + 2x[2]
        u2(x) = 2x[1]^2 - x[2]^2 + 3x[1] - x[2]
        f_a = (u, v) -> innerₕ(-1.0 * Δₕ(u(1)), v(1)) + innerₕ(u(1) - u(2), v(1)) +
                        innerₕ(mixed(u(1)), v(2)) +
                        innerₕ(-1.0 * Δₕ(u(2)) - 0.5 * mixed(u(2)), v(2))
        f_l = v -> innerₕ(x -> -4.0 + u1(x) - u2(x), v(1)) + innerₕ(x -> -2.0, v(2))
        for (name, Ωₕ, L, conflict) in MESHES
            conflict && continue
            U1, W = polygon_spaces(Ωₕ, L, u1)
            U2, _ = polygon_spaces(Ωₕ, L, u2)
            n = Bramble.ndofs(U1)
            A, F = assemble(form(U1 × U2, W × W, f_a), form(W × W, f_l))
            z = A \ F
            I, _ = labels(Ωₕ, L)
            Wp = gridspace(Ωₕ)
            @test maximum(abs, z[1:n] .- Rₕ(Wp, u1).data[I]) < 1e-9
            @test maximum(abs, z[(n + 1):end] .- Rₕ(Wp, u2).data[I]) < 1e-9
        end
    end

    @testset "contraction" begin
        for (name, Ωₕ, L, conflict) in MESHES
            U1, W = polygon_spaces(Ωₕ, L, g1)
            U2, _ = polygon_spaces(Ωₕ, L, g2)
            b = Rₕ(Bramble.padded_space(U1), bvar)
            a = form(U1 × U2, W × W, coupled2(b))
            A = assemble(a)
            rng = Xoshiro(13)
            x, y = rand(rng, size(A, 2)), rand(rng, size(A, 1))
            if conflict
                @test_throws ArgumentError a(x, y)
                # A form reading no conflict node still contracts.
                a5 = form(U1 × U2, W × W,
                    (u, v) -> innerₕ(-1.0 * Δₕ(u(1)), v(1)) + innerₕ(u(1) - u(2), v(2)))
                @test a5(x, y) ≈ dot(y, assemble(a5) * x)
            else
                @test a(x, y) ≈ dot(y, A * x)
            end
        end
    end

    @testset "warm assemble! allocates nothing" begin
        for (name, Ωₕ, L, _) in MESHES
            U1, W = polygon_spaces(Ωₕ, L, g1)
            U2, _ = polygon_spaces(Ωₕ, L, g2)
            b = Rₕ(Bramble.padded_space(U1), bvar)
            a, l = form(U1 × U2, W × W, coupled2(b)), form(W × W, load2)
            A, F = assemble(a, l)
            @test warm_bytes(A, F, a, l) == 0
        end
    end

    @testset "refusals" begin
        for (name, Ωₕ, L, _) in MESHES
            U1, W = polygon_spaces(Ωₕ, L, g1)
            U2, _ = polygon_spaces(Ωₕ, L, g2)
            V = U1 × U2
            b = Rₕ(Bramble.padded_space(U1), bvar)
            f_a = coupled2(b)
            l = form(W × W, load2)
            # The trial space is no test space, on either side of a form.
            e = @test_throws ArgumentError form(V, V, f_a)
            @test occursin("polygon", e.value.msg)
            e = @test_throws ArgumentError form(V, load2)
            @test occursin("polygon", e.value.msg)
            # Dcₓ(Dcₓ(u)) reads two nodes along x, of its own component and of another.
            e = @test_throws ArgumentError form(V, W × W,
                (u, v) -> innerₕ(Dcₓ(Dcₓ(u(1))), v(1)) + innerₕ(u(2), v(2)))
            @test occursin("polygon", e.value.msg)
            e = @test_throws ArgumentError form(V, W × W,
                (u, v) -> innerₕ(Dcₓ(Dcₓ(u(2))), v(1)) + innerₕ(u(1), v(2)))
            @test occursin("polygon", e.value.msg)
            # Each component's boundary is fixed by its own `g`.
            e = @test_throws ArgumentError assemble(form(V, W × W, f_a), l;
                dirichlet = (Symbol(L, :_boundary) => g1,))
            @test occursin("polygon", e.value.msg)
            e = @test_throws ArgumentError Bramble.constrained(V,
                u -> (:xmin => u(1) => 0.0,))
            @test occursin("polygon", e.value.msg)
            # A polygon space ghosts no facet, a plain ghost space does.
            G = gridspace(Ωₕ; ghosts = :boundary)
            e = @test_throws ArgumentError form(U1 × G, W × G,
                (u, v) -> innerₕ(u(1), v(1)) + innerₕ(u(2), v(2)))
            @test occursin("polygon", e.value.msg)
            @test_throws ArgumentError form(V, W × W,
                (u, v) -> innerₕ(u(1), v(1)) + innerₕ(u(2), v(2)) +
                          2.0 * innerₕ(x -> 1.0, u(1)) * innerₕ(x -> 1.0, v(1)))
            @test_throws ArgumentError Bramble.matrix_free_operator(form(V, W × W, f_a))
        end
    end
end

end # module
