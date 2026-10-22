module GhostLevelSetSpacesTests

# `polygon_spaces(Ωₕ, :D, g)` on the mesh of a level-set domain (#834): a level-set mesh
# must behave as a polygon mesh in every path polygon spaces support (polygon_spaces.jl). Two
# curves, one per path of the grid fit: the unit disk (closure path, crossings on nodes by
# closure) and the dented egg at 10° (orbit path, non-convex, orthogonally convex), on
# jittered grids.
#
# The class of a level-set domain is orthogonally convex: a grid line meets the curve at
# most twice. An exterior node read by two interior nodes (a V-notch, a conflict node) needs a
# line meeting the curve four times near the notch, so none occurs, and the file asserts
# that. The conflict paths of polygon_spaces.jl (the per-reader (3.2) rows, the refusal of
# a matrix-free mixed form reading one) therefore have no level-set mesh to run on and are
# not repeated here.

using Test
using Random
using SparseArrays
using LinearAlgebra: dot, mul!
using Bramble
import Bramble
using Bramble: lift, polygon, polygon_spaces, set_markers!, Dcₓ, Dcᵧ

const BOX_DISK = interval(-1.25, 1.25) × interval(-1.25, 1.25)
const BOX_EGG = interval(-1.35, 1.35) × interval(-1.35, 1.35)

disk(p) = hypot(p[1], p[2]) - 1

# r = 1 + 0.12 cos θ + 0.05 cos 2θ − 0.06 exp(−(Δθ/0.25)²), Δθ the angle to the
# diagonal direction 3π/4 (wrapped), the whole curve rotated by 10°.
function dented_egg(p)
    θ = atan(p[2], p[1]) - deg2rad(10)
    Δ = mod(θ - 3π / 4 + π, 2π) - π
    return hypot(p[1], p[2]) - (1 + 0.12 * cos(θ) + 0.05 * cos(2θ) - 0.06 * exp(-(Δ / 0.25)^2))
end

function levelset_mesh(φ, box, n, seed)
    Random.seed!(seed)
    return mesh(domain(box, :D => levelset(φ)), (n, n), (false, false))
end

# (name, mesh): the closure path and the orbit path.
const MESHES = (("disk", levelset_mesh(disk, BOX_DISK, 20, 2)),
    ("dented egg", levelset_mesh(dented_egg, BOX_EGG, 40, 3)))

# Separable-additive (u_xy = 0), so (3.2) is exact, with −Δu = −4.
uq(x) = x[1]^2 + x[2]^2 + x[1] + 2x[2]
mixed(w) = Dcₓ(Dcᵧ(w)) + Dcᵧ(Dcₓ(w))
bvar(x) = 0.3 + 0.1 * x[1] * x[2]

labels(Ωₕ) = (Bramble.index_in_marker(Ωₕ, :D), Bramble.index_in_marker(Ωₕ, :D_boundary))

# The reduced system from the plain space's matrix and load, sharing no closure code: rows
# and columns of the interior nodes, the boundary columns times `g` to the load, and each
# exterior column `p = i + (sx, sy)` of row `i` as `+1` on `i + (sx, 0)` and `i + (0, sy)`
# and `−1` on `i`; any other exterior read is an error.
function oracle(Ωₕ, f_a, f_l, g)
    Wp = gridspace(Ωₕ)
    Ap = Matrix(assemble(form(Wp, Wp, f_a)))
    Fp = assemble(form(Wp, f_l))
    I, Γ = labels(Ωₕ)
    car = CartesianIndices(Bramble.npoints(Ωₕ, Tuple))
    lin = LinearIndices(car)
    gv = [Γ[k] ? g(Bramble.point(Ωₕ, car[k])) : 0.0 for k in eachindex(I)]
    B = copy(Ap)
    for i in findall(I), p in findall(.!(I .| Γ))

        a = Ap[i, p]
        iszero(a) && continue
        sx, sy = Tuple(car[p] - car[i])
        @assert abs(sx) == 1 && abs(sy) == 1
        B[i, p] = 0
        B[i, lin[car[i] + CartesianIndex(sx, 0)]] += a
        B[i, lin[car[i] + CartesianIndex(0, sy)]] += a
        B[i, i] -= a
    end
    return B[I, I], Fp[I] .- B[I, Γ] * gv[Γ]
end

@testset "Level-set spaces (#834)" begin
    @testset "$name: exact for quadratics" for (name, Ωₕ) in MESHES
        U, W = polygon_spaces(Ωₕ, :D, uq)
        I, Γ = labels(Ωₕ)
        @test count(I) > 100
        # The nodes the scheme extends are where the polygon's oblique edges cut cells.
        @test count(U.rows.extended) > 0
        @test count(U.rows.conflict) == 0
        f_a = (w, v) -> innerₕ(-1.0 * Δₕ(w) - 0.4 * mixed(w), v)
        A, F = assemble(form(U, W, f_a), form(W, v -> innerₕ(x -> -4.0, v)))
        @test size(A) == (count(I), count(I))
        uh = lift(U, A \ F).data
        uex = Rₕ(gridspace(Ωₕ), uq).data
        @test maximum(abs, uh[I] .- uex[I]) < 1e-8
        @test uh[Γ] == uex[Γ]
        X = .!(I .| Γ)
        @test count(!iszero, uh[X]) == count(U.rows.extended) > 0
        @test uh[X .& .!U.rows.extended] == zeros(count(X .& .!U.rows.extended))
    end

    @testset "$name: (3.2) at every extended node" for (name, Ωₕ) in MESHES
        U, _ = polygon_spaces(Ωₕ, :D, uq)
        I, Γ = labels(Ωₕ)
        uex = Rₕ(gridspace(Ωₕ), uq).data
        z = rand(Xoshiro(1), count(I))
        uz = lift(U, z).data
        car = CartesianIndices(Bramble.npoints(Ωₕ, Tuple))
        lin = LinearIndices(car)
        uI = zeros(length(I))
        uI[I] .= z
        n = 0
        for p in findall(U.rows.extended)
            readers = [lin[car[p] + CartesianIndex(sx, sy)]
                       for sx in (-1, 1), sy in (-1, 1)
                       if checkbounds(Bool, car, car[p] + CartesianIndex(sx, sy)) &&
                I[lin[car[p] + CartesianIndex(sx, sy)]]]
            @test length(readers) == 1
            k = readers[1]
            s = Tuple(car[p] - car[k])
            c1, c2 = lin[car[k] + CartesianIndex(s[1], 0)], lin[car[k] + CartesianIndex(0, s[2])]
            @test Γ[c1] && Γ[c2]
            @test uz[p] ≈ uex[c1] + uex[c2] - uI[k]
            n += 1
        end
        @test n == count(U.rows.extended) > 0
    end

    @testset "$name: rows match the plain-matrix oracle" for (name, Ωₕ) in MESHES
        # Δₕ, the mixed term and a variable coefficient b read exterior nodes.
        g(x) = 1 + x[1] - 2x[2] + x[1] * x[2]
        U, W = polygon_spaces(Ωₕ, :D, g)
        bₕ = Rₕ(Bramble.padded_space(U), bvar)
        bp = Rₕ(gridspace(Ωₕ), bvar)
        f_a = (w, v) -> innerₕ(-1.0 * Δₕ(w) - Dcₓ(bₕ * Dcᵧ(w)) - Dcᵧ(bₕ * Dcₓ(w)), v)
        f_p = (w, v) -> innerₕ(-1.0 * Δₕ(w) - Dcₓ(bp * Dcᵧ(w)) - Dcᵧ(bp * Dcₓ(w)), v)
        f_l = v -> innerₕ(x -> sin(x[1]) + x[2], v)
        A, F = assemble(form(U, W, f_a), form(W, f_l))
        Aoracle, Foracle = oracle(Ωₕ, f_p, f_l, g)
        @test maximum(abs, Matrix(A) .- Aoracle) < 1e-12 * maximum(abs, Aoracle)
        @test maximum(abs, F .- Foracle) < 1e-12 * maximum(abs, Foracle)
    end

    @testset "$name: warm assemble! allocates nothing" for (name, Ωₕ) in MESHES
        U, W = polygon_spaces(Ωₕ, :D, uq)
        bₕ = Rₕ(Bramble.padded_space(U), bvar)
        a = form(U, W,
            (w, v) -> innerₕ(-1.0 * Δₕ(w) - Dcₓ(bₕ * Dcᵧ(w)) - Dcᵧ(bₕ * Dcₓ(w)), v))
        l = form(W, v -> innerₕ(x -> -4.0, v))
        A, F = assemble(a, l)
        assemble!(A, F, a, l)
        @test (@allocated assemble!(A, F, a, l)) == 0
    end

    @testset "$name: matrix-free, contract" for (name, Ωₕ) in MESHES
        rng = Xoshiro(13)
        U, W = polygon_spaces(Ωₕ, :D, uq)
        a = form(U, W, (w, v) -> innerₕ(-1.0 * Δₕ(w) - 0.4 * mixed(w), v))
        l = form(W, v -> innerₕ(x -> -4.0, v))
        A = assemble(a)
        x, y = rand(rng, size(A, 2)), rand(rng, size(A, 1))
        z = similar(y)
        mul!(z, Bramble.matrix_free_operator(a), x)
        @test z ≈ A * x
        @test a(x, y) ≈ dot(y, A * x)
        @test_throws ArgumentError semidiscretize(a, l)
    end

    @testset "$name: refusals" for (name, Ωₕ) in MESHES
        U, W = polygon_spaces(Ωₕ, :D, uq)
        l = form(W, v -> innerₕ(x -> -4.0, v))
        f_a = (w, v) -> innerₕ(-1.0 * Δₕ(w) - 0.5 * mixed(w), v)
        # The trial space is no test space: the extended rows would join the interior ones.
        @test_throws ArgumentError form(U, U, f_a)
        @test_throws ArgumentError form(U, v -> innerₕ(x -> -4.0, v))
        # Dcₓ(Dcₓ(u)) reads two nodes along x, beyond the scheme's stencil.
        @test_throws ArgumentError form(U, W, (w, v) -> innerₕ(Dcₓ(Dcₓ(w)), v))
        @test_throws ArgumentError assemble(form(U, W, (w, v) -> innerₕ(-1.0 * Δₕ(w), v)),
            l; dirichlet = (:D_boundary => uq,))
        @test_throws ArgumentError form(U, W,
            (w, v) -> innerₕ(w, v) + 2.0 * innerₕ(x -> 1.0, w) * innerₕ(x -> 1.0, v))
        @test_throws ArgumentError polygon_spaces(Ωₕ, :Q, uq)
        # Composites of level-set spaces.
        fc = (w, v) -> innerₕ(-1.0 * Δₕ(w[1]) - 0.5 * mixed(w[1]), v[1]) + innerₕ(w[2], v[2])
        @test_throws ArgumentError form(U × U, W × W, fc)
        @test_throws ArgumentError form(W × W, v -> innerₕ(x -> 1.0, v[1]))
        G = gridspace(Ωₕ; ghosts = :boundary)
        @test_throws ArgumentError form(U × G, W × G, fc)
    end

    @testset "mesh refusals" begin
        L(p) = hypot(p[1], p[2]) - 1
        M(p) = hypot(p[1] - 0.2, p[2]) - 0.5
        # One level-set marker per domain, and none beside a polygon marker.
        Ω2 = domain(BOX_DISK, :A => levelset(L), :B => levelset(M))
        @test_throws ArgumentError mesh(Ω2, (20, 20), (false, false))
        Ωp = domain(BOX_DISK, :A => levelset(L),
            :B => polygon([(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)]))
        @test_throws ArgumentError mesh(Ωp, (20, 20), (false, false))
        # The polygon is not kept on the mesh: a later marking of a level set has none.
        _, Ωₕ = MESHES[1]
        Ω = domain(BOX_DISK, :D => levelset(disk))
        @test_throws ArgumentError set_markers!(Ωₕ, markers(Ω))
    end
end

end # module
