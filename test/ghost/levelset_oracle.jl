module GhostLevelSetOracleTests

# `polygon_spaces` on a level-set mesh (#834) against FGOracle.jl, as polygon_oracle.jl does
# for a polygon. The oracle shares no code with the package. Its labels come from φ: a
# node that is not a vertex of the fitted polygon is interior or exterior by the sign of φ.
# The vertices are the mesh's boundary nodes (a nearest-node snap of the curve to the grid
# would mislabel a few of them, so φ cannot name them); each is checked to lie within the
# grid's bound of φ = 0. The library row of an interior node is the oracle's row
# times its dual-cell area, so both are compared after dividing by that area.
# The coefficients a and c sit at the edge midpoints, which on a non-uniform grid are no
# nodes: the library reads them from a grid function holding at node j the value at the
# midpoint of the edge behind it. The load is `avgₕ` of a polynomial f of degree 5, which
# both quadratures integrate exactly.
#
# Positive control: the mixed term reads exterior nodes on every grid here, and the
# library differs from the oracle that reads them as 0 instead of by (3.2). The datum `g`
# is read at the vertex, which is off the curve by up to h²/4, so a `g` projected onto the
# curve would differ from the oracle's load.

using Test
using Random
using SparseArrays: SparseMatrixCSC
using Bramble
import Bramble
using Bramble: polygon_spaces, Dcₓ, Dcᵧ, D₋ₓ, D₋ᵧ, inner₊ₓ, inner₊ᵧ

isdefined(Main, :FGOracle) || Base.include(Main, joinpath(@__DIR__, "FGOracle.jl"))
const O = Main.FGOracle

const BOX = interval(-1.25, 1.25) × interval(-1.25, 1.25)

disk(p) = hypot(p[1], p[2]) - 1

# The ellipse of semi-axes 1 and 0.6, its major axis at angle θ.
function ellipse(θ)
    c, s = cos(θ), sin(θ)
    return p -> (c * p[1] + s * p[2])^2 + ((-s * p[1] + c * p[2]) / 0.6)^2 - 1
end

a_var(x) = 1 + 0.3 * x[1] * x[2]
c_var(x) = 2 + sin(x[1]) * x[2]
b_var(x) = 0.4 + 0.1 * x[1] - 0.05 * x[2]
d_var(x) = 1 + x[1] - x[2]
q_var(x) = 1 + x[1]^2
f5(x) = 1 + x[1]^2 * x[2]^3 - x[1]^5 + 2 * x[2]^4
g_var(x) = exp(x[1]) * sin(x[2]) + x[1]^3 * x[2]

levelset_mesh(φ, n, unif) = mesh(domain(BOX, :D => levelset(φ)), (n, n), unif)
lines(Ωₕ) = (collect(points(Ωₕ(1))), collect(points(Ωₕ(2))))

# |φ| / |∇φ| at p, the gradient by central differences.
function distance(φ, p)
    x, y, δ = Float64(p[1]), Float64(p[2]), 1e-6
    gx = (φ((x + δ, y)) - φ((x - δ, y))) / 2δ
    gy = (φ((x, y + δ)) - φ((x, y - δ))) / 2δ
    return abs(φ((x, y))) / hypot(gx, gy)
end

# The oracle's labels: 1 where φ < 0, −1 where φ > 0, and 0 at the vertices `Γ`, which
# must lie within `tol` of the curve.
function phi_classes(φ, xs, ys, Γ, tol)
    cls = [φ((x, y)) < 0 ? 1 : -1 for x in xs, y in ys]
    for k in Γ
        j, l = Tuple(k)
        @test distance(φ, (xs[j], ys[l])) <= tol
        cls[k] = 0
    end
    return cls
end

# The library's labels in the oracle's encoding.
function library_classes(Ωₕ)
    xs, ys = lines(Ωₕ)
    cls = fill(-1, length(xs), length(ys))
    cls[reshape(Bramble.index_in_marker(Ωₕ, :D), size(cls))] .= 1
    cls[reshape(Bramble.index_in_marker(Ωₕ, :D_boundary), size(cls))] .= 0
    return cls
end

# Exterior diagonal reads of interior nodes, and the nodes read by more than one.
function exterior_reads(cls)
    reads = Dict{CartesianIndex{2}, Int}()
    for k in findall(==(1), cls), s in ((1, 1), (1, -1), (-1, 1), (-1, -1))

        p = k + CartesianIndex(s)
        cls[p] == -1 && (reads[p] = get(reads, p, 0) + 1)
    end
    return sum(values(reads); init = 0), count(>(1), values(reads))
end

# The library's system: a and c at the midpoint of the edge behind each node, b, d and q at
# the nodes; `b === nothing` drops the mixed term, `d` and `q` likewise.
function library_system(Ωₕ, a, b, c, f, g; d = nothing, q = nothing)
    U, W = polygon_spaces(Ωₕ, :D, g)
    S = Bramble.padded_space(U)
    xs, ys = lines(Ωₕ)
    behind(x, j) = (x[max(j - 1, 1)] + x[j]) / 2
    aₕ, cₕ = element(S), element(S)
    aₕ .= vec([a((behind(xs, j), ys[l])) for j in eachindex(xs), l in eachindex(ys)])
    cₕ .= vec([c((xs[j], behind(ys, l))) for j in eachindex(xs), l in eachindex(ys)])
    bₕ = b === nothing ? nothing : Rₕ(S, b)
    dₕ = d === nothing ? nothing : Rₕ(S, d)
    qₕ = q === nothing ? nothing : Rₕ(S, q)
    function bilinear(w, v)
        t = inner₊ₓ(aₕ * D₋ₓ(w), D₋ₓ(v)) + inner₊ᵧ(cₕ * D₋ᵧ(w), D₋ᵧ(v))
        bₕ === nothing || (t += innerₕ(-1.0 * Dcₓ(bₕ * Dcᵧ(w)) - Dcᵧ(bₕ * Dcₓ(w)), v))
        dₕ === nothing || (t += innerₕ(Dcₓ(dₕ * w), v))
        qₕ === nothing || (t += innerₕ(qₕ * w, v))
        return t
    end
    fₕ = avgₕ(S, f)
    A, F = assemble(form(U, W, bilinear), form(W, v -> innerₕ(fₕ, v)))
    return SparseMatrixCSC(A), collect(F)
end

relerr(x, y) = maximum(abs, x - y) / maximum(abs, y)

# The oracle on the library's grid, its labels from φ, both systems with the library's rows
# divided by the dual-cell area, and the zero-exterior oracle.
function compare(Ωₕ, cls, a, b, c, f, g; d = nothing, q = nothing)
    xs, ys = lines(Ωₕ)
    A, F = library_system(Ωₕ, a, b, c, f, g; d = d, q = q)
    Aoracle, Foracle, int = O.fg_system(xs, ys, cls, a, b, c, f, g; d = d, q = q)
    Az = O.fg_system(xs, ys, cls, a, b, c, f, g; d = d, q = q, exterior = :zero)[1]
    ar = O.dual_areas(xs, ys, int)
    return (; A = A ./ ar, F = F ./ ar, Aoracle, Foracle, Az, reads = exterior_reads(cls),
        xs, ys, int)
end

# The vertices' distance to φ = 0: on the closure path (the disk) every crossing is a node,
# to the root finder's precision; on the orbit path (the ellipse at 30°) a vertex is within
# h²/4, h the largest line spacing.
const CASES = (("disk", disk, h -> 1e-12), ("ellipse 30°", ellipse(π / 6), h -> h^2 / 4))

# The oracle's labels on the mesh, the vertices at the bound `tol(h)` of φ = 0.
function mesh_classes(φ, Ωₕ, tol)
    xs, ys = lines(Ωₕ)
    Γ = findall(reshape(Bramble.index_in_marker(Ωₕ, :D_boundary), length(xs), length(ys)))
    h = max(maximum(diff(xs)), maximum(diff(ys)))
    return phi_classes(φ, xs, ys, Γ, tol(h))
end

@testset "Level-set spaces against the F&G oracle" begin
    @testset "$name, n = $n (seed $seed)" for (name, φ, tol) in CASES, n in (20, 40),
        seed in (1,)
        Random.seed!(seed)
        Ωₕ = levelset_mesh(φ, n, (false, false))
        cls = mesh_classes(φ, Ωₕ, tol)
        @test cls == library_classes(Ωₕ)
        r = compare(Ωₕ, cls, a_var, b_var, c_var, f5, g_var; d = d_var, q = q_var)
        @test size(r.A, 1) == count(==(1), cls)
        @test relerr(r.A, r.Aoracle) < 1e-12
        @test relerr(r.F, r.Foracle) < 1e-12
        @test r.reads[1] > 0
        # The positive control: reading the exterior as 0 is visible.
        @test relerr(r.A, r.Az) > 1e-3
        if n == 20
            u = Matrix(r.A) \ r.F
            @test relerr(u, Matrix(r.Aoracle) \ r.Foracle) < 1e-10
        end
    end

    @testset "Ellipse 30°, no mixed term" begin
        Random.seed!(3)
        Ωₕ = levelset_mesh(ellipse(π / 6), 30, (false, false))
        cls = mesh_classes(ellipse(π / 6), Ωₕ, h -> h^2 / 4)
        @test cls == library_classes(Ωₕ)
        zero_(x) = 0.0
        one_(x) = 1.0
        r = compare(Ωₕ, cls, one_, zero_, one_, f5, g_var)
        @test relerr(r.A, r.Aoracle) < 1e-12
        @test relerr(r.F, r.Foracle) < 1e-12
    end
end

end # module
