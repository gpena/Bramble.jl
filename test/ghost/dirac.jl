module GhostDiracTests

# A `dirac` source on a ghost space (#781) loads the physical nodes the plain space
# `gridspace(Ωₕ)` loads, in the padded index space, and reduces through the closure as any
# other load: `F = E_vᵀ (F_pad − A_pad d)`. The oracles share no ghost code: the plain
# space's load, the padded points read directly for the located cell, and the closure's own
# algebra. Ghosts are on one facet, on two that meet or not, and on every facet; the points
# sit next to a ghosted facet, next to an absent corner, on a cell edge, inside, and outside
# the domain, where the plain space extrapolates on its boundary cell and the ghost space
# must do the same, never reaching a ghost. A dot product against ghost values set to the
# sentinel 1e300 stays finite only if the padded load is zero on every ghost. `reaction`
# on ghost forms refuses naming the plain space and `lift`.

using Test
using Random
using Bramble
import Bramble
using LinearAlgebra: dot

using ..TestUtils: WITH_3D_TESTS

const SENTINEL = 1.0e300
const FACETS = (:xmin, :xmax, :ymin, :ymax, :zmin, :zmax)

# A seeded non-uniform mesh of the unit `D`-cube.
function nonuniform_mesh(ns::NTuple{D, Int}; seed) where {D}
    Random.seed!(seed)
    I = interval(0.0, 1.0)
    Ω = D == 1 ? domain(I) : domain(foldl(×, ntuple(_ -> I, D)))
    return mesh(Ω, D == 1 ? ns[1] : ns, D == 1 ? false : ntuple(_ -> false, D))
end

# Robin on each ghosted facet and Dirichlet on the others, both with nonzero data so the
# closure has an affine part `d ≠ 0` the load must subtract.
function ghost_space(Ωₕ, D::Int, ghosts)
    fs = FACETS[1:(2D)]
    gs = ghosts === :boundary ? fs : ghosts
    rel = u -> ntuple(k -> fs[k] in gs ? fs[k] => u + ∂ₙ(u) => 0.5 : fs[k] => u => 1.0, 2D)
    return constrained(gridspace(Ωₕ; ghosts), rel)
end

bilinear(u, v) = innerₕ(-1.0 * Δₕ(u), v)

# The points of a 1D or nD case: a scalar in 1D, a tuple otherwise.
as_point(x::NTuple{1}) = x[1]
as_point(x) = x

# The linear indices of the physical nodes in the padded vector, in the plain space's order.
function physical_positions(G)
    return vec(LinearIndices(Bramble.npoints(G, Tuple))[Bramble.physical_indices(G)])
end

# The located padded cell `J` of `x` has both corners physical along each axis, and holds
# `x` there unless `x` is outside, read from the padded axis points alone.
function check_located_cell(G, x::NTuple{D}) where {D}
    J = Bramble.locate_cell(G, as_point(x))
    phys = Bramble.physical_indices(G)
    ok = true
    for d in 1:D
        j = D == 1 ? J : J[d]
        p = Bramble.points(G(d))
        ax = phys.indices[d]
        ok &= j in ax && j + 1 in ax
        inside = p[first(ax)] <= x[d] <= p[last(ax)]
        inside && (ok &= p[j] <= x[d] <= p[j + 1])
    end
    return ok
end

function check_case(Ωₕ, D, ghosts, pts, strengths)
    C = ghost_space(Ωₕ, D, ghosts)
    src = length(pts) == 1 ? dirac(as_point(pts[1]), strengths[1]) :
          dirac(collect(pts), collect(strengths))
    g = v -> innerₕ(src, v)
    A, F = assemble(form(C, C, bilinear), form(C, g))
    P = Bramble.padded_space(C)
    G = Bramble.mesh(P)
    Fp = assemble(form(P, g))
    Ap = assemble(form(P, P, bilinear))
    E, d = Bramble.closure(C)
    Fl = assemble(form(gridspace(Ωₕ), g))
    phys = physical_positions(G)
    rest = setdiff(eachindex(Fp), phys)

    @test G isa Bramble.GhostMesh
    @test any(!iszero, d)
    @test all(x -> check_located_cell(G, x), pts)
    # The padded load is the plain load on the physical nodes and zero on every ghost.
    @test Fp[phys] ≈ Fl rtol = 1e-14
    @test all(iszero, Fp[rest])
    @test sum(Fp) ≈ sum(Fl) rtol = 1e-13
    w = fill(SENTINEL, length(Fp))
    wl = [sin(3k) for k in eachindex(Fl)]
    w[phys] .= wl
    @test dot(Fp, w) ≈ dot(Fl, wl) rtol = 1e-12
    # The reduced load is the closure of the padded one, lifted by `d`.
    @test F ≈ transpose(E) * (Fp - Ap * d) rtol = 1e-12
    return nothing
end

# The points of the unit `D`-cube a case loads: in the cell next to the `xmax` and `xmin`
# facets, on a cell edge (a node coordinate of `Ωₕ` along `x`), inside, and outside.
function point_sets(Ωₕ, D::Int)
    xs = Bramble.points(D == 1 ? Ωₕ : Ωₕ(1))
    n = length(xs)
    pick(a, rest...) = ntuple(d -> d == 1 ? a : rest[d - 1], D)
    near_max = pick((xs[n - 1] + 2xs[n]) / 3, 0.41, 0.63)
    near_min = pick((2xs[1] + xs[2]) / 3, 0.73, 0.29)
    edge = pick(xs[n ÷ 2], 0.47, 0.38)
    inside = pick(0.35, 0.65, 0.52)
    outside = pick(1.03, 0.55, 0.44)
    corner_out = ntuple(d -> isodd(d) ? -0.02 : 1.01, D)
    # In the cell at the `xmin` end of `x` and the far end of every other axis: next to
    # the absent corner (2D) or edge (3D) where `xmin` meets a ghosted maximum facet.
    corner_in = ntuple(D) do d
        p = Bramble.points(D == 1 ? Ωₕ : Ωₕ(d))
        m = length(p)
        d == 1 ? (2p[1] + p[2]) / 3 : (p[m - 1] + 2p[m]) / 3
    end
    return (
        (pts = (near_max,), s = (3.0,)),
        (pts = (near_min,), s = (-1.5,)),
        (pts = (edge,), s = (2.0,)),
        (pts = (outside,), s = (1.25,)),
        (pts = (corner_out,), s = (0.75,)),
        (pts = (corner_in,), s = (2.5,)),
        (pts = (near_max, edge, inside, near_min, outside), s = (3.0, 2.0, -1.0, 0.5, 1.25))
    )
end

function check_dimension(ns::NTuple{D, Int}, ghost_sets; seed) where {D}
    Ωₕ = nonuniform_mesh(ns; seed)
    for ghosts in ghost_sets, c in point_sets(Ωₕ, D)

        @testset "ghosts $ghosts, $(length(c.pts)) point(s) $(c.pts)" begin
            check_case(Ωₕ, D, ghosts, c.pts, c.s)
        end
    end
    return nothing
end

@testset "dirac on ghost spaces" begin
    @testset "1D" begin
        check_dimension((13,), ((:xmax,), (:xmin,), :boundary); seed = 1781)
    end
    @testset "2D" begin
        ghosts = ((:xmax,), (:xmin, :xmax), (:xmin, :ymax), :boundary)
        check_dimension((11, 9), ghosts; seed = 2781)
    end
    WITH_3D_TESTS && @testset "3D" begin
        ghosts = ((:xmax,), (:xmin, :xmax), (:xmin, :zmax), :boundary)
        check_dimension((6, 7, 5), ghosts; seed = 3781)
    end
    # The reaction reads plain rows the relations replaced: it refuses naming the fix.
    @testset "reaction refuses on ghost forms" begin
        Ωₕ = nonuniform_mesh((11, 9); seed = 4781)
        C = ghost_space(Ωₕ, 2, (:xmax,))
        a = form(C, C, bilinear)
        l = form(C, v -> innerₕ(dirac((0.35, 0.65), 3.0), v))
        A, F = assemble(a, l)
        u = element(C)
        u .= A \ F
        @test_throws ArgumentError Bramble.reaction(a, l, u; marker = :xmin)
        msg = try
            Bramble.reaction(a, l, u; marker = :xmin)
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("gridspace", msg) && occursin("lift", msg)
    end
end

end # module
