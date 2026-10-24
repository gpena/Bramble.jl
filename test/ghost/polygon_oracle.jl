module GhostPolygonOracleTests

# `polygon_spaces` (#204) against FGOracle.jl, a hand-built system (3.1) and (3.2) of
# Ferreira and Grigorieff (2006) that shares no code with the package. The library row of
# an interior node is the oracle's row times the node's dual-cell area (the `innerₕ`
# weight), so the matrix and the load are compared after dividing each library row by that
# area. The coefficients a and c sit at the edge midpoints, which on a non-uniform grid
# are no nodes: the library reads them from a grid function holding at node j the value at
# the midpoint of the edge behind it, the edge `D₋ₓ` differences across. The load is
# `avgₕ` of a polynomial f of degree 5, which both quadratures integrate exactly.
#
# Positive controls: every grid here makes the mixed term read exterior nodes, and the
# library differs from the oracle that reads them as 0 instead of by (3.2).

using Test
using Random
using LinearAlgebra: opnorm
using Bramble
import Bramble
using Bramble: polygon, polygon_spaces, Dcₓ, Dcᵧ, D₋ₓ, D₋ᵧ, inner₊ₓ, inner₊ᵧ
using ..TestUtils: alloc_test

isdefined(Main, :FGOracle) || Base.include(Main, joinpath(@__DIR__, "FGOracle.jl"))
const O = Main.FGOracle

# Figure 1 of the paper; the V-notch, a reflex vertex at (1, 2) between two oblique edges,
# whose exterior node (0, 2) is read through the diagonals by two interior nodes.
const FIG1 = [(0.0, 2.0), (2.0, 0.0), (4.0, 0.0), (6.0, 2.0), (6.0, 3.0), (4.0, 3.0),
    (4.0, 5.0), (2.0, 5.0), (0.0, 3.0)]
const FIG1_BOX = interval(0.0, 6.0) × interval(0.0, 5.0)
const VNOTCH = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0), (0.0, 3.0), (1.0, 2.0),
    (0.0, 1.0)]
const VNOTCH_BOX = interval(0.0, 4.0) × interval(0.0, 4.0)

zero_(x) = 0.0
one_(x) = 1.0
a_var(x) = 1 + 0.3 * x[1] * x[2]
c_var(x) = 2 + sin(x[1]) * x[2]
b_var(x) = 0.4 + 0.1 * x[1] - 0.05 * x[2]
d_var(x) = 1 + x[1] - x[2]
q_var(x) = 1 + x[1]^2
f5(x) = 1 + x[1]^2 * x[2]^3 - x[1]^5 + 2 * x[2]^4
g_var(x) = exp(x[1]) * sin(x[2]) + x[1]^3 * x[2]

polygon_mesh(V, box, n, unif) = mesh(domain(box, :P => polygon(V)), n, unif)
lines(Ωₕ) = (collect(points(Ωₕ(1))), collect(points(Ωₕ(2))))

# The library's labels in the oracle's encoding.
function library_classes(Ωₕ)
    xs, ys = lines(Ωₕ)
    cls = fill(-1, length(xs), length(ys))
    cls[reshape(Bramble.index_in_marker(Ωₕ, :P), size(cls))] .= 1
    cls[reshape(Bramble.index_in_marker(Ωₕ, :P_boundary), size(cls))] .= 0
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
    U, W = polygon_spaces(Ωₕ, :P, g)
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
    return Matrix(A), collect(F)
end

relerr(x, y) = maximum(abs, x .- y) / maximum(abs, y)

# The oracle on the library's grid and labels, both systems with the library's rows divided
# by the dual-cell area, and the zero-exterior oracle.
function compare(Ωₕ, V, a, b, c, f, g; d = nothing, q = nothing)
    xs, ys = lines(Ωₕ)
    cls = O.classify(xs, ys, V)
    A, F = library_system(Ωₕ, a, b, c, f, g; d = d, q = q)
    bo = b === nothing ? zero_ : b
    Aoracle, Foracle, int = O.fg_system(xs, ys, cls, a, bo, c, f, g; d = d, q = q)
    Az = O.fg_system(xs, ys, cls, a, bo, c, f, g; d = d, q = q, exterior = :zero)[1]
    ar = O.dual_areas(xs, ys, int)
    return (; cls, A = A ./ ar, F = F ./ ar, Aoracle = Matrix(Aoracle), Foracle,
        Az = Matrix(Az), u = A \ F, uoracle = Matrix(Aoracle) \ Foracle,
        reads = exterior_reads(cls), xs, ys, int)
end

# The norms on Ω̄ₕ of #879 by hand: the values `v` at the interior nodes `int`, 0 on Γ and
# outside. The squared L² norm sums the dual-cell areas; the squared H¹ seminorm sums, over
# the grid edges with both ends in Ω̄ (`cls ≥ 0`), the dual length times the squared
# difference over the edge length.
function hand_norms(xs, ys, cls, int, v)
    V = zeros(length(xs), length(ys))
    V[int] .= v
    half(z, j) = ((j < length(z) ? z[j + 1] : z[j]) - (j > 1 ? z[j - 1] : z[j])) / 2
    s = 0.0
    for l in eachindex(ys), j in 1:(length(xs) - 1)

        cls[j, l] >= 0 && cls[j + 1, l] >= 0 || continue
        s += half(ys, l) * (V[j + 1, l] - V[j, l])^2 / (xs[j + 1] - xs[j])
    end
    for l in 1:(length(ys) - 1), j in eachindex(xs)

        cls[j, l] >= 0 && cls[j, l + 1] >= 0 || continue
        s += half(xs, j) * (V[j, l + 1] - V[j, l])^2 / (ys[l + 1] - ys[l])
    end
    return sqrt(sum(O.dual_areas(xs, ys, int) .* v .^ 2)), sqrt(s), maximum(abs, v)
end

# `Rₕ` and the norms on the trial space `U` of `polygon_spaces(Ωₕ, label, uq)`, `cls` the
# oracle's labels: `Rₕ` holds f at the interior nodes, the norms match `hand_norms`, the
# error of the quadratic solve is round-off, and the test space `W` still refuses.
function check_trial_norms(Ωₕ, label, cls)
    xs, ys = lines(Ωₕ)
    C = CartesianIndices(cls)
    int = findall(==(1), vec(cls))
    uq(x) = x[1]^2 + x[2]^2 + x[1] + 2 * x[2]
    U, W = polygon_spaces(Ωₕ, label, uq)
    r = Rₕ(U, g_var)
    @test r.space === U
    @test parent(r) == [g_var((xs[C[k][1]], ys[C[k][2]])) for k in int]
    r! = element(U)
    @test Rₕ!(r!, g_var) === r!
    @test parent(r!) == parent(r)
    n, s, m = hand_norms(xs, ys, cls, int, parent(r))
    @test normₕ(r) ≈ n rtol = 1e-13
    @test snorm₁ₕ(r) ≈ s rtol = 1e-13
    @test norm₁ₕ(r) ≈ hypot(n, s) rtol = 1e-13
    @test norminf(r) == m

    A, F = assemble(form(U, W, (w, v) -> innerₕ(-1.0 * Δₕ(w), v)),
        form(W, v -> innerₕ(x -> -4.0, v)))
    e = element(U, A \ F) - Rₕ(U, uq)
    @test normₕ(e) < 1e-9
    @test snorm₁ₕ(e) < 1e-9
    @test norm₁ₕ(e) < 1e-9
    @test norminf(e) < 1e-9
    @test normₕ(Rₕ(U, uq)) > 1   # the positive control: the norm reads the values

    @test_throws ArgumentError Rₕ(U, uq; markers = (label,))
    @test_throws ArgumentError Rₕ(W, uq)
    @test_throws ArgumentError Rₕ!(element(W), uq)
    @test_throws ArgumentError normₕ(element(W))
    @test_throws ArgumentError snorm₁ₕ(element(W))
    @test_throws ArgumentError norminf(element(W))
    @test_throws ArgumentError Rₕ(gridspace(Ωₕ; ghosts = :boundary), uq)
    return nothing
end

@testset "Polygon spaces against the F&G oracle" begin
    @testset "Rₕ and norms on the trial space" begin
        Random.seed!(2)
        M = polygon_mesh(FIG1, FIG1_BOX, (10, 9), (false, false))
        cls = O.classify(lines(M)..., FIG1)
        @test cls == library_classes(M)
        check_trial_norms(M, :P, cls)
    end

    @testset "Integer grids: Figure 1 and the V-notch" begin
        # Dual areas are 1: the rows agree unscaled.
        M = polygon_mesh(FIG1, FIG1_BOX, (7, 6), (true, true))
        r = compare(M, FIG1, one_, one_, one_, f5, g_var)
        @test r.cls == library_classes(M)
        @test size(r.A) == (13, 13)
        @test opnorm(r.A - r.Aoracle, Inf) < 1e-12
        @test relerr(r.F, r.Foracle) < 1e-12
        @test r.reads[1] > 0
        @test opnorm(r.A - r.Az, Inf) > 1e-3

        MN = polygon_mesh(VNOTCH, VNOTCH_BOX, (5, 5), (true, true))
        for b in (nothing, one_, b_var)
            r = compare(MN, VNOTCH, one_, b, one_, f5, g_var)
            @test r.cls == library_classes(MN)
            @test relerr(r.A, r.Aoracle) < 1e-12
            @test relerr(r.F, r.Foracle) < 1e-12
            @test relerr(r.u, r.uoracle) < 1e-10
            # The conflict node (0, 2) is read twice; with the mixed term, (3.2) moves it.
            @test r.reads[2] == 1
            b === nothing || @test opnorm(r.A - r.Az, Inf) > 1e-3
        end
    end

    # Variable a, b, c, a convection and a reaction term.
    @testset "Variable coefficients (seed $seed)" for seed in (3, 11)
        Random.seed!(seed)
        M = polygon_mesh(FIG1, FIG1_BOX, (9, 8), (false, false))
        @test length(points(M(1))) > 9   # the closure added lines
        r = compare(M, FIG1, a_var, b_var, c_var, f5, g_var; d = d_var, q = q_var)
        @test r.cls == library_classes(M)
        @test relerr(r.A, r.Aoracle) < 1e-12
        @test relerr(r.F, r.Foracle) < 1e-12
        @test relerr(r.u, r.uoracle) < 1e-10
        @test r.reads[1] > 0
        @test opnorm(r.A - r.Az, Inf) > 1e-3   # b ≈ 0.4..1: (3.2) is visible
    end

    @testset "V-notch closure grid (seed $seed)" for seed in (5, 17)
        Random.seed!(seed)
        MN = polygon_mesh(VNOTCH, VNOTCH_BOX, (6, 7), (false, false))
        for b in (nothing, b_var)
            r = compare(MN, VNOTCH, a_var, b, c_var, f5, g_var; d = d_var)
            @test r.cls == library_classes(MN)
            @test relerr(r.A, r.Aoracle) < 1e-12
            @test relerr(r.F, r.Foracle) < 1e-12
            @test relerr(r.u, r.uoracle) < 1e-10
            @test r.reads[2] >= 1
            b === nothing || @test opnorm(r.A - r.Az, Inf) > 1e-3
        end
    end

    @testset "Sliver grid: quadratic Poisson exact" begin
        # S3's sliver: the seed 5.7 + 1e-9 makes the closure draw the y-lines 3.3 and
        # 3.3 + 1e-9. Separable-additive u (u_xy = 0) makes (3.1) and (3.2) exact.
        P = polygon(FIG1)
        xs, ys = Bramble._grid_closure(P, [0.0, 0.3, 1.1, 2.5, 4.6, 5.7 + 1e-9, 6.0],
            [0.0, 0.45, 1.2, 2.6, 3.9, 4.4, 5.0])
        @test minimum([diff(xs); diff(ys)]) < 2e-9
        M = mesh(domain(FIG1_BOX), (length(xs), length(ys)), (true, true))
        Bramble.change_points!(M, markers(domain(FIG1_BOX, :P => P)), (xs, ys))
        uq(x) = x[1]^2 + x[2]^2 + x[1] + 2 * x[2]
        r = compare(M, FIG1, one_, one_, one_, x -> -4.0, uq)
        @test r.cls == library_classes(M)
        @test relerr(r.A, r.Aoracle) < 1e-12
        @test relerr(r.F, r.Foracle) < 1e-12
        exact = [uq((r.xs[k[1]], r.ys[k[2]]))
                 for k in CartesianIndices((length(r.xs), length(r.ys)))[r.int]]
        # Exact: each row's residual at u is round-off of that row's own terms. The solve
        # is not a fair measure, as the sliver puts cond(A) near 1e10.
        rowerr(A, F) = maximum(abs.(A * exact .- F) ./ (abs.(A) * abs.(exact) .+ abs.(F)))
        @test rowerr(r.A, r.F) < 1e-13
        @test rowerr(r.Aoracle, r.Foracle) < 1e-13
        @test relerr(r.u, exact) < 1e-6
        @test r.reads[1] > 0
        @test opnorm(r.A - r.Az, Inf) > 1e-3
    end

    # Whether a form reads the conflict node is decided by `form` (#902): the contraction of
    # a form that does not read it allocates what the padded form's own does (the one cell
    # of its sum, test/ghost/contract.jl), and one that does still throws after a warm
    # `assemble!`, which refills the pattern `form` read.
    @testset "V-notch: conflict read set by form" begin
        Random.seed!(7)
        U, W = polygon_spaces(polygon_mesh(VNOTCH, VNOTCH_BOX, (6, 7), (false, false)), :P,
            g_var)
        ap = form(U, W, (w, v) -> innerₕ(-1.0 * Δₕ(w), v))
        xn, yn = rand(Bramble.ndofs(U)), rand(Bramble.ndofs(W))
        @test ap(xn, yn) ≈ sum(yn .* (assemble(ap) * xn))
        up = element(Bramble.padded_space(U), U.E * xn)
        vp = element(Bramble.padded_space(W), W.E * yn)
        @test alloc_test(ap, xn, yn) == alloc_test(ap.padded, up, vp)
        bₕ = Rₕ(Bramble.padded_space(U), b_var)
        am = form(U, W, (w, v) -> innerₕ(-1.0 * Δₕ(w) - Dcₓ(bₕ * Dcᵧ(w)), v))
        A = assemble(am)
        assemble!(A, am)
        @test_throws ArgumentError am(xn, yn)
        @test_throws ArgumentError Bramble.matrix_free_operator(am)
    end
end

end # module
