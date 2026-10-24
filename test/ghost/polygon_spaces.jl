module GhostPolygonSpacesTests

# `polygon_spaces(Ωₕ, :L, g)` (#202, #204): the trial and test spaces of the Ferreira and
# Grigorieff scheme on a grid-fitted polygon. The trial closure keeps the interior nodes,
# fixes the boundary nodes to `g` and extends each exterior diagonal neighbour of an
# interior node by (3.2); the test closure selects the interior nodes. Solutions are exact
# for separable-additive quadratics, for which (3.2) and the scheme are exact, and a
# sign-flipped, zeroed or dropped (3.2), or the trial space as test space, breaks that.
# The reduced system is also checked row by row against an oracle built on the plain
# space's matrix: each tested row keeps its interior columns, moves its boundary columns to
# the load through `g`, and replaces each exterior column by that row's own (3.2) row, so
# a V-notch node (read by two interior nodes) is substituted per reading equation.

using Test
using Random
using SparseArrays
using LinearAlgebra: dot, mul!
using Bramble
import Bramble
using Bramble: lift, polygon, polygon_spaces, Dcₓ, Dcᵧ
using LinearSolve: KrylovJL_GMRES, solve
using Kronecker: Kronecker

# Figure 1 of the paper, its three oblique edges of slope ±1 apart; the V-notch, a reflex
# vertex at (1, 2) between two oblique edges; an L without oblique edges.
const FIG1 = [(0.0, 2.0), (2.0, 0.0), (4.0, 0.0), (6.0, 2.0), (6.0, 3.0), (4.0, 3.0),
    (4.0, 5.0), (2.0, 5.0), (0.0, 3.0)]
const VNOTCH = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0), (0.0, 3.0), (1.0, 2.0),
    (0.0, 1.0)]
const LSHAPE = [(1.0, 1.0), (5.0, 1.0), (5.0, 3.0), (3.0, 3.0), (3.0, 5.0), (1.0, 5.0)]

polygon_mesh(V, box, n, unif) = mesh(domain(box, :P => polygon(V)), n, unif)
fig1_mesh(n, unif) = polygon_mesh(FIG1, interval(0.0, 6.0) × interval(0.0, 5.0), n, unif)
vnotch_mesh(n, unif) = polygon_mesh(VNOTCH, interval(0.0, 4.0) × interval(0.0, 4.0), n,
    unif)

# Separable-additive (u_xy = 0), so (3.2) is exact, with −Δu = −4.
uq(x) = x[1]^2 + x[2]^2 + x[1] + 2x[2]
mixed(w) = Dcₓ(Dcᵧ(w)) + Dcᵧ(Dcₓ(w))

labels(Ωₕ) = (Bramble.index_in_marker(Ωₕ, :P), Bramble.index_in_marker(Ωₕ, :P_boundary))

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

@testset "Polygon spaces (#202, #204)" begin
    @testset "Figure 1: exact for quadratics" begin
        Random.seed!(2)
        Ωₕ = fig1_mesh((10, 9), (false, false))
        U, W = polygon_spaces(Ωₕ, :P, uq)
        I, Γ = labels(Ωₕ)
        f_a = (w, v) -> innerₕ(-1.0 * Δₕ(w) - 0.4 * mixed(w), v)
        A, F = assemble(form(U, W, f_a), form(W, v -> innerₕ(x -> -4.0, v)))
        @test size(A) == (count(I), count(I))
        uh = lift(U, A \ F).data
        uex = Rₕ(gridspace(Ωₕ), uq).data
        @test maximum(abs, uh[I] .- uex[I]) < 1e-9
        @test uh[Γ] == uex[Γ]
        X = .!(I .| Γ)
        @test count(!iszero, uh[X]) == count(U.rows.extended) > 0
        @test uh[X .& .!U.rows.extended] == zeros(count(X .& .!U.rows.extended))
        # The test side must be the interior selection: with the trial closure there, the
        # extended rows join the interior ones.
        @test_throws ArgumentError form(U, U, f_a)
        @test_throws ArgumentError form(U, v -> innerₕ(x -> -4.0, v))
    end

    @testset "rows match the plain-matrix oracle" begin
        Random.seed!(5)
        for (Ωₕ, f_a) in (
            (fig1_mesh((9, 8), (false, false)), (w, v) -> innerₕ(-1.0 * Δₕ(w) - mixed(w), v)),
            (vnotch_mesh((5, 5), (true, true)), (w, v) -> innerₕ(-1.0 * mixed(w), v)))
            g(x) = 1 + x[1] - 2x[2] + x[1] * x[2]
            f_l = v -> innerₕ(x -> sin(x[1]) + x[2], v)
            U, W = polygon_spaces(Ωₕ, :P, g)
            A, F = assemble(form(U, W, f_a), form(W, f_l))
            Aoracle, Foracle = oracle(Ωₕ, f_a, f_l, g)
            @test maximum(abs, Matrix(A) .- Aoracle) < 1e-12 * maximum(abs, Aoracle)
            @test maximum(abs, F .- Foracle) < 1e-12 * maximum(abs, Foracle)
        end
    end

    @testset "DOF k is the k-th interior node" begin
        Random.seed!(3)
        Ωₕ = fig1_mesh((8, 7), (false, false))
        U, W = polygon_spaces(Ωₕ, :P, x -> 0.0)
        I, _ = labels(Ωₕ)
        keep = findall(I)
        for k in (1, 7, length(keep))
            e = zeros(length(keep))
            e[k] = 1.0
            @test findall(!iszero, lift(U, e).data[I]) == [k]
            @test lift(U, e).data[keep[k]] == 1.0
            @test lift(W, e).data == (1:length(I) .== keep[k])
        end
    end

    @testset "(3.2) at every extended node" begin
        Random.seed!(4)
        Ωₕ = fig1_mesh((9, 9), (false, false))
        U, _ = polygon_spaces(Ωₕ, :P, uq)
        I, Γ = labels(Ωₕ)
        uex = Rₕ(gridspace(Ωₕ), uq).data
        z = rand(Xoshiro(1), count(I))
        uz = lift(U, z).data
        car = CartesianIndices(Bramble.npoints(Ωₕ, Tuple))
        lin = LinearIndices(car)
        uI = zeros(length(I))
        uI[I] .= z
        n = 0
        for p in findall(U.rows.extended .& .!U.rows.conflict)
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

    @testset "innerₕ weight is the dual-cell area" begin
        Random.seed!(6)
        Ωₕ = fig1_mesh((9, 8), (false, false))
        U, W = polygon_spaces(Ωₕ, :P, uq)
        I, _ = labels(Ωₕ)
        n = Bramble.npoints(Ωₕ, Tuple)
        xs = [Bramble.point(Ωₕ, (i, 1))[1] for i in 1:n[1]]
        ys = [Bramble.point(Ωₕ, (1, j))[2] for j in 1:n[2]]
        area = [(xs[i + 1] - xs[i - 1]) * (ys[j + 1] - ys[j - 1]) / 4
                for i in 2:(n[1] - 1), j in 2:(n[2] - 1)]
        full = zeros(n)
        full[2:(end - 1), 2:(end - 1)] .= area
        F = assemble(form(W, v -> innerₕ(x -> 1.0, v)))
        @test F ≈ vec(full)[I] rtol = 1e-14
    end

    @testset "no oblique edge: nothing extended" begin
        Random.seed!(7)
        Ωₕ = polygon_mesh(LSHAPE, interval(0.0, 6.0) × interval(0.0, 6.0), (9, 9),
            (false, false))
        U, W = polygon_spaces(Ωₕ, :P, uq)
        I, Γ = labels(Ωₕ)
        @test !any(U.rows.extended)
        A, F = assemble(form(U, W, (w, v) -> innerₕ(-1.0 * Δₕ(w), v)),
            form(W, v -> innerₕ(x -> -4.0, v)))
        uh = lift(U, A \ F).data
        @test maximum(abs, uh[I] .- Rₕ(gridspace(Ωₕ), uq).data[I]) < 1e-10
        @test all(iszero, uh[.!(I .| Γ)])
    end

    @testset "V-notch: per-reader (3.2) rows" begin
        Ωₕ = vnotch_mesh((5, 5), (true, true))
        g(x) = x[1]^2 + x[2]^2
        U, W = polygon_spaces(Ωₕ, :P, g)
        I, _ = labels(Ωₕ)
        @test findall(U.rows.conflict) == [LinearIndices((5, 5))[1, 3]]
        uex = Rₕ(gridspace(Ωₕ), g).data[I]
        l = form(W, v -> innerₕ(x -> -4.0, v))
        A, F = assemble(form(U, W, (w, v) -> innerₕ(-1.0 * Δₕ(w), v)), l)
        @test maximum(abs, A \ F .- uex) < 1e-12
        # Reader rows by hand: interior nodes 1 = (1, 1) and 6 = (1, 3) both read (0, 2).
        U0, W0 = polygon_spaces(Ωₕ, :P, x -> 0.0)
        AM, _ = assemble(form(U0, W0, (w, v) -> innerₕ(mixed(w), v)),
            form(W0, v -> innerₕ(x -> 0.0, v)))
        @test AM[1, 1] ≈ 0.5 && AM[6, 6] ≈ -0.5
        @test AM[1, 4] ≈ 0.5 && AM[6, 4] ≈ -0.5
        @test AM[6, 1] == 0 && AM[1, 6] == 0
        AX, FX = assemble(form(U, W, (w, v) -> innerₕ(-1.0 * Δₕ(w) - 0.5 * mixed(w), v)), l)
        @test maximum(abs, AX \ FX .- uex) < 1e-12
    end

    @testset "V-notch: variable b, random grid" begin
        Random.seed!(11)
        Ωₕ = vnotch_mesh((7, 6), (false, false))
        U, W = polygon_spaces(Ωₕ, :P, uq)
        I, _ = labels(Ωₕ)
        @test count(U.rows.conflict) > 0
        bₕ = Rₕ(Bramble.padded_space(U), x -> 0.3 + 0.1 * x[1] * x[2])
        f_a = (w, v) -> innerₕ(-1.0 * Δₕ(w) - Dcₓ(bₕ * Dcᵧ(w)) - Dcᵧ(bₕ * Dcₓ(w)), v)
        # The oracle's coefficient lives on the plain space, with the same values.
        bp = Rₕ(gridspace(Ωₕ), x -> 0.3 + 0.1 * x[1] * x[2])
        f_p = (w, v) -> innerₕ(-1.0 * Δₕ(w) - Dcₓ(bp * Dcᵧ(w)) - Dcᵧ(bp * Dcₓ(w)), v)
        f_l = v -> innerₕ(x -> -4.0 + x[1], v)
        A, F = assemble(form(U, W, f_a), form(W, f_l))
        Aoracle, Foracle = oracle(Ωₕ, f_p, f_l, uq)
        @test maximum(abs, Matrix(A) .- Aoracle) < 1e-12 * maximum(abs, Aoracle)
        @test maximum(abs, F .- Foracle) < 1e-12 * maximum(abs, Foracle)
        # Constant b: exact for the separable quadratic.
        A2, F2 = assemble(form(U, W, (w, v) -> innerₕ(-1.0 * Δₕ(w) - 0.5 * mixed(w), v)),
            form(W, v -> innerₕ(x -> -4.0, v)))
        @test maximum(abs, A2 \ F2 .- Rₕ(gridspace(Ωₕ), uq).data[I]) < 1e-10
    end

    @testset "warm assemble! allocates nothing" begin
        Random.seed!(12)
        Ωₕ = vnotch_mesh((7, 6), (false, false))
        U, W = polygon_spaces(Ωₕ, :P, uq)
        a = form(U, W, (w, v) -> innerₕ(-1.0 * Δₕ(w) - 0.5 * mixed(w), v))
        l = form(W, v -> innerₕ(x -> -4.0, v))
        A, F = assemble(a, l)
        assemble!(A, F, a, l)
        @test (@allocated assemble!(A, F, a, l)) == 0
    end

    @testset "refusals" begin
        Random.seed!(2)
        Ω = domain(interval(0.0, 6.0) × interval(0.0, 5.0), :P => polygon(FIG1))
        Ωₕ = mesh(Ω, (10, 9), (false, false))
        U, W = polygon_spaces(Ωₕ, :P, uq)
        l = form(W, v -> innerₕ(x -> -4.0, v))
        # Dcₓ(Dcₓ(u)) reads two nodes along x, beyond the scheme's stencil.
        @test_throws ArgumentError form(U, W, (w, v) -> innerₕ(Dcₓ(Dcₓ(w)), v))
        @test_throws ArgumentError assemble(form(U, W, (w, v) -> innerₕ(-1.0 * Δₕ(w), v)),
            l; dirichlet = (:P_boundary => uq,))
        e = @test_throws ArgumentError form(U, W,
            (w, v) -> innerₕ(w, v) + 2.0 * innerₕ(x -> 1.0, w) * innerₕ(x -> 1.0, v))
        @test occursin("RankOneUpdate", e.value.msg) && occursin("without it", e.value.msg)
        # A grid line the closure would not give: y = 1.3 crosses the oblique edges off
        # the nodes.
        Mf = mesh(Ω, (7, 6), (true, true))
        yf = collect(0.0:5.0)
        yf[2] = 1.3
        Bramble.change_points!(Mf, markers(Ω), (collect(0.0:6.0), yf))
        @test_throws ArgumentError polygon_spaces(Mf, :P, uq)
        M1 = polygon_mesh([(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0)],
            interval(0.0, 3.0) × interval(0.0, 3.0), (4, 4), (true, true))
        @test_throws ArgumentError polygon_spaces(M1, :P, uq)
        @test_throws ArgumentError polygon_spaces(Ωₕ, :Q, uq)
    end

    @testset "vertex within the merge tolerance" begin
        # The closure merges the line x = 1 into x = 1 + δ, so the vertex (1, 3) is off its
        # node by δ: the grid still fits, while the probe's y = 1.3 above does not.
        δ = 2.0e-10
        V = [(0.0, 0.0), (3.0, 0.0), (3.0, 1.0), (1 + δ, 1.0), (1 + δ, 2.0), (2.0, 3.0),
            (1.0, 3.0), (0.0, 2.0)]
        Ωₕ = polygon_mesh(V, interval(0.0, 3.0) × interval(0.0, 3.0), (7, 7), (true, true))
        U, W = polygon_spaces(Ωₕ, :P, uq)
        @test Bramble.ndofs(U) > 0
        I, _ = labels(Ωₕ)
        A, F = assemble(form(U, W, (w, v) -> innerₕ(-1.0 * Δₕ(w) - 0.5 * mixed(w), v)),
            form(W, v -> innerₕ(x -> -4.0, v)))
        @test maximum(abs, A \ F .- Rₕ(gridspace(Ωₕ), uq).data[I]) < 1e-8
    end

    @testset "test space as trial side" begin
        # It gives every exterior node 0, so a form reading one throws; a form reading the
        # boundary nodes reads them as 0, the trial space with g = 0.
        Ωₕ = fig1_mesh((7, 6), (true, true))
        U0, W = polygon_spaces(Ωₕ, :P, x -> 0.0)
        @test_throws ArgumentError form(W, W, (w, v) -> innerₕ(-1.0 * Δₕ(w) - mixed(w), v))
        f_a = (w, v) -> innerₕ(-1.0 * Δₕ(w), v)
        @test assemble(form(W, W, f_a)) == assemble(form(U0, W, f_a))
    end

    @testset "composites refuse polygon spaces" begin
        Random.seed!(2)
        for Ωₕ in (fig1_mesh((10, 9), (false, false)), vnotch_mesh((5, 5), (true, true)))
            U, W = polygon_spaces(Ωₕ, :P, uq)
            f_a = (w, v) -> innerₕ(-1.0 * Δₕ(w[1]) - 0.5 * mixed(w[1]), v[1]) +
                            innerₕ(w[2], v[2])
            @test_throws ArgumentError form(U × U, W × W, f_a)
            @test_throws ArgumentError form(W × W, v -> innerₕ(x -> 1.0, v[1]))
            # Beside a plain ghost space: refused before the polygon check, as a polygon
            # space ghosts no facet and the plain one does.
            G = gridspace(Ωₕ; ghosts = :boundary)
            @test_throws ArgumentError form(U × G, W × G, f_a)
            @test_throws ArgumentError form(W × G, v -> innerₕ(x -> 1.0, v[1]))
        end
    end

    @testset "matrix-free, contract, semidiscretize" begin
        rng = Xoshiro(13)
        Random.seed!(2)
        Ωₕ = fig1_mesh((10, 9), (false, false))
        U, W = polygon_spaces(Ωₕ, :P, uq)
        @test !any(U.rows.conflict)
        a = form(U, W, (w, v) -> innerₕ(-1.0 * Δₕ(w) - 0.4 * mixed(w), v))
        l = form(W, v -> innerₕ(x -> -4.0, v))
        A = assemble(a)
        x, y = rand(rng, size(A, 2)), rand(rng, size(A, 1))
        z = similar(y)
        mul!(z, Bramble.matrix_free_operator(a), x)
        @test z ≈ A * x
        @test a(x, y) ≈ dot(y, A * x)
        @test_throws ArgumentError semidiscretize(a, l)

        # On the V-notch the matrix-free product would miss the substitution of the conflict
        # node a mixed form reads; a form that does not read it applies.
        UN, WN = polygon_spaces(vnotch_mesh((5, 5), (true, true)), :P, x -> 0.0)
        am = form(UN, WN, (w, v) -> innerₕ(mixed(w), v))
        @test_throws ArgumentError Bramble.matrix_free_operator(am)
        xm, ym = rand(rng, Bramble.ndofs(UN)), rand(rng, Bramble.ndofs(WN))
        @test_throws ArgumentError am(xm, ym)
        ap = form(UN, WN, (w, v) -> innerₕ(-1.0 * Δₕ(w), v))
        xn = rand(rng, Bramble.ndofs(UN))
        zn = zeros(Bramble.ndofs(WN))
        mul!(zn, Bramble.matrix_free_operator(ap), xn)
        @test zn ≈ assemble(ap) * xn
        yn = rand(rng, Bramble.ndofs(WN))
        @test ap(xn, yn) ≈ dot(yn, assemble(ap) * xn)
    end

    @testset "linear_problem and solve (#880)" begin
        Random.seed!(2)
        Ωₕ = fig1_mesh((10, 9), (false, false))
        U, W = polygon_spaces(Ωₕ, :P, uq)
        a = form(U, W, (w, v) -> innerₕ(-1.0 * Δₕ(w) - 0.4 * mixed(w), v))
        l = form(W, v -> innerₕ(x -> -4.0, v))
        A, F = assemble(a, l)
        prob = linear_problem(a, l)
        @test prob.A == A && prob.b == F
        I, _ = labels(Ωₕ)
        uex = Rₕ(gridspace(Ωₕ), uq).data[I]
        @test maximum(abs, solve(prob).u .- uex) < 1e-9
        uₕ = solve(a, l)
        @test Bramble.space(uₕ) === U
        @test maximum(abs, parent(uₕ) .- uex) < 1e-9
        uₖ = solve(a, l; solver = KrylovJL_GMRES(), reltol = 1e-12, abstol = 0.0)
        @test maximum(abs, parent(uₖ) .- uex) < 1e-8
    end

    @testset "Kronecker, fdm, GMG refusals (#880)" begin
        Random.seed!(2)
        Ωₕ = fig1_mesh((10, 9), (false, false))
        U, W = polygon_spaces(Ωₕ, :P, uq)
        a = form(U, W, (w, v) -> innerₕ(-1.0 * Δₕ(w), v))
        F = assemble(a, form(W, v -> innerₕ(x -> -4.0, v)))[2]
        # The facet-ghost messages name relations per facet and `constrained`, neither of
        # which a polygon space has.
        polygon_msg(e) = occursin("polygon spaces", e.value.msg) &&
                         !occursin("facet", e.value.msg) &&
                         !occursin("constrained", e.value.msg)
        # A plain form's factorisation, for `fdm_factorize!` to refill from the polygon one.
        Wp = gridspace(Ωₕ)
        fp = fdm_factorize(form(Wp, Wp, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))))
        for (name, f) in (("kronecker_operator", () -> kronecker_operator(a)),
            ("fdm_solve", () -> fdm_solve(a, F)),
            ("fdm_solve", () -> fdm_solve(a, F; dirichlet = :boundary)),
            ("fdm_factorize", () -> fdm_factorize(a)),
            ("fdm_factorize!", () -> fdm_factorize!(fp, a)),
            ("fdm_preconditioner", () -> fdm_preconditioner(a)))
            e = @test_throws ArgumentError f()
            @test polygon_msg(e) && startswith(e.value.msg, name * " on a form")
        end
        e = @test_throws ArgumentError gmg_preconditioner(a)
        @test polygon_msg(e) && occursin("#846", e.value.msg)
        # The L's vertices lie on the odd lines of the 7² grid, off every line of the
        # coarse 4² grid, where `polygon_spaces` throws: the finest level is built first.
        build(W0) = (S = polygon_spaces(mesh(W0), :P, uq);
            form(S[1], S[2], (w, v) -> innerₕ(-1.0 * Δₕ(w), v)))
        for Ωg in (fig1_mesh((13, 11), (true, true)),
            polygon_mesh(LSHAPE, interval(0.0, 6.0) × interval(0.0, 6.0), (7, 7),
            (true, true)))
            e = @test_throws ArgumentError gmg_preconditioner(build, Ωg; levels = 2)
            @test polygon_msg(e) && occursin("#846", e.value.msg)
        end
        # Any other form on ghost spaces still has no one-argument method.
        G = gridspace(Ωₕ; ghosts = :boundary)
        @test_throws MethodError gmg_preconditioner(form(G, G, (u, v) -> innerₕ(u, v)))
    end
end

end # module
