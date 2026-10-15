module FormGhostCornersTests

# The corner rule of a ghost space (gpena/Bramble.jl#218, design note "Corner policy"): a
# mixed difference at a 2D corner node, or a 3D edge node, whose taps would read the absent
# padded corner beyond it is the inward one-sided mixed difference `D_{±x} D_{±y}` there,
# on physical nodes only; elsewhere nothing changes.
#
# Two oracles, neither sharing code with the library. GhostCorners' `pde_operator` with
# `closure = :facet_onesided` (the prototype's corner problem, every facet ghosted, 2D),
# and `mixed_oracle` below: the mixed difference as a product of 1D difference matrices
# lifted to the padded grid of any facet selection, in 2D and 3D, with every physical row
# that reads an absent point replaced by the one-sided row. Rows are scaled by the physical
# cell measures, as `innerₕ` weighs them. Meshes are uniform, random and graded; 3D only
# where the group allows it (#650).

using Test
using Bramble
import Bramble
using Bramble: Dcₓ, Dcᵧ, Dc₂, D̽ₓ, D̽ᵧ, D̽₂, D₋ₓ, D₋ᵧ, D₊ₓ, D₊ᵧ, D₊₂, D̃ₓ, D̃ᵧ
using LinearAlgebra: Diagonal, I, norm, ×
using SparseArrays: SparseMatrixCSC, sparse, spzeros, kron

using ..TestUtils: WITH_3D_TESTS

isdefined(Main, :GhostOracle1D) ||
    Base.include(Main, joinpath(@__DIR__, "..", "ghost", "GhostOracle1D.jl"))
isdefined(Main, :GhostCorners) ||
    Base.include(Main, joinpath(@__DIR__, "..", "ghost", "GhostCorners.jl"))
const O1 = Main.GhostOracle1D
const C = Main.GhostCorners

const tol = 1e-12

padded(Ωₕ; ghosts = :boundary) = Bramble.padded_space(gridspace(Ωₕ; ghosts = ghosts))

function random_mesh(ns; seed = 11)
    Ω = domain(foldl(×, ntuple(_ -> interval(0.0, 1.0), length(ns))))
    Bramble._seed_mesh1d_rng!(seed)
    try
        return mesh(Ω, ns, ntuple(_ -> false, length(ns)))
    finally
        Bramble._unseed_mesh1d_rng!()
    end
end

relerr(A, B) = norm(A - B) / norm(B)

#------------------------------------------------------------------------------------------#
# The oracle: 1D differences on a padded axis, lifted
#------------------------------------------------------------------------------------------#

const FACETS = (:xmin, :xmax, :ymin, :ymax, :zmin, :zmax)

# `(lo, hi)` per axis: whether `ghosts` pads its minimum and its maximum.
function pads(ghosts, D)
    names = ghosts === :boundary ? FACETS[1:(2D)] : ghosts
    return ntuple(d -> (FACETS[2d - 1] in names, FACETS[2d] in names), D)
end

function padded_axis(x, (lo, hi))
    p = copy(x)
    lo && pushfirst!(p, x[1] - (x[2] - x[1]))
    hi && push!(p, x[end] + (x[end] - x[end - 1]))
    return p
end

# The 1D difference `kind` on points `p`, with the library's truncations at the two ends.
function diff1d(kind::Symbol, p)
    N = length(p)
    A = spzeros(N, N)
    for i in 1:N
        h = i == 1 ? p[2] - p[1] : p[i] - p[i - 1]
        hf = i == N ? p[N] - p[N - 1] : p[i + 1] - p[i]
        if kind === :backward
            i > 1 && (A[i, i] = 1 / h; A[i, i - 1] = -1 / h)
        elseif kind === :forward
            i < N && (A[i, i + 1] = 1 / hf; A[i, i] = -1 / hf)
        elseif kind === :centered
            1 < i < N && (A[i, i + 1] = 1 / (h + hf); A[i, i - 1] = -1 / (h + hf))
        elseif kind === :cross
            if i == 1
                A[i, i + 1] = 1 / h
                A[i, i] = -1 / h
            elseif i == N
                A[i, i] = 1 / h
                A[i, i - 1] = -1 / h
            else
                a, b = h / ((h + hf) * hf), hf / ((h + hf) * h)
                A[i, i + 1] = a
                A[i, i] = b - a
                A[i, i - 1] = -b
            end
        end
    end
    return A
end

# `A` along axis `d` of a grid of sizes `ps` (x fastest).
function lift(A, d, ps)
    out = sparse(1.0I, 1, 1)
    for e in 1:length(ps)
        out = kron(e == d ? A : sparse(1.0I, ps[e], ps[e]), out)
    end
    return out
end

"""
    mixed_oracle(Ωₕ, (kind_d, d), (kind_e, e); ghosts, coef) -> SparseMatrixCSC

`innerₕ(A_d(c B_e(u)), v)` on the padded space of `ghosts` (rows test, columns trial), `c`
the vector `coef` on the padded points (1 when `nothing`). A physical row that reads an
absent point (outside along two axes) is `D_{s_d,d}(c D_{s_e,e}(u))` instead, `s = +1` at a
first node and `-1` at a last.
"""
function mixed_oracle(Ωₕ, (kd, d), (ke, e); ghosts = :boundary, coef = nothing)
    D = Bramble.dim(Ωₕ)
    xs = ntuple(k -> collect(Float64, points(Ωₕ(k))), D)
    pd = pads(ghosts, D)
    ps = ntuple(k -> padded_axis(xs[k], pd[k]), D)
    sz = map(length, ps)
    lo = map(t -> Int(t[1]), pd)
    ns = map(length, xs)
    c = coef === nothing ? ones(prod(sz)) : coef
    X = lift(diff1d(kd, ps[d]), d, sz) * Diagonal(c) * lift(diff1d(ke, ps[e]), e, sz)
    lin = LinearIndices(sz)
    out(J) = count(k -> J[k] <= lo[k] || J[k] > lo[k] + ns[k], 1:D)
    absent = falses(prod(sz))
    for J in CartesianIndices(sz)
        absent[lin[J]] = out(J) >= 2
    end
    W = spzeros(prod(sz), prod(sz))
    w = reduce((a, b) -> kron(b, a), map(k -> O1.cell_measures(Ωₕ(k)), 1:D))
    for (r, Iphys) in enumerate(CartesianIndices(ns))
        J = Iphys + CartesianIndex(lo)
        row = X[lin[J], :]
        if any(!iszero, row[absent])
            row = spzeros(prod(sz))
            sd = Iphys[d] == 1 ? 1 : -1
            se = Iphys[e] == 1 ? 1 : -1
            hd = ps[d][J[d] + sd] - ps[d][J[d]]
            he = ps[e][J[e] + se] - ps[e][J[e]]
            for a in (0, sd), b in (0, se)
                P = J + CartesianIndex(ntuple(k -> k == d ? a : 0, D))
                Q = P + CartesianIndex(ntuple(k -> k == e ? b : 0, D))
                row[lin[Q]] += (a == 0 ? -1 : 1) * (b == 0 ? -1 : 1) * c[lin[P]] / (hd * he)
            end
        end
        W[lin[J], :] = w[r] * row
    end
    return W
end

# The padded indices outside along two or more axes, for the facets `ghosts`.
function absent_columns(Ωₕ; ghosts = :boundary)
    D = Bramble.dim(Ωₕ)
    pd = pads(ghosts, D)
    ns = npoints(Ωₕ, Tuple)
    lo = map(t -> Int(t[1]), pd)
    sz = ntuple(k -> ns[k] + pd[k][1] + pd[k][2], D)
    lin = LinearIndices(sz)
    return [lin[J] for J in CartesianIndices(sz)
            if count(k -> J[k] <= lo[k] || J[k] > lo[k] + ns[k], 1:D) >= 2]
end

matrix(P, f) = assemble(form(P, P, f))

#------------------------------------------------------------------------------------------#
# Tests
#------------------------------------------------------------------------------------------#

@testset "Ghost corner rule (#218)" begin
    K = C.DEFAULT_K
    Kmix = [0.0 0.5; 0.5 0.0]       # pde_operator is then the mixed term X alone

    @testset "Rows match the corner oracle" begin
        for kind in (:uniform, :random, :graded), mixed in (:centered, :cross_weighted)
            Ωₕ = C.make_mesh(kind, 0; n0 = 9)
            nx, ny = C.sizes(Ωₕ)
            P = padded(Ωₕ)
            X(u) = mixed === :centered ? Dcₓ(Dcᵧ(u)) : D̽ₓ(D̽ᵧ(u))
            L(u) = K[1, 1] * D̃ₓ(D₋ₓ(u)) + 2 * K[1, 2] * X(u) + K[2, 2] * D̃ᵧ(D₋ᵧ(u))
            A = matrix(P, (u, v) -> innerₕ(L(u), v))
            rows = C.physical_padded(nx, ny)
            w = kron(O1.cell_measures(Ωₕ(2)), O1.cell_measures(Ωₕ(1)))
            R = Diagonal(w) * C.pde_operator(Ωₕ; closure = :facet_onesided, mixed, K)
            @test relerr(A[rows, :], R) <= tol
            # The rows the rule changed differ from the padded corner read, so the
            # comparison above separates the closures.
            R0 = Diagonal(w) * C.pde_operator(Ωₕ; closure = :padded_mixed, mixed, K)
            @test relerr(R0, R) > 1e-3
            cols = [C.padded(i, j, nx) for (i, j) in C.corner_points(nx, ny)]
            @test norm(A[:, cols]) == 0
            ghost_rows = setdiff(1:size(A, 1), rows)
            @test norm(A[ghost_rows, :]) == 0
        end
    end

    @testset "Transposed nesting and test side" begin
        Ωₕ = C.make_mesh(:random, 0; n0 = 9)
        nx, ny = C.sizes(Ωₕ)
        P = padded(Ωₕ)
        rows = C.physical_padded(nx, ny)
        w = kron(O1.cell_measures(Ωₕ(2)), O1.cell_measures(Ωₕ(1)))
        for (mixed, X, Xt) in ((:centered, u -> Dcₓ(Dcᵧ(u)), u -> Dcᵧ(Dcₓ(u))),
            (:cross_weighted, u -> D̽ₓ(D̽ᵧ(u)), u -> D̽ᵧ(D̽ₓ(u))))
            R = Diagonal(w) * C.pde_operator(Ωₕ; closure = :facet_onesided, mixed, K = Kmix)
            At = matrix(P, (u, v) -> innerₕ(Xt(u), v))
            @test relerr(At[rows, :], R) <= tol
            # innerₕ(u, X(v)) is the transpose of innerₕ(X(u), v)
            Av = matrix(P, (u, v) -> innerₕ(u, X(v)))
            @test relerr(Matrix(Av[:, rows]), Matrix(R')) <= tol
            @test norm(Av[:, setdiff(1:size(Av, 2), rows)]) == 0
            cols = [C.padded(i, j, nx) for (i, j) in C.corner_points(nx, ny)]
            @test norm(Av[cols, :]) == 0
            @test norm(At[:, cols]) == 0
        end
    end

    @testset "Every first-difference pair, 2D" begin
        Ωₕ = random_mesh((7, 6))
        P = padded(Ωₕ)
        absent = absent_columns(Ωₕ)
        ops = ((:backward, D₋ₓ, D₋ᵧ), (:forward, D₊ₓ, D₊ᵧ), (:centered, Dcₓ, Dcᵧ),
            (:cross, D̽ₓ, D̽ᵧ))
        for (kx, Ax, _) in ops, (ky, _, By) in ops
            A = matrix(P, (u, v) -> innerₕ(Ax(By(u)), v))
            @test relerr(A, mixed_oracle(Ωₕ, (kx, 1), (ky, 2))) <= tol
            @test norm(A[:, absent]) == 0
        end
    end

    @testset "Scales between the two differences" begin
        Ωₕ = random_mesh((6, 7); seed = 3)
        P = padded(Ωₕ)
        absent = absent_columns(Ωₕ)
        c = 1.0 .+ 0.5 .* sin.(1:Bramble.npoints(Bramble.mesh(P)))
        A = matrix(P, (u, v) -> innerₕ(Dcₓ(c * Dcᵧ(u)), v))
        @test relerr(A, mixed_oracle(Ωₕ, (:centered, 1), (:centered, 2); coef = c)) <= tol
        @test norm(A[:, absent]) == 0
        A3 = matrix(P, (u, v) -> innerₕ(D̽ᵧ(3.0 * D̽ₓ(u)), v))
        @test relerr(A3, 3.0 * mixed_oracle(Ωₕ, (:cross, 2), (:cross, 1))) <= tol
    end

    @testset "Named facets" begin
        Ωₕ = random_mesh((6, 5); seed = 5)
        for ghosts in ((:xmin, :ymin), (:xmax,), (:xmin, :xmax, :ymax), (:ymin, :ymax))
            P = padded(Ωₕ; ghosts)
            absent = absent_columns(Ωₕ; ghosts)
            for (kx, Ax, ky, By) in ((:centered, Dcₓ, :centered, Dcᵧ),
                (:cross, D̽ₓ, :cross, D̽ᵧ), (:backward, D₋ₓ, :forward, D₊ᵧ))
                A = matrix(P, (u, v) -> innerₕ(Ax(By(u)), v))
                @test relerr(A, mixed_oracle(Ωₕ, (kx, 1), (ky, 2); ghosts)) <= tol
                isempty(absent) || @test norm(A[:, absent]) == 0
            end
        end
        # A corner of a ghosted and an unghosted facet keeps today's truncation: with only
        # `:xmin` ghosted there is no absent point and the plain mixed rows stand.
        P = padded(Ωₕ; ghosts = (:xmin,))
        @test isempty(absent_columns(Ωₕ; ghosts = (:xmin,)))
        A = matrix(P, (u, v) -> innerₕ(Dcₓ(Dcᵧ(u)), v))
        @test relerr(A, mixed_oracle(Ωₕ, (:centered, 1), (:centered, 2); ghosts = (:xmin,))) <=
              tol
    end

    @testset "Plain meshes are unchanged" begin
        Ωₕ = random_mesh((7, 6); seed = 9)
        W = gridspace(Ωₕ)
        for (kx, Ax, ky, By) in ((:centered, Dcₓ, :centered, Dcᵧ), (:cross, D̽ₓ, :cross, D̽ᵧ))
            A = matrix(W, (u, v) -> innerₕ(Ax(By(u)), v))
            @test relerr(A, mixed_oracle(Ωₕ, (kx, 1), (ky, 2); ghosts = ())) <= tol
        end
        # The rule is a constant `false` off a ghost mesh, and bites at a ghost corner node.
        X = Dcₓ(Dcᵧ(Bramble.trial_function(W)))
        @test !Bramble._needs_onesided(X, Bramble.mesh(W), CartesianIndex(1, 1))
        G = Bramble.mesh(padded(Ωₕ))
        @test Bramble._needs_onesided(X, G, CartesianIndex(2, 2))
        @test !Bramble._needs_onesided(X, G, CartesianIndex(3, 2))
    end

    @testset "Warm assemble! allocates nothing" begin
        Ωₕ = C.make_mesh(:graded, 0; n0 = 9)
        P = padded(Ωₕ)
        L(u) = K[1, 1] * D̃ₓ(D₋ₓ(u)) + 2 * K[1, 2] * Dcₓ(Dcᵧ(u)) + K[2, 2] * D̃ᵧ(D₋ᵧ(u))
        a = form(P, P, (u, v) -> innerₕ(L(u), v))
        A = assemble(a)
        _alloc(A, a) = (assemble!(A, a); @allocated assemble!(A, a))
        @test _alloc(A, a) == 0
        b = form(P, P, (u, v) -> innerₕ(D̽ₓ(D̽ᵧ(u)), D̽ᵧ(D̽ₓ(v))))
        B = assemble(b)
        @test _alloc(B, b) == 0
    end

    # 3D runs nightly (#650): the edge nodes take the rule for each pair of axes.
    WITH_3D_TESTS && @testset "3D edge nodes" begin
        Ωₕ = random_mesh((5, 4, 6); seed = 13)
        P = padded(Ωₕ)
        absent = absent_columns(Ωₕ)
        for (kd, d, Ad, ke, e, Be) in ((:centered, 1, Dcₓ, :centered, 2, Dcᵧ),
            (:cross, 2, D̽ᵧ, :cross, 3, D̽₂), (:forward, 1, D₊ₓ, :centered, 3, Dc₂))
            A = matrix(P, (u, v) -> innerₕ(Ad(Be(u)), v))
            @test relerr(A, mixed_oracle(Ωₕ, (kd, d), (ke, e))) <= tol
            @test norm(A[:, absent]) == 0
        end
        A = matrix(P, (u, v) -> innerₕ(Dcₓ(Dcᵧ(u)) + D̽ᵧ(D̽₂(u)) + D₊₂(D̽ₓ(u)), v))
        @test norm(A[:, absent]) == 0
    end
end

end # module FormGhostCornersTests
