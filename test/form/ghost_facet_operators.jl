module FormGhostFacetOperatorsTests

# The facet operators `∂ₙ`, `∂ₙₙ` and `jumpₙ` (gpena/Bramble.jl#218, #205) inside `inner_Γ`
# on a ghost space, against the plain-matrix oracles' differences on the padded grid.
#
# Every expected matrix is built from the oracle alone: `Dc_d` lifted to the padded grid,
# and per facet `F = (d, s)` the diagonal `W_F` of facet weights `ω_F(I)` at its physical
# nodes. With the outward sign `η_F`, `inner_Γ(∂ₙ(u), v)` is `Σ_F η_F W_F Dc_d` (rows test,
# columns trial), the test side its transpose, and both sides `Σ_F Dc_dᵀ W_F Dc_d`. A corner
# node lies on two facets and gets a row from each selected one, never a lumped weight.
# Meshes are random (non-uniform); 3D only where the group allows it (#650).

using Test
using Bramble
import Bramble
using Bramble: Dcₓ, D₋ᵧ
using LinearAlgebra: Diagonal, dot, norm, ×
using SparseArrays: nnz, sparse, spzeros

using ..TestUtils: WITH_3D_TESTS, padded, random_mesh

isdefined(Main, :GhostOracle1D) ||
    Base.include(Main, joinpath(@__DIR__, "..", "ghost", "GhostOracle1D.jl"))
isdefined(Main, :GhostOracleND) ||
    Base.include(Main, joinpath(@__DIR__, "..", "ghost", "GhostOracleND.jl"))
const O1 = Main.GhostOracle1D
const ON = Main.GhostOracleND

const tol = 1e-12

const FACETS = (:xmin, :xmax, :ymin, :ymax, :zmin, :zmax)
facet_axis_side(f::Symbol) = (k = findfirst(==(f), FACETS); ((k + 1) ÷ 2, isodd(k) ? 1 : 2))
outward(s::Int) = s == 1 ? -1.0 : 1.0

# The 1D oracle is the ND one's axis factor: its `Dc` and unit facet weight.
function oracle_ops(Ωₕ)
    if Bramble.dim(Ωₕ) == 1
        n = length(points(Ωₕ))
        return (; Dc = (O1.operators(Ωₕ).Dc,), ns = (n,), ps = (n + 2,))
    end
    ns = ON.sizes(Ωₕ)
    return (; Dc = ON.operators(Ωₕ).Dc, ns, ps = ns .+ 2)
end

facet_weight(Ωₕ, d, I) = Bramble.dim(Ωₕ) == 1 ? 1.0 : ON.facet_weight(Ωₕ, d, I)

# `W_F`: `ω_F(I)` at the padded index of each physical node `I` of facet `(d, s)`.
function facet_weights(Ωₕ, d, s)
    (; ns, ps) = oracle_ops(Ωₕ)
    LP = LinearIndices(ps)
    w = zeros(prod(ps))
    k = s == 1 ? 1 : ns[d]
    for I in CartesianIndices(ntuple(e -> e == d ? (k:k) : (1:ns[e]), length(ns)))
        w[LP[I + oneunit(I)]] = facet_weight(Ωₕ, d, I)
    end
    return Diagonal(w)
end

# `Σ_F c_F η_F W_F N_d` over the facets `F` named, `N_d` the facet's normal flux matrix.
function trial_rows(Ωₕ, facets; c = f -> 1.0, flux = (Dc, d) -> Dc[d])
    (; Dc) = oracle_ops(Ωₕ)
    return sum(facets) do f
        d, s = facet_axis_side(f)
        c(f) * outward(s) * facet_weights(Ωₕ, d, s) * flux(Dc, d)
    end
end

relerr(A, R) = norm(A - R) / norm(R)

@testset "∂ₙ as a facet operator (#218)" begin
    @testset "1D: trial, test and both sides" begin
        Ωₕ = random_mesh((19,); seed = 7)
        P = padded(Ωₕ)
        R = sparse(trial_rows(Ωₕ, (:xmin, :xmax)))
        A = assemble(form(P, P, (u, v) -> inner_Γ(∂ₙ(u), v; markers = :boundary)))
        @test relerr(A, R) <= tol
        n = length(points(Ωₕ))
        Dc = O1.operators(Ωₕ).Dc
        # the facet node of xmin is padded index 2, and its row reads the ghost at 1
        @test A[2, 1] ≈ 1 / (2 * (points(Ωₕ)[2] - points(Ωₕ)[1]))
        @test norm(A[n + 1, :] - Dc[n + 1, :]) <= tol * norm(Dc[n + 1, :])

        B = assemble(form(P, P, (u, v) -> inner_Γ(u, ∂ₙ(v); markers = :boundary)))
        @test relerr(B, sparse(R')) <= tol
        C = assemble(form(P, P, (u, v) -> inner_Γ(∂ₙ(u), ∂ₙ(v); markers = :boundary)))
        @test relerr(C, sparse(Dc' * (facet_weights(Ωₕ, 1, 1) + facet_weights(Ωₕ, 1, 2)) *
                               Dc)) <= tol
    end

    @testset "2D trial side against the oracle" begin
        Ωₕ = random_mesh((6, 7); seed = 7)
        P = padded(Ωₕ)
        A = assemble(form(P, P, (u, v) -> inner_Γ(∂ₙ(u), v; markers = :boundary)))
        @test relerr(A, sparse(trial_rows(Ωₕ, FACETS[1:4]))) <= tol
        for f in FACETS[1:4]
            Af = assemble(form(P, P, (u, v) -> inner_Γ(∂ₙ(u), v; markers = (f,))))
            @test relerr(Af, sparse(trial_rows(Ωₕ, (f,)))) <= tol
        end
    end

    @testset "2D corner: one value per facet" begin
        Ωₕ = random_mesh((6, 7); seed = 7)
        P = padded(Ωₕ)
        (; Dc, ps) = oracle_ops(Ωₕ)
        corner = LinearIndices(ps)[2, 2]   # physical node (1, 1)
        I = CartesianIndex(1, 1)
        ωx, ωy = ON.facet_weight(Ωₕ, 1, I), ON.facet_weight(Ωₕ, 2, I)
        @test ωx != ωy   # non-uniform: a lumped weight would not hide in a coincidence
        Ax = assemble(form(P, P, (u, v) -> inner_Γ(∂ₙ(u), v; markers = (:xmin,))))
        @test norm(Ax[corner, :] + ωx * Dc[1][corner, :]) <= tol * ωx * norm(Dc[1][corner, :])
        Ab = assemble(form(P, P, (u, v) -> inner_Γ(∂ₙ(u), v; markers = :boundary)))
        rowb = -ωx * Dc[1][corner, :] - ωy * Dc[2][corner, :]
        @test norm(Ab[corner, :] - rowb) <= tol * norm(rowb)
        # the corner reads the ghosts of both facets and no absent padded point
        @test Ab[corner, LinearIndices(ps)[1, 2]] != 0 && Ab[corner, LinearIndices(ps)[2, 1]] != 0
        @test Ab[corner, LinearIndices(ps)[1, 1]] == 0
    end

    @testset "2D test side and both sides" begin
        Ωₕ = random_mesh((6, 7); seed = 7)
        P = padded(Ωₕ)
        (; Dc) = oracle_ops(Ωₕ)
        B = assemble(form(P, P, (u, v) -> inner_Γ(u, ∂ₙ(v); markers = :boundary)))
        @test relerr(B, sparse(trial_rows(Ωₕ, FACETS[1:4])')) <= tol
        C = assemble(form(P, P, (u, v) -> inner_Γ(∂ₙ(u), ∂ₙ(v); markers = :boundary)))
        RC = sum(FACETS[1:4]) do f
            d, s = facet_axis_side(f)
            Dc[d]' * facet_weights(Ωₕ, d, s) * Dc[d]
        end
        @test relerr(C, sparse(RC)) <= tol
    end

    @testset "Per-facet coefficients by markers" begin
        Ωₕ = random_mesh((6, 7); seed = 7)
        P = padded(Ωₕ)
        c = (xmin = 2.0, xmax = 0.5, ymin = 3.0, ymax = 1.5)
        a = form(P,
            P,
            (u, v) -> inner_Γ(c.xmin * ∂ₙ(u), v; markers = (:xmin,)) +
                      inner_Γ(c.xmax * ∂ₙ(u), v; markers = (:xmax,)) +
                      inner_Γ(c.ymin * ∂ₙ(u), v; markers = (:ymin,)) +
                      inner_Γ(c.ymax * ∂ₙ(u), v; markers = (:ymax,)))
        R = sparse(trial_rows(Ωₕ, FACETS[1:4]; c = f -> getfield(c, f)))
        @test relerr(assemble(a), R) <= tol

        # a Robin combination: `α u` weighted by each facet, plus `β ∂ₙ(u)`
        (; ps) = oracle_ops(Ωₕ)
        Wx = facet_weights(Ωₕ, 1, 2)
        Ar = assemble(form(P, P, (u, v) -> inner_Γ(2.0 * u + 3.0 * ∂ₙ(u), v;
            markers = (:xmax,))))
        @test relerr(Ar, sparse(2.0 * Wx + 3.0 * trial_rows(Ωₕ, (:xmax,)))) <= tol
    end

    @testset "Conormal rows with a constant K" begin
        Ωₕ = random_mesh((6, 7); seed = 7)
        P = padded(Ωₕ)
        K = [2.0 0.7; 0.4 1.5]
        conormal = (Dc, d) -> K[d, 1] * Dc[1] + K[d, 2] * Dc[2]
        A = assemble(form(P, P, (u, v) -> inner_Γ(∂ₙ(u; K = K), v; markers = :boundary)))
        @test relerr(A, sparse(trial_rows(Ωₕ, FACETS[1:4]; flux = conormal))) <= tol
        # an integer matrix is the same tensor
        Ai = assemble(form(P, P, (u, v) -> inner_Γ(∂ₙ(u; K = [2 0; 0 1]), v; markers = :boundary)))
        Ri = trial_rows(Ωₕ, FACETS[1:4]; flux = (Dc, d) -> (2, 1)[d] * Dc[d])
        @test relerr(Ai, sparse(Ri)) <= tol
    end

    @testset "Conormal at one ghosted facet's corners" begin
        # `u = y` on the padded grid: every centred difference is exact, so the conormal flux
        # at a node of xmin is `η · (K ∇u)_1 = -K[1, 2]`, and the row sum `-ω K[1, 2]`.
        Ωₕ = random_mesh((6, 7); seed = 7)
        ns = ON.sizes(Ωₕ)
        y = ON.axis_points(Ωₕ)[2]
        K = [2.0 0.7; 0.4 1.5]
        markers = (:xmin,)
        # the corners of xmin read the ymin and ymax ghosts through K[1, 2]
        Px = padded(Ωₕ; ghosts = (:xmin,))
        err = try
            form(Px, Px, (u, v) -> inner_Γ(∂ₙ(u; K), v; markers))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        msg = sprint(showerror, err)
        @test occursin(":ymin", msg) && occursin("K[1, 2]", msg)
        Pxy = padded(Ωₕ; ghosts = (:xmin, :ymin))
        @test_throws ArgumentError form(Pxy, Pxy, (u, v) -> inner_Γ(∂ₙ(u; K), v; markers))
        # no tangential coupling on xmin: one ghosted facet is enough, diagonal K or not
        for Kd in ([2.0 0.0; 0.0 1.5], [2.0 0.0; 0.4 1.5])
            Ad = assemble(form(Px, Px, (u, v) -> inner_Γ(∂ₙ(u; K = Kd), v; markers)))
            uy = [yj for _ in 1:(ns[1] + 1), yj in y][:]
            @test norm(Ad * uy) <= tol
            x = ON.axis_points(Ωₕ)[1]
            ux = [xi for xi in [2x[1] - x[2]; x], _ in y][:]
            LP = LinearIndices((ns[1] + 1, ns[2]))
            rows = [LP[2, j] for j in 1:ns[2]]
            ω = [ON.facet_weight(Ωₕ, 1, CartesianIndex(1, j)) for j in 1:ns[2]]
            @test (Ad * ux)[rows] ≈ -2.0 .* ω   # `u = x`, its ghost mirrored: Dc_x(u) = 1
            # the type is `K`'s alone: the zero is pruned at `form`, from the value
            ut = Bramble.trial_function(Px)
            @test typeof(∂ₙ(ut; K = Kd)) == typeof(∂ₙ(ut; K))
            # on the test side too, as the transpose
            At = assemble(form(Px, Px, (u, v) -> inner_Γ(u, ∂ₙ(v; K = Kd); markers)))
            @test At ≈ transpose(Ad)
            # and in a ghost relation, which checks its reach apart from `form`'s labels
            Wx = gridspace(Ωₕ; ghosts = (:xmin,))
            C = constrained(Wx, u -> :xmin => ∂ₙ(u; K = Kd) => 0.0)
            @test C isa Bramble.GhostGridSpace
        end
        @test_throws ArgumentError form(Px, Px, (u, v) -> inner_Γ(u, ∂ₙ(v; K); markers))
        Wx = gridspace(Ωₕ; ghosts = (:xmin,))
        rerr = try
            constrained(Wx, u -> :xmin => ∂ₙ(u; K) => 0.0)
            nothing
        catch e
            e
        end
        @test rerr isa ArgumentError
        @test occursin(":ymin", sprint(showerror, rerr))
        # a scale around the conormal terms, zero or a `Ref`, leaves `K`'s reach as it is
        @test_throws ArgumentError form(Px, Px,
            (u, v) -> inner_Γ(Ref(0.0) * ∂ₙ(u; K), v; markers))
        @test_throws ArgumentError form(Px, Px, (u, v) -> inner_Γ(0.0 * ∂ₙ(u; K), v; markers))
        # a y difference outside `K`'s terms is not read through `K[1, 2]`: with a diagonal
        # `K` and ghosts on xmin and ymin, `D₋ᵧ` reads the ymin ghost and an interior node
        Kd = [2.0 0.0; 0.0 1.5]
        R = assemble(form(Pxy, Pxy, (u, v) -> inner_Γ(2.0 * ∂ₙ(u), D₋ᵧ(v); markers)))
        A = assemble(form(Pxy, Pxy, (u, v) -> inner_Γ(∂ₙ(u; K = Kd), D₋ᵧ(v); markers)))
        @test nnz(R) > 0 && A ≈ R
        Ry = assemble(form(Pxy, Pxy, (u, v) -> inner_Γ(2.0 * ∂ₙ(u) + D₋ᵧ(u), v; markers)))
        Ay = assemble(form(Pxy, Pxy, (u, v) -> inner_Γ(∂ₙ(u; K = Kd) + D₋ᵧ(u), v; markers)))
        @test Ay ≈ Ry
        Wxy = gridspace(Ωₕ; ghosts = (:xmin, :ymin))
        @test constrained(Wxy, u -> :xmin => ∂ₙ(u; K = Kd) + D₋ᵧ(u) => 0.0) isa
              Bramble.GhostGridSpace
        # a read past the corner that is not `K`'s is refused at `assemble`, by its term
        lerr = try
            assemble(form(Px, Px, (u, v) -> inner_Γ(∂ₙ(u; K = Kd), Δₕ(v); markers)))
            nothing
        catch e
            e
        end
        @test lerr isa ArgumentError
        lmsg = sprint(showerror, lerr)
        @test occursin("does not exist", lmsg) && !occursin("K[1, 2]", lmsg)
        # with both y facets ghosted, the corner rows are the exact conormal flux
        P3 = padded(Ωₕ; ghosts = (:xmin, :ymin, :ymax))
        A = assemble(form(P3, P3, (u, v) -> inner_Γ(∂ₙ(u; K), v; markers)))
        py = [2y[1] - y[2]; y; 2y[end] - y[end - 1]]
        r = A * [yj for _ in 1:(ns[1] + 1), yj in py][:]
        LP = LinearIndices((ns[1] + 1, ns[2] + 2))
        for j in (1, 4, ns[2])
            @test r[LP[2, j + 1]] ≈ -ON.facet_weight(Ωₕ, 1, CartesianIndex(1, j)) * K[1, 2]
        end
    end

    @testset "Facet scalars and the element type" begin
        Ωₕ = random_mesh((6, 7); seed = 7)
        P = padded(Ωₕ)
        A2 = assemble(form(P, P, (u, v) -> inner_Γ(∂ₙ(u) / 2, v; markers = :boundary)))
        @test relerr(A2, sparse(0.5 * trial_rows(Ωₕ, FACETS[1:4]))) <= tol
        @test A2 ≈ assemble(form(P, P, (u, v) -> inner_Γ(0.5 * ∂ₙ(u), v; markers = :boundary)))
        @test relerr(assemble(form(P, P, (u, v) -> inner_Γ(u, ∂ₙ(v) / 4.0; markers = :boundary))),
            sparse(0.25 * trial_rows(Ωₕ, FACETS[1:4])')) <= tol

        Ω32 = mesh(domain(interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0)), (6, 7),
            (true, true))
        P32 = padded(Ω32)
        for t in ((u, v) -> inner_Γ(2 * ∂ₙ(u), v; markers = :boundary),
            (u, v) -> inner_Γ(∂ₙ(u; K = [2 0; 0 1]), v; markers = :boundary),
            (u, v) -> inner_Γ(∂ₙ(u) / 2, v; markers = :boundary))
            @test eltype(assemble(form(P32, P32, t))) == Float32
        end

        # a Ref divisor would freeze its value at build, so it is refused, not snapshotted
        @test_throws "freeze its value" form(P, P,
            (u, v) -> inner_Γ(∂ₙ(u) / Ref(2.0), v; markers = :boundary))
    end

    @testset "markers required, jumpₙ takes a field" begin
        Ωₕ = random_mesh((6, 7); seed = 7)
        P = padded(Ωₕ)
        # an empty selection is the plain inner_Γ's refusal, on each of the four methods
        @test_throws "needs at least one surface" form(P, P, (u, v) -> inner_Γ(∂ₙ(u), v))
        @test_throws "needs at least one surface" form(P, P, (u, v) -> inner_Γ(u, ∂ₙ(v)))
        @test_throws "needs at least one surface" form(P, P, (u, v) -> inner_Γ(∂ₙ(u), ∂ₙ(v)))
        @test_throws "needs at least one surface" form(P, v -> inner_Γ(x -> 1.0, ∂ₙ(v)))
        @test_throws ArgumentError form(P, P, (u, v) -> inner_Γ(u, v))
        # jumpₙ of a facet operator is named, not a MethodError
        @test_throws "jumpₙ takes a field expression" form(P, P, (u, v) -> inner_Γ(jumpₙ(∂ₙ(u)), v; markers = :boundary))
    end

    @testset "Linear form with ∂ₙ(v)" begin
        Ωₕ = random_mesh((6, 7); seed = 7)
        P = padded(Ωₕ)
        g(x) = 1.0 + x[1] + 2x[2]^2
        (; ns, ps) = oracle_ops(Ωₕ)
        xs = ON.axis_points(Ωₕ)
        gv = zeros(prod(ps))
        LP = LinearIndices(ps)
        for I in CartesianIndices(ns)
            gv[LP[I + oneunit(I)]] = g(ON.node_point(xs, I))
        end
        F = assemble(form(P, v -> inner_Γ(g, ∂ₙ(v); markers = (:xmin, :ymax))))
        RF = trial_rows(Ωₕ, (:xmin, :ymax))' * gv
        @test norm(F - RF) <= tol * norm(RF)
    end

    @testset "η on a ghost space names facet nodes" begin
        Ωₕ = random_mesh((6, 7); seed = 7)
        P = padded(Ωₕ)
        (; ps) = oracle_ops(Ωₕ)
        Wx = facet_weights(Ωₕ, 1, 1)
        l = assemble(form(P, v -> inner_Γ(dot((x -> 1.0, x -> 0.0), η), v;
            markers = (:xmin,))))
        @test norm(l + Wx.diag) <= tol * norm(Wx.diag)
        nₕ = Bramble.normal_vector(P, :xmin)
        @test findall(!iszero, parent(nₕ[1])) == findall(!iszero, Wx.diag)
        @test all(==(-1.0), filter(!iszero, parent(nₕ[1])))
    end

    if WITH_3D_TESTS
        @testset "3D trial and both sides" begin
            Ωₕ = random_mesh((4, 5, 6); seed = 7)
            P = padded(Ωₕ)
            (; Dc) = oracle_ops(Ωₕ)
            A = assemble(form(P, P, (u, v) -> inner_Γ(∂ₙ(u), v; markers = :boundary)))
            @test relerr(A, sparse(trial_rows(Ωₕ, FACETS))) <= tol
            C = assemble(form(P, P, (u, v) -> inner_Γ(∂ₙ(u), ∂ₙ(v); markers = (:zmax, :xmin))))
            RC = sum((:zmax, :xmin)) do f
                d, s = facet_axis_side(f)
                Dc[d]' * facet_weights(Ωₕ, d, s) * Dc[d]
            end
            @test relerr(C, sparse(RC)) <= tol
        end
    end

    @testset "Refusals" begin
        Ω1 = random_mesh((19,); seed = 7)
        Px = padded(Ω1; ghosts = (:xmin,))
        @test_throws ArgumentError form(Px, Px,
            (u, v) -> inner_Γ(∂ₙ(u), v; markers = (:xmax,)))
        # a centred difference reads the padded point beyond the facet, which Px lacks
        err = try
            assemble(form(Px, Px, (u, v) -> inner_Γ(Dcₓ(u), v; markers = (:xmax,))))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("a padded point that does not exist", err.msg)
        # the ghosted facet alone is fine
        @test form(Px, Px, (u, v) -> inner_Γ(∂ₙ(u), v; markers = (:xmin,))) isa Bramble.BilinearForm
        W = gridspace(Ω1)
        @test_throws ArgumentError form(W, W, (u, v) -> inner_Γ(∂ₙ(u), v; markers = :boundary))
        @test_throws ArgumentError form(W, W, (u, v) -> inner_Γ(u, ∂ₙ(v); markers = :boundary))

        Ω2 = random_mesh((6, 7); seed = 7)
        P2 = padded(Ω2; ghosts = (:xmin, :ymin))
        @test_throws ArgumentError form(P2, P2, (u, v) -> inner_Γ(∂ₙ(u), v; markers = :boundary))
        @test form(P2, P2, (u, v) -> inner_Γ(∂ₙ(u), v; markers = (:xmin, :ymin))) isa
              Bramble.BilinearForm
        P = padded(Ω2)
        Kf = [(x -> 1.0) (x -> 0.0); (x -> 0.0) (x -> 1.0)]
        @test_throws ArgumentError form(P, P, (u, v) -> inner_Γ(∂ₙ(u; K = Kf), v; markers = :boundary))
        @test_throws ArgumentError form(P, P, (u, v) -> inner_Γ(∂ₙ(u; K = x -> 1.0), v; markers = :boundary))
        @test_throws ArgumentError form(P, P, (u, v) -> inner_Γ(∂ₙ(u; K = ones(3, 3)), v; markers = :boundary))
        @test_throws ArgumentError form(P, P, (u, v) -> innerₕ(∂ₙ(u), v))
    end
end

# `J_F`: the inward difference `w_{I − η_F e_d} − w_I` at every padded point that has the
# inward neighbour, built entry by entry; `W_F` keeps only the facet's rows.
function inward_jump(Ωₕ, d, s)
    (; ps) = oracle_ops(Ωₕ)
    LP = LinearIndices(ps)
    step = CartesianIndex(ntuple(e -> e == d ? (s == 1 ? 1 : -1) : 0, length(ps)))
    rows, cols, vals = Int[], Int[], Float64[]
    for I in CartesianIndices(ps)
        J = I + step
        checkbounds(Bool, LP, J) || continue
        append!(rows, (LP[I], LP[I]))
        append!(cols, (LP[J], LP[I]))
        append!(vals, (1.0, -1.0))
    end
    return sparse(rows, cols, vals, prod(ps), prod(ps))
end

laps(Ωₕ) = Bramble.dim(Ωₕ) == 1 ? (O1.operators(Ωₕ).Lap,) : ON.operators(Ωₕ).Laps

# `Σ_F c_F W_F N_F` over the facets named, `N_F` the facet's operator matrix.
function facet_rows(Ωₕ, facets, N; c = f -> 1.0)
    return sum(facets) do f
        d, s = facet_axis_side(f)
        c(f) * facet_weights(Ωₕ, d, s) * N(d, s)
    end
end

@testset "∂ₙₙ and jumpₙ as facet operators" begin
    @testset "1D rows" begin
        Ωₕ = random_mesh((19,); seed = 7)
        P = padded(Ωₕ)
        n = length(points(Ωₕ))
        L = laps(Ωₕ)[1]
        A = assemble(form(P, P, (u, v) -> inner_Γ(∂ₙₙ(u), v; markers = :boundary)))
        R = spzeros(n + 2, n + 2)
        R[2, :] = L[2, :]
        R[n + 1, :] = L[n + 1, :]
        @test relerr(A, R) <= tol
        # the facet node's row reads the ghost, with the first interior spacing for it
        h2 = points(Ωₕ)[2] - points(Ωₕ)[1]
        @test A[2, 1] ≈ 1 / h2^2 && A[2, 3] ≈ 1 / h2^2
        J = assemble(form(P, P, (u, v) -> inner_Γ(jumpₙ(u), v; markers = :boundary)))
        Rj = sparse([2, 2, n + 1, n + 1], [2, 3, n + 1, n], [-1.0, 1.0, -1.0, 1.0],
            n + 2, n + 2)
        @test norm(J - Rj) <= tol
        @test norm(assemble(form(P, P, (u, v) -> inner_Γ(jumpₙ(u), jumpₙ(v); markers = :boundary))) -
                   Rj' * Rj) <= tol
    end

    @testset "2D ∂ₙₙ: trial, test, both sides" begin
        Ωₕ = random_mesh((6, 7); seed = 7)
        P = padded(Ωₕ)
        Ls = laps(Ωₕ)
        N = (d, s) -> Ls[d]
        R = facet_rows(Ωₕ, FACETS[1:4], N)
        @test relerr(assemble(form(P, P, (u, v) -> inner_Γ(∂ₙₙ(u), v; markers = :boundary))), sparse(R)) <= tol
        @test relerr(assemble(form(P, P, (u, v) -> inner_Γ(u, ∂ₙₙ(v); markers = :boundary))), sparse(R')) <= tol
        C = assemble(form(P, P, (u, v) -> inner_Γ(∂ₙₙ(u), ∂ₙₙ(v); markers = :boundary)))
        RC = sum(f -> (d = facet_axis_side(f)[1]; Ls[d]' * facet_rows(Ωₕ, (f,), N)),
            FACETS[1:4])
        @test relerr(C, sparse(RC)) <= tol
        for f in FACETS[1:4]
            Af = assemble(form(P, P, (u, v) -> inner_Γ(∂ₙₙ(u), v; markers = (f,))))
            @test relerr(Af, sparse(facet_rows(Ωₕ, (f,), N))) <= tol
        end
    end

    @testset "2D corner: each facet its own ghost" begin
        Ωₕ = random_mesh((6, 7); seed = 7)
        P = padded(Ωₕ)
        (; ps) = oracle_ops(Ωₕ)
        LP = LinearIndices(ps)
        corner = LP[2, 2]   # physical node (1, 1)
        Ax = assemble(form(P, P, (u, v) -> inner_Γ(∂ₙₙ(u), v; markers = (:xmin,))))
        @test Ax[corner, LP[1, 2]] != 0 && Ax[corner, LP[2, 1]] == 0
        Ay = assemble(form(P, P, (u, v) -> inner_Γ(∂ₙₙ(u), v; markers = (:ymin,))))
        @test Ay[corner, LP[2, 1]] != 0 && Ay[corner, LP[1, 2]] == 0
        Ab = assemble(form(P, P, (u, v) -> inner_Γ(∂ₙₙ(u), v; markers = :boundary)))
        @test Ab[corner, :] ≈ Ax[corner, :] + Ay[corner, :]
        @test Ab[corner, LP[1, 1]] == 0

        J = assemble(form(P, P, (u, v) -> inner_Γ(jumpₙ(u), v; markers = :boundary)))
        I = CartesianIndex(1, 1)
        ωx, ωy = ON.facet_weight(Ωₕ, 1, I), ON.facet_weight(Ωₕ, 2, I)
        @test ωx != ωy
        @test J[corner, LP[3, 2]] ≈ ωx && J[corner, LP[2, 3]] ≈ ωy
        @test J[corner, corner] ≈ -(ωx + ωy)
        @test count(!iszero, J[corner, :]) == 3   # no ghost read
    end

    @testset "2D jumpₙ: trial, test, both sides" begin
        Ωₕ = random_mesh((6, 7); seed = 7)
        P = padded(Ωₕ)
        N = (d, s) -> inward_jump(Ωₕ, d, s)
        R = facet_rows(Ωₕ, FACETS[1:4], N)
        @test relerr(assemble(form(P, P, (u, v) -> inner_Γ(jumpₙ(u), v; markers = :boundary))), sparse(R)) <= tol
        @test relerr(assemble(form(P, P, (u, v) -> inner_Γ(u, jumpₙ(v); markers = :boundary))), sparse(R')) <= tol
        # a max facet reads inward, never the ghost a forward jump would cross to
        Rx = facet_rows(Ωₕ, (:xmax,), N)
        Ax = assemble(form(P, P, (u, v) -> inner_Γ(jumpₙ(u), v; markers = (:xmax,))))
        @test relerr(Ax, sparse(Rx)) <= tol
        @test relerr(Ax, assemble(form(P, P, (u, v) -> inner_Γ(Bramble.jumpₓ(u), v;
            markers = (:xmax,))))) > 0.1
    end

    @testset "jumpₙ with c differing per facet" begin
        Ωₕ = random_mesh((6, 7); seed = 7)
        P = padded(Ωₕ)
        c = (xmin = 2.0, xmax = 0.5, ymin = 3.0, ymax = 1.5)
        a = form(P,
            P,
            (u, v) -> inner_Γ(c.xmin * jumpₙ(u), jumpₙ(v); markers = (:xmin,)) +
                      inner_Γ(c.xmax * jumpₙ(u), jumpₙ(v); markers = (:xmax,)) +
                      inner_Γ(c.ymin * jumpₙ(u), jumpₙ(v); markers = (:ymin,)) +
                      inner_Γ(c.ymax * jumpₙ(u), jumpₙ(v); markers = (:ymax,)))
        R = sum(FACETS[1:4]) do f
            d, s = facet_axis_side(f)
            J = inward_jump(Ωₕ, d, s)
            getfield(c, f) * J' * facet_weights(Ωₕ, d, s) * J
        end
        @test relerr(assemble(a), sparse(R)) <= tol
        Ax = assemble(form(P, P, (u, v) -> inner_Γ(c.xmin * jumpₙ(u), jumpₙ(v);
            markers = (:xmin,))))
        J1 = inward_jump(Ωₕ, 1, 1)
        @test relerr(Ax, sparse(c.xmin * J1' * facet_weights(Ωₕ, 1, 1) * J1)) <= tol
    end

    @testset "jumpₙ(Δₕ(u)) is the nesting" begin
        Ωₕ = random_mesh((6, 7); seed = 7)
        P = padded(Ωₕ)
        Lap = sum(laps(Ωₕ))
        R = facet_rows(Ωₕ, FACETS[1:4], (d, s) -> inward_jump(Ωₕ, d, s) * Lap)
        A = assemble(form(P, P, (u, v) -> inner_Γ(jumpₙ(Δₕ(u)), v; markers = :boundary)))
        @test relerr(A, sparse(R)) <= tol
        S(u) = Bramble.D̃ₓ(Bramble.D₋ₓ(u)) + Bramble.D̃ᵧ(Bramble.D₋ᵧ(u))
        @test relerr(A, assemble(form(P, P, (u, v) -> inner_Γ(jumpₙ(S(u)), v; markers = :boundary)))) <= tol
        # a max facet's row is `S₋(Δₕ u) − Δₕ u`, spelled out with the shift
        Ax = assemble(form(P, P, (u, v) -> inner_Γ(jumpₙ(Δₕ(u)), v; markers = (:xmax,))))
        Sx = assemble(form(P, P, (u, v) -> inner_Γ(Bramble.S₋ₓ(Δₕ(u)) - Δₕ(u), v;
            markers = (:xmax,))))
        @test relerr(Ax, Sx) <= tol
        B = assemble(form(P, P, (u, v) -> inner_Γ(jumpₙ(Δₕ(u)), jumpₙ(Δₕ(v)); markers = :boundary)))
        RB = sum(FACETS[1:4]) do f
            d, s = facet_axis_side(f)
            N = inward_jump(Ωₕ, d, s) * Lap
            N' * facet_weights(Ωₕ, d, s) * N
        end
        @test relerr(B, sparse(RB)) <= tol

        # on a plain space Δₕ is truncated at the boundary, and jumpₙ nests on it all the same
        W = gridspace(Ωₕ)
        Jp = assemble(form(W, W, (u, v) -> inner_Γ(jumpₙ(Δₕ(u)), v; markers = (:xmax,))))
        Sp = assemble(form(W, W, (u, v) -> inner_Γ(Bramble.S₋ₓ(Δₕ(u)) - Δₕ(u), v;
            markers = (:xmax,))))
        @test !iszero(Jp)
        @test relerr(Jp, Sp) <= tol
    end

    @testset "Linear forms and Float32" begin
        Ωₕ = random_mesh((6, 7); seed = 7)
        P = padded(Ωₕ)
        g(x) = 1.0 + x[1] + 2x[2]^2
        (; ns, ps) = oracle_ops(Ωₕ)
        xs = ON.axis_points(Ωₕ)
        gv = zeros(prod(ps))
        LP = LinearIndices(ps)
        for I in CartesianIndices(ns)
            gv[LP[I + oneunit(I)]] = g(ON.node_point(xs, I))
        end
        markers = (:xmin, :ymax)
        F = assemble(form(P, v -> inner_Γ(g, jumpₙ(v); markers)))
        RF = facet_rows(Ωₕ, markers, (d, s) -> inward_jump(Ωₕ, d, s))' * gv
        @test norm(F - RF) <= tol * norm(RF)
        G = assemble(form(P, v -> inner_Γ(g, ∂ₙₙ(v); markers)))
        RG = facet_rows(Ωₕ, markers, (d, s) -> laps(Ωₕ)[d])' * gv
        @test norm(G - RG) <= tol * norm(RG)

        Ω32 = mesh(domain(interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0)), (6, 7),
            (true, true))
        P32 = padded(Ω32)
        for t in ((u, v) -> inner_Γ(2 * ∂ₙₙ(u), v; markers = :boundary),
            (u, v) -> inner_Γ(jumpₙ(u), 3 * jumpₙ(v); markers = :boundary))
            @test eltype(assemble(form(P32, P32, t))) == Float32
        end
    end

    if WITH_3D_TESTS
        @testset "3D rows" begin
            Ωₕ = random_mesh((4, 5, 6); seed = 7)
            P = padded(Ωₕ)
            Ls = laps(Ωₕ)
            A = assemble(form(P, P, (u, v) -> inner_Γ(∂ₙₙ(u), v; markers = :boundary)))
            @test relerr(A, sparse(facet_rows(Ωₕ, FACETS, (d, s) -> Ls[d]))) <= tol
            J = assemble(form(P, P, (u, v) -> inner_Γ(jumpₙ(Δₕ(u)), v; markers = (:zmax, :xmin))))
            RJ = facet_rows(Ωₕ, (:zmax, :xmin), (d, s) -> inward_jump(Ωₕ, d, s) * sum(Ls))
            @test relerr(J, sparse(RJ)) <= tol
        end
    end

    @testset "∂ₙₙ and jumpₙ refusals" begin
        Ω1 = random_mesh((19,); seed = 7)
        Px = padded(Ω1; ghosts = (:xmin,))
        err = try
            form(Px, Px, (u, v) -> inner_Γ(∂ₙₙ(u), v; markers = (:xmax,)))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("∂ₙₙ", sprint(showerror, err)) && occursin(":xmax", sprint(showerror, err))
        @test form(Px, Px, (u, v) -> inner_Γ(∂ₙₙ(u), v; markers = (:xmin,))) isa
              Bramble.BilinearForm
        W = gridspace(Ω1)
        @test_throws ArgumentError form(W, W, (u, v) -> inner_Γ(∂ₙₙ(u), v; markers = :boundary))
        @test_throws ArgumentError form(W, W, (u, v) -> inner_Γ(u, ∂ₙₙ(v); markers = :boundary))
        @test_throws ArgumentError form(W, W, (u, v) -> inner_Γ(jumpₙ(u), ∂ₙₙ(v); markers = :boundary))
        # jumpₙ reads no ghost, so a space without any takes it
        n = length(points(Ω1))
        Jw = assemble(form(W, W, (u, v) -> inner_Γ(jumpₙ(u), v; markers = :boundary)))
        @test Jw == sparse([1, 1, n, n], [1, 2, n, n - 1], [-1.0, 1.0, -1.0, 1.0], n, n)

        Ω2 = random_mesh((6, 7); seed = 7)
        P2 = padded(Ω2; ghosts = (:xmin, :ymin))
        @test_throws ArgumentError form(P2, P2, (u, v) -> inner_Γ(∂ₙₙ(u), v; markers = :boundary))
        @test_throws ArgumentError form(P2, P2,
            (u, v) -> inner_Γ(∂ₙₙ(u), jumpₙ(v); markers = (:xmax,)))
        P = padded(Ω2)
        for f in (innerₕ, inner₊)
            @test_throws ArgumentError form(P, P, (u, v) -> f(∂ₙₙ(u), v))
            @test_throws ArgumentError form(P, P, (u, v) -> f(u, jumpₙ(v)))
        end
    end
end

end # module FormGhostFacetOperatorsTests
