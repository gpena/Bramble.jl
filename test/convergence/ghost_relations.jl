module ConvergenceGhostRelationsTests

# Second-order convergence of ghost-point boundary relations through the user's path only
# (gpena/Bramble.jl#205, #218): `gridspace(Ωₕ; ghosts)`, `constrained`, `form`,
# `assemble(a, l)` and `pde_solve`, against manufactured solutions.
#
# The cases: pure Neumann (`∂ₙ(u) => g` on every facet, solved with `mean_zero`, the exact
# solution shifted to discrete mean zero), mixed Robin and Neumann, a Helmholtz term
# `−Δu − k²u`, Dirichlet on `xmin` by both routes (a `u => g` relation and `assemble`'s
# `dirichlet` keyword), each in 1D, 2D and 3D; and the anisotropic conormal problem
# `−∇·(K∇u) = f` with `∂ₙ(u; K)` relations in 2D (the corner rule of the ghost layer).
#
# The error is the discrete H¹ norm `(‖e‖_{H,*}² + Σ_d ‖D₋_d e‖_+²)^{1/2}` on the physical
# nodes, its order the least-squares slope against `hₘₐₓ`, gated at > 1.8. Meshes are the
# oracles' `make_mesh` sequences: graded and random, uniform as an extra. On one mesh per
# case the library's solution is compared with the plain-matrix oracles' (test/ghost/),
# which share no code with the library, so that a wrong scheme which converges still fails.

using Test
using Bramble
import Bramble
using LinearAlgebra: Diagonal, dot, norm
using SparseArrays: sparse, spzeros
using ..TestUtils: _order_header, _order_row

isdefined(Main, :GhostOracle1D) ||
    Base.include(Main, joinpath(pkgdir(Bramble), "test", "ghost", "GhostOracle1D.jl"))
isdefined(Main, :GhostOracleND) ||
    Base.include(Main, joinpath(pkgdir(Bramble), "test", "ghost", "GhostOracleND.jl"))
isdefined(Main, :GhostCorners) ||
    Base.include(Main, joinpath(pkgdir(Bramble), "test", "ghost", "GhostCorners.jl"))
const O1 = Main.GhostOracle1D
const ON = Main.GhostOracleND
const OC = Main.GhostCorners

const FACETS = ON.FACETS
const β = ON.DEFAULT_β
const k² = 4.0                      # the Helmholtz term −k²u
const MIN_ORDER = 1.8
const AGREE = 1e-8                  # library against oracle, relative

relerr(a, b) = norm(a - b) / max(norm(b), eps())

#------------------------------------------------------------------------------------------#
# Geometry and the discrete H¹ error, from the mesh's points alone
#------------------------------------------------------------------------------------------#

axis_points(Ωₕ, D) = D == 1 ? (collect(points(Ωₕ)),) :
                     ntuple(d -> collect(points(Ωₕ(d))), D)

function cell_measures(x)
    h = diff(x)
    return [(i > 1 ? h[i - 1] : 0.0) / 2 + (i < length(x) ? h[i] : 0.0) / 2
            for i in eachindex(x)]
end

node_points(xs) = vec([ntuple(d -> xs[d][I[d]], length(xs))
                       for I in CartesianIndices(map(length, xs))])

hmax(xs) = maximum(x -> maximum(diff(x)), xs)

shape(v, d, D) = reshape(v, ntuple(k -> k == d ? length(v) : 1, D))

# The cell measures `ĥ_I = Π_d ĥ_d(I_d)` in mesh linear order.
function weights(xs)
    D = length(xs)
    return vec(mapreduce(d -> shape(cell_measures(xs[d]), d, D), (a, b) -> a .* b, 1:D))
end

function h1_error(xs, e)
    D = length(xs)
    ns = map(length, xs)
    E = reshape(e, ns)
    total = dot(weights(xs), abs2.(e))
    for d in 1:D
        w = mapreduce(
            k -> k == d ? shape(1 ./ diff(xs[d]), d, D) :
                 shape(cell_measures(xs[k]), k, D), (a, b) -> a .* b, 1:D)
        total += sum(w .* abs2.(diff(E; dims = d)))
    end
    return sqrt(total)
end

function lsq_order(hs, errs)
    X, Y = log.(hs), log.(errs)
    X̄, Ȳ = sum(X) / length(X), sum(Y) / length(Y)
    return sum((X .- X̄) .* (Y .- Ȳ)) / sum(abs2, X .- X̄)
end

#------------------------------------------------------------------------------------------#
# The manufactured problems
#------------------------------------------------------------------------------------------#

# The exact solution as functions of an indexable point: GhostOracle1D's eˣ cos 2x in 1D,
# GhostOracleND's exponential wave in 2D and 3D, so the oracles' data are the same.
function exact_solution(D)
    if D == 1
        ex = O1.DEFAULT_EXACT
        return (u = p -> ex.u(p[1]), grad = p -> (ex.du(p[1]),), lap = p -> ex.d2u(p[1]))
    end
    w = ON.default_exact(D)
    pt(p) = ntuple(d -> p[d], D)
    return (u = p -> ON.value(w, pt(p)), grad = p -> ON.gradient(w, pt(p)),
        lap = p -> ON.laplacian(w, pt(p)))
end

# Robin coefficients per facet (`FACETS` order): zero is Neumann.
function case_α(case, D)
    case === :neumann && return ntuple(_ -> 0.0, 6)
    case === :helmholtz && return ON.DEFAULT_α
    # Robin on the min facets, Neumann on the max ones.
    return ntuple(j -> isodd(j) ? ON.DEFAULT_α[j] : 0.0, 6)
end

case_κ(case) = case === :helmholtz ? k² : 0.0

make_mesh(D, kind, ℓ) = D == 1 ? O1.make_mesh(kind, ℓ) : ON.make_mesh(kind, ℓ; dim = D)

function facet_relation(u, ex, α, j)
    d, η = (j + 1) ÷ 2, isodd(j) ? -1.0 : 1.0
    a, b = α[j], β[j]
    g = p -> a * ex.u(p) + b * η * ex.grad(p)[d]
    return FACETS[j] => (iszero(a) ? b * ∂ₙ(u) : a * u + b * ∂ₙ(u)) => g
end

# `α u + β ∂ₙ(u) = g` on every facet, `u => g` on xmin when `dirichlet`, none there when
# `skip_xmin` (the facet left to `assemble`'s `dirichlet` keyword).
function relations(u, ex, D, α; dirichlet = false, skip_xmin = false)
    rels = Any[facet_relation(u, ex, α, j) for j in 1:(2D)]
    dirichlet && (rels[1] = :xmin => u => ex.u)
    skip_xmin && popfirst!(rels)
    return Tuple(rels)
end

"""
    library_solution(case, Ωₕ, D; route = :relation) -> Vector{Float64}

The discrete solution at every physical node (mesh order) through the public path. For
`:dirichlet`, `route = :relation` gives `xmin => u => g` in `constrained` (its nodes leave
the unknowns and take `g`) and `route = :keyword` the `dirichlet` keyword of `assemble`.
"""
function library_solution(case, Ωₕ, D; route = :relation)
    ex = exact_solution(D)
    κ = case_κ(case)
    f = p -> -ex.lap(p) - κ * ex.u(p)
    a(u, v) = iszero(κ) ? innerₕ(-1.0 * Δₕ(u), v) :
              innerₕ(-1.0 * Δₕ(u), v) - κ * innerₕ(u, v)
    if case === :dirichlet
        W = gridspace(Ωₕ; ghosts = FACETS[2:(2D)])
        if route === :relation
            C = constrained(W, u -> relations(u, ex, D, case_α(case, D); dirichlet = true))
            u = pde_solve(assemble(form(C, C, a), form(C, v -> innerₕ(f, v)))...)
            pts = node_points(axis_points(Ωₕ, D))
            fixed = [p[1] == pts[1][1] for p in pts]
            uh = similar(u, length(pts))
            uh[.!fixed] = u
            uh[fixed] = ex.u.(pts[fixed])
            return uh
        end
        C = constrained(W, u -> relations(u, ex, D, case_α(case, D); skip_xmin = true))
        A, F = assemble(form(C, C, a), form(C, v -> innerₕ(f, v));
            dirichlet = :xmin => ex.u, symmetrize = true)
        return pde_solve(A, F)
    end
    C = constrained(gridspace(Ωₕ; ghosts = :boundary),
        u -> relations(u, ex, D, case_α(case, D)))
    A, F = assemble(form(C, C, a), form(C, v -> innerₕ(f, v)))
    case === :neumann || return pde_solve(A, F)
    Wₕ = gridspace(Ωₕ)
    return pde_solve(A, F; mean_zero = assemble(form(Wₕ, v -> innerₕ(p -> 1.0, v))))
end

"""
    oracle_solution(case, Ωₕ, D) -> Vector{Float64}

The same problem through GhostOracle1D or GhostOracleND: ghosts eliminated by the relations
on the extended range, `−k²` added on the cell-measure diagonal, Dirichlet nodes lifted.
"""
function oracle_solution(case, Ωₕ, D)
    ex = exact_solution(D)
    κ = case_κ(case)
    f = p -> -ex.lap(p) - κ * ex.u(p)
    α = case_α(case, D)
    xs = axis_points(Ωₕ, D)
    pts = node_points(xs)
    ĥ = weights(xs)
    if D == 1
        rels = O1.problem_relations(:robin_a, Ωₕ; α = α[1:2], β = β[1:2])
        case === :dirichlet &&
            (rels = (xmin = O1.Dirichlet(ex.u(pts[1])), xmax = rels.xmax))
    else
        rels = ON.problem_relations(:robin_a, Ωₕ; α, β, exact = ON.default_exact(D))
    end
    A0, F0 = D == 1 ? O1.eliminated_system(rels, Ωₕ, f) : ON.eliminated_system(rels, Ωₕ, f)
    n = length(pts)
    fixed = case === :dirichlet ? [p[1] == pts[1][1] for p in pts] : falses(n)
    free = .!fixed
    if D == 1 || !any(fixed)
        A = A0 - κ * Diagonal(ĥ[free])
        F = F0
    else
        # The xmin rows go, their columns move into the load: no other row reads an xmin
        # ghost, so the remaining rows are the eliminated system with u = g there.
        A = A0[free, free] - κ * Diagonal(ĥ[free])
        F = F0[free] - A0[free, fixed] * ex.u.(pts[fixed])
    end
    if case === :neumann
        w = ĥ
        B = [sparse(A) sparse(reshape(w, :, 1)); sparse(reshape(w, 1, :)) spzeros(1, 1)]
        return (B \ vcat(F, 0.0))[1:n]
    end
    u = sparse(A) \ F
    uh = zeros(n)
    uh[free] = u
    uh[fixed] = ex.u.(pts[fixed])
    return uh
end

# The H¹ error of `uh` against the exact solution, shifted to mean zero for Neumann.
function case_error(case, Ωₕ, D, uh)
    xs = axis_points(Ωₕ, D)
    ue = exact_solution(D).u.(node_points(xs))
    case === :neumann && (ue .-= dot(weights(xs), ue) / sum(weights(xs)))
    return h1_error(xs, uh - ue)
end

function observed(case, D, kind, levels; route = :relation)
    hs, errs = Float64[], Float64[]
    for ℓ in levels
        Ωₕ = make_mesh(D, kind, ℓ)
        push!(hs, hmax(axis_points(Ωₕ, D)))
        push!(errs, case_error(case, Ωₕ, D, library_solution(case, Ωₕ, D; route)))
    end
    return hs, errs
end

#------------------------------------------------------------------------------------------#
# The anisotropic conormal problem (GhostCorners)
#------------------------------------------------------------------------------------------#

const K = OC.DEFAULT_K

# `α_F u + β_F ∂ₙ(u; K) = g_F` on every facet, the data of GhostCorners' exact solution.
function conormal_relations(u)
    ex = OC.DEFAULT_EXACT
    return ntuple(4) do F
        d, η = OC.facet_axis(F), OC.facet_normal(F)
        g = p -> begin
            ∇u = (ex.ux(p[1], p[2]), ex.uy(p[1], p[2]))
            OC.DEFAULT_α[F] * ex.u(p[1], p[2]) +
            OC.DEFAULT_β[F] * η * (K[d, 1] * ∇u[1] + K[d, 2] * ∇u[2])
        end
        OC.FACETS[F] => OC.DEFAULT_α[F] * u + OC.DEFAULT_β[F] * ∂ₙ(u; K) => g
    end
end

# `k₁₁ D̃ₓD₋ₓ + 2k₁₂ DcₓDcᵧ + k₂₂ D̃ᵧD₋ᵧ`, assembled as one form (the corner rule).
function anisotropic(u)
    xx = K[1, 1] * Bramble.D̃ₓ(Bramble.D₋ₓ(u))
    xy = 2 * K[1, 2] * Bramble.Dcₓ(Bramble.Dcᵧ(u))
    return xx + xy + K[2, 2] * Bramble.D̃ᵧ(Bramble.D₋ᵧ(u))
end

function conormal_solution(Ωₕ)
    ex = OC.DEFAULT_EXACT
    f = p -> -(K[1, 1] * ex.uxx(p[1], p[2]) + 2K[1, 2] * ex.uxy(p[1], p[2]) +
               K[2, 2] * ex.uyy(p[1], p[2]))
    C = constrained(gridspace(Ωₕ; ghosts = :boundary), conormal_relations)
    A, F = assemble(form(C, C, (u, v) -> innerₕ(-1.0 * anisotropic(u), v)),
        form(C, v -> innerₕ(f, v)))
    return pde_solve(A, F)
end

function conormal_error(Ωₕ, uh)
    xs = axis_points(Ωₕ, 2)
    return h1_error(xs, uh - [OC.DEFAULT_EXACT.u(p...) for p in node_points(xs)])
end

function conormal_observed(kind, levels)
    hs, errs = Float64[], Float64[]
    for ℓ in levels
        Ωₕ = OC.make_mesh(kind, ℓ)
        push!(hs, hmax(axis_points(Ωₕ, 2)))
        push!(errs, conormal_error(Ωₕ, conormal_solution(Ωₕ)))
    end
    return hs, errs
end

# One table per testset: `header(title)`, then `report(cells, hs, errs)` per case, `cells`
# the dimension, the case, the mesh kind and the route.
const WIDTHS = (5, 11, 9, 10)
header(title) = _order_header("Ghost relations, $title: discrete H¹ error",
    ("dim", "case", "mesh", "route"), WIDTHS)

function report(cells, hs, errs)
    r = lsq_order(hs, errs)
    _order_row(cells, r, errs, WIDTHS; gate = MIN_ORDER)
    return r
end

#------------------------------------------------------------------------------------------#
# Tests
#------------------------------------------------------------------------------------------#

const CASES = (:neumann, :mixed, :helmholtz, :dirichlet)
const KINDS = (:graded, :random, :uniform)

@testset "Ghost relations converge (#205)" begin
    @testset "1D Neumann, two random levels" begin
        hs, errs = observed(:neumann, 1, :random, 0:1)
        header("two levels")
        @test report(("1D", :neumann, :random, :relation), hs, errs) > MIN_ORDER
        Ωₕ = make_mesh(1, :random, 0)
        @test relerr(library_solution(:neumann, Ωₕ, 1), oracle_solution(:neumann, Ωₕ, 1)) <=
              AGREE
    end

    @testset "2D Robin/Neumann, two graded levels" begin
        hs, errs = observed(:mixed, 2, :graded, 1:2)
        @test report(("2D", :mixed, :graded, :relation), hs, errs) > MIN_ORDER
        Ωₕ = make_mesh(2, :graded, 0)
        @test relerr(library_solution(:mixed, Ωₕ, 2), oracle_solution(:mixed, Ωₕ, 2)) <=
              AGREE
    end

    @testset "$(D)D orders over four levels" for D in (1, 2)
        header("$(D)D, four levels")
        for case in CASES, kind in KINDS

            routes = case === :dirichlet ? (:relation, :keyword) : (:relation,)
            for route in routes
                hs, errs = observed(case, D, kind, 0:3; route)
                @test report(("$(D)D", case, kind, route), hs, errs) > MIN_ORDER
            end
        end
    end

    @testset "$(D)D agrees with the oracle" for D in (1, 2)
        for case in CASES
            Ωₕ = make_mesh(D, :random, 1)
            u0 = oracle_solution(case, Ωₕ, D)
            for route in (case === :dirichlet ? (:relation, :keyword) : (:relation,))
                @test relerr(library_solution(case, Ωₕ, D; route), u0) <= AGREE
            end
        end
    end

    @testset "3D orders over four levels" begin
        header("3D, four levels")
        for case in CASES, kind in KINDS

            routes = case === :dirichlet ? (:relation, :keyword) : (:relation,)
            for route in routes
                hs, errs = observed(case, 3, kind, 0:3; route)
                @test report(("3D", case, kind, route), hs, errs) > MIN_ORDER
            end
        end
    end

    @testset "3D agrees with the oracle" begin
        for case in CASES
            Ωₕ = make_mesh(3, :random, 0)
            u0 = oracle_solution(case, Ωₕ, 3)
            for route in (case === :dirichlet ? (:relation, :keyword) : (:relation,))
                @test relerr(library_solution(case, Ωₕ, 3; route), u0) <= AGREE
            end
        end
    end

    @testset "2D anisotropic ∂ₙ(u; K) orders" begin
        header("anisotropic ∂ₙ(u; K)")
        for kind in KINDS
            hs, errs = conormal_observed(kind, 0:3)
            @test report(("2D", :conormal, kind, :relation), hs, errs) > MIN_ORDER
        end
        Ωₕ = OC.make_mesh(:random, 0)
        u0 = OC.solve(:aniso_robin, Ωₕ; closure = :facet_onesided)
        @test relerr(conormal_solution(Ωₕ), u0) <= AGREE
    end
end

end # module ConvergenceGhostRelationsTests
