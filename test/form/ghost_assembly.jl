module FormGhostAssemblyTests

# Forms on a `GhostGridSpace` assemble `E_vᵀ A_pad E_u` in DOF order (physical nodes in
# mesh linear order, then the ghosts by facet xmin, xmax, ymin, ymax, zmin, zmax, each
# facet's column-major), with the load `E_vᵀ (F_pad − A_pad d_u)`.
#
# The operator rows (O6) are judged against the plain-matrix oracles (GhostOracle1D.jl,
# GhostOracleND.jl), which share no code with the library: at a node of a ghosted facet
# every family reads the ghost as the oracle does, and at a node of an unghosted facet it
# truncates as on `gridspace(Ωₕ)`. Rows are compared unscaled, as both sides carry the
# cell measure, the oracle's as `Diagonal(ĥ)`. Random and graded meshes only.

using Test
using Bramble
import Bramble
using Bramble: D₋ₓ, D₋ᵧ, D₋₂, D₊ₓ, D₊ᵧ, D₊₂, Dcₓ, Dcᵧ, Dc₂, D̃ₓ, D̃ᵧ, D̃₂, D̽ₓ, D̽ᵧ, D̽₂,
               Mₓ, Mᵧ, M₂, M₊ₓ, M₊ᵧ, M₊₂, jumpₓ, jumpᵧ, jump₂
using LinearAlgebra: Diagonal, norm, transpose, ×
using SparseArrays: SparseMatrixCSC, sparse, spzeros
using ForwardDiff: Dual
using ..TestUtils: WITH_3D_TESTS

isdefined(Main, :GhostOracle1D) ||
    Base.include(Main, joinpath(pkgdir(Bramble), "test", "ghost", "GhostOracle1D.jl"))
isdefined(Main, :GhostOracleND) ||
    Base.include(Main, joinpath(pkgdir(Bramble), "test", "ghost", "GhostOracleND.jl"))
const O1 = Main.GhostOracle1D
const O = Main.GhostOracleND

const tol = 1e-12

relerr(A, B) = norm(A - B) / max(norm(B), eps())

# --- References ---------------------------------------------------------------------- #

# The 1D operators of every family on the extended range `0 … n+1` of GhostOracle1D, the
# ones it lacks built here from its spacings the same way: D₊u_i = (u_{i+1} − u_i)/h_{i+1},
# M₊u_i = (u_i + u_{i+1})/2 (rows 0 … n), and D̽ (rows 1 … n) the cross-weighted
# `h_i/(h_i + h_{i+1}) D₋u_{i+1} + h_{i+1}/(h_i + h_{i+1}) D₋u_i`.
function extended_operators(Ω)
    ops = O1.operators(Ω)
    hp = O1.extended_spacings(Ω)
    N = length(hp)
    n = N - 2
    Dplus, Mplus, Dx = spzeros(N, N), spzeros(N, N), spzeros(N, N)
    for k in 1:(n + 1)              # paper i = k − 1 ∈ 0 … n
        Dplus[k, k + 1] = 1 / hp[k + 1]
        Dplus[k, k] = -1 / hp[k + 1]
        Mplus[k, k] = 0.5
        Mplus[k, k + 1] = 0.5
    end
    for k in 2:(n + 1)              # paper i ∈ 1 … n
        hi, hn = hp[k], hp[k + 1]
        s = hi + hn
        Dx[k, k + 1] += hi / (s * hn)
        Dx[k, k] += -hi / (s * hn) + hn / (s * hi)
        Dx[k, k - 1] -= hn / (s * hi)
    end
    return (; D₋ = ops.Dminus, D₊ = Dplus, Dc = ops.Dc, D̃ = ops.Dstar, D̽ = Dx,
        M₋ = ops.Mminus, M₊ = Mplus, jump = ops.Jump, D̃D₋ = ops.Lap,
        depth4 = ops.Dminus * Dplus * ops.Mminus * Mplus)
end

# The runtime families along axis d, in the order of `extended_operators`.
const AXIS_OPS = (D₋ = (D₋ₓ, D₋ᵧ, D₋₂), D₊ = (D₊ₓ, D₊ᵧ, D₊₂), Dc = (Dcₓ, Dcᵧ, Dc₂),
    D̃ = (D̃ₓ, D̃ᵧ, D̃₂), D̽ = (D̽ₓ, D̽ᵧ, D̽₂), M₋ = (Mₓ, Mᵧ, M₂), M₊ = (M₊ₓ, M₊ᵧ, M₊₂),
    jump = (jumpₓ, jumpᵧ, jump₂))

function family(name::Symbol, d::Int)
    name === :D̃D₋ && return u -> D̃_(d)(D₋_(d)(u))
    name === :depth4 && return u -> D₋_(d)(AXIS_OPS.D₊[d](AXIS_OPS.M₋[d](AXIS_OPS.M₊[d](u))))
    return AXIS_OPS[name][d]
end
D̃_(d) = AXIS_OPS.D̃[d]
D₋_(d) = AXIS_OPS.D₋[d]

const FAMILIES = (:D₋, :D₊, :Dc, :D̃, :D̽, :M₋, :M₊, :jump, :D̃D₋, :depth4)

# Families that read past the min side (a backward tap) and past the max side (forward).
const READS_MIN = (:D₋, :Dc, :D̽, :M₋, :D̃D₋, :depth4)
const READS_MAX = (:D₊, :Dc, :D̃, :D̽, :M₊, :jump, :D̃D₋, :depth4)

assembled(W, op) = Matrix(assemble(form(W, W, (u, v) -> innerₕ(op(u), v))))

const FACETS = (:xmin, :xmax, :ymin, :ymax, :zmin, :zmax)
facet(d, side) = FACETS[2(d - 1) + (side === :min ? 1 : 2)]
astuple(g::Symbol) = g === :boundary ? FACETS : (g,)
astuple(g::Tuple) = g

# The padded linear index of every DOF of `gridspace(Ωₕ; ghosts)`, from the oracle's layout:
# the physical nodes, then the named facets' ghosts in the oracle's (DOF) order.
function dof_columns(Ωₕ, ghosts)
    L = O.dof_layout(Ωₕ)
    D = length(L.ns)
    named = astuple(ghosts)
    keep = [facet(d, s) in named for (d, s, _) in L.ghosts]
    return vcat(L.phys, L.ghost[keep]), D
end

# The reference rows of `family(name, d)` on `gridspace(Ωₕ; ghosts)`: the oracle's row
# lifted to the padded grid, times the cell measure, at a node out of reach of the
# unghosted ends of axis d; the plain space's row, its ghost columns zero, at a node within
# reach of one (the end node, and the next for the depth-4 composition). Ghost rows are 0.
function reference_rows(Ωₕ, ghosts, name, d, P1)
    cols, D = dof_columns(Ωₕ, ghosts)
    ns = O.sizes(Ωₕ)
    np = prod(ns)
    ps = ns .+ 2
    Rpad = D == 1 ? Matrix(P1[name]) : Matrix(O.lift(P1[name], d, ps))
    ĥ = D == 1 ? O1.cell_measures(Ωₕ) : O.cell_measures(Ωₕ)
    R = zeros(length(cols), length(cols))
    R[1:np, :] = Diagonal(ĥ) * Rpad[cols[1:np], cols]
    named = astuple(ghosts)
    B = assembled(gridspace(Ωₕ), family(name, d))
    r = name === :depth4 ? 2 : 1    # how far in a node still reads past the end
    for (k, I) in enumerate(CartesianIndices(ns))
        unghosted = (I[d] <= r && !(facet(d, :min) in named)) ||
                    (I[d] > ns[d] - r && !(facet(d, :max) in named))
        unghosted || continue
        R[k, :] .= 0
        R[k, 1:np] = B[k, :]
    end
    return R
end

# The 1D extended operators of axis d of `Ωₕ`.
axis_operators(Ωₕ, d) = extended_operators(Bramble.dim(Ωₕ) == 1 ? Ωₕ : Ωₕ(d))

# The oracle's padded linear index of the point at padded linear index `j` of `Wg`'s mesh:
# Bramble pads the ghosted facets only, the oracle every axis at both ends.
function oracle_index(Wg, j)
    G = Bramble.mesh(Bramble.padded_space(Wg))
    D = Bramble.dim(Wg)
    J = CartesianIndices(Bramble.npoints(G, Tuple))[j]
    lo = Bramble.padded_index(G, zero(CartesianIndex{D}))
    ns = O.sizes(Bramble.physical_mesh(G))
    return LinearIndices(ns .+ 2)[J - lo + oneunit(J)]
end

# --- O11: the reduction --------------------------------------------------------------- #

@testset "1D matrix equals the oracle in DOF order" begin
    for kind in (:random, :graded)
        Ω = O1.make_mesh(kind, 1)
        n = length(points(Ω))
        Wg = gridspace(Ω; ghosts = :boundary)
        A = assemble(form(Wg, Wg, (u, v) -> innerₕ(-1.0 * D̃ₓ(D₋ₓ(u)), v)))
        @test size(A) == (n + 2, n + 2)
        perm = vcat(2:(n + 1), 1, n + 2)
        w = vcat(0.0, O1.cell_measures(Ω), 0.0)
        R = (Diagonal(w) * (-O1.operators(Ω).Lap))[perm, perm]
        @test relerr(A, R) <= tol
        @test A[1, n + 1] != 0 && A[n, n + 2] != 0   # the facet rows read their ghosts
    end
end

@testset "DOF order: physical, then by facet" begin
    for (D, ghosts) in ((1, (:xmax,)), (2, (:xmin, :ymax)), (2, :boundary))
        Ω = D == 1 ? O1.make_mesh(:random, 0) : O.make_mesh(:random, 0)
        Wg = gridspace(Ω; ghosts = ghosts)
        E, d = Bramble.closure(Wg)
        cols, _ = dof_columns(Ω, ghosts)
        @test Bramble.ndofs(Wg) == length(cols)
        @test [oracle_index(Wg, only(findall(!iszero, E[:, k]))) for k in axes(E, 2)] ==
              cols
        @test iszero(d)
    end
end

@testset "2D matrix is Eᵀ A_pad E" begin
    Ω = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (5, 7), (false, false))
    f = (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(Dcₓ(u), v) + innerₕ(u, D₊ᵧ(v))
    for ghosts in ((:xmin, :ymax), :boundary, (:ymin,))
        W = gridspace(Ω; ghosts = ghosts)
        P = Bramble.padded_space(W)
        E, _ = Bramble.closure(W)
        A = assemble(form(W, W, f))
        Ap = assemble(form(P, P, f))
        @test size(A) == (Bramble.ndofs(W), Bramble.ndofs(W))
        @test relerr(A, transpose(E) * Ap * E) <= 1e-13
    end
end

@testset "Linear form is Eᵀ F_pad" begin
    Ω = O1.make_mesh(:random, 1)
    Wg = gridspace(Ω; ghosts = :boundary)
    F = assemble(form(Wg, v -> innerₕ(x -> 1.0, v)))
    @test F ≈ vcat(O1.cell_measures(Ω), 0.0, 0.0)
    Ω2 = O.make_mesh(:random, 0)
    W2 = gridspace(Ω2; ghosts = (:xmax, :ymin))
    P2 = Bramble.padded_space(W2)
    E2, _ = Bramble.closure(W2)
    g = v -> innerₕ(x -> x[1] + x[2]^2, v) + inner_Γ(x -> 2.0 + x[1], v; markers = (:boundary,))
    @test assemble(form(W2, g)) ≈ transpose(E2) * assemble(form(P2, g))
end

# A hand-made closure with `d ≠ 0` (B8 produces real ones): a reflection Neumann closure on
# the physical DOFs alone, ghost_xmin = u_2 + d₁, ghost_xmax = u_{n−1} + d₂.
function reflected_space(Ω, d₁, d₂)
    Wg = gridspace(Ω; ghosts = :boundary)
    P = Bramble.padded_space(Wg)
    n = length(points(Ω))
    N = n + 2
    rows = vcat(2:(n + 1), 1, N)
    cols = vcat(1:n, 2, n - 1)
    E = sparse(rows, cols, ones(length(rows)), N, n)
    d = zeros(N)
    d[1], d[N] = d₁, d₂
    return Bramble.GhostGridSpace(P, E, d), P, E, d
end

@testset "Load with a lift is Eᵀ(F − A d)" begin
    Ω = O1.make_mesh(:graded, 0)
    W, P, E, d = reflected_space(Ω, 0.3, -0.7)
    fa = (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(D̃ₓ(D₋ₓ(u)), v)
    fl = v -> innerₕ(x -> exp(x[1]), v)
    a = form(W, W, fa)
    l = form(W, fl)
    A, F = assemble(a, l)
    Ap = assemble(form(P, P, fa))
    Fp = assemble(form(P, fl))
    @test relerr(A, transpose(E) * Ap * E) <= tol
    @test relerr(F, transpose(E) * (Fp - Ap * d)) <= tol
    @test norm(transpose(E) * Ap * d) > 1e-3 * norm(F)   # the lift is not negligible
    @test_throws ArgumentError assemble(l)
    @test_throws ArgumentError assemble!(similar(F), l)
    Bramble._assemble_ghost_load!(F, a, l)
    @test (@allocated Bramble._assemble_ghost_load!(F, a, l)) == 0
    @test relerr(F, transpose(E) * (Fp - Ap * d)) <= tol
end

@testset "Rectangular forms need one ghost set" begin
    Ω = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (5, 7), (false, false))
    f = (u, v) -> innerₕ(u, v) + innerₕ(D₊ᵧ(u), v)
    # equal padded sizes, rows shifted by one node: a silent misalignment before
    for (gw, gv) in (((:ymin,), (:ymax,)), ((:xmin,), (:xmax,)), ((:xmin,), :boundary))
        @test_throws ArgumentError form(gridspace(Ω; ghosts = gw), gridspace(Ω; ghosts = gv), f)
    end
    # the same ghost set on two spaces, one with another closure: E_vᵀ A_pad E_u
    W = gridspace(Ω; ghosts = (:ymin,))
    V = gridspace(Ω; ghosts = (:ymin,))
    P = Bramble.padded_space(W)
    Ev, _ = Bramble.closure(V)
    Eu = Ev[:, 1:(Bramble.ndofs(V) - 2)]
    U = Bramble.GhostGridSpace(P, Eu, zeros(size(Eu, 1)))
    Ap = assemble(form(P, Bramble.padded_space(V), f))
    for (Wt, E) in ((W, Ev), (U, Eu))
        A = assemble(form(Wt, V, f))
        @test size(A) == (Bramble.ndofs(V), size(E, 2))
        @test relerr(A, transpose(Ev) * Ap * E) <= 1e-13
    end
    B = assemble(form(gridspace(Ω), gridspace(Ω), (u, v) -> innerₕ(u, v)))
    @test relerr(assemble(form(W, V, (u, v) -> innerₕ(u, v)))[1:35, 1:35], B) <= 1e-13
end

@testset "Complex form with a real load and a lift" begin
    Ω = O1.make_mesh(:graded, 0)
    _, P, E, d = reflected_space(Ω, 0.3, -0.7)
    d[2] = 0.5     # a lift on a physical point, where the complex mass term reads it
    W = Bramble.GhostGridSpace(P, E, d)
    fa = (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ((1.0im) * u, v)
    fl = v -> innerₕ(x -> 1.0 + x[1], v)
    a, l = form(W, W, fa), form(W, fl)
    A, F = assemble(a, l)
    Ap = assemble(form(P, P, fa))
    Fp = assemble(form(P, fl))
    @test eltype(A) == ComplexF64 && eltype(F) == ComplexF64
    @test relerr(F, transpose(E) * (Fp - Ap * d)) <= tol
    @test norm(imag(F)) > 1e-6 * norm(F)   # the complex lift reaches the load
    # the real case still lifts in place
    ar, lr = form(W, W, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v))), form(W, fl)
    _, Fr = assemble(ar, lr)
    Bramble._assemble_ghost_load!(Fr, ar, lr)
    @test (@allocated Bramble._assemble_ghost_load!(Fr, ar, lr)) == 0
end

@testset "Mixed-precision trial and test spaces" begin
    Ω64 = mesh(domain(interval(0.0, 1.0)), 9, false)
    Ω32 = mesh(domain(interval(0.0f0, 1.0f0)), 9, false)
    f = (u, v) -> innerₕ(u, v) + innerₕ(D₋ₓ(u), v)
    W = gridspace(Ω64; ghosts = :boundary)
    V = gridspace(Ω32; ghosts = :boundary)
    a = form(W, V, f)
    A = assemble(a)
    Ap = assemble(form(Bramble.padded_space(W), Bramble.padded_space(V), f))
    Ev, _ = Bramble.closure(V)
    Eu, _ = Bramble.closure(W)
    @test size(A) == (11, 11)
    @test relerr(A, transpose(Ev) * Ap * Eu) <= 1e-6
    assemble!(A, a)
    @test (@allocated assemble!(A, a)) == 0
end

@testset "Refusals" begin
    Ω = O.make_mesh(:random, 0)
    Wg = gridspace(Ω; ghosts = (:xmin, :ymax))
    W = gridspace(Ω)
    f = (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v))
    fl = v -> innerₕ(x -> 1.0, v)
    a, l = form(Wg, Wg, f), form(Wg, fl)
    @test_throws ArgumentError form(Wg, W, f)
    @test_throws ArgumentError form(W, Wg, f)
    # `Wg` keeps its ghosts as unknowns, so the semidiscrete problem refuses it (the mass
    # rows of those ghosts vanish) before it reaches any other check.
    for call in (() -> semidiscretize(a, l), () -> ode_problem(a, l, zeros(Bramble.ndofs(Wg)), (0.0, 1.0)))
        err = try
            call()
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("keeps ghosts as unknowns", sprint(showerror, err))
    end
    @test_throws ArgumentError semidiscretize(form(W, W, f), l)
    @test_throws ArgumentError semidiscretize(t -> (form(W, W, f), t -> nothing), l)
    u₀ = zeros(Bramble.ndofs(Wg))
    @test_throws ArgumentError ode_problem(form(W, W, f), l, u₀, (0.0, 1.0))
    @test_throws ArgumentError Bramble.assemble_parallel!(assemble(a), a)
    @test_throws ArgumentError Bramble.assemble_parallel!(zeros(Bramble.ndofs(Wg)), l)
    @test_throws ArgumentError assemble(a; dirichlet = :boundary)
    @test_throws ArgumentError assemble(l; dirichlet = :boundary => x -> 0.0)
    @test_throws ArgumentError assemble(a, l; dirichlet = :boundary => x -> 0.0)
    @test_throws ArgumentError assemble(a, l; symmetrize = true)
    # a matrix of another pattern, and an AD element type
    @test_throws ArgumentError assemble!(assemble(form(W, W, f)), a)
    @test_throws ArgumentError assemble!(Matrix(assemble(a)), a)
    c = Ref(Dual(1.0, 1.0))
    @test_throws ArgumentError form(Wg, Wg, (u, v) -> innerₕ(c * u, v))
    @test_throws ArgumentError form(Wg, v -> innerₕ(c * v, v))
end

# gpena/Bramble.jl#818: two ghost spaces on meshes of one size and other points have equal
# closures, so the closure check alone paired a load with the wrong mesh's points.
@testset "Pair on another mesh is refused" begin
    Ωa = O.make_mesh(:random, 0; seed = 1)
    Ωb = O.make_mesh(:random, 0; seed = 2)
    Ωc = O.make_mesh(:random, 0; seed = 1)   # rebuilt alike: other object, same points
    @test npoints(Ωa, Tuple) == npoints(Ωb, Tuple) && points(Ωa) != points(Ωb)
    @test Ωa !== Ωc && points(Ωa) == points(Ωc)
    f = (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(u, v)
    fl = v -> innerₕ(x -> 1.0 + x[1] * x[2], v)
    Wa, Wb, Wc = (gridspace(Ω; ghosts = :boundary) for Ω in (Ωa, Ωb, Ωc))
    @test Bramble.closure(Wa) == Bramble.closure(Wb)   # only the mesh tells them apart
    a = form(Wa, Wa, f)
    A, F = assemble(a, form(Wa, fl))
    lb = form(Wb, fl)
    @test_throws DimensionMismatch assemble(a, lb)
    @test_throws DimensionMismatch assemble!(copy(A), copy(F), a, lb)
    # the same refusal on a composite side
    Va, Vb = Wa × Wa, Wb × Wb
    fv = (u, v) -> innerₕ(u(1), v(1)) + innerₕ(u(2), v(2))
    flv = v -> innerₕ(x -> 1.0, v(1)) + innerₕ(x -> x[1], v(2))
    av = form(Va, Va, fv)
    Av, Fv = assemble(av, form(Va, flv))
    @test_throws DimensionMismatch assemble(av, form(Vb, flv))
    @test_throws DimensionMismatch assemble!(copy(Av), copy(Fv), av, form(Vb, flv))
    @test assemble(av, form(Wc × Wc, flv))[2] == Fv
    # only the second component on another mesh: `mesh` of a composite reads the first
    for lv in (form(Wa × Wb, flv), form(Wb × Wa, flv))
        @test_throws DimensionMismatch assemble(av, lv)
        @test_throws DimensionMismatch assemble!(copy(Av), copy(Fv), av, lv)
    end
    # and on a space constrained as a whole
    Xa, Xb = (gridspace(Ω; ghosts = (:xmin, :xmax)) for Ω in (Ωa, Ωb))
    rels = u -> (:xmin => ∂ₙ(u(1)) => 0.0, :xmax => ∂ₙ(u(1)) => 0.0,
        :xmin => ∂ₙ(u(2)) => 0.0, :xmax => ∂ₙ(u(2)) => 0.0)
    Ca, Cb = constrained(Xa × Xa, rels), constrained(Xa × Xb, rels)
    @test Bramble.closure(Ca) == Bramble.closure(Cb)
    ac = form(Ca, Ca, fv)
    Ac, Fc = assemble(ac, form(Ca, flv))
    @test_throws DimensionMismatch assemble(ac, form(Cb, flv))
    @test_throws DimensionMismatch assemble!(copy(Ac), copy(Fc), ac, form(Cb, flv))
    Cc = constrained(Xa × gridspace(Ωc; ghosts = (:xmin, :xmax)), rels)
    @test assemble(ac, form(Cc, flv))[2] == Fc
    # a mesh rebuilt alike pairs, with the same load; one space stays 0 B warm
    lc = form(Wc, fl)
    @test assemble(a, lc)[2] == F
    la = form(Wa, fl)
    assemble!(A, F, a, la)
    @test (@allocated assemble!(A, F, a, la)) == 0
    # DECISION (#818): plain spaces keep today's behaviour. Their pair, `assemble(a, l)` in
    # bilinear.jl, assembles each form alone and shares no helper with `_check_pair`, so a
    # load on another mesh, even of another size, is still accepted.
    P = gridspace(Ωa)
    Ps = gridspace(O.make_mesh(:random, 0; seed = 3, n0 = 5))
    Ap, Fp = assemble(form(P, P, f), form(Ps, fl))
    @test size(Ap, 1) != length(Fp)
end

# --- O6: operator rows read the ghost at a ghosted facet ------------------------------ #

@testset "1D operator rows against oracle" begin
    for kind in (:random, :graded), ghosts in (:boundary, (:xmin,), (:xmax,))

        Ω = O1.make_mesh(kind, 0)
        n = length(points(Ω))
        P1 = extended_operators(Ω)
        Wg = gridspace(Ω; ghosts = ghosts)
        named = astuple(ghosts)
        for name in FAMILIES
            A = assembled(Wg, family(name, 1))
            R = reference_rows(Ω, ghosts, name, 1, P1)
            @test relerr(A, R) <= tol
            # the facet row reads its ghost, the last DOF column(s)
            g = n + 1
            if :xmin in named
                name in READS_MIN && @test A[1, g] != 0
                g += 1
            end
            :xmax in named && name in READS_MAX && @test A[n, g] != 0
        end
    end
end

function nd_rows_testset(D, kinds, ghost_sets)
    for kind in kinds, ghosts in ghost_sets

        Ω = O.make_mesh(kind, 0; dim = D)
        Wg = gridspace(Ω; ghosts = ghosts)
        for d in 1:D
            P1 = axis_operators(Ω, d)
            for name in FAMILIES
                A = assembled(Wg, family(name, d))
                @test relerr(A, reference_rows(Ω, ghosts, name, d, P1)) <= tol
            end
        end
        # Δ as the sum over axes: a corner of a ghosted and an unghosted facet reads one
        # ghost and truncates along the other axis
        L = u -> sum(d -> D̃_(d)(D₋_(d)(u)), 1:D)
        R = sum(d -> reference_rows(Ω, ghosts, :D̃D₋, d, axis_operators(Ω, d)), 1:D)
        @test relerr(assembled(Wg, L), R) <= tol
    end
end

@testset "2D operator rows against oracle" begin
    nd_rows_testset(2, (:random, :graded), ((:xmin, :ymax), :boundary, (:xmax, :ymin)))
end

# 3D runs nightly (`WITH_3D_TESTS`).
WITH_3D_TESTS && @testset "3D operator rows against oracle" begin
    nd_rows_testset(3, (:random,), ((:xmin, :ymax, :zmin), :boundary))
end

end # module FormGhostAssemblyTests
