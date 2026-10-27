module GhostCompositeTests

# Composite spaces with ghost components (#781): `C₁ × C₂` and `C^Val(n)` of ghost spaces
# assemble a vector problem with per-component relations. The reference of every check is
# built from the padded composite `padded(C₁) × padded(C₂)` and the block-diagonal closure
# `blockdiag(E₁, E₂)`, `vcat(d₁, d₂)` read off the components, as dense matrices, sharing no
# reduction code with the library: `A = Eᵀ A_pad E`, `F = Eᵀ (F_pad − A_pad d)`.
#
# Covered: the product and its layout, assembly and refills, `a(u, v)` and `l(v)`, `lift`,
# the `dirichlet` keyword per component against the `u => g` route, `pde_solve` against its
# block reference, and the refusals: mixed or nested composites, components ghosting
# different facets, a scalar space on the other side, rank-one terms, `semidiscretize`, and
# the matrix-free, Kronecker and multigrid operators. The second-order convergence of the
# coupled problem (GhostCoupled.jl) is in test/convergence/ghost_composite.jl.

using Test
using Bramble
import Bramble
using Bramble: D₋ₓ, D₊ᵧ, Dcₓ, Dcᵧ, Mₓ, lift, lift!
using LinearAlgebra: dot, norm
using SparseArrays: blockdiag, sparse
using Random: Xoshiro

using ..TestUtils: WITH_SLOW_TESTS, WITH_3D_TESTS, _Const, _x1, _x2, _x3, _one_plus_x1

isdefined(Main, :GhostOracleND) ||
    Base.include(Main, joinpath(pkgdir(Bramble), "test", "ghost", "GhostOracleND.jl"))
const O = Main.GhostOracleND
isdefined(Main, :GhostCoupled) || Base.include(Main, joinpath(@__DIR__, "GhostCoupled.jl"))
using Main.GhostCoupled: u2, fa_c, fl_c, coupled_solve

# Refused with an `ArgumentError` whose message holds `needle`: the message pins which
# refusal fired.
function refuses(f, needle)
    try
        f()
    catch e
        return e isa ArgumentError && occursin(needle, sprint(showerror, e))
    end
    return false
end

# --- References, from the components alone ------------------------------------------- #

padded_ref(V) = foldl(×, map(Bramble.padded_space, V.spaces))

function closure_ref(V)
    E = blockdiag(map(C -> sparse(Bramble.closure(C)[1]), V.spaces)...)
    return Matrix(E), vcat(map(C -> Vector(Bramble.closure(C)[2]), V.spaces)...)
end

function block_reference(Vu, Vv, fa, fl)
    Pu, Pv = padded_ref(Vu), padded_ref(Vv)
    Ap = Matrix(assemble(form(Pu, Pv, fa)))
    Fp = assemble(form(Pv, fl))
    Eu, du = closure_ref(Vu)
    Ev, _ = closure_ref(Vv)
    return transpose(Ev) * Ap * Eu, transpose(Ev) * (Fp - Ap * du)
end

close(a, b; rtol = 1.0e-12) = norm(a - b) <= rtol * max(norm(b), 1)

# --- Spaces ---------------------------------------------------------------------------- #

mesh2d(; n0 = 7) = O.make_mesh(:random, 0; dim = 2, n0 = n0)

const GH = (:xmin, :xmax, :ymin)

# The components are built on one ghost space `S = gridspace(Ωₕ; ghosts = GH)` unless a test
# says otherwise.
# Neumann, Robin with data, Neumann with data, Dirichlet `u => g` on the unghosted facet.
function space_a(S)
    constrained(S,
        u -> (:xmin => ∂ₙ(u) => 1.0, :xmax => 2.0 * u + ∂ₙ(u) => _x2,
            :ymin => ∂ₙ(u) => 0.5, :ymax => u => _x1))
end

# Robin, Neumann, Robin with data, and the unghosted facet left free.
function space_b(S)
    constrained(S,
        u -> (:xmin => 2.0 * u + ∂ₙ(u) => 0.5, :xmax => ∂ₙ(u) => 0.0,
            :ymin => u + ∂ₙ(u) => _x1))
end

# The mixed difference reads the ghosts beside the corners; at a corner node of a padded
# composite it takes the one-sided corner rule, as on a scalar space (tested below).
const fa2 = (u, v) -> inner₊(∇ₕ(u(1)), ∇ₕ(v(1))) + innerₕ(-1.0 * Δₕ(u(2)), v(2)) +
                      0.5 * innerₕ(u(2), v(2)) + innerₕ(u(2), v(1)) +
                      innerₕ(D₋ₓ(D₊ᵧ(u(1))), v(2))
const fl2 = v -> innerₕ(_one_plus_x1, v(1)) + innerₕ(x -> x[2]^2, v(2))

# --- The product and its layout -------------------------------------------------------- #

@testset "Composite ghost products and layout" begin
    Ωₕ = mesh2d()
    S = gridspace(Ωₕ; ghosts = GH)
    Ca, Cb = space_a(S), space_b(S)
    V = Ca × Cb
    @test V isa Bramble.CompositeGridSpace{2}
    @test V.spaces[1] === Ca && V.spaces[2] === Cb
    @test (Ca × Cb × Ca).spaces == (Ca, Cb, Ca)
    @test (Ca × (Cb × Ca)).spaces == (Ca, Cb, Ca)
    @test (V × V).spaces == (Ca, Cb, Ca, Cb)
    @test (Ca^Val(3)).spaces == (Ca, Ca, Ca)
    @test (Ca^2).spaces == (Ca, Ca) && (Ca^3).spaces == (Ca, Ca, Ca)
    @test Ca^1 === Ca && Ca^Val(1) === Ca
    @test refuses(() -> Ca^4, "Cₕ^Val(N)")

    @test ndofs(V) == ndofs(Ca) + ndofs(Cb)
    @test Bramble.component_ranges(V) == (1:ndofs(Ca), (ndofs(Ca) + 1):ndofs(V))
    @test Bramble.leaf_spaces_offsets(V) == ((Ca, 0), (Cb, ndofs(Ca)))
    @test Bramble.leaf_count(V) == 2
    @test mesh(V) === mesh(Ca) && dim(V) == 2 && eltype(V) == Float64

    u = element(V)
    @test length(u) == ndofs(V)
    u .= rand(Xoshiro(1), length(u))
    @test u(1).space === Ca && u(2).space === Cb
    @test parent(u(2)) == parent(u)[(ndofs(Ca) + 1):end]
    u(1) .= 3.0
    @test all(==(3.0), parent(u)[1:ndofs(Ca)])
    @test map(c -> c.space, Bramble.components(u)) == (Ca, Cb)

    @test Bramble.padded_space(V) === padded_ref(V)
    E, d = Bramble.closure(V)
    Er, dr = closure_ref(V)
    @test Matrix(E) == Er && d == dr
    @test d !== Ca.d && d !== Bramble.closure(V)[2]
end

# --- Assembly against the block reference --------------------------------------------- #

function check_assembly(Vu, Vv, fa, fl; zero_alloc = true)
    a, l = form(Vu, Vv, fa), form(Vv, fl)
    Aref, Fref = block_reference(Vu, Vv, fa, fl)
    A = assemble(a)
    @test close(Matrix(A), Aref)
    A2, F2 = assemble(a, l)
    @test A2 == A
    @test close(F2, Fref)
    # NaN-poisoned refills, of the matrix alone and of the pair
    A3 = copy(A)
    A3.nzval .= NaN
    @test assemble!(A3, a) == A
    F3 = fill(NaN, length(F2))
    A3.nzval .= NaN
    assemble!(A3, F3, a, l)
    @test A3 == A2 && F3 == F2
    if zero_alloc
        @test (@allocated assemble!(A3, a)) == 0
        @test (@allocated assemble!(A3, F3, a, l)) == 0
    end
    rng = Xoshiro(2)
    u, v = element(Vu), element(Vv)
    u .= rand(rng, length(u))
    v .= rand(rng, length(v))
    @test a(u, v) ≈ dot(parent(v), Aref * parent(u))
    return A, F2
end

@testset "Composite ghost assembly" begin
    Ωₕ = mesh2d()
    S = gridspace(Ωₕ; ghosts = GH)
    Ca, Cb = space_a(S), space_b(S)
    V = Ca × Cb
    a = form(V, V, fa2)
    @test a isa Bramble.GhostCompositeBilinearForm
    @test Bramble.trial_space(a) === V && Bramble.test_space(a) === V
    @test Bramble.test_space(form(V, fl2)) === V
    @test startswith(sprint(show, a), "GhostCompositeBilinearForm{2D}(CompositeGridSpace")
    @test_throws DimensionMismatch a(ones(3), ones(ndofs(V)))
    check_assembly(V, V, fa2, fl2)
    # components from two `gridspace` calls on one mesh, two padded mesh objects: the same
    # system, the coupling block's mixed difference included
    Vd = space_a(gridspace(Ωₕ; ghosts = GH)) × space_b(gridspace(Ωₕ; ghosts = GH))
    @test mesh(Vd.spaces[1]) !== mesh(Vd.spaces[2])
    @test assemble(form(Vd, Vd, fa2), form(Vd, fl2)) == assemble(a, form(V, fl2))
    if WITH_SLOW_TESTS
        # different closures on the two sides: a rectangular block system
        check_assembly(Ca × Cb, Cb × Ca, fa2, fl2)
        # three copies of one space, every pair coupled
        fa3 = (u, v) -> inner₊(∇ₕ(u(1)), ∇ₕ(v(1))) + innerₕ(-1.0 * Δₕ(u(2)), v(3)) +
                        innerₕ(u(3), v(2)) + 2.0 * innerₕ(u(3), v(3)) + innerₕ(u(1), v(2))
        fl3 = v -> innerₕ(_x1, v(1)) + innerₕ(_Const(1.0), v(3))
        check_assembly(Ca^Val(3), Ca^Val(3), fa3, fl3)

        # 1D: a constrained component beside a selection space that keeps its ghosts
        Ω1 = O.O1.make_mesh(:random, 0; n0 = 9)
        C1 = constrained(gridspace(Ω1; ghosts = :boundary),
            u -> (:xmin => ∂ₙ(u) => 1.0, :xmax => u + ∂ₙ(u) => 2.0))
        S1 = gridspace(Ω1; ghosts = :boundary)
        fa1 = (u, v) -> innerₕ(-1.0 * Δₕ(u(1)), v(1)) + innerₕ(u(2), v(2)) + innerₕ(u(1), v(2))
        fl1 = v -> innerₕ(identity, v(1)) + innerₕ(_Const(1.0), v(2))
        check_assembly(C1 × S1, C1 × S1, fa1, fl1)
    end

    # A selection space has d = 0, so its load stands alone, and l(v) is Fᵀv.
    Sb = gridspace(Ωₕ; ghosts = :boundary)
    W = Sb × Sb
    l = form(W, fl2)
    _, Fref = block_reference(W, W, (u, v) -> innerₕ(u(1), v(1)) + innerₕ(u(2), v(2)), fl2)
    F = assemble(l)
    @test close(F, Fref)
    v = element(W)
    v .= rand(Xoshiro(3), length(v))
    @test l(v) ≈ dot(Fref, parent(v))
    scratch = fill(NaN, ndofs(W))
    @test Bramble.evaluate!(scratch, l, v) ≈ dot(Fref, parent(v))
    @test close(scratch, Fref)
    # a vector of another length, named with both lengths
    w = element(gridspace(Ωₕ)^3)
    @test_throws DimensionMismatch l(w)
    @test_throws DimensionMismatch Bramble.evaluate!(zeros(ndofs(W)), l, w)
    err = try
        l(w)
        nothing
    catch e
        e
    end
    @test occursin(
        "a linear form on a composite of ghost spaces with " *
        "$(ndofs(W)) degrees of freedom cannot contract a vector of length " *
        "$(length(w))",
        sprint(showerror, err))
    # with data, the load alone refuses as on a scalar space
    @test refuses(() -> assemble(form(V, fl2)), "assemble(a, l)")

    if WITH_3D_TESTS
        Ω3 = O.make_mesh(:random, 0; dim = 3, n0 = 5)
        C3 = constrained(gridspace(Ω3; ghosts = (:xmin, :zmax)),
            u -> (:xmin => ∂ₙ(u) => 1.0, :zmax => u + ∂ₙ(u) => _x2,
                :ymin => u => 0.0))
        fa33 = (u, v) -> inner₊(∇ₕ(u(1)), ∇ₕ(v(1))) + innerₕ(-1.0 * Δₕ(u(2)), v(2)) +
                         innerₕ(u(2), v(1))
        fl33 = v -> innerₕ(_x3, v(1)) + innerₕ(_Const(1.0), v(2))
        check_assembly(C3^2, C3^2, fa33, fl33; zero_alloc = false)
    end
end

# --- The corner rule per component ------------------------------------------------------ #

# A centred mixed difference of a component reads the absent padded corner point at a corner
# node (2D) or an edge node (3D) and takes the one-sided corner rule there, as the scalar
# form does: each block of the padded composite matrix is the scalar padded matrix, and the
# reduced matrix is `Eᵀ A_pad E`.
function check_corner_rule(Ωₕ)
    S = gridspace(Ωₕ; ghosts = :boundary)
    P = Bramble.padded_space(S)
    V = S × S
    PV = Bramble.padded_space(V)
    m = ndofs(P)
    Aref = Matrix(assemble(form(P, P, (u, v) -> innerₕ(Dcₓ(Dcᵧ(u)), v))))
    for (i, j) in ((1, 1), (1, 2), (2, 1))
        A = Matrix(assemble(form(PV, PV, (u, v) -> innerₕ(Dcₓ(Dcᵧ(u(j))), v(i)))))
        B = A[((i - 1) * m + 1):(i * m), ((j - 1) * m + 1):(j * m)]
        @test close(B, Aref)
        @test count(!iszero, A) == count(!iszero, B)
    end
    # two blocks at once, with coefficients that cannot be confused
    A = Matrix(assemble(form(PV, PV,
        (u, v) -> innerₕ(Dcₓ(Dcᵧ(u(2))), v(1)) + 100.0 * innerₕ(Dcₓ(Dcᵧ(u(1))), v(2)))))
    @test close(A[1:m, (m + 1):(2m)], Aref)
    @test close(A[(m + 1):(2m), 1:m], 100.0 * Aref)
    @test iszero(A[1:m, 1:m]) && iszero(A[(m + 1):(2m), (m + 1):(2m)])
    fc = (u, v) -> innerₕ(Dcₓ(Dcᵧ(u(2))), v(1))
    E, _ = Bramble.closure(V)
    @test close(Matrix(assemble(form(V, V, fc))),
        Matrix(E' * assemble(form(PV, PV, fc)) * E))
end

@testset "Composite ghost corner rule" begin
    check_corner_rule(mesh2d())
    WITH_3D_TESTS && check_corner_rule(O.make_mesh(:random, 0; dim = 3, n0 = 5))
end

# --- lift ---------------------------------------------------------------------------- #

@testset "Composite ghost lift" begin
    Ωₕ = mesh2d()
    S = gridspace(Ωₕ; ghosts = GH)
    Ca, Cb = space_a(S), space_b(S)
    V = Ca × Cb
    E, d = closure_ref(V)
    u = rand(Xoshiro(4), ndofs(V))
    U = lift(V, u)
    @test U.space === padded_ref(V)
    @test close(parent(U), E * u + d)
    @test parent(lift(V)) == d
    u2 = rand(Xoshiro(5), ndofs(V))
    lift!(U, V, u2)
    @test close(parent(U), E * u2 + d)
    @test (@allocated lift!(U, V, u2)) == 0
    @test refuses(() -> lift(V, ones(3)), "one value per degree of freedom")
    @test refuses(() -> lift(V, "u"), "lift(Vₕ, u)")
    @test refuses(() -> lift!(lift(Ca, u[1:ndofs(Ca)]), V, u), "lives on another space")
    @test refuses(() -> lift!(U, V, u .+ 0im), "gives values of type")

    # relation data in time on one component: lift at t is the static space frozen at t
    g(x, t) = sin(t) * x[2]
    rel(h) = u -> (:xmin => ∂ₙ(u) => h, :xmax => ∂ₙ(u) => 0.0, :ymin => ∂ₙ(u) => 0.0,
        :ymax => u => 0.0)
    Ct = constrained(gridspace(Ωₕ; ghosts = GH), rel(g))
    Vt = Ct × Cb
    t = 0.7
    Ef, df = closure_ref(constrained(gridspace(Ωₕ; ghosts = GH), rel(x -> g(x, t))) × Cb)
    @test close(parent(lift(Vt, u, t)), Ef * u + df)
    Ut = lift(Vt, u, t)
    lift!(Ut, Vt, u2, t)
    @test close(parent(Ut), Ef * u2 + df)
    @test refuses(() -> lift(Vt, u), "depend on time")
    @test refuses(() -> lift(Vt), "depend on time")
    # assemble(a) reads E only; the pair needs one d
    fa = (u, v) -> inner₊(∇ₕ(u(1)), ∇ₕ(v(1))) + innerₕ(u(2), v(2))
    at = form(Vt, Vt, fa)
    @test close(Matrix(assemble(at)), transpose(Ef) * Matrix(assemble(form(
                                          padded_ref(Vt), padded_ref(Vt), fa))) * Ef)
    @test refuses(() -> assemble(at, form(Vt, fl2)), "depend on time")
    Vt2 = Cb × Ct
    @test refuses(() -> assemble(form(Vt2, Vt2, fa), form(Vt2, fl2)), "depend on time")

    # a changed mesh makes the space stale
    Ωs = mesh2d()
    Vs = space_b(gridspace(Ωs; ghosts = GH))^2
    Bramble.change_points!(Ωs, map(x -> collect(x) .^ 1.2, O.axis_points(Ωs)))
    @test refuses(() -> lift(Vs, zeros(ndofs(Vs))), "changed in place")
end

# --- Dirichlet per component ----------------------------------------------------------- #

# Facet nodes of the physical mesh in its linear order, and their points.
function facet_nodes(Ωₕ, facets)
    xs = O.axis_points(Ωₕ)
    I = vec(CartesianIndices(map(length, xs)))
    on(J) = any(
        F -> F === :ymin ? J[2] == 1 :
             F === :ymax ? J[2] == length(xs[2]) :
             F === :xmin ? J[1] == 1 : J[1] == length(xs[1]),
        facets)
    k = findall(on, I)
    return k, [(xs[1][I[j][1]], xs[2][I[j][2]]) for j in k]
end

# Row replacement and symmetrization by hand on dense matrices.
function rows_ref(A, F, rows, vals; symmetrize = false)
    A, F = copy(A), copy(F)
    for (r, g) in zip(rows, vals)
        A[r, :] .= 0
        A[r, r] = 1
        F[r] = g
    end
    if symmetrize
        for r in rows, k in axes(A, 1)

            k == r && continue
            F[k] -= A[k, r] * F[r]
            A[k, r] = 0
        end
    end
    return A, F
end

@testset "Composite ghost Dirichlet per component" begin
    Ωₕ = mesh2d()
    gD = x -> 1 + x[1] * x[2]
    # component 1 fixes ymin by a relation, so its unknowns are not its physical nodes;
    # component 2 leaves ymin and ymax free for the `dirichlet` keyword
    Ce = constrained(gridspace(Ωₕ; ghosts = (:xmin, :xmax)),
        u -> (:xmin => ∂ₙ(u) => 0.0, :xmax => ∂ₙ(u) => 0.5, :ymin => u => gD))
    Cd = constrained(gridspace(Ωₕ; ghosts = (:xmin, :xmax)),
        u -> (:xmin => ∂ₙ(u) => 1.0, :xmax => u + ∂ₙ(u) => 0.0))
    Cr = constrained(gridspace(Ωₕ; ghosts = (:xmin, :xmax)),
        u -> (:xmin => ∂ₙ(u) => 1.0, :xmax => u + ∂ₙ(u) => 0.0, :ymin => u => gD,
            :ymax => u => gD))
    fa = (u, v) -> inner₊(∇ₕ(u(1)), ∇ₕ(v(1))) + innerₕ(-1.0 * Δₕ(u(2)), v(2)) +
                   0.5 * innerₕ(u(2), v(1)) + innerₕ(u(1), v(2)) + innerₕ(u(2), v(2))
    fl = v -> innerₕ(_Const(1.0), v(1)) + innerₕ(_x1, v(2))
    V = Ce × Cd
    a, l = form(V, V, fa), form(V, fl)
    Aref, Fref = block_reference(V, V, fa, fl)
    k, pts = facet_nodes(Ωₕ, (:ymin, :ymax))
    rows = ndofs(Ce) .+ k
    vals = gD.(pts)
    bc = (:ymin => gD, :ymax => gD)

    Ar, Fr = rows_ref(Aref, Fref, rows, vals)
    A, F = assemble(a, l; dirichlet = bc, dirichlet_components = 2)
    @test close(Matrix(A), Ar) && close(F, Fr)
    @test close(Matrix(assemble(a; dirichlet = (:ymin, :ymax), dirichlet_components = (2,))),
        Ar)
    As, Fs = rows_ref(Aref, Fref, rows, vals; symmetrize = true)
    A2, F2 = assemble(a, l; dirichlet = bc, dirichlet_components = 2, symmetrize = true)
    @test close(Matrix(A2), As) && close(F2, Fs)
    assemble!(A2, F2, a, l; dirichlet = bc, dirichlet_components = 2, symmetrize = true)
    @test (@allocated assemble!(A2, F2, a, l; dirichlet = bc, dirichlet_components = 2,
        symmetrize = true)) == 0
    @test close(Matrix(A2), As) && close(F2, Fs)

    # by hand, as the keyword does it
    Ah = assemble(a)
    dirichlet_bc!(Ah, V, :ymin, :ymax; components = 2)
    @test close(Matrix(Ah), Ar)
    Ah, Fh = assemble(a, l; dirichlet = bc, dirichlet_components = 2)
    symmetrize!(Ah, Fh, V, :ymin, :ymax; components = 2)
    @test close(Matrix(Ah), As) && close(Fh, Fs)
    kb, ptsb = facet_nodes(Ωₕ, (:xmin, :xmax, :ymin, :ymax))
    Fb = copy(Fref)
    dirichlet_bc!(Fb, V, Bramble.dirichlet_constraints(Ωₕ, :boundary => gD), :boundary;
        components = 2)
    Fbr = copy(Fref)
    Fbr[ndofs(Ce) .+ kb] .= gD.(ptsb)
    @test Fb == Fbr

    # the keyword route and the `u => g` route give one solution
    u = pde_solve(A, F)
    Vr = Ce × Cr
    ur = pde_solve(assemble(form(Vr, Vr, fa), form(Vr, fl))...)
    @test close(parent(lift(V, u)), parent(lift(Vr, ur)); rtol = 1.0e-10)
    @test u[rows] ≈ vals

    # refused: a component whose unknowns are not its physical nodes
    @test refuses(() -> assemble(a; dirichlet = :ymin, dirichlet_components = 1),
        "given both ways")
    @test refuses(() -> assemble(a; dirichlet = :ymax), "fixed some nodes")
    @test refuses(() -> assemble(a, l; dirichlet = :ymax => gD, dirichlet_components = 1),
        "fixed some nodes")
    @test refuses(() -> dirichlet_bc!(assemble(a), V, :ymax), "fixed some nodes")
    @test refuses(() -> symmetrize!(Matrix(Aref), copy(Fref), V, :ymax), "fixed some nodes")
    @test refuses(
        () -> dirichlet_bc!(copy(Fref), V,
            Bramble.dirichlet_constraints(Ωₕ, :boundary => gD), :boundary), "given both ways")
    # the trial component at a selected position must have its physical nodes too
    ar = form(Cd × Ce, Cd × Cd, fa)
    @test refuses(() -> assemble(ar; dirichlet = :ymax, dirichlet_components = 2),
        "fixed some nodes")
    k1, _ = facet_nodes(Ωₕ, (:ymax,))
    A1 = Matrix(assemble(ar; dirichlet = :ymax, dirichlet_components = 1))
    @test all(r -> findall(!iszero, A1[r, :]) == [r] && A1[r, r] == 1, k1)
    # a component placed differently on the two sides has no identity column: the critic's
    # case (trial `Ce × Cd`, test `Cd × Cd`) and the roles swapped
    unpaired = "is not the row's own unknown"
    @test refuses(() -> assemble(form(Ce × Cd, Cd × Cd, fa); dirichlet = :ymax,
            dirichlet_components = 2), unpaired)
    @test refuses(() -> assemble(form(Cd × Cd, Ce × Cd, fa); dirichlet = :ymax,
            dirichlet_components = 2), unpaired)
    @test refuses(() -> assemble(form(Cd × Cd, Cd × Cd × Cd, fa); dirichlet = :ymax,
            dirichlet_components = 3),
        "the trial space has no component 3")
    # `components` counts the test side's: three trial components, two test ones
    @test refuses(
        () -> assemble(form(Cd × Cd × Cd, Cd × Cd, fa); dirichlet = :ymax,
            dirichlet_components = 3), "names leaf 3")
    @test refuses(() -> assemble(a; dirichlet = :ymin, dirichlet_components = 3),
        "names leaf 3")
    @test refuses(() -> assemble(l; dirichlet = :ymin => gD, dirichlet_components = 3),
        "names leaf 3")
    @test refuses(() -> assemble(a, l; dirichlet = (:ymin, :ymax), dirichlet_components = 2),
        "carries no values")
end

# --- pde_solve ------------------------------------------------------------------------- #

@testset "Composite ghost pde_solve" begin
    Ωₕ = O.make_mesh(:graded, 0; dim = 2, n0 = 9)
    V, u = coupled_solve(Ωₕ)
    Aref, Fref = block_reference(V, V, fa_c, fl_c)
    k, pts = facet_nodes(Ωₕ, (:ymin, :ymax))
    Ar, Fr = rows_ref(Aref, Fref, ndofs(V.spaces[1]) .+ k, u2.(pts))
    @test close(u, Ar \ Fr; rtol = 1.0e-10)
end

# --- Refusals ------------------------------------------------------------------------ #

# --- Traction spaces: pairs, the facet rule's limits, the refusals' advice ------------- #

const λt, μt = 2.0, 0.5
Kt(i, j) = [λt * (i == d) * (j == e) + μt * ((i == j) * (d == e) + (i == e) * (j == d))
            for d in 1:2, e in 1:2]
trac(u, i) = ∂ₙ(u(1); K = Kt(i, 1)) + ∂ₙ(u(2); K = Kt(i, 2))
clamped(F, u) = (F => u(1) => 0.0, F => u(2) => 0.0, F => ∂ₙₙ(u(1)) => 0.0,
    F => ∂ₙₙ(u(2)) => 0.0)
others(F, u) = mapreduce(G -> clamped(G, u), (x, y) -> (x..., y...),
    filter(!=(F), (:xmin, :xmax, :ymin, :ymax)))

# A traction on facet `F` and the other three facets clamped.
function traction_space(V, F)
    return constrained(V, u -> (F => trac(u, 1) => 1.0, F => trac(u, 2) => 0.0,
        others(F, u)...))
end

const fe = (u, v) -> inner₊(∇ₕ(u(1)), ∇ₕ(v(1))) + inner₊(∇ₕ(u(2)), ∇ₕ(v(2))) +
                     innerₕ(u(2), v(1))

@testset "Composite ghost traction pairs" begin
    Ωₕ = mesh2d()
    S = gridspace(Ωₕ; ghosts = :boundary)
    V = S × S
    Ta, Tb = traction_space(V, :xmin), traction_space(V, :ymin)
    @test ndofs(Ta) == ndofs(Tb)
    a = form(Ta, Ta, fe)
    la, lb = form(Ta, fl2), form(Tb, fl2)
    A, F = assemble(a, la)
    # one size, different kept nodes: the pair is refused, as the loads differ
    @test_throws DimensionMismatch assemble(a, lb)
    @test_throws DimensionMismatch assemble!(copy(A), copy(F), a, lb)
    @test norm(assemble(form(Tb, Tb, fe), lb)[2] - F) > 1.0e-3 * norm(F)
    # a space rebuilt alike pairs, with the same system
    @test assemble(a, form(traction_space(V, :xmin), fl2)) == (A, F)

    # the bare mixed difference takes the facet rule, so differs from the reading of the
    # closure's ghosts; with an average between the two differences there is no rule, and
    # the form reads past a corner of the padded grid
    PV = Bramble.padded_space(V)
    E = Matrix(first(Bramble.closure(Ta)))
    g = (u, v) -> innerₕ(Dcₓ(Dcᵧ(u(2))), v(1))
    Ag, Pg = Matrix(assemble(form(Ta, Ta, g))), Matrix(assemble(form(PV, PV, g)))
    @test norm(Ag - transpose(E) * Pg * E) > 1.0e-3 * norm(Ag)
    for gm in ((u, v) -> innerₕ(Dcₓ(Mₓ(Dcᵧ(u(2)))), v(1)),
        (u, v) -> innerₕ(Dcₓ(Dcᵧ(Mₓ(u(2)))), v(1)))
        @test refuses(() -> assemble(form(Ta, Ta, gm)),
            "a padded point that does not exist")
    end

    # a component fixed on the traction facet is told its ghost relation
    slide = u -> (:xmin => u(1) => 0.0, :xmin => trac(u, 2) => 0.0, others(:xmin, u)...)
    @test refuses(() -> constrained(V, slide), "takes `:xmin => ∂ₙₙ(u(i)) => h`")
    # the row replacements name the labels given
    @test refuses(() -> dirichlet_bc!(zeros(2, 2), Ta, :ymin), "dirichlet = (:ymin,)")
end

@testset "Composite ghost refusals" begin
    Ωₕ = mesh2d()
    S = gridspace(Ωₕ; ghosts = GH)
    Ca, Cb = space_a(S), space_b(S)
    V = Ca × Cb
    W = gridspace(Ωₕ)
    mixed = "takes ghost spaces only"
    # a ghost space beside a plain one, or nested in another composite
    for p in (() -> Ca × W, () -> W × Ca, () -> V × W, () -> W × V, () -> V × (W × W),
        () -> (W × W) × V, () -> Ca × (W × W), () -> (W × W) × Ca)
        @test refuses(p, mixed)
    end
    for Vm in (Bramble.CompositeGridSpace(Ca, W), Bramble.CompositeGridSpace(V, Ca))
        @test refuses(() -> form(Vm, Vm, (u, v) -> innerₕ(u(1), v(1))), mixed)
        @test refuses(() -> element(Vm)(1), mixed)
    end
    # a composite on one side only
    side = "on both sides"
    for (X, Y) in ((V, Ca), (Ca, V), (V, W), (W, V))
        @test refuses(() -> form(X, Y, (u, v) -> innerₕ(u(1), v(1))), side)
    end
    @test refuses(() -> form(V, Ca, (u, v) -> innerₕ(u(1), v(1))), "the test space, ")
    @test refuses(() -> form(W, V, (u, v) -> innerₕ(u(1), v(1))), "the trial space, ")
    # components ghosting different facets, or on meshes of different sizes
    sets = "ghost different facets"
    Cy = gridspace(Ωₕ; ghosts = (:ymax,))
    @test refuses(() -> form(S × Cy, S × Cy, (u, v) -> innerₕ(u(1), v(1))), sets)
    @test refuses(() -> form(S × Cy, v -> innerₕ(_Const(1.0), v(1))), sets)
    @test refuses(() -> form(S × S, Cy × Cy, (u, v) -> innerₕ(u(1), v(1))), sets)
    S9 = gridspace(O.make_mesh(:random, 0; dim = 2, n0 = 9); ghosts = GH)
    @test refuses(() -> form(S × S9, S × S9, (u, v) -> innerₕ(u(1), v(1))), sets)
    # one padded shape, different facets
    Sx = gridspace(Ωₕ; ghosts = (:xmin,)) × gridspace(Ωₕ; ghosts = (:xmax,))
    @test refuses(() -> form(Sx, Sx, (u, v) -> innerₕ(u(1), v(1))), sets)

    fa = (u, v) -> inner₊(∇ₕ(u(1)), ∇ₕ(v(1))) + innerₕ(u(2), v(2))
    a, l = form(V, V, fa), form(V, fl2)
    yet = "not supported yet"
    @test refuses(() -> form(V, V, (u, v) -> fa(u, v) + 2.0 * innerₕ(1, u(1)) *
                                                        innerₕ(1, v(1))), "a rank-one term")
    @test refuses(() -> form(V, V, (u, v) -> fa(u, v) + 2.0 * innerₕ(1, u(1)) *
                                                        innerₕ(1, v(1))),
        "Pose the rank-one term on each scalar ghost space")
    @test refuses(() -> semidiscretize(a, l), "Pose the time problem with semidiscretize")
    @test refuses(() -> matrix_free_operator(a), "A, F = assemble(a, l)")
    @test refuses(() -> matrix_free_operator(a), "matrix_free_operator on a composite")
    @test refuses(() -> kronecker_operator(a), "kronecker_operator on a composite")
    @test refuses(() -> chebyshev_preconditioner(a), "chebyshev_preconditioner on a composite")
    @test refuses(() -> amg_preconditioner(a), "amg_preconditioner on a composite")
    @test refuses(() -> fdm_solve(a, zeros(ndofs(V))), "fdm_solve on a composite")
    @test refuses(() -> fdm_factorize(a), "fdm_factorize on a composite")
    @test refuses(() -> fdm_preconditioner(a), "fdm_preconditioner on a composite")
    @test refuses(() -> semidiscretize(a, l), "semidiscretize on a composite")
    @test refuses(() -> ode_problem(a, l, zeros(ndofs(V)), (0.0, 1.0)),
        "ode_problem on a composite")
    build = Wl -> (C = gridspace(mesh(Wl); ghosts = (:xmin,));
        form(C × C, C × C, (u, v) -> innerₕ(u(1), v(1)) + innerₕ(u(2), v(2))))
    Ωg = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (17, 17), (true, true))
    @test refuses(() -> gmg_preconditioner(build, Ωg), "must return a BilinearForm")
    @test refuses(() -> Rₕ(V, _Const(1.0)), "Rₕ or avgₕ")
    @test refuses(() -> avgₕ(V, _Const(1.0)), "Rₕ or avgₕ")
    @test refuses(() -> Rₕ(V, (_Const(1.0), _Const(2.0))), "Rₕ or avgₕ")
    @test refuses(() -> Rₕ(Ca^2, x -> (1.0, 2.0)), "Rₕ or avgₕ")
    @test refuses(() -> matrix_free_operator(a), yet)
    @test refuses(() -> jacobi_preconditioner(a), "jacobi_preconditioner on a composite")
    v = element(V)
    v .= 1.0
    @test refuses(() -> l(v), "assemble(a, l)")
    @test refuses(() -> Bramble.evaluate!(zeros(ndofs(V)), l, v), "assemble(a, l)")
    @test refuses(() -> l(ones(ndofs(V))), "not a bare vector")
    @test refuses(() -> Bramble.reaction(a, l, v; marker = :boundary), "reaction reads")
    lp = form(W, w -> innerₕ(_Const(1.0), w))
    @test refuses(() -> semidiscretize(a, lp), "semidiscretize on a composite")
    @test refuses(() -> semidiscretize(Wl -> a, l), "semidiscretize on a composite")
    @test refuses(() -> ode_problem(a, lp, zeros(ndofs(V)), (0.0, 1.0)),
        "ode_problem on a composite")
end

end
