module GhostCompositeTests

# Composite spaces with ghost components (#781): `C₁ × C₂` and `C^Val(n)` of ghost spaces
# assemble a vector problem with per-component relations. The reference of every check is
# built from the padded composite `padded(C₁) × padded(C₂)` and the block-diagonal closure
# `blockdiag(E₁, E₂)`, `vcat(d₁, d₂)` read off the components, as dense matrices, sharing no
# reduction code with the library: `A = Eᵀ A_pad E`, `F = Eᵀ (F_pad − A_pad d)`.
#
# Covered: the product and its layout, assembly and refills, `a(u, v)` and `l(v)`, `lift`,
# the `dirichlet` keyword per component against the `u => g` route, `pde_solve` against its
# block reference, second-order convergence of a coupled two-component problem on random and
# graded meshes, and the refusals: mixed or nested composites, components ghosting different
# facets, a scalar space on the other side, rank-one terms, `semidiscretize`, and the
# matrix-free, Kronecker and multigrid operators.

using Test
using Bramble
import Bramble
using Bramble: D₋ₓ, D₊ᵧ, Dcₓ, Dcᵧ, lift, lift!
using LinearAlgebra: dot, norm
using SparseArrays: blockdiag, sparse
using Random: Xoshiro

using ..TestUtils: WITH_SLOW_TESTS, WITH_3D_TESTS

isdefined(Main, :GhostOracleND) ||
    Base.include(Main, joinpath(pkgdir(Bramble), "test", "ghost", "GhostOracleND.jl"))
const O = Main.GhostOracleND

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
        u -> (:xmin => ∂ₙ(u) => 1.0, :xmax => 2.0 * u + ∂ₙ(u) => x -> x[2],
            :ymin => ∂ₙ(u) => 0.5, :ymax => u => x -> x[1]))
end

# Robin, Neumann, Robin with data, and the unghosted facet left free.
function space_b(S)
    constrained(S,
        u -> (:xmin => 2.0 * u + ∂ₙ(u) => 0.5, :xmax => ∂ₙ(u) => 0.0,
            :ymin => u + ∂ₙ(u) => x -> x[1]))
end

# The mixed difference reads the ghosts beside the corners; at a corner node of a padded
# composite it takes the one-sided corner rule, as on a scalar space (tested below).
const fa2 = (u, v) -> inner₊(∇ₕ(u(1)), ∇ₕ(v(1))) + innerₕ(-1.0 * Δₕ(u(2)), v(2)) +
                      0.5 * innerₕ(u(2), v(2)) + innerₕ(u(2), v(1)) +
                      innerₕ(D₋ₓ(D₊ᵧ(u(1))), v(2))
const fl2 = v -> innerₕ(x -> 1 + x[1], v(1)) + innerₕ(x -> x[2]^2, v(2))

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
        fl3 = v -> innerₕ(x -> x[1], v(1)) + innerₕ(x -> 1.0, v(3))
        check_assembly(Ca^Val(3), Ca^Val(3), fa3, fl3)

        # 1D: a constrained component beside a selection space that keeps its ghosts
        Ω1 = O.O1.make_mesh(:random, 0; n0 = 9)
        C1 = constrained(gridspace(Ω1; ghosts = :boundary),
            u -> (:xmin => ∂ₙ(u) => 1.0, :xmax => u + ∂ₙ(u) => 2.0))
        S1 = gridspace(Ω1; ghosts = :boundary)
        fa1 = (u, v) -> innerₕ(-1.0 * Δₕ(u(1)), v(1)) + innerₕ(u(2), v(2)) + innerₕ(u(1), v(2))
        fl1 = v -> innerₕ(x -> x, v(1)) + innerₕ(x -> 1.0, v(2))
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
    # with data, the load alone refuses as on a scalar space
    @test refuses(() -> assemble(form(V, fl2)), "assemble(a, l)")

    if WITH_3D_TESTS
        Ω3 = O.make_mesh(:random, 0; dim = 3, n0 = 5)
        C3 = constrained(gridspace(Ω3; ghosts = (:xmin, :zmax)),
            u -> (:xmin => ∂ₙ(u) => 1.0, :zmax => u + ∂ₙ(u) => x -> x[2],
                :ymin => u => 0.0))
        fa33 = (u, v) -> inner₊(∇ₕ(u(1)), ∇ₕ(v(1))) + innerₕ(-1.0 * Δₕ(u(2)), v(2)) +
                         innerₕ(u(2), v(1))
        fl33 = v -> innerₕ(x -> x[3], v(1)) + innerₕ(x -> 1.0, v(2))
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
    fl = v -> innerₕ(x -> 1.0, v(1)) + innerₕ(x -> x[1], v(2))
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

# --- pde_solve and second-order convergence -------------------------------------------- #

# u₁ = e^{x + y/2}, u₂ = cos 2x sin(y + 1/2); −Δu₁ + u₁ + κu₂ = f₁, −Δu₂ + κu₁ = f₂.
const κ = 0.5
u1(p) = exp(p[1] + p[2] / 2)
u2(p) = cos(2p[1]) * sin(p[2] + 0.5)
dx1(p) = u1(p)
dx2(p) = -2sin(2p[1]) * sin(p[2] + 0.5)
f1(p) = -1.25u1(p) + u1(p) + κ * u2(p)
f2(p) = 5u2(p) + κ * u1(p)

const fa_c = (u, v) -> innerₕ(-1.0 * Δₕ(u(1)), v(1)) + innerₕ(u(1), v(1)) +
                       κ * innerₕ(u(2), v(1)) + innerₕ(-1.0 * Δₕ(u(2)), v(2)) +
                       κ * innerₕ(u(1), v(2))
const fl_c = v -> innerₕ(f1, v(1)) + innerₕ(f2, v(2))

# Component 1: Neumann on xmin, Robin on xmax, `u => g` on ymin and ymax; component 2:
# Robin on xmin, Neumann on xmax, Dirichlet rows on ymin and ymax (the keyword).
function coupled_spaces(Ωₕ)
    C1 = constrained(gridspace(Ωₕ; ghosts = (:xmin, :xmax)),
        u -> (:xmin => ∂ₙ(u) => p -> -dx1(p), :xmax => u + ∂ₙ(u) => p -> u1(p) + dx1(p),
            :ymin => u => u1, :ymax => u => u1))
    C2 = constrained(gridspace(Ωₕ; ghosts = (:xmin, :xmax)),
        u -> (:xmin => 2.0 * u + ∂ₙ(u) => p -> 2u2(p) - dx2(p),
            :xmax => ∂ₙ(u) => dx2))
    return C1 × C2
end

const BC2 = (:ymin => u2, :ymax => u2)

function coupled_solve(Ωₕ)
    V = coupled_spaces(Ωₕ)
    A, F = assemble(form(V, V, fa_c), form(V, fl_c); dirichlet = BC2,
        dirichlet_components = 2)
    return V, pde_solve(A, F)
end

# The discrete H¹ error `(‖e‖_{H}² + Σ_d ‖D₋_d e‖_+²)^{1/2}` at the physical nodes.
function h1_error(Ωₕ, e)
    xs = O.axis_points(Ωₕ)
    ĥ(x) = [(i > 1 ? x[i] - x[i - 1] : 0.0) / 2 + (i < length(x) ? x[i + 1] - x[i] : 0.0) / 2
            for i in eachindex(x)]
    E = reshape(e, map(length, xs))
    w = ĥ(xs[1]) .* ĥ(xs[2])'
    total = sum(w .* abs2.(E))
    total += sum((1 ./ diff(xs[1])) .* ĥ(xs[2])' .* abs2.(diff(E; dims = 1)))
    total += sum(ĥ(xs[1]) .* (1 ./ diff(xs[2]))' .* abs2.(diff(E; dims = 2)))
    return sqrt(total)
end

# The physical values of each component of a lifted solution.
function physical_values(V, U)
    out, start = Vector{Float64}[], 0
    for C in V.spaces
        G = mesh(C)
        n = prod(Bramble.npoints(G, Tuple))
        plin = LinearIndices(Bramble.npoints(G, Tuple))
        push!(out, parent(U)[start .+ vec(plin[Bramble.physical_indices(G)])])
        start += n
    end
    return out
end

function lsq_order(hs, errs)
    X, Y = log.(hs), log.(errs)
    return sum((X .- sum(X) / length(X)) .* (Y .- sum(Y) / length(Y))) /
           sum(abs2, X .- sum(X) / length(X))
end

@testset "Composite ghost pde_solve" begin
    Ωₕ = O.make_mesh(:graded, 0; dim = 2, n0 = 9)
    V, u = coupled_solve(Ωₕ)
    Aref, Fref = block_reference(V, V, fa_c, fl_c)
    k, pts = facet_nodes(Ωₕ, (:ymin, :ymax))
    Ar, Fr = rows_ref(Aref, Fref, ndofs(V.spaces[1]) .+ k, u2.(pts))
    @test close(u, Ar \ Fr; rtol = 1.0e-10)
end

@testset "Composite ghost order, $kind meshes" for kind in (:random, :graded)
    hs, errs = Float64[], NTuple{2, Float64}[]
    for ℓ in 0:2
        Ωₕ = O.make_mesh(kind, ℓ; dim = 2, n0 = 9)
        V, u = coupled_solve(Ωₕ)
        xs = O.axis_points(Ωₕ)
        nodes = vec([(x, y) for x in xs[1], y in xs[2]])
        v1, v2 = physical_values(V, lift(V, u))
        push!(hs, maximum(x -> maximum(diff(x)), xs))
        push!(errs, (h1_error(Ωₕ, v1 - u1.(nodes)), h1_error(Ωₕ, v2 - u2.(nodes))))
    end
    @test maximum(hs) / minimum(hs) > 3
    for c in 1:2
        @test lsq_order(hs, getindex.(errs, c)) > 1.8
    end
end

# --- Refusals ------------------------------------------------------------------------ #

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
    # components ghosting different facets, or on meshes of different sizes
    sets = "ghost different facets"
    Cy = gridspace(Ωₕ; ghosts = (:ymax,))
    @test refuses(() -> form(S × Cy, S × Cy, (u, v) -> innerₕ(u(1), v(1))), sets)
    @test refuses(() -> form(S × Cy, v -> innerₕ(x -> 1.0, v(1))), sets)
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
    @test refuses(() -> Rₕ(V, x -> 1.0), "Rₕ or avgₕ")
    @test refuses(() -> avgₕ(V, x -> 1.0), "Rₕ or avgₕ")
    @test refuses(() -> Rₕ(V, (x -> 1.0, x -> 2.0)), "Rₕ or avgₕ")
    @test refuses(() -> Rₕ(Ca^2, x -> (1.0, 2.0)), "Rₕ or avgₕ")
    @test refuses(() -> matrix_free_operator(a), yet)
    @test refuses(() -> jacobi_preconditioner(a), "jacobi_preconditioner on a composite")
    v = element(V)
    v .= 1.0
    @test refuses(() -> l(v), "assemble(a, l)")
    @test refuses(() -> Bramble.evaluate!(zeros(ndofs(V)), l, v), "assemble(a, l)")
    @test refuses(() -> l(ones(ndofs(V))), "not a bare vector")
    @test refuses(() -> Bramble.reaction(a, l, v; marker = :boundary), "reaction reads")
    lp = form(W, w -> innerₕ(x -> 1.0, w))
    @test refuses(() -> semidiscretize(a, lp), "semidiscretize on a composite")
    @test refuses(() -> semidiscretize(Wl -> a, l), "semidiscretize on a composite")
    @test refuses(() -> ode_problem(a, lp, zeros(ndofs(V)), (0.0, 1.0)),
        "ode_problem on a composite")
end

end
