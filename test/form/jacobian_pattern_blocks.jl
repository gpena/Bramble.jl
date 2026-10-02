module FormJacobianPatternBlocksTests

using Test
using Bramble
using SparseArrays: sparse, findnz, dropzeros
using Bramble: D₋ₓ, Mₕ, jacobian_pattern

# `jacobian_pattern` on a pair with a composite space on one side and a scalar space on the
# other. `assemble` walks such a pair block by block, the scalar side
# as a one-leaf composite; before this the pattern took the scalar path, whose `_walked_leaf`
# picks one whole space and cannot name the component a term reads on the composite side:
# a `πₕ` term threw a MethodError, and a native-only pair returned a pattern short of
# `assemble`'s (31 of 42 entries in 1D) without complaint.
#
# Kept out of jacobian_pattern.jl, which runs only in the slow and AD groups: this needs no
# weak dependency and costs little, so it runs on every push.

_m1(n) = mesh(domain(interval(0.0, 1.0)), n, true)
_m2(n) = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (n, n + 1), (true, false))

_pattern(A) = Set(zip(findnz(sparse(A))[1:2]...))
_live_pattern(A) = _pattern(dropzeros(sparse(A)))

# `Ws` is a scalar space on a second, different mesh, reached only through `πₕ`; `Wt` is the
# space every native term reads.
const _CASES = (
    ("composite trial, πₕ only", (Ws, Wt) -> (Ws × Ws, Wt),
        (U, v) -> innerₕ(πₕ(U(1)), v) + innerₕ(πₕ(U(2)), v)),
    ("composite trial, native only", (Ws, Wt) -> (Wt × Wt, Wt),
        (U, v) -> innerₕ(U(1), v) + inner₊(D₋ₓ(U(2)), D₋ₓ(v))),
    ("composite trial, native and πₕ", (Ws, Wt) -> (Wt × Ws, Wt),
        (U, v) -> inner₊(D₋ₓ(U(1)), D₋ₓ(v)) + innerₕ(πₕ(U(2)), v)),
    ("composite test, native only", (Ws, Wt) -> (Wt, Wt × Wt),
        (u, V) -> innerₕ(u, V(1)) + inner₊(D₋ₓ(u), D₋ₓ(V(2)))),
    ("composite test, native and πₕ", (Ws, Wt) -> (Wt, Wt × Ws),
        (u, V) -> inner₊(D₋ₓ(u), D₋ₓ(V(1))) + innerₕ(u, πₕ(V(2))))
)

@testset "jacobian_pattern: mixed block pairs" begin
    for (mk, ns, nt) in ((_m1, 7, 11), (_m2, 4, 6)), (name, spaces, f) in _CASES

        @testset "$name, $(dim(mk(3)))D" begin
            Wu, Wv = spaces(gridspace(mk(ns)), gridspace(mk(nt)))
            a = form(Wu, Wv, f)
            A = assemble(a)
            P = jacobian_pattern(a)
            @test size(P) == size(A)
            # A cancellation could zero an entry `assemble` still stores: the pattern must
            # match either the stored entries or the live ones.
            @test _pattern(P) == _pattern(A) || _pattern(P) == _live_pattern(A)
        end
    end
end

# A coefficient computed on the walked mesh from a trial function on another mesh
# The dependency names that computation through `πₕ`, and the
# pattern widens every row the term reaches into the columns `πₕ` reads there. The oracle
# is a dense finite-difference Jacobian of the Newton residual `A(c(u)) u`, `c` rebuilt at
# run time from the same composition the dependency names. Non-uniform meshes throughout.
_n1(n) = mesh(domain(interval(0.0, 1.0)), n, false)
_n2(n) = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (n, n + 1), (false, true))
_α(x) = 1 + x^2
_filled(W, x) = (u = element(W); u .= x; u)

function _fd_pattern(R, n)
    x = 0.5 .+ sin.(1:n) .^ 2
    r0 = R(x)
    S = Set{Tuple{Int, Int}}()
    for j in 1:n
        y = copy(x)
        y[j] += 1e-6
        d = (R(y) - r0) / 1e-6
        for i in eachindex(d)
            abs(d[i]) > 1e-8 && push!(S, (i, j))
        end
    end
    return S
end

function _check_against_oracle(a, dep, R, ntrial)
    E = _fd_pattern(R, ntrial)
    P = _pattern(jacobian_pattern(a, dep))
    @test issubset(E, P)
    @test all(((i, j),) -> 1 <= j <= ntrial, P)
    @test length(P) <= 2length(E)
end

function _refused_with_bramble_error(a, dep)
    err = try
        jacobian_pattern(a, dep)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test err isa ArgumentError && occursin("πₕ", sprint(showerror, err))
end

@testset "jacobian_pattern: dependencies via πₕ" begin
    # Each case: the form's builder over a coefficient, the dependency, and how the
    # coefficient is computed from the trial values at run time.
    @testset "1D scalar, $name" for (name, dep, build) in (
        ("πₕ(U)", U -> πₕ(U), ia -> ia), ("Mₕ(πₕ(U))", U -> Mₕ(πₕ(U)), ia -> Mₕ(ia)))
        Wa, Wb = gridspace(_n1(9)), gridspace(_n1(6))
        mk(c) = form(Wb, Wa, (u, v) -> inner₊(c * ∇ₕ(πₕ(u)), ∇ₕ(v)))
        R(x) = (c = element(Wa); c .= _α.(build(πₕ(Wa, _filled(Wb, x)))); assemble(mk(c)) * x)
        _check_against_oracle(mk(element(Wa, 1.0)), dep, R, ndofs(Wb))
    end

    @testset "2D scalar, πₕ(U)" begin
        Wa, Wb = gridspace(_n2(5)), gridspace(_n2(3))
        mk(c) = form(Wb, Wa, (u, v) -> innerₕ(c * πₕ(u), v))
        R(x) = (c = element(Wa); c .= _α.(πₕ(Wa, _filled(Wb, x))); assemble(mk(c)) * x)
        _check_against_oracle(mk(element(Wa, 1.0)), U -> πₕ(U), R, ndofs(Wb))
    end

    # Block (1) reads πₕ(U(1)); the coefficient is built from component 2.
    @testset "$(dim(mk_mesh(3)))D composite, $name" for (mk_mesh, na, nb, name, dep, build) in (
        (_n1, 9, 6, "πₕ(U(2))", U -> πₕ(U(2)), ia -> ia),
        (_n1, 9, 6, "Mₕ(πₕ(U(2)))", U -> Mₕ(πₕ(U(2))), ia -> Mₕ(ia)),
        (_n2, 5, 3, "πₕ(U(2))", U -> πₕ(U(2)), ia -> ia))
        Wa, Wb = gridspace(mk_mesh(na)), gridspace(mk_mesh(nb))
        n = ndofs(Wb)
        mk(c) = form(Wb × Wb, Wa, (U, v) -> innerₕ(c * πₕ(U(1)), v))
        R(x) = (c = element(Wa);
            c .= _α.(build(πₕ(Wa, _filled(Wb, x[(n + 1):end]))));
            assemble(mk(c)) * x)
        _check_against_oracle(mk(element(Wa, 1.0)), dep, R, 2n)
    end

    # Without πₕ a cross-mesh dependency has no anchor on the walked mesh.
    @testset "refused without πₕ, $(dim(mk_mesh(3)))D" for (mk_mesh, na, nb) in (
        (_n1, 9, 6), (_n2, 5, 3))
        Wa, Wb = gridspace(mk_mesh(na)), gridspace(mk_mesh(nb))
        _refused_with_bramble_error(form(Wb, Wa, (u, v) -> innerₕ(πₕ(u), v)), U -> Mₕ(U))
        _refused_with_bramble_error(
            form(Wb × Wb, Wa, (U, v) -> innerₕ(πₕ(U(1)), v)), U -> U(2))
    end

    # Equal point counts are not the same mesh: a leaf with as many points over another
    # extent, or over the same extent at other points, is still refused without πₕ.
    @testset "refused without πₕ, same npoints, $name" for (name, Wx) in (
        ("other extent", gridspace(mesh(domain(interval(0.3, 0.7)), 9, false))),
        ("other points", gridspace(mesh(domain(interval(0.0, 1.0)), 9, true))))
        Wa = gridspace(_n1(9))
        _refused_with_bramble_error(
            form(Wx, Wa, (u, v) -> innerₕ(πₕ(u; outside = :clamp), v)), U -> Mₕ(U))
    end

    # A walked point on a source node: the corners `locate_cell` names with weight zero add
    # no column, so the pattern stays within twice the oracle.
    @testset "same mesh through πₕ, $(dim(mk_mesh(3)))D, $name" for mk_mesh in (_n1, _n2),
        (name, dep, build) in (("πₕ(U)", U -> πₕ(U), (W, u) -> πₕ(W, u)),
            ("πₕ(U) + U", U -> πₕ(U) + U, (W, u) -> πₕ(W, u) .+ u))

        W = gridspace(mk_mesh(4))
        mk(c) = form(W, W, (u, v) -> innerₕ(c * u, v))
        R(x) = (c = element(W); c .= _α.(build(W, _filled(W, x))); assemble(mk(c)) * x)
        _check_against_oracle(mk(element(W, 1.0)), dep, R, ndofs(W))
    end

    # A data coefficient inside the dependency that is zero when the pattern is built and
    # set later: its zero weights are values, not geometry, so no column is dropped.
    @testset "data coefficient zero at build, $(dim(mk_mesh(3)))D" for (mk_mesh, na, nb) in (
        (_n1, 9, 6), (_n2, 5, 3))
        Wa, Wb = gridspace(mk_mesh(na)), gridspace(mk_mesh(nb))
        _sum(m) = m isa Tuple ? sum(m) : m
        mk(c) = form(Wb, Wa, (u, v) -> innerₕ(c * πₕ(u), v))
        g = element(Wa, 0.0)
        P = _pattern(jacobian_pattern(mk(element(Wa, 1.0)), U -> _sum(Mₕ(g * πₕ(U)))))
        g .= 1.0
        mm(z) = (m = Mₕ(z); m isa Tuple ? m[1] .+ m[2] : m)
        R(x) = (c = element(Wa);
            c .= _α.(mm(g .* πₕ(Wa, _filled(Wb, x))));
            assemble(mk(c)) * x)
        @test issubset(_fd_pattern(R, ndofs(Wb)), P)
    end

    # The scalar path walks behind a function barrier: allocations do not grow with the
    # number of walked points.
    @testset "scalar path allocations" begin
        function allocs(n)
            Wa, Wb = gridspace(_n2(n)), gridspace(_n2(n ÷ 2 + 1))
            a = form(Wb, Wa, (u, v) -> inner₊(element(Wa, 1.0) * ∇ₕ(πₕ(u)), ∇ₕ(v)))
            dep = U -> Mₕ(πₕ(U))
            jacobian_pattern(a, dep)
            return @allocations jacobian_pattern(a, dep)
        end
        @test abs(allocs(16) - allocs(8)) <= 16
    end

    # Same-mesh widening is unchanged: still a strict superset of assemble's pattern.
    @testset "same-mesh control, $name" for (name, spaces, f) in (
        ("scalar", W -> (W, W), (u, v) -> innerₕ(u, v)),
        ("mixed", W -> (W × W, W), (U, v) -> innerₕ(U(1), v) + innerₕ(U(2), v)))
        a = form(spaces(gridspace(_n1(9)))..., f)
        P = _pattern(jacobian_pattern(a, U -> Mₕ(U)))
        @test issubset(_pattern(assemble(a)), P)
        @test length(P) > length(_pattern(assemble(a)))
    end
end

# A scalar form whose test side reads πₕ: its stencil names absolute test rows on the other
# mesh, not offsets from the walked point, and the pattern must still contain the one
# `assemble` stores, with a dependency or without.
@testset "jacobian_pattern: πₕ on the test side" begin
    @testset "$(dim(mk_mesh(3)))D, $name" for (mk_mesh, na, nb) in ((_n1, 9, 6), (_n2, 5, 3)),
        (name, deps) in (("no dependency", ()), ("Mₕ(U)", (U -> Mₕ(U),)))

        Wa, Wb = gridspace(mk_mesh(na)), gridspace(mk_mesh(nb))
        c = element(Wa, 1.0)
        a = form(Wa, Wb, (u, v) -> innerₕ(c * u, πₕ(v)))
        P = jacobian_pattern(a, deps...)
        @test size(P) == size(assemble(a))
        @test issubset(_pattern(assemble(a)), _pattern(P))
    end
end

# A dependency naming a component the trial space does not have is refused by name.
@testset "jacobian_pattern: missing component" begin
    @testset "$name" for (name, dep) in (("πₕ(U(3))", U -> πₕ(U(3))), ("U(3)", U -> U(3)))
        Wa, Wb = gridspace(_n1(9)), gridspace(_n1(6))
        a = form(Wb × Wb, Wa, (U, v) -> innerₕ(element(Wa, 1.0) * πₕ(U(1)), v))
        err = try
            jacobian_pattern(a, dep)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        msg = err isa ArgumentError ? sprint(showerror, err) : ""
        @test occursin("component 3", msg) && occursin("2 components", msg)
    end
end

end # module
