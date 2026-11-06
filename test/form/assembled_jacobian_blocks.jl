module FormAssembledJacobianBlocksTests

using Test
using Bramble
using ForwardDiff: ForwardDiff
using Random: Random
using Bramble: Mₕ, Mₓ!, Mᵧ!, jacobian_form, jacobian_prototype, jacobian!, CpuSerial,
               CpuThreaded

# `jacobian_form` on a composite space (#1144): a dependency names a component leaf
# (`U -> U(2)`), so the derivative of `c1 * U(1)` in the first equation lands in the block
# of the component `c1` reads. The oracle is a dense `ForwardDiff.jacobian` of the residual
# with every coefficient computed from a `Dual` state, on a seeded non-uniform 2D mesh (2D
# only, #650). Each block is compared against the oracle relative to its own magnitude, and
# the two coefficients use different functions, so a block written into its neighbour fails.

_κ1(s) = 1 + s^2
_κ2(s) = 2 + sin(s)

function _spaces(policy = CpuSerial())
    Random.seed!(1144)
    Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
    Wc = gridspace(mesh(Ω, (9, 12), (false, false); backend = backend(policy = policy)))
    return Wc, Wc^Val(2)
end

_state(V) = [0.3 + 0.7 * sin(0.37 * i) for i in 1:ndofs(V)]

# Each case: the form's body from `(c1, c2)`, the two dependencies, and the same two
# computed on the components `cs` of a state of element type `T` for the oracle.
_diffusion(P, Q) = inner₊(∇ₕ(P(1)), ∇ₕ(Q(1))) + inner₊(∇ₕ(P(2)), ∇ₕ(Q(2)))
_avgx(Wc, u, T) = (m = element(Wc, T); Mₓ!(m, u); m)
_avgy(Wc, u, T) = (m = element(Wc, T); Mᵧ!(m, u); m)

const _CASES = (
    ("crossed",
        (c1, c2) -> (P, Q) -> _diffusion(P, Q) + innerₕ(c1 * P(1), Q(1)) +
                              innerₕ(c2 * P(2), Q(2)),
        U -> U(2), U -> U(1), (Wc, cs, T) -> cs[2], (Wc, cs, T) -> cs[1], (true, true)),
    ("diagonal",
        (c1, c2) -> (P, Q) -> _diffusion(P, Q) + innerₕ(c1 * P(1), Q(1)) +
                              innerₕ(c2 * P(2), Q(2)),
        U -> U(1), U -> U(2), (Wc, cs, T) -> cs[1], (Wc, cs, T) -> cs[2], (false, false)),
    # The dependency reads neighbouring points of the other component.
    ("averages of the other component",
        (c1, c2) -> (P, Q) -> _diffusion(P, Q) + innerₕ(c1 * P(1), Q(1)) +
                              innerₕ(c2 * P(2), Q(2)),
        U -> Mₕ(U(2))[1], U -> Mₕ(U(1))[2], (Wc, cs, T) -> _avgx(Wc, cs[2], T),
        (Wc, cs, T) -> _avgy(Wc, cs[1], T), (true, true)),
    # `T(u)` reads component 1's slice at neighbouring points.
    ("coefficient on a difference",
        (c1, c2) -> (P, Q) -> inner₊(c1 * ∇ₕ(P(1))[1], ∇ₕ(Q(1))[1]) +
                              inner₊(∇ₕ(P(1))[2], ∇ₕ(Q(1))[2]) +
                              inner₊(c2 * ∇ₕ(P(2))[2], ∇ₕ(Q(2))[2]) +
                              inner₊(∇ₕ(P(2))[1], ∇ₕ(Q(2))[1]),
        U -> U(2), U -> Mₕ(U(1))[1], (Wc, cs, T) -> cs[2],
        (Wc, cs, T) -> _avgx(Wc, cs[1], T), (true, true)),
    # Both coefficients in the first equation, one reading each component: the derivative is
    # a sum mixing components inside one inner product.
    ("mixed sum in one product",
        (c1, c2) -> (P, Q) -> _diffusion(P, Q) + innerₕ(c1 * P(1) + c2 * P(1), Q(1)) +
                              innerₕ(P(2), Q(2)),
        U -> U(2), U -> U(1), (Wc, cs, T) -> cs[2], (Wc, cs, T) -> cs[1], (true, false))
)

# Per block `(error, magnitude)` of the assembled Jacobian against the oracle, in the order
# (1,1), (1,2), (2,1), (2,2) (rows, then columns), and whether the coefficients were written.
function _block_errors(Wc, Vc, case; dirichlet = :boundary)
    _, body, d1, d2, n1, n2, _ = case
    build(c1, c2) = form(Vc, Vc, body(c1, c2))
    function R(x)
        T = eltype(x)
        xₕ = element(Vc, T)
        xₕ .= x
        cs = components(xₕ)
        c1, c2 = element(Wc, T), element(Wc, T)
        c1 .= _κ1.(n1(Wc, cs, T))
        c2 .= _κ2.(n2(Wc, cs, T))
        return assemble(build(c1, c2); dirichlet = dirichlet) * x
    end
    w = _state(Vc)
    Jfd = ForwardDiff.jacobian(R, w)
    c1, c2 = element(Wc), element(Wc)
    jf = jacobian_form(build(c1, c2), c1 => (d1, _κ1), c2 => (d2, _κ2); dirichlet = dirichlet)
    J = jacobian_prototype(jf)
    jacobian!(J, jf, w)
    n = ndofs(Wc)
    r = (1:n, (n + 1):(2n))
    errs = map(((1, 1), (1, 2), (2, 1), (2, 2))) do (i, k)
        B = Jfd[r[i], r[k]]
        return (maximum(abs, J[r[i], r[k]] - B), maximum(abs, B))
    end
    wₕ = element(Vc)
    wₕ .= w
    cs = components(wₕ)
    written = c1 ≈ _κ1.(n1(Wc, cs, Float64)) && c2 ≈ _κ2.(n2(Wc, cs, Float64))
    return errs, written
end

@testset "jacobian_form: composite blocks" begin
    for policy in (CpuSerial(), CpuThreaded()), case in _CASES

        @testset "$(case[1]), $(nameof(typeof(policy)))" begin
            Wc, Vc = _spaces(policy)
            errs, written = _block_errors(Wc, Vc, case)
            @test written
            for (e, m) in errs
                @test e <= 1e-12 * m
            end
            # The diagonal blocks always carry the diffusion; an off-diagonal block is
            # nonzero exactly when a coefficient reads the other component.
            @test errs[1][2] > 0 && errs[4][2] > 0
            @test (errs[2][2] > 0, errs[3][2] > 0) == case[7]
        end
    end
end

# The message of the `ArgumentError` `f()` throws, or `""` when it throws none.
function _message(f)
    try
        f()
    catch e
        e isa ArgumentError && return sprint(showerror, e)
    end
    return ""
end

@testset "jacobian_form: composite refusals" begin
    Wc, Vc = _spaces()
    c = element(Wc)
    indexed = form(Vc, Vc, (P, Q) -> innerₕ(c * P(1), Q(1)) + innerₕ(P(2), Q(2)))
    @test_throws ArgumentError jacobian_form(indexed, c => (U -> U(3), _κ1))
    @test occursin("2 components", _message(() -> jacobian_form(indexed, c => (U -> U(3), _κ1))))
    @test_throws ArgumentError jacobian_form(indexed, c => (U -> U, _κ1))
    @test occursin("names no component of the composite space; name one",
        _message(() -> jacobian_form(indexed, c => (U -> U, _κ1))))
    # One β cannot serve every diagonal block a coefficient on `U` reaches.
    whole = form(Vc, Vc, (P, Q) -> innerₕ(c * P, Q))
    @test_throws ArgumentError jacobian_form(whole, c => (U -> U(1), _κ1))
    @test occursin("scales a trial function naming no component",
        _message(() -> jacobian_form(whole, c => (U -> U(1), _κ1))))
    # A composite trial space with a scalar test space is not routed.
    mixed = form(Vc, Wc, (P, q) -> innerₕ(c * P(1), q) + innerₕ(P(2), q))
    @test_throws ArgumentError jacobian_form(mixed, c => (U -> U(2), _κ1))
    @test occursin("both be scalar grid spaces or both composite",
        _message(() -> jacobian_form(mixed, c => (U -> U(2), _κ1))))
end

end # module
