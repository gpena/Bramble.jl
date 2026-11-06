module FormAssembledJacobianTests

using Test
using Bramble
using ForwardDiff: ForwardDiff
using Polyester: Polyester
using Random: Random
using SparseArrays: nnz
using Bramble: D₊ₓ, D₊ᵧ, D₋ₓ, Mₓ!, Mᵧ!, jumpₓ, restrict_to, LazyOp, TrialFunction,
               OperatorAdd, OperatorScale, GridFunctionScale, BilinearProduct, ShiftNode, JumpNode,
               RegionRestriction, CenteredAverage, ForwardDifference, BackwardAverage,
               jacobian_form, jacobian_prototype, jacobian!, MissingJacobianRule,
               allocate_system_matrix, CpuSerial, CpuThreaded, CpuPolyester
using ..TestUtils: alloc_ok, alloc_test

# `jacobian_form` (#1141): the exact Jacobian of `R(u) = A(u) u` as one form `a + ∂a`. The
# oracle is a dense `ForwardDiff.jacobian` of the residual with every coefficient computed
# from a `Dual` state, on a seeded non-uniform 2D mesh (2D only, #650). Each case also checks
# that the Picard matrix `A(u)` is off the oracle by more than 1e-3, so a Jacobian that drops
# a coefficient's derivative cannot pass.

_r(s) = 1 + s^2 / 2
_α(s) = 3 + 1 / (1 + s^2)
_relerr(A, B) = maximum(abs, A - B) / maximum(abs, B)

function _space(policy = CpuSerial(); n = (9, 12))
    Random.seed!(1141)
    Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
    return gridspace(mesh(Ω, n, (false, false); backend = backend(policy = policy)))
end

_state(W) = [0.3 + 0.7 * sin(0.37 * i) for i in 1:ndofs(W)]

# Whether a node of type `T` appears anywhere in `x`, so each case is known to reach the rule
# it names (the simplifier may restructure a written form).
function _has_node(x, ::Type{T}) where {T}
    x isa T ||
        (x isa Union{LazyOp, Tuple} &&
         any(k -> _has_node(getfield(x, k), T), 1:nfields(x)))
end

# `build(cs...)` builds the form from coefficients `cs[k] == fs[k].(uₕ)` (dependency
# `U -> U`). Returns the errors of the assembled Jacobian and of the Picard matrix against
# the oracle, and the form.
function _errors(W, build, fs; dirichlet = nothing)
    u = _state(W)
    function R(x)
        T = eltype(x)
        cs = map(f -> (c = element(W, T); c .= f.(x); c), fs)
        return assemble(build(cs...); dirichlet = dirichlet) * x
    end
    Jfd = ForwardDiff.jacobian(R, u)
    cs = map(_ -> element(W, 0.0), fs)
    a = build(cs...)
    jf = jacobian_form(a, map((c, f) -> c => (U -> U, f), cs, fs)...; dirichlet = dirichlet)
    J = jacobian_prototype(jf)
    jacobian!(J, jf, u)
    written = all(k -> cs[k] ≈ fs[k].(u), eachindex(fs))
    picard = assemble(a; dirichlet = dirichlet)
    return (jac = _relerr(J, Jfd), picard = _relerr(picard, Jfd), written = written, ast = a.ast)
end

# The `:flux` diffusion form with `αx` reading the y-average and `αy` the x-average, so
# `a + ∂a` has a wider pattern than `a`. The oracle computes the averages with `Mₓ!`/`Mᵧ!`.
_flux(W, ax, ay) = form(W, W, (U, V) -> inner₊((ax * ∇ₕ(U)[1], ay * ∇ₕ(U)[2]), ∇ₕ(V)))

function _flux_errors(W; dirichlet = :boundary)
    u = _state(W)
    function R(x)
        T = eltype(x)
        uₕ = element(W, T)
        uₕ .= x
        Mx, My, ax, ay = element(W, T), element(W, T), element(W, T), element(W, T)
        Mₓ!(Mx, uₕ)
        Mᵧ!(My, uₕ)
        ax .= _α.(My)
        ay .= _α.(Mx)
        return assemble(_flux(W, ax, ay); dirichlet = dirichlet) * x
    end
    Jfd = ForwardDiff.jacobian(R, u)
    ax, ay = element(W, 0.0), element(W, 0.0)
    a = _flux(W, ax, ay)
    jf = jacobian_form(a, ax => (U -> Mₕ(U)[2], _α), ay => (U -> Mₕ(U)[1], _α);
        dirichlet = dirichlet)
    J = jacobian_prototype(jf)
    jacobian!(J, jf, u)
    picard = assemble(a; dirichlet = dirichlet)
    return _relerr(J, Jfd), _relerr(picard, Jfd), nnz(J), nnz(allocate_system_matrix(a))
end

@testset "jacobian_form: flux form, 3 policies" begin
    for policy in (CpuSerial(), CpuThreaded(), CpuPolyester()), dirichlet in (nothing, :boundary)

        e, picard, nnz_j, nnz_a = _flux_errors(_space(policy); dirichlet = dirichlet)
        @test e < 1e-12
        @test picard > 1e-3
        @test nnz_j > nnz_a                      # the swapped averages widen the pattern
    end
end

# One case per rule, with the node type it must reach.
const _RULES = (
    ("GridFunctionScale, innerₕ", GridFunctionScale,
        c -> (U, V) -> innerₕ(c * U, V) + innerₕ(U, V)),
    ("OperatorScale", OperatorScale,
        c -> (U, V) -> innerₕ(2.5 * D₋ₓ(c * U), V) + innerₕ(U, V)),
    ("OperatorAdd under the product", OperatorAdd,
        c -> (U, V) -> inner₊(c * ∇ₕ(U)[1] + 2.0 * ∇ₕ(U)[1], ∇ₕ(V)[1]) + innerₕ(U, V)),
    ("BilinearProduct, inner₊", BilinearProduct,
        c -> (U, V) -> inner₊(c * ∇ₕ(U)[2], ∇ₕ(V)[2]) + innerₕ(U, V)),
    ("TappedNode: D₊ above the coefficient", ForwardDifference,
        c -> (U, V) -> -innerₕ(D₊ₓ(c * ∇ₕ(U)[1]), V) + innerₕ(U, V)),
    ("TappedNode: an average", BackwardAverage,
        c -> (U, V) -> innerₕ(Mₕ(c * U)[1], V) + innerₕ(U, V)),
    ("TappedNode: a centred average", CenteredAverage,
        c -> (U, V) -> innerₕ(Mcₕ(c * U)[2], V) + innerₕ(U, V)),
    ("JumpNode", JumpNode,
        c -> (U, V) -> innerₕ(jumpₓ(c * U), V) + innerₕ(U, V)),
    # Shifted forward and backward, so both ends of the grid read off it (#352).
    ("ShiftNode, S₊ (off-grid read)", ShiftNode,
        c -> (U, V) -> innerₕ(S₊ₕ(c * D₋ₓ(U))[1], V) + innerₕ(U, V)),
    ("ShiftNode, S₋ (off-grid read)", ShiftNode,
        c -> (U, V) -> innerₕ(S₋ₕ(c * D₋ₓ(U))[1], V) + innerₕ(U, V)),
    # Two operators composed above the coefficient: a difference over a shift.
    ("D₊ over a shift, composed", ShiftNode,
        c -> (U, V) -> innerₕ(D₊ₓ(S₋ₕ(c * D₋ₓ(U))[1]), V) + innerₕ(U, V)),
    ("RegionRestriction above", RegionRestriction,
        c -> (U, V) -> innerₕ(restrict_to(:interior, c * U), V) + innerₕ(U, V)),
    ("RegionRestriction below", RegionRestriction,
        c -> (U, V) -> innerₕ(c * restrict_to(:boundary, U), V) + innerₕ(U, V))
)

@testset "jacobian_form: one rule each" begin
    W = _space()
    for (name, node, mk) in _RULES
        r = _errors(W, c -> form(W, W, mk(c)), (_r,))
        @test _has_node(r.ast, node)
        @test r.jac < 1e-12
        @test r.picard > 1e-3
        @test r.written
    end
end

# Two coefficients, one inside the other's operand: the outer β reads the inner one at
# neighbouring points. A forward read (`D₊`) reaches points a one-pass refill has not yet
# written, in x within a band and in y across band edges, so these pin the two sweeps.
const _NESTED = (
    (W, c, d) -> form(W, W, (U, V) -> inner₊(c * D₋ₓ(d * U), ∇ₕ(V)[1]) + innerₕ(U, V)),
    (W, c, d) -> form(W, W, (U, V) -> innerₕ(c * D₊ₓ(d * U), V) + innerₕ(U, V)),
    (W, c, d) -> form(W, W, (U, V) -> innerₕ(c * D₊ᵧ(d * U), V) + innerₕ(U, V))
)

@testset "jacobian_form: coefficient nested" begin
    for policy in (CpuSerial(), CpuThreaded(), CpuPolyester()), mk in _NESTED

        W = _space(policy)
        r = _errors(W, (c, d) -> mk(W, c, d), (_r, _α))
        @test r.jac < 1e-12
        @test r.picard > 1e-3
    end
end

@testset "jacobian_form: dependency reads" begin
    W = _space()
    c, d, e = element(W, 0.0), element(W, 0.0), element(W, 0.0)
    e .= 1 .+ 0.5 .* sin.(1:ndofs(W))
    mk(c, d) = form(W, W, (U, V) -> innerₕ(c * U, V) + innerₕ(d * U, V) + innerₕ(U, V))
    # A declared coefficient inside a dependency, another one or itself, is refused.
    for (Lc, j) in ((U -> d * U, 2), (U -> c * U, 1))
        err = try
            jacobian_form(mk(c, d), c => (Lc, _r), d => (U -> U, _α))
        catch ex
            ex
        end
        @test err isa ArgumentError
        @test occursin("reads declared coefficient $j", sprint(showerror, err))
    end
    # An undeclared grid function in a dependency is a constant: accepted, and exact.
    u = _state(W)
    function R(x)
        T = eltype(x)
        cx, dx = element(W, T), element(W, T)
        cx .= _r.(e .* x)
        dx .= _α.(x)
        return assemble(mk(cx, dx)) * x
    end
    Jfd = ForwardDiff.jacobian(R, u)
    a = mk(c, d)
    jf = jacobian_form(a, c => (U -> e * U, _r), d => (U -> U, _α))
    J = jacobian_prototype(jf)
    jacobian!(J, jf, u)
    @test _relerr(J, Jfd) < 1e-12
    @test _relerr(assemble(a), Jfd) > 1e-3
end

@testset "jacobian_form: β refill threaded" begin
    for policy in (CpuThreaded(), CpuPolyester())
        W = _space(policy)
        r = _errors(
            W, c -> form(W, W, (U, V) -> -innerₕ(D₊ₓ(c * ∇ₕ(U)[1]), V) +
                                         innerₕ(U, V)), (_r,); dirichlet = :boundary)
        @test r.jac < 1e-12
    end
end

# A test-only node with no rule, its coefficient in a container field.
struct _Holder{D, P, O <: LazyOp{D}} <: LazyOp{D}
    payload::P
    inner_op::O
end

function _refusal(f)
    try
        f()
    catch e
        return e
    end
    return nothing
end

@testset "jacobian_form: refusals" begin
    W = _space()
    c, e = element(W, 1.0), element(W, 1.0)
    dec = c => (U -> U, _r)

    test_side = form(W, W, (U, V) -> inner₊(∇ₕ(U)[1], c * ∇ₕ(V)[1]))
    err = _refusal(() -> jacobian_form(test_side, dec))
    @test err isa MissingJacobianRule
    @test err.node === :BilinearProduct
    @test occursin("BilinearProduct", sprint(showerror, err))

    # The walk itself, on nodes `form` would not take: a `NamedTuple`, a `Vector` and a
    # closure holding the coefficient are all found, and an unrelated grid function is not.
    w() = Bramble._JFWalk((Bramble._JFCoefficient(c, U -> U, _r),), W, [false], Any[])
    U = TrialFunction{2}()
    for payload in ((; k = c * U), Any[1, c], () -> sum(c))
        err = _refusal(() -> Bramble._dterm(_Holder(payload, U), w()))
        @test err isa MissingJacobianRule
        @test err.node === :_Holder
    end
    @test Bramble._dterm(_Holder((; k = e * U), U), w()) === nothing

    unread = form(W, W, (U, V) -> innerₕ(e * U, V))
    @test_throws ArgumentError jacobian_form(unread, dec)
    @test_throws ArgumentError jacobian_form(form(W, W, (U, V) -> innerₕ(c * U, V)), c => _r)
    @test_throws ArgumentError jacobian_form(form(W, W, (U, V) -> innerₕ(c * U, V)), dec, dec)
    @test_throws ArgumentError jacobian_form(
        form(W, W, (U, V) -> innerₕ(c * U, V)), c => (U -> Mₕ(U), _r))
end

@testset "jacobian_form: warm Jacobian, 0 B" begin
    W = _space()
    ax, ay = element(W, 0.0), element(W, 0.0)
    jf = jacobian_form(_flux(W, ax, ay), ax => (U -> Mₕ(U)[2], _α),
        ay => (U -> Mₕ(U)[1], _α); dirichlet = :boundary)
    J = jacobian_prototype(jf)
    @test alloc_ok(alloc_test(jacobian!, J, jf, _state(W)))
end

end # module
