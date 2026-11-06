# Design spike: the exact Jacobian of a nonlinear form assembled from its AST
# (gpena/Bramble.jl#1118, milestone v4.2.1). Prototype only, never a suite entry.
#
# A form is linear in its trial function; the state enters through coefficients the caller
# computes from it, `cᵢ = fᵢ.(Lᵢ uₕ)` with `Lᵢ` a stencil operator (`Mₕ(U)[1]`). For the
# residual R(u) = A(u) u - F the Jacobian is A(u) plus, for every `GridFunctionScale` node
# `cᵢ * T(U)` of the AST, the same tree with that node replaced by
#
#     βᵢ * Lᵢ(U),     βᵢ = fᵢ′.(Lᵢ u) .* T(u),
#
# the product rule for a pointwise product. Every node above the coefficient is linear and
# is rebuilt around the replaced child; `T(u)` and `Lᵢ u` are read with the stencils the
# assembly already uses (`local_stencil`), at the point the coefficient is read. So `J` is
# one form, `a + ∂a`, assembled into one matrix with `assemble!`: no `Dual` passes through
# assembly, and `fᵢ′` is one scalar `ForwardDiff.derivative`.
#
# A node on a path from the root to a registered coefficient with no rule here throws
# `MissingJacobianRule`, naming it; a node with no coefficient below it contributes nothing
# and needs no rule. A coefficient the walk never reaches is an `ArgumentError`.
#
# Entry points:
#
#   max_jacobian_error()   relative max-abs error of the assembled Jacobian against
#                          `AutoSparse(AutoForwardDiff())` seeded by `ast_sparsity_detector`,
#                          non-uniform 2D meshes, `-∇·(α(u) ∇u)` and a variant with an
#                          operator around the coefficient and a reaction term.
#   refusals()             the `MissingJacobianRule` messages of two forms with no rule.
#   time_case(m, policy)   one process's timings (the driver is at the end of the file).
#
# Timing, run by hand and alone (default -O, never --optimize=1):
#
#     .claude/scripts/check_power_load.sh
#     .claude/scripts/gate_preflight.sh --timing
#     BRAMBLE_JULIA_SLOTS=4 .claude/scripts/julia_slot.sh 4 julia --startup-file=no \
#         --threads=4 --project=test benchmark/assembled_jacobian_spike.jl --time
#
# The driver starts one fresh process per (size, policy, run), three runs back to back, at
# m = 100, 300, 700 (an m × (m+3) mesh) unless sizes follow `--time`. `--smoke` runs m = 40
# once. A child needs about 7.6 GB at m = 700; m = 1000 under CpuThreaded passed the 10 GB cap.
#
# Uses internal names (`local_stencil`, `_rebuilt_around`, `TappedNode`, the node structs):
# a prototype, so it reaches past the public API on purpose.

module AssembledJacobianSpike

using Bramble
using Bramble: allocate_system_matrix, D₊ₓ, D₊ᵧ, S₊ₓ, ast_sparsity_detector, type_cached_assemble!,
               local_stencil, markers, indices, Mₓ!, Mᵧ!, LazyOp, OperatorAdd,
               OperatorScale, GridFunctionScale, BilinearProduct, TrialFunction,
               TappedNode, _rebuilt_around, CpuSerial, CpuThreaded
using DifferentiationInterface: DifferentiationInterface, AutoSparse, AutoForwardDiff,
                                prepare_jacobian
using ForwardDiff: ForwardDiff
using SparseMatrixColorings: GreedyColoringAlgorithm, ncolors
using ADTypes: KnownJacobianSparsityDetector
using LinearAlgebra: mul!, lu, norm
using SparseArrays: SparseMatrixCSC, nnz
using Random: Random

const DI = DifferentiationInterface

# --- The rule set ------------------------------------------------------------------------ #

"""
    Coefficient(values, dependency, f)

A coefficient the form reads, `values == f.(L(uₕ))`, with `dependency(U) = L(U)` naming `L`
on the trial placeholder (as `jacobian_pattern` takes it) and `f` a scalar function.
"""
struct Coefficient{C, L, F}
    values::C
    dependency::L
    f::F
end

"""
    MissingJacobianRule(node, reason)

Thrown when a node on a path to a registered coefficient has no derivative rule: the term
is refused by name, never dropped.
"""
struct MissingJacobianRule <: Exception
    node::Symbol
    reason::String
end

function Base.showerror(io::IO, e::MissingJacobianRule)
    return print(io, "MissingJacobianRule: no Jacobian rule for `", e.node,
        "` above a coefficient that reads the state (", e.reason,
        "); refusing rather than dropping its term")
end

# What `_dterm` carries down the walk: the registered coefficients, which of them were
# reached, and the β buffers to refill at every state with how to refill them.
struct _Walk{C, S}
    coefficients::C
    space::S
    reached::Vector{Bool}
    refills::Vector{Any}
end

# Whether `x`'s subtree reads a registered coefficient. Generic over fields, so a node
# with no rule can never hide one.
function _reads_state(x, coefficients)
    any(c -> c.values === x, coefficients) && return true
    x isa Union{LazyOp, Tuple} || return false
    return any(f -> _reads_state(getfield(x, f), coefficients), fieldnames(typeof(x)))
end

_registered(x, coefficients) = findfirst(c -> c.values === x, coefficients)

# The derivative subtree of `op` through the coefficients, `nothing` where none is read.
function _dterm(op::LazyOp, w::_Walk)
    _reads_state(op, w.coefficients) || return nothing
    throw(MissingJacobianRule(nameof(typeof(op)), "not in the prototype's rule set"))
end

_add(::Nothing, ::Nothing) = nothing
_add(a, ::Nothing) = a
_add(::Nothing, b) = b
_add(a, b) = OperatorAdd(a, b)

function _dterm(op::OperatorAdd, w::_Walk)
    return _add(_dterm(op.left_op, w), _dterm(op.right_op, w))
end

function _dterm(op::OperatorScale, w::_Walk)
    d = _dterm(op.inner_op, w)
    return d === nothing ? nothing : OperatorScale(op.scalar, d)
end

# Differences, averages and jumps: linear, so the rule rebuilds the node around d.
function _dterm(op::TappedNode, w::_Walk)
    d = _dterm(op.inner_op, w)
    return d === nothing ? nothing : _rebuilt_around(op, d)
end

function _dterm(op::BilinearProduct{D, W}, w::_Walk) where {D, W}
    _reads_state(op.right_op, w.coefficients) &&
        throw(MissingJacobianRule(:BilinearProduct, "coefficient on the test side"))
    d = _dterm(op.left_op, w)
    d === nothing && return nothing
    return BilinearProduct{D, W, typeof(d), typeof(op.right_op)}(op.inner, d, op.right_op)
end

# The product rule: δ(c ⊙ T(u)) = (f′(L u) ⊙ T(u)) ⊙ L(δu) + c ⊙ T(δu).
function _dterm(op::GridFunctionScale, w::_Walk)
    inner = _dterm(op.inner_op, w)
    own = inner === nothing ? nothing : GridFunctionScale(op.grid_function, inner)
    k = _registered(op.grid_function, w.coefficients)
    k === nothing && return own
    c = w.coefficients[k]
    w.reached[k] = true
    dependency = c.dependency(_trial_leaf(op.inner_op))
    β = element(w.space, 0.0)
    push!(w.refills, (β, op.inner_op, dependency, c.f))
    return _add(own, β * dependency)
end

_trial_leaf(op::TrialFunction) = op
function _trial_leaf(op::LazyOp)
    for f in fieldnames(typeof(op))
        x = getfield(op, f)
        x isa LazyOp || continue
        leaf = _trial_leaf(x)
        leaf === nothing || return leaf
    end
    return nothing
end

"""
    AssembledJacobian(a, coefficients...; dirichlet)

`a + ∂a` as one form, its matrix, and the β buffers to refill. `coefficients` are
[`Coefficient`](@ref)s naming the grid functions `a` reads.
"""
struct AssembledJacobian{F, M, R, B}
    form::F
    matrix::M
    refills::R
    dirichlet::B
end

function AssembledJacobian(a, coefficients::Coefficient...; dirichlet = :boundary)
    W = a.trial_space
    w = _Walk(coefficients, W, fill(false, length(coefficients)), Any[])
    d = _dterm(a.ast, w)
    for k in eachindex(w.reached)
        w.reached[k] || throw(ArgumentError(
            "coefficient $k is not read by the form's AST: its derivative would be lost"))
    end
    ast = d === nothing ? a.ast : a.ast + d
    jform = form(W, a.test_space, (U, V) -> ast)
    J = allocate_system_matrix(jform)
    return AssembledJacobian(jform, J, Tuple(w.refills), dirichlet)
end

# Σ weight·u over a point's stencil; an entry off the grid reads 0, as assembly drops it.
@inline _stencil_dot(::Tuple{}, I, lin, u) = 0.0
@inline function _stencil_dot(st::Tuple, I, lin, u)
    off, wt = first(st)
    J = I + CartesianIndex(off)
    v = checkbounds(Bool, lin, J) ? wt * @inbounds(u[lin[J]]) : 0.0
    return v + _stencil_dot(Base.tail(st), I, lin, u)
end

@inline function _apply_at(op, W, I, mk, lin, u)
    return _stencil_dot(local_stencil(op, W, I, mk, lin[I]), I, lin, u)
end

# `f::F`: passed on uncalled, `f` would not be specialised on, and every point would
# allocate in `ForwardDiff.derivative`.
function _refill_point!(β, T, L, f::F, W, I, mk, lin, u) where {F}
    x = _apply_at(L, W, I, mk, lin, u)
    t = _apply_at(T, W, I, mk, lin, u)
    @inbounds β[lin[I]] = ForwardDiff.derivative(f, x) * t
    return nothing
end

function _refill!(refill, W, u, ::CpuSerial)
    β, T, L, f = refill
    Ωₕ = mesh(W)
    lin = LinearIndices(indices(Ωₕ))
    mk = markers(Ωₕ)
    for I in CartesianIndices(lin)
        _refill_point!(β, T, L, f, W, I, mk, lin, u)
    end
    return β
end

function _refill!(refill, W, u, ::CpuThreaded)
    β, T, L, f = refill
    Ωₕ = mesh(W)
    lin = LinearIndices(indices(Ωₕ))
    mk = markers(Ωₕ)
    CI = CartesianIndices(lin)
    Threads.@threads :static for j in axes(CI, ndims(CI))
        for I in selectdim(CI, ndims(CI), j)
            _refill_point!(β, T, L, f, W, I, mk, lin, u)
        end
    end
    return β
end

_policy(::Bramble.Backend{V, M, P}) where {V, M, P} = P()

"""
    jacobian!(aj::AssembledJacobian, u) -> SparseMatrixCSC

The exact Jacobian at `u`; the form's own coefficients must already hold their values at
`u` (the residual's refill does that).
"""
function jacobian!(aj::AssembledJacobian, u::AbstractVector)
    W = aj.form.trial_space
    policy = _policy(Bramble.backend(mesh(W)))
    foreach(r -> _refill!(r, W, u, policy), aj.refills)
    assemble!(aj.matrix, aj.form; dirichlet = aj.dirichlet)
    return aj.matrix
end

# --- The problem ------------------------------------------------------------------------- #

α(u) = 3 + 1 / (1 + u^2)
κ(u) = 1 + u^2 / 4                           # the reaction coefficient of the variant
uexact(x) = exp(x[1] + x[2])

# `-∇·(α(u)∇u) [+ κ(u) u] = g`, u = uexact on the boundary; `g` is a smooth load (the
# Jacobian check needs no manufactured solution).
g(x) = 1 + x[1] * x[2]

function _mesh(m, policy; seed = 1118)
    Random.seed!(seed)
    Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
    return mesh(Ω, (m, m + 3), (false, false); backend = backend(policy = policy))
end

# The form and its refill at state element `uₕ`, eltype from `uₕ` (a `Dual` one in the
# ForwardDiff sweeps). `variant = :flux` is the stated `inner₊(α ∇u, ∇v)`; `:strong`
# puts an operator around the coefficient, `-innerₕ(D₊(α D₋u), v)`, and adds `innerₕ(κ u, v)`.
struct Build{W, V}
    space::W
    variant::V
end

function (b::Build)(uₕ)
    W = b.space
    T = eltype(uₕ)
    Mx, My = element(W, T), element(W, T)
    αx, αy, κv = element(W, T), element(W, T), element(W, T)
    a = _diffusion_form(W, αx, αy, κv, b.variant)
    function refill!(uₕ)
        Mₓ!(Mx, uₕ)
        Mᵧ!(My, uₕ)
        αx .= α.(Mx)
        αy .= α.(My)
        b.variant === Val(:strong) && (κv .= κ.(uₕ))
        return nothing
    end
    return a, refill!, (αx, αy, κv)
end

function _diffusion_form(W, αx, αy, κv, ::Val{:flux})
    return form(W, W, (U, V) -> inner₊((αx * ∇ₕ(U)[1], αy * ∇ₕ(U)[2]), ∇ₕ(V)))
end

function _diffusion_form(W, αx, αy, κv, ::Val{:strong})
    return form(W, W, (U, V) -> -innerₕ(D₊ₓ(αx * ∇ₕ(U)[1]), V) -
                                innerₕ(D₊ᵧ(αy * ∇ₕ(U)[2]), V) + innerₕ(κv * U, V))
end

_coefficients(αx, αy, κv, ::Val{:flux}) = (
    Coefficient(αx, U -> Mₕ(U)[1], α), Coefficient(αy, U -> Mₕ(U)[2], α))
function _coefficients(αx, αy, κv, ::Val{:strong})
    (
        Coefficient(αx, U -> Mₕ(U)[1], α), Coefficient(αy, U -> Mₕ(U)[2], α),
        Coefficient(κv, U -> U, κ))
end

_dependencies(::Val{:flux}) = (U -> Mₕ(U),)
_dependencies(::Val{:strong}) = (U -> Mₕ(U), U -> U)

"""
    Problem(m, policy; variant = :flux)

Everything both Jacobians need on an `m × (m+3)` non-uniform mesh: the load, the cached
`Float64` residual (shared by both paths), the ForwardDiff residual and backend, the
assembled Jacobian.
"""
struct Problem{W, F, B, A, R, M, C, U}
    space::W
    load::F
    build::B
    form::A
    refill::R
    matrix::M
    coefficients::C
    state::U
    cache::Dict{Any, Any}
end

function Problem(m, policy; variant = :flux)
    Ωₕ = _mesh(m, policy)
    W = gridspace(Ωₕ)
    Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
    bcs = dirichlet_constraints(Ω, :boundary => uexact)
    gₕ = element(W)
    avgₕ!(gₕ, g)
    F = assemble(form(W, v -> innerₕ(gₕ, v)); dirichlet = bcs)
    b = Build(W, Val(variant))
    uₕ = element(W, 0.0)
    a, refill!, (αx, αy, κv) = b(uₕ)
    A = allocate_system_matrix(a)
    coefficients = _coefficients(αx, αy, κv, Val(variant))
    return Problem(W, F, b, a, refill!, A, coefficients, uₕ, Dict{Any, Any}())
end

# The Float64 residual the assembled path uses: refill, assemble! in place, A u - F.
function residual!(r, p::Problem, u)
    p.state .= u
    p.refill(p.state)
    assemble!(p.matrix, p.form; dirichlet = :boundary)
    mul!(r, p.matrix, u)
    r .-= p.load
    return r
end

# The residual ForwardDiff differentiates: `type_cached_assemble!`, one cached form per
# element type (the fastest documented route, docs/src/examples/poisson_nonlinear.jl).
struct FDResidual{P}
    problem::P
end

function (fr::FDResidual)(u::AbstractVector{T}) where {T}
    p = fr.problem
    uₕ = element(p.space, T)
    uₕ .= u
    build2 = x -> begin
        a, refill!, _ = p.build(x)
        return a, refill!
    end
    A = type_cached_assemble!(build2, p.cache, uₕ; dirichlet = :boundary)
    return A * u .- p.load
end

function fd_backend(p::Problem)
    detector = ast_sparsity_detector(p.form, _dependencies(p.build.variant)...)
    return AutoSparse(AutoForwardDiff(); sparsity_detector = detector,
        coloring_algorithm = GreedyColoringAlgorithm())
end

# A state that is not a solution, so every coefficient's derivative is exercised.
function _test_state(p::Problem)
    n = ndofs(p.space)
    return [0.3 + 0.7 * sin(0.37 * i) for i in 1:n]
end

function _relative_error(J1, J2)
    return maximum(abs, J1 - J2) / maximum(abs, J2)
end

"""
    max_jacobian_error() -> Float64

The largest relative max-abs difference between the assembled Jacobian and
`AutoSparse(AutoForwardDiff())` seeded by `ast_sparsity_detector`, over the two variants on
two non-uniform 2D meshes, under `CpuSerial` and `CpuThreaded`.
"""
function max_jacobian_error()
    err = 0.0
    for policy in (CpuSerial(), CpuThreaded()), variant in (:flux, :strong), m in (9, 24)
        err = max(err, jacobian_error(m, variant, policy))
    end
    return err
end

function jacobian_error(m, variant, policy)
    p = Problem(m, policy; variant = variant)
    u = _test_state(p)
    fr = FDResidual(p)
    backend = fd_backend(p)
    prep = prepare_jacobian(fr, backend, u)
    Jfd = DI.jacobian(fr, prep, backend, u)
    r = similar(u)
    residual!(r, p, u)                       # refills the coefficients at u
    aj = AssembledJacobian(p.form, p.coefficients...)
    Jast = jacobian!(aj, u)
    return _relative_error(Jast, Jfd)
end

"""
    refusals() -> Vector{String}

The messages two forms with no rule throw: a coefficient under an index shift, and one on
the test side.
"""
function refusals()
    W = gridspace(_mesh(9, CpuSerial()))
    c = element(W, 1.0)
    coef = Coefficient(c, U -> Mₕ(U)[1], α)
    shifted = form(W, W, (U, V) -> inner₊(S₊ₓ(c * ∇ₕ(U)[1]), ∇ₕ(V)[1]))
    test_side = form(W, W, (U, V) -> inner₊(∇ₕ(U)[1], c * ∇ₕ(V)[1]))
    msgs = String[]
    for a in (shifted, test_side)
        try
            AssembledJacobian(a, coef)
            push!(msgs, "no error")
        catch e
            e isa MissingJacobianRule || rethrow()
            push!(msgs, sprint(showerror, e))
        end
    end
    return msgs
end

# --- Timing ------------------------------------------------------------------------------ #

# The sparse-ForwardDiff Jacobian as the documented route builds it: prepared once, its
# `Dual` form cached by `type_cached_assemble!` in its own dictionary.
struct FDJacobian{R, B, P, M}
    residual::R
    backend::B
    prep::P
    matrix::M
end

struct FDCachedResidual{P}
    problem::P
    cache::Dict{Any, Any}
end

function (fr::FDCachedResidual)(u::AbstractVector{T}) where {T}
    p = fr.problem
    uₕ = element(p.space, T)
    uₕ .= u
    A = type_cached_assemble!(_Build2(p.build), fr.cache, uₕ; dirichlet = :boundary)
    return A * u .- p.load
end

struct _Build2{B}
    build::B
end
function (b::_Build2)(uₕ)
    a, refill!, _ = b.build(uₕ)
    return a, refill!
end

function FDJacobian(p::Problem, u; detector = nothing)
    fr = FDCachedResidual(p, Dict{Any, Any}())
    be = detector === nothing ? fd_backend(p) :
         AutoSparse(AutoForwardDiff(); sparsity_detector = detector,
        coloring_algorithm = GreedyColoringAlgorithm())
    prep = prepare_jacobian(fr, be, u)
    J = DI.jacobian(fr, prep, be, u)
    return FDJacobian(fr, be, prep, J)
end

function jacobian!(fj::FDJacobian, u::AbstractVector)
    return DI.jacobian!(fj.residual, fj.matrix, fj.prep, fj.backend, u)
end

# The assembled route's setup: the walk, the form `a + ∂a`, its pattern, and one Jacobian.
function assembled_setup(p::Problem, u)
    aj = AssembledJacobian(p.form, p.coefficients...)
    jacobian!(aj, u)
    return aj
end

_seconds(t0) = (time_ns() - t0) / 1.0e9

function _timed(f::F) where {F}
    GC.gc()
    t0 = time_ns()
    x = f()
    return _seconds(t0), x
end

function _min_time(f::F, reps) where {F}
    best = Inf
    for _ in 1:reps
        best = min(best, first(_timed(f)))
    end
    return best
end

# Residual (refilling the coefficients at u), Jacobian, factor and solve: one Newton step.
function newton_step!(du, r, p::Problem, jac, u)
    residual!(r, p, u)
    J = jacobian!(jac, u)
    du .= lu(J) \ r
    return du
end

# Newton from u = 0 to ‖R(u)‖₂ ≤ rtol ‖R(0)‖₂; returns the iteration count. Relative: the
# round-off floor of ‖R‖₂ grows with the mesh (2e-11 at 1e4 dofs, 3.5e-10 at 9e4 here).
function newton!(u, r, p::Problem, jac; rtol = 1.0e-10, maxit = 25)
    fill!(u, 0.0)
    r0 = norm(residual!(r, p, u))
    for it in 0:maxit
        residual!(r, p, u)
        norm(r) <= rtol * r0 && return it
        J = jacobian!(jac, u)
        u .-= lu(J) \ r
    end
    error("Newton did not converge in $maxit steps")
end

function _allocated_jacobian(aj, u)
    jacobian!(aj, u)
    return @allocated jacobian!(aj, u)
end

function _pattern(J::SparseMatrixCSC)
    return SparseMatrixCSC(J.m, J.n, copy(J.colptr), copy(J.rowval), fill(true, nnz(J)))
end

"""
    time_case(m, policy; reps = 5, solve_reps = 3) -> NamedTuple

One process's figures on the `m × (m+3)` non-uniform mesh, `:flux` variant: cold setup
(first in the process, compilation included), warm setup, the Jacobian alone, one Newton
step, and a Newton solve, for the assembled route and sparse ForwardDiff.
"""
function time_case(m, policy; reps = 5, solve_reps = 3)
    p = Problem(m, policy)
    u = _test_state(p)
    r = similar(u)
    residual!(r, p, u)                                   # the shared Float64 path, compiled
    cold_fd, fj = _timed(() -> FDJacobian(p, u))
    residual!(r, p, u)
    cold_ast, aj = _timed(() -> assembled_setup(p, u))
    err = _relative_error(jacobian!(aj, u), jacobian!(fj, u))
    err < 1.0e-12 || error("assembled Jacobian disagrees with sparse ForwardDiff: $err")

    warm_fd = _min_time(() -> FDJacobian(p, u), 3)
    residual!(r, p, u)
    warm_ast = _min_time(() -> assembled_setup(p, u), 3)

    ft = FDJacobian(p, u; detector = KnownJacobianSparsityDetector(_pattern(aj.matrix)))
    colors_tight = ncolors(ft.prep.coloring_result)

    jac_fd = _min_time(() -> jacobian!(fj, u), reps)
    jac_tight = _min_time(() -> jacobian!(ft, u), reps)
    residual!(r, p, u)
    jac_ast = _min_time(() -> jacobian!(aj, u), reps)
    alloc_ast = _allocated_jacobian(aj, u)

    du = similar(u)
    step_fd = _min_time(() -> newton_step!(du, r, p, fj, u), reps)
    step_ast = _min_time(() -> newton_step!(du, r, p, aj, u), reps)

    u_fd, u_ast = similar(u), similar(u)
    it_fd = newton!(u_fd, r, p, fj)
    it_ast = newton!(u_ast, r, p, aj)
    it_fd == it_ast || error("iteration counts differ: FD $it_fd, assembled $it_ast")
    maximum(abs, u_fd - u_ast) < 1.0e-10 || error("Newton solutions differ")
    solve_fd = _min_time(() -> newton!(u_fd, r, p, fj), solve_reps)
    solve_ast = _min_time(() -> newton!(u_ast, r, p, aj), solve_reps)

    return (dofs = length(u), nnz_ast = nnz(aj.matrix), nnz_fd = nnz(fj.matrix),
        colors = ncolors(fj.prep.coloring_result), colors_tight = colors_tight,
        err = err, iters = it_ast, alloc_ast = alloc_ast,
        cold_fd = cold_fd, cold_ast = cold_ast, warm_fd = warm_fd, warm_ast = warm_ast,
        jac_fd = jac_fd, jac_tight = jac_tight, jac_ast = jac_ast,
        step_fd = step_fd, step_ast = step_ast, solve_fd = solve_fd, solve_ast = solve_ast)
end

end # module

# --- Driver: one fresh process per (size, policy, run) ------------------------------------ #

if abspath(PROGRAM_FILE) == @__FILE__
    using Bramble, DifferentiationInterface, SparseConnectivityTracer,
          SparseMatrixColorings, ForwardDiff

    const _POLICIES = Dict("CpuSerial" => Bramble.CpuSerial(),
        "CpuThreaded" => Bramble.CpuThreaded())
    const _FIELDS = (:cold_fd, :cold_ast, :warm_fd, :warm_ast, :jac_fd, :jac_tight,
        :jac_ast, :step_fd, :step_ast, :solve_fd, :solve_ast)

    function _child(m, policy_name)
        set_zero_subnormals(true)
        r = AssembledJacobianSpike.time_case(m, _POLICIES[policy_name];
            solve_reps = m >= 700 ? 1 : 3)
        println("RESULT ", join(("$k=$v" for (k, v) in pairs(r)), " "),
            " maxrss_mb=", Sys.maxrss() / 2^20)
        return nothing
    end

    function _run(m, policy_name)
        julia = joinpath(Sys.BINDIR, Base.julia_exename())
        project = joinpath(@__DIR__, "..", "test")
        cmd = `$julia --startup-file=no --threads=4 --project=$project $(@__FILE__) --child $m $policy_name`
        text = read(cmd, String)
        line = only(filter(l -> startswith(l, "RESULT "), split(text, '\n')))
        return Dict(Symbol(k) => parse(Float64, v)
        for (k, v) in (split(kv, '='; limit = 2) for kv in split(line)[2:end]))
    end

    _median(x) = (s = sort(x); n = length(s); isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2)

    function _driver(sizes, runs)
        rows = []
        for m in sizes, name in ("CpuSerial", "CpuThreaded")

            results = [_run(m, name) for _ in 1:runs]
            for r in results
                println("run m=$m policy=$name ", join(("$k=$(round(r[k]; sigdigits = 4))" for k in keys(r)), " "))
                flush(stdout)
            end
            push!(rows, (m = m, policy = name, results = results))
        end
        println("\nmedian [min, max] over $runs processes, seconds")
        println("| dofs | policy | " * join(string.(_FIELDS), " | ") * " |")
        println("|---:|---|" * repeat("---:|", length(_FIELDS)))
        for row in rows
            cells = map(_FIELDS) do k
                x = [r[k] for r in row.results]
                _g(x) = string(round(x; sigdigits = 4))
                "$(_g(_median(x))) [$(_g(minimum(x))), $(_g(maximum(x)))]"
            end
            r1 = first(row.results)
            println("| $(Int(r1[:dofs])) | $(row.policy) | " * join(cells, " | ") * " |")
        end
        for row in rows
            r1 = first(row.results)
            println("dofs=$(Int(r1[:dofs])) $(row.policy): nnz_ast=$(Int(r1[:nnz_ast])) ",
                "nnz_fd=$(Int(r1[:nnz_fd])) colors=$(Int(r1[:colors])) ",
                "colors_tight=$(Int(r1[:colors_tight])) iters=$(Int(r1[:iters])) ",
                "max err=$(maximum(r[:err] for r in row.results)) ",
                "alloc_ast=$(Int.([r[:alloc_ast] for r in row.results]))")
        end
        return rows
    end

    if length(ARGS) >= 3 && ARGS[1] == "--child"
        _child(parse(Int, ARGS[2]), ARGS[3])
    elseif "--smoke" in ARGS
        _driver((40,), 1)
    elseif length(ARGS) >= 1 && ARGS[1] == "--time"
        sizes = length(ARGS) > 1 ? Tuple(parse.(Int, ARGS[2:end])) : (100, 300, 700)
        _driver(sizes, 3)
    end
end
