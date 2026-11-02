# test/ext/enzyme_policy_cases.jl: the Enzyme cases for the `_batch_*` hooks that
# `CpuPolyester` fills (ext/BramblePolyesterExt.jl), and for the `CpuThreaded` engines that
# S2.2 checks. Enzyme is not in test/Project.toml, so load it through a temporary environment
# first; benchmark/enzyme_policy_census.jl does that. Standalone:
#
#   julia --project=test -t 2 -e 'using Bramble; include("test/TestUtils.jl");
#     include("test/ext/enzyme_policy_cases.jl")'   (after the temporary environment)
#
# A case is a builder `policy -> (f, x)`: `f` a scalar function of the parameters `x`, built
# so that one public call reaches one hook. `kind` says what the hook runs: `fixed` for a
# Bramble kernel, `closure` for a closure the caller passes. Which hooks a builder reaches is
# measured by benchmark/enzyme_policy_census.jl, not asserted here.
#
# `check(hook; policy, modes)` throws unless every mode's gradient matches
# `ForwardDiff.gradient` to 1e-10 relative. The modes are `:reverse` (`Enzyme.gradient` in
# reverse mode, called directly), `:di_reverse` (the same through DifferentiationInterface)
# and `:forward` (DifferentiationInterface in forward mode). The ForwardDiff oracle goes
# through DifferentiationInterface; the Enzyme reverse call stays direct, because this file
# tests the Enzyme rules themselves.
#
# Two more modes, `BATCH_MODES`, run Enzyme directly at width `BATCH_WIDTH`, as DI's sparse
# Jacobians do when they seed several colours per pass: `:batch_forward` pushes distinct
# tangents of the parameters through one `BatchDuplicated` call, and `:batch_reverse` pulls
# distinct cotangents of the value through one. Each lane must match the ForwardDiff oracle
# to 1e-10, and under a threaded policy the hook's own rule must have been entered
# (`_rule_entries` of a Bramble Enzyme extension, status `"bypassed"` otherwise). `check`
# takes the hook and the policy as names too: `check(:_batch_for!; policy = :CpuPolyester)`.
#
# Each builder keeps the parameters small (`NPARAM`) and expands them into the field with a
# seeded basis, so a builder's own work is not a stack of other hooks.

module EnzymePolicyCases

using Bramble
using Bramble: CpuPolyester, CpuSerial, D₋ₓ, D₋ᵧ, Mₓ, Mᵧ, Mcₓ, Mcᵧ, S₊, semidiscretize_rhs
using Bramble: _sweep_for!
using Enzyme
using ForwardDiff
using Kronecker
using LinearAlgebra
using Polyester
using Random
using SparseArrays
import DifferentiationInterface as DI
using DifferentiationInterface: AutoEnzyme, AutoForwardDiff

Threads.nthreads() >= 2 ||
    error("enzyme_policy_cases.jl needs at least 2 threads, got $(Threads.nthreads())")

const MODES = (:reverse, :di_reverse, :forward)
const BATCH_MODES = (:batch_forward, :batch_reverse)
const BATCH_WIDTH = 3
const BATCH_COTANGENTS = (1.0, -0.5, 2.0)
const TOL = 1e-10
const N = 41
const NPARAM = 4
const SEED = 1234
const ERROR_CHARS = 400

# A non-uniform mesh of `ns` nodes on the unit square, on `policy`. Its interior nodes come
# from the package-local RNG, seeded, so a failing relerr reproduces across processes.
function _seeded_mesh(ns::NTuple{2, Int}, policy)
    Bramble._seed_mesh1d_rng!(SEED)
    try
        return mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), ns, (false, false);
            backend = backend(policy = policy))
    finally
        Bramble._unseed_mesh1d_rng!()
    end
end

# A non-uniform N x N mesh on the unit square, on `policy`.
_mesh(policy) = _seeded_mesh((N, N), policy)

# The field of a space is `B * p`, with `B` seeded, so every policy differentiates the same
# nodal values.
_basis(W) = randn(MersenneTwister(SEED), ndofs(W), NPARAM)
_start() = [0.5 + 0.3 * sin(k) for k in 1:NPARAM]

function _load(W, B, p::AbstractVector{T}) where {T}
    u = element(W, T)
    parent(u) .= B * p
    return u
end

function _fixed(W)
    w = element(W)
    parent(w) .= cos.(1:ndofs(W))
    return w
end

# `_batch_dot`, through `innerₕ` on two elements.
function _build_dot(policy)
    W = gridspace(_mesh(policy))
    B = _basis(W)
    w = _fixed(W)
    f(p) = innerₕ(_load(W, B, p), w)
    return f, _start()
end

# `_batch_dot_masked`, through the positional `_dot_masked` that `innerₕ(…; markers)` lowers
# to. The keyword call aborts Enzyme's compiler even under `CpuSerial`
# (`needsCacheWholeAllocation`, the kwcall of `inner` with `__markers__`), so the mask and the
# weights are built outside `f`, where they do not depend on the parameters.
function _build_dot_masked(policy)
    W = gridspace(_mesh(policy))
    B = _basis(W)
    w = _fixed(W)
    mask = Bramble._combined_marked_indices(mesh(W), (:boundary,))
    wts = Bramble.weights(W, Val(()))
    f(p) = Bramble._dot_masked(policy, _load(W, B, p).data, wts, w.data, mask)
    return f, _start()
end

# `_batch_for!`, through `Rₕ` with a closure over the parameters.
function _build_for(policy)
    W = gridspace(_mesh(policy))
    f(p) = sum(abs2, parent(Rₕ(W, x -> p[1] * sin(x[1]) + p[2] * x[2]^2 + p[3] * x[1] * x[2])))
    return f, _start()
end

# `_batch_axis_for!`: no public caller, so `_sweep_for!` on a `CartesianIndices`.
function _build_axis_for(policy)
    idxs = CartesianIndices((1:N, 1:N))
    function f(p)
        v = zeros(eltype(p), N, N)
        _sweep_for!(policy, v, idxs, x -> p[1] * x[1] + p[2] * x[2] + p[3] * sin(x[1] * x[2]))
        return sum(abs2, v)
    end
    return f, _start()
end

# `_batch_scatter_for!`, through `Rₕ` on a composite space.
function _build_scatter_for(policy)
    W = gridspace(_mesh(policy), Val(2))
    function f(p)
        u = Rₕ(W, x -> (p[1] * sin(x[1]) + p[2] * x[2], p[3] * x[1] * x[2] + p[4] * cos(x[2])))
        return sum(c -> sum(abs2, parent(c)), Bramble.components(u))
    end
    return f, _start()
end

policy_α(u) = 3 + 1 / (1 + u^2)

# A thin last axis, so `_band_count` is 0 and the bilinear and linear sweeps take their
# point-coloured path. The band cases keep the N x N mesh.
_thin_mesh(policy) = _seeded_mesh((N, 3), policy)

# A diffusion form whose coefficient is averaged from the parameters, the residual the
# sparse Enzyme cells of test/space/autodiff_policies.jl use.
function _diffusion(M)
    W = gridspace(M)
    B = _basis(W)
    function f(p)
        uₕ = _load(W, B, p)
        a = form(W, W, (U, V) -> inner₊(map(c -> policy_α.(c), Mₕ(uₕ)) * ∇ₕ(U), ∇ₕ(V)))
        return sum(abs2, nonzeros(assemble(a)))
    end
    return f, _start()
end

# `_batch_bilinear_band_replay!`, on the N x N mesh.
_build_diffusion(policy) = _diffusion(_mesh(policy))

# `_batch_bilinear_colour_replay!`, on the thin mesh.
_build_colour(policy) = _diffusion(_thin_mesh(policy))

# A linear form whose load is the parameters' field.
function _linear(M)
    W = gridspace(M)
    B = _basis(W)
    function f(p)
        uₕ = _load(W, B, p)
        return sum(abs2, assemble(form(W, v -> innerₕ(uₕ, v))))
    end
    return f, _start()
end

# `_batch_linear_band_sweep!`, on the N x N mesh.
_build_linear(policy) = _linear(_mesh(policy))

# `_batch_linear_colour_sweep!`: the test function's y-difference gives the term a colouring
# stride of 2 along the thin last axis, so `_band_count` is 0 there (a plain load's stride of
# 1 always bands, at 41 points or 3).
function _build_linear_colour(policy)
    W = gridspace(_thin_mesh(policy))
    B = _basis(W)
    function f(p)
        uₕ = _load(W, B, p)
        return sum(abs2, assemble(form(W, v -> innerₕ(uₕ, D₋ᵧ(v)))))
    end
    return f, _start()
end

# `_batch_difference_engine!`, through `D₋ₓ`.
function _build_difference(policy)
    W = gridspace(_mesh(policy))
    B = _basis(W)
    f(p) = sum(abs2, parent(D₋ₓ(_load(W, B, p))))
    return f, _start()
end

# `_batch_average_engine!`, through `Mₓ` and `Mᵧ`: the averaged axis across the bands and
# along them (the last axis, which the bands cut), whose adjoints differ.
function _build_average(policy)
    W = gridspace(_mesh(policy))
    B = _basis(W)
    function f(p)
        u = _load(W, B, p)
        return sum(abs2, parent(Mₓ(u))) + sum(abs2, parent(Mᵧ(u)))
    end
    return f, _start()
end

# `_batch_centered_average_engine!`, through `Mcₓ` and `Mcᵧ`, both axes as above.
function _build_centered(policy)
    W = gridspace(_mesh(policy))
    B = _basis(W)
    function f(p)
        u = _load(W, B, p)
        return sum(abs2, parent(Mcₓ(u))) + sum(abs2, parent(Mcᵧ(u)))
    end
    return f, _start()
end

# `_batch_run_bands!`, through the forward shift in direction 1 (`_shift_engine!`).
function _build_run_bands(policy)
    W = gridspace(_mesh(policy))
    B = _basis(W)
    f(p) = sum(abs2, parent(S₊(_load(W, B, p), 1)))
    return f, _start()
end

# A form whose matrix is not symmetric, so a reverse that applied `A` for `Aᵀ` is caught.
_nonsymmetric(W) = form(W, W, (U, V) -> inner₊(∇ₕ(U), ∇ₕ(V)) + innerₕ(U, D₋ᵧ(V)))

# `_batch_mf_bands!`: `mul!` with a matrix-free operator.
function _build_mf(policy)
    W = gridspace(_mesh(policy))
    B = _basis(W)
    op = matrix_free_operator(_nonsymmetric(W))
    function f(p)
        y = zeros(eltype(p), size(op, 1))
        mul!(y, op, parent(_load(W, B, p)))
        return sum(abs2, y)
    end
    return f, _start()
end

# `_batch_broadcast!`: a broadcast assignment between two elements.
function _build_broadcast(policy)
    W = gridspace(_mesh(policy))
    B = _basis(W)
    function f(p)
        v = _load(W, B, p)
        u = element(W, eltype(p))
        u .= v .+ 1.0
        return sum(abs2, parent(u))
    end
    return f, _start()
end

# `_batch_kron_lines!`: `mul!` with a Kronecker operator.
function _build_kron(policy)
    W = gridspace(_mesh(policy))
    B = _basis(W)
    K = kronecker_operator(_nonsymmetric(W))
    function f(p)
        y = zeros(eltype(p), size(K, 1))
        mul!(y, K, parent(_load(W, B, p)))
        return sum(abs2, y)
    end
    return f, _start()
end

# `_batch_csr_spmv!`: the right-hand side of a semidiscretisation.
function _build_csr(policy)
    W = gridspace(_mesh(policy))
    B = _basis(W)
    w = _fixed(W)
    sd = semidiscretize(_nonsymmetric(W), form(W, v -> innerₕ(w, v)))
    rhs = semidiscretize_rhs(sd)
    function f(p)
        du = zeros(eltype(p), ndofs(W))
        rhs(du, B * p, nothing, 0.0)
        return sum(abs2, du)
    end
    return f, _start()
end

# The `_batch_*` hooks, each with the builder that reaches it under `CpuPolyester`.
const CASES = (
    (hook = "_batch_dot", kind = "fixed", build = _build_dot),
    (hook = "_batch_dot_masked", kind = "fixed", build = _build_dot_masked),
    (hook = "_batch_for!", kind = "closure", build = _build_for),
    (hook = "_batch_axis_for!", kind = "closure", build = _build_axis_for),
    (hook = "_batch_scatter_for!", kind = "closure", build = _build_scatter_for),
    (hook = "_batch_bilinear_band_replay!", kind = "closure", build = _build_diffusion),
    (hook = "_batch_bilinear_colour_replay!", kind = "closure", build = _build_colour),
    (hook = "_batch_linear_colour_sweep!", kind = "closure", build = _build_linear_colour),
    (hook = "_batch_linear_band_sweep!", kind = "closure", build = _build_linear),
    (hook = "_batch_difference_engine!", kind = "fixed", build = _build_difference),
    (hook = "_batch_average_engine!", kind = "fixed", build = _build_average),
    (hook = "_batch_centered_average_engine!", kind = "fixed", build = _build_centered),
    (hook = "_batch_run_bands!", kind = "closure", build = _build_run_bands),
    (hook = "_batch_mf_bands!", kind = "fixed", build = _build_mf),
    (hook = "_batch_broadcast!", kind = "fixed", build = _build_broadcast),
    (hook = "_batch_kron_lines!", kind = "fixed", build = _build_kron),
    (hook = "_batch_csr_spmv!", kind = "fixed", build = _build_csr)
)

# The `CpuThreaded` engine entries S2.2 checks, each run under `CpuThreaded` with the builder
# whose call reaches it.
const THREADED = (
    (hook = "_static_bands!", kind = "fixed", build = _build_difference),
    (hook = "_static_for!", kind = "closure", build = _build_for),
    (hook = "_static_scatter_for!", kind = "closure", build = _build_scatter_for),
    (hook = "_static_partials!", kind = "fixed", build = _build_dot),
    (hook = "_static_band_colour!", kind = "closure", build = _build_diffusion),
    (hook = "_static_colour!", kind = "closure", build = _build_colour),
    (hook = "_static_cut_band_colour!", kind = "closure", build = _build_diffusion),
    (hook = "_kron_fused!", kind = "fixed", build = _build_kron),
    (hook = "_mf_run_bands!", kind = "fixed", build = _build_mf)
)

# `_batch_for!` under a user closure that captures an array and a `Float64`, the shape whose
# by-value call fails Enzyme's calling convention unless `_sweep_for!` passes the kernel
# behind a `Ref`. `rule` names the hook whose rule must be entered.
function _capture(policy, a::A) where {A}
    function f(p)
        v = zeros(eltype(p), N)
        c = a(p)
        _sweep_for!(policy, v, 1:N, i -> c * p[mod1(i, NPARAM)] * sin(i))
        return sum(abs2, v)
    end
    return f, _start()
end

# A constant `Float64`, and one computed from the parameters (a `MixedDuplicated` closure).
_build_const_capture(policy) = _capture(policy, p -> 2.0)
_build_mixed_capture(policy) = _capture(policy, p -> p[1]^2)

# A closure that reads the vector `_batch_for!` writes, nonlinearly and times a parameter: the
# reverse pass must see `v` as it was before the call.
function _build_reads_v(policy)
    function f(p)
        v = [p[mod1(i, NPARAM)] * i / N for i in 1:N]
        _sweep_for!(policy, v, 1:N, i -> sin(v[i]) * v[i] * p[1])
        return sum(abs2, v)
    end
    return f, _start()
end

# `_batch_mf_bands!` with the form's coefficient computed from the parameters: the reverse
# reaches the coefficient through the plan's form.
function _build_mf_coef(policy)
    W = gridspace(_mesh(policy))
    B = _basis(W)
    x = parent(_fixed(W))
    function f(p)
        κ = _load(W, B, p)
        op = matrix_free_operator(form(W, W, (U, V) -> inner₊(Mₕ(κ) * ∇ₕ(U), ∇ₕ(V))))
        y = zeros(eltype(p), size(op, 1))
        mul!(y, op, x)
        return sum(abs2, y)
    end
    return f, _start()
end

# A fused matrix-free product whose coefficient is held by reference in `κ * U`, not copied
# as `Mₕ(κ)` copies. `κ` is scaled by `_KAPPA` so the term it enters dominates `y`, and it is
# overwritten after `mul!`: a reverse that replays the parts after that (no parts tape) then
# misses the coefficient's derivative by far more than `TOL`.
const _KAPPA = 1e4
function _build_mf_overwrite_coef(policy)
    W = gridspace(_mesh(policy))
    B = _basis(W)
    B2 = reverse(B; dims = 1)
    function f(p)
        κ = _load(W, _KAPPA .* B, p)
        x = B2 * p
        op = matrix_free_operator(form(W, W,
            (U, V) -> innerₕ(κ * U, V) + inner₊(∇ₕ(U), ∇ₕ(V)) + innerₕ(U, D₋ᵧ(V))))
        y = zeros(eltype(p), size(op, 1))
        mul!(y, op, x)
        parent(κ) .= 0
        return sum(abs2, y)
    end
    return f, _start()
end

# The per-unit matrix-free walk, which an operator without a fused plan takes: an action
# sink through the bilinear replay hooks, banded on the N x N mesh and coloured on the thin
# one. No public operator lacks a plan on one mesh, so the walk is called directly.
function _unit_action(M)
    W = gridspace(M)
    B = _basis(W)
    a = _nonsymmetric(W)
    policy = Bramble.execution_policy(W)
    function f(p)
        y = zeros(eltype(p), ndofs(W))
        x = parent(_load(W, B, p))
        Bramble._mf_apply_parts!(policy, Bramble.ActionSink(y, x, one(eltype(p)), nothing), W,
            W, a.ast)
        return sum(abs2, y)
    end
    return f, _start()
end
_build_unit_band(policy) = _unit_action(_mesh(policy))
_build_unit_colour(policy) = _unit_action(_thin_mesh(policy))

# The per-unit band walk with `x` overwritten after it, the sink's `α` a parameter: the
# derivative in `α` reads `x`, which the reverse must take from the rule's own copy.
function _build_unit_overwrite_x(policy)
    W = gridspace(_mesh(policy))
    B = _basis(W)
    a = form(W, W, (U, V) -> inner₊(∇ₕ(U), ∇ₕ(V)) + innerₕ(U, D₋ᵧ(V)))
    function f(p)
        y = zeros(eltype(p), ndofs(W))
        x = B * p
        s = Bramble.ActionSink(y, x, p[1], nothing)
        Bramble._mf_apply_parts!(policy, s, W, W, a.ast)
        x .= 0
        return sum(abs2, y)
    end
    return f, _start()
end

# `_batch_mf_bands!` with a scalar parameter in the form, the operator built in `f`: its
# derivative sits in the AST itself, not in an array. With `overwrite`, `x` (a function of
# the parameters too) is overwritten after the product.
function _mf_scalar(policy, overwrite::Bool)
    W = gridspace(_mesh(policy))
    B = _basis(W)
    function f(p)
        op = matrix_free_operator(
            form(W, W, (U, V) -> p[1] * inner₊(∇ₕ(U), ∇ₕ(V)) + innerₕ(U, D₋ᵧ(V))))
        x = B * p
        y = zeros(eltype(p), size(op, 1))
        mul!(y, op, x)
        overwrite && (x .= 0)
        return sum(abs2, y)
    end
    return f, _start()
end
_build_mf_scalar_param(policy) = _mf_scalar(policy, false)
_build_mf_overwrite_x(policy) = _mf_scalar(policy, true)

# `_batch_kron_lines!` with the coefficients and `β` computed from the parameters: the
# five-argument `mul!`, `α K x + β y`.
function _build_kron_scaled(policy)
    W = gridspace(_mesh(policy))
    B = _basis(W)
    K = kronecker_operator(_nonsymmetric(W))
    w = parent(_fixed(W))
    function f(p)
        y = w .* p[4]
        mul!(y, K, parent(_load(W, B, p)), p[1]^2, p[2])
        return sum(abs2, y)
    end
    return f, _start()
end

# `_batch_csr_spmv!` with the matrix values computed from the parameters, through the
# right-hand side's product on the pattern `semidiscretize_rhs` builds.
function _build_csr_values(policy)
    W = gridspace(_mesh(policy))
    B = _basis(W)
    A0 = assemble(_nonsymmetric(W))
    csr = Bramble._rhs_csr(policy, Val(false), A0)
    C = randn(MersenneTwister(SEED + 1), nnz(A0), NPARAM)
    function f(p)
        A = SparseMatrixCSC(size(A0)..., A0.colptr, A0.rowval, nonzeros(A0) .* (C * p))
        du = zeros(eltype(p), ndofs(W))
        Bramble._rhs_spmv!(du, A, csr, parent(_load(W, B, p)))
        return sum(abs2, du)
    end
    return f, _start()
end

const CAPTURES = (
    (hook = "_batch_for_const_capture", kind = "closure", build = _build_const_capture,
        rule = "_batch_for!"),
    (hook = "_batch_for_mixed_capture", kind = "closure", build = _build_mixed_capture,
        rule = "_batch_for!"),
    (hook = "_batch_for_reads_v", kind = "closure", build = _build_reads_v,
        rule = "_batch_for!"),
    (hook = "_batch_mf_bands_coef", kind = "closure", build = _build_mf_coef,
        rule = "_batch_mf_bands!"),
    (hook = "_batch_mf_unit_band", kind = "closure", build = _build_unit_band,
        rule = "_batch_bilinear_band_replay!"),
    (hook = "_batch_mf_unit_colour", kind = "closure", build = _build_unit_colour,
        rule = "_batch_bilinear_colour_replay!"),
    (hook = "_batch_mf_scalar_param", kind = "closure", build = _build_mf_scalar_param,
        rule = "_batch_mf_bands!"),
    (hook = "_batch_mf_overwrite_x", kind = "closure", build = _build_mf_overwrite_x,
        rule = "_batch_mf_bands!"),
    (hook = "_batch_mf_unit_overwrite_x", kind = "closure", build = _build_unit_overwrite_x,
        rule = "_batch_bilinear_band_replay!"),
    (hook = "_batch_kron_scaled", kind = "fixed", build = _build_kron_scaled,
        rule = "_batch_kron_lines!"),
    (hook = "_batch_csr_values", kind = "fixed", build = _build_csr_values,
        rule = "_batch_csr_spmv!"),
    (hook = "_batch_mf_overwrite_coef", kind = "closure", build = _build_mf_overwrite_coef,
        rule = "_batch_mf_bands!")
)

function case(hook::AbstractString)
    for c in (CASES..., THREADED..., CAPTURES...)
        c.hook == hook && return c
    end
    throw(ArgumentError("no Enzyme case for hook \"$hook\""))
end

_clean(text) = first(filter(isprint, text), ERROR_CHARS)

# The first line of an Enzyme error, then where it failed: the method named after "Failure
# within method:", the custom-rule complaint, and the first frame of its stack. A long
# message is cut at `ERROR_CHARS` characters otherwise inside boilerplate.
function _summary(err)
    lines = filter(!isempty, strip.(split(sprint(showerror, err), '\n')))
    parts = [first(lines)]
    i = findfirst(==("Failure within method:"), lines)
    i === nothing || push!(parts, "in " * first(split(lines[i + 1], "(")))
    i = findfirst(startswith("Custom rule"), lines)
    i === nothing || push!(parts, lines[i])
    i = findfirst(==("Stacktrace:"), lines)
    i === nothing || i + 2 > length(lines) ||
        push!(parts, "at " * join(lines[(i + 1):(i + 2)], " "))
    return _clean(join(parts, " | "))
end

_relerr(g, ref) = maximum(abs, g .- ref) / max(maximum(abs, ref), floatmin(Float64))

function _di(mode::Symbol)
    AutoEnzyme(;
        mode = Enzyme.set_runtime_activity(mode === :forward ? Enzyme.Forward : Enzyme.Reverse),
        function_annotation = Enzyme.Const)
end

function _gradient(f, x, mode::Symbol)
    mode === :reverse && return Enzyme.gradient(
        Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(f), x)[1]
    return DI.gradient(f, _di(mode), x)
end

# Distinct tangents of the parameters, one per lane of `:batch_forward`.
_batch_tangents(x) = ntuple(k -> randn(MersenneTwister(SEED + k), length(x)), BATCH_WIDTH)

# Each lane's derivative and the oracle's value for it: a directional derivative of `f` for
# `:batch_forward`, a cotangent times the gradient for `:batch_reverse`.
function _batch(f, x, ref, mode::Symbol)
    if mode === :batch_forward
        seeds = _batch_tangents(x)
        r = Enzyme.autodiff(Enzyme.set_runtime_activity(Enzyme.Forward), Enzyme.Const(f),
            Enzyme.BatchDuplicated(copy(x), seeds))
        return [[d] for d in values(first(r))], [[dot(ref, s)] for s in seeds]
    end
    g!(y, p) = (y[1] = f(p); nothing)
    dys = ntuple(k -> [BATCH_COTANGENTS[k]], BATCH_WIDTH)
    dxs = ntuple(_ -> zero(x), BATCH_WIDTH)
    Enzyme.autodiff(Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(g!),
        Enzyme.Const, Enzyme.BatchDuplicated(zeros(1), dys),
        Enzyme.BatchDuplicated(copy(x), dxs))
    return collect(dxs), [c .* ref for c in BATCH_COTANGENTS]
end

# How many times a rule for `hook` has been entered, summed over the Bramble extensions that
# count their rules' entries.
function _rule_entries(hook::AbstractString)
    n = 0
    for name in (:BrambleEnzymePolyesterExt, :BrambleEnzymeExt)
        ext = Base.get_extension(Bramble, name)
        if ext !== nothing && isdefined(ext, :_rule_entries)
            n += ext._rule_entries(Symbol(hook))
        end
    end
    return n
end

"""
    evaluate(c; policy, mode) -> NamedTuple

Run one mode of the case `c` under `policy` and compare its derivative with the ForwardDiff
oracle, lane by lane for a batch mode. Returns `status` (`"ok"` within `TOL`, `"wrong"`
otherwise, `"throws"` when the mode raises, `"bypassed"` when a batch mode under a threaded
policy never entered the hook's rule), `relerr` (the worst lane's) and the first `error` text.
"""
function evaluate(c; policy, mode::Symbol)
    modes = (MODES..., BATCH_MODES...)
    mode in modes || throw(ArgumentError("mode must be one of $modes, got :$mode"))
    f, x = c.build(policy)
    ref = DI.gradient(f, AutoForwardDiff(), x)
    rule = get(c, :rule, c.hook)
    entries = _rule_entries(rule)
    got, want = try
        mode in BATCH_MODES ? _batch(f, x, ref, mode) : ([_gradient(f, x, mode)], [ref])
    catch err
        err isa InterruptException && rethrow()
        return (status = "throws", relerr = NaN, error = _summary(err))
    end
    r = maximum(_relerr(g, w) for (g, w) in zip(got, want))
    entered = !(mode in BATCH_MODES) || policy isa CpuSerial || _rule_entries(rule) > entries
    status = !(isfinite(r) && r <= TOL) ? "wrong" : entered ? "ok" : "bypassed"
    return (status = status, relerr = r, error = "")
end

_policy(p::Symbol) = getfield(Bramble, p)()
_policy(p) = p

"""
    check(hook; policy = CpuPolyester(), modes = MODES) -> Nothing

Throw unless every mode of the case for `hook` returns `"ok"` under `policy`. `hook` is a
string or a `Symbol`; `policy` an execution policy or the `Symbol` naming one.
"""
function check(hook::AbstractString; policy = CpuPolyester(), modes = MODES)
    c = case(hook)
    p = _policy(policy)
    for mode in modes
        r = evaluate(c; policy = p, mode)
        r.status == "ok" || error("$hook under $(nameof(typeof(p))), mode :$mode: " *
              "$(r.status), relerr $(r.relerr). $(r.error)")
    end
    return nothing
end
check(hook::Symbol; kwargs...) = check(String(hook); kwargs...)
# An entry of `CASES`, `THREADED` or `CAPTURES` itself.
check(c::NamedTuple; kwargs...) = check(c.hook; kwargs...)

end # module
