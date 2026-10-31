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
# Each builder keeps the parameters small (`NPARAM`) and expands them into the field with a
# seeded basis, so a builder's own work is not a stack of other hooks.

module EnzymePolicyCases

using Bramble
using Bramble: CpuPolyester, CpuSerial, D₋ₓ, D₋ᵧ, Mₓ, Mcₓ, S₊, semidiscretize_rhs
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
const TOL = 1e-10
const N = 41
const NPARAM = 4
const SEED = 1234
const ERROR_CHARS = 400

# A non-uniform N x N mesh on the unit square, on `policy`.
function _mesh(policy)
    return mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (N, N), (false, false);
        backend = backend(policy = policy))
end

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
function _thin_mesh(policy)
    mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (N, 3), (false, false);
        backend = backend(policy = policy))
end

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

# `_batch_average_engine!`, through `Mₓ`.
function _build_average(policy)
    W = gridspace(_mesh(policy))
    B = _basis(W)
    f(p) = sum(abs2, parent(Mₓ(_load(W, B, p))))
    return f, _start()
end

# `_batch_centered_average_engine!`, through `Mcₓ`.
function _build_centered(policy)
    W = gridspace(_mesh(policy))
    B = _basis(W)
    f(p) = sum(abs2, parent(Mcₓ(_load(W, B, p))))
    return f, _start()
end

# `_batch_run_bands!`, through the forward shift in direction 1 (`_shift_engine!`).
function _build_run_bands(policy)
    W = gridspace(_mesh(policy))
    B = _basis(W)
    f(p) = sum(abs2, parent(S₊(_load(W, B, p), 1)))
    return f, _start()
end

# `_batch_mf_bands!`: `mul!` with a matrix-free operator.
function _build_mf(policy)
    W = gridspace(_mesh(policy))
    B = _basis(W)
    op = matrix_free_operator(form(W, W, (U, V) -> inner₊(∇ₕ(U), ∇ₕ(V))))
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
    K = kronecker_operator(form(W, W, (U, V) -> inner₊(∇ₕ(U), ∇ₕ(V))))
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
    sd = semidiscretize(form(W, W, (U, V) -> inner₊(∇ₕ(U), ∇ₕ(V))), form(W, v -> innerₕ(w, v)))
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
    (hook = "_batch_run_bands!", kind = "fixed", build = _build_run_bands),
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

function case(hook::AbstractString)
    for c in CASES
        c.hook == hook && return c
    end
    for c in THREADED
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

"""
    evaluate(c; policy, mode) -> NamedTuple

Run one mode of the case `c` under `policy` and compare its gradient with the ForwardDiff
oracle. Returns `status` (`"ok"` within `TOL`, `"wrong"` otherwise, `"throws"` when the mode
raises), `relerr` and the first `error` text.
"""
function evaluate(c; policy, mode::Symbol)
    mode in MODES || throw(ArgumentError("mode must be one of $MODES, got :$mode"))
    f, x = c.build(policy)
    ref = DI.gradient(f, AutoForwardDiff(), x)
    g = try
        _gradient(f, x, mode)
    catch err
        err isa InterruptException && rethrow()
        return (status = "throws", relerr = NaN, error = _summary(err))
    end
    r = _relerr(g, ref)
    return (status = isfinite(r) && r <= TOL ? "ok" : "wrong", relerr = r, error = "")
end

"""
    check(hook; policy = CpuPolyester(), modes = MODES) -> Nothing

Throw unless every mode of the case for `hook` returns `"ok"` under `policy`.
"""
function check(hook::AbstractString; policy = CpuPolyester(), modes = MODES)
    c = case(hook)
    for mode in modes
        r = evaluate(c; policy, mode)
        r.status == "ok" || error("$hook under $(nameof(typeof(policy))), mode :$mode: " *
              "$(r.status), relerr $(r.relerr). $(r.error)")
    end
    return nothing
end

end # module
