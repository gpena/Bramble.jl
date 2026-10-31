# DifferentiationInterface against the direct backend call (gpena/Bramble.jl#1117,
# milestone v4.2.0)
#
# Part 1 (ad_backend_evaluation.jl) measured every backend through DifferentiationInterface
# (DI), as a user calls it. This script asks what DI costs and what it cannot express, for the
# two backends that survive part 1, ForwardDiff and Enzyme: each cell runs the DI call and the
# equivalent direct call, on the same problem, and says which side fails when one does. Run by
# hand, never a benchmark-suite entry (Enzyme is neither a test nor a benchmark dependency):
#
#     .claude/scripts/julia_slot.sh 2 julia --startup-file=no --threads=1 \
#         benchmark/ad_di_vs_direct.jl --smoke
#     .claude/scripts/julia_slot.sh 4 julia --startup-file=no --threads=1 \
#         benchmark/ad_di_vs_direct.jl --full
#
# The outer `julia_slot.sh` reserves the threads the cells use (2 for `--smoke`, 4 for
# `--full`). This process only schedules: every cell runs in child `julia --threads=N`
# processes at the default optimisation level, one backend loaded in each. Part 1's script is
# included, never copied or edited: its seeded non-uniform mesh, the two workloads, the
# `make_backend` configurations, the environment build (`BRAMBLE_AD_EVAL_ENV` is honoured) and
# the timeout and crash handling are its own, and `meta.base_script_sha` records the version
# reused.
#
# Timing cells (`[[timing]]`): `dense_gradient` and `adjoint_solve` with ForwardDiff and
# Enzyme under CpuSerial at several `np`, and under CpuThreaded and CpuPolyester at `np = 16`.
# The DI call is part 1's, `DI.prepare_gradient` then `DI.gradient(f, prep, ad, x)`. The
# direct call is its equivalent with the same setup: ForwardDiff's `GradientConfig` with the
# chunk DI's preparation chose, then `ForwardDiff.gradient!`; Enzyme's `autodiff` in reverse
# mode with `set_runtime_activity`, `Const(f)`, `Active` and a zeroed `Duplicated` shadow.
# Each side's status and first-call time (preparation plus first call, compilation included)
# come from its own fresh child, so one side's compiled code never helps the other and a crash
# on one side hides nothing on the other. A CpuSerial cell whose sides are both `ok` then gets a
# shared child that interleaves the two prepared calls, so compile placement and machine state
# are common to both. The policy cells record statuses, values and first-call times only.
#
# Capability cells (`[[capability]]`): one DI spelling and, where a direct one exists, the
# direct one, for each use Bramble makes of an AD backend that DI might not express.
#
# Every cell has a unique `key`, a `status_di` and `status_direct` in ok, wrong, throws,
# crash, unsupported (and `n/a` for a capability with no direct spelling), the error text of
# each failing side, and an `attribution`:
#
#   none               both sides ok
#   di                 DI throws, crashes or is wrong where the direct call is ok
#   di_cannot_express  DI is unsupported where the direct call is ok: a design limit
#   direct             only DI is ok
#   backend            both fail
#   n/a                a capability with no direct spelling

using TOML
using Dates

# Part 1's script, as a module of its own so its names stay out of this file's.
module Base1
include(joinpath(@__DIR__, "ad_backend_evaluation.jl"))
end

const ROOT = Base1.ROOT
const SCRIPT = @__FILE__
const WRONG_TOL = Base1.WRONG_TOL
const LoadFailure = Base1.LoadFailure
const BACKENDS = ("ForwardDiff", "Enzyme")

const SETTINGS = (
    smoke = (
        threads = 2, dense_nps = (2, 16), adjoint_nps = (4, 16), policy_np = 16,
        m_dense = 10, m_adjoint = 10, m_policy = Base1.SETTINGS.smoke.m_policy,
        rounds = 5, timeout = 900),
    full = (
        threads = 4, dense_nps = (2, 4, 16, 64, 256, 1024), adjoint_nps = (4, 64, 1024),
        policy_np = 16, m_dense = 34, m_adjoint = 34, m_policy = Base1.SETTINGS.full.m_policy,
        rounds = 9, timeout = 3600)
)

# The capability problems are small and identical in both modes.
const CAP_NP = 8
const CAP_M = 10
const CAP_M_SPARSE = Base1.SETTINGS.smoke.m_sparse

# case, backend, whether a direct spelling exists, what the cell compares.
const CAPABILITIES = (
    ("constant_context", "ForwardDiff", true,
        "DI.gradient(f, ad, x, DI.Constant(c)) against ForwardDiff.gradient of the " *
        "closure x -> f(x, c), on the dense-gradient problem scaled by c."),
    ("constant_context", "Enzyme", true,
        "DI.gradient(f, ad, x, DI.Constant(c)) against Enzyme.autodiff of the closure " *
        "x -> f(x, c) held in Const, with set_runtime_activity, on the dense-gradient " *
        "problem scaled by c."),
    ("chunk_size", "ForwardDiff", true,
        "AutoForwardDiff(chunksize = 4) against GradientConfig with ForwardDiff.Chunk{4}; " *
        "the DI preparation's own chunk is checked to be 4."),
    ("enzyme_forward", "Enzyme", true,
        "AutoEnzyme(mode = set_runtime_activity(Forward), function_annotation = Const) " *
        "against Enzyme.gradient with the same mode and Const(f)."),
    ("enzyme_reverse", "Enzyme", true,
        "AutoEnzyme(mode = set_runtime_activity(Reverse), function_annotation = Const) " *
        "against Enzyme.gradient with the same mode and Const(f)."),
    ("enzyme_no_runtime_activity_di", "Enzyme", true,
        "The Enzyme call without set_runtime_activity, both spellings run: does DI still " *
        "need it? AutoEnzyme(mode = Reverse, function_annotation = Const) against " *
        "Enzyme.autodiff(Reverse, Const(f), Active, Duplicated(x, dx)). Read status_di."),
    ("enzyme_no_runtime_activity_direct", "Enzyme", true,
        "The same two calls as enzyme_no_runtime_activity_di: does the direct call still " *
        "need set_runtime_activity? Read status_direct."),
    ("hessian", "ForwardDiff", true,
        "DI.hessian with AutoForwardDiff() against ForwardDiff.hessian, on the closed-form " *
        "manufactured solution sin(pi x1) sin(2 pi x2), the helper shape of " *
        "docs/src/examples/elasticity_3d.jl."),
    ("derivative", "ForwardDiff", true,
        "DI.derivative with AutoForwardDiff() against ForwardDiff.derivative, on the " *
        "closed-form t sin(2 pi t)."),
    ("two_active_arguments", "Enzyme", true,
        "Enzyme gradient of J(A, F) = sum(abs2, pde_solve(A, F)) with respect to both A " *
        "and F, direct with two Duplicated arguments (the shape of " *
        "test/ext/chainrules_enzyme_ext.jl). DI differentiates one argument: the DI " *
        "spelling makes F active and A a DI.Constant, so it returns the gradient with " *
        "respect to F only, never the one with respect to A. value_diff compares the F part."),
    ("sparse_jacobian_cpu_threaded", "ForwardDiff", false,
        "Part 1's sparse_jacobian workload at its smoke size through " *
        "AutoSparse(AutoForwardDiff()) under CpuThreaded, against the colored central " *
        "FiniteDiff Jacobian. No direct route exists. Part 1 ran it under CpuSerial only."),
    ("sparse_jacobian_cpu_polyester", "ForwardDiff", false,
        "Part 1's sparse_jacobian workload at its smoke size through " *
        "AutoSparse(AutoForwardDiff()) under CpuPolyester, against the colored central " *
        "FiniteDiff Jacobian. No direct route exists. Part 1 ran it under CpuSerial only.")
)

# ---------------------------------------------------------------------------------------
# What a child evaluates after part 1's `CHILD_CODE`, in the same module: the direct calls,
# the timing rounds and the capability spellings.
# ---------------------------------------------------------------------------------------

const SIDE_CODE = quote
    relerr(a::AbstractMatrix, ref) = maximum(abs, a .- ref) / maximum(abs, ref)

    # The problem a workload cell differentiates, and the Float64 problem its reference
    # differentiates: always on a serial mesh, so the reference does not depend on the policy.
    function problem(workload, backend_name, policy, np, m)
        pol = getfield(POLICIES, Symbol(policy))()
        if workload == "dense_gradient"
            return dense_problem(pol, m, np), dense_problem(Serial(), m, np)
        elseif workload == "adjoint_solve"
            route = backend_name == "ForwardDiff" ? Val(:sparspak) : Val(:native)
            return adjoint_problem(pol, m, np, route),
            adjoint_problem(Serial(), m, np, Val(:plain))
        end
        error("unknown workload $workload")
    end

    # A prepared direct call: it returns the gradient, freshly written. `chunk` is the
    # chunk DI's preparation chose, or 0 when none is known, which takes ForwardDiff's default.
    function direct_prepare(backend_name, f, x, chunk)
        if backend_name == "ForwardDiff"
            ch = chunk > 0 ? ForwardDiff.Chunk{chunk}() : ForwardDiff.Chunk(x)
            cfg = ForwardDiff.GradientConfig(f, x, ch)
            g = similar(x)
            return () -> ForwardDiff.gradient!(g, f, x, cfg)
        elseif backend_name == "Enzyme"
            dx = zero(x)
            mode = Enzyme.set_runtime_activity(Enzyme.Reverse)
            return function ()
                fill!(dx, 0)
                Enzyme.autodiff(mode, Enzyme.Const(f), Enzyme.Active, Enzyme.Duplicated(x, dx))
                return dx
            end
        end
        error("unknown backend $backend_name")
    end

    # The first call, through `invokelatest` as in part 1, so the clock includes whatever
    # the backend builds when the call is first dispatched.
    function first_call_cell(note_n, side, workload, backend_name, policy, np, m, chunk)
        prob, refprob = problem(workload, backend_name, policy, np, m)
        note_n(prob.n)
        f, x = prob.f, prob.x
        ref = FiniteDiff.finite_difference_gradient(refprob.f, x, Val(:central))
        k = 0
        local g
        if side == "di"
            ad = make_backend(backend_name)
            t = @elapsed begin
                prep = Base.invokelatest(DI.prepare_gradient, f, ad, x)
                g = Base.invokelatest(DI.gradient, f, prep, ad, x)
            end
            backend_name == "ForwardDiff" && (k = ForwardDiff.chunksize(prep.config))
        else
            t = @elapsed begin
                call = Base.invokelatest(direct_prepare, backend_name, f, x, chunk)
                g = Base.invokelatest(call)
            end
        end
        return Dict{String, Any}(
            "relerr" => Float64(relerr(g, ref)), "first_call_s" => t, "chunk" => k,
            "value" => collect(Float64, g))
    end

    median_of(v) = (s = sort(v); n = length(s); isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2)

    best_of(call, reps) = minimum(_ -> @elapsed(call()), 1:reps)

    # Interleaved rounds of prepared calls, each round the best of 3 per side, the side that
    # goes first alternating from round to round.
    function timing_cell(note_n, workload, backend_name, policy, np, m, rounds)
        prob, _ = problem(workload, backend_name, policy, np, m)
        note_n(prob.n)
        f, x = prob.f, prob.x
        ad = make_backend(backend_name)
        prep = Base.invokelatest(DI.prepare_gradient, f, ad, x)
        k = backend_name == "ForwardDiff" ? ForwardDiff.chunksize(prep.config) : 0
        di_call = () -> DI.gradient(f, prep, ad, x)
        direct_call = Base.invokelatest(direct_prepare, backend_name, f, x, k)
        Base.invokelatest(di_call)
        Base.invokelatest(direct_call)
        t_di, t_direct = Float64[], Float64[]
        for r in 1:rounds
            if isodd(r)
                push!(t_di, best_of(di_call, 3))
                push!(t_direct, best_of(direct_call, 3))
            else
                push!(t_direct, best_of(direct_call, 3))
                push!(t_di, best_of(di_call, 3))
            end
        end
        return Dict{String, Any}(
            "rounds" => rounds, "per_call_di_min_s" => minimum(t_di),
            "per_call_direct_min_s" => minimum(t_direct),
            "per_call_di_median_s" => median_of(t_di),
            "per_call_direct_median_s" => median_of(t_direct),
            "allocs_di_bytes" => Int(@allocated(di_call())),
            "allocs_direct_bytes" => Int(@allocated(direct_call())))
    end

    # ---- capabilities ----
    # Each case returns `(di = ..., direct = ...)`, zero-argument functions that return
    # `(value, reference, shared)`: the spelling's result, what it is checked against, and
    # the array `value_diff` compares. `direct` is `nothing` where no direct spelling exists.

    function cap_dense()
        prob = dense_problem(Serial(), $CAP_M, $CAP_NP)
        return prob.f, prob.x
    end

    function central_gradient(f, x)
        return FiniteDiff.finite_difference_gradient(f, x, Val(:central))
    end

    function cap_enzyme_dense(f, x, mode)
        dx = zero(x)
        Enzyme.autodiff(mode, Enzyme.Const(f), Enzyme.Active, Enzyme.Duplicated(x, dx))
        return dx
    end

    function cap_constant_context(backend_name)
        f, x = cap_dense()
        c = 1.7
        g = (x, c) -> c * f(x)
        h = x -> c * f(x)
        ref = central_gradient(h, x)
        ad = make_backend(backend_name)
        di = () -> (v = DI.gradient(g, ad, x, DI.Constant(c)); (v, ref, v))
        direct = if backend_name == "ForwardDiff"
            () -> (v = ForwardDiff.gradient(h, x); (v, ref, v))
        else
            mode = Enzyme.set_runtime_activity(Enzyme.Reverse)
            () -> (v = cap_enzyme_dense(h, x, mode); (v, ref, v))
        end
        return (di = di, direct = direct)
    end

    function cap_chunk_size()
        f, x = cap_dense()
        ref = central_gradient(f, x)
        ad = AutoForwardDiff(chunksize = 4)
        function di()
            prep = DI.prepare_gradient(f, ad, x)
            ForwardDiff.chunksize(prep.config) == 4 ||
                error("the preparation's chunk is $(ForwardDiff.chunksize(prep.config)), not 4")
            v = DI.gradient(f, prep, ad, x)
            return v, ref, v
        end
        function direct()
            cfg = ForwardDiff.GradientConfig(f, x, ForwardDiff.Chunk{4}())
            v = ForwardDiff.gradient(f, x, cfg)
            return v, ref, v
        end
        return (di = di, direct = direct)
    end

    function cap_enzyme_mode(mode)
        f, x = cap_dense()
        ref = central_gradient(f, x)
        ad = AutoEnzyme(mode = mode, function_annotation = Enzyme.Const)
        di = () -> (v = DI.gradient(f, ad, x); (v, ref, v))
        direct = () -> (v = Enzyme.gradient(mode, Enzyme.Const(f), x)[1]; (v, ref, v))
        return (di = di, direct = direct)
    end

    function cap_no_runtime_activity()
        f, x = cap_dense()
        ref = central_gradient(f, x)
        ad = AutoEnzyme(mode = Enzyme.Reverse, function_annotation = Enzyme.Const)
        di = () -> (v = DI.gradient(f, ad, x); (v, ref, v))
        direct = () -> (v = cap_enzyme_dense(f, x, Enzyme.Reverse); (v, ref, v))
        return (di = di, direct = direct)
    end

    function cap_hessian()
        u(x) = sin(π * x[1]) * sin(2π * x[2])
        x = [0.3, 0.6]
        s1, c1, s2, c2 = sin(π * x[1]), cos(π * x[1]), sin(2π * x[2]), cos(2π * x[2])
        h11, h12, h22 = -π^2 * s1 * s2, 2π^2 * c1 * c2, -4π^2 * s1 * s2
        ref = [h11 h12; h12 h22]
        di = () -> (H = DI.hessian(u, AutoForwardDiff(), x); (H, ref, vec(H)))
        direct = () -> (H = ForwardDiff.hessian(u, x); (H, ref, vec(H)))
        return (di = di, direct = direct)
    end

    function cap_derivative()
        v(t) = t * sin(2π * t)
        t0 = 0.37
        ref = [sin(2π * t0) + 2π * t0 * cos(2π * t0)]
        di = () -> (d = [DI.derivative(v, AutoForwardDiff(), t0)]; (d, ref, d))
        direct = () -> (d = [ForwardDiff.derivative(v, t0)]; (d, ref, d))
        return (di = di, direct = direct)
    end

    function cap_two_active()
        Random.seed!(1234)
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 21, false)
        Wₕ = gridspace(Ωₕ)
        a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
        fₕ = Rₕ(Wₕ, x -> pi^2 * sinpi(x[1]))
        l = form(Wₕ, v -> innerₕ(fₕ, v))
        A, F = assemble(a, l; dirichlet = :boundary => x -> 0.7)
        J(A, F) = sum(abs2, pde_solve(A, F))
        # The adjoint in closed form: u = A \ F, λ = A' \ 2u, dJ/dF = λ, dJ/dA = -λ u'.
        u = A \ F
        λ = A' \ (2 .* u)
        rows, cols = findnz(A)
        dA_ref = [-λ[i] * u[j] for (i, j) in zip(rows, cols)]
        function direct()
            dA = SparseMatrixCSC(
                size(A, 1), size(A, 2), copy(A.colptr), copy(A.rowval), zeros(nnz(A)))
            dF = zero(F)
            mode = Enzyme.set_runtime_activity(Enzyme.Reverse)
            Enzyme.autodiff(mode, Enzyme.Const(J), Enzyme.Active,
                Enzyme.Duplicated(A, dA), Enzyme.Duplicated(F, dF))
            return vcat(dA.nzval, dF), vcat(dA_ref, λ), dF
        end
        function di()
            ad = make_backend("Enzyme")
            dF = DI.gradient((F, A) -> J(A, F), ad, F, DI.Constant(A))
            err = relerr(dF, λ)
            why = "DI differentiates one argument: A and F cannot both be active. With F " *
                  "active and A a DI.Constant the gradient with respect to F alone is " *
                  "returned (relerr $(round(err; sigdigits = 3)) against the adjoint); the " *
                  "gradient with respect to A is not available that way."
            return dF, λ, dF, why
        end
        return (di = di, direct = direct)
    end

    function cap_sparse(policy)
        function di()
            pol = getfield(POLICIES, Symbol(policy))()
            prob = sparse_problem(pol, $CAP_M_SPARSE)
            refprob = sparse_problem(Serial(), $CAP_M_SPARSE)
            ad = sparse_wrap(make_backend("ForwardDiff"), prob.detector)
            ref_ad = sparse_wrap(AutoFiniteDiff(fdtype = Val(:central)), refprob.detector)
            ref = DI.jacobian(refprob.f, ref_ad, refprob.x)
            prep = DI.prepare_jacobian(prob.f, ad, prob.x)
            J = DI.jacobian(prob.f, prep, ad, prob.x)
            return J, ref, Float64[]
        end
        return (di = di, direct = nothing)
    end

    function cap_sides(case, backend_name)
        case == "constant_context" && return cap_constant_context(backend_name)
        case == "chunk_size" && return cap_chunk_size()
        case == "enzyme_forward" &&
            return cap_enzyme_mode(Enzyme.set_runtime_activity(Enzyme.Forward))
        case == "enzyme_reverse" &&
            return cap_enzyme_mode(Enzyme.set_runtime_activity(Enzyme.Reverse))
        startswith(case, "enzyme_no_runtime_activity") && return cap_no_runtime_activity()
        case == "hessian" && return cap_hessian()
        case == "derivative" && return cap_derivative()
        case == "two_active_arguments" && return cap_two_active()
        case == "sparse_jacobian_cpu_threaded" && return cap_sparse("CpuThreaded")
        case == "sparse_jacobian_cpu_polyester" && return cap_sparse("CpuPolyester")
        error("unknown capability $case")
    end

    function capability_cell(note_n, side, case, backend_name)
        sides = cap_sides(case, backend_name)
        call = side == "di" ? sides.di : sides.direct
        value, ref, shared, why... = Base.invokelatest(call)
        cell = Dict{String, Any}(
            "relerr" => Float64(relerr(value, ref)), "value" => collect(Float64, shared))
        # A spelling that runs but cannot express the whole question says so in a fourth item.
        isempty(why) || (cell["unsupported"] = only(why))
        return cell
    end
end

# ---------------------------------------------------------------------------------------
# One child process: runs one entry point of `SIDE_CODE` and writes its cell.
# ---------------------------------------------------------------------------------------

clean(text) = Base1.clean(text)

error_text(e) = Base1.error_text(e)

function run_child(out, cell, backend_name, entry, args...)
    # Written first, so a child that dies leaves the cell's identity behind.
    Base1.write_cell(out, cell)
    mod = Module(:AdDiCell)
    function note_n(n)
        cell["n"] = n
        Base1.write_cell(out, cell)
    end
    try
        try
            Core.eval(mod, :(import $(Symbol(backend_name))))
            Core.eval(mod, Base1.CHILD_CODE)
            Core.eval(mod, SIDE_CODE)
        catch e
            throw(LoadFailure(e))
        end
        fn = Base.invokelatest(getfield, mod, entry)
        merge!(cell, Base.invokelatest(fn, note_n, args...))
        err = get(cell, "relerr", 0.0)
        if haskey(cell, "unsupported")
            cell["status"] = "unsupported"
            cell["error"] = clean(pop!(cell, "unsupported"))
        elseif isfinite(err) && err < WRONG_TOL
            cell["status"] = "ok"
        else
            cell["status"] = "wrong"
            cell["error"] = "relerr = $err against the reference, tolerance $WRONG_TOL"
        end
    catch e
        e isa InterruptException && rethrow()
        cell["status"] = "throws"
        cell["error"] = e isa LoadFailure ? "load: " * error_text(e.err) : error_text(e)
    end
    Base1.write_cell(out, cell)
    return nothing
end

function blank_cell()
    return Dict{String, Any}(
        "status" => "running", "error" => "", "relerr" => NaN, "first_call_s" => NaN,
        "chunk" => 0, "value" => Float64[])
end

function child_main(args)
    kind, out = args[1], args[end]
    cell = blank_cell()
    if kind == "first"
        side, workload, backend_name, policy, np, m, chunk = args[2:8]
        return run_child(out, cell, backend_name, :first_call_cell, side, workload,
            backend_name, policy, parse(Int, np), parse(Int, m), parse(Int, chunk))
    elseif kind == "timing"
        workload, backend_name, policy, np, m, rounds = args[2:7]
        cell["relerr"] = 0.0
        return run_child(out, cell, backend_name, :timing_cell, workload, backend_name,
            policy, parse(Int, np), parse(Int, m), parse(Int, rounds))
    elseif kind == "capability"
        side, case, backend_name = args[2:4]
        return run_child(out, cell, backend_name, :capability_cell, side, case, backend_name)
    end
    error("unknown child kind $kind")
end

# ---------------------------------------------------------------------------------------
# The scheduler.
# ---------------------------------------------------------------------------------------

# One child to completion. A child that never finished its cell, however it ended, is a
# crash.
function run_one(args, project, s, dir, tag)
    out = joinpath(dir, "$tag.toml")
    log = joinpath(dir, "$tag.log")
    cmd = Base1.julia_cmd(s.threads, project, SCRIPT, "--child", args..., out)
    state = Base1.run_limited(cmd, log, s.timeout)
    cell = isfile(out) ? TOML.parsefile(out) : blank_cell()
    if cell["status"] == "running"
        cell["status"] = "crash"
        why = state[1] == :timeout ? "timeout after $(s.timeout) s" :
              state[1] == :signal ? "killed by signal $(state[2])" :
              "exit code $(state[2])"
        cell["error"] = clean("$why: $(Base1.log_tail(log))")
    end
    return cell
end

function attribution(status_di, status_direct)
    status_direct == "n/a" && return "n/a"
    status_di == "ok" && return status_direct == "ok" ? "none" : "direct"
    status_direct != "ok" && return "backend"
    return status_di == "unsupported" ? "di_cannot_express" : "di"
end

# The max-norm relative difference of the two sides' values; NaN when either has none.
function value_diff(di, direct)
    a, b = di["value"], direct["value"]
    (isempty(a) || length(a) != length(b)) && return NaN
    return maximum(abs, a .- b) / maximum(abs, b)
end

function sides!(cell, di, direct)
    cell["status_di"] = di["status"]
    cell["error_di"] = di["error"]
    cell["relerr_di"] = di["relerr"]
    if direct === nothing
        cell["status_direct"] = "n/a"
        cell["error_direct"] = ""
    else
        cell["status_direct"] = direct["status"]
        cell["error_direct"] = direct["error"]
        cell["relerr_direct"] = direct["relerr"]
        cell["value_diff"] = value_diff(di, direct)
    end
    cell["attribution"] = attribution(cell["status_di"], cell["status_direct"])
    return cell
end

function timing_cell(spec, project, s, dir, index)
    id = (spec.workload, spec.backend, spec.policy, string(spec.np), string(spec.m))
    di = run_one(("first", "di", id..., "0"), project, s, dir, "t$index-di")
    chunk = di["chunk"]
    direct = run_one(("first", "direct", id..., string(chunk)), project, s, dir,
        "t$index-direct")
    cell = Dict{String, Any}(
        "key" => join((spec.workload, spec.backend, spec.policy, spec.np), "/"),
        "workload" => spec.workload, "backend" => spec.backend, "policy" => spec.policy,
        "parameters" => spec.np, "n" => get(di, "n", get(direct, "n", spec.m^2)),
        "chunk" => chunk, "first_call_di_s" => di["first_call_s"],
        "first_call_direct_s" => direct["first_call_s"], "rounds" => 0,
        "per_call_di_min_s" => NaN, "per_call_direct_min_s" => NaN,
        "per_call_di_median_s" => NaN, "per_call_direct_median_s" => NaN,
        "allocs_di_bytes" => -1, "allocs_direct_bytes" => -1, "error_timing" => "")
    sides!(cell, di, direct)
    cell["di_vs_direct"] = cell["value_diff"]
    delete!(cell, "value_diff")
    if cell["attribution"] == "none" && spec.policy == "CpuSerial"
        shared = run_one(("timing", id..., string(s.rounds)), project, s, dir,
            "t$index-timing")
        if shared["status"] == "ok"
            for k in ("rounds", "per_call_di_min_s", "per_call_direct_min_s",
                "per_call_di_median_s", "per_call_direct_median_s", "allocs_di_bytes",
                "allocs_direct_bytes")
                cell[k] = shared[k]
            end
        else
            cell["error_timing"] = "$(shared["status"]): $(shared["error"])"
        end
    end
    return cell
end

function capability_cell(spec, project, s, dir, index)
    case, backend, has_direct, note = spec
    id = (case, backend)
    di = run_one(("capability", "di", id...), project, s, dir, "c$index-di")
    direct = has_direct ?
             run_one(("capability", "direct", id...), project, s, dir, "c$index-direct") :
             nothing
    cell = Dict{String, Any}(
        "key" => "capability/$case/$backend", "case" => case, "backend" => backend,
        "policy" =>
            case == "sparse_jacobian_cpu_threaded" ? "CpuThreaded" :
            case == "sparse_jacobian_cpu_polyester" ? "CpuPolyester" : "CpuSerial",
        "note" => note)
    return sides!(cell, di, direct)
end

function cells_for(s)
    timing = NamedTuple[]
    for np in s.dense_nps, b in BACKENDS

        push!(timing, (workload = "dense_gradient", backend = b, policy = "CpuSerial",
            np = np, m = s.m_dense))
    end
    for np in s.adjoint_nps, b in BACKENDS

        push!(timing, (workload = "adjoint_solve", backend = b, policy = "CpuSerial",
            np = np, m = s.m_adjoint))
    end
    for policy in ("CpuThreaded", "CpuPolyester"), w in ("dense_gradient", "adjoint_solve"),
        b in BACKENDS
        push!(timing, (workload = w, backend = b, policy = policy, np = s.policy_np,
            m = s.m_policy))
    end
    return timing
end

# The versions an upstream issue needs, read from the environment's manifest.
function versions(project)
    manifest = joinpath(project, "Manifest.toml")
    deps = isfile(manifest) ? get(TOML.parsefile(manifest), "deps", Dict()) : Dict()
    out = Dict{String, Any}("julia" => string(VERSION))
    for name in ("DifferentiationInterface", "ForwardDiff", "Enzyme")
        out[name] = haskey(deps, name) ? string(get(deps[name][1], "version", "unknown")) :
                    "unknown"
    end
    return out
end

function meta(mode, s, project)
    return Dict{String, Any}(
        "mode" => mode, "threads" => s.threads, "uniform" => false, "rounds" => s.rounds,
        "julia" => string(VERSION), "commit" => Base1.git("rev-parse", "HEAD"),
        "date" => Dates.format(Dates.now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        "script_sha" => Base1.git("hash-object", SCRIPT),
        "base_script_sha" => Base1.git("hash-object", Base1.SCRIPT),
        "load_1min" => Sys.loadavg()[1], "cell_timeout_s" => s.timeout,
        "cpu" => string(Sys.cpu_info()[1].model), "os" => string(Sys.KERNEL),
        "arch" => string(Sys.ARCH), "versions" => versions(project))
end

function write_results(path, mode, s, project, timing, capability)
    mkpath(dirname(path))
    open(path, "w") do io
        TOML.print(io,
            Dict("meta" => meta(mode, s, project), "timing" => timing,
                "capability" => capability))
    end
end

function progress(label, cell, seconds)
    println(rpad(label, 62), rpad(cell["status_di"], 12), rpad(cell["status_direct"], 12),
        rpad(cell["attribution"], 18), round(seconds; digits = 1), " s")
    flush(stdout)
end

function main(args)
    if !isempty(args) && args[1] == "--child"
        return child_main(args[2:end])
    end
    mode = "--smoke" in args ? "smoke" : "--full" in args ? "full" : nothing
    mode === nothing && error("usage: ad_di_vs_direct.jl --smoke|--full")
    s = getfield(SETTINGS, Symbol(mode))
    path = joinpath(@__DIR__, "results",
        mode == "smoke" ? "ad_di_vs_direct_smoke.toml" : "ad_di_vs_direct.toml")
    dir = mktempdir()
    try
        project = Base1.build_environment(dir, s.threads)
        specs = cells_for(s)
        total = length(specs) + length(CAPABILITIES)
        timing, capability = Dict{String, Any}[], Dict{String, Any}[]
        for (i, spec) in enumerate(specs)
            t = @elapsed cell = timing_cell(spec, project, s, dir, i)
            push!(timing, cell)
            progress("[$i/$total] $(cell["key"])", cell, t)
            # Rewritten after every cell, so an interrupted run leaves its results behind.
            write_results(path, mode, s, project, timing, capability)
        end
        for (j, spec) in enumerate(CAPABILITIES)
            t = @elapsed cell = capability_cell(spec, project, s, dir, j)
            push!(capability, cell)
            progress("[$(length(specs) + j)/$total] $(cell["key"])", cell, t)
            write_results(path, mode, s, project, timing, capability)
        end
        println("wrote ", path)
    finally
        rm(dir; recursive = true, force = true)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == SCRIPT
    main(ARGS)
end
