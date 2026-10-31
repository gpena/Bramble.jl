# Does Bramble need five AD backends? (gpena/Bramble.jl#369, milestone v4.2.0)
#
# A one-off evaluation, run by hand, that measures ForwardDiff, PolyesterForwardDiff,
# ReverseDiff, Mooncake and Enzyme on the same three workloads, each through
# DifferentiationInterface as a user calls it, and writes one TOML results file. It is not a
# benchmark-suite entry: the heavy backends are neither test nor benchmark dependencies, so
# the script builds a temporary environment that develops this checkout and adds them, and no
# Project.toml changes.
#
#     .claude/scripts/julia_slot.sh 2 julia --startup-file=no --threads=1 \
#         benchmark/ad_backend_evaluation.jl --smoke
#     .claude/scripts/julia_slot.sh 4 julia --startup-file=no --threads=1 \
#         benchmark/ad_backend_evaluation.jl --full
#
# The outer `julia_slot.sh` reserves the threads the cells use (2 for `--smoke`, 4 for
# `--full`). This process only schedules: every cell runs in a child `julia --threads=N`, so a
# crash, a timeout, or a poisoned thread pool (#369: a task failing inside `@batch` leaves
# Polyester serial for the rest of its process) is recorded as `crash` and spoils no later
# cell. The children draw their slots from a private pool of N slots, so they run one at a
# time inside the outer reservation and the machine-wide pool never sees a nested request.
#
# Workloads, all on a seeded non-uniform 2D mesh, each checked against a central FiniteDiff
# reference (`relerr`, max-norm relative; `wrong` means it returned with `relerr >= 1e-6`).
#
# `dense_gradient` differentiates a scalar functional of operator applications and a linear
# form's assembly with respect to `np` entries of a nodal coefficient. It runs under
# CpuSerial at every `np`, and under CpuThreaded and CpuPolyester at `np = 16`.
#
# `adjoint_solve` differentiates `J(κ) = ‖u(κ) - d‖²` with `u = pde_solve(assemble(a(κ)),
# F)` and `np` free entries of κ. That is the inverse-problem shape, where reverse mode
# should win as `np` grows. ForwardDiff and PolyesterForwardDiff solve with `solver =
# :sparspak` and `F` promoted to `eltype(A)`, since the default `\` overflows the stack on
# a `Dual` sparse matrix. Enzyme goes through its native `pde_solve` rule. ReverseDiff and
# Mooncake go through the `ChainRulesCore` rrule where they can import it, and are
# `unsupported` where they cannot.
#
# `sparse_jacobian` takes the `AutoSparse` Jacobian of a nonlinear-diffusion residual. Its
# reference is a colored central FiniteDiff Jacobian on the same known pattern.
#
# A cell records `first_call_s` (prepare plus the first call, compilation included),
# `per_call_s` (best of a few prepared calls) and `allocs_bytes` (one prepared call).

using TOML
using Dates

const ROOT = dirname(@__DIR__)
const SCRIPT = @__FILE__
const BACKENDS = ("ForwardDiff", "PolyesterForwardDiff", "ReverseDiff", "Mooncake", "Enzyme")
const WRONG_TOL = 1e-6
const ERROR_CHARS = 400

# Sizes are `m` points per direction, so `m^2` unknowns. `nps` is the parameter sweep, and
# `(m - 2)^2 >= maximum(nps)` interior nodes must exist. The policy cells run at `policy_np`
# on `m_policy` points, the 41 x 41 grid test/space/autodiff_policies.jl measured them on:
# the threaded kernels only engage on a grid that size, and a smaller one would pass them
# serially.
const SETTINGS = (
    smoke = (
        threads = 2, nps = (2, 16), policy_np = 16, m_dense = 10, m_adjoint = 10,
        m_sparse = 10, m_policy = 41, timeout = 900),
    full = (
        threads = 4, nps = (2, 4, 16, 64, 256, 1024), policy_np = 16, m_dense = 34,
        m_adjoint = 34, m_sparse = 100, m_policy = 41, timeout = 3600)
)

# ---------------------------------------------------------------------------------------
# One cell, run in a child process. The child's code is a quoted block evaluated into a
# module of its own once the child knows which backend it needs, so loading the five
# backends is paid per cell by the one cell that uses it.
# ---------------------------------------------------------------------------------------

const CHILD_CODE = quote
    using Bramble
    using Bramble: D₋ₓ, CpuPolyester, ast_sparsity_detector
    import Bramble: pde_solve
    using Random
    using SparseArrays
    using LinearAlgebra
    import DifferentiationInterface as DI
    using DifferentiationInterface: AutoSparse, AutoForwardDiff, AutoPolyesterForwardDiff,
                                    AutoReverseDiff, AutoMooncake, AutoEnzyme,
                                    AutoFiniteDiff
    import ForwardDiff, FiniteDiff, Sparspak, Polyester, ChainRulesCore
    import SparseConnectivityTracer, SparseMatrixColorings
    using SparseMatrixColorings: GreedyColoringAlgorithm

    const POLICIES = (
        CpuSerial = () -> Serial(), CpuThreaded = () -> Parallel(),
        CpuPolyester = () -> CpuPolyester())

    # A seeded non-uniform mesh, reseeded per call, so the cell and its reference
    # differentiate the same grid.
    function mesh2d(policy, m)
        Random.seed!(1234)
        Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
        return mesh(Ω, (m, m), (false, false); backend = backend(policy = policy))
    end

    # The nodal coefficient `κ` has `ndofs` entries; the parameters are `np` of the interior
    # ones, evenly spread (a boundary node's value never reaches an eliminated system), the
    # rest are fixed at `κ0`.
    base_kappa(nd) = [1.0 + 0.5 * sin(0.37k)^2 for k in 1:nd]
    function free_indices(m, np)
        interior = [i + (j - 1) * m for j in 2:(m - 1) for i in 2:(m - 1)]
        return interior[round.(Int, range(1, length(interior); length = np))]
    end

    function kappa(p::AbstractVector{T}, κ0, idx) where {T}
        κ = zeros(T, length(κ0))
        copyto!(κ, κ0)
        κ[idx] .= p
        return κ
    end

    function dense_problem(policy, m, np)
        Wₕ = gridspace(mesh2d(policy, m))
        nd = ndofs(Wₕ)
        κ0 = base_kappa(nd)
        idx = free_indices(m, np)
        function f(p::AbstractVector{T}) where {T}
            κₕ = element(Wₕ, T)
            κₕ .= kappa(p, κ0, idx)
            l = form(Wₕ, v -> innerₕ(κₕ, v))
            return sum(abs2, parent(D₋ₓ(κₕ))) + normₕ(κₕ)^2 + sum(abs2, assemble(l))
        end
        return (f = f, x = κ0[idx], n = nd)
    end

    # How the adjoint solve is reached, by backend. `:plain` is the default `\`, used for
    # the Float64 reference only. `:sparspak` serves `Dual` matrices, whose default `\`
    # overflows the stack, and promotes `F` to `eltype(A)`. `:native` is `pde_solve(A, F)`
    # with no keywords, the signature the Enzyme rule and the imported rrules match.
    solve(A, F, ::Val{:plain}) = pde_solve(A, F)
    solve(A, F, ::Val{:sparspak}) = pde_solve(A, convert(Vector{eltype(A)}, F);
        solver = :sparspak)
    solve(A, F, ::Val{:native}) = pde_solve(A, F)

    function adjoint_problem(policy, m, np, route)
        Wₕ = gridspace(mesh2d(policy, m))
        nd = ndofs(Wₕ)
        κ0 = base_kappa(nd)
        idx = free_indices(m, np)
        gₕ = element(Wₕ)
        avgₕ!(gₕ, x -> exp(x[1] * x[2]))
        F = assemble(form(Wₕ, v -> innerₕ(gₕ, v)); dirichlet = :boundary => x -> 0.0)
        diffusion(κₕ) = form(Wₕ, Wₕ, (U, V) -> inner₊(Mₕ(κₕ) * ∇ₕ(U), ∇ₕ(V)))
        function system(κ::AbstractVector{T}) where {T}
            κₕ = element(Wₕ, T)
            κₕ .= κ
            return assemble(diffusion(κₕ); dirichlet = :boundary)
        end
        # The data come from a perturbed coefficient, so `J` is not at its minimum.
        d = solve(system(κ0 .* [1 + 0.3 * cos(0.11k) for k in 1:nd]), F, Val(:plain))
        function f(p::AbstractVector)
            A = system(kappa(p, κ0, idx))
            return sum(abs2, solve(A, F, route) .- d)
        end
        return (f = f, x = κ0[idx], n = nd)
    end

    policy_α(u) = 3 + 1 / (1 + u^2)

    function sparse_problem(policy, m)
        Wₕ = gridspace(mesh2d(policy, m))
        nd = ndofs(Wₕ)
        gₕ = element(Wₕ)
        avgₕ!(gₕ, x -> exp(x[1] * x[2]))
        F = assemble(form(Wₕ, v -> innerₕ(gₕ, v)))
        # `Mₕ(uₕ)` is one averaged element per direction in 2D.
        diffusion(uₕ) = form(Wₕ, Wₕ,
            (U, V) -> inner₊(map(c -> policy_α.(c), Mₕ(uₕ)) * ∇ₕ(U), ∇ₕ(V)))
        function residual(u::AbstractVector{T}) where {T}
            uₕ = element(Wₕ, T)
            uₕ .= u
            return assemble(diffusion(uₕ); dirichlet = :boundary) * u .- F
        end
        detector = ast_sparsity_detector(diffusion(element(Wₕ, 0.0)), U -> Mₕ(U))
        return (f = residual, x = [0.5 + 0.3 * sin(0.01k) for k in 1:nd], n = nd,
            detector = detector)
    end

    sparse_wrap(inner, detector) = AutoSparse(
        inner; sparsity_detector = detector, coloring_algorithm = GreedyColoringAlgorithm())

    function make_backend(name)
        name == "ForwardDiff" && return AutoForwardDiff()
        name == "PolyesterForwardDiff" && return AutoPolyesterForwardDiff()
        name == "ReverseDiff" && return AutoReverseDiff()
        name == "Mooncake" && return AutoMooncake(config = nothing)
        # Enzyme's two annotations, for the reasons given in test/space/autodiff_heavy.jl.
        name == "Enzyme" && return AutoEnzyme(
            mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
            function_annotation = Enzyme.Const)
        error("unknown backend $name")
    end

    # Brings the `pde_solve` rrule to a reverse-mode backend that does not take a native
    # rule. Enzyme has its own rule in `BrambleEnzymeExt`. A failure here is `unsupported`.
    function import_rrule(name)
        if name == "Mooncake"
            Core.eval(@__MODULE__,
                :(Mooncake.@from_rrule(Mooncake.DefaultCtx,
                    Tuple{typeof(pde_solve), SparseMatrixCSC{Float64, Int}, Vector{Float64}})))
            # The import only fails when the rule is first used, so use it on a 3 x 3 system.
            A = sparse(1:3, 1:3, [2.0, 3.0, 4.0])
            probe = A -> sum(pde_solve(A, ones(3)))
            cache = Mooncake.prepare_gradient_cache(probe, A)
            Mooncake.value_and_gradient!!(cache, probe, A)
        elseif name == "ReverseDiff"
            # The macro only generates a method for `TrackedArray` arguments; the check
            # below is that one of them applies to the call the workload makes.
            Core.eval(@__MODULE__,
                :(ReverseDiff.@grad_from_chainrules(pde_solve(A::ReverseDiff.TrackedArray, F::ReverseDiff.TrackedArray))))
            call = Tuple{typeof(pde_solve), SparseMatrixCSC{<:ReverseDiff.TrackedReal, Int},
                Vector{Float64}}
            imported = filter(m -> m.module === @__MODULE__, collect(methods(pde_solve)))
            all(m -> typeintersect(m.sig, call) === Union{}, imported) && error(
                "the imported rule takes TrackedArray arguments, and the assembled A is a " *
                "SparseMatrixCSC of TrackedReal: no imported method applies")
        end
        return nothing
    end

    # The best of a few prepared calls, within a time budget, and one call's allocations.
    function time_calls(call)
        call()
        best = Inf
        spent = 0.0
        reps = 0
        while reps < 20 && (reps < 3 || spent < 2.0)
            t = @elapsed call()
            best = min(best, t)
            spent += t
            reps += 1
        end
        return best, @allocated(call())
    end

    # The first call goes through `invokelatest`, so it is dispatched when the clock runs.
    # Enzyme builds its derivative while the caller is compiled, and a call that inference
    # resolves statically would pay for it before the clock started.
    function measure_gradient(ad, f, x)
        local prep, g
        first_call = @elapsed begin
            prep = Base.invokelatest(DI.prepare_gradient, f, ad, x)
            g = Base.invokelatest(DI.gradient, f, prep, ad, x)
        end
        per_call, allocs = time_calls(() -> DI.gradient(f, prep, ad, x))
        return g, first_call, per_call, allocs
    end

    function measure_jacobian(ad, f, x)
        local prep, J
        first_call = @elapsed begin
            prep = Base.invokelatest(DI.prepare_jacobian, f, ad, x)
            J = Base.invokelatest(DI.jacobian, f, prep, ad, x)
        end
        per_call, allocs = time_calls(() -> DI.jacobian(f, prep, ad, x))
        return J, first_call, per_call, allocs
    end

    # Max-norm relative difference, on the stored entries for a sparse matrix.
    relerr(a::AbstractVector, ref) = maximum(abs, a .- ref) / maximum(abs, ref)
    function relerr(a::AbstractSparseMatrix, ref)
        return maximum(abs, nonzeros(a - ref)) / maximum(abs, nonzeros(ref))
    end

    # Measures one cell: the backend's value, the three costs and the reference. `note_n(n)`
    # records the unknown count as soon as it is known, so a cell that crashes later still
    # reports it.
    function measure_cell(note_n, workload, backend_name, policy, np, m)
        pol = getfield(POLICIES, Symbol(policy))()
        if workload == "dense_gradient"
            prob = dense_problem(pol, m, np)
            ref_prob = dense_problem(Serial(), m, np)
            note_n(prob.n)
            ad = make_backend(backend_name)
            ref = FiniteDiff.finite_difference_gradient(ref_prob.f, ref_prob.x, Val(:central))
            return (measure_gradient(ad, prob.f, prob.x)..., ref)
        elseif workload == "adjoint_solve"
            route = backend_name in ("ForwardDiff", "PolyesterForwardDiff") ? Val(:sparspak) :
                    Val(:native)
            prob = adjoint_problem(pol, m, np, route)
            note_n(prob.n)
            ad = make_backend(backend_name)
            ref = FiniteDiff.finite_difference_gradient(
                adjoint_problem(Serial(), m, np, Val(:plain)).f, prob.x, Val(:central))
            try
                import_rrule(backend_name)
            catch e
                throw(Main.Unsupported(e))
            end
            return (measure_gradient(ad, prob.f, prob.x)..., ref)
        elseif workload == "sparse_jacobian"
            prob = sparse_problem(pol, m)
            note_n(prob.n)
            ad = sparse_wrap(make_backend(backend_name), prob.detector)
            ref_ad = sparse_wrap(AutoFiniteDiff(fdtype = Val(:central)), prob.detector)
            ref = DI.jacobian(prob.f, ref_ad, prob.x)
            return (measure_jacobian(ad, prob.f, prob.x)..., ref)
        end
        error("unknown workload $workload")
    end

    function run_cell!(note_n, workload, backend_name, policy, np, m)
        value, first_call, per_call, allocs, ref = measure_cell(
            note_n, workload, backend_name, policy, np, m)
        return relerr(value, ref), first_call, per_call, allocs
    end
end

# An exception is data, not an excuse to break the TOML: printable characters only.
clean(text) = first(filter(isprint, text), ERROR_CHARS)
function error_text(e)
    text = try
        sprint(showerror, e)
    catch
        "$(typeof(e)) (its message could not be printed)"
    end
    return clean(text)
end

# Raised by a cell whose backend cannot take the `pde_solve` rule at all.
struct Unsupported <: Exception
    err::Any
end

# Raised when a backend, or the workload code, fails to load in the child.
struct LoadFailure <: Exception
    err::Any
end

function write_cell(path, cell)
    open(path, "w") do io
        TOML.print(io, cell)
    end
end

function child_main(workload, backend_name, policy, np, m, out)
    np, m = parse(Int, np), parse(Int, m)
    cell = Dict{String, Any}(
        "workload" => workload, "backend" => backend_name, "policy" => policy,
        "parameters" => workload == "sparse_jacobian" ? m^2 : np, "n" => m^2,
        "status" => "running", "relerr" => NaN, "first_call_s" => NaN,
        "per_call_s" => NaN, "allocs_bytes" => -1, "error" => "")
    # Written first, so a child that dies leaves the cell's identity (and `n`) behind.
    write_cell(out, cell)
    mod = Module(:AdEvalCell)
    function note_n(n)
        cell["n"] = n
        write_cell(out, cell)
    end
    try
        try
            Core.eval(mod, :(import $(Symbol(backend_name))))
            Core.eval(mod, CHILD_CODE)
        catch e
            throw(LoadFailure(e))
        end
        run_cell! = Base.invokelatest(getfield, mod, :run_cell!)
        err, first_call, per_call, allocs = Base.invokelatest(
            run_cell!, note_n, workload, backend_name, policy, np, m)
        cell["relerr"] = Float64(err)
        cell["first_call_s"] = first_call
        cell["per_call_s"] = per_call
        cell["allocs_bytes"] = Int(allocs)
        cell["status"] = isfinite(err) && err < WRONG_TOL ? "ok" : "wrong"
    catch e
        e isa InterruptException && rethrow()
        cell["status"] = e isa Unsupported ? "unsupported" : "throws"
        cell["error"] = e isa Unsupported ? error_text(e.err) :
                        e isa LoadFailure ? "load: " * error_text(e.err) : error_text(e)
    end
    write_cell(out, cell)
    return nothing
end

# ---------------------------------------------------------------------------------------
# The scheduler: builds the environment, runs every cell in a child, writes the results.
# ---------------------------------------------------------------------------------------

function cells_for(s)
    cells = NamedTuple[]
    for np in s.nps, b in BACKENDS

        push!(cells,
            (workload = "dense_gradient", backend = b, policy = "CpuSerial", np = np,
                m = s.m_dense))
    end
    for policy in ("CpuThreaded", "CpuPolyester"), b in BACKENDS

        push!(cells,
            (workload = "dense_gradient", backend = b, policy = policy, np = s.policy_np,
                m = s.m_policy))
    end
    for np in s.nps, b in BACKENDS

        push!(cells,
            (workload = "adjoint_solve", backend = b, policy = "CpuSerial", np = np,
                m = s.m_adjoint))
    end
    for b in BACKENDS
        push!(cells,
            (workload = "sparse_jacobian", backend = b, policy = "CpuSerial", np = 0,
                m = s.m_sparse))
    end
    return cells
end

slot_wrapper() = joinpath(ROOT, ".claude", "scripts", "julia_slot.sh")

# `julia` through the slot wrapper with a private pool of `threads` slots.
function julia_cmd(threads, project, args...)
    julia = `julia --startup-file=no --threads=$threads --project=$project $args`
    isfile(slot_wrapper()) || return julia
    pool = Dict("BRAMBLE_SLOT_DIR" => joinpath(tempdir(), "ad_backend_evaluation_slots"),
        "BRAMBLE_JULIA_SLOTS" => string(threads))
    return addenv(`$(slot_wrapper()) $threads $(julia.exec)`, pool)
end

function descendants(pid)
    kids = try
        parse.(Int, split(read(ignorestatus(`pgrep -P $pid`), String)))
    catch
        Int[]
    end
    return vcat(kids, (descendants(k) for k in kids)...)
end

function kill_tree(proc)
    pid = getpid(proc)
    for p in reverse(descendants(pid))
        run(ignorestatus(`kill -KILL $p`))
    end
    kill(proc, Base.SIGKILL)
end

# Runs `cmd`, its output going to `log`; kills the process tree after `timeout` seconds.
# Returns the exit state: `(:ok, 0)`, `(:exit, code)`, `(:signal, sig)` or `(:timeout, 0)`.
function run_limited(cmd, log, timeout)
    open(log, "w") do io
        proc = run(pipeline(cmd; stdout = io, stderr = io); wait = false)
        timed_out = Ref(false)
        timer = Timer(timeout) do _
            timed_out[] = true
            kill_tree(proc)
        end
        wait(proc)
        close(timer)
        timed_out[] && return (:timeout, 0)
        proc.termsignal != 0 && return (:signal, proc.termsignal)
        return proc.exitcode == 0 ? (:ok, 0) : (:exit, proc.exitcode)
    end
end

# The last lines of a cell's log. Enzyme can print megabytes of LLVM IR on a failure, so only
# the end of the file is read.
function log_tail(log)
    isfile(log) || return ""
    text = open(log) do io
        seekend(io)
        seek(io, max(0, position(io) - 4096))
        return read(io, String)
    end
    lines = filter(!isempty, strip.(split(text, '\n')))
    return isempty(lines) ? "" : clean(join(last(lines, 3), " | "))
end

function build_environment(dir, threads)
    custom = get(ENV, "BRAMBLE_AD_EVAL_ENV", "")
    isempty(custom) || return custom
    setup = """
        using Pkg
        Pkg.activate($(repr(dir)))
        Pkg.develop(path = $(repr(ROOT)))
        # 0.3.2 fails on aarch64; test/Project.toml pins it the same way.
        Pkg.add(name = "CpuId")
        Pkg.pin(name = "CpuId", version = "0.3.1")
        Pkg.add(["DifferentiationInterface", "ForwardDiff", "PolyesterForwardDiff",
            "ReverseDiff", "Mooncake", "Enzyme", "FiniteDiff", "Sparspak",
            "SparseConnectivityTracer", "SparseMatrixColorings", "ChainRulesCore",
            "Polyester"])
        try
            Pkg.precompile()
        catch err
            @warn "precompile failed: a backend that does not load is recorded per cell" err
        end
        """
    log = joinpath(dir, "setup.log")
    cmd = julia_cmd(threads, dir, "-e", setup)
    state = run_limited(cmd, log, 3600)
    state[1] == :ok || error("environment setup failed ($state): $(log_tail(log))")
    return dir
end

git(args...) = strip(read(`git -C $ROOT $args`, String))

function meta(mode, s)
    return Dict{String, Any}(
        "mode" => mode, "threads" => s.threads, "uniform" => false,
        "julia" => string(VERSION), "commit" => git("rev-parse", "HEAD"),
        "date" => Dates.format(Dates.now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        "script_sha" => git("hash-object", SCRIPT),
        "cell_timeout_s" => s.timeout, "cpu" => string(Sys.cpu_info()[1].model),
        "os" => string(Sys.KERNEL), "arch" => string(Sys.ARCH))
end

function run_cell(spec, project, s, dir, index)
    out = joinpath(dir, "cell$index.toml")
    log = joinpath(dir, "cell$index.log")
    args = (SCRIPT, "--child", spec.workload, spec.backend, spec.policy,
        string(spec.np), string(spec.m), out)
    state = run_limited(julia_cmd(s.threads, project, args...), log, s.timeout)
    cell = isfile(out) ? TOML.parsefile(out) :
           Dict{String, Any}(
        "workload" => spec.workload, "backend" => spec.backend,
        "policy" => spec.policy,
        "parameters" => spec.workload == "sparse_jacobian" ? spec.m^2 : spec.np,
        "n" => spec.m^2, "status" => "running", "relerr" => NaN, "first_call_s" => NaN,
        "per_call_s" => NaN, "allocs_bytes" => -1, "error" => "")
    # A child that never finished its cell, however it ended, is a crash.
    if cell["status"] == "running"
        cell["status"] = "crash"
        why = state[1] == :timeout ? "timeout after $(s.timeout) s" :
              state[1] == :signal ? "killed by signal $(state[2])" :
              "exit code $(state[2])"
        cell["error"] = clean("$why: $(log_tail(log))")
    end
    return cell
end

function write_results(path, mode, s, cells)
    mkpath(dirname(path))
    open(path, "w") do io
        TOML.print(io, Dict("meta" => meta(mode, s), "cell" => cells))
    end
end

function main(args)
    if !isempty(args) && args[1] == "--child"
        return child_main(args[2:7]...)
    end
    mode = "--smoke" in args ? "smoke" : "--full" in args ? "full" : nothing
    mode === nothing && error("usage: ad_backend_evaluation.jl --smoke|--full")
    s = getfield(SETTINGS, Symbol(mode))
    path = joinpath(@__DIR__, "results",
        mode == "smoke" ? "ad_backend_evaluation_smoke.toml" : "ad_backend_evaluation.toml")
    dir = mktempdir()
    try
        project = build_environment(dir, s.threads)
        specs = cells_for(s)
        cells = Dict{String, Any}[]
        for (i, spec) in enumerate(specs)
            t = @elapsed cell = run_cell(spec, project, s, dir, i)
            push!(cells, cell)
            println(
                rpad("[$i/$(length(specs))] $(spec.workload) $(spec.backend) " *
                     "$(spec.policy) np=$(spec.np)", 66),
                cell["status"], "  ",
                round(t; digits = 1), " s")
            flush(stdout)
            # Rewritten after every cell, so an interrupted run leaves its results behind.
            write_results(path, mode, s, cells)
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
