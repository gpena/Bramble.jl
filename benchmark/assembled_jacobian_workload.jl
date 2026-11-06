# A Jacobian-bound workload: Newton–Krylov on `-∇·(α(u) ∇u) = g` (gpena/Bramble.jl#1145,
# milestone v4.2.1). Prototype only, never a suite entry; its figures are in
# `docs/notes/internals/assembled_jacobian_workload.md`.
#
# Each Newton step: the residual `R(u) = A(u) u - F`, the Jacobian, an ILU(0) of that
# Jacobian (ILUZero) and GMRES (LinearSolve's `KrylovJL_GMRES`) to a fixed tolerance. The
# Jacobian comes from one of two routes:
#
#   assembled   `jacobian_form` (`a + ∂a`, one `assemble!` per evaluation)
#   sparse AD   `AutoSparse(AutoForwardDiff())` seeded by `ast_sparsity_detector`, the
#               sparse-FD route of `docs/src/examples/poisson_nonlinear.jl`
#
# Both routes share the same `Float64` residual. Each route factors its own Jacobian, so the
# preconditioner follows the route's pattern.
#
# Entry points:
#
#   smoke()               one small mesh: both Jacobians agree to 1e-12, and both Newton
#                         runs take the same number of iterations.
#   time_case(m, policy)  one process's figures on the m × (m+3) non-uniform mesh.
#
# Timing, run by hand and alone (default -O, never --optimize=1):
#
#     .claude/scripts/check_power_load.sh
#     .claude/scripts/gate_preflight.sh --timing
#     BRAMBLE_JULIA_SLOTS=4 .claude/scripts/julia_slot.sh 4 julia --startup-file=no \
#         --threads=4 --project=test benchmark/assembled_jacobian_workload.jl --time
#
# The driver starts one fresh process per (size, policy, run), three runs back to back, at
# m = 100, 300, 700 unless sizes follow `--time`. `--smoke` runs the smoke case once.
#
# Uses internal names (`ast_sparsity_detector`, `type_cached_assemble!`, `Mₓ!`, `Mᵧ!`):
# a prototype, so it reaches past the public API on purpose.

module AssembledJacobianWorkload

using Bramble
using Bramble: allocate_system_matrix, ast_sparsity_detector, type_cached_assemble!,
               Mₓ!, Mᵧ!, CpuSerial, CpuThreaded
using DifferentiationInterface: DifferentiationInterface, AutoSparse, AutoForwardDiff,
                                prepare_jacobian
using ForwardDiff: ForwardDiff
using SparseConnectivityTracer: SparseConnectivityTracer
using SparseMatrixColorings: GreedyColoringAlgorithm
using ILUZero: ilu0
using LinearSolve: LinearSolve, LinearProblem, KrylovJL_GMRES
using LinearAlgebra: mul!, norm
using SparseArrays: SparseMatrixCSC, nnz
using Random: Random

const DI = DifferentiationInterface

# The problem: `α(u) = 3 + 1/(1 + u²)`, the load `g = 1 + x y`, and `u = exp(x + y)` on
# the boundary. The Jacobian checks need no manufactured solution; the boundary data only
# sets the Dirichlet rows.
α(s) = 3 + 1 / (1 + s^2)
uexact(x) = exp(x[1] + x[2])
g(x) = 1 + x[1] * x[2]

# Newton stops at ‖R‖₂ ≤ NEWTON_RTOL ‖R(0)‖₂; GMRES stops at a relative residual of
# GMRES_RTOL. Both are fixed for every route and every size.
const NEWTON_RTOL = 1.0e-8
const GMRES_RTOL = 1.0e-8

function _mesh(m, policy; seed = 1145)
    Random.seed!(seed)
    Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
    return mesh(Ω, (m, m + 3), (false, false); backend = backend(policy = policy))
end

function _flux_form(W, αx, αy)
    return form(W, W, (U, V) -> inner₊((αx * ∇ₕ(U)[1], αy * ∇ₕ(U)[2]), ∇ₕ(V)))
end

# The state-dependent coefficients `αₓ = α(Mₓ u)` and `αᵧ = α(Mᵧ u)`, written in place.
struct Coefficients{V}
    Mx::V
    My::V
    αx::V
    αy::V
end

function Coefficients(W, T)
    return Coefficients(element(W, T), element(W, T), element(W, T), element(W, T))
end

function (c::Coefficients)(uₕ)
    Mₓ!(c.Mx, uₕ)
    Mᵧ!(c.My, uₕ)
    c.αx .= α.(c.Mx)
    c.αy .= α.(c.My)
    return nothing
end

# What `type_cached_assemble!` builds for one element type of the state (`Dual` in the
# sparse-AD sweeps): the form and its refill.
struct Build{W}
    space::W
end

function (b::Build)(uₕ)
    W = b.space
    c = Coefficients(W, eltype(uₕ))
    return _flux_form(W, c.αx, c.αy), c
end

struct Problem{W, F, A, C, M, V}
    space::W
    load::F
    form::A
    coefficients::C
    matrix::M
    state::V
end

function Problem(m, policy)
    W = gridspace(_mesh(m, policy))
    Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
    bcs = dirichlet_constraints(Ω, :boundary => uexact)
    gₕ = element(W)
    avgₕ!(gₕ, g)
    F = assemble(form(W, v -> innerₕ(gₕ, v)); dirichlet = bcs)
    c = Coefficients(W, Float64)
    a = _flux_form(W, c.αx, c.αy)
    return Problem(W, F, a, c, allocate_system_matrix(a), element(W, 0.0))
end

# The shared `Float64` residual: refill the coefficients at `u`, assemble `A(u)`, then
# `A(u) u - F`.
function residual!(r, p::Problem, u)
    p.state .= u
    p.coefficients(p.state)
    assemble!(p.matrix, p.form; dirichlet = :boundary)
    mul!(r, p.matrix, u)
    r .-= p.load
    return r
end

# --- The two Jacobian routes ------------------------------------------------------------- #

# `jacobian_form` declares each coefficient by its dependency `U -> L(U)` and its scalar
# function `f`. `jacobian!` refills both coefficients and their derivatives from `u`.
struct AssembledRoute{J, M}
    form::J
    matrix::M
end

function AssembledRoute(p::Problem)
    c = p.coefficients
    jf = Bramble.jacobian_form(p.form, c.αx => (U -> Mₕ(U)[1], α),
        c.αy => (U -> Mₕ(U)[2], α); dirichlet = :boundary)
    return AssembledRoute(jf, Bramble.jacobian_prototype(jf))
end

evaluate!(route::AssembledRoute, u) = Bramble.jacobian!(route.matrix, route.form, u)

# The residual sparse AD differentiates: `type_cached_assemble!`, one cached form per
# element type (`poisson_nonlinear.jl`'s route).
struct SparseADResidual{P}
    problem::P
    cache::Dict{Any, Any}
end

function (fr::SparseADResidual)(u::AbstractVector{T}) where {T}
    p = fr.problem
    uₕ = element(p.space, T)
    uₕ .= u
    A = type_cached_assemble!(Build(p.space), fr.cache, uₕ; dirichlet = :boundary)
    return A * u .- p.load
end

struct SparseADRoute{R, B, P, M}
    residual::R
    backend::B
    prep::P
    matrix::M
end

function SparseADRoute(p::Problem, u)
    residual = SparseADResidual(p, Dict{Any, Any}())
    detector = ast_sparsity_detector(p.form, U -> Mₕ(U))
    backend = AutoSparse(AutoForwardDiff(); sparsity_detector = detector,
        coloring_algorithm = GreedyColoringAlgorithm())
    prep = prepare_jacobian(residual, backend, u)
    matrix = DI.jacobian(residual, prep, backend, u)
    return SparseADRoute(residual, backend, prep, matrix)
end

function evaluate!(route::SparseADRoute, u)
    return DI.jacobian!(route.residual, route.matrix, route.prep, route.backend, u)
end

# --- Newton-Krylov ----------------------------------------------------------------------- #

# One linear solve: ILU(0) of `J` (rebuilt from this Jacobian), then GMRES to GMRES_RTOL.
# A GMRES that stops short of the tolerance is an error: its time is not a Newton step.
function linear_solve(J, rhs)
    P = ilu0(J)
    sol = LinearSolve.solve(LinearProblem(J, rhs), KrylovJL_GMRES(); Pl = P,
        abstol = 0.0, reltol = GMRES_RTOL)
    sol.stats.solved || error("GMRES stopped after $(sol.stats.niter) iterations, unsolved")
    return sol
end

# Residual, Jacobian, ILU and GMRES: one Newton step, `du` the update.
function newton_step!(du, r, p::Problem, route, u)
    residual!(r, p, u)
    J = evaluate!(route, u)
    sol = linear_solve(J, r)
    du .= sol.u
    return sol
end

# Newton from u = 0 to the tolerance. Returns the Newton and the total GMRES iterations.
function newton!(u, r, p::Problem, route; maxit = 25)
    fill!(u, 0.0)
    r0 = norm(residual!(r, p, u))
    krylov = 0
    du = similar(u)
    for it in 0:maxit
        residual!(r, p, u)
        norm(r) <= NEWTON_RTOL * r0 && return (newton = it, krylov = krylov)
        J = evaluate!(route, u)
        sol = linear_solve(J, r)
        krylov += sol.stats.niter
        u .-= sol.u
    end
    error("Newton did not converge in $maxit steps")
end

# A state that is not a solution, so every coefficient's derivative is exercised.
function _test_state(p::Problem)
    return [0.3 + 0.7 * sin(0.37 * i) for i in 1:ndofs(p.space)]
end

_relative_error(J1, J2) = maximum(abs, J1 - J2) / maximum(abs, J2)

"""
    smoke()

One small mesh (9 × 12) under both policies: the two Jacobians agree to 1e-12, and both
Newton runs take the same number of iterations and reach the same solution.
"""
function smoke()
    out = map((CpuSerial(), CpuThreaded())) do policy
        p = Problem(9, policy)
        u = _test_state(p)
        r = similar(u)
        residual!(r, p, u)
        ast = AssembledRoute(p)
        fd = SparseADRoute(p, u)
        err = _relative_error(evaluate!(ast, u), evaluate!(fd, u))
        err < 1.0e-12 || error("assembled Jacobian disagrees with sparse AD: $err")
        u_ast, u_fd = similar(u), similar(u)
        n_ast = newton!(u_ast, r, p, ast)
        n_fd = newton!(u_fd, r, p, fd)
        n_ast.newton == n_fd.newton ||
            error("Newton iterations differ: assembled $(n_ast.newton), " *
              "sparse AD $(n_fd.newton)")
        sol_diff = maximum(abs, u_ast - u_fd)
        sol_diff < 1.0e-6 || error("Newton solutions differ by $sol_diff")
        return (policy = string(typeof(policy)), err = err, newton = n_ast.newton,
            krylov_assembled = n_ast.krylov, krylov_sparse_ad = n_fd.krylov,
            solution_diff = sol_diff)
    end
    return out
end

# --- Timing ------------------------------------------------------------------------------ #

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

function _pattern(J::SparseMatrixCSC)
    return SparseMatrixCSC(J.m, J.n, copy(J.colptr), copy(J.rowval), fill(true, nnz(J)))
end

"""
    time_case(m, policy; reps = 5, solve_reps = 3) -> NamedTuple

One process's figures on the `m × (m+3)` mesh. Everything is warmed before the first timed
call, so the figures are warm: the Jacobian alone, its ILU and GMRES parts, one Newton step
(residual, Jacobian, ILU, GMRES) at a fixed state, and a whole Newton solve from `u = 0`.
Both routes run in this process.
"""
function time_case(m, policy; reps = 5, solve_reps = 3)
    p = Problem(m, policy)
    u = _test_state(p)
    r = similar(u)
    residual!(r, p, u)
    ast = AssembledRoute(p)
    fd = SparseADRoute(p, u)
    err = _relative_error(evaluate!(ast, u), evaluate!(fd, u))
    err < 1.0e-12 || error("assembled Jacobian disagrees with sparse AD: $err")
    residual!(r, p, u)
    linear_solve(ast.matrix, r)
    linear_solve(fd.matrix, r)

    res = _min_time(() -> residual!(r, p, u), reps)
    jac_fd = _min_time(() -> evaluate!(fd, u), reps)
    jac_ast = _min_time(() -> evaluate!(ast, u), reps)
    residual!(r, p, u)
    ilu_fd = _min_time(() -> ilu0(fd.matrix), reps)
    ilu_ast = _min_time(() -> ilu0(ast.matrix), reps)
    gm_fd = _min_time(() -> linear_solve(fd.matrix, r), reps)
    gm_ast = _min_time(() -> linear_solve(ast.matrix, r), reps)
    krylov_fd = linear_solve(fd.matrix, r).stats.niter
    krylov_ast = linear_solve(ast.matrix, r).stats.niter

    du = similar(u)
    step_fd = _min_time(() -> newton_step!(du, r, p, fd, u), reps)
    step_ast = _min_time(() -> newton_step!(du, r, p, ast, u), reps)

    u_fd, u_ast = similar(u), similar(u)
    n_fd = newton!(u_fd, r, p, fd)
    n_ast = newton!(u_ast, r, p, ast)
    n_fd.newton == n_ast.newton || error("iteration counts differ")
    maximum(abs, u_fd - u_ast) < 1.0e-6 || error("Newton solutions differ")
    solve_fd = _min_time(() -> newton!(u_fd, r, p, fd), solve_reps)
    solve_ast = _min_time(() -> newton!(u_ast, r, p, ast), solve_reps)

    return (dofs = length(u), nnz_ast = nnz(ast.matrix), nnz_fd = nnz(fd.matrix),
        err = err, newton = n_ast.newton, krylov_ast = n_ast.krylov,
        krylov_fd = n_fd.krylov, gmres_ast = krylov_ast, gmres_fd = krylov_fd,
        res = res, jac_fd = jac_fd, jac_ast = jac_ast, ilu_fd = ilu_fd, ilu_ast = ilu_ast,
        gm_fd = gm_fd, gm_ast = gm_ast, step_fd = step_fd, step_ast = step_ast,
        solve_fd = solve_fd, solve_ast = solve_ast)
end

end # module

# --- Driver: one fresh process per (size, policy, run) ------------------------------------ #

if abspath(PROGRAM_FILE) == @__FILE__
    using Bramble

    const _POLICIES = Dict("CpuSerial" => Bramble.CpuSerial(),
        "CpuThreaded" => Bramble.CpuThreaded())
    const _FIELDS = (:jac_fd, :jac_ast, :step_fd, :step_ast, :solve_fd, :solve_ast)

    function _child(m, policy_name)
        set_zero_subnormals(true)
        r = AssembledJacobianWorkload.time_case(m, _POLICIES[policy_name];
            solve_reps = m >= 700 ? 1 : 3)
        println("RESULT ", join(("$k=$v" for (k, v) in pairs(r)), " "),
            " maxrss_mb=", Sys.maxrss() / 2^20)
        return nothing
    end

    function _run(m, policy_name)
        julia = joinpath(Sys.BINDIR, Base.julia_exename())
        project = joinpath(@__DIR__, "..", "test")
        flags = ["--startup-file=no", "--threads=4", "--project=$project"]
        text = read(`$julia $flags $(@__FILE__) --child $m $policy_name`, String)
        line = only(filter(l -> startswith(l, "RESULT "), split(text, '\n')))
        return Dict(Symbol(k) => parse(Float64, v)
        for (k, v) in (split(kv, '='; limit = 2) for kv in split(line)[2:end]))
    end

    function _median(x)
        s = sort(x)
        n = length(s)
        return isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2
    end
    _spread(x) = (maximum(x) - minimum(x)) / _median(x)

    _fmt(x; digits = 4) = string(round(x; sigdigits = digits))

    function _driver(sizes, runs)
        rows = []
        for m in sizes, name in ("CpuSerial", "CpuThreaded")
            results = [_run(m, name) for _ in 1:runs]
            for r in results
                pairs_text = join(("$k=$(_fmt(r[k]))" for k in keys(r)), " ")
                println("run m=$m policy=$name ", pairs_text)
                flush(stdout)
            end
            push!(rows, (m = m, policy = name, results = results))
        end
        println("\nmedian, spread (max-min)/median, over $runs processes, seconds")
        for row in rows
            r1 = first(row.results)
            println("dofs=$(Int(r1[:dofs])) $(row.policy)")
            for k in _FIELDS
                x = [r[k] for r in row.results]
                println("  $k median=$(_fmt(_median(x))) spread=$(_fmt(_spread(x); digits = 2))")
            end
            shares = (("ast", :jac_ast, :step_ast), ("fd", :jac_fd, :step_fd))
            for (name, num, den) in shares
                share = [r[num] / r[den] for r in row.results]
                med, spr = _fmt(_median(share); digits = 3), _fmt(_spread(share); digits = 2)
                println("  jacobian share of step, $name: median=$med spread=$spr")
            end
            mem = maximum(r[:maxrss_mb] for r in row.results)
            err = maximum(r[:err] for r in row.results)
            println("  newton=$(Int(r1[:newton])) krylov_ast=$(Int(r1[:krylov_ast])) ",
                "krylov_fd=$(Int(r1[:krylov_fd])) nnz_ast=$(Int(r1[:nnz_ast])) ",
                "nnz_fd=$(Int(r1[:nnz_fd])) max err=$err max rss MB=$(_fmt(mem))")
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
