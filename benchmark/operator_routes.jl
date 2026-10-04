# The operator routes on four separable forms.
#
# On graded non-uniform meshes of the unit square and cube, with no Dirichlet rows, each
# route builds the same operator of each form. The forms, with `c` a separable grid
# function (it varies along the first axis only, `Rₕ(W, x -> 1 + x[1])`):
#
#   - `laplace`: `innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))`, SPD;
#   - `coefficient`: `innerₕ(u, v) + inner₊(c * ∇ₕ(u), ∇ₕ(v))`, SPD;
#   - `mixed`: `laplace` plus `innerₕ(c * D₋ₓ(D₋ᵧ(u)), v)` (#427's example), not symmetric;
#   - `advection`: `laplace` plus `innerₕ(D₋ₓ(u), v)`, not symmetric.
#
# The routes, the Kronecker and matrix-free operators side by side as alternatives:
#
#   - `assembled`: `assemble(a)`, a `SparseMatrixCSC`;
#   - `kronecker`: `kronecker_operator(a)`, a `KroneckerLinearOperator` (serial product);
#   - `kronecker_threaded`: the same on a space built with `backend(; policy = CpuThreaded())`
#     (#439), whose product threads the axis-1 lines;
#   - `matrix_free_serial`: `matrix_free_operator(a)`;
#   - `matrix_free_threaded`: `matrix_free_operator(a; policy = CpuThreaded())`.
#
# `kronecker_operator` reads a grid-function coefficient once and warns so; the script
# builds the Kronecker routes with that warning silenced, since it would repeat at every
# construction, and prints this note instead. Every table has a `form` column.
#
# Before any timing, each route's product with a random vector is compared with the
# assembled one (relative error, the `correctness` table); a route above 1e-10 aborts the run.
# Then, per form, route and size:
#
#   - `construction`: wall time (minimum over repeats, each on a fresh form) and the bytes
#     allocated while building the operator;
#   - `product`: one five-argument `mul!` (minimum over BenchmarkTools samples) and the bytes
#     the operator keeps alive (`Base.summarysize`; for the matrix-free routes, minus the form
#     it captures, as `matrix_free_spmv.jl` reports);
#   - `solve`, for the SPD forms only: unpreconditioned CG from zero to a relative residual
#     of 1e-8, the same hand-rolled loop for every route so the solver overhead is identical
#     (IterativeSolvers would call the three-argument `mul!`, which `MatrixFreeOperator`
#     lacks), with its time and iteration count. Reference rows join it: `direct` (sparse `\`
#     on the assembled matrix), and for `laplace` also `fdm_solve` (fast diagonalisation,
#     from the form), both with zero iterations.
#
# The meshes are built uniform and then moved by `change_points!` onto a cosine-clustered
# grading along the first axis and a power grading along the others, so every axis is
# non-uniform and no two axes share their nodes.
#
# Usage:
#     julia --project=benchmark --threads=4 benchmark/operator_routes.jl [--smoke] [--save PATH]
#
# The full run needs a quiet, AC-powered machine (`.claude/scripts/check_power_load.sh`) and
# is refused otherwise. `--smoke` runs two tiny sizes per dimension, skips the gate, prefixes
# every line, and is a structural check only. `--save PATH` writes the four tables through
# `save_results` (benchmark/results_io.jl). The threaded routes need more than one thread.

using Bramble
using Bramble: CpuThreaded, D₋ₓ, D₋ᵧ
using Kronecker
using BenchmarkTools
using LinearAlgebra
using Logging
using PrettyTables
using Random
using SparseArrays

include(joinpath(@__DIR__, "results_io.jl"))

const SMOKE = "--smoke" in ARGS
const SAVE_PATH = let k = findfirst(==("--save"), ARGS)
    k === nothing ? nothing : ARGS[k + 1]
end
const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const POWER_SCRIPT = joinpath(REPO_ROOT, ".claude", "scripts", "check_power_load.sh")

const LADDERS = SMOKE ? (2 => (8, 12), 3 => (5, 6)) :
                (2 => (32, 64, 128, 256, 512), 3 => (8, 16, 32, 48, 64))
const ROUTES = ("assembled", "kronecker", "kronecker_threaded", "matrix_free_serial",
    "matrix_free_threaded")
const FORMS = ("laplace", "coefficient", "mixed", "advection")
const SPD_FORMS = ("laplace", "coefficient")
const CORRECTNESS_TOL = 1e-10
const CG_RTOL = 1e-8
const CG_MAXITER = 100_000
const BUILD_REPEATS = SMOKE ? 1 : 3
const SOLVE_REPEATS = SMOKE ? 1 : 3
const BENCH_SAMPLES = SMOKE ? 3 : 50
const BENCH_SECONDS = SMOKE ? 0.1 : 2.0

# Every line of output is prefixed under `--smoke`, so a figure recorded from a smoke run
# can never be mistaken for a real measurement.
function _out(msg::AbstractString = "")
    println(SMOKE ? "[SMOKE -- STRUCTURAL CHECK ONLY, NOT A MEASUREMENT] " * msg : msg)
end

function _print_table(data, header)
    buf = IOBuffer()
    pretty_table(buf, data; column_labels = header, fit_table_in_display_horizontally = false,
        fit_table_in_display_vertically = false)
    for line in split(String(take!(buf)), '\n')
        isempty(line) || _out(line)
    end
end

# --- AC power / load gate, as in benchmark/gpu_offload.jl ------------------- #

function _power_load_state()
    isfile(POWER_SCRIPT) || return (ok = true, power = "unknown (script missing)")
    cmd = SMOKE ? `$POWER_SCRIPT --allow-battery` : `$POWER_SCRIPT --poll 600`
    io = IOBuffer()
    ok = true
    try
        run(pipeline(cmd; stdout = io, stderr = io))
    catch
        ok = false
    end
    m = match(r"Power:\s*(.+)", String(take!(io)))
    return (ok = ok || SMOKE, power = m === nothing ? "unknown" : strip(m.captures[1]))
end

# --- Problem ------------------------------------------------------------------ #

box(D) = reduce(×, ntuple(_ -> interval(0.0, 1.0), D))

# Cosine clustering on the first axis, a power grading on the others.
function graded_points(n, d)
    t = range(0.0, 1.0; length = n)
    return d == 1 ? @.(0.5 * (1 - cos(π * t))) : t .^ (1 + 0.25 * d)
end

function graded_space(D, n, be)
    Ωₕ = mesh(domain(box(D)), ntuple(_ -> n, D), ntuple(_ -> true, D); backend = be)
    Bramble.change_points!(Ωₕ, ntuple(d -> graded_points(n, d), D))
    any(d -> Bramble.is_uniform(Ωₕ(d)), 1:D) && error("$(D)D n=$n mesh is still uniform")
    return gridspace(Ωₕ)
end

# The bilinear function of `kind` on the space `W`; a coefficient is a grid function on `W`.
function form_function(kind, W)
    c = Rₕ(W, x -> 1 + x[1])
    kind == "laplace" && return (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))
    kind == "coefficient" && return (u, v) -> innerₕ(u, v) + inner₊(c * ∇ₕ(u), ∇ₕ(v))
    kind == "mixed" && return (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)) +
                                        innerₕ(c * D₋ₓ(D₋ᵧ(u)), v)
    kind == "advection" && return (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)) +
                                            innerₕ(D₋ₓ(u), v)
    error("unknown form $kind")
end

new_form(kind, W) = form(W, W, form_function(kind, W))

# `W` serial and `Wt` threaded, on the same mesh points. Every route keeps its own form:
# `assemble` fills a form's cache with scatter tables the other routes never need.
function build(kind, route, W, Wt)
    route == "kronecker_threaded" && return build_kronecker(kind, Wt)
    a = new_form(kind, W)
    route == "assembled" && return assemble(a), a
    route == "kronecker" && return build_kronecker(kind, W)
    route == "matrix_free_serial" && return matrix_free_operator(a), a
    route == "matrix_free_threaded" && return matrix_free_operator(a; policy = CpuThreaded()), a
    error("unknown route $route")
end

# The snapshot warning is silenced here only (see the header).
function build_kronecker(kind, W)
    a = new_form(kind, W)
    return with_logger(() -> kronecker_operator(a), NullLogger()), a
end

function held_bytes(route, op, a)
    bytes = Base.summarysize(op)
    startswith(route, "matrix_free") && (bytes -= Base.summarysize(a))
    return bytes
end

# --- Shared solver ------------------------------------------------------------ #

# Unpreconditioned CG from zero, through the five-argument `mul!` only.
function cg!(x, A, b; rtol = CG_RTOL, maxiter = CG_MAXITER)
    fill!(x, 0)
    r = copy(b)
    p = copy(r)
    Ap = similar(x)
    rsold = dot(r, r)
    stop = (rtol * norm(b))^2
    rsold <= stop && return x, 0
    for k in 1:maxiter
        mul!(Ap, A, p, true, false)
        α = rsold / dot(p, Ap)
        axpy!(α, p, x)
        axpy!(-α, Ap, r)
        rsnew = dot(r, r)
        rsnew <= stop && return x, k
        p .= r .+ (rsnew / rsold) .* p
        rsold = rsnew
    end
    return x, maxiter
end

function min_elapsed(f, repeats)
    f() # warm-up
    return minimum(_ -> @elapsed(f()), 1:repeats)
end

relerr(y, ref) = norm(y - ref) / norm(ref)

# --- One size ----------------------------------------------------------------- #

function row(kind, route, D, n, N; kwargs...)
    return Dict{String, Any}("form" => kind, "route" => route, "dim" => D, "n" => n,
        "ndofs" => N, "mesh" => "non-uniform", (String(k) => v for (k, v) in kwargs)...)
end

function measure!(tables, kind, D, n)
    W = graded_space(D, n, backend())
    Wt = graded_space(D, n, backend(; policy = CpuThreaded()))
    N = ndofs(W)
    Random.seed!(420)
    x = rand(N)
    b = rand(N)
    ops = Dict(route => build(kind, route, W, Wt) for route in ROUTES)
    A = first(ops["assembled"])
    is_separable(new_form(kind, W)) || error("$kind $(D)D n=$n form is not separable")

    # Correctness before any timing.
    ref = A * x
    for route in ROUTES
        y = zeros(N)
        mul!(y, first(ops[route]), x, true, false)
        e = relerr(y, ref)
        push!(tables["correctness"], row(kind, route, D, n, N; rel_error = e))
        e < CORRECTNESS_TOL ||
            error("$route disagrees with the assembled product at $kind $(D)D n=$n: " *
                  "rel error $e")
    end

    for route in ROUTES
        op, a = ops[route]
        t_build = min_elapsed(() -> build(kind, route, W, Wt), BUILD_REPEATS)
        bytes_alloc = @allocated build(kind, route, W, Wt)
        push!(tables["construction"],
            row(kind, route, D, n, N; time_s = t_build, bytes_alloc))

        y = zeros(N)
        mul!(y, op, x, true, false)
        trial = run(@benchmarkable(mul!($y, $op, $x, true, false); samples = BENCH_SAMPLES,
            evals = 1, seconds = BENCH_SECONDS))
        push!(tables["product"], row(kind, route, D, n, N; time_s = minimum(trial.times) / 1e9,
            bytes_held = held_bytes(route, op, a)))

        kind in SPD_FORMS || continue
        u = zeros(N)
        t_solve = min_elapsed(() -> cg!(u, op, b), SOLVE_REPEATS)
        _, iterations = cg!(u, op, b)
        iterations < CG_MAXITER || error("CG did not converge for $route at $kind $(D)D n=$n")
        push!(tables["solve"], row(kind, route, D, n, N; time_s = t_solve, iterations,
            rel_residual = relerr(A * u, b)))
    end

    kind in SPD_FORMS || return nothing
    # Reference solves: fast diagonalisation from the form (the plain Laplacian only), and
    # sparse `\`.
    if kind == "laplace"
        a_fdm = new_form(kind, W)
        t_fdm = min_elapsed(() -> fdm_solve(a_fdm, b), SOLVE_REPEATS)
        u_fdm = fdm_solve(a_fdm, b)
        push!(tables["solve"], row(kind, "fdm_solve", D, n, N; time_s = t_fdm, iterations = 0,
            rel_residual = relerr(A * u_fdm, b)))
    end
    t_direct = min_elapsed(() -> A \ b, SOLVE_REPEATS)
    u_direct = A \ b
    push!(tables["solve"], row(kind, "direct", D, n, N; time_s = t_direct, iterations = 0,
        rel_residual = relerr(A * u_direct, b)))
    return nothing
end

# --- Driver ------------------------------------------------------------------- #

function print_tables(tables)
    for (name, cols) in (
        "correctness" => ("rel_error",),
        "construction" => ("time_s", "bytes_alloc"),
        "product" => ("time_s", "bytes_held"),
        "solve" => ("time_s", "iterations", "rel_residual"))
        rows = tables[name]
        header = ["form", "route", "dim", "n", "ndofs", cols...]
        data = [r[h] isa AbstractFloat ? round(r[h]; sigdigits = 4) : r[h]
                for r in rows, h in header]
        _out()
        _out("=== $name ===")
        _print_table(data, header)
    end
end

function main()
    if !SMOKE
        gate = _power_load_state()
        if !gate.ok
            _out("REFUSED: $(POWER_SCRIPT) reports the machine is not ready for measurement " *
                 "(battery power, or load above threshold, after waiting). Plug in and/or " *
                 "quiet the machine, then re-run, or pass --smoke for a structural check only.")
            exit(1)
        end
    end
    set_zero_subnormals(true)
    nt = Threads.nthreads()
    _out("Operator routes on four separable forms, non-uniform meshes -- #420, #427, #439")
    _out("Julia $(VERSION), $nt threads" *
         (nt == 1 ? "  WARNING: the threaded routes run on one thread" : "") *
         (SMOKE || nt == 4 ? "" : "  WARNING: expected 4 (bramble-benchmarks §1)"))

    tables = Dict(name => Dict{String, Any}[]
    for name in ("correctness", "construction", "product", "solve"))
    _out("kronecker_operator's snapshot warning for grid-function coefficients is silenced")
    for (D, ladder) in LADDERS, n in ladder, kind in FORMS

        _out("measuring $kind $(D)D n=$n")
        measure!(tables, kind, D, n)
        GC.gc()
    end
    print_tables(tables)

    if SAVE_PATH !== nothing
        save_results(SAVE_PATH, basename(@__FILE__), tables; smoke = SMOKE)
        _out()
        _out("saved $(SAVE_PATH)")
    end
    _out()
    _out("OK-OPERATOR-ROUTES")
    return nothing
end

main()
