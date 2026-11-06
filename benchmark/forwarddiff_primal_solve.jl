# Primal-once against the Dual-factor solve: time and peak memory (gpena/Bramble.jl#1123,
# milestone v4.2.1)
#
# The system is a non-uniform 2D Poisson-type form, -∇·(κ ∇u) = g on the unit square,
# u = 0 on the boundary, with κ a `ForwardDiff.Dual` of B partials. Three paths solve it,
# each in its own fresh julia process:
#
#   new          pde_solve(A, F)                 the primal factored once, by the
#                                                default `Float64` route
#   new_sparspak pde_solve(A, F; solver = :sparspak)
#                                                the same with Sparspak, the old path's
#                                                solver: separates the primal-once effect
#                                                from the change of solver
#   old          sparspak_factorize(A) \ F       the Dual matrix factored by Sparspak
#                                                (the oracle of the extension's tests)
#
# Every case is one warmed solve on a fresh system, timed alone. The warm-up is a 16 × 16
# system, checked entrywise against the old path before the timing. The driver runs the
# three back-to-back runs of each row and reads each case's footprint from `top`, the figure
# `julia_slot.sh` watches. Its 10 GB watchdog watches these processes too, so a case over
# the cap is killed and its row says so. Run by hand, alone, never a suite entry:
#
#     .claude/scripts/check_power_load.sh
#     .claude/scripts/gate_preflight.sh --timing
#     BRAMBLE_JULIA_SLOTS=4 .claude/scripts/julia_slot.sh 4 julia --startup-file=no \
#         --threads=4 --project=test benchmark/forwarddiff_primal_solve.jl --full
#
# `--smoke` runs two small sizes once (a correctness and plumbing check, not a timing).
# The default `-O` is kept: never time at `--optimize=1`.

using Bramble
using ForwardDiff: ForwardDiff
using Random: Random
using Printf: @printf, @sprintf
using SparseArrays: SparseMatrixCSC, issymmetric
using Statistics: median
using Sparspak
using SuiteSparse
if Sys.isapple()
    using AppleAccelerate
end

const ROOT = normpath(joinpath(@__DIR__, ".."))
const SCRIPT = @__FILE__
const PATHS = ("new", "new_sparspak", "old")
const SETTINGS = (
    smoke = (sizes = (32, 64), partials = (1, 4), runs = 1, timeout = 900),
    full = (sizes = (64, 128, 256, 512, 1000), partials = (1, 4), runs = 3, timeout = 3600),
)
const CAP_MB = 1024 * parse(Float64, get(ENV, "BRAMBLE_JULIA_MAX_GB", "10"))

# A coefficient κ(x, y) ≥ 1 with B partials: κ = κ₀ + Σₖ pₖ φₖ at p = 0.
function coefficient(::Type{T}, x, y, B) where {T}
    value = 1 + 0.5 * sin(3x)^2 * (1 + y)
    partials = ntuple(k -> 0.2 * sin(k * π * x) * sin(π * y), B)
    return T(value, ForwardDiff.Partials(partials))
end

# The Dual matrix A and the Float64 load F of an m × m node system (m² dofs).
function dual_system(m, B)
    Random.seed!(1234)
    Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (m, m), (false, false))
    Wₕ = gridspace(Ωₕ)
    T = ForwardDiff.Dual{:primal, Float64, B}
    x, y = points(Ωₕ)
    κₕ = element(Wₕ, T)
    κₕ .= vec([coefficient(T, x[I[1]], y[I[2]], B) for I in CartesianIndices((m, m))])
    diffusion(κ) = form(Wₕ, Wₕ, (U, V) -> inner₊(Mₕ(κ) * ∇ₕ(U), ∇ₕ(V)))
    A = assemble(diffusion(κₕ); dirichlet = :boundary)
    gₕ = element(Wₕ)
    avgₕ!(gₕ, x -> exp(x[1] * x[2]))
    F = assemble(form(Wₕ, v -> innerₕ(gₕ, v)); dirichlet = :boundary => x -> 0.0)
    return A, F
end

function solve_path(path::String, A, F)
    path == "new" && return pde_solve(A, F)
    path == "new_sparspak" && return pde_solve(A, F; solver = :sparspak)
    path == "old" && return Bramble.sparspak_factorize(A) \ F
    throw(ArgumentError("unknown path $path, expected one of $PATHS"))
end

# `top`'s MEM column in MB: the footprint Activity Monitor shows, as julia_slot.sh reads it.
function parse_mem(s::AbstractString)
    s = replace(s, r"[+-]" => "")
    unit = s[end]
    v = parse(Float64, s[1:(end - 1)])
    unit == 'K' && return v / 1024
    unit == 'M' && return v
    unit == 'G' && return v * 1024
    throw(ArgumentError("unknown top MEM unit in \"$s\""))
end

function footprint_mb(pid::Integer)
    out = read(ignorestatus(`top -l 1 -pid $pid -stats mem`), String)
    lines = filter(!isempty, strip.(split(out, '\n')))
    return parse_mem(last(lines))
end

# Entrywise agreement of two Dual solutions: values and every partial, to rounding.
function check_agree(x, y, B)
    ok = isapprox(
        ForwardDiff.value.(x), ForwardDiff.value.(y); rtol = 1.0e-10, atol = 1.0e-12
    )
    for k in 1:B
        ok &= isapprox(
            ForwardDiff.partials.(x, k), ForwardDiff.partials.(y, k);
            rtol = 1.0e-10, atol = 1.0e-12
        )
    end
    ok || error("path disagrees with the direct Sparspak Dual factor")
    return nothing
end

function child_main(m::Int, path::String, B::Int)
    A16, F16 = dual_system(16, B)
    check_agree(solve_path(path, A16, F16), solve_path("old", A16, F16), B)
    A, F = dual_system(m, B)
    A0 = SparseMatrixCSC(A.m, A.n, A.colptr, A.rowval, ForwardDiff.value.(A.nzval))
    accelerate = Bramble._default_wants_accelerate(A0, :auto)
    sym = issymmetric(A0)
    before = footprint_mb(getpid())
    GC.gc()
    t0 = time_ns()
    x = solve_path(path, A, F)
    t = (time_ns() - t0) / 1.0e9
    after = footprint_mb(getpid())
    vnorm = sqrt(sum(abs2, ForwardDiff.value.(x)))
    pnorm = sqrt(sum(k -> sum(abs2, ForwardDiff.partials.(x, k)), 1:B))
    println(
        "RESULT m=$m path=$path B=$B dofs=$(length(F)) t=$t before_mb=$before ",
        "after_mb=$after symmetric=$sym accelerate=$accelerate vnorm=$vnorm pnorm=$pnorm",
    )
    return nothing
end

# One case in its own process. The sampler records the largest footprint seen while the
# case runs; a case past `timeout` is killed. Returns a NamedTuple, status "ok" when the
# case printed its RESULT line.
function run_case(m, path, B, timeout)
    julia = joinpath(Sys.BINDIR, Base.julia_exename())
    project = joinpath(ROOT, "test")
    args = ["--startup-file=no", "--threads=4", "--project=$project", SCRIPT]
    cmd = `$julia $args --child $m $path $B`
    proc = open(cmd, "r")
    pid = getpid(proc)
    out = @async read(proc, String)
    peak = 0.0
    t0 = time()
    killed_by_driver = false
    while process_running(proc)
        try
            peak = max(peak, footprint_mb(pid))
        catch
        end
        if time() - t0 > timeout
            killed_by_driver = true
            kill(proc)
            sleep(5)
            process_running(proc) && kill(proc, 9)
            break
        end
        sleep(1)
    end
    wait(proc)
    text = fetch(out)
    fields = Dict{String, String}()
    for line in split(text, '\n')
        startswith(line, "RESULT ") || continue
        for kv in split(line)[2:end]
            k, v = split(kv, '='; limit = 2)
            fields[k] = v
        end
    end
    status = if killed_by_driver
        "timeout"
    elseif proc.termsignal != 0
        "killed (signal $(proc.termsignal))"
    elseif proc.exitcode != 0
        "failed (exit $(proc.exitcode))"
    elseif !haskey(fields, "t")
        "failed (no RESULT)"
    else
        "ok"
    end
    return (
        status = status,
        t = status == "ok" ? parse(Float64, fields["t"]) : NaN,
        before = status == "ok" ? parse(Float64, fields["before_mb"]) : NaN,
        after = status == "ok" ? parse(Float64, fields["after_mb"]) : NaN,
        peak = peak,
        symmetric = get(fields, "symmetric", "?"),
        accelerate = get(fields, "accelerate", "?"),
        vnorm = status == "ok" ? parse(Float64, fields["vnorm"]) : NaN,
        pnorm = status == "ok" ? parse(Float64, fields["pnorm"]) : NaN,
    )
end

function driver_main(settings)
    println("# cap $(round(CAP_MB / 1024; digits = 1)) GB; sampler: top MEM once a second")
    rows = []
    failed = Dict(p => false for p in PATHS)
    for m in settings.sizes, B in settings.partials, path in PATHS
        # After a path has failed at a smaller size, its larger sizes get one attempt each.
        runs = failed[path] ? 1 : settings.runs
        results = [run_case(m, path, B, settings.timeout) for _ in 1:runs]
        for r in results
            r.status == "ok" || (failed[path] = true)
            t = @sprintf("%9.3f", r.t)
            before = @sprintf("%8.1f", r.before)
            peak = @sprintf("%8.1f", r.peak)
            println(
                "run m=$m dofs=$(m^2) B=$B path=$(rpad(path, 12)) ",
                "status=$(rpad(r.status, 18)) t=$t s before=$before MB peak=$peak MB ",
                "sym=$(r.symmetric) acc=$(r.accelerate)",
            )
            flush(stdout)
        end
        push!(rows, (m = m, B = B, path = path, results = results))
    end
    print_table(rows)
    return rows
end

function print_table(rows)
    println("\n| dofs | partials | path | time median (s) | spread |" *
            " peak MB, median (max) | status |")
    println("|---:|---:|---|---:|---:|---:|---|")
    for row in rows
        oks = [r for r in row.results if r.status == "ok"]
        status = join(unique([r.status for r in row.results]), "; ")
        peaks = [r.peak for r in row.results]
        peak = @sprintf("%.0f (%.0f)", median(peaks), maximum(peaks))
        if isempty(oks)
            @printf(
                "| %d | %d | %s | — | — | %s | %s |\n",
                row.m^2, row.B, row.path, peak, status,
            )
        else
            ts = [r.t for r in oks]
            spread = 100 * (maximum(ts) - minimum(ts)) / median(ts)
            @printf(
                "| %d | %d | %s | %.2f | %.1f %% (%d runs) | %s | %s |\n",
                row.m^2, row.B, row.path, median(ts), spread, length(oks), peak, status,
            )
        end
    end
    println("\nfingerprints (vnorm, pnorm) of the ok runs, relative spread across paths:")
    for m in unique(row.m for row in rows), B in unique(row.B for row in rows)
        oks = [
            r for row in rows if row.m == m && row.B == B
                for r in row.results if r.status == "ok"
        ]
        isempty(oks) && continue
        v = [r.vnorm for r in oks]
        p = [r.pnorm for r in oks]
        @printf(
            "m=%d B=%d  vnorm rel spread %.2e  pnorm rel spread %.2e\n", m, B,
            (maximum(v) - minimum(v)) / maximum(v), (maximum(p) - minimum(p)) / maximum(p),
        )
    end
    return nothing
end

if length(ARGS) == 4 && ARGS[1] == "--child"
    child_main(parse(Int, ARGS[2]), ARGS[3], parse(Int, ARGS[4]))
elseif length(ARGS) == 1 && ARGS[1] in ("--smoke", "--full")
    driver_main(SETTINGS[Symbol(ARGS[1][3:end])])
else
    error("usage: julia --project=test forwarddiff_primal_solve.jl --smoke | --full")
end
