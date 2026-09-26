#===========================================================================#
# Portable CPU-policy crossover benchmark (user request, 2026-09-26)
#
#     julia --threads=N --project=benchmark benchmark/policy_crossover.jl [--smoke] [--out results.md]
#
# For nine workloads (`Rₕ!` unmasked/masked, `avgₕ!`, `innerₕ` unmasked/masked,
# `D₋ₓ!`, a warmed broadcast axpy, and three assembly sweeps -- first `assemble`,
# warmed `assemble!` of the Laplace bilinear form `inner₊(∇ₕu, ∇ₕv)`, and warmed
# `assemble!` of the linear form `innerₕ(f, v)`), each in 1D/2D/3D on a
# non-uniform mesh, this script sweeps a common total-DOF grid and reports the
# smallest size at which `CpuThreaded`/`Parallel` and `CpuPolyester` start
# beating `CpuSerial`/`Serial` -- "the crossover", using the same "twice
# running" confirmation rule `benchmark/polyester_crossover.jl` uses (a win is
# trusted only once the next larger size also wins).
#
# Every (workload, size) pair runs all three policies on *separately built*
# `Backend`s/meshes that nonetheless carry identical interior points: the
# non-uniform 1D/nD mesh constructor draws random points, so `Random.seed!` is
# called with the same seed immediately before each policy's `mesh(...)` call
# (`src/mesh/mesh1d.jl`'s own `_generate_random_points!`, unarmed, reads
# straight off `Random.default_rng()` -- exactly what an ordinary
# `Random.seed!(N)` call ahead of a non-uniform mesh build already controls
# elsewhere in this repo's tests).
#
# Correctness before timing (bramble-verification): every non-serial arm's
# result is checked against the serial one before its timing is trusted, at
# every (workload, D, size). Elementwise workloads (`Rₕ!`, `avgₕ!`, `D₋ₓ!`,
# broadcast axpy) are compared at `rtol = atol = 1e-12` (broadcast axpy at
# exact `==`, since the threaded/batched broadcast runs Base's own loop per
# band, so it equals serial bit for bit); the `innerₕ` reductions
# at `rtol = atol = 1e-9` (summation order differs across policies); assembled
# matrices/vectors structurally plus `nzval`/entries at `rtol = atol = 1e-11`.
# A mismatch prints `MISMATCH: ...`, withholds that arm's timing, and the
# script never prints an all-clear marker at the end.
#
# One machine-readable line per (workload, D):
#
#     CROSSOVER | <workload> | <1D|2D|3D> | threads=<DOFs or none> | polyester=<DOFs or none> | polyester-vs-threads=<DOFs or none>
#
# plus one context line per D -- an end-to-end assemble + sparse direct solve
# (`A \ F`) of the same Laplace system with a homogeneous Dirichlet boundary,
# at a single moderate size (not a sweep: `\` does not dispatch on the
# execution policy, so it has no crossover of its own to look for):
#
#     CONTEXT | assemble+solve | <1D|2D|3D> | assemble=<ms> | solve=<ms> | solve share=<pct>
#
# A summary recommendation table (per workload x dimension: use Serial below X,
# Polyester above Y, Threads above Z) closes the run. `--out file.md` also
# writes every table as Markdown.
#
# ## Portability (this script must run on any machine, not just this one's CI)
#
# Power source and 1-minute load average are read where the OS exposes them
# (macOS: `pmset -g batt`, `sysctl -n vm.loadavg`; Linux:
# `/sys/class/power_supply/*/online`, `/proc/loadavg`) and printed as a
# warning banner when on battery, under load, or unreadable -- this script
# never refuses to run, unlike the AC-gated benchmarks elsewhere in this
# directory, since its whole point is to run on whatever machine the caller
# has in front of them.
#
# `Polyester` is a benchmark/Project.toml dependency, loaded in a `try`; if it
# is missing for any reason, the Polyester column and every
# polyester/polyester-vs-threads crossover print as `none` rather than the
# script failing.
#
# ## Sizes
#
# The elementwise workloads (`Rₕ!`, `avgₕ!`, `innerₕ`, `D₋ₓ!`, broadcast axpy)
# sweep total-DOF targets from 10^2 up to 10^6; the three assembly workloads
# use the same target list, also capped at 10^6 (assembly is far more
# expensive per DOF than an elementwise sweep, and a 3D 10^7-DOF matrix would
# risk exhausting memory on exactly the small/older machine this script is
# meant to run on -- bramble-verification's "measurements, not aspirations").
# `--smoke` shrinks both lists to a handful of tiny sizes, seconds total, for
# structural validation only.
#===========================================================================#

using Bramble
using Bramble: CpuPolyester, allocate_system_matrix, D₋ₓ!
using PrettyTables
using SparseArrays
using Random

set_zero_subnormals(true)

# --- CLI ------------------------------------------------------------------- #

const SMOKE = "--smoke" in ARGS

const OUT_PATH = let i = findfirst(==("--out"), ARGS)
    if i === nothing
        nothing
    elseif i == length(ARGS)
        error("--out requires a file path argument")
    else
        ARGS[i + 1]
    end
end

# Default cap: 1e6 total dofs, for memory safety on whatever machine this runs on (a 3D
# 1e7-dof sweep risks exhausting memory on a small/older machine -- see the header
# comment). `--max-dofs N` (accepts scientific notation, e.g. `--max-dofs 1e7`) raises it
# for a caller who knows their machine can take it.
const MAX_DOFS = let i = findfirst(==("--max-dofs"), ARGS)
    if i === nothing
        1_000_000
    elseif i == length(ARGS)
        error("--max-dofs requires a value, e.g. --max-dofs 1e7")
    else
        round(Int, parse(Float64, ARGS[i + 1]))
    end
end

const MD_BUF = IOBuffer()

function _out(msg::AbstractString = "")
    line = SMOKE ? "[SMOKE -- STRUCTURAL CHECK ONLY, NOT A MEASUREMENT] " * msg : msg
    println(line)
    OUT_PATH === nothing || println(MD_BUF, msg)
    return nothing
end

function _print_table(data; column_labels)
    buf = IOBuffer()
    pretty_table(buf, data; column_labels = column_labels, fit_table_in_display_horizontally = false)
    for line in split(String(take!(buf)), '\n')
        isempty(line) || _out(line)
    end
    if OUT_PATH !== nothing
        println(MD_BUF, pretty_table(String, data; column_labels = column_labels, backend = :markdown))
        println(MD_BUF)
    end
    return nothing
end

# --- Polyester: optional -------------------------------------------------- #

const POLYESTER_OK = try
    using Polyester
    true
catch e
    println("NOTE: Polyester unavailable ($(sprint(showerror, e))) -- skipping the Polyester column.")
    false
end

# --- Portable power / load read (warn, never refuse) ----------------------- #

function _power_state()
    if Sys.isapple()
        try
            out = strip(read(`pmset -g batt`, String))
            return (on_battery = occursin("Battery Power", out), raw = out, readable = true)
        catch e
            return (on_battery = false, raw = "pmset failed: $(sprint(showerror, e))", readable = false)
        end
    elseif Sys.islinux()
        try
            base = "/sys/class/power_supply"
            isdir(base) || return (on_battery = false, raw = "no $base", readable = false)
            online_files = [
                joinpath(base, d, "online") for d in readdir(base) if isfile(joinpath(base, d, "online"))
            ]
            isempty(online_files) &&
                return (on_battery = false, raw = "no */online file under $base", readable = false)
            online = any(strip(read(f, String)) == "1" for f in online_files)
            files_list = join(online_files, ", ")
            return (on_battery = !online, raw = "online=$online ($files_list)", readable = true)
        catch e
            return (on_battery = false, raw = "sysfs read failed: $(sprint(showerror, e))", readable = false)
        end
    else
        return (on_battery = false, raw = "unsupported OS for a power check ($(Sys.KERNEL))", readable = false)
    end
end

function _load1()
    if Sys.isapple()
        try
            out = strip(read(`sysctl -n vm.loadavg`, String))
            nums = filter(!isempty, split(replace(out, r"[{}]" => ""), ' '))
            return (load1 = parse(Float64, nums[1]), raw = out, readable = true)
        catch e
            return (load1 = NaN, raw = "sysctl failed: $(sprint(showerror, e))", readable = false)
        end
    elseif Sys.islinux()
        try
            out = strip(read("/proc/loadavg", String))
            return (load1 = parse(Float64, split(out)[1]), raw = out, readable = true)
        catch e
            return (load1 = NaN, raw = "/proc/loadavg failed: $(sprint(showerror, e))", readable = false)
        end
    else
        return (load1 = NaN, raw = "unsupported OS for a load check ($(Sys.KERNEL))", readable = false)
    end
end

# --- Header ----------------------------------------------------------------- #

power = _power_state()
load = _load1()

_out("Portable CPU-policy crossover benchmark -- user request 2026-09-26")
_out()
_out("Julia version : $(VERSION)")
_out("CPU           : $(Sys.cpu_info()[1].model)")
_out("OS            : $(Sys.KERNEL) ($(Sys.MACHINE))")
_out("Julia threads : $(Threads.nthreads())")
_out(
    "Run this at the thread count you actually intend to use " *
    "(`julia --threads=N ...`) -- the crossovers below are specific to N.",
)
_out()
_out("Power         : $(power.raw)")
power.readable ||
    _out("  WARNING: power source could not be read on this OS; proceeding without a battery check.")
power.on_battery && _out(
    "  WARNING: on battery power. CPU frequency scaling/thermal throttling can skew " *
    "absolute timings; prefer the ratio columns.",
)
_out("1-min load    : $(load.raw)")
load.readable ||
    _out("  WARNING: 1-minute load average could not be read on this OS; proceeding without a load check.")
if load.readable && load.load1 >= Sys.CPU_THREADS / 2
    _out(
        "  WARNING: load $(round(load.load1; digits = 2)) is at least half of $(Sys.CPU_THREADS) " *
        "cores; absolute timings below may be noisier than usual.",
    )
end
POLYESTER_OK ||
    _out("NOTE: Polyester is unavailable in this environment; its column reads 'none' throughout.")
_out(
    "DOF cap       : $(MAX_DOFS) total dofs (default 1e6; raise with --max-dofs N, e.g. " *
    "--max-dofs 1e7, on a machine that can take a bigger sweep).",
)
_out()

# --- Geometry / sources, dimension-generic (backends.jl's own pattern) ------ #

_unit_cube(::Val{1}) = interval(0.0, 1.0)
_unit_cube(::Val{D}) where {D} = reduce(×, ntuple(_ -> interval(0.0, 1.0), Val(D)))

_grid(::Val{1}, Ωd, n; backend) = mesh(Ωd, n, false; backend = backend)
function _grid(::Val{D}, Ωd, n; backend) where {D}
    mesh(Ωd, ntuple(_ -> n, Val(D)), ntuple(_ -> false, Val(D)); backend = backend)
end

_f(::Val{1}) = x -> sin(2π * x)
_f(::Val{D}) where {D} = x -> prod(sin(2π * xᵢ) for xᵢ in x)
_g(::Val{1}) = x -> cos(3π * x) + 1.0
_g(::Val{D}) where {D} = x -> prod(cos(3π * xᵢ) + 1.0 for xᵢ in x)

_poisson(u, v) = inner₊(∇ₕ(u), ∇ₕ(v))

const SEED = 20260926
const ZERO_BC = :dir => (x -> 0.0)

# --- Sizes ------------------------------------------------------------------ #

# Ladders up to 1e8, filtered down to whatever `MAX_DOFS` allows -- `--max-dofs` raises the
# effective top of the sweep without needing a code change (S6.1 review #2). The assembly
# ladder is coarser than the elementwise one: assembly costs far more per DOF, so the same
# fine spacing would multiply an already-expensive sweep for little extra crossover
# precision.
const _CHEAP_LADDER = (
    100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000,
    10_000_000, 30_000_000, 100_000_000
)
const _ASSEMBLE_LADDER = (100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000)

const CHEAP_TARGETS = SMOKE ? (100, 1_000, 10_000) :
                      Tuple(filter(t -> t <= MAX_DOFS, _CHEAP_LADDER))
const ASSEMBLE_TARGETS = SMOKE ? (100, 1_000) : Tuple(filter(t -> t <= MAX_DOFS, _ASSEMBLE_LADDER))
const CONTEXT_TARGET = SMOKE ? 1_000 : min(100_000, MAX_DOFS)

_n_for(D::Int, target::Int) = max(2, round(Int, target^(1 / D)))

# --- Correctness ------------------------------------------------------------ #

_agree_arr(a, b) = isapprox(a, b; rtol = 1e-12, atol = 1e-12)
_agree_scalar(a, b) = isapprox(a, b; rtol = 1e-9, atol = 1e-9)
_agree_exact(a, b) = a == b

function _agree_mat(A::SparseMatrixCSC, B::SparseMatrixCSC; rtol = 1e-11, atol = 1e-11)
    (A.colptr == B.colptr && A.rowval == B.rowval) || return false
    return isapprox(A.nzval, B.nzval; rtol = rtol, atol = atol)
end

const ALL_OK = Ref(true)

function _check(label::AbstractString, ok::Bool)
    ok || (_out("  MISMATCH: $label"); ALL_OK[] = false)
    return ok
end

# --- Timing: warm-up, then minimum over k samples scaled to >=50ms total --- #

function _sample_count(t1::Float64)
    SMOKE && return 2
    return clamp(ceil(Int, 0.05 / max(t1, 1e-7)), 3, 300)
end

function _min_ms(f::F) where {F}
    f()  # warm-up: JIT compilation happens here, excluded from the trial
    t1 = @elapsed f()
    best = t1
    for _ in 1:_sample_count(t1)
        t = @elapsed f()
        best = min(best, t)
    end
    return best * 1000
end

# --- One row per (workload, D, size) ---------------------------------------- #

struct Row
    workload::String
    D::Int
    n::Int
    dofs::Int
    t_serial::Float64
    t_threads::Float64
    t_poly::Float64
end

# Some Polyester stencil paths are still mid-flight in this repo (gpena/Bramble.jl#356:
# S7.1/S7.2 thread the difference/average engines under CpuThreaded/CpuPolyester but have
# not landed for every operator yet), so a Polyester arm can raise instead of just being
# slow. Caught here rather than left to crash this script: that cell's Polyester column
# reports "none"/"-" instead, exactly as if Polyester were unavailable altogether.
function _poly_try(label::AbstractString, f::F) where {F}
    try
        return f(), true
    catch e
        _out("NOTE: Polyester path not yet available for \"$label\" ($(sprint(showerror, e))) -- reporting polyester=none for this cell.")
        return nothing, false
    end
end

function _bench!(
        rows::Vector{Row}, label::AbstractString, D::Int, n::Int, dofs::Int,
        use_poly::Bool, agree::Function,
        value_s, action_s!::Fs,
        value_t, action_t!::Ft,
        value_b, action_b!::Fb
) where {Fs, Ft, Fb}
    ok_t = _check("$label D=$D dofs=$dofs Threads", agree(value_t, value_s))
    ok_b = use_poly ? _check("$label D=$D dofs=$dofs Polyester", agree(value_b, value_s)) : true
    t_s = _min_ms(action_s!)
    t_t = ok_t ? _min_ms(action_t!) : NaN
    t_b = (use_poly && ok_b) ? _min_ms(action_b!) : NaN
    push!(rows, Row(label, D, n, dofs, t_s, t_t, t_b))
    return nothing
end

# --- Spaces: identical points across policies (seed reset before each mesh) - #

function _spaces(D::Int, n::Int, use_poly::Bool)
    Iᴰ = _unit_cube(Val(D))
    Ωd = domain(Iᴰ, :dir => boundary_symbols(Iᴰ))
    function _build(policy)
        Random.seed!(SEED)
        be = backend(Float64; policy = policy)
        return gridspace(_grid(Val(D), Ωd, n; backend = be))
    end
    Ws = _build(Serial())
    Wt = _build(Parallel())
    Wb = use_poly ? _build(CpuPolyester()) : nothing
    return Ws, Wt, Wb
end

# --- Cheap (elementwise) workloads ------------------------------------------ #

function _run_cheap(D::Int, n::Int)
    use_poly = POLYESTER_OK
    Ws, Wt, Wb = _spaces(D, n, use_poly)
    dofs = ndofs(Ws)
    f = _f(Val(D))
    g = _g(Val(D))
    rows = Row[]

    # --- Rₕ! unmasked ---
    u_s = element(Ws, Float64)
    Rₕ!(u_s, f)
    u_t = element(Wt, Float64)
    Rₕ!(u_t, f)
    u_b, u_b_ok = use_poly ?
                  _poly_try("Rₕ! unmasked D=$D n=$n", () -> (e = element(Wb, Float64); Rₕ!(e, f); e)) :
                  (nothing, false)
    poly1 = use_poly && u_b_ok
    _bench!(
        rows, "Rₕ! unmasked", D, n, dofs, poly1, _agree_arr,
        parent(u_s), () -> Rₕ!(u_s, f),
        parent(u_t), () -> Rₕ!(u_t, f),
        poly1 ? parent(u_b) : nothing, poly1 ? (() -> Rₕ!(u_b, f)) : (() -> nothing)
    )

    # --- Rₕ! masked (:boundary, every mesh's own automatic marker) ---
    um_s = element(Ws, Float64)
    Rₕ!(um_s, f; markers = (:boundary,))
    um_t = element(Wt, Float64)
    Rₕ!(um_t, f; markers = (:boundary,))
    um_b, um_b_ok = use_poly ?
                    _poly_try(
        "Rₕ! masked D=$D n=$n", () -> (e = element(Wb, Float64); Rₕ!(e, f; markers = (:boundary,)); e)
    ) : (nothing, false)
    poly2 = use_poly && um_b_ok
    _bench!(
        rows, "Rₕ! masked", D, n, dofs, poly2, _agree_arr,
        parent(um_s), () -> Rₕ!(um_s, f; markers = (:boundary,)),
        parent(um_t), () -> Rₕ!(um_t, f; markers = (:boundary,)),
        poly2 ? parent(um_b) : nothing,
        poly2 ? (() -> Rₕ!(um_b, f; markers = (:boundary,))) : (() -> nothing)
    )

    # --- avgₕ! ---
    w_s = element(Ws, Float64)
    avgₕ!(w_s, f)
    w_t = element(Wt, Float64)
    avgₕ!(w_t, f)
    w_b, w_b_ok = use_poly ?
                  _poly_try("avgₕ! D=$D n=$n", () -> (e = element(Wb, Float64); avgₕ!(e, f); e)) :
                  (nothing, false)
    poly3 = use_poly && w_b_ok
    _bench!(
        rows, "avgₕ!", D, n, dofs, poly3, _agree_arr,
        parent(w_s), () -> avgₕ!(w_s, f),
        parent(w_t), () -> avgₕ!(w_t, f),
        poly3 ? parent(w_b) : nothing, poly3 ? (() -> avgₕ!(w_b, f)) : (() -> nothing)
    )

    # Serial/Threads operands shared across the four workloads below (neither policy
    # fails today, so sharing is safe); the Polyester operands are rebuilt from scratch
    # inside each workload's own `_poly_try`, below, so one workload's Polyester arm
    # raising can never affect another's -- arms must be independent (S6.1 review #1).
    fu_s = Rₕ(Ws, f)
    gv_s = Rₕ(Ws, g)
    fu_t = Rₕ(Wt, f)
    gv_t = Rₕ(Wt, g)

    # --- innerₕ ---
    ih_b, ih_b_ok = use_poly ?
                    _poly_try("innerₕ operands D=$D n=$n", () -> (Rₕ(Wb, f), Rₕ(Wb, g))) :
                    (nothing, false)
    poly_ih = use_poly && ih_b_ok
    ih_fu_b, ih_gv_b = poly_ih ? ih_b : (nothing, nothing)
    s_s = innerₕ(fu_s, gv_s)
    s_t = innerₕ(fu_t, gv_t)
    s_b = poly_ih ? innerₕ(ih_fu_b, ih_gv_b) : nothing
    _bench!(
        rows, "innerₕ", D, n, dofs, poly_ih, _agree_scalar,
        s_s, () -> innerₕ(fu_s, gv_s),
        s_t, () -> innerₕ(fu_t, gv_t),
        s_b, poly_ih ? (() -> innerₕ(ih_fu_b, ih_gv_b)) : (() -> nothing)
    )

    # --- innerₕ masked (:boundary) -- independent Polyester operands from innerₕ's ---
    ihm_b, ihm_b_ok = use_poly ?
                      _poly_try("innerₕ masked operands D=$D n=$n", () -> (Rₕ(Wb, f), Rₕ(Wb, g))) :
                      (nothing, false)
    poly_ihm = use_poly && ihm_b_ok
    ihm_fu_b, ihm_gv_b = poly_ihm ? ihm_b : (nothing, nothing)
    sm_s = innerₕ(fu_s, gv_s; markers = (:boundary,))
    sm_t = innerₕ(fu_t, gv_t; markers = (:boundary,))
    sm_b = poly_ihm ? innerₕ(ihm_fu_b, ihm_gv_b; markers = (:boundary,)) : nothing
    _bench!(
        rows, "innerₕ masked", D, n, dofs, poly_ihm, _agree_scalar,
        sm_s, () -> innerₕ(fu_s, gv_s; markers = (:boundary,)),
        sm_t, () -> innerₕ(fu_t, gv_t; markers = (:boundary,)),
        sm_b, poly_ihm ? (() -> innerₕ(ihm_fu_b, ihm_gv_b; markers = (:boundary,))) : (() -> nothing)
    )

    # --- D₋ₓ! (in place; now threaded under CpuThreaded and batched under
    # CpuPolyester -- #356) --
    # independent Polyester operand: kept independent so a failure there must
    # not gate any other workload's Polyester arm.
    dx_s = element(Ws, Float64)
    D₋ₓ!(dx_s, fu_s)
    dx_t = element(Wt, Float64)
    D₋ₓ!(dx_t, fu_t)
    dx_b, dx_b_ok = use_poly ?
                    _poly_try(
        "D₋ₓ! D=$D n=$n",
        () -> (fu_b = Rₕ(Wb, f); e = element(Wb, Float64); D₋ₓ!(e, fu_b); (e, fu_b))
    ) : (nothing, false)
    poly_dx = use_poly && dx_b_ok
    dx_e, dx_fu_b = poly_dx ? dx_b : (nothing, nothing)
    _bench!(
        rows, "D₋ₓ!", D, n, dofs, poly_dx, _agree_arr,
        parent(dx_s), () -> D₋ₓ!(dx_s, fu_s),
        parent(dx_t), () -> D₋ₓ!(dx_t, fu_t),
        poly_dx ? parent(dx_e) : nothing, poly_dx ? (() -> D₋ₓ!(dx_e, dx_fu_b)) : (() -> nothing)
    )

    # --- broadcast axpy: vₕ .= a .* uₕ .+ wₕ (now threaded/batched, #357) ---
    # independent Polyester operands from every workload above.
    α = 1.7
    ax_s = element(Ws, Float64)
    ax_s .= α .* fu_s .+ gv_s
    ax_t = element(Wt, Float64)
    ax_t .= α .* fu_t .+ gv_t
    ax_b, ax_b_ok = use_poly ?
                    _poly_try(
        "broadcast axpy D=$D n=$n",
        () -> begin
            ax_fu_b = Rₕ(Wb, f)
            ax_gv_b = Rₕ(Wb, g)
            e = element(Wb, Float64)
            e .= α .* ax_fu_b .+ ax_gv_b
            (e, ax_fu_b, ax_gv_b)
        end
    ) : (nothing, false)
    poly_ax = use_poly && ax_b_ok
    ax_e, ax_fu_b, ax_gv_b = poly_ax ? ax_b : (nothing, nothing, nothing)
    _bench!(
        rows, "broadcast axpy", D, n, dofs, poly_ax, _agree_exact,
        parent(ax_s), () -> (ax_s .= α .* fu_s .+ gv_s),
        parent(ax_t), () -> (ax_t .= α .* fu_t .+ gv_t),
        poly_ax ? parent(ax_e) : nothing,
        poly_ax ? (() -> (ax_e .= α .* ax_fu_b .+ ax_gv_b)) : (() -> nothing)
    )

    return rows
end

# --- Assembly workloads ------------------------------------------------------ #

function _run_assemble(D::Int, n::Int)
    use_poly = POLYESTER_OK
    Ws, Wt, Wb = _spaces(D, n, use_poly)
    dofs = ndofs(Ws)
    rows = Row[]

    # --- assemble bilinear first (a fresh matrix each call: record cost) ---
    Aref_s = assemble(form(Ws, Ws, _poisson))
    Aref_t = assemble(form(Wt, Wt, _poisson))
    Aref_b, Aref_b_ok = use_poly ?
                        _poly_try(
        "assemble bilinear first D=$D n=$n", () -> assemble(form(Wb, Wb, _poisson))
    ) : (nothing, false)
    poly_a1 = use_poly && Aref_b_ok
    _bench!(
        rows, "assemble bilinear first", D, n, dofs, poly_a1, _agree_mat,
        Aref_s, () -> assemble(form(Ws, Ws, _poisson)),
        Aref_t, () -> assemble(form(Wt, Wt, _poisson)),
        Aref_b, poly_a1 ? (() -> assemble(form(Wb, Wb, _poisson))) : (() -> nothing)
    )

    # --- assemble! bilinear (warmed: record once, then replay) ---
    a_s = form(Ws, Ws, _poisson)
    A_s = allocate_system_matrix(a_s)
    assemble!(A_s, a_s)
    a_t = form(Wt, Wt, _poisson)
    A_t = allocate_system_matrix(a_t)
    assemble!(A_t, a_t)
    ab_pair, ab_ok = use_poly ?
                     _poly_try(
        "assemble! bilinear D=$D n=$n",
        () -> (a = form(Wb, Wb, _poisson); A = allocate_system_matrix(a); assemble!(A, a); (a, A))
    ) : (nothing, false)
    poly_a2 = use_poly && ab_ok
    a_b, A_b = poly_a2 ? ab_pair : (nothing, nothing)
    _bench!(
        rows, "assemble! bilinear", D, n, dofs, poly_a2, _agree_mat,
        A_s, () -> assemble!(A_s, a_s),
        A_t, () -> assemble!(A_t, a_t),
        poly_a2 ? A_b : nothing, poly_a2 ? (() -> assemble!(A_b, a_b)) : (() -> nothing)
    )

    # --- assemble! linear (warmed) ---
    f = _f(Val(D))
    fh_s = Rₕ(Ws, f)
    l_s = form(Ws, v -> innerₕ(fh_s, v))
    b_s = zeros(dofs)
    assemble!(b_s, l_s)
    fh_t = Rₕ(Wt, f)
    l_t = form(Wt, v -> innerₕ(fh_t, v))
    b_t = zeros(dofs)
    assemble!(b_t, l_t)
    lb_pair, lb_ok = use_poly ?
                     _poly_try(
        "assemble! linear D=$D n=$n",
        () -> begin
            fh_b = Rₕ(Wb, f)
            l_b = form(Wb, v -> innerₕ(fh_b, v))
            b_b = zeros(dofs)
            assemble!(b_b, l_b)
            (l_b, b_b)
        end
    ) : (nothing, false)
    poly_a3 = use_poly && lb_ok
    l_b, b_b = poly_a3 ? lb_pair : (nothing, nothing)
    _bench!(
        rows, "assemble! linear", D, n, dofs, poly_a3, _agree_arr,
        b_s, () -> assemble!(b_s, l_s),
        b_t, () -> assemble!(b_t, l_t),
        poly_a3 ? b_b : nothing, poly_a3 ? (() -> assemble!(b_b, l_b)) : (() -> nothing)
    )

    return rows
end

# --- Context: assemble + solve, one size per D, not a crossover ------------- #

function _context_once(D::Int, n::Int)
    Random.seed!(SEED)
    be = backend(Float64; policy = Serial())
    Iᴰ = _unit_cube(Val(D))
    Ωd = domain(Iᴰ, :dir => boundary_symbols(Iᴰ))
    Ωₕ = _grid(Val(D), Ωd, n; backend = be)
    Wₕ = gridspace(Ωₕ)
    fₕ = Rₕ(Wₕ, _f(Val(D)))
    a = form(Wₕ, Wₕ, _poisson)
    l = form(Wₕ, v -> innerₕ(fₕ, v))
    t_a = (@elapsed ((A, F) = assemble(a, l; dirichlet = ZERO_BC, symmetrize = true))) * 1000
    t_s = (@elapsed (A \ F)) * 1000
    return t_a, t_s
end

function _run_context(D::Int)
    _context_once(D, 4)  # warm-up: pays JIT compilation on a throwaway tiny problem
    n = _n_for(D, CONTEXT_TARGET)
    t_a, t_s = _context_once(D, n)
    total = t_a + t_s
    share = total > 0 ? 100 * t_s / total : NaN
    _out()
    _out(
        "Context (dim=$D, n=$n): assemble+solve does not dispatch on the execution " *
        "policy -- `A \\ F` and the record/replay assembly path both run identically " *
        "regardless of any backend policy chosen elsewhere in this script, so this is one " *
        "measurement per dimension, not a crossover sweep.",
    )
    _out(
        "CONTEXT | assemble+solve | $(D)D | assemble=$(round(t_a; digits = 3)) | " *
        "solve=$(round(t_s; digits = 3)) | solve share=$(round(share; digits = 1))%",
    )
    return nothing
end

# --- Crossover: smallest DOF count confirmed by the next larger size ------- #

function _crossover_dofs(dofs::Vector{Int}, ratios::Vector{Float64})
    m = length(dofs)
    for i in 1:m
        isnan(ratios[i]) && continue
        if ratios[i] < 1
            if i == m || (!isnan(ratios[i + 1]) && ratios[i + 1] < 1)
                return dofs[i]
            end
        end
    end
    return nothing
end

_fmt_crossover(v) = v === nothing ? "none" : string(v)

function _print_workload_table(label::AbstractString, D::Int, rows::Vector{Row})
    isempty(rows) && return
    header = [
        "dofs", "n/axis", "serial (ms)", "threads (ms)", "polyester (ms)",
        "threads/serial", "polyester/serial"
    ]
    data = Matrix{Any}(undef, length(rows), length(header))
    for (i, r) in enumerate(rows)
        rt = r.t_threads / r.t_serial
        rb = r.t_poly / r.t_serial
        data[i, :] = [
            r.dofs, r.n, round(r.t_serial; digits = 5),
            isnan(r.t_threads) ? "-" : round(r.t_threads; digits = 5),
            isnan(r.t_poly) ? "-" : round(r.t_poly; digits = 5),
            isnan(rt) ? "-" : round(rt; digits = 3),
            isnan(rb) ? "-" : round(rb; digits = 3)
        ]
    end
    _out()
    _out("=== $label ($(D)D) ===")
    _print_table(data; column_labels = header)
    return nothing
end

function _print_crossover(label::AbstractString, D::Int, rows::Vector{Row})
    if isempty(rows)
        _out(
            "CROSSOVER | $label | $(D)D | threads=none | polyester=none | polyester-vs-threads=none",
        )
        return (workload = label, D = D, threads = nothing, polyester = nothing, pvt = nothing)
    end
    dofs = [r.dofs for r in rows]
    rt = [r.t_threads / r.t_serial for r in rows]
    rb = [r.t_poly / r.t_serial for r in rows]
    rpt = [r.t_poly / r.t_threads for r in rows]
    threads_x = _crossover_dofs(dofs, rt)
    poly_x = _crossover_dofs(dofs, rb)
    pvt_x = _crossover_dofs(dofs, rpt)
    _out(
        "CROSSOVER | $label | $(D)D | threads=$(_fmt_crossover(threads_x)) | " *
        "polyester=$(_fmt_crossover(poly_x)) | polyester-vs-threads=$(_fmt_crossover(pvt_x))",
    )
    # A "none" only hints at "raise --max-dofs" when the sweep actually produced usable
    # timings for that arm all the way to the cap and still never found a crossover --
    # not when the arm was never measured at all (Polyester unavailable, or every size
    # raised). Distinguishing those two "none"
    # reasons is exactly why arms must stay independent (S6.1 review #1): a workload with
    # zero valid Polyester timings prints no hint about the cap, since the cap was never
    # the reason.
    threads_measured = any(r -> !isnan(r.t_threads), rows)
    poly_measured = any(r -> !isnan(r.t_poly), rows)
    capped = String[]
    threads_x === nothing && threads_measured && push!(capped, "threads")
    poly_x === nothing && poly_measured && push!(capped, "polyester")
    if !isempty(capped)
        joined = join(capped, ", ")
        _out(
            "  hint: $joined crossover not found up to $(dofs[end]) dofs -- may lie above " *
            "--max-dofs=$(MAX_DOFS) (raise the cap to search further)",
        )
    end
    return (workload = label, D = D, threads = threads_x, polyester = poly_x, pvt = pvt_x)
end

function _print_summary(recs)
    _out()
    _out("=== Summary recommendation (per workload x dimension) ===")
    header = ["Workload", "D", "Use Serial below", "Use Polyester above", "Use Threads above"]
    data = Matrix{Any}(undef, length(recs), length(header))
    for (i, r) in enumerate(recs)
        best = filter(x -> x !== nothing, (r.threads, r.polyester))
        below = isempty(best) ? "no crossover found in sweep" : "$(minimum(best)) dofs"
        data[i, :] = [
            r.workload, "$(r.D)D", below,
            r.polyester === nothing ? "n/a" : "$(r.polyester) dofs",
            r.threads === nothing ? "n/a" : "$(r.threads) dofs"
        ]
    end
    _print_table(data; column_labels = header)
    return nothing
end

# --- Driver ------------------------------------------------------------------ #

function main()
    cheap_rows = Row[]
    assemble_rows = Row[]
    for D in 1:3
        for target in CHEAP_TARGETS
            append!(cheap_rows, _run_cheap(D, _n_for(D, target)))
        end
        for target in ASSEMBLE_TARGETS
            append!(assemble_rows, _run_assemble(D, _n_for(D, target)))
        end
    end

    cheap_labels = (
        "Rₕ! unmasked", "Rₕ! masked", "avgₕ!", "innerₕ", "innerₕ masked", "D₋ₓ!", "broadcast axpy"
    )
    assemble_labels = ("assemble bilinear first", "assemble! bilinear", "assemble! linear")

    recs = []
    for label in cheap_labels, D in 1:3
        rows = filter(r -> r.workload == label && r.D == D, cheap_rows)
        _print_workload_table(label, D, rows)
        push!(recs, _print_crossover(label, D, rows))
    end
    for label in assemble_labels, D in 1:3
        rows = filter(r -> r.workload == label && r.D == D, assemble_rows)
        _print_workload_table(label, D, rows)
        push!(recs, _print_crossover(label, D, rows))
    end

    for D in 1:3
        _run_context(D)
    end

    _print_summary(recs)

    _out()
    if ALL_OK[]
        _out("OK-POLICY-CROSSOVER")
    else
        _out("One or more arms MISMATCHED serial output -- see MISMATCH lines above.")
    end

    if OUT_PATH !== nothing
        open(OUT_PATH, "w") do io
            write(io, String(take!(MD_BUF)))
        end
        _out("Markdown report written to $OUT_PATH")
    end

    return nothing
end

main()
