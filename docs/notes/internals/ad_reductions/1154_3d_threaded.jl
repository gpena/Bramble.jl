# Reproducer for gpena/Bramble.jl#1154: the `assemble_averaged_3d` cell of
# test/space/ad_coverage.jl (`build_assemble_averaged_3d`: `inner₊(Mₕ(κ) * ∇ₕ(U), ∇ₕ(V))` on a
# non-uniform 5 x 5 x 5 mesh), one execution policy and one mode per process, at Enzyme's
# default `maxtypeoffset` (the test's `Enzyme.API.maxtypeoffset!(1024)` is not set here).
#
# Run from the repository root, with two threads (the threaded policies need more than one):
#
#     /usr/bin/time -l .claude/scripts/julia_slot.sh 2 julia --startup-file=no --project=. \
#         --threads=2 --optimize=1 --check-bounds=auto \
#         docs/notes/internals/ad_reductions/1154_3d_threaded.jl <policy> <mode>
#
# `<policy>` is CpuSerial, CpuThreaded or CpuPolyester; `<mode>` is rev or fwd. The mesh, the
# basis and the loss are the cell's (`_mesh3d`, `_basis`, `_field`, `build_assemble_averaged_3d`),
# built here so the cell runs without the test file's child, which skips `:skip` cells and hides
# which call failed.
#
# The reference is ForwardDiff on the same policy, through DifferentiationInterface as the test
# does. A cell is `ok` when its relative error, the test's `_relerr` (the largest absolute
# difference over the largest absolute reference), is at most 1e-8; `fails` when Enzyme throws;
# `wrong` when it returns a gradient off that bound.
#
# Enzyme and ForwardDiff are weak dependencies of the root project, so they are loaded from a
# temporary environment, as 1157_ghost_load.jl does. The root Manifest.toml must hold them.
#
# Set `BRAMBLE_VERBOSE_ERRORS=1` to turn on `Enzyme.Compiler.VERBOSE_ERRORS[]` and print the
# whole error message, which names the value and the instruction Enzyme cannot type.
#
# Output: `step <name> t=<seconds>` lines as the cell runs, then one line
# `RESULT <policy> <mode> <status> <relerr> <seconds> <detail>`. Exit status is 0 for `ok` and
# 1 otherwise. `/usr/bin/time -l` adds the peak (`maximum resident set size`, in bytes).
println("pid ", getpid())
flush(stdout)
_e = mktempdir()
write(joinpath(_e, "Project.toml"),
    "[deps]\n" *
    "DifferentiationInterface = \"a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63\"\n" *
    "Enzyme = \"7da242da-08ed-463a-9acd-ee780be4f1d9\"\n" *
    "ForwardDiff = \"f6369f11-7733-5829-9624-2563aa707210\"\n" *
    "Polyester = \"f517fe37-dbe3-4b94-8317-1923a5111588\"\n")
push!(LOAD_PATH, _e)
using Bramble, DifferentiationInterface, Enzyme, ForwardDiff, Polyester, Random, SparseArrays
using Logging

const VERBOSE = get(ENV, "BRAMBLE_VERBOSE_ERRORS", "") == "1"
VERBOSE && (Enzyme.Compiler.VERBOSE_ERRORS[] = true)

# Enzyme warns once per stored value it cannot find a use for, thousands of lines of IR.
global_logger(NullLogger())

const T0 = time()
step(name) = (println("step ", name, " t=", round(time() - T0; digits = 1)); flush(stdout))

length(ARGS) == 2 || error("usage: 1154_3d_threaded.jl <policy> <rev|fwd>")
const POLICY_NAME, MODE = ARGS
const POLICIES = Dict("CpuSerial" => Serial(), "CpuThreaded" => Parallel(),
    "CpuPolyester" => Bramble.CpuPolyester())
const POLICY = POLICIES[POLICY_NAME]
MODE in ("rev", "fwd") || error("mode must be rev or fwd, not ", MODE)
Threads.nthreads() >= 2 || POLICY_NAME == "CpuSerial" ||
    error("a threaded policy needs --threads=2 or more")

const SEED = 1234
const NP = 3
_basis(n) = randn(MersenneTwister(SEED), n, NP)
_start() = [0.5 + 0.3 * sin(k) for k in 1:NP]

# The test's `_mesh3d(policy, :nonuniform)`.
function _mesh3d(policy)
    Bramble._seed_mesh1d_rng!(SEED)
    try
        return mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0)),
            (5, 5, 5), (false, false, false); backend = backend(policy = policy))
    finally
        Bramble._unseed_mesh1d_rng!()
    end
end

# A grid function holding `shift + B p`.
function _field(W, B, p::AbstractVector{T}; shift = 0.0) where {T}
    u = element(W, T)
    parent(u) .= shift .+ B * p
    return u
end

# The cell's `build_assemble_averaged_3d`, `Mₕ(κ)` on a 3D mesh.
function build_assemble_averaged_3d(policy)
    W = gridspace(_mesh3d(policy))
    B = _basis(ndofs(W))
    function f(p)
        κ = _field(W, B, p; shift = 2.0)
        return sum(abs2, nonzeros(assemble(form(W, W, (U, V) -> inner₊(Mₕ(κ) * ∇ₕ(U), ∇ₕ(V))))))
    end
    return f, _start()
end
const loss, P0 = build_assemble_averaged_3d(POLICY)

const AD = if MODE == "rev"
    AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
        function_annotation = Enzyme.Const)
else
    AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Forward),
        function_annotation = Enzyme.Const)
end

println("cell ", POLICY_NAME, " ", MODE, ", Enzyme ", pkgversion(Enzyme), ", Julia ", VERSION,
    ", threads ", Threads.nthreads(), ", optimize ", Base.JLOptions().opt_level,
    ", check-bounds ", Base.JLOptions().check_bounds, " (0 auto, 1 yes, 2 no)",
    ", maxtypeoffset default")
flush(stdout)

ref = DifferentiationInterface.gradient(loss, AutoForwardDiff(), P0)
step("reference")
if minimum(abs, ref) <= 1e-6 * maximum(abs, ref)
    println("RESULT ", POLICY_NAME, " ", MODE, " weakref NaN ", round(time() - T0; digits = 1),
        " reference ", ref)
    exit(1)
end

# Top-level `try` blocks do not hand their assignments back, so the cell runs in a function.
function measure(ref, t_ref)
    status, relerr, detail, g = "ok", NaN, "", Float64[]
    step("enzyme start")
    try
        g = DifferentiationInterface.gradient(loss, AD, P0)
        relerr = maximum(abs, g .- ref) / maximum(abs, ref)
        relerr <= 1e-8 || (status = "wrong")
        step("enzyme done")
    catch err
        err isa InterruptException && rethrow()
        status = "fails"
        msg = sprint(showerror, err)
        detail = string(nameof(typeof(err)), ": ", first(replace(msg, '\n' => ' '), 400))
        println(VERBOSE ? msg : first(msg, 1500))
        frames = filter(l -> occursin(r"^ *(\[\d+\]|@ |cannot handle|No create)", l),
            split(msg, '\n'))
        println(join(first(frames, 40), '\n'))
        step("enzyme failed")
    end
    return (; status, relerr, detail, g, secs = time() - T0 - t_ref)
end

# `secs` counts from the reference to the end of the Enzyme call, compile included, the same
# span as the note's `Compile:` line.
r = measure(ref, time() - T0)
println("RESULT ", POLICY_NAME, " ", MODE, " ", r.status, " ",
    isnan(r.relerr) ? "NaN" : string(r.relerr), " ", round(r.secs; digits = 1), " ", r.detail)
println("enzyme gradient ", isempty(r.g) ? "none" : string(r.g), " reference ", ref)
flush(stdout)
exit(r.status == "ok" ? 0 : 1)
