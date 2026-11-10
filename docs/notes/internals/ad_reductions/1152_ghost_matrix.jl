# Reproducer for gpena/Bramble.jl#1152: the `ghost_assemble` cell of test/space/ad_coverage.jl
# (`build_ghost_assemble`: a `Δₕ` form on a ghost space with Neumann and Robin sides),
# one execution policy and one mode per process.
#
# Run from the repository root, with two threads (the threaded policies need more than one):
#
#     julia --startup-file=no --project=. --threads=2 --optimize=1 --check-bounds=auto \
#         docs/notes/internals/ad_reductions/1152_ghost_matrix.jl <policy> <mode>
#
# `<policy>` is CpuSerial, CpuThreaded or CpuPolyester; `<mode>` is rev or fwd. The space and
# the loss are the cell's (`_ghost_space` and `build_ghost_assemble`), built here so the cell
# runs without the test file's child, which hides the failing call.
#
# The reference is ForwardDiff on the same policy, through DifferentiationInterface as the
# test does. A cell is `ok` when its relative error, the test's `_relerr` (the largest
# absolute difference over the largest absolute reference), is at most 1e-8; `fails` when
# Enzyme throws; `wrong` when it returns a gradient off that bound.
#
# Enzyme and ForwardDiff are weak dependencies of the root project, so they are loaded from a
# temporary environment, as 1157_ghost_load.jl does. The root Manifest.toml must hold them.
#
# Set `BRAMBLE_VERBOSE_ERRORS=1` to turn on `Enzyme.Compiler.VERBOSE_ERRORS[]` and print the
# whole error message, which names the value and the instruction Enzyme cannot type.
#
# Output: `step <name> t=<seconds>` lines as the cell runs, then one line
# `RESULT <policy> <mode> <status> <relerr> <seconds> <detail>`. Exit status is 0 for `ok` and
# 1 otherwise.
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
using Bramble, DifferentiationInterface, Enzyme, ForwardDiff, Polyester, SparseArrays
using Logging

const VERBOSE = get(ENV, "BRAMBLE_VERBOSE_ERRORS", "") == "1"
VERBOSE && (Enzyme.Compiler.VERBOSE_ERRORS[] = true)

# Enzyme warns once per stored value it cannot find a use for, thousands of lines of IR.
global_logger(NullLogger())

const T0 = time()
step(name) = (println("step ", name, " t=", round(time() - T0; digits = 1)); flush(stdout))

length(ARGS) == 2 || error("usage: 1152_ghost_matrix.jl <policy> <rev|fwd>")
const POLICY_NAME, MODE = ARGS
const POLICIES = Dict("CpuSerial" => Serial(), "CpuThreaded" => Parallel(),
    "CpuPolyester" => Bramble.CpuPolyester())
const POLICY = POLICIES[POLICY_NAME]
MODE in ("rev", "fwd") || error("mode must be rev or fwd, not ", MODE)

# The test harness's `ghost_mesh(policy, (7, 6))`, copied from test/TestUtils.jl.
function ghost_mesh(pol, ns::NTuple{D, Int}; T = Float64, seed = 11) where {D}
    Bramble._seed_mesh1d_rng!(seed)
    be = backend(vector_type = Vector{T}, matrix_type = SparseMatrixCSC{T, Int},
        policy = pol)
    Ωd = domain(foldl(×, ntuple(d -> interval(T(0), T(d)), D)))
    Ω = mesh(Ωd, D == 1 ? only(ns) : ns, D == 1 ? false : ntuple(_ -> false, D);
        backend = be)
    Bramble._unseed_mesh1d_rng!()
    return Ω
end

# The cell's space, as `_ghost_space(policy, :nonuniform)`.
const C = constrained(gridspace(ghost_mesh(POLICY, (7, 6)); ghosts = (:xmin, :xmax, :ymax)),
    u -> (:xmin => ∂ₙ(u) => 0.0, :xmax => 2.0 * u + ∂ₙ(u) => 1.0,
        :ymin => u => 0.0, :ymax => ∂ₙ(u) => 1.0))
# The cell's load, built outside the differentiated function as `build_ghost_assemble` does.
const L = form(C, v -> innerₕ(x -> 1.0 + x[1], v))

# The cell's `f` of `build_ghost_assemble`, behind a builder function as the tests build it.
function build_ghost_assemble(C, l)
    function f(p)
        a = form(C, C, (U, V) -> p[1] * innerₕ(-Δₕ(U), V) + p[2]^2 * innerₕ(U, V))
        A, F = assemble(a, l)
        return sum(abs2, nonzeros(A)) + 1e4 * p[3] * sum(F)
    end
    return f
end
const loss = build_ghost_assemble(C, L)

const P0 = [0.5 + 0.3 * sin(k) for k in 1:3]
const AD = if MODE == "rev"
    AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
        function_annotation = Enzyme.Const)
else
    AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Forward),
        function_annotation = Enzyme.Const)
end

println("cell ", POLICY_NAME, " ", MODE, ", Enzyme ", pkgversion(Enzyme), ", Julia ", VERSION,
    ", threads ", Threads.nthreads(), ", optimize ", Base.JLOptions().opt_level,
    ", check-bounds ", Base.JLOptions().check_bounds, " (0 auto, 1 yes, 2 no)")
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
