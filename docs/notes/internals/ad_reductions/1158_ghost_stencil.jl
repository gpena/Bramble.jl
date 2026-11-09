# Reproducer for gpena/Bramble.jl#1158: Enzyme on `centered_difference!` of a `GhostVector`.
#
# The cell `build_ghost_runtime` of test/space/ad_coverage.jl with
# `centered_difference!(out, v, Val(1))` in place of `Δₕ!`: the same ghost space (7 by 6
# non-uniform mesh, Neumann and Robin sides), the same parameters and basis, and the
# gradient taken the way the coverage test takes it, through DifferentiationInterface with
# `AutoEnzyme(mode = set_runtime_activity(...), function_annotation = Const)`, against
# ForwardDiff on the same policy.
#
# Run from the repository root, with Bramble from the root project:
#
#     julia --project=. --threads=2 --optimize=1 --check-bounds=auto \
#         docs/notes/internals/ad_reductions/1158_ghost_stencil.jl [variant]
#
# Enzyme, DifferentiationInterface, ForwardDiff and Polyester are weak dependencies of
# Bramble, so they are loaded from a temporary environment, the way the test harness loads
# Enzyme. The root Manifest.toml must hold them.
#
# A variant is `policy_mode[_dim][_op]`: policy `CpuSerial`, `CpuThreaded` or
# `CpuPolyester`; mode `rev` or `fwd`; dim `1` (the default) or `2`, the `Val` of the
# stencil; op `centered` (the default) or `laplacian`, `Δₕ!` as in the coverage test's own
# `ghost_runtime` cell, a control. The default variant is `CpuSerial_rev`.
#
# Exit status: 0 when the gradient matches ForwardDiff to 1e-8 relative, 1 when Enzyme
# throws or the gradient is wrong. The script prints the gradient call's time twice, the
# first with the compile, and on a throw the Bramble frames of the backtrace.
_e = mktempdir()
write(joinpath(_e, "Project.toml"),
    """
    [deps]
    DifferentiationInterface = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Polyester = "f517fe37-dbe3-4b94-8317-1923a5111588"
    """)
push!(LOAD_PATH, _e)
println("pid ", getpid())
flush(stdout)
using Bramble, Enzyme, DifferentiationInterface, ForwardDiff, Polyester
using Random, SparseArrays, Logging
using Bramble: Δₕ!, GhostVector, update_ghosts!, CpuPolyester, centered_difference!

# Enzyme warns once per stored value it cannot find a use for, thousands of lines of IR.
global_logger(NullLogger())

const VARIANT = isempty(ARGS) ? "CpuSerial_rev" : only(ARGS)
const PARTS = split(VARIANT, "_")
const POLICY = PARTS[1]
const MODE = PARTS[2]
const DIM = something(tryparse(Int, get(PARTS, 3, "1")), 1)
const OP = PARTS[end] in ("centered", "laplacian") ? PARTS[end] : "centered"
const POLICIES = Dict("CpuSerial" => Serial(), "CpuThreaded" => Parallel(),
    "CpuPolyester" => CpuPolyester())
MODE in ("rev", "fwd") || error("unknown mode ", MODE)

# The test harness's `ghost_mesh(policy, (7, 6))` and `_ghost_space`.
function ghost_space(pol)
    Bramble._seed_mesh1d_rng!(11)
    be = backend(vector_type = Vector{Float64}, matrix_type = SparseMatrixCSC{Float64, Int},
        policy = pol)
    Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (7, 6), (false, false);
        backend = be)
    Bramble._unseed_mesh1d_rng!()
    return constrained(gridspace(Ωₕ; ghosts = (:xmin, :xmax, :ymax)),
        u -> (:xmin => ∂ₙ(u) => 0.0, :xmax => 2.0 * u + ∂ₙ(u) => 1.0,
            :ymin => u => 0.0, :ymax => ∂ₙ(u) => 1.0))
end

const C = ghost_space(POLICIES[POLICY])
const B = randn(MersenneTwister(1234), 42, 3)
const x0 = [0.5 + 0.3 * sin(k) for k in 1:3]

apply!(out, v, ::Val{:centered}) = centered_difference!(out, v, Val(DIM))
apply!(out, v, ::Val{:laplacian}) = Δₕ!(out, v)

function f(p)
    v = update_ghosts!(GhostVector(C, B * p), C)
    out = zeros(eltype(p), 42)
    apply!(out, v, Val(Symbol(OP)))
    return sum(abs2, out)
end

println("variant ", VARIANT, " (", OP, ", Val(", DIM, ")), Enzyme ", pkgversion(Enzyme),
    ", Julia ", VERSION, ", threads ", Threads.nthreads(), ", optimize ",
    Base.JLOptions().opt_level, ", check-bounds ", Base.JLOptions().check_bounds,
    " (0 auto, 1 yes, 2 no), BrambleEnzymePolyesterExt ",
    Base.get_extension(Bramble, :BrambleEnzymePolyesterExt) !== nothing)
flush(stdout)

ref = DifferentiationInterface.gradient(f, AutoForwardDiff(), x0)
minimum(abs, ref) > 1e-6 * maximum(abs, ref) || error("weak reference ", ref)
m = Enzyme.set_runtime_activity(MODE == "rev" ? Enzyme.Reverse : Enzyme.Forward)
ad = AutoEnzyme(; mode = m, function_annotation = Enzyme.Const)

failed = false
t1 = @elapsed try
    g = DifferentiationInterface.gradient(f, ad, x0)
    t2 = @elapsed DifferentiationInterface.gradient(f, ad, x0)
    r = maximum(abs, g .- ref) / maximum(abs, ref)
    global failed = !(isfinite(r) && r <= 1e-8)
    println(failed ? "WRONG" : "OK", ": gradient ", g, ", ForwardDiff ", ref,
        ", relerr ", r, ", second call ", round(t2; digits = 3), " s")
catch e
    global failed = true
    msg = sprint(showerror, e)
    println("FAILURE: ", typeof(e))
    println(first(msg, 400))
    keep = filter(l -> occursin(r"^ *(\[\d+\]|@ |Failure within|cannot handle|No create)", l),
        split(msg, '\n'))
    println(join(first(keep, 30), '\n'))
    bt = catch_backtrace()
    frames = filter(fr -> occursin("Bramble", string(fr.file)), stacktrace(bt))
    println("Bramble frames:")
    foreach(fr -> println("  ", fr.func, " at ", fr.file, ":", fr.line), first(frames, 20))
end
println("first call seconds ", round(t1; digits = 1))
exit(failed ? 1 : 0)
