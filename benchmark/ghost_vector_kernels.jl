#===========================================================================#
# The stencil cost over a `GhostVector` against the contiguous kernel
# (gpena/Bramble.jl#776, third criterion).
#
# Run from a worktree root, alone on a quiet machine:
#
#     julia --project=test --threads=4 benchmark/ghost_vector_kernels.jl
#
# For each size (2D 256² and 1024², 3D 64³) it builds a seeded non-uniform mesh with ghosts
# on every facet and times, behind a function barrier, `op!(out, v::GhostVector)` against
# `op!(vₕ, uₕ)` on `gridspace(Ωₕ)`, for `Δₕ!`, `D₋ₓ!`, `Dcₓ!` and `D̽ₓ!`. A figure is the
# median over five array placements of the minimum of many `time_ns` samples per placement,
# the two kernels sampled alternately so drift hits both.
# Each kernel is asserted to allocate 0 B after warm-up (CpuSerial and CpuPolyester; the
# threaded policy spawns, so its bytes are printed, not asserted). Per row it prints
#
#     ROW<TAB>dims<TAB>op<TAB>ghost_ns<TAB>plain_ns<TAB>ratio<TAB>vec_ghost<TAB>vec_plain
#
# where `vec_*` counts the lines of the kernel's optimised LLVM IR (whole module) holding a
# `<N x double>` vector type. `GHOST_KERNELS_SMOKE=1` shrinks it to 16² and 8³ with two
# samples: it checks that the script runs, not a figure. `GHOST_KERNELS_POLICY` (serial,
# threaded) picks the mesh policy; the default is the backend's default policy.
#===========================================================================#
using Bramble
using Bramble: D₋ₓ!, Dcₓ!, D̽ₓ!, Δₕ!, CpuSerial, CpuThreaded
using InteractiveUtils: code_llvm
using Random: MersenneTwister
using SparseArrays: SparseMatrixCSC

const SMOKE = get(ENV, "GHOST_KERNELS_SMOKE", "") == "1"
# -O1 (the suite's level) changes which loops vectorise, so a timing run refuses it.
SMOKE || Base.JLOptions().opt_level >= 2 ||
    throw(ArgumentError("run without --optimize=1: timings below the default level are not figures"))
const SIZES = SMOKE ? [(16, 16), (8, 8, 8)] : [(256, 256), (1024, 1024), (64, 64, 64)]
const LAYOUTS = SMOKE ? 2 : 5
const MIN_SAMPLES = SMOKE ? 2 : 20
const BUDGET_NS = SMOKE ? 0 : 300_000_000
const OPS = (("Δₕ!", Δₕ!, false), ("D₋ₓ!", D₋ₓ!, true), ("Dcₓ!", Dcₓ!, true),
    ("D̽ₓ!", D̽ₓ!, true))

function make_mesh(ns::NTuple{D, Int}) where {D}
    pol = get(ENV, "GHOST_KERNELS_POLICY", "")
    kw = pol == "serial" ? (; backend = Bramble.backend(policy = CpuSerial())) :
         pol == "threaded" ? (; backend = Bramble.backend(policy = CpuThreaded())) : (;)
    Bramble._seed_mesh1d_rng!(11)
    Ωd = domain(foldl(×, ntuple(d -> interval(0.0, Float64(d)), D)))
    Ω = mesh(Ωd, ns, ntuple(_ -> false, D); kw...)
    Bramble._unseed_mesh1d_rng!()
    return Ω
end

# One call of `op` on the barrier's arguments, so each sample times a call and nothing else.
@noinline run_ghost!(op::F, out, v) where {F} = op(out, v)
@noinline run_plain!(op::F, out, u) where {F} = op(out, u)

function sample(f::F, out, x) where {F}
    t0 = time_ns()
    f(out, x)
    return time_ns() - t0
end

# Minimum time of each kernel over alternating samples: at least `MIN_SAMPLES` each and
# until `BUDGET_NS` has passed.
function race(opg::F, outg, v, opp::G, outp, u) where {F, G}
    tg = tp = typemax(UInt64)
    start = time_ns()
    n = 0
    while n < MIN_SAMPLES || time_ns() - start < BUDGET_NS
        tg = min(tg, sample((o, x) -> run_ghost!(opg, o, x), outg, v))
        tp = min(tp, sample((o, x) -> run_plain!(opp, o, x), outp, u))
        n += 1
    end
    return Float64(tg), Float64(tp)
end

alloc(f::F, op::O, a, b) where {F, O} = (f(op, a, b); f(op, a, b); @allocated f(op, a, b))

function vector_lines(f, types)
    io = IOBuffer()
    code_llvm(io, f, types; debuginfo = :none, optimize = true, dump_module = true)
    return count(l -> occursin(r"<\d+ x double>", l), eachline(IOBuffer(String(take!(io)))))
end

# The operands of one placement: every array is allocated after a pad of `k` pages of 16 KiB,
# so the layouts land at different address offsets (one fixed placement can make a kernel
# look 2x slower from cache aliasing alone, #475). `keep` holds the pads against the GC.
function operands(W, Wp, np, k, keep)
    rng = MersenneTwister(4321)
    push!(keep, zeros(UInt8, 16384 * k))
    v = GhostVector(W, rand(rng, np))
    Bramble.ghost_values(v) .= rand(rng, Bramble.nghosts(mesh(W)))
    out = zeros(np)
    push!(keep, zeros(UInt8, 16384 * (k + 1)))
    return v, out, element(Wp, rand(rng, np)), element(Wp)
end

median(x) = sort(x)[(length(x) + 1) ÷ 2]

for ns in SIZES
    Ω = make_mesh(ns)
    W = gridspace(Ω; ghosts = :boundary)
    Wp = gridspace(Ω)
    np = prod(ns)
    dims = join(ns, "x")
    pol = Bramble.execution_policy(Bramble.backend(Ω))
    println("# mesh ", dims, " policy ", typeof(pol), " threads ", Threads.nthreads(), " ghosts ",
        Bramble.nghosts(mesh(W)), " layouts ", LAYOUTS)
    keep = Any[]
    ops = [operands(W, Wp, np, k, keep) for k in 1:LAYOUTS]
    for (name, op, _) in OPS
        v, out, uₕ, outₕ = ops[1]
        bg = alloc(run_ghost!, op, out, v)
        bp = alloc(run_plain!, op, outₕ, uₕ)
        if !(pol isa CpuThreaded)
            (bg == 0 && bp == 0) || error("$name on $dims allocates: ghost $bg B, plain $bp B")
        else
            println("# ", name, " threaded bytes: ghost ", bg, " plain ", bp)
        end
        mins = [race(op, out, v, op, outₕ, uₕ) for (v, out, uₕ, outₕ) in ops]
        tg, tp = median(first.(mins)), median(last.(mins))
        vg = vector_lines(op, Tuple{typeof(ops[1][2]), typeof(ops[1][1])})
        vp = vector_lines(op, Tuple{typeof(ops[1][4]), typeof(ops[1][3])})
        println("ROW\t", dims, "\t", name, "\t", tg, "\t", tp, "\t", round(tg / tp; digits = 4),
            "\t", vg, "\t", vp)
    end
end
