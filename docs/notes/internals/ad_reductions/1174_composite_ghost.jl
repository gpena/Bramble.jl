# Reproducer for gpena/Bramble.jl#1174: Enzyme reverse of the ghost `Δₕ` assembly on the
# composite space `C^2` of a 2D ghost space, under `CpuSerial`, did not finish compiling.
# One cell and one mode per process, with a wall-clock limit the script enforces itself.
#
# Run from the repository root, with one thread, at Julia's default optimisation level:
#
#     julia --startup-file=no --project=<checkout> --threads=1 \
#         docs/notes/internals/ad_reductions/1174_composite_ghost.jl <cell> <rev|fwd|ref>
#
# `<checkout>` is the Bramble tree under test (`.` for this one, a scratch worktree of the
# base commit for the other); the script itself is the same file for both.
#
# Cells, all on the (6, 5) non-uniform mesh of the test harness (seed 11) with ghosts on the
# xmin and ymax sides, a Neumann side, a Robin side and two Dirichlet sides. That ghost space
# `C` is the one of cell `comp` in the v4-2-4 critic's script.
#   comp    the issue's cell, `V = C^2` with `p[1] * innerₕ(-Δₕ(u(1)), v(1))
#           + p[2]^2 * innerₕ(-Δₕ(u(2)), v(2)) + innerₕ(u(2), v(1))`, assembled with the
#           load `innerₕ(x -> x[1] + 1.0, v(1)) + innerₕ(x -> 1.0, v(2))`.
#   comp1   `C^2` with `Δₕ` on component 1 only, `p[1] * innerₕ(-Δₕ(u(1)), v(1))
#           + p[2]^2 * innerₕ(u(2), v(2)) + innerₕ(u(2), v(1))`, same load.
#   diag    `C^2` without the coupling, `p[1] * innerₕ(-Δₕ(u(1)), v(1))
#           + p[2]^2 * innerₕ(-Δₕ(u(2)), v(2))`, same load.
#   one     the control, the scalar space `C` with `p[1] * innerₕ(-Δₕ(U), V)
#           + p[2]^2 * innerₕ(U, V)` and the load `innerₕ(x -> x[1] + 1.0, v)`.
# The loss is `sum(abs2, nonzeros(A)) + 1e4 * p[3] * sum(F)` for `A, F = assemble(a, l)`,
# with the form built from the parameter inside the differentiated function so its scalars
# are active.
#
# Mode `ref` runs the ForwardDiff reference only (no Enzyme call), a check that the cell
# builds on the tree under test. Mode `infer`, no Enzyme call either, prints the type
# inference gives `simplify_ast` (src/ast/simplifier.jl) on the cell's resolved sum, alone
# as Enzyme compiles it, and the type it gives `loss`.
#
# The limit is `S1174_LIMIT` seconds (default 1200). With one thread a Julia task cannot
# interrupt a compile that never yields, so the script starts a shell watchdog on its own
# pid, which prints `WATCHDOG timeout` to stderr and sends SIGKILL at the limit, and exits
# within a second when the run ends first.
#
# Enzyme and ForwardDiff are weak dependencies of the root project: they load from a
# temporary environment, as 1177_ghost_3d.jl does.
#
# Output: `step <name> t=<seconds>` lines, then
# `RESULT <cell> <mode> <status> <relerr> <enzyme seconds> <detail>`. The enzyme seconds
# are the first `gradient` call alone, its compile included. Exit 0 for `ok`. A timeout or
# a kill prints no RESULT line.
const LIMIT = parse(Int, get(ENV, "S1174_LIMIT", "1200"))
println("pid ", getpid(), " limit ", LIMIT, " s")
flush(stdout)
const WATCHDOG_SH = "i=0; while [ \$i -lt $(LIMIT) ]; do sleep 1; " *
                    "kill -0 $(getpid()) 2>/dev/null || exit 0; i=\$((i + 1)); done; " *
                    "echo WATCHDOG timeout >&2; kill -9 $(getpid())"
run(pipeline(`sh -c $WATCHDOG_SH`; stderr = stderr); wait = false)

_e = mktempdir()
write(joinpath(_e, "Project.toml"),
    "[deps]\n" *
    "DifferentiationInterface = \"a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63\"\n" *
    "Enzyme = \"7da242da-08ed-463a-9acd-ee780be4f1d9\"\n" *
    "ForwardDiff = \"f6369f11-7733-5829-9624-2563aa707210\"\n")
push!(LOAD_PATH, _e)
using Bramble, DifferentiationInterface, Enzyme, ForwardDiff, SparseArrays
using Logging
const B = Bramble

# Enzyme warns once per stored value it cannot find a use for, thousands of lines of IR.
global_logger(NullLogger())

const T0 = time()
step(name) = (println("step ", name, " t=", round(time() - T0; digits = 1)); flush(stdout))

length(ARGS) == 2 ||
    error("usage: 1174_composite_ghost.jl <comp|comp1|diag|one> <rev|fwd|ref|infer>")
const CELL, MODE = ARGS
CELL in ("comp", "comp1", "diag", "one") ||
    error("cell must be comp, comp1, diag or one, not ", CELL)
MODE in ("rev", "fwd", "ref", "infer") ||
    error("mode must be rev, fwd, ref or infer, not ", MODE)

# The test harness's `ghost_mesh(policy, ns)`, copied from test/TestUtils.jl.
function ghost_mesh(pol, ns::NTuple{D, Int}) where {D}
    B._seed_mesh1d_rng!(11)
    be = backend(vector_type = Vector{Float64}, matrix_type = SparseMatrixCSC{Float64, Int},
        policy = pol)
    Ω = mesh(domain(foldl(×, ntuple(d -> interval(0.0, Float64(d)), D))), ns,
        ntuple(_ -> false, D); backend = be)
    B._unseed_mesh1d_rng!()
    return Ω
end

const C = constrained(
    gridspace(ghost_mesh(B.CpuSerial(), (6, 5)); ghosts = (:xmin, :ymax)),
    u -> (:xmin => ∂ₙ(u) => 1.0, :ymax => u + ∂ₙ(u) => 0.0, :xmax => u => 0.0,
        :ymin => u => 0.0))
const W = CELL == "one" ? C : C^2
const L = CELL == "one" ? form(W, v -> innerₕ(x -> x[1] + 1.0, v)) :
          form(W, v -> innerₕ(x -> x[1] + 1.0, v(1)) + innerₕ(x -> 1.0, v(2)))
bilinear(p) =
    if CELL == "comp"
        (u, v) -> p[1] * innerₕ(-Δₕ(u(1)), v(1)) + p[2]^2 * innerₕ(-Δₕ(u(2)), v(2)) +
                  innerₕ(u(2), v(1))
    elseif CELL == "comp1"
        (u, v) -> p[1] * innerₕ(-Δₕ(u(1)), v(1)) + p[2]^2 * innerₕ(u(2), v(2)) +
                  innerₕ(u(2), v(1))
    elseif CELL == "diag"
        (u, v) -> p[1] * innerₕ(-Δₕ(u(1)), v(1)) + p[2]^2 * innerₕ(-Δₕ(u(2)), v(2))
    else
        (U, V) -> p[1] * innerₕ(-Δₕ(U), V) + p[2]^2 * innerₕ(U, V)
    end

function loss(p)
    a = form(W, W, bilinear(p))
    A, F = assemble(a, L)
    return sum(abs2, nonzeros(A)) + 1e4 * p[3] * sum(F)
end

const P0 = [0.5 + 0.3 * sin(k) for k in 1:3]

println("cell ", CELL, " ", MODE, ", ndofs ", ndofs(W), ", Enzyme ", pkgversion(Enzyme),
    ", Bramble ", pkgversion(Bramble), " at ", pkgdir(Bramble), ", Julia ", VERSION,
    ", threads ", Threads.nthreads(), ", optimize ", Base.JLOptions().opt_level)
flush(stdout)

if MODE == "infer"
    Pw = B.padded_space(W)  # the space `form` on a ghost space builds the tree on
    raw = bilinear(P0)(B.trial_function(Pw), B.test_function(Pw))
    resolved = B._resolve_surfaces_for_space(B.resolve_ast(raw), Pw, Pw)
    for (name, f, args) in (("simplify_ast", B.simplify_ast, (typeof(resolved),)),
        ("loss", loss, (Vector{Float64},)))
        println("INFER ", name, " => ", only(Base.return_types(f, args)))
    end
    exit(0)
end

ref = DifferentiationInterface.gradient(loss, AutoForwardDiff(), P0)
step("reference")
println("reference ", ref)
if minimum(abs, ref) <= 1e-6 * maximum(abs, ref)
    println("RESULT ", CELL, " ", MODE, " weakref NaN 0.0 reference ", ref)
    exit(1)
end
MODE == "ref" && (println("RESULT ", CELL, " ref ok 0.0 0.0"); exit(0))

const AD = AutoEnzyme(;
    mode = Enzyme.set_runtime_activity(MODE == "rev" ? Enzyme.Reverse : Enzyme.Forward),
    function_annotation = Enzyme.Const)

# Top-level `try` blocks do not hand their assignments back, so the call runs in a function.
# Its clock starts outside: Enzyme's generated thunk compiles while `measure` is inferred,
# before its first line runs.
function measure(ref, t)
    status, relerr, detail, g = "ok", NaN, "", Float64[]
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
        println(first(msg, 1500))
        step("enzyme failed")
    end
    return (; status, relerr, detail, g, secs = time() - t)
end

step("enzyme start")
r = measure(ref, time())
println("RESULT ", CELL, " ", MODE, " ", r.status, " ",
    isnan(r.relerr) ? "NaN" : string(r.relerr), " ", round(r.secs; digits = 1), " ", r.detail)
println("enzyme gradient ", isempty(r.g) ? "none" : string(r.g), " reference ", ref)
flush(stdout)
exit(r.status == "ok" ? 0 : 1)
