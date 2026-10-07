#===========================================================================#
# One side of a base-vs-head A/B of restricted forms (gpena/Bramble.jl#437,
# subplan S6.3). Run from a worktree root, with that worktree's test
# environment, 4 threads and -O1:
#
#     julia --project=<test env> --threads=4 --optimize=1 --startup-file=no \
#         <head>/benchmark/restricted_forms.jl <policies> <case>...
#     julia ... benchmark/restricted_forms.jl --list
#
# `<policies>` is a comma list of `serial`, `threaded`, `polyester`. For each
# case, policy and size the case chooses, the script prints
# `AB<TAB>case<TAB>policy<TAB>n<TAB>ns`: the minimum over repeated warm calls
# (at least 5, up to ~0.2 s or 20000 calls), then the median of that minimum
# over `AB_LAYOUTS` array placements (default 5, see below). An unknown case
# prints `AB-MISSING<TAB>case` and exits 1. `--list` prints
# `case<TAB>dim<TAB>hratio`, hratio being the largest over the smallest mesh
# spacing of the jittered mesh the case runs on. It reads the mesh, it is not
# a constant. The meshes are non-uniform on every axis.
#
# The cases are the restricted forms S3 changed (a `RegionRestriction` reads
# the mesh's marker words instead of a `Dict` per point):
#
#     "<form> <D>D <kind>"    form: boundary, two-region; D: 2, 3;
#                             kind: refill (`assemble!` into a built matrix),
#                                   product (matrix-free `mul!`)
#     "GMG 50 V-cycles"       one loop of 50 `v_cycle!` calls under
#                             `CpuPolyester` on a 2D jittered mesh, the
#                             "boundary" form on every level
#
# "boundary" is `innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))` plus one term
# restricted to `:boundary`. "two-region" is `innerₕ(u, v)` plus one term
# restricted to `:interior` and one to `:boundary`. Both are timed under the
# serial and threaded policies. The GMG case also prints
# `GC<TAB>case<TAB>policy<TAB>n<TAB>gc_ns<TAB>time_ns` from `@timed` of one
# warm 50-cycle loop. There is no interface-integral row: that term never
# reads the marker table, so a row would show nothing.
#
# Everything called here exists on the base as well: the base side runs this
# script against the base's Bramble.
#
# Array placement. One fixed placement of the input and output arrays can make
# a loop 2 to 4 times slower at certain address offsets (a store followed by a
# load whose addresses collide modulo a page), on base and head alike, and
# that reads as a code change. So before each case this allocates a pad of
# whole 16 KiB pages, a seeded random number of them, `AB_LAYOUTS` times, and
# reports the median of the per-placement minima. The seed depends only on the
# case, policy and size, so both sides draw the same pads. The GMG case uses
# the same count: one 50-cycle loop takes ~0.5 s at 513^2, so its minimum is
# over 3 loops, not the ~0.2 s budget of the other cases, and a whole process
# stays within half a minute.
#===========================================================================#
using Bramble, Polyester
using Bramble: CpuPolyester, CpuSerial, CpuThreaded, change_points!, allocate_system_matrix,
               D₋ₓ, restrict_to, v_cycle!
using LinearAlgebra: mul!
using Random: Xoshiro

# A jittered, non-uniform `D`-dimensional mesh of `n` points per axis on the unit box, the
# interior points moved by up to 0.3h, with the box's own boundary markers.
function jittered_mesh(n, D, policy; seed = 4321)
    rng = Xoshiro(seed)
    X = reduce(×, ntuple(_ -> interval(0.0, 1.0), D))
    Ω = mesh(domain(X, :dir => boundary_symbols(X)), ntuple(_ -> n, D), ntuple(_ -> true, D);
        backend = backend(policy = policy))
    h = 1 / (n - 1)
    function pts()
        x = collect(range(0.0, 1.0; length = n)) .+ 0.3h .* (2 .* rand(rng, n) .- 1)
        x[1], x[end] = 0.0, 1.0
        return sort!(x)
    end
    change_points!(Ω, ntuple(_ -> pts(), D))
    return Ω
end

# Largest over smallest spacing of any axis of the mesh.
function hratio(Ω, D)
    hs = (diff(collect(points(Ω(i)))) for i in 1:D)
    return maximum(maximum, hs) / minimum(minimum, hs)
end

# The restricted forms. `κ` is a grid function, as in the matrix-free tests of restricted
# forms, so the restricted terms carry data.
function boundary_form(W)
    κ = Rₕ(W, x -> 1 + sum(abs2, x))
    return form(W, W,
        (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)) +
                  innerₕ(κ * u, restrict_to(:boundary, v)))
end

function two_region_form(W)
    κ = Rₕ(W, x -> 1 + sum(abs2, x))
    return form(W, W,
        (u, v) -> innerₕ(u, v) + innerₕ(restrict_to(:interior, D₋ₓ(u)), v) +
                  innerₕ(κ * u, restrict_to(:boundary, v)))
end

const FORMS = Dict("boundary" => boundary_form, "two-region" => two_region_form)

# Sizes per dimension: a small and a large mesh. 3D has fewer points per axis (65^3 = 274625
# points, about as many as 513^2).
const SIZES = Dict(2 => (65, 513), 3 => (17, 65))

# Each case is a name, its dimension, its policies' sizes and a setup `(n, policy) -> call`.
function refill(form_of, D)
    return (n, p) -> begin
        a = form_of(gridspace(jittered_mesh(n, D, p)))
        A = allocate_system_matrix(a)
        () -> assemble!(A, a)
    end
end

function product(form_of, D)
    return (n, p) -> begin
        op = matrix_free_operator(form_of(gridspace(jittered_mesh(n, D, p))))
        x = rand(Xoshiro(1), size(op, 2))
        y = similar(x)
        () -> mul!(y, op, x)
    end
end

const GMG_N = 513
const GMG_CYCLES = 50

# 50 V-cycles of the boundary form from a zero iterate: resetting it keeps every loop on
# the same arithmetic.
function gmg(n, p)
    Ω = jittered_mesh(n, 2, p)
    P = gmg_preconditioner(boundary_form, Ω; cycle = :V)
    b = rand(Xoshiro(3), npoints(Ω))
    x = zeros(length(b))
    return () -> begin
        fill!(x, 0.0)
        for _ in 1:GMG_CYCLES
            v_cycle!(x, P, b)
        end
        x
    end
end

struct Case
    dim::Int
    sizes::Tuple{Vararg{Int}}
    policies::Vector{String}
    setup::Function
end
const CASES = Dict{String, Case}()
for (fname, f) in FORMS, D in (2, 3)

    CASES["$fname $(D)D refill"] = Case(D, SIZES[D], ["serial", "threaded"], refill(f, D))
    CASES["$fname $(D)D product"] = Case(D, SIZES[D], ["serial", "threaded"], product(f, D))
end
CASES["GMG 50 V-cycles"] = Case(2, (GMG_N,), ["polyester"], gmg)

if ARGS == ["--list"]
    for name in sort!(collect(keys(CASES)))
        c = CASES[name]
        println(name, "\t", c.dim, "\t", hratio(jittered_mesh(c.sizes[1], c.dim, CpuSerial()), c.dim))
    end
    exit(0)
end

Threads.nthreads() >= 4 || error("run with --threads=4")
const POLICIES = Dict("serial" => CpuSerial(), "threaded" => CpuThreaded(),
    "polyester" => CpuPolyester())
const POL = split(ARGS[1], ',')
const NAMES = ARGS[2:end]
all(in(keys(POLICIES)), POL) || error("unknown policy in $(ARGS[1])")

@noinline function once(call::F) where {F}
    t = time_ns()
    call()
    return time_ns() - t
end
# Minimum over repeated warm calls; `budget` ns bounds the repeats, `minreps` is the floor.
@noinline function best(call::F, budget, minreps) where {F}
    call()
    call()
    t1 = once(call)
    reps = clamp(ceil(Int, budget / max(t1, 1)), minreps, 20_000)
    return minimum(once(call) for _ in 1:reps)
end

# The page pads, kept until the next case so they are not reused while the arrays live.
const PADS = Vector{UInt8}[]

# `AB_LAYOUTS` placements, each built after a fresh pad; the median of their minima.
function placed(name, pol, n, layouts, setup, budget, minreps)
    rng = Xoshiro(hash((20261006, name, pol, n)))
    empty!(PADS)
    ts = UInt64[]
    for _ in 1:layouts
        push!(PADS, Vector{UInt8}(undef, 16384 * rand(rng, 1:512)))
        call = setup(n, POLICIES[pol])
        push!(ts, best(call, budget, minreps))
    end
    return sort!(ts)[(length(ts) + 1) ÷ 2]
end

missing_any = false
for name in NAMES
    haskey(CASES, name) || (println("AB-MISSING\t", name); global missing_any = true)
end
missing_any && exit(1)

const LAYOUTS = parse(Int, get(ENV, "AB_LAYOUTS", "5"))
for name in NAMES
    c = CASES[name]
    gmg_case = name == "GMG 50 V-cycles"
    for pol in POL
        pol in c.policies || continue
        for n in c.sizes
            t = if gmg_case
                placed(name, pol, n, LAYOUTS, c.setup, 0.0, 3)
            else
                placed(name, pol, n, LAYOUTS, c.setup, 2e8, 5)
            end
            println("AB\t", name, "\t", pol, "\t", n, "\t", t)
            if gmg_case
                call = c.setup(n, POLICIES[pol])
                call()
                r = @timed call()
                println("GC\t", name, "\t", pol, "\t", n, "\t", round(Int, r.gctime * 1e9), "\t",
                    round(Int, r.time * 1e9))
            end
        end
    end
end
