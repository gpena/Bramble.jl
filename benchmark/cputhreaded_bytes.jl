#===========================================================================#
# Warm bytes per call of the in-place paths under `CpuThreaded`, and what the bytes are
# (gpena/Bramble.jl#436, subplan S5.1).
#
# Run from a worktree root with four threads and the benchmark environment:
#
#     julia --startup-file=no --threads=4 --project=benchmark benchmark/cputhreaded_bytes.jl
#
# It prints the table `docs/notes/cputhreaded_bytes.md` holds, one row per path and mesh
# size, as markdown. `--list` prints the path names and exits.
#
# Each path runs on a graded 2D mesh of 65 × 65 and of 513 × 513 points: along axis `d` the
# nodes are `t^(1 + d/4)`, so no two axes share their nodes. A path is called three times,
# then `REPS` more under `@allocated`; the figure is the median, and the spread (largest
# minus smallest) is printed when it is not zero. `@allocated` counts every thread, so the
# bytes of a spawned task count.
#
# The split comes from one more warm call under `Profile.Allocs` at sample rate 1, repeated
# `SPLIT_REPS` times, the median of each class. A class is the type of the allocated object:
#
#     Task     a `Task`
#     closure  an object whose type is a `Function` subtype (a closure with captured
#              variables, which `Threads.@threads` and `@spawn` build per call)
#     rest     everything else (`Ref`, `Tuple`, `Memory`, `Core.Box`, ...)
#
# `Profile.Allocs` does not see every byte `@allocated` counts: 80 B per `Threads.@threads`
# region and 16 B per `@spawn` (the calibration rows show it). `rest` is the bytes minus
# Task and closure, so the three columns add up to the bytes and the unseen part is in
# `rest`.
#===========================================================================#
using Bramble, Profile
using Bramble: CpuSerial, CpuThreaded, allocate_system_matrix, change_points!, restrict_to,
               v_cycle!, jacobian!, jacobian_form, jacobian_prototype
using ForwardDiff: ForwardDiff
using LinearAlgebra: mul!, ldiv!
using Statistics: median

const SIZES = (65, 513)
const REPS = 11
const SPLIT_REPS = 5

# The unit square of `n` × `n` points, graded, under `policy`, with a stored `:dir` marker
# on the whole boundary.
function graded_mesh(policy, n)
    X = interval(0.0, 1.0) × interval(0.0, 1.0)
    Ω = mesh(domain(X, :dir => boundary_symbols(X)), (n, n), (false, false);
        backend = backend(policy = policy))
    change_points!(Ω, ntuple(d -> range(0.0, 1.0; length = n) .^ (1 + 0.25d), 2))
    return Ω
end

f_of_x(x) = 1 + sum(abs2, x) + prod(x)

# ---------------------------------------------------------------------------------------
# The paths. A setup takes the mesh and returns the call, a closure of no arguments.

stencil(name) = Ω -> begin
    W = gridspace(Ω)
    u = Rₕ(W, f_of_x)
    w = similar(u)
    f = getproperty(Bramble, name)
    () -> f(w, u)
end

function calculus(name, dest, source)
    return Ω -> begin
        W = gridspace(Ω)
        V = gridspace(Ω, Val(2))
        d = dest(W)
        s = source(W, V)
        f = getproperty(Bramble, name)
        () -> f(d, s)
    end
end
scalar_dest(W) = element(W)
grad_dest(W) = (element(W), element(W))
strain_dest(W) = ((element(W), element(W)), (element(W), element(W)))
scalar_source(W, V) = Rₕ(W, f_of_x)
field_source(W, V) = Rₕ(V, (f_of_x, f_of_x))

broadcast_fused(Ω) = begin
    W = gridspace(Ω)
    u = Rₕ(W, f_of_x)
    v = similar(u)
    () -> (v .= 2.0 .* u .+ u .* u; nothing)
end
broadcast_two(Ω) = begin
    W = gridspace(Ω)
    u = Rₕ(W, f_of_x)
    w = Rₕ(W, f_of_x)
    v = similar(u)
    () -> (v .= u .+ w; nothing)
end

# A bilinear form: mass plus stiffness, and with one term masked to the boundary.
mass_stiffness(u, v) = inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(u, v)
function masked_form(W)
    κ = Rₕ(W, f_of_x)
    return (u, v) -> mass_stiffness(u, v) + innerₕ(κ * u, restrict_to(:dir, v))
end

# `innerₕ(f, v)` linear forms: plain, masked by `markers`, masked by `restrict_to`.
function linear_form(kind)
    return Ω -> begin
        W = gridspace(Ω)
        f = Rₕ(W, f_of_x)
        l = if kind === :plain
            form(W, v -> innerₕ(f, v))
        elseif kind === :markers
            form(W, v -> innerₕ(f, v; markers = (:dir,)))
        else
            form(W, v -> innerₕ(f, restrict_to(:dir, v)))
        end
        b = assemble(l)
        () -> assemble!(b, l)
    end
end

function bilinear_form(kind)
    return Ω -> begin
        W = gridspace(Ω)
        a = form(W, W, kind === :plain ? mass_stiffness : masked_form(W))
        A = allocate_system_matrix(a)
        assemble!(A, a)
        () -> assemble!(A, a)
    end
end

function bilinear_dirichlet(Ω)
    W = gridspace(Ω)
    a = form(W, W, mass_stiffness)
    A = assemble(a; dirichlet = :dir)
    return () -> assemble!(A, a; dirichlet = :dir)
end

# `a + ∂a` at the state `u`: the matrix `jacobian!` fills.
_jac_α(s) = 3 + 1 / (1 + s^2)
function jacobian_path(Ω)
    W = gridspace(Ω)
    c = element(W, 0.0)
    a = form(W, W, (U, V) -> innerₕ(c * U, V) + inner₊(∇ₕ(U), ∇ₕ(V)))
    jf = jacobian_form(a, c => (U -> U, _jac_α); dirichlet = :dir)
    J = jacobian_prototype(jf)
    u = [0.3 + 0.7 * sin(0.37 * i) for i in 1:ndofs(W)]
    return () -> jacobian!(J, jf, u)
end

function matrix_free_product(kind)
    return Ω -> begin
        W = gridspace(Ω)
        a = form(W, W, kind === :plain ? mass_stiffness : masked_form(W))
        op = matrix_free_operator(a)
        x = fill(0.5, ndofs(W))
        y = zeros(ndofs(W))
        () -> mul!(y, op, x)
    end
end

# Mass plus variable diffusion, symmetric positive definite, on every level.
struct VariableDiffusion end
function (::VariableDiffusion)(W)
    κ = Rₕ(W, x -> 1 + sum(abs2, x))
    return form(W, W, (u, v) -> innerₕ(u, v) + inner₊(κ * ∇ₕ(u), ∇ₕ(v)))
end

function multigrid(call)
    return Ω -> begin
        # The coarsest level keeps 17 points per axis, so the large mesh has more levels.
        levels = round(Int, log2(npoints(Ω) |> sqrt) - 1) - 2
        P = gmg_preconditioner(VariableDiffusion(), Ω; levels)
        n = npoints(Ω)
        x = ones(n)
        b = fill(0.5, n)
        () -> call(x, P, b)
    end
end

const PATHS = Tuple{String, String, Function}[
    ("difference", "D₋ₓ!", stencil(:D₋ₓ!)),
    ("difference", "D₋ᵧ!", stencil(:D₋ᵧ!)),
    ("difference", "D₊ₓ!", stencil(:D₊ₓ!)),
    ("difference", "Dcₓ!", stencil(:Dcₓ!)),
    ("difference", "D̃ₓ!", stencil(:D̃ₓ!)),
    ("average", "Mₓ!", stencil(:Mₓ!)),
    ("average", "M₊ₓ!", stencil(:M₊ₓ!)),
    ("average", "Mcₓ!", stencil(:Mcₓ!)),
    ("calculus", "∇cₕ!", calculus(:∇cₕ!, grad_dest, scalar_source)),
    ("calculus", "Δₕ!", calculus(:Δₕ!, scalar_dest, scalar_source)),
    ("calculus", "divₕ!", calculus(:divₕ!, scalar_dest, field_source)),
    ("calculus", "divcₕ!", calculus(:divcₕ!, scalar_dest, field_source)),
    ("calculus", "diṽₕ!", calculus(:diṽₕ!, scalar_dest, field_source)),
    ("calculus", "curlₕ!", calculus(:curlₕ!, scalar_dest, field_source)),
    ("calculus", "curlcₕ!", calculus(:curlcₕ!, scalar_dest, field_source)),
    ("calculus", "εₕ!", calculus(:εₕ!, strain_dest, field_source)),
    ("broadcast", "v .= 2u + u*u", broadcast_fused),
    ("broadcast", "v .= u + w", broadcast_two),
    ("innerₕ", "assemble! innerₕ(f, v)", linear_form(:plain)),
    ("innerₕ", "assemble! innerₕ(f, v; markers)", linear_form(:markers)),
    ("innerₕ", "assemble! innerₕ(f, restrict_to(v))", linear_form(:restrict)),
    ("assemble!", "assemble! bilinear", bilinear_form(:plain)),
    ("assemble!", "assemble! bilinear, masked term", bilinear_form(:masked)),
    ("assemble!", "assemble! bilinear, dirichlet", bilinear_dirichlet),
    ("assemble!", "jacobian!", jacobian_path),
    ("matrix-free", "mul! plain", matrix_free_product(:plain)),
    ("matrix-free", "mul! masked term", matrix_free_product(:masked)),
    ("multigrid", "v_cycle!", multigrid(v_cycle!)),
    ("multigrid", "ldiv! multigrid", multigrid((x, P, b) -> ldiv!(x, P, b)))
]

if ARGS == ["--list"]
    foreach(p -> println(p[2]), PATHS)
    exit(0)
end
Threads.nthreads() == 4 || error("run with --threads=4")

# ---------------------------------------------------------------------------------------
# Measurement

# The bytes of `REPS` warm calls.
@noinline function warm_bytes(call::F) where {F}
    for _ in 1:3
        call()
    end
    b = Vector{Int}(undef, REPS)
    for i in 1:REPS
        b[i] = @allocated call()
    end
    return b
end

# Bytes per class of one call under `Profile.Allocs`: (Task, closure, rest).
@noinline function profiled_split(call::F) where {F}
    Profile.Allocs.clear()
    Profile.Allocs.@profile sample_rate = 1.0 call()
    task = closure = rest = 0
    for a in Profile.Allocs.fetch().allocs
        T = a.type
        if T === Task
            task += a.size
        elseif T isa DataType && T <: Function
            closure += a.size
        else
            rest += a.size
        end
    end
    Profile.Allocs.clear()
    return (task, closure, rest)
end

function measure(setup, n)
    call = setup(graded_mesh(CpuThreaded(), n))
    bytes = warm_bytes(call)
    splits = [profiled_split(call) for _ in 1:SPLIT_REPS]
    med(k) = round(Int, median(s[k] for s in splits))
    total = round(Int, median(bytes))
    return (; bytes = total, spread = maximum(bytes) - minimum(bytes),
        task = med(1), closure = med(2), rest = total - med(1) - med(2))
end

# Two bare regions, to price the machinery without any work in it. The setup ignores the mesh.
function empty_region(_)
    return () -> begin
        Threads.@threads :static for _ in 1:4
        end
        return nothing
    end
end
function spawn_and_wait(_)
    return () -> begin
        wait(Threads.@spawn nothing)
        return nothing
    end
end
const CALIBRATION = Tuple{String, Function}[
    ("empty Threads.@threads :static region", empty_region),
    ("Threads.@spawn and wait", spawn_and_wait)
]

println("| Group | Path | Mesh | Bytes | Task | Closure | Rest |")
println("|---|---|---|---:|---:|---:|---:|")
function row(group, name, mesh, r)
    println("| $group | `$name` | $mesh | $(r.bytes) | $(r.task) | $(r.closure) | $(r.rest) |")
    r.spread == 0 || println(stderr, "SPREAD\t$name\t$mesh\t$(r.spread)")
    return nothing
end
for (name, setup) in CALIBRATION
    row("calibration", name, "none", measure(setup, SIZES[1]))
end
for (group, name, setup) in PATHS, n in SIZES

    row(group, name, "$n × $n", measure(setup, n))
end
