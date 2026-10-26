#===========================================================================#
# Warm assembly and products on non-uniform meshes, base against branch (gpena/Bramble.jl#928
# and #930, S8.1 of .claude/plans/v4-1-3-types.md): do the type cleanups move the warm path?
#
# Plain Julia, Base, stdlib and TOML only, launched with no `--project` of its own: it spawns
# child `julia` processes (`Base.julia_cmd()`) against the project at `--tree`, so one run
# per worktree gives the base and the branch figures, and the script itself is the same
# file on both. It uses only API present on main, so it runs on base unchanged.
#
#     julia --startup-file=no benchmark/type_cleanups_ab.jl [--tree DIR] [--launches 5]
#         [--threads 1] [--rows A,B] [--section NAME --out FILE] [--smoke]
#
# It refuses to run below `--optimize=2`: at -O1 the SIMD loops do not vectorise and the
# figures are not comparable (the children inherit the level of the parent's `julia_cmd`).
#
# Rows, all on seeded non-uniform meshes (seed 5, the ghost tests' `ghost_mesh`), serial
# policy:
#
#   - plain_2d, plain_3d: `gridspace`, `innerₕ(-1.0 * Δₕ(u), v) + innerₕ(0.5 * Dcₓ(u), v)`;
#   - ghost_2d:           the same form on `gridspace(Ωₕ; ghosts = :boundary)`, 2D;
#   - kron_2d:            Kronecker-separable mass + stiffness,
#                         `innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))`, 2D, product by
#                         `kronecker_operator`;
#   - kron_2d_mf:         the same form, product by `matrix_free_operator`.
#
# Per row: warm `assemble!(A, a)` and warm matrix-free `mul!` (the minimum of 30 calls, in ns
# per DOF) and the allocations of one warm call of each, in bytes. The parent runs the
# children `--launches` times, interleaved by round (one child per row and round), and prints
# the median of each figure per row. `--smoke` runs one child per row, once, on small meshes.
# `--section NAME --out FILE` also writes (merging into FILE) a TOML table `[NAME.<row>]`.
#
# Smoke figures, and figures taken while other jobs run the machine, are not measurements.
#===========================================================================#

using TOML

const ALL_ROWS = ["plain_2d", "plain_3d", "ghost_2d", "kron_2d", "kron_2d_mf"]
const SEED = 5
const SAMPLES = 30
# Mesh points per axis: the measurement size and the smoke size.
const SIZES = Dict(2 => (300, 300), 3 => (40, 40, 40))
const SMOKE_SIZES = Dict(2 => (40, 40), 3 => (12, 12, 12))

if "--child" in ARGS
    # --child <row> [--smoke]
    row = ARGS[2]
    smoke = "--smoke" in ARGS
    using Bramble
    using Bramble: Dcₓ, kronecker_operator
    using LinearAlgebra: mul!

    Base.JLOptions().opt_level >= 2 ||
        throw(ArgumentError("run without --optimize=1: timings below the default level are not figures"))

    # A seeded non-uniform mesh on [0, 1] × [0, 2] × [0, 3] (its first `D` axes), the
    # construction of `ghost_mesh` in `test/ghost/matrix_free.jl`.
    function seeded_mesh(D)
        ns = (smoke ? SMOKE_SIZES : SIZES)[D]
        Bramble._seed_mesh1d_rng!(SEED)
        Ωd = domain(foldl(×, ntuple(d -> interval(0.0, Float64(d)), D)))
        Ωₕ = mesh(Ωd, ns, ntuple(_ -> false, D))
        Bramble._unseed_mesh1d_rng!()
        return Ωₕ
    end

    form_f(u, v) = innerₕ(-1.0 * Δₕ(u), v) + innerₕ(0.5 * Dcₓ(u), v)
    kron_f(u, v) = innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))

    # Warm once, then the minimum of `n` calls in seconds.
    function tmin(g::F, n = SAMPLES) where {F}
        g()
        return minimum(@elapsed(g()) for _ in 1:n)
    end

    # Allocations of one warm call.
    function allocs(g::F) where {F}
        g()
        return @allocated g()
    end

    function build(row)
        row == "plain_3d" && return gridspace(seeded_mesh(3)), form_f, matrix_free_operator
        row == "plain_2d" && return gridspace(seeded_mesh(2)), form_f, matrix_free_operator
        row == "ghost_2d" &&
            return gridspace(seeded_mesh(2); ghosts = :boundary), form_f, matrix_free_operator
        row == "kron_2d" && return gridspace(seeded_mesh(2)), kron_f, kronecker_operator
        return gridspace(seeded_mesh(2)), kron_f, matrix_free_operator
    end

    W, f, opf = build(row)
    a = form(W, W, f)
    A = assemble(a)
    n = ndofs(W)
    asm! = () -> assemble!(A, a)
    op = opf(a)
    x = rand(n)
    y = similar(x)
    m! = () -> mul!(y, op, x)
    tasm, tmul = tmin(asm!), tmin(m!)
    aasm, amul = allocs(asm!), allocs(m!)
    println("RESULT ", n, " ", 1.0e9 * tasm / n, " ", 1.0e9 * tmul / n, " ", aasm, " ", amul)
    exit(0)
end

Base.JLOptions().opt_level >= 2 ||
    throw(ArgumentError("run without --optimize=1: timings below the default level are not figures"))

function arg(name, default)
    i = findfirst(==(name), ARGS)
    return i === nothing ? default : ARGS[i + 1]
end

function launch(tree, row, threads, smoke)
    cmd = `$(Base.julia_cmd()) --project=$tree --threads=$threads --startup-file=no
        $(abspath(@__FILE__)) --child $row $(smoke ? "--smoke" : "")`
    for line in eachline(cmd)
        startswith(line, "RESULT ") && return parse.(Float64, split(line)[2:6])
    end
    error("child for $row printed no RESULT line")
end

function median(v)
    s = sort(v)
    return isodd(length(s)) ? s[(end + 1) ÷ 2] : (s[end ÷ 2] + s[end ÷ 2 + 1]) / 2
end

function main()
    tree = abspath(expanduser(arg("--tree", joinpath(@__DIR__, ".."))))
    smoke = "--smoke" in ARGS
    n = smoke ? 1 : parse(Int, arg("--launches", "5"))
    threads = parse(Int, arg("--threads", "1"))
    rows = String.(split(arg("--rows", join(ALL_ROWS, ",")), ","))
    for r in rows
        r in ALL_ROWS || error("unknown row $r; rows are $(join(ALL_ROWS, ", "))")
    end
    runs = Dict(r => Vector{Vector{Float64}}() for r in rows)
    for _ in 1:n, r in rows

        push!(runs[r], launch(tree, r, threads, smoke))
    end
    println("median of $n launch(es), min of $SAMPLES calls: tree $tree, $threads thread(s), ",
        "opt_level ", Base.JLOptions().opt_level)
    smoke && println("SMOKE: one launch on small meshes, not a measurement")
    println(rpad("row", 14), lpad("dofs", 9), lpad("asm! ns/dof", 13), lpad("mul! ns/dof", 13),
        lpad("asm! B", 9), lpad("mul! B", 9))
    table = Dict{String, Any}()
    for r in rows
        m = [median([x[k] for x in runs[r]]) for k in 1:5]
        println(rpad(r, 14), lpad(round(Int, m[1]), 9), lpad(round(m[2]; digits = 2), 13),
            lpad(round(m[3]; digits = 2), 13), lpad(round(Int, m[4]), 9),
            lpad(round(Int, m[5]), 9))
        table[r] = Dict("dofs" => round(Int, m[1]), "assemble_ns_per_dof" => m[2],
            "mul_ns_per_dof" => m[3], "assemble_bytes" => round(Int, m[4]),
            "mul_bytes" => round(Int, m[5]), "launches" => n, "threads" => threads)
    end
    section, out = arg("--section", nothing), arg("--out", nothing)
    if section !== nothing && out !== nothing
        data = isfile(out) ? TOML.parsefile(out) : Dict{String, Any}()
        data[section] = table
        open(io -> TOML.print(io, data), out, "w")
    end
    return nothing
end

main()
