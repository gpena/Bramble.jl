#===========================================================================#
# First-call and warm times of multi-term forms, each in a fresh child process
# (gpena/Bramble.jl#763). Rows: `elasticity` (the 3D elasticity form on a 5^3 mesh, so
# compile cost does not depend on the size), `vector_laplacian` (`inner₊(∇ₕ(u), ∇ₕ(v))` on
# `gridspace(Ωₕ)^Val(3)`) and `scalar_2d` (a small scalar 2D Laplacian, the no-regression
# row). Four scale rows, `scale_float64`, `scale_float32`, `scale_pi` and `scale_rational`,
# time `c * inner₊(∇ₕ(u), ∇ₕ(v))` with `c` a `Float64`, `Float32`, `π` and `1 // 2`, on the
# graded 9x11 2D mesh of `benchmark/snoop_instances.jl` (same `graded`).
#
# Plain Julia, Base and stdlib only, launched with no `--project` of its own: it spawns
# child `julia` processes (`Base.julia_cmd()`) against the project at `--tree`, so C4 can
# A/B commits by pointing it at another worktree.
#
#     julia --startup-file=no benchmark/multiterm_ttfx.jl [--tree DIR] [--launches 5]
#         [--threads 1] [--rows A,B] [--section NAME --out FILE] [--smoke]
#
# Each child times the first `form`, the first `assemble`, then `assemble` warm, `form`
# warm and `assemble!(A, a)` warm (each the min of 5). `--rows` picks rows by name,
# comma-separated; the default is the first three. The parent runs the children
# `--launches` times, interleaved by round, and prints the median of each figure per row, in
# seconds. `--smoke` runs one child per form, once.
# `--section NAME --out FILE` also writes (merging into FILE) a TOML table `[NAME.<row>]`.
#
# Smoke figures taken while other jobs run the machine are not measurements.
#===========================================================================#

using TOML

const DEFAULT_ROWS = ["elasticity", "vector_laplacian", "scalar_2d"]
const SCALARS = Dict("scale_float64" => 2.0, "scale_float32" => 2.0f0, "scale_pi" => π,
    "scale_rational" => 1 // 2)
const ALL_ROWS = vcat(DEFAULT_ROWS, ["scale_float64", "scale_float32", "scale_pi", "scale_rational"])

if "--child" in ARGS
    # --child <row>
    row = ARGS[2]
    using Bramble
    import Bramble: change_points!

    # The points of an axis: the exponent differs per axis, so no two axes agree.
    graded(n, axis) = collect(range(0.0, 1.0; length = n)) .^ (1.2 + 0.2 * axis)

    function build(row)
        if haskey(SCALARS, row)
            n = (9, 11)
            Ωₕ = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), n)
            change_points!(Ωₕ, ntuple(d -> graded(n[d], d), 2))
            c = SCALARS[row]
            return gridspace(Ωₕ), (u, v) -> c * inner₊(∇ₕ(u), ∇ₕ(v))
        end
        if row == "scalar_2d"
            Ωₕ = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (9, 9), (false, false))
            return gridspace(Ωₕ), (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v))
        end
        Ωc = mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (5, 5, 5),
            (false, false, false))
        W = gridspace(Ωc)^Val(3)
        row == "vector_laplacian" && return W, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v))
        μ, λ = 1.0, 2.0
        return W, (u, v) -> 2μ * inner₊(εₕ(u), εₕ(v)) + λ * inner₊(divₘₕ(u), divₘₕ(v))
    end

    W, fun = build(row)
    t_form = @elapsed a = form(W, W, fun)
    A = nothing
    t_asm = @elapsed A = assemble(a)
    t_warm = minimum(@elapsed(Base.donotdelete(assemble(a))) for _ in 1:5)
    t_warm_form = minimum(@elapsed(Base.donotdelete(form(W, W, fun))) for _ in 1:5)
    t_warm_bang = minimum(@elapsed(Base.donotdelete(assemble!(A, a))) for _ in 1:5)
    println("RESULT ", t_form, " ", t_asm, " ", t_warm, " ", t_warm_form, " ", t_warm_bang)
    exit(0)
end

function arg(name, default)
    i = findfirst(==(name), ARGS)
    return i === nothing ? default : ARGS[i + 1]
end

function launch(tree, row, threads)
    cmd = `$(Base.julia_cmd()) --project=$tree --threads=$threads --startup-file=no
        $(abspath(@__FILE__)) --child $row`
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
    rows = String.(split(arg("--rows", join(DEFAULT_ROWS, ",")), ","))
    for r in rows
        r in ALL_ROWS || error("unknown row $r; rows are $(join(ALL_ROWS, ", "))")
    end
    runs = Dict(r => Vector{Vector{Float64}}() for r in rows)
    for _ in 1:n, r in rows

        push!(runs[r], launch(tree, r, threads))
    end
    println("median of $n launch(es), seconds: tree $tree, $threads thread(s)")
    smoke && println("SMOKE: one launch, not a measurement")
    println(rpad("row", 18), rpad("form", 12), rpad("assemble", 12), rpad("warm assemble", 15),
        rpad("warm form", 12), "warm assemble!")
    table = Dict{String, Any}()
    for r in rows
        m = [median([x[k] for x in runs[r]]) for k in 1:5]
        println(rpad(r, 18), rpad(round(m[1]; sigdigits = 4), 12),
            rpad(round(m[2]; sigdigits = 4), 12), rpad(round(m[3]; sigdigits = 4), 15),
            rpad(round(m[4]; sigdigits = 4), 12), round(m[5]; sigdigits = 4))
        table[r] = Dict("form" => m[1], "assemble" => m[2], "warm_assemble" => m[3],
            "warm_form" => m[4], "warm_assemble!" => m[5], "launches" => n,
            "threads" => threads)
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
