#===========================================================================#
# Warm assembly of the 3D elasticity form against two reference forms, each in a fresh
# child process (gpena/Bramble.jl#765). Rows:
#
# - `elasticity`: `2.0 inner₊(εₕ(u), εₕ(v)) + 2.0 inner₊(divₘₕ(u), divₘₕ(v))` on
#   `gridspace(Ωc)^Val(3)`;
# - `vector_laplacian`: `inner₊(∇ₕ(u), ∇ₕ(v))` on the same composite space;
# - `scalar_laplacian`: `inner₊(∇ₕ(u), ∇ₕ(v))` on `gridspace(Ωc)`.
#
# `Ωc` is a seeded non-uniform, non-cubic mesh of the unit cube with `(n + 1, n, n - 1)`
# points, `n` the `--size` (30 or 45), so `--size 30` is about 30³ points.
#
# Plain Julia, Base and stdlib only, launched with no `--project` of its own: it spawns
# child `julia` processes (`Base.julia_cmd()`, default `-O`) against the project at `--tree`,
# so W3 can A/B commits by pointing it at another worktree.
#
#     julia --startup-file=no benchmark/warm_composite.jl [--tree DIR] [--size 30]
#         [--launches 3] [--threads 1] [--section NAME --out FILE]
#     julia --startup-file=no benchmark/warm_composite.jl --profile [--row elasticity]
#         [--size 30] [--tree DIR] [--seconds 4]
#
# Timing mode. Each child builds its form, fills twice to warm every path, then takes the
# min of 5 of `assemble!` into the matrix the form recorded (`fill`), of `assemble` (the
# pattern copy plus the fill), and of three stages alone, the pattern copy
# (`_allocate_assembled`), the stored-value zeroing (`_zero_stored!`) and the replay
# (`_replay_bilinear_core!`). The parent runs the children `--launches` times, interleaved
# by round, and prints one table row per form, the median over launches of the per-launch
# figures. `--section NAME --out FILE` also writes (merging into FILE) a TOML
# table `[NAME.<row>]` holding these keys, times in seconds:
#
#     fill, fill_launches     median and per-launch min of 5 warm `assemble!`
#     assemble                median of the per-launch min of 5 warm `assemble`
#     copy, zero, replay      median of the per-launch min of 5 of each stage
#     ns_per_unknown          1e9 * fill / ndofs
#     ns_per_add              1e9 * replay / adds
#     bytes_fill, bytes_assemble   bytes allocated by one warm call
#     ndofs, nnz, units, adds      unknowns, stored entries, recorded (term, block) units,
#                                  value additions per fill (sum of recorded positions)
#     points, launches, threads, size
#
# Profile mode (`--profile`) runs one child that profiles warm `assemble` of `--row` for
# about `--seconds` and prints, over the samples inside `assemble` only:
#
#     FUNCTIONS   by function (inlined frames included, C frames left out), inclusive and
#                 self shares, each over 0.5 %
#     STAGES      inclusive shares: pattern copy, zeroing, kept-pattern copy, replay; inside
#                 the replay the per-block setup (`_bind_interp_spaces`,
#                 `_check_block_meshes`, `host_weights`), the interior core
#                 (`_visit_interior!`), the guarded boundary shell (`_visit_guarded_region!`),
#                 stencil evaluation (`local_stencil`), the bounds guards and the value
#                 addition (`_replay_add!`); and garbage collection (a `gc` C frame)
#     UNITS       per (term, block) unit: the margin the walk peels at, the margin its
#                 entries need (walking every point), its reach, its entries per point, its
#                 pair half (-1 alone), and the boundary-shell share of points at the two
#                 margins; then the value additions at an interior point as the units make
#                 them and as one stencil per block (their sum) would
#
# Figures taken while other jobs run the machine are not measurements.
#===========================================================================#

# `Profile` here, not in the child block: that block is lowered whole, before its `using`
# runs, and `Profile.@profile` must expand there.
using Profile, TOML

const ROWS = ["elasticity", "vector_laplacian", "scalar_laplacian"]

if "--child" in ARGS || "--child-profile" in ARGS
    using Bramble, Random

    function build(row, n)
        Random.seed!(20260903)
        Ωc = mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (n + 1, n, n - 1),
            (false, false, false))
        lap = (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v))
        row == "scalar_laplacian" && return Ωc, gridspace(Ωc), lap
        W = gridspace(Ωc)^Val(3)
        row == "vector_laplacian" && return Ωc, W, lap
        row == "elasticity" && return Ωc, W,
        (u, v) -> 2.0 * inner₊(εₕ(u), εₕ(v)) + 2.0 * inner₊(divₘₕ(u), divₘₕ(v))
        error("unknown row $row")
    end

    minof5(f) = minimum(@elapsed(f()) for _ in 1:5)

    function adds(a)
        return sum(s -> length(s.positions) + length(s.positions_t), a.cache.segments)
    end

    replay!(B, a) = Bramble._replay_bilinear_core!(Bramble._SerialReplay(), B,
        a.trial_space, a.test_space, a.ast, a.cache.segments, true)

    # Allocation of one warm call, behind a function barrier.
    bytes_fill(B, a) = @allocated assemble!(B, a)
    bytes_assemble(a) = @allocated assemble(a)

    function measure(row, n)
        Ωc, W, fun = build(row, n)
        a = form(W, W, fun)
        A = assemble(a)      # records, keeps the pattern
        assemble(a)          # the pattern-copy path, warm
        t_asm = minof5(() -> assemble(a))
        b_asm = bytes_assemble(a)
        # The matrix the recording is against from here on.
        B = Bramble._allocate_assembled(a)
        assemble!(B, a)
        t_fill = minof5(() -> assemble!(B, a))
        b_fill = bytes_fill(B, a)
        t_zero = minof5(() -> Bramble._zero_stored!(B))
        t_replay = minof5(() -> replay!(B, a))
        # Last: each copy moves the recording to the new matrix.
        t_copy = minof5(() -> Bramble._allocate_assembled(a))
        nd = size(A, 2)
        npts = length(Bramble.indices(Ωc))
        return (; fill = t_fill, assemble = t_asm, copy = t_copy, zero = t_zero,
            replay = t_replay, bytes_fill = b_fill, bytes_assemble = b_asm, ndofs = nd,
            nnz = length(A.nzval), units = length(a.cache.segments), adds = adds(a),
            points = npts)
    end

    # Function name of a frame, the inlined ones included.
    fname(sf) = String(sf.func)

    # Per (term, block) unit: the margin the walk peels at (`_stencil_margin`), the margin
    # the unit needs (one more than the depth of the deepest point with an entry outside
    # the grid, walking every point), the stencil's reach, its entries per point at the
    # middle point, and the share of grid points in the boundary shell at the two margins.
    function units(a, Ωc)
        ax = size(Bramble.indices(Ωc))
        shell(m) = round(100 * (1 - prod(max.(ax .- 2m, 0)) / prod(ax)); digits = 1)
        rows = NTuple{5, Int}[]
        fused = Dict{NTuple{2, Int}, Set{Any}}()
        Bramble._foreach_unit(a.trial_space, a.test_space, a.ast) do term, sp, ro, co, dr, dc, half
            grid = Bramble.indices(Bramble.mesh(sp))
            lin = LinearIndices(grid)
            need, reach = 0, 0
            inside(J) = all(1 .<= Tuple(J) .<= ax)
            for I in grid
                st = Bramble.local_stencil(term, sp, I, nothing, lin[I])
                for (ou, ov) in Bramble.entry_offsets(st)
                    reach = max(reach, maximum(abs, Tuple(ou)), maximum(abs, Tuple(ov)))
                    inside(I + CartesianIndex(ou)) && inside(I + CartesianIndex(ov)) &&
                        continue
                    need = max(need, 1 + minimum(min.(Tuple(I) .- 1, ax .- Tuple(I))))
                end
            end
            I = CartesianIndex(ax .÷ 2)
            offs = Bramble.entry_offsets(Bramble.local_stencil(term, sp, I, nothing, lin[I]))
            for (ou, ov) in offs
                push!(get!(Set{Any}, fused, (ro, co)), (Tuple(ou), Tuple(ov)))
                # A pair's second term: the transposed entry, in its own block.
                half == 0 &&
                    push!(get!(Set{Any}, fused, (co + dr, ro + dc)), (Tuple(ov), Tuple(ou)))
            end
            push!(rows, (Bramble._stencil_margin(term), need, reach, length(offs), half))
        end
        println("UNITS (k: margin, needed margin, reach, entries per point, pair half; ",
            "shell % at margin, at needed margin)")
        for (k, (m, nd, r, e, h)) in enumerate(rows)
            println("  ", k, ": ", m, ", ", nd, ", ", r, ", ", e, ", ", h, "; ", shell(m),
                ", ", shell(nd))
        end
        # What one stencil per block, the units' sums, would add at an interior point.
        walked = sum(((_, _, _, e, h),) -> h == 0 ? 2e : e, rows)
        distinct = sum(length, values(fused))
        println("ADDS PER INTERIOR POINT: units ", walked, ", one stencil per block ",
            distinct, " (", length(fused), " blocks)")
        return nothing
    end

    function profile(row, n, seconds)
        Ωc, W, fun = build(row, n)
        a = form(W, W, fun)
        assemble(a)
        assemble(a)
        t = minof5(() -> assemble(a))
        reps = max(1, round(Int, seconds / t))
        Profile.init(; n = 10^7, delay = 0.0005)
        Profile.clear()
        Profile.@profile for _ in 1:reps
            assemble(a)
        end
        data = Profile.fetch(; include_meta = false)
        lidict = Profile.getdict(data)
        incl = Dict{String, Int}()
        self = Dict{String, Int}()
        gc = 0
        total = 0
        bt = UInt64[]
        # Samples outside `assemble` (the profiler's own task, the loop) are left out.
        flush_bt() = begin
            frames = reduce(vcat, (lidict[ip] for ip in bt); init = Base.StackFrame[])
            empty!(bt)
            jl = filter(sf -> !sf.from_c, frames)
            any(sf -> fname(sf) == "assemble", jl) || return
            total += 1
            any(sf -> sf.from_c && occursin("gc", fname(sf)), frames) && (gc += 1)
            # Self: the innermost frame that names a function, not an inlined loop body.
            k = findfirst(sf -> fname(sf) != "macro expansion", jl)
            k === nothing || (self[fname(jl[k])] = get(self, fname(jl[k]), 0) + 1)
            for name in unique(fname.(jl))
                incl[name] = get(incl, name, 0) + 1
            end
        end
        for ip in data
            ip == 0 ? flush_bt() : push!(bt, ip)
        end
        isempty(bt) || flush_bt()
        pct(k) = round(100 * k / total; digits = 1)
        println("PROFILE row=$row size=$n reps=$reps samples=$total min_assemble_s=$t")
        println("FUNCTIONS (share of the samples inside `assemble`)")
        println(rpad("function", 44), lpad("incl %", 8), lpad("self %", 8))
        for (name, c) in sort(collect(incl); by = last, rev = true)
            c < 0.005 * total && break
            println(rpad(name, 44), lpad(pct(c), 8), lpad(pct(get(self, name, 0)), 8))
        end
        println("STAGES (share of the samples inside `assemble`, inclusive)")
        for (label, name) in (("pattern copy", "_allocate_assembled"),
            ("zero stored values", "_zero_stored!"), ("keep pattern", "_keep_pattern!"),
            ("replay", "_replay_bilinear_core!"),
            ("  setup: _bind_interp_spaces", "_bind_interp_spaces"),
            ("  setup: _check_block_meshes", "_check_block_meshes"),
            ("  setup: host_weights", "host_weights"),
            ("  interior core", "_visit_interior!"),
            ("  guarded boundary shell", "_visit_guarded_region!"),
            ("  stencil evaluation", "local_stencil"),
            ("  guard: _test_inbounds", "_test_inbounds"),
            ("  guard: _trial_inbounds", "_trial_inbounds"),
            ("  value addition", "_replay_add!"))
            println(rpad(label, 34), lpad(pct(get(incl, name, 0)), 8))
        end
        println(rpad("garbage collection", 34), lpad(pct(gc), 8))
        units(a, Ωc)
        return nothing
    end

    function argc(name, default)
        i = findfirst(==(name), ARGS)
        return i === nothing ? default : ARGS[i + 1]
    end

    if "--child-profile" in ARGS
        profile(argc("--row", "elasticity"), parse(Int, argc("--size", "30")),
            parse(Float64, argc("--seconds", "4")))
    else
        r = measure(argc("--child", ""), parse(Int, argc("--size", "30")))
        println("RESULT ", join(("$k=$v" for (k, v) in pairs(r)), " "))
    end
    exit(0)
end

function arg(name, default)
    i = findfirst(==(name), ARGS)
    return i === nothing ? default : ARGS[i + 1]
end

function launch(tree, row, n, threads)
    cmd = `$(Base.julia_cmd()) --project=$tree --threads=$threads --startup-file=no
        $(abspath(@__FILE__)) --child $row --size $n`
    for line in eachline(cmd)
        startswith(line, "RESULT ") || continue
        kv = (split(s, "=") for s in split(line)[2:end])
        return Dict(String(k) => parse(Float64, v) for (k, v) in kv)
    end
    error("child for $row printed no RESULT line")
end

function median(v)
    s = sort(v)
    return isodd(length(s)) ? s[(end + 1) ÷ 2] : (s[end ÷ 2] + s[end ÷ 2 + 1]) / 2
end

function main()
    tree = abspath(expanduser(arg("--tree", joinpath(@__DIR__, ".."))))
    n = parse(Int, arg("--size", "30"))
    threads = parse(Int, arg("--threads", "1"))
    if "--profile" in ARGS
        run(`$(Base.julia_cmd()) --project=$tree --threads=$threads --startup-file=no
            $(abspath(@__FILE__)) --child-profile --row $(arg("--row", "elasticity"))
            --size $n --seconds $(arg("--seconds", "4"))`)
        return nothing
    end
    launches = parse(Int, arg("--launches", "3"))
    runs = Dict(r => Vector{Dict{String, Float64}}() for r in ROWS)
    for _ in 1:launches, r in ROWS
        push!(runs[r], launch(tree, r, n, threads))
    end
    println("median of $launches launch(es) of min of 5: tree $tree, size $n, ",
        "$threads thread(s)")
    println(rpad("row", 18), lpad("fill s", 11), lpad("ns/unknown", 12),
        lpad("ns/add", 9), lpad("assemble s", 12), lpad("copy s", 10), lpad("replay s", 10),
        lpad("units", 7), lpad("adds/nnz", 10), lpad("ndofs", 9), lpad("nnz", 10))
    table = Dict{String, Any}()
    for r in ROWS
        med(k) = median([x[k] for x in runs[r]])
        x = runs[r][1]
        fill = med("fill")
        row = Dict{String, Any}("fill" => fill,
            "fill_launches" => [y["fill"] for y in runs[r]],
            "assemble" => med("assemble"), "copy" => med("copy"), "zero" => med("zero"),
            "replay" => med("replay"), "ns_per_unknown" => 1e9 * fill / x["ndofs"],
            "ns_per_add" => 1e9 * med("replay") / x["adds"],
            "bytes_fill" => Int(med("bytes_fill")),
            "bytes_assemble" => Int(med("bytes_assemble")))
        for k in ("ndofs", "nnz", "units", "adds", "points")
            row[k] = Int(x[k])
        end
        merge!(row, Dict("launches" => launches, "threads" => threads, "size" => n))
        table[r] = row
        sig(v) = round(v; sigdigits = 4)
        println(rpad(r, 18), lpad(sig(fill), 11), lpad(sig(row["ns_per_unknown"]), 12),
            lpad(sig(row["ns_per_add"]), 9), lpad(sig(row["assemble"]), 12),
            lpad(sig(row["copy"]), 10), lpad(sig(row["replay"]), 10),
            lpad(row["units"], 7), lpad(sig(row["adds"] / row["nnz"]), 10),
            lpad(row["ndofs"], 9), lpad(row["nnz"], 10))
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
