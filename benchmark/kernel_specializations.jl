#===========================================================================#
# Tabulates the compiled instances of each replay kernel of the bilinear assembly
# (gpena/Bramble.jl#916). Each row runs in a fresh child process: it loads Bramble, builds a
# graded 5x5x5 mesh and `gridspace(Ωₕ)^Val(3)`, then calls `form` and `assemble` once. For
# every kernel in `KERNELS` and every method of it, the child walks
# `Base.specializations(m)` and records the instances and, per argument, the number of
# distinct concrete types that argument takes across them.
#
# The count is of all compiled instances in the process after one form and assemble, the
# ones loaded from the package image included. `baseline_instances` is the same count taken
# after `using Bramble` and before the mesh and form are built: subtract it to get what the
# form and assemble add over loading alone.
#
# Rows are those of benchmark/trace_compile.jl: `elasticity` and `vector_laplacian`. Plain
# Julia, Base and TOML only, launched with no `--project` of its own: the children run
# against the project at `--tree`, at the default `-O` (asserted).
#
#     julia --startup-file=no benchmark/kernel_specializations.jl [--tree DIR]
#         [--rows a,b] [--threads 1] [--out FILE]
#
# Per row, in TOML `<row>.kernels.<name>`
#   instances, baseline_instances    summed over the kernel's methods
#   args    an array of tables, one per method and argument position (`#self#` left out):
#           `method` (file:line), `method_instances`, `position`, `name` (from
#           `Base.method_argnames`), `distinct_types` (distinct concrete types in the slot),
#           `abstract_types` (distinct non-concrete ones, from signatures compiled without
#           full specialisation, mostly image entries) and `labels`, the outer type names of
#           the slot, e.g. `OperatorScale`, `ReplaySink`
# and `<row>.baseline_instances`, `<row>.instances`, `commit`, `julia`, `threads`.
#
# A name in `KERNELS` that Bramble does not define stops the script, so a rename is caught;
# `OPTIONAL_KERNELS` are included only when defined.
#===========================================================================#

const KERNELS = ["_foreach_summand_unit", "_foreach_block_unit", "_replay_summands!",
    "_replay_blocks!", "_replay_pair!", "_foreach_pair_block_unit", "_replay_unit!",
    "_replay_pair_unit!"]
const OPTIONAL_KERNELS = ["_replay_pair_blocks!", "_visit_guarded_region!"]

using TOML

# --- Child: one fresh process, one form and assemble ----------------------- #

if "--child" in ARGS
    # --child <row> <out.toml>
    row, child_out = ARGS[2], ARGS[3]
    Base.JLOptions().opt_level == 2 ||
        error("needs the default -O (2), got $(Base.JLOptions().opt_level)")
    using Bramble
    import Bramble: change_points!

    function kernel_functions()
        names = copy(KERNELS)
        for n in KERNELS
            isdefined(Bramble, Symbol(n)) ||
                error("kernel $n is not defined in Bramble: renamed? update KERNELS")
        end
        append!(names, filter(n -> isdefined(Bramble, Symbol(n)), OPTIONAL_KERNELS))
        return [n => getfield(Bramble, Symbol(n)) for n in names]
    end

    instances_of(m::Method) = [mi for mi in Base.specializations(m) if mi !== nothing]

    count_instances(f) = sum(length(instances_of(m)) for m in methods(f); init = 0)

    # The outer type name of a slot type, without parameters.
    function label(@nospecialize(T))
        T isa Core.TypeofVararg && return "Vararg"
        T isa Union && return "Union"
        T isa UnionAll && return label(Base.unwrap_unionall(T))
        T isa DataType || return string(T)
        T <: Type && T !== Type && return "Type"
        T <: Function && isdefined(T, :instance) && return "typeof($(T.instance))"
        return string(nameof(T))
    end

    function tabulate(f)
        args = Dict{String, Any}[]
        total = 0
        for m in methods(f)
            mis = instances_of(m)
            total += length(mis)
            argnames = Base.method_argnames(m)
            sigs = [Base.unwrap_unionall(mi.specTypes).parameters for mi in mis]
            for p in 2:length(argnames)
                slot = Set{Any}(s[p] for s in sigs if length(s) >= p)
                concrete = count(isconcretetype, slot)
                push!(args,
                    Dict{String, Any}("method" => "$(m.file):$(m.line)",
                        "method_instances" => length(mis), "position" => p - 1,
                        "name" => string(argnames[p]), "distinct_types" => concrete,
                        "abstract_types" => length(slot) - concrete,
                        "labels" => sort!(unique!([label(T) for T in slot]))))
            end
        end
        return total, args
    end

    # The points of an axis: the exponent differs per axis, so no two axes agree.
    graded(n, axis) = collect(range(0.0, 1.0; length = n)) .^ (1.2 + 0.2 * axis)

    function build(row)
        n = (5, 5, 5)
        Ωₕ = mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), n)
        change_points!(Ωₕ, ntuple(d -> graded(n[d], d), 3))
        W = gridspace(Ωₕ)^Val(3)
        row == "vector_laplacian" && return W, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v))
        row == "elasticity" || error("unknown row $row")
        μ, λ = 1.0, 2.0
        return W, (u, v) -> 2μ * inner₊(εₕ(u), εₕ(v)) + λ * inner₊(divₘₕ(u), divₘₕ(v))
    end

    kernels = Base.invokelatest(kernel_functions)
    baseline = Dict(n => Base.invokelatest(count_instances, f) for (n, f) in kernels)

    # Behind `invokelatest`, so no inference of a caller compiles anything ahead of the call.
    W, fun = Base.invokelatest(build, row)
    a = Base.invokelatest(form, W, W, fun)
    Base.invokelatest(assemble, a)

    table = Dict{String, Any}()
    for (n, f) in kernels
        total, args = Base.invokelatest(tabulate, f)
        table[n] = Dict{String, Any}("instances" => total,
            "baseline_instances" => baseline[n], "args" => args)
    end
    open(io -> TOML.print(io, Dict("kernels" => table); sorted = true), child_out, "w")
    exit(0)
end

# --- Parent ---------------------------------------------------------------- #

const ROWS = ["elasticity", "vector_laplacian"]

function arg(name, default)
    i = findfirst(==(name), ARGS)
    return i === nothing ? default : ARGS[i + 1]
end

function git_commit(tree)
    try
        return strip(read(`git -C $tree rev-parse --short HEAD`, String))
    catch
        return "unknown"
    end
end

function run_child(tree, row, threads)
    out = joinpath(mktempdir(), "$row.toml")
    cmd = `$(Base.julia_cmd()) --project=$tree --threads=$threads --startup-file=no
        $(abspath(@__FILE__)) --child $row $out`
    run(cmd)
    return TOML.parsefile(out)
end

function print_table(row, table)
    k = table["kernels"]
    println("\n### ", row, "\n")
    println("instances ", table["instances"], " (", table["baseline_instances"],
        " before the form), commit ", table["commit"], ", Julia ", table["julia"])
    println("\n| kernel | instances | baseline | method (instances) | argument | ",
        "distinct_types | abstract_types | labels |")
    println("|---|---:|---:|---|---|---:|---:|---|")
    for n in sort!(collect(keys(k)))
        v = k[n]
        first_line = true
        for a in v["args"]
            head = first_line ? "`$n` | $(v["instances"]) | $(v["baseline_instances"])" :
                   " | | "
            println("| ", head, " | ", basename(a["method"]), " (", a["method_instances"],
                ") | ", a["position"], " `", a["name"], "` | ", a["distinct_types"], " | ",
                a["abstract_types"], " | ", join(a["labels"], ", "), " |")
            first_line = false
        end
        first_line && println("| `", n, "` | ", v["instances"], " | ",
            v["baseline_instances"], " | | | | | |")
    end
    return nothing
end

function main()
    tree = abspath(expanduser(arg("--tree", joinpath(@__DIR__, ".."))))
    rows = split(arg("--rows", join(ROWS, ",")), ",")
    all(in(ROWS), rows) || error("--rows must be among $(join(ROWS, ", "))")
    threads = parse(Int, arg("--threads", "1"))
    out = arg("--out", nothing)
    commit = git_commit(tree)
    tables = Dict{String, Any}()
    for r in rows
        table = run_child(tree, r, threads)
        k = table["kernels"]
        table["instances"] = sum(v["instances"] for v in values(k); init = 0)
        table["baseline_instances"] = sum(v["baseline_instances"] for v in values(k);
            init = 0)
        table["commit"] = commit
        table["julia"] = string(VERSION)
        table["threads"] = threads
        tables[r] = table
        print_table(r, table)
    end
    if out !== nothing
        data = isfile(out) ? TOML.parsefile(out) : Dict{String, Any}()
        merge!(data, tables)
        open(io -> TOML.print(io, data; sorted = true), out, "w")
        println("\nwrote ", out)
    end
    return nothing
end

main()
