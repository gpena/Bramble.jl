#===========================================================================#
# Lists the predicates of Bramble that infer a return type other than `Bool`
# (gpena/Bramble.jl#913). A predicate that infers `Any` or `Union{Bool,Nothing}` at some
# call site sends its caller through a runtime dispatch or a boxed branch, so each such
# instance is a site worth reading.
#
# One process, run under `--project=test` (JET lives there). Per row it builds the form
# once, runs `form` and `assemble` through `Base.invokelatest` (so inference happens at call
# time), then measures twice.
#
# Rows
#   elasticity   as benchmark/trace_compile.jl: a graded 5x5x5 mesh and
#                `gridspace(Ωₕ)^Val(3)`, `2μ * inner₊(εₕ(u), εₕ(v)) +
#                λ * inner₊(divₘₕ(u), divₘₕ(v))`
#   sum40        a graded 9x11 scalar space, `foldl(+, ntuple(k -> (k + 0.5) *
#                inner₊(∇ₕ(u), ∇ₕ(v)), Val(40)))`. The `Float64` coefficients carry
#                data, so no like terms fold: the row records `summands`, counted on the
#                simplified AST (`form.ast`), and the script stops when it is not 40.
#
# Per row, in the TOML table `<row>`
#   summands                the summands of `form.ast`, the leaves of its `OperatorAdd` tree
#   jet_form_reports        `JET.report_opt` on `form(W, W, fun)` with
#   jet_assemble_reports    `assemble(a)`, `target_modules = (Bramble,)`: the reports
#   sites                   an array of tables `function` (`Module.name`), `signature` (the
#                           instance's `specTypes`), `rettype`, `callers` (see below), from
#                           the detector
#   commit, julia
#
# The detector
#   For every function owned by Bramble or a submodule (a value bound in the module, found
#   through `names(m; all = true)`, whose `parentmodule` is in Bramble; closures and keyword
#   bodies included), walk `Base.specializations` of each of its methods and read the
#   return type of the instance from its CodeInstance chain (`mi.cache`, then `.next`), so
#   instances inferred but never compiled count too. Only a CodeInstance of the native
#   cache (`owner === nothing`, never the one JET fills) valid in the current world counts.
#   A function is a predicate when at least one instance returns `Bool` and none returns a
#   concrete non-`Bool` type. A site is an instance of a predicate whose return type is not
#   `Bool`. An instance that infers `Union{}` (it only throws) is no site and no `Bool`
#   instance: it never produces a value, so it cannot box a branch, and one throwing
#   specialisation would not turn a predicate into a non-predicate either. `callers` are
#   the methods whose runtime-dispatch reports in the JET runs name the function (the first
#   element of the report's call signature), the report's innermost frame; empty when JET
#   saw no dispatch to it.
#
# Not covered are the methods Bramble adds to a function of another module (`Base.isempty`
# and the like). The detector cannot enumerate them without scanning every loaded module,
# and they would mix unrelated methods under one name. Rows share one process and its
# cache, so a second row lists the instances the first added too. Run `--rows` with one row
# for a clean table. A name pattern (`is_*`) for predicates was rejected, since it misses
# `_holds_rank_one` and `_mixes_components`.
#
#     julia --startup-file=no --project=test benchmark/bool_audit.jl [--control]
#         [--rows elasticity,sum40] [--out FILE]
#
# `--control` defines `AuditControl` with a predicate that infers `Any` for one instance and
# `Bool` for another (`isp`) and one that only infers `Bool` (`isq`), calls them, runs the
# detector on that module, and stops unless `isp` is a site and `isq` is not. The result is
# `control.sites` in the output. `--out FILE` merges the tables into FILE. JET segfaults
# intermittently on Julia 1.13: run again once before calling it a failure.
#===========================================================================#

using Bramble
using JET
using TOML
import Bramble: change_points!

# --- The control ----------------------------------------------------------- #

# `x > 0` on an `Any` argument infers `Any`, and `@nospecialize` makes that the method's
# only instance. `isp(::Int)` adds the `Bool` instance a predicate needs to be one.
module AuditControl
isp(@nospecialize x) = x > 0
isp(x::Int) = x > 0
isq(x::Int) = x > 0
end

function run_control()
    Base.invokelatest(AuditControl.isp, 1)
    Base.invokelatest(AuditControl.isp, 1.5)
    Base.invokelatest(AuditControl.isq, 1)
    return nothing
end

# --- The detector ---------------------------------------------------------- #

function submodules(root::Module)
    found = Module[root]
    for n in names(root; all = true, imported = false)
        isdefined(root, n) || continue
        v = getfield(root, n)
        v isa Module && v !== root && parentmodule(v) === root &&
            append!(found, submodules(v))
    end
    return found
end

function within(m::Module, root::Module)
    return m === root || (m !== parentmodule(m) && within(parentmodule(m), root))
end

# `Module.name`, the module path without `Main`.
function qualified(m::Module, name)
    path = [n for n in fullname(m) if n !== :Main]
    return join([string.(path); string(name)], ".")
end

function owned_functions(root::Module)
    fns = Any[]
    for m in submodules(root), n in names(m; all = true, imported = false)
        isdefined(m, n) || continue
        f = getfield(m, n)
        f isa Function && within(parentmodule(f), root) && !any(g -> g === f, fns) &&
            push!(fns, f)
    end
    return fns
end

# The return type of an instance from its native, valid CodeInstances, or `nothing`.
function inferred_rettype(mi::Core.MethodInstance, world)
    isdefined(mi, :cache) || return nothing
    best = nothing
    ci = mi.cache
    while true
        if ci.owner === nothing && ci.min_world <= world <= ci.max_world &&
           (best === nothing || ci.min_world > best.min_world)
            best = ci
        end
        isdefined(ci, :next) || break
        ci = ci.next
    end
    return best === nothing ? nothing : best.rettype
end

# The instances of every owned function with a return type: `(f, mi, rettype)`.
function inferred_instances(root::Module)
    world = Base.get_world_counter()
    out = Tuple{Any, Core.MethodInstance, Any}[]
    for f in owned_functions(root), m in methods(f), mi in Base.specializations(m)
        mi isa Core.MethodInstance || continue
        rt = inferred_rettype(mi, world)
        rt === nothing || push!(out, (f, mi, rt))
    end
    return out
end

# The sites in `root`: `(f, mi, rettype)` of every non-`Bool` instance of a predicate.
function detect_sites(root::Module)
    instances = inferred_instances(root)
    has_bool = Dict{Any, Bool}()
    has_other = Dict{Any, Bool}()
    for (f, _, rt) in instances
        rt === Bool && (has_bool[f] = true)
        isconcretetype(rt) && rt !== Bool && (has_other[f] = true)
    end
    return [(f, mi, rt) for (f, mi, rt) in instances
            if get(has_bool, f, false) && !get(has_other, f, false) && rt !== Bool &&
               rt !== Union{}]
end

# --- JET ------------------------------------------------------------------- #

function dispatch_reports(result)
    return filter(r -> r isa JET.RuntimeDispatchReport, JET.get_reports(result))
end

# The callers named by the runtime-dispatch reports of `f`: the method of each report's
# innermost frame.
function callers_of(f, reports)
    callers = String[]
    for r in reports
        tt = Base.unwrap_unionall(r.sig.tt)
        (tt isa DataType && !isempty(tt.parameters) && tt.parameters[1] === typeof(f)) ||
            continue
        mi = r.vst[end].linfo
        mi isa Core.MethodInstance && mi.def isa Method &&
            push!(callers, qualified(mi.def.module, mi.def.name))
    end
    return sort!(unique!(callers))
end

# --- Rows ------------------------------------------------------------------ #

# The points of an axis: the exponent differs per axis, so no two axes agree.
graded(n, axis) = collect(range(0.0, 1.0; length = n)) .^ (1.2 + 0.2 * axis)

function build(row)
    if row == "sum40"
        n = (9, 11)
        Ωₕ = mesh(domain(box((0.0, 0.0), (1.0, 1.0))), n)
        change_points!(Ωₕ, ntuple(d -> graded(n[d], d), 2))
        fun = (u, v) -> foldl(+,
            ntuple(k -> (k + 0.5) * inner₊(∇ₕ(u), ∇ₕ(v)), Val(40)))
        return gridspace(Ωₕ), fun
    end
    row == "elasticity" || error("unknown row $row")
    n = (5, 5, 5)
    Ωₕ = mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), n)
    change_points!(Ωₕ, ntuple(d -> graded(n[d], d), 3))
    μ, λ = 1.0, 2.0
    fun = (u, v) -> 2μ * inner₊(εₕ(u), εₕ(v)) +
                    λ * inner₊(divₘₕ(u), divₘₕ(v))
    return gridspace(Ωₕ)^Val(3), fun
end

function summand_count(op::Bramble.OperatorAdd)
    return summand_count(op.left_op) + summand_count(op.right_op)
end
summand_count(::Any) = 1

function site_tables(sites, reports)
    return [Dict{String, Any}("function" => qualified(parentmodule(f), nameof(f)),
                "signature" => string(mi.specTypes), "rettype" => string(rt),
                "callers" => callers_of(f, reports)) for (f, mi, rt) in sites]
end

function audit_row(row)
    W, fun = Base.invokelatest(build, row)
    a = Base.invokelatest(form, W, W, fun)
    Base.invokelatest(assemble, a)
    summands = summand_count(a.ast)
    row == "sum40" && summands != 40 &&
        error("sum40 holds $summands summands after simplification, not 40")
    jet_form = JET.report_opt(form, Tuple{typeof(W), typeof(W), typeof(fun)};
        target_modules = (Bramble,))
    jet_assemble = JET.report_opt(assemble, Tuple{typeof(a)}; target_modules = (Bramble,))
    reports = [dispatch_reports(jet_form); dispatch_reports(jet_assemble)]
    sites = Base.invokelatest(detect_sites, Bramble)
    return Dict{String, Any}("summands" => summands,
        "jet_form_reports" => length(JET.get_reports(jet_form)),
        "jet_assemble_reports" => length(JET.get_reports(jet_assemble)),
        "sites" => site_tables(sites, reports))
end

# --- Parent ---------------------------------------------------------------- #

function arg(name, default)
    i = findfirst(==(name), ARGS)
    return i === nothing ? default : ARGS[i + 1]
end

function git_commit()
    try
        return strip(read(`git -C $(@__DIR__) rev-parse --short HEAD`, String))
    catch
        return "unknown"
    end
end

function control_table()
    run_control()
    sites = Base.invokelatest(detect_sites, AuditControl)
    found = [qualified(parentmodule(f), nameof(f)) for (f, _, _) in sites]
    "AuditControl.isp" in found || error("the control predicate isp was not found: $found")
    "AuditControl.isq" in found && error("the control predicate isq was reported: $found")
    return Dict{String, Any}("sites" => site_tables(sites, []))
end

function print_sites(title, sites)
    println("\n### ", title, ": ", length(sites), " site(s)\n")
    for s in sites
        println("- `", s["function"], "` returns `", s["rettype"], "`, callers ",
            isempty(s["callers"]) ? "none named" : join(s["callers"], ", "))
    end
    return nothing
end

function write_out(out, tables)
    out === nothing && return nothing
    data = isfile(out) ? TOML.parsefile(out) : Dict{String, Any}()
    merge!(data, tables)
    open(io -> TOML.print(io, data; sorted = true), out, "w")
    println("\nwrote ", out)
    return nothing
end

function main()
    rows = String.(split(arg("--rows", "elasticity,sum40"), ","))
    all(in(["elasticity", "sum40"]), rows) ||
        error("--rows must be among elasticity, sum40")
    out = arg("--out", nothing)
    tables = Dict{String, Any}()
    if "--control" in ARGS
        tables["control"] = control_table()
        print_sites("control", tables["control"]["sites"])
    end
    commit = git_commit()
    for row in rows
        t = audit_row(row)
        t["commit"], t["julia"] = commit, string(VERSION)
        tables[row] = t
        println("\n", row, ": ", t["summands"], " summands, JET reports ",
            t["jet_form_reports"], " (form), ", t["jet_assemble_reports"], " (assemble)")
        print_sites(row, t["sites"])
    end
    write_out(out, tables)
    return nothing
end

main()
