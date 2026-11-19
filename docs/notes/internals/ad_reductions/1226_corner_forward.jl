# Reproducer for gpena/Bramble.jl#1226: Enzyme forward mode, with runtime activity, gives a
# wrong gradient of `closure(constrained(S × S, ...))` when a parameter enters the operator
# of a traction relation on `:xmin` and `:ymin` (the `β` case of the issue). One case per
# process.
#
# Run from the repository root, with one thread, at Julia's default optimisation level:
#
#     julia --startup-file=no --project=. --threads=1 \
#         docs/notes/internals/ad_reductions/1226_corner_forward.jl <case>
#
# Cases, all on the issue's seeded non-uniform (7, 6) mesh with ghosts on every facet
# under `CpuSerial`, and `p0 = [2.0]` unless said otherwise.
#   beta      the issue's reproducer, `β * u(1) + trac(u, 1)` on `:xmin`, `β * u(2) +
#             trac(u, 2) => 0.3` on `:ymin`, `:xmax` and `:ymax` clamped, the loss
#             `sum(abs2, nonzeros(E)) + sum(abs2, d)` of `closure`. Width 1.
#   beta2     the same loss with a second, unused parameter, `p0 = [2.0, 0.5]`, width 2.
#   betag     the issue's width-2 form, `p0 = [2.0, 0.3]`, the `:ymin` datum `p[2]`.
#   nora      `beta` without `set_runtime_activity`.
#   inline    `beta` with `_composite_relations` (src/assembly/ghost_relations.jl)
#             redefined from this script with its body unchanged and `@inline` added.
#   nocorner  `beta` with the `:ymin` relations reading one component each (`β * u(2) +
#             ∂ₙ(u(2)) => 0.3` and `∂ₙ(u(1)) => 0.0`), so no corner is one-sided.
#   rels      the smallest: the four traction relations of `beta` parsed by
#             `_composite_relations` on the padded meshes, the fourth (`:ymin`, datum 0.3)
#             passed to a `@noinline` function that returns `([β], [datum])`, and the loss
#             `β² + datum`. Its derivative is 2β = 4. No space, form or closure is built
#             inside the differentiated function.
#   relsinl   `rels` with the `@inline` redefinition of `inline`.
#   relsnc    `rels` on the four relations of `nocorner`.
#   relsnora  `rels` without `set_runtime_activity`.
#   relsfld   `rels` with the relation rebuilt from its fields (`_FacetRelation(r.d, r.s,
#             r.name, r.op, r.data, r.c)`) before the `@noinline` call.
#
# Every case calls Enzyme the way the issue does, `DifferentiationInterface.gradient` with
# `AutoEnzyme(; mode = set_runtime_activity(Forward), function_annotation = Const)` (no
# runtime activity for `nora` and `relsnora`). The reference is ForwardDiff on the same
# loss. The relative error is the largest absolute difference over the largest absolute
# reference; `ok` is at most 1e-8, `wrong` past it, `fails` when Enzyme throws.
#
# The limit is `S1226_LIMIT` seconds (default 1200), enforced by a shell watchdog on the
# script's own pid, as in 1174_composite_ghost.jl. Enzyme, DifferentiationInterface and
# ForwardDiff load from a temporary environment.
#
# Output: `Enzyme: <version>`, then one line
# `Case: <case> | fwd | width <w> | <ok, wrong or fails> | <relerr>`, then the gradients.
const LIMIT = parse(Int, get(ENV, "S1226_LIMIT", "1200"))
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

const CASES = ("beta", "beta2", "betag", "nora", "inline", "nocorner", "rels", "relsinl",
    "relsnc", "relsnora", "relsfld")
length(ARGS) == 1 && ARGS[1] in CASES ||
    error("usage: 1226_corner_forward.jl <", join(CASES, "|"), ">")
const CASE = ARGS[1]

# `_composite_relations` as at 12d859e1, with `@inline` added and nothing else changed.
if CASE in ("inline", "relsinl")
    @eval B @inline function _composite_relations(Gs::Tuple, out)
        pairs = _relation_pairs(out)
        isempty(pairs) && _throw_relation_shape(out)
        keys = map(p -> _pair_key(Val(dim(first(Gs))), p, true), pairs)
        rels = _parse_in_order(p -> _composite_relation(Gs, p), pairs) do r, i
            _is_dirichlet(r) && any(j -> keys[j] == keys[i], 1:(i - 1)) &&
                _throw_facet_twice(r.name, true, r.c)
            return nothing
        end
        key(r) = (r.c, r.d, r.s, _is_dirichlet(r))
        return _with_components(rels,
            _attach_relations(collect(map(key, rels)), length(Gs)))
    end
end

function gm(ns)
    B._seed_mesh1d_rng!(11)
    Ω = mesh(domain(foldl(×, ntuple(d -> interval(0.0, Float64(d)), length(ns)))), ns,
        ntuple(_ -> true, length(ns)); backend = backend(policy = B.CpuSerial()))
    B._unseed_mesh1d_rng!()
    return Ω
end
Kt(i, j, λ) = [λ * (i == d) * (j == e) + 0.5 * ((i == j) * (d == e) + (i == e) * (j == d))
               for d in 1:2, e in 1:2]
trac(u, i, λ) = ∂ₙ(u(1); K = Kt(i, 1, λ)) + ∂ₙ(u(2); K = Kt(i, 2, λ))
clamped(F, u) = (F => u(1) => 0.0, F => u(2) => 0.0, F => ∂ₙₙ(u(1)) => 0.0,
    F => ∂ₙₙ(u(2)) => 0.0)
const S = gridspace(gm((7, 6)); ghosts = :boundary)
sq(C) = ((E, d) = B.closure(C); sum(abs2, nonzeros(E)) + sum(abs2, d))

tractions(β, g = 0.3) = u -> (:xmin => β * u(1) + trac(u, 1, 2.0) => 1.0,
    :xmin => trac(u, 2, 2.0) => 0.0, :ymin => trac(u, 1, 2.0) => 0.0,
    :ymin => β * u(2) + trac(u, 2, 2.0) => g)
fβ(p) = sq(constrained(S × S, u -> (tractions(p[1])(u)..., clamped(:xmax, u)...,
    clamped(:ymax, u)...)))
fβ2(p) = fβ(p[1:1])
fβg(p) = sq(constrained(S × S, u -> (tractions(p[1], p[2])(u)..., clamped(:xmax, u)...,
    clamped(:ymax, u)...)))
single(β) = u -> (:xmin => β * u(1) + trac(u, 1, 2.0) => 1.0,
    :xmin => trac(u, 2, 2.0) => 0.0, :ymin => ∂ₙ(u(1)) => 0.0,
    :ymin => β * u(2) + ∂ₙ(u(2)) => 0.3)
fno(p) = sq(constrained(S × S, u -> (single(p[1])(u)..., clamped(:xmax, u)...,
    clamped(:ymax, u)...)))

const P = B.padded_space(S × S)
const GS = map(((C, _, _),) -> mesh(C), B._padded_leaves(S × S))
# The relation's `β`: the scalar of `β * u(2)`, its first term on the facet.
@noinline scalar_and_datum(r) = ([_beta(r.op)], [r.data])
_beta(op::B.FacetOperator) = op.ops[1][1].left_op.scalar
_beta(op) = op.left_op.scalar
function frels(p, rels = tractions; rebuild = false)
    r = B._composite_relations(GS, rels(p[1])(B.trial_function(P)))[4]
    rebuild && (r = B._FacetRelation(r.d, r.s, r.name, r.op, r.data, r.c))
    x, y = scalar_and_datum(r)
    return sum(abs2, x) + sum(y)
end

const f, p0 = CASE in ("rels", "relsinl", "relsnora") ? (frels, [2.0]) :
              CASE == "relsnc" ? (p -> frels(p, single), [2.0]) :
              CASE == "relsfld" ? (p -> frels(p; rebuild = true), [2.0]) :
              CASE == "beta2" ? (fβ2, [2.0, 0.5]) :
              CASE == "betag" ? (fβg, [2.0, 0.3]) :
              CASE == "nocorner" ? (fno, [2.0]) : (fβ, [2.0])
const MODE = CASE in ("nora", "relsnora") ? Enzyme.Forward :
             Enzyme.set_runtime_activity(Enzyme.Forward)
const AD = AutoEnzyme(; mode = MODE, function_annotation = Enzyme.Const)

println("Enzyme: ", pkgversion(Enzyme), ", Julia ", VERSION, ", threads ",
    Threads.nthreads(), ", Bramble at ", pkgdir(Bramble))
flush(stdout)
ref = ForwardDiff.gradient(f, p0)

function measure(ref)
    try
        g = DifferentiationInterface.gradient(f, AD, p0)
        relerr = maximum(abs, g .- ref) / maximum(abs, ref)
        return (relerr <= 1e-8 ? "ok" : "wrong"), string(relerr), g
    catch err
        err isa InterruptException && rethrow()
        println(first(sprint(showerror, err), 1500))
        return "fails", "NaN", nothing
    end
end

t = time()
status, relerr, g = measure(ref)
println("Case: ", CASE, " | fwd | width ", length(p0), " | ", status, " | ", relerr)
println("enzyme ", g, " forwarddiff ", ref, " seconds ", round(time() - t; digits = 1))
flush(stdout)
exit(status == "ok" ? 0 : 1)
