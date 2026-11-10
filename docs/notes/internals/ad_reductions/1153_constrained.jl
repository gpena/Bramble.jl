# Reproducer for gpena/Bramble.jl#1153: Enzyme through `constrained(W, relations)` when the
# data of a relation depend on the parameter. No form of the user and no assembly: the
# differentiated function builds the constrained space and reads its closure. One step of
# the construction, one parameter-dependent case and one mode per process.
#
# Run from the repository root, with one thread:
#
#     julia --startup-file=no --project=. --threads=1 --optimize=1 --check-bounds=auto \
#         docs/notes/internals/ad_reductions/1153_constrained.jl <step> <case> <fwd|rev>
#
# Cases. The space is `1127_ghost.jl`'s, a 7 by 6 non-uniform mesh with ghosts on xmin, xmax
# and ymax, and the relations are those of `constrained_with` there, one datum replaced by
# p[1].
#   data       the Neumann datum of xmax, `β * u + ∂ₙ(u) => p[1]` (`data_rev` of 1127).
#   robin      the Robin coefficient of xmax, `p[1] * u + ∂ₙ(u) => 1.0` (`robin_rev`).
#   dirichlet  the Dirichlet datum of ymin, `u => p[1]`.
#   conormal   a space with ghosts on every facet, the relations `∂ₙ(u; K) => x -> p[1] *
#              (x[1] - x[2])` on the four facets, K = [2 1; 1 2] (the test of
#              test/form/ghost_rank_one.jl). The ghosts of a corner node are coupled, so
#              `_block_solve` takes its dense branch at run time.
#   const      no parameter in the relations, `constrained` run outside, for `downstream` only.
#
# Steps, in the order `constrained` runs them (src/assembly/ghost_relations.jl). A step is a
# function of the parameter. What the steps before it give and the parameter does not touch
# is computed outside, at the parameter's value, and passed in. What the parameter does
# touch is rebuilt inside as `a0 + p[1] * a1`, the affine map of the step's earlier
# outputs, exact here because the data and the coefficient enter the relations linearly
# (`inputs` is the control that this rebuilding itself differentiates). The steps
# `parse`, `relation`, `rows` and `space` rebuild the relations from the parameter, since
# the parameter lives in the relations' types.
#   inputs         the control, the affine rebuild with nothing of Bramble.
#   parse          `_facet_relations`, `_ghost_relations`, `_dirichlet_relations`.
#   relation       `_assemble_relation` of the xmax relation (forms on the padded space).
#   relation_load, relation_matrix, relation_transpose
#                  the parts of `_assemble_relation` alone: the data's load, the operator's
#                  matrix, the transposition of that matrix to the rows' layout.
#   rows           `_relation_rows`.
#   dirichlet      `_dirichlet_values`.
#   kept           `_kept_points`, integers only, the parameter scales the sum.
#   bookkeeping    the union-find and the `Dict` of `_block_solve`, integers only.
#   lu             the dense kernel of `_block_solve` (lines 1217 to 1231), `lu`, `F \ Bb`
#                  and `F \ b` on a 2 by 2 block, whose entries are active (the cases'
#                  own blocks hold one ghost, so the kernel is not run for them).
#   block_solve    `_block_solve` as `_relation_closure` calls it.
#   closure        `_relation_closure`.
#   closure_data   `_closure_matrix` and `_closure_data`.
#   space          `_constrained_space`, the relations parsed inside.
#   constrained    `constrained` itself, the loss of `data_rev` and `robin_rev`.
#   downstream     case `const` only: a form and its assembly on the constrained space
#                  built outside, whose relations carry no parameter.
#
# The reference is a central difference of the same loss (the losses are rational in the
# parameter and h = 1e-5 leaves about 1e-10). A step is `ok` when the relative error, the
# largest absolute difference over the largest absolute reference, is at most 1e-6.
#
# Enzyme is not a dependency of the root project, so it is loaded from a temporary
# environment, as 1127_ghost.jl does. The root Manifest.toml must hold Enzyme.
#
# Output: `step <name> t=<seconds>` lines as it runs, then one line
# `RESULT <step> <case> <mode> <status> <relerr> <seconds> <detail>`. Exit status is 0 for
# `ok`, 1 for `fails` or `wrong`.
println("pid ", getpid())
flush(stdout)
_e = mktempdir()
write(joinpath(_e, "Project.toml"),
    "[deps]\nEnzyme = \"7da242da-08ed-463a-9acd-ee780be4f1d9\"\n")
push!(LOAD_PATH, _e)
using Bramble, Enzyme, LinearAlgebra, SparseArrays, Logging

# Enzyme warns once per stored value it cannot find a use for, thousands of lines of IR.
global_logger(NullLogger())

const T0 = time()
step(name) = (println("step ", name, " t=", round(time() - T0; digits = 1)); flush(stdout))

length(ARGS) == 3 || error("usage: 1153_constrained.jl <step> <case> <fwd|rev>")
const STEP, CASE_NAME, MODE = ARGS
const CASE = Symbol(CASE_NAME)
MODE in ("fwd", "rev") || error("mode must be fwd or rev, not ", MODE)
CASE in (:data, :robin, :dirichlet, :conormal, :const) || error("unknown case ", CASE)

# The `1127_ghost.jl` space.
const Ωₕ = let
    Bramble._seed_mesh1d_rng!(11)
    m = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (7, 6), (false, false))
    Bramble._unseed_mesh1d_rng!()
    m
end
const W = gridspace(Ωₕ; ghosts = CASE === :conormal ? :boundary : (:xmin, :xmax, :ymax))
const P = Bramble.padded_space(W)
const G = mesh(P)
const T = eltype(P)
const BE = backend(P)

# The relations of `constrained_with` of 1127_ghost.jl, the datum `x` in the place `CASE`
# names.
function relations_of(case::Symbol, x)
    if case === :conormal
        K = [2.0 1.0; 1.0 2.0]
        return u -> map(F -> F => ∂ₙ(u; K) => (y -> x * (y[1] - y[2])),
            (:xmin, :xmax, :ymin, :ymax))
    end
    β = case === :robin ? x : 2.0
    g = case === :data ? x : 1.0
    h = case === :dirichlet ? x : 0.0
    return u -> (:xmin => ∂ₙ(u) => 0.0, :xmax => β * u + ∂ₙ(u) => g, :ymin => u => h,
        :ymax => ∂ₙ(u) => 1.0)
end

# What `constrained` does first (src/assembly/ghost_relations.jl, line 110).
function parse_relations(case::Symbol, x)
    rels = Bramble._facet_relations(G, relations_of(case, x)(Bramble.trial_function(P)))
    return Bramble._ghost_relations(rels), Bramble._dirichlet_relations(rels)
end

# `_relation_closure`'s sorting of the rows `R` into `C` (the ghosts' columns), `B` (the
# kept points', as `Bt`) and the right-hand side `bg`, a copy of lines 1113 to 1148 there
# for what the relations here read (no refusal branch).
function split_blocks(R, b, ghosts, keep, fixed, g)
    plin = LinearIndices(npoints(G, Tuple))
    gpts = Int[plin[Bramble._ghost_point(G, k)] for k in ghosts]
    Npad = size(R, 2)
    m, n = length(gpts), length(keep)
    gpos, kpos = zeros(Int, Npad), zeros(Int, Npad)
    gpos[gpts] .= 1:m
    kpos[keep] .= 1:n
    bg = Vector{promote_type(eltype(b), eltype(R), eltype(g))}(b)
    Ci, Cj, Cv = Int[], Int[], eltype(R)[]
    Bi, Bj, Bv = Int[], Int[], eltype(R)[]
    for p in 1:Npad, q in nzrange(R, p)

        v = nonzeros(R)[q]
        iszero(v) && continue
        k = rowvals(R)[q]
        if gpos[p] != 0
            push!(Ci, k), push!(Cj, gpos[p]), push!(Cv, v)
        elseif kpos[p] != 0
            push!(Bi, k), push!(Bj, kpos[p]), push!(Bv, -v)
        elseif fixed[p]
            bg[k] -= v * g[p]
        else
            error("a relation reads an absent point")
        end
    end
    return sparse(Ci, Cj, Cv, m, m), sparse(Bj, Bi, Bv, n, m), bg
end

# `_block_solve`'s union-find and its `Dict` of members (lines 1187 to 1201 there), a copy.
function block_members(C::SparseMatrixCSC)
    m = size(C, 1)
    root = collect(1:m)
    find(k) = (while root[k] != k
            root[k] = root[root[k]]
            k = root[k]
        end; k)
    for j in 1:m, q in nzrange(C, j)

        a, c = find(rowvals(C)[q]), find(j)
        a == c || (root[max(a, c)] = min(a, c))
    end
    members = Dict{Int, Vector{Int}}()
    for k in 1:m
        push!(get!(members, find(k), Int[]), k)
    end
    return members
end

# The value `a0 + x * a1` of something affine in the parameter, measured at two values.
struct Affine{A}
    a0::A
    a1::A
end
function Affine(f1, f2)
    a1 = f2 - f1
    return Affine(f1 - a1, a1)
end
at(a::Affine, p) = a.a0 .+ p[1] .* a.a1

# The same for a sparse matrix of one pattern.
struct SparseAffine
    m::Int
    n::Int
    colptr::Vector{Int}
    rowval::Vector{Int}
    nz::Affine{Vector{Float64}}
end
function SparseAffine(A1::SparseMatrixCSC, A2::SparseMatrixCSC)
    A1.colptr == A2.colptr && A1.rowval == A2.rowval ||
        error("the pattern of the matrix depends on the parameter")
    return SparseAffine(size(A1)..., A1.colptr, A1.rowval,
        Affine(Vector{Float64}(nonzeros(A1)), Vector{Float64}(nonzeros(A2))))
end
function at(a::SparseAffine, p)
    return SparseMatrixCSC(a.m, a.n, a.colptr, a.rowval, at(a.nz, p))
end

# Everything the steps after `parse` take, at the datum `x`.
function stage(case::Symbol, x)
    gr, dr = parse_relations(case, x)
    R, b, ghosts, _ = Bramble._relation_rows(P, gr)
    fixed, g, _, _ = Bramble._dirichlet_values(G, dr, T)
    keep = Bramble._kept_points(G, ghosts, fixed)
    C, Bt, bg = split_blocks(R, b, ghosts, keep, fixed, g)
    E, d, _ = Bramble._relation_closure(G, R, b, ghosts, keep, fixed, g)
    return (; gr, dr, R, b, ghosts, fixed, g, keep, C, Bt, bg, E, d)
end

# The pairs of affine maps the steps rebuild their inputs from.
function affine_inputs(case::Symbol)
    s1, s2 = stage(case, 1.0), stage(case, 2.0)
    return (; s = s2,
        R = SparseAffine(s1.R, s2.R), b = Affine(s1.b, s2.b), g = Affine(Float64.(s1.g),
            Float64.(s2.g)),
        C = SparseAffine(s1.C, s2.C), Bt = SparseAffine(s1.Bt, s2.Bt),
        bg = Affine(s1.bg, s2.bg), E = SparseAffine(s1.E, s2.E), d = Affine(s1.d, s2.d))
end

# What the parameter is in the relation `parse_relations` returns, read back as a number.
function the_datum(case::Symbol, gr, dr)
    case === :data && return gr[2].data
    case === :robin && return gr[2].op.ops[1][2].left_op.scalar
    return dr[1].data
end

const A = CASE === :const ? nothing : affine_inputs(CASE)

sq(x) = sum(abs2, x)

loss_inputs(p) = sq(nonzeros(at(A.R, p))) + sq(at(A.b, p)) + sq(at(A.g, p))

function loss_parse(p)
    gr, dr = parse_relations(CASE, p[1])
    return the_datum(CASE, gr, dr)^2
end

function loss_relation(p)
    gr, _ = parse_relations(CASE, p[1])
    Rt, bF = Bramble._assemble_relation(P, gr[2])
    return sq(nonzeros(Rt)) + sq(bF)
end

# The two forms of `_assemble_relation` (lines 798 and 799 there) on their own, for the
# :xmax relation (`d = 1`, `s = 2`, mask `_relation_mask(Val(1), Val(2), Val(2))`): the
# data's load, and the operator's matrix.
function loss_relation_load(p)
    gr, _ = parse_relations(CASE, p[1])
    mask = Bramble._relation_mask(Val(1), Val(2), Val(2))
    data = gr[2].data
    b = assemble(form(P, v -> Bramble._facet_inner(mask, data, Bramble._facet_identity(v))))
    return sq(b)
end

function loss_relation_matrix(p)
    gr, _ = parse_relations(CASE, p[1])
    mask = Bramble._relation_mask(Val(1), Val(2), Val(2))
    op = gr[2].op
    R = assemble(form(P, P, (u, v) -> Bramble._facet_inner(mask, op, v)))
    return sq(nonzeros(R))
end

# `sparse(transpose(R))` of `_assemble_relation` (its last line) on a matrix of the
# parameter, rebuilt as `inputs` does.
function loss_relation_transpose(p)
    return sq(nonzeros(sparse(transpose(at(A.R, p)))))
end

function loss_rows(p)
    gr, _ = parse_relations(CASE, p[1])
    R, b, _, _ = Bramble._relation_rows(P, gr)
    return sq(nonzeros(R)) + sq(b)
end

function loss_dirichlet(p)
    _, dr = parse_relations(CASE, p[1])
    _, g, _, _ = Bramble._dirichlet_values(G, dr, T)
    return sq(g)
end

loss_kept(p) = p[1] * sum(Float64, Bramble._kept_points(G, A.s.ghosts, A.s.fixed))

function loss_bookkeeping(p)
    members = block_members(at(A.C, p))
    return p[1] * sum(length, values(members))
end

# The relations here couple no two ghosts, so `_block_solve` never takes its dense branch at
# run time (every block has one ghost). `lu` runs that branch's kernel on a block of two,
# with entries linear in the parameter.
const CB = Affine([3.0 0.5; 0.25 2.0], [0.2 0.1; 0.05 0.3])
const BB = Affine([1.0 2.0 0.5; 0.3 1.0 2.0], [0.1 0.4 0.2; 0.3 0.1 0.5])
const BV = Affine([1.0, 2.0], [0.5, 0.25])

function loss_lu(p)
    Cb, Bb, bb = at(CB, p), at(BB, p), at(BV, p)
    F = lu(Cb; check = false)
    return sq(F \ Bb) + sq(F \ bb)
end

function loss_block_solve(p)
    _, _, vals, dg = Bramble._block_solve(Symbol[:x for _ in 1:size(A.s.C, 1)],
        at(A.C, p), at(A.Bt, p), at(A.bg, p), nothing)
    return sq(vals) + sq(dg)
end

function loss_closure(p)
    E, d, _ = Bramble._relation_closure(G, at(A.R, p), at(A.b, p), A.s.ghosts, A.s.keep,
        A.s.fixed, at(A.g, p))
    return sq(nonzeros(E)) + sq(d)
end

function loss_closure_data(p)
    Ec = Bramble._closure_matrix(BE, T, at(A.E, p))
    dc = Bramble._closure_data(BE, T, at(A.d, p))
    return sq(nonzeros(Ec)) + sq(dc)
end

function loss_space(p)
    gr, dr = parse_relations(CASE, p[1])
    Wc = Bramble._constrained_space(W, gr, dr)
    return sq(Wc.d) + sq(nonzeros(Wc.E))
end

function loss_constrained(p)
    Wc = constrained(W, relations_of(CASE, p[1]))
    return sq(Wc.d) + sq(nonzeros(Wc.E))
end

# Case `const`: the same relations with constant data, `constrained` run outside, and the
# form and assembly of `1127_ghost.jl`'s `loss_assemble` on it, the parameter in the form only.
const C0 = CASE === :const ? constrained(W, relations_of(:const, 0.0)) : nothing
const L0 = CASE === :const ? form(C0, v -> innerₕ(x -> 1.0 + x[1], v)) : nothing
function loss_downstream(p)
    a = form(C0, C0, (U, V) -> p[1] * innerₕ(-Δₕ(U), V) + p[1]^2 * innerₕ(U, V))
    M, F = assemble(a, L0)
    return sq(nonzeros(M)) + 1e4 * p[1] * sum(F)
end

const LOSSES = Dict("inputs" => loss_inputs, "parse" => loss_parse,
    "relation" => loss_relation, "relation_load" => loss_relation_load,
    "relation_matrix" => loss_relation_matrix,
    "relation_transpose" => loss_relation_transpose, "rows" => loss_rows,
    "dirichlet" => loss_dirichlet,
    "kept" => loss_kept, "bookkeeping" => loss_bookkeeping, "lu" => loss_lu,
    "block_solve" => loss_block_solve, "closure" => loss_closure,
    "closure_data" => loss_closure_data, "space" => loss_space,
    "constrained" => loss_constrained, "downstream" => loss_downstream)
haskey(LOSSES, STEP) || error("unknown step ", STEP)
(STEP == "downstream") == (CASE === :const) ||
    error("`downstream` runs the case `const`, alone")

const LOSS = LOSSES[STEP]
const P0 = [0.5 + 0.3 * sin(1)]

function central_difference(f, p; h = 1e-5)
    return [(f(p .+ h .* (1:length(p) .== k)) - f(p .- h .* (1:length(p) .== k))) / 2h
            for k in eachindex(p)]
end

println("step ", STEP, " case ", CASE, " mode ", MODE, ", Enzyme ", pkgversion(Enzyme),
    ", Julia ", VERSION, ", threads ", Threads.nthreads(), ", optimize ",
    Base.JLOptions().opt_level, ", check-bounds ", Base.JLOptions().check_bounds,
    " (0 auto, 1 yes, 2 no)")
flush(stdout)

ref = central_difference(LOSS, P0)
step("reference")
if abs(only(ref)) <= 1e-8
    println("RESULT ", STEP, " ", CASE, " ", MODE, " weakref NaN ",
        round(time() - T0; digits = 1), " reference ", ref)
    exit(1)
end

# Top-level `try` blocks do not hand their assignments back, so the step runs in a function.
function measure(ref, t_ref)
    status, relerr, detail, g = "ok", NaN, "", Float64[]
    step("enzyme start")
    try
        if MODE == "fwd"
            mode = Enzyme.set_runtime_activity(Enzyme.Forward)
            g = [only(Enzyme.autodiff(mode, Enzyme.Const(LOSS), Enzyme.Duplicated,
                Enzyme.Duplicated(copy(P0), [1.0])))]
        else
            g = zeros(1)
            mode = Enzyme.set_runtime_activity(Enzyme.Reverse)
            Enzyme.autodiff(mode, Enzyme.Const(LOSS), Enzyme.Active,
                Enzyme.Duplicated(copy(P0), g))
        end
        relerr = maximum(abs, g .- ref) / maximum(abs, ref)
        relerr <= 1e-6 || (status = "wrong")
        step("enzyme done")
    catch err
        err isa InterruptException && rethrow()
        status = "fails"
        msg = sprint(showerror, err)
        detail = string(nameof(typeof(err)), ": ", first(replace(msg, '\n' => ' '), 400))
        println(first(msg, 1500))
        frames = filter(l -> occursin(r"^ *(\[\d+\]|@ |cannot handle|No create)", l),
            split(msg, '\n'))
        println(join(first(frames, 40), '\n'))
        step("enzyme failed")
    end
    return (; status, relerr, detail, g, secs = time() - T0 - t_ref)
end

# `secs` counts from the reference to the end of the Enzyme call, compile included, the same
# span as the note's `Compile:` line.
r = measure(ref, time() - T0)
println("RESULT ", STEP, " ", CASE, " ", MODE, " ", r.status, " ",
    isnan(r.relerr) ? "NaN" : string(r.relerr), " ", round(r.secs; digits = 1), " ", r.detail)
println("enzyme gradient ", isempty(r.g) ? "none" : string(r.g), " reference ", ref)
flush(stdout)
exit(r.status == "ok" ? 0 : 1)
