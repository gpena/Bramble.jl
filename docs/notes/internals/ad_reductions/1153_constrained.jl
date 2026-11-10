# Reproducer for gpena/Bramble.jl#1153: Enzyme through `constrained(W, relations)` when the
# data of a relation depend on the parameter. No form of the user and no assembly: the
# differentiated function builds the constrained space and reads its closure. One step of
# the construction, one parameter-dependent case and one mode per process.
#
# Run from the repository root, with one thread:
#
#     julia --startup-file=no --project=. --threads=1 --optimize=1 --check-bounds=auto \
#         docs/notes/internals/ad_reductions/1153_constrained.jl <step> <case> <fwd|rev> \
#         [rule|split]
#
# The fourth argument prototypes a fix without changing `src/`: `rule` is a reverse rule
# over `_assemble_relation`, `split` a replacement of that method that splits the operator
# (both below, after the inputs).
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
#   relation_forms, relation_forms_check, relation_forms_transpose, relation_at_facet
#                  the bisection of `_assemble_relation`: its two forms in one function on
#                  a constant facet, the same with the reach check or the transposition
#                  added, and the two forms through `_at_facet` on the runtime facet.
#   split_load, split_matrix, split_matrix_r
#                  the linearity split on the facet :xmax: the load of the unit datum times
#                  the datum, and the matrices of `u` and `∂ₙ(u)` combined by the Robin
#                  coefficient, the two operators rebuilt from the space or read out of the
#                  relation.
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
#   cell           the loss of the `ghost_data` and `ghost_robin` cells of
#                  test/space/ad_coverage.jl: `constrained`, then a form and its assembly.
#   downstream     case `const` only: a form and its assembly on the constrained space
#                  built outside, whose relations carry no parameter.
#
# The reference is a central difference of the same loss (the losses are rational in the
# parameter and h = 1e-5 leaves about 1e-10). A step is `ok` when the relative error, the
# largest absolute difference over the largest absolute reference, is at most 1e-6. The step
# `cell` compares with ForwardDiff instead, to 1e-8.
#
# Enzyme and ForwardDiff are not dependencies of the root project, so they are loaded from
# a temporary environment, as 1127_ghost.jl does. The root Manifest.toml must hold both.
#
# Output: `step <name> t=<seconds>` lines as it runs, then one line
# `RESULT <step> <case> <mode> <status> <relerr> <seconds> <detail>`. Exit status is 0 for
# `ok`, 1 for `fails` or `wrong`.
println("pid ", getpid())
flush(stdout)
_e = mktempdir()
write(joinpath(_e, "Project.toml"),
    "[deps]\nEnzyme = \"7da242da-08ed-463a-9acd-ee780be4f1d9\"\n" *
    "ForwardDiff = \"f6369f11-7733-5829-9624-2563aa707210\"\n")
push!(LOAD_PATH, _e)
using Bramble, Enzyme, ForwardDiff, LinearAlgebra, SparseArrays, Logging

# Enzyme warns once per stored value it cannot find a use for, thousands of lines of IR.
global_logger(NullLogger())

const T0 = time()
step(name) = (println("step ", name, " t=", round(time() - T0; digits = 1)); flush(stdout))

length(ARGS) in (3, 4) ||
    error("usage: 1153_constrained.jl <step> <case> <fwd|rev> [rule|split]")
const STEP, CASE_NAME, MODE = ARGS[1:3]
# A fourth argument prototypes a fix with no change to `src/`, before anything is
# differentiated: `rule` defines a reverse rule over `_assemble_relation`, `split` replaces
# the method by one that assembles forms of operands without the parameter and scales them.
const FIX = length(ARGS) == 4 ? ARGS[4] : ""
FIX in ("", "rule", "split") || error("the fourth argument is `rule` or `split`")
const RULE = FIX == "rule"
const SPLIT = FIX == "split"
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

# The rule prototyped for `_assemble_relation`, no change to `src/`: reverse mode does not
# differentiate the forms. The primal is the function's own call; the pullback takes one
# forward derivative of the same function per float of the relation (its datum, its Robin
# coefficient) and contracts it with the cotangents of the matrix and the load. Forward
# mode compiles the function in well under a minute, where reverse takes 732 s.
has_float(::Type{<:AbstractFloat}) = true
function has_float(::Type{T}) where {T}
    return isconcretetype(T) && !ismutabletype(T) && any(has_float, fieldtypes(T))
end
rebuilt(x::T, fields) where {T} = ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), T,
    fields, fieldcount(T))::T
# `x` with its `k[]`-th float replaced by one and every other by zero; `k` counts down.
unit_leaf(x::AbstractFloat, k) = (k[] -= 1; k[] == 0 ? one(x) : zero(x))
function unit_leaf(x::T, k) where {T}
    (fieldcount(T) == 0 || !has_float(T)) && return x
    return rebuilt(x, Any[unit_leaf(getfield(x, i), k) for i in 1:fieldcount(T)])
end
# `x` with its floats replaced by those of `vals`, in order.
fill_leaves(x::AbstractFloat, vals, k) = oftype(x, vals[k[] += 1])
function fill_leaves(x::T, vals, k) where {T}
    (fieldcount(T) == 0 || !has_float(T)) && return x
    return rebuilt(x, Any[fill_leaves(getfield(x, i), vals, k) for i in 1:fieldcount(T)])
end
count_leaves(x::AbstractFloat) = 1
function count_leaves(x::T) where {T}
    (fieldcount(T) == 0 || !has_float(T)) && return 0
    return sum(i -> count_leaves(getfield(x, i)), 1:fieldcount(T))
end

function relation_pullback(f, P, r, dR, db)
    mode = Enzyme.set_runtime_activity(Enzyme.Forward)
    vals = zeros(count_leaves(r))
    for l in eachindex(vals)
        Ṙ, ḃ = only(Enzyme.autodiff(mode, Enzyme.Const(f), Enzyme.Duplicated,
            Enzyme.Const(P), Enzyme.Duplicated(r, unit_leaf(r, Ref(l)))))
        vals[l] = dot(nonzeros(Ṙ), nonzeros(dR)) + dot(ḃ, db)
    end
    return fill_leaves(r, vals, Ref(0))
end

# The linearity split prototyped as a replacement of `_assemble_relation` on a scalar space:
# the component of the operator on the facet splits into its terms (`_relation_terms`), each
# term gives up its scalar factors (`_peel`), the matrix of each bare term is assembled on
# its own, and the relation's matrix is their combination by the factors. The load is the
# load of the unit datum times the datum (a number here). The space's closure is compared
# with the unsplit method's first.
if SPLIT
    const UNSPLIT = constrained(W, relations_of(CASE, 0.5 + 0.3 * sin(1)))
    @eval Bramble begin
        @inline function _peel(t::OperatorScale{D, <:Number}) where {D}
            c, u = _peel(t.inner_op)
            return t.scalar * c, u
        end
        @inline _peel(t) = (true, t)
        @inline function _unit_matrix(P, mask, op, ::Val{i}) where {i}
            F = _map_facets(c -> last(_peel(_relation_terms(c)[i])), op)
            return assemble(form(P, P, (u, v) -> _facet_inner(mask, F, v)))
        end
        @inline _combine(cs::Tuple{Any}, Ms::Tuple{Any}) = first(cs) * first(Ms)
        @inline function _combine(cs::Tuple, Ms::Tuple)
            return first(cs) * first(Ms) + _combine(Base.tail(cs), Base.tail(Ms))
        end
        function _assemble_relation(P::ScalarGridSpace{D}, r::_FacetRelation) where {D}
            op, data = r.op, r.data
            return _at_facet(Val(D), r.d, r.s) do name, d, s
                _check_facet_reach(mesh(P), _reads_ghosts(op)[_val(d)], _val(d), _val(s),
                    (_facet_component(op, d, s),))
                mask = _relation_mask(d, s, Val(D))
                terms = _relation_terms(_facet_component(op, d, s))
                Ms = map(i -> _unit_matrix(P, mask, op, i), ntuple(Val, Val(length(terms))))
                R = _combine(map(t -> first(_peel(t)), terms), Ms)
                w = assemble(form(P,
                    v -> _facet_inner(mask, one(eltype(P)), _facet_identity(v))))
                return sparse(transpose(R)), w .* data
            end
        end
    end
    let C = constrained(W, relations_of(CASE, 0.5 + 0.3 * sin(1)))
        println("split closure: E pattern equal ",
            rowvals(C.E) == rowvals(UNSPLIT.E) &&
            SparseArrays.getcolptr(C.E) == SparseArrays.getcolptr(UNSPLIT.E),
            ", E bits equal ", nonzeros(C.E) == nonzeros(UNSPLIT.E), ", max diff ",
            maximum(abs, C.E - UNSPLIT.E), "; d bits equal ", C.d == UNSPLIT.d,
            ", max diff ", maximum(abs, C.d - UNSPLIT.d))
        flush(stdout)
    end
end

if RULE
    function Enzyme.EnzymeRules.augmented_primal(config::Enzyme.EnzymeRules.RevConfig,
            func::Enzyme.Const{typeof(Bramble._assemble_relation)}, ::Type{RT},
            P::Enzyme.Annotation, r::Enzyme.Active) where {RT}
        P isa Enzyme.Const || P.dval === P.val || error("the space is active: ", typeof(P))
        Enzyme.EnzymeRules.width(config) == 1 || error("width 1 only")
        R, b = func.val(P.val, r.val)
        shadow = (Enzyme.make_zero(R), Enzyme.make_zero(b))
        return Enzyme.EnzymeRules.AugmentedReturn(
            Enzyme.EnzymeRules.needs_primal(config) ? (R, b) : nothing,
            Enzyme.EnzymeRules.needs_shadow(config) ? shadow : nothing, shadow)
    end
    function Enzyme.EnzymeRules.reverse(config::Enzyme.EnzymeRules.RevConfig,
            func::Enzyme.Const{typeof(Bramble._assemble_relation)}, ::Type{RT}, tape,
            P::Enzyme.Annotation, r::Enzyme.Active) where {RT}
        dR, db = tape
        r̄ = relation_pullback(func.val, P.val, r.val, dR, db)
        fill!(nonzeros(dR), 0)
        fill!(db, 0)
        return (nothing, r̄)
    end
end

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

# The bisection of `_assemble_relation`'s reverse cost, 732 s against about 200 s for its
# parts alone. `relation_forms` runs the two forms in one function on the constant facet
# :xmax, without `_check_facet_reach` and without the transposition; `_check` and
# `_transpose` add one of those back; `relation_at_facet` runs the two forms through
# `_at_facet` on the relation's runtime `(d, s)`, as `_assemble_relation` does, so the
# compiled code holds one copy of the forms per facet of the mesh (four in 2D).
@inline function relation_forms(op, data, d::Val, s::Val)
    mask = Bramble._relation_mask(d, s, Val(2))
    R = assemble(form(P, P, (u, v) -> Bramble._facet_inner(mask, op, v)))
    b = assemble(form(P, v -> Bramble._facet_inner(mask, data, Bramble._facet_identity(v))))
    return R, b
end

function loss_relation_forms(p)
    gr, _ = parse_relations(CASE, p[1])
    R, b = relation_forms(gr[2].op, gr[2].data, Val(1), Val(2))
    return sq(nonzeros(R)) + sq(b)
end

function loss_relation_forms_check(p)
    gr, _ = parse_relations(CASE, p[1])
    op = gr[2].op
    Bramble._check_facet_reach(G, Bramble._reads_ghosts(op)[1], 1, 2,
        (Bramble._facet_component(op, Val(1), Val(2)),))
    R, b = relation_forms(op, gr[2].data, Val(1), Val(2))
    return sq(nonzeros(R)) + sq(b)
end

function loss_relation_forms_transpose(p)
    gr, _ = parse_relations(CASE, p[1])
    R, b = relation_forms(gr[2].op, gr[2].data, Val(1), Val(2))
    return sq(nonzeros(sparse(transpose(R)))) + sq(b)
end

function loss_relation_at_facet(p)
    gr, _ = parse_relations(CASE, p[1])
    r = gr[2]
    op, data = r.op, r.data
    R, b = Bramble._at_facet(Val(2), r.d, r.s) do name, d, s
        return relation_forms(op, data, d, s)
    end
    return sq(nonzeros(R)) + sq(b)
end

# The linearity split, prototyped here with no change to `src/`: the forms are assembled on
# operands that hold no parameter, and the parameter scales what they give.
# `split_load` (data): the load of the unit datum, the facet weights, times the datum.
# `split_matrix` (robin): the matrices of `u` and of `∂ₙ(u)`, operators rebuilt from the
# space alone, combined as `β * Mu + Mn`. `split_matrix_r` is the same with the two
# operators read out of the parsed relation, which also holds the parameter.
# Each prints, before Enzyme runs, whether the split gives the bits of the unsplit form.
function unit_matrices(Fu, Fn)
    mask = Bramble._relation_mask(Val(1), Val(2), Val(2))
    Mu = assemble(form(P, P, (u, v) -> Bramble._facet_inner(mask, Fu, v)))
    Mn = assemble(form(P, P, (u, v) -> Bramble._facet_inner(mask, Fn, v)))
    return Mu, Mn
end

function loss_split_load(p)
    gr, _ = parse_relations(CASE, p[1])
    mask = Bramble._relation_mask(Val(1), Val(2), Val(2))
    w = assemble(form(P, v -> Bramble._facet_inner(mask, one(T), Bramble._facet_identity(v))))
    return sq(w .* gr[2].data)
end

function split_matrix(p)
    gr, _ = parse_relations(CASE, p[1])
    Fn = ∂ₙ(Bramble.trial_function(P))
    u0 = Bramble.trial_function(P)
    Mu, Mn = unit_matrices(Bramble._map_facets(c -> u0, Fn), Fn)
    return gr[2].op.ops[1][2].left_op.scalar * Mu + Mn
end
loss_split_matrix(p) = sq(nonzeros(split_matrix(p)))

function split_matrix_r(p)
    gr, _ = parse_relations(CASE, p[1])
    op = gr[2].op
    Mu, Mn = unit_matrices(Bramble._map_facets(c -> c.left_op.inner_op, op),
        Bramble._map_facets(c -> c.right_op, op))
    return op.ops[1][2].left_op.scalar * Mu + Mn
end
loss_split_matrix_r(p) = sq(nonzeros(split_matrix_r(p)))

# Whether the split gives the unsplit form's bits and pattern, printed for the split steps.
function split_primal_check()
    gr, _ = parse_relations(CASE, P0[1])
    mask = Bramble._relation_mask(Val(1), Val(2), Val(2))
    op, data = gr[2].op, gr[2].data
    R = assemble(form(P, P, (u, v) -> Bramble._facet_inner(mask, op, v)))
    b = assemble(form(P, v -> Bramble._facet_inner(mask, data, Bramble._facet_identity(v))))
    w = assemble(form(P, v -> Bramble._facet_inner(mask, one(T), Bramble._facet_identity(v))))
    println("split load: bits equal ", (w .* data) == b, ", max diff ",
        maximum(abs, w .* data .- b))
    for (name, S) in (("split_matrix", split_matrix(P0)), ("split_matrix_r", split_matrix_r(P0)))
        println(name, ": pattern equal ", rowvals(S) == rowvals(R) &&
                                          SparseArrays.getcolptr(S) == SparseArrays.getcolptr(R),
            ", nnz ", nnz(S), " against ", nnz(R), ", bits equal ", S == R, ", max diff ",
            maximum(abs, S - R))
    end
    return flush(stdout)
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

# The loss of the `ghost_data` and `ghost_robin` cells of test/space/ad_coverage.jl
# (`build_ghost_param`): `constrained` on the parameter, then a form and its assembly on the
# space it gives.
function loss_cell(p)
    C = constrained(W, relations_of(CASE, p[1]))
    a = form(C, C, (U, V) -> innerₕ(-Δₕ(U), V) + innerₕ(U, V))
    M, F = assemble(a, form(C, v -> innerₕ(x -> 1.0 + x[1], v)))
    return sq(nonzeros(M)) + sq(F)
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
    "relation_transpose" => loss_relation_transpose,
    "relation_forms" => loss_relation_forms,
    "relation_forms_check" => loss_relation_forms_check,
    "relation_forms_transpose" => loss_relation_forms_transpose,
    "relation_at_facet" => loss_relation_at_facet, "split_load" => loss_split_load,
    "split_matrix" => loss_split_matrix, "split_matrix_r" => loss_split_matrix_r,
    "rows" => loss_rows,
    "dirichlet" => loss_dirichlet,
    "kept" => loss_kept, "bookkeeping" => loss_bookkeeping, "lu" => loss_lu,
    "block_solve" => loss_block_solve, "closure" => loss_closure,
    "closure_data" => loss_closure_data, "space" => loss_space,
    "constrained" => loss_constrained, "cell" => loss_cell,
    "downstream" => loss_downstream)
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
startswith(STEP, "split") && split_primal_check()

# The step `cell` compares with ForwardDiff to 1e-8, as the cells of
# test/space/ad_coverage.jl do: its loss holds the squares of a matrix the parameter does
# not reach, whose cancellation leaves a central difference about 1e-6.
const TOL = STEP == "cell" ? 1e-8 : 1e-6
ref = STEP == "cell" ? ForwardDiff.gradient(LOSS, P0) : central_difference(LOSS, P0)
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
        relerr <= TOL || (status = "wrong")
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
