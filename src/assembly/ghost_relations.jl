# ghost_relations.jl: `constrained`, the ghost space whose ghosts are tied to boundary
# relations written `operator => data`, one per facet.
#
# `constrained(Wₕ, u -> (:xmin => α * u + β * ∂ₙ(u) => g, …))` calls the lambda once with
# the trial placeholder of the padded space. Each relation's operator is a facet operator
# (`∂ₙ`, `∂ₙₙ`, their sums with `u`), and its rows on the nodes of its facet are assembled
# by forms on the padded space, `inner_Γ(rel, v; markers = (F,))` and
# `inner_Γ(g, v; markers = (F,))`, so each row carries that facet's weight on both sides
# and the weight cancels when the row is solved for the ghost. A ghost's row reads its own
# ghost, physical nodes and, at a corner of the conormal `∂ₙ(u; K)`, the ghost of the other
# facet there: the eliminated ghosts couple in blocks (one entry per ghost, a 2×2 block at a
# 2D corner), each solved on its own, so the whole closure stays sparse (design D7).
#
# The closure `(E, d)` of the result keeps the DOF order of the ghost space: the physical
# nodes, then the ghosts of the facets that carry no relation, which stay unknowns.
#
# See also: `GhostGridSpace`, `FacetOperator`, `GhostBilinearForm`

"""
    constrained(Wₕ, relations) -> GhostGridSpace

The space `Wₕ`, built by `gridspace(Ωₕ; ghosts = ...)`, with the ghosts of some facets
eliminated through boundary relations. `relations` is a function of the unknown `u` that
returns one pair `facet => (operator => data)` per related facet, or a tuple of them:

```julia
Cₕ = constrained(Wₕ, u -> (:xmin => ∂ₙ(u) => 0.0,
    :xmax => 2.0 * u + 0.5 * ∂ₙ(u) => x -> exp(x[1])))
```

The operator is a facet operator that reads the facet's ghost: [`∂ₙ`](@ref) (conormal
`∂ₙ(u; K)` included), [`∂ₙₙ`](@ref), and their linear combinations with `u`. The data is a
number or a function of the point. At each node of the facet, the relation is solved for
the node's ghost; at a 2D corner where two conormal relations read each other's ghost, the
two are solved together. A ghosted facet without a relation keeps its ghosts as unknowns.

The degrees of freedom of the result are the physical nodes, in the mesh's linear order,
then the ghosts that remain unknowns, in the order of `Wₕ`. Its closure `(E, d)` maps them
to the padded values `E u + d`, where `d` carries the data; so a load needs the bilinear
form, `assemble(a, l)`.

Refused with an `ArgumentError`: a space without ghosts, an already constrained space, a
facet named twice or that carries no ghosts, a relation that does not read its facet's
ghost (`β = 0`, or a `jumpₙ` alone), and a singular block at a corner (a singular `K`).

# Examples

```julia
Ωₕ = mesh(domain(interval(0.0, 1.0)), 11, false)
Wₕ = gridspace(Ωₕ; ghosts = :boundary)
Cₕ = constrained(Wₕ, u -> (:xmin => ∂ₙ(u) => 1.0, :xmax => u + ∂ₙ(u) => 2.0))
a = form(Cₕ, Cₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
l = form(Cₕ, v -> innerₕ(x -> 1.0, v))
A, F = assemble(a, l)
```

See also: [`gridspace`](@ref), [`∂ₙ`](@ref), [`∂ₙₙ`](@ref)
"""
function constrained(Wₕ::GhostGridSpace, relations)
    _check_unconstrained(Wₕ)
    P = padded_space(Wₕ)
    rels = _facet_relations(mesh(P), relations(trial_function(P)))
    return _constrained_space(Wₕ, rels)
end

@noinline function constrained(Wₕ, relations)
    throw(
        ArgumentError(
        "constrained takes a space with ghosts, built by gridspace(Ωₕ; ghosts = ...); " *
        "got $(summary(Wₕ)). Build the space with the facets the relations need ghosted.",
    ),
    )
end

# The relations close the ghosts of the selection closure of `gridspace(Ωₕ; ghosts)`, which
# has one degree of freedom per physical node and per ghost and no affine part.
function _check_unconstrained(Wₕ::GhostGridSpace)
    G = mesh(padded_space(Wₕ))
    ndofs(Wₕ) == npoints(physical_mesh(G)) + nghosts(G) && iszero(Wₕ.d) && return nothing
    throw(
        ArgumentError(
        "constrained takes the space of gridspace(Ωₕ; ghosts = ...), and $(Wₕ) is already " *
        "constrained. Give every relation in one call on the unconstrained space.",
    ),
    )
end

# --- Parsing the relations ----------------------------------------------------------- #

# One parsed relation: the facet `(d, s)` (`s` 1 the minimum, 2 the maximum), its
# canonical name, the facet operator and the data.
struct _FacetRelation{O, T}
    d::Int
    s::Int
    name::Symbol
    op::O
    data::T
end

_relation_pairs(p::Pair) = (p,)
_relation_pairs(t::Tuple) = t
_relation_pairs(x) = _throw_relation_shape(x)

function _facet_relations(G::GhostMesh{D}, out) where {D}
    pairs = _relation_pairs(out)
    isempty(pairs) && _throw_relation_shape(out)
    rels = Any[]
    for p in pairs
        r = _facet_relation(G, p)
        any(q -> q.d == r.d && q.s == r.s, rels) && _throw_facet_twice(r.name)
        push!(rels, r)
    end
    return rels
end

function _facet_relation(G::GhostMesh{D}, p::Pair{Symbol, <:Pair}) where {D}
    face = _find_face(Val(D), first(p))
    face === nothing && _throw_not_a_relation_facet(first(p), D)
    d, s = face
    name = _facet_name(d, s)
    has_ghosts(G, d, _GHOST_SIDES[s]) || _throw_relation_unghosted(name)
    op, data = last(p)
    _check_relation_operator(op, name)
    _check_relation_data(data, name)
    return _FacetRelation(d, s, name, op, data)
end
_facet_relation(G::GhostMesh, p) = _throw_relation_shape(p)

_check_relation_operator(::FacetOperator, name::Symbol) = nothing
@noinline function _check_relation_operator(op, name::Symbol)
    throw(
        ArgumentError(
        "the relation on :$name has the operator $(summary(op)), which reads no ghost, so " *
        "it cannot be solved for the ghosts of :$name. Write a facet operator that reads " *
        "the facet's ghost, such as ∂ₙ(u), α * u + β * ∂ₙ(u) with β ≠ 0, or ∂ₙₙ(u).",
    ),
    )
end

_check_relation_data(::Union{Number, Function}, ::Symbol) = nothing
@noinline function _check_relation_data(data, name::Symbol)
    throw(
        ArgumentError(
        "the data of the relation on :$name is $(summary(data)); give a number or a " *
        "function of the point, as in `:$name => ∂ₙ(u) => x -> x[1]`.",
    ),
    )
end

@noinline function _throw_relation_shape(x)
    throw(
        ArgumentError(
        "constrained(Wₕ, relations) expects `relations(u)` to return pairs " *
        "`facet => operator => data`, or a nonempty tuple of them (at least one " *
        "relation), such as " *
        "`u -> (:xmin => ∂ₙ(u) => 0.0, :xmax => ∂ₙ(u) => 1.0)`; got $(summary(x)).",
    ),
    )
end

@noinline function _throw_not_a_relation_facet(s, D::Int)
    faces = join((":$c" for (c, _, _, _) in _boundary_faces(Val(D))), ", ")
    throw(
        ArgumentError(
        "a relation is keyed by a facet name, and $(repr(s)) does not name a facet in " *
        "$(D)D. Use one of $faces (or an alias).",
    ),
    )
end

@noinline function _throw_facet_twice(name::Symbol)
    throw(
        ArgumentError(
        "the facet :$name has two relations; give one per facet, summing the operators " *
        "into one relation if both hold.",
    ),
    )
end

@noinline function _throw_relation_unghosted(name::Symbol)
    throw(
        ArgumentError(
        "the facet :$name carries no ghosts on this space, so a relation there has no " *
        "ghost to solve for. Name :$name in `gridspace(Ωₕ; ghosts = ...)` (or use " *
        "`ghosts = :boundary`).",
    ),
    )
end

# --- The relation rows --------------------------------------------------------------- #

# The facet operator equal to `v` on every facet: the test side of the data's row.
@inline function _facet_identity(v::LazyOp{D}) where {D}
    return _facet_operator(Val(ntuple(_ -> ntuple(_ -> false, Val(D)), Val(D))),
        ntuple(_ -> (v, v), Val(D)))
end

# The rows of the relations of `rels` on the padded grid, one per ghost they eliminate,
# in ghost order: `R[k, :]` is the facet weight times the relation of ghost `ghosts[k]`'s
# facet at its node, and `b[k]` the weight times the data there.
function _relation_rows(P::ScalarGridSpace, rels)
    G = mesh(P)
    plin = LinearIndices(npoints(G, Tuple))
    facet = zeros(Int, nghosts(G))            # the relation of each ghost, 0 if none
    for g in 1:nghosts(G)
        _, d, side = boundary_node(G, g)
        facet[g] = something(findfirst(r -> r.d == d && _GHOST_SIDES[r.s] === side, rels), 0)
    end
    ghosts = findall(!iszero, facet)
    blocks = map(r -> _assemble_relation(P, r), rels)
    rows, cols, vals = Int[], Int[], promote_type(map(x -> eltype(first(x)), blocks)...)[]
    b = zeros(promote_type(eltype(vals), map(x -> eltype(last(x)), blocks)...), length(ghosts))
    for (k, g) in enumerate(ghosts)
        Rt, bF = blocks[facet[g]]
        i = plin[padded_index(G, first(boundary_node(G, g)))]
        for q in nzrange(Rt, i)
            push!(rows, k)
            push!(cols, rowvals(Rt)[q])
            push!(vals, nonzeros(Rt)[q])
        end
        b[k] = bF[i]
    end
    return sparse(rows, cols, vals, length(ghosts), length(plin)), b, ghosts
end

# The relation's rows and data on the nodes of its facet, by forms on the padded space;
# the rows transposed, so each node's row is a column.
function _assemble_relation(P::ScalarGridSpace, r::_FacetRelation)
    op, data, name = r.op, r.data, r.name
    R = assemble(form(P, P, (u, v) -> inner_Γ(op, v; markers = (name,))))
    b = assemble(form(P, v -> inner_Γ(data, _facet_identity(v); markers = (name,))))
    return sparse(transpose(R)), b
end

# --- The closure --------------------------------------------------------------------- #

function _constrained_space(Wₕ::GhostGridSpace, rels)
    P = padded_space(Wₕ)
    G = mesh(P)
    R, b, ghosts = _relation_rows(P, rels)
    keep = _kept_points(G, ghosts)
    E, d = _relation_closure(G, R, b, ghosts, keep, rels)::Tuple{SparseMatrixCSC, Vector}
    T = eltype(P)
    return GhostGridSpace(P, SparseMatrixCSC{T, Int}(E), Vector{T}(d))
end

# The padded points of the degrees of freedom that remain: the physical nodes, then the
# ghosts no relation eliminates, in the order of the ghost space.
function _kept_points(G::GhostMesh{D}, ghosts::Vector{Int}) where {D}
    plin = LinearIndices(npoints(G, Tuple))
    keep = vec([plin[J] for J in physical_indices(G)])
    eliminated = falses(nghosts(G))
    eliminated[ghosts] .= true
    for g in 1:nghosts(G)
        eliminated[g] || push!(keep, plin[_ghost_point(G, g)])
    end
    return keep
end

@inline function _ghost_point(G::GhostMesh{D}, g::Int) where {D}
    I, d, side = boundary_node(G, g)
    step = side === :min ? -1 : 1
    return padded_index(G, I) + CartesianIndex(ntuple(e -> e == d ? step : 0, Val(D)))
end

# `u_pad = E u + d`: a kept point copies its degree of freedom; the eliminated ghosts solve
# `C u_g = b − B u`, `C` their columns of the rows and `B` the kept ones. `C` couples only
# the ghosts of one node, so it is solved block by block into a sparse inverse.
function _relation_closure(G::GhostMesh, R::SparseMatrixCSC, b, ghosts, keep, rels)
    Npad = size(R, 2)
    m, n = length(ghosts), length(keep)
    gpos = zeros(Int, Npad)
    kpos = zeros(Int, Npad)
    gpts = [LinearIndices(npoints(G, Tuple))[_ghost_point(G, g)] for g in ghosts]
    gpos[gpts] .= 1:m
    kpos[keep] .= 1:n
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
        else
            _throw_relation_reads_absent(_relation_name(G, ghosts[k]))
        end
    end
    C = sparse(Ci, Cj, Cv, m, m)
    for k in 1:m
        iszero(C[k, k]) && _throw_relation_misses_ghost(_relation_name(G, ghosts[k]))
    end
    Cinv = _block_inverse(G, C, ghosts)
    Eg = Cinv * sparse(Bi, Bj, Bv, m, n)
    dg = Cinv * b
    rows, cols, vals = collect(keep), collect(1:n), ones(eltype(Eg), n)
    for j in 1:n, q in nzrange(Eg, j)
        push!(rows, gpts[rowvals(Eg)[q]]), push!(cols, j), push!(vals, nonzeros(Eg)[q])
    end
    d = zeros(eltype(dg), Npad)
    d[gpts] .= dg
    return sparse(rows, cols, vals, Npad, n), d
end

@inline function _relation_name(G::GhostMesh, g::Int)
    _, d, side = boundary_node(G, g)
    return _facet_name(d, _ghost_side(side))
end

# The inverse of `C`, block by block: the blocks are the connected sets of ghosts the rows
# couple, at most one per node of the mesh, so each is a small dense solve.
function _block_inverse(G::GhostMesh, C::SparseMatrixCSC, ghosts)
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
    rows, cols, vals = Int[], Int[], eltype(C)[]
    for idx in values(members)
        if length(idx) == 1
            k = only(idx)
            push!(rows, k), push!(cols, k), push!(vals, inv(C[k, k]))
            continue
        end
        Cb = Matrix(C[idx, idx])
        F = lu(Cb; check = false)
        _nonsingular(F, Cb) ||
            _throw_singular_corner(map(k -> _relation_name(G, ghosts[k]), idx))
        Ib = inv(F)
        for (a, ka) in enumerate(idx), (c, kc) in enumerate(idx)
            push!(rows, ka), push!(cols, kc), push!(vals, Ib[a, c])
        end
    end
    return sparse(rows, cols, vals, m, m)
end

# A block whose smallest pivot is at the level of rounding is singular: a singular `K`
# leaves one of that size rather than an exact zero. The pivots are judged on the block
# with each row, then each column, scaled to a largest entry of one, since the facet
# weights and spacings scale rows and columns by the mesh's aspect alone.
function _nonsingular(F, Cb::AbstractMatrix)
    F.info == 0 || return false
    S = Cb ./ maximum(abs, Cb; dims = 2)
    S = S ./ maximum(abs, S; dims = 1)
    Fs = lu(S; check = false)
    Fs.info == 0 || return false
    tol = 1000 * size(S, 1) * eps(real(eltype(S)))
    return all(i -> abs(Fs.U[i, i]) > tol, axes(S, 1))
end

@noinline function _throw_relation_misses_ghost(name::Symbol)
    throw(
        ArgumentError(
        "the relation on :$name does not read the ghost of :$name (a zero coefficient on " *
        "the normal derivative, or an operator such as jumpₙ that stays inside), so it " *
        "cannot be solved for it. Give the relation a nonzero normal-derivative term, " *
        "such as α * u + β * ∂ₙ(u) with β ≠ 0.",
    ),
    )
end

@noinline function _throw_singular_corner(names)
    facets = join((":$s" for s in unique(names)), " and ")
    throw(
        ArgumentError(
        "the relations on $facets cannot be solved together for the ghosts of a corner " *
        "node: their block is singular, as it is for a singular conormal tensor K in " *
        "∂ₙ(u; K). Use a nonsingular K, or drop the tangential terms of one relation.",
    ),
    )
end

@noinline function _throw_relation_reads_absent(name::Symbol)
    throw(
        ArgumentError(
        "the relation on :$name reads a padded corner point, which is no node and no " *
        "ghost. Write the relation with operators that read only the facet's own ghosts " *
        "and the physical nodes.",
    ),
    )
end
