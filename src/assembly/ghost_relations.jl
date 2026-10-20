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
# 2D corner), each solved on its own, so the whole closure stays sparse.
#
# Data that depend on time, `(x, t) -> …`, leave `E` as it is and make `d` depend on time.
# `d` is linear in the data values at the related facet nodes and the Dirichlet nodes, so
# the same block solves give, once, the sparse `L` with `d(t) = L g(t)`, which the space
# stores with the points where `g` is read (`_TimeData`, in ghost_gridspace.jl).
#
# On a composite of ghost spaces, `constrained(C₁ × C₂, relations)` (ghost_composite.jl),
# a relation may read several components, as a traction `Σⱼ ∂ₙ(u(j); K = Kᵢⱼ)` does. One
# `inner_Γ` holds one trial component, so its rows are the sum of one form per component it
# reads (`_relation_pieces`). A facet's ghost relations are not keyed by component: each
# is attached to one component's ghosts (a facet with a relation reading several has one
# per component), and the rows of a node are solved together in its block, which must be
# nonsingular as a whole. At a node shared by two facets that each have a relation reading
# several components (a corner of two tractions), the rows of each facet read the other
# facet's ghosts through the one-sided difference along the facet (`_one_sided_corner!`),
# so each facet's corner rows read its own ghosts alone and the corner block splits in two:
# a joint block would be singular for isotropic tractions, whose shear relations are one
# equation there. A form on the result reads the ghosts of such a facet through its mixed
# differences by extrapolation from the nodes inward instead (the facet rule of
# stencil_eval.jl, `_multi_component_facets` in ghost_composite.jl), since read through the
# relation's tangential difference they made the reduced matrix unstable on cells stretched
# across the facet. The closure itself is the same.
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

The operator is either `u` itself, a Dirichlet relation, or a facet operator that reads the
facet's ghost: [`∂ₙ`](@ref) (conormal `∂ₙ(u; K)` included), [`∂ₙₙ`](@ref), and their linear
combinations with `u`. The data is a number, a function of the point `x -> …`, or a
function of the point and the time `(x, t) -> …` (told apart by the methods it has). At
each node of the facet, a ghost relation is solved for the node's ghost; at a 2D corner
where two conormal relations read each other's ghost, the two are solved together. A
Dirichlet relation `:F => u => g` removes the nodes of `F` from the unknowns and fixes them
to `g`; its facet needs no ghosts, and if it has them it needs one ghost relation besides,
which solves for them. A ghosted facet without any relation keeps its ghosts as unknowns.

The degrees of freedom of the result are the physical nodes no Dirichlet relation fixes, in
the mesh's linear order, then the ghosts that remain unknowns, in the order of `Wₕ`. Its
closure `(E, d)` maps them to the padded values `E u + d`, where `d` carries the data. The
load of a pair whose trial space has `d ≠ 0` reads the bilinear form, so `assemble(a, l)`
builds the two together, its matrix `Eᵀ A E` symmetric when `A` is; `assemble(l)` alone
refuses a test space with `d ≠ 0`, which is exact when both sides are one space. Dirichlet
conditions can instead replace rows, `assemble(a, l; dirichlet, symmetrize)`, on a space
whose ghosts are all eliminated and none of whose nodes are fixed by `u => g`: its
unknowns are then the physical nodes, as on `gridspace(Ωₕ)`.

When some data depend on time, `E` stays the same and only `d` depends on time, `d(t)`.
[`lift`](@ref)`(Cₕ, u, t)` gives `E u + d(t)`; `lift(Cₕ, u)`, `lift(Cₕ)` and
`assemble(a, l)` refuse with an `ArgumentError`, as there is no single `d`, while `form`
and `assemble(a)`, which read `E` only, work as on a static space. A time-dependent problem
on such a space is posed through `semidiscretize`. `constrained` evaluates the data at
`t = 0`, so they must be defined there, and `closure(Cₕ)` gives `d(0)` as its `d`.
On a composite of ghost spaces, `constrained(C₁ × C₂, relations)` relates the components
together, a relation reading several of them (a traction) included: see
`constrained(::CompositeGridSpace, relations)`.

Dirichlet data in time that meet at a corner are checked to agree at `t = 0` only; the
corner then takes the first facet's datum at every `t`. Under `CpuPolyester` a datum that
closes over an array crosses each task whole and allocates on every evaluation; one that
closes over numbers only, or over nothing, allocates nothing.

Refused with an `ArgumentError`: a space without ghosts, an already constrained space, a
facet with two Dirichlet or two ghost relations, a ghost relation on a facet that carries
no ghosts, a ghosted facet with `u => g` and no ghost relation, a relation that does not
read its facet's ghost (`β = 0`, or a `jumpₙ` alone), two Dirichlet relations disagreeing
at a shared node, and a singular block at a corner (a singular `K`).

# Examples

```julia
Ωₕ = mesh(domain(interval(0.0, 1.0)), 11, false)
Wₕ = gridspace(Ωₕ; ghosts = :boundary)
Cₕ = constrained(Wₕ, u -> (:xmin => ∂ₙ(u) => 1.0, :xmax => u + ∂ₙ(u) => 2.0))
a = form(Cₕ, Cₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
l = form(Cₕ, v -> innerₕ(x -> 1.0, v))
A, F = assemble(a, l)

Dₕ = constrained(gridspace(Ωₕ; ghosts = (:xmax,)), u -> (:xmin => u => 0.0,
    :xmax => ∂ₙ(u) => x -> x[1]))  # 10 unknowns: the node at xmin is fixed

Tₕ = constrained(Wₕ, u -> (:xmin => ∂ₙ(u) => (x, t) -> sin(t), :xmax => u + ∂ₙ(u) => 2.0))
Bramble.lift(Tₕ, zeros(Bramble.ndofs(Tₕ)), 0.5)   # E u + d(0.5)
```

See also: [`gridspace`](@ref), [`∂ₙ`](@ref), [`∂ₙₙ`](@ref)
"""
function constrained(Wₕ::GhostGridSpace, relations)
    _check_unconstrained(Wₕ)
    P = padded_space(Wₕ)
    rels = _facet_relations(mesh(P), relations(trial_function(P)))
    return _constrained_space(Wₕ, filter(!_is_dirichlet, rels), filter(_is_dirichlet, rels))
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
    ndofs(Wₕ) == npoints(physical_mesh(G)) + nghosts(G) && iszero(Wₕ.d) &&
        !_is_time_dependent(Wₕ) && return nothing
    throw(
        ArgumentError(
        "constrained takes the space of gridspace(Ωₕ; ghosts = ...), and $(Wₕ) is already " *
        "constrained. Give every relation in one call on the unconstrained space.",
    ),
    )
end

# --- Parsing the relations ----------------------------------------------------------- #

# One parsed relation: the facet `(d, s)` (`s` 1 the minimum, 2 the maximum), its
# canonical name, the facet operator (the trial function itself for `u => g`), the data and
# the component it constrains (1 on a scalar space).
struct _FacetRelation{O, T}
    d::Int
    s::Int
    name::Symbol
    op::O
    data::T
    c::Int
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
        any(q -> q.d == r.d && q.s == r.s && _is_dirichlet(q) == _is_dirichlet(r), rels) &&
            _throw_facet_twice(r.name, _is_dirichlet(r))
        push!(rels, r)
    end
    return rels
end

@inline _is_dirichlet(r::_FacetRelation) = r.op isa Union{TrialFunction,
    IndexedTrialFunction}

# Data depending on time take the time as a second argument, read off their methods as
# `dirichlet_constraints` reads its conditions (`_accepts`, geometry/marker.jl).
@inline _is_time_datum(data) = data isa Function && _accepts(data, Val(2))
@inline _is_time_relation(r::_FacetRelation) = _is_time_datum(r.data)

# The datum at the point `x` at time `t`.
@inline _datum(data::Number, x, t) = data
@inline _datum(data, x, t) = _is_time_datum(data) ? data(x, t) : data(x)

function _facet_relation(G::GhostMesh{D}, p::Pair{Symbol, <:Pair}) where {D}
    face = _find_face(Val(D), first(p))
    face === nothing && _throw_not_a_relation_facet(first(p), D)
    d, s = face
    name = _facet_name(d, s)
    op, data = last(p)
    _check_relation_operator(op, name)
    op isa TrialFunction || has_ghosts(G, d, _GHOST_SIDES[s]) || _throw_relation_unghosted(name)
    _check_relation_data(data, name)
    return _FacetRelation(d, s, name, op, data, 1)
end
_facet_relation(G::GhostMesh, p) = _throw_relation_shape(p)

_check_relation_operator(::FacetOperator, name::Symbol) = nothing
_check_relation_operator(::TrialFunction, name::Symbol) = nothing   # Dirichlet, `u => g`
@noinline function _check_relation_operator(op, name::Symbol)
    throw(
        ArgumentError(
        "the relation on :$name has the operator $(summary(op)), which reads no ghost, so " *
        "it cannot be solved for the ghosts of :$name. Write a facet operator that reads " *
        "the facet's ghost, such as ∂ₙ(u), α * u + β * ∂ₙ(u) with β ≠ 0, or ∂ₙₙ(u); or " *
        "`u` alone, as in `:$name => u => g`, for a Dirichlet condition.",
    ),
    )
end

_check_relation_data(::Union{Number, Function}, ::Symbol) = nothing
@noinline function _check_relation_data(data, name::Symbol)
    throw(
        ArgumentError(
        "the data of the relation on :$name is $(summary(data)); give a number or a " *
        "function of the point, as in `:$name => ∂ₙ(u) => x -> x[1]`, or of the point " *
        "and the time, as in `:$name => ∂ₙ(u) => (x, t) -> t * x[1]`.",
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

@noinline function _throw_facet_twice(name::Symbol, dirichlet::Bool, c::Int = 0)
    what = dirichlet ? "two Dirichlet relations `u => g`" : "two relations reading its ghost"
    whose = c == 0 ? "" : " of component $c"
    throw(
        ArgumentError(
        "the facet :$name$whose has $what; give at most one of each kind per facet, " *
        "summing the operators into one relation if both hold.",
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

# --- Relations on a composite of ghost spaces ---------------------------------------- #

# The relations of `constrained(C₁ × C₂, relations)`, `Gs` the components' ghost meshes. A
# Dirichlet relation `u(i) => g` names its component, which has at most one per facet. The
# ghost relations of a facet are not keyed by component. A relation reading one component
# alone (`∂ₙ(u(2))`) is attached to that component's ghosts, at most one per component, and
# a component without one keeps its ghosts there as unknowns. A facet with a relation
# reading several components (a traction) has one ghost relation per component, the others
# attached one by one to the components left. The rows of a node, or of a corner, are solved
# together, so which ghost a row is attached to changes no value. Data depending on time
# are refused (they would need a `d(t)` of the whole).
function _composite_relations(Gs::Tuple, out)
    pairs = _relation_pairs(out)
    isempty(pairs) && _throw_relation_shape(out)
    rels = Any[]
    for p in pairs
        r = _composite_relation(Gs, p)
        _is_dirichlet(r) &&
            any(q -> _is_dirichlet(q) && q.c == r.c && q.d == r.d && q.s == r.s, rels) &&
            _throw_facet_twice(r.name, true, r.c)
        push!(rels, r)
    end
    return _attach_relations(rels, length(Gs))
end

# Each facet's ghost relations attached to distinct components: first each that reads one
# component alone to that one, then the rest to the components left, in order. `r.c` of a
# ghost relation is the component it reads alone, `0` for several. A facet of one-component
# relations takes at most one per component; one with a relation reading several takes one
# per component.
function _attach_relations(rels::Vector{Any}, N::Int)
    out = copy(rels)
    for (i, r) in enumerate(rels)
        _is_dirichlet(r) && continue
        any(q -> !_is_dirichlet(q) && q.d == r.d && q.s == r.s, view(rels, 1:(i - 1))) &&
            continue
        idx = findall(q -> !_is_dirichlet(q) && q.d == r.d && q.s == r.s, rels)
        if any(k -> rels[k].c == 0, idx)
            length(idx) == N || _throw_facet_relation_count(r.name, length(idx), N)
        else
            for (a, k) in enumerate(idx)
                c = rels[k].c
                any(j -> rels[j].c == c, view(idx, 1:(a - 1))) &&
                    _throw_facet_twice(r.name, false, c)
            end
            continue                        # each already attached to its own component
        end
        taken = falses(N)
        attached = zeros(Int, N)
        for (a, k) in enumerate(idx)
            c = rels[k].c
            c != 0 && !taken[c] && (attached[a] = c; taken[c] = true)
        end
        for a in 1:N
            attached[a] == 0 || continue
            c = findfirst(!, taken)::Int
            attached[a], taken[c] = c, true
        end
        for (a, k) in enumerate(idx)
            q = rels[k]
            out[k] = _FacetRelation(q.d, q.s, q.name, q.op, q.data, attached[a])
        end
    end
    return out
end

function _composite_relation(Gs::Tuple{Vararg{GhostMesh{D}}},
        p::Pair{Symbol, <:Pair}) where {D}
    face = _find_face(Val(D), first(p))
    face === nothing && _throw_not_a_relation_facet(first(p), D)
    d, s = face
    name = _facet_name(d, s)
    op, data = last(p)
    c = _relation_component(op, d, s, name)
    c == 0 || _check_component_range(c, length(Gs))
    op isa IndexedTrialFunction || has_ghosts(first(Gs), d, _GHOST_SIDES[s]) ||
        _throw_relation_unghosted(name)
    _check_relation_data(data, name)
    _is_time_datum(data) && _throw_composite_time(name)
    return _FacetRelation(d, s, name, op, data, c)
end
_composite_relation(::Tuple, p) = _throw_relation_shape(p)

# The component a relation reads on its own facet: its trial leaf's, or `0` for a facet
# operator reading several, such as a traction `∂ₙ(u(1); K = K₁₁) + ∂ₙ(u(2); K = K₁₂)`.
_relation_component(op::IndexedTrialFunction, ::Int, ::Int, ::Symbol) = op.component_idx
function _relation_component(op::FacetOperator, d::Int, s::Int, name::Symbol)
    c = _trial_component_class(_facet_component(op, d, s))
    c isa Int && return c
    c isa _MixedComponents && return 0
    return _throw_relation_components(name)
end
function _relation_component(::TrialFunction, ::Int, ::Int, name::Symbol)
    return _throw_relation_components(name)
end
_relation_component(op, ::Int, ::Int, name::Symbol) = _check_relation_operator(op, name)

@noinline function _throw_relation_components(name::Symbol)
    throw(
        ArgumentError(
        "the relation on :$name names no component of the composite. Write it of the " *
        "components u(i), such as `:$name => ∂ₙ(u(1)) => g`, " *
        "`:$name => ∂ₙ(u(1); K = K₁₁) + ∂ₙ(u(2); K = K₁₂) => g` (a traction), or " *
        "`:$name => u(2) => g` for a Dirichlet condition.",
    ),
    )
end

@noinline function _throw_facet_relation_count(name::Symbol, n::Int, N::Int)
    throw(
        ArgumentError(
        "the facet :$name has $n relation$(n == 1 ? "" : "s") reading its ghosts, and " *
        "each of its nodes has a ghost in each of the $N components. Give :$name one " *
        "ghost relation per component, solved together at each node, as a relation " *
        "there reads several components (a traction gives one per component, " *
        "`:$name => ∂ₙ(u(1); K = K₁₁) + ∂ₙ(u(2); K = K₁₂) => g₁`, …); or relations of " *
        "one component each, at most one per component. A component fixed on :$name " *
        "by `u(i) => g` takes `:$name => ∂ₙₙ(u(i)) => h` as its ghost relation.",
    ),
    )
end

# The pieces of a ghost relation's operator on its facet, one per component it reads, each
# as a facet operator with that piece on every facet (`markers` selects its own): one
# `inner_Γ` holds one trial component (block_extract.jl), so the relation's row is the sum
# of one per piece, as `_strain_pieces` keeps the terms of `εₕ` apart (difference.jl).
function _relation_pieces(op::FacetOperator{D, G}, d::Int, s::Int, name::Symbol,
        N::Int) where {D, G}
    pieces = Any[]
    for j in 1:N
        p = _trial_piece(_facet_component(op, d, s), j, name)
        p === nothing || push!(pieces, _facet_operator(Val(G), ntuple(_ -> (p, p), Val(D))))
    end
    return pieces
end

# The terms of `op` that read component `j`, or `nothing` if none does: the leaves of the
# other components are dropped, with the sums, scales and differences around them.
function _trial_piece(op::IndexedTrialFunction, j::Int, ::Symbol)
    return op.component_idx == j ? op : nothing
end
for T in (
    :BackwardDifference,
    :ForwardDifference,
    :CenteredDifference,
    :StarDifference,
    :CrossWeightedDifference,
    :SecondDifference,
    :JumpNode,
    :BackwardAverage,
    :ForwardAverage,
    :CenteredAverage
)
    @eval function _trial_piece(op::$T{D, Dim}, j::Int, name::Symbol) where {D, Dim}
        inner = _trial_piece(op.inner_op, j, name)
        inner === nothing && return nothing
        return $T{D, Dim, typeof(inner)}(inner)
    end
end
function _trial_piece(op::ShiftNode{D, Dim}, j::Int, name::Symbol) where {D, Dim}
    inner = _trial_piece(op.inner_op, j, name)
    inner === nothing && return nothing
    return ShiftNode{D, Dim, typeof(inner)}(op.shift_amount, inner)
end
function _trial_piece(op::OperatorScale{D, S}, j::Int, name::Symbol) where {D, S}
    inner = _trial_piece(op.inner_op, j, name)
    inner === nothing && return nothing
    return OperatorScale{D, S, typeof(inner)}(op.scalar, inner)
end
function _trial_piece(op::GridFunctionScale{D, V}, j::Int, name::Symbol) where {D, V}
    inner = _trial_piece(op.inner_op, j, name)
    inner === nothing && return nothing
    return GridFunctionScale{D, V, typeof(inner)}(op.grid_function, inner)
end
function _trial_piece(op::OperatorAdd{D}, j::Int, name::Symbol) where {D}
    l = _trial_piece(op.left_op, j, name)
    r = _trial_piece(op.right_op, j, name)
    l === nothing && return r
    r === nothing && return l
    return OperatorAdd{D, typeof(l), typeof(r)}(l, r)
end
function _trial_piece(op, j::Int, name::Symbol)
    c = _trial_component_class(op)
    c isa Int || _throw_relation_unsplit(name, op)
    return c == j ? op : nothing
end

@noinline function _throw_relation_unsplit(name::Symbol, op)
    throw(
        ArgumentError(
        "the relation on :$name holds the term $(summary(op)), which cannot be split into " *
        "one term per component it reads. Write the relation as a sum of facet operators " *
        "of one component each, such as `∂ₙ(u(1); K = K₁₁) + ∂ₙ(u(2); K = K₁₂)`.",
    ),
    )
end

@noinline function _throw_composite_time(name::Symbol)
    throw(
        ArgumentError(
        "the data of the relation on :$name depend on time, which constrained does not " *
        "take on a composite of ghost spaces. Give data of the point alone, x -> …, or " *
        "constrain each component on its own and take the product, " *
        "constrained(W₁, r₁) × constrained(W₂, r₂), whose components take data in time.",
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
# facet at its node, and `b[k]` the weight times the data there (the weight alone, for data
# depending on time); `rel[k]` is the index in `rels` of that relation.
function _relation_rows(P::ScalarGridSpace, rels)
    R, b, ghosts, rel = _relation_rows(P, ((mesh(P), 0),), rels)
    return R, b, Int[g for (_, g) in ghosts], rel
end

# The same on a padded composite, `comps` holding each component's `GhostMesh` and the
# offset of its padded values: a relation of component `c` eliminates the ghosts of its
# facet in that component, and `ghosts[k]` is the pair `(c, g)`, ghost `g` of component `c`,
# component by component. The columns of `R` run over the whole padded composite. The rows
# on a facet of `onesided`, at a node it shares with another, read the other's ghosts
# through the one-sided difference along the facet (`_one_sided_corner!`).
function _relation_rows(P, comps::Tuple, rels; onesided = ())
    T = eltype(P)
    isempty(rels) && return spzeros(T, 0, ndofs(P)), T[], Tuple{Int, Int}[], Int[]
    ghosts, rel = Tuple{Int, Int}[], Int[]
    for (c, (G, _)) in enumerate(comps), g in 1:nghosts(G)

        _, d, side = boundary_node(G, g)
        i = findfirst(r -> r.c == c && r.d == d && _GHOST_SIDES[r.s] === side, rels)
        i === nothing || (push!(ghosts, (c, g)); push!(rel, i))
    end
    blocks = map(r -> _assemble_relation(P, r), rels)
    rows, cols, vals = Int[], Int[], promote_type(map(x -> eltype(first(x)), blocks)...)[]
    b = zeros(promote_type(eltype(vals), map(x -> eltype(last(x)), blocks)...), length(ghosts))
    for (k, (c, g)) in enumerate(ghosts)
        G, offset = comps[c]
        Rt, bF = blocks[rel[k]]
        plin = LinearIndices(npoints(G, Tuple))
        I = first(boundary_node(G, g))
        i = offset + plin[padded_index(G, I)]
        start = length(rows) + 1
        for q in nzrange(Rt, i)
            push!(rows, k)
            push!(cols, rowvals(Rt)[q])
            push!(vals, nonzeros(Rt)[q])
        end
        r = rels[rel[k]]
        r.name in onesided &&
            _one_sided_corner!(rows, cols, vals, start, k, comps, I, r.d, onesided)
        b[k] = bF[i]
    end
    return sparse(rows, cols, vals, length(ghosts), ndofs(P)), b, ghosts, rel
end

# The row `k`, its entries `start:end` of `(rows, cols, vals)`, of a relation on a facet of
# axis `d` at its node `I`: on each other facet of `facets` that `I` lies on, the row's
# entries at that facet's ghosts beyond `I`, one per component, move to the three physical
# nodes along the facet, by the quadratic through them extrapolated to the ghost
# (`_one_sided_weights`). The centred difference along the facet, `(u_g − u₁) / (x_g − x₁)`
# with the ghost `g` mirroring the first neighbour `1` about the node, then reads the
# derivative of that quadratic at the node: the one-sided difference into the domain,
# second order on the non-uniform spacing.
function _one_sided_corner!(rows, cols, vals, start::Int, k::Int, comps::Tuple,
        I::CartesianIndex{D}, d::Int, facets) where {D}
    n = npoints(physical_mesh(first(first(comps))), Tuple)
    for e in 1:D, t in 1:2

        e != d && I[e] == (t == 1 ? 1 : n[e]) && _facet_name(e, t) in facets || continue
        n[e] >= 3 || _throw_corner_axis_too_short(_facet_name(e, t), e, n[e])
        step = CartesianIndex(ntuple(a -> a == e ? (t == 1 ? -1 : 1) : 0, Val(D)))
        for (G, offset) in comps
            has_ghosts(G, e, _GHOST_SIDES[t]) || continue
            plin = LinearIndices(npoints(G, Tuple))
            J = padded_index(G, I)
            ghost = offset + plin[J + step]
            v = zero(eltype(vals))
            for q in start:lastindex(cols)
                cols[q] == ghost && (v += vals[q]; vals[q] = zero(v))
            end
            iszero(v) && continue
            x = host_points(_padded(G)(e))
            at(m) = J[e] + m * step[e]
            w = _one_sided_weights(x[at(1)], x[at(0)], x[at(-1)], x[at(-2)])
            for m in 0:2
                push!(rows, k), push!(cols, offset + plin[J - m * step])
                push!(vals, v * w[m + 1])
            end
        end
    end
    return nothing
end

# The weights at the nodes `x₀`, `x₁`, `x₂` of the quadratic through them, evaluated at `x`:
# the one-sided second-order difference of a corner, as `_one_sided_corner!` applies them,
# and the facet rule's extrapolation to a ghost (`_extrapolated`, stencil_eval.jl).
@inline function _one_sided_weights(x, x₀, x₁, x₂)
    return ((x - x₁) * (x - x₂) / ((x₀ - x₁) * (x₀ - x₂)),
        (x - x₀) * (x - x₂) / ((x₁ - x₀) * (x₁ - x₂)),
        (x - x₀) * (x - x₁) / ((x₂ - x₀) * (x₂ - x₁)))
end

@noinline function _throw_corner_axis_too_short(name::Symbol, e::Int, n::Int)
    throw(
        ArgumentError(
        "the relations on :$name and on a facet it meets each read several components " *
        "(tractions), so at their corner each reads the other's ghosts through a " *
        "one-sided difference over three nodes along axis $e, which has $n. Use a mesh " *
        "with at least three nodes along axis $e.",
    ),
    )
end

# The relation's rows and data on the nodes of its facet, by forms on the padded space;
# the rows transposed, so each node's row is a column. Data depending on time give the
# facet weights instead, which `L` multiplies the data at each time by.
function _assemble_relation(P::ScalarGridSpace, r::_FacetRelation)
    op, name = r.op, r.name
    data = _is_time_relation(r) ? one(eltype(P)) : r.data
    R = assemble(form(P, P, (u, v) -> inner_Γ(op, v; markers = (name,))))
    b = assemble(form(P, v -> inner_Γ(data, _facet_identity(v); markers = (name,))))
    return sparse(transpose(R)), b
end

# On a padded composite, tested against the component `r.c` whose ghosts the relation is
# attached to: the sum of one `inner_Γ` per component the relation reads.
function _assemble_relation(P::CompositeGridSpace{N}, r::_FacetRelation) where {N}
    name, c = r.name, r.c
    R = sum(_relation_pieces(r.op, r.d, r.s, name, N)) do F
        return assemble(form(P, P, (u, v) -> inner_Γ(F, v(c); markers = (name,))))
    end
    b = assemble(form(P, v -> inner_Γ(r.data, _facet_identity(v(c)); markers = (name,))))
    return sparse(transpose(R)), b
end

# --- The closure --------------------------------------------------------------------- #

function _constrained_space(Wₕ::GhostGridSpace, rels, drels)
    P = padded_space(Wₕ)
    G = mesh(P)
    for r in drels
        has_ghosts(G, r.d, _GHOST_SIDES[r.s]) || continue
        any(q -> q.d == r.d && q.s == r.s, rels) || _throw_dirichlet_orphan_ghosts(r.name)
    end
    R, b, ghosts, rel = _relation_rows(P, rels)
    fixed, g, order, owner = _dirichlet_values(G, drels, eltype(P))
    keep = _kept_points(G, ghosts, fixed)
    T = eltype(P)
    be = backend(P)
    if !any(_is_time_relation, rels) && !any(_is_time_relation, drels)
        E, d, _ = _relation_closure(G, R, b, ghosts, keep, fixed,
            g)::Tuple{SparseMatrixCSC, Vector, Nothing}
        return GhostGridSpace(P, _closure_matrix(be, T, E), _closure_data(be, T, d))
    end
    w = map(k -> _is_time_relation(rels[rel[k]]) ? b[k] : one(eltype(b)), eachindex(b))
    E, _, L = _relation_closure(G, R, b, ghosts, keep, fixed, g, (w, order))
    td = _time_data(be, T, G, L, rels, drels, b, ghosts, rel, g, order, owner)
    d = _backend_vector(be, T, size(E, 1))
    W = GhostGridSpace(P, _closure_matrix(be, T, E), d, td)
    _lift_affine!(d, W, zero(real(T)))      # the stored `d` is `d(0)`
    return W
end

# The closure of the padded composite `P` under the ghost relations `rels` and the
# Dirichlet ones `drels`, `comps` each component's `GhostMesh` and padded offset: the rows,
# kept points and fixed points of every component, at its offset, solved together by the
# core below, so `E` couples two components wherever a relation does. The degrees of
# freedom are each component's, in the order `constrained` gives it alone, one after the
# other.
function _composite_closure(P::CompositeGridSpace, comps::Tuple, rels, drels)
    for r in drels
        has_ghosts(first(comps[r.c]), r.d, _GHOST_SIDES[r.s]) || continue
        any(q -> q.c == r.c && q.d == r.d && q.s == r.s, rels) ||
            _throw_dirichlet_orphan_ghosts(r.name)
    end
    reads_several(r) = _relation_component(r.op, r.d, r.s, r.name) == 0
    multi = unique!(Symbol[r.name for r in rels if reads_several(r)])
    R, b, ghosts, _ = _relation_rows(P, comps, rels; onesided = multi)
    fixed, gs, keep = falses(ndofs(P)), Any[], Int[]
    for (c, (G, offset)) in enumerate(comps)
        f, g, _, _ = _dirichlet_values(G, filter(r -> r.c == c, drels), eltype(P))
        own = Int[k for (j, k) in ghosts if j == c]
        append!(keep, offset .+ _kept_points(G, own, f))
        fixed[offset .+ eachindex(f)] .= f
        push!(gs, g)
    end
    gpts = Int[comps[c][2] + LinearIndices(npoints(first(comps[c]), Tuple))[
                   _ghost_point(first(comps[c]), k)] for (c, k) in ghosts]
    names = Symbol[_relation_name(first(comps[c]), k) for (c, k) in ghosts]
    owner = Int[c for (c, _) in ghosts]
    E, d, _ = _relation_closure(gpts, names, R, b, keep, fixed, reduce(vcat, gs);
        comps = owner)::Tuple{SparseMatrixCSC, Vector, Nothing}
    return E, d
end

# The record of `d(t) = L g(t)`: the sources are the eliminated ghosts (`1:m`, in the order
# of `ghosts`), then the fixed points in `order`. `g` starts with the static values, the
# facet-weighted data `b[k]` of a static relation and the value of a static Dirichlet datum;
# each relation with data depending on time gets a group of its sources, which are
# contiguous: a facet's ghosts follow one another, and so do the points a Dirichlet
# relation fixes first.
function _time_data(be, ::Type{T}, G::GhostMesh, L::SparseMatrixCSC, rels, drels, b,
        ghosts, rel, gfix, order, owner) where {T}
    Ωₕ = physical_mesh(G)
    m = length(ghosts)
    g = zeros(T, m + length(order))
    for k in 1:m
        _is_time_relation(rels[rel[k]]) || (g[k] = b[k])
    end
    for (q, p) in enumerate(order)
        _is_time_relation(drels[owner[q]]) || (g[m + q] = gfix[p])
    end
    groups = Any[]
    for (i, r) in enumerate(rels)
        _is_time_relation(r) || continue
        ks = findall(==(i), rel)
        isempty(ks) && continue
        pts = [point(Ωₕ, first(boundary_node(G, ghosts[k]))) for k in ks]
        push!(groups, _time_group(be, r.data, pts, first(ks)))
    end
    pin = CartesianIndices(npoints(G, Tuple))
    phys = first(physical_indices(G)) - oneunit(first(pin))
    for (i, r) in enumerate(drels)
        _is_time_relation(r) || continue
        qs = findall(==(i), owner)
        isempty(qs) && continue
        pts = [point(Ωₕ, pin[order[q]] - phys) for q in qs]
        push!(groups, _time_group(be, r.data, pts, m + first(qs)))
    end
    Lt = sparse(transpose(SparseMatrixCSC{T, Int}(L)))
    stored(x) = copyto!(_backend_vector(be, eltype(x), length(x)), x)
    return _TimeData(Tuple(groups), stored(SparseArrays.getcolptr(Lt)), stored(rowvals(Lt)),
        stored(nonzeros(Lt)), stored(g), stored(g))
end

function _time_group(be, f, pts::Vector, first::Int)
    return _TimeGroup(f, copyto!(_backend_vector(be, eltype(pts), length(pts)), pts), first)
end

# The padded points the Dirichlet relations fix, as a mask over the padded grid, and their
# values: the data at each physical node of the facet, `data(point(Ωₕ, I))` as `dirichlet`
# evaluates it (at `t = 0` for data depending on time). A node of two such facets (a
# corner) takes the first value, and the second must agree with it to a relative `√eps` of
# the larger of the two, plus `10 eps` of the largest datum of all, so that a rounding zero
# (`sin(π x)` at `x = 1`) meets an exact one. `order` lists the fixed points as first
# fixed, and `owner` the index in `drels` of the relation that fixed each.
function _dirichlet_values(G::GhostMesh{D}, drels, ::Type{T}) where {D, T}
    Ωₕ = physical_mesh(G)
    plin = LinearIndices(npoints(G, Tuple))
    n = npoints(Ωₕ, Tuple)
    pts, vals, names, rids = Int[], Any[], Symbol[], Int[]
    t0 = zero(real(T))
    for (i, r) in enumerate(drels)
        k = r.s == 1 ? 1 : n[r.d]
        for I in CartesianIndices(ntuple(e -> e == r.d ? (k:k) : (1:n[e]), Val(D)))
            push!(pts, plin[padded_index(G, I)])
            push!(vals, _datum(r.data, point(Ωₕ, I), t0))
            push!(names, r.name)
            push!(rids, i)
        end
    end
    S = mapreduce(typeof, promote_type, vals; init = T)
    fixed = falses(length(plin))
    g = zeros(S, length(plin))::Vector{S}
    first_name = Vector{Symbol}(undef, length(plin))
    floor = 10 * eps(real(S)) * maximum(abs, Iterators.filter(isfinite, vals);
        init = zero(real(S)))
    order, owner, first_rel = Int[], Int[], zeros(Int, length(plin))
    for (p, v, name, i) in zip(pts, vals, names, rids)
        if fixed[p]
            isequal(g[p], v) && continue
            isfinite(g[p]) || _throw_dirichlet_not_finite(drels[first_rel[p]], g[p])
            isfinite(v) || _throw_dirichlet_not_finite(drels[i], v)
            abs(g[p] - v) <= sqrt(eps(real(S))) * max(abs(g[p]), abs(v)) + floor ||
                _throw_dirichlet_disagree(first_name[p], name, g[p], v)
        else
            fixed[p], g[p], first_name[p], first_rel[p] = true, v, name, i
            push!(order, p), push!(owner, i)
        end
    end
    return fixed, g, order, owner
end

# Two equal values agree, `Inf` included; a value that is not finite and differs from the
# other cannot be compared, so the datum itself is named.
@noinline function _throw_dirichlet_not_finite(r::_FacetRelation, v)
    when = _is_time_relation(r) ? " at t = 0, where constrained evaluates it" : ""
    throw(
        ArgumentError(
        "the Dirichlet datum on :$(r.name) is not finite$when ($v) at a node it shares " *
        "with another Dirichlet facet, so the two cannot be checked to agree there. Give " *
        "data that are finite at the corners of their facets.",
    ),
    )
end

# The ghosts of a ghosted facet whose nodes `u => g` fixes would be unknowns that no
# equation reads: the fixed nodes' rows are gone and no relation solves for them.
@noinline function _throw_dirichlet_orphan_ghosts(name::Symbol)
    throw(
        ArgumentError(
        "the facet :$name carries ghosts and only the Dirichlet relation `u => g`, so its " *
        "ghosts would be unknowns that no equation reads (a singular system). Drop :$name " *
        "from `gridspace(Ωₕ; ghosts = ...)`, or give it a ghost relation as well, such as " *
        "`:$name => ∂ₙ(u) => h` or `:$name => ∂ₙₙ(u) => h`.",
    ),
    )
end

@noinline function _throw_dirichlet_disagree(a::Symbol, b::Symbol, ga, gb)
    throw(
        ArgumentError(
        "the Dirichlet relations on :$a and :$b give a node they share (a corner) the " *
        "values $ga and $gb. Give data that agree where the facets meet.",
    ),
    )
end

# The padded points of the degrees of freedom that remain: the physical nodes no Dirichlet
# relation fixes, then the ghosts no relation eliminates, in the order of the ghost space.
function _kept_points(G::GhostMesh{D}, ghosts::Vector{Int}, fixed::BitVector) where {D}
    plin = LinearIndices(npoints(G, Tuple))
    keep = Int[plin[J] for J in physical_indices(G) if !fixed[plin[J]]]
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

# `u_pad = E u + d`: a kept point copies its degree of freedom, a fixed point holds its
# Dirichlet value `g`, and the eliminated ghosts solve `C u_g = b − B u − R_fixed g`, `C`
# their columns of the rows and `B` the kept ones. `C` couples only the ghosts of one node,
# so it is solved block by block, each block's rows of `B` with its own LU.
# With `sources = (w, order)` it also returns `L`, `d = L g` for the sources `g` of
# `_time_data`: column `k` of `L` is the solve for `b = w[k] e_k`, column `m + q` the solve
# for `b = −R[:, p]` plus the 1 at `p = order[q]`; otherwise `L` is `nothing`.
function _relation_closure(G::GhostMesh, R::SparseMatrixCSC, b, ghosts, keep,
        fixed::BitVector, g::AbstractVector, sources = nothing)
    plin = LinearIndices(npoints(G, Tuple))
    gpts = Int[plin[_ghost_point(G, k)] for k in ghosts]
    names = Symbol[_relation_name(G, k) for k in ghosts]
    return _relation_closure(gpts, names, R, b, keep, fixed, g, sources)
end

# The closure on index sets alone, as a padded composite gives them: the eliminated ghosts'
# padded points `gpts`, the names of their relations, for the refusals, and the kept points.
# On a composite, `comps` holds the component of each eliminated ghost: a row may read
# other components' ghosts and none of its own (a traction attached to a component whose
# normal derivative it does not read), so only each block as a whole must be nonsingular.
# A scalar space's row must read its own ghost.
function _relation_closure(gpts::Vector{Int}, names::Vector{Symbol}, R::SparseMatrixCSC,
        b, keep, fixed::BitVector, g::AbstractVector, sources = nothing;
        comps::Union{Nothing, Vector{Int}} = nothing)
    Npad = size(R, 2)
    m, n = length(gpts), length(keep)
    gpos = zeros(Int, Npad)
    kpos = zeros(Int, Npad)
    gpos[gpts] .= 1:m
    kpos[keep] .= 1:n
    bg = Vector{promote_type(eltype(b), eltype(R), eltype(g))}(b)
    Ci, Cj, Cv = Int[], Int[], eltype(R)[]
    Bi, Bj, Bv = Int[], Int[], eltype(R)[]
    Fi, Fj, Fv = Int[], Int[], eltype(R)[]   # `−R_fixed`, by fixed point
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
            push!(Fi, k), push!(Fj, p), push!(Fv, -v)
        else
            _throw_relation_reads_absent(names[k])
        end
    end
    C = sparse(Ci, Cj, Cv, m, m)
    if comps === nothing
        for k in 1:m
            iszero(C[k, k]) && _throw_relation_misses_ghost(names[k])
        end
    end
    Bt = sparse(Bj, Bi, Bv, n, m)            # the rows of `B` as columns
    Ei, Ej, Ev, dg = _block_solve(names, C, Bt, bg, comps)
    rows = vcat(keep, gpts[Ei])
    cols = vcat(1:n, Ej)
    vals = vcat(ones(eltype(Ev), n), Ev)
    d = zeros(promote_type(eltype(dg), eltype(g)), Npad)
    d[gpts] .= dg
    d[fixed] .= g[fixed]
    L = _affine_map(names, C, gpts, Fi, Fj, Fv, Npad, sources)
    return sparse(rows, cols, vals, Npad, n), d, L
end

_affine_map(names, C, gpts, Fi, Fj, Fv, Npad, ::Nothing) = nothing
function _affine_map(names::Vector{Symbol}, C::SparseMatrixCSC, gpts, Fi, Fj, Fv, Npad::Int,
        (w, order)::Tuple)
    m, nf = size(C, 1), length(order)
    src = zeros(Int, Npad)
    src[order] .= m .+ (1:nf)
    St = sparse(vcat(1:m, src[Fj]), vcat(1:m, Fi), vcat(w, Fv), m + nf, m)
    Li, Lj, Lv, _ = _block_solve(names, C, St, zeros(eltype(St), m))
    return sparse(vcat(gpts[Li], order), vcat(Lj, m .+ (1:nf)),
        vcat(Lv, ones(eltype(Lv), nf)), Npad, m + nf)
end

@inline function _relation_name(G::GhostMesh, g::Int)
    _, d, side = boundary_node(G, g)
    return _facet_name(d, _ghost_side(side))
end

# `C \ B` and `C \ b`, block by block: the blocks are the connected sets of ghosts the rows
# couple, at most one per node of the mesh, so each is a small dense LU solve on the
# columns its rows of `B` touch. The result is the triplets `(k, j, v)` of the ghosts' rows
# of `E` and the ghosts' `d`. A solve, not a multiplication by an inverse: on a block near
# singularity (a nearly singular `K`) the inverse leaves the relations unsatisfied.
# `names[k]` is the relation of ghost `k`, which a singular block names, with, on a
# composite, the components of its ghosts.
function _block_solve(names::Vector{Symbol}, C::SparseMatrixCSC, Bt::SparseMatrixCSC, b,
        comps::Union{Nothing, Vector{Int}} = nothing)
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
    T = promote_type(eltype(C), eltype(Bt))
    rows, cols, vals = Int[], Int[], T[]
    dg = zeros(promote_type(eltype(C), eltype(b)), m)
    for idx in values(members)
        if length(idx) == 1
            k = only(idx)
            c = C[k, k]
            iszero(c) && _throw_singular_block(names, comps, idx)
            for q in nzrange(Bt, k)
                push!(rows, k), push!(cols, rowvals(Bt)[q]), push!(vals, nonzeros(Bt)[q] / c)
            end
            dg[k] = b[k] / c
            continue
        end
        Cb = Matrix(C[idx, idx])
        F = lu(Cb; check = false)
        _nonsingular(F, Cb) || _throw_singular_block(names, comps, idx)
        js = sort!(unique!(reduce(vcat, (rowvals(Bt)[nzrange(Bt, k)] for k in idx))))
        Bb = zeros(T, length(idx), length(js))
        for (a, k) in enumerate(idx), q in nzrange(Bt, k)

            Bb[a, searchsortedfirst(js, rowvals(Bt)[q])] = nonzeros(Bt)[q]
        end
        X = F \ Bb
        for (c, j) in enumerate(js), (a, k) in enumerate(idx)

            push!(rows, k), push!(cols, j), push!(vals, X[a, c])
        end
        dg[idx] .= F \ b[idx]
    end
    return rows, cols, vals, dg
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

_throw_singular_block(names, ::Nothing, idx) = _throw_singular_corner(names[idx])
function _throw_singular_block(names, owner::Vector{Int}, idx)
    order = (:xmin, :xmax, :ymin, :ymax, :zmin, :zmax)
    facets = sort!(unique(names[idx]); by = f -> findfirst(==(f), order))
    return _throw_composite_singular(facets, sort!(unique(owner[idx])))
end

function _component_list(cs::Vector{Int})
    length(cs) == 1 && return "component $(only(cs))"
    return "components $(join(cs, ", ", " and "))"
end

@noinline function _throw_composite_singular(facets, cs)
    names = join((":$s" for s in facets), " and ")
    throw(
        ArgumentError(
        "the relations on $names cannot be solved for the ghosts of " *
        "$(_component_list(cs)) at a node: their block is singular, the relations " *
        "reading those ghosts in fewer independent combinations than there are ghosts " *
        "(two relations of a facet that agree there, or none reading a component's " *
        "ghost). Give each component a relation whose rows read its ghost independently " *
        "of the others, such as `∂ₙ(u(i))`, or a traction Σⱼ ∂ₙ(u(j); K = Kᵢⱼ) with " *
        "Kᵢᵢ[d, d] ≠ 0 on the facet of axis d.",
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
