# ghost_assembly.jl: the assembly of forms posed on a `GhostGridSpace`, which eliminates
# the ghost values through the space's closure so that the assembled matrix acts on the
# space's degrees of freedom.
#
# A form on ghost spaces is the form on their padded spaces (the same placeholders, the
# same AST) wrapped with the closures `(E, d)` of both sides. `assemble` fills the padded
# matrix `A_pad` and reduces it to `E_vᵀ A_pad E_u` in DOF order. The load of a linear form
# is `E_vᵀ (F_pad − A_pad d_u)`, so it needs the bilinear form: `assemble(a, l)` builds it,
# `assemble!(A, F, a, l)` refills it, and `assemble(l)` alone refuses a test space with
# `d ≠ 0`, which is exact when both sides are one space (it cannot see the trial space's
# `d_u`). A map from `A_pad`'s stored values to the reduced matrix's, built once with the
# form, does the reduction, so a warm `assemble!` allocates nothing at all. Substituting `E`
# at each stencil tap instead would rewrite the guarded walk, the pattern recorder and the
# replay.
#
# The reduction, the load and `lift` run in the bands of the mesh's policy (`_run_bands!`):
# each band owns a range of output entries and sums every contribution to them itself, so
# no two bands write one entry, and every policy adds an entry's terms in the same order and
# gives the serial result bitwise. The reduction's contributions are grouped by reduced
# entry, `A_pad d` is stored by padded row over the columns where `d` can be nonzero (at
# any time, for data depending on time), `E_vᵀ w` reads the columns of `E_v` and `E u`
# those of the stored `Eᵀ`. The band functions take arrays and isbits scalars only, so that
# a `CpuPolyester` task boxes nothing.
#
# `dirichlet` and `symmetrize` replace rows of the reduced system, through the masks of the
# physical mesh, when its unknowns are the physical nodes in mesh order: every ghost
# eliminated by a relation and no node fixed by a Dirichlet relation `u => g`. Any other
# ghost space refuses them, naming `u => g` as the route that works there. A label is a
# marker of the physical mesh or, as in `constrained`, a facet name or its alias.
#
# On a space whose relation data depend on time, `assemble(a, l)`, `assemble!(A, F, a, l)`
# and the load alone refuse, naming `lift(Cₕ, u, t)` and `semidiscretize`, as there is no
# single `d` to lift with. `form` and `assemble(a)` read `E` only and work as on a static
# space.
#
# `semidiscretize` and `ode_problem` take ghost forms in `ghost_semidiscrete.jl`, whose load
# is the one built here with `d(t)` in place of `d`.
#
# A rank-one term `σ * innerₕ(g, u) * innerₕ(h, v)` gives a `RankOneForm` whose parts are
# ghost forms, reduced in `rank_one.jl`: the other terms as here, and each functional's
# padded vector through its own side's closure, `wₛ = E_vᵀ wₛ_pad` and `wₜ = E_uᵀ wₜ_pad`.
#
# A form on composites of ghost spaces, `C₁ × C₂`, is a `GhostCompositeBilinearForm` or
# `GhostCompositeLinearForm`, whose sides are `_CompositeClosure`s: the padded composite
# `padded(C₁) × padded(C₂)` with the closure `(blockdiag(E₁, E₂), vcat(d₁, d₂))`, or, on a
# `ConstrainedCompositeSpace`, the closure `constrained` stored. The reduction and the load
# below run on it unchanged (`_AnyGhostBilinearForm`); `ghost_composite.jl` builds it and
# holds its `dirichlet` rows and refusals. A type of its own keeps it out of the methods
# other files define on `GhostBilinearForm`.
#
# Still refused with an `ArgumentError`: device backends, matrix types other than
# `SparseMatrixCSC`, AD element types, composite spaces beyond what `ghost_composite.jl`
# supports and `semidiscretize` on composites. The matrix-free and Kronecker operators and
# the contractions of a form live in `ghost_matrix_free.jl`, `ghost_kronecker.jl` and
# `ghost_contract.jl`. The table `Support on ghost spaces` in `docs/src/internals/space.md`
# lists every operation.
#
# See also: `GhostGridSpace`, `BilinearForm`, `LinearForm`

"""
    ConstrainedCompositeSpace{N} <: AbstractSpaceType{N}

A composite of `N` ghost spaces constrained as a whole, built by
`constrained(C₁ × C₂, relations)` (ghost_composite.jl): the unconstrained composite, its
padded space and the closure `(E, d)` the relations give, stored once. `E` is general, not
block diagonal: the degrees of freedom are each component's, one component after the
other, and its padded values are `E * u + d`. It also stores the facets with a relation
reading several components, whose ghosts a form's mixed differences read by extrapolation
from the nodes inward (the facet rule, stencil_eval.jl). `form` applies the rule when
there is one. That is a value, not a type parameter, so `constrained` infers.
"""
struct ConstrainedCompositeSpace{N, S <: CompositeGridSpace{N}, P <: CompositeGridSpace{N},
    ET <: SparseMatrixCSC, DT <: AbstractVector, FT <: Tuple} <: AbstractSpaceType{N}
    "the unconstrained composite of ghost spaces the relations were given on."
    space::S
    "the composite of the components' padded spaces."
    padded::P
    "the closure matrix, padded values by degrees of freedom."
    E::ET
    "the affine part of the closure, one entry per padded value."
    d::DT
    "the transpose of `E`, whose columns give `lift` its padded rows."
    Et::ET
    "`facets[d][s]`: facet `(d, s)`, `s` 1 the minimum, has a relation reading several."
    facets::FT
end

function ConstrainedCompositeSpace(space::CompositeGridSpace{N},
        padded::CompositeGridSpace{N}, E::SparseMatrixCSC, d::AbstractVector,
        Et::SparseMatrixCSC, facets::Tuple) where {N}
    S, P = typeof(space), typeof(padded)
    return ConstrainedCompositeSpace{N, S, P, typeof(E), typeof(d), typeof(facets)}(
        space, padded, E, d, Et, facets)
end

"""
    _CompositeClosure

One side of a form on a composite of ghost spaces (`C₁ × C₂`, built in
`ghost_composite.jl`): the composite, its padded space `padded(C₁) × padded(C₂)` and the
closure of the whole: `E = blockdiag(E₁, E₂)` and `d = vcat(d₁, d₂)` when each component
is constrained by its own relations, the stored closure of a `ConstrainedCompositeSpace`.
It has the fields of a [`GhostGridSpace`](@ref) that the reduction reads, so a
`GhostCompositeBilinearForm` or `GhostCompositeLinearForm` holds it as a side.
"""
struct _CompositeClosure{
    S <: Union{CompositeGridSpace{<:Any, <:Tuple{Vararg{GhostGridSpace}}},
        ConstrainedCompositeSpace},
    P <: CompositeGridSpace{<:Any, <:Tuple{Vararg{ScalarGridSpace}}}, ET <: SparseMatrixCSC,
    DT <: AbstractVector}
    "the composite of ghost spaces, or one constrained as a whole."
    space::S
    "the composite of the components' padded spaces."
    padded::P
    "the closure matrix."
    E::ET
    "the affine part, the components' in order."
    d::DT
    "the transpose of `E`."
    Et::ET
end

# A side of a form on ghost spaces: a scalar ghost space or a composite of them.
const _GhostSide = Union{GhostGridSpace, _CompositeClosure}

# Row replacement on a form's spaces: the rows of each space are its physical nodes, and the
# facet masks are those of the test space's physical mesh (`_RowsDirichlet`, in
# ghost_gridspace.jl). A form on a polygon space is a polygon form, and it keeps the
# conflict nodes of its trial space, the last of `spaces`, which the matrix-free operator
# refuses.
function _RowsDirichlet(spaces::GhostGridSpace...)
    return _RowsDirichlet(all(W -> W.rows.physical, spaces), first(spaces).rows.facets,
        any(W -> W.rows.polygon, spaces), false, BitVector(), BitVector(),
        last(spaces).rows.conflict)
end

"""
    GhostBilinearForm

A bilinear form on two [`GhostGridSpace`](@ref)s, built by `form(Wₕ, Vₕ, f)`: the form on
their padded spaces, the padded matrix it fills, and the map that reduces that matrix to
`E_vᵀ A_pad E_u`, the matrix on the degrees of freedom.
"""
struct GhostBilinearForm{D, F <: BilinearForm{D}, WT <: GhostGridSpace,
    VT <: GhostGridSpace, MT <: AbstractMatrix, IV <: AbstractArray{Int, 1},
    CV <: AbstractVector}
    "the form on the padded spaces."
    padded::F
    trial_space::WT
    test_space::VT
    "the padded matrix, filled by every assembly."
    A_pad::MT
    "the stored position in `A_pad` of each contribution, grouped by reduced entry."
    src::IV
    "the contributions `dst[e]:(dst[e + 1] - 1)` that sum into stored reduced entry `e`."
    dst::IV
    "the closure weight `E_v[i, r] E_u[j, c]` of each contribution."
    coef::CV
    "the reduced pattern."
    colptr::IV
    rowval::IV
    "the terms `liftptr[i]:(liftptr[i + 1] - 1)` of row `i` of `A_pad d_u`."
    liftptr::IV
    "the stored position in `A_pad` and the column of each term of `A_pad d_u`."
    liftsrc::IV
    liftcol::IV
    "row replacement on the spaces, for `dirichlet`."
    rows::_RowsDirichlet
end

"""
    GhostLinearForm

A linear form on a [`GhostGridSpace`](@ref), built by `form(Vₕ, f)`: the form on the padded
space and the padded load vector it fills.
"""
struct GhostLinearForm{D, F <: LinearForm{D}, VT <: GhostGridSpace, BT <: AbstractVector}
    "the form on the padded space."
    padded::F
    test_space::VT
    "the padded load vector, filled by every assembly."
    F_pad::BT
    "row replacement on the space, for `dirichlet`."
    rows::_RowsDirichlet
end

"""
    GhostCompositeBilinearForm

A bilinear form on two composites of ghost spaces (`C₁ × C₂`), built by `form(Wₕ, Vₕ, f)`:
the fields of a [`GhostBilinearForm`](@ref) without its `rows`, the sides
`_CompositeClosure`s.
`dirichlet` replaces the rows of each selected component (ghost_composite.jl).
"""
struct GhostCompositeBilinearForm{D, F <: BilinearForm{D}, WT <: _CompositeClosure,
    VT <: _CompositeClosure, MT <: AbstractMatrix, IV <: AbstractArray{Int, 1},
    CV <: AbstractVector}
    padded::F
    trial_space::WT
    test_space::VT
    A_pad::MT
    src::IV
    dst::IV
    coef::CV
    colptr::IV
    rowval::IV
    liftptr::IV
    liftsrc::IV
    liftcol::IV
end

"""
    GhostCompositeLinearForm

A linear form on a composite of ghost spaces (`C₁ × C₂`), built by `form(Vₕ, f)`: the form
on the padded composite and the padded load vector it fills.
"""
struct GhostCompositeLinearForm{D, F <: LinearForm{D}, VT <: _CompositeClosure,
    BT <: AbstractVector}
    padded::F
    test_space::VT
    F_pad::BT
end

# The forms the assembly below takes: on scalar ghost spaces or on composites of them.
const _AnyGhostBilinearForm = Union{GhostBilinearForm, GhostCompositeBilinearForm}
const _AnyGhostLinearForm = Union{GhostLinearForm, GhostCompositeLinearForm}

trial_space(a::GhostBilinearForm) = a.trial_space
test_space(a::GhostBilinearForm) = a.test_space
test_space(l::GhostLinearForm) = l.test_space

function Base.show(io::IO, a::GhostBilinearForm{D}) where {D}
    print(io, "GhostBilinearForm{$(D)D}(", a.trial_space, " → ", a.test_space, ")")
    return nothing
end

function Base.show(io::IO, l::GhostLinearForm{D}) where {D}
    print(io, "GhostLinearForm{$(D)D}(", l.test_space, ")")
    return nothing
end

# --- Construction -------------------------------------------------------------------- #

function form(Wₕ::GhostGridSpace, Vₕ::GhostGridSpace, f)
    Vₕ.rows.trial_only && _throw_polygon_trial_as_test()
    _check_ghost_host(Wₕ)
    _check_ghost_host(Vₕ)
    _check_same_ghosts(Wₕ, Vₕ)
    a = form(padded_space(Wₕ), padded_space(Vₕ), f)
    # A rank-one term makes `a` a `RankOneForm`, whose parts are reduced one by one; behind
    # an inference barrier as in `form`, so a form without one still infers.
    if a isa RankOneForm
        (Wₕ.rows.polygon || Vₕ.rows.polygon) && _throw_polygon_rank_one()
        return Base.inferencebarrier(_ghost_rank_one_form)(Wₕ, Vₕ, a)  # rank_one.jl
    end
    return _ghost_bilinear_form(Wₕ, Vₕ, a)
end

# The ghost form of `a`, a form on the padded spaces of `Wₕ` and `Vₕ`.
function _ghost_bilinear_form(Wₕ::GhostGridSpace, Vₕ::GhostGridSpace, a::BilinearForm)
    return GhostBilinearForm(a, Wₕ, Vₕ, _ghost_maps(Wₕ, Vₕ, a)..., _RowsDirichlet(Vₕ, Wₕ))
end
function _ghost_bilinear_form(Wₕ::_CompositeClosure, Vₕ::_CompositeClosure,
        a::BilinearForm)
    (_has_polygon(Wₕ) || _has_polygon(Vₕ)) && _throw_polygon_composite()
    return GhostCompositeBilinearForm(a, Wₕ, Vₕ, _ghost_maps(Wₕ, Vₕ, a)...)
end

# The padded matrix of `a`, the map of its reduction and the map of `A_pad d_u`.
function _ghost_maps(Wₕ::_GhostSide, Vₕ::_GhostSide, a::BilinearForm)
    _check_ghost_eltype(_matrix_eltype(a, a.ast))
    A_pad = allocate_system_matrix(a)::SparseMatrixCSC  # `_check_ghost_host`
    moves = _exterior_moves(Wₕ, Vₕ, A_pad)
    src, dst, coef, colptr, rowval = _ghost_reduction_map(
        backend(Vₕ), A_pad, Vₕ.E, Wₕ.E, moves)
    liftptr, liftsrc, liftcol = _ghost_lift_map(backend(Vₕ), A_pad, _affine_support(Wₕ),
        moves)
    return (A_pad, src, dst, coef, colptr, rowval, liftptr, liftsrc, liftcol)
end

# The moves of the entries of `A_pad` at conflict columns, `(k, j, s)`: stored entry `k`, at
# a tested row `i` and a conflict column, enters the reduction as `s` times an entry at
# column `j` (polygon_space.jl). There are none on a space without a polygon trial side.
const _Moves = Tuple{Vector{Int}, Vector{Int}, Vector{Int}}
_exterior_moves(::_GhostSide, ::_GhostSide, ::SparseMatrixCSC) = (Int[], Int[], Int[])

# The reach guard and the moves on a space of polygon_spaces as the trial side. The test
# space there extends nothing, so every exterior node is unclosed. A tested row (one `E_v`
# reads) may read an exterior node only at a diagonal neighbour the (3.2) extension gives.
# An entry at an unclosed node, or at an extended one off the diagonals, is a stencil wider
# than the scheme's, and throws. An entry `a` of row `i` at a conflict node `i + (sx, sy)`
# moves onto the reader's own (3.2) row: `+a` at `i + (sx, 0)` and `i + (0, sy)`, the two
# Γ corners, and `−a` at `i`.
function _exterior_moves(Wₕ::GhostGridSpace, Vₕ::GhostGridSpace, A_pad::SparseMatrixCSC)
    rows = Wₕ.rows
    moves = (Int[], Int[], Int[])
    rows.polygon || return moves
    un, ext, cf = rows.unclosed, rows.extended, rows.conflict
    cp = SparseArrays.getcolptr(Vₕ.Et)
    car = CartesianIndices(npoints(mesh(Wₕ), Tuple))
    lin = LinearIndices(car)
    rv = rowvals(A_pad)
    for j in axes(A_pad, 2)
        (un[j] || ext[j]) || continue
        for k in nzrange(A_pad, j)
            i = rv[k]
            cp[i + 1] > cp[i] || continue          # a row no test function reads
            un[j] && _throw_polygon_reach(car[i], car[j])
            sx, sy = Tuple(car[j] - car[i])
            abs(sx) == 1 && abs(sy) == 1 || _throw_polygon_reach(car[i], car[j])
            cf[j] || continue
            I = car[i]
            for (q, s) in ((lin[I[1] + sx, I[2]], 1), (lin[I[1], I[2] + sy], 1), (i, -1))
                (un[q] || ext[q]) && _throw_polygon_reach(car[i], car[j])
                push!(moves[1], k), push!(moves[2], q), push!(moves[3], s)
            end
        end
    end
    return moves
end

@noinline function _throw_polygon_reach(I::CartesianIndex, J::CartesianIndex)
    throw(
        ArgumentError(
        "the form reads, in the equation of the interior node $(Tuple(I)), the exterior " *
        "node $(Tuple(J)), to which the trial space gives no value: only the trial space " *
        "of polygon_spaces extends exterior nodes, by (3.2), and only the diagonal " *
        "neighbours of interior nodes. Pose the form as form(U, V, f) with " *
        "`U, V = polygon_spaces(...)`, reading the 5-point and diagonal neighbours, as " *
        "Δₕ, Dcₓ(Dcᵧ(u)) and first differences do.",
    ),
    )
end

@noinline function _throw_polygon_trial_as_test()
    throw(
        ArgumentError(
        "the trial space of polygon_spaces is not a test space: its closure extends " *
        "the exterior nodes, and the equations are those of the interior nodes alone. " *
        "Pass the test space polygon_spaces returns, `U, V = polygon_spaces(...)`, " *
        "form(U, V, f) and form(V, f).",
    ),
    )
end

@noinline function _throw_polygon_rank_one()
    throw(ArgumentError("forms on polygon spaces do not support rank-one terms yet."))
end

function form(Vₕ::GhostGridSpace, f)
    Vₕ.rows.trial_only && _throw_polygon_trial_as_test()
    _check_ghost_host(Vₕ)
    return _ghost_linear_form(Vₕ, form(padded_space(Vₕ), f))
end

# The ghost form of `l`, a form on the padded space of `Vₕ`.
function _ghost_linear_form(Vₕ::GhostGridSpace, l::LinearForm)
    return GhostLinearForm(l, Vₕ, _ghost_load_buffer(Vₕ, l), _RowsDirichlet(Vₕ))
end
function _ghost_linear_form(Vₕ::_CompositeClosure, l::LinearForm)
    _has_polygon(Vₕ) && _throw_polygon_composite()
    return GhostCompositeLinearForm(l, Vₕ, _ghost_load_buffer(Vₕ, l))
end

# Whether a composite side has a component built by `polygon_spaces`: the reach guard and
# the substitution of conflict nodes (`_exterior_moves`) see scalar sides only.
_has_polygon(C::_CompositeClosure) = any(W -> W.rows.polygon, _ghost_components(C.space))

@noinline function _throw_polygon_composite()
    throw(
        ArgumentError(
        "polygon spaces are not supported in composites: the reduction checks the reach " *
        "of a form and substitutes conflict nodes on a scalar polygon space only. Pose " *
        "each field on its own pair from polygon_spaces.",
    ),
    )
end

# The padded load vector of `l`.
function _ghost_load_buffer(Vₕ::_GhostSide, l::LinearForm)
    T = _assembled_eltype(l.ast, padded_space(Vₕ))
    _check_ghost_eltype(T)
    return _backend_vector(backend(Vₕ), T, ndofs(padded_space(Vₕ)))
end

form(::GhostGridSpace, Vₕ, f) = _throw_ghost_mixed("test", Vₕ)
form(Wₕ, ::GhostGridSpace, f) = _throw_ghost_mixed("trial", Wₕ)

# `side` names the side without ghosts, `Xₕ` the space it holds.
@noinline function _throw_ghost_mixed(side::String, Xₕ)
    throw(
        ArgumentError(
        "a form takes ghost spaces on both sides or on neither: the reduction to the " *
        "degrees of freedom needs a closure on each, and the $side space, " *
        "$(summary(Xₕ)), has no ghosts. Build both spaces with " *
        "gridspace(Ωₕ; ghosts = ...).",
    ),
    )
end

# A composite space that mixes ghost and plain components, or nests a composite with a ghost
# component: only a flat composite of ghost spaces has a closure (ghost_composite.jl).
@noinline function _throw_ghost_composite()
    throw(
        ArgumentError(
        "a composite space with a ghost-space component takes ghost spaces only, side by " *
        "side: build it as C₁ × C₂ or C^Val(n) from spaces built by " *
        "gridspace(Ωₕ; ghosts = ...) or constrained, not mixed with plain spaces and not " *
        "nested in another CompositeGridSpace.",
    ),
    )
end

# Padded index `k` names the same point on both sides only when both pad the same facets:
# `(:ymin,)` and `(:ymax,)` pad grids of one size whose rows are shifted by one node.
function _check_same_ghosts(Wₕ::GhostGridSpace, Vₕ::GhostGridSpace)
    Gw, Gv = mesh(padded_space(Wₕ)), mesh(padded_space(Vₕ))
    Gw === Gv || Gw.active == Gv.active || _throw_ghost_sets(Gw.active, Gv.active)
    return nothing
end

@noinline function _throw_ghost_sets(aw, av)
    throw(
        ArgumentError(
        "the trial and test ghost spaces ghost different facets ($aw and $av as per-axis " *
        "(min, max) flags), so a padded index names different points on each side. " *
        "Forms coupling different ghost sets are not supported: build both spaces with " *
        "the same `ghosts`.",
    ),
    )
end

function _check_ghost_host(Wₕ::GhostGridSpace)
    P = padded_space(Wₕ)
    (locality(backend(P)) isa DeviceLocality || execution_policy(P) isa GpuPolicy) &&
        _throw_ghost_unsupported("a device backend")
    matrix_type(backend(P)) <: SparseMatrixCSC ||
        _throw_ghost_unsupported("the matrix type $(matrix_type(backend(P)))")
    return nothing
end

# Plain real and complex floating-point entries only: AD types are out of scope.
_check_ghost_eltype(::Type{<:Union{AbstractFloat, Complex{<:AbstractFloat}}}) = nothing
_check_ghost_eltype(T::Type) = _throw_ghost_unsupported("the element type $T (AD)")

@noinline function _throw_ghost_unsupported(what)
    throw(ArgumentError("forms on a ghost space do not support $what yet."))
end

# The map of `A = E_vᵀ A_pad E_u`: `A[r, c] = Σ E_v[i, r] A_pad[i, j] E_u[j, c]`, one
# contribution per stored `(i, j)` of `A_pad` and pair of stored entries in rows `i` of
# `E_v` and `j` of `E_u`. The reduced pattern holds every such `(r, c)`.
# Closures of different element types (a Float32 and a Float64 space) give weights of the
# promoted type. The map is built on the host and stored in the backend's vector type.
# The contributions are then grouped by reduced entry, in the order the walk met them (a
# stable sort), and `dst` is the pointer over the groups, so a band owns whole entries.
# Each move `(k, j, s)` (`_exterior_moves`) adds the contributions of stored entry `k` as
# if it sat, times `s`, at column `j` of its row.
function _ghost_reduction_map(be, A_pad::SparseMatrixCSC, Ev::SparseMatrixCSC,
        Eu::SparseMatrixCSC, moves::_Moves = (Int[], Int[], Int[]))
    Evt = sparse(transpose(Ev))
    Eut = sparse(transpose(Eu))
    src, rs, cs = Int[], Int[], Int[]
    coef = promote_type(eltype(Ev), eltype(Eu))[]
    rv, ev, eu = rowvals(A_pad), nonzeros(Evt), nonzeros(Eut)
    function contribute!(k, i, j, s)
        for p in nzrange(Evt, i), q in nzrange(Eut, j)

            push!(src, k)
            push!(rs, rowvals(Evt)[p])
            push!(cs, rowvals(Eut)[q])
            push!(coef, s * ev[p] * eu[q])
        end
    end
    for j in axes(A_pad, 2), k in nzrange(A_pad, j)

        contribute!(k, rv[k], j, true)
    end
    for (k, j, s) in zip(moves...)
        contribute!(k, rv[k], j, s)
    end
    S = sparse(rs, cs, trues(length(rs)), size(Ev, 2), size(Eu, 2), |)
    srv = rowvals(S)
    entry = similar(src)
    for t in eachindex(entry)
        range = nzrange(S, cs[t])
        entry[t] = first(range) - 1 + searchsortedfirst(view(srv, range), rs[t])
    end
    p = sortperm(entry)                 # stable
    return map(v -> copyto!(_backend_vector(be, eltype(v), length(v)), v),
        (src[p], _group_pointer(entry, length(srv)), coef[p], SparseArrays.getcolptr(S),
            srv))
end

# The pointer over `n` groups of `keys`, each in `1:n`: group `g` is the entries
# `ptr[g]:(ptr[g + 1] - 1)` of `keys` once sorted.
function _group_pointer(keys::Vector{Int}, n::Int)
    ptr = zeros(Int, n + 1)
    for g in keys
        ptr[g + 1] += 1
    end
    ptr[1] = 1
    return cumsum!(ptr, ptr)
end

# `A_pad d` by padded row, over the columns `j` where `d[j]` can be nonzero only (`support`,
# `_affine_support`): the stored position of each term and its column, in column order
# within a row, as `A_pad * d` adds them. A move `(k, j, 1)` adds stored entry `k` at
# column `j` after them; a move `(k, j, -1)` lands on its reader, a degree of freedom,
# where `d` is 0.
function _ghost_lift_map(be, A_pad::SparseMatrixCSC, support::AbstractVector{Bool},
        moves::_Moves = (Int[], Int[], Int[]))
    rv = rowvals(A_pad)
    rows, src, cols = Int[], Int[], Int[]
    for j in axes(A_pad, 2)
        support[j] || continue
        for k in nzrange(A_pad, j)
            push!(rows, rv[k])
            push!(src, k)
            push!(cols, j)
        end
    end
    for (k, j, s) in zip(moves...)
        (s == 1 && support[j]) || continue
        push!(rows, rv[k])
        push!(src, k)
        push!(cols, j)
    end
    p = sortperm(rows)                  # stable
    return map(v -> copyto!(_backend_vector(be, eltype(v), length(v)), v),
        (_group_pointer(rows, size(A_pad, 1)), src[p], cols[p]))
end

# --- Bilinear assembly --------------------------------------------------------------- #

function assemble(a::_AnyGhostBilinearForm; dirichlet = nothing,
        dirichlet_components = nothing)
    labels, _ = _normalize_dirichlet(dirichlet)
    _check_rows(a, labels, dirichlet_components)
    A = _assemble_ghost!(_reduced_matrix(a.A_pad, a), a)
    return _form_rows!(A, a, labels, dirichlet_components)
end

# A matrix of the reduced pattern, of the type of `A_pad`'s entries times the closure
# weights; `_assemble_ghost!` zeroes it.
function _reduced_matrix(A_pad::SparseMatrixCSC, a::_AnyGhostBilinearForm)
    T = promote_type(eltype(A_pad), eltype(a.coef))
    return SparseMatrixCSC(ndofs(a.test_space), ndofs(a.trial_space), copy(a.colptr),
        copy(a.rowval), similar(a.coef, T, length(a.rowval)))
end

function assemble!(A::AbstractMatrix, a::_AnyGhostBilinearForm; dirichlet = nothing,
        dirichlet_components = nothing)
    labels, _ = _normalize_dirichlet(dirichlet)
    _check_rows(a, labels, dirichlet_components)
    _check_reduced_pattern(A, a)
    _assemble_ghost!(A, a)
    return _form_rows!(A, a, labels, dirichlet_components)
end

@inline function _check_reduced_pattern(A::AbstractMatrix, a::_AnyGhostBilinearForm)
    A isa SparseMatrixCSC && size(A) == (ndofs(a.test_space), ndofs(a.trial_space)) &&
    SparseArrays.getcolptr(A) == a.colptr && rowvals(A) == a.rowval ||
        _throw_ghost_pattern(A, a)
    return nothing
end

function _assemble_ghost!(A::SparseMatrixCSC, a::_AnyGhostBilinearForm)
    assemble!(a.A_pad, a.padded)
    _run_bands!(_ghost_policy(a), _reduce_band!, nonzeros(A), nonzeros(a.A_pad),
        a.dst, a.src, a.coef)
    return A
end

# The policy of a ghost space, and of a ghost form read from its padded form, which the
# form's banded passes use: that form's type is concrete wherever the form's is, while a
# caller that infers the ghost space only partly (the workload's `constrained` spaces)
# leaves `test_space` abstract. Its policy then infers as `Any`, inference visits all three
# `_run_bands!` methods with the band's concrete arrays, and the `CpuPolyester` one records
# an edge into the `_batch_run_bands!` stub that `BramblePolyesterExt` invalidates
# (gpena/Bramble.jl#414).
@inline _ghost_policy(Wₕ::GhostGridSpace) = execution_policy(padded_space(Wₕ))
@inline _ghost_policy(a::Union{_AnyGhostBilinearForm, _AnyGhostLinearForm}) = execution_policy(
    test_space(a.padded))

# The reduced entries of band `b` of `nbands`: entry `e` sums its contributions in order.
@noinline function _reduce_band!(nz, pv, ptr, src, coef, nbands::Int, b::Int)
    T = eltype(nz)
    @inbounds for e in _band_range(1:length(nz), nbands, b)
        s = zero(T)
        for t in ptr[e]:(ptr[e + 1] - 1)
            s = convert(T, s + coef[t] * pv[src[t]])
        end
        nz[e] = s
    end
    return nothing
end

@noinline function _throw_ghost_pattern(A::AbstractMatrix, a::_AnyGhostBilinearForm)
    throw(
        ArgumentError(
        "assemble! on a ghost-space form fills a SparseMatrixCSC with the pattern of the " *
        "matrix assemble(a) returns, its size " *
        "$((ndofs(a.test_space), ndofs(a.trial_space))) and its stored entries; got a " *
        "$(typeof(A)) of size $(size(A)). Call A = assemble(a) once and refill that.",
    ),
    )
end

# --- Dirichlet by row replacement ---------------------------------------------------- #

# Route (a): the reduced unknowns are the physical nodes in mesh order exactly when there
# are as many as physical nodes and column `k` of `E` holds the 1 of physical node `k`; then
# the physical mesh's label masks index the reduced rows and columns unchanged.
function _rows_are_physical(G::GhostMesh, E::AbstractMatrix)
    phys = physical_indices(G)
    size(E, 2) == length(phys) || return false
    plin = LinearIndices(npoints(G, Tuple))
    for (k, J) in enumerate(phys)
        E[plin[J], k] == 1 || return false
    end
    return true
end

# The physical nodes of each facet, as masks over the physical mesh's linear order.
function _facet_masks(Ωₕ::AbstractMeshType, ::Val{D}) where {D}
    n = npoints(Ωₕ, Tuple)
    lin = LinearIndices(n)
    masks = BitVector[]
    for d in 1:D, s in 1:2

        mask = falses(length(lin))
        k = s == 1 ? 1 : n[d]
        for I in CartesianIndices(ntuple(e -> e == d ? (k:k) : (1:n[e]), Val(D)))
            mask[lin[I]] = true
        end
        push!(masks, mask)
    end
    return masks
end

# The nodes of a label on the physical mesh: its marker, or else the nodes of the facet it
# names (canonical or alias), which `constrained` and `inner_Γ` accept as labels too.
@inline function _label_mask(Ωₕ::AbstractMeshType, rows::_RowsDirichlet, label::Symbol)
    haskey(markers(Ωₕ), label) && return index_in_marker(Ωₕ, label)
    face = _find_face(Val(dim(Ωₕ)), label)
    face === nothing && return index_in_marker(Ωₕ, label)   # its error lists the labels
    d, s = face
    return rows.facets[2(d - 1) + s]
end

# Without `dirichlet_components`, the call of `matrix_free_operator`, which applies
# `E_vᵀ A_pad E_u` and so refuses a form that reads a conflict node of its trial space in
# a tested row: the substitution per reading equation lives in the reduction map alone.
function _check_rows(a::GhostBilinearForm, labels)
    _reads_conflict(a) && _throw_polygon_conflict("the matrix-free operator")
    return _check_rows(a.rows, (a.test_space, a.trial_space), labels)
end

# Whether `a` reads, in a tested row, a conflict node of its polygon trial space; the
# matrix-free operator and the contraction `a(u, v)` (ghost_contract.jl) refuse such a form.
_reads_conflict(::GhostCompositeBilinearForm) = false
function _reads_conflict(a::GhostBilinearForm)
    cf = a.rows.conflict
    any(cf) || return false
    cp, rv = SparseArrays.getcolptr(a.test_space.Et), rowvals(a.A_pad)
    for j in findall(cf), k in nzrange(a.A_pad, j)

        cp[rv[k] + 1] > cp[rv[k]] && return true
    end
    return false
end
_check_rows(l::GhostLinearForm, labels) = _check_rows(l.rows, (l.test_space,), labels)
# With the `dirichlet_components` of the call: a scalar space validates them with its rows
# (`_form_rows!`), a composite one reads them here (ghost_composite.jl).
_check_rows(a::GhostBilinearForm, labels, components) = _check_rows(
    a.rows, (a.test_space, a.trial_space), labels)
_check_rows(l::GhostLinearForm, labels, components) = _check_rows(l, labels)

@noinline function _throw_polygon_conflict(what::String)
    throw(
        ArgumentError(
        "$what applies E_vᵀ A_pad E_u, and this form reads a conflict node of its " *
        "polygon trial space (an exterior node two interior nodes extend by different " *
        "(3.2) rows), which only the assembled reduction substitutes per equation. " *
        "Assemble the matrix with assemble(a).",
    ),
    )
end
@inline _check_rows(::_RowsDirichlet, ::Tuple, ::Nothing) = nothing
@inline function _check_rows(rows::_RowsDirichlet, spaces::Tuple, labels::Tuple)
    isempty(labels) || rows.physical || _throw_rows_dirichlet(spaces, labels)
    return nothing
end

# Why the rows are not the physical nodes: a label's node fixed by `u => g` (the label
# given both ways), another node fixed by `u => g`, or ghosts left unknown.
@noinline function _throw_rows_dirichlet(spaces::Tuple, labels::Tuple)
    any(W -> W.rows.polygon, spaces) && throw(
        ArgumentError(
        "`dirichlet` replaces rows of the physical nodes, and the rows of a polygon space " *
        "are its interior nodes: polygon_spaces(Ωₕ, :L, g) already fixes the polygon's " *
        "boundary nodes to g. Drop `dirichlet`.",
    ),
    )
    Wₕ = first(spaces)
    for W in spaces                 # the first whose rows are not its physical nodes
        W.rows.physical || (Wₕ = W; break)
    end
    G = mesh(Wₕ)
    Ωₕ = physical_mesh(G)
    plin = LinearIndices(npoints(G, Tuple))
    held = falses(length(plin))
    held[rowvals(Wₕ.E)] .= true
    fixed = vec([!held[plin[J]] for J in physical_indices(G)])
    rows = _RowsDirichlet(false, _facet_masks(Ωₕ, Val(dim(Ωₕ))))
    for label in labels
        any(fixed .& _label_mask(Ωₕ, rows, label)) || continue
        throw(
            ArgumentError(
            "the Dirichlet condition on :$label is given both ways: `dirichlet` replaces " *
            "rows of nodes that a relation `u => g` in constrained already fixed. Give it " *
            "once, either as `u => g` or as `dirichlet`.",
        ),
        )
    end
    any(fixed) && throw(
        ArgumentError(
        "`dirichlet` replaces rows of the physical nodes, and this space has fixed some " *
        "nodes by a Dirichlet relation `u => g`, so its unknowns are not the physical " *
        "nodes. Give every Dirichlet condition as `u => g` in constrained, or every one " *
        "as `dirichlet`.",
    ),
    )
    throw(
        ArgumentError(
        "`dirichlet` replaces rows of the physical nodes, and $(Wₕ) keeps ghosts as " *
        "unknowns. Eliminate every ghost by a relation in constrained first, or give the " *
        "Dirichlet condition as `:F => u => g` in constrained.",
    ),
    )
end

# The row replacements of `gridspace(Ωₕ)` through the masks of the physical mesh, on
# reduced rows that are the physical nodes (checked by the caller).
function _rows_dirichlet!(A::AbstractMatrix, Ωₕ, rows::_RowsDirichlet, labels::Tuple)
    for lbl in labels
        _dirichlet_bc_indices!(A, _label_mask(Ωₕ, rows, lbl))
    end
    return A
end

# The values of `bcs` at the nodes of `labels`, written from entry `offset + 1` of `v` on (a
# component of a composite); unrolled over the conditions, as `_apply_conditions!` is, so a
# warm call allocates nothing.
@inline _rows_conditions!(::Tuple{}, v, Ωₕ, rows, labels, offset::Int = 0) = v
@inline function _rows_conditions!(ms::Tuple, v, Ωₕ, rows, labels, offset::Int = 0)
    m = first(ms)
    label(m) in labels &&
        _dirichlet_bc_indices!(v, Ωₕ, _label_mask(Ωₕ, rows, label(m)), identifier(m),
            offset)
    return _rows_conditions!(Base.tail(ms), v, Ωₕ, rows, labels, offset)
end

function _rows_symmetrize!(A::AbstractMatrix, F::AbstractVector, Ωₕ, rows::_RowsDirichlet,
        labels::Tuple)
    _check_symmetrize_square(A)
    for lbl in labels
        symmetrize!(A, F, _label_mask(Ωₕ, rows, lbl), 0)
    end
    return nothing
end

@inline _form_rows!(A, a, ::Nothing, components) = A
function _form_rows!(A::AbstractMatrix, a::GhostBilinearForm, labels::Tuple, components)
    _validate_scalar_components(components)
    return _rows_dirichlet!(A, physical_mesh(mesh(a.test_space)), a.rows, labels)
end

@inline _form_rows!(b, l, ::Nothing, bcs, components) = b
function _form_rows!(b::AbstractVector, l::GhostLinearForm, labels::Tuple, bcs, components)
    isempty(labels) && return b
    bcs === nothing && _throw_labels_without_conditions(labels)
    _validate_scalar_components(components)
    _check_labels_have_conditions(conditions(bcs), labels)
    return _rows_conditions!(conditions(bcs), b, physical_mesh(mesh(l.test_space)), l.rows,
        labels)
end

# `dirichlet_bc!` and `symmetrize!` by hand on a ghost space whose reduced rows are its
# physical nodes, as on `gridspace(Ωₕ)`.
function dirichlet_bc!(A::AbstractMatrix, Wₕ::GhostGridSpace, labels::Symbol...;
        components = nothing)
    _validate_scalar_components(components)
    _check_rows(Wₕ.rows, (Wₕ,), labels)
    return _rows_dirichlet!(A, physical_mesh(mesh(Wₕ)), Wₕ.rows, labels)
end

function dirichlet_bc!(v::AbstractVector, Wₕ::GhostGridSpace,
        bcs::Union{ConstraintMarkers, EvaluatedParametricDomainMarkers}, labels::Symbol...;
        components = nothing)
    _validate_scalar_components(components)
    _check_rows(Wₕ.rows, (Wₕ,), labels)
    _check_labels_have_conditions(conditions(bcs), labels)
    return _rows_conditions!(conditions(bcs), v, physical_mesh(mesh(Wₕ)), Wₕ.rows, labels)
end

function symmetrize!(A::AbstractMatrix, F::AbstractVector, Wₕ::GhostGridSpace,
        labels::Symbol...; components = nothing)
    _validate_scalar_components(components)
    _check_rows(Wₕ.rows, (Wₕ,), labels)
    return _rows_symmetrize!(A, F, physical_mesh(mesh(Wₕ)), Wₕ.rows, labels)
end

# --- Linear assembly ----------------------------------------------------------------- #

function assemble(l::_AnyGhostLinearForm; dirichlet = nothing,
        dirichlet_components = nothing)
    _is_time_dependent(l.test_space) && _throw_ghost_time("assemble(l)")
    b = similar(l.F_pad, ndofs(l.test_space))
    return assemble!(b, l; dirichlet = dirichlet, dirichlet_components = dirichlet_components)
end

function assemble!(b::AbstractVector, l::_AnyGhostLinearForm; dirichlet = nothing,
        dirichlet_components = nothing)
    labels, bcs = _normalize_dirichlet(dirichlet)
    _check_rows(l, labels, dirichlet_components)
    _is_time_dependent(l.test_space) && _throw_ghost_time("assemble!(b, l)")
    any(!iszero, l.test_space.d) && _throw_ghost_lift()
    _check_linear_length(b, l.test_space)
    assemble!(l.F_pad, l.padded)
    _closure_transpose_mul!(b, l, l.F_pad)
    return _form_rows!(b, l, labels, bcs, dirichlet_components)
end

@noinline function _throw_ghost_lift()
    throw(
        ArgumentError(
        "the closure of this ghost space has an affine part d ≠ 0, so its load is " *
        "Eᵀ(F − A d) and needs the bilinear form: assemble both with assemble(a, l).",
    ),
    )
end

function assemble(a::_AnyGhostBilinearForm, l::_AnyGhostLinearForm; dirichlet = nothing,
        dirichlet_components = nothing, symmetrize::Bool = false)
    labels, bcs = _normalize_dirichlet(dirichlet)
    _check_pair(a, l, labels, symmetrize, "assemble(a, l)", dirichlet_components)
    A = _reduced_matrix(a.A_pad, a)
    F = similar(l.F_pad, promote_type(eltype(l.F_pad), eltype(A)), ndofs(l.test_space))
    return _assemble_pair!(A, F, a, l, labels, bcs, dirichlet_components, symmetrize)
end

"""
    assemble!(A, F, a::GhostBilinearForm, l::GhostLinearForm;
        dirichlet = nothing, dirichlet_components = nothing, symmetrize = false)

Refill `A` and `F` with the matrix and the load that `A, F = assemble(a, l)` returns, and
return `A, F`. `A` keeps the pattern and `F` the element type that `assemble(a, l)` gave,
so call that once and refill its two outputs. The call allocates nothing warm.

Both forms are assembled together because, when the closure of the trial space has an
affine part `d ≠ 0`, the load is `E_vᵀ(F − A d)` and reads the matrix. `assemble!(b, l)`
alone refuses a test space with `d ≠ 0`, which is exact when both sides are one space: on a
pair whose trial space has data and whose test space has none, it gives `E_vᵀ F` without
the lift. `dirichlet`, `dirichlet_components` and `symmetrize` act as in `assemble(a, l)`.

It throws an `ArgumentError` when the trial space of `a` or the test space of `l` is
time-dependent (pose the problem with `semidiscretize`), for `symmetrize = true` without
`dirichlet`, for `A` that is not a `SparseMatrixCSC` with the reduced pattern of `a`, for a
`dirichlet` the rows of the space do not allow, and for an `F` whose element type cannot
hold the load's. A `DimensionMismatch` follows an `F` of the wrong length and forms whose
test spaces differ: not the same space, and closures `(E, d)` that are not equal.
"""
function assemble!(A::AbstractMatrix, F::AbstractVector, a::_AnyGhostBilinearForm,
        l::_AnyGhostLinearForm; dirichlet = nothing, dirichlet_components = nothing,
        symmetrize::Bool = false)
    labels, bcs = _normalize_dirichlet(dirichlet)
    _check_pair(a, l, labels, symmetrize, "assemble!(A, F, a, l)", dirichlet_components)
    _check_reduced_pattern(A, a)
    _check_linear_length(F, l.test_space)
    T = promote_type(eltype(l.F_pad), eltype(A))
    promote_type(eltype(F), T) === eltype(F) || _throw_pair_eltype(F, T)
    return _assemble_pair!(A, F, a, l, labels, bcs, dirichlet_components, symmetrize)
end

# The refusal of a load whose lift depends on time (there is no single `d`).
@noinline function _throw_ghost_time(call::AbstractString)
    throw(
        ArgumentError(
        "$call has no time, and the relation data of the space depend on time, so the " *
        "load's lift Eᵀ A d(t) does too. Pose the problem in time with semidiscretize, " *
        "or lift a solution at a time with lift(Cₕ, u, t); form and assemble(a) work as " *
        "they are.",
    ),
    )
end

function _check_pair(a::_AnyGhostBilinearForm, l::_AnyGhostLinearForm, labels,
        symmetrize::Bool, call::String = "assemble(a, l)", components = nothing)
    _is_time_dependent(a.trial_space) && _throw_ghost_time(call)
    _is_time_dependent(l.test_space) && _throw_ghost_time(call)
    symmetrize && labels === nothing && _throw_symmetrize_without_dirichlet()
    _check_rows(a, labels, components)
    _same_closure(a.test_space, l.test_space) || _throw_pair_spaces(a, l)
    return nothing
end

# Whether two test sides reduce alike: one space (a pointer compare, so a warm `assemble!`
# stays 0 B), or closures `(E, d)` equal entry by entry, as two spaces built alike give.
# Equal sizes are not enough: Dirichlet relations on different facets keep as many nodes,
# different ones. A composite side compares the space its `_CompositeClosure` was built
# from, since `form` builds a closure per call.
@inline function _same_closure(W::GhostGridSpace, V::GhostGridSpace)
    return W === V || _equal_closure(W, V)
end
@inline function _same_closure(W::_CompositeClosure, V::_CompositeClosure)
    return W.space === V.space || _equal_closure(W, V)
end
_same_closure(::_GhostSide, ::_GhostSide) = false   # a scalar side and a composite one
# Static closures only: a space with data in time stores `d(0)`, which a static space with
# zero data matches.
@noinline function _equal_closure(W, V)
    return !_is_time_dependent(W) && !_is_time_dependent(V) && W.E == V.E && W.d == V.d
end

@noinline function _throw_pair_spaces(a::_AnyGhostBilinearForm, l::_AnyGhostLinearForm)
    throw(
        DimensionMismatch(
        "the forms' test spaces differ: $(a.test_space) and $(l.test_space), whose " *
        "closures (E, d) are not equal. Build the linear form on the bilinear form's " *
        "test space, form(test_space(a), f).",
    ),
    )
end

@noinline function _throw_pair_eltype(F::AbstractVector, T::Type)
    throw(
        ArgumentError(
        "assemble!(A, F, a, l) fills a load of element type $T, and F holds " *
        "$(eltype(F)); call A, F = assemble(a, l) once and refill those.",
    ),
    )
end

function _assemble_pair!(A::SparseMatrixCSC, F::AbstractVector, a::_AnyGhostBilinearForm,
        l::_AnyGhostLinearForm, labels, bcs, components, symmetrize::Bool)
    _assemble_ghost!(A, a)
    _assemble_ghost_load!(F, a, l)
    # The load fills all of `F`, so it comes before `_form_rows!` writes the boundary rows;
    # its lift reads `A_pad`, which they leave alone.
    _form_rows!(A, a, labels, components)
    _form_rows!(F, l, labels, bcs, components)
    # `_check_pair` refused `symmetrize` without labels.
    symmetrize && labels isa Tuple && _form_symmetrize!(A, F, a, labels, components)
    return A, F
end

# `symmetrize` on the rows `_form_rows!` replaced (a composite's in ghost_composite.jl).
function _form_symmetrize!(A, F, a::GhostBilinearForm, labels::Tuple, components)
    return _rows_symmetrize!(A, F, physical_mesh(mesh(a.test_space)), a.rows, labels)
end

# `E_vᵀ (F_pad − A_pad d_u)`, `A_pad` as the last assembly of `a` left it, in one banded
# pass that forms each `w_i = F_pad[i] − (A_pad d_u)[i]` where a column of `E_v` reads it:
# no padded intermediate, so no buffer whose type would have to hold both forms' types,
# and `F_pad` stays the load. `d` is the trial space's `d`, or `d(t)` for `semidiscretize`.
function _assemble_ghost_load!(F::AbstractVector, a::_AnyGhostBilinearForm,
        l::_AnyGhostLinearForm, d::AbstractVector = a.trial_space.d)
    assemble!(l.F_pad, l.padded)
    isempty(a.liftsrc) && return _closure_transpose_mul!(F, l, l.F_pad)
    E = l.test_space.E
    z = zero(promote_type(eltype(l.F_pad), eltype(a.A_pad), eltype(d)))
    _run_bands!(_ghost_policy(l), _lifted_load_band!, F,
        SparseArrays.getcolptr(E), rowvals(E), nonzeros(E), l.F_pad, nonzeros(a.A_pad),
        a.liftptr, a.liftsrc, a.liftcol, d, z)
    return F
end

# Entries `r` of band `b`: `y[r] = Σ E[i, r] w_i` over column `r` of `E`, with
# `w_i = F[i] − Σ A_pad[i, j] d[j]` summed over the stored terms of row `i` in the type of
# `z`. A row read by several columns is summed again for each, in the same order.
@noinline function _lifted_load_band!(y, colptr, rowval, nz, F, pv, ptr, src, col, d, z,
        nbands::Int, b::Int)
    T = eltype(y)
    S = typeof(z)
    @inbounds for r in _band_range(1:length(y), nbands, b)
        acc = zero(T)
        for k in colptr[r]:(colptr[r + 1] - 1)
            i = rowval[k]
            w = convert(S, F[i])
            for t in ptr[i]:(ptr[i + 1] - 1)
                w = convert(S, w - pv[src[t]] * d[col[t]])
            end
            acc = convert(T, acc + nz[k] * w)
        end
        y[r] = acc
    end
    return nothing
end

# `y = Eᵀ x` for the closure `E` of the test space of `l`, entry `r` from column `r` of `E`.
function _closure_transpose_mul!(y::AbstractVector, l::_AnyGhostLinearForm,
        x::AbstractVector)
    E = l.test_space.E
    _run_bands!(_ghost_policy(l), _csc_columns_band!, y, SparseArrays.getcolptr(E),
        rowvals(E), nonzeros(E), x)
    return y
end

# --- Refused paths ------------------------------------------------------------------- #

function assemble_parallel!(::AbstractVecOrMat,
        ::Union{_AnyGhostBilinearForm, _AnyGhostLinearForm})
    _throw_ghost_assemble_parallel()
end

@noinline function _throw_ghost_assemble_parallel()
    throw(
        ArgumentError(
        "assemble_parallel! does not take a ghost-space form: call assemble! instead, " *
        "which runs in parallel under the policy of the mesh's backend.",
    ),
    )
end

# The AST of the form on the padded spaces.
resolve_form_ast(a::Union{_AnyGhostBilinearForm, _AnyGhostLinearForm}) = resolve_form_ast(
    a.padded)
