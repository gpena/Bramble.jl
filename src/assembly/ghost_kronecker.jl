# ghost_kronecker.jl: `kronecker_operator` on a form over ghost spaces (#781): the reduced
# matrix `E_vᵀ A_pad E_u` of `assemble(a)` as a `KroneckerLinearOperator`, from 1D factors.
#
# With one relation per facet along its normal (Dirichlet, Neumann, Robin with a constant
# coefficient), the closure `E` is the tensor product `E_D ⊗ … ⊗ E_1` of 1D closures, one
# per axis: `E_d` keeps the axis's unfixed physical points and writes each ghost of the axis
# as a combination of them, with the same coefficients at every node of its facet. The
# padded form is a sum of Kronecker products of padded 1D factors `F_d`, each assembled on
# the padded 1D ghost space of axis `d` (physical-only weights, as the padded space's are
# their product), so each term of the reduced matrix is `⊗_d E_{v,d}ᵀ F_d E_{u,d}`.
#
# The 1D closures are read from `E` itself, along one line per axis, and `E` is then
# checked against their Kronecker product, so no relation is parsed here. A closure that is
# not such a product is refused, naming `assemble`: a conormal `∂ₙ(u; K)` with off-diagonal
# `K` (its ghost reads tangential neighbours, and at a corner the other facet's ghost), a
# coefficient varying along the facet, or a ghost left as an unknown. The product differs
# from `E` at the absent padded points (beyond two ghosted facets), where `E` is 0: a term
# whose factors reach ghost rows (or columns) on two axes would read them, and is refused.
#
# The data `d` enter the load only, so they do not bear on the operator. A region
# restriction and a grid-function coefficient on the padded mesh have no ghost factors here
# and are refused.
#
# See also: `kronecker_operator`, `KroneckerLinearOperator`, `GhostBilinearForm`

function kronecker_operator(a::GhostBilinearForm{D}) where {D}
    D == 1 && _throw_not_separable_dim(D)
    _ghost_kron_polygon(a) && _throw_ghost_kron_polygon("kronecker_operator")
    Wu, Wv = trial_space(a), test_space(a)
    _ghost_kron_same_mesh(mesh(Wu), mesh(Wv))
    _kron_check_spaces(a.padded)
    G = mesh(Wu)
    Eu = _ghost_kron_closures(Wu)
    Ev = Wv === Wu ? Eu : _ghost_kron_closures(Wv)
    map(E -> size(E, 2), Eu) == map(E -> size(E, 2), Ev) || _throw_ghost_kron_rectangular(Wu, Wv)
    Ωp = _padded(G)
    spaces = ntuple(d -> _ghost_kron_axis_space(G, d), Val(D))
    mass = ntuple(Val(D)) do d
        M = Ev[d]' * Diagonal(weights(spaces[d], Innerh())) * Eu[d]
        Diagonal([M[i, i] for i in axes(M, 1)])
    end
    seen = ntuple(_ -> Any[], Val(D))
    T = eltype(mass[1])
    reads_coef = false
    terms = ()
    for (scales, term) in _kron_leaves(resolve_form_ast(a.padded), ())
        P = _ghost_kron_project(term, Ωp, spaces)
        P === nothing && _throw_ghost_not_separable_term(term, Ωp)
        reads_coef |= _kron_reads_coef(term)
        for (live, padded) in P
            _ghost_kron_reads_absent(padded, G) &&
                _throw_ghost_not_separable(Wu,
                    "a term reads the absent padded points (beyond two ghosted facets, as a " *
                    "mixed derivative does), where the separable closure is not 0")
            # `_kron_cached!` returns a factor of the type of the one it is given, which
            # the assertion tells inference, as `seen` holds factors of any type.
            factors = ntuple(Val(D)) do d
                F = _kron_factor(sparse(Ev[d]' * padded[d] * Eu[d]), mass[d])
                return _kron_cached!(seen[d], F)::typeof(F)
            end
            T = promote_type(T, map(eltype, factors)...)
            T = promote_type(T, typeof(_kron_coeff(T, (scales..., live...))))
            terms = (terms..., _kron_term((scales..., live...), factors))
        end
    end
    reads_coef && _warn_kron_coefficient()
    dims = map(E -> size(E, 2), Eu)
    policy = execution_policy(backend(Wu))
    return KroneckerLinearOperator{T, D, typeof(terms), typeof(policy), typeof(G)}(
        terms, dims, prod(dims), policy, G, _mesh_version(_walk_mesh(G)))
end

# The trial and test spaces may come from two `gridspace` calls, each with a `GhostMesh` of
# its own: the factors are built on the trial one, so both must pad one physical mesh at the
# same facets.
function _ghost_kron_same_mesh(Gu::GhostMesh, Gv::GhostMesh)
    Gu === Gv && return nothing
    physical_mesh(Gu) === physical_mesh(Gv) || _throw_ghost_kron_mesh_objects()
    Gu.active == Gv.active ||
        _throw_ghost_kron_facets(Gu.active, Gv.active)
    return nothing
end

# Equal meshes that are two objects are refused too: the operator records one mesh and its
# version, to throw after an in-place change of it.
@noinline function _throw_ghost_kron_mesh_objects()
    throw(
        ArgumentError(
        "kronecker_operator: the trial and test ghost spaces are built on different mesh " *
        "objects (even if they are equal point for point), and the operator is tied to " *
        "one mesh object, whose in-place changes it checks for. Build both spaces from " *
        "one mesh, or assemble the form instead, assemble(a).",
    ),
    )
end

@noinline function _throw_ghost_kron_facets(au, av)
    throw(
        ArgumentError(
        "kronecker_operator: the trial and test ghost spaces ghost different facets ($au " *
        "and $av as per-axis (min, max) flags), so the form is not separable into 1D " *
        "factors on one padded mesh. Build both with the same `ghosts`, or assemble the " *
        "form instead, assemble(a).",
    ),
    )
end

# A Dirichlet relation on one side only fixes points of one space and not the other: the
# reduced system is rectangular, and every factor of a `KroneckerLinearOperator` is square.
@noinline function _throw_ghost_kron_rectangular(Wu::GhostGridSpace, Wv::GhostGridSpace)
    throw(
        ArgumentError(
        "kronecker_operator: the trial space has $(ndofs(Wu)) degrees of freedom and the " *
        "test space $(ndofs(Wv)), as their Dirichlet relations fix different points. The " *
        "reduced system is rectangular, and a KroneckerLinearOperator is a sum of " *
        "Kronecker products of square 1D factors. Give both spaces the same Dirichlet " *
        "relations, or assemble the form instead, assemble(a).",
    ),
    )
end

# The padded 1D space of axis `d`: the padded space of the physical axis, ghosted on the
# sides `G` ghosts along `d` (on none, the axis itself), weighing the physical points only.
# One type for every axis, so the tuple of them is concrete. Built on a serial host
# backend, as `_kron_axis_space` builds the factors of a plain space.
function _ghost_kron_axis_space(G::GhostMesh, d::Int)
    m = physical_mesh(G)(d)
    execution_policy(backend(m)) isa CpuSerial ||
        (m = _rebackend(m, backend(eltype(m)), identity))
    sides = Tuple(s for (s, side) in ((:xmin, :min), (:xmax, :max)) if has_ghosts(G, d, side))
    Gd = GhostMesh(m; ghosts = sides)
    return _gridspace(Gd, _ghost_space_weights(Gd))
end

# `_kron_project` with the given axis spaces: the factors of `term` on the padded axes.
_ghost_kron_project(::LazyOp, ::Any, ::Any) = nothing

function _ghost_kron_project(term::BilinearProduct{D}, Ωp::MeshnD{D}, spaces) where {D}
    inners = _kron_inners(term.inner, Ωp)
    inners === nothing && return nothing
    ls = _kron_split(term.left_op)
    ls === nothing || (ls = _kron_coefs(ls, Ωp))
    ls === nothing && return nothing
    rs = _kron_split(term.right_op)
    rs === nothing || (rs = _kron_coefs(rs, Ωp))
    rs === nothing && return nothing
    (any(_kron_restricts, ls) || any(_kron_restricts, rs)) && return nothing
    return _kron_project_terms(ls, rs, inners, spaces)
end

# Whether the padded factors of one Kronecker term reach an absent padded point: some row,
# or some column, nonzero at a ghost index on two axes or more.
function _ghost_kron_reads_absent(factors::Tuple, G::GhostMesh)
    phys = physical_indices(G).indices
    ghost_rows = 0
    ghost_cols = 0
    for d in eachindex(factors)
        F = sparse(factors[d])
        r, c, v = SparseArrays.findnz(F)
        out(i) = !(i in phys[d])
        ghost_rows += any(k -> !iszero(v[k]) && out(r[k]), eachindex(v))
        ghost_cols += any(k -> !iszero(v[k]) && out(c[k]), eachindex(v))
    end
    return ghost_rows > 1 || ghost_cols > 1
end

"""
    _ghost_kron_closures(Wₕ::GhostGridSpace{D}) -> NTuple{D, SparseMatrixCSC}

The 1D closures `E_d` whose Kronecker product `E_D ⊗ … ⊗ E_1` is the closure `E` of `Wₕ` at
every padded point that is not absent; throws an `ArgumentError` if there are none. `E_d` is
`P_d × m_d`: `P_d` padded points, `m_d` physical ones that no Dirichlet relation fixes. The
degrees of freedom must be the product of those, in the mesh's linear order, so no ghost
is left as an unknown. Each ghost's row of `E_d` is read on one line, through the first
unfixed point of every other axis. Then `E` is checked against the product at every padded
point that is not absent, to `64 eps` times the larger of 1 and the largest entry of `E`.
"""
function _ghost_kron_closures(Wₕ::GhostGridSpace{D, T}) where {D, T}
    refuse() = _throw_ghost_not_separable(Wₕ, _GHOST_KRON_CLOSURE)
    G = mesh(Wₕ)
    E = Wₕ.E
    Et = Wₕ.Et
    n = size(E, 2)
    pd = npoints(G, Tuple)
    plin = LinearIndices(pd)
    phys = physical_indices(G)
    cp, rv, nz = SparseArrays.getcolptr(Et), rowvals(Et), nonzeros(Et)
    # The unfixed physical points: rows of `E` that are one degree of freedom, weight 1.
    units = Tuple{CartesianIndex{D}, Int}[]
    for P in phys
        i = plin[P]
        k = cp[i]:(cp[i + 1] - 1)
        if length(k) == 1 && isone(nz[first(k)])
            push!(units, (P, rv[first(k)]))
        elseif any(q -> !iszero(nz[q]), k)
            refuse()
        end
    end
    kept = ntuple(d -> sort!(unique!([P[d] for (P, _) in units])), Val(D))
    m = map(length, kept)
    (prod(m) == n == length(units) && n > 0) || refuse()
    loc = ntuple(d -> Dict(p => j for (j, p) in enumerate(kept[d])), Val(D))
    dof = LinearIndices(m)
    all(((P, j),) -> j == dof[ntuple(d -> loc[d][P[d]], Val(D))...], units) || refuse()
    Es = ntuple(Val(D)) do d
        rows = Int[]
        cols = Int[]
        vals = T[]
        for (j, p) in enumerate(kept[d])
            push!(rows, p)
            push!(cols, j)
            push!(vals, one(T))
        end
        for g in 1:pd[d]
            g in phys.indices[d] && continue
            Q = CartesianIndex(ntuple(e -> e == d ? g : kept[e][1], Val(D)))
            i = plin[Q]
            for q in cp[i]:(cp[i + 1] - 1)
                J = CartesianIndices(m)[rv[q]]
                all(e -> e == d || J[e] == 1, 1:D) || refuse()
                push!(rows, g)
                push!(cols, J[d])
                push!(vals, nz[q])
            end
        end
        sparse(rows, cols, vals, pd[d], m[d])
    end
    present = Diagonal(vec([!is_absent(G, J) for J in CartesianIndices(pd)]))
    Δ = E - present * foldl(kron, reverse(Es))
    R = real(T)
    tol = 64 * eps(R) * max(one(R), maximum(abs, nonzeros(E); init = zero(R)))
    all(x -> abs(x) <= tol, nonzeros(Δ)) || refuse()
    return Es
end

# Why a closure is not the product of 1D ones, and what makes it one.
const _GHOST_KRON_CLOSURE = "its closure is not separable (it is when every ghost is " *
                            "eliminated by one relation per facet along its normal: u => g, " *
                            "∂ₙ(u), α * u + β * ∂ₙ(u) with constant α and β, or ∂ₙ(u; K) " *
                            "with diagonal K; a conormal K with off-diagonal entries " *
                            "couples the axes at the corners)"

@noinline function _throw_ghost_not_separable(Wₕ::GhostGridSpace, what::String)
    throw(
        ArgumentError(
        "kronecker_operator on $(Wₕ): $what, so the reduced matrix is not a sum of " *
        "Kronecker products of 1D factors. Assemble the form instead, assemble(a), or " *
        "apply it with matrix_free_operator(a).",
    ),
    )
end

# A polygon space (#202) extends an exterior node by (3.2) from the interior nodes beside
# it, node by node along the polygon's edges, not by one relation per facet along an axis,
# so its closure has no 1D factors whatever the form. `fdm_*` (BrambleKroneckerExt) asks
# the same question.
_ghost_kron_polygon(a::GhostBilinearForm) = trial_space(a).rows.polygon ||
                                            test_space(a).rows.polygon

@noinline function _throw_ghost_kron_polygon(caller::String)
    throw(
        ArgumentError(
        "$caller on a form over polygon spaces: their closure extends each exterior " *
        "node from its interior neighbours along the polygon's edges, which no product " *
        "of 1D closures gives, so the reduced matrix has no Kronecker factors. Assemble " *
        "the form instead, assemble(a, l), and solve it with pde_solve, or precondition " *
        "it with ilu_preconditioner or amg_preconditioner.",
    ),
    )
end

# What in `term` has no factors on the padded axes: the node `_kron_refused_node` names,
# or, when every node projects, a region restriction, which the padded axes do not carry.
function _ghost_kron_refused_node(term::BilinearProduct, Ωp)
    projects(side) = (c = _kron_split(side); c !== nothing && _kron_coefs(c, Ωp) !== nothing)
    _kron_inners(term.inner, Ωp) !== nothing && projects(term.left_op) &&
        projects(term.right_op) &&
        return "a region restriction, which the padded axes have no factors for,"
    return _kron_refused_node(term, Ωp) * ", which has no projection onto the padded axes,"
end
function _ghost_kron_refused_node(term::LazyOp, Ωp)
    _kron_refused_node(term, Ωp) * ", which has " *
    "no projection onto the padded axes,"
end

@noinline function _throw_ghost_not_separable_term(term, Ωp)
    throw(
        ArgumentError(
        "kronecker_operator on a ghost space: the term $(nameof(typeof(term))) is not " *
        "separable into 1D factors: it holds $(_ghost_kron_refused_node(term, Ωp)) so the " *
        "reduced matrix is not a sum of Kronecker products. Assemble the form instead, " *
        "assemble(a), or apply it with matrix_free_operator(a).",
    ),
    )
end
