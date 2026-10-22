# levelset_space.jl: the trial and test spaces of the Shortley-Weller scheme on a level set
# marked on a 2D mesh (`:L => levelset(φ)`), as two `GhostGridSpace`s over the mesh itself,
# `GhostMesh(Ωₕ; ghosts = ())`, so that forms assemble through the ghost-space reduction.
#
# The trial closure keeps the interior nodes, fixes the boundary nodes to `g` and gives every
# exterior node an empty row of `E` and `d = 0`. An exterior node `p` that an interior
# 5-point neighbour `k` reads has no single value: in the equation of `k` it takes the value
# at `p` of the quadratic along their grid line through the node behind `k` (or, when that
# node is exterior, the crossing on that side with its datum), `k`, and the crossing between
# `k` and `p` with its datum (`_ArmRules`, ghost_gridspace.jl). The reduction moves each read
# onto that value (`_exterior_moves`, ghost_assembly.jl). On a 3-point difference along the
# line, as in `Δₕ`, that is the Shortley-Weller row with the arm shortened to its crossing,
# since a 3-point second difference is exact on quadratics.
#
# See also: `polygon_spaces`, `GhostGridSpace`, `_ArmRules`

"""
    levelset_spaces(Ωₕ::MeshnD{2}, Ω::Domain, label::Symbol, g) -> (U, W)

The trial space `U` and the test space `W` of the Shortley-Weller scheme on the level set
marked `label` in `Ω`, `label => levelset(φ)`, on `Ωₕ`, a mesh of `Ω`, with the Dirichlet
datum `g(x)` on the curve `φ = 0`.

Both are [`GhostGridSpace`](@ref)s over `Ωₕ`, whose degrees of freedom are the interior
nodes (the label `label`), in the mesh's linear order. `lift(U, u)` is the grid function on
all of `Ωₕ`: `u` at the interior nodes, `g` at the boundary nodes
(`Symbol(label, :_boundary)`) and 0 at the exterior ones. In the equation of an interior
node `k`, an exterior 5-point neighbour `p` takes the value at `p` of the quadratic along
their grid line through the node behind `k`, `k`, and the crossing of `φ = 0` between `k`
and `p`, where it is `g`; when the node behind `k` is exterior too, the crossing on that
side replaces it. Each equation reads its own value, so an exterior node read by two
interior nodes along the two axes has two values. On `-Δₕ` this is the Shortley-Weller
scheme, each arm the curve cuts shortened to its crossing. `W` selects the interior nodes.
Assemble with `form(U, W, f)` and `form(W, f)`:

```julia
U, W = Bramble.levelset_spaces(Ωₕ, Ω, :D, g)
A, F = assemble(form(U, W, (u, v) -> innerₕ(-1.0 * Δₕ(u), v)),
                form(W, v -> innerₕ(f, v)))
uₕ = Bramble.lift(U, A \\ F)
```

A form may read, in the equation of an interior node, its 5-point neighbours only (`Δₕ`,
`D̃ₓ(a * D₋ₓ(u))`, first and central differences, reaction terms): a diagonal read
(`Dcₓ(Dcᵧ(u))`) or one two nodes away (`D₋ₓ(D₋ₓ(u))`) throws.

# Throws

  - `ArgumentError`: if `Ω` has no level-set marker `label`; if the labels of `Ωₕ` differ
    from those `φ` gives its nodes (a mesh not built from `Ω`); if the curve crosses all
    four edges of a cell, which the grid cannot resolve (refine it); if an interior node
    lies on the grid's boundary; if there is no interior node. `form(U, U, f)` and
    `form(U, f)` throw, as `U` is no test space, and so does `dirichlet =` in `assemble`:
    `g` fixes the boundary.
"""
function levelset_spaces(Ωₕ::MeshnD{2}, Ω::Domain, label::Symbol, g)
    return _levelset_spaces(Ωₕ, regions(markers(Ω)), label, g)
end

# The marker `name` among the regions of the domain, unrolled over the tuple so that the
# type of each marker is known where it is matched.
_levelset_spaces(Ωₕ, ::Tuple{}, name::Symbol, g) = _throw_levelset_marker(name)
function _levelset_spaces(Ωₕ, regs::Tuple, name::Symbol, g)
    m = first(regs)
    m isa _LevelSetMarker && label(m) === name && return _levelset_spaces(Ωₕ, m, g)
    return _levelset_spaces(Ωₕ, Base.tail(regs), name, g)
end

function _levelset_spaces(Ωₕ, marker::_LevelSetMarker, g)
    name = label(marker)
    labels = _polygon_labels(marker)
    m = markers(Ωₕ)
    all(l -> haskey(m, l), labels) || _throw_levelset_mismatch(name)
    n = npoints(Ωₕ, Tuple)
    car, lin = CartesianIndices(n), LinearIndices(n)
    fresh = MeshMarkers(l => falses(length(lin)) for l in labels)
    pts = host_points(Ωₕ)
    _set_region!(fresh, marker, Ωₕ, pts, lin)
    all(l -> fresh[l] == m[l], labels) || _throw_levelset_mismatch(name)
    inn, Γ = BitVector(m[labels[1]]), BitVector(m[labels[2]])
    ext = .!(inn .| Γ)
    L = identifier(marker)
    _check_levelset_cells([L((x, y)) for x in pts[1], y in pts[2]], name)
    for k in findall(inn)
        i, j = Tuple(car[k])
        1 < i < n[1] && 1 < j < n[2] || _throw_levelset_outer(name, (i, j))
    end
    any(inn) || _throw_levelset_empty(name)

    # With no ghosts, the padded grid is the mesh: padded and mesh linear indices agree.
    G = GhostMesh(Ωₕ; ghosts = ())
    return _levelset_spaces(Ωₕ, _gridspace(G, _ghost_space_weights(G)), L, g, inn, Γ, ext,
        car, lin)
end

# The spaces on the padded space `P`, behind a function barrier so that its element type
# `T` is known.
function _levelset_spaces(Ωₕ, P::ScalarGridSpace{2, T}, L::LevelSet, g, inn::BitVector,
        Γ::BitVector, ext::BitVector, car, lin) where {T}
    be = backend(Ωₕ)
    N = length(lin)
    keep = findall(inn)
    E = sparse(keep, 1:length(keep), fill(one(T), length(keep)), N, length(keep))
    gv = zeros(T, N)
    for k in findall(Γ)
        gv[k] = g(point(Ωₕ, car[k]))
    end
    U0 = GhostGridSpace(P, _closure_matrix(be, T, E), _closure_data(be, T, gv))
    arms, read = _arm_rules(host_points(Ωₕ), L, g, inn, ext, car, lin, T)
    rows = _RowsDirichlet(false, U0.rows.facets, true, true, ext, falses(N), read, arms)
    U = GhostGridSpace(U0.padded, U0.E, U0.d, U0.Et, rows, U0.td, U0.gu, U0.grid_function)

    V0 = GhostGridSpace(P, _closure_matrix(be, T, E), _closure_data(be, T, zeros(T, N)))
    # As a trial side, `V` gives every exterior node 0 and has no rule: the reach guard
    # refuses any read.
    norules = _ArmRules(ones(Int, N + 1), Int[], Int[], T[])
    vrows = _RowsDirichlet(false, V0.rows.facets, true, false, ext, falses(N), falses(N),
        norules)
    V = GhostGridSpace(V0.padded, V0.E, V0.d, V0.Et, vrows, V0.td, V0.gu, V0.grid_function)
    return U, V
end

# The rule of each exterior node `p = k ± e_a` in the equation of each interior node `k`
# reading it (`_ArmRules`), grouped by `p`, and the mask of the nodes read. Positions are
# taken along the axis `a` from `x_k`: `zp` at `p`, `c` at the crossing towards `p`, `b` at
# the node behind `k` or its crossing.
function _arm_rules(pts, L::LevelSet, g, inn::BitVector, ext::BitVector, car, lin,
        ::Type{T}) where {T}
    node(I) = (pts[1][I[1]], pts[2][I[2]])
    ps, reader, other, coef = Int[], Int[], Int[], T[]
    for k in findall(inn), a in 1:2, s in (-1, 1)
        K = car[k]
        e = CartesianIndex(a == 1 ? (s, 0) : (0, s))
        p = lin[K + e]
        ext[p] || continue
        o = lin[K - e]
        xk = node(K)
        zp = T(node(K + e)[a] - xk[a])
        tp, gp = _arm_crossing(L, g, xk, node(K + e), T)
        c = tp * zp
        if ext[o]
            to, go = _arm_crossing(L, g, xk, node(K - e), T)
            wo, wk, wc = _quadratic_weights(to * T(node(K - e)[a] - xk[a]), c, zp, tp)
            push!(other, 0)
            push!(coef, wk, zero(T), wc * gp + wo * go)
        else
            wo, wk, wc = _quadratic_weights(T(node(K - e)[a] - xk[a]), c, zp, tp)
            push!(other, o)
            push!(coef, wk, wo, wc * gp)
        end
        push!(ps, p)
        push!(reader, k)
    end
    perm = sortperm(ps)                 # stable
    read = falses(length(lin))
    read[ps] .= true
    coef3 = vec(reshape(coef, 3, :)[:, perm])
    return _ArmRules(_group_pointer(ps, length(lin)), reader[perm], other[perm], coef3), read
end

# The parameter `t` of the crossing of `φ = 0` on the edge `x → y` and the datum there.
function _arm_crossing(L::LevelSet, g, x, y, ::Type{T}) where {T}
    t = T(_crossing(L, x, y))
    return t, T(g((x[1] + t * (y[1] - x[1]), x[2] + t * (y[2] - x[2]))))
end

# The Lagrange weights at `z` of the quadratic through the positions `b`, 0 and `c = t z`, in
# that order; `z − c = z (1 − t)`.
function _quadratic_weights(b::T, c::T, z::T, t::T) where {T}
    zc = z * (1 - t)
    return z * zc / (b * (b - c)), (z - b) * zc / (b * c), z * (z - b) / (c * (c - b))
end

# A cell whose four edges each join nodes where `φ` has opposite signs: the curve enters and
# leaves it twice (a saddle of `φ`), and the crossings on its edges do not say how.
function _check_levelset_cells(vals::AbstractMatrix, name::Symbol)
    cut(v, w) = v * w < 0
    for j in 1:(size(vals, 2) - 1), i in 1:(size(vals, 1) - 1)

        a, b, c, d = vals[i, j], vals[i + 1, j], vals[i, j + 1], vals[i + 1, j + 1]
        cut(a, b) && cut(c, d) && cut(a, c) && cut(b, d) &&
            _throw_levelset_thin(name, (i, j))
    end
    return nothing
end

@noinline function _throw_levelset_marker(name::Symbol)
    throw(
        ArgumentError(
        "levelset_spaces needs a level-set marker `:$name => levelset(φ)` in the domain; " *
        "the domain has none labelled :$name. Pass the domain the mesh was built from.",
    ),
    )
end

@noinline function _throw_levelset_mismatch(name::Symbol)
    throw(
        ArgumentError(
        "the labels :$name, :$(name)_boundary and :$(name)_exterior of the mesh differ " *
        "from those the level set :$name of the domain gives its nodes, so the mesh was " *
        "not built from this domain. Build it with mesh(Ω, npts, unif), or move its " *
        "points with change_points!(Ωₕ, markers(Ω), pts).",
    ),
    )
end

@noinline function _throw_levelset_thin(name::Symbol, cell::NTuple{2, Int})
    throw(
        ArgumentError(
        "the level set :$name crosses all four edges of the cell at node $cell, so the " *
        "curve passes through it twice and the grid cannot tell how (a saddle of φ, or " *
        "a part of the domain thinner than the cell). Refine the grid.",
    ),
    )
end

@noinline function _throw_levelset_outer(name::Symbol, node::NTuple{2, Int})
    throw(
        ArgumentError(
        "the interior node $node of the level set :$name lies on the boundary of the " *
        "grid, where the scheme has no neighbour beyond it. Enlarge the box so that the " *
        "curve φ = 0 stays inside the grid.",
    ),
    )
end

@noinline function _throw_levelset_empty(name::Symbol)
    throw(
        ArgumentError(
        "the level set :$name has no interior node on this grid, so its spaces have no " *
        "degree of freedom; refine the grid.",
    ),
    )
end
