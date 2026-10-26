# polygon_space.jl: the trial and test spaces of the Ferreira and Grigorieff scheme on a
# polygon (`:L => polygon(V)`) or level-set curve (`:L => levelset(φ)`) marked on a 2D mesh,
# as two `GhostGridSpace`s over the mesh itself, `GhostMesh(Ωₕ; ghosts = ())`, so that
# forms assemble through the ghost-space reduction.
#
# The test closure selects the interior nodes with `d = 0` (the paper's W₀,H): the trial
# closure on the test side would add the rows of the extended nodes to the interior ones.
# Every other exterior node is 0 and must not be read (`_exterior_moves`,
# ghost_assembly.jl).
#
# See also: `GhostGridSpace`, `GhostMesh`, `_relation_closure`

"""
    polygon_spaces(Ωₕ::MeshnD{2}, label::Symbol, g) -> (U, W)

The trial space `U` and the test space `W` of the scheme of Ferreira and Grigorieff (2006)
on the region marked `label` on `Ωₕ`, a mesh of a domain with the marker
`label => polygon(vertices)` or `label => levelset(φ)`, with the Dirichlet datum `g(x)` on
the region's boundary nodes. For a level-set marker the region is the polygon that `mesh`
inscribes in the curve ``φ = 0`` (its vertices are the boundary nodes), and `g` is read at
those vertices, not projected onto the curve: a vertex lies off the curve by ``O(h^2)``.

Both are [`GhostGridSpace`](@ref)s over `Ωₕ`, whose degrees of freedom are the interior
nodes of the polygon (the label `label`), in the mesh's linear order. `lift(U, u)` is the
grid function on all of `Ωₕ`: `u` at the interior nodes, `g` at the boundary nodes
(`Symbol(label, :_boundary)`), the extension (3.2) of the paper at each exterior diagonal
neighbour `p = k + (sx, sy)` of an interior node `k`,
`u_p = u_{k+(sx,0)} + u_{k+(0,sy)} − u_k`, and 0 at every other node. `W` selects the
interior nodes.
Assemble with `form(U, W, f)` and `form(W, f)`:

```julia
U, W = Bramble.polygon_spaces(Ωₕ, :L, g)
A, F = assemble(form(U, W, (u, v) -> innerₕ(-1.0 * Δₕ(u), v)),
                form(W, v -> innerₕ(f, v)))
uₕ = Bramble.lift(U, A \\ F)
```

A form may read, in the equation of an interior node, its 5-point neighbours and its
diagonal neighbours (`Δₕ`, `Dcₓ(Dcᵧ(u))`, first differences): an exterior node beyond
them has no value and the form throws. An exterior node two interior nodes extend
differently (a V-notch) is substituted per reading equation in the reduction.

`Rₕ(U, f)` is the element of `U` holding `f` at the interior nodes, and [`normₕ`](@ref),
[`snorm₁ₕ`](@ref), [`norm₁ₕ`](@ref) and [`norminf`](@ref) of an element of `U` measure
the grid function on Ω̄ₕ that is the element at the interior nodes and 0 at the boundary
nodes, so that the error of a solve on the region is

```julia
e = element(U, A \\ F) - Rₕ(U, u)
normₕ(e), snorm₁ₕ(e), norminf(e)
```

`W` refuses `Rₕ` and the norms.

# Throws

  - `ArgumentError`: if `Ωₕ` has no polygon or level-set marker `label`; if the grid does
    not fit the region (a cell holding interior and exterior nodes other than across a diagonal
    between two boundary nodes, as after `change_points!` to lines the closure would not
    give); if the region has no interior node. `form(U, U, f)` and `form(U, f)` throw, as
    `U` is no test space, and so does `dirichlet =` in `assemble`: `g` fixes the boundary.
"""
function polygon_spaces(Ωₕ::MeshnD{2}, label::Symbol, g)
    labels = _region_marker_labels(label)
    m = markers(Ωₕ)
    all(L -> haskey(m, L), labels) || _throw_polygon_label(label)
    inn, Γ = BitVector(m[labels[1]]), BitVector(m[labels[2]])
    ext = .!(inn .| Γ)
    n = npoints(Ωₕ, Tuple)
    car, lin = CartesianIndices(n), LinearIndices(n)
    _check_polygon_fit(inn, Γ, ext, car, label)
    any(inn) || _throw_polygon_empty(label)

    # With no ghosts, the padded grid is the mesh: padded and mesh linear indices agree.
    G = GhostMesh(Ωₕ; ghosts = ())
    return _polygon_spaces(Ωₕ, _gridspace(G, _ghost_space_weights(G)), label, g, inn, Γ,
        ext, car, lin)
end

# The spaces on the padded space `P`, behind a function barrier so that its element type
# `T` is known.
function _polygon_spaces(Ωₕ, P::ScalarGridSpace{2, T}, label::Symbol, g, inn::BitVector,
        Γ::BitVector, ext::BitVector, car, lin) where {T}
    be = backend(Ωₕ)
    N = length(lin)
    keep = findall(inn)
    gv = zeros(T, N)
    for k in findall(Γ)
        gv[k] = g(point(Ωₕ, car[k]))
    end

    # The interior readers of each exterior node, through the diagonals.
    nread, reader = zeros(Int, N), zeros(Int, N)
    for k in keep, sx in (-1, 1), sy in (-1, 1)
        p = lin[car[k] + CartesianIndex(sx, sy)]
        ext[p] || continue
        nread[p] += 1
        reader[p] = k
    end
    gpts = findall(==(1), nread)
    Ri, Rj, Rv = Int[], Int[], T[]
    for (r, p) in enumerate(gpts)
        K = car[reader[p]]
        sx, sy = Tuple(car[p] - K)
        for (q, v) in ((p, 1), (lin[K[1] + sx, K[2]], -1), (lin[K[1], K[2] + sy], -1),
            (reader[p], 1))
            push!(Ri, r), push!(Rj, q), push!(Rv, v)
        end
    end
    R = sparse(Ri, Rj, Rv, length(gpts), N)
    E, d, _ = _relation_closure(gpts, fill(label, length(gpts)), R, zeros(T, length(gpts)),
        keep, Γ, gv)

    extended = nread .> 0
    conflict = nread .> 1
    U0 = GhostGridSpace(P, _closure_matrix(be, T, E), _closure_data(be, T, d))
    rows = _RowsDirichlet(false, U0.rows.facets, true, true, ext .& .!extended, extended,
        conflict)
    U = GhostGridSpace(U0.padded, U0.E, U0.d, U0.Et, rows, U0.td, U0.gu, U0.grid_function)

    Ev = sparse(keep, 1:length(keep), fill(one(T), length(keep)), N, length(keep))
    V0 = GhostGridSpace(P, _closure_matrix(be, T, Ev), _closure_data(be, T, zeros(T, N)))
    # As a trial side, `V` gives every exterior node 0: the reach guard refuses any read.
    vrows = _RowsDirichlet(false, V0.rows.facets, true, false, ext, falses(N), falses(N))
    V = GhostGridSpace(V0.padded, V0.E, V0.d, V0.Et, vrows, V0.td, V0.gu, V0.grid_function)
    return U, V
end

# Whether `Wₕ` is the trial space of `polygon_spaces`: its degrees of freedom are the
# interior nodes, and `Rₕ` and the discrete norms act on them. The test space and the
# ghost spaces of `gridspace(Ωₕ; ghosts)` are not.
@inline _is_polygon_trial(Wₕ::GhostGridSpace) = Wₕ.rows.polygon && Wₕ.rows.trial_only

# The mesh linear index of each degree of freedom of a polygon trial space, its interior
# node: outside the exterior, the closure's row of an interior node is that degree of
# freedom alone and the row of a boundary node is empty. With no ghosts, padded and mesh
# linear indices agree.
function _polygon_interior(U::GhostGridSpace)
    Et = U.Et
    ext = U.rows.unclosed .| U.rows.extended
    cp, rv = SparseArrays.getcolptr(Et), rowvals(Et)
    idx = Vector{Int}(undef, size(Et, 1))
    for p in eachindex(ext)
        ext[p] || cp[p + 1] == cp[p] || (idx[rv[cp[p]]] = p)
    end
    return idx
end

# `Rₕ` on a polygon trial space: `f` at each interior node, the degrees of freedom.
function _polygon_restrict!(uₕ::VectorElement{<:GhostGridSpace}, f)
    Ωₕ = physical_mesh(mesh(uₕ.space))
    car = CartesianIndices(npoints(Ωₕ, Tuple))
    u = parent(uₕ)
    for (k, p) in enumerate(_polygon_interior(uₕ.space))
        u[k] = f(point(Ωₕ, car[p]))
    end
    return uₕ
end

# The element type of `Rₕ(U, f)`: the backend's, promoted with that of `f` at the first
# interior node, as `Rₕ` on a plain space samples `f` at one point.
function _polygon_restriction_eltype(U::GhostGridSpace, f)
    Ωₕ = physical_mesh(mesh(U))
    p = CartesianIndices(npoints(Ωₕ, Tuple))[first(_polygon_interior(U))]
    return promote_type(eltype(backend(U)), _restricted_value_type(f, point(Ωₕ, p)))
end

# The grid function on Ω̄ₕ of an element of a polygon trial space, on a plain space of the
# mesh: the element at the interior nodes and 0 at every other node, the boundary nodes
# included. The norms read it: no grid edge joins an interior node to an exterior one (the
# grid fits the region), so the box sums are those over Ω̄ₕ. The padded space's weights are
# read first, so a moved mesh throws as on any ghost space.
function _polygon_element(uₕ::VectorElement{<:GhostGridSpace})
    U = uₕ.space
    weights(U.padded)   # throws once the mesh has moved
    Ωₕ = physical_mesh(mesh(U))
    u = parent(uₕ)
    v = fill!(similar(u, npoints(Ωₕ)), zero(eltype(u)))
    for (k, p) in enumerate(_polygon_interior(U))
        v[p] = u[k]
    end
    return VectorElement(v, gridspace(Ωₕ))
end

@noinline function _throw_polygon_restriction_markers(op::Symbol)
    throw(
        ArgumentError(
        "$op on a polygon trial space takes no `markers`: its degrees of freedom are the " *
        "interior nodes of the region, and $op holds f at each of them. Restrict on " *
        "`gridspace(Ωₕ)` for the values at marked nodes of the mesh.",
    ),
    )
end

# The grid fits the polygon (the paper's (Geom)) when every cell holding an interior and an
# exterior node holds them at opposite corners, the other two on the boundary: the boundary
# crosses the cell along that diagonal and nowhere else.
function _check_polygon_fit(inn::BitVector, Γ::BitVector, ext::BitVector,
        car::CartesianIndices{2}, label::Symbol)
    lin = LinearIndices(car)
    n = size(car)
    across(p, q, r, s) = (inn[p] & ext[q] | ext[p] & inn[q]) & Γ[r] & Γ[s]
    for j in 1:(n[2] - 1), i in 1:(n[1] - 1)

        a, b, c, d = lin[i, j], lin[i + 1, j], lin[i, j + 1], lin[i + 1, j + 1]
        (inn[a] | inn[b] | inn[c] | inn[d]) &&
        (ext[a] | ext[b] | ext[c] | ext[d]) || continue
        across(a, d, b, c) || across(b, c, a, d) || _throw_polygon_unfitted(label, (i, j))
    end
    return nothing
end

@noinline function _throw_polygon_label(label::Symbol)
    throw(
        ArgumentError(
        "polygon_spaces needs the labels :$label, :$(label)_boundary and " *
        ":$(label)_exterior of a polygon marker `:$label => polygon(vertices)` or a " *
        "level-set marker `:$label => levelset(φ)` on the mesh; build the mesh from a " *
        "domain that holds one.",
    ),
    )
end

@noinline function _throw_polygon_unfitted(label::Symbol, cell::NTuple{2, Int})
    throw(
        ArgumentError(
        "the grid does not fit the region :$label: the cell at node $cell holds " *
        "interior and exterior nodes other than across a diagonal between two boundary " *
        "nodes, so " *
        "the region's boundary crosses a grid line off the nodes. Build the mesh with " *
        "mesh(Ω, npts, unif), which fits the grid lines to the polygon or level-set " *
        "curve, or move the points to lines that fit gives.",
    ),
    )
end

@noinline function _throw_polygon_empty(label::Symbol)
    throw(
        ArgumentError(
        "the region :$label has no interior node on this grid, so its spaces have no " *
        "degree of freedom; refine the grid.",
    ),
    )
end
