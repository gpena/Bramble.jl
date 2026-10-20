# ghost_gauge.jl: `ghost_kernel(A, Wₕ)`, the sparse basis of the ghost-only kernel of a matrix
# assembled on a ghost space, which `pde_solve(A, F; gauge)` imposes as a constraint.
#
# The kernel is found node by node: the ghosts that remain unknowns at one physical node
# (one per ghosted facet through it, so two at a 2D corner and up to three at a 3D corner)
# give the columns of `A` at those ghosts, and the null vectors of that small dense block,
# restricted to its nonzero rows, are the kernel's columns at the node. No global null
# space is formed: it would be dense, O(N³), and mix physical modes in.
#
# See also: `GhostGridSpace`, `constrained`

"""
    ghost_kernel(A, Wₕ::GhostGridSpace; rtol = 64 eps) -> SparseMatrixCSC

A sparse `ndofs(Wₕ) × k` matrix whose columns span the ghost-only kernel of `A`, a matrix
assembled on the ghost space `Wₕ` (built by `gridspace(Ωₕ; ghosts = ...)` or
[`constrained`](@ref)): the vectors `z` with `A z = 0` that are zero at every physical
degree of freedom.

The kernel is computed one physical node at a time. The ghosts at a node that remain
unknowns, one for each ghosted facet through it without a relation, give `m` columns of
`A`; restricted to their nonzero rows they form a small dense block, and each of its right
singular vectors whose singular value is at most `rtol` times the larger of the block's
largest one and the largest column 2-norm of `A` becomes a column of the result, with at
most `m` entries. The second scale makes a ghost column that is zero only to rounding (a
conormal `K` that cancels a coefficient) a kernel vector, as the dense spectrum calls it.
The columns have unit 2-norm and their first nonzero entry is real and positive; the nodes
come in the mesh's linear order, so the result depends on `A` and `Wₕ` only.

For a form that reads the ghosts at a corner through `Δₕ` alone, such as
`innerₕ(Δₕ(u), Δₕ(v)) + innerₕ(u, v)`, the corner's ghost columns are parallel, so each 2D
corner gives one column and each 3D corner two, and each node inside a 3D edge one: `k = 4`
in 2D, `k = 4 Σₑ (nₑ - 2) + 16` in 3D, with `nₑ` the points along axis `e`. With a single
ghosted facet no node carries two ghosts, and `k = 0` for that form. A form that never
reads a ghost leaves it in the kernel, one column each.

A kernel vector spread over the ghosts of several nodes is not found. Such vectors arise
when the form differences the ghosts along their own facet: with `ghosts = (:xmin,)`,
`innerₕ(∇ₕ(Δₕ(u))[2], v)` annihilates the vector that is constant on the `:xmin` ghosts,
and `ghost_kernel` returns no column for it. A gauged solve with such a `Z` is not
refused and usually factors: the physical part is still right, and the components left out
are arbitrary.

`ghost_kernel` is public, not exported: write `Bramble.ghost_kernel`.

Throws an `ArgumentError` when `A` is not an `AbstractMatrix` of size `ndofs(Wₕ) × ndofs(Wₕ)`,
or when `Wₕ` is not a [`GhostGridSpace`](@ref). A space whose relations eliminate every
ghost has no ghost-only kernel, and the result has no columns.

# Examples

```julia
Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (6, 7), (false, false))
Wₕ = gridspace(Ωₕ; ghosts = :boundary)
A = assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(Δₕ(u), Δₕ(v)) + innerₕ(u, v)))
Z = Bramble.ghost_kernel(A, Wₕ)    # 68 × 4, two entries per column, one column per corner
norm(A * Z)                        # rounding
```
"""
function ghost_kernel(A::AbstractMatrix, Wₕ::GhostGridSpace;
        rtol::Real = 64 * eps(real(float(eltype(A)))))
    n = ndofs(Wₕ)
    size(A) == (n, n) || _throw_ghost_kernel_size(A, Wₕ)
    S = A isa SparseMatrixCSC ? A : sparse(A)
    Tv = float(eltype(S))
    nodes, dofs = _ghost_dofs_by_node(Wₕ)
    Zi, Zj, Zv = Int[], Int[], Tv[]
    scale = _max_column_norm(S)
    k = 0
    lo = 1
    while lo <= length(nodes)
        hi = lo
        while hi < length(nodes) && nodes[hi + 1] == nodes[lo]
            hi += 1
        end
        k = _node_kernel!(Zi, Zj, Zv, k, S, view(dofs, lo:hi), rtol * scale, rtol)
        lo = hi + 1
    end
    return sparse(Zi, Zj, Zv, n, k)
end

@noinline function ghost_kernel(A, Wₕ::GhostGridSpace; kwargs...)
    throw(
        ArgumentError(
        "ghost_kernel(A, Wₕ) takes an `AbstractMatrix` A; got a $(nameof(typeof(A))).",
    ),
    )
end

@noinline function ghost_kernel(A, Wₕ; kwargs...)
    throw(
        ArgumentError(
        "ghost_kernel(A, Wₕ) takes a matrix assembled on a space with ghosts, built by " *
        "gridspace(Ωₕ; ghosts = ...) or constrained; got a $(nameof(typeof(Wₕ))). A " *
        "space without ghosts has no ghost-only kernel: build it with " *
        "`ghosts = :boundary` or name its facets.",
    ),
    )
end

@noinline function _throw_ghost_kernel_size(A, Wₕ::GhostGridSpace)
    throw(
        ArgumentError(
        "ghost_kernel(A, Wₕ) takes the $(ndofs(Wₕ)) × $(ndofs(Wₕ)) matrix assembled on " *
        "$(Wₕ); got a $(join(size(A), " × ")) matrix. Pass `assemble(form(Wₕ, Wₕ, ...))` " *
        "on this very space, not one on its padded or physical mesh.",
    ),
    )
end

# The ghost degrees of freedom of `Wₕ` and the linear index of the physical node each one
# sits beyond, sorted by node and then by degree of freedom. The kept physical nodes are
# the physical points whose closure row is not empty (a node `u => g` fixes has none) and
# come first; the kept ghosts follow in ghost order, each one's row a single 1 in its own
# column, which the walk below matches in that order.
function _ghost_dofs_by_node(Wₕ::GhostGridSpace)
    G = mesh(Wₕ)
    E, _ = closure(Wₕ)
    n = size(E, 2)
    plin = LinearIndices(npoints(G, Tuple))
    count = zeros(Int, size(E, 1))
    for r in rowvals(E)
        count[r] += 1
    end
    nphys = 0
    for J in physical_indices(G)
        count[plin[J]] > 0 && (nphys += 1)
    end
    Et = sparse(transpose(E))                  # the rows of `E` as columns
    nodes, dofs = Int[], Int[]
    nlin = LinearIndices(npoints(physical_mesh(G), Tuple))
    j = nphys + 1
    for g in 1:nghosts(G)
        j > n && break
        p = plin[_ghost_point(G, g)]
        r = nzrange(Et, p)
        length(r) == 1 && rowvals(Et)[first(r)] == j && isone(nonzeros(Et)[first(r)]) ||
            continue
        push!(nodes, nlin[first(boundary_node(G, g))])
        push!(dofs, j)
        j += 1
    end
    j == n + 1 || _throw_ghost_dofs_unmatched(Wₕ)
    perm = sortperm(nodes)                      # stable: ghost order within a node
    return nodes[perm], dofs[perm]
end

@noinline function _throw_ghost_dofs_unmatched(Wₕ::GhostGridSpace)
    throw(
        ArgumentError(
        "ghost_kernel could not match the ghost degrees of freedom of $(Wₕ) to its ghosts " *
        "through its closure. Build the space with gridspace(Ωₕ; ghosts = ...) or " *
        "constrained, without editing its closure.",
    ),
    )
end

# The largest 2-norm of a column of `S`, the scale of `A` the local blocks are judged by:
# within a factor `√n` of its largest singular value, and `O(nnz)`.
function _max_column_norm(S::SparseMatrixCSC)
    c = zero(real(float(eltype(S))))
    for j in axes(S, 2)
        c = max(c, norm(view(nonzeros(S), nzrange(S, j))))
    end
    return c
end

# The null vectors of the columns `dofs` of `S` (the ghosts of one node), restricted to
# their nonzero rows, appended as columns `k + 1, …` of the triplets; returns the new count.
# A singular value is null when at most `atol` (`rtol` times the scale of `A`) or `rtol`
# times the block's largest, so a column that is zero only to rounding is null too.
function _node_kernel!(Zi::Vector{Int}, Zj::Vector{Int}, Zv::Vector{T}, k::Int,
        S::SparseMatrixCSC, dofs, atol::Real, rtol::Real) where {T}
    m = length(dofs)
    rows = Int[]
    for j in dofs, q in nzrange(S, j)

        iszero(nonzeros(S)[q]) || push!(rows, rowvals(S)[q])
    end
    unique!(sort!(rows))
    B = fill!(Matrix{T}(undef, length(rows), m), zero(T))
    for (c, j) in enumerate(dofs), q in nzrange(S, j)

        v = nonzeros(S)[q]
        iszero(v) || (B[searchsortedfirst(rows, rowvals(S)[q]), c] = v)
    end
    # A block with no nonzero row (the form never reads these ghosts) is all kernel.
    σ, V = isempty(rows) ? (real(T)[], Matrix{T}(I, m, m)) :
           (F = svd(B; full = true); (F.S, F.V))
    σ₁ = isempty(σ) ? zero(real(T)) : first(σ)
    for i in 1:m
        i <= length(σ) && σ[i] > max(atol, rtol * σ₁) && continue
        z = V[:, i]
        z ./= norm(z)
        s = findfirst(!iszero, z)
        if s !== nothing                         # the first nonzero entry real positive
            z .*= abs(z[s]) / z[s]
            z[s] = abs(z[s])
        end
        k += 1
        for (c, j) in enumerate(dofs)
            iszero(z[c]) && continue
            push!(Zi, j), push!(Zj, k), push!(Zv, z[c])
        end
    end
    return k
end
