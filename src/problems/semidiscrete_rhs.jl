# semidiscrete_rhs.jl: `SemidiscretizeRHS`, the matrix-free explicit right-hand side
# `uₕ' = M⁻¹(F(t) - A uₕ)` for a `Semidiscretization` (`semidiscrete.jl`) with no Dirichlet
# constraints, `M`'s diagonal folded into a precomputed scaling instead of left for a solver
# to factorise at every step.

# --- The matrix-free explicit right-hand side --------------------------------------- #
#
# `M uₕ' = F(t) - A uₕ` (the residual above) is deliberately *not* divided by `M`: that
# residual is handed to SciML as an `ODEFunction` carrying `M` as its own `mass_matrix`,
# and the solver factorises `M` itself -- the only correct way to do it once any Dirichlet
# row is constrained, since a constrained row of `M` is zero and dividing by it is nonsense
# (that zero row *is* the algebraic constraint `0 = g(x, t) - uₕ[i]`, not an equation for
# `uₕ'[i]`).
#
# `M` is genuinely invertible everywhere only when there is no such row at all --
# `NoConstraints`, the plain `M = diag(weights)` the discrete `L²` inner product always
# assembles into. Only then does `uₕ' = M⁻¹(F(t) - A uₕ)` mean the same thing as the
# constrained residual above, and only then is dividing by `M`'s diagonal, once, outside
# the time loop, a valid substitute for a mass-matrix-aware solver -- gpena/Bramble.jl#163's
# whole premise ("H is diagonal") holds for this one case, not in general: a caller's own
# `mass` keyword to `semidiscretize` can be any `BilinearForm`, including a non-diagonal
# one, so this is checked here rather than assumed.

"""
    SemidiscretizeRHS{S,V,C}

The right-hand side of `uₕ' = M⁻¹(F(t) - A uₕ)`, `M` folded into a precomputed diagonal
scaling rather than solved for -- for a [`Semidiscretization`](@ref) with no Dirichlet
constraints, where that fold is valid. Built by [`semidiscretize_rhs`](@ref); callable with
the `(du, u, p, t)` signature a plain (non-mass-matrix) `SciMLBase.ODEProblem` expects.

Named `SemidiscretizeRHS`/[`semidiscretize_rhs`](@ref) rather than the free function
`semidiscretize_rhs!(du, u, p, t)` gpena/Bramble.jl#163 proposes: that signature is exactly
`SciMLBase.ODEFunction`'s own, which leaves no argument to pass a
[`Semidiscretization`](@ref) or the precomputed scaling through -- both have to be closed
over somehow, and a callable struct is what every other stateful callable in this package
does instead of a closure (`Semidiscretization` itself, `TypeCachedOperator`).

The `csr` field holds the compressed-row pattern of `A` that a [`CpuPolyester`](@ref) space
with `reassemble = false` multiplies by (see [`semidiscretize_rhs`](@ref)), and is
`nothing` otherwise.
"""
struct SemidiscretizeRHS{S <: Semidiscretization, V <: AbstractVector, C}
    sd::S
    inv_mass_diag::V
    csr::C
end

# `M`'s own storage, not `weights(space(sd))`: `mass` is a caller-supplied keyword to
# `semidiscretize`, so the diagonal actually assembled -- which the default `innerₕ(u, v)`
# happens to make equal to `weights`, but a coefficient-scaled custom `mass` would not --
# is what has to be inverted.
function _diagonal_or_throw(M::SparseMatrixCSC)
    n = size(M, 1)
    d = zeros(eltype(M), n)
    nz = nonzeros(M)
    rv = rowvals(M)
    for j in 1:n
        rng = nzrange(M, j)
        if length(rng) > 1 || (length(rng) == 1 && rv[first(rng)] != j)
            throw(
                ArgumentError(
                "semidiscretize_rhs requires a diagonal mass matrix; column $j of the " *
                "assembled `mass` form has an off-diagonal entry.",
            ),
            )
        end
        length(rng) == 1 && (d[j] = nz[first(rng)])
    end
    return d
end

# Generic `AbstractMatrix` fallback (S1.2's extension of the matrix-type seam,
# gpena/Bramble.jl#12): no `nonzeros`/`rowvals` to walk on a dense-backend matrix, so the
# off-diagonal check reads every entry directly instead.
function _diagonal_or_throw(M::AbstractMatrix)
    n = size(M, 1)
    d = zeros(eltype(M), n)
    for j in 1:n, i in 1:n

        if i == j
            d[j] = M[i, j]
        elseif !iszero(M[i, j])
            throw(
                ArgumentError(
                "semidiscretize_rhs requires a diagonal mass matrix; column $j of the " *
                "assembled `mass` form has an off-diagonal entry.",
            ),
            )
        end
    end
    return d
end

"""
    semidiscretize_rhs(sd::Semidiscretization) -> SemidiscretizeRHS

Build the matrix-free right-hand side of `uₕ' = M⁻¹(F(t) - A uₕ)` from `sd`, folding `M`'s
diagonal into a precomputed scaling instead of leaving `M` for a solver to factorise at
every step -- valid only because `M` is diagonal and, here, invertible everywhere.

Requires `sd` to have been built with `dirichlet = nothing`: any Dirichlet label makes the
corresponding row of `M` zero by construction (see [`semidiscretize`](@ref)), which is an
algebraic constraint on `uₕ`, not an equation for `uₕ'` -- there is no `uₕ'` value that
divides it away. Also requires `mass_matrix(sd)` to actually be diagonal: the default
`mass` (the discrete `L²` inner product) always assembles diagonally, but a caller-supplied
`mass` keyword to [`semidiscretize`](@ref) need not.

When the space's execution policy is [`CpuPolyester`](@ref) (`using Polyester` required),
`sd` was built with `reassemble = false` and `A` is a `SparseMatrixCSC`, the product `A u`
runs one row per `Polyester.@batch` iteration, each row writing its own entry of `du`. Rows
need `A`'s pattern in compressed-row order, built here once: two `Int` arrays of length
`nnz(A)` (the column of each entry, and its position in `nonzeros(A)`) and one of length
`size(A, 1) + 1`, costing about three products to build. The values are read from `A`
itself at every call, so editing `nonzeros(operator_matrix(sd))` afterwards is seen; a
change to `A`'s stored pattern is not, and if it changes `nnz(A)` the call falls back to
the serial product. The row sums run in a different order than the serial column product,
so the result agrees with [`CpuSerial`](@ref)'s to rounding, not bit for bit. With
`reassemble = true` `A` changes at every step and a fresh copy would cost more than the
threads save, so that case, like every other policy, keeps the serial product.

# Examples

```julia
using Bramble: semidiscretize_rhs
sd = semidiscretize(a, l)  # no `dirichlet` keyword: NoConstraints
rhs = semidiscretize_rhs(sd)
prob = ODEProblem(rhs, parent(u₀), (0.0, 1.0))  # no mass_matrix to factorise
sol = solve(prob, Tsit5())
```

See also [`semidiscretize`](@ref), [`ode_problem`](@ref).
"""
function semidiscretize_rhs(sd::Semidiscretization)
    sd.constraints isa NoConstraints || throw(
        ArgumentError(
        "semidiscretize_rhs requires a Semidiscretization with dirichlet = nothing " *
        "(NoConstraints): a Dirichlet row's zero mass-matrix entry is an algebraic " *
        "constraint, not something `M⁻¹` can be taken through. Got constraints of " *
        "kind $(typeof(sd.constraints)).",
    ),
    )
    d = _diagonal_or_throw(mass_matrix(sd))
    any(iszero, d) && throw(
        ArgumentError("semidiscretize_rhs: the assembled mass matrix has a zero diagonal entry."),
    )
    A = operator_matrix(sd)
    return SemidiscretizeRHS(sd, inv.(d), _rhs_csr(execution_policy(sd.space), sd.reassemble, A))
end

# The compressed-row pattern of `A` the `CpuPolyester` product walks: `rowptr[i]:rowptr[i +
# 1] - 1` are row `i`'s entries, `colval[k]` the column of entry `k` and `perm[k]` its index
# in `nonzeros(A)`. Only the pattern is kept, not the values, so the product reads `A`'s
# current values and an in-place edit of them is never stale.
struct _CSRPattern
    rowptr::Vector{Int}
    colval::Vector{Int}
    perm::Vector{Int}
end

# Transposing a copy of `A` whose values are their own positions `1:nnz(A)` gives the
# row-major pattern and, as its values, the permutation into `nonzeros(A)`. Every other case
# keeps no pattern and runs the serial column product.
_rhs_csr(policy, reassemble, A) = nothing
_nstored(A::SparseMatrixCSC) = Int(A.colptr[end]) - 1
function _rhs_csr(::CpuPolyester, ::Val{false}, A::SparseMatrixCSC)
    n = _nstored(A)
    P = SparseMatrixCSC(size(A)..., Vector{Int}(A.colptr), Vector{Int}(rowvals(A)[1:n]),
        collect(1:n))
    Pt = copy(transpose(P))
    return _CSRPattern(Pt.colptr, rowvals(Pt), nonzeros(Pt))
end

# `du .-= A u`: the serial column product, or the row-parallel one over the pattern. A
# pattern whose entry count no longer matches `A`'s (an entry inserted or dropped since
# `semidiscretize_rhs`) would index past `nonzeros(A)`, so that case runs serially.
_rhs_spmv!(du, A, ::Nothing, u) = mul!(du, A, u, -1, 1)
# The row loop runs under `@inbounds`, so the lengths `mul!` would check are checked here.
function _rhs_spmv!(du, A::SparseMatrixCSC, csr::_CSRPattern, u)
    length(csr.perm) == _nstored(A) || return mul!(du, A, u, -1, 1)
    (length(du) == size(A, 1) && length(u) == size(A, 2)) || _throw_rhs_dims(du, A, u)
    _late(_batch_csr_spmv!, du, csr.rowptr, csr.colval, csr.perm, nonzeros(A), u)
    return du
end

@noinline function _throw_rhs_dims(du, A, u)
    throw(
        DimensionMismatch(
        "the operator matrix has size $(size(A)), but du has length $(length(du)) and u " *
        "has length $(length(u))",
    ),
    )
end

"""
    _batch_csr_spmv!(du, rowptr, colval, perm, nzval, u) -> Nothing

[`CpuPolyester`](@ref)'s explicit right-hand-side product, filled by `BramblePolyesterExt`:
`du[i] -= Σₖ nzval[perm[k]] * u[colval[k]]` over `k in rowptr[i]:(rowptr[i + 1] - 1)`, with
`(rowptr, colval, perm)` the compressed-row pattern of `A` and `nzval = nonzeros(A)`, one row
`i` per `Polyester.@batch` iteration. The only `src/` method errors naming Polyester.
"""
@noinline function _batch_csr_spmv!(du, rowptr, colval, perm, nzval, u)
    return _throw_cpubatch_without_polyester(:_batch_csr_spmv!)
end

"""
    (rhs::SemidiscretizeRHS)(du, u, p, t) -> du

Evaluate `du = M⁻¹(F(t) - A u)`, the explicit right-hand side [`semidiscretize_rhs`](@ref)
built.

Under [`CpuSerial`](@ref) it allocates nothing (**0 bytes**) under the same condition the
underlying [`Semidiscretization`](@ref) residual does: `eltype(du)` and `typeof(t)`
matching the assembled element type. Under [`CpuPolyester`](@ref) each call allocates a
small constant amount, the same on every grid, for the argument boxes Polyester sends to
its threads.
"""
function (rhs::SemidiscretizeRHS)(du::AbstractVector, u::AbstractVector, p, t)
    sd = rhs.sd
    _sync_state!(sd.state, u)
    _update_coefficients!(sd.update_coefficients, p, t)

    F = _source_buffer(sd, du, t)
    _assemble_source!(F, sd, sd.constraints, p, t)

    A = _refresh_operator!(sd, sd.reassemble, du, t)

    copyto!(du, F)
    _rhs_spmv!(du, A, rhs.csr, u)
    du .*= rhs.inv_mass_diag
    return du
end
