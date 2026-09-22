# ext/BrambleSparseMatricesCSRExt.jl: the `SparseMatrixCSR` backend (S3.1,
# gpena/Bramble.jl#214, .agents/plans/v3-3-0-memory-scaling.md).
#
# Plugs `SparseMatricesCSR.jl`'s `SparseMatrixCSR{1,T,Int}` (the one-based variant, matching
# Bramble's own indexing) into the matrix-type seam S1.1 opened in `src/form/`:
# `_scatter_position`, `_scatter_add!` (bilinear_traversal.jl), `_allocate_from_pattern`
# (bilinear_pattern.jl) and `_zero_stored!` (bilinear.jl); and into the Dirichlet/symmetrize
# fast paths `dirichlet_constraints.jl` already carries a `SparseMatrixCSC` specialisation of
# beside its `AbstractMatrix` fallback.
#
# `assemble_parallel!` needs no method here. The band-coloured threaded sweep in
# `bilinear_execution.jl` (`_scatter_point!`, `_sweep_bilinear_colour!`, `_sweep_bilinear!`,
# `_assemble_blocks_parallel!`, both `_assemble_bilinear_parallel_core!` overloads) is typed
# `A::SparseMatrixCSC` throughout, not `A::AbstractMatrix`, so it cannot be reused for CSR
# without widening those signatures -- a file this subplan does not own. A `SparseMatrixCSR`
# therefore falls through to the existing generic
# `_assemble_bilinear_parallel_core!(A::AbstractMatrix, ...)` fallback (the ordinary serial
# record pass, `bilinear_execution.jl`), exactly like every other non-CSC backend today; see
# the integrator report for the widening this would need.
#
# `SparseMatrixCSR`'s own `setindex!` throws on an entry outside the sparsity pattern rather
# than growing it the way `SparseMatrixCSC`'s does (`A[i,i] = one(T)`), so the Dirichlet and
# symmetrize methods below assume the diagonal is already a stored pattern entry at every
# constrained row/column -- true for every finite-difference stencil this package builds,
# since the point itself is always part of its own stencil -- and throw a clear error rather
# than silently doing nothing if it is not.
module BrambleSparseMatricesCSRExt

using Bramble:
               Bramble,
               Backend,
               ExecutionPolicy,
               csr_backend,
               domain,
               interval,
               ×,
               mesh,
               gridspace,
               form,
               assemble,
               assemble!,
               inner₊,
               ∇ₕ,
               boundary_symbols
using SparseMatricesCSR: SparseMatricesCSR, SparseMatrixCSR, sparsecsr
using PrecompileTools: @setup_workload, @compile_workload

# --- backend construction (S1.4's stub) -------------------------------------------- #

# `T <: Number` (rather than an unconstrained `T`) so this is a genuine specialisation of
# the stub in `backend.jl` -- `::Type` there means `Type{T} where T`, so an unconstrained
# `T` here would have the identical signature and count as redefining the same method,
# which precompilation refuses ("Method overwriting is not permitted"), confirmed by
# hitting exactly that error before adding the bound.
function Bramble._csr_backend(::Type{T}, policy::ExecutionPolicy) where {T <: Number}
    return Backend{Vector{T}, SparseMatrixCSR{1, T, Int}, typeof(policy)}()
end

# `SparseMatrixCSR` has neither an `(undef, n, m)` nor an `(n, m)` constructor, so
# `_undef_or_sized` (backend.jl) cannot reach it -- the same reason `Tridiagonal` and
# `SymTridiagonal` get their own `matrix`/`_backend_eye`/`_backend_zeros` methods there
# instead of relying on `supports_undef_construction`. An empty (all-zero, no stored
# entries) `n × m` matrix: `rowptr` all `1` means every row's segment is empty.
@inline function Bramble.matrix(
        ::Backend{VT, MT, EP}, n::Integer, m::Integer
) where {VT, T, Ti, MT <: SparseMatrixCSR{1, T, Ti}, EP}
    return SparseMatrixCSR{1}(Int(n), Int(m), ones(Ti, n + 1), Ti[], T[])
end

@inline function Bramble._backend_eye(
        ::Type{<:SparseMatrixCSR{1, T, Ti}}, n::Integer
) where {T, Ti}
    return SparseMatrixCSR{1}(
        Int(n), Int(n), Vector{Ti}(1:(n + 1)), Vector{Ti}(1:n), ones(T, n)
    )
end

@inline function Bramble._backend_zeros(
        ::Type{<:SparseMatrixCSR{1, T, Ti}}, n::Integer
) where {T, Ti}
    return SparseMatrixCSR{1}(Int(n), Int(n), ones(Ti, n + 1), Ti[], T[])
end

# --- the matrix-type seam (S1.1) ---------------------------------------------------- #

# Built directly with `sparsecsr` rather than via the `SparseMatrixCSC` `sparse!` already
# builds for the CSC backend, converted afterwards: `sparsecsr(I, J, V, m, n, combine)`
# is `SparseMatrixCSR(transpose(sparse(J, I, V, n, m, combine)))` -- one sparse-matrix
# construction (the sort-and-combine `sparse` already does for CSC) plus a zero-copy
# `transpose` wrap, since `SparseMatrixCSR(a::Transpose{...})` takes ownership of `a`'s
# arrays directly. Converting an already-built `SparseMatrixCSC` to CSR instead
# (`SparseMatrixCSR(A::SparseMatrixCSC) = SparseMatrixCSR(transpose(sparse(transpose(A))))`)
# pays that sort-and-combine twice: once to build `A`, once more to materialise
# `sparse(transpose(A))`. Measured directly (best of 15, warmed): a 200x200 2D
# five-point Poisson pattern (40,000 dofs, 199,200 entries pre-combine) built in 1.12 ms via
# `sparsecsr(Val(1), I, J, V, n, n, +)` against 2.59 ms via
# `SparseMatrixCSR(sparse!(I, J, V, n, n, +))` -- direct `sparsecsr` about 2.3x faster.
function Bramble._allocate_from_pattern(
        ::Type{MT},
        nrows::Int,
        ncols::Int,
        I_vec::Vector{Int},
        J_vec::Vector{Int},
        V_vec::AbstractVector
) where {MT <: SparseMatrixCSR}
    return sparsecsr(Val(1), I_vec, J_vec, V_vec, nrows, ncols, +)
end

# The row-major mirror of `_scatter_position(A::SparseMatrixCSC, row, col)`
# (bilinear_traversal.jl): a linear scan of the row's segment when it holds few entries, a
# binary search otherwise, relying on `SparseMatrixCSR`'s own invariant that `colval` is
# sorted within each row (guaranteed by how `_allocate_from_pattern` above builds it, through
# the same `sparse` combine-and-sort `SparseMatrixCSC` gets).
@inline function Bramble._scatter_position(A::SparseMatrixCSR{1}, row::Int, col::Int)
    rowptr = A.rowptr
    colval = A.colval
    p1 = rowptr[row]
    p2 = rowptr[row + 1] - 1

    if (p2 - p1) < 32
        idx = p1
        @inbounds while idx <= p2
            colval[idx] == col && return idx
            idx += 1
        end
    else
        lo = p1
        hi = p2
        @inbounds while lo <= hi
            mid = (lo + hi) >>> 1
            mid_col = colval[mid]
            if mid_col < col
                lo = mid + 1
            elseif mid_col > col
                hi = mid - 1
            else
                return mid
            end
        end
    end
    return 0
end

@inline function Bramble._scatter_add!(A::SparseMatrixCSR{1}, pos::Int, val)
    @inbounds A.nzval[pos] += val
    return nothing
end

@inline function Bramble._zero_stored!(A::SparseMatrixCSR{1})
    fill!(A.nzval, zero(eltype(A)))
    return A
end

# --- Dirichlet and symmetrize (dirichlet_constraints.jl's existing pattern) --------- #

@noinline function _throw_missing_csr_diagonal(i::Int)
    throw(
        ArgumentError(
        "Dirichlet row/column $i has no stored diagonal entry in this `SparseMatrixCSR`. " *
        "Unlike `SparseMatrixCSC`, whose `A[i, i] = one(T)` grows the pattern on demand, " *
        "`SparseMatrixCSR`'s sparsity pattern cannot grow after `allocate_system_matrix`, " *
        "so the form's own stencil must already carry the diagonal at every constrained " *
        "row/column.",
    ),
    )
end

# Zeros each marked row's whole `nzval` segment in one pass -- CSR's row-major storage
# gives this for free where the CSC method (dirichlet_constraints.jl) has to sweep every
# column checking which rows it touches. The diagonal is written where the same pass meets
# it, exactly like the CSC method; see the module docstring above for why a missing one
# throws here instead of being inserted.
function Bramble._dirichlet_bc_rows!(A::SparseMatrixCSR{1}, entries::Tuple)
    T = eltype(A)
    rowptr = A.rowptr
    colval = A.colval
    vals = A.nzval
    for (mask, offset, _, active) in entries
        active || continue
        Bramble._each_marked(mask, offset) do r
            diagonal_found = false
            @inbounds for k in rowptr[r]:(rowptr[r + 1] - 1)
                if colval[k] == r
                    vals[k] = one(T)
                    diagonal_found = true
                else
                    vals[k] = zero(T)
                end
            end
            diagonal_found || _throw_missing_csr_diagonal(r)
            return nothing
        end
    end
    return A
end

# The scalar-space counterpart of `_dirichlet_bc_rows!` above, matching
# `_dirichlet_bc_indices!(A::SparseMatrixCSC, ...)`'s contract against a flat mask.
function Bramble._dirichlet_bc_indices!(A::SparseMatrixCSR{1}, index_in_marker::BitVector)
    T = eltype(A)
    rowptr = A.rowptr
    colval = A.colval
    vals = A.nzval
    Bramble._each_marked(index_in_marker, 0) do i
        diagonal_found = false
        @inbounds for k in rowptr[i]:(rowptr[i + 1] - 1)
            if colval[k] == i
                vals[k] = one(T)
                diagonal_found = true
            else
                vals[k] = zero(T)
            end
        end
        diagonal_found || _throw_missing_csr_diagonal(i)
        return nothing
    end
    return A
end

# `symmetrize!(A::SparseMatrixCSC, ...)` walks each marked *column*'s stored rows directly,
# cheap because CSC stores columns contiguously. CSR stores rows contiguously instead, so
# there is no equally cheap way to visit one column: this sweeps every stored entry once
# (`O(nnz)` rather than the CSC method's "boundary size x column density") and acts on the
# ones whose column is marked, which is where CSC's `elseif index_in_marker[row]` branch
# looks at rows -- here the roles are swapped, and the diagonal (`col == i`) is met from its
# own row's pass instead of from a per-marked-column search.
function Bramble.symmetrize!(
        A::SparseMatrixCSR{1}, F::AbstractVector, mask::BitVector, offset::Int = 0
)
    T = eltype(A)
    rowptr = A.rowptr
    colval = A.colval
    vals = A.nzval
    nr = A.m
    nmask = length(mask)

    @inbounds for i in 1:nr
        i_marked = 1 <= (i - offset) <= nmask && mask[i - offset]
        diagonal_found = false
        for k in rowptr[i]:(rowptr[i + 1] - 1)
            col = colval[k]
            j = col - offset
            (1 <= j <= nmask && mask[j]) || continue
            if col == i
                vals[k] = one(T)
                diagonal_found = true
            else
                dirichlet_val = F[col]
                iszero(dirichlet_val) || (F[i] -= vals[k] * dirichlet_val)
                vals[k] = zero(T)
            end
        end
        i_marked && !diagonal_found && _throw_missing_csr_diagonal(i)
    end
    return A
end

# Warms this extension's own entry points -- `csr_backend`, `Bramble.matrix` for the CSR
# backend, and `assemble`/`assemble!`/`assemble(...; dirichlet)` of the Laplacian form
# against a `SparseMatrixCSR` destination -- since the CSR-typed method instances of the
# core assembly seam (`_allocate_from_pattern`, `_scatter_position`, `_scatter_add!`,
# `_zero_stored!`, `_dirichlet_bc_rows!`/`_dirichlet_bc_indices!` above) are only reachable
# once `SparseMatricesCSR` is loaded, so only this extension's own precompile pass reaches
# them. Covers both 1D and 2D meshes since `assemble` specialises on the form/grid-space
# type, which depends on the mesh dimension, even though the assembled `SparseMatrixCSR{1,
# Float64, Int}` itself does not. Not named in gpena/Bramble.jl#196; added for
# gpena/Bramble.jl#284.
if Bramble.PRECOMPILE_WORKLOAD
    @setup_workload begin
        be = csr_backend()
        systems = map((1, 2)) do D
            S = D == 1 ? interval(0.0, 1.0) : interval(0.0, 1.0) × interval(0.0, 1.0)
            Ω = domain(S, :boundary => boundary_symbols(S))
            Ωₕ = D == 1 ? mesh(Ω, 8, false; backend = be) : mesh(Ω, (8, 8), (false, false); backend = be)
            Wₕ = gridspace(Ωₕ)
            a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
            (; a,)
        end

        precompile(Bramble.matrix, (typeof(be), Int, Int))

        @compile_workload begin
            be2 = csr_backend()
            Bramble.matrix(be2, 5, 5)

            for sys in systems
                A = assemble(sys.a)
                assemble!(A, sys.a)
                assemble(sys.a; dirichlet = :boundary)
            end
        end
    end
end

end # module BrambleSparseMatricesCSRExt
