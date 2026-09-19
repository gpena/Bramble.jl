# ext/BrambleBandedMatricesExt.jl: the `BandedMatrix`/`BandedBlockBandedMatrix` backends
# (S4.2, gpena/Bramble.jl#175 #216, .agents/plans/v3-3-0-memory-scaling.md).
#
# Plugs `BandedMatrices.jl`'s `BandedMatrix` into the matrix-type seam S1.1 opened in
# `src/form/`: `_scatter_position`, `_scatter_add!` (bilinear_traversal.jl),
# `_allocate_from_pattern` (bilinear_pattern.jl) and `_zero_stored!` (bilinear.jl); and into
# the Dirichlet/symmetrize fast paths `dirichlet_constraints.jl` already carries a
# `SparseMatrixCSC` specialisation of, beside its `AbstractMatrix` fallback -- mirroring
# `ext/BrambleSparseMatricesCSRExt.jl` (S3.1), the worked example for a storage-type
# extension.
#
# `block_banded_backend` (#216) names `BlockBandedMatrix` in the issue, but a
# lexicographically ordered Cartesian stencil is block-banded *with banded sub-blocks* --
# `BandedBlockBandedMatrix` -- not the looser `BlockBandedMatrix` (block-banded with dense
# sub-blocks), which would store zeros a Cartesian mesh never produces. `backend.jl`'s own
# `block_banded_backend` docstring (S1.4) already says as much; this file builds the type it
# names.
#
# The one place `BandedBlockBandedMatrix` cannot reuse the `_allocate_from_pattern` seam
# as-is: `allocate_system_matrix` hands that seam only `(nrows, ncols, I, J, V)`, and a block
# partition is not recoverable from those alone (many divisors of `nrows` are possible; the
# pattern does not say which one is the mesh's own). The route taken here needs no `src/`
# edit: a specialised `Bramble.allocate_system_matrix` method, dispatching on the *type* of
# the form's test space -- specifically, on the matrix type buried inside
# `TestSpace`'s mesh type's `Backend` type parameter (`Mesh1D`/`MeshnD` both carry their
# `Backend` as a type parameter, so this is ordinary, not run-time, dispatch) -- reads the
# block sizes from `npoints(mesh(test_space(form)), Tuple)` and the bandwidths from
# `Bramble.blockbandwidths(form)` (S4.1), both already on hand from the AST alone, and
# otherwise repeats the walk `allocate_system_matrix`'s own generic method runs (same
# internal calls: `_bind_interp_spaces`, `_check_block_meshes`, `_walked_leaf`,
# `PatternSink`, `visit_bilinear_stencil`) before building the matrix directly instead of
# handing coordinates to `_allocate_from_pattern`. An alternative considered: teaching
# `_allocate_from_pattern` itself a block-size argument -- rejected, since every other
# backend's method has the same four-argument shape and widening it would be a `src/` edit
# to a public-facing internal this subplan does not own.
#
# `assemble_parallel!` needs no method here, for the same reason `ext/BrambleSparseMatricesCSRExt.jl`
# needs none: the band-coloured threaded sweep in `bilinear_execution.jl` is typed
# `A::SparseMatrixCSC` throughout, so both matrix types here fall through to the generic
# `_assemble_bilinear_parallel_core!(A::AbstractMatrix, ...)` fallback (the serial record
# pass).
module BrambleBandedMatricesExt

using Bramble: Bramble, Backend, ExecutionPolicy
using BandedMatrices: BandedMatrices, BandedMatrix, Zeros
using BlockBandedMatrices: BlockBandedMatrices, BandedBlockBandedMatrix
using LinearAlgebra: LinearAlgebra, I

# --- backend construction (S1.4's stub) -------------------------------------------- #

# `T <: Number` (rather than an unconstrained `T`) for the same reason
# `BrambleSparseMatricesCSRExt._csr_backend` needs the bound: an unconstrained `T` here has
# the identical signature to the stub in `backend.jl` (`::Type` there means `Type{T} where
# T`), and precompilation refuses redefining the same method.
function Bramble._banded_backend(::Type{T}, policy::ExecutionPolicy) where {T <: Number}
    return Backend{Vector{T}, BandedMatrix{T, Matrix{T}, Base.OneTo{Int}}, typeof(policy)}()
end

function Bramble._block_banded_backend(
        ::Type{T}, policy::ExecutionPolicy
) where {T <: Number}
    return Backend{Vector{T}, BandedBlockBandedMatrix{T}, typeof(policy)}()
end

# `matrix`, `_backend_eye`, `_backend_zeros`: `BandedMatrix` has neither an `(undef, n, m)`
# nor an `(n, m)` constructor (a band needs a bandwidth to mean anything), so
# `_undef_or_sized` (backend.jl) cannot reach it -- the same reason `Tridiagonal` and
# `SparseMatrixCSR` get their own methods there instead. `(-1, -1)` is
# `BandedMatrices.jl`'s own spelling of "nothing is stored" (a zero-row `data`), the exact
# analogue of `SparseMatrixCSR`'s `nnz == 0` empty matrix.
@inline function Bramble.matrix(
        ::Backend{VT, MT, EP}, n::Integer, m::Integer
) where {VT, T, MT <: BandedMatrix{T}, EP}
    return BandedMatrix(Zeros{T}(Int(n), Int(m)), (-1, -1))
end

@inline function Bramble._backend_eye(::Type{<:BandedMatrix{T}}, n::Integer) where {T}
    return BandedMatrix{T}(I, (Int(n), Int(n)), (0, 0))
end

@inline function Bramble._backend_zeros(::Type{<:BandedMatrix{T}}, n::Integer) where {T}
    return BandedMatrix(Zeros{T}(Int(n), Int(n)), (-1, -1))
end

# The block-banded backend's `matrix`/`_backend_eye`/`_backend_zeros` have no mesh to read a
# block partition from (see the module docstring above on `allocate_system_matrix` for why
# that partition matters and where it actually comes from). A bare `matrix(backend, n, m)`
# call, with nothing else to go on, gets the coarsest legal partition -- one row-block and
# one column-block -- under which a `BandedBlockBandedMatrix` behaves exactly like a plain
# `BandedMatrix` wrapped in block form; real system matrices never take this path; they are
# built by [`allocate_system_matrix`](@ref) below.
@inline function Bramble.matrix(
        ::Backend{VT, MT, EP}, n::Integer, m::Integer
) where {VT, T, MT <: BandedBlockBandedMatrix{T}, EP}
    return BandedBlockBandedMatrix(Zeros{T}(Int(n), Int(m)), [Int(n)], [Int(m)], (-1, -1), (-1, -1))
end

@inline function Bramble._backend_eye(
        ::Type{<:BandedBlockBandedMatrix{T}}, n::Integer
) where {T}
    return BandedBlockBandedMatrix(Matrix{T}(I, Int(n), Int(n)), [Int(n)], [Int(n)], (0, 0), (0, 0))
end

@inline function Bramble._backend_zeros(
        ::Type{<:BandedBlockBandedMatrix{T}}, n::Integer
) where {T}
    return BandedBlockBandedMatrix(Zeros{T}(Int(n), Int(n)), [Int(n)], [Int(n)], (-1, -1), (-1, -1))
end

# --- the matrix-type seam (S1.1): BandedMatrix -------------------------------------- #

"""
    _allocate_from_pattern(::Type{<:BandedMatrix}, nrows, ncols, I, J, V) -> BandedMatrix

The bandwidths `(l, u)` a Cartesian stencil's own pattern needs are exactly the pattern's
own coordinate extremes -- `l = max(row - col)`, `u = max(col - row)` -- over every `(I[k],
J[k])` the walk collected, so no separate bandwidth analysis (`Bramble.bandwidths`, S4.1) is
needed to allocate: the pattern already carries the answer. Duplicate `(row, col)` pairs
(the sum [`_allocate_from_pattern`](@ref)'s own docstring promises) are combined by
[`_scatter_add!`](@ref) below, one at a time, rather than by a sort-and-combine pass the way
`sparse!`/`sparsecsr` do it for the sparse backends: a `BandedMatrix`'s storage has no
notion of "the same slot twice" to sort away, so accumulating directly is both correct and
the cheaper option.
"""
function Bramble._allocate_from_pattern(
        ::Type{MT},
        nrows::Int,
        ncols::Int,
        I_vec::Vector{Int},
        J_vec::Vector{Int},
        V_vec::AbstractVector
) where {MT <: BandedMatrix}
    T = eltype(V_vec)
    l = 0
    u = 0
    @inbounds for k in eachindex(I_vec, J_vec)
        d = J_vec[k] - I_vec[k]
        l = max(l, -d)
        u = max(u, d)
    end
    A = BandedMatrix(Zeros{T}(nrows, ncols), (l, u))
    @inbounds for k in eachindex(I_vec, J_vec, V_vec)
        pos = Bramble._scatter_position(A, I_vec[k], J_vec[k])
        Bramble._scatter_add!(A, pos, V_vec[k])
    end
    return A
end

"""
    _scatter_position(A::BandedMatrix, row::Int, col::Int) -> Int

The LAPACK band-storage index `BandedMatrices.jl` itself uses for `A.data`: column-major,
`u + 1` rows of padding above the diagonal, so entry `(row, col)` lives at
`A.data[u + 1 + row - col, col]` -- `0` when `(row, col)` falls outside `(l, u)`, matching
`SparseMatrixCSC`'s own "not stored" answer.

`O(1)`, unlike the sparse backends' column/row scan: a band's storage says directly where an
entry is, with nothing to search for.
"""
@inline function Bramble._scatter_position(A::BandedMatrix, row::Int, col::Int)
    l, u = BandedMatrices.bandwidths(A)
    (-l <= (col - row) <= u) || return 0
    stride = size(A.data, 1)
    return (u + 1 + row - col) + (col - 1) * stride
end

@inline function Bramble._scatter_add!(A::BandedMatrix, pos::Int, val)
    @inbounds A.data[pos] += val
    return nothing
end

@inline function Bramble._zero_stored!(A::BandedMatrix)
    fill!(A.data, zero(eltype(A)))
    return A
end

# --- the matrix-type seam (S1.1): BandedBlockBandedMatrix --------------------------- #

# Both axes are one uniform block size throughout this file (mixing block sizes -- an
# unequal partition of a Cartesian mesh's own axis -- never arises: [`allocate_system_matrix`](@ref)
# below always builds one). `axes(A, d).lasts[1]` is the length of that axis's first block,
# equal to every other block's for a uniform partition, and reading it costs one field
# access into the `BlockedOneTo` -- no allocation, unlike `BlockBandedMatrices.blocksize(A)`.
@inline function _uniform_block_sizes(A::BandedBlockBandedMatrix)
    return (@inbounds axes(A, 1).lasts[1]), (@inbounds axes(A, 2).lasts[1])
end

"""
    _scatter_position(A::BandedBlockBandedMatrix, row::Int, col::Int) -> Int

The block-banded analogue of `_scatter_position(A::BandedMatrix, ...)`: `A.data` stores only
the `l + u + 1` block-bands that can hold an entry, each `λ + μ + 1` rows tall in its own
banded layout, with the *global* column axis otherwise untouched (`BandedBlockBandedMatrix`
docstring, `BlockBandedMatrices.jl`). Reduced to plain arithmetic on a uniform Cartesian
partition: `(row, col)` maps to block indices `(K, J)` and within-block offsets `(kloc,
jloc)`, `0` when either the block-level band (`-l <= J - K <= u`) or the within-block band
(`-λ <= jloc - kloc <= μ`) is missed, and otherwise to
`(u + K - J) * (λ + μ + 1) + (μ + 1 + kloc - jloc)` in `A.data`'s own row axis, column `col`
unchanged. Verified directly against `BandedBlockBandedMatrix`'s own `getindex`/`setindex!`
on every in-band and out-of-band case of a small test matrix before use here: `A[i, j] =
v; A.data[pos] == v` agreed in every case, and 64 B/call of the generic
`AbstractMatrix` fallback (`A[LinearIndices(A)[row,col]] += val`) dropped to 0 B/call.
"""
@inline function Bramble._scatter_position(A::BandedBlockBandedMatrix, row::Int, col::Int)
    l, u, λ, μ = A.l, A.u, A.λ, A.μ
    blksz_r, blksz_c = _uniform_block_sizes(A)
    K = (row - 1) ÷ blksz_r + 1
    J = (col - 1) ÷ blksz_c + 1
    (-l <= (J - K) <= u) || return 0
    kloc = (row - 1) % blksz_r + 1
    jloc = (col - 1) % blksz_c + 1
    (-λ <= (jloc - kloc) <= μ) || return 0
    band_row = u + K - J + 1
    raw_row = (band_row - 1) * (λ + μ + 1) + (μ + 1 + kloc - jloc)
    stride = size(parent(A.data), 1)
    return raw_row + (col - 1) * stride
end

@inline function Bramble._scatter_add!(A::BandedBlockBandedMatrix, pos::Int, val)
    @inbounds parent(A.data)[pos] += val
    return nothing
end

@inline function Bramble._zero_stored!(A::BandedBlockBandedMatrix)
    fill!(parent(A.data), zero(eltype(A)))
    return A
end

# --- allocate_system_matrix: the block-banded route (see the module docstring) ------ #

# Matches only a `BilinearForm` whose *test* space's mesh was built on a `block_banded_backend()`
# -- `TestSpace`'s own type parameters spell this out (`Mesh1D`/`MeshnD` carry their
# `Backend` as a type parameter, `ScalarGridSpace` carries its mesh type the same way), so
# this is compile-time dispatch on a type that already exists, not a run-time branch on
# `matrix_type(backend(...))`. `D` is shared with the enclosing `BilinearForm{D, ...}` so a
# 1D `Mesh1D`-backed space (block-banded makes no sense there; `Bramble.blockbandwidths`
# already treats `D == 1` as trivial) never matches this method and falls through to the
# ordinary `allocate_system_matrix`, which then hits `_allocate_from_pattern`'s generic
# `BandedBlockBandedMatrix` fallback below and reports the missing block-size problem this
# method exists to solve.
const _BlockBandedTestSpace{D} = Bramble.ScalarGridSpace{
    D, <:Any, <:Any,
    <:Bramble.MeshnD{D, <:Backend{<:Any, <:BandedBlockBandedMatrix}}
}

"""
    allocate_system_matrix(form::BilinearForm) -> BandedBlockBandedMatrix

Specialised for a form whose test space's backend matrix type is `BandedBlockBandedMatrix`
(built by [`block_banded_backend`](@ref)).

The generic `allocate_system_matrix` (`bilinear_pattern.jl`) hands `_allocate_from_pattern`
only `(nrows, ncols, I, J, V)`, from which a block partition cannot be recovered: many
divisors of `nrows` are possible, and the pattern does not say which one is the mesh's own
(see the module docstring above). This method runs the same walk the generic one does --
`_bind_interp_spaces`, `_check_block_meshes`, `_walked_leaf`, `PatternSink`,
`visit_bilinear_stencil`, `_matrix_eltype` are the same internal calls, in the same order --
but reads the block sizes from `npoints(mesh(test_space(form)), Tuple)` (the `D - 1` leading
axes are one block, the `D`-th axis is the block count -- [`Bramble.blockbandwidths`](@ref)'s
own layout, S4.1) and the bandwidths from `Bramble.blockbandwidths(form)`, then builds the
`BandedBlockBandedMatrix` and scatters directly instead of calling `_allocate_from_pattern`.
"""
function Bramble.allocate_system_matrix(
        form::Bramble.BilinearForm{D, TrialSpace, TestSpace, AST}, ast = form.ast
) where {D, TrialSpace, AST, TestSpace <: _BlockBandedTestSpace{D}}
    ast = Bramble._bind_interp_spaces(ast, form.trial_space, form.test_space)
    Bramble._check_block_meshes(ast, form.trial_space, form.test_space)
    space = Bramble._walked_leaf(ast, form.trial_space, form.test_space)
    Ωₕ = Bramble.mesh(space)
    mesh_markers = Bramble.markers(Ωₕ)
    Bramble._validate_term_markers(ast, mesh_markers, "the form's space")
    lin_indices = LinearIndices(Bramble.indices(Ωₕ))

    I_vec = Int[]
    J_vec = Int[]
    hint = Bramble._pattern_size_hint(ast, space, mesh_markers, lin_indices)
    sizehint!(I_vec, hint)
    sizehint!(J_vec, hint)
    Bramble.visit_bilinear_stencil(Bramble.PatternSink(I_vec, J_vec), ast, space, 0, 0)

    nrows = Bramble.ndofs(form.test_space)
    ncols = Bramble.ndofs(form.trial_space)
    V_vec = Bramble._zeros_of(Bramble._matrix_eltype(ast, form), length(I_vec))

    (l_blk, u_blk), (l_sub, u_sub) = Bramble.blockbandwidths(form)
    n = Bramble.npoints(Ωₕ, Tuple)
    blksz = prod(Base.front(n))
    nblk = last(n)
    rdims = fill(blksz, nblk)
    cdims = fill(blksz, nblk)

    A = BandedBlockBandedMatrix(
        Zeros{eltype(V_vec)}(nrows, ncols), rdims, cdims, (l_blk, u_blk), (l_sub, u_sub)
    )
    @inbounds for k in eachindex(I_vec, J_vec, V_vec)
        pos = Bramble._scatter_position(A, I_vec[k], J_vec[k])
        Bramble._scatter_add!(A, pos, V_vec[k])
    end
    return A
end

# The 1D (`Mesh1D`) or otherwise-not-block-sized case that falls through to here: a form's
# own `Bramble.blockbandwidths` treats `D == 1` as trivial (one point per block), which this
# backend is not built for, and every other case is a programming error (`block_banded_backend`
# used somewhere `allocate_system_matrix`'s specialised method above should have matched
# instead). Named clearly rather than left as a `MethodError` naming `_allocate_from_pattern`,
# which would point at the wrong function.
function Bramble._allocate_from_pattern(
        ::Type{MT}, ::Int, ::Int, ::Vector{Int}, ::Vector{Int}, ::AbstractVector
) where {MT <: BandedBlockBandedMatrix}
    throw(
        ArgumentError(
        "BandedBlockBandedMatrix needs a mesh to size its blocks from, which " *
        "`allocate_system_matrix(form::BilinearForm)`'s specialised method reads from the " *
        "form's test space; this fallback runs only when that specialisation did not match " *
        "(a 1D `Mesh1D`-backed space, where `block_banded_backend` is not meaningful -- " *
        "use `banded_backend` instead).",
    ),
    )
end

# --- Dirichlet and symmetrize: BandedMatrix (dirichlet_constraints.jl's own pattern) --- #

# Unlike `SparseMatrixCSR` (S3.1), a `BandedMatrix` never has a "missing" diagonal to guard
# against: every `(row, col)` inside `bandwidths(A)` is `A`'s own pattern by construction,
# and [`_allocate_from_pattern`](@ref) above always keeps the diagonal in band (`l, u >= 0`
# always holds for a real stencil, since a point is always part of its own stencil). Both
# methods below therefore walk only the marked rows/columns via [`Bramble._each_marked`](@ref)
# -- costing the boundary cardinality times the bandwidth, not `ndofs` -- and write the
# diagonal unconditionally rather than searching for it first.
function Bramble._dirichlet_bc_indices!(A::BandedMatrix, index_in_marker::BitVector)
    T = eltype(A)
    l, u = BandedMatrices.bandwidths(A)
    ncols = size(A, 2)
    data = A.data
    Bramble._each_marked(index_in_marker, 0) do r
        clo = max(1, r - l)
        chi = min(ncols, r + u)
        @inbounds for c in clo:chi
            pos = Bramble._scatter_position(A, r, c)
            data[pos] = (c == r) ? one(T) : zero(T)
        end
        return nothing
    end
    return A
end

# The composite-space counterpart, matching `_dirichlet_bc_rows!(A::SparseMatrixCSR, ...)`'s
# contract against an `entries` tuple of `(mask, offset, _, active)`.
function Bramble._dirichlet_bc_rows!(A::BandedMatrix, entries::Tuple)
    T = eltype(A)
    l, u = BandedMatrices.bandwidths(A)
    ncols = size(A, 2)
    data = A.data
    for (mask, offset, _, active) in entries
        active || continue
        Bramble._each_marked(mask, offset) do r
            clo = max(1, r - l)
            chi = min(ncols, r + u)
            @inbounds for c in clo:chi
                pos = Bramble._scatter_position(A, r, c)
                data[pos] = (c == r) ? one(T) : zero(T)
            end
            return nothing
        end
    end
    return A
end

# `symmetrize!` acts on *columns* of the marked set (restoring symmetry after the row
# elimination above), matching `symmetrize!(A::SparseMatrixCSC, ...)`'s own contract: zero
# the off-diagonal entries of column `i`, folding each one's contribution into `F` first.
function Bramble.symmetrize!(
        A::BandedMatrix, F::AbstractVector, mask::BitVector, offset::Int = 0
)
    T = eltype(A)
    l, u = BandedMatrices.bandwidths(A)
    nrows = size(A, 1)
    data = A.data
    Bramble._each_marked(mask, offset) do i
        dirichlet_val = F[i]
        value_is_zero = iszero(dirichlet_val)
        klo = max(1, i - u)
        khi = min(nrows, i + l)
        @inbounds for k in klo:khi
            pos = Bramble._scatter_position(A, k, i)
            if k == i
                data[pos] = one(T)
            else
                value_is_zero || (F[k] -= data[pos] * dirichlet_val)
                data[pos] = zero(T)
            end
        end
        return nothing
    end
    return A
end

# --- Dirichlet and symmetrize: BandedBlockBandedMatrix ------------------------------ #

function Bramble._dirichlet_bc_indices!(
        A::BandedBlockBandedMatrix, index_in_marker::BitVector
)
    T = eltype(A)
    l, u, λ, μ = A.l, A.u, A.λ, A.μ
    blksz_r, blksz_c = _uniform_block_sizes(A)
    nblk_c = size(A, 2) ÷ blksz_c
    data = parent(A.data)
    Bramble._each_marked(index_in_marker, 0) do r
        K = (r - 1) ÷ blksz_r + 1
        kloc = (r - 1) % blksz_r + 1
        Jlo = max(1, K - l)
        Jhi = min(nblk_c, K + u)
        @inbounds for J in Jlo:Jhi
            jloc_lo = max(1, kloc - λ)
            jloc_hi = min(blksz_c, kloc + μ)
            for jloc in jloc_lo:jloc_hi
                c = (J - 1) * blksz_c + jloc
                pos = Bramble._scatter_position(A, r, c)
                data[pos] = (c == r) ? one(T) : zero(T)
            end
        end
        return nothing
    end
    return A
end

function Bramble._dirichlet_bc_rows!(A::BandedBlockBandedMatrix, entries::Tuple)
    T = eltype(A)
    l, u, λ, μ = A.l, A.u, A.λ, A.μ
    blksz_r, blksz_c = _uniform_block_sizes(A)
    nblk_c = size(A, 2) ÷ blksz_c
    data = parent(A.data)
    for (mask, offset, _, active) in entries
        active || continue
        Bramble._each_marked(mask, offset) do r
            K = (r - 1) ÷ blksz_r + 1
            kloc = (r - 1) % blksz_r + 1
            Jlo = max(1, K - l)
            Jhi = min(nblk_c, K + u)
            @inbounds for J in Jlo:Jhi
                jloc_lo = max(1, kloc - λ)
                jloc_hi = min(blksz_c, kloc + μ)
                for jloc in jloc_lo:jloc_hi
                    c = (J - 1) * blksz_c + jloc
                    pos = Bramble._scatter_position(A, r, c)
                    data[pos] = (c == r) ? one(T) : zero(T)
                end
            end
            return nothing
        end
    end
    return A
end

function Bramble.symmetrize!(
        A::BandedBlockBandedMatrix, F::AbstractVector, mask::BitVector, offset::Int = 0
)
    T = eltype(A)
    l, u, λ, μ = A.l, A.u, A.λ, A.μ
    blksz_r, blksz_c = _uniform_block_sizes(A)
    nblk_r = size(A, 1) ÷ blksz_r
    data = parent(A.data)
    Bramble._each_marked(mask, offset) do i
        dirichlet_val = F[i]
        value_is_zero = iszero(dirichlet_val)
        J = (i - 1) ÷ blksz_c + 1
        jloc = (i - 1) % blksz_c + 1
        Klo = max(1, J - u)
        Khi = min(nblk_r, J + l)
        @inbounds for K in Klo:Khi
            kloc_lo = max(1, jloc - μ)
            kloc_hi = min(blksz_r, jloc + λ)
            for kloc in kloc_lo:kloc_hi
                r = (K - 1) * blksz_r + kloc
                pos = Bramble._scatter_position(A, r, i)
                if r == i
                    data[pos] = one(T)
                else
                    value_is_zero || (F[r] -= data[pos] * dirichlet_val)
                    data[pos] = zero(T)
                end
            end
        end
        return nothing
    end
    return A
end

end # module BrambleBandedMatricesExt
