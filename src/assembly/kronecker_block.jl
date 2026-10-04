# kronecker_block.jl
#
# `KroneckerBlockOperator`: the Kronecker operator of a bilinear form over a composite space
# whose leaves all share one mesh (gpena/Bramble.jl#427). The form splits into its
# (test leaf, trial leaf) blocks exactly as assembly splits it (`block_of`,
# `block_extract.jl`); each nonzero block is a scalar form on the shared mesh, so it gets
# its own `KroneckerLinearOperator`, and the block operator applies each one to the slices
# of `x` and `y` its leaves own.
#
# One operator with the leaf index as one more tensor axis was rejected: leaves may differ
# in their number of points per axis (staggered spaces), so the leaf index is not a tensor
# axis. Leaves on different meshes are refused, as a scalar form's two spaces on different
# meshes are: the factors are built on one mesh's axes.

const _KronBlockSpace = Union{ScalarGridSpace, CompositeGridSpace}

# --- Splitting the form into blocks -------------------------------------------------- #

"""
    _kron_unindex(op::LazyOp) -> LazyOp

`op` with every `IndexedTrialFunction`/`IndexedTestFunction` replaced by the plain
`TrialFunction`/`TestFunction`: within its block a term is a scalar form on the block's
leaves, which `_kron_project` reads. A node `_kron_project` has no projection for is kept as
it is (its indexed leaves too), so the walk still refuses it.
"""
_kron_unindex(::IndexedTrialFunction{D}) where {D} = TrialFunction{D}()
_kron_unindex(::IndexedTestFunction{D}) where {D} = TestFunction{D}()
function _kron_unindex(op::Union{OperatorAdd{D}, BilinearProduct{D}}) where {D}
    l = _kron_unindex(op.left_op)
    r = _kron_unindex(op.right_op)
    return _kron_rebuild(op, l, r)
end
# A node with one operand is rebuilt with `_kron_rewrap`, which every node the walk projects
# has (`kronecker_projection.jl`); `applicable` keeps any other node as it is, and runs only
# while the operator is built.
function _kron_unindex(op::LazyOp)
    hasproperty(op, :inner_op) || return op
    x = _kron_unindex(op.inner_op)
    x === op.inner_op && return op
    return applicable(_kron_rewrap, op, x) ? _kron_rewrap(op, x) : op
end

_kron_rebuild(::OperatorAdd{D}, l, r) where {D} = OperatorAdd{D, typeof(l), typeof(r)}(l, r)
function _kron_rebuild(::BilinearProduct{D, I}, l, r) where {D, I}
    return BilinearProduct{D, I, typeof(l), typeof(r)}(l, r)
end

"""
    _kron_block_leaves(a, trial_leaves, test_leaves) -> Vector

One `(trial, test, scales, term)` per addend of `a`'s resolved AST and block it belongs to
(`block_of`): a term naming neither component once per diagonal block. `term` is unindexed
(`_kron_unindex`) and `scales` its constant coefficients (`_kron_leaves`).
"""
function _kron_block_leaves(a::BilinearForm, trial_leaves::Tuple, test_leaves::Tuple)
    nt, ns = length(trial_leaves), length(test_leaves)
    out = []
    for (scales, term) in _kron_leaves(resolve_form_ast(a), ())
        bc = block_of(term, nt, ns)
        u = _kron_unindex(term)
        for (tc, sc) in (bc === nothing ? [(c, c) for c in 1:min(nt, ns)] : [bc])
            push!(out, (tc, sc, scales, u))
        end
    end
    return out
end

# The mesh every leaf of both spaces lives on, or `nothing` when two differ.
function _kron_shared_mesh(trial_leaves::Tuple, test_leaves::Tuple)
    Ω = mesh(first(first(trial_leaves)))
    return all(l -> mesh(first(l)) === Ω, (trial_leaves..., test_leaves...)) ? Ω : nothing
end

function _kron_separable(Wu::_KronBlockSpace, Wv::_KronBlockSpace, a::BilinearForm)
    tl, sl = leaf_spaces_offsets(Wu), leaf_spaces_offsets(Wv)
    Ω = _kron_shared_mesh(tl, sl)
    Ω === nothing && return false
    _kron_check_spaces(a)
    Ωₕ = _host_mirror_mesh(Ω)
    return all(l -> _kron_project(l[4], Ωₕ) !== nothing, _kron_block_leaves(a, tl, sl))
end

# --- The operator -------------------------------------------------------------------- #

# One nonzero block: its operator, the rows (test leaf) and columns (trial leaf) it owns,
# and whether it is the first block on its rows, the one that applies `β` to them.
struct _KronBlock{K <: KroneckerLinearOperator}
    op::K
    rows::UnitRange{Int}
    cols::UnitRange{Int}
    first::Bool
end

"""
    KroneckerBlockOperator{T, B <: Tuple, R <: Tuple} <: AbstractMatrix{T}

The [`kronecker_operator`](@ref) of a bilinear form over a composite space whose leaves
share one mesh, not exported: one [`KroneckerLinearOperator`](@ref) per nonzero
(test leaf, trial leaf) block, each applied to the slice of `x` its trial leaf owns and
accumulated into the slice of `y` its test leaf owns. Rows of a test leaf no block reaches
(`idle`) are zero.

Supports `size`, `eltype`, `getindex`, three- and five-argument `mul!`, `Base.:*`,
`LinearAlgebra.issymmetric` and `SparseMatrixCSC(K)`, like `KroneckerLinearOperator`. Each
block applies under the execution policy of the form's backend, and the blocks run in a
fixed order, so the product is the same bit for bit under every host policy and a warm one
allocates nothing. `issymmetric` is `true` when every diagonal block is symmetric and every
other block is, term by term and factor by factor, the transpose of its mirror block.
Every block checks its mesh, so after an in-place mesh mutation the operator throws as a
stale `KroneckerLinearOperator` does.
"""
struct KroneckerBlockOperator{T, B <: Tuple, R <: Tuple} <: AbstractMatrix{T}
    blocks::B
    idle::R
    nrows::Int
    ncols::Int
end

@noinline function _throw_kron_block_meshes(trial_leaves::Tuple, test_leaves::Tuple)
    Ω = mesh(first(first(trial_leaves)))
    k = findfirst(l -> mesh(first(l)) !== Ω, trial_leaves)
    side, c = k === nothing ? ("test", findfirst(l -> mesh(first(l)) !== Ω, test_leaves)) :
              ("trial", k)
    throw(
        ArgumentError(
        "kronecker_operator: $side component $c lives on another mesh than trial " *
        "component 1; the Kronecker factors of a composite form are built on the one mesh " *
        "every leaf shares.",
    ),
    )
end

function _kron_operator(Wu::_KronBlockSpace, Wv::_KronBlockSpace, a::BilinearForm)
    tl, sl = leaf_spaces_offsets(Wu), leaf_spaces_offsets(Wv)
    Ω = _kron_shared_mesh(tl, sl)
    Ω === nothing && _throw_kron_block_meshes(tl, sl)
    _kron_check_spaces(a)
    cache = _kron_cache(Ω)
    be = backend(Wu)
    leaves = _kron_block_leaves(a, tl, sl)
    # Row by row (test leaf), then column by column (trial leaf).
    keys = sort!(unique([(l[2], l[1]) for l in leaves]))
    reads_coef = false
    blocks = _KronBlock[]
    for (sc, tc) in keys
        group = [(l[3], l[4]) for l in leaves if l[1] == tc && l[2] == sc]
        Wt, co = tl[tc]
        Ws, ro = sl[sc]
        K, r = _kron_build(cache, be, ndofs(Ws, Tuple), group,
            " (block trial component $tc, test component $sc)")
        reads_coef |= r
        first_on_rows = all(b -> b.rows != ro .+ (1:ndofs(Ws)), blocks)
        push!(blocks, _KronBlock(K, ro .+ (1:ndofs(Ws)), co .+ (1:ndofs(Wt)), first_on_rows))
    end
    reads_coef && _warn_kron_coefficient()
    idle = Tuple(o .+ (1:ndofs(W)) for (c, (W, o)) in enumerate(sl) if all(k -> k[1] != c, keys))
    T = promote_type(eltype(cache.mass[1]), map(b -> eltype(b.op), blocks)...)
    bs = Tuple(blocks)
    return KroneckerBlockOperator{T, typeof(bs), typeof(idle)}(bs, idle, ndofs(Wv), ndofs(Wu))
end

Base.size(K::KroneckerBlockOperator) = (K.nrows, K.ncols)

# --- The product --------------------------------------------------------------------- #

# `β == 0` overwrites (a `NaN` in `v` does not survive), `β == 1` leaves `v` as it is.
function _kron_scale!(v, β)
    if iszero(β)
        fill!(v, zero(eltype(v)))
    elseif !isone(β)
        v .*= β
    end
    return v
end

@inline _kron_blocks_apply!(loc, y, ::Tuple{}, x, α, β) = y
@inline function _kron_blocks_apply!(loc, y, bs::Tuple, x, α, β)
    b = bs[1]
    _kron_apply!(loc, view(y, b.rows), b.op, view(x, b.cols), α, b.first ? β : one(β))
    return _kron_blocks_apply!(loc, y, Base.tail(bs), x, α, β)
end

@noinline function _throw_kron_block_dimmismatch(K::KroneckerBlockOperator, x, y)
    throw(
        DimensionMismatch(
        "KroneckerBlockOperator of size $(size(K)) cannot multiply a vector of length " *
        "$(length(x)) into one of length $(length(y))",
    ),
    )
end

# `y = α * K * x + β * y` with `LinearAlgebra`'s semantics. Rows no block reaches are scaled
# by `β` here; the first block on a test leaf's rows applies `β` to them and every later one
# accumulates. The locality is `y`'s, passed down because each block sees a view
# (`_kron_apply!`). No three-argument method: `LinearAlgebra`'s own forwards here with
# `α = true, β = false`, and one of ours would be ambiguous against ReverseDiff's (see
# `kronecker.jl`).
function mul!(
        y::AbstractVector, K::KroneckerBlockOperator, x::AbstractVector, α::Number, β::Number
)
    # Every block, before the size check (a refined mesh changes the sizes) and before any
    # block writes: a stale one must leave `y` untouched.
    _kron_check_fresh(K)
    # Each block views `y` and `x` from 1.
    Base.require_one_based_indexing(y, x)
    (length(x) == K.ncols && length(y) == K.nrows) || _throw_kron_block_dimmismatch(K, x, y)
    for r in K.idle
        _kron_scale!(view(y, r), β)
    end
    return _kron_blocks_apply!(locality(typeof(y)), y, K.blocks, x, α, β)
end

# Whether every block's factors still describe the mesh (`kronecker.jl`).
@inline _kron_is_fresh(K::KroneckerBlockOperator) = all(b -> _kron_is_fresh(b.op), K.blocks)

function Base.show(io::IO, m::MIME"text/plain", K::KroneckerBlockOperator)
    _kron_is_fresh(K) && return invoke(show, Tuple{IO, MIME"text/plain", AbstractMatrix}, io, m, K)
    return _kron_show_stale(io, K)
end

# --- Inspection ---------------------------------------------------------------------- #

function Base.getindex(K::KroneckerBlockOperator{T}, i::Int, j::Int) where {T}
    _kron_check_fresh(K)
    @boundscheck checkbounds(K, i, j)
    total = zero(T)
    for b in K.blocks
        (i in b.rows && j in b.cols) || continue
        total += b.op[i - first(b.rows) + 1, j - first(b.cols) + 1]
    end
    return total
end

function SparseArrays.SparseMatrixCSC(K::KroneckerBlockOperator{T}) where {T}
    _kron_check_fresh(K)
    I = Int[]
    J = Int[]
    V = T[]
    for b in K.blocks
        i, j, v = SparseArrays.findnz(SparseMatrixCSC(b.op))
        append!(I, i .+ (first(b.rows) - 1))
        append!(J, j .+ (first(b.cols) - 1))
        append!(V, v)
    end
    return sparse(I, J, V, K.nrows, K.ncols)
end

# Whether `F` is `transpose(G)`, for two host factors; a device factor answers `false`.
_kron_is_transpose(F::AbstractMatrix, G::AbstractMatrix) = F == transpose(G)
_kron_is_transpose(::Any, ::Any) = false

# Whether two terms' coefficients are equal now and stay so: the same scales (a `Ref` the
# same object), or constant numbers with the same product (`1.0 * a` against `a`).
function _kron_same_scales(s::Tuple, t::Tuple)
    s == t && return true
    return all(c -> c isa Number, (s..., t...)) && _kron_coeff(s) == _kron_coeff(t)
end

# Whether `K` is `transpose(L)` term by term: the same coefficients, factor by factor
# transposed. A sum of terms that happens to match in another order answers `false`.
function _kron_transposes(K::KroneckerLinearOperator, L::KroneckerLinearOperator)
    length(K.terms) == length(L.terms) || return false
    return all(map(K.terms, L.terms) do s, t
        _kron_same_scales(s.scales, t.scales) && all(map(_kron_is_transpose, s.factors, t.factors))
    end)
end

function issymmetric(K::KroneckerBlockOperator)
    K.nrows == K.ncols || return false
    return all(K.blocks) do b
        b.rows == b.cols && return issymmetric(b.op)
        return any(c -> c.rows == b.cols && c.cols == b.rows && _kron_transposes(b.op, c.op),
            K.blocks)
    end
end
