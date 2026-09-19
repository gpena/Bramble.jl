# kronecker.jl
#
# `is_separable` and `KroneckerLinearOperator`: a matrix-free operator for a `BilinearForm`
# whose assembled matrix is an exact sum of Kronecker products of one-dimensional factors
# (gpena/Bramble.jl#162), so a 200^3 problem stores `3 * 200` numbers per factor instead of
# an `8_000_000^2`-entry sparse matrix.
#
# What this file recognises is deliberately narrow: a term is separable here only when it is
# `innerₕ(u, v)` (identity, i.e. a mass factor, on every axis) or `inner₊` of a backward
# difference along one axis on both sides (what `∇ₕ(u)`/`∇ₕ(v)` expand into, one term per
# axis, once `form` has resolved and simplified the AST -- see `simplifier.jl` and
# `operators/inner.jl`'s `inner_plus`). Both shapes factor as `H_D ⊗ ... ⊗ A_d ⊗ ... ⊗ H_1`:
# a difference/mass matrix on the touched axis, the plain mass (cell-measure) matrix on
# every other axis. A scalar coefficient (literal or `Ref`) wrapping a term does not break
# this -- it factors out of the whole Kronecker product -- so it is stripped and carried
# separately rather than being part of the shape match.
#
# Everything else is refused rather than approximated: a `GridFunctionScale` coefficient (no
# tensor structure), a `RegionRestriction` (Dirichlet rows included -- S5.2 handles
# constraints through the `Kronecker.jl` extension, not here), an `InterpolationNode`
# (cross-mesh, no per-axis submesh), an `InnerGamma` surface weight (a `(D-1)`-dimensional
# integral, no full-`D` factorisation), a composite space (leaves can have different
# meshes), a 1D mesh (nothing to factor), and any node this file does not explicitly
# recognise (forward/centered/star/cross-weighted differences, averages, jumps, a mixed
# multi-axis composition). A false negative here only forgoes the fast path; a false
# positive would build an operator that silently computes the wrong matrix-vector product.
#
# Dirichlet rows are out of scope for this operator: it has no boundary constraint of its
# own. `bramble-plan`'s v3.3.0 subplan S5.2 layers that on top, through the `Kronecker.jl`
# extension.

# --- Flattening a sum into (coefficient, term) pairs -------------------------------- #

"""
    _kron_leaves(op, scales::Tuple) -> Tuple

Flatten `op`'s top-level `OperatorAdd` sum into `(scales, term)` pairs, one per addend,
pushing every scalar factor found along the way -- a literal number or a `Ref` -- into
`scales` instead of leaving it wrapped around the sum. `simplify_ast` (`simplifier.jl`)
already lifts a term's own scalar all the way out (`⟨c * u, v⟩ -> c * ⟨u, v⟩`) and factors a
shared one out of a sum (`c*A + c*B -> c*(A+B)`), so a coefficient can sit above several
addends at once; this walk is what puts it back beside each one without rebuilding the AST.

`_separable_axis` classifies `term` alone; [`is_separable`](@ref) and
[`kronecker_operator`](@ref) multiply by `scales` (via `_kron_coeff`) at `mul!` time, the
same way a live `Ref` coefficient stays live through `assemble!`.
"""
@inline _kron_leaves(op::OperatorAdd, scales::Tuple) = (
    _kron_leaves(op.left_op, scales)..., _kron_leaves(op.right_op, scales)...
)
@inline _kron_leaves(op::OperatorScale, scales::Tuple) = _kron_leaves(op.inner_op, (scales..., op.scalar))
@inline _kron_leaves(op::LazyOp, scales::Tuple) = ((scales, op),)

@inline _kron_coeff_factor(c::Number) = c
@inline _kron_coeff_factor(c::Base.RefValue{<:Number}) = c[]

# `init = 1.0` rather than `true` (the usual empty-product identity elsewhere in this
# package): every use multiplies straight into a `Float64` accumulator, and an empty
# `scales` tuple is the common case (a term with no wrapping scalar at all).
@inline _kron_coeff(scales::Tuple) = prod(_kron_coeff_factor, scales; init = 1.0)

# --- Term classification ------------------------------------------------------------- #

"""
    _separable_axis(term) -> Union{Nothing, Some{Union{Int, Nothing}}}

The axis a recognised separable `term` acts along, wrapped in `Some` to tell "valid, no
axis" apart from "not recognised": `Some(nothing)` for a mass term (`innerₕ(u, v)`,
identity on every axis) and `Some(d)` for a single-direction term (`inner₊` of a backward
difference along axis `d` on both sides -- what `∇ₕ` expands to, one term per axis). Plain
`nothing` for anything else.

Conservative by construction: only the two `BilinearProduct` shapes below have a method:
a bare mass product over plain (non-indexed) trial/test functions, and a directional
product over a `BackwardDifference` wrapping each. Every other node -- including a
`GridFunctionScale`, `RegionRestriction`, `InterpolationNode`, an `InnerGamma` weight, an
indexed (composite) trial or test function, or any other difference/average/jump family --
falls through to the `LazyOp` fallback and answers `nothing`, never a guess.
"""
_separable_axis(::LazyOp) = nothing

@inline function _separable_axis(
        ::BilinearProduct{D, InnerH, <:TrialFunction{D}, <:TestFunction{D}}
) where {D}
    return Some(nothing)
end

@inline function _separable_axis(
        ::BilinearProduct{
        D, InnerPlus{Dim}, <:BackwardDifference{D, Dim, <:TrialFunction{D}},
        <:BackwardDifference{D, Dim, <:TestFunction{D}}
}
) where {D, Dim}
    return Some(Dim)
end

"""
    is_separable(a::BilinearForm) -> Bool

Whether `a`'s resolved AST is a sum of terms each expressible as a Kronecker product of
one-dimensional factors, `H_D ⊗ ... ⊗ A_d ⊗ ... ⊗ H_1`, over a `MeshnD`.

`true` requires every one of the following:

  - `a`'s trial and test space are both a (non-composite) [`ScalarGridSpace`](@ref) sharing
    one mesh, and that mesh is at least two-dimensional (a 1D mesh has nothing to factor).
  - Every addend of the resolved AST, after stripping any constant (literal or `Ref`)
    scalar coefficient, is `innerₕ(u, v)` or `inner₊` of a `D₋` backward difference along
    one axis on both sides -- what `innerₕ(u, v)` and `inner₊(∇ₕ(u), ∇ₕ(v))` resolve to.

A grid-function coefficient, a region restriction (Dirichlet included), an interpolation, a
surface (`InnerGamma`) weight, or any operator family this file does not explicitly
recognise (forward/centered/star/cross-weighted differences, averages, jumps, a mixed
multi-axis composition) all answer `false` -- conservatively: a false negative only forgoes
the Kronecker fast path, so this never claims separability it cannot back up with factors.

# Examples

```julia
Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 7), (false, false))
Wₕ = gridspace(Ωₕ)
is_separable(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))))  # true

fₕ = Rₕ(Wₕ, x -> 1 + x[1])
is_separable(form(Wₕ, Wₕ, (u, v) -> innerₕ(fₕ * u, v)))  # false: grid-function coefficient
```

See also: [`kronecker_operator`](@ref), [`KroneckerLinearOperator`](@ref).
"""
function is_separable(a::BilinearForm{D}) where {D}
    D == 1 && return false
    Wu = trial_space(a)
    Wv = test_space(a)
    Wu isa ScalarGridSpace || return false
    Wv isa ScalarGridSpace || return false
    mesh(Wu) === mesh(Wv) || return false
    for (_, term) in _kron_leaves(resolve_form_ast(a), ())
        _separable_axis(term) === nothing && return false
    end
    return true
end

# --- KroneckerTerm: one addend's per-axis factors ------------------------------------ #

"""
    KroneckerTerm{D, S <: Tuple, F <: Tuple}

One separable addend's `D` one-dimensional factor matrices and the (possibly still-`Ref`)
scalar coefficients multiplying it: [`KroneckerLinearOperator`](@ref)'s building block, not
exported. `factors[d]` is either the assembled 1D difference matrix on the touched axis or
the diagonal mass matrix on every other axis, in the order [`kronecker_operator`](@ref)
built them; `scales` is read at `mul!` time through `_kron_coeff` so a `Ref` coefficient
stays live, matching a `BilinearForm`'s own contract.
"""
struct KroneckerTerm{D, S <: Tuple, F <: Tuple}
    scales::S
    factors::F
end

"""
    KroneckerLinearOperator{T, D, TermsT <: Tuple}

A matrix-free linear operator for a separable [`BilinearForm`](@ref) (see
[`is_separable`](@ref)): the sum, over its terms, of a Kronecker product of `D`
one-dimensional factor matrices, applied by sum factorisation
(`LinearAlgebra.mul!(y, K, x)`) rather than ever materialising the `D`-dimensional matrix.
For a `200^3` mesh the factors together hold `O(200)` numbers per axis instead of the
assembled matrix's `O(200^3)` stored entries.

Build one with [`kronecker_operator`](@ref). Subtypes `AbstractMatrix{T}` so it plugs into
`LinearProblem`/`KrylovJL_CG` (`LinearSolve.jl`) the same way an assembled matrix does, and
supports `size`, `eltype`, `getindex`, `Base.:*`, `LinearAlgebra.issymmetric`, and
`SparseMatrixCSC(K)` (an explicit `kron` of the factors, for testing and inspection -- the
very matrix this operator avoids forming).

The two `n`-length work buffers `mul!` needs for sum factorisation are grown on the first
call rather than at construction (`resize!`, starting from an empty vector), so a freshly
built `K` that has never multiplied anything costs only its `D` one-dimensional factors --
this is what keeps `Base.summarysize(K)` small immediately after
[`kronecker_operator`](@ref) returns. `mul!` still allocates nothing **after that first
call**, matching every other zero-allocation refill in this package.

Dirichlet rows are out of scope: this operator carries no boundary constraint of its own.
`bramble-plan`'s v3.3.0 subplan S5.2 layers that on top, through the `Kronecker.jl`
extension and its fast-diagonalisation solve.

See also: [`is_separable`](@ref), [`kronecker_operator`](@ref).
"""
struct KroneckerLinearOperator{T, D, TermsT <: Tuple} <: AbstractMatrix{T}
    terms::TermsT
    dims::NTuple{D, Int}
    n::Int
    buf1::Vector{T}
    buf2::Vector{T}
end

@noinline function _throw_not_separable_dim(D::Int)
    throw(
        ArgumentError(
        "kronecker_operator needs at least two dimensions to factor a Kronecker product " *
        "from; got a $(D)D form, which has nothing to factor.",
    ),
    )
end

@noinline function _throw_not_separable_space(Wu, Wv)
    throw(
        ArgumentError(
        "kronecker_operator only supports a scalar (non-composite) trial and test space " *
        "sharing one mesh; got $(typeof(Wu)) and $(typeof(Wv)).",
    ),
    )
end

@noinline function _throw_not_separable_term(term)
    throw(
        ArgumentError(
        "kronecker_operator: the term $(typeof(term)) is not one of the recognised " *
        "separable shapes (innerₕ(u, v), or inner₊ of a backward difference along one " *
        "axis on both sides -- what ∇ₕ(u)/∇ₕ(v) expand into). A grid-function " *
        "coefficient, a region restriction (Dirichlet included), an interpolation, a " *
        "surface weight, a composite space, or a mixed/forward/centered/averaged/jump " *
        "operator all fall outside what this file builds Kronecker factors for.",
    ),
    )
end

"""
    kronecker_operator(a::BilinearForm) -> KroneckerLinearOperator

Build a matrix-free [`KroneckerLinearOperator`](@ref) for the separable bilinear form `a`
(see [`is_separable`](@ref)), without ever assembling the `D`-dimensional matrix.

For each axis `d`, the per-axis mass factor is the diagonal matrix of `d`'s cell measures
(`weights(gridspace(Ωₕ(d)), Innerh())`); the factor on a term's touched axis is instead the
assembled 1D operator `assemble(form(Wₕd, Wₕd, (u, v) -> inner₊(D₋ₓ(u), D₋ₓ(v))))` over
`Wₕd = gridspace(Ωₕ(d))`, cached across terms that share an axis.

# Throws

  - `ArgumentError`: `a` is not separable, naming the offending term (or dimension, or
    space) -- the same check [`is_separable`](@ref) runs, made specific.

# Examples

```julia
Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 7), (false, false))
Wₕ = gridspace(Ωₕ)
a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
K = kronecker_operator(a)
x = rand(ndofs(Wₕ))
K * x ≈ assemble(a) * x
```

See also: [`is_separable`](@ref), [`KroneckerLinearOperator`](@ref).
"""
function kronecker_operator(a::BilinearForm{D}) where {D}
    D == 1 && _throw_not_separable_dim(D)
    Wu = trial_space(a)
    Wv = test_space(a)
    (Wu isa ScalarGridSpace && Wv isa ScalarGridSpace) || _throw_not_separable_space(Wu, Wv)
    mesh(Wu) === mesh(Wv) || _throw_not_separable_space(Wu, Wv)

    Ωₕ = mesh(Wu)
    axis_spaces = ntuple(d -> gridspace(Ωₕ(d)), Val(D))
    mass_vecs = ntuple(d -> weights(axis_spaces[d], Innerh()), Val(D))

    # Cached across terms that touch the same axis, since two directional terms along the
    # same axis (rare, but not disallowed) would otherwise assemble the identical 1D
    # operator twice.
    diff_mats = Dict{Int, Any}()

    leaves = _kron_leaves(resolve_form_ast(a), ())
    terms = map(leaves) do leaf
        scales, term = leaf
        axis_opt = _separable_axis(term)
        axis_opt === nothing && _throw_not_separable_term(term)
        axis = something(axis_opt)
        factors = ntuple(Val(D)) do d
            if axis !== nothing && d == axis
                get!(diff_mats, d) do
                    Wd = axis_spaces[d]
                    assemble(form(Wd, Wd, (u, v) -> inner₊(D₋ₓ(u), D₋ₓ(v))))
                end
            else
                Diagonal(mass_vecs[d])
            end
        end
        KroneckerTerm{D, typeof(scales), typeof(factors)}(scales, factors)
    end

    dims = ndofs(Wu, Tuple)
    n = ndofs(Wu)
    T = eltype(mass_vecs[1])
    # Empty, not `Vector{T}(undef, n)`: `mul!` grows them to `n` on its first call
    # (`_kron_ensure_buffers!`), so a freshly built `K` costs only its factors.
    return KroneckerLinearOperator{T, D, typeof(terms)}(
        terms, dims, n, Vector{T}(undef, 0), Vector{T}(undef, 0)
    )
end

# --- Sum factorisation: mode-d contraction on a flat buffer -------------------------- #
#
# A `D`-array `X` of shape `dims`, flattened column-major into a `Vector`, is contracted
# along axis `d` by a factor `F` (size `dims[d] x dims[d]`) by viewing the flat index as
# `(i, l, k)` with `i in 1:pre`, `l in 1:dims[d]`, `k in 1:post` (`pre = prod(dims[1:d-1])`,
# `post = prod(dims[d+1:end])`) and computing `Y[i, j, k] = sum_l F[j, l] * X[i, l, k]`.
# Written against the flat index directly (`base + offset`) rather than `reshape`, which
# allocates a new `Array` header on every call -- the same reason `_fold_taps`
# (`stencil_eval.jl`) recurses over tuple structure instead of building an intermediate
# array. `_kron_apply_axes!` peels one axis off `factors`/`dims` per call the same way
# `_fold_taps` peels one tap: the compiler sees each `factors[1]` at a concrete,
# non-abstract type and unrolls the whole `D`-axis loop with no dynamic dispatch.

@inline function _kron_apply_mode!(
        Y::Vector{T}, F::Diagonal, X::Vector{T}, pre::Int, m::Int, post::Int
) where {T}
    d = F.diag
    @inbounds for k in 0:(post - 1)
        base = k * pre * m
        for l in 1:m
            dl = d[l]
            off = base + (l - 1) * pre
            @simd for i in 1:pre
                Y[off + i] = dl * X[off + i]
            end
        end
    end
    return Y
end

@inline function _kron_apply_mode!(
        Y::Vector{T}, F::SparseMatrixCSC, X::Vector{T}, pre::Int, m::Int, post::Int
) where {T}
    fill!(Y, zero(T))
    rows = rowvals(F)
    vals = nonzeros(F)
    @inbounds for k in 0:(post - 1)
        base = k * pre * m
        for l in 1:m
            xoff = base + (l - 1) * pre
            for idx in nzrange(F, l)
                j = rows[idx]
                v = vals[idx]
                yoff = base + (j - 1) * pre
                @simd for i in 1:pre
                    Y[yoff + i] += v * X[xoff + i]
                end
            end
        end
    end
    return Y
end

# One-axis-left base case: writes the result into `nxt` and returns it.
@inline function _kron_apply_axes!(
        cur::Vector{T}, nxt::Vector{T}, factors::Tuple{Any}, dims::Tuple{Any}, pre::Int, n::Int
) where {T}
    m = dims[1]
    post = n ÷ (pre * m)
    _kron_apply_mode!(nxt, factors[1], cur, pre, m, post)
    return nxt
end

@inline function _kron_apply_axes!(
        cur::Vector{T}, nxt::Vector{T}, factors::Tuple, dims::Tuple, pre::Int, n::Int
) where {T}
    m = dims[1]
    post = n ÷ (pre * m)
    _kron_apply_mode!(nxt, factors[1], cur, pre, m, post)
    return _kron_apply_axes!(nxt, cur, Base.tail(factors), Base.tail(dims), pre * m, n)
end

# Copies `x` into `buf1` (the first axis reads from it, never mutating `x` itself), then
# ping-pongs `buf1`/`buf2` one axis at a time, returning whichever one ends up holding the
# `D`-th axis's result -- `mul!` reads it back rather than assuming a fixed parity.
@inline function _kron_apply_term!(
        buf1::Vector{T}, buf2::Vector{T}, term::KroneckerTerm, x::AbstractVector, dims::Tuple
) where {T}
    copyto!(buf1, x)
    return _kron_apply_axes!(buf1, buf2, term.factors, dims, 1, length(buf1))
end

# One term's contribution accumulated into `y`, peeling `K.terms` (a heterogeneous `Tuple`
# -- each term's factors are a different concrete `SparseMatrixCSC`/`Diagonal` mix) the same
# way `_kron_apply_axes!` peels `factors`, rather than `for term in K.terms`: the latter
# gives `term` a small-`Union` type across iterations, and this package's zero-allocation
# contract is measured, not assumed (`bramble-verification` #1) -- a `for` loop here
# measured a nonzero `@allocated` on the very mixed-factor terms this operator exists for.
@inline function _kron_accumulate!(
        y::AbstractVector, buf1::Vector{T}, buf2::Vector{T}, terms::Tuple{Any}, x::AbstractVector, dims::Tuple, n::Int
) where {T}
    term = terms[1]
    cur = _kron_apply_term!(buf1, buf2, term, x, dims)
    c = _kron_coeff(term.scales)
    @inbounds @simd for i in 1:n
        y[i] += c * cur[i]
    end
    return y
end

@inline function _kron_accumulate!(
        y::AbstractVector, buf1::Vector{T}, buf2::Vector{T}, terms::Tuple, x::AbstractVector, dims::Tuple, n::Int
) where {T}
    term = terms[1]
    cur = _kron_apply_term!(buf1, buf2, term, x, dims)
    c = _kron_coeff(term.scales)
    @inbounds @simd for i in 1:n
        y[i] += c * cur[i]
    end
    return _kron_accumulate!(y, buf1, buf2, Base.tail(terms), x, dims, n)
end

@noinline function _throw_kron_dimmismatch(K::KroneckerLinearOperator, x, y)
    throw(
        DimensionMismatch(
        "KroneckerLinearOperator of size $(size(K)) cannot multiply a vector of length " *
        "$(length(x)) into one of length $(length(y))",
    ),
    )
end

# Grows the two scratch buffers to `n` on the first call and never again -- see
# `KroneckerLinearOperator`'s own docstring for why they start empty.
@inline function _kron_ensure_buffers!(K::KroneckerLinearOperator, n::Int)
    if length(K.buf1) != n
        resize!(K.buf1, n)
        resize!(K.buf2, n)
    end
    return nothing
end

function mul!(y::AbstractVector, K::KroneckerLinearOperator{T}, x::AbstractVector) where {T}
    n = K.n
    (length(x) == n && length(y) == n) || _throw_kron_dimmismatch(K, x, y)
    _kron_ensure_buffers!(K, n)
    fill!(y, zero(eltype(y)))
    _kron_accumulate!(y, K.buf1, K.buf2, K.terms, x, K.dims, n)
    return y
end

Base.size(K::KroneckerLinearOperator) = (K.n, K.n)
Base.size(K::KroneckerLinearOperator, i::Integer) = i in (1, 2) ? K.n : 1
Base.eltype(::KroneckerLinearOperator{T}) where {T} = T

"""
    getindex(K::KroneckerLinearOperator, i::Int, j::Int) -> Number

The `(i, j)` entry of the assembled matrix `K` stands for, read off the Kronecker
structure directly (`sum` over terms of the coefficient times the product of each factor's
own `(i_d, j_d)` entry) rather than through `mul!` -- `AbstractMatrix`'s minimal interface,
so `K` prints and indexes like the matrix it factors.
"""
function Base.getindex(K::KroneckerLinearOperator{T, D}, i::Int, j::Int) where {T, D}
    @boundscheck checkbounds(K, i, j)
    Ic = CartesianIndices(K.dims)[i]
    Jc = CartesianIndices(K.dims)[j]
    total = zero(T)
    for term in K.terms
        c = _kron_coeff(term.scales)
        p = one(T)
        for d in 1:D
            p *= term.factors[d][Ic[d], Jc[d]]
        end
        total += c * p
    end
    return total
end

function Base.:*(K::KroneckerLinearOperator{T}, x::AbstractVector) where {T}
    y = Vector{promote_type(T, eltype(x))}(undef, K.n)
    mul!(y, K, x)
    return y
end

"""
    issymmetric(K::KroneckerLinearOperator) -> Bool

Always `true`: [`kronecker_operator`](@ref) only ever builds a term from `innerₕ(u, v)` or
`inner₊` of the *same* backward difference on the trial and the test side, so every
Kronecker factor -- and therefore every term, and their sum -- is symmetric.
"""
issymmetric(::KroneckerLinearOperator) = true

_kron_as_sparse(F::SparseMatrixCSC) = F
_kron_as_sparse(F::Diagonal) = sparse(F)

"""
    SparseMatrixCSC(K::KroneckerLinearOperator) -> SparseMatrixCSC

Materialise `K` as an explicit sparse matrix: the sum, over its terms, of the coefficient
times the Kronecker product of its `D` one-dimensional factors, last axis leftmost
(`A_2D = H_y ⊗ A_x + A_y ⊗ H_x`, matching gpena/Bramble.jl#162's own formula). For testing
and inspection only -- this is exactly the `D`-dimensional matrix [`kronecker_operator`](@ref)
is built to avoid forming.
"""
function SparseArrays.SparseMatrixCSC(K::KroneckerLinearOperator{T}) where {T}
    A = spzeros(T, K.n, K.n)
    for term in K.terms
        c = _kron_coeff(term.scales)
        Aterm = foldl(kron, reverse(map(_kron_as_sparse, term.factors)))
        A = A + c * Aterm
    end
    return A
end
