# Rank-one penalty terms `σ * innerₕ(g, u) * innerₕ(h, v)`.
#
# A product of a scalar functional of the trial function and one of the test function is
# the matrix `σ wₛ wₜᵀ`, with `wₛ` the vector of the test functional and `wₜ` that of the
# trial one. Every entry is nonzero, so it never reaches the stencil walk: `form` splits it
# off the expression (`_rank_one_form`), assembles its two vectors through the linear path,
# and `assemble` returns the sparse part and the two vectors together, as a
# `RankOneUpdate`. Storing it in the sparse matrix would cost `N²` entries.

# --- The term ------------------------------------------------------------------------ #

"""
    RankOneTerm{D, S, FT, FS} <: LazyOp{D}

The product `σ * ft(u) * fs(v)` of a scalar functional `ft` of the trial function and a
scalar functional `fs` of the test function, as `*` builds it from
`σ * innerₕ(g, u) * innerₕ(h, v)`. `trial` and `test` are the two functionals, each an
expression of `LinearProduct`s (`innerₕ(g, ·)`, `inner₊(g, ·)`, sums, scales and restrictions
of them). `scale` is `σ` as a tuple, a number first and then any `Ref`s, read at assembly:
every constant factor of the product is folded into the number when `*` builds the term.

`form` splits it off the expression before the expression is resolved, so no other part of
the form layer sees it.
"""
struct RankOneTerm{D, S <: Tuple, FT <: LazyOp{D}, FS <: LazyOp{D}} <: LazyOp{D}
    scale::S
    trial::FT
    test::FS
end

const _RankScalar = Union{Number, Base.RefValue{<:Number}}

# `σ`'s value: the number times every `Ref`'s current value, so a `Ref` stays live.
@inline _rank_scale_value(s::Tuple) = _rank_scale_value(s[1], Base.tail(s)...)
@inline _rank_scale_value(c::Number) = c
@inline _rank_scale_value(c::Number, r::Base.RefValue, rs...) = _rank_scale_value(c * r[], rs...)

@inline _push_scale(s::Tuple, c::Number) = (s[1] * c, Base.tail(s)...)
@inline _push_scale(s::Tuple, r::Base.RefValue) = (s..., r)
@inline _join_scales(a::Tuple, b::Tuple) = (a[1] * b[1], Base.tail(a)..., Base.tail(b)...)

# The constant and `Ref` factors around a functional, peeled off into `σ`.
@inline _peel_scales(op::OperatorScale{D, <:_RankScalar}, s::Tuple) where {D} = _peel_scales(
    op.inner_op, _push_scale(s, op.scalar))
@inline _peel_scales(op, s::Tuple) = (s, op)

# Whether the node type `T` or a node below it is an `N`. Only node parameters are read,
# never a coefficient's (a closure's captured form holds `LinearProduct`s of its own).
# Defined before the generated functions below, which call it.
function _type_holds(T, ::Type{N}) where {N}
    T isa DataType && T <: LazyOp || return false
    T <: N && return true
    return any(P -> _type_holds(P, N), T.parameters)
end

# Which function the leaves of `T` are: `(holds a trial leaf, holds a test leaf)`.
function _type_leaf_sides(T)
    T isa DataType && T <: LazyOp || return (false, false)
    T <: Union{TrialFunction, IndexedTrialFunction} && return (true, false)
    T <: Union{TestFunction, IndexedTestFunction} && return (false, true)
    trial = test = false
    for P in T.parameters
        a, b = _type_leaf_sides(P)
        trial |= a
        test |= b
    end
    return (trial, test)
end

# `:trial` or `:test` when `T` is a scalar functional of that function alone: a
# `LinearProduct` over it, or a sum, constant scale or restriction of such products of one
# side. `:none` for anything else (a bilinear product, a bare function, a grid function
# scale around a functional, a functional of both, a product inside a functional's operand).
function _type_functional_side(T)
    T isa DataType || return :none
    if T <: LinearProduct
        _type_holds(T.parameters[4], Union{BilinearProduct, LinearProduct}) && return :none
        trial, test = _type_leaf_sides(T.parameters[4])
        return trial == test ? :none : (trial ? :trial : :test)
    elseif T <: OperatorAdd
        l = _type_functional_side(T.parameters[2])
        return l === _type_functional_side(T.parameters[3]) ? l : :none
    elseif T <: OperatorScale
        T.parameters[2] <: _RankScalar || return :none
        return _type_functional_side(T.parameters[3])
    elseif T <: RegionRestriction
        return _type_functional_side(T.parameters[3])
    end
    return :none
end

@generated _functional_side(::T) where {T} = :(Val{$(QuoteNode(_type_functional_side(T)))}())

# The product of two operators, accepted only as a rank-one term: a scalar functional of the
# trial function times one of the test function, in either order, each with any constant or
# `Ref` factors, which are folded into `σ`. Two functionals of one function, or a factor that
# is no functional (a bilinear product, a bare trial or test function, a rank-one term),
# throw an `ArgumentError` naming the form a rank-one term takes.
@inline function Base.:*(a::LazyOp{D}, b::LazyOp{D}) where {D}
    sa, fa = _peel_scales(a, (1,))
    sb, fb = _peel_scales(b, (1,))
    return _rank_one_product(_functional_side(fa), _functional_side(fb), sa, sb, fa, fb)
end

@inline function _rank_one_product(::Val{:trial}, ::Val{:test}, sa, sb, fa, fb)
    s = _join_scales(sa, sb)
    return RankOneTerm{_rank_dim(fa), typeof(s), typeof(fa), typeof(fb)}(s, fa, fb)
end
@inline function _rank_one_product(::Val{:test}, ::Val{:trial}, sa, sb, fa, fb)
    s = _join_scales(sa, sb)
    return RankOneTerm{_rank_dim(fa), typeof(s), typeof(fb), typeof(fa)}(s, fb, fa)
end
@inline _rank_one_product(::Val{S}, ::Val{S}, sa, sb, fa, fb) where {S} = _throw_same_side(S)
@inline _rank_one_product(::Val{:none}, ::Val{:none}, sa, sb, fa, fb) = _throw_not_functional(
    "left", fa)
@inline _rank_one_product(::Val{:none}, ::Val, sa, sb, fa, fb) = _throw_not_functional(
    "left", fa)
@inline _rank_one_product(::Val, ::Val{:none}, sa, sb, fa, fb) = _throw_not_functional(
    "right", fb)

@inline _rank_dim(::LazyOp{D}) where {D} = D

@noinline function _throw_same_side(side::Symbol)
    f = side === :trial ? "u" : "v"
    throw(
        ArgumentError(
        "innerₕ(g, $f) * innerₕ(h, $f) multiplies two functionals of the $side function, " *
        "so it is not bilinear in (u, v). A rank-one term multiplies a functional of the " *
        "trial function by one of the test function: write σ * innerₕ(g, u) * innerₕ(h, v).",
    ),
    )
end

@noinline function _throw_not_functional(position::String, @nospecialize(op))
    throw(
        ArgumentError(
        "a product of two operators in a form is only accepted as a rank-one term, a " *
        "scalar times a functional of the trial function times a functional of the test " *
        "function: σ * innerₕ(g, u) * innerₕ(h, v), with g and h data. Its $position " *
        "factor, a $(nameof(typeof(op))), is not such a functional. Scale an operator by a " *
        "grid function as g * u, and write a functional times a bilinear term as two " *
        "separate terms.",
    ),
    )
end

# --- Splitting a form's expression ---------------------------------------------------- #

# Read off the type, so a form without the term pays one constant `false`.
@generated _holds_rank_one(::T) where {T} = _type_holds(T, RankOneTerm)
@generated _holds_functional(::T) where {T} = _type_holds(T, LinearProduct)

# `(rest, terms)`: the expression without its rank-one terms (`nothing` if none is left),
# and those terms, each with the constant and `Ref` scales above it folded into its `σ`.
# Sums and scales are rebuilt around what is left; any other node is returned as it was
# built, and refused if a term sits below it.
function _split_rank_one(op)
    _holds_rank_one(op) || return (op, ())
    return _throw_nested_rank_one(op)
end
_split_rank_one(t::RankOneTerm) = (nothing, (t,))
function _split_rank_one(op::OperatorAdd)
    l, tl = _split_rank_one(op.left_op)
    r, tr = _split_rank_one(op.right_op)
    return (_add_rest(l, r), (tl..., tr...))
end
function _split_rank_one(op::OperatorScale{D, <:_RankScalar}) where {D}
    r, ts = _split_rank_one(op.inner_op)
    rest = r === nothing ? nothing : OperatorScale(op.scalar, r)
    return (rest, map(t -> _scale_term(t, op.scalar), ts))
end

@inline _scale_term(t::RankOneTerm{D}, c) where {D} = (s = _push_scale(t.scale, c);
    RankOneTerm{D, typeof(s), typeof(t.trial), typeof(t.test)}(s, t.trial, t.test))

_add_rest(::Nothing, ::Nothing) = nothing
_add_rest(l, ::Nothing) = l
_add_rest(::Nothing, r) = r
_add_rest(l, r) = OperatorAdd(l, r)

@noinline function _throw_nested_rank_one(@nospecialize(op))
    throw(
        ArgumentError(
        "a rank-one term σ * innerₕ(g, u) * innerₕ(h, v) can only be added to a form's " *
        "other terms or scaled by a number or a Ref, and here it sits inside a " *
        "$(nameof(typeof(op))). Write it as a separate term of the form.",
    ),
    )
end

@noinline function _throw_several_rank_one(n::Int)
    throw(
        ArgumentError(
        "this form holds $n rank-one terms and takes at most one. Terms sharing their " *
        "functionals are one term with their scalars added: " *
        "(σ₁ + σ₂) * innerₕ(g, u) * innerₕ(h, v). A sum of terms with different " *
        "functionals is a rank-$n update, which assemble does not store.",
    ),
    )
end

# A functional alone in a bilinear form is linear in one argument only. Refused at `form`
# rather than at the walk, which has no stencil for it.
@inline _check_bilinear_terms(op) = _holds_functional(op) ? _throw_lone_functional() : nothing

@noinline function _throw_lone_functional()
    throw(
        ArgumentError(
        "a bilinear form's terms have to be linear in both u and v, and this one holds a " *
        "functional of one of them alone (innerₕ(g, u) or innerₕ(g, v)). A source term " *
        "belongs in a linear form, form(Vₕ, v -> innerₕ(g, v)); a functional of u times a " *
        "functional of v is a rank-one term, σ * innerₕ(g, u) * innerₕ(h, v).",
    ),
    )
end

# The trial functional as a functional of the test function of the same space, so the
# linear path assembles its vector.
_as_test(::TrialFunction{D, N}) where {D, N} = TestFunction{D, N}()
_as_test(op::IndexedTrialFunction{D}) where {D} = IndexedTestFunction{D}(op.component_idx)
function _as_test(op::LinearProduct{D, W, L}) where {D, W, L}
    r = _as_test(op.right_op)
    return LinearProduct{D, W, L, typeof(r)}(op.left_op, r)
end
_as_test(op::OperatorAdd) = OperatorAdd(_as_test(op.left_op), _as_test(op.right_op))
_as_test(op::UnaryWrapper) = _rewrap_inner(op, _as_test(op.inner_op))

# Ghost spaces assemble on their padded space and reduce; the rank-one part's reduction is
# not written yet.
_holds_ghost_leaf(Wₕ) = any(((s, _),) -> mesh(s) isa GhostMesh, leaf_spaces_offsets(Wₕ))

@noinline function _throw_rank_one_ghost()
    throw(
        ArgumentError(
        "rank-one terms σ * innerₕ(g, u) * innerₕ(h, v) are not supported on ghost " *
        "spaces yet. Build the form on a space without ghosts, or impose the condition " *
        "the term penalises with pde_solve(A, F; mean_zero = w).",
    ),
    )
end

# --- The form ------------------------------------------------------------------------- #

"""
    RankOneForm{D, B, S, LS, LT}

What [`form`](@ref) returns for a bilinear expression holding a rank-one term
`σ * innerₕ(g, u) * innerₕ(h, v)`: the rest of the expression as a [`BilinearForm`](@ref)
(`bilinear`), `σ` as the term holds it (`scale`, see [`RankOneTerm`](@ref)), and the two
functionals as [`LinearForm`](@ref)s, `test_functional` over the test space and
`trial_functional` over the trial space. [`assemble`](@ref) gives a
[`RankOneUpdate`](@ref).
"""
struct RankOneForm{D, B <: BilinearForm{D}, S <: Tuple, LS <: LinearForm, LT <: LinearForm}
    bilinear::B
    scale::S
    test_functional::LS
    trial_functional::LT
end

trial_space(a::RankOneForm) = trial_space(a.bilinear)
test_space(a::RankOneForm) = test_space(a.bilinear)

function _rank_one_form(Wₕ, Vₕ, raw_ast)
    (_holds_ghost_leaf(Wₕ) || _holds_ghost_leaf(Vₕ)) && _throw_rank_one_ghost()
    rest, terms = _split_rank_one(raw_ast)
    length(terms) == 1 || _throw_several_rank_one(length(terms))
    t = only(terms)
    ft = _as_test(t.trial)
    D = dim(Wₕ)
    a = _bilinear_form(Wₕ, Vₕ, rest === nothing ? ZeroOperator{D, Nothing}(nothing) : rest)
    ls = form(Vₕ, Returns(t.test))
    lt = form(Wₕ, Returns(ft))
    return RankOneForm{D, typeof(a), typeof(t.scale), typeof(ls), typeof(lt)}(
        a, t.scale, ls, lt)
end

function Base.show(io::IO, a::RankOneForm{D}) where {D}
    print(io, "RankOneForm{$(D)D, ", ndofs(test_space(a)), "×", ndofs(trial_space(a)), "}")
    return nothing
end

# The form's other terms as a `BilinearForm` shows them, then the rank-one term's `σ`.
function Base.show(io::IO, ::MIME"text/plain", a::RankOneForm)
    println(io, "RankOneForm: the bilinear form below + σ wₛ wₜᵀ, σ = ",
        _rank_scale_value(a.scale))
    show(io, MIME"text/plain"(), a.bilinear)
    return nothing
end

# --- The assembled operator ------------------------------------------------------------ #

"""
    RankOneUpdate{T, M, VS, VT} <: AbstractMatrix{T}
    RankOneUpdate(A, σ, ws, wt)
    RankOneUpdate{T}(A, σ, ws, wt)

The matrix `A + σ wₛ wₜᵀ`, stored as its four parts and never as a dense block: what
[`assemble`](@ref) returns for a form holding a rank-one term
`σ * innerₕ(g, u) * innerₕ(h, v)`. `A` is the matrix of the form's other terms, `wₛ` the
vector of the test functional `innerₕ(h, ·)` and `wₜ` that of the trial functional
`innerₕ(g, ·)`, each assembled by the linear path (`assemble(form(Vₕ, v -> innerₕ(h, v)))`).

# Fields
- `A::M`: the sparse part, an `AbstractMatrix` of size `(length(ws), length(wt))`.
- `σ::T`: the scalar, read when the form is assembled; a `Ref` factor stays live, so
  [`assemble!`](@ref) reads its new value.
- `ws::VS`: the test-side (row) vector.
- `wt::VT`: the trial-side (column) vector.

`assemble!(K, a)` refills `A`, `ws`, `wt` and `σ` in place, allocating nothing.
`RankOneUpdate(A, σ, ws, wt)` takes `T` from the four parts' element types; `σ` is
converted to `T`. Not exported: write `Bramble.RankOneUpdate`.

Entry `(i, j)` is `A[i, j] + σ * (ws[i] * wt[j])`. `mul!(y, K, x, α, β)` computes
`α (A x + σ ws (wtᵀ x)) + β y` with one product by `A` and two passes over the vectors, and
allocates nothing; so do `K * x`, `K * X` and `mul!` on matrices, apart from their result.
`Matrix(K)` is the dense matrix. `adjoint(K)` and `transpose(K)` are rank-one updates of
`A`'s adjoint and transpose with the two vectors swapped, sharing `A` and the vectors (the
adjoint conjugates `σ` and complex vectors). `similar(K)` is a rank-one update with
`similar` parts, and `copy(K)` one with copied parts.

With `dirichlet` labels, `assemble` makes each Dirichlet row of `A` an identity row, as
[`dirichlet_bc!`](@ref) does, and zeroes `ws` there, so the row is the identity row of
`K`; the columns keep their entries, as `A`'s do. [`symmetrize!`](@ref) then moves those
columns' entries, of `A` and of `σ wₛ wₜᵀ`, into the load and zeroes `wt` at the
Dirichlet indices.

`issymmetric(K)` is `true` when `A` is symmetric and `ws == wt` (or `σ` is zero), the case
of a symmetric form with one functional on both sides. Any other `K` reports `false`
without comparing its `N²` entries, even if they happen to be symmetric (`ws` a multiple of
`wt`, say).

# Examples
```jldoctest
using Bramble, LinearAlgebra
Wₕ = gridspace(mesh(domain(interval(0.0, 1.0)), 6, false))
a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + 2.0 * innerₕ(1, u) * innerₕ(1, v))
K = assemble(a)
w = assemble(form(Wₕ, v -> innerₕ(1, v)))
A = assemble(form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v))))
(K isa Bramble.RankOneUpdate, Matrix(K) ≈ Matrix(A) + 2.0 * w * w')

# output
(true, true)
```
"""
mutable struct RankOneUpdate{T, M <: AbstractMatrix, VS <: AbstractVector,
    VT <: AbstractVector} <: AbstractMatrix{T}
    const A::M
    σ::T
    const ws::VS
    const wt::VT

    function RankOneUpdate{T, M, VS, VT}(A::M, σ, ws::VS, wt::VT) where {T, M, VS, VT}
        length(ws) == size(A, 1) && length(wt) == size(A, 2) ||
            _throw_rank_one_sizes(size(A), length(ws), length(wt))
        return new{T, M, VS, VT}(A, σ, ws, wt)
    end
end

RankOneUpdate{T}(A::AbstractMatrix, σ::Number, ws::AbstractVector, wt::AbstractVector) where {T} = RankOneUpdate{
    T, typeof(A), typeof(ws), typeof(wt)}(A, convert(T, σ), ws, wt)
RankOneUpdate(A::AbstractMatrix, σ::Number, ws::AbstractVector, wt::AbstractVector) = RankOneUpdate{
    promote_type(eltype(A), typeof(σ), eltype(ws), eltype(wt))}(A, σ, ws, wt)

@noinline function _throw_rank_one_sizes(sz, m, n)
    throw(
        DimensionMismatch(
        "a rank-one update of a $(sz[1])×$(sz[2]) matrix needs a row vector of length " *
        "$(sz[1]) and a column vector of length $(sz[2]), got $m and $n.",
    ),
    )
end

Base.size(K::RankOneUpdate) = size(K.A)

Base.@propagate_inbounds function Base.getindex(K::RankOneUpdate{T}, i::Int, j::Int) where {T}
    @boundscheck checkbounds(K, i, j)
    return convert(T, @inbounds K.A[i, j] + K.σ * (K.ws[i] * K.wt[j]))
end

# `Σⱼ w[j] x[j]`, unconjugated: the term is `σ ws wtᵀ`, a transpose and not an adjoint.
@inline function _tdot(w::AbstractVector, x::AbstractVector)
    s = zero(promote_type(eltype(w), eltype(x)))
    @inbounds @simd for j in eachindex(w, x)
        s += w[j] * x[j]
    end
    return s
end

@inline function _tdot(w::AbstractVector, X::AbstractMatrix, k::Int)
    s = zero(promote_type(eltype(w), eltype(X)))
    @inbounds @simd for j in eachindex(w)
        s += w[j] * X[j, k]
    end
    return s
end

function mul!(
        y::AbstractVector, K::RankOneUpdate, x::AbstractVector, α::Number, β::Number)
    length(x) == size(K, 2) || _throw_mul_size(size(K), length(x))
    iszero(α) && return _scale_only!(y, β)
    c = α * K.σ * _tdot(K.wt, x)
    mul!(y, K.A, x, α, β)
    y .+= c .* K.ws
    return y
end

function mul!(
        Y::AbstractMatrix, K::RankOneUpdate, X::AbstractMatrix, α::Number, β::Number)
    size(X, 1) == size(K, 2) || _throw_mul_size(size(K), size(X, 1))
    iszero(α) && return _scale_only!(Y, β)
    mul!(Y, K.A, X, α, β)
    for k in axes(X, 2)
        c = α * K.σ * _tdot(K.wt, X, k)
        @inbounds for i in eachindex(K.ws)
            Y[i, k] += c * K.ws[i]
        end
    end
    return Y
end

# `α = 0`: `x` is not read, so an `Inf` in it gives no `NaN`, and with `β = 0` neither is
# `y`, as a matrix's own `mul!` behaves.
function _scale_only!(y::AbstractArray, β::Number)
    iszero(β) ? fill!(y, zero(eltype(y))) : (y .*= β)
    return y
end

@noinline function _throw_mul_size(sz, n)
    throw(DimensionMismatch("a $(sz[1])×$(sz[2]) rank-one update times $n rows."))
end

function Base.Matrix{S}(K::RankOneUpdate) where {S}
    M = Matrix{S}(K.A)
    @inbounds for j in axes(M, 2), i in axes(M, 1)
        M[i, j] += K.σ * (K.ws[i] * K.wt[j])
    end
    return M
end
Base.Matrix(K::RankOneUpdate{T}) where {T} = Matrix{T}(K)
Base.Array(K::RankOneUpdate) = Matrix(K)

Base.adjoint(K::RankOneUpdate) = RankOneUpdate(adjoint(K.A), conj(K.σ), conj(K.wt), conj(K.ws))
Base.transpose(K::RankOneUpdate) = RankOneUpdate(transpose(K.A), K.σ, K.wt, K.ws)

Base.similar(K::RankOneUpdate{T}) where {T} = similar(K, T)
Base.similar(K::RankOneUpdate, ::Type{S}) where {S} = RankOneUpdate{S}(
    similar(K.A, S), K.σ, similar(K.ws, S), similar(K.wt, S))
Base.copy(K::RankOneUpdate{T}) where {T} = RankOneUpdate{T}(
    copy(K.A), K.σ, copy(K.ws), copy(K.wt))

issymmetric(K::RankOneUpdate) = (iszero(K.σ) || K.ws == K.wt) && issymmetric(K.A)

function Base.show(io::IO, K::RankOneUpdate{T}) where {T}
    m, n = size(K)
    print(io, "RankOneUpdate{$T}($(m)×$(n) A + σ wₛ wₜᵀ, σ = ", K.σ, ")")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", K::RankOneUpdate)
    m, n = size(K)
    println(io, m, "×", n, " ", typeof(K).name.name, "{", eltype(K), "}: A + σ wₛ wₜᵀ")
    print(io, "  A: ")
    summary(io, K.A)
    print(io, "\n  σ = ", K.σ, "\n  wₛ: ")
    summary(io, K.ws)
    print(io, "\n  wₜ: ")
    summary(io, K.wt)
    return nothing
end

# --- Assembly ---------------------------------------------------------------------- #

"""
    assemble(a::RankOneForm; dirichlet = nothing, dirichlet_components = nothing) -> RankOneUpdate

Assemble a form holding a rank-one term into a [`RankOneUpdate`](@ref): its other terms
as [`assemble`](@ref) assembles a [`BilinearForm`](@ref), with `dirichlet` applied to their
matrix, and the two functionals' vectors through the linear path. `wₛ` is zero at the
Dirichlet rows, so they are identity rows of the result.
"""
function assemble(a::RankOneForm; dirichlet = nothing, dirichlet_components = nothing)
    A = assemble(a.bilinear; dirichlet = dirichlet,
        dirichlet_components = dirichlet_components)
    ws = assemble(a.test_functional)
    wt = assemble(a.trial_functional)
    K = RankOneUpdate(A, _rank_scale_value(a.scale), ws, wt)
    _zero_dirichlet_rows!(K.ws, test_space(a), dirichlet, dirichlet_components)
    return K
end

"""
    assemble!(K::RankOneUpdate, a::RankOneForm; dirichlet = nothing, dirichlet_components = nothing) -> RankOneUpdate

Refill `K`, from an earlier `assemble(a)`, in place: `A`, both vectors and `σ`, allocating
nothing on the host. A `Ref` factor of `σ` and every coefficient of the functionals are read
afresh.
"""
function assemble!(
        K::RankOneUpdate, a::RankOneForm; dirichlet = nothing, dirichlet_components = nothing)
    assemble!(K.A, a.bilinear; dirichlet = dirichlet,
        dirichlet_components = dirichlet_components)
    assemble!(K.ws, a.test_functional)
    assemble!(K.wt, a.trial_functional)
    K.σ = _rank_scale_value(a.scale)
    _zero_dirichlet_rows!(K.ws, test_space(a), dirichlet, dirichlet_components)
    return K
end

assemble!(A::AbstractMatrix, ::RankOneForm; kwargs...) = _throw_rank_one_target(A)
assemble!(K::RankOneUpdate, ::BilinearForm; kwargs...) = _throw_rank_one_target(K)

# Entry points written for a matrix the form's sparsity covers, refused by name.
assemble_add!(::AbstractMatrix, ::RankOneForm, α...) = _throw_rank_one_unsupported(
    "assemble_add!")
allocate_system_matrix(::RankOneForm, ast...) = _throw_rank_one_unsupported(
    "allocate_system_matrix")
matrix_free_operator(::RankOneForm; kwargs...) = _throw_rank_one_unsupported(
    "matrix_free_operator")

@noinline function _throw_rank_one_unsupported(name::String)
    throw(
        ArgumentError(
        "$name does not take a form holding a rank-one term σ * innerₕ(g, u) * " *
        "innerₕ(h, v) yet: its matrix is dense. Assemble the form with assemble(a), which " *
        "returns the sparse part and the term apart as a RankOneUpdate.",
    ),
    )
end

# `semidiscretize(a, l)`, `semidiscretize(build, l)` with `build` returning one, and a rank-one
# `mass`: each reaches one of these.
semidiscretize(::RankOneForm, ::LinearForm; kwargs...) = _throw_rank_one_semidiscrete()
_validate_semidiscrete_spaces(::RankOneForm, ::LinearForm; kwargs...) = _throw_rank_one_semidiscrete()
_validate_form_space(::RankOneForm, name, sp) = _throw_rank_one_semidiscrete()

@noinline function _throw_rank_one_semidiscrete()
    throw(
        ArgumentError(
        "semidiscretize does not take a form holding a rank-one term σ * innerₕ(g, u) * " *
        "innerₕ(h, v): the term has no semidiscretisation. Write the condition it " *
        "penalises another way (a constraint, or the penalty as a separate equation), or " *
        "assemble the steady form with assemble(a) and solve that.",
    ),
    )
end

@noinline function _throw_rank_one_target(@nospecialize(A))
    throw(
        ArgumentError(
        "a form holding a rank-one term assembles into the RankOneUpdate its own assemble " *
        "returns, and a form without one into a matrix; got a $(nameof(typeof(A))). " *
        "Call K = assemble(a) once, then assemble!(K, a).",
    ),
    )
end

"""
    assemble(a::RankOneForm, l::LinearForm; dirichlet = nothing, dirichlet_components = nothing, symmetrize::Bool = false) -> (K, F)

The rank-one counterpart of `assemble(a::BilinearForm, l::LinearForm)`: `K` a
[`RankOneUpdate`](@ref) and `F` the load, with the same `dirichlet` constraints applied
to both. `symmetrize = true` runs [`symmetrize!`](@ref) on them.
"""
function assemble(
        a::RankOneForm,
        l::LinearForm;
        dirichlet = nothing,
        dirichlet_components = nothing,
        symmetrize::Bool = false
)
    K = assemble(a; dirichlet = dirichlet, dirichlet_components = dirichlet_components)
    F = assemble(l; dirichlet = dirichlet, dirichlet_components = dirichlet_components)
    if symmetrize
        labels, _ = _normalize_dirichlet(dirichlet)
        labels === nothing && _throw_symmetrize_without_dirichlet()
        symmetrize!(K, F, test_space(a), labels...; components = dirichlet_components)
    end
    return K, F
end

# The Dirichlet rows of `ws`, zeroed one label at a time: zeroing is idempotent, so a point
# two labels share needs no combined mask, and nothing is allocated.
function _zero_dirichlet_rows!(ws::AbstractVector, Vₕ, dirichlet, components)
    labels, _ = _normalize_dirichlet(dirichlet)
    labels === nothing && return ws
    _each_dirichlet_mask(Vₕ, labels, components) do mask, offset
        _each_marked(mask, offset) do i
            @inbounds ws[i] = zero(eltype(ws))
        end
    end
    return ws
end

# Calls `f(mask, offset)` once per label and selected leaf of `Vₕ`.
function _each_dirichlet_mask(f::F, Vₕ::ScalarGridSpace, labels, components) where {F}
    _validate_scalar_components(components)
    for label in labels
        f(index_in_marker(mesh(Vₕ), label), 0)
    end
    return nothing
end

function _each_dirichlet_mask(f::F, Vₕ::CompositeGridSpace, labels, components) where {F}
    leaves = leaf_spaces_offsets(Vₕ)
    _validate_dirichlet_components(components, length(leaves))
    _each_selected_leaf(leaves, components) do sp, offset
        for label in labels
            f(index_in_marker(mesh(sp), label), offset)
        end
    end
    return nothing
end

"""
    symmetrize!(K::RankOneUpdate, F::AbstractVector, Wₕ, labels::Symbol...; components = nothing)

[`symmetrize!`](@ref) for a [`RankOneUpdate`](@ref) `K = A + σ wₛ wₜᵀ`, after the
Dirichlet rows are applied (`assemble(a; dirichlet)`). `A` is symmetrized as a matrix is,
and the rank-one columns of the Dirichlet indices `D` move into the load:
`F -= σ wₛ Σⱼ∈D wₜ[j] F[j]`, then `wₜ[D] = 0`. As `wₛ` is zero at `D`, the Dirichlet rows
of `F` keep their values.
"""
function symmetrize!(K::RankOneUpdate, F::AbstractVector, Wₕ::ScalarGridSpace,
        labels::Symbol...; components = nothing)
    return _symmetrize_rank_one!(K, F, Wₕ, labels, components)
end

function symmetrize!(K::RankOneUpdate, F::AbstractVector, Wₕ::CompositeGridSpace,
        labels::Symbol...; components = nothing)
    return _symmetrize_rank_one!(K, F, Wₕ, labels, components)
end

function _symmetrize_rank_one!(K::RankOneUpdate, F::AbstractVector, Wₕ, labels, components)
    _check_symmetrize_square(K)
    c = Ref(zero(promote_type(eltype(K.wt), eltype(F))))
    _each_dirichlet_mask(Wₕ, labels, components) do mask, offset
        _each_marked(mask, offset) do j
            c[] += K.wt[j] * F[j]
            K.wt[j] = zero(eltype(K.wt))
        end
    end
    F .-= (K.σ * c[]) .* K.ws
    symmetrize!(K.A, F, Wₕ, labels...; components = components)
    return nothing
end
