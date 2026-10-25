# symmetry.jl

# ==============================================================================
# Structural symmetry / SPD detection from the AST
# ==============================================================================

#=
`innerₕ(L(u), L(v))` with the same `L` applied to the trial and test argument assembles to
`LᵀWL`: symmetric by construction, since `W` (the quadrature weight `InnerH`/`InnerPlus`
carries) is a positive diagonal, and `AᵀWA` is symmetric whatever `A` is. A sum of such
terms is symmetric, and scaling one by a real number preserves that. A term naming a
different operator on either side, such as `inner₊(u, D₋ₓ(v))`, generally is not.

A term with different operators either side can still sit in a symmetric sum: ⟨Au, Bv⟩
assembles to `BᵀWA` and ⟨Bu, Av⟩ to its transpose, so a transposed pair is symmetric whatever
`A` and `B` are, provided both terms carry the same scalings. `issymmetric` pairs such terms
anywhere in a sum, expanding a sum inside either side first (a pair whose half is written
⟨Au, (B + C)v⟩ is still found); `isposdef` does not, since a pair need not be
semi-definite.

The check walks the same AST `local_stencil` walks to assemble, asking whether `left_op`
and `right_op` (or, for a pair, one term's `left_op` and the other's `right_op`) are the same
operator chain up to substituting `TrialFunction` for `TestFunction` at the leaves. It
evaluates nothing: two coefficients compare equal only when they are the same object, which
is exactly what happens when `L` is written once and applied to both arguments, so both sides
then close over the identical variable. A numerically equal
but distinct coefficient is deliberately not recognised: this answers "is it this
pattern" and not "does it happen to work out".

None of this means anything unless the trial and test argument range over the same space:
"symmetric" presupposes a square matrix, and `form(Wₕ, Vₕ, ...)` with `Wₕ ≠ Vₕ` need not even
produce one; nor, if it happens to be square, does the `off_u`/`off_v` swap the argument
relies on correspond to an actual matrix transpose unless both sides share the same mesh
indexing. So both predicates below check `trial_space(a) === test_space(a)` before walking
the expression at all, and answer `false` immediately otherwise.
=#

function _same_operator_shape(::TrialFunction{D, N}, ::TestFunction{D, M}) where {D, N, M}
    (N === M || N === nothing || M === nothing)
end
function _same_operator_shape(
        a::IndexedTrialFunction{D}, b::IndexedTestFunction{D}
) where {D}
    return a.component_idx == b.component_idx
end

# Every other node wraps one (or two) inner operators, and that inner operator is exactly
# where a trial/test pair stops being the same Julia type: `D₋ₓ(u)` is a
# `BackwardDifference{D,Dim,TrialFunction{D}}`, `D₋ₓ(v)` a
# `BackwardDifference{D,Dim,TestFunction{D}}` (different `OpType`, so `typeof(a) ==
# typeof(b)` is false for the one case this trait exists to recognise). Each method below
# fixes every type parameter except the wrapped operator's own type, and recurses into it
# instead of requiring it to match structurally.
for W in (
    :BackwardDifference,
    :ForwardDifference,
    :CenteredDifference,
    :StarDifference,
    :CrossWeightedDifference,
    :SecondDifference,
    :BackwardAverage,
    :ForwardAverage,
    :CenteredAverage,
    :JumpNode
)
    @eval _same_operator_shape(a::$W{D, Dim}, b::$W{D, Dim}) where {D, Dim} = _same_operator_shape(a.inner_op, b.inner_op)
end

# `shift_amount` is a field, not a type parameter, so, like `RegionRestriction.region` and
# `OperatorScale.scalar` below, it is compared explicitly rather than folded into the
# `where` clause. Two shifts by different amounts are not the same operator: leaving this to
# the generic loop above compared only `D`/`Dim`, so `shift_op(u, 1, 1)` and
# `shift_op(v, 1, 2)` read as identical, and the fast path this trait guards then evaluates
# one side only and mirrors it into a matrix that is not what the form asked for.
function _same_operator_shape(a::ShiftNode{D, Dim}, b::ShiftNode{D, Dim}) where {D, Dim}
    return a.shift_amount == b.shift_amount && _same_operator_shape(a.inner_op, b.inner_op)
end

# The region a restriction names is a field, not a type parameter with a fixed set of
# values, so it is compared explicitly rather than folded into the `where` clause.
function _same_operator_shape(a::RegionRestriction{D}, b::RegionRestriction{D}) where {D}
    return a.region === b.region && _same_operator_shape(a.inner_op, b.inner_op)
end

function _same_operator_shape(a::OperatorScale{D}, b::OperatorScale{D}) where {D}
    return a.scalar == b.scalar && _same_operator_shape(a.inner_op, b.inner_op)
end

# By identity, not value: a coefficient compares equal here only when both sides close over
# the identical object, which is what happens when `L` is written once and applied twice:
# see the module-level note above. A numerically equal but distinct array is not this case.
function _same_operator_shape(a::GridFunctionScale{D}, b::GridFunctionScale{D}) where {D}
    return a.grid_function === b.grid_function &&
           _same_operator_shape(a.inner_op, b.inner_op)
end

function _same_operator_shape(a::OperatorAdd{D}, b::OperatorAdd{D}) where {D}
    return _same_operator_shape(a.left_op, b.left_op) &&
           _same_operator_shape(a.right_op, b.right_op)
end

function _same_operator_shape(a::IdentityOperator{D}, b::IdentityOperator{D}) where {D}
    return a.space === b.space
end
_same_operator_shape(a::ZeroOperator{D}, b::ZeroOperator{D}) where {D} = a.space === b.space

# Anything else: different node types, a Trial/Test pair with mismatched indices, or a
# shape this does not recognise, is not verified as the same operator.
_same_operator_shape(a, b) = false

# ==============================================================================
# The assembly-level consumer: skip half the multiplications `multiply_stencils_bilinear`
# does when the two sides are known (by `_same_operator_shape`, in
# `local_stencil(::BilinearProduct, …)`, operators/inner.jl) to produce the same stencil.
# ==============================================================================

# For `i <= j`, `stencil[i][2]*stencil[j][2]*vol` is computed once and bound to a local; for
# `i > j` the mirrored entry reuses that same binding instead of recomputing the (identical,
# since multiplication commutes) product. The output is still every `(i, j)` pair (same
# length, same values) `multiply_stencils_bilinear` would give for `left ≡ right`, just
# built from `N(N+1)/2` multiplications rather than `N²`.
@generated function multiply_stencils_bilinear_symmetric(stencil::Tuple, vol::Number)
    N = length(stencil.parameters)
    assigns = Expr[]
    slot = Matrix{Symbol}(undef, N, N)
    for i in 1:N, j in i:N

        s = Symbol(:w_, i, :_, j)
        push!(assigns, :($s = stencil[$i][2] * stencil[$j][2] * vol))
        slot[i, j] = s
    end
    exprs = Expr[]
    for i in 1:N, j in 1:N

        s = i <= j ? slot[i, j] : slot[j, i]
        push!(exprs, :((stencil[$i][1], stencil[$j][1], $s)))
    end
    return Expr(:block, assigns..., Expr(:tuple, exprs...))
end

# --- Transposed pairs ------------------------------------------------------------ #
#
# ⟨Au, Bv⟩ assembles to `BᵀWA` and ⟨Bu, Av⟩ to `AᵀWB = (BᵀWA)ᵀ`, so the two together are
# symmetric whatever `A` and `B` are. Both terms must carry the same scalings (the same
# object, or equal numbers), for the same reason a single term's coefficient must be the
# same object on both sides.

# Whether `q` is `p` with its two sides swapped (trial and test exchanged at the leaves).
# The weights must be equal, not only of one type: a surface weight holds its faces as data
# (`InnerGamma`, operators/inner.jl), so an `:xmin` term and an `:xmax` one share a type.
function _is_transposed_pair(p::BilinearProduct{D, K}, q::BilinearProduct{D, K}) where {D, K}
    return p.inner == q.inner && _same_operator_shape(p.left_op, q.right_op) &&
           _same_operator_shape(q.left_op, p.right_op)
end
_is_transposed_pair(p, q) = false

# A product with a sum on either side, expanded into one product per pair of summands:
# ⟨Au, (B + C)v⟩ is ⟨Au, Bv⟩ + ⟨Au, Cv⟩, so a pair one of whose halves is written with a sum
# inside a side is still paired. Runtime only (`issymmetric`), so it builds values freely.
_side_summands(op::OperatorAdd) = (_side_summands(op.left_op)..., _side_summands(op.right_op)...)
_side_summands(op) = (op,)
function _expand_product(p::BilinearProduct{D, K}) where {D, K}
    return [BilinearProduct{D, K, typeof(l), typeof(r)}(p.inner, l, r)
            for l in _side_summands(p.left_op) for r in _side_summands(p.right_op)]
end

# Every summand of `op` as `(scalings, term)`, the scalings being every `OperatorScale`
# factor above the term, outermost first. `false` when a summand is neither a product nor a
# collapsed zero.
function _collect_scaled_terms!(out::Vector{Tuple{Vector{Any}, Any}}, scales::Vector{Any}, op)
    if op isa OperatorAdd
        return _collect_scaled_terms!(out, scales, op.left_op) &&
               _collect_scaled_terms!(out, scales, op.right_op)
    elseif op isa OperatorScale
        return _collect_scaled_terms!(out, Any[scales..., op.scalar], op.inner_op)
    elseif op isa ZeroOperator
        # `simplify_ast` collapses a product with a zero side to a bare `ZeroOperator`: the
        # zero matrix, symmetric.
        return true
    elseif op isa BilinearProduct
        if _same_operator_shape(op.left_op, op.right_op)
            push!(out, (scales, op))
        else
            foreach(t -> push!(out, (scales, t)), _expand_product(op))
        end
        return true
    end
    return false
end

# Whether two scaling lists scale by the same amount. Equal factor by factor, as before;
# otherwise the `Ref` factors must be the same objects in the same order and the plain
# numbers must have the same exact product. `simplify_ast` leaves `(A / 2) / 3` as two
# `Rational` scalings (gpena/Bramble.jl#633), which must still partner `A' / 6`. The product
# is a `Rational{BigInt}`, so it cannot overflow; a non-finite float, a non-real factor, or a
# zero beside `1 // 0` has no exact product and does not partner.
function _same_scalings(a::Vector{Any}, b::Vector{Any})
    length(a) == length(b) && all(map(==, a, b)) && return true
    isref(c) = c isa Base.RefValue
    ra, rb = filter(isref, a), filter(isref, b)
    length(ra) == length(rb) && all(map(===, ra, rb)) || return false
    na, nb = filter(!isref, a), filter(!isref, b)
    all(_has_exact_value, na) && all(_has_exact_value, nb) || return false
    _exact_product(na) === nothing && return false
    _exact_product(nb) === nothing && return false
    return _exact_product(na) == _exact_product(nb)
end
_has_exact_value(c) = c isa Union{Integer, Rational} || (c isa AbstractFloat && isfinite(c))
function _exact_product(v::Vector{Any})
    R = Rational{BigInt}
    has_zero = any(iszero, v)
    has_inf = any(c -> c isa Rational && iszero(denominator(c)), v)
    has_zero && has_inf && return nothing
    return prod(c -> R(c), v; init = one(R))
end

# Symmetric terms need no partner; every other term needs its own transposed partner with
# the same scalings, and each term partners at most one other.
function _is_symmetric_sum(ast)
    terms = Tuple{Vector{Any}, Any}[]
    _collect_scaled_terms!(terms, Any[], ast) || return false
    used = falses(length(terms))
    for i in eachindex(terms)
        used[i] && continue
        s, p = terms[i]
        _same_operator_shape(p.left_op, p.right_op) && continue
        j = findfirst(eachindex(terms)) do k
            k > i && !used[k] && _same_scalings(terms[k][1], s) &&
                _is_transposed_pair(p, terms[k][2])
        end
        j === nothing && return false
        used[j] = true
    end
    return true
end

# --- Transposed pairs at the type level (assembly) ------------------------------- #
#
# Assembly compiles one kernel per summand, so pairing has to be decided from the summand
# types alone, before any value is looked at: `_pair_plan` pairs summand `i` with the first
# later summand `j` whose product type (under any scalings) is `i`'s with its sides swapped
# and trial and test exchanged at the leaves. Only products whose values carry nothing but
# component indices are paired (`_pairable_type`): the type then fixes every stencil weight,
# and a component index only chooses the block (`bilinear_execution.jl`).
#
# Both helpers below run only in `_foldl_pairs`'s generator, on types as values. One method
# each, `@nospecialize`: dispatching on `Type{T}` compiled one instance per node type of
# every summand.
function _transpose_type(@nospecialize(T))
    T isa DataType || return T
    P = T.parameters
    isempty(P) && return T
    w = T.name.wrapper
    w === TrialFunction && return TestFunction{P...}
    w === TestFunction && return TrialFunction{P...}
    w === IndexedTrialFunction && return IndexedTestFunction{P...}
    w === IndexedTestFunction && return IndexedTrialFunction{P...}
    w === BilinearProduct &&
        return BilinearProduct{P[1], P[2], _transpose_type(P[4]), _transpose_type(P[3])}
    return w{map(p -> p isa Type ? _transpose_type(p) : p, P)...}
end

# The product under any scalings, or `nothing` for a summand that is not one.
function _bare_product_type(@nospecialize(T))
    T isa Type || return nothing
    T <: BilinearProduct && return T
    T isa DataType && T <: OperatorScale && return _bare_product_type(T.parameters[3])
    return nothing
end

# Singleton types, component-indexed trial and test functions, and nodes built only from
# these. A coefficient, a scalar inside a side, a shift amount, a region or an interpolation's
# space is a value the type does not fix.
function _pairable_type(T)
    T isa DataType || return false
    Base.issingletontype(T) && return true
    (T <: IndexedTrialFunction || T <: IndexedTestFunction) && return true
    isconcretetype(T) && fieldcount(T) > 0 || return false
    return all(_pairable_type, fieldtypes(T))
end

# `((i, j), (k, 0), ...)`: one group per summand kept, `j == 0` for a summand walked alone.
function _pair_plan(Ts)
    n = length(Ts)
    partner = zeros(Int, n)
    bare = map(_bare_product_type, Ts)
    for i in 1:n
        (partner[i] == 0 && bare[i] !== nothing) || continue
        _pairable_type(bare[i]) || continue
        tr = try
            _transpose_type(bare[i])
        catch
            nothing
        end
        (tr === nothing || tr == bare[i]) && continue
        for j in (i + 1):n
            if partner[j] == 0 && bare[j] == tr
                partner[i] = j
                partner[j] = -1
                break
            end
        end
    end
    return Tuple((i, partner[i]) for i in 1:n if partner[i] >= 0)
end

# `acc = single(acc, ts[i])` or `acc = pair(acc, ts[i], ts[j])` per group of `_pair_plan`,
# unrolled with literal indices so every call is type-stable.
@generated function _foldl_pairs(single::F1, pair::F2, acc, ts::Tuple) where {F1, F2}
    body = Expr(:block)
    for (i, j) in _pair_plan(Tuple(ts.parameters))
        push!(body.args,
            j == 0 ? :(acc = single(acc, ts[$i])) : :(acc = pair(acc, ts[$i], ts[$j])))
    end
    push!(body.args, :(return acc))
    return body
end

# The product a pair kernel walks, and the scaling its entries are multiplied by.
@inline _bare_product(op::OperatorScale) = _bare_product(op.inner_op)
@inline _bare_product(op) = op
@inline _term_scale(op::OperatorScale{D, <:Base.RefValue}) where {D} = op.scalar[] * _term_scale(op.inner_op)
@inline _term_scale(op::OperatorScale) = op.scalar * _term_scale(op.inner_op)
@inline _term_scale(op) = true

# Whether a term is `LᵀWL` by the argument above (transposed pairs are not: they need not be
# semi-definite), with every `OperatorScale` along the way carrying a positive scalar:
# `LᵀWL` is positive semi-definite, and a negative or zero scale would flip or collapse that.
#
# A coefficient inside a side must also be real: assembly does not conjugate the test side,
# so `innerₕ(g * u, g * v)` assembles to `LᵀGWGL`, and a non-real `g` makes `G²` indefinite
# or negative (`g = im` gives `-LᵀWL`). Both sides are walked, since `OperatorScale` sides
# match by `==` and `1.0 == 1.0 + 0im`.
#
# The weight `W` must be non-negative too: `InnerGammaNormal` carries the outward sign
# (`-` on a minimum facet), so `inner_Γ(u * η[1], v)` is indefinite (#865). The weights
# are whitelisted, so a signed weight added later answers `false` until it is listed.
function _is_posdef_term(op::BilinearProduct{D, W}) where {D, W}
    return _has_nonnegative_weight(W) && _same_operator_shape(op.left_op, op.right_op) &&
           _is_real_side(op.left_op) && _is_real_side(op.right_op)
end
_has_nonnegative_weight(::Type{<:Union{InnerH, InnerPlus, InnerPlusSet, InnerGamma,
    InnerGammaFacet}}) = true
_has_nonnegative_weight(::Type) = false
function _is_posdef_term(op::OperatorAdd)
    return _is_posdef_term(op.left_op) && _is_posdef_term(op.right_op)
end
function _is_posdef_term(op::OperatorScale)
    return _is_positive_scale(op.scalar) && _is_posdef_term(op.inner_op)
end
# A `Ref` is read now, so the answer follows its current value; a non-real scale is never
# taken as positive.
_is_positive_scale(c::Real) = c > 0
_is_positive_scale(c::Base.RefValue) = _is_positive_scale(c[])
_is_positive_scale(c) = false
_is_posdef_term(::ZeroOperator) = true
_is_posdef_term(op) = false

# Whether every coefficient inside a side is real: each `GridFunctionScale` has a real
# element type and each `OperatorScale` a real scalar (a `Ref` read now). Walks every node
# `_same_operator_shape` recognises; anything else answers `false`, as it does there.
function _is_real_side(::Union{TrialFunction, TestFunction, IndexedTrialFunction, IndexedTestFunction,
        IdentityOperator, ZeroOperator})
    true
end
for W in (
    :BackwardDifference,
    :ForwardDifference,
    :CenteredDifference,
    :StarDifference,
    :CrossWeightedDifference,
    :SecondDifference,
    :BackwardAverage,
    :ForwardAverage,
    :CenteredAverage,
    :JumpNode,
    :ShiftNode,
    :RegionRestriction
)
    @eval _is_real_side(op::$W) = _is_real_side(op.inner_op)
end
_is_real_side(op::OperatorAdd) = _is_real_side(op.left_op) && _is_real_side(op.right_op)
_is_real_side(op::OperatorScale) = _is_real_scale(op.scalar) && _is_real_side(op.inner_op)
function _is_real_side(op::GridFunctionScale)
    return eltype(op.grid_function) <: Real && _is_real_side(op.inner_op)
end
_is_real_side(op) = false
_is_real_scale(c::Base.RefValue) = _is_real_scale(c[])
_is_real_scale(c) = c isa Real

"""
    issymmetric(a::BilinearForm) -> Bool

Whether `a` is symmetric by construction: a sum or scaling of terms `innerₕ(L(u), L(v))`,
with the *same* `L` written once and applied to both the trial and test argument, and of
transposed pairs `innerₕ(A(u), B(v)) + innerₕ(B(u), A(v))`, which may sit anywhere in the sum.
The two terms of a pair must carry the same coefficients: the same object (a `Ref`, a grid
function), or numbers whose exact products are equal (`(A / 2) / 3` partners `B / 6`), or
none on either. A term with no partner makes the answer `false`.

Purely structural: this walks `a`'s expression and never assembles a matrix. It is also
conservative: a term that happens to produce a symmetric matrix through some other route
answers `false`, the same as one that is not symmetric at all.

This describes the *unconstrained* operator. [`dirichlet_bc!`](@ref) zeros a row without
touching its column, so a matrix assembled with `dirichlet` is not symmetric even when
`issymmetric(a)` is `true`, until [`symmetrize!`](@ref) restores it. `true` here is a
claim about `a`'s expression, not about whatever matrix a particular call to `assemble`
produced.

# Examples
```julia
using Bramble: D₋ₓ, D₋ᵧ, inner₊ₓ
using LinearAlgebra: issymmetric
a = form(Wₕ, Wₕ, (u, v) -> inner₊ₓ(D₋ₓ(u), D₋ₓ(v)))
issymmetric(a)  # true: the same D₋ₓ on both sides

b = form(Wₕ, Wₕ, (u, v) -> inner₊(u, D₋ₓ(v)))
issymmetric(b)  # false: different operators either side, and no transposed partner

c = form(Wₕ, Wₕ, (u, v) -> innerₕ(D₋ₓ(u), D₋ᵧ(v)) + innerₕ(D₋ᵧ(u), D₋ₓ(v)))
issymmetric(c)  # true: a transposed pair

issymmetric(Matrix(assemble(a)))                                # true
issymmetric(Matrix(assemble(a; dirichlet = :boundary)))   # false: rows zeroed, columns not
```
"""
function issymmetric(a::BilinearForm)
    trial_space(a) === test_space(a) || return false
    return _is_symmetric_sum(resolve_form_ast(a))
end

"""
    isposdef(a::BilinearForm) -> Bool

Whether `a` is symmetric positive semi-definite by the `LᵀWL` construction
`issymmetric` checks: true only when every term is `innerₕ(L(u), L(v))` (a transposed pair
need not be semi-definite, so one makes the answer `false`) and every scaling along the way
is by a positive number, which is what keeps that positivity from being flipped or collapsed.
A `Ref` scale is read when `isposdef` is called, so the answer follows its value at that
moment and may change after the `Ref` is updated. A non-real scale or coefficient, complex
included, answers `false`.

Purely structural, like `issymmetric`, and for the same reason conservative: this does not
prove positive-definite (which also needs `L` to have trivial kernel), only that the
assembled matrix is symmetric positive semi-definite, enough to make `cholesky` worth
attempting first rather than a general factorization.

Describes the *unconstrained* operator, exactly as `issymmetric` does: a matrix assembled
with `dirichlet` needs [`symmetrize!`](@ref) after [`dirichlet_bc!`](@ref) before
either symmetry or positive-definiteness holds of it, `isposdef(a)` being `true` notwithstanding.
"""
function isposdef(a::BilinearForm)
    trial_space(a) === test_space(a) || return false
    return _is_posdef_term(resolve_form_ast(a))
end

# --- Display ---------------------------------------------------------------------- #
#
# The detailed `show` reports the spaces first, then an `Expression` row (`expression(form)`,
# src/ast/expression.jl). It lives here because it reports `issymmetric`, defined above.

function Base.show(io::IO, l::LinearForm{D}) where {D}
    print(io, "LinearForm{$(D)D, ", ndofs(test_space(l)), "}")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", l::LinearForm{D}) where {D}
    return show_block(io) do io
        pp = PrettyPrinter(io)
        Vₕ = test_space(l)

        printstyled(io, "LinearForm"; bold = true, color = :cyan)
        print(io, " {")
        printstyled(io, "$(D)D"; color = :yellow)
        print(io, ", ")
        printstyled(io, "$(eltype(Vₕ))"; color = :yellow)
        println(io, "}:")

        pp_indented = with_indent(pp, 1)
        print_key_value(pp_indented, "Test space", sprint(show, Vₕ); separator = ": ")
        print_key_value(pp_indented, "Vector", string(ndofs(Vₕ)); separator = ": ")
        return print_key_value(pp_indented, "Expression", expression(l); separator = ": ")
    end
end

function Base.show(io::IO, a::BilinearForm{D}) where {D}
    print(io, "BilinearForm{$(D)D, ", ndofs(test_space(a)), "×", ndofs(trial_space(a)), "}")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", a::BilinearForm{D}) where {D}
    return show_block(io) do io
        pp = PrettyPrinter(io)
        Uₕ = trial_space(a)
        Vₕ = test_space(a)

        printstyled(io, "BilinearForm"; bold = true, color = :cyan)
        print(io, " {")
        printstyled(io, "$(D)D"; color = :yellow)
        print(io, ", ")
        printstyled(io, "$(eltype(Vₕ))"; color = :yellow)
        println(io, "}:")

        pp_indented = with_indent(pp, 1)
        print_key_value(pp_indented, "Trial space", sprint(show, Uₕ); separator = ": ")

        # `(same as trial)` rather than repeating the line: trial === test is the
        # overwhelmingly common case, and the interesting information is which of the two
        # it is.
        test_text = if Uₕ === Vₕ
            sprint(show, Vₕ) * "  (same as trial)"
        else
            sprint(show, Vₕ)
        end
        print_key_value(pp_indented, "Test space", test_text; separator = ": ")
        print_key_value(
            pp_indented, "Matrix", "$(ndofs(Vₕ)) × $(ndofs(Uₕ))"; separator = ": "
        )
        print_key_value(
            pp_indented, "Symmetric", issymmetric(a) ? "yes" : "no"; separator = ": "
        )
        return print_key_value(pp_indented, "Expression", expression(a); separator = ": ")
    end
end
