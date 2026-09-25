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
anywhere in a sum, expanding a sum inside either side first (⟨Au, (B + C)v⟩ is how
`simplify_ast`'s factoring can store one half of a pair); `isposdef` does not, since a pair
need not be semi-definite.

The check walks the same AST `local_stencil` walks to assemble, asking whether `left_op`
and `right_op` (or, for a pair, one term's `left_op` and the other's `right_op`) are the same
operator chain up to substituting `TrialFunction` for `TestFunction` at the leaves. It
evaluates nothing: two coefficients compare equal only when they are the same object, which
is exactly what happens when `L` is written once and applied to both arguments: both sides
then close over the identical variable. A numerically equal
but distinct coefficient is deliberately not recognised: this answers "is it this
pattern", not "does it happen to work out".

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
# `local_stencil(::BilinearProduct, …)`, form/operators/inner.jl) to produce the same stencil.
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
function _is_transposed_pair(p::BilinearProduct{D, K}, q::BilinearProduct{D, K}) where {D, K}
    return _same_operator_shape(p.left_op, q.right_op) &&
           _same_operator_shape(q.left_op, p.right_op)
end
_is_transposed_pair(p, q) = false

# A product with a sum on either side, expanded into one product per pair of summands:
# ⟨Au, (B + C)v⟩ is ⟨Au, Bv⟩ + ⟨Au, Cv⟩, which is how `simplify_ast`'s factoring can reshape
# one half of a transposed pair. Runtime only (`issymmetric`), so it builds values freely.
_side_summands(op::OperatorAdd) = (_side_summands(op.left_op)..., _side_summands(op.right_op)...)
_side_summands(op) = (op,)
function _expand_product(p::BilinearProduct{D, K}) where {D, K}
    return [BilinearProduct{D, K, typeof(l), typeof(r)}(l, r)
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
            k > i && !used[k] && length(terms[k][1]) == length(s) &&
                all(map(==, terms[k][1], s)) && _is_transposed_pair(p, terms[k][2])
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

_transpose_type(::Type{TrialFunction{D, N}}) where {D, N} = TestFunction{D, N}
_transpose_type(::Type{TestFunction{D, N}}) where {D, N} = TrialFunction{D, N}
_transpose_type(::Type{IndexedTrialFunction{D}}) where {D} = IndexedTestFunction{D}
_transpose_type(::Type{IndexedTestFunction{D}}) where {D} = IndexedTrialFunction{D}
function _transpose_type(::Type{BilinearProduct{D, K, L, R}}) where {D, K, L, R}
    return BilinearProduct{D, K, _transpose_type(R), _transpose_type(L)}
end
function _transpose_type(T::DataType)
    isempty(T.parameters) && return T
    return T.name.wrapper{map(p -> p isa Type ? _transpose_type(p) : p, T.parameters)...}
end
_transpose_type(T) = T

# The product under any scalings, or `nothing` for a summand that is not one.
_bare_product_type(::Type{<:OperatorScale{D, S, T}}) where {D, S, T} = _bare_product_type(T)
_bare_product_type(T::Type{<:BilinearProduct}) = T
_bare_product_type(::Type) = nothing

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
_is_posdef_term(op::BilinearProduct) = _same_operator_shape(op.left_op, op.right_op)
function _is_posdef_term(op::OperatorAdd)
    return _is_posdef_term(op.left_op) && _is_posdef_term(op.right_op)
end
_is_posdef_term(op::OperatorScale) = op.scalar > 0 && _is_posdef_term(op.inner_op)
_is_posdef_term(::ZeroOperator) = true
_is_posdef_term(op) = false

"""
    issymmetric(a::BilinearForm) -> Bool

Whether `a` is symmetric by construction: a sum or scaling of terms `innerₕ(L(u), L(v))`,
with the *same* `L` written once and applied to both the trial and test argument, and of
transposed pairs `innerₕ(A(u), B(v)) + innerₕ(B(u), A(v))`, which may sit anywhere in the sum.
The two terms of a pair must carry the same coefficients: the same object (a `Ref`, a grid
function) or equal numbers, or none on either. A term with no partner makes the answer
`false`.

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
# Both form types fell through to Julia's default `show`, which printed the whole resolved
# AST type -- every operator node and its parameters -- ahead of the spaces, which are
# what a caller actually wants to check. The detailed `show` below reports the spaces first,
# then an `Expression` row rendered via `expression(form)`. That rendering logic itself
# lives in `src/form/expression.jl`, with the per-node-type methods colocated with each
# node's struct.
#
# Lives in this file rather than `linear.jl`/`bilinear.jl` because `issymmetric` below is
# what the detailed bilinear block reports, and it is defined here.

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
