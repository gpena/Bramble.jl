# symmetry.jl

# ==============================================================================
# Structural symmetry / SPD detection from the AST
# ==============================================================================

#=
`innerₕ(L(u), L(v))` with the same `L` applied to the trial and test argument assembles to
`LᵀWL`: symmetric by construction, since `W` — the quadrature weight `InnerH`/`InnerPlus`
carries — is a positive diagonal, and `AᵀWA` is symmetric whatever `A` is. A sum of such
terms is symmetric, and scaling one by a real number preserves that. A term naming a
different operator on either side, such as `inner₊(u, D₋ₓ(v))`, generally is not.

The check walks the same AST `local_stencil` walks to assemble, asking whether `left_op`
and `right_op` are the same operator chain up to substituting `TrialFunction` for
`TestFunction` at the leaves. It evaluates nothing: two coefficients compare equal only when
they are the same object, which is exactly what happens when `L` is written once and applied
to both arguments — both sides then close over the identical variable. A numerically equal
but distinct coefficient is deliberately not recognised: this answers "is it *this*
pattern", not "does it happen to work out", the same conservative stance point 14 was
removed for not having.

None of this means anything unless the trial and test argument range over the same space:
"symmetric" presupposes a square matrix, and `form(Wₕ, Vₕ, ...)` with `Wₕ ≠ Vₕ` need not even
produce one — nor, if it happens to be square, does the `off_u`/`off_v` swap the argument
relies on correspond to an actual matrix transpose unless both sides share the same mesh
indexing. So both predicates below check `trial_space(a) === test_space(a)` before walking
the expression at all, and answer `false` immediately otherwise.
=#

_same_operator_shape(::TrialFunction{D}, ::TestFunction{D}) where {D} = true
_same_operator_shape(a::IndexedTrialFunction{D}, b::IndexedTestFunction{D}) where {D} = a.component_idx ==
                                                                                         b.component_idx

# Every other node wraps one (or two) inner operators, and that inner operator is exactly
# where a trial/test pair stops being the same Julia type: `D₋ₓ(u)` is a
# `BackwardDifference{D,Dim,TrialFunction{D}}`, `D₋ₓ(v)` a
# `BackwardDifference{D,Dim,TestFunction{D}}` — different `OpType`, so `typeof(a) ==
# typeof(b)` is false for the one case this trait exists to recognise. Each method below
# fixes every type parameter *except* the wrapped operator's own type, and recurses into it
# instead of requiring it to match structurally.
for W in (:BackwardDifference, :ForwardDifference, :CenteredDifference,
    :StarDifference, :CrossWeightedDifference, :BackwardAverage,
    :ForwardAverage, :ShiftNode, :JumpNode)
    @eval _same_operator_shape(a::$W{D, Dim}, b::$W{D, Dim}) where {D, Dim} = _same_operator_shape(
        a.inner_op, b.inner_op)
end

# The region a restriction names is a field, not a type parameter with a fixed set of
# values, so it is compared explicitly rather than folded into the `where` clause.
_same_operator_shape(a::RegionRestriction{D}, b::RegionRestriction{D}) where {D} = a.region ===
                                                                                    b.region &&
                                                                                    _same_operator_shape(
    a.inner_op, b.inner_op)

_same_operator_shape(a::OperatorScale{D}, b::OperatorScale{D}) where {D} = a.scalar ==
                                                                            b.scalar &&
                                                                            _same_operator_shape(
    a.inner_op, b.inner_op)

# By identity, not value: a coefficient compares equal here only when both sides close over
# the identical object, which is what happens when `L` is written once and applied twice —
# see the module-level note above. A numerically equal but distinct array is not this case.
_same_operator_shape(a::GridFunctionScale{D}, b::GridFunctionScale{D}) where {D} = a.grid_function ===
                                                                                    b.grid_function &&
                                                                                    _same_operator_shape(
    a.inner_op, b.inner_op)

_same_operator_shape(a::OperatorAdd{D}, b::OperatorAdd{D}) where {D} = _same_operator_shape(
    a.left_op, b.left_op) && _same_operator_shape(a.right_op, b.right_op)

_same_operator_shape(a::IdentityOperator{D}, b::IdentityOperator{D}) where {D} = a.space ===
                                                                                  b.space
_same_operator_shape(a::ZeroOperator{D}, b::ZeroOperator{D}) where {D} = a.space === b.space

# Anything else — different node types, a Trial/Test pair with mismatched indices, or a
# shape this does not recognise — is not verified as the same operator.
_same_operator_shape(a, b) = false

# ==============================================================================
# The assembly-level consumer: skip half the multiplications `multiply_stencils_bilinear`
# does when the two sides are known (by `_same_operator_shape`, in
# `local_stencil(::BilinearProduct, …)`, form/operators/inner.jl) to produce the same stencil.
# ==============================================================================

# For `i <= j`, `stencil[i][2]*stencil[j][2]*vol` is computed once and bound to a local; for
# `i > j` the mirrored entry reuses that same binding instead of recomputing the (identical,
# since multiplication commutes) product. The output is still every `(i, j)` pair — same
# length, same values — `multiply_stencils_bilinear` would give for `left ≡ right`, just
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

# Whether one term of a resolved bilinear AST is symmetric by the `LᵀWL` argument above.
# Only the three shapes that argument covers answer true; anything else — including a shape
# this does not recognise but which happens to be symmetric some other way — answers false.
_is_symmetric_term(op::BilinearProduct) = _same_operator_shape(op.left_op, op.right_op)
function _is_symmetric_term(op::OperatorAdd)
    _is_symmetric_term(op.left_op) && _is_symmetric_term(op.right_op)
end
_is_symmetric_term(op::OperatorScale) = _is_symmetric_term(op.inner_op)
_is_symmetric_term(op) = false

# As `_is_symmetric_term`, but additionally requires every `OperatorScale` along the way to
# carry a positive scalar: `LᵀWL` is positive semi-definite, and a negative or zero scale
# would flip or collapse that.
_is_posdef_term(op::BilinearProduct) = _same_operator_shape(op.left_op, op.right_op)
function _is_posdef_term(op::OperatorAdd)
    _is_posdef_term(op.left_op) && _is_posdef_term(op.right_op)
end
_is_posdef_term(op::OperatorScale) = op.scalar > 0 && _is_posdef_term(op.inner_op)
_is_posdef_term(op) = false

"""
    issymmetric(a::BilinearForm)

Whether `a` is symmetric by construction — `innerₕ(L(u), L(v))`, or a sum or scaling of such
terms, with the *same* `L` written once and applied to both the trial and test argument.

Purely structural: this walks `a`'s expression and never assembles a matrix. It is also
conservative — a term that happens to produce a symmetric matrix through some other route
answers `false`, the same as one that is not symmetric at all.

This describes the *unconstrained* operator. [`dirichlet_bc!`](@ref) zeros a row without
touching its column, so a matrix assembled with `dirichlet_labels` is not symmetric even when
`issymmetric(a)` is `true` — only after [`symmetrize!`](@ref) restores it. `true` here is a
claim about `a`'s expression, not about whatever matrix a particular call to `assemble`
produced.

# Examples
```julia
a = form(Wₕ, Wₕ, (u, v) -> inner₊ₓ(D₋ₓ(u), D₋ₓ(v)))
issymmetric(a)  # true — the same D₋ₓ on both sides

b = form(Wₕ, Wₕ, (u, v) -> inner₊(u, D₋ₓ(v)))
issymmetric(b)  # false — different operators either side

issymmetric(Matrix(assemble(a)))                                # true
issymmetric(Matrix(assemble(a; dirichlet_labels = :boundary)))   # false — rows zeroed, columns not
```
"""
function issymmetric(a::BilinearForm)
    trial_space(a) === test_space(a) || return false
    return _is_symmetric_term(resolve_form_ast(a))
end

"""
    isposdef(a::BilinearForm)

Whether `a` is symmetric positive semi-definite by the same `LᵀWL` construction
`issymmetric` checks — true only when, in addition, every scaling along the way is by a
positive number, which is what keeps that positivity from being flipped or collapsed.

Purely structural, like `issymmetric`, and for the same reason conservative: this does not
prove positive-*definite* (which also needs `L` to have trivial kernel), only that the
assembled matrix is symmetric positive semi-definite — enough to make `cholesky` worth
attempting first rather than a general factorization.

Describes the *unconstrained* operator, exactly as `issymmetric` does: a matrix assembled
with `dirichlet_labels` needs [`symmetrize!`](@ref) after [`dirichlet_bc!`](@ref) before
either symmetry or positive-definiteness holds of it, `isposdef(a)` being `true` notwithstanding.
"""
function isposdef(a::BilinearForm)
    trial_space(a) === test_space(a) || return false
    return _is_posdef_term(resolve_form_ast(a))
end
