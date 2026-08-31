##############################################################################
#                                                                            #
#              Lazy operator algebra used by the form layer                  #
#                                                                            #
##############################################################################

#=
# linear_operators.jl

The symbolic operator nodes the form layer builds its abstract syntax tree from. A
`LazyOp` records *what* to apply without applying it, so a bilinear form can be written
down as an expression and assembled later.

This is the subset of the file removed in e93d638 that `src/form/` actually needs, and no
more. Sufficiency is not a guess: with this file in place the whole form layer, 2,914
lines across twelve files, compiles.

Left out, and why:

  - `GradientOperator`, and the `scalar` and `codomaintype` traits. Nothing in
    `src/form/` names them. Its twelve uses of `scalar` are reads of the
    `OperatorScale.scalar` field, not calls. `GradientOperator` is also the one piece
    that would reach back into the concrete operators — it was defined as
    `∇₋ₕ(IdentityOperator(space))`, giving `∇₋ₕ` a symbolic method alongside its grid
    function one. Whether the form layer wants that is a design question, so it is not
    restored by default.
  - the `innerₕ` and `inner₊` methods that evaluated these products directly on a
    `VectorElement`. They no longer compile: they read `uₕ.values` where a
    `VectorElement` now stores `data`, and call `innerh_weights(space(uₕ))` and
    `innerplus_weights(W, i)`, which are now `weights(Wₕ, Innerh())` and
    `weights(Wₕ, Innerplus(), i)`. Rewriting them against the current interface is work
    the form layer does not need done to load.

`get_innermost_dim` is kept, unlike the rest of that group:
`src/form/operators/difference.jl` adds methods to it, so the function has to exist for it
to extend them. Its companion `get_derivative_matrix_and_scale` was kept alongside it for
a while and then removed: nothing ever called it — not `src/`, and not `bilinear.jl` or
`linear.jl` either — so it was a matrix-assembly path with no assembler. The form layer
assembles from `local_stencil`. If a matrix path is wanted when `bilinear.jl` returns, it
should be written against what that file actually needs rather than against a guess, and
every operator family now has a matrix form to build it from.

The arithmetic below is written as `Base.:*` and so on rather than as bare `*`. Bramble
does not import those operators, so a bare definition would create a `Bramble.*` distinct
from `Base.*` and shadow it for anyone doing `using Bramble` — the same trap that
`Bramble.values` fell into.
=#

"""
	OperatorType

Supertype of everything the form layer treats as an operator. Carries a `space` field.
"""
abstract type OperatorType end

"""
	LazyOp{D} <: OperatorType

A node of the symbolic operator tree over a `D`-dimensional space. Records an operation
without performing it, so that a form can be written as an expression and assembled later.
"""
abstract type LazyOp{D} <: OperatorType end

@inline space(op::OperatorType) = op.space
@inline eltype(op::OperatorType) = eltype(space(op))

# --- The nodes ------------------------------------------------------------------- #

"""
	IdentityOperator(Wₕ::AbstractSpaceType)

The identity on `Wₕ`, as a symbolic node.
"""
struct IdentityOperator{D, S} <: LazyOp{D}
    space::S
end

"""
	ZeroOperator(Wₕ::AbstractSpaceType)

The zero operator on `Wₕ`, as a symbolic node. Absorbs multiplication by a scalar.
"""
struct ZeroOperator{D, S} <: LazyOp{D}
    space::S
end

@inline IdentityOperator(space::AbstractSpaceType) = IdentityOperator{
    dim(space), typeof(space)}(space)
@inline ZeroOperator(space::AbstractSpaceType) = ZeroOperator{
    dim(space), typeof(space)}(space)

"""
	OperatorScale(α, op::LazyOp)

`op` scaled by the number `α`, as a symbolic node.
"""
struct OperatorScale{D, ScalarType, OpType <: LazyOp{D}} <: LazyOp{D}
    scalar::ScalarType
    inner_op::OpType

    # Written out so that Julia does not also generate a default outer
    # constructor, which would collide with the one defined below.
    function OperatorScale{D, ScalarType, OpType}(
            scalar::ScalarType, inner_op::OpType) where {D, ScalarType, OpType}
        return new{D, ScalarType, OpType}(scalar, inner_op)
    end
end

"""
	GridFunctionScale(vₕ, op::LazyOp)

`op` scaled pointwise by the grid function or function `vₕ`, as a symbolic node.
"""
struct GridFunctionScale{D, VType, OpType <: LazyOp{D}} <: LazyOp{D}
    grid_function::VType
    inner_op::OpType

    # Written out so that Julia does not also generate a default outer
    # constructor, which would collide with the one defined below.
    function GridFunctionScale{D, VType, OpType}(
            grid_function::VType, inner_op::OpType) where {D, VType, OpType}
        return new{D, VType, OpType}(grid_function, inner_op)
    end
end

"""
	OperatorAdd(left::LazyOp, right::LazyOp)

The sum of two symbolic nodes over the same space.
"""
struct OperatorAdd{D, LeftType <: LazyOp{D}, RightType <: LazyOp{D}} <: LazyOp{D}
    left_op::LeftType
    right_op::RightType

    # Written out so that Julia does not also generate a default outer
    # constructor, which would collide with the one defined below.
    function OperatorAdd{D, LeftType, RightType}(
            left_op::LeftType, right_op::RightType) where {D, LeftType, RightType}
        return new{D, LeftType, RightType}(left_op, right_op)
    end
end

@inline OperatorScale(scalar::S, op::LazyOp{D}) where {D, S} = OperatorScale{
    D, S, typeof(op)}(scalar, op)
@inline GridFunctionScale(grid_function::V, op::LazyOp{D}) where {D, V} = GridFunctionScale{
    D, V, typeof(op)}(grid_function, op)
@inline OperatorAdd(left::LazyOp{D}, right::LazyOp{D}) where {D} = OperatorAdd{
    D, typeof(left), typeof(right)}(left, right)

# --- Symbolic or not --------------------------------------------------------------- #

"""
	is_symbolic(op) -> Bool

Whether `op` still contains a symbolic placeholder, such as a trial or test function, and
so cannot be evaluated until one is substituted.

The base cases are here; `src/form/common.jl` adds the methods for its own AST nodes.
"""
function is_symbolic end

is_symbolic(::LazyOp) = false
is_symbolic(ops::Tuple) = any(is_symbolic, ops)

is_symbolic(op::OperatorScale) = is_symbolic(op.inner_op)
is_symbolic(op::GridFunctionScale) = is_symbolic(op.inner_op)
is_symbolic(op::OperatorAdd) = is_symbolic(op.left_op) || is_symbolic(op.right_op)

# --- Display ----------------------------------------------------------------------- #

show(io::IO, ::IdentityOperator) = print(io, "I")
show(io::IO, ::ZeroOperator) = print(io, "0")

# --- Algebra ----------------------------------------------------------------------- #
# Written as `Base.:op` rather than bare `op`; see the note at the top of the file.
#
# Two groups from the original are not here, both because they are unnecessary and
# ambiguous rather than merely unnecessary:
#
#   - the zero-operator absorption, `*(α, ::ZeroOperator) = op`. Written with an untyped
#     `α` it ties with every method below it: `(::ZeroOperator, ::Any)` and
#     `(::LazyOp, ::Number)` are neither more specific than the other. It could be
#     written out once per scalar type, but `src/form/` does not depend on the
#     simplification — it uses `ZeroOperator` only in `resolve_ast` and one evaluation
#     method — and `α * 0` wrapped in an `OperatorScale` is still zero.
#   - the scaling of an `NTuple{D, LazyOp{D}}`. At `D = 0` the empty tuple is both an
#     `NTuple{0, LazyOp{0}}` and an `NTuple{0, VectorElement}`, so those methods tie with
#     the existing tuple arithmetic on grid functions. `src/form/operators/inner.jl`
#     works with tuples of nodes but defines its own `inner₊` over them rather than
#     scaling them with `*`.
#
# Both are user-facing sugar for writing forms down. They belong with whatever syntax the
# form layer settles on, written so as not to be ambiguous, rather than restored blind.

@inline Base.:+(op1::LazyOp{D}, op2::LazyOp{D}) where {D} = OperatorAdd(op1, op2)
# `-1` rather than `-one(Float64)`: the factor promotes against whatever the space's
# element type is, instead of baking a Float64 into the tree and dragging a Float32 or
# extended-precision assembly up to Float64 with it. Same defect as the `* 0.5` that used
# to promote the averaging matrices.
@inline Base.:-(op1::LazyOp{D}, op2::LazyOp{D}) where {D} = op1 + OperatorScale(-1, op2)

@inline Base.:*(c::Number, op::LazyOp) = OperatorScale(c, op)
@inline Base.:*(op::LazyOp, c::Number) = OperatorScale(c, op)
@inline Base.:/(op::LazyOp, c::Number) = OperatorScale(one(c) / c, op)

@inline Base.:*(vₕ::AbstractVector, op::LazyOp) = GridFunctionScale(vₕ, op)
@inline Base.:*(op::LazyOp, vₕ::AbstractVector) = GridFunctionScale(vₕ, op)

@inline Base.:*(vₕ::Function, op::LazyOp) = GridFunctionScale(vₕ, op)
@inline Base.:*(op::LazyOp, vₕ::Function) = GridFunctionScale(vₕ, op)

# --- Traits the form layer extends -------------------------------------------------- #
#
# `src/form/operators/difference.jl` adds the methods for its own difference nodes; what
# lives here is the declaration and the recursion through the two scaling nodes, which
# belongs to this algebra rather than to the form layer.

"""
	get_innermost_dim(op::LazyOp)

The coordinate direction of the difference node at the bottom of `op`.
"""
function get_innermost_dim end

@inline get_innermost_dim(op::OperatorScale) = get_innermost_dim(op.inner_op)
@inline get_innermost_dim(op::GridFunctionScale) = get_innermost_dim(op.inner_op)
