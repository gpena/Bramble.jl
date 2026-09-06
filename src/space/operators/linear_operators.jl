##############################################################################
#                                                                            #
#              Lazy operator algebra used by the form layer                  #
#                                                                            #
##############################################################################

#=
# linear_operators.jl

The symbolic operator nodes the form layer builds its abstract syntax tree from. A
`LazyOp` records *what* to apply without applying it, allowing a bilinear form to be written
down as an expression and assembled later into a matrix or local stencil.

All algebra over symbolic operators extends `Base.:+`, `Base.:-`, `Base.:*`, and `Base.:/`
directly rather than defining separate unexported aliases.
=#

"""
    OperatorType

Supertype of everything the form layer treats as an operator.
"""
abstract type OperatorType end

"""
    LazyOp{D} <: OperatorType

A node of the symbolic operator tree over a `D`-dimensional space. Records an operation
without performing it, so that a form can be written as an expression and assembled later.
"""
abstract type LazyOp{D} <: OperatorType end

# `space` is implemented only by nodes that carry a concrete space reference (`IdentityOperator`,
# `ZeroOperator`), whereas purely algebraic or symbolic nodes (`TestFunction`, `TrialFunction`)
# receive space context during form assembly.

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

@inline space(op::IdentityOperator) = op.space
@inline space(op::ZeroOperator) = op.space
@inline ZeroOperator(space::AbstractSpaceType) = ZeroOperator{
    dim(space), typeof(space)}(space)

"""
    OperatorScale(α, op::LazyOp)

`op` scaled by the number `α`, as a symbolic node.
"""
struct OperatorScale{D, ScalarType, OpType <: LazyOp{D}} <: LazyOp{D}
    scalar::ScalarType
    inner_op::OpType

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

@inline Base.:+(op1::LazyOp{D}, op2::LazyOp{D}) where {D} = OperatorAdd(op1, op2)
# `-1` promotes against whatever the space's element type is, preserving precision.
@inline Base.:-(op1::LazyOp{D}, op2::LazyOp{D}) where {D} = op1 + OperatorScale(-1, op2)

@inline Base.:*(c::Number, op::LazyOp) = OperatorScale(c, op)
@inline Base.:*(op::LazyOp, c::Number) = OperatorScale(c, op)
@inline Base.:/(op::LazyOp, c::Number) = OperatorScale(one(c) / c, op)

@inline Base.:*(c::Base.RefValue{<:Number}, op::LazyOp) = OperatorScale(c, op)
@inline Base.:*(op::LazyOp, c::Base.RefValue{<:Number}) = OperatorScale(c, op)

@inline Base.:*(vₕ::AbstractVector, op::LazyOp) = GridFunctionScale(vₕ, op)
@inline Base.:*(op::LazyOp, vₕ::AbstractVector) = GridFunctionScale(vₕ, op)

@inline Base.:*(vₕ::Function, op::LazyOp) = GridFunctionScale(vₕ, op)
@inline Base.:*(op::LazyOp, vₕ::Function) = GridFunctionScale(vₕ, op)
