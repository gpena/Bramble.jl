##############################################################################
#                                                                            #
#         Rendering a resolved form AST in Bramble's operator notation      #
#                                                                            #
##############################################################################

#=
# expression.jl

`expression(form)` / `expression(ast::LazyOp)` render a resolved `LazyOp` tree (the
post-`simplify_ast` AST stored in `form.ast`) as a string in Bramble's own operator notation,
e.g. `innerₕ(D₋ₓ(u), v)`, instead of the raw nested struct type.

One `expression` method per node type, dispatched on the node's concrete type. This file
covers only the algebra nodes (`OperatorAdd`, `OperatorScale`, `GridFunctionScale`) and the two
space leaves (`IdentityOperator`, `ZeroOperator`); every other node type (trial/test functions,
inner products, directional differences, interpolation, restriction, ...) is added by later
files, each adding its own `expression` methods without touching this one.
=#

"""
    expression(form::Union{LinearForm,BilinearForm}) -> String

Render `form`'s resolved AST (`form.ast`) in Bramble's own operator notation.

Left untyped on purpose: `LinearForm`/`BilinearForm` are defined in `form/linear.jl` and
`form/bilinear.jl`, included well after this file, so a `Union{LinearForm,BilinearForm}`
annotation here would need those names to exist at `include` time. Every `LazyOp` node has its
own, more specific `expression` method, so this untyped fallback only ever catches forms.
"""
expression(form) = expression(form.ast)

# --- Scalar formatting ------------------------------------------------------------- #

# Shared by every node that carries a numeric coefficient: reads through a `Base.RefValue`,
# and drops a trailing `.0` from integer-valued reals (`2.0` -> `"2"`, `-1.0` -> `"-1"`).
_format_scalar(c::Base.RefValue) = _format_scalar(c[])
_format_scalar(c::Real) = isinteger(c) ? string(Integer(c)) : string(c)
_format_scalar(c) = string(c)

# --- Parenthesization -------------------------------------------------------------- #

# An `OperatorAdd` nested inside an `OperatorAdd`/`OperatorScale` needs parens to keep
# precedence unambiguous; a scale or a leaf never does (its own rendering already delimits it).
_paren_if_add(op::OperatorAdd) = "($(expression(op)))"
_paren_if_add(op::LazyOp) = expression(op)

# --- Space leaves ------------------------------------------------------------------- #

expression(::IdentityOperator) = "I"
expression(::ZeroOperator) = "0"

# --- Scale nodes -------------------------------------------------------------------- #

function expression(op::OperatorScale)
    inner = _paren_if_add(op.inner_op)
    formatted = _format_scalar(op.scalar)
    return formatted == "-1" ? "-$(inner)" : "$(formatted) * $(inner)"
end

# No name to recover for an arbitrary vector/function coefficient -- always the placeholder.
expression(op::GridFunctionScale) = "vₕ * $(expression(op.inner_op))"

# --- Sum nodes ------------------------------------------------------------------------ #

function expression(op::OperatorAdd)
    left = _paren_if_add(op.left_op)
    right_op = op.right_op
    if right_op isa OperatorScale && right_op.scalar isa Real && right_op.scalar < 0
        flip = expression(OperatorScale(-right_op.scalar, right_op.inner_op))
        return "$(left) - $(flip)"
    end
    return "$(left) + $(_paren_if_add(right_op))"
end
