##############################################################################
#                                                                            #
#              Lazy operator algebra used by the form layer                  #
#                                                                            #
##############################################################################

#=
# ast.jl

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

@inline IdentityOperator(space::AbstractSpaceType) = IdentityOperator{dim(space), typeof(space)}(space)

@inline space(op::IdentityOperator) = op.space
@inline space(op::ZeroOperator) = op.space
@inline ZeroOperator(space::AbstractSpaceType) = ZeroOperator{dim(space), typeof(space)}(space)

"""
    OperatorScale(α, op::LazyOp)

`op` scaled by the number `α`, as a symbolic node.
"""
struct OperatorScale{D, ScalarType, OpType <: LazyOp{D}} <: LazyOp{D}
    scalar::ScalarType
    inner_op::OpType

    function OperatorScale{D, ScalarType, OpType}(
            scalar::ScalarType, inner_op::OpType
    ) where {D, ScalarType, OpType}
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
            grid_function::VType, inner_op::OpType
    ) where {D, VType, OpType}
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
            left_op::LeftType, right_op::RightType
    ) where {D, LeftType, RightType}
        return new{D, LeftType, RightType}(left_op, right_op)
    end
end

@inline OperatorScale(scalar::S, op::LazyOp{D}) where {D, S} = OperatorScale{D, S, typeof(op)}(scalar, op)
@inline GridFunctionScale(grid_function::V, op::LazyOp{D}) where {D, V} = GridFunctionScale{D, V, typeof(op)}(grid_function, op)
@inline OperatorAdd(left::LazyOp{D}, right::LazyOp{D}) where {D} = OperatorAdd{D, typeof(left), typeof(right)}(left, right)

"""
    DiracSource{D, P, S} <: LazyOp{D}

An AST node representing a point (Dirac delta) distribution or collection of point sources.

A `DiracSource` represents the linear functional:
```math
\\ell(v) = \\langle S \\, \\delta_{x_0}, v \\rangle = S \\cdot v(x_0)
```
Discretized on a grid space ``W_h``, an off-node source ``x_0`` falling into cell ``\\text{cell}(idx)``
with fractional coordinates ``t \\in [0, 1]^D`` distributes its strength ``S`` across the ``2^D`` surrounding
cell vertices via multilinear corner weights:
```math
w_c = \\prod_{d=1}^D \\bigl( c_d t_d + (1 - c_d)(1 - t_d) \\bigr), \\qquad c \\in \\{0, 1\\}^D
```
The resulting load vector ``b = \\text{assemble}(l)`` satisfies ``b_{idx + c} = S w_c`` with ``\\sum_c b_{idx + c} = S``,
reproducing exact on-node delta placement (when ``x_0 = x_I``, ``b_I = S``) and second-order accurate functional
contraction ``\\ell(v_h) = S v_h(x_0)`` against discrete grid functions.

See also: [`dirac`](@ref), [`innerₕ`](@ref), [`inner₊`](@ref), [`LinearForm`](@ref).
"""
struct DiracSource{D, P, S} <: LazyOp{D}
    points::P
    strengths::S

    function DiracSource{D, P, S}(points::P, strengths::S) where {D, P, S}
        return new{D, P, S}(points, strengths)
    end
end

@inline DiracSource{D}(points::P, strengths::S) where {D, P, S} = DiracSource{D, P, S}(points, strengths)

"""
    dirac(x0::Union{Real, NTuple{D, Real}, AbstractVector{<:Real}}, strength = 1.0) -> DiracSource{D}
    dirac(points::AbstractVector, strengths = 1.0) -> DiracSource{D}

Construct a symbolic point (Dirac delta) source term at coordinate `x0` with scalar `strength`,
or a collection of point sources at `points` with corresponding `strengths`.

# Mathematical formulation
A point source represents the continuous linear functional:
```math
\\ell(v) = \\int_\\Omega S \\, \\delta(x - x_0) \\, v(x) \\, dx = S \\cdot v(x_0)
```
When assembled into a [`LinearForm`](@ref) via [`innerₕ`](@ref) or [`inner₊`](@ref):
```julia
l = form(Wₕ, v -> innerₕ(dirac(x0, strength), v))
```
the strength ``S`` is distributed over the ``2^D`` bounding cell corners using multilinear
interpolation weights ``w_c``:
```math
b_{idx + c} = S \\cdot w_c, \\qquad \\sum_{c \\in \\{0, 1\\}^D} w_c = 1
```
- **On-grid source** (``x_0 = x_I``): Exactly 1 nonzero entry ``b_I = S``.
- **Off-grid source**: Load distributes locally to the surrounding cell vertices, preserving total integral ``\\sum b_i = S``.
- **Contraction**: Contracting the linear form against a discrete grid function ``v_h \\in W_h`` evaluates ``\\ell(v_h) = b \\cdot v_h \\approx S \\cdot v_h(x_0)`` with ``\\mathcal{O}(h^2)`` accuracy.

# Arguments
- `x0`: Point coordinate: scalar number (1D), `NTuple{D, Real}`, or `AbstractVector{<:Real}`.
- `strength`: Source intensity (default: `1.0`). Accepts:
  - A constant `Number` (e.g. `2.5`).
  - A dynamic `Ref(val)`: Enables live in-place updates (`strength[] = new_val`) in time-stepping loops without rebuilding the form and with **0 heap allocations**.
  - A zero-argument function thunk: `() -> f(t)` for time-dependent sources.
- `points`: Collection of point coordinates for multiple simultaneous sources.
- `strengths`: Matching collection of intensities, or a single scalar broadcast to all points.

# Examples
```julia
using Bramble

Ωₕ = mesh(domain(interval(0.0, 1.0)), 11)
Wₕ = gridspace(Ωₕ)

# 1. On-grid point source
l_on = form(Wₕ, v -> innerₕ(dirac(0.3, 2.0), v))
b_on = assemble(l_on)
sum(b_on) ≈ 2.0

# 2. Dynamic time-stepping point source (0 allocations)
s_live = Ref(1.0)
l_live = form(Wₕ, v -> innerₕ(dirac(0.45, s_live), v))
b = assemble(l_live)

for step in 1:10
    s_live[] = sin(0.1 * step)
    assemble!(b, l_live) # 0 bytes allocated
end
```

See also: [`DiracSource`](@ref), [`innerₕ`](@ref), [`inner₊`](@ref), [`form`](@ref), [`assemble!`](@ref).
"""
function dirac(x0::Real, strength = 1.0)
    pt = (Float64(x0),)
    return DiracSource{1, typeof(pt), typeof(strength)}(pt, strength)
end

function dirac(x0::NTuple{D, Real}, strength = 1.0) where {D}
    pt = map(Float64, x0)
    return DiracSource{D, typeof(pt), typeof(strength)}(pt, strength)
end

function dirac(x0::AbstractVector{<:Real}, strength = 1.0)
    D = length(x0)
    pt = ntuple(d -> Float64(x0[d]), D)
    return DiracSource{D, typeof(pt), typeof(strength)}(pt, strength)
end

@inline _to_tuple_pt(::Val{1}, p::Real) = (Float64(p),)
@inline _to_tuple_pt(::Val{D}, p::NTuple{D, Real}) where {D} = map(Float64, p)
@inline _to_tuple_pt(::Val{D}, p::AbstractVector{<:Real}) where {D} = ntuple(d -> Float64(p[d]), Val(D))

function dirac(pts::AbstractVector{<:Union{Real, NTuple, AbstractVector}}, strengths = 1.0)
    isempty(pts) && throw(ArgumentError("dirac requires at least one point location"))
    first_pt = first(pts)
    D = first_pt isa Real ? 1 : length(first_pt)
    normalized_pts = [_to_tuple_pt(Val(D), p) for p in pts]
    normalized_strengths = if strengths isa Number || strengths isa Base.RefValue || strengths isa Function
        fill(strengths, length(pts))
    else
        length(strengths) == length(pts) || throw(
            ArgumentError(
            "length of strengths ($(length(strengths))) must match length of points ($(length(pts)))",
        ),
        )
        collect(strengths)
    end
    return DiracSource{D, typeof(normalized_pts), typeof(normalized_strengths)}(
        normalized_pts, normalized_strengths
    )
end

# --- Symbolic or not --------------------------------------------------------------- #

"""
    is_symbolic(op) -> Bool

Whether `op` still contains a symbolic placeholder, such as a trial or test function, and
so cannot be evaluated until one is substituted.

The base cases are here; `src/form/stencil_eval.jl` adds the methods for the concrete AST
nodes, once every node type exists.
"""
function is_symbolic end

is_symbolic(::LazyOp) = false
is_symbolic(ops::Tuple) = any(is_symbolic, ops)

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

# Operator tuple scaling: c * ∇ₕ(u) and (c1, c2) * ∇ₕ(u)
#
# `Tuple{LazyOp, Vararg{LazyOp}}` rather than `Tuple{Vararg{LazyOp}}`: the latter also
# matches the empty tuple `()`, which is ambiguous with `vectorelement.jl`'s own
# `NTuple{D, VectorElement}` scaling methods at `D = 0`. Excluding it here is free -- an
# empty operator tuple is not a real value ∇ₕ ever produces.
@inline Base.:*(
    c::Union{Number, AbstractVector, Function, Base.RefValue{<:Number}},
    ops::Tuple{LazyOp, Vararg{LazyOp}}
) = map(op -> c * op, ops)
@inline Base.:*(
    ops::Tuple{LazyOp, Vararg{LazyOp}},
    c::Union{Number, AbstractVector, Function, Base.RefValue{<:Number}}
) = c * ops
@inline Base.:*(coeffs::Tuple, ops::Tuple{LazyOp, Vararg{LazyOp}}) = map((c, op) -> c * op, coeffs, ops)
