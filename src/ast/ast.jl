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
    LazyOp{D}

A node of the symbolic operator tree over a `D`-dimensional space. Records an operation
without performing it, so that a form can be written as an expression and assembled later.
"""
abstract type LazyOp{D} end

# `space` is implemented only by the node that carries a concrete space reference
# (`IdentityOperator`), whereas purely algebraic or symbolic nodes (`TestFunction`,
# `TrialFunction`) receive space context during form assembly.

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

The zero operator on `Wₕ`, as a symbolic node. Absorbs multiplication by a scalar. Keeps
only the dimension of `Wₕ`: two zero operators of one dimension are the same node.
"""
struct ZeroOperator{D} <: LazyOp{D} end

@inline IdentityOperator(space::AbstractSpaceType) = IdentityOperator{dim(space), typeof(space)}(space)

@inline space(op::IdentityOperator) = op.space
@inline ZeroOperator(space::AbstractSpaceType) = ZeroOperator{dim(space)}()

"""
    OperatorScale(α, op::LazyOp)

`op` scaled by the number `α`, as a symbolic node.
"""
struct OperatorScale{D, ScalarType, OpType <: LazyOp{D}} <: LazyOp{D}
    scalar::ScalarType
    inner_op::OpType
end

"""
    GridFunctionScale(vₕ, op::LazyOp)

`op` scaled pointwise by the grid function or function `vₕ`, as a symbolic node.
"""
struct GridFunctionScale{D, VType, OpType <: LazyOp{D}} <: LazyOp{D}
    grid_function::VType
    inner_op::OpType
end

"""
    OperatorAdd(left::LazyOp, right::LazyOp)

The sum of two symbolic nodes over the same space.
"""
struct OperatorAdd{D, LeftType <: LazyOp{D}, RightType <: LazyOp{D}} <: LazyOp{D}
    left_op::LeftType
    right_op::RightType
end

# A sum built by the `εₕ`/`divₕ`/`divₘₕ` `inner₊` builders (operators/difference.jl), with
# the component pair of each term in its type. `P` mirrors the `OperatorAdd`s of `sum`, a
# leaf `(i, j)` for a term whose trial leaves are all `u(i)` and test leaves all `v(j)`, a
# join `(L, R)` for a `+`. `simplify_ast` (simplifier.jl) reads `P` to distribute a scalar
# over the sum by type where `_mixes_components` compares runtime indices, so a captured
# coefficient gives `form` one type (gpena/Bramble.jl#939); an unscaled one is just `sum`.
#
# Nothing past `simplify_ast` sees it. `resolve_ast` and `_resolve_surfaces` (inner.jl)
# rebuild it around their result, and the other walkers before them read its type or return
# it as built. Only a sum whose pairs differ is wrapped, and such a sum names more than one
# component, so a difference, average or restriction over it distributes and simplifies its
# operand (`_LinearWrapper`, simplifier.jl) rather than keep it whole, marker and all.
struct _ComponentSum{D, P, S <: LazyOp{D}} <: LazyOp{D}
    sum::S
end

@inline function _component_sum(s::LazyOp{D}, ::Val{P}) where {D, P}
    return _pairs_differ(P) ? _ComponentSum{D, P, typeof(s)}(s) : s
end

# Whether the leaves of the label tree `P` hold two different pairs. Folds: `P` is a constant.
@inline _pairs_differ(P) = _differs_from(P, _first_pair(P))
@inline _first_pair(P::Tuple{Int, Int}) = P
@inline _first_pair(P::Tuple) = _first_pair(P[1])
@inline _differs_from(P::Tuple{Int, Int}, q) = P !== q
@inline _differs_from(P::Tuple, q) = _differs_from(P[1], q) || _differs_from(P[2], q)

# A scale node: a scalar or a grid function times an operator.
const _Scale = Union{OperatorScale, GridFunctionScale}

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
end

@inline DiracSource{D}(points::P, strengths::S) where {D, P, S} = DiracSource{D, P, S}(points, strengths)

"""
    dirac(x0::Union{Real, NTuple{D, Real}, AbstractVector{<:Real}}, strength = 1) -> DiracSource{D}
    dirac(points::AbstractVector, strengths = 1) -> DiracSource{D}

Construct a symbolic point (Dirac delta) source term at coordinate `x0` with scalar `strength`,
or a collection of point sources at `points` with corresponding `strengths`.

Tuples are the type-stable spelling of a point: `dirac((0.3, 0.4))` and
`dirac([(0.3, 0.4), (0.7, 0.2)])` infer a concrete `DiracSource{2}`. A vector point is accepted,
but its dimension is read from its length at run time, so `dirac([0.3, 0.4])` and
`dirac([[0.3, 0.4], [0.7, 0.2]])` are not inferable; they normalise to the same tuples and
assemble the same load vector. A vector with a real element type (`[0.2, 0.7]`, `Real[0.2, 0.7]`),
or an untyped `Any[0.2, 0.7]` holding only reals, is one point, here the 2D point `(0.2, 0.7)`;
two 1D points are spelt `dirac([(0.2,), (0.7,)])`. A list of points may mix tuples and vectors.

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
- `x0`: Point coordinate, a scalar number (1D), `NTuple{D, Real}` (type-stable), or
  `AbstractVector{<:Real}` (one `D`-dimensional point, not inferable). An empty tuple or vector
  throws an `ArgumentError`.
- `strength`: Source intensity (default: `1`, which takes the space's element type). Accepts any of the following.
  - A constant `Number` (e.g. `2.5`).
  - A dynamic `Ref(val)`, which enables live in-place updates (`strength[] = new_val`) in time-stepping loops without rebuilding the form and with **0 heap allocations**.
  - A zero-argument function thunk, `() -> f(t)`, for time-dependent sources.
- `points`: Collection of point coordinates for multiple simultaneous sources. The first point sets
  the dimension `D`; a later point with a different number of coordinates, or of a kind that
  cannot have `D` coordinates, throws an `ArgumentError` naming its index.
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
function dirac(x0::Real, strength = 1)
    pt = (Float64(x0),)
    return DiracSource{1, typeof(pt), typeof(strength)}(pt, strength)
end

function dirac(x0::NTuple{D, Real}, strength = 1) where {D}
    pt = map(Float64, x0)
    return DiracSource{D, typeof(pt), typeof(strength)}(pt, strength)
end

dirac(::Tuple{}, strength = 1) = _dirac_empty_point("tuple")

function dirac(x0::AbstractVector{<:Real}, strength = 1)
    isempty(x0) && _dirac_empty_point("vector")
    D = length(x0)
    pt = ntuple(d -> Float64(x0[d]), D)
    return DiracSource{D, typeof(pt), typeof(strength)}(pt, strength)
end

# Normalise point `i` of a multi-point `dirac` to an `NTuple{D, Float64}`; a point whose
# length or kind does not match the first one's dimension `D` falls to the named error
@inline _to_tuple_pt(::Val{1}, p::Real, i) = (Float64(p),)
@inline _to_tuple_pt(::Val{D}, p::NTuple{D, Real}, i) where {D} = map(Float64, p)
@inline function _to_tuple_pt(::Val{D}, p::AbstractVector{<:Real}, i) where {D}
    length(p) == D || _dirac_point_mismatch(Val(D), p, i)
    return ntuple(d -> Float64(p[d]), Val(D))
end
_to_tuple_pt(::Val{D}, p, i) where {D} = _dirac_point_mismatch(Val(D), p, i)

@noinline function _dirac_empty_point(kind)
    throw(ArgumentError("dirac requires a point with at least one coordinate, got an empty $kind"))
end

@noinline function _dirac_point_mismatch(::Val{D}, p, i) where {D}
    throw(ArgumentError("dirac point $i is $(repr(p)), but the first point sets the dimension $D: " *
                        "every point needs $D coordinate(s), all spelt as a number (1D), a tuple or a vector"))
end

dirac(pts::AbstractVector{<:Union{Real, NTuple, AbstractVector}}, strengths = 1) = _dirac_points(pts, strengths)

# A list with no narrower element type, such as `[(0.1, 0.2), [0.3, 0.4]]` (a `Vector{Any}`):
# one holding only reals is one point, as for a `Vector{<:Real}`, and any other is a list of points
function dirac(pts::AbstractVector, strengths = 1)
    if !isempty(pts) && _all_real(pts)
        return dirac(map(Float64, pts), strengths)
    end
    return _dirac_points(pts, strengths)
end

# Whether every entry is a real, as a loop: `all` over an untyped list can infer wider than
# `Bool` once an extension adds methods (JET saw `Union{Nothing, Some{Bool}, Bool}`)
function _all_real(pts)
    for p in pts
        p isa Real || return false
    end
    return true
end

function _dirac_points(pts, strengths)
    isempty(pts) && throw(ArgumentError("dirac requires at least one point location"))
    first_pt = first(pts)
    first_pt isa Union{Real, Tuple, AbstractVector} || throw(
        ArgumentError("dirac point 1 is $(repr(first_pt)): a point is a number (1D), a tuple or a vector"),
    )
    D = first_pt isa Real ? 1 : length(first_pt)
    D >= 1 || throw(ArgumentError("dirac point 1 is empty: a point needs at least one coordinate"))
    normalized_pts = [_to_tuple_pt(Val(D), p, i) for (i, p) in enumerate(pts)]
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

The base cases are here; `src/assembly/stencil_eval.jl` adds the methods for the concrete AST
nodes, once every node type exists.
"""
function is_symbolic end

is_symbolic(::LazyOp) = false
is_symbolic(ops::Tuple) = any(is_symbolic, ops)

is_symbolic(op::OperatorAdd) = is_symbolic(op.left_op) || is_symbolic(op.right_op)

# --- Display ----------------------------------------------------------------------- #

show(io::IO, ::IdentityOperator) = print(io, "I")
show(io::IO, ::ZeroOperator) = print(io, "0")

# Debugging-level render, not tuned for large point clouds.
expression(op::DiracSource) = "dirac($(string(op.points)), $(string(op.strengths)))"

# --- Algebra ----------------------------------------------------------------------- #

@inline Base.:+(op1::LazyOp{D}, op2::LazyOp{D}) where {D} = OperatorAdd(op1, op2)
# The sign of `-op` and `a - b`: the `-1` as a singleton type, so `simplify_ast` reads that
# it is neither zero nor one off the type, even when the operator is a runtime value (as in
# `constrained`). An `Int` `-1` stored in the node is collapsed by value
# (gpena/Bramble.jl#240) and makes `form` a `Union` there. Not a `Number`, so Base never
# promotes it and it adds no method that would invalidate Base code on load. `simplify_ast`
# folds it (`_fold_mul`) and `form` lowers what is left to the `Int` `-1` (`_lower_signs`);
# the readers of a raw tree (`expression`, `scale_stencil`, the rank-one split) take it as
# `-1`.
struct _MinusOne end
Base.show(io::IO, ::_MinusOne) = print(io, -1)

@inline Base.:-(op1::LazyOp{D}, op2::LazyOp{D}) where {D} = op1 + OperatorScale(
    _MinusOne(), op2)
@inline Base.:-(op::LazyOp) = OperatorScale(_MinusOne(), op)

@inline Base.:*(c::Number, op::LazyOp) = OperatorScale(c, op)
@inline Base.:*(op::LazyOp, c::Number) = OperatorScale(c, op)
@inline Base.:/(op::LazyOp, c::Number) = OperatorScale(one(c) / c, op)
# An integer divisor that fits in an `Int` scales by the exact `Rational` `1 // c`, which
# promotes against the space's element type as `-1` does above: `one(c) / c` is a `Float64`
# and would widen a `Float32` form (gpena/Bramble.jl#633). `simplify_ast` never multiplies
# or adds a `Rational` scale, so no fold can overflow. `op / 0` is `1 // 0`, an `Inf` as
# before. `typemin(Int)` has no negation in `Int`; `-1 // typemax(Int)` converts to the same
# `-2.0^-63` in `Float64` and `Float32`. `UInt64`, `Int128`, `UInt128` and `BigInt` divisors
# keep the `Number` method above. Limitation: a term paired with its transpose, or a product
# with a scaled argument on each side, multiplies its nested rational scales exactly when
# assembled, so `(innerₕ(D₋ₓ(u), v) / 2^40) / 2^40 + innerₕ(u, D₋ₓ(v))` throws an
# `OverflowError` once their denominators pass `typemax(Int)`.
const _IntDivisor = Union{Bool, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32}
@inline _int_reciprocal(c::Int) = c == typemin(Int) ? -1 // typemax(Int) : 1 // c
@inline Base.:/(op::LazyOp, c::_IntDivisor) = OperatorScale(_int_reciprocal(Int(c)), op)

@inline Base.:*(c::Base.RefValue{<:Number}, op::LazyOp) = OperatorScale(c, op)
@inline Base.:*(op::LazyOp, c::Base.RefValue{<:Number}) = OperatorScale(c, op)

@inline Base.:*(vₕ::AbstractVector, op::LazyOp) = GridFunctionScale(vₕ, op)
@inline Base.:*(op::LazyOp, vₕ::AbstractVector) = GridFunctionScale(vₕ, op)

@inline Base.:*(vₕ::Function, op::LazyOp) = GridFunctionScale(vₕ, op)
@inline Base.:*(op::LazyOp, vₕ::Function) = GridFunctionScale(vₕ, op)

# Operator tuple scaling: c * ∇ₕ(u) and (c1, c2) * ∇ₕ(u)
#
# A coefficient tuple (numbers, grid functions, normal components) scales the operator
# tuple entry by entry, so the two lengths must match: a mismatch throws a
# `DimensionMismatch` instead of letting `map` cut the result to the shorter tuple.
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
@inline function Base.:*(coeffs::Tuple, ops::Tuple{LazyOp, Vararg{LazyOp}})
    length(coeffs) == length(ops) || _throw_coeff_count(length(coeffs), length(ops))
    return map((c, op) -> c * op, coeffs, ops)
end
@inline Base.:-(ops::Tuple{LazyOp, Vararg{LazyOp}}) = map(-, ops)

# Operator tuple sums: 0.1 * ∇ₕ(v) + 3 * ∇ₕ(D₋ₓ(v)) adds entry by entry, and unequal
# lengths throw a `DimensionMismatch` for the same reason as the coefficient tuples above.
@inline function Base.:+(a::Tuple{LazyOp, Vararg{LazyOp}}, b::Tuple{LazyOp, Vararg{LazyOp}})
    length(a) == length(b) || _throw_tuple_lengths(length(a), length(b))
    return map(+, a, b)
end
@inline function Base.:-(a::Tuple{LazyOp, Vararg{LazyOp}}, b::Tuple{LazyOp, Vararg{LazyOp}})
    length(a) == length(b) || _throw_tuple_lengths(length(a), length(b))
    return map(-, a, b)
end

@noinline _throw_tuple_lengths(na::Int, nb::Int) = throw(DimensionMismatch(
    "cannot add or subtract operator tuples of lengths $na and $nb: they must have equal length"))

@noinline _throw_coeff_count(nc::Int, no::Int) = throw(DimensionMismatch(
    "got $nc coefficients for $no operators: a coefficient tuple holds one entry per operator"))
