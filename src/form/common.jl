# common.jl

# ==============================================================================
# 1. Zero-Allocation Spacing & Tuple Utilities
# ==============================================================================

@inline _get_component(x::Tuple, dim::Int) = x[dim]
@inline _get_component(x::Number, dim::Int) = x

"""
    get_spacing(mesh, I, dim::Int)

Gets the grid spacing in a given coordinate direction `dim` at Cartesian index `I`.
"""
@inline get_spacing(mesh, I, dim::Int) = _get_component(spacing(mesh, I), dim)

"""
    get_forward_spacing(mesh, I, dim::Int)

Gets the forward grid spacing in a given coordinate direction `dim` at Cartesian index `I`.
"""
@inline get_forward_spacing(mesh, I, dim::Int) = _get_component(forward_spacing(mesh, I), dim)

"""
    get_half_spacing(mesh, I, dim::Int)

Gets the half-grid spacing in a given coordinate direction `dim` at Cartesian index `I`.
"""
@inline get_half_spacing(mesh, I, dim::Int) = _get_component(half_spacing(mesh, I), dim)

"""
    shift_offset(offset::NTuple{D,Int}, dim::Int, delta::Int) where D

Shifts a Cartesian offset tuple by `delta` in dimension `dim`.
"""
@inline shift_offset(offset::NTuple{D, Int}, dim::Int, delta::Int) where {D} = ntuple(
    i -> i == dim ? offset[i] + delta : offset[i], Val(D))

"""
    zero_offset(::Val{D}) where D

Returns a D-tuple of zeros.
"""
@inline zero_offset(::Val{D}) where {D} = ntuple(x -> 0, Val(D))

"""
    shift_stencil(inner::Tuple, ::Val{Dim}, delta)

Shifts all coordinates in a stencil tuple by `delta` in dimension `Dim`.
"""
@generated function shift_stencil(inner::Tuple, ::Val{Dim}, ::Val{Delta}) where {Dim, Delta}
    N = length(inner.parameters)
    exprs = Expr[]
    for i in 1:N
        push!(exprs, :((shift_offset(inner[$i][1], Dim, Delta), inner[$i][2])))
    end
    return Expr(:tuple, exprs...)
end

@generated function shift_stencil(inner::Tuple, ::Val{Dim}, delta::Int) where {Dim}
    N = length(inner.parameters)
    exprs = Expr[]
    for i in 1:N
        push!(exprs, :((shift_offset(inner[$i][1], Dim, delta), inner[$i][2])))
    end
    return Expr(:tuple, exprs...)
end

@generated function concatenate_stencils(left::Tuple, right::Tuple)
    N_left = length(left.parameters)
    N_right = length(right.parameters)
    exprs = Expr[]
    for i in 1:N_left
        push!(exprs, :(left[$i]))
    end
    for i in 1:N_right
        push!(exprs, :(right[$i]))
    end
    return Expr(:tuple, exprs...)
end

@generated function multiply_stencils_bilinear(left::Tuple, right::Tuple, vol::Number)
    N_l = length(left.parameters)
    N_r = length(right.parameters)
    exprs = Expr[]
    for i in 1:N_l, j in 1:N_r

        push!(exprs, :((left[$i][1], right[$j][1], left[$i][2] * right[$j][2] * vol)))
    end
    return Expr(:tuple, exprs...)
end

@generated function multiply_stencils_linear(left::Tuple, right::Tuple, vol::Number)
    N_l = length(left.parameters)
    N_r = length(right.parameters)
    exprs = Expr[]
    for i in 1:N_l, j in 1:N_r

        push!(exprs, :((right[$j][1], left[$i][2] * right[$j][2] * vol)))
    end
    return Expr(:tuple, exprs...)
end

@generated function scale_stencil(inner::Tuple, scalar::Number)
    N = length(inner.parameters)
    exprs = Expr[]
    for i in 1:N
        push!(exprs, :((Base.front(inner[$i])..., inner[$i][end] * scalar)))
    end
    return Expr(:tuple, exprs...)
end

# ==============================================================================
# 2. Abstract Syntax Tree (AST) Nodes
# ==============================================================================

"""
    TrialFunction{D} <: LazyOp{D}

An AST node representing the symbolic trial function \$u\$ in a bilinear form.
"""
struct TrialFunction{D} <: LazyOp{D} end

"""
    TestFunction{D} <: LazyOp{D}

An AST node representing the symbolic test function \$v\$ in a form.
"""
struct TestFunction{D} <: LazyOp{D} end

"""
    IndexedTrialFunction{D} <: LazyOp{D}

An AST node representing the symbolic trial function for a specific **component**
of a composite trial space. Carries a runtime `component_idx` identifying which
leaf scalar space (1-based, depth-first order) it belongs to. Used by
a coupled form to route stencil contributions to the correct block.
"""
struct IndexedTrialFunction{D} <: LazyOp{D}
    component_idx::Int
end

"""
    IndexedTestFunction{D} <: LazyOp{D}

An AST node representing the symbolic test function for a specific **component**
of a composite test space. Carries a runtime `component_idx`. Used by
a coupled form to route stencil contributions to the correct block.
"""
struct IndexedTestFunction{D} <: LazyOp{D}
    component_idx::Int
end

"""
    SourceFunction{D,F} <: LazyOp{D}

An AST node representing a source term defined by a continuous function.
"""
struct SourceFunction{D, F} <: LazyOp{D}
    func::F
end

"""
    SourceVector{D,VType} <: LazyOp{D}

An AST node representing a source term defined by a discrete vector of values.

Note the division of labour with `GridFunctionScale`, which also carries values per
grid point. A `SourceFunction` holds a function of *position*, `f(x)`, evaluated at the
point. A `Function` inside a `GridFunctionScale` is something else entirely: a
**zero-argument thunk** returning the vector or number to scale by, called as `f()` both
here and in `resolve_ast`. It defers building that vector until the form is resolved.

So `(x -> x[1]) * D₋ₓ(u)` does not do what it reads as — the thunk call fails, because the
function wants a point. A function of position belongs in a `SourceFunction`, or should be
restricted to the grid with `Rₕ` first and passed as the vector it becomes.
"""
struct SourceVector{D, VType <: AbstractVector} <: LazyOp{D}
    vec::VType
end

"""
    AbsoluteColumn

A stencil entry's trial slot, naming a column of the trial space directly rather than an
offset from the point being evaluated.

Every other node's stencil says "this many points from here, on the mesh being walked", which
is what lets `shift_stencil` compose operators by relabelling. An interpolation cannot say
that: the trial degrees of freedom it reaches live on a *different* mesh, and which ones
depends on where the point falls (`locate_cell`). So it names them outright, and the three
bilinear consumers resolve the two kinds of entry by dispatch — see
`form/operators/interpolation.jl` (point 61).
"""
struct AbsoluteColumn
    col::Int
end

# ==============================================================================
# 3. Form API & Bramble Standard Mapping
# ==============================================================================

# Indexing a node by component — `v(1)`, and the distribution through whatever is built on
# top of it — lives in form/component.jl, included after the operator files because it needs
# every node type in its signatures.

"""
    trial_function(::Val{D}) where D

Constructs a `TrialFunction` of dimension `D`.
"""
trial_function(::Val{D}) where {D} = TrialFunction{D}()

"""
    test_function(::Val{D}) where D

Constructs a `TestFunction` of dimension `D`.
"""
test_function(::Val{D}) where {D} = TestFunction{D}()

"""
    source_function(f, ::Val{D}) where D

Constructs a `SourceFunction` wrapping function `f`.
"""
source_function(f, ::Val{D}) where {D} = SourceFunction{D, typeof(f)}(f)

# Import modularized operator and product logic
include("operators/difference.jl")
include("operators/jump.jl")
include("operators/average.jl")
include("operators/restriction.jl")
include("operators/inner.jl")
include("operators/interpolation.jl")

# ==============================================================================
# 4. Zero-Allocation Stencil Evaluators
# ==============================================================================

@inline local_stencil(
    ::TrialFunction{D}, space, I::CartesianIndex{D}, markers, lin_idx::Int) where {D} = ((
    zero_offset(Val(D)), 1),)
@inline local_stencil(
    ::TestFunction{D}, space, I::CartesianIndex{D}, markers, lin_idx::Int) where {D} = ((
    zero_offset(Val(D)), 1),)
@inline local_stencil(::IndexedTrialFunction{D}, space, I::CartesianIndex{D},
    markers, lin_idx::Int) where {D} = ((zero_offset(Val(D)), 1),)
@inline local_stencil(::IndexedTestFunction{D}, space, I::CartesianIndex{D},
    markers, lin_idx::Int) where {D} = ((zero_offset(Val(D)), 1),)

@inline function local_stencil(
        op::SourceFunction{D}, space, I::CartesianIndex{D}, markers, lin_idx::Int) where {D}
    m = mesh(space)
    x = point(m, I)
    return ((zero_offset(Val(D)), op.func(x)),)
end

@inline function local_stencil(
        op::SourceVector{D}, space, I::CartesianIndex{D}, markers, lin_idx::Int) where {D}
    return ((zero_offset(Val(D)), op.vec[lin_idx]),)
end

@inline function local_stencil(
        op::OperatorAdd, space, I::CartesianIndex{D}, markers, lin_idx::Int) where {D}
    left_stencil = local_stencil(op.left_op, space, I, markers, lin_idx)
    right_stencil = local_stencil(op.right_op, space, I, markers, lin_idx)
    return concatenate_stencils(left_stencil, right_stencil)
end

@inline function local_stencil(
        op::OperatorScale, space, I::CartesianIndex{D}, markers, lin_idx::Int) where {D}
    inner = local_stencil(op.inner_op, space, I, markers, lin_idx)
    return scale_stencil(inner, op.scalar)
end

@inline function local_stencil(
        op::OperatorScale{D, <:Base.RefValue}, space, I::CartesianIndex{D}, markers, lin_idx::Int) where {D}
    inner = local_stencil(op.inner_op, space, I, markers, lin_idx)
    return scale_stencil(inner, op.scalar[])
end

@inline function local_stencil(
        op::GridFunctionScale, space, I::CartesianIndex{D}, markers, lin_idx::Int) where {D}
    inner = local_stencil(op.inner_op, space, I, markers, lin_idx)
    grid_fn = op.grid_function
    local_val = if grid_fn isa Function
        val = grid_fn()
        val isa Number ? val : val[lin_idx]
    else
        grid_fn isa Number ? grid_fn : grid_fn[lin_idx]
    end
    return scale_stencil(inner, local_val)
end

@inline local_stencil(op::IdentityOperator{D}, space, I::CartesianIndex{D},
    markers, lin_idx::Int) where {D} = ((zero_offset(Val(D)), 1),)
@inline local_stencil(
    op::ZeroOperator{D}, space, I::CartesianIndex{D}, markers, lin_idx::Int) where {D} = ((
    zero_offset(Val(D)), 0),)

# ==============================================================================
# 5. AST Resolution & Thunk Eval
# ==============================================================================

resolve_ast(op::TrialFunction) = op
resolve_ast(op::TestFunction) = op
resolve_ast(op::IndexedTrialFunction) = op
resolve_ast(op::IndexedTestFunction) = op
resolve_ast(op::SourceFunction) = op
resolve_ast(op::SourceVector) = op

function resolve_ast(op::OperatorAdd{D}) where {D}
    OperatorAdd{D, typeof(resolve_ast(op.left_op)), typeof(resolve_ast(op.right_op))}(
        resolve_ast(op.left_op), resolve_ast(op.right_op))
end
function resolve_ast(op::OperatorScale{D}) where {D}
    OperatorScale{D, typeof(op.scalar), typeof(resolve_ast(op.inner_op))}(op.scalar, resolve_ast(op.inner_op))
end

function resolve_ast(op::GridFunctionScale{D, VType}) where {D, VType}
    GridFunctionScale{D, VType, typeof(resolve_ast(op.inner_op))}(op.grid_function, resolve_ast(op.inner_op))
end

function resolve_ast(op::GridFunctionScale{D, <:Function}) where {D}
    vec = op.grid_function()
    return GridFunctionScale{D, typeof(vec), typeof(resolve_ast(op.inner_op))}(vec, resolve_ast(op.inner_op))
end

resolve_ast(op::IdentityOperator) = op
resolve_ast(op::ZeroOperator) = op

resolve_ast(ops::NTuple{N, Any}) where {N} = map(resolve_ast, ops)
resolve_ast(op::Any) = op

# ==============================================================================
# 6. Symbolic AST Traits
# ==============================================================================

# Note: is_symbolic base function is declared in linear_operators.jl

is_symbolic(::TrialFunction) = true
is_symbolic(::TestFunction) = true
is_symbolic(::IndexedTrialFunction) = true
is_symbolic(::IndexedTestFunction) = true
is_symbolic(::SourceFunction) = true
is_symbolic(::SourceVector) = true
is_symbolic(op::BilinearProduct) = true
is_symbolic(op::LinearProduct) = true

is_symbolic(op::BackwardDifference) = is_symbolic(op.inner_op)
is_symbolic(op::ForwardDifference) = is_symbolic(op.inner_op)

is_symbolic(op::BackwardAverage) = is_symbolic(op.inner_op)
is_symbolic(op::ForwardAverage) = is_symbolic(op.inner_op)
is_symbolic(op::ShiftNode) = is_symbolic(op.inner_op)

is_symbolic(op::RegionRestriction) = is_symbolic(op.inner_op)

is_symbolic(op::CenteredDifference) = is_symbolic(op.inner_op)
is_symbolic(op::StarDifference) = is_symbolic(op.inner_op)
is_symbolic(op::CrossWeightedDifference) = is_symbolic(op.inner_op)
is_symbolic(op::JumpNode) = is_symbolic(op.inner_op)

"""
    _is_source_only(op::LazyOp) -> Bool

Whether a `LazyOp` subtree is *source-only*: built entirely from sources
(`SourceFunction`/`SourceVector`) and the plain operators that wrap them, never bottoming
out in a `TrialFunction`/`IndexedTrialFunction` leaf.

`innerₕ`'s `l::Function`/`l::Number`/`l::VectorElement` overloads (`operators/inner.jl`)
never need this: those three types are never anything *but* a source, so wrapping them in a
`LinearProduct` is unconditional. The question only exists for an argument that already
arrived as a `LazyOp` — `πₕ(uₕ)` ([`interpolate_at`](@ref), point 25) or `D₋ₓ(πₕ(uₕ))` are
sources too, just already wrapped, and the generic `innerₕ(::LazyOp, ::LazyOp)` used to build
a `BilinearProduct` regardless, which is the wrong AST shape for a `LinearForm`'s assembly
walk — a `BilinearProduct`'s stencil carries a *pair* of offsets (trial and test), where
`_scatter_term!` (`form/linear.jl`) expects one.

A missing case defaults to `false` (the fallback `::LazyOp` method below) — conservative,
since that is exactly the behaviour every node had before this predicate existed (always
`BilinearProduct`) for anything not explicitly listed as source-only.
"""
_is_source_only(::TrialFunction) = false
_is_source_only(::TestFunction) = false
_is_source_only(::IndexedTrialFunction) = false
_is_source_only(::IndexedTestFunction) = false
_is_source_only(::SourceFunction) = true
_is_source_only(::SourceVector) = true

_is_source_only(op::BackwardDifference) = _is_source_only(op.inner_op)
_is_source_only(op::ForwardDifference) = _is_source_only(op.inner_op)
_is_source_only(op::CenteredDifference) = _is_source_only(op.inner_op)
_is_source_only(op::StarDifference) = _is_source_only(op.inner_op)
_is_source_only(op::CrossWeightedDifference) = _is_source_only(op.inner_op)

_is_source_only(op::BackwardAverage) = _is_source_only(op.inner_op)
_is_source_only(op::ForwardAverage) = _is_source_only(op.inner_op)
_is_source_only(op::ShiftNode) = _is_source_only(op.inner_op)
_is_source_only(op::JumpNode) = _is_source_only(op.inner_op)
_is_source_only(op::RegionRestriction) = _is_source_only(op.inner_op)

_is_source_only(op::OperatorScale) = _is_source_only(op.inner_op)
_is_source_only(op::GridFunctionScale) = _is_source_only(op.inner_op)
function _is_source_only(op::OperatorAdd)
    _is_source_only(op.left_op) &&
        _is_source_only(op.right_op)
end

# A product (whichever kind) is its own thing, not a bare source to route again.
_is_source_only(::BilinearProduct) = false
_is_source_only(::LinearProduct) = false

_is_source_only(::LazyOp) = false

#===========================================================================#
# The *value* of a source-only subtree at a grid point.
#
# `local_stencil` describes an operator as `(offset, coefficient)` pairs relative to the
# point being evaluated, and composes operators by *relabelling* those offsets
# (`shift_stencil`) rather than re-evaluating the inner operator at the shifted point.
# That is exact for anything translation-invariant — a trial function's stencil is the
# same at every point, so relabelling and re-evaluating agree — and it is the whole
# reason the stencil algebra works.
#
# A source is not translation-invariant. Its stencil coefficient *is* a value, read at the
# current point (`op.func(point(m, I))`), so relabelling its offset does not change which
# point it was read at. `D₋ₓ(f)` therefore stencils as `((0, f(xᵢ)/h), ((-1,), -f(xᵢ)/h))`
# — the same value twice — and `multiply_stencils_linear` then contracts the left factor by
# *summing its coefficients* (it keeps only the right operand's offsets), so the two
# cancel and the term assembles to exactly zero. An average sums to `f(xᵢ)`, i.e. the
# operator is silently dropped. Measured, before this existed: `innerₕ(D₋ₓ(f), v)` gave the
# zero vector and `innerₕ(M₋ₓ(f), v)` reproduced `innerₕ(f, v)`.
#
# So a source-only subtree needs the other formulation: evaluate it to its *value* at the
# point, re-reading the source at whatever neighbouring points the operator reaches. Each
# method below mirrors the `local_stencil` of the same node — same masks, same spacings,
# same boundary conventions — differing only in that it reads the source again instead of
# relabelling a coefficient. Both spellings of every operator therefore have to agree, and
# `test/form/source_operators.jl` pins each one against the numeric layer, which is a third,
# independent implementation of the same arithmetic.
#
# `I` moves as the recursion descends, so the linear index is derived from it here rather
# than threaded in: a shifted read needs the shifted index.
#===========================================================================#

@inline _source_lin(space, I) = LinearIndices(indices(mesh(space)))[I]

@inline _in_grid(space, I::CartesianIndex{D}) where {D} = checkbounds(
    Bool, LinearIndices(indices(mesh(space))), I)

@inline function _source_value(
        op::SourceFunction{D}, space, I::CartesianIndex{D}, markers) where {D}
    return op.func(point(mesh(space), I))
end

@inline function _source_value(
        op::SourceVector{D}, space, I::CartesianIndex{D}, markers) where {D}
    return op.vec[_source_lin(space, I)]
end

@inline function _source_value(
        op::OperatorAdd, space, I::CartesianIndex{D}, markers) where {D}
    return _source_value(op.left_op, space, I, markers) +
           _source_value(op.right_op, space, I, markers)
end

@inline function _source_value(
        op::OperatorScale, space, I::CartesianIndex{D}, markers) where {D}
    return op.scalar * _source_value(op.inner_op, space, I, markers)
end

@inline function _source_value(op::OperatorScale{D, <:Base.RefValue}, space,
        I::CartesianIndex{D}, markers) where {D}
    return op.scalar[] * _source_value(op.inner_op, space, I, markers)
end

# The coefficient is read at the point being evaluated, which under an outer difference is
# the *shifted* point — the reading its `local_stencil` twin cannot give, since a relabelled
# offset carries the coefficient found at the unshifted point.
@inline function _source_value(
        op::GridFunctionScale, space, I::CartesianIndex{D}, markers) where {D}
    grid_fn = op.grid_function
    lin = _source_lin(space, I)
    local_val = if grid_fn isa Function
        val = grid_fn()
        val isa Number ? val : val[lin]
    else
        grid_fn isa Number ? grid_fn : grid_fn[lin]
    end
    return local_val * _source_value(op.inner_op, space, I, markers)
end

@noinline function _throw_no_source_value(op)
    throw(ArgumentError(
        "no _source_value method for $(typeof(op)). `_is_source_only` accepted this node " *
        "as a source, so a linear form will try to contract it to a value, and every node " *
        "it can accept needs a value spelling alongside its `local_stencil`. Add one next " *
        "to that node's stencil method, mirroring its masks and spacings, and pin it " *
        "against the numeric operator in test/form/source_operators.jl."))
end

_source_value(op::LazyOp, space, I, markers) = _throw_no_source_value(op)
