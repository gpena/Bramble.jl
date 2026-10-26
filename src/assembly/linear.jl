# --- Struct definitions ----------------------------------------------------------- #

"""
    LinearForm{D, TestSpace, AST}

Represents a linear form defined over a test space.

# Arguments
- `test_space::TestSpace`: Space for the test function.
- `ast::AST`: Resolved expression tree.

The form resolves its expression tree `ast` once at construction, referencing the underlying
storage of any coefficient grid functions (`VectorElement`). In-place updates via `Rₕ!(fₕ, ...)`
or `parent(fₕ) .= ...` are automatically seen by subsequent assemblies with zero heap allocations.
The expression itself is not retained: downstream routines evaluate the resolved AST directly.

Constant scalar coefficients can be written directly as numbers (e.g. `2.5 * innerₕ(fₕ, v)`).
`Ref` is only needed if a dynamic scalar coefficient changes across loop iterations:
```julia
α = Ref(1.0)
l = form(Wₕ, v -> α * innerₕ(fₕ, v))
# Inside time loop:
α[] = 2.5
assemble!(b, l) # zero allocations, evaluates with α = 2.5
```
"""
struct LinearForm{D, TestSpace, AST}
    test_space::TestSpace
    ast::AST
end

"""
    test_space(form::LinearForm)

Return the test space of the linear form.
"""
test_space(form::LinearForm) = form.test_space

# A linear form is a functional on its test space, contracting against an element of that space.
#
# Requiring a `VectorElement` prevents errors on composite spaces where bare vector lengths
# do not convey block alignments.
@inline function (form::LinearForm)(vₕ::VectorElement)
    v = parent(vₕ)
    _check_contract_length(form.test_space, v)
    return _contract_linear(locality(backend(form.test_space)), form, v)
end

@inline function _contract_linear(::HostLocality, form::LinearForm, v::AbstractVector)
    ast = form.ast
    space = form.test_space
    T = promote_type(_assembled_eltype(ast, space), eltype(v))

    return _contract_linear_core(space, ast, v, zero(T))
end

# The fused sweep reads `v` and the space one grid point at a time, which a device array
# refuses. A device test space assembles the vector instead (`assemble`, a host-mirror
# sweep and one upload) and contracts it against `v` without conjugating: a real vector
# through `dot`, the device's own reduction, anything else on host copies of both.
function _contract_linear(::DeviceLocality, form::LinearForm, v::AbstractVector)
    b = assemble(form)
    eltype(b) <: Real && return _contract_values(b, v)
    return _contract_values(Array(b), Array(v))
end

# The contraction reads `v` with `@inbounds` at the rows the sweep reaches, so an element of
# another space must be refused before it starts: a short one would be read out of bounds, a
# long one would be summed as if its tail were absent.
@inline function _check_contract_length(space, v::AbstractVector)
    Base.require_one_based_indexing(v)
    length(v) == ndofs(space) || _throw_contract_length(ndofs(space), length(v))
    return nothing
end

@noinline function _throw_contract_length(expected::Int, actual::Int)
    throw(
        DimensionMismatch(
        "a linear form on a test space with $expected degrees of freedom cannot contract " *
        "a vector of length $actual",
    ),
    )
end

@noinline function (form::LinearForm)(v::AbstractVector)
    throw(
        ArgumentError(
        "a linear form contracts against an element of its test space, not a bare vector: " *
        "the length of a vector says nothing about whether its blocks match the components " *
        "the form routes to. Name the space first, with l(element(test_space(l), v)).",
    ),
    )
end

"""
    evaluate!(scratch::AbstractVector, form::LinearForm, vₕ::VectorElement) -> Number

Evaluate `form` at `vₕ`, assembling into `scratch` rather than into a newly allocated vector, and
return the resulting contracted scalar value.

Useful when both the assembled vector and the scalar value are needed (such as a Newton step
requiring the residual vector and its norm). `scratch` is overwritten and contains the right-hand
side upon return.

For the scalar value alone, `form(vₕ)` fuses the contraction into the assembly sweep in a single pass.

# Examples
```julia
using Bramble: evaluate!
l = form(Wₕ, v -> innerₕ(fₕ, v))
scratch = zeros(ndofs(Wₕ))
for step in 1:nsteps
    Rₕ!(fₕ, source_at(step))          # modified in-place
    value = evaluate!(scratch, l, uₕ)
end
```
"""
@inline function evaluate!(
        scratch::AbstractVector, form::LinearForm, vₕ::VectorElement
)
    _assemble_linear!(scratch, form, form.ast, nothing, nothing)
    return _contract_values(scratch, parent(vₕ))
end

# `l(vₕ)` contracts without conjugating, so neither may this: `dot` conjugates its first
# argument and would return conj(l(vₕ)) for a complex source. A real `a` has nothing to
# conjugate, and keeps `dot` (BLAS on the host, and a device array's own reduction).
_contract_values(a::AbstractVector{<:Real}, b::AbstractVector) = dot(a, b)

function _contract_values(a::AbstractVector, b::AbstractVector)
    acc = zero(promote_type(eltype(a), eltype(b)))
    @inbounds @simd for i in eachindex(a, b)
        acc += a[i] * b[i]
    end
    return acc
end

@noinline function evaluate!(::AbstractVector, ::LinearForm, v::AbstractVector)
    throw(
        ArgumentError(
        "a linear form contracts against an element of its test space, not a bare vector: " *
        "the length of a vector says nothing about whether its blocks match the components " *
        "the form routes to. Name the space first, with " *
        "evaluate!(scratch, l, element(test_space(l), v)).",
    ),
    )
end

"""
    resolve_form_ast(form::LinearForm)

Return the resolved AST stored inside the linear form.
"""
@inline resolve_form_ast(form::LinearForm) = form.ast

# The kind (`Val(:linear)` or `Val(:bilinear)`) only picks the error message, so the
# success path builds no string. The throwers take it untyped here, which keeps the
# matching-dimension method the most specific one for every kind.
@inline _validate_form_expression(::LazyOp{D}, ::Val{D}, ::Val) where {D} = nothing
@inline _validate_form_expression(bad, ::Val, kind) = _throw_form_value(bad, kind)
@inline _validate_form_expression(::LazyOp{E}, ::Val{D}, kind) where {E, D} = _throw_form_dimension(E, D, kind)

@noinline function _throw_form_value(bad, ::Val{:linear})
    throw(
        ArgumentError(
        "a linear form's expression has to build an operator over its test argument, and " *
        "this one returned a $(typeof(bad)). Write it as a function of the test argument (`v -> innerₕ(fₕ, v)`) " *
        "rather than as a value.",
    ),
    )
end

@noinline function _throw_form_value(bad, ::Val{:bilinear})
    throw(
        ArgumentError(
        "the expression of a bilinear form has to build an operator over its trial and test " *
        "arguments, and this one returned a $(typeof(bad)). Write it as a function of both " *
        "(`(u, v) -> innerₕ(u, v)`) rather than as a value.",
    ),
    )
end

@noinline function _throw_form_dimension(E, D, ::Val{:linear})
    throw(
        ArgumentError(
        "a linear form's expression is $(E)-dimensional and its test space is $(D). The " *
        "operators in the expression have to come from the same space the form is built " *
        "over.",
    ),
    )
end

@noinline function _throw_form_dimension(E, D, ::Val{:bilinear})
    throw(
        ArgumentError(
        "the expression of a bilinear form is $(E)-dimensional and its trial space is $(D). " *
        "The operators in the expression have to come from the spaces the form is built " *
        "over.",
    ),
    )
end

# Two node types `_lower_sources` (ast/common.jl) needs its own method for, defined here
# rather than alongside the rest of the family there because both are included after
# common.jl -- a type named in a method *signature* has to already exist (bramble-performance
# skill, "include-order rule"), unlike a name used only in a body.

# `LinearProduct`'s left side is always a source (`_is_source_only`), its right side never
# is, so only the left needs lowering.
function _lower_sources(op::LinearProduct{D, W}, space) where {D, W}
    left = _lower_sources(op.left_op, space)
    return left === op.left_op ? op :
           LinearProduct{D, W, typeof(left), typeof(op.right_op)}(op.inner, left, op.right_op)
end

function _lower_sources(op::ShiftNode{D, Dim}, space) where {D, Dim}
    inner = _lower_sources(op.inner_op, space)
    return inner === op.inner_op ? op : ShiftNode{D, Dim, typeof(inner)}(op.shift_amount, inner)
end

# --- Eager source lowering across a CompositeGridSpace's leaves --- #
#
# A non-composite space has exactly one leaf -- itself -- so `_lower_sources` (ast/common.jl,
# and the methods above) runs directly against `Wₕ`, unambiguous. A composite space's terms
# follow the routing rule the assembly walks use (`_routed_target`, below): a term naming one
# component lowers against that leaf's own space; a term naming none goes to *every* leaf,
# which may have different meshes, so there is no single space to sample against and lowering
# is skipped for it -- the same `SourceFunction` it already was, evaluated fresh per leaf at
# assembly time.
# Missing this optimisation for that one shape is the safe choice: sampling against the wrong
# leaf's mesh would silently assemble the wrong numbers, not merely run slower. A component
# outside the space throws here, at `form`, through `_routed_target`'s check; returning the
# term unchanged for it made the result a Union and deferred the error to assembly.
# On a non-composite space a term can still name a component, by re-indexing its one leaf
# (`v(1)(2)`, ast/component.jl), and nothing downstream reads it, so it is checked against
# the space's one component here too (gpena/Bramble.jl#638).
function _lower_sources_for_space(ast, Wₕ)
    _check_scalar_components(ast, test_component_or_nothing, "test")
    return _lower_term(ast, Wₕ)
end

# Each term's component on one side, read by `side_component`, against the one component a
# non-composite space has. `bilinear.jl`'s `form` runs it for a non-composite side, too.
function _check_scalar_components(op::OperatorAdd, side_component::F, side) where {F}
    _check_scalar_components(op.left_op, side_component, side)
    _check_scalar_components(op.right_op, side_component, side)
    return nothing
end

_check_scalar_components(term, side_component::F, side) where {F} = _check_component(
    side_component(term), 1, side)

function _lower_sources_for_space(ast, Wₕ::CompositeGridSpace)
    return _lower_sources_over_leaves(ast, leaf_spaces_offsets(Wₕ))
end

function _lower_sources_over_leaves(op::OperatorAdd, leaves)
    left = _lower_sources_over_leaves(op.left_op, leaves)
    right = _lower_sources_over_leaves(op.right_op, leaves)
    return left === op.left_op && right === op.right_op ? op : OperatorAdd(left, right)
end

function _lower_sources_over_leaves(term::TERM, leaves) where {TERM}
    target = _routed_target(term, length(leaves))
    target === nothing && return term
    return _at_leaf(leaf -> _lower_term(term, first(leaf)), leaves, target)
end

# A term's sources lowered against the leaf `sp` it assembles on, then its restricted
# sources typed against the same leaf. Host spaces only: the probe reads a point and a
# source value one at a time, a scalar read a device array refuses.
_lower_term(term, sp) = _typed_restrictions(
    locality(backend(sp)), _lower_sources(term, sp), sp)
function _typed_restrictions(::HostLocality, op, sp)
    _any_restriction(op) || return op
    return _typed_restrictions(op, sp, trues(length(indices(mesh(sp)))))
end
_typed_restrictions(::DeviceLocality, op, sp) = op

# Whether `_typed_restrictions` below would reach a `RegionRestriction`: without one, `form`
# allocates no point mask (the `form (linear, 2D)` allocation bound in benchmark/).
@inline _any_restriction(::RegionRestriction) = true
@inline _any_restriction(op::OperatorAdd) = _any_restriction(op.left_op) ||
                                            _any_restriction(op.right_op)
@inline _any_restriction(op::LinearProduct) = _any_restriction(op.left_op)
@inline _any_restriction(op::UnaryWrapper) = _any_restriction(op.inner_op)
@inline _any_restriction(::Any) = false

# --- A restricted source keeps one stencil type (gpena/Bramble.jl#639) --- #
#
# Outside its region a `RegionRestriction` answers `()` and inside it its operand's stencil,
# so a restricted source's `local_stencil` infers a `Union` of the two, and every node above
# it inherits it. Here, once, at `form`, each restriction of a source-only operand
# (`_is_source_only`) is evaluated at one point of its region, on the space it assembles on,
# and stores a zero stencil of what it found there (`_zero_entries`): outside the region it
# answers that instead of `()` (`_outside_stencil`, operators/region_restriction.jl), so its
# stencil has one length and one type at every point. No inference is run on the source.
# The type is decided by `form`, not per walk, so a walk's bound term has one type whatever
# the regions hold when it runs; region membership itself is still read live, per walk.
#
# The source is called only at a point where an assembly calls it anyway: `reach` marks the
# points (linear indices) at which the walk evaluates the node, narrowed by every enclosing
# restriction and moved by every enclosing `ShiftNode`, so a source undefined outside its
# region, or outside an enclosing one, is never called there. No such point (an empty
# region, a label the mesh lacks yet) leaves the node as it was, answering `()`. The source
# itself is not sampled: it is called again at every point of its region on every assembly.
# Nested restrictions are typed inner first, so an outer probe outside an inner region reads
# the inner zero stencil.
_typed_restrictions(op, sp, reach) = op

function _typed_restrictions(op::UnaryWrapper, sp, reach)
    return _rewrap_inner(op, _typed_restrictions(op.inner_op, sp, reach))
end

# A shift evaluates its operand at `I + shift` for each `I` it is evaluated at.
function _typed_restrictions(op::ShiftNode{D, Dim}, sp, reach) where {D, Dim}
    step = _stencil_step(Val(Dim), Val(D)) * op.shift_amount
    moved = _shift_reach(CartesianIndices(indices(mesh(sp))), step, reach)
    return _rewrap_inner(op, _typed_restrictions(op.inner_op, sp, moved))
end

@noinline function _shift_reach(cart::CartesianIndices, step::CartesianIndex, reach::BitVector)
    lin_of = LinearIndices(cart)
    moved = falses(length(reach))
    for lin in findall(reach)
        J = cart[lin] + step
        checkbounds(Bool, lin_of, J) && (moved[lin_of[J]] = true)
    end
    return moved
end

function _typed_restrictions(op::OperatorAdd{D}, sp, reach) where {D}
    left = _typed_restrictions(op.left_op, sp, reach)
    right = _typed_restrictions(op.right_op, sp, reach)
    return left === op.left_op && right === op.right_op ? op :
           OperatorAdd{D, typeof(left), typeof(right)}(left, right)
end

function _typed_restrictions(op::LinearProduct{D, W}, sp, reach) where {D, W}
    left = _typed_restrictions(op.left_op, sp, reach)
    return left === op.left_op ? op :
           LinearProduct{D, W, typeof(left), typeof(op.right_op)}(op.inner, left, op.right_op)
end

function _typed_restrictions(op::RegionRestriction, sp, reach)
    Ωₕ = mesh(sp)
    inside = _probe_reach(_marker_words(Ωₕ), _probe_ids(op.region, Ωₕ), reach)
    inner = _typed_restrictions(op.inner_op, sp, inside)
    _is_source_only(inner) || return _rebuild_restriction(op, op.region, inner)
    lin = findfirst(inside)
    lin === nothing && return _rebuild_restriction(op, op.region, inner)
    I = CartesianIndices(indices(Ωₕ))[lin]
    bound, mk = _bind_walk(inner, Ωₕ)
    z = _zero_entries(local_stencil(bound, sp, I, mk, lin))
    return RegionRestriction{_dim(op), typeof(op.region), typeof(inner), typeof(z)}(
        op.region, inner, z)
end

# The points of `reach` inside the region: out of line, typed by the region and the word
# matrix only, so a new source does not compile it again.
@noinline function _probe_reach(words::AbstractMatrix{UInt64}, ids, reach::BitVector)
    inside = falses(length(reach))
    for lin in findall(reach)
        _in_probe_region(words, ids, lin) && (inside[lin] = true)
    end
    return inside
end

@inline _dim(::LazyOp{D}) where {D} = D

# The zero of every coefficient, offsets kept: `zero` rather than a `false` scale, so the
# zero is `+0.0` whatever the sign of the probed value.
# `_each_entry` rather than `map`, which boxes past 32 entries.
@inline _zero_entries(s::Tuple) = _each_entry(t -> (Base.front(t)..., zero(t[end])), s)

# A label's column in the word matrix, or 0 for a label the mesh lacks: the walk reports
# that one (`_validate_term_markers`), not `form`.
@inline _probe_ids(label::Symbol, Ωₕ) = get(_marker_ids(Ωₕ), label, 0)
@inline _probe_ids(labels::NTuple{N, Symbol}, Ωₕ) where {N} = map(l -> _probe_ids(l, Ωₕ), labels)

# Whether point `lin` is in the region; a label the mesh lacks (id 0) marks no point.
@inline _in_probe_region(words, id::Int, lin::Int) = id != 0 && _is_marked(words, id, lin)
@inline _in_probe_region(words, ids::Tuple, lin::Int) = any(id -> _in_probe_region(words, id, lin), ids)

"""
    form(Wₕ, f) -> LinearForm

Construct a `LinearForm` over the test space `Wₕ` using the linear expression `f`.

Construction resolves the AST once and runs [`simplify_ast`](@ref) over it -- dropping zero
terms, folding nested constant scalings, lifting scalars out of inner products, and
distributing sums that mix components -- before it is stored; a sum is otherwise kept as
written.
Grid partitioning for parallel assembly is determined from the resolved AST during assembly
(see `_term_colour_strides`).

Every `SourceFunction` reachable from the simplified AST -- a source term built directly from
a plain function, `f(x)`, rather than a [`VectorElement`](@ref) or a `Ref` -- is then
sampled once against its own leaf's space and lowered to a `SourceVector`
because a brand-new closure passed as a source would force a full
recompilation of the assembly pipeline on every distinct closure, since
Julia gives it its own type; assembling against the fixed `SourceVector` shape instead means
that cost is paid once, here, not on every later `assemble!`/`assemble` call. This changes
what a raw closure that captures mutable state does: it is evaluated once, now, not
re-evaluated on later assemblies. `VectorElement` and `Ref` coefficients are unaffected --
see the note on `_lower_sources` in `ast/common.jl` for why, and for the documented
alternative (`update_coefficients!`) a source meant to keep varying should use instead.

The one exception is an interpolant [`πₕ`](@ref)`(uₕ)`, which is never sampled: every
`assemble`/`assemble!` evaluates it on `uₕ`'s current values, and throws an `ArgumentError`
once `uₕ`'s mesh has moved.

A source under `restrict_to` is also called once here, at the first point of its region
that the assembly reads (inside every enclosing region, moved by every enclosing shift), so
that outside its region it answers a zero of the type it has inside and its stencil keeps
one type at every point. It is never called at a point no assembly reads, with one
exception: the point comes from the markers as they stand at `form`, so after `markers!`
moves the region the source has been called once at a point of the old one. A region with
no such point at `form` keeps the empty stencil. In return:

  - a source that throws at a point of its region throws here, not at the first assembly;
  - a coefficient that is not finite outside the region gives `NaN` there when it scales a
    shift, a difference, an average or a sum holding the restricted source (directly over
    the restriction, or over restrictions and scales nested in it, it still gives 0);
  - [`assemble_add!`](@ref) into entries holding `-0.0` may turn them into `+0.0` there.

# Examples

```jldoctest
using Bramble
Wₕ = gridspace(mesh(domain(interval(0.0, 1.0)), 11))
fₕ = Rₕ(Wₕ, x -> 1.0)
l = form(Wₕ, v -> innerₕ(fₕ, v))    # l(v) = (fₕ, v)ₕ
isapprox(sum(assemble(l)), 1.0; atol = 1.0e-12)

# output
true
```
"""
function form(Wₕ, f)
    D = dim(Wₕ)
    raw_ast = f(test_function(Wₕ))
    _validate_form_expression(raw_ast, Val(D), Val(:linear))
    E = eltype(Wₕ)
    # Real coefficients made `Float64` first where that keeps the bits (bilinear.jl, #912).
    ast = _split_restricted_products(
        _normalise_scalars(
            _lower_signs(simplify_ast(_resolve_surfaces_for_space(resolve_ast(raw_ast), Wₕ))),
            _leaf_eltype(Wₕ)),
        E)
    _check_centered_meshes(ast, Wₕ)
    ast = _lower_sources_for_space(ast, Wₕ)
    _check_source_meshes(ast, Wₕ)
    return LinearForm{D, typeof(Wₕ), typeof(ast)}(Wₕ, ast)
end

# The type of the weights the stencil of `T` holds. A constant, `Ref` or grid function
# scale multiplies its operand's by its value type, a difference, average or jump by an `E`
# (the spaces' element type, as `1/h` and `1/2` are), a linear product its test side's by
# its source's and the `E` volume, and a test or trial function's weight is an `L`. `Any`
# where a value type is not known from the type (a function thunk, a point source), and
# nothing is then rewritten or regrouped. With `L = E` it is the type the weights join to
# once each is multiplied by an `E` volume, the target the widening converts scales to
# (`π * 1f0` is a `Float32`); `_type_weight` is the type they have. Defined here, before
# the generators that read it, bilinear.jl's `_needs_rewrite` and `_values_differ` below.
function _type_value(T, E, L = E)
    T isa DataType || return Any
    if T <: OperatorAdd
        return promote_type(
            _type_value(T.parameters[2], E, L), _type_value(T.parameters[3], E, L))
    elseif T <: OperatorScale
        S = T.parameters[2]
        S isa DataType && S <: Base.RefValue && (S = S.parameters[1])
        return Base.promote_op(*, S, _type_value(T.parameters[3], E, L))
    elseif T <: GridFunctionScale
        V = T.parameters[2]
        S = V <: Number ? V : V <: AbstractArray ? eltype(V) : Any
        return Base.promote_op(*, S, _type_value(T.parameters[3], E, L))
    elseif T <: LinearProduct
        return Base.promote_op(
            *, Base.promote_op(*, _type_value(T.parameters[3], E, L),
                _type_value(T.parameters[4], E, L)), E)
    elseif T <: Union{_DifferenceNode, _AverageNode}
        return Base.promote_op(*, E, _type_value(fieldtype(T, :inner_op), E, L))
    elseif T <: UnaryWrapper
        return _type_value(fieldtype(T, :inner_op), E, L)
    elseif T <: SourceVector
        return eltype(T.parameters[2])
    elseif T <: SourceConstant
        return T.parameters[2]
    elseif T <: Union{SourceFunction, DiracSource}
        return Any
    elseif T <: Union{TestFunction, TrialFunction, IndexedTestFunction,
        IndexedTrialFunction, IdentityOperator}
        return L
    end
    return E
end
# The type of `T`'s weights once multiplied by an `E` volume: a test function's weight is
# the `Int` `1`, so `π * v` has `Float64` weights on a Float32 mesh beside the `Float32` of
# `3 * D₋ₓ(v)`.
_type_weight(T, E) = Base.promote_op(*, _type_value(T, E, Int), E)

# A product's test side holding three or more restricted summands is split as a bilinear
# product's is (`_split_restricted_products`, bilinear.jl, gpena/Bramble.jl#727): grouped
# by type, at most two a piece, every piece one `LinearProduct` with the same source. Its
# source is left as it is, since a restricted source keeps one stencil type once
# `_typed_restrictions` (above) has typed it. The test side's summands, when their weights
# differ in type, are widened as a bilinear operand's are (`_uniform_scales`), to the type
# their weights join to and never narrowed below an `E` weight. Without a restriction that
# happens only when the type is bits (`_widened_product`, 14112 B before for `0.1 * v +
# 3 * D₋ₓ(v)` on a Float32 mesh), and a form whose test sides' weights agree folds to
# itself at compile time (`_needs_rewrite`, which reads the test side through the method
# below, defined before bilinear.jl's generator that calls it).
_type_mixed_scales(T::Type{<:LinearProduct}, E) = T isa DataType &&
                                                  _type_scales_differ(T.parameters[4], E)

function _split_restricted_products(op::LinearProduct{D, W}, ::Type{E}) where {D, W, E}
    _needs_rewrite(op, E) || return op
    _holds_restriction(op.right_op) || return _widened_product(op, E)
    r = _uniform_scales(op.right_op, E)
    rs = _operand_groups(_expanded_summands(r), Val(2))
    length(rs) > 16 && return op
    if length(rs) == 1 || _mixed_scales(r)
        _mixed_scales(r) || (r = _hoisted(r, rs))
        return r === op.right_op ? op :
               LinearProduct{D, W, typeof(op.left_op), typeof(r)}(op.inner, op.left_op, r)
    end
    return _linear_product_sum(op, rs)
end

function _widened_product(op::LinearProduct{D, W}, ::Type{E}) where {D, W, E}
    r = _bits_uniform_scales(op.right_op, E)
    return LinearProduct{D, W, typeof(op.left_op), typeof(r)}(op.inner, op.left_op, r)
end

# One `LinearProduct` per group, summed left to right; recursive on the tuple's type, so the
# sum infers concretely.
@inline _linear_product_sum(op::LinearProduct, rs::Tuple{Any}) = _linear_piece(op, rs[1])
@inline _linear_product_sum(op::LinearProduct, rs::Tuple) = OperatorAdd(
    _linear_product_sum(op, Base.front(rs)), _linear_piece(op, rs[end]))
@inline _linear_piece(op::LinearProduct{D, W}, r) where {D, W} = LinearProduct{
    D, W, typeof(op.left_op), typeof(r)}(op.inner, op.left_op, r)

# --- A VectorElement source must live on the leaf it is read on --- #
#
# `local_stencil(::SourceVector)` reads `op.vec[lin_idx]` at the walked leaf's linear index,
# and `local_stencil(::GridFunctionScale)` its `grid_function[lin_idx]`, so a source or a
# coefficient from another mesh is read at the wrong points: a finer one silently, a coarser
# one past its end. Only an array that kept its `VectorElement` (`_as_source`,
# operators/inner.jl, or `Rₕ` as a coefficient) knows its mesh; a plain vector, from a
# lowered closure or built by hand, carries none and is skipped. Routed as
# `_check_centered_meshes` routes, on the leaf the sweep walks; a linear form's test
# components outside the space never reach here, as `_lower_sources_for_space` runs first
# in `form` and throws for them. The walk reaches what `_host_sources` does. Once, at
# `form`: points moved later (`change_points!`) are not caught, as in `_same_mesh`.
_check_source_meshes(ast, Wₕ) = _check_walked_meshes(_check_source_mesh, ast, Wₕ, Wₕ)

_check_source_mesh(op, Ωₕ) = nothing

function _check_source_mesh(op::SourceVector{D, <:VectorElement}, Ωₕ) where {D}
    Ωs = mesh(space(op.vec))
    _same_mesh(Ωs, Ωₕ) || _throw_source_mesh(Ωs, Ωₕ)
    return nothing
end

function _check_source_mesh(op::GridFunctionScale{D, <:VectorElement}, Ωₕ) where {D}
    Ωs = mesh(space(op.grid_function))
    _same_mesh(Ωs, Ωₕ) || _throw_source_mesh(Ωs, Ωₕ)
    return _check_source_mesh(op.inner_op, Ωₕ)
end

function _check_source_mesh(op::Union{OperatorAdd, LinearProduct, BilinearProduct}, Ωₕ)
    _check_source_mesh(op.left_op, Ωₕ)
    _check_source_mesh(op.right_op, Ωₕ)
    return nothing
end

# The operand of an `InterpolationNode` lives on another mesh, so it is not entered.
_check_source_mesh(op::InterpolationNode, Ωₕ) = nothing
_check_source_mesh(op::UnaryWrapper, Ωₕ) = _check_source_mesh(op.inner_op, Ωₕ)

@noinline function _throw_source_mesh(Ωs, Ωₕ)
    ns, nh = npoints(Ωs, Tuple), npoints(Ωₕ, Tuple)
    # Equal counts would print two equal tuples, which show no reason for the refusal.
    which = ns == nh ? "the source is on another mesh of the same shape, $(ns) points" :
            "the source's mesh has $(ns) points and the one it is read on $(nh)"
    throw(
        ArgumentError(
        "a grid-function source or coefficient of the form lives on another mesh than the " *
        "space it is read on: $(which). It is read at that space's own points, which " *
        "name other places on another mesh. Interpolate it, as in `innerₕ(πₕ(uₕ), v)` " *
        "or `πₕ(Wₕ, uₕ) * u`, or build it with `Rₕ` on the space the form walks.",
    ),
    )
end

# --- A centered difference needs three points along its direction --- #
#
# `Dc` and `D̽` read a neighbour on each side, so on an axis with fewer than three points
# their symbolic stencils truncate at every point and the form assembles to zero, where the
# runtime operators throw (gpena/Bramble.jl#629). Checked once, at `form`, per term, on the
# leaf the sweep walks (`_walked_leaf`): the test leaf, or the trial leaf when the term
# interpolates on the test side. Routed as `each_routed_leaf` and `blocks` route: a term
# naming a component goes to that leaf alone, one naming none to every leaf the assembly
# walks, so a composite whose collapsed leaf the term never touches still forms. A
# component outside the space matches no leaf here and is refused where it was before. The
# operand of an `InterpolationNode` lives on another mesh, so it is not entered. A linear
# form walks its test space; `bilinear.jl`'s `form` passes both of its spaces and runs both
# checks in one walk (`_check_form_meshes`), since no source is lowered between them there.
# The walk is `_check_walked_meshes`, shared with `_check_source_meshes` above.
_check_centered_meshes(ast, Vₕ) = _check_centered_meshes(ast, Vₕ, Vₕ)
_check_centered_meshes(ast, Wₕ, Vₕ) = _check_walked_meshes(_check_centered_mesh, ast, Wₕ, Vₕ)

_check_form_meshes(ast, Wₕ, Vₕ) = _check_walked_meshes(_check_term_meshes, ast, Wₕ, Vₕ)

function _check_term_meshes(term, Ωₕ)
    _check_centered_mesh(term, Ωₕ)
    _check_source_mesh(term, Ωₕ)
    return nothing
end

# Calls `check(term, Ωₕ)` for each term of `ast` on the mesh of every leaf the sweep walks
# it over, routed as described above. A sum's terms are visited from one generated body,
# not one method per partial sum, and a term whose type holds no node either check reads
# (`_reads_meshes`) is routed but not checked, so neither the leaf loop nor the checks
# compile for it (gpena/Bramble.jl#763). The routing queries stay: `assemble` compiles the
# same ones per term (`block_of`), so skipping them here only moves them there.
_check_walked_meshes(check::C, ast, Wₕ, Vₕ) where {C} = _check_summand_meshes(
    check, ast, Wₕ, Vₕ)

# Whether a node of type `T` may hold one the checks above read: a centered or
# cross-weighted difference (`_check_centered_mesh`), or a field holding a `VectorElement`
# (`_check_source_mesh`). Read off the type alone and folded at compile time, as
# `_may_restrict` is (block_extract.jl); a field whose type is not concrete answers `true`.
Base.@nospecializeinfer Base.@assume_effects :foldable function _reads_meshes(
        @nospecialize(T))
    T isa DataType || return true
    T <: Union{CenteredDifference, CrossWeightedDifference} && return true
    for i in 1:fieldcount(T)
        F = fieldtype(T, i)
        isconcretetype(F) || return true
        F <: VectorElement && return true
        F <: Union{LazyOp, Tuple} && _reads_meshes(F) && return true
    end
    return false
end

# The field paths from a sum of type `T`, reached as `path`, to each of its summands.
function _summand_paths(@nospecialize(T), path)
    T <: OperatorAdd || return Any[path]
    return vcat(_summand_paths(fieldtype(T, :left_op), :($path.left_op)),
        _summand_paths(fieldtype(T, :right_op), :($path.right_op)))
end

@generated function _check_summand_meshes(check::C, op::OperatorAdd, Wₕ, Vₕ) where {C}
    calls = [:(_check_summand_meshes(check, $p, Wₕ, Vₕ)) for p in _summand_paths(op, :op)]
    return :($(calls...); nothing)
end

function _check_summand_meshes(check::C, term, Wₕ, Vₕ) where {C}
    # A term naming neither side is assembled on the diagonal blocks alone, the first
    # `min` of the two leaf counts (`_diagonal_blocks`), so only those leaves are checked.
    # A linear form passes its test space twice: every leaf, as `each_routed_leaf` routes.
    unnamed = trial_component_or_nothing(term) === nothing &&
              test_component_or_nothing(term) === nothing
    nmax = unnamed ? min(leaf_count(Wₕ), leaf_count(Vₕ)) : typemax(Int)
    _has_test_interp(term) &&
        return _check_on_side(check, term, Wₕ, trial_component_or_nothing, nmax)
    return _check_on_side(check, term, Vₕ, test_component_or_nothing, nmax)
end

@inline function _check_on_side(check::C, term, space, side_component::F,
        nmax) where {C, F}
    _reads_meshes(typeof(term)) || return nothing
    return check(term, mesh(space))
end

@inline function _check_on_side(check::C, term, space::CompositeGridSpace,
        side_component::F, nmax) where {C, F}
    target = side_component(term)
    _reads_meshes(typeof(term)) || return nothing
    _foldl_leaves(nothing, leaf_spaces_offsets(space)) do _, c, leaf
        (target === nothing ? c <= nmax : target == c) && check(term, mesh(first(leaf)))
        return nothing
    end
    return nothing
end

_check_centered_mesh(op, Ωₕ) = nothing
_check_centered_mesh(op::InterpolationNode, Ωₕ) = nothing
_check_centered_mesh(op::UnaryWrapper, Ωₕ) = _check_centered_mesh(op.inner_op, Ωₕ)

function _check_centered_mesh(op::Union{OperatorAdd, LinearProduct, BilinearProduct}, Ωₕ)
    _check_centered_mesh(op.left_op, Ωₕ)
    _check_centered_mesh(op.right_op, Ωₕ)
    return nothing
end

function _check_centered_mesh(
        op::Union{CenteredDifference{D, Dim}, CrossWeightedDifference{D, Dim}},
        Ωₕ
) where {D, Dim}
    n = npoints(Ωₕ, Tuple)[Dim]
    n >= 3 || _throw_centered_too_few_points(Dim, n)
    return _check_centered_mesh(op.inner_op, Ωₕ)
end

# --- Assembly implementations ----------------------------------------------------- #

# Nothing to do unless labels are provided. `dirichlet_conditions` defaults to `nothing`
# to prevent allocations when boundary constraints are absent.
function apply_dirichlet_conditions!(
        b::AbstractVector,
        form::LinearForm,
        dirichlet_conditions,
        dirichlet_labels,
        dirichlet_components = nothing
)
    dirichlet_labels === nothing && return b

    dirichlet_conditions === nothing && _throw_labels_without_conditions(dirichlet_labels)

    if dirichlet_labels isa Symbol
        dirichlet_bc!(
            b,
            test_space(form),
            dirichlet_conditions,
            dirichlet_labels;
            components = dirichlet_components
        )
    elseif dirichlet_labels isa Tuple && !isempty(dirichlet_labels)
        dirichlet_bc!(
            b,
            test_space(form),
            dirichlet_conditions,
            dirichlet_labels...;
            components = dirichlet_components
        )
    end
    return b
end

@noinline function _throw_labels_without_conditions(labels)
    throw(
        ArgumentError(
        "dirichlet names label(s) $labels but carries no values for them. A linear " *
        "form needs both: pass a `label => f` Pair (or a Tuple of them, or constraints " *
        "from dirichlet_constraints) as `dirichlet`, not a bare label.",
    ),
    )
end

"""
    assemble(form::LinearForm; dirichlet = nothing, dirichlet_components = nothing) -> AbstractVector

Assemble the system vector of the `LinearForm`, applying the boundary values `dirichlet`
carries on the regions it names. `dirichlet` accepts a `label => f` `Pair`, a `Tuple` of
such `Pair`s, or constraints from [`dirichlet_constraints`](@ref) -- a bare label or
`Tuple` of labels has no values to write and raises an error (see
[`_normalize_dirichlet`](@ref) for every accepted form). `dirichlet_components` restricts
which leaf components of a composite test space they bind to (see [`dirichlet_bc!`](@ref)).

Runs serially or across threads following `test_space(form)`'s backend
[`execution_policy`](@ref): [`Serial`](@ref) (the default) or [`Parallel`](@ref).
[`assemble_parallel!`](@ref) always threads regardless of the backend policy.
"""
function assemble(
        form::LinearForm; dirichlet = nothing, dirichlet_components = nothing
)
    space = test_space(form)
    # `parent(element(space, T))` reuses the space's backend container type.
    b = parent(element(space, _assembled_eltype(form.ast, space)))
    return _assemble_linear!(b, form, form.ast, dirichlet, dirichlet_components)
end

# The element type of the assembled vector is the one the form's own weights have, promoted
# against the space's (not the space's outright), supporting autodiff types like `ForwardDiff.Dual`.
function _assembled_eltype(ast, space)
    return _folded_eltype(ast, space, eltype(space))
end

# Composite: terms naming components are routed and folded on their respective leaf spaces,
# and a term naming none on every leaf.
function _assembled_eltype(ast, space::CompositeGridSpace)
    leaves = leaf_spaces_offsets(space)
    return _routed_eltype(ast, leaves, leaves, eltype(space))
end

# A space's leaves as a tuple, a scalar space being its own one leaf.
@inline _leaves_of(space::CompositeGridSpace) = leaf_spaces_offsets(space)
@inline _leaves_of(space) = ((space, 0),)

# The leading entries of `a` that have a partner in `b`: the diagonal blocks. Dispatch on
# the tuple lengths keeps it static.
@inline _paired(::Tuple{}, ::Tuple{}) = ()
@inline _paired(::Tuple{}, ::Tuple) = ()
@inline _paired(::Tuple, ::Tuple{}) = ()
@inline _paired(a::Tuple, b::Tuple) = (first(a), _paired(Base.tail(a), Base.tail(b))...)

# The type every leaf of `term` can put in an entry, promoted against `T`.
#
# Read from the node types wherever they carry it (a `Number` or `Ref{T}` scale, an array
# coefficient or source), and from a value only where nothing else reveals it: a thunk is
# called once, and so is every strength of a `Vector{Function}` Dirac source. Never from a
# stencil probed at an arbitrary point (a restricted source is probed inside its region,
# below): a restricted term, or a Dirac point far from that point, answers there with
# nothing, or with a narrower type than it writes elsewhere, and the
# vector it is assembled into cannot hold what the other points write. Static types keep
# `@inferred assemble` stable; `assemble!` sizes nothing, so a refill never comes here.
#
# `sp` is the leaf the term is routed to (each leaf in turn for a term naming no component,
# `_every_leaf_eltype`), and its own type is promoted in: a composite's
# `eltype` is its first leaf's alone, so a Float64 or `Dual` second leaf would otherwise be
# rounded or refused. `sp` is typed `::ScalarGridSpace` for the reason `_pattern_size_hint`
# gives (`bilinear_pattern.jl`): an untyped `sp` lets JET reach `host_weights(::SeparableWeights)`.
@inline _folded_eltype(term, sp::ScalarGridSpace, T) = _leaf_eltype(
    promote_type(T, eltype(sp)), term, sp)

# A leaf that carries no coefficient: trial and test functions, identity, zero.
@inline _leaf_eltype(T, op, sp) = T

# One operand, and nothing of its own that is not already in `T` (the space's spacings).
const _ELTYPE_WRAPPERS = Union{
    ShiftNode, InterpolationNode, BackwardDifference, ForwardDifference,
    CenteredDifference, StarDifference, CrossWeightedDifference, SecondDifference,
    BackwardAverage, ForwardAverage, CenteredAverage, JumpNode}

@inline _leaf_eltype(T, op::_ELTYPE_WRAPPERS, sp) = _leaf_eltype(T, op.inner_op, sp)

# A restricted source is typed at the point `form` probed it (`_typed_restrictions`), the
# first point of its region the walk reads, not the middle point a bare `SourceFunction` is
# sampled at, where a source defined only on its region may throw or answer another type.
# A probed node holds a zero stencil of that value, so its coefficient types are read and
# nothing is called. An unprobed one (a term naming no component on a composite, which
# `_lower_sources_over_leaves` leaves alone) is left as it was built and read on the leaf
# `sp`. Its operand is evaluated, as the walk does, at the points of its region in turn
# until one answers a non-empty stencil (an enclosed restriction or a shift off the grid
# answers `()` without calling the source). Whether one does is known only at run time, so
# the answer starts from the type the operand's stencil infers with its restrictions taken
# off (`_static_coeff_eltype`). An empty region and a found point then give the same type,
# and `@inferred assemble` holds.
# A device leaf is not probed by `form` either, so its restriction still types its operand.
@inline _leaf_eltype(T, op::RegionRestriction{<:Any, <:Any, <:Any, <:Tuple}, sp) = _stencil_eltype(
    T, op.zero_stencil)
@inline _leaf_eltype(T, op::RegionRestriction{<:Any, <:Any, <:Any, Nothing}, sp) = _unprobed_eltype(
    locality(backend(sp)), T, op, sp)

_unprobed_eltype(::DeviceLocality, T, op, sp) = _leaf_eltype(T, op.inner_op, sp)
function _unprobed_eltype(::HostLocality, T, op, sp)
    _is_source_only(op.inner_op) || return _leaf_eltype(T, op.inner_op, sp)
    Ωₕ = mesh(sp)
    cart = CartesianIndices(indices(Ωₕ))
    S = _static_coeff_eltype(T, op.inner_op, sp, first(cart))
    bound, mk = _bind_walk(op.inner_op, Ωₕ)
    inside = _probe_reach(_marker_words(Ωₕ), _probe_ids(op.region, Ωₕ), trues(length(cart)))
    for lin in findall(inside)
        s = local_stencil(bound, sp, cart[lin], mk, lin)
        s === () || return _stencil_eltype(S, s)
    end
    return S
end

# `T` promoted with the coefficient types `op`'s stencil infers once its restrictions are
# taken off (a restriction answers `()` outside its region, so its stencil infers a
# `Union`). The type comes from broadcasting over no point: the source is not called.
function _static_coeff_eltype(T, op, sp, I)
    bound, mk = _bind_walk(_unrestricted(op), mesh(sp))
    return _stencil_type_eltype(T, eltype((lin -> local_stencil(bound, sp, I, mk, lin)).(Int[])))
end

_unrestricted(op) = op
_unrestricted(op::UnaryWrapper) = _rewrap_inner(op, _unrestricted(op.inner_op))
_unrestricted(op::RegionRestriction) = _unrestricted(op.inner_op)
function _unrestricted(op::OperatorAdd{D}) where {D}
    left, right = _unrestricted(op.left_op), _unrestricted(op.right_op)
    return OperatorAdd{D, typeof(left), typeof(right)}(left, right)
end

# `T` promoted with the coefficient types of a concrete stencil type. Only concrete number
# types count; anything else is left to the value read at the probe point. The generator
# only lists the types: `promote_type` runs in the caller's world, where the promotion
# rules of packages loaded after Bramble (`ForwardDiff.Dual`) are seen.
@generated function _stencil_type_eltype(::Type{T}, ::Type{S}) where {T, S}
    Cs = Any[]
    for E in (S <: Tuple && isconcretetype(S) ? fieldtypes(S) : ())
        C = E <: Tuple && fieldcount(E) > 0 ? fieldtype(E, fieldcount(E)) : Any
        isconcretetype(C) && C <: Number && push!(Cs, C)
    end
    return :(promote_type(T, $(Cs...)))
end

@inline _stencil_eltype(T, ::Tuple{}) = T
@inline _stencil_eltype(T, s::Tuple) = _stencil_eltype(
    promote_type(T, typeof(first(s)[end])), Base.tail(s))

@inline _leaf_eltype(T, op::Union{OperatorAdd, BilinearProduct, LinearProduct}, sp) = _leaf_eltype(
    _leaf_eltype(T, op.left_op, sp), op.right_op, sp)

@inline _leaf_eltype(T, op::OperatorScale, sp) = _leaf_eltype(
    _value_eltype(T, op.scalar), op.inner_op, sp)
@inline _leaf_eltype(T, op::GridFunctionScale, sp) = _leaf_eltype(
    _value_eltype(T, op.grid_function), op.inner_op, sp)

@inline _leaf_eltype(T, op::SourceVector, sp) = _value_eltype(T, op.vec)
@inline _leaf_eltype(T, op::SourceConstant, sp) = _value_eltype(T, op.value)

# A function of position reveals its type only through a value: sampled once, at an
# interior grid point, on the host copy of the mesh (`host_weights`).
# Interior, not the first point: a function defined on a smaller mesh (an interpolant
# `πₕ(uₕ; outside = fill)` under a difference) answers a boundary point with its fill,
# whose type need not be its values'.
@inline function _leaf_eltype(T, op::SourceFunction, sp::ScalarGridSpace)
    Ωₕ = mesh(host_weights(sp))
    grid_inds = indices(Ωₕ)
    I = grid_inds[length(grid_inds) ÷ 2 + 1]
    return promote_type(T, typeof(op.func(point(Ωₕ, I))))
end

# An interpolant `πₕ(uₕ)` is left unsampled, so probing it at a
# point would answer with the fill's type wherever that point lies outside `uₕ`'s mesh:
# its type is read off `uₕ`, the walked mesh's coordinates and the fill instead.
@inline _leaf_eltype(T, op::SourceFunction{D, <:GridInterpolant}, sp::ScalarGridSpace) where {D} = promote_type(
    T, _interpolant_type(op.func, eltype(sp)))

# Each strength's weight type (`_dirac_weight_type`, `stencil_eval.jl`), every one of them.
@inline _leaf_eltype(T, op::DiracSource, sp) = promote_type(
    T, eltype(mesh(sp)), _strengths_eltype(op.strengths))

@inline _strengths_eltype(s) = typeof(_point_strength_val(s))
@inline _strengths_eltype(s::AbstractVector{<:Number}) = _value_eltype(Union{}, s)
@inline _strengths_eltype(s::AbstractVector{<:Base.RefValue{<:Number}}) = isconcretetype(eltype(s)) ?
                                                                          eltype(eltype(s)) :
                                                                          _each_strength_eltype(s)
@inline _strengths_eltype(s::AbstractVector) = _each_strength_eltype(s)

function _each_strength_eltype(s)
    R = Union{}
    for sₖ in s
        R = promote_type(R, typeof(_point_strength_val(sₖ)))
    end
    return R
end

# The type a coefficient's values have: its own for a number, its element type for a `Ref` or
# an array, a thunk's result for a thunk. Only an abstract element type is read value by value.
@inline _value_eltype(T, x::Number) = promote_type(T, typeof(x))
@inline _value_eltype(T, x::Base.RefValue) = isconcretetype(eltype(x)) ?
                                             promote_type(T, eltype(x)) : _value_eltype(T, x[])
@inline _value_eltype(T, x::AbstractArray) = isconcretetype(eltype(x)) ?
                                             promote_type(T, eltype(x)) :
                                             mapreduce(typeof, promote_type, x; init = T)
@inline _value_eltype(T, f::Function) = _value_eltype(T, f())
@inline _value_eltype(T, x) = promote_type(T, typeof(x))

# `diagonal` holds the leaves an unnamed term is assembled on: every leaf for a linear form,
# those with a trial partner for a bilinear one.
function _routed_eltype(op::OperatorAdd, leaves, diagonal, T)
    return promote_type(
        _routed_eltype(op.left_op, leaves, diagonal, T),
        _routed_eltype(op.right_op, leaves, diagonal, T)
    )
end

function _routed_eltype(term, leaves, diagonal, T)
    target = test_component_or_nothing(term)
    _check_component(target, length(leaves), "test")
    target === nothing && return _every_leaf_eltype(term, diagonal, T)
    return _at_leaf(leaf -> _folded_eltype(term, first(leaf), T), leaves, target)
end

# A term naming no component is assembled on every diagonal block, so each of their leaves
# decides its type, not the first alone: a wider later leaf (Float64 behind Float32, a
# `Dual`-coordinate mesh) would otherwise be rounded or refused. Tail
# recursion over the leaf tuple keeps the fold static.
@inline _every_leaf_eltype(term, leaves::Tuple{Any}, T) = _folded_eltype(
    term, first(first(leaves)), T)
@inline _every_leaf_eltype(term, leaves::Tuple, T) = _every_leaf_eltype(
    term, Base.tail(leaves), _folded_eltype(term, first(first(leaves)), T))

function _trial_routed_eltype(op::OperatorAdd, leaves, diagonal, T)
    return promote_type(
        _trial_routed_eltype(op.left_op, leaves, diagonal, T),
        _trial_routed_eltype(op.right_op, leaves, diagonal, T)
    )
end

function _trial_routed_eltype(term, leaves, diagonal, T)
    target = trial_component_or_nothing(term)
    _check_component(target, length(leaves), "trial")
    target === nothing && return _every_leaf_type(diagonal, T)
    return _at_leaf(leaf -> promote_type(T, eltype(first(leaf))), leaves, target)
end

# An unnamed trial function stands on every diagonal block, so each of their leaves' types
# counts.
@inline _every_leaf_type(leaves::Tuple, T) = promote_type(
    T, map(eltype ∘ first, leaves)...)

# --- Helper cores for function barrier optimization ------------------------------- #

# The scalar case is `_scatter_term!` (below) at offset zero: one leaf, the whole space.
@inline function _assemble_linear_core!(
        b::AbstractVector, space, ast::AST_TYPE, α = true) where {AST_TYPE}
    _each_walk_unit(t -> _scatter_term!(b, space, t, 0, α), ast, eltype(space))
    return b
end

# --- Walk units (gpena/Bramble.jl#727) --------------------------------------------- #
#
# A scalar space walks its whole form at once, every term's stencil concatenated at each
# point, so the shapes of all restricted summands multiply: past two the `Union` is not
# split and every point boxes, however `form` split the products
# (`_split_restricted_products`). Such a form is walked one summand at a time instead,
# each summand a walk of its own, as a composite space's terms already are. A form with at
# most two restricted summands, the count read off its type, is walked once, as before.
#
# Terms whose weights differ in type (`innerₕ(f, 2 * v) + innerₕ(1.0, v)` on a Float32
# mesh, Float32 and Float64) are walked one value type at a time too, each group of terms
# of one type a walk of its own (`_value_groups`): concatenated, the stencil's entries
# differ in type and the loop over them boxed at every point (27216 B on a 7x9 grid). Both
# are read off the types (`_type_value`), so a form of one weight type walks as before.
@inline function _each_walk_unit(f::F, ast, ::Type{E}) where {F, E}
    if _unit_bits(ast) > 2
        map(f, _walk_summands(ast))
    elseif _values_differ(_walk_summands(ast), E)
        map(f, _value_groups(_walk_summands(ast), E))
    else
        f(ast)
    end
    return nothing
end
@inline function _fold_walk_units(f::F, ast, acc, ::Type{E}) where {F, E}
    _unit_bits(ast) > 2 && return _fold_summands(f, _walk_summands(ast), acc)
    _values_differ(_walk_summands(ast), E) &&
        return _fold_summands(f, _value_groups(_walk_summands(ast), E), acc)
    return f(ast, acc)
end

# Whether the terms `ts` have weights of more than one type, each known.
@generated function _values_differ(ts::Tuple, ::Type{E}) where {E}
    values = Any[_type_weight(T, E) for T in fieldtypes(ts)]
    return all(isconcretetype, values) && !allequal(values)
end
# The terms `ts` summed in groups of one weight type, in the order each type first comes.
@generated function _value_groups(ts::Tuple, ::Type{E}) where {E}
    values = Any[_type_weight(T, E) for T in fieldtypes(ts)]
    groups = Expr(:tuple)
    for v in unique(values)
        idx = findall(isequal(v), values)
        ex = :(ts[$(idx[1])])
        for i in idx[2:end]
            ex = :(OperatorAdd($ex, ts[$i]))
        end
        push!(groups.args, ex)
    end
    return groups
end

# The summands walked apart: a scale over a sum (`2.0 * (l₁ + l₂ + l₃)`, `-(l₁ + l₂ + l₃)`)
# is looked through, each summand walked under its own copy of the scale. The tree `form`
# stored is not changed, so the scale is rebuilt per summand only here, in the walk.
@inline _walk_summands(op::OperatorAdd) = (
    _walk_summands(op.left_op)..., _walk_summands(op.right_op)...)
@inline function _walk_summands(op::_Scale)
    ts = _walk_summands(op.inner_op)
    return ts isa Tuple{Any} ? (op,) : map(t -> _rewrap_inner(op, t), ts)
end
@inline _walk_summands(op) = (op,)
@inline _fold_summands(f::F, ::Tuple{}, acc) where {F} = acc
@inline _fold_summands(f::F, ts::Tuple, acc) where {F} = _fold_summands(
    f, Base.tail(ts), f(first(ts), acc))

# The restricted summands of a linear form, as `_shape_bits` (bilinear.jl) counts an
# operand's, over both sides of each product and every restriction wrapping a product.
@inline _unit_bits(op::OperatorAdd) = _unit_bits(op.left_op) + _unit_bits(op.right_op)
@inline _unit_bits(op::LinearProduct) = _shape_bits(op.left_op) + _shape_bits(op.right_op)
@inline _unit_bits(op::RegionRestriction) = (op.zero_stencil === nothing) +
                                            _unit_bits(op.inner_op)
@inline _unit_bits(op::UnaryWrapper) = _unit_bits(op.inner_op)
@inline _unit_bits(op) = _shape_bits(op)

# --- Parallel assembly partitioning ------------------------------------------------ #

"""
    _colour_strides(offsets) -> NTuple{D, Int}

Per-dimension stride separating grid points that a parallel assembly may write concurrently,
for an operator reaching `offsets`.

Two points of one colour differ by a multiple of the stride in some dimension (at least
`span + 1` there, where each writes a footprint `span` wide about itself). Beyond one width apart,
the footprints do not overlap: no two points in a colour ever target the same row, enabling
lock-free parallel assembly.

An operator reaching only its own point (such as `innerₕ(fₕ, v)` or any form whose test
argument carries no difference) strides by 1 in every dimension, resulting in a single colour.
"""
@inline _colour_strides(offsets::Vector{NTuple{D, Int}}) where {D} = _bounds_strides(
    _offset_bounds(offsets))

# `_colour_strides(stencil_offsets(op))` without the offset `Vector`s, so a threaded refill or
# product costs no heap allocation per unit and call. `_row_bounds(op)` is the corner pair
# `(lo, hi)` of the box around `stencil_offsets(op)`, read off the same tree by the same rules:
# a tap or a shift moves the child's box along its own dimension by its least and greatest
# step, a sum takes the box around both summands, and every other node passes its child's
# through. A node with no method here falls back to `stencil_offsets` itself, so the bounds
# always agree with the offsets, at worst at the old cost.
@inline _row_bounds(op) = _offset_bounds(stencil_offsets(op))

@inline function _offset_bounds(offsets::Vector{NTuple{D, Int}}) where {D}
    isempty(offsets) && return (ntuple(_ -> 0, D), ntuple(_ -> -1, D))
    lo = hi = first(offsets)
    for o in offsets
        lo = min.(lo, o)
        hi = max.(hi, o)
    end
    return (lo, hi)
end

@inline _row_bounds(::Union{
    TrialFunction{D}, TestFunction{D}, IndexedTrialFunction{D}, IndexedTestFunction{D},
    SourceFunction{D}, SourceVector{D}, SourceConstant{D}, DiracSource{D},
    IdentityOperator{D}, ZeroOperator{D}}) where {D} = (ntuple(_ -> 0, D), ntuple(_ -> 0, D))

@inline function _moved_bounds(
        (lo, hi)::NTuple{2, NTuple{D, Int}}, ::Val{Dim}, steps::Tuple
) where {D, Dim}
    smin, smax = minimum(steps), maximum(steps)
    return (ntuple(i -> i == Dim ? lo[i] + smin : lo[i], Val(D)),
        ntuple(i -> i == Dim ? hi[i] + smax : hi[i], Val(D)))
end

@inline _row_bounds(op::TappedNode{D, Dim}) where {D, Dim} = _moved_bounds(
    _row_bounds(op.inner_op), Val(Dim), map(_shift_delta, _stencil_taps(op)))
@inline _row_bounds(op::ShiftNode{D, Dim}) where {D, Dim} = _moved_bounds(
    _row_bounds(op.inner_op), Val(Dim), (op.shift_amount,))
@inline _row_bounds(op::Union{OperatorScale, GridFunctionScale, RegionRestriction,
    InterpolationNode}) = _row_bounds(op.inner_op)
@inline _row_bounds(op::Union{LinearProduct, BilinearProduct}) = _row_bounds(op.right_op)
@inline _row_bounds(op::OperatorAdd) = _union_bounds(
    _row_bounds(op.left_op), _row_bounds(op.right_op))

@inline _union_bounds((lo1, hi1), (lo2, hi2)) = (min.(lo1, lo2), max.(hi1, hi2))

# The strides of `_colour_strides` over the rows one term reaches, or two terms together
# (a transposed pair, whose rows are both terms' rows).
@inline _term_colour_strides(op) = _bounds_strides(_row_bounds(op))
@inline _term_colour_strides(p1, p2) = _bounds_strides(
    _union_bounds(_row_bounds(p1), _row_bounds(p2)))
@inline _bounds_strides((lo, hi)) = max.(hi .- lo .+ 1, 1)

# One colour of `grid_inds`, represented as a strided subgrid without allocating index vectors.
@inline function _colour_subgrid(
        grid_inds::CartesianIndices{D}, c::CartesianIndex{D}, strides::NTuple{D, Int}
) where {D}
    return CartesianIndices(ntuple(d -> c[d]:strides[d]:last(axes(grid_inds, d)), D))
end

# The threaded pass over one colour, writing directly into `b`.
"""
    _scatter_linear_point!(b, sp, term, I, lin_indices, mesh_markers, offset, α = true) -> Nothing

Add one grid point's stencil contributions to `b`.

Shared by the banded and the point-coloured sweep, so the two cannot drift apart.
"""
@inline function _scatter_linear_point!(
        b::AbstractVector,
        sp,
        term::TERM,
        I::CartesianIndex,
        lin_indices,
        mesh_markers,
        offset::Int,
        α = true
) where {TERM}
    stencil = local_stencil(term, sp, I, mesh_markers, lin_indices[I])

    for (off_v, weight) in stencil
        Iv = I + CartesianIndex(off_v)

        if checkbounds(Bool, lin_indices, Iv)
            @inbounds b[lin_indices[Iv] + offset] += α * weight
        end
    end
    return nothing
end

# One point of a right-hand-side sweep (`_sweep_coloured!`, bilinear_execution.jl):
# `_scatter_linear_point!` into `b` at `I`.
struct _LinearPoint{B, S, M, L, K, A}
    b::B
    sp::S
    term::M
    lin_indices::L
    mesh_markers::K
    offset::Int
    α::A
end
@inline (p::_LinearPoint)(I) = _scatter_linear_point!(
    p.b, p.sp, p.term, I, p.lin_indices, p.mesh_markers, p.offset, p.α)

@inline _sweep_bands!(policy, p::_LinearPoint, ax, bidx, nbands::Int, rest) = _sweep_linear_band_colour!(
    policy, p.b, p.sp, p.term, ax, bidx, nbands, rest, p.lin_indices, p.mesh_markers,
    p.offset, p.α)
@inline _sweep_points!(policy, p::_LinearPoint, idxs) = _sweep_colour!(
    policy, p.b, p.sp, p.term, idxs, p.lin_indices, p.mesh_markers, p.offset, p.α)

"""
    _sweep_linear_band_colour!(policy, b, sp, term, ax, bidx, nbands, rest, lin_indices, mesh_markers, offset, α = true) -> Nothing

Scatter one band colour of `term` into `b` across threads.

Two points collide only when they write the same entry of `b`, which needs their difference
to lie inside the stencil's reach in every axis at once. Being at least `strides[D]` apart
along the banded axis rules that out on its own, so alternate slabs never race. A term that
reaches only its own point cannot collide at all, and then `bidx` is every band at once.
"""
@noinline function _sweep_linear_band_colour!(
        ::CpuThreaded,
        b::AbstractVector,
        sp,
        term::TERM,
        ax,
        bidx,
        nbands::Int,
        rest,
        lin_indices,
        mesh_markers,
        offset::Int,
        α = true
) where {TERM}
    point = _LinearPoint(b, sp, term, lin_indices, mesh_markers, offset, α)
    return _threaded_bands!(point, ax, bidx, nbands, rest)
end

@noinline function _sweep_linear_band_colour!(
        ::CpuPolyester,
        b::AbstractVector,
        sp,
        term::TERM,
        ax,
        bidx,
        nbands::Int,
        rest,
        lin_indices,
        mesh_markers,
        offset::Int,
        α = true
) where {TERM}
    return _late(
        _batch_linear_band_sweep!,
        b, sp, term, ax, bidx, nbands, rest, lin_indices, mesh_markers, offset, α
    )
end

"""
    _batch_linear_band_sweep!(b, sp, term, ax, bidx, nbands, rest, lin_indices, mesh_markers, offset, α) -> Nothing

[`CpuPolyester`](@ref)'s counterpart of the `Threads.@threads` body in
[`_sweep_linear_band_colour!`](@ref), filled by `BramblePolyesterExt`.
Its `src/` method only throws, naming Polyester; `BramblePolyesterExt` defines the one that
runs.
"""
@noinline function _batch_linear_band_sweep!(
        b, sp, term, ax, bidx, nbands, rest, lin_indices, mesh_markers, offset, α
)
    return _throw_cpubatch_without_polyester(:_batch_linear_band_sweep!)
end

# Dispatches on the *effective* execution policy (`_sweep_parallel!` computes it):
# `CpuThreaded` runs the shared `Threads.@threads` loop; `CpuPolyester` reaches its own hook
# instead, so it never silently threads with the wrong mechanism.
# `CpuSerial` never reaches this function -- `_effective_parallel_policy` only ever hands it
# `CpuThreaded` or `CpuPolyester`.
@noinline function _sweep_colour!(
        ::CpuThreaded, b::AbstractVector, sp, term::TERM, idxs, lin_indices, mesh_markers, offset::Int, α = true
) where {TERM}
    point = _LinearPoint(b, sp, term, lin_indices, mesh_markers, offset, α)
    return _threaded_colour!(point, idxs)
end

@noinline function _sweep_colour!(
        ::CpuPolyester, b::AbstractVector, sp, term::TERM, idxs, lin_indices, mesh_markers, offset::Int, α = true
) where {TERM}
    return _late(_batch_linear_colour_sweep!, b, sp, term, idxs, lin_indices, mesh_markers, offset, α)
end

"""
    _batch_linear_colour_sweep!(b, sp, term, idxs, lin_indices, mesh_markers, offset, α) -> Nothing

[`CpuPolyester`](@ref)'s counterpart of the `Threads.@threads` body in `_sweep_colour!`,
filled by `BramblePolyesterExt`. Its `src/` method only throws,
naming Polyester; `BramblePolyesterExt` defines the one that runs.
"""
@noinline function _batch_linear_colour_sweep!(b, sp, term, idxs, lin_indices, mesh_markers, offset, α)
    return _throw_cpubatch_without_polyester(:_batch_linear_colour_sweep!)
end

# Every colour in turn (`_sweep_coloured!`, bilinear_execution.jl). `policy` is the
# *effective* policy (`_effective_parallel_policy(sp)`, computed once here): `CpuSerial` is
# coerced to `CpuThreaded` since every call into this function is already on the
# forced-threaded path (`_assemble_linear_parallel_core!`, entered from a non-`CpuSerial`
# branch, or from `assemble_parallel!`'s own "regardless of policy" contract);
# `CpuPolyester` passes through unchanged so the colour/band sweeps reach their own hook
# instead of `Threads.@threads`.
function _sweep_parallel!(
        b::AbstractVector, sp, term::TERM, grid_inds, strides, offset::Int, α = true
) where {TERM}
    lin_indices = LinearIndices(indices(mesh(sp)))
    bound, mesh_markers = _bind_walk(term, sp)
    point = _LinearPoint(b, sp, bound, lin_indices, mesh_markers, offset, α)
    _sweep_coloured!(_effective_parallel_policy(sp), point, grid_inds, strides)
    return b
end

function _assemble_linear_parallel_core!(
        b::AbstractVector, space, ast::AST_TYPE, α = true
) where {AST_TYPE}
    _each_walk_unit(ast, eltype(space)) do t
        strides = _term_colour_strides(t)
        return _sweep_parallel!(b, space, t, indices(mesh(space)), strides, 0, α)
    end
    return b
end

function _assemble_linear_core!(
        b::AbstractVector, space::CompositeGridSpace{N}, ast::AST_TYPE, α = true
) where {N, AST_TYPE}
    return _route_terms!(b, ast, leaf_spaces_offsets(space), α)
end

# A term naming a component the space does not have used to contribute nothing, in silence:
# the loops below match `target` against each leaf in turn, so a target past the end simply
# never matched. On a two-block space `innerₕ(1.0, v(3))` assembled to zeros, and
# `innerₕ(1.0, v(1)) + innerₕ(2.0, v(9))` dropped the second term and kept the first.
#
# That is the failure mode the composite tests exist to prevent, so it is checked once per
# term rather than left to the reader of the answer.
# `side` ("test" or "trial") names the space the component is counted against.
@inline _check_component(::Nothing, ::Int, ::String) = nothing

@inline function _check_component(target::Int, nblocks::Int, side::String)
    1 <= target <= nblocks || _throw_component_out_of_range(target, nblocks, side)
    return nothing
end

@noinline function _throw_component_out_of_range(target::Int, nblocks::Int, side::String)
    throw(
        ArgumentError(
        "a term of this form names component $target, and its $side space has $nblocks. " *
        "Components are numbered 1 to $nblocks; a term written for a space with more of " *
        "them contributes nothing here, which is why this is an error rather than a zero.",
    ),
    )
end

# --- the routing rule ------------------------------------------------------------- #

#=
The rule that carries the semantics of composite linear forms: **a term naming no component
goes to every leaf; a term naming one goes to that leaf alone.** It used to be stated in a
comment and then implemented three times over, once per consumer -- scatter into `b`,
contract into an accumulator, sweep threaded. It is now written once
here, and the three consumers differ only in what they do per leaf.

Recursing the tree rather than flattening it into a vector of terms first avoids allocation
(see `_visit_operator_add2` and `_fold_operator_add` in assembly/stencil_eval.jl).
=#

# Which leaves a term goes to. Resolved once per term, not once per (term, leaf).
@inline function _routed_target(term, nleaves::Int)
    target = test_component_or_nothing(term)
    _check_component(target, nleaves, "test")
    return target
end

@inline _goes_to_leaf(::Nothing, ::Int) = true
@inline _goes_to_leaf(target::Int, c::Int) = target == c

# `f(leaf_space, offset)`, for each leaf the term routes to.
@inline function each_routed_leaf(f::F, term, leaves) where {F}
    target = _routed_target(term, length(leaves))
    _foldl_leaves(nothing, leaves) do _, c, leaf
        _goes_to_leaf(target, c) && f(first(leaf), last(leaf))
        return nothing
    end
    return nothing
end

# `acc = f(leaf_space, offset, acc)`, for each leaf the term routes to. Separate from
# `each_routed_leaf` rather than expressed through it: threading the accumulator as a return
# value is what keeps it concretely typed instead of captured and boxed.
@inline function fold_routed_leaves(f::F, term, leaves, acc::T) where {F, T}
    target = _routed_target(term, length(leaves))
    return _foldl_leaves(acc, leaves) do a, c, leaf
        return _goes_to_leaf(target, c) ? f(first(leaf), last(leaf), a) : a
    end
end

# --- the three consumers ---------------------------------------------------------- #

function _route_terms!(b::AbstractVector, op::OperatorAdd, leaves, α = true)
    return _visit_operator_add2(_route_terms!, b, op, leaves, α)
end

function _route_terms!(b::AbstractVector, term::TERM, leaves, α = true) where {TERM}
    each_routed_leaf(term, leaves) do sp, offset
        return _scatter_term!(b, sp, term, offset, α)
    end
    return b
end

# --- Contraction: accumulating a scalar without vector allocation ------------------ #

# `l(vₕ)` evaluates to a scalar by fusing the stencil evaluation with contraction against `vₕ`.
# The scalar case is `_contract_term` (below) at offset zero: one leaf, the whole space.
@inline _contract_linear_core(
    space, ast::AST_TYPE, v::AbstractVector, acc::T
) where {AST_TYPE, T} = _fold_walk_units(
    (t, a) -> _contract_term(space, t, 0, v, a), ast, acc, eltype(space))

function _contract_linear_core(
        space::CompositeGridSpace{N}, ast::AST_TYPE, v::AbstractVector, acc::T
) where {N, AST_TYPE, T}
    return _route_terms_contract(ast, acc, leaf_spaces_offsets(space), v)
end

# The counterpart of `_scatter_term!`, functioning as a barrier.
function _contract_term(
        sp, term::TERM, offset::Int, v::AbstractVector, acc::T
) where {TERM, T}
    Ωsp = mesh(sp)
    lin_indices = LinearIndices(indices(Ωsp))
    bound, mesh_markers = _bind_walk(term, sp)
    for I in indices(Ωsp)
        lin_idx = lin_indices[I]
        stencil = local_stencil(bound, sp, I, mesh_markers, lin_idx)

        for (off_v, weight) in stencil
            Iv = I + CartesianIndex(off_v)

            if checkbounds(Bool, lin_indices, Iv)
                @inbounds acc += weight * v[lin_indices[Iv] + offset]
            end
        end
    end
    return acc
end

function _route_terms_contract(op::OperatorAdd, acc, leaves, v)
    return _fold_operator_add(_route_terms_contract, op, acc, leaves, v)
end

function _route_terms_contract(term::TERM, acc::T, leaves, v) where {TERM, T}
    return fold_routed_leaves(term, leaves, acc) do sp, offset, a
        return _contract_term(sp, term, offset, v, a)
    end
end

# Threaded routing by term: hoists component resolution outside the inner loop.
function _route_terms_parallel!(b::AbstractVector, op::OperatorAdd, leaves, α = true)
    return _visit_operator_add2(_route_terms_parallel!, b, op, leaves, α)
end

function _route_terms_parallel!(b::AbstractVector, term::TERM, leaves, α = true) where {TERM}
    # Hoisted out of the per-leaf call, as before: the colouring depends on the term's
    # stencil, not on which leaf it lands in.
    strides = _term_colour_strides(term)
    each_routed_leaf(term, leaves) do sp, offset
        return _sweep_parallel!(b, sp, term, indices(mesh(sp)), strides, offset, α)
    end
    return b
end

# Function barrier for term scattering.
function _scatter_term!(b::AbstractVector, sp, term::TERM, offset::Int, α = true) where {TERM}
    Ωsp = mesh(sp)
    lin_indices = LinearIndices(indices(Ωsp))
    bound, mesh_markers = _bind_walk(term, sp)
    for I in indices(Ωsp)
        _scatter_linear_point!(b, sp, bound, I, lin_indices, mesh_markers, offset, α)
    end
    return b
end

function _assemble_linear_parallel_core!(
        b::AbstractVector, space::CompositeGridSpace{N}, ast::AST_TYPE, α = true
) where {N, AST_TYPE}
    return _route_terms_parallel!(b, ast, leaf_spaces_offsets(space), α)
end

"""
    assemble!(b::AbstractVector, form::LinearForm; dirichlet = nothing,
              dirichlet_components = nothing) -> AbstractVector

Refill `b` with the assembled `form` and return it with zero allocations (**0 bytes**).
A product with three or more region-restricted summands on its test side is split by `form`
into products of at most two (gpena/Bramble.jl#727); one it leaves whole, as it would split
into more than 16 or its constant scales keep several types, allocates per grid point.

On a test space whose backend lives on a device (a Metal space, say), `b` is filled on the
host and uploaded in one `copyto!`, as system matrices are: that path
allocates a host buffer and host mirrors of the space on every call, so the zero-allocation
guarantee is the host path's alone.

`assemble!` uses the pre-resolved `form.ast` stored directly inside the form.

## Live coefficients
- Grid functions: the stored AST retains references to source `VectorElement` storage. Mutating values in-place (`Rₕ!(uₕ, ...)` or `parent(uₕ) .= ...`) between steps automatically updates the assembled vector without needing to rebuild the form. Nested scales such as `uₕ * (wₕ * v)` stay live too, because each grid function is read at assembly time and never fused into a copy when the form is built.
- Dynamic scalars: plain numbers work directly for constant scalars. To update a scalar dynamically across loop iterations, wrap it in a `Ref(val)` (e.g. `α = Ref(1.0); l = form(Wₕ, v -> α * innerₕ(uₕ, v))`). Mutating `α[] = new_val` evaluates live during assembly with 0 allocations.

# Arguments
- `b`: Vector to refill, with length `ndofs(test_space(form))`; any other length raises a `DimensionMismatch` before anything is written.
- `form`: Linear form to assemble.

# Keywords
- `dirichlet`: Boundary values to impose after assembly, and the labels to impose them on, together -- a `label => f` `Pair`, a `Tuple` of such `Pair`s, or constraints from [`dirichlet_constraints`](@ref) (default: `nothing`; see [`_normalize_dirichlet`](@ref) for every accepted form).
- `dirichlet_components`: Restricts which leaf components of a composite `test_space(form)` the labels bind to (see [`dirichlet_bc!`](@ref); default: `nothing`, targeting all leaves).

Runs serially or across threads following `test_space(form)`'s backend [`execution_policy`](@ref):
[`Serial`](@ref) or [`Parallel`](@ref). [`assemble_parallel!`](@ref) forces threaded execution
regardless of backend policy.

See also [`assemble`](@ref), [`assemble_parallel!`](@ref), and [`evaluate!`](@ref).
"""
function assemble!(
        b::AbstractVector,
        form::LinearForm;
        dirichlet = nothing,
        dirichlet_components = nothing
)
    return _assemble_linear!(b, form, form.ast, dirichlet, dirichlet_components)
end

# The shared core behind `assemble!` and `assemble`: takes `ast` positionally, the public
# entry points passing `form.ast`, the device path its host-sourced copy (`_host_sources`).
# Split on where the test space lives, a type-level choice the compiler folds away: a host
# space sweeps `b` directly, a device one goes through a host buffer
# (`_assemble_linear_device!`, below). The length check comes first, before any write and
# before the device path builds its buffer.
@inline function _assemble_linear!(
        b::AbstractVector, form::LinearForm, ast, dirichlet, dirichlet_components
)
    _check_linear_length(b, form.test_space)
    return _assemble_linear!(
        locality(backend(form.test_space)), b, form, ast, dirichlet, dirichlet_components
    )
end

@inline function _assemble_linear!(
        ::DeviceLocality, b::AbstractVector, form::LinearForm, ast, dirichlet,
        dirichlet_components
)
    return _assemble_linear_device!(b, form, ast, dirichlet, dirichlet_components)
end

function _assemble_linear!(
        ::HostLocality, b::AbstractVector, form::LinearForm, ast, dirichlet,
        dirichlet_components
)
    dirichlet_labels, dirichlet_conditions = _normalize_dirichlet(dirichlet)
    fill!(b, zero(eltype(b)))
    space = form.test_space
    _validate_term_markers(ast, markers(mesh(space)), "the form's space")

    # A genuine 3-way dispatch, not a binary `isa CpuSerial` check:
    # `CpuPolyester` is neither `CpuSerial` nor `CpuThreaded`'s `Threads.@threads` path, and
    # Two branches, not three: `CpuSerial` runs the serial core, and everything else goes
    # to `_assemble_linear_parallel_core!`, whose own `_sweep_parallel!` computes the
    # effective policy and dispatches to `Threads.@threads` for `CpuThreaded` or to the
    # `_batch_*` hook for `CpuPolyester`. A `CpuPolyester` fast-fail used to sit here, on the
    # reasoning that failing before the call chain was more honest; it was neither, since
    # it fired even with Polyester loaded and the hooks implemented, so `assemble` on a
    # `LinearForm` could never work under `CpuPolyester` at all. Without
    # Polyester the hook still raises, one frame deeper, naming the package.
    policy = execution_policy(space)
    if policy isa CpuSerial
        _assemble_linear_core!(b, space, ast)
    else
        _assemble_linear_parallel_core!(b, space, ast)
    end

    apply_dirichlet_conditions!(
        b, form, dirichlet_conditions, dirichlet_labels, dirichlet_components
    )
    return b
end

# --- Device test spaces: host fill, one upload ---------------- #
#
# The sweep reads weights, spacings and source values one grid point at a time, which a
# device-backed space refuses. So a device test space is assembled the way matrices are:
# the whole sweep, Dirichlet values included, runs on a host mirror of the space
# (`host_weights` per leaf) into a host buffer, and `b` receives it in one `copyto!`. Not a
# device scatter kernel, and not scalar writes under `allowscalar`, which would be correct
# and silently slow. The buffer and the mirrors are rebuilt on every call, so unlike the
# host path this one allocates; a mirror is a snapshot, which is also what keeps live
# coefficients live: a `SourceVector` over device storage is copied fresh each time.
function _assemble_linear_device!(
        b::AbstractVector, form::LinearForm{D}, ast, dirichlet, dirichlet_components
) where {D}
    hspace = _host_mirror_space(form.test_space)
    hast = _host_sources(DeviceLocality(), ast)
    hform = LinearForm{D, typeof(hspace), typeof(hast)}(hspace, hast)
    hb = Vector{eltype(b)}(undef, length(b))
    _assemble_linear!(HostLocality(), hb, hform, hast, dirichlet, dirichlet_components)
    copyto!(_vector_storage(b), hb)
    return b
end

# The sweep writes `b` with `@inbounds`, so a vector of the wrong length would be written out
# of bounds (a short one) or left with a stale tail (a long one). Every linear entry point
# (`assemble!`, `assemble_add!`, `assemble_parallel!`) checks it before writing anything.
@inline function _check_linear_length(b::AbstractVector, space)
    length(b) == ndofs(space) || _throw_linear_length(length(b), ndofs(space))
    return nothing
end

@noinline function _throw_linear_length(actual::Int, expected::Int)
    throw(
        DimensionMismatch(
        "the vector has length $actual, but the form's test space has $expected degrees of freedom",
    ),
    )
end

# The storage a device path downloads from and uploads to. A `VectorElement` passed as `b`
# is unwrapped, since copying through the wrapper indexes the device array one scalar at a
# time; any other vector (a device vector, or a strided view of one) is copied as it is.
# Not `parent` in general: a view's parent is the whole array, not the vector.
@inline _vector_storage(b::AbstractVector) = b
@inline _vector_storage(b::VectorElement) = parent(b)

# Leaf by leaf, so a composite mirror keeps the leaf order (and so the offsets) the lowered
# sources were sampled against.
@inline _host_mirror_space(Wₕ::ScalarGridSpace) = host_weights(Wₕ)
@inline _host_mirror_space(Wₕ::CompositeGridSpace) = CompositeGridSpace(
    map(_host_mirror_space, Wₕ.spaces))

# Host copies of the per-point data a linear form's AST carries: `SourceVector` storage (a
# lowered `f(x)` sampled on the device space, or a `VectorElement` source) and a
# `GridFunctionScale`'s coefficient (`u * v`, where `u` is a `VectorElement` whose own type
# says nothing about where its storage lives, so its `parent` decides; the copy is a plain
# `Vector`, which `_grid_function_value` indexes the same way). The walk covers the nodes a
# source reaches through (`_lower_sources`'s own set plus `RegionRestriction`) and every
# `@node_family` node, so a coefficient under `Mₓ(u * v)` or `D₋ₓ(u * v)` is reached too
#; a node outside it is returned as it is, and device data left inside
# one fails loudly at its first scalar read rather than assembling wrong numbers.
@inline _host_sources(op) = op

# The same, keyed on where the space lives: only a device space's AST is walked.
@inline _host_sources(::DeviceLocality, op) = _host_sources(op)

@inline _host_array(x) = x
@inline _host_array(v::AbstractArray) = _host_array(locality(typeof(v)), v)
@inline _host_array(::HostLocality, v::AbstractArray) = v
@inline _host_array(::DeviceLocality, v::AbstractArray) = Array(v)
# A host `VectorElement` stays itself, so a bilinear walk binding it (`stencil_eval.jl`)
# keeps its type and copies nothing; only device storage is copied, to a plain `Vector`.
@inline _host_array(v::VectorElement) = _host_array(locality(typeof(parent(v))), v)
@inline _host_array(::DeviceLocality, v::VectorElement) = Array(parent(v))

function _host_sources(op::SourceVector{D}) where {D}
    vec = _host_array(op.vec)
    return SourceVector{D, typeof(vec)}(vec)
end

function _host_sources(op::GridFunctionScale{D}) where {D}
    gf = _host_array(op.grid_function)
    inner = _host_sources(op.inner_op)
    return GridFunctionScale{D, typeof(gf), typeof(inner)}(gf, inner)
end

function _host_sources(op::OperatorAdd{D}) where {D}
    left = _host_sources(op.left_op)
    right = _host_sources(op.right_op)
    return OperatorAdd{D, typeof(left), typeof(right)}(left, right)
end

function _host_sources(op::LinearProduct{D, W}) where {D, W}
    left = _host_sources(op.left_op)
    right = _host_sources(op.right_op)
    return LinearProduct{D, W, typeof(left), typeof(right)}(op.inner, left, right)
end

# The nodes holding one `inner_op` and nothing to copy: `OperatorScale`, `ShiftNode`,
# `RegionRestriction` (its region and zero stencil kept) and the `@node_family` nodes
# (`operators/node_family.jl`), each rebuilt around the walked operand by `_rewrap_inner`.
function _host_sources(op::Union{OperatorScale, ShiftNode, RegionRestriction,
        BackwardDifference, ForwardDifference, CenteredDifference, StarDifference,
        CrossWeightedDifference, SecondDifference, BackwardAverage, ForwardAverage,
        CenteredAverage, JumpNode})
    return _rewrap_inner(op, _host_sources(op.inner_op))
end

"""
    assemble_parallel!(b::AbstractVector, form::LinearForm) -> AbstractVector

Refill `b` with the assembled `form` across threads and return it, regardless of
`test_space(form)`'s backend execution policy. Unlike [`assemble!`](@ref), does not
apply Dirichlet conditions. A stencil reaching neighbours is swept by colour, so the entries
can differ from the serial sweep in the last bits; repeated threaded calls agree exactly.
On a device test space the threaded sweep runs on a host
mirror into a host buffer, uploaded to `b` in one `copyto!`, as in [`assemble!`](@ref).
"""
function assemble_parallel!(b::AbstractVector, form::LinearForm)
    _check_linear_length(b, form.test_space)
    return _assemble_linear_parallel!(
        locality(backend(form.test_space)), b, form.test_space, form.ast)
end

function _assemble_linear_parallel!(::HostLocality, b::AbstractVector, space, ast)
    _validate_term_markers(ast, markers(mesh(space)), "the form's space")

    fill!(b, zero(eltype(b)))
    _assemble_linear_parallel_core!(b, space, ast)

    return b
end

function _assemble_linear_parallel!(::DeviceLocality, b::AbstractVector, space, ast)
    hb = Vector{eltype(b)}(undef, length(b))
    _assemble_linear_parallel!(
        HostLocality(), hb, _host_mirror_space(space), _host_sources(ast))
    copyto!(_vector_storage(b), hb)
    return b
end
