# --- Struct definitions ----------------------------------------------------------- #

"""
    LinearForm{D, TestSpace, AST}

Represents a linear form defined over a test space.

# Arguments
- `test_space::TestSpace`: Space for the test function.
- `ast::AST`: Resolved expression tree.

The form resolves its expression tree `ast` once at construction, referencing the underlying
storage of any coefficient grid functions (`VectorElement`). In-place updates via `Rₕ!(fₕ, ...)`
or `values(fₕ) .= ...` are automatically seen by subsequent assemblies with zero heap allocations.
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
    ast = form.ast
    space = form.test_space
    Ωₕ = mesh(space)
    T = promote_type(_assembled_eltype(ast, space), eltype(values(vₕ)))

    return _contract_linear_core(space, ast, LinearIndices(indices(Ωₕ)), markers(Ωₕ),
        values(vₕ), zero(T))
end

@noinline function (form::LinearForm)(v::AbstractVector)
    throw(ArgumentError(
        "a linear form contracts against an element of its test space, not a bare vector: " *
        "the length of a vector says nothing about whether its blocks match the components " *
        "the form routes to. Name the space first, with l(element(test_space(l), v))."))
end

"""
    evaluate!(scratch::AbstractVector, form::LinearForm, vₕ::VectorElement; ast = resolve_form_ast(form)) -> Number

Evaluate `form` at `vₕ`, assembling into `scratch` rather than into a newly allocated vector, and
return the resulting contracted scalar value.

Useful when both the assembled vector and the scalar value are needed (such as a Newton step
requiring the residual vector and its norm). `scratch` is overwritten and contains the right-hand
side upon return.

For the scalar value alone, `form(vₕ)` fuses the contraction into the assembly sweep in a single pass.
Pass `ast` to reuse the pre-resolved expression tree across iterations for zero allocations.

# Examples
```julia
l = form(Wₕ, v -> innerₕ(fₕ, v))
scratch = zeros(ndofs(Wₕ))
ast = resolve_form_ast(l)
for step in 1:nsteps
    Rₕ!(fₕ, source_at(step))          # modified in-place
    value = evaluate!(scratch, l, uₕ; ast = ast)
end
```
"""
@inline function evaluate!(scratch::AbstractVector, form::LinearForm, vₕ::VectorElement;
        ast = form.ast)
    assemble!(scratch, form; ast = ast)
    return dot(scratch, values(vₕ))
end

@noinline function evaluate!(::AbstractVector, ::LinearForm, v::AbstractVector; kwargs...)
    throw(ArgumentError(
        "a linear form contracts against an element of its test space, not a bare vector: " *
        "the length of a vector says nothing about whether its blocks match the components " *
        "the form routes to. Name the space first, with " *
        "evaluate!(scratch, l, element(test_space(l), v))."))
end

"""
    resolve_form_ast(form::LinearForm)

Return the resolved AST stored inside the linear form.
"""
@inline resolve_form_ast(form::LinearForm) = form.ast

@inline _validate_form_expression(::LazyOp{D}, ::Val{D}) where {D} = nothing

@noinline function _validate_form_expression(bad, ::Val{D}) where {D}
    throw(ArgumentError(
        "a linear form's expression has to build an operator over its test argument, and " *
        "this one returned a $(typeof(bad)). Write it as a function of the test argument (`v -> innerₕ(fₕ, v)`) " *
        "rather than as a value."))
end

@noinline function _validate_form_expression(::LazyOp{E}, ::Val{D}) where {E, D}
    throw(ArgumentError(
        "a linear form's expression is $(E)-dimensional and its test space is $(D). The " *
        "operators in the expression have to come from the same space the form is built " *
        "over."))
end

"""
    form(Wₕ, f) -> LinearForm

Construct a `LinearForm` over the test space `Wₕ` using the linear expression `f`.

Construction resolves the AST once; grid partitioning for parallel assembly is determined
from the resolved AST during assembly (see `_colour_strides`).

# Examples
```julia
# 1D linear form: l(v) = (f, v)
l = form(Wₕ, v -> innerₕ(fₕ, v))
```
"""
function form(Wₕ, f)
    D = dim(Wₕ)
    raw_ast = f(TestFunction{D}())
    _validate_form_expression(raw_ast, Val(D))
    ast = resolve_ast(raw_ast)
    return LinearForm{D, typeof(Wₕ), typeof(ast)}(Wₕ, ast)
end

# --- Assembly implementations ----------------------------------------------------- #

# Nothing to do unless labels are provided. `dirichlet_conditions` defaults to `nothing`
# to prevent allocations when boundary constraints are absent.
function apply_dirichlet_conditions!(
        b::AbstractVector, form::LinearForm, dirichlet_conditions, dirichlet_labels,
        dirichlet_components = nothing)
    dirichlet_labels === nothing && return b

    dirichlet_conditions === nothing && _throw_labels_without_conditions(dirichlet_labels)

    if dirichlet_labels isa Symbol
        dirichlet_bc!(b, test_space(form), dirichlet_conditions, dirichlet_labels;
            components = dirichlet_components)
    elseif dirichlet_labels isa Tuple && !isempty(dirichlet_labels)
        dirichlet_bc!(b, test_space(form), dirichlet_conditions, dirichlet_labels...;
            components = dirichlet_components)
    end
    return b
end

@noinline function _throw_labels_without_conditions(labels)
    throw(ArgumentError(
        "dirichlet_labels = $labels was given without dirichlet_conditions. Pass the " *
        "constraints as well: assemble(form; dirichlet_conditions = bcs, " *
        "dirichlet_labels = $labels)."))
end

"""
    assemble(form::LinearForm; dirichlet_conditions = nothing, dirichlet_labels = nothing, dirichlet_components = nothing, ast = form.ast) -> AbstractVector

Assemble the system vector of the `LinearForm`, applying `dirichlet_conditions` on the
regions named by `dirichlet_labels` when both are provided. `dirichlet_components` restricts which
leaf components of a composite test space they bind to (see [`dirichlet_bc!`](@ref)).

Runs serially or across threads following `test_space(form)`'s backend
[`execution_policy`](@ref): [`Serial`](@ref) (the default) or [`Parallel`](@ref).
[`assemble_parallel!`](@ref) always threads regardless of the backend policy.
"""
function assemble(form::LinearForm; dirichlet_conditions = nothing,
        dirichlet_labels = nothing, dirichlet_components = nothing, ast = form.ast)
    _validate_dirichlet_labels(dirichlet_labels)
    space = test_space(form)
    # `values(element(space, T))` reuses the space's backend container type.
    b = values(element(space, _assembled_eltype(ast, space)))
    return assemble!(b, form; ast = ast, dirichlet_conditions = dirichlet_conditions,
        dirichlet_labels = dirichlet_labels, dirichlet_components = dirichlet_components)
end

# The element type of the assembled vector is the one the form's own weights have, promoted
# against the space's (not the space's outright), supporting autodiff types like `ForwardDiff.Dual`.
function _assembled_eltype(ast, space)
    return _probed_eltype(ast, space, eltype(space))
end

# Composite: terms naming components are routed and probed on their respective leaf spaces.
function _assembled_eltype(ast, space::CompositeGridSpace)
    return _routed_eltype(ast, leaf_spaces_offsets(space), eltype(space))
end

# An interior point, so a truncated stencil does not decide the type. A restriction can
# still answer with nothing, in which case the space's type is used.
function _probed_eltype(term, sp, T)
    Ωₕ = mesh(sp)
    grid_inds = indices(Ωₕ)
    lin_indices = LinearIndices(grid_inds)
    I = grid_inds[length(grid_inds) ÷ 2 + 1]
    st = local_stencil(term, sp, I, markers(Ωₕ), lin_indices[I])
    isempty(st) && return T
    return promote_type(T, typeof(last(first(st))))
end

function _routed_eltype(op::OperatorAdd, leaves, T)
    promote_type(
        _routed_eltype(op.left_op, leaves, T), _routed_eltype(op.right_op, leaves, T))
end

function _routed_eltype(term, leaves, T)
    target = test_component_or_nothing(term)
    _check_component(target, length(leaves))
    sp = target === nothing ? first(first(leaves)) : first(leaves[target])
    return _probed_eltype(term, sp, T)
end

# --- Helper cores for function barrier optimization ------------------------------- #

function _assemble_linear_core!(
        b::AbstractVector, space, ast::AST_TYPE, lin_indices, mesh_markers) where {AST_TYPE}
    for I in indices(mesh(space))
        lin_idx = lin_indices[I]
        stencil = local_stencil(ast, space, I, mesh_markers, lin_idx)

        for (off_v, weight) in stencil
            Iv = I + CartesianIndex(off_v)

            if checkbounds(Bool, lin_indices, Iv)
                row = lin_indices[Iv]
                @inbounds b[row] += weight
            end
        end
    end
    return b
end

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
@inline function _colour_strides(offsets::Vector{NTuple{D, Int}}) where {D}
    isempty(offsets) && return ntuple(_ -> 1, D)

    lo = first(offsets)
    hi = first(offsets)
    for o in offsets
        lo = min.(lo, o)
        hi = max.(hi, o)
    end
    return hi .- lo .+ 1
end

# One colour of `grid_inds`, represented as a strided subgrid without allocating index vectors.
@inline function _colour_subgrid(grid_inds::CartesianIndices{D}, c::CartesianIndex{D},
        strides::NTuple{D, Int}) where {D}
    return CartesianIndices(ntuple(d -> c[d]:strides[d]:last(axes(grid_inds, d)), D))
end

# The threaded pass over one colour, writing directly into `b`.
@noinline function _sweep_colour!(b::AbstractVector, sp, term::TERM, idxs, lin_indices,
        mesh_markers, offset::Int) where {TERM}
    Threads.@threads for I in idxs
        stencil = local_stencil(term, sp, I, mesh_markers, lin_indices[I])

        for (off_v, weight) in stencil
            Iv = I + CartesianIndex(off_v)

            if checkbounds(Bool, lin_indices, Iv)
                @inbounds b[lin_indices[Iv] + offset] += weight
            end
        end
    end
    return nothing
end

# Every colour in turn.
function _sweep_parallel!(b::AbstractVector, sp, term::TERM, grid_inds, strides,
        offset::Int) where {TERM}
    Ωsp = mesh(sp)
    lin_indices = LinearIndices(indices(Ωsp))
    mesh_markers = markers(Ωsp)

    if prod(strides) == 1
        _sweep_colour!(b, sp, term, grid_inds, lin_indices, mesh_markers, offset)
        return b
    end

    for c in CartesianIndices(strides)
        _sweep_colour!(b, sp, term, _colour_subgrid(grid_inds, c, strides), lin_indices,
            mesh_markers, offset)
    end
    return b
end

function _assemble_linear_parallel_core!(b::AbstractVector, space, ast::AST_TYPE,
        lin_indices, mesh_markers) where {AST_TYPE}
    strides = _colour_strides(stencil_offsets(ast))
    _sweep_parallel!(b, space, ast, indices(mesh(space)), strides, 0)
    return b
end

function _assemble_linear_core!(
        b::AbstractVector, space::CompositeGridSpace{N}, ast::AST_TYPE,
        lin_indices, mesh_markers) where {N, AST_TYPE}
    leaves = leaf_spaces_offsets(space)

    if !routes_by_component(ast)
        for (sp, offset) in leaves
            _scatter_term!(b, sp, ast, offset)
        end
        return b
    end

    _route_terms!(b, ast, leaves)
    return b
end

# A term naming a component the space does not have used to contribute nothing, in silence:
# the loops below match `target` against each leaf in turn, so a target past the end simply
# never matched. On a two-block space `innerₕ(1.0, v(3))` assembled to zeros, and
# `innerₕ(1.0, v(1)) + innerₕ(2.0, v(9))` dropped the second term and kept the first.
#
# That is the failure mode the composite tests exist to prevent, so it is checked once per
# term rather than left to the reader of the answer.
@inline _check_component(::Nothing, ::Int) = nothing

@inline function _check_component(target::Int, nblocks::Int)
    1 <= target <= nblocks || _throw_component_out_of_range(target, nblocks)
    return nothing
end

@noinline function _throw_component_out_of_range(target::Int, nblocks::Int)
    throw(ArgumentError(
        "a term of this form names component $target, and its test space has $nblocks. " *
        "Components are numbered 1 to $nblocks; a term written for a space with more of " *
        "them contributes nothing here, which is why this is an error rather than a zero."))
end

# Walk the sum and send each term to the blocks it belongs to. Recursing the tree rather
# than flattening it into a vector of terms first avoids allocation (see `_visit_operator_add2`
# in form/common.jl).
function _route_terms!(b::AbstractVector, op::OperatorAdd, leaves)
    _visit_operator_add2(
        _route_terms!, b, op, leaves)
end

function _route_terms!(b::AbstractVector, term::TERM, leaves) where {TERM}
    target = test_component_or_nothing(term)
    _check_component(target, length(leaves))
    for (c, leaf) in enumerate(leaves)
        # a term naming a component goes to that block alone; one naming none goes to every
        # block, so the two spellings can be mixed in a single form
        (target === nothing || target == c) || continue
        _scatter_term!(b, first(leaf), term, last(leaf))
    end
    return b
end

# --- Contraction: accumulating a scalar without vector allocation ------------------ #

# `l(vₕ)` evaluates to a scalar by fusing the stencil evaluation with contraction against `vₕ`.
function _contract_linear_core(space, ast::AST_TYPE, lin_indices, mesh_markers,
        v::AbstractVector, acc::T) where {AST_TYPE, T}
    for I in indices(mesh(space))
        lin_idx = lin_indices[I]
        stencil = local_stencil(ast, space, I, mesh_markers, lin_idx)

        for (off_v, weight) in stencil
            Iv = I + CartesianIndex(off_v)

            if checkbounds(Bool, lin_indices, Iv)
                @inbounds acc += weight * v[lin_indices[Iv]]
            end
        end
    end
    return acc
end

function _contract_linear_core(space::CompositeGridSpace{N}, ast::AST_TYPE, lin_indices,
        mesh_markers, v::AbstractVector, acc::T) where {N, AST_TYPE, T}
    leaves = leaf_spaces_offsets(space)

    if !routes_by_component(ast)
        for (sp, offset) in leaves
            acc = _contract_term(sp, ast, offset, v, acc)
        end
        return acc
    end

    return _route_terms_contract(ast, leaves, v, acc)
end

# The counterpart of `_scatter_term!`, functioning as a barrier.
function _contract_term(sp, term::TERM, offset::Int,
        v::AbstractVector, acc::T) where {TERM, T}
    Ωsp = mesh(sp)
    lin_indices = LinearIndices(indices(Ωsp))
    mesh_markers = markers(Ωsp)
    for I in indices(Ωsp)
        lin_idx = lin_indices[I]
        stencil = local_stencil(term, sp, I, mesh_markers, lin_idx)

        for (off_v, weight) in stencil
            Iv = I + CartesianIndex(off_v)

            if checkbounds(Bool, lin_indices, Iv)
                @inbounds acc += weight * v[lin_indices[Iv] + offset]
            end
        end
    end
    return acc
end

function _route_terms_contract(op::OperatorAdd, leaves, v, acc)
    acc = _route_terms_contract(op.left_op, leaves, v, acc)
    return _route_terms_contract(op.right_op, leaves, v, acc)
end

function _route_terms_contract(term::TERM, leaves, v, acc::T) where {TERM, T}
    target = test_component_or_nothing(term)
    _check_component(target, length(leaves))
    for (c, leaf) in enumerate(leaves)
        (target === nothing || target == c) || continue
        acc = _contract_term(first(leaf), term, last(leaf), v, acc)
    end
    return acc
end

# Threaded routing by term: hoists component resolution outside the inner loop.
function _route_terms_parallel!(b::AbstractVector, op::OperatorAdd, leaves)
    _visit_operator_add2(
        _route_terms_parallel!, b, op, leaves)
end

function _route_terms_parallel!(b::AbstractVector, term::TERM, leaves) where {TERM}
    target = test_component_or_nothing(term)
    _check_component(target, length(leaves))
    strides = _colour_strides(stencil_offsets(term))

    for (c, leaf) in enumerate(leaves)
        (target === nothing || target == c) || continue
        sp = first(leaf)
        _sweep_parallel!(b, sp, term, indices(mesh(sp)), strides, last(leaf))
    end
    return b
end

# Function barrier for term scattering.
function _scatter_term!(b::AbstractVector, sp, term::TERM, offset::Int) where {TERM}
    Ωsp = mesh(sp)
    lin_indices = LinearIndices(indices(Ωsp))
    mesh_markers = markers(Ωsp)
    for I in indices(Ωsp)
        lin_idx = lin_indices[I]
        stencil = local_stencil(term, sp, I, mesh_markers, lin_idx)

        for (off_v, weight) in stencil
            Iv = I + CartesianIndex(off_v)

            if checkbounds(Bool, lin_indices, Iv)
                @inbounds b[lin_indices[Iv] + offset] += weight
            end
        end
    end
    return b
end

function _assemble_linear_parallel_core!(b::AbstractVector,
        space::CompositeGridSpace{N}, ast::AST_TYPE,
        lin_indices, mesh_markers) where {N, AST_TYPE}
    leaves = leaf_spaces_offsets(space)
    strides = _colour_strides(stencil_offsets(ast))

    if !routes_by_component(ast)
        for (sp, offset) in leaves
            _sweep_parallel!(b, sp, ast, indices(mesh(sp)), strides, offset)
        end
        return b
    end

    _route_terms_parallel!(b, ast, leaves)
    return b
end

"""
    assemble!(b::AbstractVector, form::LinearForm; dirichlet_conditions = nothing,
              dirichlet_labels = nothing, dirichlet_components = nothing, ast = form.ast) -> AbstractVector

Refill `b` with the assembled `form` and return it with zero allocations (**0 bytes**).

By default `assemble!` uses the pre-resolved `form.ast` stored directly inside the form.

## Live coefficients
- Grid functions: the stored AST retains references to source `VectorElement` storage. Mutating values in-place (`Rₕ!(uₕ, ...)` or `values(uₕ) .= ...`) between steps automatically updates the assembled vector without needing to rebuild the form.
- Dynamic scalars: plain numbers work directly for constant scalars. To update a scalar dynamically across loop iterations, wrap it in a `Ref(val)` (e.g. `α = Ref(1.0); l = form(Wₕ, v -> α * innerₕ(uₕ, v))`). Mutating `α[] = new_val` evaluates live during assembly with 0 allocations.

# Arguments
- `b`: Vector to refill, with length `ndofs(test_space(form))`.
- `form`: Linear form to assemble.

# Keywords
- `dirichlet_conditions`, `dirichlet_labels`: Boundary values to impose after assembly and the region labels to impose them on (both default to `nothing`).
- `dirichlet_components`: Restricts which leaf components of a composite `test_space(form)` the labels bind to (see [`dirichlet_bc!`](@ref); default: `nothing`, targeting all leaves).
- `ast`: Optional custom AST override (defaults to `form.ast`).

Runs serially or across threads following `test_space(form)`'s backend [`execution_policy`](@ref):
[`Serial`](@ref) or [`Parallel`](@ref). [`assemble_parallel!`](@ref) forces threaded execution
regardless of backend policy.

See also [`assemble`](@ref), [`assemble_parallel!`](@ref), and [`evaluate!`](@ref).
"""
function assemble!(b::AbstractVector, form::LinearForm{D, TestSpace, AST};
        dirichlet_conditions = nothing,
        dirichlet_labels = nothing,
        dirichlet_components = nothing,
        ast = form.ast) where {D, TestSpace, AST}
    _validate_dirichlet_labels(dirichlet_labels)
    fill!(b, zero(eltype(b)))
    space = form.test_space
    Ωₕ = mesh(space)
    mesh_markers = markers(Ωₕ)
    _validate_term_markers(ast, mesh_markers, "the form's space")
    lin_indices = LinearIndices(indices(Ωₕ))

    if execution_policy(space) isa Serial
        _assemble_linear_core!(b, space, ast, lin_indices, mesh_markers)
    else
        _assemble_linear_parallel_core!(b, space, ast, lin_indices, mesh_markers)
    end

    apply_dirichlet_conditions!(b, form, dirichlet_conditions, dirichlet_labels,
        dirichlet_components)
    return b
end

"""
    assemble_parallel!(b::AbstractVector, form::LinearForm, ast = form.ast) -> AbstractVector

Refill `b` with the assembled `form` across threads and return it, regardless of
`test_space(form)`'s backend execution policy. Unlike [`assemble!`](@ref), does not
apply Dirichlet conditions.
"""
function assemble_parallel!(b::AbstractVector,
        form::LinearForm{D, TestSpace, AST},
        ast = form.ast) where {D, TestSpace, AST}
    space = form.test_space
    Ωₕ = mesh(space)
    mesh_markers = markers(Ωₕ)
    _validate_term_markers(ast, mesh_markers, "the form's space")
    lin_indices = LinearIndices(indices(Ωₕ))

    fill!(b, zero(eltype(b)))
    _assemble_linear_parallel_core!(b, space, ast, lin_indices, mesh_markers)

    return b
end

# update_ast_grid_coefficients! has been deprecated and deleted.
