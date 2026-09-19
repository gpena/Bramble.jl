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
    ast = form.ast
    space = form.test_space
    T = promote_type(_assembled_eltype(ast, space), eltype(parent(vₕ)))

    return _contract_linear_core(space, ast, parent(vₕ), zero(T))
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
l = form(Wₕ, v -> innerₕ(fₕ, v))
scratch = zeros(ndofs(Wₕ))
for step in 1:nsteps
    Rₕ!(fₕ, source_at(step))          # modified in-place
    value = evaluate!(scratch, l, uₕ)
end
```
"""
@inline function evaluate!(
        scratch::AbstractVector, form::LinearForm, vₕ::VectorElement; ast = nothing
)
    resolved_ast = ast === nothing ? form.ast : (_warn_ast_keyword(:evaluate!); ast)
    _assemble_linear!(scratch, form, resolved_ast, nothing, nothing)
    return dot(scratch, parent(vₕ))
end

@noinline function evaluate!(::AbstractVector, ::LinearForm, v::AbstractVector; kwargs...)
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

@inline _validate_form_expression(::LazyOp{D}, ::Val{D}) where {D} = nothing

@noinline function _validate_form_expression(bad, ::Val{D}) where {D}
    throw(
        ArgumentError(
        "a linear form's expression has to build an operator over its test argument, and " *
        "this one returned a $(typeof(bad)). Write it as a function of the test argument (`v -> innerₕ(fₕ, v)`) " *
        "rather than as a value.",
    ),
    )
end

@noinline function _validate_form_expression(::LazyOp{E}, ::Val{D}) where {E, D}
    throw(
        ArgumentError(
        "a linear form's expression is $(E)-dimensional and its test space is $(D). The " *
        "operators in the expression have to come from the same space the form is built " *
        "over.",
    ),
    )
end

# Two node types `_lower_sources` (form/common.jl) needs its own method for, defined here
# rather than alongside the rest of the family there because both are included after
# common.jl -- a type named in a method *signature* has to already exist (bramble-performance
# skill, "include-order rule"), unlike a name used only in a body.

# `LinearProduct`'s left side is always a source (`_is_source_only`), its right side never
# is, so only the left needs lowering.
function _lower_sources(op::LinearProduct{D, W}, space) where {D, W}
    left = _lower_sources(op.left_op, space)
    return left === op.left_op ? op :
           LinearProduct{D, W, typeof(left), typeof(op.right_op)}(left, op.right_op)
end

function _lower_sources(op::ShiftNode{D, Dim}, space) where {D, Dim}
    inner = _lower_sources(op.inner_op, space)
    return inner === op.inner_op ? op : ShiftNode{D, Dim, typeof(inner)}(op.shift_amount, inner)
end

# --- Eager source lowering across a CompositeGridSpace's leaves (gpena/Bramble.jl#197) --- #
#
# A non-composite space has exactly one leaf -- itself -- so `_lower_sources` above runs
# directly against `Wₕ`, unambiguous. A composite space's terms follow the same routing rule
# `_route_terms!` assembles by (`_routed_target`, above): a term naming one component lowers
# against that leaf's own space; a term naming none goes to *every* leaf, which may have
# different meshes, so there is no single space to sample against and lowering is skipped for
# it -- the same `SourceFunction` it already was, evaluated fresh per leaf at assembly time
# exactly as today. Missing this optimisation for that one shape is the safe choice: sampling
# against the wrong leaf's mesh would silently assemble the wrong numbers, not merely run
# slower.
_lower_sources_for_space(ast, Wₕ) = _lower_sources(ast, Wₕ)

function _lower_sources_for_space(ast, Wₕ::CompositeGridSpace)
    return _lower_sources_over_leaves(ast, leaf_spaces_offsets(Wₕ))
end

function _lower_sources_over_leaves(op::OperatorAdd, leaves)
    left = _lower_sources_over_leaves(op.left_op, leaves)
    right = _lower_sources_over_leaves(op.right_op, leaves)
    return left === op.left_op && right === op.right_op ? op : OperatorAdd(left, right)
end

function _lower_sources_over_leaves(term::TERM, leaves) where {TERM}
    target = test_component_or_nothing(term)
    (target === nothing || target < 1 || target > length(leaves)) && return term
    return _lower_sources(term, first(leaves[target]))
end

"""
    form(Wₕ, f) -> LinearForm

Construct a `LinearForm` over the test space `Wₕ` using the linear expression `f`.

Construction resolves the AST once and runs [`simplify_ast`](@ref) over it -- factoring
common scalings, combining like terms, and eliding zero-scaled ones -- before it is stored.
Grid partitioning for parallel assembly is determined from the resolved AST during assembly
(see `_colour_strides`).

Every `SourceFunction` reachable from the simplified AST -- a source term built directly from
a plain function, `f(x)`, rather than a [`VectorElement`](@ref) or a `Ref` -- is then
sampled once against its own leaf's space and lowered to a `SourceVector`
(gpena/Bramble.jl#197): a brand-new closure passed as a source used to force a full
recompilation of the assembly pipeline (~11ms measured) on every distinct closure, since
Julia gives it its own type; assembling against the fixed `SourceVector` shape instead means
that cost is paid once, here, not on every later `assemble!`/`assemble` call. This changes
what a raw closure that captures mutable state does: it is evaluated once, now, not
re-evaluated on later assemblies. `VectorElement` and `Ref` coefficients are unaffected --
see the note on `_lower_sources` in `form/common.jl` for why, and for the documented
alternative (`update_coefficients!`) a source meant to keep varying should use instead.

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
    _validate_form_expression(raw_ast, Val(D))
    ast = simplify_ast(resolve_ast(raw_ast))
    ast = _lower_sources_for_space(ast, Wₕ)
    return LinearForm{D, typeof(Wₕ), typeof(ast)}(Wₕ, ast)
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
        form::LinearForm; dirichlet = nothing, dirichlet_components = nothing, ast = nothing
)
    resolved_ast = ast === nothing ? form.ast : (_warn_ast_keyword(:assemble); ast)
    space = test_space(form)
    # `parent(element(space, T))` reuses the space's backend container type.
    b = parent(element(space, _assembled_eltype(resolved_ast, space)))
    return _assemble_linear!(b, form, resolved_ast, dirichlet, dirichlet_components)
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
    return promote_type(
        _routed_eltype(op.left_op, leaves, T), _routed_eltype(op.right_op, leaves, T)
    )
end

function _routed_eltype(term, leaves, T)
    target = test_component_or_nothing(term)
    _check_component(target, length(leaves))
    sp = target === nothing ? first(first(leaves)) : first(leaves[target])
    return _probed_eltype(term, sp, T)
end

# --- Helper cores for function barrier optimization ------------------------------- #

# The scalar case is `_scatter_term!` (below) at offset zero: one leaf, the whole space.
@inline _assemble_linear_core!(b::AbstractVector, space, ast::AST_TYPE, α = true) where {AST_TYPE} = _scatter_term!(
    b, space, ast, 0, α)

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
@inline function _colour_subgrid(
        grid_inds::CartesianIndices{D}, c::CartesianIndex{D}, strides::NTuple{D, Int}
) where {D}
    return CartesianIndices(ntuple(d -> c[d]:strides[d]:last(axes(grid_inds, d)), D))
end

# The threaded pass over one colour, writing directly into `b`.
"""
    _scatter_linear_point!(b, sp, term, I, lin_indices, mesh_markers, offset) -> Nothing

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

"""
    _sweep_linear_band_colour!(b, sp, term, ax, parity, nbands, rest, lin_indices, mesh_markers, offset) -> Nothing

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
    Threads.@threads for k in bidx
        for I in CartesianIndices((rest..., _band_range(ax, nbands, k)))
            _scatter_linear_point!(b, sp, term, I, lin_indices, mesh_markers, offset, α)
        end
    end
    return nothing
end

@noinline function _sweep_linear_band_colour!(
        ::CpuBatch,
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
    return _batch_linear_band_sweep!(
        b, sp, term, ax, bidx, nbands, rest, lin_indices, mesh_markers, offset, α
    )
end

"""
    _batch_linear_band_sweep!(b, sp, term, ax, bidx, nbands, rest, lin_indices, mesh_markers, offset, α) -> Nothing

[`CpuBatch`](@ref)'s counterpart of the `Threads.@threads` body in
[`_sweep_linear_band_colour!`](@ref), filled by `BramblePolyesterExt`
(gpena/Bramble.jl#190). The only `src/` method errors naming Polyester.
"""
@noinline function _batch_linear_band_sweep!(
        b, sp, term, ax, bidx, nbands, rest, lin_indices, mesh_markers, offset, α
)
    return _throw_cpubatch_without_polyester(:_batch_linear_band_sweep!)
end

# Dispatches on the *effective* execution policy (`_sweep_parallel!` computes it):
# `CpuThreaded` keeps `Threads.@threads` exactly as before; `CpuBatch` reaches its own hook
# instead, so it never silently threads with the wrong mechanism (gpena/Bramble.jl#190).
# `CpuSerial` never reaches this function -- `_effective_parallel_policy` only ever hands it
# `CpuThreaded` or `CpuBatch`.
@noinline function _sweep_colour!(
        ::CpuThreaded, b::AbstractVector, sp, term::TERM, idxs, lin_indices, mesh_markers, offset::Int, α = true
) where {TERM}
    Threads.@threads for I in idxs
        _scatter_linear_point!(b, sp, term, I, lin_indices, mesh_markers, offset, α)
    end
    return nothing
end

@noinline function _sweep_colour!(
        ::CpuBatch, b::AbstractVector, sp, term::TERM, idxs, lin_indices, mesh_markers, offset::Int, α = true
) where {TERM}
    return _batch_linear_colour_sweep!(b, sp, term, idxs, lin_indices, mesh_markers, offset, α)
end

"""
    _batch_linear_colour_sweep!(b, sp, term, idxs, lin_indices, mesh_markers, offset, α) -> Nothing

[`CpuBatch`](@ref)'s counterpart of the `Threads.@threads` body in `_sweep_colour!`,
filled by `BramblePolyesterExt` (gpena/Bramble.jl#190). The only `src/` method errors
naming Polyester.
"""
@noinline function _batch_linear_colour_sweep!(b, sp, term, idxs, lin_indices, mesh_markers, offset, α)
    return _throw_cpubatch_without_polyester(:_batch_linear_colour_sweep!)
end

# Every colour in turn. `policy` is the *effective* policy (`_effective_parallel_policy(sp)`,
# computed once here): `CpuSerial` is coerced to `CpuThreaded` since every call into this
# function is already on the forced-threaded path (`_assemble_linear_parallel_core!`,
# entered from a non-`CpuSerial` branch, or from `assemble_parallel!`'s own "regardless of
# policy" contract); `CpuBatch` passes through unchanged so the colour/band sweeps below
# reach their own hook instead of `Threads.@threads` (gpena/Bramble.jl#190).
function _sweep_parallel!(
        b::AbstractVector, sp, term::TERM, grid_inds, strides, offset::Int, α = true
) where {TERM}
    Ωsp = mesh(sp)
    lin_indices = LinearIndices(indices(Ωsp))
    mesh_markers = markers(Ωsp)
    policy = _effective_parallel_policy(sp)

    # Bands before colours, for the reason spelled out in `_sweep_bilinear!`: two slabs
    # instead of `prod(strides)` strided colours, each walked contiguously.
    inds = grid_inds.indices
    D = length(strides)
    ax = inds[D]
    nbands = _band_count(length(ax), strides[D], Threads.nthreads())

    if nbands != 0
        rest = Base.front(inds)
        # One pass over every band when nothing can collide; see `_sweep_bilinear!`.
        bands = prod(strides) == 1 ? (1:1:nbands,) : (1:2:nbands, 2:2:nbands)
        for bidx in bands
            _sweep_linear_band_colour!(
                policy, b, sp, term, ax, bidx, nbands, rest, lin_indices, mesh_markers, offset, α
            )
        end
        return b
    end

    if prod(strides) == 1
        _sweep_colour!(policy, b, sp, term, grid_inds, lin_indices, mesh_markers, offset, α)
        return b
    end

    for c in CartesianIndices(strides)
        _sweep_colour!(
            policy,
            b,
            sp,
            term,
            _colour_subgrid(grid_inds, c, strides),
            lin_indices,
            mesh_markers,
            offset,
            α
        )
    end
    return b
end

function _assemble_linear_parallel_core!(
        b::AbstractVector, space, ast::AST_TYPE, α = true
) where {AST_TYPE}
    strides = _colour_strides(stencil_offsets(ast))
    _sweep_parallel!(b, space, ast, indices(mesh(space)), strides, 0, α)
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
@inline _check_component(::Nothing, ::Int) = nothing

@inline function _check_component(target::Int, nblocks::Int)
    1 <= target <= nblocks || _throw_component_out_of_range(target, nblocks)
    return nothing
end

@noinline function _throw_component_out_of_range(target::Int, nblocks::Int)
    throw(
        ArgumentError(
        "a term of this form names component $target, and its test space has $nblocks. " *
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
contract into an accumulator, sweep threaded (gpena/Bramble.jl#55). It is now written once
here, and the three consumers differ only in what they do per leaf.

Recursing the tree rather than flattening it into a vector of terms first avoids allocation
(see `_visit_operator_add2` and `_fold_operator_add` in form/stencil_eval.jl).
=#

# Which leaves a term goes to. Resolved once per term, not once per (term, leaf).
@inline function _routed_target(term, nleaves::Int)
    target = test_component_or_nothing(term)
    _check_component(target, nleaves)
    return target
end

@inline _goes_to_leaf(::Nothing, ::Int) = true
@inline _goes_to_leaf(target::Int, c::Int) = target == c

# `f(leaf_space, offset)`, for each leaf the term routes to.
@inline function each_routed_leaf(f::F, term, leaves) where {F}
    target = _routed_target(term, length(leaves))
    for (c, leaf) in enumerate(leaves)
        _goes_to_leaf(target, c) || continue
        f(first(leaf), last(leaf))
    end
    return nothing
end

# `acc = f(leaf_space, offset, acc)`, for each leaf the term routes to. Separate from
# `each_routed_leaf` rather than expressed through it: threading the accumulator as a return
# value is what keeps it concretely typed instead of captured and boxed.
@inline function fold_routed_leaves(f::F, term, leaves, acc::T) where {F, T}
    target = _routed_target(term, length(leaves))
    for (c, leaf) in enumerate(leaves)
        _goes_to_leaf(target, c) || continue
        acc = f(first(leaf), last(leaf), acc)
    end
    return acc
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
) where {AST_TYPE, T} = _contract_term(space, ast, 0, v, acc)

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
    strides = _colour_strides(stencil_offsets(term))
    each_routed_leaf(term, leaves) do sp, offset
        return _sweep_parallel!(b, sp, term, indices(mesh(sp)), strides, offset, α)
    end
    return b
end

# Function barrier for term scattering.
function _scatter_term!(b::AbstractVector, sp, term::TERM, offset::Int, α = true) where {TERM}
    Ωsp = mesh(sp)
    lin_indices = LinearIndices(indices(Ωsp))
    mesh_markers = markers(Ωsp)
    for I in indices(Ωsp)
        lin_idx = lin_indices[I]
        stencil = local_stencil(term, sp, I, mesh_markers, lin_idx)

        for (off_v, weight) in stencil
            Iv = I + CartesianIndex(off_v)

            if checkbounds(Bool, lin_indices, Iv)
                @inbounds b[lin_indices[Iv] + offset] += α * weight
            end
        end
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

`assemble!` uses the pre-resolved `form.ast` stored directly inside the form.

## Live coefficients
- Grid functions: the stored AST retains references to source `VectorElement` storage. Mutating values in-place (`Rₕ!(uₕ, ...)` or `parent(uₕ) .= ...`) between steps automatically updates the assembled vector without needing to rebuild the form.
- Dynamic scalars: plain numbers work directly for constant scalars. To update a scalar dynamically across loop iterations, wrap it in a `Ref(val)` (e.g. `α = Ref(1.0); l = form(Wₕ, v -> α * innerₕ(uₕ, v))`). Mutating `α[] = new_val` evaluates live during assembly with 0 allocations.

# Arguments
- `b`: Vector to refill, with length `ndofs(test_space(form))`.
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
        dirichlet_components = nothing,
        ast = nothing
)
    resolved_ast = ast === nothing ? form.ast : (_warn_ast_keyword(:assemble!); ast)
    return _assemble_linear!(b, form, resolved_ast, dirichlet, dirichlet_components)
end

# The shared core behind `assemble!` and `assemble`: takes its `ast` positionally, already
# resolved and already past the deprecation check, so neither public entry point warns twice
# calling into the other.
function _assemble_linear!(
        b::AbstractVector, form::LinearForm, ast, dirichlet, dirichlet_components
)
    dirichlet_labels, dirichlet_conditions = _normalize_dirichlet(dirichlet)
    fill!(b, zero(eltype(b)))
    space = form.test_space
    _validate_term_markers(ast, markers(mesh(space)), "the form's space")

    # A genuine 3-way dispatch, not a binary `isa CpuSerial` check (gpena/Bramble.jl#190):
    # `CpuBatch` is neither `CpuSerial` nor `CpuThreaded`'s `Threads.@threads` path, and
    # Two branches, not three: `CpuSerial` runs the serial core, and everything else goes
    # to `_assemble_linear_parallel_core!`, whose own `_sweep_parallel!` computes the
    # effective policy and dispatches to `Threads.@threads` for `CpuThreaded` or to the
    # `_batch_*` hook for `CpuBatch`. A `CpuBatch` fast-fail used to sit here, on the
    # reasoning that failing before the call chain was more honest; it was neither, since
    # it fired even with Polyester loaded and the hooks implemented, so `assemble` on a
    # `LinearForm` could never work under `CpuBatch` at all (gpena/Bramble.jl#190). Without
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

"""
    assemble_parallel!(b::AbstractVector, form::LinearForm) -> AbstractVector

Refill `b` with the assembled `form` across threads and return it, regardless of
`test_space(form)`'s backend execution policy. Unlike [`assemble!`](@ref), does not
apply Dirichlet conditions.
"""
function assemble_parallel!(b::AbstractVector, form::LinearForm, ast = nothing)
    resolved_ast = ast === nothing ? form.ast : (_warn_ast_keyword(:assemble_parallel!); ast)
    space = form.test_space
    _validate_term_markers(resolved_ast, markers(mesh(space)), "the form's space")

    fill!(b, zero(eltype(b)))
    _assemble_linear_parallel_core!(b, space, resolved_ast)

    return b
end
