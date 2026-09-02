
# ==============================================================================
# Struct Definitions
# ==============================================================================

"""
    LinearForm{D,TestSpace,FType,AST}

Represents a linear form defined over a test space.

# Fields
- `test_space::TestSpace`: The space for the test function.
- `f::FType`: The expression, as a function of a test argument.
- `ast::AST`: The resolved expression tree.

The form resolves its expression tree `ast` once at construction, referencing the underlying
storage of any coefficient grid functions (`VectorElement`). In-place updates via `Rₕ!(fₕ, ...)`
or `values(fₕ) .= ...` are automatically seen by subsequent assemblies with zero heap allocations.

Constant scalar coefficients can be written directly as plain numbers (e.g. `2.5 * innerₕ(fₕ, v)`).
`Ref` is only needed if you want a **dynamic scalar coefficient** that changes across iterations in a loop:
```julia
α = Ref(1.0)
l = form(Wₕ, v -> α * innerₕ(fₕ, v))
# Inside time loop:
α[] = 2.5
assemble!(b, l) # allocates 0 bytes and evaluates with α = 2.5
```
"""
struct LinearForm{D, TestSpace, FType, AST}
    test_space::TestSpace
    f::FType
    ast::AST
end

"""
    test_space(form::LinearForm)

Returns the test space of the linear form.
"""
test_space(form::LinearForm) = form.test_space

# A linear form is a functional on its test space, so what it contracts against is an
# element of that space.
#
# `dot` would take a bare `Vector` of the right length just as happily, and on a composite
# space that is precisely where it goes quietly wrong: the length says nothing about whether
# the blocks line up with the components the form routes its terms to, so a vector assembled
# for one component ordering contracts against another without complaint. Requiring a
# `VectorElement` makes the caller name the space it belongs to.
# Deliberately one argument. `assemble!` and `evaluate!` take an `ast` keyword so a caller
# assembling in a loop can resolve once; this does not, because here it buys nothing worth an
# interface: measured from 2,500 to 1,000,000 degrees of freedom, resolving a form over a
# plain grid function is below the resolution of the clock and the keyword changed the total
# by 0.01%. What it costs is the 160 B of the tree, per call, which is why the loop-shaped
# entry points keep it and the convenience one does not.
#
# The exception is a source carrying an operator, where resolving *computes*: `D₋ₓ(fₕ)` on
# grid data is 349 us and a full-length element at a million degrees of freedom, and the
# keyword would have saved 18.8%. Hoisting it out of the form is the better answer and needs
# no keyword at all —
#
#     dfₕ = D₋ₓ(fₕ)                          # once, and visibly
#     l = form(Wₕ, v -> innerₕ(dfₕ, v))      # the source is plain data again
#
# which also keeps the form live with respect to `dfₕ`'s values, where a cached tree would
# not.
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
    evaluate!(scratch, form::LinearForm, vₕ; ast = resolve_form_ast(form))

Evaluates `form` at `vₕ`, assembling into `scratch` rather than into a fresh vector, and
returns the value.

Use this when the assembled vector is wanted as well as the value — a Newton step needs the
residual and its norm, and assembling once serves both. `scratch` is overwritten on every
call and holds the right-hand side when this returns.

For the value *alone*, call the form instead: `form(vₕ)` fuses the contraction into the
assembly walk and is faster — 1,528 us against 2,143 us at a million degrees of freedom,
because it makes one pass where this writes a full-length vector and then reads it back. It
takes no `ast`, so it resolves on every call and allocates the 160 B of the tree; this is
the one to reach for when a loop must allocate nothing at all.

Pass `ast` as well to resolve the expression once across the loop; resolving is 160 B per
call otherwise. That caches the expression, so the loop must write through the same elements
rather than rebind them — see the note above `assemble!` for what a reused `ast` does
and does not notice.

Returns the value rather than `scratch`, which departs from the rule that a mutating
function with a single destination returns it. The departure is the point: `scratch` is not
the result here, it is the space the result was computed in.

# Examples
```julia
l = form(Wₕ, v -> innerₕ(fₕ, v))
scratch = zeros(ndofs(Wₕ))
ast = resolve_form_ast(l)
for step in 1:nsteps
    Rₕ!(fₕ, source_at(step))          # written through, not rebound
    value = evaluate!(scratch, l, uₕ; ast = ast)
    # `scratch` is the right-hand side for this step, and `value` its contraction
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

Returns the resolved AST stored inside the linear form.
"""
@inline resolve_form_ast(form::LinearForm) = form.ast

@inline _validate_form_expression(::LazyOp{D}, ::Val{D}) where {D} = nothing

@noinline function _validate_form_expression(bad, ::Val{D}) where {D}
    throw(ArgumentError(
        "a linear form's expression has to build an operator over its test argument, and " *
        "this one returned a $(typeof(bad)). Write it as a function of the test argument — " *
        "`v -> innerₕ(fₕ, v)` — rather than as a value."))
end

@noinline function _validate_form_expression(::LazyOp{E}, ::Val{D}) where {E, D}
    throw(ArgumentError(
        "a linear form's expression is $(E)-dimensional and its test space is $(D). The " *
        "operators in the expression have to come from the same space the form is built " *
        "over."))
end

"""
    form(Wₕ, f)

Constructs a `LinearForm` over the test space `Wₕ` using the linear expression `f`.

Construction is cheap: it resolves nothing and measures nothing. What a parallel assembly
needs to partition the grid is read from the AST when it assembles — see
`_colour_strides`.

# Examples
```julia
# 1D linear form: l(v) = (f, v)
l = form(Wh, v -> innerₕ(fh, v))
```
"""
function form(Wₕ, f)
    D = dim(Wₕ)
    raw_ast = f(TestFunction{D}())
    _validate_form_expression(raw_ast, Val(D))
    ast = resolve_ast(raw_ast)
    return LinearForm{D, typeof(Wₕ), typeof(f), typeof(ast)}(Wₕ, f, ast)
end

# ==============================================================================
# Assembly Implementations
# ==============================================================================

# Nothing to do unless labels were named. `dirichlet_conditions` defaults to `nothing`
# rather than to an empty constraint set, because the empty set was built on every call and
# then discarded by exactly this test — 2,080 B per assembly, for an argument that went
# unread. Naming labels without conditions is a usage error and says so.
function apply_dirichlet_conditions!(
        b::AbstractVector, form::LinearForm, dirichlet_conditions, dirichlet_labels)
    dirichlet_labels === nothing && return b

    dirichlet_conditions === nothing && _throw_labels_without_conditions(dirichlet_labels)

    if dirichlet_labels isa Symbol
        dirichlet_bc!(b, test_space(form), dirichlet_conditions, dirichlet_labels)
    elseif dirichlet_labels isa Tuple && !isempty(dirichlet_labels)
        dirichlet_bc!(b, test_space(form), dirichlet_conditions, dirichlet_labels...)
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
    assemble(form::LinearForm; dirichlet_conditions = nothing, dirichlet_labels = nothing)

Assembles the system vector of the `LinearForm`, applying `dirichlet_conditions` on the
regions `dirichlet_labels` names when both are given.

Assembles serially. [`assemble_parallel!`](@ref) is the threaded counterpart, and which of
the two wins depends on the size of the problem. On four threads the parallel sweep runs
1.2x to 2.2x faster from roughly 250,000 degrees of freedom upward, and loses below about
100,000, where spawning the tasks costs more than the whole sweep.

`assemble` does not pick between them. The threshold moves with the thread count, the
dimension and the form, so the choice belongs to the caller rather than to a constant
compiled in here.

For most of this package's history the parallel path lost at *every* size — 76x at 2,500
degrees of freedom, 3.0x at 250,000 — because it gave each thread a full-length buffer and
reduced them afterwards, paying O(n · threads) of memory traffic against the O(n · stencil)
the assembly itself costs. The buffers are gone; see `_colour_strides`.
"""
function assemble(form::LinearForm; dirichlet_conditions = nothing,
        dirichlet_labels = nothing, ast = form.ast)
    _validate_dirichlet_labels(dirichlet_labels)
    space = test_space(form)
    # `values(element(space, T))` rather than `zeros(T, ndofs(space))`: the latter always
    # gives a plain `Vector` regardless of what backend the space was built with, so a form
    # over a sparse-vector or GPU backend silently assembled into a dense CPU array instead.
    # `element` is the one place that already turns a space's `Backend` into a concrete
    # container, so this reuses it rather than re-deriving the same construction here.
    b = values(element(space, _assembled_eltype(ast, space)))
    return assemble!(b, form; ast = ast, dirichlet_conditions = dirichlet_conditions,
        dirichlet_labels = dirichlet_labels)
end

# The element type of the assembled vector is the one the form's own weights have, promoted
# against the space's — not the space's outright.
#
# It used to be `eltype(test_space(form))`, which made a Float64 space able to assemble only
# Float64 right-hand sides, and that is what blocked differentiating an assembled residual.
# Writing a `ForwardDiff.Dual` weight into a Float64 vector met
# `MethodError: no method matching Float64(::Dual)`. It is the same rule `Rₕ` uses and the
# same defect `dirichlet_constraints` had: read the type from the data, not from the space.
#
# Promoted rather than taken outright, so an integer-valued source still assembles into
# Float64 on a Float64 space, while a Dual-valued one gives a Dual over the same,
# undifferentiated, geometry. Read from one stencil evaluation, which is one extra call per
# assembly against inferring a type the compiler may not know.
function _assembled_eltype(ast, space)
    T = eltype(space)
    sp = first_space(space)
    Ωₕ = mesh(sp)
    grid_inds = indices(Ωₕ)
    lin_indices = LinearIndices(grid_inds)

    # An interior point, so a truncated stencil does not decide the type. A restriction can
    # still answer with nothing, in which case the space's type is all there is to go on.
    I = grid_inds[length(grid_inds) ÷ 2 + 1]
    st = local_stencil(ast, sp, I, markers(Ωₕ), lin_indices[I])
    isempty(st) && return T
    return promote_type(T, typeof(last(first(st))))
end

# ==============================================================================
# Helper Cores for Function Barrier Optimization
# ==============================================================================

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

# ==============================================================================
# The partition a parallel assembly walks
# ==============================================================================

"""
	_colour_strides(offsets) -> NTuple{D, Int}

The per-dimension stride separating grid points a parallel assembly may write at the same
time, for an operator reaching `offsets`.

Two points of one colour differ by a multiple of the stride in some dimension, so by at
least `span + 1` there, while each writes a footprint `span` wide about itself. More than a
width apart, the footprints cannot overlap: no two points in a colour ever target the same
row, and the sweep needs no coordination of any kind.

An operator reaching only its own point — `innerₕ(fₕ, v)`, and every form whose test
argument carries no difference — strides by 1 in every dimension, so it has one colour: the
whole grid in a single flat parallel pass.
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

# One colour of `grid_inds`, as a strided sub-grid rather than a materialised list of
# indices. The colouring is a range, so it costs nothing to build — the version this
# replaces binned every index into a vector of vectors — and the writes within a colour
# still run in ascending order.
@inline function _colour_subgrid(grid_inds::CartesianIndices{D}, c::CartesianIndex{D},
        strides::NTuple{D, Int}) where {D}
    return CartesianIndices(ntuple(d -> c[d]:strides[d]:last(axes(grid_inds, d)), D))
end

# The threaded pass over one colour, writing straight into `b`.
#
# `AbstractVector` rather than `Vector`, and no element type named anywhere, so a
# Dual-valued assembly takes this path as readily as a Float64 one. The per-thread buffers
# this replaces were `Vector{Float64}` outright, which is what stopped the parallel path
# differentiating at all — the same defect, read the type from the data and not from the
# space, that `Rₕ` and `dirichlet_constraints` each had.
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

# Every colour in turn. Written as a loop over an explicitly passed grid rather than as a
# higher-order function taking the sweep as a closure, which is not a style choice: the
# closure captured the `sp` of the caller's `for (sp, offset) in leaves`, and a closure over
# a loop variable is boxed, so `local_stencil` became a dynamic call at every grid point of
# every leaf. It cost a composite assembly 7x — measured at 0.08x of serial, against 1.9x
# for the same form on a scalar space, which is how the boxing was found.
function _sweep_parallel!(b::AbstractVector, sp, term::TERM, grid_inds, strides,
        lin_indices, mesh_markers, offset::Int) where {TERM}
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
    _sweep_parallel!(b, space, ast, indices(mesh(space)), strides, lin_indices,
        mesh_markers, 0)
    return b
end

function _assemble_linear_core!(
        b::AbstractVector, space::CompositeGridSpace{N}, ast::AST_TYPE,
        lin_indices, mesh_markers) where {N, AST_TYPE}
    # `leaf_spaces_offsets` rather than a Vector of offsets accumulated with `push!`, for
    # two reasons. It allocates nothing, where the Vector cost 128 B on every call — small,
    # but this is what a time loop calls each step. And it walks the *leaves*, where
    # `space.spaces` walks the top-level components: for a composite whose component is
    # itself composite, `indices(mesh(sp))` covers one leaf's grid while `ndofs(sp)` counts
    # the whole nested block, so the old loop wrote into a fraction of the range it had
    # reserved.
    leaves = leaf_spaces_offsets(space)

    # A form written without component indices means the same integrand in every block, and
    # takes the path that allocates nothing. A form that names components —
    # `innerₕ(uₕ(1), v(1)) + innerₕ(uₕ(2), v(2))` — has to be split, and splitting builds a
    # vector of terms, so the question is asked once here rather than paid for always.
    if !routes_by_component(ast)
        for (sp, offset) in leaves
            _scatter_term!(b, sp, ast, lin_indices, mesh_markers, offset)
        end
        return b
    end

    _route_terms!(b, ast, leaves, lin_indices, mesh_markers)
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

# Walk the sum and send each term to the blocks it belongs to.
#
# Recursing the tree rather than flattening it into a vector of terms first. A flattened
# vector is a `Vector{Any}`, which allocates and makes every term a dynamic read; recursing
# keeps each term concretely typed at its own call, so this is inferable end to end and costs
# nothing — 544 B per assembly became 0.
function _route_terms!(
        b::AbstractVector, op::OperatorAdd, leaves, lin_indices, mesh_markers)
    _route_terms!(b, op.left_op, leaves, lin_indices, mesh_markers)
    _route_terms!(b, op.right_op, leaves, lin_indices, mesh_markers)
    return b
end

function _route_terms!(b::AbstractVector, term::TERM, leaves, lin_indices,
        mesh_markers) where {TERM}
    target = test_component_or_nothing(term)
    _check_component(target, length(leaves))
    for (c, leaf) in enumerate(leaves)
        # a term naming a component goes to that block alone; one naming none goes to every
        # block, so the two spellings can be mixed in a single form
        (target === nothing || target == c) || continue
        _scatter_term!(b, first(leaf), term, lin_indices, mesh_markers, last(leaf))
    end
    return b
end

# ==============================================================================
# Contraction: the same walk, accumulating a number instead of filling a vector
# ==============================================================================

# `l(vₕ)` answers with a scalar, and used to build the whole right-hand side to get it:
# `dot(assemble(form), values(vₕ))` allocated one full-length vector per call to produce one
# number. On a 90,000-point grid that was 721,168 B for 8 bytes of answer, and two vectors
# when the source carried an operator, since resolving evaluates `D₋ₓ(uₕ)` into an element
# of its own.
#
# The vector was never needed. `l(vₕ) = Σᵢ bᵢ vᵢ`, and `bᵢ` is built by accumulating stencil
# weights into row `i`, so the sum can be taken as the walk goes: multiply each weight by
# `v` at the row it would have been written to. Same arithmetic, reassociated, and one pass
# rather than a write pass and a read pass.
#
# `acc` is passed in already typed rather than started at `zero(T)` here, so the accumulator
# type is fixed by the caller and this stays inferable whatever the weights promote to.
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
    # The same split the assembling core makes, for the same reasons.
    leaves = leaf_spaces_offsets(space)

    if !routes_by_component(ast)
        for (sp, offset) in leaves
            acc = _contract_term(sp, ast, lin_indices, mesh_markers, offset, v, acc)
        end
        return acc
    end

    return _route_terms_contract(ast, leaves, lin_indices, mesh_markers, v, acc)
end

# The counterpart of `_scatter_term!`, and a function barrier for the same reason.
function _contract_term(sp, term::TERM, lin_indices, mesh_markers, offset::Int,
        v::AbstractVector, acc::T) where {TERM, T}
    for I in indices(mesh(sp))
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

# The counterpart of `_route_terms!`. Recursive rather than over a flattened vector for the
# same reason, and threading `acc` through the recursion rather than summing the branches
# keeps the accumulator type fixed across the whole walk.
function _route_terms_contract(op::OperatorAdd, leaves, lin_indices, mesh_markers, v, acc)
    acc = _route_terms_contract(op.left_op, leaves, lin_indices, mesh_markers, v, acc)
    return _route_terms_contract(op.right_op, leaves, lin_indices, mesh_markers, v, acc)
end

function _route_terms_contract(term::TERM, leaves, lin_indices, mesh_markers,
        v, acc::T) where {TERM, T}
    target = test_component_or_nothing(term)
    _check_component(target, length(leaves))
    for (c, leaf) in enumerate(leaves)
        (target === nothing || target == c) || continue
        acc = _contract_term(first(leaf), term, lin_indices, mesh_markers, last(leaf),
            v, acc)
    end
    return acc
end

# The threaded counterpart of `_route_terms!`, and deliberately the same shape: term
# outside, grid inside.
#
# The obvious parallel shape is the other order — one pass over the grid, routing every term
# at each point — and it measured 0.08x of the serial sweep at a million degrees of freedom,
# against 1.9x for a comparable form on a scalar space. Routing per point redoes the
# component walk and the leaf search at every point; routing per term hoists both out of the
# grid loop and leaves each sweep a tight, concretely typed one. The order of the two loops
# is the whole of the difference.
#
# Strides come from the term rather than from the whole form, so a term reaching only its own
# point still sweeps in a single colour even when some other term in the same form does not.
function _route_terms_parallel!(b::AbstractVector, op::OperatorAdd, leaves, lin_indices,
        mesh_markers)
    _route_terms_parallel!(b, op.left_op, leaves, lin_indices, mesh_markers)
    _route_terms_parallel!(b, op.right_op, leaves, lin_indices, mesh_markers)
    return b
end

function _route_terms_parallel!(b::AbstractVector, term::TERM, leaves, lin_indices,
        mesh_markers) where {TERM}
    target = test_component_or_nothing(term)
    _check_component(target, length(leaves))
    strides = _colour_strides(stencil_offsets(term))

    for (c, leaf) in enumerate(leaves)
        (target === nothing || target == c) || continue
        sp = first(leaf)
        _sweep_parallel!(
            b, sp, term, indices(mesh(sp)), strides, lin_indices, mesh_markers,
            last(leaf))
    end
    return b
end

# The scatter, behind a function barrier: a term is only concretely typed once it is an
# argument, so without this `local_stencil` would be a dynamic call at every grid point
# rather than once per term.
function _scatter_term!(b::AbstractVector, sp, term::TERM, lin_indices, mesh_markers,
        offset::Int) where {TERM}
    for I in indices(mesh(sp))
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

    # The same split the serial core makes: a form whose terms mean the same thing on every
    # block sweeps each leaf with the whole AST, and one whose terms name components has to
    # be routed term by term.
    if !routes_by_component(ast)
        for (sp, offset) in leaves
            _sweep_parallel!(b, sp, ast, indices(mesh(sp)), strides, lin_indices,
                mesh_markers, offset)
        end
        return b
    end

    _route_terms_parallel!(b, ast, leaves, lin_indices, mesh_markers)
    return b
end

# Not cached on the form on purpose: a `GridFunctionScale` over a thunk has its values read
# when the AST resolves, so resolving eagerly at construction would freeze coefficients a
# caller may still be changing. Resolving per call costs 160 B.

"""
    assemble!(b::AbstractVector, form::LinearForm; dirichlet_conditions = nothing,
              dirichlet_labels = nothing, ast = form.ast) -> b

Refills `b` with the assembled `form` and returns it, allocating nothing (**0 bytes**).

This is the call a time loop wants. [`assemble`](@ref) allocates a fresh vector on every
step; `assemble!` writes into one that already exists, and a vector fill into an existing
buffer measures 0 bytes.

By default `assemble!` uses the pre-resolved `form.ast` stored directly inside the form,
achieving zero heap allocations.

## Live Coefficients
- **Grid functions**: The stored AST retains references to source `VectorElement` storage. Mutating values in-place (`Rₕ!(uₕ, ...)` or `values(uₕ) .= ...`) between steps automatically updates the assembled vector without needing to rebuild the form.
- **Dynamic scalars**: Plain numbers work directly for constant scalars. To update a scalar dynamically across loop iterations, wrap it in a `Ref(val)` (e.g. `α = Ref(1.0); l = form(Wₕ, v -> α * innerₕ(uₕ, v))`). Mutating `α[] = new_val` evaluates live during assembly with 0 allocations.

# Arguments

  - `b`: the vector to refill. Its length must be `ndofs(test_space(form))`.
  - `form`: the linear form to assemble.

# Keywords

  - `dirichlet_conditions`, `dirichlet_labels`: boundary values to impose after assembling,
    and the labels to impose them on. Both default to `nothing`, which imposes nothing.
  - `ast`: an optional custom AST override (defaults to `form.ast`).

See also [`assemble`](@ref), [`assemble_parallel!`](@ref) for the threaded sweep, and
[`evaluate!`](@ref) when the contraction is wanted alongside the vector.
"""
function assemble!(b::AbstractVector, form::LinearForm{D, TestSpace, FType, AST};
        dirichlet_conditions = nothing,
        dirichlet_labels = nothing,
        ast = form.ast) where {D, TestSpace, FType, AST}
    _validate_dirichlet_labels(dirichlet_labels)
    fill!(b, zero(eltype(b)))
    space = form.test_space
    Ωₕ = mesh(space)
    mesh_markers = markers(Ωₕ)
    _validate_term_markers(ast, mesh_markers, "the form's space")
    lin_indices = LinearIndices(indices(Ωₕ))

    _assemble_linear_core!(b, space, ast, lin_indices, mesh_markers)

    apply_dirichlet_conditions!(b, form, dirichlet_conditions, dirichlet_labels)
    return b
end

"""
    assemble_parallel!(b, form::LinearForm, ast = form.ast) -> b

Refills `b` with the assembled `form` across threads, and returns it.
"""
function assemble_parallel!(b::AbstractVector,
        form::LinearForm{D, TestSpace, FType, AST},
        ast = form.ast) where {D, TestSpace, FType, AST}
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
