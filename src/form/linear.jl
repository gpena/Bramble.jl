
# ==============================================================================
# Struct Definitions
# ==============================================================================

# `ParallelWorkspace` lives in form/parallel_workspace.jl, included before both
# assembly files, so this file no longer depends on bilinear.jl for it.
"""
    LinearForm{D,TestSpace,ExprType,FType}

Represents a linear form defined over a test space.

# Fields
- `test_space::TestSpace`: The space for the test function.
- `ast::ExprType`: The symbolic expression AST representation of the form.
- `f::FType`: The user-defined lambda function representing the form.
- `workspace::ParallelWorkspace{D}`: Preallocated coordinate partitions for lock-free parallel assembly.
"""
struct LinearForm{D, TestSpace, ExprType <: LazyOp{D}, FType}
    test_space::TestSpace
    ast::ExprType
    f::FType
    workspace::ParallelWorkspace{D}
end

"""
    test_space(form::LinearForm)

Returns the test space of the linear form.
"""
test_space(form::LinearForm) = form.test_space

@inline (form::LinearForm)(v) = dot(assemble(form), v)

"""
    resolve_form_ast(form::LinearForm)

Fully resolves grid coefficient functions and scales inside the linear form's AST.
"""
@inline resolve_form_ast(form::LinearForm{D, TestSpace, ExprType,
    FType}) where {D, TestSpace, ExprType, FType} = resolve_ast(form.f(TestFunction{D}()))

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
    ast = f(TestFunction{D}())

    # No colouring and no buffers built here, on purpose. Both used to be: every `form`
    # call evaluated a sample stencil, binned the whole grid into a
    # `Vector{Vector{CartesianIndex{D}}}` one `push!` at a time, and allocated a
    # full-length `Float64` buffer per thread — 64 MB at a million degrees of freedom on
    # eight threads, paid whether or not the caller ever assembled in parallel. The bins
    # were then never read by anything, and the buffers are what made the parallel path
    # slower than the serial one at every size.
    #
    # The partition is a property of the AST and the grid, both of which assembly has in
    # hand, so it is computed there instead and costs nothing here.
    workspace = ParallelWorkspace{D}(Vector{CartesianIndex{D}}[])

    return LinearForm{D, typeof(Wₕ), typeof(ast), typeof(f)}(Wₕ, ast, f, workspace)
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
        dirichlet_labels = nothing)
    ast = resolve_form_ast(form)
    space = test_space(form)
    b = zeros(_assembled_eltype(ast, space), ndofs(space))
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
        b::Vector, space, ast::AST_TYPE, lin_indices, mesh_markers) where {AST_TYPE}
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

function _assemble_linear_core!(b::Vector, space::CompositeGridSpace{N}, ast::AST_TYPE,
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

# Walk the sum and send each term to the blocks it belongs to.
#
# Recursing the tree rather than calling `flatten_sum` and iterating the result: that answers
# with a `Vector{Any}`, which allocates and makes every term a dynamic read. Recursing keeps
# each term concretely typed at its own call, so this is inferable end to end and costs
# nothing — 544 B per assembly became 0.
function _route_terms!(b::Vector, op::OperatorAdd, leaves, lin_indices, mesh_markers)
    _route_terms!(b, op.left_op, leaves, lin_indices, mesh_markers)
    _route_terms!(b, op.right_op, leaves, lin_indices, mesh_markers)
    return b
end

function _route_terms!(b::Vector, term::TERM, leaves, lin_indices,
        mesh_markers) where {TERM}
    target = test_component_or_nothing(term)
    for (c, leaf) in enumerate(leaves)
        # a term naming a component goes to that block alone; one naming none goes to every
        # block, so the two spellings can be mixed in a single form
        (target === nothing || target == c) || continue
        _scatter_term!(b, first(leaf), term, lin_indices, mesh_markers, last(leaf))
    end
    return b
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

# The threaded core's counterpart of `_route_terms!`: one point, every term, routed to the
# blocks each belongs to. Recursive for the same reason — no vector, nothing dynamic.
function _route_point!(
        dest::AbstractVector, op::OperatorAdd, leaves, I, lin_idx::Int, lin_indices,
        mesh_markers)
    _route_point!(dest, op.left_op, leaves, I, lin_idx, lin_indices, mesh_markers)
    _route_point!(dest, op.right_op, leaves, I, lin_idx, lin_indices, mesh_markers)
    return dest
end

function _route_point!(
        dest::AbstractVector, term::TERM, leaves, I, lin_idx::Int, lin_indices,
        mesh_markers) where {TERM}
    target = test_component_or_nothing(term)
    for (c, leaf) in enumerate(leaves)
        (target === nothing || target == c) || continue
        _scatter_point!(dest, term, first(leaf), I, lin_idx, lin_indices, mesh_markers,
            last(leaf))
    end
    return dest
end

# One point's contribution, behind a function barrier for the same reason as
# `_scatter_term!`: a term read out of a `Vector{Any}` is only concretely typed once it is
# an argument.
@inline function _scatter_point!(
        dest::AbstractVector, term::TERM, sp, I, lin_idx::Int, lin_indices,
        mesh_markers, offset::Int) where {TERM}
    stencil = local_stencil(term, sp, I, mesh_markers, lin_idx)
    for (off_v, weight) in stencil
        Iv = I + CartesianIndex(off_v)
        if checkbounds(Bool, lin_indices, Iv)
            @inbounds dest[lin_indices[Iv] + offset] += weight
        end
    end
    return dest
end

# The scatter, behind a function barrier. `flatten_sum` answers with a `Vector{Any}`, so a
# term read out of it is only concretely typed once it is an argument: without this
# `local_stencil` would be a dynamic call at every grid point rather than once per term.
function _scatter_term!(b::Vector, sp, term::TERM, lin_indices, mesh_markers,
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

# `ast` is a keyword so a caller assembling repeatedly can resolve once and hand it in;
# resolving is 160 B per call otherwise. It is not cached on the form on purpose: a
# `GridFunctionScale` over a thunk has its values read when the AST resolves, so resolving
# eagerly at construction would freeze coefficients a caller may still be changing.
function assemble!(b::Vector, form::LinearForm{D, TestSpace, ExprType, FType};
        dirichlet_conditions = nothing,
        dirichlet_labels = nothing,
        ast = resolve_form_ast(form)) where {D, TestSpace, ExprType, FType}
    _validate_dirichlet_labels(dirichlet_labels)
    fill!(b, zero(eltype(b)))
    space = form.test_space
    Ωₕ = mesh(space)
    mesh_markers = markers(Ωₕ)
    lin_indices = LinearIndices(indices(Ωₕ))

    _assemble_linear_core!(b, space, ast, lin_indices, mesh_markers)

    apply_dirichlet_conditions!(b, form, dirichlet_conditions, dirichlet_labels)
    return b
end

"""
    assemble_parallel!(b, form::LinearForm, ast = resolve_form_ast(form))

Assembles the `LinearForm` into `b` across threads, writing directly and without
coordination by taking one colour of the grid at a time.

See `_colour_strides` for why that is safe, and [`assemble`](@ref) for when it is
worth choosing over the serial sweep.
"""
function assemble_parallel!(b::AbstractVector,
        form::LinearForm{D, TestSpace, ExprType, FType},
        ast = resolve_form_ast(form)) where {D, TestSpace, ExprType, FType}
    space = form.test_space
    Ωₕ = mesh(space)
    mesh_markers = markers(Ωₕ)
    lin_indices = LinearIndices(indices(Ωₕ))

    # The sweep accumulates, where the version this replaces overwrote `b` from the buffer
    # reduction, so `b` has to start at zero.
    fill!(b, zero(eltype(b)))
    _assemble_linear_parallel_core!(b, space, ast, lin_indices, mesh_markers)

    return b
end

# update_ast_grid_coefficients! has been deprecated and deleted.
