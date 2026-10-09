# block_extract.jl
# Reading which block of a coupled form a term belongs to.
#
# Determines component routing for coupled assembly: trial and test component indices
# inspected via `trial_component_or_nothing` and `test_component_or_nothing`, which
# `block_of` converts to a block coordinate `(trial, test)` or validates. Routing by
# leaf index supports arbitrary composite function spaces.

"""
    test_component_or_nothing(op) -> Union{Int, Nothing}

The component `op` is written against, or `nothing` when it names none.

Used when assembling composite right-hand sides, where a term built from an indexed test
function belongs to one block while a term built from an unindexed one belongs to all of them.
Written as a query rather than catching an exception because it is evaluated once per
term per assembly in time-stepping loops.
"""
test_component_or_nothing(op::IndexedTestFunction) = op.component_idx
# Every node wrapping one operand answers what that operand answers. One method instead of
# thirteen registrations (gpena/Bramble.jl#52); see [`UnaryWrapper`](@ref).
test_component_or_nothing(op::UnaryWrapper) = test_component_or_nothing(op.inner_op)
test_component_or_nothing(op::LinearProduct) = test_component_or_nothing(op.right_op)
test_component_or_nothing(op::BilinearProduct) = test_component_or_nothing(op.right_op)
# A sum inside one inner product, `innerₕ(uₕ, v + 2 * D₋ₓ(v) - Mₓ(v))`, is still one
# term of the form, and every test leaf in it names the same component or none. So the
# component of a sum is the component its sides agree on.
#
# Without this check the fallback returned `nothing` and the term broadcast to every block.
function test_component_or_nothing(op::OperatorAdd)
    l = test_component_or_nothing(op.left_op)
    r = test_component_or_nothing(op.right_op)
    l === r && return l
    return _throw_mixed_components(l, r)
end

# Sides naming different components cannot be one term of one block:
# `innerₕ(uₕ, v(1) + v(2))` is ill-formed, so an error is raised.
@noinline function _throw_mixed_components(l, r)
    throw(
        ArgumentError(
        "the two sides of a sum inside one inner product name different components " *
        "($l and $r). Each inner product belongs to one component: write the sum of " *
        "products instead, innerₕ(u, v(1)) + innerₕ(u, v(2)).",
    ),
    )
end

test_component_or_nothing(::Any) = nothing

"""
    trial_component_or_nothing(op) -> Union{Int, Nothing}

The trial component `op` is written against, or `nothing` when it names none.

The trial-side mirror of [`test_component_or_nothing`](@ref), and needed for the same
reason one step further on: a bilinear form's term belongs to a *block*, which takes a
component from each side. A term naming neither is the same integrand in every diagonal
block; a term naming both is one block; and a term naming one but not the other is not
something the mathematics can express, so it is an error rather than a guess.
"""
trial_component_or_nothing(op::IndexedTrialFunction) = op.component_idx
trial_component_or_nothing(op::UnaryWrapper) = trial_component_or_nothing(op.inner_op)
trial_component_or_nothing(op::BilinearProduct) = trial_component_or_nothing(op.left_op)

function trial_component_or_nothing(op::OperatorAdd)
    l = trial_component_or_nothing(op.left_op)
    r = trial_component_or_nothing(op.right_op)
    l === r && return l
    return _throw_mixed_components(l, r)
end

trial_component_or_nothing(::Any) = nothing

"""
    block_of(term, nblocks_trial, nblocks_test) -> Union{Nothing, Tuple{Int, Int}}

The `(trial, test)` block `term` belongs to, or `nothing` when it belongs to every diagonal
block.

A term naming neither side is the same integrand on each block, which for a matrix means the
diagonal: `Σᵢ innerₕ(uᵢ, vᵢ)` is block diagonal, not full. A term naming both is one block.
A term naming one and not the other is refused: `innerₕ(u(1), v)` is not something written
in a variational formulation, and reading it as a whole row or column of blocks would be a
guess about what was meant.
"""
function block_of(term, nblocks_trial::Int, nblocks_test::Int)
    tc = trial_component_or_nothing(term)
    sc = test_component_or_nothing(term)

    tc === nothing && sc === nothing && return nothing
    # An unnamed side on a single-leaf space can only mean its one leaf.
    tc === nothing && sc !== nothing && nblocks_trial == 1 && (tc = 1)
    sc === nothing && tc !== nothing && nblocks_test == 1 && (sc = 1)
    (tc === nothing || sc === nothing) && _throw_half_named_block(tc, sc)

    1 <= tc <= nblocks_trial || _throw_block_out_of_range("trial", tc, nblocks_trial)
    1 <= sc <= nblocks_test || _throw_block_out_of_range("test", sc, nblocks_test)
    return (tc, sc)
end

@noinline function _throw_half_named_block(tc, sc)
    named, missing_side = tc === nothing ? ("test", "trial") : ("trial", "test")
    throw(
        ArgumentError(
        "a term of this form names its $named component but not its $missing_side one. " *
        "A block of a bilinear form takes a component from each side: write " *
        "innerₕ(u(i), v(j)) for one block, or innerₕ(u, v) for the same integrand on " *
        "every diagonal block.",
    ),
    )
end

@noinline function _throw_block_out_of_range(side::String, c::Int, n::Int)
    throw(
        ArgumentError(
        "a term of this form names $side component $c, and that side has $n blocks. " *
        "Components are numbered 1 to $n; a term written for a wider space contributes " *
        "nothing here, which is why this is an error rather than an empty block.",
    ),
    )
end

"""
    Block{TrialLeaf, TestLeaf}

One leaf-space pair's rectangle within a composite system matrix: the concrete trial and
test leaf spaces a term couples, and the row/column offset each contributes to that
rectangle's position in the assembled matrix.

Matrix rows are indexed by the test function (see `bilinear.jl`'s file header), so
`row_offset` always comes from `test_leaf`'s offset in `leaf_spaces_offsets`, and
`col_offset` from `trial_leaf`'s. That asymmetry used to be carried by convention across
six call sites, each unpacking a bare `(tc, sc)` tuple into `first`/`last` calls in the
right order -- one of which got it backwards (gpena/Bramble.jl#48). Naming it here means a
caller reads `blk.row_offset`/`blk.col_offset` off the type instead of re-deriving which
positional element means which.
"""
struct Block{TrialLeaf, TestLeaf}
    trial_leaf::TrialLeaf
    test_leaf::TestLeaf
    row_offset::Int
    col_offset::Int
end

@inline _block_from_indices(trial_leaves, test_leaves, tc::Int, sc::Int) = Block(
    first(trial_leaves[tc]),
    first(test_leaves[sc]),
    last(test_leaves[sc]),
    last(trial_leaves[tc])
)

@inline function _diagonal_blocks(trial_leaves::Tuple, test_leaves::Tuple)
    n = min(length(trial_leaves), length(test_leaves))
    return ntuple(c -> _block_from_indices(trial_leaves, test_leaves, c, c), n)
end

"""
    blocks(term, trial_leaves, test_leaves) -> Tuple{Vararg{Block}}

Every [`Block`](@ref) `term` must be assembled into.

A term naming both sides (via [`block_of`](@ref)) resolves to the one `Block` it names. A
term naming neither is the same integrand on every diagonal block, so it resolves to one
`Block` per diagonal leaf pair. `trial_leaves`/`test_leaves` are `leaf_spaces_offsets`
results.
"""
@inline function blocks(term, trial_leaves::Tuple, test_leaves::Tuple)
    blk = block_of(term, length(trial_leaves), length(test_leaves))
    blk === nothing && return _diagonal_blocks(trial_leaves, test_leaves)
    tc, sc = blk
    return (_block_from_indices(trial_leaves, test_leaves, tc, sc),)
end

"""
    _collect_region_labels(op) -> NTuple{N, Symbol}

Every marker label a `RegionRestriction` anywhere in `op` names: from `restrict_to` calls
written directly, or from the `markers = (...)` keyword on `innerₕ`/`inner₊` and friends.
Flattened into one tuple; a term naming several restrictions (nested, or one on each side of
a product) reports all of them, since every one has to exist on every leaf the term reaches
for assembly to mean what it says.

Recurses the same way [`trial_component_or_nothing`](@ref)/[`test_component_or_nothing`](@ref)
do, so a marker nested behind any operator those already see through is found here too.
"""
function _collect_region_labels(op::RegionRestriction)
    return (_region_labels(op.region)..., _collect_region_labels(op.inner_op)...)
end

_region_labels(region::Symbol) = (region,)
_region_labels(region::NTuple{N, Symbol}) where {N} = region

# `RegionRestriction`'s own method above wins on specificity, so it is untouched by this
# collapse; only `InterpolationNode`'s separate method (operators/interpolation.jl)
# becomes redundant, since it recursed the same way.
_collect_region_labels(op::UnaryWrapper) = _collect_region_labels(op.inner_op)

function _collect_region_labels(op::Union{BilinearProduct, LinearProduct, OperatorAdd})
    return (_collect_region_labels(op.left_op)..., _collect_region_labels(op.right_op)...)
end

_collect_region_labels(op) = ()

"""
    _validate_term_markers(term, mesh_markers, context::String)

Throws if `term` names, via `restrict_to` or `markers = (...)`, a label that does not exist
in `mesh_markers` (the mesh a term is about to be scattered against). Checked once, while the
sparsity pattern is built (`allocate_system_matrix`/`_coord_walk!`), rather than left to
`RegionRestriction`'s own `local_stencil`: that answers `false` for a missing key the same way
it does for "not marked", so a typo'd or leaf-missing label would otherwise assemble to a
silent all-zero contribution instead of failing loudly.
"""
function _validate_term_markers(term, mesh_markers, context::String)
    for label in _collect_region_labels(term)
        haskey(mesh_markers, label) || _throw_marker_not_on_space(label, mesh_markers, context)
    end
    return nothing
end

@noinline function _throw_marker_not_on_space(label::Symbol, mesh_markers, context::String)
    known = sort!(collect(keys(mesh_markers)))
    avail = isempty(known) ? "(none)" : join(map(s -> ":$s", known), ", ")
    throw(
        ArgumentError(
        "the marker :$label is not defined on $context. Available marker labels on this space are: $avail. " *
        "A marker named in restrict_to or markers = (...) must exist on every space a term reaches; " *
        "if it is only defined on some of a composite space's leaves, write the term per component instead, " *
        "one innerₕ(u(i), v(i)) per leaf with that leaf's own markers, rather than one term " *
        "naming a marker not every leaf it reaches has.",
    ),
    )
end

"""
    _bind_walk(term, Ωₕ::AbstractMeshType) -> Tuple{bound_term, Union{AbstractMatrix{UInt64}, Nothing}}
    _bind_walk(term, sp::ScalarGridSpace) -> Tuple{bound_term, Union{AbstractMatrix{UInt64}, Nothing}}

`term` with every `RegionRestriction` it holds bound to `Ωₕ`'s marker ids
([`_bind_marker_ids`](@ref)), and the marker table the walk over `Ωₕ` passes to
`local_stencil`: `Ωₕ`'s word matrix, any `AbstractMatrix{UInt64}` ([`_marker_words`](@ref)),
or `nothing` when `term` restricts nothing, decided from its type.

Called at every walk entry, once per walk, so the ids are read from `Ωₕ`'s label table as it
is when the walk runs: `assemble` on a form built before a `markers!` call binds against the
new table, and a label removed since throws the `ArgumentError` of
[`_validate_term_markers`](@ref). Editing `markers(Ωₕ)` in place is unsupported, and `markers!`
changes the labels. A label written in place has no id and throws too
(`_throw_marker_not_bound`). Given the walked leaf `sp` instead, it binds against
`mesh(sp)` after checking `sp`'s weights ([`weights`](@ref)), since its stencils read the
weights unchecked.
"""
@inline function _bind_walk(term, Ωₕ::AbstractMeshType)
    isempty(_collect_region_labels(term)) && return term, nothing
    return _bind_marker_ids(term, Ωₕ), _marker_words(Ωₕ)
end

# The form every walk entry calls, with the leaf it walks: `sp`'s weights are checked
# against its mesh here, once per walk, and the stencil then reads them unchecked at every
# point (`_stored_weights`, gpena/Bramble.jl#437). A stale `sp` throws `weights`'s error;
# a bilinear walk's trial leaf is checked in `_check_block_meshes` (#466).
@inline function _bind_walk(term, sp::ScalarGridSpace)
    weights(sp)
    return _bind_walk(term, mesh(sp))
end

"""
    _bind_marker_ids(op, Ωₕ::AbstractMeshType) -> LazyOp

`op` rebuilt with each `RegionRestriction` region replaced by the column of its label in
`Ωₕ`'s word matrix ([`_marker_id`](@ref)): a `Symbol` becomes an `Int`, a tuple of them an
`NTuple{N, Int}`. Recurses exactly where [`_collect_region_labels`](@ref) does, so every
label validated is bound. Labels are read as stored, no boundary alias resolved.
"""
_bind_marker_ids(op, Ωₕ::AbstractMeshType) = op

function _bind_marker_ids(op::RegionRestriction{D}, Ωₕ::AbstractMeshType) where {D}
    inner = _bind_marker_ids(op.inner_op, Ωₕ)
    region = _region_ids(op.region, Ωₕ)
    return RegionRestriction{D, typeof(region), typeof(inner)}(region, inner)
end

function _bind_marker_ids(op::UnaryWrapper, Ωₕ::AbstractMeshType)
    return _rewrap_inner(op, _bind_marker_ids(op.inner_op, Ωₕ))
end

function _bind_marker_ids(op::OperatorAdd{D}, Ωₕ::AbstractMeshType) where {D}
    left = _bind_marker_ids(op.left_op, Ωₕ)
    right = _bind_marker_ids(op.right_op, Ωₕ)
    return OperatorAdd{D, typeof(left), typeof(right)}(left, right)
end

function _bind_marker_ids(op::BilinearProduct{D, W}, Ωₕ::AbstractMeshType) where {D, W}
    left = _bind_marker_ids(op.left_op, Ωₕ)
    right = _bind_marker_ids(op.right_op, Ωₕ)
    return BilinearProduct{D, W, typeof(left), typeof(right)}(left, right)
end

function _bind_marker_ids(op::LinearProduct{D, W}, Ωₕ::AbstractMeshType) where {D, W}
    left = _bind_marker_ids(op.left_op, Ωₕ)
    right = _bind_marker_ids(op.right_op, Ωₕ)
    return LinearProduct{D, W, typeof(left), typeof(right)}(left, right)
end

@inline function _region_ids(label::Symbol, Ωₕ::AbstractMeshType)
    id = get(_marker_ids(Ωₕ), label, 0)
    id == 0 && _throw_marker_not_bound(label, Ωₕ)
    return id
end

# A label missing from the id table is either absent from the mesh (removed since the form
# was built) or present only in the `markers(Ωₕ)` dictionary, written there in place. The
# dictionary is a read-only view: only `markers!`/`set_markers!` rebuild the table.
@noinline function _throw_marker_not_bound(label::Symbol, Ωₕ::AbstractMeshType)
    haskey(markers(Ωₕ), label) || _throw_marker_not_on_space(
        label, markers(Ωₕ), "the space being assembled")
    throw(
        ArgumentError(
        "the marker :$label was added to markers(Ωₕ) in place, so assembly cannot see it. " *
        "markers(Ωₕ) and index_in_marker are read-only views of the mesh's labels; " *
        "add or change a label with markers!(Ωₕ, ...) or set_markers!(Ωₕ, ...), which " *
        "rebuild the marker state assembly reads.",
    ),
    )
end

@inline _region_ids(labels::NTuple{N, Symbol}, Ωₕ::AbstractMeshType) where {N} = map(
    label -> _region_ids(label, Ωₕ), labels)

# A `UnaryWrapper` rebuilt around `inner`: the type parameter its `inner_op` field is declared
# with is replaced by `typeof(inner)`, every other field and parameter kept. Generated from
# the type alone, so one method serves every wrapper, and an `inner` of the operand's own
# type returns `op` itself.
@generated function _rewrap_inner(op::T, inner::I) where {T, I}
    fieldtype(T, :inner_op) === I && return :op
    wrapper = Base.typename(T).wrapper
    decl = fieldtype(Base.unwrap_unionall(wrapper), :inner_op)
    params = Any[T.parameters...]
    params[findfirst(p -> p === decl, Base.unwrap_unionall(wrapper).parameters)] = I
    args = map(fieldnames(T)) do f
        f === :inner_op ? :inner : :(getfield(op, $(QuoteNode(f))))
    end
    return :($(wrapper{params...})($(args...)))
end
