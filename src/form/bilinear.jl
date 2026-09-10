# --- Struct definitions ----------------------------------------------------------- #

# Every scattered contribution's nzval position, for one term routed into one block: the
# same shape `add_to_sparse!` used to re-derive by searching on every call (gpena/Bramble.jl#26).
# `point_ptr[lin_idx]:point_ptr[lin_idx + 1] - 1` is the slice of `positions` holding the
# positions for grid point `lin_idx`'s own (guard-passing, non-deduplicated) stencil entries,
# in the same order a scatter walk visits them -- addressed per point rather than by a shared
# running counter so a future caller could read it without a race even if the walk over grid
# points were threaded (today's cached path is serial-only; see below).
const NzvalSegment = Tuple{Vector{Int},Vector{Int}}

# One `BilinearForm`'s nzval-position cache: valid only for the exact matrix object last
# assembled into (`A === cache.A`), one `NzvalSegment` per (term, block) the serial assembly
# walk visits, in visitation order. A companion *value*, not a type parameter of
# `BilinearForm` -- so it can be filled in lazily, on the first `assemble!` call, without the
# form itself needing to be mutable or its type to depend on whether a cache exists yet.
mutable struct _AssemblyCache
    A::Union{Nothing,SparseMatrixCSC}
    ast::Any
    segments::Vector{NzvalSegment}
end

# Every fresh `BilinearForm` starts pointing at this one, shared, empty vector rather than
# allocating its own: it is never mutated in place (a cache miss *replaces* `cache.segments`
# wholesale, see `_assemble_bilinear_core_cached!`, rather than `empty!`ing whatever it
# currently references), so sharing it across every not-yet-assembled form is safe. Keeps
# `form(Wₕ, Vₕ, f)` itself allocation-free: only the `_AssemblyCache` wrapper is a genuine
# per-form cost (one allocation, since it is a `mutable struct` and therefore always
# heap-boxed), not a second one for an empty vector nothing has scattered into yet.
const _NO_SEGMENTS = NzvalSegment[]

_AssemblyCache() = _AssemblyCache(nothing, nothing, _NO_SEGMENTS)

"""
    BilinearForm{D, TrialSpace, TestSpace, AST}

Represents a bilinear form defined over a trial space and test space.

# Arguments
- `trial_space::TrialSpace`: Space for the trial function.
- `test_space::TestSpace`: Space for the test function.
- `ast::AST`: Resolved expression tree.

The form resolves its expression tree `ast` once at construction, referencing the underlying
storage of any coefficient grid functions (`VectorElement`). In-place updates via `Rₕ!(cₕ, ...)`
or `parent(cₕ) .= ...` are automatically seen by subsequent assemblies with zero heap allocations.
The expression itself is not kept: downstream routines evaluate the resolved AST directly.

Constant scalar coefficients can be written directly as numbers (e.g. `2.0 * innerₕ(D₋ₓ(u), D₋ₓ(v))`).
`Ref` is only needed if a dynamic scalar coefficient changes across loop iterations:
```julia
β = Ref(1.0)
a = form(Wₕ, Wₕ, (u, v) -> innerₕ(β * D₋ₓ(u), D₋ₓ(v)))
# Inside time loop:
β[] = 3.0
assemble!(A, a) # zero allocations, evaluates with β = 3.0
```
"""
struct BilinearForm{D,TrialSpace,TestSpace,AST}
    trial_space::TrialSpace
    test_space::TestSpace
    ast::AST
    cache::_AssemblyCache
end

"""
    trial_space(form::BilinearForm)

Return the trial space of the bilinear form.
"""
trial_space(form::BilinearForm) = form.trial_space

"""
    test_space(form::BilinearForm)

Return the test space of the bilinear form.
"""
test_space(form::BilinearForm) = form.test_space

# `a(uₕ, vₕ) = vᵀ A u`. Assembles a whole matrix per call: intended for testing/convenience.
@inline (form::BilinearForm)(u, v) = dot(v, assemble(form) * u)

"""
    resolve_form_ast(form::BilinearForm)

Return the resolved AST stored inside the bilinear form.
"""
@inline resolve_form_ast(form::BilinearForm) = form.ast

"""
    form(Wₕ, Vₕ, f) -> BilinearForm

Construct a `BilinearForm` over the trial space `Wₕ` and the test space `Vₕ` from the
bilinear expression `f` (a function of trial and test arguments `(u, v)`).

# Examples
```julia
# a(u, v) = (∇₋ₕu, ∇₋ₕv)₊
a = form(Wₕ, Wₕ, (u, v) -> inner₊ₓ(D₋ₓ(u), D₋ₓ(v)))

# a coupled system, one term per block
a = form(Vₕ, Vₕ, (u, v) -> inner₊ₓ(D₋ₓ(u(1)), D₋ₓ(v(1))) + innerₕ(u(2), v(1)))
```
"""
function form(Wₕ, Vₕ, f)
    D = dim(Wₕ)
    raw_ast = f(TrialFunction{D}(), TestFunction{D}())
    _validate_form_expression(raw_ast, Val(D))
    ast = resolve_ast(raw_ast)
    return BilinearForm{D,typeof(Wₕ),typeof(Vₕ),typeof(ast)}(Wₕ, Vₕ, ast, _AssemblyCache())
end

# --- Utility helpers -------------------------------------------------------------- #

# The nzval index storing (row, col) in A, or 0 if it names no stored entry. A linear scan
# of the column when it holds few entries, a binary search otherwise -- both rely on
# `SparseMatrixCSC`'s own invariant that `rowval` is sorted within each column.
@inline function _find_nzval_position(A::SparseMatrixCSC, row::Int, col::Int)
    p1 = A.colptr[col]
    p2 = A.colptr[col + 1] - 1

    if (p2 - p1) < 32
        idx = p1
        @inbounds while idx <= p2
            A.rowval[idx] == row && return idx
            idx += 1
        end
    else
        lo = p1
        hi = p2
        @inbounds while lo <= hi
            mid = (lo + hi) >>> 1
            mid_row = A.rowval[mid]
            if mid_row < row
                lo = mid + 1
            elseif mid_row > row
                hi = mid - 1
            else
                return mid
            end
        end
    end
    return 0
end

@inline function add_to_sparse!(A::SparseMatrixCSC, row::Int, col::Int, val::Number)
    pos = _find_nzval_position(A, row, col)
    pos == 0 && return nothing
    @inbounds A.nzval[pos] += val
    return nothing
end

# Dispatches on `::Type{T}` to ensure concrete vector return type.
@inline _zeros_of(::Type{T}, n::Int) where {T} = zeros(T, n)

# Whether an earlier entry of this stencil already named the same pair of offsets.
@inline function _offsets_seen_before(stencil, k::Int, off_u, off_v)
    @inbounds for l in 1:(k - 1)
        stencil[l][1] == off_u && stencil[l][2] == off_v && return true
    end
    return false
end

# Which column of the trial block a stencil entry's trial slot names, or `0` when it names
# none of them.
#
# Ordinary offsets are bounds-checked and dropped on boundaries. An interpolation entry
# (`AbsoluteColumn`) names a source column directly.
@inline function _trial_column(lin_indices, I::CartesianIndex, off_u)
    Iu = I + CartesianIndex(off_u)
    return checkbounds(Bool, lin_indices, Iu) ? lin_indices[Iu] : 0
end

@inline _trial_column(lin_indices, I::CartesianIndex, off_u::AbsoluteColumn) = off_u.col

# --- one traversal, pluggable sinks (gpena/Bramble.jl#50) -------------------------- #

#=
Five sweeps used to re-derive this same walk: `mesh` -> `markers` -> `LinearIndices`, loop
the grid, evaluate `local_stencil`, decide each entry's (row, col), guard it, act. Two of
them recorded the sparsity pattern and three wrote values into it, and the property that
matters -- *every (row, col) the scatter touches is present in the pattern* -- held only
because two independently written traversals agreed. `add_to_sparse!` has no else branch,
so an entry the pattern lacks is dropped rather than raised.

The walk is now written once. A sink says what to do per entry, and declares by dispatch
whether it wants the de-duplication the pattern passes need and the value passes must not
have (repeated offsets accumulate).

Every rule that used to live in five places lives here: which entries are dropped
(out-of-range rows, and `_trial_column` answering 0), that `off_u` may be an
`AbsoluteColumn` while `off_v` never is, and that the fifth argument to `local_stencil` is
the *leaf's* linear index -- the wrong leaf reads the wrong `SourceVector`, silently.
=#

"""
    _sink_dedups(sink) -> Bool

Whether `sink` wants repeated `(off_u, off_v)` pairs within one point's stencil collapsed
to a single entry.

The sparsity pattern wants each `(row, col)` once however many stencil entries name it, and
a value sink wants every one of them, because repeated entries accumulate. Getting this
backwards gives a wrong matrix rather than an error, so it is answered by dispatch on the
sink type: the default is `false`, and the branch folds away at compile time, leaving the
value sweeps with no de-duplication scan at all.

See also: [`visit_bilinear_stencil`](@ref), [`PatternSink`](@ref).
"""
@inline _sink_dedups(::Any) = false

"""
    _sink_point!(sink, lin_idx::Int) -> Nothing

Announce grid point `lin_idx` to `sink` before any of that point's entries.

The default does nothing. [`RecordSink`](@ref) uses it to open that point's slice of the
position list, and [`ReplaySink`](@ref) to seek to it, which is what lets a replay stay
correct regardless of the order grid points are visited in.

See also: [`visit_bilinear_stencil`](@ref).
"""
@inline _sink_point!(::Any, ::Int) = nothing

"""
    _entry_target(lin_indices, I::CartesianIndex, off_u, off_v,
                  row_offset::Int, col_offset::Int) -> Tuple{Int,Int}

The matrix position a stencil entry writes to, or `(0, 0)` when it writes nowhere.

The row comes from the test offset `off_v` and is dropped when it leaves the grid. The
column comes from [`_trial_column`](@ref), which answers `0` for a trial offset outside the
grid and reads an [`AbsoluteColumn`](@ref) directly, since an interpolation entry names its
source column rather than an offset from `I`. Both are then shifted into the block by
`row_offset` and `col_offset`.

`(0, 0)` is a sentinel rather than `nothing` so the return type stays concrete on the
assembly hot path.

# Returns
- `Tuple{Int,Int}`: The `(row, col)` to write, or `(0, 0)` to skip the entry.
"""
@inline function _entry_target(
    lin_indices, I::CartesianIndex, off_u, off_v, row_offset::Int, col_offset::Int
)
    Iv = I + CartesianIndex(off_v)
    checkbounds(Bool, lin_indices, Iv) || return (0, 0)
    col = _trial_column(lin_indices, I, off_u)
    col == 0 && return (0, 0)
    return (lin_indices[Iv] + row_offset, col + col_offset)
end

"""
    visit_bilinear_stencil(sink, term, sp, row_offset::Int, col_offset::Int) -> sink

Walk every grid point of `sp`, evaluate `term`'s local stencil there, and hand each entry
that lands inside the matrix to `sink`.

The walk that five separate sweeps used to re-derive: mesh, markers, linear indices, the
loop over grid points, the stencil evaluation, and the decision of where each entry lands
([`_entry_target`](@ref)). A sink supplies only what to do with an entry, so the pattern
passes and the value passes share one traversal instead of agreeing by coincidence.

The fifth argument handed to `local_stencil` is the *leaf's* linear index. A term routed to
the wrong leaf reads the wrong `SourceVector` without complaint, which is why the index is
derived here rather than by each caller.

# Arguments
- `sink`: What to do per entry. See [`PatternSink`](@ref), [`RecordSink`](@ref) and
  [`ReplaySink`](@ref), and the contract in [`_sink_entry!`](@ref),
  [`_sink_point!`](@ref) and [`_sink_dedups`](@ref).
- `term`: The AST node whose stencil is evaluated at each point.
- `sp`: The test leaf whose grid is walked and whose markers the stencil sees.
- `row_offset`, `col_offset`: The block's origin in the assembled matrix, `0` for a scalar
  space.

# Returns
- `sink`: The same sink, so a caller can read what it collected.

See also: [`allocate_system_matrix`](@ref), [`add_to_sparse!`](@ref).
"""
@inline function visit_bilinear_stencil(
    sink::SINK, term::TERM, sp, row_offset::Int, col_offset::Int
) where {SINK,TERM}
    Ωₕ = mesh(sp)
    mesh_markers = markers(Ωₕ)
    lin_indices = LinearIndices(indices(Ωₕ))

    @inbounds for I in indices(Ωₕ)
        lin_idx = lin_indices[I]
        stencil = local_stencil(term, sp, I, mesh_markers, lin_idx)
        _sink_point!(sink, lin_idx)

        for k in eachindex(stencil)
            off_u, off_v, weight = stencil[k]
            if _sink_dedups(sink) && _offsets_seen_before(stencil, k, off_u, off_v)
                continue
            end
            row, col = _entry_target(lin_indices, I, off_u, off_v, row_offset, col_offset)
            row == 0 && continue
            _sink_entry!(sink, row, col, weight)
        end
    end
    return sink
end

# --- the sinks --------------------------------------------------------------------- #

"""
    PatternSink(I_vec::Vector{Int}, J_vec::Vector{Int})

Collect the `(row, col)` coordinates a term can reach, for building a sparsity pattern.

Appends each coordinate to `I_vec` and `J_vec`, which [`allocate_system_matrix`](@ref) then
hands to `sparse!`. The only sink that de-duplicates ([`_sink_dedups`](@ref)): a coordinate
named twice by one point's stencil is one entry of the pattern, and the weights it carries
are not read here at all.

See also: [`visit_bilinear_stencil`](@ref), [`RecordSink`](@ref).
"""
struct PatternSink
    I_vec::Vector{Int}
    J_vec::Vector{Int}
end
@inline _sink_dedups(::PatternSink) = true

"""
    _sink_entry!(sink, row::Int, col::Int, weight) -> Nothing

Act on one stencil entry landing at `(row, col)` with coefficient `weight`.

The one method each sink has to supply. Called by [`visit_bilinear_stencil`](@ref) only for
entries that land inside the matrix, so a sink never has to guard the index itself.
"""
@inline function _sink_entry!(sink::PatternSink, row::Int, col::Int, _)
    push!(sink.I_vec, row)
    push!(sink.J_vec, col)
    return nothing
end

"""
    RecordSink(A::SparseMatrixCSC, term, point_ptr::Vector{Int}, positions::Vector{Int})

Add a term's values to `A` and record where each entry landed, building the replay cache.

For each entry it searches `A` for the `(row, col)`'s slot in `nzval`, adds the weight
there, and appends the slot to `positions`. [`_sink_point!`](@ref) opens each grid point's
own slice of that list in `point_ptr`, so a later replay can address a point directly
instead of relying on the walk order.

The search is the expensive half of assembly, which is why it is done once and replayed by
[`ReplaySink`](@ref) afterwards.

# Throws
- `ArgumentError`: A `(row, col)` the pattern does not contain. This pass builds the cache,
  so a pattern that cannot hold the term is reported rather than skipped, unlike
  [`add_to_sparse!`](@ref) on the threaded path.

See also: [`visit_bilinear_stencil`](@ref), [`NzvalSegment`](@ref).
"""
struct RecordSink{M<:SparseMatrixCSC,TERM}
    A::M
    term::TERM
    point_ptr::Vector{Int}
    positions::Vector{Int}
end
@inline _sink_point!(sink::RecordSink, lin_idx::Int) =
    (@inbounds sink.point_ptr[lin_idx] = length(sink.positions) + 1; nothing)
@inline function _sink_entry!(sink::RecordSink, row::Int, col::Int, weight)
    pos = _find_nzval_position(sink.A, row, col)
    pos == 0 && _throw_missing_pattern_entry(sink.term)
    @inbounds sink.A.nzval[pos] += weight
    push!(sink.positions, pos)
    return nothing
end

"""
    ReplaySink(A::SparseMatrixCSC, point_ptr::Vector{Int}, positions::Vector{Int}, p::Int)

Add a term's values to `A` using slots recorded earlier by [`RecordSink`](@ref).

The same walk and the same fresh stencil evaluation, because weights may be live: a
coefficient grid function updated in place through `Rₕ!` is seen by the next assembly. Only
the slot lookup is skipped, taken from `positions` rather than searched for, which is what
the cache buys. `row` and `col` are ignored for that reason.

Mutable because `p` advances through `positions` as the walk proceeds;
[`_sink_point!`](@ref) resets it to each grid point's own slice.

See also: [`visit_bilinear_stencil`](@ref).
"""
mutable struct ReplaySink{M<:SparseMatrixCSC}
    A::M
    point_ptr::Vector{Int}
    positions::Vector{Int}
    p::Int
end
@inline _sink_point!(sink::ReplaySink, lin_idx::Int) =
    (@inbounds sink.p = sink.point_ptr[lin_idx]; nothing)
@inline function _sink_entry!(sink::ReplaySink, ::Int, ::Int, weight)
    @inbounds sink.A.nzval[sink.positions[sink.p]] += weight
    sink.p += 1
    return nothing
end

# Refuse cross-mesh coupling unless an explicit mapping (such as interpolation) is provided.
@noinline function _throw_cross_mesh_block(term, Ωu, Ωv)
    throw(
        ArgumentError(
            "a bilinear term coupling two leaves over different meshes has no assembly: the " *
            "trial leaf has $(npoints(Ωu, Tuple)) points and the test leaf $(npoints(Ωv, Tuple)), " *
            "so an index on one names no point on the other. Got $(typeof(term)). Couple leaves " *
            "that share a mesh, or wrap the trial function in an interpolation operator: `πₕ(Wtrial, u)`.",
        ),
    )
end

@inline function _check_block_meshes(term, trial_leaf, test_leaf)
    _check_interp_spaces(term, trial_leaf)
    _all_trial_interpolated(term) && return nothing

    Ωu = mesh(trial_leaf)
    Ωv = mesh(test_leaf)
    npoints(Ωu, Tuple) == npoints(Ωv, Tuple) || _throw_cross_mesh_block(term, Ωu, Ωv)
    return nothing
end

@inline function _check_one_interp_space(term, Wsrc, trial_leaf)
    Ωsrc = mesh(Wsrc)
    Ωu = mesh(trial_leaf)
    npoints(Ωsrc, Tuple) == npoints(Ωu, Tuple) ||
        _throw_interp_space_mismatch(term, Ωsrc, Ωu)
    return nothing
end

@noinline function _throw_interp_space_mismatch(term, Ωsrc, Ωu)
    throw(
        ArgumentError(
            "the interpolation operator in a bilinear term names a space that is not the trial " *
            "function's: `πₕ` was given a space over a mesh of $(npoints(Ωsrc, Tuple)) points, " *
            "while the trial leaf this term assembles into has $(npoints(Ωu, Tuple)). Got " *
            "$(typeof(term)). `πₕ(Wsrc, u)` interpolates from the space the trial function " *
            "lives on, so `Wsrc` must be that space.",
        ),
    )
end

@inline _check_block_meshes(op::OperatorAdd, trial_leaf, test_leaf) =
    _visit_operator_add1(_check_block_meshes, op, trial_leaf, test_leaf)

# The element type is the one the form's own weights have, promoted against the trial
# space's (supporting automatic differentiation dual numbers). One place for this rule:
# reading it from the space alone instead of promoting against the data broke ForwardDiff in
# four separate places, each with the same symptom (`MethodError: no method matching
# Float64(::Dual)`), each time only on the AD path (bramble-verification §4).
@inline _matrix_eltype(ast, form::BilinearForm) =
    promote_type(_assembled_eltype(ast, form.test_space), eltype(form.trial_space))

# A hint for `sizehint!`, not a real bound: `local_stencil` can return a longer stencil at a
# boundary point than at this representative interior one, so this can undercount. Cheap to
# get wrong, since the only cost is a reallocation of `I_vec`/`J_vec` -- computing the true
# maximum (over the boundary stencils too) would cost more than the reallocation it saves.
# Named for what it is after gpena/Bramble.jl#41 pointed out that "upper bound" was a
# guarantee this never gave.
function _pattern_size_hint(ast::AST_TYPE, sp, mesh_markers, lin_indices) where {AST_TYPE}
    grid_inds = indices(mesh(sp))
    npts = length(grid_inds)
    I = grid_inds[length(grid_inds) ÷ 2 + 1]
    return npts * length(local_stencil(ast, sp, I, mesh_markers, lin_indices[I]))
end

"""
    allocate_system_matrix(form::BilinearForm, ast = resolve_form_ast(form)) -> SparseMatrixCSC

Build the sparse matrix a `BilinearForm` assembles into: the appropriate size, correct sparsity
pattern, and stored zeros throughout.

The pattern follows from the stencil rather than coefficient values, remaining invariant while the mesh
and expression structure are unchanged. Preallocating the matrix once outside loops allows zero-allocation
in-place assembly:

```julia
A = allocate_system_matrix(a)
for step in 1:nsteps
    assemble!(A, a)          # refills values in-place with zero allocations
end
```

Only the structure is preallocated here; all stored entries are zero until `assemble!` fills them.

See also [`assemble`](@ref) and [`assemble!`](@ref).
"""
function allocate_system_matrix(
    form::BilinearForm{D,TrialSpace,TestSpace,AST}, ast=form.ast
) where {D,TrialSpace,TestSpace,AST}
    # The test space: matrix rows are indexed by the test function and the quadrature weight
    # belongs to the integral over the test space mesh.
    space = form.test_space
    _check_block_meshes(ast, form.trial_space, form.test_space)
    Ωₕ = mesh(space)
    mesh_markers = markers(Ωₕ)
    _validate_term_markers(ast, mesh_markers, "the form's space")
    lin_indices = LinearIndices(indices(Ωₕ))

    I_vec = Int[]
    J_vec = Int[]
    hint = _pattern_size_hint(ast, space, mesh_markers, lin_indices)
    sizehint!(I_vec, hint)
    sizehint!(J_vec, hint)

    visit_bilinear_stencil(PatternSink(I_vec, J_vec), ast, space, 0, 0)

    V_vec = _zeros_of(_matrix_eltype(ast, form), length(I_vec))
    return sparse!(I_vec, J_vec, V_vec, ndofs(form.test_space), ndofs(form.trial_space), +)
end

# Which entries a term can reach, block by block.
function _pattern_term!(
    I_vec::Vector{Int},
    J_vec::Vector{Int},
    term::TERM,
    trial_leaf,
    test_leaf,
    row_offset::Int,
    col_offset::Int,
) where {TERM}
    Ωₕ = mesh(test_leaf)
    mesh_markers = markers(Ωₕ)
    _validate_term_markers(term, mesh_markers, "one of the composite space's leaves")
    visit_bilinear_stencil(
        PatternSink(I_vec, J_vec), term, test_leaf, row_offset, col_offset
    )
    return nothing
end

# Recursion shape shared via `_visit_operator_add3` (form/common.jl).
function _pattern_blocks!(
    I_vec::Vector{Int}, J_vec::Vector{Int}, op::OperatorAdd, trial_leaves, test_leaves
)
    return _visit_operator_add3(
        _pattern_blocks!, I_vec, J_vec, op, trial_leaves, test_leaves
    )
end

function _pattern_blocks!(
    I_vec::Vector{Int}, J_vec::Vector{Int}, term::TERM, trial_leaves, test_leaves
) where {TERM}
    for blk in blocks(term, trial_leaves, test_leaves)
        _check_block_meshes(term, blk.trial_leaf, blk.test_leaf)
        _pattern_term!(
            I_vec,
            J_vec,
            term,
            blk.trial_leaf,
            blk.test_leaf,
            blk.row_offset,
            blk.col_offset,
        )
    end
    return nothing
end

function allocate_system_matrix(
    form::BilinearForm{D,TrialSpace,TestSpace,AST}, ast=form.ast
) where {D,TrialSpace<:CompositeGridSpace,TestSpace<:CompositeGridSpace,AST}
    trial_leaves = leaf_spaces_offsets(form.trial_space)
    test_leaves = leaf_spaces_offsets(form.test_space)

    I_vec = Int[]
    J_vec = Int[]

    sp = first(first(test_leaves))
    Ωₛ = mesh(sp)
    hint =
        length(test_leaves) *
        _pattern_size_hint(ast, sp, markers(Ωₛ), LinearIndices(indices(Ωₛ)))
    sizehint!(I_vec, hint)
    sizehint!(J_vec, hint)

    _pattern_blocks!(I_vec, J_vec, ast, trial_leaves, test_leaves)

    ncols = ndofs(form.trial_space)
    nrows = ndofs(form.test_space)
    V_vec = _zeros_of(_matrix_eltype(ast, form), length(I_vec))
    return sparse!(I_vec, J_vec, V_vec, nrows, ncols, +)
end

# --- Assembly implementations ----------------------------------------------------- #

function apply_dirichlet_labels!(
    A::AbstractMatrix, form::BilinearForm, dirichlet_labels, dirichlet_components=nothing
)
    if dirichlet_labels !== nothing
        if dirichlet_labels isa Symbol
            dirichlet_bc!(
                A, test_space(form), dirichlet_labels; components=dirichlet_components
            )
        elseif dirichlet_labels isa Tuple
            if !isempty(dirichlet_labels)
                dirichlet_bc!(
                    A,
                    test_space(form),
                    dirichlet_labels...;
                    components=dirichlet_components,
                )
            end
        end
    end
end

"""
    assemble(form::BilinearForm; dirichlet = nothing, dirichlet_components = nothing) -> SparseMatrixCSC

Allocate a matrix with the form's sparsity pattern and assemble into it.

**Call this once, then assemble into what it returns.** Building the sparsity pattern is the
larger part of the work (at 250,000 degrees of freedom it is 9,700 us and 52 MB against 1,500 us
and zero allocations to refill the matrix), and the pattern does not change between assemblies.
A time loop or Newton iteration benefits from preallocating the pattern once:

```julia
A = assemble(a)                        # once: pattern, allocation and initial fill
for step in 1:nsteps
    Rₕ!(cₕ, coefficient_at(step))      # modified in-place
    assemble!(A, a)                    # or assemble_parallel!(A, a)
end
```

Runs serially or across threads following `form.trial_space`'s backend
[`execution_policy`](@ref): [`Serial`](@ref) by default. Optional `dirichlet` applies
boundary conditions to the matrix -- a label `Symbol`, a `Tuple` of labels, a `label => f`
`Pair` (values ignored on the matrix side), or constraints from
[`dirichlet_constraints`](@ref); see [`_normalize_dirichlet`](@ref) for every accepted
form. `dirichlet_components` restricts which leaf components of a composite trial space
they bind to (see [`dirichlet_bc!`](@ref)).
"""
function assemble(form::BilinearForm; dirichlet=nothing, dirichlet_components=nothing)
    ast_resolved = form.ast
    A = allocate_system_matrix(form, ast_resolved)
    assemble!(
        A,
        form;
        dirichlet=dirichlet,
        dirichlet_components=dirichlet_components,
        ast=ast_resolved,
    )
    return A
end

# --- Helper cores for function barrier optimization ------------------------------- #
#
# Two ways to fill a term's block, serially: `_record_segment!` searches for each entry's
# nzval position (as every call used to) and also records it; `_replay_segment!` reads a
# previously recorded position back instead of searching. `_assemble_bilinear_core_cached!`
# picks between them by whether `A` is the exact matrix object the form's cache was last
# built against (gpena/Bramble.jl#26).
#
# The parallel path (`_sweep_bilinear!`/`_sweep_bilinear_colour!`, below) is untouched: it
# always threads, and recording is an inherently serial, one-time pass (`push!` from
# multiple threads would race), so caching it would mean its first call silently stopped
# threading -- breaking `assemble_parallel!`'s own documented contract ("always threads").

@noinline function _throw_missing_pattern_entry(term)
    throw(
        ArgumentError(
            "assembling $(typeof(term)) reached a matrix entry outside its preallocated " *
            "sparsity pattern. `A` was not built by `allocate_system_matrix` for this exact " *
            "form, or the form's `ast` changed after `A` was built.",
        ),
    )
end

# One term into one block, serially, searching for each entry's nzval position (once) and
# recording it into a fresh `NzvalSegment` alongside performing the (first) real scatter.
# `row_offset` comes from the test leaf and `col_offset` from the trial leaf: a matrix row
# is indexed by the test function.
function _record_segment!(
    A::SparseMatrixCSC, term::TERM, sp, row_offset::Int, col_offset::Int
) where {TERM}
    n = length(indices(mesh(sp)))
    sink = RecordSink(A, term, Vector{Int}(undef, n + 1), Int[])
    visit_bilinear_stencil(sink, term, sp, row_offset, col_offset)
    @inbounds sink.point_ptr[n + 1] = length(sink.positions) + 1
    return (sink.point_ptr, sink.positions)::NzvalSegment
end

# The replay counterpart: same walk, same fresh stencil evaluation (weights may be live --
# only positions are fixed), but each entry's nzval index comes from `segment` instead of a
# search. `point_ptr[lin_idx]` addresses each point's own slice of `positions` directly, so
# this stays correct regardless of what order grid points are visited in.
function _replay_segment!(
    A::SparseMatrixCSC,
    term::TERM,
    sp,
    row_offset::Int,
    col_offset::Int,
    segment::NzvalSegment,
) where {TERM}
    point_ptr, positions = segment
    visit_bilinear_stencil(
        ReplaySink(A, point_ptr, positions, 1), term, sp, row_offset, col_offset
    )
    return nothing
end

# The scalar case: one block, no offsets, so exactly one segment either way.
function _record_bilinear_core!(
    A::SparseMatrixCSC,
    trial_space,
    test_space,
    ast::AST_TYPE,
    segments::Vector{NzvalSegment},
) where {AST_TYPE}
    _check_block_meshes(ast, trial_space, test_space)
    push!(segments, _record_segment!(A, ast, test_space, 0, 0))
    return nothing
end

function _replay_bilinear_core!(
    A::SparseMatrixCSC,
    trial_space,
    test_space,
    ast::AST_TYPE,
    segments::Vector{NzvalSegment},
) where {AST_TYPE}
    _check_block_meshes(ast, trial_space, test_space)
    _replay_segment!(A, ast, test_space, 0, 0, segments[1])
    return nothing
end

function _record_blocks!(
    A::SparseMatrixCSC,
    op::OperatorAdd,
    trial_leaves,
    test_leaves,
    segments::Vector{NzvalSegment},
)
    _record_blocks!(A, op.left_op, trial_leaves, test_leaves, segments)
    _record_blocks!(A, op.right_op, trial_leaves, test_leaves, segments)
    return nothing
end

function _record_blocks!(
    A::SparseMatrixCSC,
    term::TERM,
    trial_leaves,
    test_leaves,
    segments::Vector{NzvalSegment},
) where {TERM}
    for blk in blocks(term, trial_leaves, test_leaves)
        _check_block_meshes(term, blk.trial_leaf, blk.test_leaf)
        push!(
            segments,
            _record_segment!(A, term, blk.test_leaf, blk.row_offset, blk.col_offset),
        )
    end
    return nothing
end

# `next` is threaded through by value and returned, rather than via a mutable `Ref`, so
# this stays allocation-free: the segment index the *next* leaf-term/block should consume,
# in the same left-then-right order `_record_blocks!` built `segments` in.
function _replay_blocks!(
    A::SparseMatrixCSC,
    op::OperatorAdd,
    trial_leaves,
    test_leaves,
    segments::Vector{NzvalSegment},
    next::Int,
)
    next = _replay_blocks!(A, op.left_op, trial_leaves, test_leaves, segments, next)
    next = _replay_blocks!(A, op.right_op, trial_leaves, test_leaves, segments, next)
    return next
end

function _replay_blocks!(
    A::SparseMatrixCSC,
    term::TERM,
    trial_leaves,
    test_leaves,
    segments::Vector{NzvalSegment},
    next::Int,
) where {TERM}
    for blk in blocks(term, trial_leaves, test_leaves)
        _check_block_meshes(term, blk.trial_leaf, blk.test_leaf)
        next += 1
        _replay_segment!(
            A, term, blk.test_leaf, blk.row_offset, blk.col_offset, segments[next]
        )
    end
    return next
end

function _record_bilinear_core!(
    A::SparseMatrixCSC,
    trial_space::CompositeGridSpace,
    test_space::CompositeGridSpace,
    ast::AST_TYPE,
    segments::Vector{NzvalSegment},
) where {AST_TYPE}
    _record_blocks!(
        A, ast, leaf_spaces_offsets(trial_space), leaf_spaces_offsets(test_space), segments
    )
    return nothing
end

function _replay_bilinear_core!(
    A::SparseMatrixCSC,
    trial_space::CompositeGridSpace,
    test_space::CompositeGridSpace,
    ast::AST_TYPE,
    segments::Vector{NzvalSegment},
) where {AST_TYPE}
    _replay_blocks!(
        A,
        ast,
        leaf_spaces_offsets(trial_space),
        leaf_spaces_offsets(test_space),
        segments,
        0,
    )
    return nothing
end

# Picks recording (cache miss: a fresh `A`, a changed `ast` -- e.g. `assemble!(A, form;
# ast = ...)` given something other than `form.ast` -- or the first call ever) or replay
# (cache hit) and keeps `cache` in step with whichever one ran. `ast` is checked as well as
# `A`: the cache's positions are only valid for the exact stencil shape they were recorded
# against, and a different `ast` can visit a different number of entries per point.
function _assemble_bilinear_core_cached!(
    A::SparseMatrixCSC, trial_space, test_space, ast::AST_TYPE, cache::_AssemblyCache
) where {AST_TYPE}
    if cache.A === A && cache.ast === ast
        _replay_bilinear_core!(A, trial_space, test_space, ast, cache.segments)
    else
        # A fresh vector, not `empty!` on whatever `cache.segments` currently references:
        # that reference may be `_NO_SEGMENTS`, shared with every other not-yet-assembled
        # form, and `empty!`ing it in place would corrupt all of them.
        segments = NzvalSegment[]
        _record_bilinear_core!(A, trial_space, test_space, ast, segments)
        cache.segments = segments
        cache.A = A
        cache.ast = ast
    end
    return A
end

# One grid point's stencil, scattered into the matrix. Used only by the parallel path
# below: it always searches (never caches), so a serial recording pass is never required
# before a `Parallel()`-backend form's first assembly.
@inline function _scatter_point!(
    A::SparseMatrixCSC,
    term::TERM,
    sp,
    I::CartesianIndex,
    lin_indices,
    mesh_markers,
    row_offset::Int,
    col_offset::Int,
) where {TERM}
    stencil = local_stencil(term, sp, I, mesh_markers, lin_indices[I])

    # One point rather than the whole grid, so this cannot call `visit_bilinear_stencil`
    # itself -- the threaded sweep owns the grid loop. It shares the entry rule instead, so
    # the guard and the `AbsoluteColumn` case are still stated in exactly one place.
    for (off_u, off_v, weight) in stencil
        row, col = _entry_target(lin_indices, I, off_u, off_v, row_offset, col_offset)
        row == 0 && continue
        add_to_sparse!(A, row, col, weight)
    end
    return nothing
end

# One colour, threaded, writing directly into the matrix.
@noinline function _sweep_bilinear_colour!(
    A::SparseMatrixCSC,
    sp,
    term::TERM,
    idxs,
    lin_indices,
    mesh_markers,
    row_offset::Int,
    col_offset::Int,
) where {TERM}
    Threads.@threads for I in idxs
        _scatter_point!(A, term, sp, I, lin_indices, mesh_markers, row_offset, col_offset)
    end
    return nothing
end

# Every colour in turn, using strided subgrids.
function _sweep_bilinear!(
    A::SparseMatrixCSC, sp, term::TERM, strides, row_offset::Int, col_offset::Int
) where {TERM}
    Ωₕ = mesh(sp)
    grid_inds = indices(Ωₕ)
    lin_indices = LinearIndices(grid_inds)
    mesh_markers = markers(Ωₕ)

    if prod(strides) == 1
        _sweep_bilinear_colour!(
            A, sp, term, grid_inds, lin_indices, mesh_markers, row_offset, col_offset
        )
        return A
    end

    for c in CartesianIndices(strides)
        _sweep_bilinear_colour!(
            A,
            sp,
            term,
            _colour_subgrid(grid_inds, c, strides),
            lin_indices,
            mesh_markers,
            row_offset,
            col_offset,
        )
    end
    return A
end

function _assemble_blocks_parallel!(
    A::SparseMatrixCSC, op::OperatorAdd, trial_leaves, test_leaves
)
    return _visit_operator_add2(
        _assemble_blocks_parallel!, A, op, trial_leaves, test_leaves
    )
end

function _assemble_blocks_parallel!(
    A::SparseMatrixCSC, term::TERM, trial_leaves, test_leaves
) where {TERM}
    for blk in blocks(term, trial_leaves, test_leaves)
        _check_block_meshes(term, blk.trial_leaf, blk.test_leaf)
        _sweep_bilinear!(
            A,
            blk.test_leaf,
            term,
            _colour_strides(stencil_offsets(term)),
            blk.row_offset,
            blk.col_offset,
        )
    end
    return A
end

function _assemble_bilinear_parallel_core!(
    A::SparseMatrixCSC, trial_space, test_space, ast::AST_TYPE
) where {AST_TYPE}
    _check_block_meshes(ast, trial_space, test_space)
    _sweep_bilinear!(A, test_space, ast, _colour_strides(stencil_offsets(ast)), 0, 0)
    return A
end

function _assemble_bilinear_parallel_core!(
    A::SparseMatrixCSC,
    trial_space::CompositeGridSpace,
    test_space::CompositeGridSpace,
    ast::AST_TYPE,
) where {AST_TYPE}
    _assemble_blocks_parallel!(
        A, ast, leaf_spaces_offsets(trial_space), leaf_spaces_offsets(test_space)
    )
    return A
end

"""
    assemble!(A::SparseMatrixCSC, form::BilinearForm; dirichlet = nothing, dirichlet_components = nothing, ast = form.ast) -> SparseMatrixCSC

Assemble the `BilinearForm` into the preallocated sparse matrix `A`, allocating nothing (**0 bytes**).

Runs serially or across threads following `form.trial_space`'s backend
[`execution_policy`](@ref): [`Serial`](@ref) (the default) or [`Parallel`](@ref).
[`assemble_parallel!`](@ref) is a separate, lower-level entry point that always threads,
ignoring the backend's policy.

By default `assemble!` uses the pre-resolved `form.ast` stored directly inside the form.

## Live coefficients
- Grid functions: the stored AST retains references to source `VectorElement` storage. Mutating values in-place (`Rₕ!(cₕ, ...)` or `parent(cₕ) .= ...`) between steps automatically updates the matrix entries with 0 allocations.
- Dynamic scalars: plain numbers work directly for constant scalars. To update a scalar dynamically across loop iterations, wrap it in a `Ref(val)` (e.g. `β = Ref(1.0); a = form(Wₕ, Wₕ, (u, v) -> innerₕ(β * D₋ₓ(u), D₋ₓ(v)))`). Mutating `β[] = new_val` evaluates live during assembly with 0 allocations.
"""
function assemble!(
    A::SparseMatrixCSC,
    form::BilinearForm{D,TrialSpace,TestSpace,AST};
    dirichlet=nothing,
    dirichlet_components=nothing,
    ast=form.ast,
) where {D,TrialSpace,TestSpace,AST}
    dirichlet_labels, _ = _normalize_dirichlet(dirichlet)
    fill!(nonzeros(A), zero(eltype(nonzeros(A))))

    if execution_policy(form.trial_space) isa Serial
        _assemble_bilinear_core_cached!(
            A, form.trial_space, form.test_space, ast, form.cache
        )
    else
        _assemble_bilinear_parallel_core!(A, form.trial_space, form.test_space, ast)
    end

    apply_dirichlet_labels!(A, form, dirichlet_labels, dirichlet_components)
    return A
end

"""
    assemble_parallel!(A::SparseMatrixCSC, form::BilinearForm, ast = form.ast) -> SparseMatrixCSC

Refill `A` with the assembled `form` across threads and return it, regardless of
`form.trial_space`'s backend policy. `A` must already carry the correct sparsity pattern from
[`allocate_system_matrix`](@ref) or a previous [`assemble`](@ref). Unlike `assemble!`, does
not apply `dirichlet_labels`.

Colouring on the test side ensures thread safety when updating stored matrix values concurrently.
"""
function assemble_parallel!(
    A::SparseMatrixCSC, form::BilinearForm{D,TrialSpace,TestSpace,AST}, ast=form.ast
) where {D,TrialSpace,TestSpace,AST}
    fill!(nonzeros(A), zero(eltype(nonzeros(A))))

    _assemble_bilinear_parallel_core!(A, form.trial_space, form.test_space, ast)

    return A
end

"""
    assemble(a::BilinearForm, l::LinearForm; dirichlet = nothing, dirichlet_components = nothing, symmetrize::Bool = false) -> (A, F)

Assemble both the matrix and the vector in one call, applying the same `dirichlet`
constraints to each -- the common case of building a system to solve `A \\ F` against.
`a` and `l` should share the same test space; boundary values are read from `dirichlet`
once rather than once per form.

`symmetrize = true` additionally restores symmetry in `A` after `dirichlet_bc!` (which
only clears rows, not the columns) and updates `F` to match, via [`symmetrize!`](@ref).
Requires `dirichlet` to name at least one label -- there is nothing to symmetrize against
otherwise.

```julia
A, F = assemble(a, l; dirichlet = :boundary => sol, symmetrize = true)
u = A \\ F
```

is the one-call equivalent of assembling `a` and `l` separately and calling
[`symmetrize!`](@ref) by hand.
"""
function assemble(
    a::BilinearForm,
    l::LinearForm;
    dirichlet=nothing,
    dirichlet_components=nothing,
    symmetrize::Bool=false,
)
    A = assemble(a; dirichlet=dirichlet, dirichlet_components=dirichlet_components)
    F = assemble(l; dirichlet=dirichlet, dirichlet_components=dirichlet_components)

    if symmetrize
        dirichlet_labels, _ = _normalize_dirichlet(dirichlet)
        dirichlet_labels === nothing && _throw_symmetrize_without_dirichlet()
        symmetrize!(
            A, F, test_space(a), dirichlet_labels...; components=dirichlet_components
        )
    end

    return A, F
end

@noinline function _throw_symmetrize_without_dirichlet()
    throw(
        ArgumentError(
            "symmetrize = true has nothing to symmetrize against without dirichlet naming " *
            "at least one label.",
        ),
    )
end
