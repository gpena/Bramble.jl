# bilinear_pattern.jl: matrix allocation and sparsity pattern discovery for a
# `BilinearForm`. Each (term, block) unit is walked once for the `(row, col)` of every entry
# it writes (`_form_coordinates`); the coordinates build the pattern (`sparse!`) and, searched
# in the new matrix, the replay cache's positions (`bilinear_execution.jl`).

# Dispatches on `::Type{T}` to ensure concrete vector return type.
@inline _zeros_of(::Type{T}, n::Int) where {T} = zeros(T, n)

# `_assembled_eltype`/`_probed_eltype` (`form/linear.jl`, not owned by this subplan) probe
# the same representative point `_pattern_size_hint` does, through the same `local_stencil`,
# so they hit the identical device scalar-indexing wall (gpena/Bramble.jl#94 S4.0) -- a third
# call site reaching it, discovered running S4.1's own CHECK rather than assumed. `host_weights`
# is only defined for a `ScalarGridSpace`; a `CompositeGridSpace` test space is passed through
# unchanged; `_routed_eltype` probes its own leaves internally and is out of reach from here.
@inline _pattern_probe_space(sp::ScalarGridSpace) = host_weights(sp)
@inline _pattern_probe_space(sp) = sp

# The element type is the one the form's own weights have, promoted against the trial
# space's (supporting automatic differentiation dual numbers). One place for this rule:
# reading it from the space alone instead of promoting against the data broke ForwardDiff in
# four separate places, each with the same symptom (`MethodError: no method matching
# Float64(::Dual)`), each time only on the AD path (bramble-verification §4).
#
# Probed one summand at a time, never through the whole sum's fused stencil: a fused probe
# compiles every term's stencil into one method a second time. The interpolation in a term
# is bound to the first leaves, which every leaf answers the same way for a weight's type.
function _matrix_eltype(form::BilinearForm, ast)
    probe = _bind_interp_spaces(ast, _first_leaf(form.trial_space), _first_leaf(form.test_space))
    return promote_type(
        _summands_eltype(_summands(probe), _pattern_probe_space(form.test_space)),
        eltype(form.trial_space)
    )
end

# Tail recursion, one small method per remaining length, rather than `mapreduce`'s one
# unrolled fold over the whole summand tuple.
@inline _summands_eltype(ts::Tuple{Any}, space) = _assembled_eltype(first(ts), space)
@inline _summands_eltype(ts::Tuple, space) = promote_type(
    _assembled_eltype(first(ts), space), _summands_eltype(Base.tail(ts), space)
)

@inline _first_leaf(sp::CompositeGridSpace) = first(first(leaf_spaces_offsets(sp)))
@inline _first_leaf(sp) = sp

"""
    _allocate_from_pattern(::Type{MT}, nrows::Int, ncols::Int, I::Vector{Int}, J::Vector{Int}, V::AbstractVector) -> MT

Build the `nrows × ncols` matrix a form's sparsity pattern describes, from the coordinate
triplet `(I, J, V)` the coordinate walk (`_form_coordinates`) collected -- summing `V[k]` into
any `(row, col)` that `I`/`J` name more than once, matching `sparse!`'s own combiner.

The one place a fresh system matrix is born (S1.1, gpena/Bramble.jl#12): every backend's
matrix type implements exactly this to be usable with [`allocate_system_matrix`](@ref).
`SparseMatrixCSC`'s method is `sparse!` itself, consuming `I`/`J` in place. The generic
`AbstractMatrix` fallback -- the dense `Matrix{Float64}` positive control among them --
allocates zeros and scatters into it, since a dense matrix has no sparsity pattern to build.
"""
@inline function _allocate_from_pattern(
        ::Type{MT}, nrows::Int, ncols::Int, I_vec::Vector{Int}, J_vec::Vector{Int},
        V_vec::AbstractVector
) where {MT <: SparseMatrixCSC}
    return sparse!(I_vec, J_vec, V_vec, nrows, ncols, +)
end

function _allocate_from_pattern(
        ::Type{MT}, nrows::Int, ncols::Int, I_vec::Vector{Int}, J_vec::Vector{Int},
        V_vec::AbstractVector{T}
) where {MT <: AbstractMatrix, T}
    # Built via `Array{T}(undef, ...)` + `fill!`, not `zeros(T, nrows, ncols)`: `zeros`
    # dispatches on its first argument as an ordinary value, and analysed abstractly (a
    # `V_vec` too generic to pin `T` down at inference time, as `report_package` does) that
    # argument's own inferred type is `Any`, which inference can't rule out being another
    # `Integer` dimension rather than a type -- so it also considers `zeros(dims::Integer...)`,
    # producing a phantom `Array{Float64, 3}` no backend ever actually returns
    # (gpena/Bramble.jl#12, JET gate). `Array{T}` is `Core.apply_type`, not a value-dispatched
    # call, so it carries no such ambiguity.
    A = Array{T}(undef, nrows, ncols)
    fill!(A, zero(T))
    @inbounds for k in eachindex(I_vec, J_vec, V_vec)
        A[I_vec[k], J_vec[k]] += V_vec[k]
    end
    return A
end

# A hint for `sizehint!`, not a real bound, used by the Jacobian pattern
# (`form/jacobian_pattern.jl`): `local_stencil` can return a longer stencil at a boundary
# point than at this representative interior one, so this can undercount. Cheap to get
# wrong, since the only cost is a reallocation of the coordinate vectors.
#
# `host_weights` (gpena/Bramble.jl#94 S4.0): `local_stencil` reads `sp`'s weights and its
# mesh's spacings one grid point at a time, which a device-backed `sp` refuses outright --
# a no-op on a host-backed `sp`, so the CPU path pays one locality check and nothing else.
#
# `sp` is typed `::ScalarGridSpace` (every call site passes a walked leaf, which bottoms
# out at one) rather than left generic: `host_weights` also has a method for
# `SeparableWeights`, and an untyped `sp` makes JET consider that branch reachable here too,
# reporting `mesh(::SeparableWeights)` as unresolved below even though nothing ever calls
# this with one -- a static inference artefact, not a live path.
function _pattern_size_hint(
        ast::AST_TYPE, sp::ScalarGridSpace, mesh_markers, lin_indices
) where {AST_TYPE}
    hp = host_weights(sp)
    grid_inds = indices(mesh(hp))
    npts = length(grid_inds)
    I = grid_inds[length(grid_inds) ÷ 2 + 1]
    return npts * length(local_stencil(ast, hp, I, mesh_markers, lin_indices[I]))
end

# --- One setup walk per term: coordinates for the pattern and the replay cache ----- #
#
# Every (term, block) unit is walked by `_coord_walk!` twice: a counting pass sizes the
# coordinate vectors and fills the unit's `point_ptr`, a filling pass writes each entry's
# `(row, col)`. The same coordinates build the sparsity pattern (`sparse!`) and, searched in
# the new matrix, the replay cache's positions, so a term's stencil is compiled for this one
# walk and for the replay walk, and the first values come from the replay.
#
# A transposed pair's unit (`_foreach_unit`, `half >= 0`) also writes, after its own
# coordinates, the transposed coordinate `(col + dr, row + dc)` of every entry: the entry the
# pair's second term writes (`_PairReplaySink`). `half` is `0` for both halves, `1` for the
# first term's only and `2` for the transposed only, as on a pair whose blocks sit on two
# leaf objects; a half that is skipped is neither in the pattern nor searched.

# Unit `u`'s share of the coordinate vectors: `n` direct entries unless only the transposed
# half is written, and `n` transposed entries for a pair writing that half.
@inline _direct_count(n::Int, half::Int) = half == 2 ? 0 : n
@inline _transposed_count(n::Int, half::Int) = (half == 0 || half == 2) ? n : 0

# Per-unit state for the two coordinate passes.
mutable struct _CoordPass
    const I::Vector{Int}
    const J::Vector{Int}
    const ptrs::Vector{Vector{Int}}
    const counts::Vector{Int}
    const margins::Vector{Int}
    const halves::Vector{Int}
    const context::String
    fill::Bool
    unit::Int
    base::Int
end
function _CoordPass(context::String)
    return _CoordPass(Int[], Int[], Vector{Int}[], Int[], Int[], Int[], context, false, 0, 0)
end

# One unit, in whichever pass `p` is in. `half` is `-1` for a term walked alone.
function (p::_CoordPass)(
        term::TERM, sp, row_offset::Int, col_offset::Int, dr::Int, dc::Int, half::Int
) where {TERM}
    hp = host_weights(sp)
    if p.fill
        p.unit += 1
        u = p.unit
        n = p.counts[u]
        nd = _direct_count(n, half)
        sink = _CoordSink(p.ptrs[u], p.I, p.J, p.base, p.base + nd, dr, dc, half, true, 0)
        _coord_walk!(sink, term, hp, row_offset, col_offset)
        p.base += nd + _transposed_count(n, half)
    else
        _validate_term_markers(term, markers(mesh(sp)), p.context)
        npts = length(indices(mesh(sp)))
        ptr = Vector{Int}(undef, npts + 1)
        sink = _CoordSink(ptr, p.I, p.J, 0, 0, dr, dc, half, false, 0)
        _coord_walk!(sink, term, hp, row_offset, col_offset)
        @inbounds ptr[npts + 1] = sink.n + 1
        push!(p.ptrs, ptr)
        push!(p.counts, sink.n)
        push!(p.margins, _stencil_margin(term))
        push!(p.halves, half)
    end
    return nothing
end

# Interior box first, then the boundary slabs, all through the one guarded region walk: the
# visit order `visit_bilinear_stencil` gives, so a unit's coordinates line up with its replay
# entry for entry, and one compiled copy of the term's stencil instead of an unguarded one
# beside it.
@noinline function _coord_walk!(
        sink::_CoordSink, term::TERM, sp, row_offset::Int, col_offset::Int
) where {TERM}
    Ωₕ = mesh(sp)
    mesh_markers = markers(Ωₕ)
    grid_inds = indices(Ωₕ)
    lin_indices = LinearIndices(grid_inds)
    ax = axes(grid_inds)
    margin = _stencil_margin(term)
    if _peelable(ax, margin)
        interior = CartesianIndices(map(r -> _interior_range(r, margin), ax))
        _visit_guarded_region!(
            sink, term, sp, mesh_markers, lin_indices, interior, row_offset, col_offset
        )
        for slab in _boundary_shell_slabs(ax, margin)
            _visit_guarded_region!(
                sink, term, sp, mesh_markers, lin_indices, slab, row_offset, col_offset
            )
        end
    else
        whole = CartesianIndices(map(_full_range, ax))
        _visit_guarded_region!(
            sink, term, sp, mesh_markers, lin_indices, whole, row_offset, col_offset
        )
    end
    return nothing
end

# Count, size, fill: every unit's coordinates in one pair of vectors, in the order
# `_replay_bilinear_core!` consumes segments.
function _form_coordinates(trial_space, test_space, ast)
    p = _CoordPass(
        _is_block_pair(trial_space, test_space) ? "one of the composite space's leaves" :
        "the form's space"
    )
    _foreach_unit(p, trial_space, test_space, ast)
    total = 0
    for u in eachindex(p.counts)
        n = p.counts[u]
        total += _direct_count(n, p.halves[u]) + _transposed_count(n, p.halves[u])
    end
    resize!(p.I, total)
    resize!(p.J, total)
    p.fill = true
    _foreach_unit(p, trial_space, test_space, ast)
    return p
end

# Coordinates become positions in `A`, in place in `p.I`. `ast` only names the form in the
# error.
function _coordinates_to_positions!(A::AbstractMatrix, p::_CoordPass, ast)
    I = p.I
    J = p.J
    @inbounds for k in eachindex(I, J)
        pos = _scatter_position(A, I[k], J[k])
        pos == 0 && _throw_missing_pattern_entry(ast)
        I[k] = pos
    end
    return nothing
end

# One `Segment{D}` per unit, cut from the positions.
function _segments_from_positions(::Val{D}, p::_CoordPass) where {D}
    segments = Segment{D}[]
    base = 0
    for u in eachindex(p.counts)
        n = p.counts[u]
        half = p.halves[u]
        if half < 0
            positions = p.I[(base + 1):(base + n)]
            push!(segments, _unit_segment(Val(D), p.margins[u], p.ptrs[u], positions))
            base += n
        else
            nd = _direct_count(n, half)
            nt = _transposed_count(n, half)
            push!(
                segments,
                Segment{D}(
                    false, p.ptrs[u], p.I[(base + 1):(base + nd)], Int[], Int[], 0,
                    _empty_interior(Val(D)), p.I[(base + nd + 1):(base + nd + nt)]
                )
            )
            base += nd + nt
        end
    end
    return segments
end

# Only 1D segments can be diagonal (`_diagonal_replay`), and a 1D unit's grid is exactly its
# own points, so each unit (a cross-mesh leaf included) is checked against its own grid.
_unit_segment(::Val{D}, _, ptr, positions) where {D} = _flat_segment(Val(D), ptr, positions)
function _unit_segment(::Val{1}, margin::Int, ptr::Vector{Int}, positions::Vector{Int})
    grid = CartesianIndices((Base.OneTo(length(ptr) - 1),))
    return _try_diagonal_segment(margin, grid, ptr, positions)
end

# The pattern from coordinates, leaving `I`/`J` intact for the positions search.
function _allocate_keeping(
        ::Type{MT}, nrows::Int, ncols::Int, I::Vector{Int}, J::Vector{Int}, V::Vector{Tv}
) where {MT <: SparseMatrixCSC, Tv}
    m = length(I)
    return sparse!(
        I, J, V, nrows, ncols, +, Vector{Int}(undef, ncols), Vector{Int}(undef, nrows + 1),
        Vector{Int}(undef, m), Vector{Tv}(undef, m), Vector{Int}(undef, ncols + 1), Int[], Tv[]
    )
end

"""
    allocate_system_matrix(form::BilinearForm, ast = resolve_form_ast(form)) -> AbstractMatrix

Build the matrix a `BilinearForm` assembles into: the appropriate size, correct sparsity
pattern, and stored zeros throughout, in `matrix_type(backend(test_space(form)))` --
`SparseMatrixCSC{Float64,Int}` by default, or whatever [`backend`](@ref) the space's mesh was
built with (see [`_allocate_from_pattern`](@ref)). On a [`metal_backend`](@ref) space this
returns a device-resident `BrambleMetalExt.MetalSparseMatrixCSR{Float32,Int64}`, not a host
`SparseMatrixCSC` -- verified against real triplets (gpena/Bramble.jl#94 S4.1).

The pattern follows from the stencil rather than coefficient values, remaining invariant while the mesh
and expression structure are unchanged. Preallocating the matrix once outside loops allows zero-allocation
in-place assembly:

```julia
using Bramble: allocate_system_matrix
A = allocate_system_matrix(a)
for step in 1:nsteps
    assemble!(A, a)          # refills values in-place with zero allocations
end
```

Only the structure is preallocated here; all stored entries are zero until `assemble!` fills them.

See also [`assemble`](@ref) and [`assemble!`](@ref).
"""
function allocate_system_matrix(form::BilinearForm{D}, ast = form.ast) where {D}
    p = _form_coordinates(form.trial_space, form.test_space, ast)
    MT = matrix_type(backend(form.test_space))
    V = _zeros_of(_matrix_eltype(form, ast), length(p.I))
    nrows = ndofs(form.test_space)
    ncols = ndofs(form.trial_space)
    # The walk's coordinates are the replay cache's positions too, once searched in the new
    # matrix, so a serial sparse form keeps them: its first fill is then a replay.
    keep = MT <: SparseMatrixCSC && ast === form.ast &&
           execution_policy(form.trial_space) isa CpuSerial
    keep || return _allocate_from_pattern(MT, nrows, ncols, p.I, p.J, V)
    A = _allocate_keeping(MT, nrows, ncols, p.I, p.J, V)
    _coordinates_to_positions!(A, p, ast)
    _store_recording!(form.cache, ast, _segments_from_positions(Val(D), p), A)
    return A
end
