# ==============================================================================
# Struct Definitions
# ==============================================================================

# of a `LinearForm` field, which has to resolve when the struct is defined rather than when
# it is called, so keeping it here made unlocking that file alone impossible.

"""
    BilinearForm{D,TrialSpace,TestSpace,AST}

Represents a bilinear form defined over a trial space and test space.

# Fields
- `trial_space::TrialSpace`: The space for the trial function.
- `test_space::TestSpace`: The space for the test function.
- `ast::AST`: The resolved expression tree.

The form resolves its expression tree `ast` once at construction, referencing the underlying
storage of any coefficient grid functions (`VectorElement`). In-place updates via `Rₕ!(cₕ, ...)`
or `values(cₕ) .= ...` are automatically seen by subsequent assemblies with zero heap allocations.
The expression itself is not kept: nothing downstream ever calls it again, only the AST it
built once.

Constant scalar coefficients can be written directly as plain numbers (e.g. `2.0 * innerₕ(D₋ₓ(u), D₋ₓ(v))`).
`Ref` is only needed if you want a **dynamic scalar coefficient** that changes across iterations in a loop:
```julia
β = Ref(1.0)
a = form(Wₕ, Wₕ, (u, v) -> innerₕ(β * D₋ₓ(u), D₋ₓ(v)))
# Inside time loop:
β[] = 3.0
assemble!(A, a) # allocates 0 bytes and evaluates with β = 3.0
```
"""
struct BilinearForm{D, TrialSpace, TestSpace, AST}
    trial_space::TrialSpace
    test_space::TestSpace
    ast::AST
end

"""
    trial_space(form::BilinearForm)

Returns the trial space of the bilinear form.
"""
trial_space(form::BilinearForm) = form.trial_space

"""
    test_space(form::BilinearForm)

Returns the test space of the bilinear form.
"""
test_space(form::BilinearForm) = form.test_space

# `a(uₕ, vₕ) = vᵀ A u`. Assembles a whole matrix per call, which is what makes it a
# convenience rather than something to put in a loop.
@inline (form::BilinearForm)(u, v) = dot(v, assemble(form) * u)

"""
    resolve_form_ast(form::BilinearForm)

Returns the resolved AST stored inside the bilinear form.
"""
@inline resolve_form_ast(form::BilinearForm) = form.ast

"""
    form(Wₕ, Vₕ, f)

Constructs a `BilinearForm` over the trial space `Wₕ` and the test space `Vₕ` from the
bilinear expression `f`, a function of a trial and a test argument.

A composite space needs no separate constructor and no separate type. Its blocks are
addressed by leaf index, `u(i)` and `v(j)`, whatever the nesting: `leaf_spaces_offsets`
flattens a space of spaces into leaves, so a two-by-two nesting is four blocks numbered one
to four. A term naming neither side is the same integrand on every diagonal block, and one
naming both is the single block it belongs to, off-diagonal included.

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
    return BilinearForm{D, typeof(Wₕ), typeof(Vₕ), typeof(ast)}(Wₕ, Vₕ, ast)
end

# ==============================================================================
# Utility Helpers
# ==============================================================================

@inline function add_to_sparse!(A::SparseMatrixCSC, row::Int, col::Int, val::Number)
    p1 = A.colptr[col]
    p2 = A.colptr[col + 1] - 1

    if (p2 - p1) < 32
        idx = p1
        @inbounds while idx <= p2
            if A.rowval[idx] == row
                A.nzval[idx] += val
                return
            end
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
                A.nzval[mid] += val
                return
            end
        end
    end
end

# `zeros(T, n)` with `T` a value rather than a literal type matches two methods —
# `zeros(::Type, ::Integer)` giving a vector and `zeros(::Integer, ::Integer)` giving a
# matrix — so the result infers as a union and `sparse` has no method for half of it.
# Dispatching on `::Type{T}` settles which one, and the vector comes back concrete.
@inline _zeros_of(::Type{T}, n::Int) where {T} = zeros(T, n)

# Whether an earlier entry of this stencil already named the same pair of offsets.
#
# The pattern wants each position once, however many terms write a value there. A form like
# `innerₕ(u, v) + inner₊ₓ(D₋ₓ(u), D₋ₓ(v))` has both products contributing at `(0, 0)`, so
# its stencil is five entries over four distinct pairs — a fifth of the coordinates it
# generated were the same entry twice, and `sparse!` then paid to sort and merge them.
#
# A linear scan over the entries already seen, which is the whole stencil: a handful, not a
# collection worth a set.
@inline function _offsets_seen_before(stencil, k::Int, off_u, off_v)
    @inbounds for l in 1:(k - 1)
        stencil[l][1] == off_u && stencil[l][2] == off_v && return true
    end
    return false
end

# Which column of the trial block a stencil entry's trial slot names, or `0` when it names
# none of them.
#
# Two kinds of entry, resolved by dispatch on the slot rather than by a runtime flag. The
# ordinary kind is an offset from the point being visited, read out of the index space being
# walked and dropped when it falls off the grid — the boundary truncation the whole assembly
# has always relied on. An interpolation's kind (`AbsoluteColumn`, point 61) is already a
# column of the *source* space, chosen by `locate_cell`, and needs neither the arithmetic nor
# the bounds check: `locate_cell` clamps to a real cell, so every column it names exists.
#
# `0` for "no column" rather than `nothing`: column indices are one-based, so the sentinel
# cannot collide with an answer, and the callers stay free of a union to unwrap in their
# innermost loop.
@inline function _trial_column(lin_indices, I::CartesianIndex, off_u)
    Iu = I + CartesianIndex(off_u)
    return checkbounds(Bool, lin_indices, Iu) ? lin_indices[Iu] : 0
end

@inline _trial_column(lin_indices, I::CartesianIndex, off_u::AbsoluteColumn) = off_u.col

# Whether a term's two leaves can be coupled at all.
#
# A coupled term is assembled by walking the *test* leaf's grid and reading the trial column
# out of that same index space, offset into the trial leaf's block:
# `lin_indices[I + off_u] + col_offset`. That is only meaningful when the two leaves share an
# index space — which they always did until heterogeneous composite spaces arrived, since a
# space built by repeating one space (`Wₕ^Val(N)`) hands every leaf the same mesh object.
#
# On leaves over differently-sized meshes there is no correspondence between an index on one
# and an index on the other, and the arithmetic fails in whichever direction the sizes run:
# a smaller trial block overruns (a loud `ArgumentError` from `sparse!`, naming column
# indices rather than the real problem) and a larger one lands on in-range but *wrong*
# columns, silently. Neither is an answer to give, because the term has no meaning until
# something says how to map between the two meshes. That something is the symbolic
# interpolation operator `πₕ(Wsrc, u)` (point 61), and a term that carries one is exempt from
# this check for exactly that reason: its trial entries are absolute columns of `Wsrc`
# (`AbsoluteColumn`, `_trial_column`), so it never does the index arithmetic this refuses.
# Every other cross-mesh term is still refused by name, which is the only honest option — the
# same reasoning as "a missing component is an error, not a zero".
@noinline function _throw_cross_mesh_block(term, Ωu, Ωv)
    throw(ArgumentError(
        "a bilinear term coupling two leaves over different meshes has no assembly: the " *
        "trial leaf has $(npoints(Ωu, Tuple)) points and the test leaf $(npoints(Ωv, Tuple)), " *
        "so an index on one names no point on the other. Got $(typeof(term)). Couple leaves " *
        "that share a mesh, or say how to map between them: wrap the trial function in the " *
        "interpolation operator, `πₕ(Wtrial, u)`, which reads the trial field at the test " *
        "mesh's own points and is what gives such a block a meaning. On the source side of a " *
        "*linear* form, `πₕ(uₕ)` does the same for a field whose values are already known."))
end

@inline function _check_block_meshes(term, trial_leaf, test_leaf)
    # Every interpolation the term carries has to name the leaf whose columns it writes into.
    # Checked first and unconditionally: it applies whether or not the rest of the term
    # interpolates, and folds to nothing when no interpolation is present.
    _check_interp_spaces(term, trial_leaf)

    # The mesh correspondence is only dispensable when *every* trial column the term
    # contributes is an absolute one. A term mixing the two — `πₕ(Wsrc, u) + u` — still reads
    # the bare `u`'s column out of the index space being walked, so it still needs the two
    # leaves to share one. Asking "does an interpolation appear anywhere" instead is what let
    # such a mix assemble against wrong columns in silence.
    _all_trial_interpolated(term) && return nothing

    Ωu = mesh(trial_leaf)
    Ωv = mesh(test_leaf)
    npoints(Ωu, Tuple) == npoints(Ωv, Tuple) || _throw_cross_mesh_block(term, Ωu, Ωv)
    return nothing
end

# An interpolating term names its columns outright, so the two leaves are free to be over
# different meshes — that is the whole point of point 61. What is *not* free is which space
# those columns belong to: they are numbered in `Wsrc`, and the block writes them into the
# trial leaf's own column range. Get that pairing wrong and a too-small `Wsrc` writes into a
# corner of the block while a too-large one runs past its end, which is the same silent-wrong
# answer point 69 refused for the un-interpolated case. So it is checked, not assumed.
@inline function _check_one_interp_space(term, Wsrc, trial_leaf)
    Ωsrc = mesh(Wsrc)
    Ωu = mesh(trial_leaf)
    npoints(Ωsrc, Tuple) == npoints(Ωu, Tuple) ||
        _throw_interp_space_mismatch(term, Ωsrc, Ωu)
    return nothing
end

@noinline function _throw_interp_space_mismatch(term, Ωsrc, Ωu)
    throw(ArgumentError(
        "the interpolation operator in a bilinear term names a space that is not the trial " *
        "function's: `πₕ` was given a space over a mesh of $(npoints(Ωsrc, Tuple)) points, " *
        "while the trial leaf this term assembles into has $(npoints(Ωu, Tuple)). Got " *
        "$(typeof(term)). `πₕ(Wsrc, u)` interpolates *from* the space the trial function " *
        "lives on, so `Wsrc` must be that space — the form's trial space, or, on a composite " *
        "space, the leaf the index picks out: `πₕ(leaf, u(i))`."))
end

# A sum is checked term by term. The composite paths route each term to its block first and
# so never reach here with a sum, but the scalar ones check the whole form's AST at once —
# and there `_bears_interpolation` of the sum is true as soon as *one* summand interpolates,
# which would exempt the others along with it. Recursion shape shared via
# `_visit_operator_add1` (form/common.jl).
@inline _check_block_meshes(op::OperatorAdd, trial_leaf, test_leaf) = _visit_operator_add1(
    _check_block_meshes, op, trial_leaf, test_leaf)

# How many entries the pattern can hold at most: one interior stencil's worth per grid
# point. An upper bound, because truncation at a boundary drops entries and never adds any.
#
# Worth computing because `push!` grows by doubling: filling two 1,250,000-element vectors
# cost 28.4 MB each where the data is 9.5 MB, so a third of the pattern's memory was
# reallocation. `sizehint!` is the whole fix for that part.
function _pattern_upper_bound(ast::AST_TYPE, sp, mesh_markers, lin_indices) where {AST_TYPE}
    grid_inds = indices(mesh(sp))
    npts = length(grid_inds)
    I = grid_inds[length(grid_inds) ÷ 2 + 1]
    return npts * length(local_stencil(ast, sp, I, mesh_markers, lin_indices[I]))
end

"""
    allocate_system_matrix(form::BilinearForm, ast = resolve_form_ast(form)) -> SparseMatrixCSC

Builds the sparse matrix a `BilinearForm` assembles into: the right size, the right sparsity
pattern, and every stored value zero.

The pattern follows from the stencil rather than from the numbers, so it is known before any
value is computed, and it does not change while the mesh and the expression do not. That is
what makes the two-step idiom worth using — build the pattern once, outside a time loop, and
refill it inside:

```julia
A = allocate_system_matrix(a)
for step in 1:nsteps
    assemble!(A, a)          # refills the values, allocates nothing
    # …
end
```

against calling [`assemble`](@ref) each step, which allocates a new matrix every time.

Only the structure is computed here. The entries are all zero on return, so a matrix from
this is not usable until `assemble!` has filled it.

See also [`assemble`](@ref) and [`assemble!`](@ref).
"""
function allocate_system_matrix(
        form::BilinearForm{D, TrialSpace, TestSpace, AST},
        ast = form.ast) where {D, TrialSpace, TestSpace, AST}
    # The *test* space, because a matrix row is indexed by the test function and the
    # quadrature weight belongs to the integral, which is over the test space's mesh. The
    # composite path has always walked its test leaf (`first(test_leaves[…])`); the two
    # spaces coincide in every form where one space serves both, so this only starts to
    # matter — and only then differs — once an interpolating term makes them differ (point 61).
    space = form.test_space
    _check_block_meshes(ast, form.trial_space, form.test_space)
    Ωₕ = mesh(space)
    mesh_markers = markers(Ωₕ)
    _validate_term_markers(ast, mesh_markers, "the form's space")
    lin_indices = LinearIndices(indices(Ωₕ))
    n = length(lin_indices)

    I_vec = Int[]
    J_vec = Int[]
    hint = _pattern_upper_bound(ast, space, mesh_markers, lin_indices)
    sizehint!(I_vec, hint)
    sizehint!(J_vec, hint)

    @inbounds for I in indices(Ωₕ)
        lin_idx = lin_indices[I]
        stencil = local_stencil(ast, space, I, mesh_markers, lin_idx)

        for k in eachindex(stencil)
            off_u, off_v, _ = stencil[k]
            _offsets_seen_before(stencil, k, off_u, off_v) && continue

            Iv = I + CartesianIndex(off_v)
            col = _trial_column(lin_indices, I, off_u)
            if checkbounds(Bool, lin_indices, Iv) && col != 0
                push!(I_vec, lin_indices[Iv])
                push!(J_vec, col)
            end
        end
    end

    # The element type is the one the form's own weights have, promoted against the space's
    # — the same rule `assemble` uses for a right-hand side, and for the same two reasons.
    # It is what lets a `Dual` through, and taking it from the space instead left it
    # inferred as a union: `zeros(eltype(space), n)` can be `zeros(n₁, n₂)` when `eltype`
    # of an unconstrained type parameter is not known to be a type, so `V_vec` came out as
    # `Union{Vector{Float64}, Matrix{Float64}}` and `sparse` had no method for half of it.
    # JET found that; nothing else would have, since the bad half is unreachable in
    # practice.
    # `sparse!` rather than `sparse`: it uses the coordinate vectors as its own scratch
    # instead of copying them, and they are discarded here either way. Worth 13.4 MB of the
    # 64.9 the pattern cost at 250,000 degrees of freedom.
    V_vec = _zeros_of(
        promote_type(_assembled_eltype(ast, space), eltype(form.trial_space)), length(I_vec))
    # rows are indexed by the test function and columns by the trial one. The check at the
    # top is what makes these equal; naming them separately keeps the shape honest rather
    # than resting on an `n` that happens to serve both.
    return sparse!(I_vec, J_vec, V_vec, ndofs(form.test_space), ndofs(form.trial_space), +)
end

# Which entries a term can reach, block by block.
#
# The version this replaces walked `space.spaces` with one offset used for both the row and
# the column, so the pattern it built had diagonal blocks and nothing else — and
# `add_to_sparse!` searches for an entry and returns quietly when it is missing, so every
# off-diagonal contribution was dropped without a word. `innerₕ(u(1), v(2))` assembled to
# zeros.
function _pattern_term!(I_vec::Vector{Int}, J_vec::Vector{Int}, term::TERM, trial_leaf,
        test_leaf, row_offset::Int, col_offset::Int) where {TERM}
    Ωₕ = mesh(test_leaf)
    mesh_markers = markers(Ωₕ)
    _validate_term_markers(term, mesh_markers, "one of the composite space's leaves")
    lin_indices = LinearIndices(indices(Ωₕ))

    @inbounds for I in indices(Ωₕ)
        stencil = local_stencil(term, test_leaf, I, mesh_markers, lin_indices[I])

        for k in eachindex(stencil)
            off_u, off_v, _ = stencil[k]
            _offsets_seen_before(stencil, k, off_u, off_v) && continue

            Iv = I + CartesianIndex(off_v)
            col = _trial_column(lin_indices, I, off_u)

            if checkbounds(Bool, lin_indices, Iv) && col != 0
                push!(I_vec, lin_indices[Iv] + row_offset)
                push!(J_vec, col + col_offset)
            end
        end
    end
    return nothing
end

# Recursion shape shared via `_visit_operator_add3` (form/common.jl).
function _pattern_blocks!(I_vec::Vector{Int}, J_vec::Vector{Int}, op::OperatorAdd,
        trial_leaves, test_leaves)
    _visit_operator_add3(
        _pattern_blocks!, I_vec, J_vec, op, trial_leaves, test_leaves)
end

function _pattern_blocks!(I_vec::Vector{Int}, J_vec::Vector{Int}, term::TERM,
        trial_leaves, test_leaves) where {TERM}
    blk = block_of(term, length(trial_leaves), length(test_leaves))

    if blk === nothing
        for c in 1:min(length(trial_leaves), length(test_leaves))
            _check_block_meshes(term, first(trial_leaves[c]), first(test_leaves[c]))
            _pattern_term!(I_vec, J_vec, term, first(trial_leaves[c]),
                first(test_leaves[c]), last(test_leaves[c]), last(trial_leaves[c]))
        end
        return nothing
    end

    tc, sc = blk
    _check_block_meshes(term, first(trial_leaves[tc]), first(test_leaves[sc]))
    _pattern_term!(I_vec, J_vec, term, first(trial_leaves[tc]), first(test_leaves[sc]),
        last(test_leaves[sc]), last(trial_leaves[tc]))
    return nothing
end

function allocate_system_matrix(
        form::BilinearForm{D, TrialSpace, TestSpace, AST},
        ast = form.ast) where {D, TrialSpace <: CompositeGridSpace,
        TestSpace <: CompositeGridSpace, AST}
    trial_leaves = leaf_spaces_offsets(form.trial_space)
    test_leaves = leaf_spaces_offsets(form.test_space)

    I_vec = Int[]
    J_vec = Int[]

    # One block's worth per block is the bound: a term reaching every diagonal block reaches
    # as many as there are leaves, and one naming a block reaches exactly one.
    sp = first(first(test_leaves))
    Ωₛ = mesh(sp)
    hint = length(test_leaves) *
           _pattern_upper_bound(ast, sp, markers(Ωₛ), LinearIndices(indices(Ωₛ)))
    sizehint!(I_vec, hint)
    sizehint!(J_vec, hint)

    _pattern_blocks!(I_vec, J_vec, ast, trial_leaves, test_leaves)

    ncols = ndofs(form.trial_space)
    nrows = ndofs(form.test_space)
    # `first_space(form.trial_space)` used to be passed here, which is a *scalar* leaf —
    # `_assembled_eltype` then dispatched to its generic, non-composite method, which probes
    # the whole (possibly multi-leaf) `ast` with a single `local_stencil` call and reads only
    # the type of that stencil's *first* entry. A `GridFunctionScale` coefficient (a
    # `ForwardDiff.Dual`, differentiating through a coupled nonlinear system) contributed by a
    # later leaf's term was invisible to that single probe: the matrix still allocated as
    # `Float64`, and the first attempt to scatter a `Dual` value into it threw
    # `MethodError: no method matching Float64(::Dual)` from deep inside `add_to_sparse!`,
    # far from this line. Passing the composite space itself dispatches to the
    # `CompositeGridSpace` method instead, which probes every term against the leaf it
    # actually routes to and promotes across all of them — the same thing the scalar path
    # above already gets from `promote_type(_assembled_eltype(ast, space), ...)`.
    V_vec = _zeros_of(
        promote_type(_assembled_eltype(ast, form.test_space), eltype(form.trial_space)),
        length(I_vec))
    return sparse!(I_vec, J_vec, V_vec, nrows, ncols, +)
end

# ==============================================================================
# Assembly Implementations
# ==============================================================================

function apply_dirichlet_labels!(
        A::AbstractMatrix, form::BilinearForm, dirichlet_labels, dirichlet_components = nothing)
    if dirichlet_labels !== nothing
        if dirichlet_labels isa Symbol
            dirichlet_bc!(A, trial_space(form), dirichlet_labels; components = dirichlet_components)
        elseif dirichlet_labels isa Tuple
            if !isempty(dirichlet_labels)
                dirichlet_bc!(A, trial_space(form), dirichlet_labels...;
                    components = dirichlet_components)
            end
        end
    end
end

"""
    assemble(form::BilinearForm; dirichlet_labels = nothing)

Allocates a matrix with the form's sparsity pattern and assembles into it.

**Call this once, then assemble into what it returns.** Building the sparsity pattern is by
far the larger half of the work — at 250,000 degrees of freedom it is 9,700 us and 52 MB
against 1,500 us and nothing to fill the matrix — and the pattern does not change between
assemblies. So a time loop or a Newton iteration written with `assemble` pays for the same
pattern on every step:

```julia
A = assemble(a)                        # once: pattern, allocation and the first fill
for step in 1:nsteps
    Rₕ!(cₕ, coefficient_at(step))      # written through, so the form still sees it
    assemble!(A, a)                    # or assemble_parallel!(A, a)
end
```

which is seven times cheaper per step. It assumes the pattern stays the same, which it does
as long as the form's operators do: a coefficient changes values, not which entries exist.

`assemble!` also takes an `ast`, and for a form like the one above there is no reason to use
it — resolving is 11 us against 550 to assemble, and 288 bytes. It earns its place in one
case, where a *coefficient* carries an operator: `innerₕ(D₋ₓ(cₕ) * u, v)` recomputes
`D₋ₓ(cₕ)` on every resolve, 2 MB of it, where the same operator on the test argument is
symbolic and free. Hoisting is the better answer there, and keeps the form live —

```julia
dcₕ = D₋ₓ(cₕ)                          # computed once, and still written through
a = form(Wₕ, Wₕ, (u, v) -> innerₕ(dcₕ * u, v))
```

Runs serially or across threads following `form.trial_space`'s backend
[`execution_policy`](@ref), the same as [`assemble!`](@ref) — [`Serial`](@ref) by default.
Optional `dirichlet_labels` applies boundary conditions to the matrix; `dirichlet_components`
restricts which leaf(-ves) of a composite trial space they bind to — see
[`dirichlet_bc!`](@ref).
"""
function assemble(form::BilinearForm; dirichlet_labels = nothing, dirichlet_components = nothing)
    _validate_dirichlet_labels(dirichlet_labels)
    ast_resolved = form.ast
    A = allocate_system_matrix(form, ast_resolved)
    assemble!(A, form; dirichlet_labels = dirichlet_labels,
        dirichlet_components = dirichlet_components, ast = ast_resolved)
    return A
end

# ==============================================================================
# Helper Cores for Function Barrier Optimization
# ==============================================================================

# The scalar case: one block, no offsets.
function _assemble_bilinear_core!(A::SparseMatrixCSC, trial_space, test_space,
        ast::AST_TYPE) where {AST_TYPE}
    _check_block_meshes(ast, trial_space, test_space)
    _scatter_block!(A, ast, test_space, 0, 0)
    return A
end

# One term into one block, serially. `row_offset` comes from the test leaf and `col_offset`
# from the trial leaf: a matrix row is indexed by the test function.
function _scatter_block!(A::SparseMatrixCSC, term::TERM, sp, row_offset::Int,
        col_offset::Int) where {TERM}
    Ωₕ = mesh(sp)
    mesh_markers = markers(Ωₕ)
    lin_indices = LinearIndices(indices(Ωₕ))

    @inbounds for I in indices(Ωₕ)
        stencil = local_stencil(term, sp, I, mesh_markers, lin_indices[I])

        for (off_u, off_v, weight) in stencil
            Iv = I + CartesianIndex(off_v)
            col = _trial_column(lin_indices, I, off_u)

            if checkbounds(Bool, lin_indices, Iv) && col != 0
                add_to_sparse!(A, lin_indices[Iv] + row_offset, col + col_offset, weight)
            end
        end
    end
    return A
end

# Walk the sum and send each term to the blocks it belongs to. The same shape as
# `_route_terms!` on the vector side, and for the same reason: recursing keeps each term
# concretely typed at its own call, where flattening into a vector makes every one a dynamic
# read.
#
# A term naming neither side goes to the diagonal blocks, since `Σᵢ innerₕ(uᵢ, vᵢ)` is block
# diagonal and not full. A term naming both goes to one block, off-diagonal included — which
# is what the version this replaces could not do: it walked the top-level components with a
# single offset for row and column, so off-diagonal terms had nowhere to land and
# `add_to_sparse!` dropped them in silence. Recursion shape shared via `_visit_operator_add2`
# (form/common.jl).
function _assemble_blocks!(A::SparseMatrixCSC, op::OperatorAdd, trial_leaves, test_leaves)
    _visit_operator_add2(
        _assemble_blocks!, A, op, trial_leaves, test_leaves)
end

function _assemble_blocks!(A::SparseMatrixCSC, term::TERM, trial_leaves,
        test_leaves) where {TERM}
    blk = block_of(term, length(trial_leaves), length(test_leaves))

    if blk === nothing
        for c in 1:min(length(trial_leaves), length(test_leaves))
            _check_block_meshes(term, first(trial_leaves[c]), first(test_leaves[c]))
            _scatter_block!(A, term, first(test_leaves[c]), last(test_leaves[c]),
                last(trial_leaves[c]))
        end
        return A
    end

    tc, sc = blk
    _check_block_meshes(term, first(trial_leaves[tc]), first(test_leaves[sc]))
    _scatter_block!(A, term, first(test_leaves[sc]), last(test_leaves[sc]),
        last(trial_leaves[tc]))
    return A
end

function _assemble_bilinear_core!(A::SparseMatrixCSC, trial_space::CompositeGridSpace,
        test_space::CompositeGridSpace, ast::AST_TYPE) where {AST_TYPE}
    _assemble_blocks!(A, ast, leaf_spaces_offsets(trial_space),
        leaf_spaces_offsets(test_space))
    return A
end

# The safe stride for a matrix assembly, read from a sample stencil.
#
# A bilinear stencil writes to `(I + off_v, I + off_u)`, so two points collide on an entry
# only if their *row* footprints overlap: rows disjoint implies entries disjoint whatever the
# columns do. So the span of the test-side offsets is enough, which is the same quantity
# `_colour_strides` computes for a vector assembly.
#
# Taken from an evaluated stencil rather than from `stencil_offsets`, which refuses a
# `BilinearProduct` on purpose: its offsets are pairs, and what is wanted here is one side.
function _bilinear_colour_strides(ast::AST_TYPE, sp, ::Val{D}) where {AST_TYPE, D}
    Ωₕ = mesh(sp)
    grid_inds = indices(Ωₕ)
    lin_indices = LinearIndices(grid_inds)
    I = grid_inds[length(grid_inds) ÷ 2 + 1]
    stencil = local_stencil(ast, sp, I, markers(Ωₕ), lin_indices[I])
    isempty(stencil) && return ntuple(_ -> 1, D)

    lo = stencil[1][2]
    hi = stencil[1][2]
    for (_, off_v, _) in stencil
        lo = min.(lo, off_v)
        hi = max.(hi, off_v)
    end
    return hi .- lo .+ 1
end

# One colour, threaded, writing straight into the matrix. The colouring is what makes that
# safe: `add_to_sparse!` searches a column and updates in place, so two threads landing on
# one entry would race on the value.
@noinline function _sweep_bilinear_colour!(A::SparseMatrixCSC, sp, term::TERM, idxs,
        lin_indices, mesh_markers, row_offset::Int, col_offset::Int) where {TERM}
    Threads.@threads for I in idxs
        stencil = local_stencil(term, sp, I, mesh_markers, lin_indices[I])

        for (off_u, off_v, weight) in stencil
            Iv = I + CartesianIndex(off_v)
            col = _trial_column(lin_indices, I, off_u)

            if checkbounds(Bool, lin_indices, Iv) && col != 0
                add_to_sparse!(A, lin_indices[Iv] + row_offset, col + col_offset, weight)
            end
        end
    end
    return nothing
end

# Every colour in turn, as strided sub-grids rather than a materialised list of indices. The
# version this replaces binned the whole grid into a `Vector{Vector{CartesianIndex}}` at
# construction — 9.3 MB at 90,000 degrees of freedom, before a single entry was assembled.
function _sweep_bilinear!(A::SparseMatrixCSC, sp, term::TERM, strides, row_offset::Int,
        col_offset::Int) where {TERM}
    Ωₕ = mesh(sp)
    grid_inds = indices(Ωₕ)
    lin_indices = LinearIndices(grid_inds)
    mesh_markers = markers(Ωₕ)

    if prod(strides) == 1
        _sweep_bilinear_colour!(A, sp, term, grid_inds, lin_indices, mesh_markers,
            row_offset, col_offset)
        return A
    end

    for c in CartesianIndices(strides)
        _sweep_bilinear_colour!(A, sp, term, _colour_subgrid(grid_inds, c, strides),
            lin_indices, mesh_markers, row_offset, col_offset)
    end
    return A
end

# The threaded counterpart of `_assemble_blocks!`, term outside and grid inside for the same
# reason the vector side is: routing per point redoes the component walk at every one.
# Recursion shape shared via `_visit_operator_add2` (form/common.jl).
function _assemble_blocks_parallel!(A::SparseMatrixCSC, op::OperatorAdd, trial_leaves,
        test_leaves, dim_val::Val)
    _visit_operator_add2(
        _assemble_blocks_parallel!, A, op, trial_leaves, test_leaves, dim_val)
end

function _assemble_blocks_parallel!(A::SparseMatrixCSC, term::TERM, trial_leaves,
        test_leaves, dim_val::Val) where {TERM}
    blk = block_of(term, length(trial_leaves), length(test_leaves))

    if blk === nothing
        for c in 1:min(length(trial_leaves), length(test_leaves))
            _check_block_meshes(term, first(trial_leaves[c]), first(test_leaves[c]))
            sp = first(test_leaves[c])
            _sweep_bilinear!(A, sp, term, _bilinear_colour_strides(term, sp, dim_val),
                last(test_leaves[c]), last(trial_leaves[c]))
        end
        return A
    end

    tc, sc = blk
    _check_block_meshes(term, first(trial_leaves[tc]), first(test_leaves[sc]))
    sp = first(test_leaves[sc])
    _sweep_bilinear!(A, sp, term, _bilinear_colour_strides(term, sp, dim_val),
        last(test_leaves[sc]), last(trial_leaves[tc]))
    return A
end

function _assemble_bilinear_parallel_core!(A::SparseMatrixCSC, trial_space, test_space,
        ast::AST_TYPE, dim_val::Val) where {AST_TYPE}
    _check_block_meshes(ast, trial_space, test_space)
    _sweep_bilinear!(A, test_space, ast,
        _bilinear_colour_strides(ast, test_space, dim_val), 0, 0)
    return A
end

function _assemble_bilinear_parallel_core!(A::SparseMatrixCSC,
        trial_space::CompositeGridSpace, test_space::CompositeGridSpace,
        ast::AST_TYPE, dim_val::Val) where {AST_TYPE}
    _assemble_blocks_parallel!(A, ast, leaf_spaces_offsets(trial_space),
        leaf_spaces_offsets(test_space), dim_val)
    return A
end

"""
    assemble!(A::SparseMatrixCSC, form::BilinearForm; dirichlet_labels = nothing, dirichlet_components = nothing, ast = form.ast)

Assembles the `BilinearForm` into the preallocated sparse matrix `A`, allocating nothing (**0 bytes**).

Runs serially or across threads following `form.trial_space`'s backend
[`execution_policy`](@ref) — [`Serial`](@ref) (the default) or [`Parallel`](@ref).
[`assemble_parallel!`](@ref) is a separate, lower-level entry point that always threads,
ignoring the backend's policy.

By default `assemble!` uses the pre-resolved `form.ast` stored inside the form.

## Live coefficients
- Grid functions: the stored AST retains references to source `VectorElement` storage. Mutating values in-place (`Rₕ!(cₕ, ...)` or `values(cₕ) .= ...`) between steps automatically updates the matrix entries with 0 allocations.
- Dynamic scalars: plain numbers work directly for constant scalars. To update a scalar dynamically across loop iterations, wrap it in a `Ref(val)` (e.g. `β = Ref(1.0); a = form(Wₕ, Wₕ, (u, v) -> innerₕ(β * D₋ₓ(u), D₋ₓ(v)))`). Mutating `β[] = new_val` evaluates live during assembly with 0 allocations.
"""
function assemble!(
        A::SparseMatrixCSC, form::BilinearForm{D, TrialSpace, TestSpace, AST};
        dirichlet_labels = nothing,
        dirichlet_components = nothing,
        ast = form.ast) where {D, TrialSpace, TestSpace, AST}
    _validate_dirichlet_labels(dirichlet_labels)
    fill!(nonzeros(A), zero(eltype(nonzeros(A))))

    if execution_policy(form.trial_space) isa Serial
        _assemble_bilinear_core!(A, form.trial_space, form.test_space, ast)
    else
        _assemble_bilinear_parallel_core!(A, form.trial_space, form.test_space, ast, Val(D))
    end

    apply_dirichlet_labels!(A, form, dirichlet_labels, dirichlet_components)
    return A
end

"""
    assemble_parallel!(A::SparseMatrixCSC, form::BilinearForm, ast = resolve_form_ast(form)) -> A

Refills `A` with the assembled `form` across threads, and returns it, regardless of
`form.trial_space`'s backend policy — a lower-level entry point than [`assemble!`](@ref) for
forcing a threaded sweep explicitly (benchmarking, or a one-off forced comparison). `A` must
already carry the right sparsity pattern, from [`allocate_system_matrix`](@ref) or a previous
[`assemble`](@ref). Unlike `assemble!`, does not apply `dirichlet_labels`.

Colouring is what makes this correct rather than merely fast. Filling an entry goes through a
column search and an in-place update, so two threads landing on the same entry would race on
the *value*, not just on the structure.

A matrix colours on the **test side alone**. A bilinear stencil writes to
`(I + off_v, I + off_u)`, so two points collide on an entry only if their row footprints
overlap — rows disjoint implies entries disjoint, whatever the columns do — which is the same
span a vector assembly uses.

Takes `ast` positionally, matching the vector form and unlike the keyword on
[`assemble!`](@ref).
"""
function assemble_parallel!(
        A::SparseMatrixCSC, form::BilinearForm{D, TrialSpace, TestSpace, AST},
        ast = form.ast) where {D, TrialSpace, TestSpace, AST}
    fill!(nonzeros(A), zero(eltype(nonzeros(A))))

    _assemble_bilinear_parallel_core!(
        A, form.trial_space, form.test_space, ast, Val(D))

    return A
end
