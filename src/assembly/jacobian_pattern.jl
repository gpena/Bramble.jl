# jacobian_pattern.jl
#
# Sparsity pattern of a Newton residual's Jacobian, read off a BilinearForm's AST -- no AD
# tracing.
#
# The residual in the nonlinear worked examples (docs/src/examples/poisson_nonlinear.jl,
# coupled_reaction_diffusion.jl) has the shape `A(u) * u - F`, where `a` (the `BilinearForm`
# that assembles `A`) has a `GridFunctionScale` coefficient computed *outside* the AST, e.g.
# `αvals = α.(Mₕ(uₕ))`. `∂residual/∂u` therefore has two contributions at every entry `a`'s
# own pattern reaches: `A` acting on the explicit `u` (that's `a`'s own sparsity, already
# exact -- see allocate_system_matrix), plus the chain-rule term through `αvals`'s own
# dependence on `u`. `local_stencil`'s `GridFunctionScale` case (form/common.jl) reads that
# coefficient as `grid_fn[lin_idx]` -- the SAME point `I` the assembly loop is currently at,
# not shifted by the term's own offsets -- so the second contribution's reach is exactly `I`
# widened by whatever stencil op the coefficient was itself built from (e.g. Mₕ's `{0,-1}`),
# at every row the term reaches from `I`. That composition is what this file adds.
#
# Left out of the matrix-type seam: both methods below always
# return a `SparseMatrixCSC{Bool}`, regardless of `a`'s own backend. A Jacobian sparsity
# pattern for `ADTypes.AbstractSparsityDetector` is a sparsity pattern, not a system matrix
# the backend controls, and neither method here ever reads `matrix_type(backend(...))` --
# they only walk `a`'s AST and push `(row, col)` pairs into `sparse!`. A dense-backend form
# is expected to produce the same nonzero *count* as CSC, not the same matrix type.

# Whether an earlier stencil entry already named this row offset -- so the coefficient's own
# reach is added once per point per distinct row, not once per (off_u, off_v) pair.
@inline function _row_offset_seen_before(stencil, k::Int, off_v)
    @inbounds for l in 1:(k - 1)
        stencil[l][2] == off_v && return true
    end
    return false
end

# `dep(U)` may return a single `LazyOp` (e.g. `Mₕ(U)` in 1D) or a `D`-tuple of them (e.g.
# `∇ₕ(U)` in D > 1); normalized to always iterate as a tuple.
@inline _as_op_tuple(result::Tuple) = result
@inline _as_op_tuple(result) = (result,)

# Every node the dependencies return, evaluated once against a fresh symbolic trial
# placeholder and flattened across dependencies and across whatever tuple a
# multi-dimensional stencil op (`∇ₕ`, `Mₕ` in D > 1) returns.
_dependency_nodes(::Tuple{}, U) = ()
function _dependency_nodes(deps::Tuple, U)
    return (_as_op_tuple(first(deps)(U))..., _dependency_nodes(Base.tail(deps), U)...)
end

# The split between the two kinds of dependency, decided by each node's type alone
# (`_has_trial_interp`, operators/interpolation.jl). A *relative* node's reach is a set of
# offsets from the walked point, read on the walked mesh. A *point-dependent* node carries
# a `πₕ`: which columns it reads depends on where the walked point falls on the source mesh,
# so it is evaluated at each point instead (`_push_point_dependencies!`, below).
_relative_nodes(::Tuple{}) = ()
function _relative_nodes(ops::Tuple)
    rest = _relative_nodes(Base.tail(ops))
    return _has_trial_interp(first(ops)) ? rest : (first(ops), rest...)
end

_point_dependent_nodes(::Tuple{}) = ()
function _point_dependent_nodes(ops::Tuple)
    rest = _point_dependent_nodes(Base.tail(ops))
    return _has_trial_interp(first(ops)) ? (first(ops), rest...) : rest
end

# The union of every relative node's reach -- the same offsets `stencil_offsets` already
# gives any stencil op, since a coefficient dependency is written exactly the way a form
# term is (a function of `U`).
function _coefficient_offsets(::Val{D}, ops::Tuple) where {D}
    offs = NTuple{D, Int}[]
    for op in ops
        for o in stencil_offsets(op)
            o in offs || push!(offs, o)
        end
    end
    return sort!(offs)
end

"""
    jacobian_pattern(a::BilinearForm, coefficient_dependencies::Function...) -> SparseMatrixCSC{Bool}

Sparsity pattern of the Jacobian of a Newton residual `A(u) * u - F`, where `a` is the
[`BilinearForm`](@ref) that assembles `A(u)` and each function in
`coefficient_dependencies` names, symbolically, the stencil operator one of `a`'s live
coefficients was itself computed from -- written the same way a form term names an
operator, as a function of the trial placeholder. If `a`'s diffusion coefficient was built
as `αvals = α.(Mₕ(uₕ))`, pass `U -> Mₕ(U)`.

Derived entirely from `a`'s AST and each dependency's own stencil reach -- no AD
tracing, no coefficient values needed. A safe superset of the exact pattern (a pointwise
nonlinear `α` can never narrow the reach its argument already has), suitable for
[`ADTypes.KnownJacobianSparsityDetector`](https://github.com/SciML/ADTypes.jl):

```julia
using Bramble: jacobian_pattern
a = form(Wₕ, Wₕ, (U, V) -> inner₊(αvals * ∇ₕ(U), ∇ₕ(V)))   # αvals = α.(Mₕ(uₕ))
pattern = jacobian_pattern(a, U -> Mₕ(U))
sparse_ad = AutoSparse(AutoForwardDiff();
    sparsity_detector = KnownJacobianSparsityDetector(pattern),
    coloring_algorithm = GreedyColoringAlgorithm())
```

## Composite trial/test spaces

A dependency may also name a *different* leaf, the same way a form term does --
`U -> U(2)` for a coefficient that is component 2's own value (no stencil op),
or `U -> Mₕ(U(2))` for one built from a stencil op applied to that other component.
`nothing` named (`U -> Mₕ(U)`, no `(k)`) means the coefficient depends on *this block's
own* trial leaf, exactly like the non-composite case above. Every dependency still applies
to every block the walk visits, whichever leaf it names -- a safe superset stays safe
however many blocks end up seeing an entry they did not strictly need.

```julia
using Bramble: jacobian_pattern
# a = form(Vₕ, Vₕ, (p, q) -> inner₊(∇ₕ(p(1)), ∇ₕ(q(1))) + innerₕ(v_c * p(1), q(1)) +
#                            inner₊(∇ₕ(p(2)), ∇ₕ(q(2))) - innerₕ(u_c * p(2), q(2)))
pattern = jacobian_pattern(a, U -> U(2), U -> U(1))   # block (1,1) reads U(2), (2,2) reads U(1)
```

## Dependencies through πₕ

A dependency like the ones above is read at offsets from the point being walked, so the
leaf it names must be discretised on the mesh the term is walked on. When the coefficient
lives on that mesh but was computed from a trial function on another one, the dependency
says how, through [`πₕ`](@ref): `U -> πₕ(U)` for `cvals = α.(πₕ(Wₐ, uₕ))`,
`U -> Mₕ(πₕ(U))` for `cvals = α.(Mₕ(πₕ(Wₐ, uₕ)))`, and `U -> πₕ(U(2))` for a coefficient
interpolated from component 2 of a composite trial space. At each walked point the pattern
then adds, to every row the term reaches there, the source-mesh columns that composition
reads: the corners of the cell the point falls in, and for an operator applied outside
`πₕ` the corners at each neighbour that operator reaches. A dependency naming a leaf on
another mesh without `πₕ` (`U -> Mₕ(U)`) is refused with an `ArgumentError`.
So is one naming a component the trial space does not have (`U -> U(3)` on `W × W`).

```julia
using Bramble: jacobian_pattern
# uₕ ∈ Wᵦ, cvals = α.(Mₕ(πₕ(Wₐ, uₕ))) ∈ Wₐ
a = form(Wᵦ, Wₐ, (u, v) -> inner₊(cvals * ∇ₕ(πₕ(u)), ∇ₕ(v)))
pattern = jacobian_pattern(a, U -> Mₕ(πₕ(U)))
```
"""
function jacobian_pattern(
        form::BilinearForm{D, TrialSpace, TestSpace, AST}, coefficient_dependencies::Function...
) where {D, TrialSpace, TestSpace, AST}
    # A composite space on either side is walked block by block, as `assemble` walks it.
    _is_block_pair(form.trial_space, form.test_space) &&
        return _jacobian_pattern_blocks(form, coefficient_dependencies...)
    ast = _bind_interp_spaces(form.ast, form.trial_space, form.test_space)
    _check_block_meshes(ast, form.trial_space, form.test_space)
    space = _walked_leaf(ast, form.trial_space, form.test_space)
    Ωₕ = mesh(space)
    _validate_term_markers(ast, markers(Ωₕ), "the form's space")
    bound, mesh_markers = _bind_walk(ast, Ωₕ)
    lin_indices = LinearIndices(indices(Ωₕ))

    nodes = _dependency_nodes(coefficient_dependencies, TrialFunction{D}())
    coeff_offsets = _coefficient_offsets(Val(D), _relative_nodes(nodes))
    isempty(coeff_offsets) || _same_mesh_or_throw(nothing, form.trial_space, Ωₕ)
    point_deps = map(_point_dependent_nodes(nodes)) do op
        return _bind_point_dependency(op, nothing, form.trial_space, 0, Ωₕ)
    end

    I_vec = Int[]
    J_vec = Int[]
    hint = _pattern_size_hint(bound, space, mesh_markers, lin_indices) *
           (1 + length(coeff_offsets))
    sizehint!(I_vec, hint)
    sizehint!(J_vec, hint)
    _scalar_jacobian_walk!(
        I_vec, J_vec, bound, space, mesh_markers, lin_indices, coeff_offsets, point_deps
    )

    n = ndofs(form.test_space)
    m = ndofs(form.trial_space)
    return sparse!(I_vec, J_vec, fill(true, length(I_vec)), n, m, |)
end

# The scalar path's point walk, behind a function barrier: `jacobian_pattern` is not
# specialised on its `Function...` arguments, so the bound point-dependent nodes are only
# concretely typed from this call on, and the walk dispatches statically at every point.
function _scalar_jacobian_walk!(
        I_vec, J_vec, ast, space, mesh_markers, lin_indices, coeff_offsets, point_deps
)
    @inbounds for I in CartesianIndices(lin_indices)
        lin_idx = lin_indices[I]
        stencil = local_stencil(ast, space, I, mesh_markers, lin_idx)

        for k in eachindex(stencil)
            off_u, off_v, _ = stencil[k]
            _offsets_seen_before(stencil, k, off_u, off_v) && continue

            row = _test_row(lin_indices, I, off_v)
            col = _trial_column(lin_indices, I, off_u)
            (row == 0 || col == 0) && continue
            push!(I_vec, row)
            push!(J_vec, col)
        end

        _push_point_dependencies!(
            I_vec, J_vec, point_deps, stencil, lin_indices, I, 0, space, mesh_markers, lin_idx
        )
        isempty(coeff_offsets) && continue

        for k in eachindex(stencil)
            off_u, off_v, _ = stencil[k]
            _row_offset_seen_before(stencil, k, off_v) && continue

            row = _test_row(lin_indices, I, off_v)
            row == 0 && continue

            for δ in coeff_offsets
                Ic = I + CartesianIndex(δ)
                checkbounds(Bool, lin_indices, Ic) || continue
                push!(I_vec, row)
                push!(J_vec, lin_indices[Ic])
            end
        end
    end
    return nothing
end

# --- composite trial/test spaces --------------------------------------------------- #
#
# A dependency op's own reach (`stencil_offsets`) is unchanged by which leaf it names --
# `Mₕ(U(2))`'s reach is exactly `Mₕ(U)`'s, since `component` only replaces the leaf type,
# never wraps it in anything `stencil_offsets` sees. What is new is *where* that reach
# lands: `trial_component_or_nothing` (form/block_extract.jl) reads which leaf a resolved
# op names, `nothing` meaning "the block currently being widened, not a different one."

# One resolved dependency: which leaf it targets (`nothing` = the block's own trial leaf)
# paired with its own stencil reach.
const _DependencyOp{D} = Tuple{Union{Int, Nothing}, Vector{NTuple{D, Int}}}

# Every relative node, one entry per node, not unioned by target, so a point-anchored
# composition (below) can pull each entry's own `lin_indices`/`col_offset` independently.
function _resolve_dependency_ops(::Val{D}, ops::Tuple) where {D}
    entries = _DependencyOp{D}[]
    for op in ops
        push!(entries, (trial_component_or_nothing(op), stencil_offsets(op)))
    end
    return entries
end

@noinline function _throw_cross_leaf_dependency_mesh(target::Union{Int, Nothing})
    named = target === nothing ? "the term's own trial function" : "component $target"
    throw(
        ArgumentError(
        "a coefficient dependency named $named, whose leaf is discretised on another mesh " *
        "than the one the term is walked on. A dependency without πₕ is read at offsets " *
        "from the walked point, which name nothing on another mesh. If the coefficient was " *
        "computed by interpolating onto the walked mesh, say so: wrap the trial function " *
        "in πₕ, as in `U -> πₕ(U)` or `U -> Mₕ(πₕ(U(2)))`.",
    ),
    )
end

# A dependency read at offsets from the walked point names columns of `leaf` only when the
# two are the same mesh: the same object, or one with the same points. Equal point counts
# are not enough, since an offset then names the right column index at the wrong place.
# Once per block, so comparing the points costs nothing that matters.
@inline function _same_mesh_or_throw(target, leaf, Ωₕ)
    Ωleaf = mesh(leaf)
    Ωleaf === Ωₕ ||
        (npoints(Ωleaf, Tuple) == npoints(Ωₕ, Tuple) &&
         host_points(Ωleaf) == host_points(Ωₕ)) ||
        _throw_cross_leaf_dependency_mesh(target)
    return nothing
end

@noinline function _throw_dependency_component(target::Int, ncomponents::Int)
    throw(
        ArgumentError(
        "a coefficient dependency named component $target of the trial space, which has " *
        "$ncomponents components.",
    ),
    )
end

# The trial leaf and column offset of component `target`, refused with an `ArgumentError`
# when the trial space has no such component.
@inline function _dependency_component(trial_leaves, target::Int)
    1 <= target <= length(trial_leaves) ||
        _throw_dependency_component(target, length(trial_leaves))
    return trial_leaves[target]
end

# `nothing` -> the block's own trial leaf. An explicit component -> that leaf, looked up
# from `trial_leaves`. Either way the leaf must share the walked mesh `Ωₕ`: the offsets are
# taken from the walked point.
function _dependency_leaf(
        target::Union{Int, Nothing}, trial_leaves, own_leaf, own_col_offset, Ωₕ
)
    leaf_space, leaf_col_offset = target === nothing ? (own_leaf, own_col_offset) :
                                  _dependency_component(trial_leaves, target)
    _same_mesh_or_throw(target, leaf_space, Ωₕ)
    return (LinearIndices(indices(mesh(leaf_space))), leaf_col_offset)
end

# --- Dependencies through πₕ -------------------------------------------------------- #
#
# A point-dependent node is bound to the leaf its `πₕ` reads (`_bind_interp_spaces`, exactly
# as a term's own interpolation is bound per block) and evaluated with `local_stencil` at
# each walked point, which is how the assembly evaluates an interpolated trial term: its
# entries name `AbsoluteColumn`s on that leaf, the corners `locate_cell` finds, and an
# operator composed outside (`Mₕ(πₕ(U))`) re-evaluates the interpolation at each neighbour
# it reaches (`stencil_shift_trait`). A node mixing `πₕ` with a bare trial function
# (`πₕ(U) + U`) also carries relative offsets, so it still needs the leaf on the walked mesh.
function _bind_point_dependency(op, target, leaf, col_offset, Ωₕ)
    _all_trial_interpolated(op) || _same_mesh_or_throw(target, leaf, Ωₕ)
    bound = _bind_interp_spaces(op, leaf, leaf)
    return (bound, LinearIndices(indices(mesh(leaf))), col_offset)
end

# Every column each bound point-dependent node reads at `I`, added to every row the term's
# own `stencil` reaches from `I`. One call per node, recursing on the tuple, so each node's
# stencil is concretely typed.
@inline _push_point_dependencies!(I_vec, J_vec, ::Tuple{}, args...) = nothing
@inline function _push_point_dependencies!(
        I_vec, J_vec, deps::Tuple, stencil, lin_indices, I, row_offset, space, markers,
        lin_idx
)
    _push_point_dependency!(
        I_vec, J_vec, first(deps)..., stencil, lin_indices, I, row_offset, space, markers,
        lin_idx
    )
    return _push_point_dependencies!(
        I_vec, J_vec, Base.tail(deps), stencil, lin_indices, I, row_offset, space, markers,
        lin_idx
    )
end

# Whether a node carries data -- a grid-function coefficient, a scalar (possibly a `Ref`
# the caller changes later) or a source -- that `local_stencil` folds into its weights.
# Decided by the node's type alone.
_carries_data(::LazyOp) = false
_carries_data(op::UnaryWrapper) = _carries_data(op.inner_op)
_carries_data(::GridFunctionScale) = true
_carries_data(::OperatorScale) = true
_carries_data(::Union{SourceFunction, SourceVector, SourceConstant, DiracSource}) = true
_carries_data(op::OperatorAdd) = _carries_data(op.left_op) || _carries_data(op.right_op)

# An entry whose weight is zero for geometric reasons alone: a corner `locate_cell` names
# but the blend does not read (a walked point on a source node), or a tap an operator masks
# off. Dropping it keeps the pattern a safe superset. In a node carrying data, a zero weight
# may be a value that changes after the pattern is built (`Mₕ(g * πₕ(U))` while `g == 0`),
# so every entry of such a node is kept.
@inline _structural_zero(op, w) = !_carries_data(op) && iszero(w)

function _push_point_dependency!(
        I_vec, J_vec, op, dep_lin_indices, dep_col_offset, stencil, lin_indices, I,
        row_offset, space, markers, lin_idx
)
    dep_stencil = local_stencil(op, space, I, markers, lin_idx)
    @inbounds for k in eachindex(stencil)
        off_v = stencil[k][2]
        _row_offset_seen_before(stencil, k, off_v) && continue
        row = _test_row(lin_indices, I, off_v)
        row == 0 && continue
        for entry in dep_stencil
            _structural_zero(op, entry[2]) && continue
            col = _trial_column(dep_lin_indices, I, entry[1])
            col == 0 && continue
            push!(I_vec, row + row_offset)
            push!(J_vec, col + dep_col_offset)
        end
    end
    return nothing
end

# One term's contribution to one block, mirroring the coordinate walk (`_coord_walk!`,
# form/bilinear_pattern.jl) for the base pattern, plus the same per-point coefficient widening
# the scalar `jacobian_pattern`
# does above -- resolved once per block (not per point) into `(lin_indices, col_offset)`
# pairs, since neither depends on the grid point being visited.
function _pattern_term_jacobian!(
        I_vec::Vector{Int},
        J_vec::Vector{Int},
        term::TERM,
        trial_leaf,
        test_leaf,
        row_offset::Int,
        col_offset::Int,
        trial_leaves,
        deps::Tuple{Vector{_DependencyOp{D}}, Tuple}
) where {TERM, D}
    sp = _walked_leaf(term, trial_leaf, test_leaf)
    Ωₕ = mesh(sp)
    _validate_term_markers(term, markers(Ωₕ), "one of the composite space's leaves")
    bound, mesh_markers = _bind_walk(term, Ωₕ)
    lin_indices = LinearIndices(indices(Ωₕ))
    dep_ops, point_nodes = deps

    resolved = map(dep_ops) do (target, offsets)
        leaf_lin_indices, leaf_col_offset = _dependency_leaf(
            target, trial_leaves, trial_leaf, col_offset, Ωₕ
        )
        return (leaf_lin_indices, leaf_col_offset, offsets)
    end
    point_deps = map(point_nodes) do op
        target = trial_component_or_nothing(op)
        leaf, leaf_col_offset = target === nothing ? (trial_leaf, col_offset) :
                                _dependency_component(trial_leaves, target)
        return _bind_point_dependency(op, target, leaf, leaf_col_offset, Ωₕ)
    end

    @inbounds for I in indices(Ωₕ)
        stencil = local_stencil(bound, sp, I, mesh_markers, lin_indices[I])

        for k in eachindex(stencil)
            off_u, off_v, _ = stencil[k]
            _offsets_seen_before(stencil, k, off_u, off_v) && continue

            row = _test_row(lin_indices, I, off_v)
            col = _trial_column(lin_indices, I, off_u)
            if row != 0 && col != 0
                push!(I_vec, row + row_offset)
                push!(J_vec, col + col_offset)
            end
        end

        _push_point_dependencies!(
            I_vec, J_vec, point_deps, stencil, lin_indices, I, row_offset, sp, mesh_markers,
            lin_indices[I]
        )
        isempty(resolved) && continue

        for k in eachindex(stencil)
            off_u, off_v, _ = stencil[k]
            _row_offset_seen_before(stencil, k, off_v) && continue

            row_base = _test_row(lin_indices, I, off_v)
            row_base == 0 && continue
            row = row_base + row_offset

            for (dep_lin_indices, dep_col_offset, offsets) in resolved, δ in offsets

                Ic = I + CartesianIndex(δ)
                checkbounds(Bool, dep_lin_indices, Ic) || continue
                push!(I_vec, row)
                push!(J_vec, dep_lin_indices[Ic] + dep_col_offset)
            end
        end
    end
    return nothing
end

# Recursion shape shared via `_visit_operator_add3` (form/common.jl), the same one
# `_foreach_unit` (form/bilinear_pattern.jl) uses for the base (non-Jacobian) pattern.
function _pattern_blocks_jacobian!(
        I_vec::Vector{Int},
        J_vec::Vector{Int},
        op::OperatorAdd,
        trial_leaves,
        test_leaves,
        deps
)
    return _visit_operator_add3(
        _pattern_blocks_jacobian!, I_vec, J_vec, op, trial_leaves, test_leaves, deps
    )
end

function _pattern_blocks_jacobian!(
        I_vec::Vector{Int}, J_vec::Vector{Int}, term::TERM, trial_leaves, test_leaves, deps
) where {TERM}
    for blk in blocks(term, trial_leaves, test_leaves)
        bound = _bind_interp_spaces(term, blk.trial_leaf, blk.test_leaf)
        _check_block_meshes(bound, blk.trial_leaf, blk.test_leaf)
        _pattern_term_jacobian!(
            I_vec,
            J_vec,
            bound,
            blk.trial_leaf,
            blk.test_leaf,
            blk.row_offset,
            blk.col_offset,
            trial_leaves,
            deps
        )
    end
    return nothing
end

# Any pair `_is_block_pair` (form/bilinear_execution.jl) accepts: composite on both sides, or
# on one side with the scalar side as a one-leaf composite (`leaf_spaces_offsets`), the same
# walk `assemble` makes. The scalar path's `_walked_leaf` would pick one whole space and could
# not name the component a term reads on the composite side.
function _jacobian_pattern_blocks(
        form::BilinearForm{D}, coefficient_dependencies::Function...
) where {D}
    ast = form.ast
    trial_leaves = leaf_spaces_offsets(form.trial_space)
    test_leaves = leaf_spaces_offsets(form.test_space)

    nodes = _dependency_nodes(coefficient_dependencies, TrialFunction{D}())
    deps = (_resolve_dependency_ops(Val(D), _relative_nodes(nodes)),
        _point_dependent_nodes(nodes))

    I_vec = Int[]
    J_vec = Int[]
    _pattern_blocks_jacobian!(I_vec, J_vec, ast, trial_leaves, test_leaves, deps)

    n = ndofs(form.test_space)
    m = ndofs(form.trial_space)
    return sparse!(I_vec, J_vec, fill(true, length(I_vec)), n, m, |)
end

"""
    ast_sparsity_detector(a::BilinearForm, coefficient_dependencies::Function...)

An `ADTypes.AbstractSparsityDetector` that supplies [`jacobian_pattern`](@ref)`(a,
coefficient_dependencies...)` directly as a Newton residual's Jacobian sparsity, in place of
one detected by tracing:

```julia
using Bramble: ast_sparsity_detector
sparse_ad = AutoSparse(AutoForwardDiff();
    sparsity_detector = ast_sparsity_detector(a, U -> Mₕ(U)),
    coloring_algorithm = GreedyColoringAlgorithm())
```

Requires [ADTypes.jl](https://github.com/SciML/ADTypes.jl); call `using ADTypes` before
calling this function.
"""
function ast_sparsity_detector(a::BilinearForm, coefficient_dependencies::Function...)
    return _ast_sparsity_detector(a, coefficient_dependencies...)
end

# Errors by default, same idiom as `export_vtk`/`_export_vtk` and `metal_backend`/
# `_metal_backend`: a helpful message rather than a bare `MethodError` when the weak
# dependency has not been loaded. `BrambleSparseADExt` overrides this with the real
# implementation, gated on `ADTypes` alone -- the only package the detector interface
# (`ADTypes.AbstractSparsityDetector`/`jacobian_sparsity`) is actually defined in;
# `DifferentiationInterface` depends on `ADTypes`, so loading it loads this extension too.
#
# The first argument is `::Any` here, not `::BilinearForm`: the extension's method has to be
# a strict *specialization* of this one rather than an identical signature, or loading it
# overwrites a method during precompilation, which Julia refuses (`export_vtk`'s own
# `_export_vtk(::AbstractString, ::Any, ::Pair...)` fallback is loosened the same way).
function _ast_sparsity_detector(::Any, ::Function...)
    return error(
        "ast_sparsity_detector requires ADTypes.jl. Add `using ADTypes` before calling " *
        "this function.",
    )
end
