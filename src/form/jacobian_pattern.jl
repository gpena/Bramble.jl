# jacobian_pattern.jl
#
# Sparsity pattern of a Newton residual's Jacobian, read off a BilinearForm's AST -- no AD
# tracing (see gpena/Bramble.jl#21).
#
# The residual in the nonlinear worked examples (docs/src/examples/poisson_nonlinear.md,
# coupled_reaction_diffusion.md) has the shape `A(u) * u - F`, where `a` (the `BilinearForm`
# that assembles `A`) has a `GridFunctionScale` coefficient computed *outside* the AST, e.g.
# `αvals = α.(M₋ₕ(uₕ))`. `∂residual/∂u` therefore has two contributions at every entry `a`'s
# own pattern reaches: `A` acting on the explicit `u` (that's `a`'s own sparsity, already
# exact -- see allocate_system_matrix), plus the chain-rule term through `αvals`'s own
# dependence on `u`. `local_stencil`'s `GridFunctionScale` case (form/common.jl) reads that
# coefficient as `grid_fn[lin_idx]` -- the SAME point `I` the assembly loop is currently at,
# not shifted by the term's own offsets -- so the second contribution's reach is exactly `I`
# widened by whatever stencil op the coefficient was itself built from (e.g. M₋ₕ's `{0,-1}`),
# at every row the term reaches from `I`. That composition is what this file adds.

# Whether an earlier stencil entry already named this row offset -- so the coefficient's own
# reach is added once per point per distinct row, not once per (off_u, off_v) pair.
@inline function _row_offset_seen_before(stencil, k::Int, off_v)
    @inbounds for l in 1:(k - 1)
        stencil[l][2] == off_v && return true
    end
    return false
end

# `dep(U)` may return a single `LazyOp` (e.g. `M₋ₕ(U)` in 1D) or a `D`-tuple of them (e.g.
# `∇₋ₕ(U)` in D > 1); normalized to always iterate as a tuple.
@inline _as_op_tuple(result::Tuple) = result
@inline _as_op_tuple(result) = (result,)

# The union of every dependency's reach, evaluated once against a fresh symbolic trial
# placeholder -- the same offsets `stencil_offsets` already gives any stencil op, since a
# coefficient dependency is written exactly the way a form term is (a function of `U`).
function _coefficient_offsets(::Val{D}, deps::Tuple, U) where {D}
    offs = NTuple{D, Int}[]
    for dep in deps, op in _as_op_tuple(dep(U))

        for o in stencil_offsets(op)
            o in offs || push!(offs, o)
        end
    end
    return sort!(offs)
end

"""
    jacobian_pattern(a::BilinearForm, coefficient_dependencies::Function...) -> SparseMatrixCSC{Bool}

Sparsity pattern of the Jacobian of a Newton residual `A(u) * u - F`, where `a` is the
[`BilinearForm`](@ref) that assembles `A(u)` (e.g. `diffusion_matrix` in
[the nonlinear Poisson example](examples/poisson_nonlinear.md)) and each function in
`coefficient_dependencies` names, symbolically, the stencil operator one of `a`'s live
coefficients was itself computed from -- written the same way a form term names an
operator, as a function of the trial placeholder. If `a`'s diffusion coefficient was built
as `αvals = α.(M₋ₕ(uₕ))`, pass `U -> M₋ₕ(U)`.

Derived entirely from `a`'s AST and each dependency's own stencil reach -- no AD
tracing, no coefficient values needed. A safe superset of the exact pattern (a pointwise
nonlinear `α` can never narrow the reach its argument already has), suitable for
[`ADTypes.KnownJacobianSparsityDetector`](https://github.com/SciML/ADTypes.jl):

```julia
a = form(Wₕ, Wₕ, (U, V) -> inner₊(αvals * ∇₋ₕ(U), ∇₋ₕ(V)))   # αvals = α.(M₋ₕ(uₕ))
pattern = jacobian_pattern(a, U -> M₋ₕ(U))
sparse_ad = AutoSparse(AutoForwardDiff();
    sparsity_detector = KnownJacobianSparsityDetector(pattern),
    coloring_algorithm = GreedyColoringAlgorithm())
```

## Composite trial/test spaces

A dependency may also name a *different* leaf, the same way a form term does --
`U -> U(2)` for a coefficient that is component 2's own value (no stencil op, as in
[the coupled reaction-diffusion example](examples/coupled_reaction_diffusion.md)'s `v_c`),
or `U -> M₋ₕ(U(2))` for one built from a stencil op applied to that other component.
`nothing` named (`U -> M₋ₕ(U)`, no `(k)`) means the coefficient depends on *this block's
own* trial leaf, exactly like the non-composite case above. Every dependency still applies
to every block the walk visits, whichever leaf it names -- a safe superset stays safe
however many blocks end up seeing an entry they did not strictly need.

```julia
# a = form(Vₕ, Vₕ, (p, q) -> inner₊(∇₋ₕ(p(1)), ∇₋ₕ(q(1))) + innerₕ(v_c * p(1), q(1)) +
#                            inner₊(∇₋ₕ(p(2)), ∇₋ₕ(q(2))) - innerₕ(u_c * p(2), q(2)))
pattern = jacobian_pattern(a, U -> U(2), U -> U(1))   # block (1,1) reads U(2), (2,2) reads U(1)
```
"""
function jacobian_pattern(
        form::BilinearForm{D, TrialSpace, TestSpace, AST},
        coefficient_dependencies::Function...) where {D, TrialSpace, TestSpace, AST}
    ast = form.ast
    space = form.test_space
    _check_block_meshes(ast, form.trial_space, form.test_space)
    Ωₕ = mesh(space)
    mesh_markers = markers(Ωₕ)
    _validate_term_markers(ast, mesh_markers, "the form's space")
    lin_indices = LinearIndices(indices(Ωₕ))

    coeff_offsets = _coefficient_offsets(Val(D), coefficient_dependencies, TrialFunction{D}())

    I_vec = Int[]
    J_vec = Int[]
    hint = _pattern_size_hint(ast, space, mesh_markers, lin_indices) *
           (1 + length(coeff_offsets))
    sizehint!(I_vec, hint)
    sizehint!(J_vec, hint)

    @inbounds for I in indices(Ωₕ)
        lin_idx = lin_indices[I]
        stencil = local_stencil(ast, space, I, mesh_markers, lin_idx)

        for k in eachindex(stencil)
            off_u, off_v, _ = stencil[k]
            _offsets_seen_before(stencil, k, off_u, off_v) && continue

            Iv = I + CartesianIndex(off_v)
            checkbounds(Bool, lin_indices, Iv) || continue
            col = _trial_column(lin_indices, I, off_u)
            col == 0 && continue
            push!(I_vec, lin_indices[Iv])
            push!(J_vec, col)
        end

        isempty(coeff_offsets) && continue

        for k in eachindex(stencil)
            off_u, off_v, _ = stencil[k]
            _row_offset_seen_before(stencil, k, off_v) && continue

            Iv = I + CartesianIndex(off_v)
            checkbounds(Bool, lin_indices, Iv) || continue
            row = lin_indices[Iv]

            for δ in coeff_offsets
                Ic = I + CartesianIndex(δ)
                checkbounds(Bool, lin_indices, Ic) || continue
                push!(I_vec, row)
                push!(J_vec, lin_indices[Ic])
            end
        end
    end

    n = ndofs(form.test_space)
    m = ndofs(form.trial_space)
    return sparse!(I_vec, J_vec, fill(true, length(I_vec)), n, m, |)
end

# --- composite trial/test spaces --------------------------------------------------- #
#
# A dependency op's own reach (`stencil_offsets`) is unchanged by which leaf it names --
# `M₋ₕ(U(2))`'s reach is exactly `M₋ₕ(U)`'s, since `component` only replaces the leaf type,
# never wraps it in anything `stencil_offsets` sees. What is new is *where* that reach
# lands: `trial_component_or_nothing` (form/block_extract.jl) reads which leaf a resolved
# op names, `nothing` meaning "the block currently being widened, not a different one."

# One resolved dependency: which leaf it targets (`nothing` = the block's own trial leaf)
# paired with its own stencil reach.
const _DependencyOp{D} = Tuple{Union{Int, Nothing}, Vector{NTuple{D, Int}}}

# Every `(dep(U))` node, flattened across dependencies and across whatever tuple a
# multi-dimensional stencil op (`∇₋ₕ`, `M₋ₕ` in D > 1) returns -- one entry per node, not
# unioned by target, so a point-anchored composition (below) can pull each entry's own
# `lin_indices`/`col_offset` independently.
function _resolve_dependency_ops(::Val{D}, deps::Tuple, U) where {D}
    entries = _DependencyOp{D}[]
    for dep in deps, op in _as_op_tuple(dep(U))

        push!(entries, (trial_component_or_nothing(op), stencil_offsets(op)))
    end
    return entries
end

@noinline function _throw_cross_leaf_dependency_mesh(target::Int)
    throw(ArgumentError(
        "a coefficient dependency named component $target, whose leaf does not share the " *
        "reaching term's own mesh. jacobian_pattern's composite case assumes every leaf a " *
        "dependency can name is discretised on the same mesh as the term it widens, the " *
        "same assumption allocate_system_matrix's own composite method makes."))
end

# `nothing` -> the block's own trial leaf, its own `lin_indices`/`col_offset` (already in
# hand from the block being widened). An explicit component -> that leaf's own, looked up
# from `trial_leaves`, guarded the same way `_check_block_meshes` guards a term's own leaves.
function _dependency_leaf(target::Union{Int, Nothing}, trial_leaves,
        own_lin_indices, own_col_offset, own_mesh)
    target === nothing && return (own_lin_indices, own_col_offset)
    leaf_space, leaf_col_offset = trial_leaves[target]
    leaf_mesh = mesh(leaf_space)
    npoints(leaf_mesh, Tuple) == npoints(own_mesh, Tuple) ||
        _throw_cross_leaf_dependency_mesh(target)
    return (LinearIndices(indices(leaf_mesh)), leaf_col_offset)
end

# One term's contribution to one block, mirroring `_pattern_term!` (form/bilinear.jl) for
# the base pattern, plus the same per-point coefficient widening the scalar `jacobian_pattern`
# does above -- resolved once per block (not per point) into `(lin_indices, col_offset)`
# pairs, since neither depends on the grid point being visited.
function _pattern_term_jacobian!(I_vec::Vector{Int}, J_vec::Vector{Int}, term::TERM,
        trial_leaf, test_leaf, row_offset::Int, col_offset::Int, trial_leaves,
        dep_ops::Vector{_DependencyOp{D}}) where {TERM, D}
    Ωₕ = mesh(test_leaf)
    mesh_markers = markers(Ωₕ)
    _validate_term_markers(term, mesh_markers, "one of the composite space's leaves")
    lin_indices = LinearIndices(indices(Ωₕ))
    Ωu = mesh(trial_leaf)

    resolved = map(dep_ops) do (target, offsets)
        leaf_lin_indices, leaf_col_offset = _dependency_leaf(
            target, trial_leaves, lin_indices, col_offset, Ωu)
        (leaf_lin_indices, leaf_col_offset, offsets)
    end

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

        isempty(resolved) && continue

        for k in eachindex(stencil)
            off_u, off_v, _ = stencil[k]
            _row_offset_seen_before(stencil, k, off_v) && continue

            Iv = I + CartesianIndex(off_v)
            checkbounds(Bool, lin_indices, Iv) || continue
            row = lin_indices[Iv] + row_offset

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
# `_pattern_blocks!` (form/bilinear.jl) uses for the base (non-Jacobian) pattern.
function _pattern_blocks_jacobian!(I_vec::Vector{Int}, J_vec::Vector{Int}, op::OperatorAdd,
        trial_leaves, test_leaves, dep_ops)
    _visit_operator_add3(
        _pattern_blocks_jacobian!, I_vec, J_vec, op, trial_leaves, test_leaves, dep_ops)
end

function _pattern_blocks_jacobian!(I_vec::Vector{Int}, J_vec::Vector{Int}, term::TERM,
        trial_leaves, test_leaves, dep_ops) where {TERM}
    for blk in blocks(term, trial_leaves, test_leaves)
        _check_block_meshes(term, blk.trial_leaf, blk.test_leaf)
        _pattern_term_jacobian!(I_vec, J_vec, term, blk.trial_leaf, blk.test_leaf,
            blk.row_offset, blk.col_offset, trial_leaves, dep_ops)
    end
    return nothing
end

function jacobian_pattern(
        form::BilinearForm{D, TrialSpace, TestSpace, AST},
        coefficient_dependencies::Function...) where {D, TrialSpace <: CompositeGridSpace,
        TestSpace <: CompositeGridSpace, AST}
    ast = form.ast
    trial_leaves = leaf_spaces_offsets(form.trial_space)
    test_leaves = leaf_spaces_offsets(form.test_space)

    dep_ops = _resolve_dependency_ops(Val(D), coefficient_dependencies, TrialFunction{D}())

    I_vec = Int[]
    J_vec = Int[]
    _pattern_blocks_jacobian!(I_vec, J_vec, ast, trial_leaves, test_leaves, dep_ops)

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
sparse_ad = AutoSparse(AutoForwardDiff();
    sparsity_detector = ast_sparsity_detector(a, U -> M₋ₕ(U)),
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
    error("ast_sparsity_detector requires ADTypes.jl. Add `using ADTypes` before calling " *
          "this function.")
end
