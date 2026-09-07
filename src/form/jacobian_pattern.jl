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

Scoped to a bilinear form over a single (non-composite) grid space: a composite trial
space's leaves can couple through *another* component (`coupled_reaction_diffusion.md`'s
`v_c * p(1)`), which needs the block/leaf routing `allocate_system_matrix`'s composite
method has and this function does not yet reuse.
"""
function jacobian_pattern(
        form::BilinearForm{D, TrialSpace, TestSpace, AST},
        coefficient_dependencies::Function...) where {D, TrialSpace, TestSpace, AST}
    TrialSpace <: CompositeGridSpace && throw(ArgumentError(
        "jacobian_pattern does not support a composite trial space yet: a coefficient " *
        "that depends on a *different* leaf (as in coupled_reaction_diffusion.md) needs " *
        "block routing this function does not do. See gpena/Bramble.jl#21."))

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
