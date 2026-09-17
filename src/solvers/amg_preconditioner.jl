# amg_preconditioner.jl
#
# `amg_preconditioner`: algebraic multigrid hierarchies (smoothed aggregation or classical
# Ruge-Stüben) for the symmetric positive-definite matrices Bramble's SBP forms assemble.
# Implemented in `BrambleAlgebraicMultigridExt`, same underscored-fallback idiom as
# `ode_problem`/`linear_problem` (form/semidiscrete_problems.jl) and `nonlinear_problem`
# (form/nonlinear_problem.jl): the extension's method narrows on `A::AbstractMatrix`, a
# strict specialisation of this file's `::Any` fallback.
#
# `amg_preconditioner` returns the `MultiLevel` hierarchy itself, not a ready preconditioner:
# `AlgebraicMultigrid.aspreconditioner` is what turns a hierarchy into the object with
# `ldiv!` that `LinearSolve`'s `Pl`/`Pr` expect, and a caller may want the bare hierarchy (to
# inspect the coarsening, or build more than one preconditioner from one setup) without
# always paying for that wrapping. `_amg_operator` below is the thin wrapper that does call
# `aspreconditioner`, kept separate so that `BrambleSciMLExt`'s `preconditioner = :amg`
# reaches AMG through this package's own dispatch instead of depending on
# `AlgebraicMultigrid` itself.

"""
    amg_preconditioner(A::AbstractMatrix; method = :smoothed_aggregation, kwargs...) -> MultiLevel
    amg_preconditioner(a::BilinearForm; method = :smoothed_aggregation, dirichlet = nothing,
                        dirichlet_components = nothing, kwargs...) -> MultiLevel

Build an algebraic multigrid hierarchy for the symmetric positive-definite matrix `A` -- or
for `a` assembled with the given Dirichlet conditions -- ready to turn into a preconditioner
with `AlgebraicMultigrid.aspreconditioner`.

The elliptic SBP forms Bramble assembles (Poisson, variable-coefficient diffusion, Helmholtz)
have condition numbers scaling as `O(h^-2)`: a direct sparse solve suffers severe fill-in
past a few hundred thousand degrees of freedom, and an unpreconditioned Krylov method needs
`O(h^-1)` iterations. Algebraic multigrid builds its coarse-grid hierarchy from the graph of
`A` alone, giving a preconditioned Krylov method grid-independent, `O(1)` iteration counts
instead.

# Keywords
- `method`: `:smoothed_aggregation` (default) or `:ruge_stuben`, `AlgebraicMultigrid`'s two
  hierarchy constructions.
- Every other keyword forwards to the chosen `AlgebraicMultigrid` constructor.

The `BilinearForm` method matches `assemble(a::BilinearForm; ...)`'s own default and does not
symmetrize -- Dirichlet rows become `eₖ`, but the matching columns are left alone, so `A` is
not exactly symmetric even though the underlying operator is. AMG still builds a usable
hierarchy from it, but for the SPD matrix the theory assumes, assemble through
[`assemble`](@ref)`(a, l; symmetrize = true)` and pass that matrix to the `AbstractMatrix`
method instead.

Requires [AlgebraicMultigrid.jl](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl);
call `using AlgebraicMultigrid` before calling this function.

# Examples

```julia
using AlgebraicMultigrid

A = assemble(a; dirichlet = :boundary)
ml = amg_preconditioner(A)
P = aspreconditioner(ml)
sol = solve(LinearProblem(A, F), KrylovJL_CG(); Pl = P)

# Or, directly from the form:
uₕ = solve(a, l; dirichlet = bcs, preconditioner = :amg, solver = KrylovJL_CG())
```

See also [`linear_problem`](@ref), [`assemble`](@ref).
"""
function amg_preconditioner(A::AbstractMatrix; kwargs...)
    return _amg_preconditioner(A; kwargs...)
end

function amg_preconditioner(
        a::BilinearForm; dirichlet = nothing, dirichlet_components = nothing, kwargs...
)
    A = assemble(a; dirichlet = dirichlet, dirichlet_components = dirichlet_components)
    return _amg_preconditioner(A; kwargs...)
end

function _amg_preconditioner(::Any; kwargs...)
    return error(
        "amg_preconditioner requires AlgebraicMultigrid.jl. Add `using AlgebraicMultigrid` " *
        "before calling this function.",
    )
end

function _amg_operator(::Any; kwargs...)
    return error(
        "preconditioner = :amg requires AlgebraicMultigrid.jl. Add `using AlgebraicMultigrid` " *
        "before calling this function.",
    )
end
