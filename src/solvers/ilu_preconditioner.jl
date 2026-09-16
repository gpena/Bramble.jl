# ilu_preconditioner.jl
#
# `ilu_preconditioner`: zero-fill incomplete LU (ILU(0)) preconditioning for the unsymmetric,
# convection-dominated matrices Bramble's SBP forms assemble, where algebraic multigrid does
# not degrade gracefully (gpena/Bramble.jl#244). Implemented in `BrambleILUZeroExt`, the same
# underscored-fallback idiom `amg_preconditioner` uses (solvers/amg_preconditioner.jl):
# the extension's method narrows on `A::AbstractMatrix`, a strict specialisation of this
# file's `::Any` fallback.
#
# Unlike `amg_preconditioner`, which returns a bare `MultiLevel` hierarchy that still needs
# `AlgebraicMultigrid.aspreconditioner` to gain `ldiv!`, `ILUZero.ilu0` already returns an
# `ILU0Precon` with `ldiv!` defined directly -- so `ilu_preconditioner` and the underscored
# `_ilu_operator` `BrambleSciMLExt`'s `preconditioner = :ilu0` reaches are the same call, kept
# as two names only for parity with the AMG pair.

"""
    ilu_preconditioner(A::AbstractMatrix; kwargs...) -> ILUZero.ILU0Precon
    ilu_preconditioner(a::BilinearForm; dirichlet = nothing,
                        dirichlet_components = nothing, kwargs...) -> ILUZero.ILU0Precon

Build a zero-fill incomplete LU (ILU(0)) preconditioner for the matrix `A` -- or for `a`
assembled with the given Dirichlet conditions -- ready to use directly as `LinearSolve`'s
`Pl`.

Convection-dominated forms (advection large relative to diffusion) assemble unsymmetric
matrices far from an M-matrix, where algebraic multigrid ([`amg_preconditioner`](@ref)) does
not degrade gracefully: measured on a 2D convection-diffusion system with diffusion `1e-2`
against unit advection, `ruge_stuben` AMG failed to converge in 2000 GMRES iterations while
ILU(0) converged in 18 (gpena/Bramble.jl#244). ILU(0) reuses `A`'s own sparsity pattern for
its factors, so it has no fill-in parameter to tune and is cheap to build, at the cost of a
weaker preconditioner than a tuned incomplete factorization on harder systems.

# Keywords
Forwarded to `ILUZero.ilu0`, which currently takes none beyond `A` itself.

The `BilinearForm` method matches `assemble(a::BilinearForm; ...)`'s own default and does not
symmetrize; ILU(0) does not assume symmetry, so this is the natural entry point for the
unsymmetric, convection-dominated forms this preconditioner targets.

Requires [ILUZero.jl](https://github.com/mohamed82008/ILUZero.jl); call `using ILUZero`
before calling this function.

# Examples

```julia
using ILUZero

A = assemble(a; dirichlet = :boundary)
P = ilu_preconditioner(A)
sol = solve(LinearProblem(A, F), KrylovJL_GMRES(); Pl = P)

# Or, directly from the form:
uₕ = solve(a, l; dirichlet = bcs, preconditioner = :ilu0, solver = KrylovJL_GMRES())
```

See also [`amg_preconditioner`](@ref), [`linear_problem`](@ref), [`assemble`](@ref).
"""
function ilu_preconditioner(A::AbstractMatrix; kwargs...)
    return _ilu_preconditioner(A; kwargs...)
end

function ilu_preconditioner(
        a::BilinearForm; dirichlet = nothing, dirichlet_components = nothing, kwargs...
)
    A = assemble(a; dirichlet = dirichlet, dirichlet_components = dirichlet_components)
    return _ilu_preconditioner(A; kwargs...)
end

function _ilu_preconditioner(::Any; kwargs...)
    return error(
        "ilu_preconditioner requires ILUZero.jl. Add `using ILUZero` before calling this function.",
    )
end

function _ilu_operator(::Any; kwargs...)
    return error(
        "preconditioner = :ilu0 requires ILUZero.jl. Add `using ILUZero` before calling this function.",
    )
end
