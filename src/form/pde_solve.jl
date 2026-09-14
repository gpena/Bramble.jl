# pde_solve.jl
#
# `pde_solve`: a name for `A \\ F` to hang a reverse-mode adjoint rule on.
#
# No source-level AD tool -- forward or reverse, `ForwardDiff`, `ReverseDiff`, `Mooncake`,
# `Enzyme` -- can differentiate through `\\` itself: it dispatches into compiled
# BLAS/SuiteSparse code, opaque to every one of them, the same reason none can differentiate
# through any other library call written in C or Fortran. Everything *around* the solve --
# `assemble`, `dirichlet_bc!` -- is already reverse-mode-differentiable (`docs/src/internals/
# autodiff.md`), so this one function is the entire gap between "Bramble can be forward-mode
# differentiated" (true today) and "Bramble can be used inside a gradient-based inverse
# problem or a PDE-constrained training loop without paying one factorisation per parameter"
# (what closes gpena/Bramble.jl#228).
#
# `pde_solve` itself needs no weak dependency: without `ChainRulesCore` loaded it is exactly
# `\\`, same result, same cost, just not differentiable. `BrambleChainRulesExt` supplies the
# `ChainRulesCore.rrule`; a caller on `Enzyme` additionally needs
# `Enzyme.@import_rrule(typeof(pde_solve), SparseMatrixCSC, AbstractVector)` -- Enzyme does not
# adopt an arbitrary `ChainRulesCore.rrule` merely because `ChainRulesCore` is loaded, so this
# is spelled out on both `pde_solve`'s own docstring and the rrule's, not left to be
# discovered from an `IllegalTypeAnalysisException`. `Mooncake` cannot currently be bridged
# this way at all: `Mooncake.@from_rrule`/`build_rrule` both accept the bridge without
# complaint, but the first time the rule actually *runs* it needs a real tangent value for
# the `SparseMatrixCSC` argument, and `Mooncake` has no `increment_and_get_rdata!` for that
# type combination (verified directly, not assumed) -- a real gap in `Mooncake.jl` itself,
# not something this package can route around, so it is left unsupported and documented
# rather than worked around.

"""
    pde_solve(A::SparseMatrixCSC, F::AbstractVector; solver = :default, sym = :auto, kwargs...) -> Vector
    pde_solve(fact::MUMPSFactorization, F::AbstractVector) -> Vector

Solve `A u = F` (or `fact u = F`) and return `u`.

When called with no keyword arguments (or `solver = :default`), identical to `A \\ F` --
this default path exists to provide a stable name for reverse-mode automatic differentiation
tools to attach adjoint rules to.

# Solvers
- `:default` or `:suitesparse`: standard SuiteSparse sparse direct solve (`\\`).
- `:mumps`: MUMPS multifrontal direct solver (requires [MUMPS.jl](https://github.com/lruthotto/MUMPS.jl)).

# Symmetry options (`sym`)
For `solver = :mumps`:
- `:auto` (default): automatic detection.
- `:spd`, `:definite`, or `1`: symmetric positive definite.
- `:symmetric` or `2`: general symmetric.
- `:unsymmetric` or `0`: general unsymmetric.

# Reverse-mode differentiation

Requires [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl) loaded (`using
ChainRulesCore` or any package that re-exports/bridges to it) for the
`ChainRulesCore.rrule(::typeof(pde_solve), A, F)` this package's `BrambleChainRulesExt`
defines to take effect.

For a functional `J(u)` of the solution, the rule solves the adjoint system `Aᵀ λ = ∂J/∂u`
once -- reusing the same LU factorisation the forward solve already computed, via `fact' \\ b`,
whether or not `A` is symmetric -- and returns `∂J/∂A = -λ uᵀ`, restricted to `A`'s own
sparsity pattern and never densified, and `∂J/∂F = λ`. Since `assemble`/`dirichlet_bc!` are
themselves already reverse-mode-differentiable, wrapping only this one function is enough to
differentiate an entire `θ -> assemble(a(θ), l(θ); dirichlet = θ) -> pde_solve -> J(u)` chain
end to end -- including gradients with respect to a Dirichlet boundary value.

!!! note "Enzyme needs one extra line, and ChainRulesCore loaded first"
    `Enzyme` does not automatically adopt a `ChainRulesCore.rrule`. `using ChainRulesCore`
    before `using Enzyme` (`Enzyme`'s own bridge lives in its `EnzymeChainRulesCoreExt`, a
    weak-dependency extension of `Enzyme` gated on `ChainRulesCore` being loaded -- without
    it, `Enzyme.@import_rrule` itself is undefined), then add
    `Enzyme.@import_rrule(typeof(pde_solve), SparseMatrixCSC, AbstractVector)` once, before
    calling `Enzyme.gradient`/`Enzyme.autodiff` on code that reaches `pde_solve` -- without
    it, Enzyme tries to trace into `lu`'s internals directly and raises
    `IllegalTypeAnalysisException`, not merely a slow or wrong answer.

!!! warning "Mooncake is not supported"
    `Mooncake.@from_rrule`/`build_rrule` both accept a bridge for `pde_solve` without
    complaint, but running the resulting rule fails: `Mooncake` has no
    `increment_and_get_rdata!` method to represent a `SparseMatrixCSC` cotangent, regardless
    of whether `∂J/∂A` is ever requested. This is a gap in `Mooncake.jl` itself; revisit if a
    future release adds sparse-array tangent support.

# Examples

```julia
A, F = assemble(a, l; dirichlet = :boundary => x -> 0.0)
u = pde_solve(A, F)

# Using MUMPS
using MUMPS
u_mumps = pde_solve(A, F; solver = :mumps)
```

See also [`assemble`](@ref), [`mumps_solve`](@ref), [`mumps_factorize`](@ref),
[`linear_problem`](@ref).
"""
function pde_solve(A::SparseMatrixCSC, F::AbstractVector; solver::Symbol = :default, sym = :auto, kwargs...)
    if solver === :default || solver === :suitesparse
        return A \ F
    elseif solver === :mumps
        return _mumps_solve(A, F; sym = sym, kwargs...)
    else
        throw(ArgumentError("Unknown solver: $solver. Expected :default, :suitesparse, or :mumps."))
    end
end

pde_solve(fact::MUMPSFactorization, F::AbstractVector) = fact \ F
