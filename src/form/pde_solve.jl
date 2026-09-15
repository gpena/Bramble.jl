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
# `ChainRulesCore.rrule`, and `BrambleEnzymeExt` a native `EnzymeRules` rule -- Enzyme gets
# its own rather than a bridged one because `Enzyme.@import_rrule`'s bridge corrupts the
# shadow of a sparse `A` whenever the cotangent carries an explicit zero
# (gpena/Bramble.jl#240, and this file's docstring below). `Mooncake` cannot currently be bridged
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

!!! note "Enzyme needs nothing beyond `using Enzyme`"
    `using Enzyme` is enough: `BrambleEnzymeExt` defines a native `EnzymeRules` reverse rule
    for `pde_solve`, so `Enzyme.gradient`/`Enzyme.autodiff` reach the adjoint directly.

    Do **not** call `Enzyme.@import_rrule(typeof(pde_solve), ...)`. Besides now defining a
    second rule for the same signature, that bridge is itself unsound here
    (gpena/Bramble.jl#240): merging the rrule's returned `SparseMatrixCSC` into Enzyme's
    shadow drops the cotangent's explicit zeros from `nzval` while leaving `colptr`/`rowval`
    unchanged, so the shadow stops being a well-formed sparse matrix and the gradient comes
    back wrong without any error. A homogeneous Dirichlet problem produces such a zero
    routinely, since a constrained row's solution entry is exactly its boundary value.

    A closure that captures a grid space or a form still needs `Enzyme.Const(f)` and
    `Enzyme.set_runtime_activity(Enzyme.Reverse)`, the same two annotations
    `docs/src/tutorials/autodiff.md` documents for the ordinary (non-solve) path.

!!! warning "Gradients with respect to an operator's own coefficient are not supported"
    A gradient with respect to a *Dirichlet value or source term* works for any form: `θ`
    reaches `F` through `dirichlet_bc!`'s value-writing path, never the assembly engine.

    A gradient with respect to the operator's own coefficient (`θ` scaling the bilinear form
    itself) additionally needs `Enzyme` to differentiate `assemble`'s recording pass, and
    that is not currently reliable. Two separate limits bite:

    - From a fully inferred call site, a `Union` still left in the assembly path raises
      `IllegalTypeAnalysisException`. It happens to compile when the same call is
      dynamically dispatched instead -- from a script's untyped globals, say -- which is far
      too fragile a distinction to rely on.
    - Above roughly a dozen machine words of stencil (any difference operator in 2D or 3D,
      larger sums of terms in 1D), Enzyme cannot type the mixed offset/weight tuple once it
      is passed through memory, and raises `EnzymeNoTypeError`. Raising
      `Enzyme.API.maxtypeoffset!`/`maxtypedepth!` does not move that threshold, and
      `looseTypeAnalysis!` buys compilation at the price of silently wrong answers, so it is
      not a workaround.

    Splitting the stencil's `Int` offsets from its `Float64` weights, so what Enzyme
    differentiates is uniformly typed, is the fix; it is follow-up work on
    gpena/Bramble.jl#240. Until then, use `ForwardDiff` for coefficient sensitivities, or
    differentiate with respect to boundary/source data.

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
