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

With no keyword arguments this is `A \\ F`, unless `AppleAccelerate.jl` is loaded on macOS (see
`:default` below), in which case it is `accelerate_solve(A, F; sym)`. The name exists so that
reverse-mode AD tools have one function to attach an adjoint rule to, which is what makes a
whole `θ -> assemble -> pde_solve -> J(u)` chain differentiable.

# Keywords
- `solver`: `:default` -- `A \\ F` on Linux, Windows, and macOS without `AppleAccelerate.jl`
  loaded; on macOS with `using AppleAccelerate` in effect, dispatches to `accelerate_solve`
  instead (same as passing `solver = :accelerate` explicitly). This only ever narrows which
  solver runs on macOS; Linux and Windows are never affected, and a matrix Accelerate cannot
  factor (e.g. non-square) throws from `accelerate_solve` exactly as `solver = :accelerate`
  would -- `:default` never silently falls back to `\\` after picking Accelerate.
  `:suitesparse` (CHOLMOD/UMFPACK), `:spqr` (sparse QR, for a least-squares or rectangular
  `A`), `:accelerate` (Apple `libSparse`, needs `AppleAccelerate.jl`), `:mumps` (needs
  `MUMPS.jl`), `:sparspak` (pure Julia, needs `Sparspak.jl`).
- `sym`: symmetry hint for `:suitesparse`, `:accelerate` and `:mumps`. `:auto` (default)
  detects it; `:spd`/`:definite`/`1`, `:symmetric`/`2` and `:unsymmetric`/`0` state it.

# Returns
- `Vector`: the solution, of the promoted element type of `A` and `F`.

# Reverse-mode differentiation

With `ChainRulesCore.jl` loaded, `BrambleChainRulesExt`'s `rrule` solves the adjoint system
`Aᵀ λ = ∂J/∂u` once, reusing the forward solve's own factorisation, and returns
`∂J/∂A = -λ uᵀ` restricted to `A`'s sparsity (never densified) and `∂J/∂F = λ`. `using Enzyme`
is enough for the same adjoint through `BrambleEnzymeExt`'s native `EnzymeRules` rule; do not
call `Enzyme.@import_rrule`, whose bridge drops the cotangent's explicit zeros from `nzval`
and returns a wrong gradient without any error. `Mooncake` cannot represent a
`SparseMatrixCSC` cotangent at all and is unsupported.

A closure capturing a grid space or a form needs `Enzyme.Const(f)` and
`Enzyme.set_runtime_activity(Enzyme.Reverse)`, as the automatic
differentiation tutorial describes for every other path. A gradient with
respect to a *runtime* scalar coefficient of the form should be a `Float64` or a `Ref`: an
`Integer` coefficient known only at run time costs `form` its inferred return type, which
Enzyme's type analysis rejects.

# Examples

```julia
A, F = assemble(a, l; dirichlet = :boundary => x -> 0.0)
u = pde_solve(A, F)

using SuiteSparse
u_spd = pde_solve(A, F; solver = :suitesparse, sym = :spd)
```

See also [`assemble`](@ref), [`sparse_factorize`](@ref), [`suitesparse_solve`](@ref),
[`suitesparse_qr_solve`](@ref), [`accelerate_solve`](@ref), [`mumps_solve`](@ref),
[`sparspak_solve`](@ref), [`linear_problem`](@ref).
"""
function pde_solve(A::SparseMatrixCSC, F::AbstractVector; solver::Symbol = :default, sym = :auto, kwargs...)
    if solver === :default
        if Sys.isapple() && Base.get_extension(Bramble, :BrambleAppleAccelerateExt) !== nothing
            return accelerate_solve(A, F; sym = sym, kwargs...)
        end
        return A \ F
    elseif solver === :suitesparse
        return suitesparse_solve(A, F; sym = sym, kwargs...)
    elseif solver === :spqr
        return suitesparse_qr_solve(A, F; kwargs...)
    elseif solver === :accelerate
        return accelerate_solve(A, F; sym = sym, kwargs...)
    elseif solver === :mumps
        return mumps_solve(A, F; sym = sym, kwargs...)
    elseif solver === :sparspak
        return sparspak_solve(A, F)
    else
        throw(
            ArgumentError(
            "Unknown solver: $solver. Expected :default, :suitesparse, :spqr, :accelerate, :mumps, or :sparspak.",
        ),
        )
    end
end

pde_solve(fact::Factorization, F::AbstractVector) = fact \ F
