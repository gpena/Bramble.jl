# pde_solve.jl
#
# `pde_solve`: the solve entry point, a plain `A \ F` unless `solver` or the macOS
# Accelerate route says otherwise, and the one function a reverse-mode adjoint rule attaches to.
#
# No source-level AD tool -- forward or reverse, `ForwardDiff`, `ReverseDiff`, `Mooncake`,
# `Enzyme` -- can differentiate through `\` itself: it dispatches into compiled
# BLAS/SuiteSparse code, opaque to every one of them, the same reason none can differentiate
# through any other library call written in C or Fortran. Everything *around* the solve --
# `assemble`, `dirichlet_bc!` -- is already reverse-mode-differentiable (`docs/notes/internals/
# autodiff.md`), so this one function is the entire gap between "Bramble can be forward-mode
# differentiated" (true today) and "Bramble can be used inside a gradient-based inverse
# problem or a PDE-constrained training loop without paying one factorisation per parameter"
# (what closes gpena/Bramble.jl#228).
#
# `pde_solve` itself needs no weak dependency: without `ChainRulesCore` it is the same solve,
# just not differentiable. `BrambleChainRulesExt` supplies the
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

# `:default` takes `accelerate_solve` on macOS only for a symmetric `A`: Accelerate loses on
# unsymmetric systems. An explicit `sym` hint is trusted unchecked, and `:unsymmetric` is the
# one that routes away.
function _default_wants_accelerate(A, sym)
    sym === :spd || sym === :definite || sym == 1 ||
        sym === :symmetric || sym == 2 ||
        (sym === :auto && issymmetric(A))
end

"""
    pde_solve(A::SparseMatrixCSC, F::AbstractVector; solver = :default, sym = :auto, mean_zero = nothing, kwargs...) -> Vector
    pde_solve(fact::Factorization, F::AbstractVector) -> Vector

Solve `A u = F` (or `fact u = F`) and return `u`.

With no keyword arguments this is `A \\ F`, unless `AppleAccelerate.jl` is loaded on macOS
*and* the system is symmetric (see `:default` below), in which case it is
`accelerate_solve(A, F; sym)`. The name exists so that reverse-mode AD tools have one function
to attach an adjoint rule to, which is what makes a whole `θ -> assemble -> pde_solve -> J(u)`
chain differentiable.

# Keywords
- `solver`: `:default` -- `A \\ F` on Linux, Windows, macOS without `AppleAccelerate.jl`
  loaded, and macOS on an unsymmetric `A`; on macOS with `using AppleAccelerate` in effect and
  `A` symmetric, dispatches to `accelerate_solve` instead (same as passing
  `solver = :accelerate` explicitly). Symmetry is `issymmetric(A)` under `sym = :auto` (the
  default), or trusted outright from an explicit `sym = :spd`/`:definite`/`:symmetric` (and
  conversely `:unsymmetric` skips straight to `A \\ F`) -- the caller has already asserted the
  property, so it is not checked again. The narrowing exists because Accelerate is only a win
  on the symmetric factorisations it reaches (SPD/Cholesky, LDLᵀ): on forms assembled on 2D
  grid spaces it won on a symmetric Poisson-plus-mass system and lost on an unsymmetric
  convection-diffusion one. This only ever narrows which solver runs on macOS; Linux and
  Windows are never affected, and a matrix Accelerate cannot factor (e.g. non-square) throws
  from `accelerate_solve` exactly as `solver = :accelerate` would -- `:default` never silently
  falls back to `\\` after picking Accelerate. `:suitesparse` (CHOLMOD/UMFPACK), `:spqr`
  (sparse QR, for a least-squares or rectangular `A`), `:accelerate` (Apple `libSparse`, needs
  `AppleAccelerate.jl`, honoured unconditionally regardless of symmetry), `:mumps` (needs
  `MUMPS.jl`), `:sparspak` (pure Julia, needs `Sparspak.jl`).
- `kwargs`: passed to the chosen backend (for `:accelerate` and `:mumps` the keywords of
  [`accelerate_solve`](@ref) and [`mumps_solve`](@ref)), which throws on one it does not
  take. `:default` takes only `kind` and `:sparspak` takes none: both throw an `ArgumentError`
  on any other keyword. `kind` and `sym` are read only when the Accelerate route runs; on
  `A \\ F` they are accepted and ignored, so `kind = :bogus` throws only where Accelerate is
  taken.
- `sym`: symmetry hint for `:suitesparse`, `:accelerate` and `:mumps` (and, on macOS, for
  `:default`'s own choice of solver -- see above). `:auto` (default) detects it;
  `:spd`/`:definite`/`1`, `:symmetric`/`2` and `:unsymmetric`/`0` state it.
- `mean_zero`: a weight vector `w` (for instance the assembled `innerₕ(1, v)`), or `nothing`
  (default). With `w`, `pde_solve` solves the bordered system `[A w; wᵀ 0] [u; λ] = [F; 0]`
  and returns `u`, the solution with discrete zero average ``w^T u = 0`` of a pure-Neumann
  system whose `A` annihilates constants, of the promoted float type of `A`, `F` and `w`;
  `A` is not modified. The bordered matrix is indefinite, so `sym` is not used: for a real
  symmetric `A` and real `w`, `:default` factors it by LU (or by Accelerate's pivoting
  ``LDL^T``) and `:accelerate` and `:mumps` by ``LDL^T``; for any other `A` they use LU, as
  do `:suitesparse` (UMFPACK) and `:sparspak`. `:spqr` throws an `ArgumentError`, and a `w`,
  `F` or `A` of mismatched size a `DimensionMismatch`. Without `mean_zero`, a square `A`
  with `‖A 1‖∞ ≤ 1e-10 ‖A‖∞` (constants in its kernel) logs a warning naming `mean_zero`,
  and the solve proceeds.

# Returns
- `Vector`: the solution, of the promoted element type of `A` and `F`.

A `SparseMatrixCSR` (`SparseMatricesCSR.jl`) `A` is also accepted: converted to
`SparseMatrixCSC` first (see `docs/src/internals/csr_solvers.md`), then solved exactly as
above.

# Reverse-mode differentiation

With `ChainRulesCore.jl` loaded, `BrambleChainRulesExt`'s `rrule` solves the adjoint system
`Aᴴ λ = ∂J/∂u` once and returns `∂J/∂A = -λ u'` (conjugate transpose, the ChainRules
convention for a complex `A`; `-λ uᵀ` for real data) restricted to `A`'s sparsity (never
densified) and `∂J/∂F = λ`. Without keywords the adjoint solve reuses the forward solve's own
factorisation. The rule accepts the keywords of `pde_solve`: it forwards them to the primal
solve and to the adjoint solve of `sparse(A')`, so `solver` and `sym` apply to `Aᴴ` as well,
and a `sym` hint must hold for `Aᴴ` too. A `SparseMatrixCSR` `A` has no rule (ChainRules sees
a method with no `rrule`, and differentiates the conversion and the solver's internals).
`using Enzyme` is enough for the same adjoint through `BrambleEnzymeExt`'s native
`EnzymeRules` rule, which takes no keywords (it cannot run before v4.3.0); do not
call `Enzyme.@import_rrule`, whose bridge drops the cotangent's explicit zeros from `nzval`
and returns a wrong gradient without any error. The Enzyme rule keeps the unconjugated
`-λ uᵀ` and is not validated for complex `A`, whose convention differs. `Mooncake` cannot
represent a `SparseMatrixCSC` cotangent at all and is unsupported.

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

# pure Neumann: fix the constant by a discrete zero average
w = assemble(form(W, v -> innerₕ(x -> 1.0, v)))
u0 = pde_solve(A_neumann, F_neumann; mean_zero = w)
```

See also [`assemble`](@ref), [`sparse_factorize`](@ref), [`suitesparse_solve`](@ref),
[`suitesparse_qr_solve`](@ref), [`accelerate_solve`](@ref), [`mumps_solve`](@ref),
[`sparspak_solve`](@ref), [`linear_problem`](@ref).
"""
function pde_solve(A::SparseMatrixCSC, F::AbstractVector; solver::Symbol = :default, sym = :auto,
        mean_zero::Union{Nothing, AbstractVector} = nothing, kwargs...)
    mean_zero === nothing || return _mean_zero_solve(A, F, mean_zero, solver, kwargs)
    _warn_if_annihilates_constants(A)
    if solver === :default
        _reject_unknown_kwargs(:pde_solve, "solver = :default", kwargs, (:kind,))
        if Sys.isapple() && Base.get_extension(Bramble, :BrambleAppleAccelerateExt) !== nothing &&
           _default_wants_accelerate(A, sym)
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
        _reject_unknown_kwargs(:pde_solve, "solver = :sparspak", kwargs)
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

# A pure-Neumann matrix sends the constant vector to zero, so `A \ F` is singular: one product
# `A 1` detects it, and the warning names the keyword that fixes the constant. Not an error,
# since a singular consistent system is legitimate for some solvers.
function _warn_if_annihilates_constants(A::SparseMatrixCSC)
    (size(A, 1) == size(A, 2) && !isempty(A)) || return nothing
    if norm(A * ones(eltype(A), size(A, 2)), Inf) <= 1e-10 * norm(A, Inf)
        @warn "pde_solve: A annihilates constants (‖A 1‖∞ ≤ 1e-10 ‖A‖∞), so the system is " *
              "singular, as for pure Neumann conditions; pass `mean_zero = w` to fix the " *
              "constant by wᵀu = 0."
    end
    return nothing
end

# `[A w; wᵀ 0]` is symmetric only for a real symmetric `A` and a real `w` (a ghost-point
# Neumann `A` is not symmetric); otherwise a symmetric route would read one triangle, or refuse.
function _bordered_sym(A::SparseMatrixCSC, w::AbstractVector)
    return eltype(A) <: Real && eltype(w) <: Real && issymmetric(A) ? :symmetric : :unsymmetric
end

# `mean_zero = w` solves the bordered system `[A w; wᵀ 0] [u; λ] = [F; 0]` and returns `u`.
# The border is never added to `A` as a rank-one term, which would be dense. The bordered
# matrix is indefinite: a symmetric route that pivots (Accelerate and MUMPS `LDLᵀ`) for a real
# symmetric `A`, LU otherwise, never Cholesky; `:default` calls `lu` directly, because `\` on
# a symmetric matrix tries Cholesky first. The caller's `sym` describes `A`, not the bordered
# matrix, so it is not used.
function _mean_zero_solve(A::SparseMatrixCSC, F::AbstractVector, w::AbstractVector, solver::Symbol,
        kwargs)
    n = size(A, 1)
    (size(A, 2) == n && length(w) == n && length(F) == n) || throw(
        DimensionMismatch(
        "pde_solve(mean_zero = w) needs a square A and F and w of its size; got A of size " *
        "$(size(A)), F of length $(length(F)) and w of length $(length(w)).",
    ),
    )
    solver === :spqr && throw(
        ArgumentError(
        "pde_solve(solver = :spqr) does not accept mean_zero: the bordered system is square, " *
        "not a least-squares problem. Use :default, :suitesparse, :accelerate, :mumps or :sparspak.",
    ),
    )
    T = promote_type(eltype(A), eltype(w))
    S = float(promote_type(T, eltype(F)))  # an all-`Int` system has a fractional solution
    n == 0 && return S[]
    ws = SparseMatrixCSC{T, Int}(reshape(w, n, 1))
    B = [A ws; transpose(ws) spzeros(T, 1, 1)]
    b = Vector{S}(undef, n + 1)  # dense: the factorisations take no sparse `F`
    b[1:n] .= F
    b[n + 1] = zero(S)
    bsym = _bordered_sym(A, w)
    if solver === :default
        _reject_unknown_kwargs(:pde_solve, "solver = :default", kwargs, (:kind,))
        x = if bsym === :symmetric && Sys.isapple() &&
               Base.get_extension(Bramble, :BrambleAppleAccelerateExt) !== nothing
            accelerate_solve(B, b; sym = bsym, kwargs...)
        else
            lu(B) \ b
        end
    elseif solver === :suitesparse
        x = suitesparse_solve(B, b; sym = :unsymmetric, kwargs...)
    elseif solver === :accelerate
        x = accelerate_solve(B, b; sym = bsym, kwargs...)
    elseif solver === :mumps
        x = mumps_solve(B, b; sym = bsym, kwargs...)
    elseif solver === :sparspak
        _reject_unknown_kwargs(:pde_solve, "solver = :sparspak", kwargs)
        x = sparspak_solve(B, b)
    else
        throw(
            ArgumentError(
            "Unknown solver: $solver. Expected :default, :suitesparse, :accelerate, :mumps, or :sparspak.",
        ),
        )
    end
    # Copied into a `Vector{S}`: the backends return `Float64` for `Float32` data, and the
    # branches return different types, which would leave the result uninferred.
    u = Vector{S}(undef, n)
    copyto!(u, 1, x, 1, n)
    return u
end

# `SparseMatrixCSR` support, via the same CSC-conversion fallback `sparse_factorize` uses
# (`_is_csr`/`_csr_to_csc`, src/solvers/sparse_solvers.jl -- see that file's comment for why
# `SparseMatrixCSR` cannot be named as a compile-time type here). A non-CSR, non-CSC
# `AbstractMatrix` (e.g. a dense `Matrix`) still throws a plain `MethodError`, the same as
# before this method existed.
function pde_solve(A::AbstractMatrix, F::AbstractVector; kwargs...)
    _is_csr(A) && return pde_solve(_csr_to_csc(A), F; kwargs...)
    throw(MethodError(pde_solve, (A, F)))
end
