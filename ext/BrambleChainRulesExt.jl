module BrambleChainRulesExt

using Bramble: Bramble, pde_solve
using ChainRulesCore: ChainRulesCore, NoTangent, unthunk
using SparseArrays: SparseMatrixCSC, rowvals, nonzeros, nzrange, sparse
using LinearAlgebra: lu
using PrecompileTools: @setup_workload, @compile_workload

# Gated on `ChainRulesCore` alone, with no `SciMLBase` or `LinearSolve` coupling, so a caller
# who only wants gradients through a Bramble solve needs neither. `pde_solve`'s own docstring
# (`src/solvers/pde_solve.jl`) explains why this one function carries the whole adjoint:
# nothing else in the assemble/Dirichlet path needs a hand-written rule.

"""
    ChainRulesCore.rrule(::typeof(pde_solve), A::SparseMatrixCSC, F::AbstractVector; kwargs...)

Adjoint rule for [`pde_solve`](@ref): for a scalar functional `J` of the solution `u`, the
pullback solves `Aᴴ λ = ∂J/∂u` once and returns `∂J/∂A = -λ u'` (`u'` the conjugate
transpose, so a complex `A` conjugates `u`; for real data this is `-λ uᵀ`) restricted to `A`'s
own stored entries (never densified) and `∂J/∂F = λ`. `∂J/∂A` has the element type
`promote_type(eltype(A), eltype(λ), eltype(u))`, so a real `A` with a complex `F` gets a
complex `∂J/∂A`. The primal `u` has the element type `pde_solve(A, F)` returns.

Without keywords the adjoint solve is `fact' \\ b` against the *same* LU factorisation the
forward solve already computed, which is exact and requires no second factorisation whether or
not `A` is symmetric (a constrained row's `eₖ` replacement, `dirichlet_bc!`'s own convention,
makes even a naturally symmetric form structurally asymmetric). With keywords (`solver`, `sym`,
and the backend's own) the primal is `pde_solve(A, F; kwargs...)` and the adjoint solve is
`pde_solve(sparse(A'), ∂J/∂u; kwargs...)`, so the same backend and hint apply to `Aᴴ`. A `sym`
hint must therefore hold for `Aᴴ` as well as `A`, which it does for the symmetric, Hermitian
and positive definite cases it describes. The adjoint solve conjugates `mean_zero` and `gauge`,
the border of `Aᴴ`'s bordered system. A rectangular `A` (a `solver = :spqr` least-squares
solve) throws an `ArgumentError`: its pullback has a residual term this adjoint omits.
"""
function ChainRulesCore.rrule(
        ::typeof(pde_solve), A::SparseMatrixCSC, F::AbstractVector; kwargs...)
    Bramble._check_square_for_adjoint(A)
    isempty(kwargs) && return _lu_rrule(A, F)
    u = pde_solve(A, F; kwargs...)
    return u, _pde_solve_pullback(A, u, ȳ -> Bramble._pde_solve_adjoint(A, ȳ; kwargs...))
end

# One factorisation serves both solves. `fact \ F` can be wider than `pde_solve`'s result
# (`Float64` for `Float32` data), so `u` is converted to the type `pde_solve` returns.
function _lu_rrule(A::SparseMatrixCSC, F::AbstractVector)
    fact = lu(A)
    u = convert(AbstractVector{float(promote_type(eltype(A), eltype(F)))}, fact \ F)
    return u, _pde_solve_pullback(A, u, ȳ -> fact' \ ȳ)
end

# `Ā` takes the promoted type of `A`, `λ` and `u`: `-λ uᴴ` is complex for a real `A` and a
# complex `F`.
function _pde_solve_pullback(A::SparseMatrixCSC, u::AbstractVector, adjoint_solve)
    function pde_solve_pullback(ȳ)
        λ = adjoint_solve(unthunk(ȳ))
        Ā = similar(A, promote_type(eltype(A), eltype(λ), eltype(u)))
        rows = rowvals(A)
        Āv = nonzeros(Ā)
        for j in axes(A, 2), k in nzrange(A, j)

            Āv[k] = -λ[rows[k]] * conj(u[j])
        end
        return (NoTangent(), Ā, λ)
    end
    return pde_solve_pullback
end

# Warms `pde_solve`'s forward path and the rrule's construction/pullback -- both only reachable
# once `ChainRulesCore` is loaded, so only this extension's own precompile pass reaches them.
if Bramble.PRECOMPILE_WORKLOAD
    @setup_workload begin
        A = sparse(1:3, 1:3, ones(3))
        F = ones(3)

        @compile_workload begin
            u = pde_solve(A, F)
            _, pb = ChainRulesCore.rrule(pde_solve, A, F)
            pb(u)
        end
    end
end

end
