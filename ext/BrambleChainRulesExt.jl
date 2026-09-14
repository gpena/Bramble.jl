module BrambleChainRulesExt

using Bramble: Bramble, pde_solve
using ChainRulesCore: ChainRulesCore, NoTangent, unthunk
using SparseArrays: SparseMatrixCSC, rowvals, nonzeros, nzrange, sparse
using LinearAlgebra: lu
using PrecompileTools: @setup_workload, @compile_workload

# Gated on `ChainRulesCore` alone -- no `SciMLBase`/`LinearSolve` coupling, so a caller who
# only wants gradients through a Bramble solve never needs either. `pde_solve`'s own docstring
# (`src/form/pde_solve.jl`) explains why this one function is the entire adjoint story: nothing
# else in the assemble/Dirichlet path needs a hand-written rule, since it is already
# reverse-mode-differentiable on its own.

"""
    ChainRulesCore.rrule(::typeof(pde_solve), A::SparseMatrixCSC, F::AbstractVector)

Adjoint rule for [`pde_solve`](@ref): for a scalar functional `J` of the solution `u`, the
pullback solves `Aᵀ λ = ∂J/∂u` once -- via `fact' \\ b` against the *same* LU factorisation the
forward solve already computed, which is exact and requires no second factorisation whether or
not `A` is symmetric (a constrained row's `eₖ` replacement, `dirichlet_bc!`'s own convention,
makes even a naturally symmetric form structurally asymmetric) -- and returns `∂J/∂A = -λ uᵀ`
restricted to `A`'s own stored entries (never densified) and `∂J/∂F = λ`.
"""
function ChainRulesCore.rrule(::typeof(pde_solve), A::SparseMatrixCSC, F::AbstractVector)
    fact = lu(A)
    u = fact \ F
    function pde_solve_pullback(ȳ)
        λ = fact' \ unthunk(ȳ)
        Ā = similar(A)
        rows = rowvals(A)
        Āv = nonzeros(Ā)
        for j in axes(A, 2), k in nzrange(A, j)
            Āv[k] = -λ[rows[k]] * u[j]
        end
        return (NoTangent(), Ā, λ)
    end
    return u, pde_solve_pullback
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
