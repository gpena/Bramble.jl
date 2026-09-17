module BrambleILUZeroExt

using Bramble: Bramble
using ILUZero: ILUZero, ilu0, ILU0Precon
using LinearAlgebra: LinearAlgebra, ldiv!
using PrecompileTools: @setup_workload, @compile_workload
using SparseArrays: SparseMatrixCSC

# `ilu_preconditioner`/`_ilu_operator`: `solvers/ilu_preconditioner.jl` explains the
# underscored-fallback idiom and why `_ilu_operator` exists separately from the public
# `ilu_preconditioner` -- it is what `BrambleSciMLExt`'s `preconditioner = :ilu0` calls,
# through `Bramble`'s own dispatch, without that extension ever depending on `ILUZero` itself.
# `ilu0` already returns an object with `ldiv!`, unlike AMG's hierarchy, so both underscored
# functions do the same thing.

function Bramble._ilu_preconditioner(A::SparseMatrixCSC; kwargs...)
    return ilu0(A; kwargs...)
end

function Bramble._ilu_operator(A::SparseMatrixCSC; kwargs...)
    return ilu0(A; kwargs...)
end

# `ILU0Precon <: Factorization`, so `ILUZero`'s own
# `ldiv!(::AbstractVector{M}, ::ILU0Precon{T, N, M}, ::AbstractVector{M})` and Bramble's
# `ldiv!(::VectorElement, ::Factorization, ::AbstractVector)` are equally specific for a
# `VectorElement` destination. This method is more specific than both, so applying an ilu0
# preconditioner into a `VectorElement` resolves instead of erroring. The element-type
# parameters `M` and `N` are repeated from `ILUZero`'s signature: a method left generic in
# either would itself be ambiguous with that one.
function LinearAlgebra.ldiv!(
        uₕ::Bramble.VectorElement{<:Any, M}, P::ILU0Precon{T, N, M}, b::AbstractVector{M}
) where {T, N <: Integer, M}
    ldiv!(parent(uₕ), P, b)
    return uₕ
end

# Warms `ilu_preconditioner` on a small unsymmetric system, plus the `_ilu_operator` path
# `BrambleSciMLExt`'s `preconditioner = :ilu0` reaches -- both only reachable once `ILUZero`
# is loaded, so only this extension's own precompile pass, not the core package's, reaches
# them.
if Bramble.PRECOMPILE_WORKLOAD
    @setup_workload begin
        Ωₕ = Bramble.mesh(Bramble.domain(Bramble.interval(0.0, 1.0)), 5, true)
        Wₕ = Bramble.gridspace(Ωₕ)
        a = Bramble.form(
            Wₕ, Wₕ, (u, v) -> Bramble.inner₊(Bramble.∇ₕ(u), Bramble.∇ₕ(v)) + Bramble.innerₕ(Bramble.D₊ₓ(u), v)
        )
        A = Bramble.assemble(a; dirichlet = :boundary)

        @compile_workload begin
            Bramble.ilu_preconditioner(A)
            Bramble._ilu_operator(A)
        end
    end
end

end # module
