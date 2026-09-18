module BrambleAlgebraicMultigridExt

using Bramble: Bramble
using AlgebraicMultigrid: AlgebraicMultigrid, smoothed_aggregation, ruge_stuben, aspreconditioner
using PrecompileTools: @setup_workload, @compile_workload

# `amg_preconditioner`/`_amg_operator`: `solvers/amg_preconditioner.jl` explains the
# underscored-fallback idiom and why `_amg_operator` exists separately from the public
# `amg_preconditioner` -- it is what `BrambleSciMLExt`'s `preconditioner = :amg` calls,
# through `Bramble`'s own dispatch, without that extension ever depending on
# `AlgebraicMultigrid` itself.

function _hierarchy(A::AbstractMatrix; method::Symbol = :smoothed_aggregation, kwargs...)
    if method === :smoothed_aggregation
        return smoothed_aggregation(A; kwargs...)
    elseif method === :ruge_stuben
        return ruge_stuben(A; kwargs...)
    else
        throw(ArgumentError("Unknown AMG method: $method. Expected :smoothed_aggregation or :ruge_stuben."))
    end
end

function Bramble._amg_preconditioner(A::AbstractMatrix; kwargs...)
    return _hierarchy(A; kwargs...)
end

function Bramble._amg_operator(A::AbstractMatrix; kwargs...)
    return aspreconditioner(_hierarchy(A; kwargs...))
end

# Warms `amg_preconditioner` and the `:smoothed_aggregation`/`:ruge_stuben` branches it
# dispatches to, plus the `aspreconditioner` wrapping `_amg_operator` does -- all only
# reachable once `AlgebraicMultigrid` is loaded, so only this extension's own precompile
# pass, not the core package's, ever reaches them.
if Bramble.PRECOMPILE_WORKLOAD
    @setup_workload begin
        Ωₕ = Bramble.mesh(Bramble.domain(Bramble.interval(0.0, 1.0)), 5, true)
        Wₕ = Bramble.gridspace(Ωₕ)
        a = Bramble.form(Wₕ, Wₕ, (u, v) -> Bramble.inner₊(Bramble.∇ₕ(u), Bramble.∇ₕ(v)))
        A = Bramble.assemble(a; dirichlet = :boundary)

        @compile_workload begin
            Bramble.amg_preconditioner(A)
            Bramble.amg_preconditioner(A; method = :ruge_stuben)
            Bramble._amg_operator(A)
        end
    end
end

end # module
