# nonlinear_problem.jl
#
# `nonlinear_problem`: the SciMLBase handoff for a steady nonlinear residual, alongside
# `ode_problem`/`linear_problem` (form/semidiscrete_problems.jl). Implemented in `BrambleSciMLExt`,
# same underscored-fallback idiom as those two and `ast_sparsity_detector`/`export_vtk`: the
# extension's method narrows on `u0` (`::AbstractVector`, which a `VectorElement` already is)
# rather than on `residual`, which has no Bramble type to narrow to -- an arbitrary residual
# function, not a `BilinearForm`/`Semidiscretization` this package builds itself. Narrowing on
# either argument makes the extension's method a strict specialisation of this file's
# `::Any, ::Any` fallback, which is all Julia's method-overwrite check requires (the same
# trick `_export_vtk` uses, narrowing its *second* argument rather than its first).
#
# Unlike `ode_problem`, no consistent-initial-condition step is needed here: every Bramble
# worked example builds `residual` from an assembled `BilinearForm`, `A(u) * u - F`, and a
# Dirichlet row is already the identity there (see `assemble`) -- so `R_i(u) = u_i - g_i`
# corrects any initial guess in the very first Newton step, with nothing extra to enforce.
# `ode_problem`'s consistency step exists only because a differential-algebraic solver's very
# first step assumes it, which a root-finder never does.

"""
    nonlinear_problem(residual, u0; jacobian = nothing, jac_prototype = nothing, kwargs...) -> NonlinearProblem

Wrap `residual` -- the residual of a steady nonlinear discretisation `F(u) = 0`, as `F(u, p)`
or in-place `F!(res, u, p)` -- into the `NonlinearProblem` that `NonlinearSolve.solve` takes.

`u0`, the initial guess, is a [`VectorElement`](@ref) or a plain vector; copied, never
mutated. Every Bramble worked example builds `residual` from an assembled
[`BilinearForm`](@ref), `A(u) * u - F` evaluated at the current iterate, so Dirichlet rows
already carry their own identity/value pair and a Newton step corrects them from any initial
guess -- no separate consistency step, unlike [`ode_problem`](@ref)'s differential-algebraic
system.

# Keywords

  - `jacobian`: `jac(J, u, p)` if `residual` is in-place, or `J = jac(u, p)` if it is not --
    the two conventions cannot be mixed, the same requirement `NonlinearFunction` itself
    enforces. Default `nothing`: the solver differentiates `residual` itself.
  - `jac_prototype`: sparsity for the solver's Jacobian cache, e.g. built from
    [`jacobian_pattern`](@ref). Default `nothing` (dense).
  - Every other keyword forwards to `NonlinearProblem`.

Requires [SciMLBase.jl](https://github.com/SciML/SciMLBase.jl); call `using SciMLBase` (or
any package that loads it, such as `NonlinearSolve`) before calling this function.

# Examples

```julia
function residual!(res, u_vec::AbstractVector{T}, p) where {T}
    uₕ = element(Wₕ, T)
    uₕ .= u_vec
    A = assemble(diffusion_form(uₕ); dirichlet = :boundary)
    mul!(res, A, u_vec)
    return res .-= F
end

prob = nonlinear_problem(residual!, zeros(ndofs(Wₕ)))
sol = solve(prob, NewtonRaphson())
```

See also [`ode_problem`](@ref), [`linear_problem`](@ref), [`jacobian_pattern`](@ref).
"""
function nonlinear_problem(residual, u0; kwargs...)
    return _nonlinear_problem(residual, u0; kwargs...)
end

function _nonlinear_problem(residual, ::Any; kwargs...)
    return error(
        "nonlinear_problem requires SciMLBase.jl. Add `using SciMLBase` (or a package " *
        "that loads it, such as NonlinearSolve) before calling this function.",
    )
end
