# semidiscrete_rhs.jl: `SemidiscretizeRHS`, the matrix-free explicit right-hand side
# `uₕ' = M⁻¹(F(t) - A uₕ)` for a `Semidiscretization` (`semidiscrete.jl`) with no Dirichlet
# constraints, `M`'s diagonal folded into a precomputed scaling instead of left for a solver
# to factorise at every step.

# --- The matrix-free explicit right-hand side --------------------------------------- #
#
# `M uₕ' = F(t) - A uₕ` (the residual above) is deliberately *not* divided by `M`: that
# residual is handed to SciML as an `ODEFunction` carrying `M` as its own `mass_matrix`,
# and the solver factorises `M` itself -- the only correct way to do it once any Dirichlet
# row is constrained, since a constrained row of `M` is zero and dividing by it is nonsense
# (that zero row *is* the algebraic constraint `0 = g(x, t) - uₕ[i]`, not an equation for
# `uₕ'[i]`).
#
# `M` is genuinely invertible everywhere only when there is no such row at all --
# `NoConstraints`, the plain `M = diag(weights)` the discrete `L²` inner product always
# assembles into. Only then does `uₕ' = M⁻¹(F(t) - A uₕ)` mean the same thing as the
# constrained residual above, and only then is dividing by `M`'s diagonal, once, outside
# the time loop, a valid substitute for a mass-matrix-aware solver -- gpena/Bramble.jl#163's
# whole premise ("H is diagonal") holds for this one case, not in general: a caller's own
# `mass` keyword to `semidiscretize` can be any `BilinearForm`, including a non-diagonal
# one, so this is checked here rather than assumed.

"""
    SemidiscretizeRHS{S,V}

The right-hand side of `uₕ' = M⁻¹(F(t) - A uₕ)`, `M` folded into a precomputed diagonal
scaling rather than solved for -- for a [`Semidiscretization`](@ref) with no Dirichlet
constraints, where that fold is valid. Built by [`semidiscretize_rhs`](@ref); callable with
the `(du, u, p, t)` signature a plain (non-mass-matrix) `SciMLBase.ODEProblem` expects.

Named `SemidiscretizeRHS`/[`semidiscretize_rhs`](@ref) rather than the free function
`semidiscretize_rhs!(du, u, p, t)` gpena/Bramble.jl#163 proposes: that signature is exactly
`SciMLBase.ODEFunction`'s own, which leaves no argument to pass a
[`Semidiscretization`](@ref) or the precomputed scaling through -- both have to be closed
over somehow, and a callable struct is what every other stateful callable in this package
does instead of a closure (`Semidiscretization` itself, `TypeCachedOperator`).
"""
struct SemidiscretizeRHS{S <: Semidiscretization, V <: AbstractVector}
    sd::S
    inv_mass_diag::V
end

# `M`'s own storage, not `weights(space(sd))`: `mass` is a caller-supplied keyword to
# `semidiscretize`, so the diagonal actually assembled -- which the default `innerₕ(u, v)`
# happens to make equal to `weights`, but a coefficient-scaled custom `mass` would not --
# is what has to be inverted.
function _diagonal_or_throw(M::SparseMatrixCSC)
    n = size(M, 1)
    d = zeros(eltype(M), n)
    nz = nonzeros(M)
    rv = rowvals(M)
    for j in 1:n
        rng = nzrange(M, j)
        if length(rng) > 1 || (length(rng) == 1 && rv[first(rng)] != j)
            throw(
                ArgumentError(
                "semidiscretize_rhs requires a diagonal mass matrix; column $j of the " *
                "assembled `mass` form has an off-diagonal entry.",
            ),
            )
        end
        length(rng) == 1 && (d[j] = nz[first(rng)])
    end
    return d
end

"""
    semidiscretize_rhs(sd::Semidiscretization) -> SemidiscretizeRHS

Build the matrix-free right-hand side of `uₕ' = M⁻¹(F(t) - A uₕ)` from `sd`, folding `M`'s
diagonal into a precomputed scaling instead of leaving `M` for a solver to factorise at
every step -- valid only because `M` is diagonal and, here, invertible everywhere.

Requires `sd` to have been built with `dirichlet = nothing`: any Dirichlet label makes the
corresponding row of `M` zero by construction (see [`semidiscretize`](@ref)), which is an
algebraic constraint on `uₕ`, not an equation for `uₕ'` -- there is no `uₕ'` value that
divides it away. Also requires `mass_matrix(sd)` to actually be diagonal: the default
`mass` (the discrete `L²` inner product) always assembles diagonally, but a caller-supplied
`mass` keyword to [`semidiscretize`](@ref) need not.

# Examples

```julia
sd = semidiscretize(a, l)  # no `dirichlet` keyword: NoConstraints
rhs = semidiscretize_rhs(sd)
prob = ODEProblem(rhs, parent(u₀), (0.0, 1.0))  # no mass_matrix to factorise
sol = solve(prob, Tsit5())
```

See also [`semidiscretize`](@ref), [`ode_problem`](@ref).
"""
function semidiscretize_rhs(sd::Semidiscretization)
    sd.constraints isa NoConstraints || throw(
        ArgumentError(
        "semidiscretize_rhs requires a Semidiscretization with dirichlet = nothing " *
        "(NoConstraints): a Dirichlet row's zero mass-matrix entry is an algebraic " *
        "constraint, not something `M⁻¹` can be taken through. Got constraints of " *
        "kind $(typeof(sd.constraints)).",
    ),
    )
    d = _diagonal_or_throw(mass_matrix(sd))
    any(iszero, d) && throw(
        ArgumentError("semidiscretize_rhs: the assembled mass matrix has a zero diagonal entry."),
    )
    return SemidiscretizeRHS(sd, inv.(d))
end

"""
    (rhs::SemidiscretizeRHS)(du, u, p, t) -> du

Evaluate `du = M⁻¹(F(t) - A u)`, the explicit right-hand side [`semidiscretize_rhs`](@ref)
built.

Allocates nothing (**0 bytes**) under the same condition the underlying
[`Semidiscretization`](@ref) residual does: `eltype(du)` and `typeof(t)` matching the
assembled element type.
"""
function (rhs::SemidiscretizeRHS)(du::AbstractVector, u::AbstractVector, p, t)
    sd = rhs.sd
    _sync_state!(sd.state, u)
    _update_coefficients!(sd.update_coefficients, p, t)

    F = _source_buffer(sd, du, t)
    _assemble_source!(F, sd, sd.constraints, p, t)

    A = _refresh_operator!(sd, sd.reassemble, du, t)

    copyto!(du, F)
    mul!(du, A, u, -1, 1)
    du .*= rhs.inv_mass_diag
    return du
end
