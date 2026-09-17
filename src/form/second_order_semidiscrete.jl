#=
# second_order_semidiscrete.jl

Method-of-lines semidiscretisation of second-order-in-time systems: turns a stiffness
bilinear form, an optional damping bilinear form, and a source linear form into the
second-order ODE a `SecondOrderODEProblem`-aware time stepper consumes.

## Mathematical background

Given the spatial stiffness form `K`, an optional damping form `C`, and the source `l`, method
of lines keeps time continuous and discretises only space, leaving one second-order ordinary
differential equation per degree of freedom:

```math
M \\ddot{u}_h + C \\dot{u}_h + K u_h = F(t),
```

where `K`/`C` are `a`/`c` assembled, `F(t)` is `l` assembled at time `t`, and `M` is the mass
matrix of the discrete inner product -- the same roles `A`/`M`/`F` play in the first-order
[`Semidiscretization`](@ref) (`semidiscrete.jl`), one order up.

## Why this maps onto `SciMLBase.SecondOrderODEProblem` state as `(v, u)`

`SecondOrderODEProblem(f, du0, u0, tspan)` only ever asks for the **acceleration** equation
`v' = f(v, u, p, t)`; the kinematic equation `u' = v` is filled in for you. Rewritten with `v`
standing in for `u̇`, the system above is

```math
v' = F(t) - C v - K u \\qquad (\\text{mass-matrix form, unmultiplied by } M^{-1}),
\\qquad u' = v,
```

which is exactly `(sd::SecondOrderSemidiscretization)(dv, v, u, p, t)` below, paired with the
`2n × 2n` block mass matrix [`block_mass_matrix`](@ref)`(sd) = blockdiag(M, I)` handed to
`DynamicalODEFunction` in `BrambleSciMLExt` -- the `I` block leaves `u' = v` a plain ODE, and
`M` carries whatever Dirichlet row-zeroing `sd` was built with.

## Dirichlet conditions constrain `u`, and `v` follows for free

A constrained row `i` carries `K[i, :] = eₖ`, `C[i, :] = 0`, `M[i, :] = 0`, and `F[i] = g(x_i,
t)` -- [`assemble`](@ref) and [`_assemble_mass_matrix`](@ref) already produce exactly this
shape (`_assemble_mass_matrix`'s job -- assemble a form, zero its Dirichlet rows -- is not
mass-specific, so it is reused unchanged for `C` too). Row `i` of the stacked system then
reads `0 = g(x_i, t) - u_h[i]`: the algebraic constraint pins `u_h[i] = g(x_i, t)` at every
`t`, and differentiating that identity gives `v_h[i] = ġ(x_i, t)` -- velocity at a constrained
dof is never independently prescribed, it falls out of position being held exactly. Prescribing
`v` *without* constraining `u` (a "driven velocity" boundary, position left to free
integration) needs a non-identity kinematic equation, which this abstraction does not build;
it is a different, more invasive mechanism than reusing `SecondOrderODEProblem`'s own default.

!!! warning "Explicit and symplectic solvers cannot be used at all"
    Every `SecondOrderSemidiscretization` carries a `mass_matrix` -- `blockdiag(M, I)`, `M`
    being whatever [`mass`](@ref semidiscretize_second_order) assembled to, `innerₕ(u, v)` by
    default. `OrdinaryDiffEqCore` refuses any explicit/symplectic solver
    (`VelocityVerlet`, the whole `OrdinaryDiffEqSymplecticRK`/Nyström family) unless the mass
    matrix is *exactly* `I` (a value check, `mass_matrix == I`, not merely non-singular) --
    checked empirically against `OrdinaryDiffEqCore` 1.x/2.x. A discrete `innerₕ` mass matrix
    is never exactly `I` on a Cartesian grid (its boundary rows always carry a half-weight,
    trapezoid-rule style), and there is no unweighted bilinear-form primitive in this package
    to build an identity mass some other way. **Solve with a mass-matrix-aware solver instead**
    -- `Rodas5P`, `FBDF`, `RadauIIA5` -- the same family [`ode_function`](@ref)'s own docstring
    points to for the first-order case; this restriction holds whether or not `sd` has any
    Dirichlet labels at all.

## Why the residual lives here and not in the extension

Same reasoning as [`Semidiscretization`](@ref): nothing on this page needs `SciMLBase`, a
`SecondOrderSemidiscretization` is a callable with the `(dv, v, u, p, t)` signature and a
triple of matrices, and `BrambleSciMLExt` only wraps it in a `DynamicalODEFunction`/
`SecondOrderODEProblem`.

See also: [`semidiscretize_second_order`](@ref), [`second_order_ode_function`](@ref),
[`second_order_ode_problem`](@ref)
=#

"""
    SecondOrderSemidiscretization{...} <: AbstractSemidiscretization

Method-of-lines semidiscretisation of `M üₕ + C u̇ₕ + K uₕ = F(t)`, callable with the
`(dv, v, u, p, t)` signature a second-order time stepper expects.

Built by [`semidiscretize_second_order`](@ref); read back with [`mass_matrix`](@ref),
[`damping_matrix`](@ref) and [`stiffness_matrix`](@ref).

# Fields
- `stiffness`: spatial [`BilinearForm`](@ref), assembled into `K`.
- `damping`: spatial [`BilinearForm`](@ref), assembled into `C`, or `nothing` for an undamped system.
- `source`: source [`LinearForm`](@ref), assembled into `F(t)` at each step.
- `space`: the test space every form shares.
- `stiffness_matrix`: `K`, with `eₖ` rows on the constrained degrees of freedom.
- `damping_matrix`: `C`, with zero rows on the constrained degrees of freedom, or `nothing`.
- `mass_matrix`: `M`, with zero rows on the constrained degrees of freedom.
- `source_vector`: the reusable `F` buffer, refilled in place.
- `constraints`: how the source's boundary values are reached -- one of `NoConstraints`, `LabelsOnly`, `StaticConstraints` or `TimeDependentConstraints`.
- `labels`: constrained boundary labels.
- `components`: leaf components the labels bind to, or `nothing` for all.
"""
struct SecondOrderSemidiscretization{K, C, L, S, MT, CM, VT, BC, LB, CP} <:
       AbstractSemidiscretization
    stiffness::K
    damping::C
    source::L
    space::S
    stiffness_matrix::MT
    damping_matrix::CM
    mass_matrix::MT
    source_vector::VT
    constraints::BC
    labels::LB
    components::CP
end

"""
    mass_matrix(sd::SecondOrderSemidiscretization) -> SparseMatrixCSC

Return the constant mass matrix `M`, whose constrained rows are zero.
"""
@inline mass_matrix(sd::SecondOrderSemidiscretization) = sd.mass_matrix

"""
    damping_matrix(sd::SecondOrderSemidiscretization) -> Union{SparseMatrixCSC, Nothing}

Return the assembled damping operator `C`, whose constrained rows are zero, or `nothing` for
an undamped system.
"""
@inline damping_matrix(sd::SecondOrderSemidiscretization) = sd.damping_matrix

"""
    stiffness_matrix(sd::SecondOrderSemidiscretization) -> SparseMatrixCSC

Return the assembled spatial stiffness operator `K`, whose constrained rows are `eₖ`.
"""
@inline stiffness_matrix(sd::SecondOrderSemidiscretization) = sd.stiffness_matrix

"""
    semidiscretize_second_order(K::BilinearForm, l::LinearForm; kwargs...) -> SecondOrderSemidiscretization

Semidiscretise `M üₕ + C u̇ₕ + K uₕ = F(t)` from the spatial stiffness form `K` and the source
`l`, both posed on the same test space.

Write `K` as the steady problem is written: `assemble(K)` is the same matrix
[`semidiscretize`](@ref) calls `A`.

# Keywords
- `mass`: [`BilinearForm`](@ref) defining `M` (default: `innerₕ(u, v)`, the discrete `L²` inner product, which is diagonal).
- `damping`: [`BilinearForm`](@ref) defining `C` (default: `nothing`, an undamped system).
- `dirichlet`: constrained labels and, where they carry values, the values -- every form [`assemble`](@ref) accepts, including time-dependent constraints from [`dirichlet_constraints`](@ref)`(Ωₕ, I, :label => (x, t) -> ...)` (default: `nothing`).
- `dirichlet_components`: leaf components of a composite space the labels bind to (default: `nothing`, all leaves).

Dirichlet conditions constrain `u`; the consistent velocity at a constrained dof follows from
differentiating that constraint in time and is never prescribed independently -- see this
file's header comment for why, and for the solver-compatibility warning that follows from it.

# Examples

```julia
Ωₕ = mesh(domain(interval(0.0, 1.0)), 101)
Wₕ = gridspace(Ωₕ)
I = interval(0.0, 1.0)

fₕ = Rₕ(Wₕ, x -> 0.0)
K = form(Wₕ, Wₕ, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
l = form(Wₕ, v -> innerₕ(fₕ, v))

sd = semidiscretize_second_order(K, l)
```

See also [`second_order_ode_function`](@ref), [`second_order_ode_problem`](@ref).
"""
function semidiscretize_second_order(
        K::BilinearForm,
        l::LinearForm;
        mass = nothing,
        damping = nothing,
        dirichlet = nothing,
        dirichlet_components = nothing
)
    sp = test_space(l)
    _validate_semidiscrete_spaces(K, l)

    labels, constraint_values = _normalize_dirichlet(dirichlet)
    constraints = _source_constraints(labels, constraint_values)

    Kmat = assemble(K; dirichlet = labels, dirichlet_components = dirichlet_components)
    M = _assemble_mass_matrix(
        mass === nothing ? _default_mass_form(sp) : mass, labels, dirichlet_components
    )
    Cmat = damping === nothing ? nothing :
           _assemble_mass_matrix(damping, labels, dirichlet_components)
    F = parent(element(sp, eltype(Kmat)))

    return SecondOrderSemidiscretization(
        K,
        damping,
        l,
        sp,
        Kmat,
        Cmat,
        M,
        F,
        constraints,
        labels === nothing ? () : labels,
        dirichlet_components
    )
end

# --- The residual ------------------------------------------------------------------- #

"""
    (sd::SecondOrderSemidiscretization)(dv, v, u, p, t) -> dv

Evaluate `dv = F(t) - C v - K u`, the right-hand side of `M v' = F(t) - C v - K u` (with
`u' = v` supplied by [`second_order_ode_function`](@ref)).
"""
function (sd::SecondOrderSemidiscretization)(
        dv::AbstractVector, v::AbstractVector, u::AbstractVector, p, t
)
    F = _source_buffer(sd, dv, t)
    _assemble_source!(F, sd, sd.constraints, p, t)

    copyto!(dv, F)
    mul!(dv, sd.stiffness_matrix, u, -1, 1)
    sd.damping_matrix === nothing || mul!(dv, sd.damping_matrix, v, -1, 1)
    return dv
end

"""
    block_mass_matrix(sd::SecondOrderSemidiscretization) -> SparseMatrixCSC

Return the `2n × 2n` block-diagonal `blockdiag(mass_matrix(sd), I)`, the mass matrix
[`second_order_ode_function`](@ref) hands `DynamicalODEFunction`: the `I` block leaves the
kinematic equation `u' = v` an ordinary ODE, and the `mass_matrix(sd)` block carries whatever
Dirichlet row-zeroing `sd` was built with.
"""
@inline function block_mass_matrix(sd::SecondOrderSemidiscretization)
    n = size(sd.mass_matrix, 1)
    return blockdiag(sd.mass_matrix, sparse(I, n, n))
end

# --- Display ------------------------------------------------------------------------ #

function Base.show(io::IO, sd::SecondOrderSemidiscretization)
    n = ndofs(sd.space)
    print(io, "SecondOrderSemidiscretization{", n, " dofs, ", length(sd.labels), " constrained label")
    print(io, length(sd.labels) == 1 ? "" : "s", "}")
    return nothing
end

function Base.show(
        io::IO, ::MIME"text/plain",
        sd::SecondOrderSemidiscretization{K, C, L, S, MT, CM, VT, BC, LB, CP}
) where {K, C, L, S, MT, CM, VT, BC, LB, CP}
    return show_block(io) do io
        pp = PrettyPrinter(io)
        printstyled(io, "SecondOrderSemidiscretization"; bold = true, color = :cyan)
        print(io, " {")
        printstyled(io, "M üₕ + C u̇ₕ + K uₕ = F(t)"; color = :yellow)
        println(io, "}:")

        pp_indented = with_indent(pp, 1)
        print_key_value(pp_indented, "Space", sprint(show, sd.space); separator = ": ")
        print_key_value(
            pp_indented, "Stiffness", _operator_description(sd.stiffness_matrix); separator = ": "
        )
        print_key_value(
            pp_indented,
            "Damping",
            sd.damping_matrix === nothing ? "none" : _operator_description(sd.damping_matrix);
            separator = ": "
        )
        return print_key_value(
            pp_indented, "Constraints", _constraints_description(sd); separator = ": "
        )
    end
end

Base.summary(sd::SecondOrderSemidiscretization) = sprint(show, sd)

# --- SciMLBase handoff -------------------------------------------------------------- #
#
# Implemented in `BrambleSciMLExt`, mirroring `ode_function`/`ode_problem`'s own
# underscored-fallback idiom -- see `semidiscrete.jl`'s header comment for why.

"""
    second_order_ode_function(sd::SecondOrderSemidiscretization) -> DynamicalODEFunction
    second_order_ode_function(K::BilinearForm, l::LinearForm; kwargs...) -> DynamicalODEFunction

Wrap a second-order semidiscretisation as a `DynamicalODEFunction` carrying its block mass
matrix, ready for `OrdinaryDiffEq`.

The two-form method builds the [`SecondOrderSemidiscretization`](@ref) first, forwarding
every keyword to [`semidiscretize_second_order`](@ref).

Requires [SciMLBase.jl](https://github.com/SciML/SciMLBase.jl); call `using SciMLBase` (or
any package that loads it, such as `OrdinaryDiffEq`) before calling this function.

See also [`second_order_ode_problem`](@ref), [`semidiscretize_second_order`](@ref).
"""
function second_order_ode_function(sd::SecondOrderSemidiscretization)
    return _second_order_ode_function(sd)
end

function second_order_ode_function(K::BilinearForm, l::LinearForm; kwargs...)
    sd = semidiscretize_second_order(K, l; kwargs...)
    return _second_order_ode_function(sd)
end

function _second_order_ode_function(::Any)
    return error(
        "second_order_ode_function requires SciMLBase.jl. Add `using SciMLBase` (or a " *
        "package that loads it, such as OrdinaryDiffEq) before calling this function.",
    )
end

"""
    second_order_ode_problem(sd::SecondOrderSemidiscretization, du₀, u₀, I::CartesianProduct{1}) -> SecondOrderODEProblem
    second_order_ode_problem(K::BilinearForm, l::LinearForm, du₀, u₀, I; kwargs...) -> SecondOrderODEProblem

Build the `SecondOrderODEProblem` stepping `sd` over the time domain `I`, from the initial
velocity `du₀` and initial displacement `u₀` -- each a [`VectorElement`](@ref) or a plain
vector. `I` may equally be a `(t₀, t₁)` tuple.

`u₀` is copied, never mutated, and the copy is made consistent with the Dirichlet rows at `t₀`
(see [`dirichlet_bc!`](@ref)). `du₀` is passed through unchanged: velocity at a constrained
dof is not an independent algebraic unknown here (see this file's header comment).

Keywords are those of [`second_order_ode_function`](@ref); the two-form method also forwards
to [`semidiscretize_second_order`](@ref).

Requires [SciMLBase.jl](https://github.com/SciML/SciMLBase.jl).

# Examples

```julia
sd = semidiscretize_second_order(K, l; dirichlet = bcs)
prob = second_order_ode_problem(sd, Rₕ(Wₕ, x -> 0.0), Rₕ(Wₕ, x -> sinpi(x[1])), interval(0.0, 1.0))
sol = solve(prob, Rodas5P(); reltol = 1e-11, abstol = 1e-13)
```

See also [`second_order_ode_function`](@ref), [`semidiscretize_second_order`](@ref).
"""
function second_order_ode_problem(sd::SecondOrderSemidiscretization, du₀, u₀, I; kwargs...)
    return _second_order_ode_problem(sd, du₀, u₀, I; kwargs...)
end

function second_order_ode_problem(
        K::BilinearForm, l::LinearForm, du₀, u₀, I; kwargs...
)
    sd = semidiscretize_second_order(K, l; kwargs...)
    return _second_order_ode_problem(sd, du₀, u₀, I)
end

function _second_order_ode_problem(::Any, du₀, u₀, I; kwargs...)
    return error(
        "second_order_ode_problem requires SciMLBase.jl. Add `using SciMLBase` (or a " *
        "package that loads it, such as OrdinaryDiffEq) before calling this function.",
    )
end
