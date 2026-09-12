#=
# semidiscrete.jl

Method-of-lines semidiscretisation: turns a spatial bilinear form and a source linear form
into the ODE/DAE system a time stepper consumes.

## Mathematical background

Given the spatial form `a` and the source `l`, the method of lines keeps time continuous and
discretises only space, leaving one ordinary differential equation per degree of freedom:

```math
M \\frac{\\mathrm{d} u_h}{\\mathrm{d} t} = F(t) - A u_h,
```

where `A` is `a` assembled, `F(t)` is `l` assembled at time `t`, and `M` is the mass matrix
of the discrete inner product. Writing `a` the way the steady problem is written -- positive
definite, as in `inner₊(∇₋ₕ(u), ∇₋ₕ(v))` -- makes the steady state of this system the
solution of `A u = F`, which is the sign convention every example here already uses.

## Dirichlet conditions are algebraic constraints, not eliminated rows

A constrained row `i` carries `A[i, :] = eₖ` and `F[i] = g(x_i, t)` -- exactly what
[`assemble`](@ref) already produces. Zeroing row `i` of `M` therefore turns that row into

```math
0 = g(x_i, t) - u_h[i],
```

the constraint itself. The system becomes a differential-algebraic equation of index 1 with
a constant, singular mass matrix, which is what the stiff solvers in the SciML stack expect.

The alternative -- keeping `M[i, i]` and prescribing `u_h'[i] = ∂ₜg` -- needs the time
derivative of the boundary data, which nothing here has. This route needs only `g`.

## Why the residual lives here and not in the extension

Nothing on this page needs SciMLBase: a [`Semidiscretization`](@ref) is a callable with the
`(du, u, p, t)` signature and a pair of matrices. `BrambleSciMLExt` only wraps it in an
`ODEFunction`/`ODEProblem`. Keeping the logic here means the main test suite covers it
without the weak dependency loaded, and only the wrapper is gated.

See also: [`semidiscretize`](@ref), [`ode_function`](@ref), [`ode_problem`](@ref)
=#

# --- How the source's boundary values are reached at each step ---------------------- #
#
# Which of these a `Semidiscretization` stores is decided once, at construction, and kept as
# a field whose concrete type selects the `_assemble_source!` method -- rather than a flag
# tested per call inside the residual, which runs once per stage of every step.

"""
    NoConstraints

Source constraint carrier for a problem with no Dirichlet labels at all.
"""
struct NoConstraints end

"""
    LabelsOnly

Source constraint carrier for labels given without values (`dirichlet = :boundary`), which
constrain `u_h` to zero there.
"""
struct LabelsOnly end

"""
    StaticConstraints{C}

Source constraint carrier for time-independent boundary values `g(x)`.
"""
struct StaticConstraints{C}
    constraints::C
end

"""
    TimeDependentConstraints{C}

Source constraint carrier for boundary values `g(x, t)`, evaluated at each step through
`(dm::DomainMarkers)(t)`.
"""
struct TimeDependentConstraints{C}
    constraints::C
end

# Time dependence is read off the conditions' arity, the same test
# `_validate_time_dependent_arity` uses to check a time domain was passed honestly. A
# spatial condition `x -> ...` has no two-argument method, so the two never collide.
function _dirichlet_is_time_dependent(constraints::ConstraintMarkers)
    conds = conditions(constraints)
    isempty(conds) && return false
    return all(m -> hasmethod(identifier(m), Tuple{Any,Any}), conds)
end

@inline _source_constraints(::Nothing, ::Nothing) = NoConstraints()
@inline _source_constraints(::Any, ::Nothing) = LabelsOnly()
@inline _source_constraints(::Any, constraints) =
    if _dirichlet_is_time_dependent(constraints)
        TimeDependentConstraints(constraints)
    else
        StaticConstraints(constraints)
    end

# --- Walking the constrained rows --------------------------------------------------- #

# Mirrors `dirichlet_bc!`'s own route into `_each_marked` (dirichlet_constraints.jl) so the
# rows zeroed in the mass matrix are exactly the rows `assemble` writes `eₖ` into. Scanning
# for them instead would cost `ndofs` rather than the boundary cardinality.
function _each_dirichlet_row(f::F, space::ScalarGridSpace, labels, components) where {F}
    _validate_scalar_components(components)
    Ωₕ = mesh(space)
    for p in labels
        _each_marked(f, index_in_marker(Ωₕ, p), 0)
    end
    return nothing
end

function _each_dirichlet_row(f::F, space::CompositeGridSpace, labels, components) where {F}
    leaves = leaf_spaces_offsets(space)
    _validate_dirichlet_components(components, length(leaves))
    for p in labels
        for (mask, offset, _, active) in _leaf_entries(leaves, p, components)
            active || continue
            _each_marked(f, mask, offset)
        end
    end
    return nothing
end

# --- The semidiscretisation --------------------------------------------------------- #

"""
    Semidiscretization{...}

Method-of-lines semidiscretisation of `M u_h' = F(t) - A u_h`, callable with the
`(du, u, p, t)` signature a time stepper expects.

Built by [`semidiscretize`](@ref); read back with [`mass_matrix`](@ref) and
[`operator_matrix`](@ref).

# Fields
- `operator`: spatial [`BilinearForm`](@ref), assembled into `A`.
- `source`: source [`LinearForm`](@ref), assembled into `F(t)` at each step.
- `space`: the test space both forms share.
- `operator_matrix`: `A`, with `eₖ` rows on the constrained degrees of freedom.
- `mass_matrix`: `M`, with zero rows on the constrained degrees of freedom.
- `source_vector`: the reusable `F` buffer, refilled in place.
- `constraints`: how the source's boundary values are reached -- one of `NoConstraints`, `LabelsOnly`, `StaticConstraints` or `TimeDependentConstraints`.
- `labels`: constrained boundary labels.
- `components`: leaf components the labels bind to, or `nothing` for all.
- `state`: [`VectorElement`](@ref) receiving `u` before assembly, or `nothing`.
- `update_coefficients`: callable invoked with `t` before assembly, or `nothing`.
- `reassemble`: `Val(true)` to refill `A` at every step.
"""
struct Semidiscretization{A,L,S,MT,VT,BC,LB,CP,ST,TR,R}
    operator::A
    source::L
    space::S
    operator_matrix::MT
    mass_matrix::MT
    source_vector::VT
    constraints::BC
    labels::LB
    components::CP
    state::ST
    update_coefficients::TR
    reassemble::Val{R}
end

"""
    mass_matrix(sd::Semidiscretization) -> SparseMatrixCSC

Return the constant mass matrix `M`, whose constrained rows are zero.
"""
@inline mass_matrix(sd::Semidiscretization) = sd.mass_matrix

"""
    operator_matrix(sd::Semidiscretization) -> SparseMatrixCSC

Return the assembled spatial operator `A`, whose constrained rows are `eₖ`.
"""
@inline operator_matrix(sd::Semidiscretization) = sd.operator_matrix

"""
    space(sd::Semidiscretization)

Return the test space the semidiscretisation was built on.
"""
@inline space(sd::Semidiscretization) = sd.space

"""
    semidiscretize(a::BilinearForm, l::LinearForm; kwargs...) -> Semidiscretization

Semidiscretise `M u_h' = F(t) - A u_h` from the spatial form `a` and the source `l`, both
posed on the same test space.

Write `a` as the steady problem is written: `assemble(a)` is `A`, so the steady state of the
returned system solves `A u_h = F`.

# Keywords
- `mass`: [`BilinearForm`](@ref) defining `M` (default: `innerₕ(u, v)`, the discrete `L²` inner product, which is diagonal).
- `dirichlet`: constrained labels and, where they carry values, the values -- every form [`assemble`](@ref) accepts, including time-dependent constraints from [`dirichlet_constraints`](@ref)`(Ωₕ, I, :label => (x, t) -> ...)` (default: `nothing`).
- `dirichlet_components`: leaf components of a composite space the labels bind to (default: `nothing`, all leaves).
- `state`: [`VectorElement`](@ref) the current `u` is copied into before each assembly, for forms whose coefficients read it (default: `nothing`).
- `update_coefficients!`: called with the current `t` before each assembly, for coefficients that vary in time -- `t -> Rₕ!(fₕ, x -> f(x, t))` for a time-dependent source, or `t -> (α[] = t)` for a scalar `Ref` (default: `nothing`).
- `reassemble`: refill `A` at every step, for an operator whose coefficients change with `t` or `u` (default: `false`).

Time dependence of the Dirichlet values is detected by arity, exactly as
[`dirichlet_constraints`](@ref) validates it: conditions accepting `(x, t)` are evaluated at
each step, conditions accepting `(x)` are not.

# Examples

```julia
Ωₕ = mesh(domain(interval(0.0, 1.0)), 101)
Wₕ = gridspace(Ωₕ)
I = interval(0.0, 1.0)

fₕ = Rₕ(Wₕ, x -> 1.0)
a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
l = form(Wₕ, v -> innerₕ(fₕ, v))
bcs = dirichlet_constraints(Ωₕ, I, :boundary => (x, t) -> 0.0)

sd = semidiscretize(a, l; dirichlet = bcs)
```

See also [`ode_function`](@ref), [`ode_problem`](@ref), [`jacobian!`](@ref).
"""
function semidiscretize(
    a::BilinearForm,
    l::LinearForm;
    mass=nothing,
    dirichlet=nothing,
    dirichlet_components=nothing,
    state=nothing,
    (update_coefficients!)=nothing,
    reassemble::Bool=false,
)
    sp = test_space(l)
    _validate_semidiscrete_spaces(a, l)

    labels, constraint_values = _normalize_dirichlet(dirichlet)
    constraints = _source_constraints(labels, constraint_values)

    A = assemble(a; dirichlet=labels, dirichlet_components=dirichlet_components)
    M = _assemble_mass_matrix(
        mass === nothing ? _default_mass_form(sp) : mass, labels, dirichlet_components
    )
    F = parent(element(sp, eltype(A)))

    return Semidiscretization(
        a,
        l,
        sp,
        A,
        M,
        F,
        constraints,
        labels === nothing ? () : labels,
        dirichlet_components,
        state,
        update_coefficients!,
        Val(reassemble),
    )
end

@inline _default_mass_form(sp) = form(sp, sp, (u, v) -> innerₕ(u, v))

function _validate_semidiscrete_spaces(a::BilinearForm, l::LinearForm)
    n = ndofs(test_space(l))
    if ndofs(test_space(a)) != n || ndofs(trial_space(a)) != n
        throw(
            ArgumentError(
                "semidiscretize: `a` and `l` must be posed on the same space; got " *
                "$(ndofs(trial_space(a)))×$(ndofs(test_space(a))) for `a` and $n for `l`.",
            ),
        )
    end
    return nothing
end

# `dirichlet_bc!` writes `eₖ` into the constrained rows; the diagonal is then cleared, which
# leaves the stored entry in place rather than dropping it. Keeping it structurally present
# matters: the mass matrix is handed over once as a constant, and a solver that refactorises
# `M/γ - J` reads its pattern, not just its values.
function _assemble_mass_matrix(mass::BilinearForm, labels, components)
    sp = test_space(mass)
    M = assemble(mass)
    (labels === nothing || isempty(labels)) && return M

    dirichlet_bc!(M, sp, labels...; components=components)
    _each_dirichlet_row(sp, labels, components) do i
        return M[i, i] = zero(eltype(M))
    end
    return M
end

# --- The residual ------------------------------------------------------------------- #

"""
    (sd::Semidiscretization)(du, u, p, t) -> du

Evaluate `du = F(t) - A u`, the right-hand side of `M u_h' = F(t) - A u_h`.

Allocates nothing (**0 bytes**) whenever `eltype(du)` and `typeof(t)` match the assembled
element type -- the path every step of a real solve takes. A wider element type (a
`ForwardDiff.Dual` `t`, from the time gradient a Rosenbrock method needs) is met by
allocating a matching source buffer for that call alone.
"""
function (sd::Semidiscretization)(du::AbstractVector, u::AbstractVector, p, t)
    _sync_state!(sd.state, u)
    _update_coefficients!(sd.update_coefficients, t)

    F = _source_buffer(sd, du, t)
    _assemble_source!(F, sd, sd.constraints, t)

    A = _refresh_operator!(sd, sd.reassemble)

    # `du = F - A u` via the 5-argument `mul!(C, A, B, α, β) = α A B + β C`: fuses the
    # subtraction into the matrix-vector product itself (`du = -1 * A * u + 1 * du`),
    # saving the extra read-and-write pass over `du`/`F` that a separate combine step
    # would cost on every stage of a time integrator. `copyto!` first so that fused
    # pass reads `F`'s values back out of `du`, not `F` itself.
    copyto!(du, F)
    mul!(du, A, u, -1, 1)
    return du
end

@inline _sync_state!(::Nothing, u) = nothing
@inline function _sync_state!(state, u)
    copyto!(parent(state), u)
    return nothing
end

@inline _update_coefficients!(::Nothing, t) = nothing
@inline function _update_coefficients!(update!::F, t) where {F}
    update!(t)
    return nothing
end

# `eltype(F)` and `T` are both fixed by the argument types, so the comparison folds away and
# only one branch is ever compiled into a given specialisation.
@inline function _source_buffer(sd::Semidiscretization, du::AbstractVector, t)
    T = promote_type(eltype(du), typeof(t))
    F = sd.source_vector
    return eltype(F) === T ? F : similar(du, T)
end

@inline _assemble_source!(F::AbstractVector, sd::Semidiscretization, ::NoConstraints, t) =
    assemble!(F, sd.source)

# The conditions are applied through `apply_dirichlet_conditions!` rather than `assemble!`'s
# `dirichlet` keyword, which would re-run `_normalize_dirichlet` on every call: for
# constraints that is `Tuple(labels(bcs))` over a generator, and it allocates once per step.
# The labels were normalised once, in `semidiscretize`, and are passed straight through.
@inline function _assemble_source!(
    F::AbstractVector, sd::Semidiscretization, c::StaticConstraints, t
)
    assemble!(F, sd.source)
    return apply_dirichlet_conditions!(
        F, sd.source, c.constraints, sd.labels, sd.components
    )
end

@inline function _assemble_source!(
    F::AbstractVector, sd::Semidiscretization, c::TimeDependentConstraints, t
)
    assemble!(F, sd.source)
    return apply_dirichlet_conditions!(
        F, sd.source, c.constraints(t), sd.labels, sd.components
    )
end

# Labels without values constrain `u_h` to zero, which `assemble!` cannot express: handed a
# bare label it has nothing to write and raises. The rows are cleared here instead.
function _assemble_source!(F::AbstractVector, sd::Semidiscretization, ::LabelsOnly, t)
    assemble!(F, sd.source)
    _each_dirichlet_row(sd.space, sd.labels, sd.components) do i
        return @inbounds F[i] = zero(eltype(F))
    end
    return F
end

@inline _refresh_operator!(sd::Semidiscretization, ::Val{false}) = sd.operator_matrix
@inline function _refresh_operator!(sd::Semidiscretization, ::Val{true})
    A = sd.operator_matrix
    assemble!(A, sd.operator; dirichlet=sd.labels, dirichlet_components=sd.components)
    return A
end

# --- The Jacobian ------------------------------------------------------------------- #

"""
    jacobian!(J, sd::Semidiscretization, u, p, t) -> J

Fill `J` with `∂/∂u (F(t) - A u) = -A` and return it.

Exact whenever the operator's coefficients do not read `u` -- that is, whenever `sd` was
built without `state`. With a `state` the operator also varies with `u`, the term
`-(∂A/∂u) u` is missing, and this is a Picard linearisation rather than a Jacobian: pass
`jacobian = nothing` to [`ode_function`](@ref) and let the solver build it by sparse
automatic differentiation instead, seeded from [`jacobian_pattern`](@ref).

See also [`jacobian_prototype`](@ref).
"""
function jacobian!(J, sd::Semidiscretization, u, p, t)
    _sync_state!(sd.state, u)
    _update_coefficients!(sd.update_coefficients, t)
    A = _refresh_operator!(sd, sd.reassemble)
    return _negate_into!(J, A)
end

# Same pattern by construction (`jacobian_prototype` hands the solver a copy of `A`), so the
# stored values line up one for one and no structural search is needed.
@inline function _negate_into!(J::SparseMatrixCSC, A::SparseMatrixCSC)
    nzJ = nonzeros(J)
    nzA = nonzeros(A)
    if length(nzJ) == length(nzA) && rowvals(J) == rowvals(A)
        @. nzJ = -nzA
        return J
    end
    return _negate_into_generic!(J, A)
end

@inline _negate_into!(J, A) = _negate_into_generic!(J, A)

function _negate_into_generic!(J, A)
    copyto!(J, A)
    @. J = -J
    return J
end

"""
    jacobian_prototype(sd::Semidiscretization) -> SparseMatrixCSC

Return a matrix carrying the sparsity of `∂/∂u (F(t) - A u)`, for a solver to use as its
Jacobian cache.

This is the pattern of the assembled operator, Dirichlet rows included, which is what the
system's Jacobian has -- [`jacobian_pattern`](@ref) answers the different question of what a
*Newton residual*'s Jacobian looks like when the form's coefficients depend on the solution.

See also [`jacobian!`](@ref).
"""
@inline jacobian_prototype(sd::Semidiscretization) = copy(sd.operator_matrix)

# --- Consistent initial conditions -------------------------------------------------- #

"""
    dirichlet_bc!(u::AbstractVector, sd::Semidiscretization, t::Number) -> u

Write `sd`'s Dirichlet values at time `t` into `u` and return it.

An index-1 differential-algebraic system needs its initial condition to satisfy the
algebraic rows already: a `u` disagreeing with `g(x, 0)` on the boundary is inconsistent,
and a stiff solver either rejects it or absorbs it into the first step.
[`ode_problem`](@ref) applies this to a copy of the initial condition it is handed.
"""
function dirichlet_bc!(u::AbstractVector, sd::Semidiscretization, t::Number)
    return _apply_initial_constraints!(u, sd, sd.constraints, t)
end

@inline _apply_initial_constraints!(
    u::AbstractVector, sd::Semidiscretization, ::NoConstraints, t
) = u

function _apply_initial_constraints!(
    u::AbstractVector, sd::Semidiscretization, ::LabelsOnly, t
)
    _each_dirichlet_row(sd.space, sd.labels, sd.components) do i
        return @inbounds u[i] = zero(eltype(u))
    end
    return u
end

@inline _apply_initial_constraints!(
    u::AbstractVector, sd::Semidiscretization, c::StaticConstraints, t
) = dirichlet_bc!(u, sd.space, c.constraints, sd.labels...; components=sd.components)

@inline _apply_initial_constraints!(
    u::AbstractVector, sd::Semidiscretization, c::TimeDependentConstraints, t
) = dirichlet_bc!(u, sd.space, c.constraints(t), sd.labels...; components=sd.components)

# --- Display ------------------------------------------------------------------------ #
#
# Without these a `Semidiscretization` falls through to Julia's default and prints every
# nested type parameter of both forms and the space -- over 1500 characters, the same defect
# `ScalarGridSpace` carries its own pair of methods for (gpena/Bramble.jl#17). Two-argument
# `show` is the embeddable one-liner; `MIME"text/plain"` is the detailed block
# (gpena/Bramble.jl#45).

function Base.show(io::IO, sd::Semidiscretization)
    n = ndofs(sd.space)
    print(io, "Semidiscretization{", n, " dofs, ", length(sd.labels), " constrained label")
    print(io, length(sd.labels) == 1 ? "" : "s", "}")
    return nothing
end

function Base.show(
    io::IO, ::MIME"text/plain", sd::Semidiscretization{A,L,S,MT,VT,BC,LB,CP,ST,TR,R}
) where {A,L,S,MT,VT,BC,LB,CP,ST,TR,R}
    return show_block(io) do io
        pp = PrettyPrinter(io)
        printstyled(io, "Semidiscretization"; bold=true, color=:cyan)
        print(io, " {")
        printstyled(io, "M uₕ' = F(t) - A uₕ"; color=:yellow)
        println(io, "}:")

        pp_indented = with_indent(pp, 1)
        print_key_value(pp_indented, "Space", sprint(show, sd.space); separator=": ")
        print_key_value(
            pp_indented,
            "Operator",
            _operator_description(sd.operator_matrix);
            separator=": ",
        )
        print_key_value(
            pp_indented, "Constraints", _constraints_description(sd); separator=": "
        )
        return print_key_value(
            pp_indented, "Reassembled", R ? "every step" : "once"; separator=": "
        )
    end
end

@inline _operator_description(A) =
    string(size(A, 1), "×", size(A, 2), ", ", length(nonzeros(A)), " stored")

function _constraints_description(sd::Semidiscretization)
    isempty(sd.labels) && return "none"
    return string(
        _constraint_description(sd.constraints),
        " on ",
        join((string(":", l) for l in sd.labels), ", "),
    )
end

@inline _constraint_description(::NoConstraints) = "none"
@inline _constraint_description(::LabelsOnly) = "zero"
@inline _constraint_description(::StaticConstraints) = "g(x)"
@inline _constraint_description(::TimeDependentConstraints) = "g(x, t)"

Base.summary(sd::Semidiscretization) = sprint(show, sd)

# --- SciMLBase handoff -------------------------------------------------------------- #
#
# The three entry points below need SciMLBase and are implemented in `BrambleSciMLExt`. Each
# forwards to an underscored fallback whose first argument is `::Any` rather than the
# concrete type, so the extension's method is a strict specialisation: an identical signature
# would overwrite a method during precompilation, which Julia refuses. Same idiom as
# `ast_sparsity_detector`/`_ast_sparsity_detector` and `export_vtk`/`_export_vtk`.

"""
    ode_function(sd::Semidiscretization; kwargs...) -> ODEFunction
    ode_function(a::BilinearForm, l::LinearForm; kwargs...) -> ODEFunction

Wrap a semidiscretisation as an `ODEFunction` carrying its mass matrix, Jacobian and
sparsity, ready for `OrdinaryDiffEq`.

The two-form method builds the [`Semidiscretization`](@ref) first, forwarding every keyword
to [`semidiscretize`](@ref).

# Keywords
- `jacobian`: `jacobian!` (the default) to hand the solver the exact `-A`, or `nothing` to let it build one by automatic differentiation from `jac_prototype`.
- `jac_prototype`: sparsity for the solver's Jacobian cache (default: [`jacobian_prototype`](@ref)`(sd)`).

The resulting system is a differential-algebraic one whenever any Dirichlet label is
constrained, since those rows of the mass matrix are zero. Solve it with a method that
admits a singular mass matrix -- `FBDF`, `QNDF`, `Rodas5P`, `RadauIIA5` -- not an explicit
one.

!!! note "Rosenbrock methods and `update_coefficients!`"
    A Rosenbrock method (`Rodas5P`, `Rosenbrock23`) also needs `∂f/∂t`, which it builds by
    differentiating through `t`. That works when the only time dependence is the Dirichlet
    data, since those values reach the assembled vector as its element type. An
    `update_coefficients!` hook writing into a `Float64` [`VectorElement`](@ref) -- the usual
    `t -> Rₕ!(fₕ, x -> f(x, t))` -- cannot take a `ForwardDiff.Dual` time, and the solve
    fails on the first step with a time-gradient error. Either pass
    `Rodas5P(autodiff = AutoFiniteDiff())`, or use a BDF method, which needs no `∂f/∂t` at
    all. `FBDF` and `QNDF` are unaffected either way.

Requires [SciMLBase.jl](https://github.com/SciML/SciMLBase.jl); call `using SciMLBase` (or
any package that loads it, such as `OrdinaryDiffEq`) before calling this function.

See also [`ode_problem`](@ref), [`linear_problem`](@ref).
"""
function ode_function(sd::Semidiscretization; kwargs...)
    return _ode_function(sd; kwargs...)
end

function ode_function(
    a::BilinearForm, l::LinearForm; jacobian=jacobian!, jac_prototype=nothing, kwargs...
)
    sd = semidiscretize(a, l; kwargs...)
    return _ode_function(sd; jacobian=jacobian, jac_prototype=jac_prototype)
end

function _ode_function(::Any; kwargs...)
    return error(
        "ode_function requires SciMLBase.jl. Add `using SciMLBase` (or a package that " *
        "loads it, such as OrdinaryDiffEq) before calling this function.",
    )
end

"""
    ode_problem(sd::Semidiscretization, u₀, I::CartesianProduct{1}; kwargs...) -> ODEProblem
    ode_problem(a::BilinearForm, l::LinearForm, u₀, I; kwargs...) -> ODEProblem

Build the `ODEProblem` stepping `sd` over the time domain `I`, from the initial condition
`u₀` -- a [`VectorElement`](@ref) or a plain vector. `I` may equally be a `(t₀, t₁)` tuple.

`u₀` is copied, never mutated, and the copy is made consistent with the Dirichlet rows at
`t₀` (see [`dirichlet_bc!`](@ref)).

Keywords are those of [`ode_function`](@ref); the two-form method also forwards to
[`semidiscretize`](@ref).

Requires [SciMLBase.jl](https://github.com/SciML/SciMLBase.jl).

# Examples

```julia
sd = semidiscretize(a, l; dirichlet = bcs)
prob = ode_problem(sd, Rₕ(Wₕ, x -> sinpi(x[1])), interval(0.0, 1.0))
sol = solve(prob, FBDF())
```

See also [`ode_function`](@ref), [`semidiscretize`](@ref).
"""
function ode_problem(sd::Semidiscretization, u₀, I; kwargs...)
    return _ode_problem(sd, u₀, I; kwargs...)
end

function ode_problem(
    a::BilinearForm,
    l::LinearForm,
    u₀,
    I;
    jacobian=jacobian!,
    jac_prototype=nothing,
    kwargs...,
)
    sd = semidiscretize(a, l; kwargs...)
    return _ode_problem(sd, u₀, I; jacobian=jacobian, jac_prototype=jac_prototype)
end

function _ode_problem(::Any, u₀, I; kwargs...)
    return error(
        "ode_problem requires SciMLBase.jl. Add `using SciMLBase` (or a package that " *
        "loads it, such as OrdinaryDiffEq) before calling this function.",
    )
end

"""
    linear_problem(a::BilinearForm, l::LinearForm; kwargs...) -> LinearProblem

Assemble `a` and `l` into the `LinearProblem` that `LinearSolve.solve` takes, so the steady
system reaches the factorisations, iterative solvers and preconditioners in that stack
without being assembled by hand first.

# Keywords
- `dirichlet`, `dirichlet_components`: as [`assemble`](@ref) takes them.
- `symmetrize`: restore symmetry after imposing the conditions (default: `false`; see [`symmetrize!`](@ref)).

Requires [SciMLBase.jl](https://github.com/SciML/SciMLBase.jl), which defines
`LinearProblem`; `LinearSolve` itself is needed only to solve the result.

# Examples

```julia
using LinearSolve, IncompleteLU

prob = linear_problem(a, l; dirichlet = bcs)
sol = solve(prob, KrylovJL_GMRES())
```

See also [`assemble`](@ref), [`ode_problem`](@ref).
"""
function linear_problem(a::BilinearForm, l::LinearForm; kwargs...)
    return _linear_problem(a, l; kwargs...)
end

function _linear_problem(::Any, l; kwargs...)
    return error(
        "linear_problem requires SciMLBase.jl. Add `using SciMLBase` (or a package that " *
        "loads it, such as LinearSolve) before calling this function.",
    )
end
