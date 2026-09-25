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
definite, as in `inner₊(∇ₕ(u), ∇ₕ(v))` -- makes the steady state of this system the
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

Second-order systems `M ü_h + C u̇_h + K u_h = F(t)` -- wave equations, elastodynamics -- are
[`SecondOrderSemidiscretization`](@ref) (`second_order_semidiscrete.jl`), built by
[`semidiscretize_second_order`](@ref).

The Dirichlet constraint carriers (`NoConstraints`/`LabelsOnly`/`StaticConstraints`/...),
`AbstractSemidiscretization` and their application to a vector live in
`semidiscrete_constraints.jl`, included just before this file. The matrix-free explicit
right-hand side (`SemidiscretizeRHS`) and the `ODEFunction`/`ODEProblem`/`LinearProblem`
construction glue live in `semidiscrete_rhs.jl` and `semidiscrete_problems.jl`, included
just after.

See also: [`semidiscretize`](@ref), [`ode_function`](@ref), [`ode_problem`](@ref)
=#

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
- `constraints`: how the source's boundary values are reached -- one of `NoConstraints`, `LabelsOnly`, `StaticConstraints`, `TimeDependentConstraints` or `TimeParamDependentConstraints`.
- `labels`: constrained boundary labels.
- `components`: leaf components the labels bind to, or `nothing` for all.
- `state`: [`VectorElement`](@ref) receiving `u` before assembly, or `nothing`.
- `update_coefficients`: callable invoked with `t` (or, wrapped in `ParametricUpdate`, with `t` and the residual's own `p`) before assembly, or `nothing`.
- `reassemble`: `Val(true)` to refill `A` at every step.
"""
struct Semidiscretization{A, L, S, MT, VT, BC, LB, CP, ST, TR, R} <: AbstractSemidiscretization
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
    mass_matrix(sd::Semidiscretization) -> AbstractMatrix

Return the constant mass matrix `M`, whose constrained rows are zero.
"""
@inline mass_matrix(sd::Semidiscretization) = sd.mass_matrix

"""
    operator_matrix(sd::Semidiscretization) -> AbstractMatrix

Return the assembled spatial operator `A`, whose constrained rows are `eₖ`.
"""
@inline operator_matrix(sd::Semidiscretization) = sd.operator_matrix

"""
    semidiscretize(a::BilinearForm, l::LinearForm; kwargs...) -> Semidiscretization

Semidiscretise `M u_h' = F(t) - A u_h` from the spatial form `a` and the source `l`, both
posed on the same test space.

Write `a` as the steady problem is written: `assemble(a)` is `A`, so the steady state of the
returned system solves `A u_h = F`.

# Keywords
- `mass`: [`BilinearForm`](@ref) defining `M` (default: `innerₕ(u, v)`, the discrete `L²` inner product, which is diagonal).
- `dirichlet`: constrained labels and, where they carry values, the values -- every form [`assemble`](@ref) accepts, including time-dependent constraints from [`dirichlet_constraints`](@ref)`(Ωₕ, I, :label => (x, t) -> ...)`, or parameter-dependent ones from `(x, t, p) -> ...` for a value reached by the residual's own `p` (default: `nothing`).
- `dirichlet_components`: leaf components of a composite space the labels bind to (default: `nothing`, all leaves).
- `state`: [`VectorElement`](@ref) the current `u` is copied into before each assembly, for forms whose coefficients read it (default: `nothing`).
- `update_coefficients!`: called with the current `t` before each assembly, for coefficients that vary in time -- `t -> Rₕ!(fₕ, x -> f(x, t))` for a time-dependent source, or `t -> (α[] = t)` for a scalar `Ref` (default: `nothing`). A two-argument `(t, p) -> ...` also reaches the residual's own `p`, the same way a three-argument Dirichlet condition does.
- `reassemble`: refill `A` at every step, for an operator whose coefficients change with `t` or `u` (default: `false`).

Time and parameter dependence of the Dirichlet values is detected by arity, exactly as
[`dirichlet_constraints`](@ref) validates it: conditions accepting `(x, t, p)` are evaluated
at each step against the residual's own `p`, conditions accepting `(x, t)` are evaluated at
each step without it, and conditions accepting `(x)` are not evaluated per step at all.
`update_coefficients!` is detected the same way, between its one- and two-argument forms.

!!! note "`p` reaches only the source, never the operator"
    `A` (and therefore [`jacobian!`](@ref)) stays fixed for a given `t`, regardless of `p`:
    an operator coefficient that itself depends on `p` is not read by anything here. Use the
    `build`-based method below for an operator that depends on `t`; a `p`-dependent operator
    is not yet supported by either method.

# Examples

```julia
Ωₕ = mesh(domain(interval(0.0, 1.0)), 101)
Wₕ = gridspace(Ωₕ)
I = interval(0.0, 1.0)

fₕ = Rₕ(Wₕ, x -> 1.0)
a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
l = form(Wₕ, v -> innerₕ(fₕ, v))
bcs = dirichlet_constraints(Ωₕ, I, :boundary => (x, t) -> 0.0)

sd = semidiscretize(a, l; dirichlet = bcs)
```

See also [`ode_function`](@ref), [`ode_problem`](@ref), [`jacobian!`](@ref).
"""
function semidiscretize(
        a::BilinearForm,
        l::LinearForm;
        mass = nothing,
        dirichlet = nothing,
        dirichlet_components = nothing,
        state = nothing,
        (update_coefficients!) = nothing,
        reassemble::Bool = false
)
    sp = test_space(l)
    _validate_semidiscrete_spaces(a, l)

    labels, constraint_values = _normalize_dirichlet(dirichlet)
    constraints = _source_constraints(labels, constraint_values)

    A = assemble(a; dirichlet = labels, dirichlet_components = dirichlet_components)
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
        _wrap_update(update_coefficients!),
        Val(reassemble)
    )
end

"""
    semidiscretize(build, l::LinearForm; kwargs...) -> Semidiscretization

Semidiscretise `M u_h' = F(t) - A(t) u_h` from a spatial operator that genuinely depends on
`t` -- built fresh, once per element type `t` is ever reached at, instead of the single
`BilinearForm` the other method assembles into one fixed matrix.

`build(t)` is called once for each element type `t` is ever seen at (`Float64` on a normal
step, a `ForwardDiff.Dual` while a Rosenbrock stepper's `tgrad` differentiates the residual
through `t`), and must return `(a, refill!)`: the [`BilinearForm`](@ref) to assemble, built
around whatever live coefficient buffer(s) it needs, and a *one*-argument `refill!(t)`
updating those buffers from the current `t` -- called on every step, cache hit or miss, so a
later step at an already-seen type still sees the new `t` rather than the one `build` first
saw. Same build/refill!/cache discipline as [`type_cached_assemble!`](@ref), keyed on `t`
instead of a `VectorElement` iterate.

`build` should be a named function, not a closure literal written inline -- the same reason
[`type_cached_assemble!`](@ref)'s own docstring gives: a `do ... end` block re-literalised on
every call allocates a new closure each time.

This is what lets a Rosenbrock method (`Rodas5P`, `Rosenbrock23`) capture `-Ȧ(t) u_h` by
differentiating through `t` directly: the other method's fixed `BilinearForm` closes over a
`Float64`-typed coefficient buffer, which cannot hold a `Dual` and throws `InexactError`
under exactly that AD sweep.

# Keywords
Same as the `BilinearForm` method, except `reassemble` defaults to `true` -- a `build`-based
operator exists specifically to be rebuilt at every step.

# Examples

```julia
function build_diffusion_operator(t)
    αₕ = element(Wₕ, typeof(t))
    a = form(Wₕ, Wₕ, (u, v) -> inner₊(αₕ * ∇ₕ(u), ∇ₕ(v)))
    refill!(t) = (Rₕ!(αₕ, x -> α(x, t)); nothing)
    return a, refill!
end

sd = semidiscretize(build_diffusion_operator, l; dirichlet = bcs)
```

See also [`ode_function`](@ref), [`ode_problem`](@ref).
"""
function semidiscretize(
        build::B,
        l::LinearForm;
        mass = nothing,
        dirichlet = nothing,
        dirichlet_components = nothing,
        state = nothing,
        (update_coefficients!) = nothing,
        reassemble::Bool = true
) where {B <: Function}
    sp = test_space(l)

    labels, constraint_values = _normalize_dirichlet(dirichlet)
    constraints = _source_constraints(labels, constraint_values)

    t0 = 0.0
    a0, refill0! = build(t0)
    _validate_semidiscrete_spaces(a0, l)
    refill0!(t0)
    A0 = allocate_system_matrix(a0)
    assemble!(A0, a0; dirichlet = labels, dirichlet_components = dirichlet_components)

    operator = TypeCachedOperator(build, Dict{DataType, Any}(Float64 => (a0, refill0!, A0)))

    M = _assemble_mass_matrix(
        mass === nothing ? _default_mass_form(sp) : mass, labels, dirichlet_components
    )
    F = parent(element(sp, eltype(A0)))

    return Semidiscretization(
        operator,
        l,
        sp,
        A0,
        M,
        F,
        constraints,
        labels === nothing ? () : labels,
        dirichlet_components,
        state,
        _wrap_update(update_coefficients!),
        Val(reassemble)
    )
end

"""
    TypeCachedOperator{F}

A time-dependent spatial operator for [`semidiscretize`](@ref), built by `build(t)` once per
element type `t` is ever reached at, exactly the discipline [`type_cached_assemble!`](@ref)
uses for a coefficient that depends on the current iterate: the *pattern* of a time-dependent
operator is as fixed across element types as it is across time steps, only the coefficient's
own values differ, and only because they were evaluated at a different `t`.

Not constructed directly: [`semidiscretize`](@ref)`(build, l; ...)` builds one.
"""
struct TypeCachedOperator{F}
    build::F
    cache::Dict{DataType, Any}
end

# Mirrors `type_cached_assemble!`'s own cache-dict discipline, with `t` -- rather than a
# `VectorElement` iterate -- as the thing a fresh element type is ever reached through: the
# only way a `Semidiscretization`'s operator becomes `T`-typed is a Rosenbrock stepper
# differentiating its right-hand side through `t` (see `TypeCachedOperator`'s docstring).
function _fetch_or_build!(op::TypeCachedOperator, ::Type{T}, t) where {T}
    a, refill!, A = if haskey(op.cache, T)
        op.cache[T]
    else
        a, refill! = op.build(t)
        # Populated before the pattern walk, not after -- same reason `type_cached_assemble!`
        # does: a coefficient buffer built `undef` and read before `refill!` ever ran throws
        # inside `allocate_system_matrix` for a non-`isbits` `T`.
        refill!(t)
        entry = (a, refill!, allocate_system_matrix(a))
        op.cache[T] = entry
        entry
    end
    refill!(t)
    return a, A
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

    dirichlet_bc!(M, sp, labels...; components = components)
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

`p` reaches [`update_coefficients!`](@ref semidiscretize) and a time-dependent Dirichlet
condition written as `(x, t, p) -> ...`, if either was given one of those shapes; otherwise
it is unused, exactly as before this was possible. See the "`p` reaches only the source"
note on [`semidiscretize`](@ref).
"""
function (sd::Semidiscretization)(du::AbstractVector, u::AbstractVector, p, t)
    _sync_state!(sd.state, u)
    _update_coefficients!(sd.update_coefficients, p, t)

    F = _source_buffer(sd, du, t)
    _assemble_source!(F, sd, sd.constraints, p, t)

    A = _refresh_operator!(sd, sd.reassemble, du, t)

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

"""
    ParametricUpdate{F}

Wraps an `update_coefficients!` callable that reads the residual's own `p`, i.e. one given to
[`semidiscretize`](@ref) as `(t, p) -> ...` rather than `t -> ...`. Built once by
[`_wrap_update`](@ref), never directly -- the same "decided once, at construction" discipline
[`_source_constraints`](@ref) uses for the Dirichlet carriers above, so `_update_coefficients!`
below dispatches on it rather than re-checking arity on every residual call.
"""
struct ParametricUpdate{F}
    f::F
end

@inline _wrap_update(::Nothing) = nothing
@inline _wrap_update(f) = hasmethod(f, Tuple{Any, Any}) ? ParametricUpdate(f) : f

@inline _update_coefficients!(::Nothing, p, t) = nothing
# The existing one-argument path, untouched: a concrete `ParametricUpdate` method is more
# specific than this `where {F}` fallback, so wrapping only the new case above does not
# change how a plain `t -> ...` closure -- stored raw, exactly as before -- dispatches here.
@inline function _update_coefficients!(update!::F, p, t) where {F}
    update!(t)
    return nothing
end
@inline function _update_coefficients!(update!::ParametricUpdate, p, t)
    update!.f(t, p)
    return nothing
end

# `eltype(F)` and `T` are both fixed by the argument types, so the comparison folds away and
# only one branch is ever compiled into a given specialisation.
@inline function _source_buffer(sd::AbstractSemidiscretization, du::AbstractVector, t)
    T = promote_type(eltype(du), typeof(t))
    F = sd.source_vector
    return eltype(F) === T ? F : similar(du, T)
end

@inline _assemble_source!(F::AbstractVector, sd::AbstractSemidiscretization, ::NoConstraints, p, t) = assemble!(F, sd.source)

# The conditions are applied through `apply_dirichlet_conditions!` rather than `assemble!`'s
# `dirichlet` keyword, which would re-run `_normalize_dirichlet` on every call: for
# constraints that is `Tuple(labels(bcs))` over a generator, and it allocates once per step.
# The labels were normalised once, in `semidiscretize`, and are passed straight through.
@inline function _assemble_source!(
        F::AbstractVector, sd::AbstractSemidiscretization, c::StaticConstraints, p, t
)
    assemble!(F, sd.source)
    return apply_dirichlet_conditions!(
        F, sd.source, c.constraints, sd.labels, sd.components
    )
end

@inline function _assemble_source!(
        F::AbstractVector, sd::AbstractSemidiscretization, c::TimeDependentConstraints, p, t
)
    assemble!(F, sd.source)
    return apply_dirichlet_conditions!(
        F, sd.source, c.constraints(t), sd.labels, sd.components
    )
end

@inline function _assemble_source!(
        F::AbstractVector, sd::AbstractSemidiscretization, c::TimeParamDependentConstraints, p, t
)
    assemble!(F, sd.source)
    return apply_dirichlet_conditions!(
        F, sd.source, c.constraints(t, p), sd.labels, sd.components
    )
end

# Labels without values constrain `u_h` to zero, which `assemble!` cannot express: handed a
# bare label it has nothing to write and raises. The rows are cleared here instead.
function _assemble_source!(F::AbstractVector, sd::AbstractSemidiscretization, ::LabelsOnly, p, t)
    assemble!(F, sd.source)
    _each_dirichlet_row(sd.space, sd.labels, sd.components) do i
        return @inbounds F[i] = zero(eltype(F))
    end
    return F
end

@inline _refresh_operator!(sd::Semidiscretization, ::Val{false}, x, t) = sd.operator_matrix
@inline function _refresh_operator!(sd::Semidiscretization, ::Val{true}, x, t)
    return _reassemble_operator!(sd.operator, sd, x, t)
end

# The `BilinearForm` path: one fixed, `Float64`-typed matrix, refilled in place every step --
# unaffected by, and exactly as before, `TypeCachedOperator` existing.
@inline function _reassemble_operator!(op::BilinearForm, sd::Semidiscretization, x, t)
    A = sd.operator_matrix
    assemble!(A, op; dirichlet = sd.labels, dirichlet_components = sd.components)
    return A
end

# The `build`-based path: a fresh matrix, of whichever element type `x` and `t` are running
# at, fetched from (or added to) `op`'s cache -- see `TypeCachedOperator`.
function _reassemble_operator!(op::TypeCachedOperator, sd::Semidiscretization, x, t)
    T = promote_type(eltype(x), typeof(t))
    a, A = _fetch_or_build!(op, T, t)
    assemble!(A, a; dirichlet = sd.labels, dirichlet_components = sd.components)
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
    _update_coefficients!(sd.update_coefficients, p, t)
    A = _refresh_operator!(sd, sd.reassemble, u, t)
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
    jacobian_prototype(sd::Semidiscretization) -> AbstractMatrix

Return a matrix carrying the sparsity of `∂/∂u (F(t) - A u)`, for a solver to use as its
Jacobian cache.

This is the pattern of the assembled operator, Dirichlet rows included, which is what the
system's Jacobian has -- [`jacobian_pattern`](@ref) answers the different question of what a
*Newton residual*'s Jacobian looks like when the form's coefficients depend on the solution.

See also [`jacobian!`](@ref).
"""
@inline jacobian_prototype(sd::Semidiscretization) = copy(sd.operator_matrix)

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
        io::IO, ::MIME"text/plain",
        sd::Semidiscretization{A, L, S, MT, VT, BC, LB, CP, ST, TR, R}
) where {A, L, S, MT, VT, BC, LB, CP, ST, TR, R}
    return show_block(io) do io
        pp = PrettyPrinter(io)
        printstyled(io, "Semidiscretization"; bold = true, color = :cyan)
        print(io, " {")
        printstyled(io, "M uₕ' = F(t) - A uₕ"; color = :yellow)
        println(io, "}:")

        pp_indented = with_indent(pp, 1)
        print_key_value(pp_indented, "Space", sprint(show, sd.space); separator = ": ")
        print_key_value(
            pp_indented,
            "Operator",
            _operator_description(sd.operator_matrix);
            separator = ": "
        )
        print_key_value(
            pp_indented, "Constraints", _constraints_description(sd); separator = ": "
        )
        return print_key_value(
            pp_indented, "Reassembled", R ? "every step" : "once"; separator = ": "
        )
    end
end

# `nonzeros` only exists for a `SparseMatrixCSC`; a dense-backend matrix (S1.2's extension of
# the matrix-type seam, gpena/Bramble.jl#12) has no separate notion of "stored" entries, so
# every entry counts.
@inline _stored_count(A::SparseMatrixCSC) = length(nonzeros(A))
@inline _stored_count(A::AbstractMatrix) = length(A)

@inline _operator_description(A) = string(size(A, 1), "×", size(A, 2), ", ", _stored_count(A), " stored")

Base.summary(sd::Semidiscretization) = sprint(show, sd)
