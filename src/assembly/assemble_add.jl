#=
# assemble_add.jl

Additive accumulation into an already-filled matrix or vector (gpena/Bramble.jl#231).

## Why this needed no new sink or traversal

`assemble!`'s own sweep already *adds* every stencil entry's weight into `A.nzval`/`b`
(`add_to_sparse!`, and every sink in `bilinear_traversal.jl`) -- that is what lets two
stencil taps that land on the same matrix entry accumulate correctly within one term. The
only thing that makes `assemble!` read as "replace" from the outside is the `fill!(A/b,
0)` immediately before that sweep (`_assemble_bilinear!`/`_assemble_linear!`). `assemble_add!`
is therefore the *same* serial-cached/parallel core `assemble!` already calls, minus that
`fill!` and minus the Dirichlet pass -- reusing the coordinate-walk/replay cache, the threaded
band-coloured sweep, and the "pattern must already contain this entry" error
(`add_to_sparse!`'s `ArgumentError`) exactly as they already existed.

## The scale factor

`α` is threaded through as a plain multiplier at the point each entry's weight is about to
be added (`sink.α * weight` in every `ReplaySink`/`DiagonalReplaySink`, `α *
weight` in the parallel scatter and in the linear-form scatter) -- never folded into the
AST as an `OperatorScale` node. Wrapping the AST would build a *new* object on every call,
which the replay cache is keyed on by identity (`cache.ast === ast`): every
`assemble_add!(A, a, α)` call would then miss the cache and re-record, allocating exactly
the positions array a cache exists to avoid. Threading `α` as a runtime value instead keeps
the cache keyed on `(A, a.ast)` alone -- entirely unaffected by `α`, since which `nzval`
slots a term touches never depends on how the term is scaled -- so a caller is free to
change `α` (typically a `Ref`'s current value) on every call and still replay from cache
with 0 allocations.

Every internal function this threads `α` through (`_replay_segment!`,
`_scatter_point!`, `_scatter_term!`, ... ) defaults it to `true`, so `assemble!`/`assemble`'s
own call sites need no changes at all: `weight * true` is the exact algebraic identity
(unlike `weight * false`, which is not, for `NaN`/`Inf`/signed zero), so this is the same
optimization `_wrap_scale` (`form/simplifier.jl`) already relies on for `isone`/`iszero`
short-circuiting elsewhere in this package.

## Dirichlet interaction

`assemble_add!` never applies `dirichlet_bc!`/`apply_dirichlet_conditions!` -- accumulating
into a matrix whose rows a previous `dirichlet_bc!` call already replaced with `eₖ` adds the
new term's contribution *on top of* that identity row, which is essentially always a
mistake (the constrained row stops reading `u_k = g_k` and starts reading `u_k = g_k +
(\text{new term's diagonal entry})\, u_k + \ldots`). Assemble every piece with
`assemble_add!` first, and apply `dirichlet_bc!`/`symmetrize!` last, exactly once.
=#

@inline _scale_value(α::Base.RefValue) = α[]
@inline _scale_value(α::Number) = α

"""
    assemble_add!(A::AbstractMatrix, a::BilinearForm) -> AbstractMatrix
    assemble_add!(A::AbstractMatrix, a::BilinearForm, α) -> AbstractMatrix

Add `a`'s contribution to the preallocated `A`, in place, **without** first zeroing it --
unlike [`assemble!`](@ref), which refills `A` from scratch. The scaled form adds `α *`
`a`'s contribution instead; `α` may be a plain number or a `Base.RefValue`, dereferenced
once per call, so a time-stepping loop can vary it (a changing step size, a continuation
parameter) without rebuilding anything.

`A`'s sparsity pattern must already contain every entry `a` touches. When several pieces
accumulate into the same `A` (a mass and a stiffness form, say, with different stencils),
[`allocate_system_matrix`](@ref) against a form whose AST is their *sum* -- built once,
for the pattern alone, never assembled itself -- gives a matrix wide enough for every
piece: `PatternSink` (`bilinear_traversal.jl`) already walks an `OperatorAdd` as the union
of both sides' entries, exactly what this needs. A missing entry raises an `ArgumentError`
naming it, exactly as [`assemble!`](@ref)'s own first call does; nothing is silently
dropped or reallocated.

Threads following `trial_space(a)`'s backend [`execution_policy`](@ref), same as
[`assemble!`](@ref). **0 additional bytes** allocated on every call after the very first
one against a given `(A, a)` pair -- the same record/replay cache `assemble!` uses, shared
with it: calling `assemble!(A, a)` and `assemble_add!(A, a, α)` against the same `(A, a)`
both replay from whichever one recorded first.

Never applies `dirichlet_bc!` -- see this file's own header for why accumulating into an
already-constrained row is almost always wrong. Assemble every piece additively first,
constrain once, last.

# Examples

```julia
using Bramble: allocate_system_matrix
# M/Δt + θK, without ever assembling M or K into a temporary and adding it in.
Δt = 0.01
θ = Ref(1.0)

# For the pattern alone: never assembled itself, just wide enough for both pieces.
wide_form = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
A = allocate_system_matrix(wide_form)

for step in 1:nsteps
    fill!(nonzeros(A), 0)
    assemble_add!(A, m_form, 1 / Δt)   # A = M/Δt
    assemble_add!(A, k_form, θ)        # A += θ[] * K

    # ... solve, then update θ[] for the next step, no rebuild needed ...
end
```

See also: [`assemble!`](@ref), [`assemble`](@ref).
"""
# `A::AbstractMatrix`, not `A::SparseMatrixCSC` (S1.2's extension of the matrix-type seam,
# gpena/Bramble.jl#12): `_assemble_bilinear_core_cached!`/`_assemble_bilinear_parallel_core!`
# already dispatch on the matrix type themselves (S1.1), so widening this signature is all
# that is needed for a dense-backend form.
function assemble_add!(A::AbstractMatrix, a::BilinearForm)
    if execution_policy(a.trial_space) isa CpuSerial
        _assemble_bilinear_core_cached!(A, a.trial_space, a.test_space, a.ast, a.cache)
    else
        _assemble_bilinear_parallel_core!(A, a.trial_space, a.test_space, a.ast)
    end
    return A
end

function assemble_add!(A::AbstractMatrix, a::BilinearForm, α)
    αv = _scale_value(α)
    if execution_policy(a.trial_space) isa CpuSerial
        _assemble_bilinear_core_cached!(A, a.trial_space, a.test_space, a.ast, a.cache, αv)
    else
        _assemble_bilinear_parallel_core!(A, a.trial_space, a.test_space, a.ast, αv)
    end
    return A
end

"""
    assemble_add!(F::AbstractVector, l::LinearForm) -> AbstractVector
    assemble_add!(F::AbstractVector, l::LinearForm, α) -> AbstractVector

Add `l`'s contribution to the preallocated `F`, in place, without first zeroing it -- the
linear-form counterpart of [`assemble_add!`](@ref)`(A, a)`. A linear form's own assembly
scatters directly into a dense `F` (no sparsity pattern, no record/replay cache), so this
is simply the existing scatter with the initial `fill!` and the Dirichlet pass left out,
same as the bilinear case; see this file's header for the scale factor and the Dirichlet
note, both of which apply identically here.

# Examples

```julia
F = zeros(ndofs(Wₕ))
assemble_add!(F, l_mass)
assemble_add!(F, l_source, θ)   # F += θ[] * l_source's contribution
```

See also: [`assemble!`](@ref), [`assemble`](@ref).
"""
function assemble_add!(F::AbstractVector, l::LinearForm)
    space = test_space(l)
    _validate_term_markers(l.ast, markers(mesh(space)), "the form's space")
    if execution_policy(space) isa CpuSerial
        _assemble_linear_core!(F, space, l.ast)
    else
        _assemble_linear_parallel_core!(F, space, l.ast)
    end
    return F
end

function assemble_add!(F::AbstractVector, l::LinearForm, α)
    αv = _scale_value(α)
    space = test_space(l)
    _validate_term_markers(l.ast, markers(mesh(space)), "the form's space")
    if execution_policy(space) isa CpuSerial
        _assemble_linear_core!(F, space, l.ast, αv)
    else
        _assemble_linear_parallel_core!(F, space, l.ast, αv)
    end
    return F
end
