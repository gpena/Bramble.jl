# type_cached_assemble.jl
#
# Caching a coefficient-dependent BilinearForm's assembly by element type (see
# gpena/Bramble.jl#20).
#
# A Newton residual generic over `T` (`Float64` on a plain call, `ForwardDiff.Dual` while an
# AD backend's sparse Jacobian sweep is probing it) cannot reuse one preallocated matrix the
# way the Picard loop in poisson_nonlinear.jl does: a matrix allocated for one element type
# cannot hold the other, so the doc's own `diffusion_matrix` rebuilds a *fresh* matrix,
# pattern and values both, on every call. But the *pattern* is exactly as fixed across
# element types as it is across Newton iterations -- only the coefficient's own values
# differ, and only because it was evaluated at a different `T`. This file gives that pattern
# a place to live per type it is ever reached at, instead of nowhere.

# `cache` holds one `_TypeCachedEntry` per element type. Its fields are left untyped on
# purpose: each one holds its value already boxed, so reading it back out of an `Any`
# entry costs nothing, where a concrete field read through `Any` would box a fresh copy.
mutable struct _TypeCachedEntry
    const a::Any
    const refill!::Any
    const A::Any
    args::Any
end

"""
    type_cached_assemble!(build, cache::AbstractDict, uₕ::VectorElement;
        dirichlet = nothing, dirichlet_components = nothing) -> AbstractMatrix

Assembles a coefficient-dependent [`BilinearForm`](@ref) into a matrix whose sparsity
pattern is built once per distinct element type `uₕ` is ever passed at, rather than on
every call, as `diffusion_matrix` in the nonlinear worked example does without it:
[`assemble`](@ref)/[`allocate_system_matrix`](@ref) rebuild the whole matrix, pattern
included, every time otherwise.

`build(uₕ)` is called once for each element type `uₕ` is ever seen at, and must return
`(a, refill!)`: the `BilinearForm` to assemble, built around whatever live coefficient
buffer(s) it needs (see the
[forms tutorial](tutorials/form.md#Live-grid-coefficients-and-dynamic-scalars)), and a
*one*-argument function `refill!(uₕ)` updating those buffers from the current `uₕ` --
called on every invocation, cache hit or miss, so a later call at an already-seen type
still sees the new guess rather than the one `build` first saw.

`build` itself should be a named function defined once, not a closure literal written
inside whatever function calls `type_cached_assemble!`, the same reason the Picard loop
in `poisson_nonlinear.jl` builds its own form once, outside the loop, rather than on every
iteration: a `do ... end` block re-literalized on every call allocates a new closure each
time, which is exactly the cost this function exists to avoid paying more than once.

```julia
using Bramble: Mₓ!, type_cached_assemble!
function build_diffusion(uₕ)
    Mu = element(Wₕ, eltype(uₕ))     # scratch for Mₓ!'s own output
    αvals = element(Wₕ, eltype(uₕ))
    a = form(Wₕ, Wₕ, (U, V) -> inner₊(αvals * ∇ₕ(U), ∇ₕ(V)))
    refill!(uₕ) = begin
        Mₓ!(Mu, uₕ)          # in place: `Mₓ(uₕ)` alone would allocate a fresh result
        αvals .= α.(Mu)
    end
    return a, refill!
end

cache = Dict()
diffusion_matrix(uₕ) = type_cached_assemble!(
    build_diffusion, cache, uₕ; dirichlet = :boundary)
```

`refill!` reaches for `Mₓ!` rather than the non-mutating `Mₓ`/`Mₕ` deliberately: the
latter allocates a fresh result every call (the same `similar`-based cost every allocating
stencil operator has), which would silently reintroduce an O(n) allocation this function's
whole point is to stop paying repeatedly. `Mₓ!` alone covers the 1D case above; a
D-dimensional coefficient needs one scratch buffer and one `Mₓ!`/`Mᵧ!`/`M₂!` call per
direction, the same way `poisson_nonlinear.jl`'s own `nonlinear_series` builds a
D-dimensional coefficient tuple.

`cache` is shared across an entire Newton (or Picard) loop, one `Dict` per residual. The
first call at a given type pays for `build` and [`allocate_system_matrix`](@ref). After
that, a call at the same type looks the entry up, runs `refill!` and refills the cached
matrix in place with [`assemble!`](@ref). If it passes the same kinds of `dirichlet`
arguments as the call before, it allocates nothing beyond what those two do: with the
`build` above, 0 bytes. The entry holds on to the last `uₕ` and Dirichlet arguments,
which stay reachable while `cache` is.

Not thread-safe. `cache` is a bare `Dict` with no lock, one per residual. Assembling from
more than one task needs a lock or one cache per task.
"""
function type_cached_assemble!(
        build::F,
        cache::AbstractDict,
        uₕ::VectorElement{S, T};
        dirichlet = nothing,
        dirichlet_components = nothing
) where {F, S, T}
    found = get(cache, T, nothing)
    entry = found === nothing ? _type_cached_build!(build, cache, uₕ) :
            found::_TypeCachedEntry
    # The call's arguments go across the barrier inside a typed `Ref` kept on the entry.
    # Passed directly they would each be boxed. The `Ref` is refilled in place while the
    # argument types match the previous call's, and replaced (one allocation) when not.
    args = (uₕ, dirichlet, dirichlet_components)
    slot = entry.args
    if slot isa Base.RefValue{typeof(args)}
        slot[] = args
    else
        slot = Ref(args)
        entry.args = slot
    end
    # This is the one dynamic call. Everything it receives is already on the heap and it
    # returns `nothing`, so it boxes nothing; past it, the refill and `assemble!` infer
    # concretely.
    # `inferencebarrier` keeps inference from also walking `_type_cached_fill!` at the
    # abstract `Any` signature, a path no call ever takes.
    Base.inferencebarrier(_type_cached_fill!)(entry.a, entry.refill!, entry.A, slot)
    return entry.A
end

function _type_cached_build!(build::F, cache, uₕ::VectorElement{S, T}) where {F, S, T}
    a, refill! = build(uₕ)
    # Populated *before* the pattern walk, not after: a coefficient buffer built with
    # `element(Wₕ, T)` starts `undef`, which for a non-`isbits` `T` (a tracer type, as
    # `ast_sparsity_detector`'s own `SparseConnectivityTracer` fallback reaches for) is
    # a genuinely unassigned reference, not merely an arbitrary bit pattern -- reading
    # it before `refill!` ever ran throws `UndefRefError` inside `allocate_system_matrix`.
    refill!(uₕ)
    entry = _TypeCachedEntry(a, refill!, allocate_system_matrix(a), nothing)
    cache[T] = entry
    return entry
end

function _type_cached_fill!(a, refill!::R, A, slot) where {R}
    uₕ, dirichlet, dirichlet_components = slot[]
    refill!(uₕ)
    assemble!(A, a; dirichlet = dirichlet, dirichlet_components = dirichlet_components)
    return nothing
end
