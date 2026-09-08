# type_cached_assemble.jl
#
# Caching a coefficient-dependent BilinearForm's assembly by element type (see
# gpena/Bramble.jl#20).
#
# A Newton residual generic over `T` (`Float64` on a plain call, `ForwardDiff.Dual` while an
# AD backend's sparse Jacobian sweep is probing it) cannot reuse one preallocated matrix the
# way the Picard loop in poisson_nonlinear.md does: a matrix allocated for one element type
# cannot hold the other, so the doc's own `diffusion_matrix` rebuilds a *fresh* matrix,
# pattern and values both, on every call. But the *pattern* is exactly as fixed across
# element types as it is across Newton iterations -- only the coefficient's own values
# differ, and only because it was evaluated at a different `T`. This file gives that pattern
# a place to live per type it is ever reached at, instead of nowhere.

"""
    type_cached_assemble!(build, cache::AbstractDict, uₕ::VectorElement;
        dirichlet = nothing, dirichlet_components = nothing) -> SparseMatrixCSC

Assembles a coefficient-dependent [`BilinearForm`](@ref) into a matrix whose sparsity
pattern is built once per distinct element type `uₕ` is ever passed at, rather than on
every call -- the fix `diffusion_matrix`-style Newton residuals in the nonlinear worked
examples name and deliberately defer, since [`assemble`](@ref)/[`allocate_system_matrix`](@ref)
rebuild the whole matrix, pattern included, every time otherwise.

`build(uₕ)` is called once for each element type `uₕ` is ever seen at, and must return
`(a, refill!)`: the `BilinearForm` to assemble, built around whatever live coefficient
buffer(s) it needs (see the
[forms tutorial](tutorials/form.md#Live-grid-coefficients-and-dynamic-scalars)), and a
*one*-argument function `refill!(uₕ)` updating those buffers from the current `uₕ` --
called on every invocation, cache hit or miss, so a later call at an already-seen type
still sees the new guess rather than the one `build` first saw.

`build` itself should be a named function defined once, not a closure literal written
inside whatever function calls `type_cached_assemble!` — the same reason the Picard loop
in `poisson_nonlinear.md` builds its own form once, outside the loop, rather than on every
iteration: a `do ... end` block re-literalized on every call allocates a new closure each
time, which is exactly the cost this function exists to avoid paying more than once.

```julia
function build_diffusion(uₕ)
    Mu = element(Wₕ, eltype(uₕ))     # scratch for M₋ₓ!'s own output
    αvals = element(Wₕ, eltype(uₕ))
    a = form(Wₕ, Wₕ, (U, V) -> inner₊(αvals * ∇₋ₕ(U), ∇₋ₕ(V)))
    refill!(uₕ) = begin
        M₋ₓ!(Mu, uₕ)          # in place: `M₋ₓ(uₕ)` alone would allocate a fresh result
        αvals .= α.(Mu)
    end
    return a, refill!
end

cache = Dict()
diffusion_matrix(uₕ) = type_cached_assemble!(
    build_diffusion, cache, uₕ; dirichlet = :boundary)
```

`refill!` reaches for `M₋ₓ!` rather than the non-mutating `M₋ₓ`/`M₋ₕ` deliberately: the
latter allocates a fresh result every call (the same `similar`-based cost every allocating
stencil operator has), which would silently reintroduce an O(n) allocation this function's
whole point is to stop paying repeatedly. `M₋ₓ!` alone covers the 1D case above; a
D-dimensional coefficient needs one scratch buffer and one `M₋ₓ!`/`M₋ᵧ!`/`M₋₂!` call per
direction, the same way `poisson_nonlinear.md`'s own `nonlinear_series` builds a
D-dimensional coefficient tuple.

`cache` is shared across an entire Newton (or Picard) loop, one `Dict` per residual: the
first call at a given type pays `build`'s own cost plus [`allocate_system_matrix`](@ref)'s;
every later call at that same type pays only [`assemble!`](@ref)'s refill plus a small,
fixed dictionary/dynamic-dispatch overhead fetching the cached entry back out (a few KB,
independent of `ndofs`) -- not the `O(ndofs)` pattern rebuild a cache miss (or no cache at
all) pays every time.

Not thread-safe: `cache` is a plain, unlocked `Dict`, sized for the one-cache-per-residual
usage above. A form assembled from more than one task needs a lock or a per-task cache,
the same as any other shared mutable `Dict`.
"""
function type_cached_assemble!(build::F, cache::AbstractDict, uₕ::VectorElement{S, T};
        dirichlet = nothing, dirichlet_components = nothing) where {F, S, T}
    # `haskey`/`cache[T]` rather than `get!(f, cache, T) do ... end`: the do-block form
    # has to construct its closure before `get!` can even decide whether to call it, so it
    # allocates on every call, cache hit or miss -- exactly the cost this function exists
    # to avoid paying more than once.
    a, refill!, A = if haskey(cache, T)
        cache[T]
    else
        a, refill! = build(uₕ)
        # Populated *before* the pattern walk, not after: a coefficient buffer built with
        # `element(Wₕ, T)` starts `undef`, which for a non-`isbits` `T` (a tracer type, as
        # `ast_sparsity_detector`'s own `SparseConnectivityTracer` fallback reaches for) is
        # a genuinely unassigned reference, not merely an arbitrary bit pattern -- reading
        # it before `refill!` ever ran throws `UndefRefError` inside `allocate_system_matrix`.
        refill!(uₕ)
        entry = (a, refill!, allocate_system_matrix(a))
        cache[T] = entry
        entry
    end
    refill!(uₕ)
    assemble!(A, a; dirichlet = dirichlet,
        dirichlet_components = dirichlet_components)
    return A
end
