#===========================================================================#
# The restriction operator Rₕ.
#
# Evaluates a function at the grid points. It lives with the other operators rather
# than with the element it produces, alongside the stencil families and avgₕ.
#===========================================================================#

########################
#                      #
# Restriction operator #
#                      #
########################

"""
    Rₕ!(uₕ::VectorElement, f; markers = ()) -> VectorElement

In-place version of the restriction operator [`Rₕ`](@ref). Evaluates `f` at the
grid points and writes the result into `uₕ`. Returns `uₕ`.

# Arguments

  - `uₕ::VectorElement`: pre-allocated element to write into.
  - `f`: function of one grid point. It receives a scalar on a 1D mesh and an
    `NTuple{D}` on a `D`-dimensional one, never an `SVector`.

# Keywords

  - `markers::NTuple{N,Symbol}`: restrict evaluation to the named marked
    regions, leaving every other entry zero. Several markers act as a union.

# Examples

```julia
Rₕ!(uₕ, x -> sin(x))                  # 1D: x is a Float64
Rₕ!(uₕ, x -> sin(x[1]) * cos(x[2]))   # 2D: x is a Tuple{Float64,Float64}

# only the points carrying the :left marker; the rest stay zero
Rₕ!(uₕ, x -> 1.0; markers = (:left,))
```

For an `N`-component element either shape of `f` works and both give the same result;
the single vector-valued function is evaluated once per grid point when every
component shares the same mesh, whereas the tuple always evaluates each component
function separately. On a heterogeneous composite — components built over different
meshes — the single-function form is instead re-evaluated once per component, since
there is no grid point shared by every component to evaluate it at only once:

```julia
Rₕ!(uₕ, (f₁, f₂))                     # one function per component
Rₕ!(uₕ, x -> (f₁(x), f₂(x)))          # one function returning all components
```

See also: [`Rₕ`](@ref), [`avgₕ!`](@ref), [`element`](@ref)
"""
@inline Rₕ!(uₕ::VectorElement{<:ScalarGridSpace}, f::F) where {F} = _Rₕ_parallel!(uₕ, f)
@inline Rₕ!(uₕ::VectorElement{<:CompositeGridSpace}, f::F) where {F} = _Rₕ_scatter_parallel!(
    uₕ, f)
@inline function Rₕ!(uₕ::VectorElement{<:CompositeGridSpace}, f::Tuple)
    # `map` over `components(uₕ)` and `f` together, rather than `ntuple(…, Val(NC))`
    # indexing both by a shared count: it unrolls exactly the same way for tuples, needs
    # no leaf count of its own, stays correct under any nesting, and — since `map` requires
    # equal-length tuples — errors the same way a length mismatch always would have.
    map(Rₕ!, components(uₕ), f)
    return uₕ
end

# A concretely typed kernel for per-point restriction calls, avoiding anonymous closure
# captures over (`f`, `Ωₕ`, `idxs`). A named callable struct eliminates compiler indirection
# and achieves performance parity with a flat loop.
struct _RₕKernel{F, M, IX}
    f::F
    Ω::M
    idxs::IX
end
@inline (k::_RₕKernel)(i) = k.f(point(k.Ω, k.idxs[i]))

# The two plain methods above ensure that unmasked restriction calls (the primary path
# during time stepping) resolve directly without invoking Julia's keyword argument dispatch
# machinery. They take precedence over the generic `uₕ::VectorElement` keyword method for
# concrete scalar and composite elements.
#
# A function barrier, typing `f` as its own free parameter `F`. Takes `uₕ` itself and
# re-derives `Ωₕ`/`raw`/`idxs` inside the typed function, rather than being handed
# pre-extracted locals from an untyped caller.
@inline function _Rₕ_parallel!(uₕ::VectorElement{<:ScalarGridSpace}, f::F) where {F}
    (; space) = uₕ
    Ωₕ = mesh(space)
    raw = values(uₕ)
    idxs = indices(Ωₕ)
    n = length(idxs)
    _cpu_threaded_for!(execution_policy(space), raw, 1:n, _RₕKernel(f, Ωₕ, idxs))
    return uₕ
end

# When every leaf shares one mesh, `f` is evaluated once per grid point and its tuple
# scattered across every leaf's storage in a single pass — the docstring's "evaluated
# once" claim. A heterogeneous composite (leaves on different meshes) has no such shared
# "grid point i"; `mesh(Wₕ::CompositeGridSpace)` always resolves to the first leaf's
# regardless (see `vector_gridspace.jl`), so taking the shared-evaluation path
# unconditionally silently mis-sized every leaf after the first (gpena/Bramble.jl#78).
# `f` is instead re-evaluated at each leaf's own grid points through `_Rₕ_parallel!`,
# keeping only that leaf's entry of the tuple it returns.
@inline function _Rₕ_scatter_parallel!(
        uₕ::VectorElement{<:CompositeGridSpace}, f::F) where {F}
    comps = components(uₕ)
    if _shares_one_mesh(comps)
        sp = space(uₕ)
        Ωₕ = mesh(sp)
        raws = map(values, comps)
        idxs = indices(Ωₕ)
        n = length(idxs)
        _cpu_threaded_scatter_for!(execution_policy(sp), raws, 1:n, _RₕKernel(f, Ωₕ, idxs))
    else
        ntuple(k -> (_Rₕ_parallel!(comps[k], pt -> f(pt)[k]); nothing), Val(length(comps)))
    end
    return uₕ
end

# A one-component space is a scalar space, so generic code that builds an
# NC-tuple of functions still works when NC == 1.
@inline Rₕ!(uₕ::VectorElement{<:ScalarGridSpace{D}}, f::Tuple{Any}) where {D} = Rₕ!(uₕ, f[1])
@inline Rₕ!(uₕ::VectorElement{<:ScalarGridSpace}, f::Tuple{Any};
    markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {N} = Rₕ!(uₕ, f[1]; markers = markers)

# One function per component: each is already independent, so restrict each
# component with its own function. Masked restriction routes componentwise through _Rₕ_masked!.
@inline function _Rₕ_masked!(uₕ::VectorElement{<:CompositeGridSpace}, f::Tuple,
        markers::NTuple{N, Symbol}) where {N}
    map((c, g) -> _Rₕ_masked!(c, g, markers), components(uₕ), f)
    return uₕ
end

# The general keyword method, typed as broadly as `VectorElement` so it stays less specific
# than every plain method above, matching the split `avgₕ!` uses. The `N == 0` case never
# actually runs (the plain methods intercept a no-kwarg call before this method is even
# looked up), but is kept as a fallback for an explicit `markers = ()`.
Base.@constprop :aggressive function Rₕ!(uₕ::VectorElement, f::F;
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {F, N}
    if N > 0
        @debug "Using marker-based restriction" markers
    end

    if N == 0
        return Rₕ!(uₕ, f)
    end

    return _Rₕ_masked!(uₕ, f, markers)
end

function _Rₕ_masked!(uₕ::VectorElement{<:ScalarGridSpace}, f::F, markers::NTuple{
        N, Symbol}) where {F, N}
    (; space) = uₕ
    Ωₕ = mesh(space)
    raw = values(uₕ)
    idxs = indices(Ωₕ)
    n = length(idxs)

    fill!(raw, zero(eltype(raw)))
    for m in markers
        mask = index_in_marker(Ωₕ, m)
        @inbounds for i in 1:n
            if mask[i]
                raw[i] = f(point(Ωₕ, idxs[i]))
            end
        end
    end
    return uₕ
end

# As `_Rₕ_scatter_parallel!` above: only valid as a single shared-mesh pass when every
# leaf sits on the same mesh; a heterogeneous composite instead uses each leaf's own
# marker mask and grid points, re-evaluating `f` per leaf through the scalar
# `_Rₕ_masked!` and keeping only that leaf's tuple entry (gpena/Bramble.jl#78).
function _Rₕ_masked!(uₕ::VectorElement{<:CompositeGridSpace}, f::F,
        markers::NTuple{N, Symbol}) where {F, N}
    comps = components(uₕ)
    if _shares_one_mesh(comps)
        Ωₕ = mesh(space(uₕ))
        raws = map(values, comps)
        idxs = indices(Ωₕ)
        n = length(idxs)

        for raw in raws
            fill!(raw, zero(eltype(raw)))
        end
        for m in markers
            mask = index_in_marker(Ωₕ, m)
            @inbounds for i in 1:n
                if mask[i]
                    vals = f(point(Ωₕ, idxs[i]))
                    _write_components!(raws, vals, i)
                end
            end
        end
    else
        ntuple(
            k -> (_Rₕ_masked!(comps[k], pt -> f(pt)[k], markers); nothing),
            Val(length(comps)))
    end
    return uₕ
end

# The coefficient type of a restriction is the one `f` returns, promoted against the
# backend's. Promoted rather than taken outright so that an integer-valued `f` still gives
# a Float64 element on a Float64 backend, while a ForwardDiff.Dual-valued one gives a Dual
# element over the same, undifferentiated, Float64 mesh.
#
# The type is read from one evaluation at the first grid point. That is one extra call to
# `f` per restriction, against inferring it, which would have to guess at a return type
# the compiler may not know.
@inline _scalar_value_type(::Type{T}) where {T} = T
# The `eltype` of a tuple type is the join of its field types, so `eltype(Tuple{Float64, Int})`
# is `Real` (abstract), which would cause `element(Wₕ, Real)` to allocate boxed pointers
# lacking memory contiguity and SIMD optimization. An integer literal among the components
# is enough to trigger it: `Rₕ(Vₕ, x -> (1.0, 2))` would infer `eltype = Real`.
#
# Calling `promote_type` across the component field types preserves concrete numeric types
# (e.g. `Float64`), consistent with scalar arithmetic, and is unchanged for homogeneous tuples.
@inline _scalar_value_type(::Type{T}) where {T <: Tuple} = promote_type(fieldtypes(T)...)

@inline _restricted_value_type(f, p) = _scalar_value_type(typeof(f(p)))
@inline _restricted_value_type(f::Tuple, p) = promote_type(map(
    g -> _scalar_value_type(typeof(g(p))), f)...)

# Selects a sample point where `f` is evaluated to determine its coefficient return type.
# When markers are specified, the point must reside within the marked region because `f`
# may be undefined outside this domain (e.g., functions valid only on a specific boundary).
#
# If no index is marked, nothing is written and the element type cannot matter, so the
# first grid point is as good as any.
@inline _probe_point(Ωₕ, ::NTuple{0, Symbol}) = point(Ωₕ, first(indices(Ωₕ)))

function _probe_point(Ωₕ, markers::NTuple{N, Symbol}) where {N}
    idxs = indices(Ωₕ)
    lin = LinearIndices(idxs)
    for m in markers
        mask = index_in_marker(Ωₕ, m)
        @inbounds for idx in idxs
            mask[lin[idx]] && return point(Ωₕ, idx)
        end
    end
    return point(Ωₕ, first(idxs))
end

@inline function _restriction_eltype(Wₕ::AbstractSpaceType, f,
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {N}
    Ωₕ = mesh(Wₕ)
    return promote_type(
        eltype(backend(Wₕ)), _restricted_value_type(f, _probe_point(Ωₕ, markers)))
end

"""
    Rₕ(Wₕ::AbstractSpaceType, f; markers = ()) -> VectorElement

Standard nodal restriction operator. Evaluates `f` at the grid points of `mesh(Wₕ)`
and returns the result as a [`VectorElement`](@ref).

# Arguments

  - `Wₕ::AbstractSpaceType`: grid space on which to restrict `f`.
  - `f`: function of one grid point. It receives a scalar on a 1D mesh and an
    `NTuple{D}` on a `D`-dimensional one, never an `SVector`.

# Keywords

  - `markers::NTuple{N,Symbol}`: restrict evaluation to the named marked
    regions, leaving every other entry zero.

# Examples

```julia
Rₕ(Wₕ, x -> sin(x))                # 1D: x is a Float64
Rₕ(Wₕ, x -> sin(x[1]) * x[2])      # 2D: x is a Tuple{Float64,Float64}

# Vector-valued spaces:
Rₕ(Vₕ, (f₁, f₂))                   # one function per component
Rₕ(Vₕ, x -> (f₁(x), f₂(x)))        # one function returning all components
```

Prefer `x -> (f₁(x), f₂(x))` when components share computation, as it evaluates once
per grid point on a space whose components share one mesh, whereas `(f₁, f₂)` always
evaluates each component function separately. On a heterogeneous composite — components
built over different meshes — the single-function form gives up that advantage, since
there is no grid point shared by every component to evaluate it at only once.

See also: [`Rₕ!`](@ref), [`avgₕ`](@ref).
"""
function Rₕ(Wₕ::AbstractSpaceType, f; markers::NTuple{N, Symbol} = NTuple{
        0, Symbol}()) where {N}
    uₕ = element(Wₕ, _restriction_eltype(Wₕ, f, markers))
    return Rₕ!(uₕ, f; markers = markers)
end
