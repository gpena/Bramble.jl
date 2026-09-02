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
	Rₕ!(uₕ::VectorElement, f; markers = ())

In-place version of the restriction operator [`Rₕ`](@ref). Evaluates `f` at the
grid points and writes the result into `uₕ`. Returns `uₕ`.

# Arguments

  - `uₕ::VectorElement`: pre-allocated element to write into.
  - `f`: function of one grid point. It receives a scalar on a 1D mesh and an
    `NTuple{D}` on a `D`-dimensional one -- never an `SVector`.

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

For an `N`-component element either shape of `f` works and both give the same
result; the single vector-valued function is evaluated once per grid point,
whereas the tuple evaluates each component function separately:

```julia
Rₕ!(uₕ, (f₁, f₂))                     # one function per component
Rₕ!(uₕ, x -> (f₁(x), f₂(x)))          # one function returning all components
```

See also: [`Rₕ`](@ref), [`avgₕ!`](@ref), [`element`](@ref)
"""
@inline Rₕ!(uₕ::VectorElement{<:ScalarGridSpace}, f::F) where {F} = _Rₕ_parallel!(uₕ, f)
@inline Rₕ!(uₕ::VectorElement{<:CompositeGridSpace{NC}}, f::F) where {
    NC, F} = _Rₕ_scatter_parallel!(
    uₕ, f)

# A named, concretely-typed kernel for the per-point restriction call, rather than an
# anonymous closure over the same three captures (`f`, `Ωₕ`, `idxs`).
#
# `avgₕ!` already made this move (`_AvgKernel1`/`_AvgKernelD`, point 51) for an allocation
# flake that this closure, with fewer captures, never showed. It still cost real time: point
# 53's investigation measured Rₕ! itself, closure and all, at ~16% slower than the identical
# computation written as one flat loop with no closure indirection -- and a named struct
# closes essentially the whole gap, the same lever point 51 already pulled for a different
# reason (point 67). One fewer compiler decision between the data and the call either way.
struct _RₕKernel{F, M, IX}
    f::F
    Ω::M
    idxs::IX
end
@inline (k::_RₕKernel)(i) = k.f(point(k.Ω, k.idxs[i]))

# The two plain methods just above exist so that the no-markers call -- by far the common
# case, every time step of a PDE solve -- resolves directly to one of them, never touching
# the keyword-sorter machinery the generic method further below generates. Going through that
# machinery still cost 96 B here, no matter how the callee itself was typed, deep enough under
# a keyword-taking caller that escape analysis gave up on the `_cpu_threaded_for!` closure
# regardless. They stay more specific than the keyword method's bare `uₕ::VectorElement`, so
# they take priority for a concrete Scalar/Composite element without colliding with the
# keyword-sorter's own auto-generated no-kwarg stub. Matches `_avgₕ!`'s plain/keyword split.
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

@inline function _Rₕ_scatter_parallel!(
        uₕ::VectorElement{<:CompositeGridSpace{NC}}, f::F) where {NC, F}
    sp = space(uₕ)
    Ωₕ = mesh(sp)
    comps = components(uₕ)
    raws = ntuple(i -> values(comps[i]), Val(NC))
    idxs = indices(Ωₕ)
    n = length(idxs)
    _cpu_threaded_scatter_for!(execution_policy(sp), raws, 1:n, _RₕKernel(f, Ωₕ, idxs))
    return uₕ
end

# A one-component space is a scalar space, so generic code that builds an
# NC-tuple of functions still works when NC == 1.
@inline Rₕ!(uₕ::VectorElement{<:ScalarGridSpace{D}}, f::Tuple{Any}) where {D} = Rₕ!(uₕ, f[1])
@inline Rₕ!(uₕ::VectorElement{<:ScalarGridSpace}, f::Tuple{Any};
    markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {N} = Rₕ!(uₕ, f[1]; markers = markers)

# One function per component: each is already independent, so restrict each
# component with its own function. No plain-method counterpart: this recurses into the
# scalar `Rₕ!` per component, which already has the fast no-kwarg path above, so there is
# nothing left to gain from also bypassing the kwsorter at this outer level.
@inline function Rₕ!(uₕ::VectorElement{<:CompositeGridSpace{NC}}, f::Tuple;
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {NC, N}
    comps = components(uₕ)
    ntuple(i -> Rₕ!(comps[i], f[i]; markers = markers), Val(NC))
    return uₕ
end

# The general keyword method, typed as broadly as `VectorElement` so it stays less specific
# than every plain method above -- exactly the split `avgₕ!` uses. The `N == 0` case never
# actually runs (the plain methods intercept a no-kwarg call before this method is even
# looked up), but is kept as a fallback for an explicit `markers = ()`.
Base.@constprop :aggressive function Rₕ!(uₕ::VectorElement, f;
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {N}
    if N > 0
        @debug "Using marker-based restriction" markers
    end

    if N == 0
        return Rₕ!(uₕ, f)
    end

    return _Rₕ_masked!(uₕ, f, markers)
end

function _Rₕ_masked!(uₕ::VectorElement{<:ScalarGridSpace}, f, markers::NTuple{
        N, Symbol}) where {N}
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

function _Rₕ_masked!(uₕ::VectorElement{<:CompositeGridSpace{NC}}, f,
        markers::NTuple{N, Symbol}) where {NC, N}
    Ωₕ = mesh(space(uₕ))
    comps = components(uₕ)
    raws = ntuple(i -> values(comps[i]), Val(NC))
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
                _scatter_comp!(raws, vals, i)
            end
        end
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
# `eltype` of a tuple type is the join of its fields, so `eltype(Tuple{Float64, Int})` is
# `Real` — abstract, which would make `element(Wₕ, Real)` allocate a `Vector{Real}` of boxed
# pointers with no contiguity and no SIMD. An integer literal among the components is enough
# to trigger it: `Rₕ(Vₕ, x -> (1.0, 2))` measured `eltype = Real`.
#
# `promote_type` over the field types gives what the arithmetic would give anyway — `Float64`
# there — and is unchanged for a homogeneous tuple.
@inline _scalar_value_type(::Type{T}) where {T <: Tuple} = promote_type(fieldtypes(T)...)

@inline _restricted_value_type(f, p) = _scalar_value_type(typeof(f(p)))
@inline _restricted_value_type(f::Tuple, p) = promote_type(map(
    g -> _scalar_value_type(typeof(g(p))), f)...)

# Where to sample `f` to learn its return type. With markers it has to be a point the
# caller actually selected: `f` need not be defined anywhere else, and probing the first
# grid point regardless turned working calls into errors — `Rₕ(Wₕ, x -> sqrt(x - 0.5);
# markers = (:right,))` threw a DomainError at x = 0 while `Rₕ!` with the same arguments
# succeeded, since the in-place form never probes.
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
	Rₕ(Wₕ::AbstractSpaceType, f; markers = ())

Standard nodal restriction operator. It returns a [`VectorElement`](@ref) with the result of evaluating the function `f` at the points of `mesh(Wₕ)`.

# The shape of `f`

`f` is called with the coordinates of one grid point: a scalar for a 1D mesh and
an `NTuple{D}` for a `D`-dimensional one. It is never passed an `SVector`.

```julia
Rₕ(Wₕ, x -> sin(x))                # 1D:  x is a Float64
Rₕ(Wₕ, x -> sin(x[1]) * x[2])      # 2D:  x is a Tuple{Float64,Float64}
```

For an `N`-component space either form works and both give the same result:

```julia
Rₕ(Vₕ, (f₁, f₂))                   # one function per component
Rₕ(Vₕ, x -> (f₁(x), f₂(x)))        # one function returning all components
```

Prefer the second when the components share work: it is evaluated once per grid
point, while the first evaluates each component function separately.

`markers` restricts evaluation to the named marked regions, leaving every other
entry zero. [`avgₕ`](@ref) takes the same keyword; it additionally takes
`quad_points`, which has no meaning here because nodal restriction involves no
quadrature.

See also: [`Rₕ!`](@ref), [`avgₕ`](@ref).
"""
function Rₕ(Wₕ::AbstractSpaceType, f; markers::NTuple{N, Symbol} = NTuple{
        0, Symbol}()) where {N}
    uₕ = element(Wₕ, _restriction_eltype(Wₕ, f, markers))
    return Rₕ!(uₕ, f; markers = markers)
end
