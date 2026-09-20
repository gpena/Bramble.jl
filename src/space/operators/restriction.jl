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

On a Metal (or other device) backend, `f` runs *on the device*, evaluated once per grid
point inside a compiled kernel (gpena/Bramble.jl#94, #174). It must therefore be
GPU-compilable: `x -> sin(x[1])` is, but a closure boxing a captured value (a `Ref`, a
`mutable struct` field, a global variable) or calling a function the device compiler cannot
inline is not, and fails at kernel-compile time with a `GPUCompiler.InvalidIRError` naming
the offending call -- inherent to GPU execution, not specific to `Rₕ!`. A masked call
(`markers` non-empty) and a mesh of more than one dimension currently fall back to the same
per-index sweep the CPU backend uses, which is not device-compatible either.

For an `N`-component element either shape of `f` works and both give the same result;
the single vector-valued function is evaluated once per grid point when every
component shares the same mesh, whereas the tuple always evaluates each component
function separately. On a heterogeneous composite, components built over different
meshes, the single-function form is instead re-evaluated once per component, since
there is no grid point shared by every component to evaluate it at only once:

```julia
Rₕ!(uₕ, (f₁, f₂))                     # one function per component
Rₕ!(uₕ, x -> (f₁(x), f₂(x)))          # one function returning all components
```

See also: [`Rₕ`](@ref), [`avgₕ!`](@ref), [`element`](@ref)
"""
@inline Rₕ!(uₕ::VectorElement{<:ScalarGridSpace}, f::F) where {F} = project!(uₕ, PointValue(f))
@inline Rₕ!(uₕ::VectorElement{<:CompositeGridSpace}, f::F) where {F} = project!(uₕ, PointValue(f))
@inline Rₕ!(uₕ::VectorElement{<:CompositeGridSpace}, f::Tuple) = project!(uₕ, map(PointValue, f))

# A concretely typed kernel for per-point restriction calls, avoiding anonymous closure
# captures over (`f`, `Ωₕ`, `idxs`). A named callable struct eliminates compiler indirection
# and achieves performance parity with a flat loop.
struct _RₕKernel{F, M, IX}
    f::F
    Ω::M
    idxs::IX
end
@inline (k::_RₕKernel)(i) = k.f(point(k.Ω, k.idxs[i]))

# The plain methods above ensure that unmasked restriction calls (the primary path during
# time stepping) resolve directly without invoking Julia's keyword argument dispatch
# machinery. They take precedence over the generic `uₕ::VectorElement` keyword method for
# concrete scalar and composite elements.
#
# `PointValue`'s side of the `project!` contract (`operators/projection.jl`). The same
# kernel serves the scalar and the scattered case: `f` returns a scalar on a scalar space
# and the leaves' tuple on a composite one, which is exactly what each sweep wants.
@inline _rule_kernel(rule::PointValue, sp) = _RₕKernel(rule.f, mesh(sp), indices(mesh(sp)))

@inline _rule_scatter_kernel(rule::PointValue, sp, ::Val{NC}) where {NC} = _RₕKernel(rule.f, mesh(sp), indices(mesh(sp)))

@inline _rule_component(rule::PointValue, k) = PointValue(pt -> rule.f(pt)[k])

#------------------------------------------------------------------------------------------#
# Device kernel launch stubs (gpena/Bramble.jl#94, #174, S2.3 of
# .agents/plans/metal-and-apple-silicon-acceleration.md)
#
# `PointValue`'s side of the `_device_project!`/`_device_scatter_project!` contract
# (`operators/projection.jl`): a 1D leaf space evaluates `rule.f` directly against the
# mesh's own coordinate vector `points(Ωₕ)`, through `ka_device` and one of the launchers
# below, which `ext/BrambleKernelAbstractionsExt.jl` implements as a
# `KernelAbstractions.@kernel`. Without that extension loaded, the launcher throws a named
# diagnostic instead of failing several frames later on a scalar index.
#------------------------------------------------------------------------------------------#

@noinline function _throw_no_ka_projection_kernel(fname::String)
    return error(
        "$fname requires KernelAbstractions.jl to fill a device-backed VectorElement. Add " *
        "`using KernelAbstractions` (and the package providing this backend's device, e.g. " *
        "`using Metal`) before calling Rₕ!/Rₕ on this backend.",
    )
end

# Deliberately untyped on the array/kernel arguments (matching the `ka_device`/
# `_launch_uniform_points!` fallback idiom): the extension's methods are typed on
# `AbstractVector`/`Tuple`, and a fallback with the same signature would overwrite them
# instead of adding a genuinely more specific dispatch.
_launch_restriction!(v, pts, f, dev) = _throw_no_ka_projection_kernel("_launch_restriction!")
_launch_restriction_scatter!(mats, pts, f, dev) = _throw_no_ka_projection_kernel("_launch_restriction_scatter!")

# The `D >= 2` counterparts: a tensor-product mesh stores one coordinate vector per axis
# (`points(Ωₕ::MeshnD) -> NTuple{D,AbstractVector}`), so the point at Cartesian index `I`
# is `ntuple(d -> pts[d][I[d]], Val(D))` -- built *inside* the kernel from `pts` (a `Tuple`
# of top-level device arrays, which `Adapt.jl` does convert element-wise) and `idxs`
# (`indices(Ωₕ)`, a bits `CartesianIndices`), never from the mesh object itself. See the
# module comment at the top of `ext/BrambleKernelAbstractionsExt.jl` for why that
# distinction is load-bearing on a device.
_launch_restriction_nd!(v, pts, idxs, f, dev) = _throw_no_ka_projection_kernel("_launch_restriction_nd!")
function _launch_restriction_scatter_nd!(mats, pts, idxs, f, dev)
    _throw_no_ka_projection_kernel(
        "_launch_restriction_scatter_nd!"
    )
end

"""
    _device_project!(::GpuPolicy, rule::PointValue, raw::AbstractVector, sp::ScalarGridSpace{1}) -> Bool
    _device_project!(::GpuPolicy, rule::PointValue, raw::AbstractVector, sp::ScalarGridSpace{D}) where {D} -> Bool

Fills `raw` with `rule.f` evaluated at every point of the mesh `mesh(sp)`, via a device
kernel (gpena/Bramble.jl#94, #174, S2.3). The 1D method reads the mesh's own coordinate
vector directly; the `D`-dimensional method (`D` here is never `1`, since the method above
is strictly more specific and wins dispatch for it) builds each point from the `D` per-axis
coordinate vectors instead. `rule.f` runs on the device either way: see [`_gpu_for!`](@ref)
for what that requires of it.
"""
@inline function _device_project!(
        ::GpuPolicy, rule::PointValue, raw::AbstractVector, sp::ScalarGridSpace{1}
)
    Ωₕ = mesh(sp)
    dev = ka_device(backend(sp))
    _launch_restriction!(raw, points(Ωₕ), rule.f, dev)
    return true
end

@inline function _device_project!(
        ::GpuPolicy, rule::PointValue, raw::AbstractVector, sp::ScalarGridSpace{D}
) where {D}
    Ωₕ = mesh(sp)
    dev = ka_device(backend(sp))
    _launch_restriction_nd!(raw, points(Ωₕ), indices(Ωₕ), rule.f, dev)
    return true
end

"""
    _device_scatter_project!(::GpuPolicy, rule::PointValue, raws::Tuple, sp, ::Val{NC}) -> Bool

The scatter counterpart of [`_device_project!`](@ref) above, for an `NC`-component
composite space `sp` whose leaves share one mesh: `rule.f` is evaluated once per point and
its `NC` components scattered into `raws` in the same device kernel. `sp` is typed
generically (not `ScalarGridSpace`) because the caller is always the composite branch of
[`project!`](@ref); the dimension the mesh has determines which launcher runs (1D or the
`D`-dimensional counterpart), checked on `mesh(sp)` at runtime since it cannot be expressed
as a type constraint on the composite space itself.
"""
@inline function _device_scatter_project!(
        ::GpuPolicy, rule::PointValue, raws::Tuple, sp, ::Val{NC}
) where {NC}
    Ωₕ = mesh(sp)
    dev = ka_device(backend(sp))
    if Ωₕ isa AbstractMeshType{1}
        _launch_restriction_scatter!(raws, points(Ωₕ), rule.f, dev)
    else
        _launch_restriction_scatter_nd!(raws, points(Ωₕ), indices(Ωₕ), rule.f, dev)
    end
    return true
end

# A one-component space is a scalar space, so generic code that builds an
# NC-tuple of functions still works when NC == 1.
@inline Rₕ!(uₕ::VectorElement{<:ScalarGridSpace{D}}, f::Tuple{Any}) where {D} = Rₕ!(uₕ, f[1])
@inline Rₕ!(
    uₕ::VectorElement{<:ScalarGridSpace},
    f::Tuple{Any};
    markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {N} = Rₕ!(uₕ, f[1]; markers = markers)

# The general keyword method, typed as broadly as `VectorElement` so it stays less specific
# than every plain method above, matching the split `avgₕ!` uses. The `N == 0` case never
# actually runs (the plain methods intercept a no-kwarg call before this method is even
# looked up), but is kept as a fallback for an explicit `markers = ()`.
Base.@constprop :aggressive function Rₕ!(
        uₕ::VectorElement, f::F; markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {F, N}
    if N > 0
        @debug "Using marker-based restriction" markers
    end

    if N == 0
        return Rₕ!(uₕ, f)
    end

    return project!(uₕ, _point_rule(f), markers)
end

# A tuple of functions is a rule per leaf; anything else is one rule.
@inline _point_rule(f::Tuple) = map(PointValue, f)
@inline _point_rule(f) = PointValue(f)

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
@inline _restricted_value_type(f::Tuple, p) = promote_type(map(g -> _scalar_value_type(typeof(g(p))), f)...)

# Selects a sample point where `f` is evaluated to determine its coefficient return type.
# When markers are specified, the point must reside within the marked region because `f`
# may be undefined outside this domain (e.g., functions valid only on a specific boundary).
#
# If no index is marked, nothing is written and the element type cannot matter, so the
# first grid point is as good as any.
@inline _probe_point(Ωₕ, ::NTuple{0, Symbol}) = _probe_point_value(Ωₕ, first(indices(Ωₕ)))

function _probe_point(Ωₕ, markers::NTuple{N, Symbol}) where {N}
    idxs = indices(Ωₕ)
    for m in markers
        # `findfirst` on a `BitVector` already scans a word (64 bits) at a time via
        # `trailing_zeros`, rather than testing one Cartesian index per iteration.
        i = findfirst(index_in_marker(Ωₕ, m))
        i === nothing || return _probe_point_value(Ωₕ, idxs[i])
    end
    return _probe_point_value(Ωₕ, first(idxs))
end

# `point(Ωₕ, idx)` (src/mesh/mesh1d.jl) is `points(Ωₕ)[idx]`, a scalar `getindex` that
# `GPUArraysCore` disallows outside a kernel once `points(Ωₕ)` is device-backed
# (gpena/Bramble.jl#94, #174, S2.3 of .agents/plans/metal-and-apple-silicon-acceleration.md).
# `Rₕ`/`avgₕ` (never `Rₕ!`/`avgₕ!`) reach this once per call, on the host, purely to sample
# `f`'s return type -- not a hot path -- so the fix is a single-element array-to-array copy
# rather than teaching `point` itself about the backend.
@inline _probe_point_value(Ωₕ, idx) = point(Ωₕ, idx)
@inline function _probe_point_value(Ωₕ::AbstractMeshType{1}, idx)
    return _probe_point_value(points(Ωₕ), Ωₕ, idx)
end
@inline _probe_point_value(::Array, Ωₕ, idx) = point(Ωₕ, idx)
@inline function _probe_point_value(pts::AbstractVector, Ωₕ, idx)
    i = _extract_linear_index(idx)
    return @inbounds Array(view(pts, i:i))[1]
end

# `MeshnD` (`D >= 2`): `point(Ωₕ, idx)` (src/mesh/meshnd.jl) combines each axis's own
# scalar read via `@generate_mesh_ntuple_func_with_idx point`, one `getindex` per axis on
# that axis's own coordinate vector -- the identical failure the method above fixes for a
# 1D mesh. Rather than a second array-copy implementation, this probes each submesh (a
# `Mesh1D`, so `Ωₕ(d)` reaches the very method above) with its own share of `idx`. Coexists
# with the `AbstractMeshType{1}` method without ambiguity: that one is strictly more
# specific and wins dispatch whenever `D == 1`.
@inline function _probe_point_value(Ωₕ::AbstractMeshType{D}, idx) where {D}
    return ntuple(d -> _probe_point_value(Ωₕ(d), idx[d]), Val(D))
end

@inline function _restriction_eltype(
        Wₕ::AbstractSpaceType, f, markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {N}
    Ωₕ = mesh(Wₕ)
    return promote_type(
        eltype(backend(Wₕ)), _restricted_value_type(f, _probe_point(Ωₕ, markers))
    )
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
evaluates each component function separately. On a heterogeneous composite (components
built over different meshes), the single-function form gives up that advantage, since
there is no grid point shared by every component to evaluate it at only once.

See also: [`Rₕ!`](@ref), [`avgₕ`](@ref).
"""
function Rₕ(
        Wₕ::AbstractSpaceType, f; markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {N}
    uₕ = element(Wₕ, _restriction_eltype(Wₕ, f, markers))
    return Rₕ!(uₕ, f; markers = markers)
end
