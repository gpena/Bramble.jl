##############################################################################
#                                                                            #
#            Projecting a function onto the space of grid functions          #
#                                                                            #
##############################################################################

#=
# projection.jl

`CONTEXT.md` defines restriction and cell average as the same operation with different
rules: both project a continuous function onto the space of grid functions, one by
evaluating it at the mesh points, the other by averaging it over each cell. This file is
that shared operation, with the rule as an argument (gpena/Bramble.jl#51).

The driver decides the three things neither rule should have an opinion about -- how the
space is shaped (scalar, composite on one mesh, composite across several), whether a marker
mask applies, and whether the sweep is threaded -- and asks the rule only for a per-point
kernel. `Rₕ!` (`restriction.jl`) and `avgₕ!` (`cell_average.jl`) keep their own public
signatures and docstrings and become thin wrappers over `project!`.

**The threading trap this closes.** Before this, the masked paths of both families were
plain serial loops that ignored `execution_policy`, while the unmasked ones threaded. A
caller who added `markers = (...)` silently lost threading, with nothing in either
interface saying so. Masking is now a property of the kernel (`_MaskedKernel` below), not
of the sweep, so both go through the same `_sweep_for!` and the policy is honoured
either way.
=#

"""
    ProjectionRule

Supertype of the rules [`project!`](@ref) can apply: what a grid function's value at a
point should be, given a continuous function. [`PointValue`](@ref) evaluates at the mesh
point, [`CellAverage`](@ref) averages over the cell.
"""
abstract type ProjectionRule end

"""
    PointValue(f)

Project `f` by evaluating it at each mesh point. The rule behind [`Rₕ!`](@ref).
"""
struct PointValue{F} <: ProjectionRule
    f::F
end

"""
    CellAverage(f, nq::Val)

Project `f` by averaging it over each cell with `nq` quadrature points per direction. The
rule behind [`avgₕ!`](@ref).
"""
struct CellAverage{F, NQ} <: ProjectionRule
    f::F
    nq::Val{NQ}
end

# --- what a rule has to provide, implemented in each rule's own file --------------- #

"""
    _rule_kernel(rule, space) -> callable

A concretely typed callable mapping a linear grid index to the scalar value `rule` gives
at that point. Kept a named struct per rule rather than a closure: measured, an anonymous
closure over the captures here takes a miscompiled path that allocates per grid point
(gpena/Bramble.jl#64).
"""
function _rule_kernel end

"""
    _rule_scatter_kernel(rule, space, ::Val{NC}) -> callable

As [`_rule_kernel`](@ref), for a rule whose function returns all `NC` leaf values at once,
so it is evaluated once per point and scattered across the leaves.
"""
function _rule_scatter_kernel end

"""
    _rule_component(rule, k) -> ProjectionRule

`rule` restricted to leaf `k` of a composite space whose leaves do not share one mesh, so
there is no shared grid point to evaluate once and scatter.
"""
function _rule_component end

# --- the device fast path (gpena/Bramble.jl#94, #174, S2.3 of
# .agents/plans/metal-and-apple-silicon-acceleration.md) ------------------------------- #
#
# `_rule_kernel`/`_rule_scatter_kernel` build a callable closing over the mesh itself
# (`_RₕKernel`, `_AvgKernel`, `_AvgScatterKernel`), which is exactly right for the CPU sweep
# below but not GPU-compilable: a mesh is a `mutable struct` carrying a `Dict` of markers,
# and passing one into a `KernelAbstractions.@kernel` fails to compile with "passing
# non-bitstype argument" the moment the kernel is launched, well before `f` itself is ever
# reached (confirmed against a real Metal device while designing this). `_device_project!`/
# `_device_scatter_project!` are the alternative a rule can offer instead: a kernel built
# only from the mesh's own coordinate *arrays*, passed to the device launcher as their own
# top-level arguments rather than nested inside a wrapper struct.
#
# Both default to `false` -- "no device kernel for this rule/space combination" -- so
# `project!` falls back to the generic sweep above unchanged: on `HostLocality` that is
# every call (the CPU path never even asks); on `DeviceLocality` every `PointValue`/
# `CellAverage`, masked or not, on a mesh of any dimension answers `true`. A future rule
# this file has not been taught about falls through to `_sweep_for!`/`_sweep_scatter_for!`'s
# own `DeviceLocality` handling, which either runs (if `f`'s fields happen to be GPU-safe)
# or fails with a real device compiler diagnostic -- never silently the wrong answer.
#
# Both take an optional last argument `sel`: `nothing` for an unmasked call, or a device
# index vector naming the only linear indices to fill (see `_device_masked_project!` below).

"""
    _device_project!(loc, rule, raw, sp, sel = nothing) -> Bool

Attempt a dedicated device kernel filling `raw` (a leaf's coefficient vector) according to
`rule` on space `sp`. Returns `true` when it did; `false` when no such kernel exists for
this `rule`/`sp` combination on this locality, so the caller runs the generic
[`_sweep_for!`](@ref) sweep instead. Always `false` on [`HostLocality`](@ref): the
CPU sweep already indexes `raw` directly and never needs this path. Legality here is a
locality question, not a policy question (gpena/Bramble.jl#298): a device kernel either
exists for this destination's storage or it doesn't, regardless of which strategy the
policy names within that locality. A device index vector `sel` restricts the fill to those
linear indices and leaves every other entry of `raw` untouched.
"""
@inline _device_project!(::HostLocality, rule, raw, sp, sel = nothing) = false
@inline _device_project!(::DeviceLocality, rule, raw, sp, sel = nothing) = false

"""
    _device_scatter_project!(loc, rule, raws::Tuple, sp, ::Val{NC}, sel = nothing) -> Bool

The scatter counterpart of [`_device_project!`](@ref): a dedicated device kernel filling
every leaf array in `raws` (`NC` of them, all sharing `sp`'s mesh) in one launch. Same
`true`/`false` contract, keyed on locality for the same reason.
"""
@inline _device_scatter_project!(::HostLocality, rule, raws, sp, ::Val, sel = nothing) = false
@inline _device_scatter_project!(::DeviceLocality, rule, raws, sp, ::Val, sel = nothing) = false

# --- the masked device path (gpena/Bramble.jl#297) --------------------------------- #
#
# The host path folds the mask into the kernel (`_MaskedKernel` below), which closes over the
# mesh's `BitVector`s -- a struct nesting arrays, so not a kernel argument on a device, and a
# `BitVector` could not be one anyway. The device path launches over an index list instead:
# the marked linear indices, gathered on the host from the mesh's own (host-resident) masks
# once per call and uploaded one-way as an `Int32` vector, a top-level kernel argument of its
# own. A `:boundary` marker is a vanishing fraction of the grid, so launching over the list
# rather than over every point with a predicate is also the cheaper of the two. Nothing is
# cached on the device: a persistent copy would go stale under `set_markers!`.
#
# The destination is zeroed on the device first, so off-region entries end up zero exactly
# as the host `_MaskedKernel` path leaves them. The zero fill and the scatter are queued on
# the same device stream, so the scatter always lands after it.

"""
    _marker_index_list(like, Ωₕ, markers) -> AbstractVector{Int32}

The linear indices of `Ωₕ`'s points lying in any of the `markers` regions, gathered on the
host from [`index_in_marker`](@ref)'s masks and copied into a new `Int32` vector allocated
like `like` (so on `like`'s device).

# Throws
- `ErrorException`: the mesh has more points than `typemax(Int32)`.
"""
function _marker_index_list(like::AbstractArray, Ωₕ, markers::NTuple{N, Symbol}) where {N}
    masks = _marker_masks(Ωₕ, markers)
    mask = N == 1 ? masks[1] : reduce(.|, masks)
    length(mask) <= typemax(Int32) || error(
        "mesh too large for a masked device projection: $(length(mask)) points, past " *
        "typemax(Int32) = $(typemax(Int32))",
    )
    host = Int32.(findall(mask))
    sel = similar(like, Int32, length(host))
    copyto!(sel, host)
    return sel
end

"""
    _device_masked_project!(loc, rule, raw, sp, markers) -> Bool
    _device_masked_project!(loc, rule, raws::Tuple, sp, markers, ::Val{NC}) -> Bool

The masked counterparts of [`_device_project!`](@ref)/[`_device_scatter_project!`](@ref):
on [`DeviceLocality`](@ref), zero the destination(s) on the device and fill only the points
in the union of the `markers` regions, through the rule's own device kernel launched over
[`_marker_index_list`](@ref). Always `false` on [`HostLocality`](@ref), where the generic
sweep folds the mask into its kernel instead.
"""
@inline _device_masked_project!(::HostLocality, rule, raw, sp, markers) = false
function _device_masked_project!(loc::DeviceLocality, rule, raw, sp, markers)
    fill!(raw, zero(eltype(raw)))
    sel = _marker_index_list(raw, mesh(sp), markers)
    isempty(sel) && return true
    return _device_project!(loc, rule, raw, sp, sel)
end

@inline _device_masked_project!(::HostLocality, rule, raws, sp, markers, ::Val) = false
function _device_masked_project!(loc::DeviceLocality, rule, raws, sp, markers, nc::Val)
    foreach(r -> fill!(r, zero(eltype(r))), raws)
    sel = _marker_index_list(raws[1], mesh(sp), markers)
    isempty(sel) && return true
    return _device_scatter_project!(loc, rule, raws, sp, nc, sel)
end

# --- the offloaded path (gpena/Bramble.jl#324) ------------------------------------- #
#
# A space whose backend carries a `GpuOffload` policy keeps host storage and a host mesh, so
# `raw` answers `HostLocality` and the dispatch above never reaches a device kernel. Instead,
# `project!` fills a device buffer allocated per call through the wrapped device backend,
# with the same `_device_project!`/`_device_masked_project!` machinery a device-resident
# space uses, then copies the result back into the host destination and drops the buffer.
# The rules' device methods upload the mesh axes they need with `_on_device` (a no-op for a
# device-resident mesh) and launch on `_device_backend`'s device. Nothing is cached on the
# space, the mesh or the policy: a persistent device copy would go stale under a mesh or
# marker change, the hazard #313 removed.

"""
    _offload_backend(backend::Backend) -> Union{Backend, Nothing}

The device backend a [`GpuOffload`](@ref) policy wraps, or `nothing` for any other policy,
so [`project!`](@ref) knows whether to fill through the device.
"""
@inline _offload_backend(::Backend{VT, MT, EP}) where {VT, MT, EP} = _offload_backend(EP)
@inline _offload_backend(::Type{<:ExecutionPolicy}) = nothing
@inline _offload_backend(::Type{GpuOffload{I, DB}}) where {I, DB} = DB()

"""
    _device_backend(backend::Backend) -> Backend

The backend whose device a projection kernel launches on: the wrapped device backend for a
[`GpuOffload`](@ref) policy, otherwise `backend` itself.
"""
@inline _device_backend(be::Backend) = something(_offload_backend(be), be)

"""
    _on_device(like, x) -> AbstractVector or Tuple

`x` (a mesh coordinate vector, or a tuple of them) as device arrays allocated like `like`:
returned as is when already device-resident, otherwise copied into a new device array.
"""
@inline _on_device(like, x::Tuple) = map(a -> _on_device(like, a), x)
@inline _on_device(like, x::AbstractVector) = _on_device(locality(typeof(x)), like, x)
@inline _on_device(::DeviceLocality, like, x) = x
@inline _on_device(::HostLocality, like, x) = copyto!(similar(like, eltype(x), length(x)), x)

"""
    _offload_project!(db, rule, raw, sp, markers) -> Bool
    _offload_project!(db, rule, raws::Tuple, sp, markers, ::Val{NC}) -> Bool

Fill the host destination(s) through device backend `db`: allocate a device buffer per
destination, run the rule's device kernel on it (masked when `markers` is non-empty) and
copy it back. `false`, with the destination untouched, when the rule has no device kernel
or the destination's element type is not the device's (`_offload_representable`).
"""
function _offload_project!(db::Backend, rule, raw, sp, markers::NTuple{N, Symbol}) where {N}
    _offload_representable(db, raw) || return false
    draw = vector(db, length(raw))
    loc = locality(typeof(draw))
    done = N == 0 ? _device_project!(loc, rule, draw, sp) : _device_masked_project!(loc, rule, draw, sp, markers)
    done && _copy_back!(raw, draw)
    return done
end

function _offload_project!(db::Backend, rule, raws::Tuple, sp, markers::NTuple{N, Symbol}, nc::Val) where {N}
    _offload_representable(db, raws[1]) || return false
    draws = map(r -> vector(db, length(r)), raws)
    loc = locality(typeof(draws[1]))
    done = N == 0 ? _device_scatter_project!(loc, rule, draws, sp, nc) :
           _device_masked_project!(loc, rule, draws, sp, markers, nc)
    done && foreach(_copy_back!, raws, draws)
    return done
end

# A destination of another element type than the device buffer (`Rₕ` of a `Float64`-valued
# function on a `Float32` space) stays on the host: the device buffer would round every value
# to its own type and the copy back would hand the rounded values over silently.
@inline _offload_representable(db::Backend, raw) = eltype(vector_type(db)) === eltype(raw)

# A composite's leaves are views into one shared host vector, which a device array cannot
# `copyto!` into without scalar indexing, so those go through a host copy first.
@inline _copy_back!(dst::Array, src) = copyto!(dst, src)
@inline _copy_back!(dst, src) = copyto!(dst, Array(src))

# --- masking, as a property of the kernel rather than of the sweep ----------------- #

# The marker masks, as a tuple whose length is a type parameter so the `||` chain below
# unrolls. `index_in_marker` is a dictionary lookup returning the mesh's own array, so this
# copies nothing.
@inline _marker_masks(Ωₕ, markers::NTuple{N, Symbol}) where {N} = ntuple(i -> index_in_marker(Ωₕ, markers[i]), Val(N))

@inline _in_any_marker(::Tuple{}, i) = false
@inline _in_any_marker(masks::Tuple, i) = (@inbounds masks[1][i]) || _in_any_marker(Base.tail(masks), i)

# Off-region entries are written as zero rather than skipped. The serial loops this
# replaces did `fill!(raw, 0)` first and then wrote only the masked entries, which leaves
# exactly the same array -- but writing every entry is what lets the masked sweep be the
# same threaded sweep as the unmasked one.
struct _MaskedKernel{K, M, Z}
    kernel::K
    masks::M
    zeroval::Z
end

@inline (mk::_MaskedKernel)(i) = _in_any_marker(mk.masks, i) ? mk.kernel(i) : mk.zeroval

# --- the driver -------------------------------------------------------------------- #

"""
    project!(uₕ::VectorElement, rule, markers = ()) -> uₕ

Project onto `uₕ` in place according to `rule`, optionally restricted to the union of the
labelled marker regions, leaving every other entry zero.

Handles the space's shape and the execution policy; `rule` supplies only the per-point
value. See [`PointValue`](@ref) and [`CellAverage`](@ref), and [`Rₕ!`](@ref) / [`avgₕ!`](@ref)
for the public spellings.
"""
function project! end

@inline function project!(
        uₕ::VectorElement{<:ScalarGridSpace},
        rule::ProjectionRule,
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {N}
    sp = space(uₕ)
    Ωₕ = mesh(sp)
    raw = parent(uₕ)
    policy = execution_policy(sp)
    db = _offload_backend(backend(sp))
    if db !== nothing && _offload_project!(db, rule, raw, sp, markers)
        return uₕ
    end
    loc = locality(typeof(raw))
    if N == 0 ? _device_project!(loc, rule, raw, sp) : _device_masked_project!(loc, rule, raw, sp, markers)
        return uₕ
    end
    n = length(indices(Ωₕ))
    kernel = _rule_kernel(rule, sp)
    _sweep_for!(
        policy, raw, 1:n, _apply_mask(kernel, Ωₕ, markers, eltype(raw))
    )
    return uₕ
end

@inline function project!(
        uₕ::VectorElement{<:CompositeGridSpace},
        rule::ProjectionRule,
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {N}
    comps = components(uₕ)
    if _shares_one_mesh(comps)
        sp = space(uₕ)
        Ωₕ = mesh(sp)
        raws = map(parent, comps)
        NC = length(comps)
        policy = execution_policy(sp)
        db = _offload_backend(backend(sp))
        if db !== nothing && _offload_project!(db, rule, raws, sp, markers, Val(NC))
            return uₕ
        end
        loc = locality(typeof(raws[1]))
        if N == 0 ? _device_scatter_project!(loc, rule, raws, sp, Val(NC)) :
           _device_masked_project!(loc, rule, raws, sp, markers, Val(NC))
            return uₕ
        end
        n = length(indices(Ωₕ))
        kernel = _rule_scatter_kernel(rule, sp, Val(NC))
        zeros_nc = ntuple(_ -> zero(eltype(first(raws))), Val(NC))
        _sweep_scatter_for!(
            policy, raws, 1:n, _apply_mask(kernel, Ωₕ, markers, zeros_nc)
        )
    else
        # No shared grid point exists, so `rule`'s function is re-evaluated at each leaf's
        # own points, keeping only that leaf's entry (gpena/Bramble.jl#78).
        ntuple(
            k -> (project!(comps[k], _rule_component(rule, k), markers); nothing),
            Val(length(comps))
        )
    end
    return uₕ
end

# One rule per leaf: each is already independent. `map` over both tuples rather than
# `ntuple` indexing a shared count -- it needs no leaf count, stays correct under any
# nesting, and errors on a length mismatch the way it always did.
@inline function project!(
        uₕ::VectorElement{<:CompositeGridSpace},
        rules::Tuple,
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {N}
    map((c, r) -> project!(c, r, markers), components(uₕ), rules)
    return uₕ
end

# A one-component space is a scalar space, so a 1-tuple of rules must still work.
@inline project!(
    uₕ::VectorElement{<:ScalarGridSpace},
    rules::Tuple{Any},
    markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {N} = project!(uₕ, rules[1], markers)

# Unmasked stays the bare kernel, so the common path carries no mask check at all.
@inline _apply_mask(kernel, Ωₕ, ::NTuple{0, Symbol}, zeroval) = kernel
@inline _apply_mask(kernel, Ωₕ, markers::NTuple{N, Symbol}, zeroval) where {N} = _MaskedKernel(
    kernel, _marker_masks(Ωₕ, markers), _zero_of(zeroval))

@inline _zero_of(::Type{T}) where {T} = zero(T)
@inline _zero_of(z::Tuple) = z
