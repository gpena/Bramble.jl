################################################################################
#                      Discrete L² Inner Product and Norm                      #
################################################################################

#=
Which argument each product accepts, and why they differ.

  innerₕ, normₕ, norm₁ₕ, snorm₁ₕ   a grid function on a scalar space
  inner₊, norm₊                    a tuple of grid functions, one per direction

The cell-measure product weights one value per grid point, so it is defined for a
scalar grid function. A component of a composite grid function is itself a scalar
grid function -- `components(uₕ)[i]` has a ScalarGridSpace -- so those are accepted
and are the way to take the product of one component.

The staggered product weights a different direction per entry, so its argument is a
tuple with one grid function per direction: the gradient ∇ₕ(uₕ) is exactly that
shape. In one dimension a one-tuple and the grid function coincide, so the scalar
form is accepted there.

`inner₊` and `innerₕ` are each spelled twice in this package, and the two meanings do not
live in the same layer. Here they take grid functions and return a **number**. In
`src/ast/operators/inner.jl` they take operators and return an **AST node** for a form to
be assembled from. CONTEXT.md draws that line at the domain level: a form is symbolic, a
grid function is data.

The two families are kept from colliding by the `NTuple{N,<:Tuple}` restriction on the
symbolic tuple overload: a tuple of grid functions is not a tuple of tuples, so it cannot
reach the symbolic method, and the `@generated` methods below stay reachable. Widen either
side and the collision is real. The constraint is asserted in
`test/form/inner_products.jl`, testset "Symbolic and numeric families stay apart"
(gpena/Bramble.jl#60), so a change to either signature fails a test rather than silently
returning the wrong kind of thing.

A composite grid function is deliberately not accepted by either. It is a stack of
scalar functions with no single weighting of its own, and summing over its
components is a choice the caller should make explicitly rather than have inferred.
The operators are the other way round: Rₕ, avgₕ and every difference, jump and
average apply componentwise and take any grid function.
=#

#=
`markers`, on `innerₕ` and the directional products, follows the `Rₕ!`/`avgₕ!`
`markers::NTuple{N,Symbol} = NTuple{0,Symbol}()` precedent exactly: the default empty tuple
is a zero-cost, compile-time-known branch to the original unmasked sum, and a non-empty tuple
restricts the sum to the union of the labelled regions' points.

This is a *masked sum of the existing cell measures*, not a surface integral; the two are
not interchangeable and differ by a factor of `h`. Measured on a 5×5 uniform mesh of the unit
square, restricted to `:bottom`: the masked sum here gives 0.125, which scales like `h` and
vanishes under refinement, where the true boundary integral `∫_Γ u v ds` gives 1.0 and is
mesh-independent. See `inner_Γ` below (gpena/Bramble.jl#157).
=#

@inline function _combined_mask(Ωₕ, markers::NTuple{1, Symbol})
    return index_in_marker(Ωₕ, markers[1])
end

# `N > 1`: the stored per-marker masks are combined by union into a fresh `BitVector`, since
# a sum (unlike `Rₕ!`'s per-marker write) must not double-count a point two markers both
# cover. `N == 1` above skips this allocation, since the mesh's own stored mask can be read
# directly without being unioned with anything.
#
# `dirichlet_bc!` (form/dirichlet_constraints.jl) is the other caller of this exact function,
# and its `SparseMatrixCSC` path needs random-access `getindex` on the result (`index_in_marker[j]`
# for an arbitrary column `j`, not a walk), which a lazy union can't offer without also
# duplicating `BitVector`'s own bit-indexing math. So this stays eager; `_combined_marked_indices`
# below is the innerₕ-only, allocation-free replacement (gpena/Bramble.jl#149), used solely by
# code that only ever walks the result instead of indexing into it.
function _combined_mask(Ωₕ, markers::NTuple{N, Symbol}) where {N}
    mask = copy(index_in_marker(Ωₕ, markers[1]))
    for k in 2:N
        mask .|= index_in_marker(Ωₕ, markers[k])
    end
    return mask
end

@inline function _combined_marked_indices(Ωₕ, markers::NTuple{1, Symbol})
    return index_in_marker(Ωₕ, markers[1])
end

# `N > 1`: only ever handed to `_dot_masked`, which walks it and never indexes into it, so
# the union can stay a lazy `MarkedIndicesUnion` (utils/linear_algebra.jl) instead of a
# freshly allocated `BitVector` -- unlike `_combined_mask` above, this has exactly one
# caller family (`innerₕ`/`_directional_inner_plus` below) and can afford to (gpena/Bramble.jl#149).
@inline function _combined_marked_indices(Ωₕ, markers::NTuple{N, Symbol}) where {N}
    return MarkedIndicesUnion(ntuple(k -> index_in_marker(Ωₕ, markers[k]), Val(N)))
end

"""
    innerₕ(uₕ::VectorElement, vₕ::VectorElement; markers = ()) -> Real

Returns the discrete ``L^2`` inner product of the grid functions `uₕ` and `vₕ`, weighting each point by its cell measure.

  - 1D case

```math
(\\textrm{u}_h, \\textrm{v}_h)_h \\vcentcolon = \\sum_{i=1}^N |\\square_{i}| \\textrm{u}_h(x_i) \\textrm{v}_h(x_i)
```

  - 2D case

```math
(\\textrm{u}_h, \\textrm{v}_h)_h \\vcentcolon = \\sum_{i=1}^{N_x} \\sum_{j=1}^{N_y} |\\square_{i,j}| \\textrm{u}_h(x_i,y_j) \\textrm{v}_h(x_i,y_j)
```

  - 3D case

```math
(\\textrm{u}_h, \\textrm{v}_h)_h \\vcentcolon = \\sum_{i=1}^{N_x} \\sum_{j=1}^{N_y}  \\sum_{l=1}^{N_z}  |\\square_{i,j,l}| \\textrm{u}_h(x_i,y_j) \\textrm{v}_h(x_i,y_j)
```

On a [`CompositeGridSpace`](@ref) it is the inner product of the product space, the sum of
the component-wise products:

```math
(\\textrm{u}_h, \\textrm{v}_h)_h = \\sum_{c=1}^{N_c} (\\textrm{u}_h^{(c)}, \\textrm{v}_h^{(c)})_h
```

which is the only meaning it can have, so there is nothing ambiguous about accepting one.
The two grid functions must have the same number of components.

`markers` restricts the sum to the union of the labelled regions' points (a masked sum of
the same cell measures, not a surface integral; see the note above `_combined_mask`).

# Examples

```jldoctest
using Bramble
Wₕ = gridspace(mesh(domain(interval(0.0, 1.0)), 101))
uₕ = Rₕ(Wₕ, x -> 1.0)
isapprox(innerₕ(uₕ, uₕ), 1.0; atol = 1.0e-12) && isapprox(normₕ(uₕ), 1.0; atol = 1.0e-12)

# output
true
```
"""
@inline function innerₕ(
        uₕ::VectorElement{<:ScalarGridSpace},
        vₕ::VectorElement{<:ScalarGridSpace};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {N}
    return inner₊(uₕ, vₕ, Val(()); markers = markers)
end

# Summed over the *leaves* (`components` flattens any nesting), unrolled via `map` over the
# two tuples exactly as the old `ntuple(…, Val(NC))` did over the space's own structural
# type parameter -- which is what a nested composite's u(i)/components disagree with (see
# CONTEXT.md, gpena/Bramble.jl#64), so this can no longer read the leaf count off a type
# parameter shared by both arguments; it has to check the actual leaf counts instead.
# `markers` threads through unchanged: each leaf is restricted to the same regions.
@inline function innerₕ(
        uₕ::VectorElement{<:CompositeGridSpace},
        vₕ::VectorElement{<:CompositeGridSpace};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {N}
    uc, vc = components(uₕ), components(vₕ)
    length(uc) == length(vc) || _throw_innerh_leaf_mismatch(length(uc), length(vc))
    return sum(map((u, v) -> innerₕ(u, v; markers = markers), uc, vc))
end

@noinline function _throw_innerh_leaf_mismatch(n::Int, m::Int)
    throw(
        DimensionMismatch(
        "innerₕ needs the same number of leaf components on both sides; got $n and $m"
    ),
    )
end

"""
    inner_Γ(uₕ::VectorElement, vₕ::VectorElement, labels::Symbol...) -> Real

Returns the ``(D-1)``-dimensional boundary integral of `uₕ * vₕ` over the grid faces `labels`
name,

```math
(\\textrm{u}_h, \\textrm{v}_h)_\\Gamma \\vcentcolon =
\\int_\\Gamma \\textrm{u}_h \\textrm{v}_h \\, ds
\\approx \\sum_{I \\in \\Gamma_h} \\omega(I) \\textrm{u}_h(I) \\textrm{v}_h(I)
```

with the lumped surface weight ``\\omega`` described in `mesh/queries.jl`: the sum, over the
normal directions of the surface pieces through a point, of the product of the *transverse*
half-spacings. In 1D a face is a point of measure 1, so `inner_Γ(uₕ, vₕ, :boundary)` is
`u(x₀)v(x₀) + u(x_N)v(x_N)`, with no spacing anywhere -- the correct pairing for a 1D Neumann
term, and not comparable dimensionally with the 2D and 3D cases.

**Not the same quantity as `innerₕ(uₕ, vₕ; markers = labels)`.** That is a masked sum of the
existing cell measures (a ``D``-dimensional quantity restricted to a set of points), and it
scales like `h`, vanishing under refinement: 0.125 on a 5×5 mesh of the unit square restricted
to `:bottom`, against 1.0 here on the same mesh and region, on every mesh.

`labels` name whole coordinate faces: `:boundary`, the canonical `:xmin`…`:zmax`, or their
viewpoint aliases. A user-defined marker covering part of a face, an interior interface or a
staircase is a genuinely more general surface -- the weight stops factorising where the
surface is cut at an interior transverse index, and needs the explicit one-sided face sum --
and is refused rather than silently given the factorised weight.

The symbolic twin, for use inside a form, is `inner_Γ(g, v; markers = …)`
(`ast/operators/inner.jl`). It takes its regions as a keyword, to sit beside `innerₕ`; this
one keeps the positional labels it has always had.

# Examples

```jldoctest
Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 5), (true, true))
Wₕ = gridspace(Ωₕ)
uₕ = Rₕ(Wₕ, x -> 1.0)

# the length of the bottom edge, on this mesh and on every refinement of it
inner_Γ(uₕ, uₕ, :bottom)

# output

1.0
```

See also: [`innerₕ`](@ref)
"""
function inner_Γ(
        uₕ::VectorElement{<:ScalarGridSpace{D}},
        vₕ::VectorElement{<:ScalarGridSpace{D}},
        labels::Symbol...
) where {D}
    Ωₕ = mesh(space(uₕ))
    mask = _face_mask(Val(D), labels)
    _no_faces(mask) && _throw_no_surface_labels()
    _check_surface_is_thin(Ωₕ, mask)
    policy = execution_policy(space(uₕ))
    return _surface_sum(policy, Ωₕ, mask, parent(uₕ), parent(vₕ))
end

@noinline function _throw_no_surface_labels()
    throw(
        ArgumentError(
        "inner_Γ needs at least one surface to integrate over; name `:boundary` or a " *
        "coordinate face such as `:xmin`.",
    ),
    )
end

# Walks the whole grid rather than the marked points alone: `ω` is zero off the surface, the
# grid walk is the one every other inner product here does, and the boundary is a vanishing
# fraction of the points, so the arithmetic saved by a marked-index walk is not worth a
# second traversal shape. The same weight is read per point by the symbolic path
# (`compute_weight(::InnerGamma, …)`, ast/operators/inner.jl), so the two layers compute the
# same number by construction.
@inline function _surface_sum(Ωₕ, mask, u, v)
    acc = zero(eltype(u)) * zero(eltype(v)) * zero(eltype(Ωₕ))
    lin = LinearIndices(indices(Ωₕ))
    @inbounds for I in indices(Ωₕ)
        w = _surface_weight(Ωₕ, mask, I)
        iszero(w) && continue
        k = lin[I]
        acc += w * u[k] * v[k]
    end
    return acc
end

# Policy-dispatched entry point (gpena/Bramble.jl#311), the same shape as `_dot`/`_dot_masked`
# (src/utils/linear_algebra.jl): a `CpuPolicy` reaches the host loop above unchanged, whatever
# its concrete flavour, since `_surface_sum` above does no threading of its own; a `GpuPolicy`
# reaches the device method below instead of scalar-indexing `u`/`v` and `half_spacings`.
@inline _surface_sum(policy::ExecutionPolicy, Ωₕ, mask, u, v) = _surface_sum(
    locality(policy), policy, Ωₕ, mask, u, v)

@inline _surface_sum(::HostLocality, ::CpuSerial, Ωₕ, mask, u, v) = _surface_sum(Ωₕ, mask, u, v)
@inline _surface_sum(::HostLocality, ::CpuThreaded, Ωₕ, mask, u, v) = _surface_sum(Ωₕ, mask, u, v)
@inline _surface_sum(::HostLocality, ::CpuPolyester, Ωₕ, mask, u, v) = _surface_sum(Ωₕ, mask, u, v)

@noinline _surface_sum(loc::Locality, policy, Ωₕ, mask, u, v) = _throw_locality_mismatch(loc, policy)

"""
    _surface_sum(::DeviceLocality, ::GpuPolicy, Ωₕ, mask, u, v) -> Real

Device counterpart of [`_surface_sum`](@ref) above: a canonical face (or a union of them) is
a set of at most `2D` axis-aligned `(D-1)`-dimensional slices of the grid, so the CPU walk
over every grid point -- reading `half_spacings` and `u`/`v` one scalar at a time -- becomes,
per active face, one broadcasted reduction over that slice.

`u` and `v` are reshaped to the grid's own `dims` (a view, not a copy). For each axis `d` and
side (`min`/`max`) the mask marks, `selectdim` takes the boundary slice directly -- no mask
array is built or walked here, unlike [`_dot_masked`](@ref)'s device path, since a coordinate
face is already a contiguous slice rather than an arbitrary marked set. The transverse surface
measure for that slice is built once from [`host_half_spacings`](@ref) (never read on the
device), then copied onto `u`'s own device in one transfer via
[`_transverse_weight_device`](@ref); the reduction itself is `sum(su .* sv .* w)`, no `@kernel`.

Faces are summed independently rather than through one shared mask, exactly as the host
`_surface_weight_dir` recursion does: a corner point on two active faces receives both
faces' contributions, which is the intended, non-double-counting definition documented above
`_face_mask`.
"""
@noinline function _surface_sum(
        ::DeviceLocality, ::GpuPolicy, Ωₕ::AbstractMeshType{D}, mask,
        u::AbstractVector, v::AbstractVector
) where {D}
    dims = npoints(Ωₕ, Tuple)
    ur = reshape(u, dims)
    vr = reshape(v, dims)
    return _surface_sum_dirs(Ωₕ, mask, ur, vr, u, dims, Val(D), Val(D))
end

@inline _surface_sum_dirs(Ωₕ, mask, ur, vr, u, dims, ::Val{0}, ::Val{D}) where {D} = zero(eltype(ur)) *
                                                                                     zero(eltype(vr)) * zero(eltype(Ωₕ))

# One direction per rung, recursing on `Val(d)` for the same reason `_surface_weight_dir`
# does: a closure over a runtime `d` would box it (gpena/Bramble.jl#146), and while this is
# called `2D` times rather than once per grid point, there is no reason to give up the idiom
# the host recursion already uses right next to it.
@inline function _surface_sum_dirs(Ωₕ, mask, ur, vr, u, dims, ::Val{d}, ::Val{D}) where {d, D}
    s = zero(eltype(ur)) * zero(eltype(vr)) * zero(eltype(Ωₕ))
    mask[d][1] && (s += _boundary_face_sum(Ωₕ, ur, vr, u, Val(d), 1, Val(D)))
    mask[d][2] && (s += _boundary_face_sum(Ωₕ, ur, vr, u, Val(d), dims[d], Val(D)))
    return s + _surface_sum_dirs(Ωₕ, mask, ur, vr, u, dims, Val(d - 1), Val(D))
end

# `su`/`sv` and `w` share the same `(D-1)`-dimensional shape (the grid's `dims` with axis `d`
# dropped), so the masked sum from `_surface_weight_dir` becomes one broadcasted reduction
# over the slice instead of a per-point lookup.
@inline function _boundary_face_sum(Ωₕ, ur, vr, u, ::Val{d}, idx::Int, ::Val{D}) where {d, D}
    su = selectdim(ur, d, idx)
    sv = selectdim(vr, d, idx)
    w = _transverse_weight_device(Ωₕ, u, Val(d), Val(D))
    return sum(su .* sv .* w)
end

# 1D has no transverse axis at all -- a face is a point of measure 1 (see the note above
# `_face_of_symbol`) -- so this returns the same scalar `_transverse_measure` does, with no
# device array built or transferred.
@inline _transverse_weight_device(Ωₕ::AbstractMeshType{1}, u, ::Val{d}, ::Val{1}) where {d} = one(eltype(Ωₕ))

"""
    _transverse_weight_device(Ωₕ::AbstractMeshType{D}, u, ::Val{d}, ::Val{D}) -> AbstractArray

The `(D-1)`-dimensional transverse surface measure for the face normal to axis `d`, as a
device array matching `u`'s own backend.

Built the way `_separable_weights_full` builds the full weight tensor -- one
reshape per transverse axis so broadcasting multiplies them out to the full transverse shape
-- except every factor here comes from [`host_half_spacings`](@ref) rather than a
`SeparableWeights` factor that is already device-resident, since this is a fresh geometric
quantity with nothing cached to read. The host tensor is built once, from vectors of length
`O(n^{1/D})`, then copied onto `u`'s device in a single transfer -- never read back scalar by
scalar the way `half_spacing` would on the host path.
"""
@inline function _transverse_weight_device(
        Ωₕ::AbstractMeshType{D}, u, ::Val{d}, ::Val{D}
) where {d, D}
    factors = ntuple(Val(D - 1)) do k
        e = k < d ? k : k + 1
        hs = _apply_hs_logic.(host_half_spacings(Ωₕ(e)))
        reshape(hs, ntuple(j -> j == k ? length(hs) : 1, D - 1))
    end
    hostw = reduce((a, b) -> a .* b, factors)
    devw = similar(u, eltype(hostw), length(hostw))
    copyto!(devw, vec(hostw))
    return reshape(devw, size(hostw))
end

"""
    normₕ(uₕ::VectorElement) -> Real

Returns the discrete ``L^2`` norm of the grid function `uₕ`, defined as

```math
\\Vert \\textrm{u}_h \\Vert_h \\vcentcolon = \\sqrt{(\\textrm{u}_h, \\textrm{u}_h)_h}
```

On a [`CompositeGridSpace`](@ref) it is the norm of the product space, which follows from
the inner product there: the square root of the sum of the components' squared norms.
"""
@inline normₕ(uₕ::VectorElement{<:ScalarGridSpace}) = sqrt(innerₕ(uₕ, uₕ))
@inline normₕ(uₕ::VectorElement{<:CompositeGridSpace}) = sqrt(innerₕ(uₕ, uₕ))

"""
    norminf(uₕ::VectorElement) -> Real
    norminf(uₕ::NTuple{D, VectorElement}) -> Real

Returns the discrete maximum norm of the grid function `uₕ`, defined as

```math
\\Vert \\textrm{u}_h \\Vert_{h,\\infty} \\vcentcolon = \\max_{I} \\vert \\textrm{u}_h(I) \\vert
```

Unlike the ``L^2``-type norms, this one carries no quadrature weight, so the same expression
serves a [`ScalarGridSpace`](@ref) and a [`CompositeGridSpace`](@ref): the maximum over a
composite's degrees of freedom is the maximum over its components. On an `NTuple` of grid
functions -- what the vectorial aliases such as [`∇ₕ`](@ref) return -- it is the maximum over
every entry of every component.

The element type comes from the data, not from the space, so a `ForwardDiff.Dual`-valued grid
function returns a `Dual`.

See also: [`normₕ`](@ref), [`norm₁ₕ`](@ref)
"""
@inline function norminf(uₕ::VectorElement)
    data = parent(uₕ)
    return mapreduce(abs, max, data; init = abs(zero(eltype(data))))
end

@inline norminf(uₕ::NTuple{<:Any, VectorElement}) = maximum(norminf, uₕ)

"""
    norm(uₕ::VectorElement, kind::AbstractString) -> Real

Returns the discrete norm of `uₕ` that `kind` names: `"h"` is [`normₕ`](@ref), `"1h"` is
[`norm₁ₕ`](@ref) and `"∞"` is [`norminf`](@ref). Any other `kind` throws an `ArgumentError`.

The one-argument `norm(uₕ)` is unchanged: a [`VectorElement`](@ref) is an `AbstractVector`,
so it is still `LinearAlgebra`'s Euclidean norm of the values, which carries no quadrature
weight.
"""
function norm(uₕ::VectorElement, kind::AbstractString)
    kind == "h" && return normₕ(uₕ)
    kind == "1h" && return norm₁ₕ(uₕ)
    kind == "∞" && return norminf(uₕ)
    throw(ArgumentError("the norm must be \"h\", \"1h\" or \"∞\", got \"$kind\""))
end

################################################################################
#                 Discrete Modified L² Inner Product and Norm                  #
################################################################################

# Weight-side specializations of `_dot`/`_dot_masked` (src/utils/linear_algebra.jl) for a
# lazy `SeparableWeights` (src/space/scalar_gridspace.jl): `weights(Wₕ, Val(S))` returns one
# of these for every staggered set `S` that is neither `()` nor a singleton
# (gpena/Bramble.jl#115, #234).
#
# The unmasked `_dot` walks the grid one axis-1 line at a time. The outer loop runs over
# `CartesianIndices(w.dims[2:D])`; for each line it forms the scalar
# `c = ∏_{d ≥ 2} factors[d][I_d]` once, and the inner `@inbounds @simd` loop walks the
# line's contiguous `dims[1]` entries, accumulating `u[k] * v[k] * factors[1][i₁]` with the
# same `muladd` shape as the dense `_dot`. The line sum is then folded in as
# `s = muladd(c, line_sum, s)`. For D = 1 there is one line and `c = one(T)`. Nothing is
# allocated and the weight tensor is never formed. Measured on non-uniform 1000² and 100³
# grids (minimum of 15 runs): innerₕ and inner₊ₓ each take 0.74-0.78x the time of a dense
# `_dot` over the collected weights, which has a third full-length vector to read.
#
# The masked methods only visit marked indices, so they convert each linear index to a
# `CartesianIndex` and read `w` through its `CartesianIndex` `getindex`, which multiplies
# the per-axis factors directly instead of dividing by each axis length in turn.
#
# No separate `CpuPolyester` override is needed here: `inner₊(uₕ, vₕ, Val(S))` calls the
# policy-dispatched `_dot`/`_dot_masked(policy, u, v, w[, mask])` (S7.1,
# `src/utils/linear_algebra.jl`), whose `CpuSerial` method falls through to the plain
# three/four-argument methods below -- where ordinary dispatch on the weight argument's
# runtime type reaches this specialization -- and whose `CpuThreaded` method likewise falls
# through to `_threaded_dot`/`_threaded_dot_masked`, reaching the `CpuThreaded`
# specializations of those two names declared further below (gpena/Bramble.jl#301 S2.2)
# rather than the dense `_threaded_dot`/`_threaded_dot_masked` in `src/utils/linear_algebra.jl`.
# Its `CpuPolyester` method calls `_batch_dot`/`_batch_dot_masked` directly, before the
# weight's type is ever consulted, so a `CpuPolyester` policy reaches S7.1's Polyester hook
# (or its "not loaded" error) regardless of whether the weight is dense or a
# `SeparableWeights`, never this loop.
@inline function _dot(
        u::AbstractVector, w::SeparableWeights{D, <:Any, VT}, v::AbstractVector
) where {D, VT}
    # The factors are indexed directly below, so repeat `getindex`'s device guard (#310).
    locality(VT) isa DeviceLocality && _throw_device_scalar_weights()
    n = length(w)
    (length(u) == n == length(v)) || _throw_dot_dim_error(length(u), n, length(v))
    T = promote_type(eltype(u), eltype(w), eltype(v))
    s = zero(T)
    f₁ = first(w.factors)
    n₁ = first(w.dims)
    offset = 0
    @inbounds for J in CartesianIndices(Base.tail(w.dims))
        c = one(T)
        for d in 2:D
            c *= T(w.factors[d][J[d - 1]])
        end
        line_sum = zero(T)
        @simd for i₁ in 1:n₁
            k = offset + i₁
            line_sum = muladd(T(u[k]) * T(v[k]), T(f₁[i₁]), line_sum)
        end
        s = muladd(c, line_sum, s)
        offset += n₁
    end
    return s
end

@inline function _dot_masked(
        u::AbstractVector, w::SeparableWeights{D}, v::AbstractVector, mask::BitVector
) where {D}
    n = length(w)
    (length(u) == n == length(v) == length(mask)) ||
        _throw_dot_dim_error(length(u), n, length(v), length(mask))
    T = promote_type(eltype(u), eltype(w), eltype(v))
    s = zero(T)
    cart = CartesianIndices(w.dims)
    @inbounds for i in MarkedIndices(mask)
        s = muladd(T(u[i]) * T(v[i]), T(w[cart[i]]), s)
    end
    return s
end

@inline function _dot_masked(
        u::AbstractVector, w::SeparableWeights{D}, v::AbstractVector, mask::MarkedIndicesUnion
) where {D}
    n = length(w)
    (length(u) == n == length(v) == mask.len) ||
        _throw_dot_dim_error(length(u), n, length(v), mask.len)
    T = promote_type(eltype(u), eltype(w), eltype(v))
    s = zero(T)
    cart = CartesianIndices(w.dims)
    @inbounds for i in mask
        s = muladd(T(u[i]) * T(v[i]), T(w[cart[i]]), s)
    end
    return s
end

# `CpuThreaded` specializations of the three serial methods above (gpena/Bramble.jl#301 S2.2,
# #288). `_dot`/`_dot_masked(policy, u, w, v[, mask])` (S7.1, `src/utils/linear_algebra.jl`)
# dispatch `CpuThreaded` to `_threaded_dot`/`_threaded_dot_masked`, and ordinary dispatch on
# the weight argument's runtime type reaches these rather than the dense
# `_threaded_dot(u::AbstractVector, v::AbstractVector, w::AbstractVector)` in
# `src/utils/linear_algebra.jl` -- exactly as `CpuSerial` reaches the serial specializations
# above instead of the dense kernel there. Without these, a `SeparableWeights` under
# `CpuThreaded` fell through to that dense method, which reads `w` through its linear `Int`
# `getindex` -- an `O(D)` `divrem` per point (gpena/Bramble.jl#288) -- once per grid point
# rather than once per line. `CpuPolyester` is unaffected: it reaches
# `_batch_dot`/`_batch_dot_masked` directly, before the weight's type is ever consulted (see
# the comment above the serial methods).
#
# The dense method splits `CartesianIndices(Base.tail(w.dims))` along its last axis into
# `Threads.nthreads()` static bands via `_last_axis_chunks` (`src/utils/linear_algebra.jl`,
# already used by `_threaded_axis_for!` to split the same way), each band's task
# walking its lines with the exact serial line-sum structure -- the trailing-factor product
# `c` and the `@inbounds @simd` line over axis 1 -- and landing one partial sum in a fixed
# `Threads.nthreads()`-length buffer, summed serially at the end (the same shape
# `_threaded_dot` in `src/utils/linear_algebra.jl` uses). A line's offset into `u`/`v` is
# `(LinearIndices(tail_dims)[J] - 1) * n₁`, computed directly from `J` rather than
# accumulated across iterations as the serial loop does, since bands are visited out of
# order. `D == 1` has no tail axis to split -- `Base.tail((n₁,))` is `()` -- so that case
# instead splits the single line itself (`1:n₁`) into bands via `_band_range`, with the
# trailing product `c` fixed at `one(T)`.
#
# The masked methods split the mask's 64-bit words into bands exactly as
# `_threaded_dot_masked` (`src/utils/linear_algebra.jl`) does, and read `w` through a
# `CartesianIndex` (`w[cart[i]]`), never the linear `Int` `getindex` this specialization
# exists to avoid.
@noinline function _threaded_dot(
        u::AbstractVector, w::SeparableWeights{D, <:Any, VT}, v::AbstractVector
) where {D, VT}
    # The factors are indexed directly below, so repeat `getindex`'s device guard (#310).
    locality(VT) isa DeviceLocality && _throw_device_scalar_weights()
    n = length(w)
    (length(u) == n == length(v)) || _throw_dot_dim_error(length(u), n, length(v))
    T = promote_type(eltype(u), eltype(w), eltype(v))
    f₁ = first(w.factors)
    n₁ = first(w.dims)
    nchunks = Threads.nthreads()
    partials = zeros(T, nchunks)

    if D == 1
        ax = 1:n₁
        Threads.@threads :static for b in 1:nchunks
            rng = _band_range(ax, nchunks, b)
            s = zero(T)
            @inbounds @simd for i₁ in rng
                s = muladd(T(u[i₁]) * T(v[i₁]), T(f₁[i₁]), s)
            end
            @inbounds partials[b] = s
        end
    else
        tail_dims = Base.tail(w.dims)
        lin = LinearIndices(tail_dims)
        blocks = _last_axis_chunks(CartesianIndices(tail_dims), nchunks)
        nblocks = length(blocks)
        Threads.@threads :static for b in 1:nblocks
            block = blocks[b]
            s = zero(T)
            @inbounds for J in block
                c = one(T)
                for d in 2:D
                    c *= T(w.factors[d][J[d - 1]])
                end
                offset = (lin[J] - 1) * n₁
                line_sum = zero(T)
                @simd for i₁ in 1:n₁
                    k = offset + i₁
                    line_sum = muladd(T(u[k]) * T(v[k]), T(f₁[i₁]), line_sum)
                end
                s = muladd(c, line_sum, s)
            end
            @inbounds partials[b] = s
        end
    end
    return sum(partials)
end

@noinline function _threaded_dot_masked(
        u::AbstractVector, w::SeparableWeights{D}, v::AbstractVector, mask::BitVector
) where {D}
    n = length(w)
    (length(u) == n == length(v) == length(mask)) ||
        _throw_dot_dim_error(length(u), n, length(v), length(mask))
    T = promote_type(eltype(u), eltype(w), eltype(v))
    cart = CartesianIndices(w.dims)
    chunks = mask.chunks
    nwords = length(chunks)
    nchunks = Threads.nthreads()
    partials = zeros(T, nchunks)
    ax = 1:nwords
    Threads.@threads :static for b in 1:nchunks
        rng = _band_range(ax, nchunks, b)
        s = zero(T)
        @inbounds for widx in rng
            word = chunks[widx]
            base = (widx - 1) * 64
            while word != zero(UInt64)
                i = base + trailing_zeros(word) + 1
                s = muladd(T(u[i]) * T(v[i]), T(w[cart[i]]), s)
                word &= word - 1
            end
        end
        @inbounds partials[b] = s
    end
    return sum(partials)
end

@noinline function _threaded_dot_masked(
        u::AbstractVector, w::SeparableWeights{D}, v::AbstractVector, mask::MarkedIndicesUnion
) where {D}
    n = length(w)
    (length(u) == n == length(v) == mask.len) ||
        _throw_dot_dim_error(length(u), n, length(v), mask.len)
    T = promote_type(eltype(u), eltype(w), eltype(v))
    cart = CartesianIndices(w.dims)
    nwords = length(mask.chunks[1])
    nchunks = Threads.nthreads()
    partials = zeros(T, nchunks)
    ax = 1:nwords
    Threads.@threads :static for b in 1:nchunks
        rng = _band_range(ax, nchunks, b)
        s = zero(T)
        @inbounds for widx in rng
            word = _reduce_or_chunk(mask.chunks, widx)
            base = (widx - 1) * 64
            while word != zero(UInt64)
                i = base + trailing_zeros(word) + 1
                s = muladd(T(u[i]) * T(v[i]), T(w[cart[i]]), s)
                word &= word - 1
            end
        end
        @inbounds partials[b] = s
    end
    return sum(partials)
end

# Device counterparts of the three specializations above (gpena/Bramble.jl#94, #174, S2.5 of
# .agents/plans/metal-and-apple-silicon-acceleration.md). A device-backed space's
# `SeparableWeights.factors` are themselves device arrays (S2.2), so the
# `CartesianIndices`/scalar-`getindex` walk above would scalar-index the device once per
# grid point -- not merely slow, an error under `GPUArrays`' default scalar-indexing guard.
#
# `_separable_weights_full` materializes the D-dimensional weight tensor instead, one
# reshape-and-broadcast per axis rather than a `@kernel`: each `factors[d]` is reshaped to
# broadcast only along its own axis (singleton elsewhere), and multiplying the D reshaped
# factors together broadcasts out to the full `dims` shape, exactly `SeparableWeights`'s own
# `__prod`, computed once for the whole grid instead of once per point. `reshape(u, dims)`
# is a view, not a copy, so the reduction below is one `GPUArrays` `sum` over a broadcasted
# expression (S2.5 notes: reductions need no kernel).
@inline function _separable_weights_full(w::SeparableWeights{D}) where {D}
    dims = w.dims
    shaped = ntuple(D) do d
        reshape(w.factors[d], ntuple(k -> k == d ? dims[d] : 1, D))
    end
    return reduce((a, b) -> a .* b, shaped)
end

"""
    _dot(::DeviceLocality, ::GpuPolicy, u, w::SeparableWeights, v) -> Real

The device counterpart of [`_dot`](@ref)`(u, w::SeparableWeights, v)` above: the weight
tensor is materialized once via `_separable_weights_full`, then the product with
`u` and `v` (reshaped to the grid's own `dims`, not copied) is one broadcasted reduction.
"""
@noinline function _dot(
        ::DeviceLocality, ::GpuPolicy, u::AbstractVector, w::SeparableWeights{D}, v::AbstractVector
) where {D}
    n = length(w)
    (length(u) == n == length(v)) || _throw_dot_dim_error(length(u), n, length(v))
    dims = w.dims
    wfull = _separable_weights_full(w)
    return sum(reshape(u, dims) .* reshape(v, dims) .* wfull)
end

"""
    _dot_masked(::DeviceLocality, ::GpuPolicy, u, w::SeparableWeights, v, mask::BitVector) -> Real

The device counterpart of the `BitVector`-masked specialization above: `mask` (always host
memory, gpena/Bramble.jl#298) is copied onto `u`'s own device once, reshaped to `dims`
alongside `u` and `v`, and the masked sum is one broadcasted reduction.
"""
@noinline function _dot_masked(
        ::DeviceLocality, ::GpuPolicy, u::AbstractVector, w::SeparableWeights{D},
        v::AbstractVector, mask::BitVector
) where {D}
    n = length(w)
    (length(u) == n == length(v) == length(mask)) ||
        _throw_dot_dim_error(length(u), n, length(v), length(mask))
    dims = w.dims
    wfull = _separable_weights_full(w)
    md = similar(u, Bool)
    copyto!(md, Vector{Bool}(mask))
    return sum(reshape(u, dims) .* reshape(v, dims) .* wfull .* reshape(md, dims))
end

"""
    _dot_masked(::DeviceLocality, ::GpuPolicy, u, w::SeparableWeights, v, mask::MarkedIndicesUnion) -> Real

The multi-marker counterpart: `mask` is walked once, on the host, into a plain `BitVector`,
then handled exactly as the `BitVector` method above.
"""
@noinline function _dot_masked(
        ::DeviceLocality, ::GpuPolicy, u::AbstractVector, w::SeparableWeights{D},
        v::AbstractVector, mask::MarkedIndicesUnion
) where {D}
    n = length(w)
    (length(u) == n == length(v) == mask.len) ||
        _throw_dot_dim_error(length(u), n, length(v), mask.len)
    hostmask = falses(mask.len)
    @inbounds for i in mask
        hostmask[i] = true
    end
    dims = w.dims
    wfull = _separable_weights_full(w)
    md = similar(u, Bool)
    copyto!(md, Vector{Bool}(hostmask))
    return sum(reshape(u, dims) .* reshape(v, dims) .* wfull .* reshape(md, dims))
end

"""
    inner₊(uₕ::VectorElement, vₕ::VectorElement, ::Val{S}; markers = ()) -> Real

Returns the discrete inner product of the grid functions `uₕ` and `vₕ` weighted by the
staggered set `S ⊆ 1:D`, [`weights`](@ref)`(space(uₕ), Val(S))`: entry `I` weighs
``\\prod_{d \\in S} h_d(I_d) \\cdot \\prod_{d \\notin S} h_d(I_d + 1/2)`` (gpena/Bramble.jl#115,
#234).

`Val(())` is [`innerₕ`](@ref), and `Val((1,))`/`Val((2,))`/`Val((3,))` are
[`inner₊ₓ`](@ref)/[`inner₊ᵧ`](@ref)/[`inner₊₂`](@ref) respectively -- those four functions
are aliases of this one, sharing its implementation, and return the identical number to it
because [`weights`](@ref)`(Wₕ, Val(S))` returns the very same dense vector for those four
`S`, not a recomputed copy. Every other `S` -- a pair, or the full `1:D` combination --
reduces against a lazily-computed [`SeparableWeights`](@ref) instead, through the
`_dot`/`_dot_masked` specializations above, so no `O(n^D)` vector is ever materialised for
it.

`markers` restricts the sum to the union of the labelled regions' points, as it does for
[`innerₕ`](@ref) (a masked sum of the weight above, not a surface integral).

Defined for grid functions of a [`ScalarGridSpace`](@ref) only, the same restriction every
other `inner₊*` in this file has: a composite grid function has no single weighting of its
own; take a scalar component of it with [`components`](@ref) first, which is itself a
scalar grid function and is accepted.

# Examples

```jldoctest
using Bramble
Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (6, 5), (true, true))
Wₕ = gridspace(Ωₕ)
uₕ = Rₕ(Wₕ, x -> 1.0)
inner₊(uₕ, uₕ, Val(())) == innerₕ(uₕ, uₕ)

# output
true
```

See also: [`inner₊`](@ref), [`weights`](@ref), [`SeparableWeights`](@ref).
"""
@inline function inner₊(
        uₕ::VectorElement{<:ScalarGridSpace},
        vₕ::VectorElement{<:ScalarGridSpace},
        ::Val{S};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {S, N}
    w = weights(space(uₕ), Val(S))
    policy = execution_policy(space(uₕ))
    N == 0 && return _dot(policy, uₕ.data, w, vₕ.data)
    mask = _combined_marked_indices(mesh(space(uₕ)), markers)
    return _dot_masked(policy, uₕ.data, w, vₕ.data, mask)
end

@inline function _directional_inner_plus(
        uₕ::VectorElement{<:ScalarGridSpace},
        vₕ::VectorElement{<:ScalarGridSpace},
        ::Val{DIM};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {DIM, N}
    return inner₊(uₕ, vₕ, Val((DIM,)); markers = markers)
end

"""
    inner₊ₓ(uₕ::VectorElement, vₕ::VectorElement; markers = ()) -> Real

Returns the discrete modified ``L^2`` inner product of the grid functions `uₕ` and `vₕ` associated with the first variable.

For [`VectorElement`](@ref)s, it is defined as

  - 1D case

```math
(\\textrm{u}_h, \\textrm{v}_h)_+ \\vcentcolon = \\sum_{i=1}^{N_x} h_{i} \\textrm{u}_h(x_i) \\textrm{v}_h(x_i)
```

  - 2D case

```math
(\\textrm{u}_h, \\textrm{v}_h)_{+x} \\vcentcolon = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y}  h_{x,i} h_{y,j+1/2}  \\textrm{u}_h(x_i,y_j) \\textrm{v}_h(x_i,y_j)
```

  - 3D case

```math
(\\textrm{u}_h, \\textrm{v}_h)_{+x} \\vcentcolon = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y}\\sum_{l=1}^{N_z}   h_{x,i} h_{y,j+1/2} h_{z,l+1/2}  \\textrm{u}_h(x_i,y_j,z_l) \\textrm{v}_h(x_i,y_j,z_l).
```

Defined for grid functions of a [`ScalarGridSpace`](@ref) only. A grid function of a
composite grid space is rejected at dispatch; take a scalar component of it with
[`components`](@ref) first, which is itself a scalar grid function and is accepted.

`markers` restricts the sum as it does for [`innerₕ`](@ref) (a masked sum, not a surface
integral).
"""
@inline inner₊ₓ(
    uₕ::VectorElement{<:ScalarGridSpace},
    vₕ::VectorElement{<:ScalarGridSpace};
    markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {N} = _directional_inner_plus(uₕ, vₕ, Val(1); markers = markers)

"""
    inner₊ᵧ(uₕ::VectorElement, vₕ::VectorElement; markers = ()) -> Real

Returns the discrete modified ``L^2`` inner product of the grid functions `uₕ` and `vₕ`
associated with the second variable, the ``y`` direction.

For [`VectorElement`](@ref)s, it is defined as

  - 2D case

```math
(\\textrm{u}_h, \\textrm{v}_h)_{+y} \\vcentcolon = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y}   h_{x,i} h_{y,j+1/2}   \\textrm{u}_h(x_i,y_j) \\textrm{v}_h(x_i,y_j)
```

  - 3D case

```math
(\\textrm{u}_h, \\textrm{v}_h)_{+y} \\vcentcolon = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y}\\sum_{l=1}^{N_z}   h_{x,i+1/2} h_{y,j} h_{z,l+1/2} \\textrm{u}_h(x_i,y_j,z_l) \\textrm{v}_h(x_i,y_j,z_l).
```

Defined for grid functions of a [`ScalarGridSpace`](@ref) only. A grid function of a
composite grid space is rejected at dispatch; take a scalar component of it with
[`components`](@ref) first, which is itself a scalar grid function and is accepted.

`markers` restricts the sum as it does for [`innerₕ`](@ref) (a masked sum, not a surface
integral).
"""
@inline inner₊ᵧ(
    uₕ::VectorElement{<:ScalarGridSpace},
    vₕ::VectorElement{<:ScalarGridSpace};
    markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {N} = _directional_inner_plus(uₕ, vₕ, Val(2); markers = markers)

"""
    inner₊₂(uₕ::VectorElement, vₕ::VectorElement; markers = ()) -> Real

Returns the discrete modified ``L^2`` inner product of the grid functions `uₕ` and `vₕ` associated with the `z` variable

```math
(\\textrm{u}_h, \\textrm{v}_h)_{+z} \\vcentcolon = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y}\\sum_{l=1}^{N_z}  h_{x,i+1/2} h_{y,j+1/2} h_{z,l} \\textrm{u}_h(x_i,y_j,z_l) \\textrm{v}_h(x_i,y_j,z_l).
```

Defined for grid functions of a [`ScalarGridSpace`](@ref) only. A grid function of a
composite grid space is rejected at dispatch; take a scalar component of it with
[`components`](@ref) first, which is itself a scalar grid function and is accepted.

`markers` restricts the sum as it does for [`innerₕ`](@ref) (a masked sum, not a surface
integral).
"""
@inline inner₊₂(
    uₕ::VectorElement{<:ScalarGridSpace},
    vₕ::VectorElement{<:ScalarGridSpace};
    markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {N} = _directional_inner_plus(uₕ, vₕ, Val(3); markers = markers)

@noinline _throw_inner_plus_bounds(d) = throw(BoundsError(inner₊, d))

# Selects between literal `Val`s, one arm per direction -- the same shape as
# `_dispatch_dim` (stencil.jl): a runtime `Integer` never reaches `Val` directly, since
# `Val(d)` built from a non-constant `d` boxes it and nothing downstream can constant-fold
# (gpena/Bramble.jl#146). Always three arms regardless of the mesh a caller eventually
# applies the result to, exactly as `_vectorial_index_expr`'s `getindex` is -- an out-of-range
# direction is caught downstream, by the operator the index yields.
@inline function _inner_plus_val(d::Integer)
    d == 1 && return Val(1)
    d == 2 && return Val(2)
    d == 3 && return Val(3)
    _throw_inner_plus_bounds(d)
end

"""
    inner₊(uₕ::VectorElement, vₕ::VectorElement, d; markers = ()) -> Real

Returns [`inner₊ₓ`](@ref)/[`inner₊ᵧ`](@ref)/[`inner₊₂`](@ref)`(uₕ, vₕ; markers)`, selected by
`d`: an `Integer` (`1`, `2` or `3`) or a `Symbol` (`:x`, `:y` or `:z`) (gpena/Bramble.jl#341).

This is the preferred spelling: `inner₊(uₕ, vₕ, :x)` over `inner₊ₓ(uₕ, vₕ)`, which stays
reachable as a plain alias.

An `Integer` outside `1:3` throws a `BoundsError`; a `Symbol` that is not `:x`/`:y`/`:z`
throws an `ArgumentError`.

`inner₊` also destructures and indexes like the vectorial operator aliases
(gpena/Bramble.jl#340): `ix, iy, iz = inner₊` binds `inner₊ₓ`, `inner₊ᵧ`, `inner₊₂` (the
very same function objects, not copies), and `inner₊[1]`/`inner₊[:x]` (through
`length`/`firstindex`/`lastindex`/`getindex`/iteration) index into that same triple.

# Examples

```jldoctest
using Bramble
Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (6, 5), (true, true))
Wₕ = gridspace(Ωₕ)
uₕ = Rₕ(Wₕ, x -> 1.0)
inner₊(uₕ, uₕ, :x) == inner₊(uₕ, uₕ, 1)

# output
true
```

See also: [`inner₊ₓ`](@ref), [`inner₊ᵧ`](@ref), [`inner₊₂`](@ref).
"""
@inline function inner₊(
        uₕ::VectorElement{<:ScalarGridSpace},
        vₕ::VectorElement{<:ScalarGridSpace},
        d::Integer;
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {N}
    return _directional_inner_plus(uₕ, vₕ, _inner_plus_val(d); markers = markers)
end

@inline function inner₊(
        uₕ::VectorElement{<:ScalarGridSpace},
        vₕ::VectorElement{<:ScalarGridSpace},
        s::Symbol;
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {N}
    return inner₊(uₕ, vₕ, _dim_index(s); markers = markers)
end

# --- Iteration and indexing protocol, following `_vectorial_index_expr` -------------- #
# (stencil.jl, gpena/Bramble.jl#340): `inner₊ₓ`/`inner₊ᵧ`/`inner₊₂` are spliced in as
# literals, so `inner₊[2]` is a call to a `@inline` function of a literal `Int` against
# literal comparisons -- the compiler constant-folds it to the concrete function, same as
# any vectorial alias in stencil.jl. Written by hand rather than generated, since `inner₊`
# is not built by `@operator_family`.
@inline Base.iterate(::typeof(inner₊)) = (inner₊ₓ, 2)
@inline Base.iterate(::typeof(inner₊), state::Int) = state == 2 ? (inner₊ᵧ, 3) :
                                                     state == 3 ? (inner₊₂, 4) : nothing
@inline Base.length(::typeof(inner₊)) = 3
@inline Base.eltype(::Type{typeof(inner₊)}) = Function
@inline Base.firstindex(::typeof(inner₊)) = 1
@inline Base.lastindex(::typeof(inner₊)) = 3
@inline function Base.getindex(::typeof(inner₊), i::Integer)
    i == 1 && return inner₊ₓ
    i == 2 && return inner₊ᵧ
    i == 3 && return inner₊₂
    throw(BoundsError(inner₊, i))
end
@inline function Base.getindex(::typeof(inner₊), s::Symbol)
    s === :x && return inner₊ₓ
    s === :y && return inner₊ᵧ
    s === :z && return inner₊₂
    throw(ArgumentError("the coordinate direction must be :x, :y or :z, got :$s"))
end

get_dimension_from_type(::Type{<:NTuple{D, Any}}) where {D} = D
get_dimension_from_type(::Type{<:VectorElement{S}}) where {S} = dim(mesh_type(S))
get_dimension_from_type(::Type) = nothing

# The spatial dimension of a tuple's *elements*, which is not the tuple's arity: a mixed
# term such as (Dx*u, Mx*u) on a one-dimensional mesh is a 2-tuple whose elements are 1D.
#
# Read off the first element through `fieldtypes`, which is public and defined for every
# tuple type. Reading `.parameters[2]` instead threw a BoundsError on a 1-tuple, since
# `Tuple{VectorElement{…}}` has one parameter, and did so from inside a generated
# function, so `inner₊((uₕ,), (vₕ,))` failed at code generation rather than returning a
# number.
function _tuple_element_dim(::Type{T}) where {T <: Tuple}
    ft = fieldtypes(T)
    return isempty(ft) ? nothing : get_dimension_from_type(first(ft))
end

# Whether `@generated` is load-bearing here, or just this file's house style, was asked in
# gpena/Bramble.jl#61 and measured rather than asserted (bramble-verification §1): an
# ordinary function computing `D`/`mesh_dim` this same way and passing them on as
# `Val(D)`/`Val(mesh_dim)` inferred to `Any` and allocated (112–592 B across 1D/2D/3D,
# scalar and tuple inputs), where the `@generated` version below infers concretely and
# allocates 0. The difference is constant propagation, not type stability: `D`/`mesh_dim`
# come out of a several-branch `if`/`something` chain, and while that chain's return type is
# already concrete (`Tuple{Int,Int}`), Julia's inliner does not reliably fold it down to the
# *specific* compile-time value `Val(D)` needs from inside a caller, unlike a plain `map`
# or `sum` over an already concretely-sized `Tuple`, which Julia unrolls and specializes on
# its own (see `form/common.jl`'s stencil primitives, sibling functions in this same
# investigation that turned out not to need `@generated` at all). So: load-bearing here,
# incidental there.
function _generate_inner_plus_body(u_type, v_type, result_kind::Symbol)
    dim_u = get_dimension_from_type(u_type)
    dim_v = get_dimension_from_type(v_type)

    u_is_tuple = u_type <: NTuple
    v_is_tuple = v_type <: NTuple

    # Prefer tuple arity when tuples are provided (e.g., inner₊((a,b), (c,d)) even in 1D).
    D = if u_type <: NTuple
        dim_u
    elseif v_type <: NTuple
        dim_v
    elseif !isnothing(dim_u) && !isnothing(dim_v)
        if dim_u == dim_v
            dim_u
        else
            # The message is built here and spliced in as a String. Writing the
            # interpolation inside the quoted string would defer it to run time,
            # where dim_u and dim_v do not exist, and the call would raise
            # UndefVarError instead of the intended error.
            return :(throw(DimensionMismatch($("Dimensions $dim_u and $dim_v do not match"))))
        end
    elseif !isnothing(dim_u)
        dim_u
    elseif !isnothing(dim_v)
        dim_v
    else
        return :(throw(
            ArgumentError(
            $("Could not determine dimension from input types $u_type and $v_type")
        ),
        ))
    end

    # Direction count for the underlying space (fallback to 1 if unknown).
    # For tuple inputs we want the *spatial* dimension of the element type, not
    # the tuple arity (which can exceed the mesh dimension in mixed terms such as
    # `(Dx*u, Mx*u)` on 1D meshes).
    u_elem_dim = u_is_tuple ? _tuple_element_dim(u_type) : nothing
    v_elem_dim = v_is_tuple ? _tuple_element_dim(v_type) : nothing
    mesh_dim = something(
        u_elem_dim,
        v_elem_dim,
        (!u_is_tuple && !isnothing(dim_u)) ? dim_u : nothing,
        (!v_is_tuple && !isnothing(dim_v)) ? dim_v : nothing,
        1
    )

    terms = map(1:D) do i
        u_component = u_is_tuple ? :(uₕ[$i]) : :uₕ
        v_component = v_is_tuple ? :(vₕ[$i]) : :vₕ
        dir = min(i, mesh_dim) # avoid out-of-bounds when tuples are longer than spatial dim
        return :(_directional_inner_plus($u_component, $v_component, Val($dir)))
    end

    if result_kind === :sum
        return :(+($(terms...)))
    elseif result_kind === :tuple
        return :($(Expr(:tuple, terms...)))
    else
        return :(throw(ArgumentError("Invalid result kind for code generation.")))
    end
end

"""
    inner₊(uₕ::VectorElement, vₕ::VectorElement) -> Real
    inner₊(uₕ::VectorElement, vₕ::VectorElement, ::Type{Tuple}) -> Tuple
    inner₊(uₕ::NTuple{D, VectorElement}, vₕ::NTuple{D, VectorElement}) -> Real

Returns the discrete modified ``L^2`` inner product of the grid functions `uₕ` and `vₕ`.

If the `Tuple` argument is given, it returns a `D`-tuple of all ``\\textrm{inner}_{x_i,+}`` applied to its input arguments, where `D` is the topological dimension of the mesh associated with the elements.

If `NTuple`s of [`VectorElement`](@ref) are passed as input arguments, it returns the sum of all inner products ``(\\textrm{u}_h[i],\\textrm{v}_h[i])_{+x_i}``.

For [`VectorElement`](@ref)s, the definition is given by

  - 1D case

```math
(\\textrm{u}_h, \\textrm{v}_h)_+ \\vcentcolon = \\sum_{i=1}^{N_x} h_{i} \\textrm{u}_h(x_i) \\textrm{v}_h(x_i)
```

  - 2D case

```math
(\\textrm{u}_h, \\textrm{v}_h)_+ \\vcentcolon = (\\textrm{u}_h, \\textrm{v}_h)_{+x} + (\\textrm{u}_h, \\textrm{v}_h)_{+y}
```

  - 3D case

```math
(\\textrm{u}_h, \\textrm{v}_h)_+ \\vcentcolon = (\\textrm{u}_h, \\textrm{v}_h)_{+x} + (\\textrm{u}_h, \\textrm{v}_h)_{+y} + (\\textrm{u}_h, \\textrm{v}_h)_{+z}.
```

See the definitions of [`inner₊ₓ`](@ref), [`inner₊ᵧ`](@ref), and [`inner₊₂`](@ref) for more details.

Defined for grid functions of a [`ScalarGridSpace`](@ref) only, and for tuples whose
entries are such grid functions. A grid function of a composite grid space raises a
`MethodError`; take a scalar component of it with [`components`](@ref) first, which is
itself a scalar grid function and is accepted.
"""
@generated inner₊(uₕ, vₕ) = :($(_generate_inner_plus_body(uₕ, vₕ, :sum)))
@generated inner₊(uₕ, vₕ, ::Type{Tuple}) = :($(_generate_inner_plus_body(uₕ, vₕ, :tuple)))

"""
    norm₊(uₕ::VectorElement) -> Real
    norm₊(uₕ::NTuple{D, VectorElement}) -> Real

Returns the discrete modified ``L^2`` norm of the grid function `uₕ`. It also accepts an `NTuple` of [`VectorElement`](@ref)s.

For [`VectorElement`](@ref)s `uₕ`, it is defined as

```math
\\Vert \\textrm{u}_h \\Vert_+ = \\sqrt{(\\textrm{u}_h,\\textrm{u}_h)_+}.
```

and for `NTuple`s of [`VectorElement`](@ref)s it returns

```math
\\Vert \\textrm{u}_h \\Vert_+ \\vcentcolon = \\sqrt{ \\sum_{i=1}^D(\\textrm{u}_h[i],\\textrm{u}_h[i])_{+,x_i}}.
```

Defined for grid functions of a [`ScalarGridSpace`](@ref) only, and for tuples whose
entries are such grid functions. A grid function of a composite grid space raises a
`MethodError`; take a scalar component of it with [`components`](@ref) first, which is
itself a scalar grid function and is accepted.
"""
@inline norm₊(uₕ::Union{VectorElement, NTuple{<:Any, VectorElement}}) = sqrt(inner₊(uₕ, uₕ))

################################################################################
#                        Discrete H¹ Norm and Seminorm                         #
################################################################################
# The squared seminorm along one direction. `d` arrives as a `Val` so the backward-neighbour
# offset and the loop shape are fixed at compile time, and the spacing and weight are read
# once per direction rather than once per grid point.
#
# The loop mirrors `_dot`'s `SeparableWeights` specialization above: the outer loop runs
# over the axis-1 lines (`CartesianIndices` of `dims[2:D]`), and the inner `@inbounds @simd`
# loop walks each line's contiguous entries. The boundary slice contributes nothing (the
# backward difference is truncated to zero there, so its square is zero), so only the
# interior is walked:
#   - d = 1: the line runs over `i₁ ∈ 2:n₁`, reading the weight factor `factors[1][i₁]` and
#     dividing by the spacing `h[i₁]` per point; the backward neighbour is the previous entry.
#     The division is kept: on 2D 1000² this pass takes 1.21x a dense `_dot`, and 1.07x
#     with a multiplication in its place, so it does not limit the speed.
#   - d ≥ 2: lines with `I_d = 1` are skipped. Along a line the spacing `h[I_d]` and every
#     weight factor but the first are constant, so the line sum is `Σ factors[1][i₁] δ²`
#     over the undivided differences `δ = u[k] - u[k - stride_d]`, and it is scaled once by
#     `c · inv(h[I_d])²`, where `c` is the product of the other axes' factors and
#     `stride_d = n₁ ⋯ n_{d-1}`.
# The weight factors are read directly, so `getindex`'s device guard is repeated (#310).
# Measured on non-uniform 1000² and 100³ grids (minimum of 15 runs): snorm₁ₕ takes 2.56x
# (2D) and 3.82x (3D) the time of a dense `_dot` over the collected `innerₕ` weights, about
# 1.3x per direction; the previous point-wise walk over `CartesianIndices(interior)`, whose
# per-point division did not vectorise, took 12.7x in 2D.
@inline function _seminorm_sq_along(data, space, Ωₕ, li, ::Val{d}, ::Val{D}) where {d, D}
    h = backward_spacings_for_derivative(Ωₕ(d))
    w = weights(space, Innerplus(), d)
    locality(typeof(first(w.factors))) isa DeviceLocality && _throw_device_scalar_weights()
    dims = size(li)
    T = promote_type(eltype(data), eltype(w), eltype(h))
    f₁ = first(w.factors)
    n₁ = first(dims)
    stride = d == 1 ? 1 : prod(ntuple(k -> dims[k], Val(d - 1)))
    lines = CartesianIndices(ntuple(
        k -> k + 1 == d ? (2:dims[k + 1]) : (1:dims[k + 1]), Val(D - 1)))

    s = zero(T)
    @inbounds for J in lines
        c = one(T)
        for k in 2:D
            c *= T(w.factors[k][J[k - 1]])
        end
        offset = li[CartesianIndex(1, Tuple(J)...)] - 1
        line_sum = zero(T)
        if d == 1
            @simd for i₁ in 2:n₁
                k = offset + i₁
                δ = (T(data[k]) - T(data[k - 1])) / T(h[i₁])
                line_sum = muladd(T(f₁[i₁]), δ * δ, line_sum)
            end
        else
            @simd for i₁ in 1:n₁
                k = offset + i₁
                δ = T(data[k]) - T(data[k - stride])
                line_sum = muladd(T(f₁[i₁]), δ * δ, line_sum)
            end
            ih = inv(T(h[J[d - 1]]))
            c *= ih * ih
        end
        s = muladd(c, line_sum, s)
    end

    return s
end

# `D` is taken from the space's type parameter rather than from `dim(Ωₕ)`: building a
# `Val` out of a value returned at run time is a dynamic dispatch, which cost 224 bytes
# per call on a 2D grid.
# Summed by recursion on `Val(d)` rather than through `ntuple`: the closure `ntuple`
# needs captures four locals, and capturing them cost a small allocation per call.
@inline _sum_dirs(data, space, Ωₕ, li, ::Val{0}, ::Val{D}) where {D} = zero(eltype(data))
@inline _sum_dirs(data, space, Ωₕ, li, ::Val{d}, ::Val{D}) where {d, D} = _seminorm_sq_along(
    data, space, Ωₕ, li, Val(d), Val(D)) +
                                                                          _sum_dirs(
    data, space, Ωₕ, li, Val(d - 1), Val(D))

# Device counterpart of `_seminorm_sq_along`/`_sum_dirs` above (gpena/Bramble.jl#94, #174,
# S2.5): `data`, `h` and `w`'s factors are all device arrays for a device-backed space, so
# the scalar `@inbounds @simd` walk above would scalar-index the device once per grid
# point. `_seminorm_sq_along_device` reshapes `data` to the grid's own `dims` and takes the
# backward difference along axis `d` as one strided slice minus another -- exactly the
# `interior` the host version walks, since truncating the first index along `d` is the same
# set of points the host loop's `interior` range already restricts to -- divides by `h`
# reshaped to broadcast along that axis alone, and weights the result by the same
# `SeparableWeights` tensor `_dot`'s device path builds, restricted along axis `d` to match.
# One broadcasted `sum`, no `@kernel`.
@inline function _seminorm_sq_along_device(
        data, w::SeparableWeights{D}, h::AbstractVector, dims::NTuple{D, Int}, ::Val{d}
) where {D, d}
    datar = reshape(data, dims)
    lo = selectdim(datar, d, 1:(dims[d] - 1))
    hi = selectdim(datar, d, 2:dims[d])
    hinterior = @view h[2:dims[d]]
    hr = reshape(hinterior, ntuple(k -> k == d ? length(hinterior) : 1, D))
    δ = (hi .- lo) ./ hr
    factors = ntuple(D) do k
        f = k == d ? (@view w.factors[k][2:dims[k]]) : w.factors[k]
        reshape(f, ntuple(j -> j == k ? length(f) : 1, D))
    end
    wfull = reduce((a, b) -> a .* b, factors)
    return sum(wfull .* δ .* δ)
end

@inline _sum_dirs_device(data, space, Ωₕ, dims, ::Val{0}, ::Val{D}) where {D} = zero(eltype(data))
@inline function _sum_dirs_device(data, space, Ωₕ, dims, ::Val{d}, ::Val{D}) where {d, D}
    h = backward_spacings_for_derivative(Ωₕ(d))
    w = weights(space, Innerplus(), d)
    return _seminorm_sq_along_device(data, w, h, dims, Val(d)) +
           _sum_dirs_device(data, space, Ωₕ, dims, Val(d - 1), Val(D))
end

@inline function _snorm₁ₕ_sq(uₕ::VectorElement{<:ScalarGridSpace{D}}) where {D}
    (; data, space) = uₕ
    Ωₕ = mesh(space)
    dims = npoints(Ωₕ, Tuple)
    locality(typeof(data)) isa DeviceLocality &&
        return _sum_dirs_device(data, space, Ωₕ, dims, Val(D), Val(D))
    li = LinearIndices(dims)
    return _sum_dirs(data, space, Ωₕ, li, Val(D), Val(D))
end

"""
    snorm₁ₕ(uₕ::VectorElement) -> Real

Returns the discrete ``H^1`` seminorm of the grid function `uₕ`,

```math
|\\textrm{u}_h|_{1h} \\vcentcolon = \\Vert \\nabla_h \\textrm{u}_h \\Vert_+
```

so that `snorm₁ₕ(uₕ) == norm₊(∇ₕ(uₕ))` in one, two and three dimensions. The argument is
the grid function itself; the backward gradient is taken internally, and without
materialising it, so this allocates nothing.

See also: [`norm₁ₕ`](@ref), [`norm₊`](@ref), [`∇ₕ`](@ref).

Defined for grid functions of a [`ScalarGridSpace`](@ref) only. A grid function of a
composite grid space is rejected at dispatch; take a scalar component of it with
[`components`](@ref) first, which is itself a scalar grid function and is accepted.
"""
@inline snorm₁ₕ(uₕ::VectorElement{<:ScalarGridSpace}) = sqrt(_snorm₁ₕ_sq(uₕ))

"""
    norm₁ₕ(uₕ::VectorElement) -> Real

Returns the discrete version of the standard ``H^1`` norm of [`VectorElement`](@ref) `uₕ`.

```math
\\Vert \\textrm{u}_h \\Vert_{1h} \\vcentcolon = \\sqrt{ \\Vert \\textrm{u}_h \\Vert_h^2 +  \\Vert \\nabla_h \\textrm{u}_h \\Vert_h^2   }
```

Built from the squared quantities directly: taking `normₕ` and `snorm₁ₕ` and squaring
them back up would compute two square roots only to undo them.

Defined for grid functions of a [`ScalarGridSpace`](@ref) only. A grid function of a
composite grid space is rejected at dispatch; take a scalar component of it with
[`components`](@ref) first, which is itself a scalar grid function and is accepted.
"""
@inline norm₁ₕ(uₕ::VectorElement{<:ScalarGridSpace}) = sqrt(innerₕ(uₕ, uₕ) + _snorm₁ₕ_sq(uₕ))
