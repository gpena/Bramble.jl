#===========================================================================#
# The cell average operator avgₕ, and its quadrature.
#
# Averages a function over the cell around each grid point with a tensor-product
# Gauss-Legendre rule.
#
# Distinct from operators/average.jl, which is the two-point average of a grid point with
# its neighbour. That one is a stencil; this one integrates over a cell.
#===========================================================================#

######################
#                    #
# Averaging operator #
#                    #
######################

# The coefficient type of the result comes from `f`, as it does in `Rₕ` and for the same
# reason: the average of a Dual-valued function is Dual-valued, over a mesh that stays
# Float64. The quadrature weights are the mesh's type and promote against it.
#
# Placed above the docstring, not between it and the definition. A comment there detaches
# the docstring from `avgₕ` and binds it to nothing, which the exported-names-are-
# documented check in test/quality catches.
"""
    avgₕ(Wₕ::AbstractSpaceType, f; quad_points = AVG_QUAD_POINTS, markers = ()) -> VectorElement

Returns a [`VectorElement`](@ref) with the average of function `f` with respect to the [`cell_measure`](@ref) of `mesh(Wₕ)` around each grid point.

Each cell average is a tensor-product Gauss-Legendre rule with `quad_points`
points per direction, exact for polynomials of degree `2 * quad_points - 1`.

# Arguments

  - `Wₕ::AbstractSpaceType`: grid space on which to average `f`.
  - `f`: function of one grid point. Receives coordinates as a scalar on 1D
    meshes or an `NTuple{D}` on `D`-dimensional meshes, never an `SVector`.

# Keywords

  - `quad_points::Union{Integer, Val}`: points per direction, per cell.
    Defaults to `Val(AVG_QUAD_POINTS)`.
  - `markers::NTuple{N, Symbol}`: restrict evaluation to the named marked
    regions, leaving every other entry zero.

# Examples

```julia
avgₕ(Wₕ, x -> sin(x))
avgₕ(Wₕ, x -> sin(x[1]) * x[2]; quad_points = Val(4))
```

On a Metal (or other device) backend, `f` runs *on the device* inside the quadrature
kernel, evaluated `quad_points^D` times per cell (gpena/Bramble.jl#94, #174), so it must be
GPU-compilable in the same sense [`Rₕ!`](@ref)'s docstring describes. A masked call and a
mesh of more than one dimension currently fall back to the CPU-only per-index sweep.

See also: [`avgₕ!`](@ref), [`Rₕ`](@ref).
"""
Base.@constprop :aggressive function avgₕ(
        Wₕ::AbstractSpaceType,
        f;
        quad_points::Union{Integer, Val} = Val(AVG_QUAD_POINTS),
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {N}
    uₕ = element(Wₕ, _restriction_eltype(Wₕ, f, markers))
    return avgₕ!(uₕ, f; quad_points = quad_points, markers = markers)
end

"""
    avgₕ!(uₕ::VectorElement, f; quad_points = AVG_QUAD_POINTS, markers = ()) -> VectorElement

In-place version of the averaging operator [`avgₕ`](@ref). Returns `uₕ`.

Evaluates the tensor-product Gauss-Legendre cell average of `f` and writes
the result into `uₕ`.

# Arguments

  - `uₕ::VectorElement`: pre-allocated element to write into.
  - `f`: function of one grid point. Receives coordinates as a scalar on 1D
    meshes or an `NTuple{D}` on `D`-dimensional meshes.

# Keywords

  - `quad_points::Union{Integer, Val}`: points per direction, per cell.
    Defaults to `Val(AVG_QUAD_POINTS)`. Using a `Val` allows compile-time
    specialization of the quadrature nodes and weights without boxing.
  - `markers::NTuple{N, Symbol}`: restrict evaluation to the named marked
    regions, leaving every other entry zero.

See also: [`avgₕ`](@ref), [`Rₕ!`](@ref).
"""
@inline avgₕ!(uₕ::VectorElement{<:ScalarGridSpace{D}}, f::Tuple{Any}) where {D} = avgₕ!(uₕ, f[1])

@inline avgₕ!(uₕ::VectorElement{<:ScalarGridSpace}, f::F) where {F} = project!(uₕ, _average_rule(f, Val(AVG_QUAD_POINTS)))

@inline avgₕ!(uₕ::VectorElement{<:CompositeGridSpace}, f::Tuple) = project!(uₕ, _average_rule(f, Val(AVG_QUAD_POINTS)))

@inline avgₕ!(uₕ::VectorElement{<:CompositeGridSpace}, f::F) where {F} = project!(uₕ, _average_rule(f, Val(AVG_QUAD_POINTS)))

@inline avgₕ!(uₕ::VectorElement{<:ScalarGridSpace}, f::F, nq::Val{NQ}) where {F, NQ} = project!(uₕ, _average_rule(f, nq))

@inline avgₕ!(uₕ::VectorElement{<:CompositeGridSpace}, f::Tuple, nq::Val{NQ}) where {NQ} = project!(uₕ, _average_rule(f, nq))

@inline avgₕ!(uₕ::VectorElement{<:CompositeGridSpace}, f::F, nq::Val{NQ}) where {F, NQ} = project!(uₕ, _average_rule(f, nq))

# A one-component space is a scalar space, so an NC-tuple of functions with
# NC == 1 must still work.
#
# This method and the positional `Tuple{Any}` one at the top of the file are a pair, and the
# `ScalarGridSpace{D}` on that one is load-bearing rather than leftover: written as a bare
# `ScalarGridSpace` its positional signature becomes identical to this method's, and a
# keyword method with the same positional signature *overwrites* the positional one instead
# of coexisting with it -- precompilation then fails with "Method overwriting is not
# permitted". Its `D` is otherwise unused.
@inline avgₕ!(
    uₕ::VectorElement{<:ScalarGridSpace},
    f::Tuple{Any};
    quad_points::Union{Integer, Val} = Val(AVG_QUAD_POINTS),
    markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {N} = avgₕ!(uₕ, f[1]; quad_points = quad_points, markers = markers)

# `NQ` is a compile-time constant here (a type parameter), so the check folds away,
# costing nothing on the hot path. Without it, `quad_points = Val(0)` reached QuadGK's
# internal error message rather than validating the argument early.
@inline function _to_quad_val(nq::Val{NQ}) where {NQ}
    NQ >= 1 || throw(ArgumentError("quad_points must be >= 1, got $NQ"))
    return nq
end
@inline function _to_quad_val(nq::Integer)
    nq >= 1 || throw(ArgumentError("quad_points must be >= 1, got $nq"))
    return Val(Int(nq))
end

Base.@constprop :aggressive function avgₕ!(
        uₕ::VectorElement,
        f::F;
        quad_points::Union{Integer, Val} = Val(AVG_QUAD_POINTS),
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()
) where {F, N}
    nq = _to_quad_val(quad_points)

    if N > 0
        return project!(uₕ, _average_rule(f, nq), markers)
    end

    return project!(uₕ, _average_rule(f, nq))
end

# A tuple of functions is a rule per leaf; anything else is one rule.
@inline _average_rule(f::Tuple, nq::Val) = map(g -> CellAverage(g, nq), f)
@inline _average_rule(f, nq::Val) = CellAverage(f, nq)

# A concretely typed kernel (`_AvgKernel`) for the quadrature loop, avoiding anonymous
# closures over captures (`f`, `x`, `idxs`, `nodes`, `wts`). Explicit struct types ensure
# predictable inlining and eliminate allocation flakes inside parallel loop dispatch.
#
# No accumulator seed is built here: `_cell_average` seeds itself from `f`'s own first
# evaluation, so the mesh's element type `T` never leaks into the quadrature sum
# (gpena/Bramble.jl#148).
struct _AvgKernel{F, X, IX, NQ, T}
    f::F
    x::X
    idxs::IX
    nodes::NTuple{NQ, T}
    wts::NTuple{NQ, T}
end
@inline (k::_AvgKernel{F, X, IX, NQ, T})(i) where {F, X, IX, NQ, T} = _cell_average(
    k.f, k.x, k.idxs[i], k.nodes, k.wts)

# Same reasoning, for the tuple-valued (composite) quadrature call. `NC` is the space's
# *leaf* count (`length(components(uₕ))`, which flattens any nesting), not the space's own
# structural type parameter; kept as a type parameter here for a concrete struct even
# though `_cell_average` no longer needs it to build a seed.
struct _AvgScatterKernel{F, X, IX, NQ, T, NC}
    f::F
    x::X
    idxs::IX
    nodes::NTuple{NQ, T}
    wts::NTuple{NQ, T}
end
@inline (k::_AvgScatterKernel{F, X, IX, NQ, T, NC})(i) where {F, X, IX, NQ, T, NC} = _cell_average(
    k.f, k.x, k.idxs[i], k.nodes, k.wts)

# `CellAverage`'s side of the `project!` contract (`operators/projection.jl`). The rule
# carries the quadrature order; the driver decides the space's shape, the masking and the
# execution policy.
@inline function _rule_kernel(rule::CellAverage{F, NQ}, sp) where {F, NQ}
    Ωₕ = mesh(sp)
    nodes, wts = _gauss_rule(rule.nq, eltype(Ωₕ))
    return _AvgKernel(rule.f, half_points(Ωₕ), indices(Ωₕ), nodes, wts)
end

@inline function _rule_scatter_kernel(
        rule::CellAverage{F, NQ}, sp, ::Val{NC}
) where {F, NQ, NC}
    Ωₕ = mesh(sp)
    T = eltype(Ωₕ)
    nodes, wts = _gauss_rule(rule.nq, T)
    x = half_points(Ωₕ)
    idxs = indices(Ωₕ)
    return _AvgScatterKernel{typeof(rule.f), typeof(x), typeof(idxs), NQ, T, NC}(
        rule.f, x, idxs, nodes, wts
    )
end

@inline _rule_component(rule::CellAverage, k) = CellAverage(pt -> rule.f(pt)[k], rule.nq)

#------------------------------------------------------------------------------------------#
# Device kernel launch stubs (gpena/Bramble.jl#94, #174, S2.3 of
# .agents/plans/metal-and-apple-silicon-acceleration.md)
#
# `CellAverage`'s side of the `_device_project!`/`_device_scatter_project!` contract
# (`operators/projection.jl`), keyed on `DeviceLocality` so `project!`'s locality-based
# dispatch reaches them: the quadrature loop is exactly `_cell_average` above, called
# per point with the mesh's own `half_points(Ωₕ)` vector as a top-level kernel argument
# rather than nested inside a wrapper struct -- see `restriction.jl`'s stubs for why that
# distinction matters on a device. Without the KA extension loaded, the launcher throws a
# named diagnostic instead of failing several frames later on a scalar index.
#------------------------------------------------------------------------------------------#

"""
    _launch_cell_average!(v::AbstractVector, x::AbstractVector, nodes, wts, f, dev) -> Nothing

Fills `v[i]` with the [`_cell_average`](@ref) of `f` over the 1D cell around grid point `i`,
using the Gauss-Legendre `nodes`/`wts` and the mesh's half points `x`, via a
`KernelAbstractions.@kernel` launch on `dev`, filled by
`ext/BrambleKernelAbstractionsExt.jl`. `f` runs on the device: same GPU-compilability
requirement as [`_gpu_for!`](@ref).

# Throws
- `ErrorException`: no `KernelAbstractions` extension is loaded, so there is no device
  kernel to reach (`_throw_no_ka_projection_kernel`).
"""
_launch_cell_average!(v, x, nodes, wts, f, dev) = _throw_no_ka_projection_kernel("_launch_cell_average!")

"""
    _launch_cell_average_scatter!(mats::Tuple, x::AbstractVector, nodes, wts, f, dev) -> Nothing

The scatter counterpart of [`_launch_cell_average!`](@ref): computes the
[`_cell_average`](@ref) of `f` over each 1D cell once and scatters its components into the
destination tuple `mats`, via a `KernelAbstractions.@kernel` launch on `dev`. Same
GPU-compilability requirement on `f`.

# Throws
- `ErrorException`: no `KernelAbstractions` extension is loaded (`_throw_no_ka_projection_kernel`).
"""
function _launch_cell_average_scatter!(mats, x, nodes, wts, f, dev)
    _throw_no_ka_projection_kernel(
        "_launch_cell_average_scatter!"
    )
end

# The `D >= 2` counterparts: `half_points(Ωₕ::MeshnD) -> NTuple{D,AbstractVector}` is
# already the exact shape `_cell_average`'s 2D/3D methods take as `x` -- one coordinate
# vector per axis -- so the device kernel calls it directly with `x` (a `Tuple` of
# top-level device arrays) and `idxs` (`indices(Ωₕ)`, a bits `CartesianIndices`), the same
# non-nesting rule `restriction.jl`'s `_nd` launchers follow.
"""
    _launch_cell_average_nd!(v::AbstractVector, x::Tuple, idxs, nodes, wts, f, dev) -> Nothing

The `D >= 2` counterpart of [`_launch_cell_average!`](@ref): fills `v[i]` with the
[`_cell_average`](@ref) of `f` over the cell at Cartesian index `idxs[i]`, using the
per-axis half-point vectors `x` and the Gauss-Legendre `nodes`/`wts`, via a
`KernelAbstractions.@kernel` launch on `dev`. Same GPU-compilability requirement on `f`.

# Throws
- `ErrorException`: no `KernelAbstractions` extension is loaded (`_throw_no_ka_projection_kernel`).
"""
_launch_cell_average_nd!(v, x, idxs, nodes, wts, f, dev) = _throw_no_ka_projection_kernel("_launch_cell_average_nd!")

"""
    _launch_cell_average_scatter_nd!(mats::Tuple, x::Tuple, idxs, nodes, wts, f, dev) -> Nothing

The `D >= 2`, scatter counterpart of [`_launch_cell_average!`](@ref): computes the
[`_cell_average`](@ref) of `f` over the cell at Cartesian index `idxs[i]` once and scatters
its components into the destination tuple `mats`, via a `KernelAbstractions.@kernel` launch
on `dev`. Same GPU-compilability requirement on `f`.

# Throws
- `ErrorException`: no `KernelAbstractions` extension is loaded (`_throw_no_ka_projection_kernel`).
"""
function _launch_cell_average_scatter_nd!(mats, x, idxs, nodes, wts, f, dev)
    _throw_no_ka_projection_kernel(
        "_launch_cell_average_scatter_nd!"
    )
end

"""
    _device_project!(::DeviceLocality, rule::CellAverage, raw::AbstractVector, sp::ScalarGridSpace{1}) -> Bool
    _device_project!(::DeviceLocality, rule::CellAverage, raw::AbstractVector, sp::ScalarGridSpace{D}) where {D} -> Bool

Fills `raw` with the cell average of `rule.f` over every cell of the mesh `mesh(sp)`, via a
device kernel that calls the same [`_cell_average`](@ref) quadrature the CPU sweep uses
(gpena/Bramble.jl#94, #174, S2.3) -- the 1D method reads `half_points(Ωₕ)` directly, the
`D`-dimensional one (never reached for `D == 1`, since the method above is strictly more
specific) hands it the per-axis tuple `_cell_average`'s own 2D/3D methods already expect.
`rule.f` runs on the device either way: see [`_gpu_for!`](@ref) for what that requires of it.
"""
@inline function _device_project!(
        ::DeviceLocality, rule::CellAverage, raw::AbstractVector, sp::ScalarGridSpace{1}
)
    Ωₕ = mesh(sp)
    nodes, wts = _gauss_rule(rule.nq, eltype(Ωₕ))
    dev = ka_device(backend(sp))
    _launch_cell_average!(raw, half_points(Ωₕ), nodes, wts, rule.f, dev)
    return true
end

@inline function _device_project!(
        ::DeviceLocality, rule::CellAverage, raw::AbstractVector, sp::ScalarGridSpace{D}
) where {D}
    Ωₕ = mesh(sp)
    nodes, wts = _gauss_rule(rule.nq, eltype(Ωₕ))
    dev = ka_device(backend(sp))
    _launch_cell_average_nd!(raw, half_points(Ωₕ), indices(Ωₕ), nodes, wts, rule.f, dev)
    return true
end

"""
    _device_scatter_project!(::DeviceLocality, rule::CellAverage, raws::Tuple, sp, ::Val{NC}) -> Bool

The scatter counterpart of [`_device_project!`](@ref) above, for an `NC`-component
composite space `sp` whose leaves share one mesh. `sp` is typed generically for the same
reason `restriction.jl`'s counterpart is: the mesh's own dimension, checked on `mesh(sp)`
at runtime, picks the 1D or `D`-dimensional launcher.
"""
@inline function _device_scatter_project!(
        ::DeviceLocality, rule::CellAverage, raws::Tuple, sp, ::Val{NC}
) where {NC}
    Ωₕ = mesh(sp)
    nodes, wts = _gauss_rule(rule.nq, eltype(Ωₕ))
    dev = ka_device(backend(sp))
    if Ωₕ isa AbstractMeshType{1}
        _launch_cell_average_scatter!(raws, half_points(Ωₕ), nodes, wts, rule.f, dev)
    else
        _launch_cell_average_scatter_nd!(raws, half_points(Ωₕ), indices(Ωₕ), nodes, wts, rule.f, dev)
    end
    return true
end

#=
Cell averages are computed with a fixed tensor-product Gauss-Legendre rule per
cell. Every cell integral is independent, low dimensional and over a smooth
integrand, so a small fixed rule is both cheaper and allocation free.

Writing the cell integral on the reference cube,

    1/|C| ∫_C f = ∫_{[0,1]^D} f(a + t ⊙ (b - a)) dt,

because |C| = ∏ₖ (bₖ - aₖ) exactly: the cell around a grid point spans
consecutive half points, whose spacing is the half spacing that
`cell_measure` returns. The quadrature weights below sum to one, so the
weighted sum is the average and no measure division is needed.
=#

"""
    AVG_QUAD_POINTS

Default number of Gauss-Legendre points per direction, per cell, used by
[`avgₕ`](@ref). Six points are exact for polynomials up to degree eleven.

Unlike an adaptive rule, a fixed one does not tighten itself on coarse cells, so
the default is chosen to be accurate on cells far coarser than any practical
grid. Measured on 4 points spanning [-1, 4] with a function varying by a factor
of e^5 across the domain (deliberately harsher than a real mesh), the worst
error over 30 random grids was

    points   1D        2D        3D        evaluations per cell (3D)
    3        6.1e-5    1.6e-4    3.1e-4     27
    4        2.8e-7    8.1e-7    3.4e-6     64
    5        1.9e-9    4.1e-9    7.8e-9    125
    6        5.5e-12   1.4e-11   1.6e-11   216

Cost is `quad_points^D` evaluations per cell. On a fine grid three points are
usually ample; lower it with the `quad_points` keyword when the integrand is
cheap to resolve and the cells are small.
"""
const AVG_QUAD_POINTS = 6

"""
    _gauss_rule(::Val{N}, ::Type{T})

Returns `(nodes, weights)` for the `N`-point Gauss-Legendre rule on `[0, 1]` as
`NTuple{N,T}`, so the per-cell loop that consumes them does not allocate.

The rule is built by `QuadGK.gauss` in the requested element type, so `Float32`
and `BigFloat` grids get a rule at their own precision rather than a rounded
`Float64` one. Weights sum to one, which makes the weighted sum over a cell the
cell average directly.
"""
@generated function _gauss_rule(::Val{N}, ::Type{T}) where {N, T}
    # When `T` is an isbits float its precision is fixed by the type, so the rule
    # depends only on (N, T) and is folded into a compile-time constant: obtaining it
    # then costs nothing at all. This covers Float16/32/64 and equally the stack
    # allocated extended precision types such as Double64 or Float64x2.
    #
    # Non-isbits types (notably BigFloat, whose precision is a runtime setting)
    # fall back to building the rule dynamically per call at current precision.
    if isbitstype(T)
        try
            x, w = gauss(T, N, zero(T), one(T))
            nodes = Expr(:tuple, x...)
            wts = Expr(:tuple, w...)
            return :(($nodes, $wts))
        catch
            # fall through to the run-time rule
        end
    end
    return :(_gauss_rule_runtime(Val($N), $T))
end

@inline function _gauss_rule_runtime(::Val{N}, ::Type{T}) where {N, T}
    x, w = gauss(T, N, zero(T), one(T))
    return NTuple{N, T}(x), NTuple{N, T}(w)
end

"""
    _cell_average(f, x::AbstractVector, idx::CartesianIndex{1}, nodes::NTuple{NQ, T}, wts::NTuple{NQ, T}) where {NQ, T}
    _cell_average(f, x::AbstractVector, i::Int, nodes::NTuple{NQ, T}, wts::NTuple{NQ, T}) where {NQ, T}
    _cell_average(f, x::NTuple{D}, idx::CartesianIndex{D}, nodes::NTuple{NQ, T}, wts::NTuple{NQ, T}) where {D, NQ, T}

`NQ`-point Gauss-Legendre average of `f` over the cell spanned by the half points around
`idx`, along every axis at once. Scalar or composite (`NTuple`) by `f`'s own return type:
broadcasting `.+`/`.*` over a `Number` accumulator computes exactly the scalar arithmetic
`+`/`*` would, and over an `NTuple` accumulator the per-component sum, so no `::Val{NC}` or
caller-built seed is needed to tell the two apart (gpena/Bramble.jl#102, #148). The
accumulator is seeded from `f`'s own first evaluation rather than `zero(T)`, so `T` (the
mesh's element type) never leaks into the sum when `f` returns something else, such as a
`ForwardDiff.Dual` under AD (gpena/Bramble.jl#148).

A one-dimensional mesh answers `half_points` with a plain vector but indexes with
`CartesianIndex{1}`; the `CartesianIndex{1}` method unwraps that to the `Int` the
`x::AbstractVector` method takes, which is the only thing the 1D/`D`-dimensional split in
this file ever did (gpena/Bramble.jl#69). The 1D, 2D and 3D methods are specialised for the
meshes this package builds (`D <= 3`); the generic `NTuple{D}` method exists for dispatch
correctness at any `D` and is tested directly rather than through a mesh.

# Arguments
- `f`: Function evaluated at quadrature points, returning a scalar or an `NTuple`.
- `x`: Half points along each axis (`AbstractVector` in 1D, `NTuple{D}` otherwise).
- `idx`/`i`: The cell's `CartesianIndex` (or, in 1D, its `Int`).
- `nodes`, `wts`: Gauss-Legendre nodes and weights on `[0, 1]` from [`_gauss_rule`](@ref).

# Returns
- The cell average of `f`, matching `f`'s own return shape.
"""
@inline _cell_average(
    f,
    x::AbstractVector,
    idx::CartesianIndex{1},
    nodes::NTuple{NQ, T},
    wts::NTuple{NQ, T}
) where {NQ, T} = _cell_average(f, x, idx[1], nodes, wts)

# Average of `f` over the 1D cell spanned by `x[i] .. x[i+1]`.
#
# The accumulator starts from `f`'s own first evaluation, not `zero(T)`: `T` is the mesh's
# element type, but `f` can return something else entirely -- a `ForwardDiff.Dual` under
# AD, say -- and seeding from the mesh forced a mid-loop type change the moment `f`'s
# result first arrived (gpena/Bramble.jl#148). Element type comes from the data, never from
# the space, same rule `Rₕ` follows. `zero.(...)` broadcasts over both the scalar and the
# `NTuple` (composite) return shape, so one method still covers both.
@inline function _cell_average(
        f, x::AbstractVector, i::Int, nodes::NTuple{NQ, T}, wts::NTuple{NQ, T}
) where {NQ, T}
    @inbounds a = T(x[i])
    @inbounds d = T(x[i + 1]) - a

    @inbounds s = zero.(f(a + nodes[1] * d))
    @inbounds for q in 1:NQ
        s = s .+ wts[q] .* f(a + nodes[q] * d)
    end
    return s
end

# 2D cell average, scalar or composite by `f`'s own return type -- see the 1D method above.
#
# The composite case is also what a 1D mesh's single-function-returning-all-components
# form needs on a composite space; the 1D method above covers it directly (a 1D mesh
# answers `half_points` with a plain vector, not a one-tuple of vectors, so it does not
# reach this NTuple{2}-indexed method in the first place).
@inline function _cell_average(
        f, x::NTuple{2}, idx::CartesianIndex{2}, nodes::NTuple{NQ, T}, wts::NTuple{NQ, T}
) where {NQ, T}
    @inbounds i, j = idx[1], idx[2]
    @inbounds a1 = T(x[1][i])
    @inbounds d1 = T(x[1][i + 1]) - a1
    @inbounds a2 = T(x[2][j])
    @inbounds d2 = T(x[2][j + 1]) - a2

    # `p1` does not depend on `q2`, so evaluating it inside the `q2` loop recomputed it
    # NQ² times instead of NQ (gpena/Bramble.jl#114). Precomputed once per axis instead;
    # `ntuple` over a `Val`-known `NQ` unrolls to a stack tuple, no heap allocation.
    p1s = ntuple(q -> a1 + nodes[q] * d1, Val(NQ))
    p2s = ntuple(q -> a2 + nodes[q] * d2, Val(NQ))

    # Accumulator seeded from `f`'s own first evaluation, not `zero(T)` -- see the 1D
    # method above (gpena/Bramble.jl#148).
    @inbounds s = zero.(f((p1s[1], p2s[1])))
    @inbounds for q2 in 1:NQ
        w2 = wts[q2]
        p2 = p2s[q2]
        for q1 in 1:NQ
            w1 = wts[q1] * w2
            p1 = p1s[q1]
            s = s .+ w1 .* f((p1, p2))
        end
    end
    return s
end

# 3D cell average, scalar or composite by `f`'s own return type -- see the 1D method above.
@inline function _cell_average(
        f, x::NTuple{3}, idx::CartesianIndex{3}, nodes::NTuple{NQ, T}, wts::NTuple{NQ, T}
) where {NQ, T}
    @inbounds i, j, k = idx[1], idx[2], idx[3]
    @inbounds a1 = T(x[1][i])
    @inbounds d1 = T(x[1][i + 1]) - a1
    @inbounds a2 = T(x[2][j])
    @inbounds d2 = T(x[2][j + 1]) - a2
    @inbounds a3 = T(x[3][k])
    @inbounds d3 = T(x[3][k + 1]) - a3

    # `p1` (independent of `q2`/`q3`) was recomputed NQ³ times instead of NQ, and `p2`
    # (independent of `q3`) NQ² times instead of NQ (gpena/Bramble.jl#114). Both
    # precomputed once per axis instead, same zero-allocation `ntuple` idiom as the 2D
    # method above.
    p1s = ntuple(q -> a1 + nodes[q] * d1, Val(NQ))
    p2s = ntuple(q -> a2 + nodes[q] * d2, Val(NQ))
    p3s = ntuple(q -> a3 + nodes[q] * d3, Val(NQ))

    # Accumulator seeded from `f`'s own first evaluation, not `zero(T)` -- see the 1D
    # method above (gpena/Bramble.jl#148).
    @inbounds s = zero.(f((p1s[1], p2s[1], p3s[1])))
    @inbounds for q3 in 1:NQ
        w3 = wts[q3]
        p3 = p3s[q3]
        for q2 in 1:NQ
            w23 = wts[q2] * w3
            p2 = p2s[q2]
            for q1 in 1:NQ
                w1 = wts[q1] * w23
                p1 = p1s[q1]
                s = s .+ w1 .* f((p1, p2, p3))
            end
        end
    end
    return s
end

# Average of `f` over the D-dimensional cell around `idx`, whose corners are the half
# points `x[k][idx[k]]` and `x[k][idx[k] + 1]` along each axis -- scalar or composite by
# `f`'s own return type, see the 1D method above. No mesh this package builds is more than
# 3D, so the 1D/2D/3D specialized methods above always take priority in practice; this
# generic one exists for dispatch correctness at any `D`, tested directly rather than
# through a mesh.
@inline function _cell_average(
        f, x::NTuple{D}, idx::CartesianIndex{D}, nodes::NTuple{NQ, T}, wts::NTuple{NQ, T}
) where {D, NQ, T}
    a = ntuple(k -> @inbounds(T(x[k][idx[k]])), Val(D))
    b = ntuple(k -> @inbounds(T(x[k][idx[k] + 1])), Val(D))

    # Accumulator seeded from `f`'s own first evaluation, not `zero(T)` -- see the 1D
    # method above (gpena/Bramble.jl#148).
    pt1 = ntuple(k -> a[k] + nodes[1] * (b[k] - a[k]), Val(D))
    s = zero.(f(pt1))
    @inbounds for q in CartesianIndices(ntuple(_ -> NQ, Val(D)))
        w = one(T)
        for k in 1:D
            w *= wts[q[k]]
        end
        pt = ntuple(k -> a[k] + nodes[q[k]] * (b[k] - a[k]), Val(D))
        s = s .+ w .* f(pt)
    end
    return s
end
