# vector_calculus.jl
# The higher-level discrete vector calculus operators: divergence, curl and the Laplacian
# (gpena/Bramble.jl#158). Each is a contraction of the directional differences
# `operators/difference.jl` already provides, and each used to be written out by hand at
# every call site.
#
# Three things are deliberate here.
#
# The names follow the gradient's. `∇ₕ` is the *backward* gradient and `∇₊ₕ` the forward one
# (operators/difference.jl), so `divₕ`/`curlₕ` are backward and `div₊ₕ`/`curl₊ₕ` forward --
# not the `div₋ₕ` the issue asked for, which would read as a fourth naming convention for the
# same choice. `Δₕ` has no such pair: it is the conservative composition of the two.
#
# Each operator accumulates into its destination in one traversal per direction rather than
# composing the public aliases, which would need a temporary per direction. That is what lets
# every `!` form allocate nothing without a caller-supplied scratch buffer.
#
# The vector field is an `NTuple{D, VectorElement}` -- what `∇ₕ` returns -- or a composite
# grid function with `D` leaves. Both spellings reach the same kernels.

# --- Device dispatch for the fused vector-calculus kernels (gpena/Bramble.jl#306, #302,
# S12, part 4) ------------------------------------------------------------------------- #
#
# Every accumulating engine below (`_accumulate_backward!`, `_accumulate_forward!`,
# `_accumulate_laplacian!`, `_avg_backward_inplace!`) is a host `@inbounds @simd for` loop
# that scalar-indexes its destination, which a device-backed `VectorElement` refuses
# outright. `divₕ!`/`div₊ₕ!`/`curlₕ!`/`curl₊ₕ!`/`Δₕ!`/`εₕ!` each check `locality` on their
# destination first and, on a device array, route to one fused `KernelAbstractions.@kernel`
# launch that reads every component it needs exactly once per grid point -- instead of the
# CPU path's one pass per spatial direction (or, for `εₕ!`'s off-diagonal entries, one pass
# each for a difference and the average composed onto it) -- rather than raising
# `ScalarIndexingDisallowed` partway through the first accumulation. `ext/BrambleKernelAbstractionsExt.jl`
# fills in each launcher; without that extension loaded, the launcher throws a named
# diagnostic instead, matching every other device stub in this package.
function _throw_no_ka_vector_calculus_kernel(fname::String)
    error(
        "$fname requires KernelAbstractions.jl to apply a vector-calculus operator to a " *
        "device-backed VectorElement. Add `using KernelAbstractions` (and the package " *
        "providing this backend's device, e.g. `using Metal`) before calling it on this " *
        "backend.",
    )
end

"""
    _launch_fused_divergence!(out, comps::Tuple, hs::Tuple, dims::Tuple, dir::GridDirection, dev) -> Nothing

Fills `out` with the discrete divergence of the `D`-component vector field `comps`
(`divₕ`/`div₊ₕ`'s `Backward`/`Forward` forms), one fused device kernel reading every
component once per grid point, filled by `ext/BrambleKernelAbstractionsExt.jl`.
"""
_launch_fused_divergence!(out, comps, hs, dims, dir, dev) = _throw_no_ka_vector_calculus_kernel("divₕ!/div₊ₕ!")

"""
    _launch_fused_curl2d!(out, u1, u2, h1, h2, dims::Tuple, dir::GridDirection, dev) -> Nothing

Fills `out` with the 2D scalar curl `D_{dir,x}(u2) - D_{dir,y}(u1)`, one fused device
kernel, filled by `ext/BrambleKernelAbstractionsExt.jl`.
"""
_launch_fused_curl2d!(out, u1, u2, h1, h2, dims, dir, dev) = _throw_no_ka_vector_calculus_kernel("curlₕ!/curl₊ₕ!")

"""
    _launch_fused_curl3d!(out1, out2, out3, u1, u2, u3, h1, h2, h3, dims::Tuple, dir::GridDirection, dev) -> Nothing

Fills the three components of the 3D curl, one fused device kernel reading `u1`, `u2` and
`u3` once per grid point, filled by `ext/BrambleKernelAbstractionsExt.jl`.
"""
function _launch_fused_curl3d!(out1, out2, out3, u1, u2, u3, h1, h2, h3, dims, dir, dev)
    _throw_no_ka_vector_calculus_kernel(
        "curlₕ!/curl₊ₕ!"
    )
end

"""
    _launch_fused_laplacian!(out, u, hbs::Tuple, hss::Tuple, dims::Tuple, dev) -> Nothing

Fills `out` with the conservative discrete Laplacian of `u`, one fused device kernel
summing every direction's flux difference per grid point, filled by
`ext/BrambleKernelAbstractionsExt.jl`.
"""
_launch_fused_laplacian!(out, u, hbs, hss, dims, dev) = _throw_no_ka_vector_calculus_kernel("Δₕ!")

"""
    _launch_fused_strain_offdiag!(out, ui, uj, hi, hj, dims::Tuple, dim_i::Val, dim_j::Val, dev) -> Nothing

Fills `out` with the off-diagonal strain tensor entry
`(M₋ᵢ(D₋ⱼ(uᵢ)) + M₋ⱼ(D₋ᵢ(uⱼ))) / 2`, one fused device kernel composing the difference and
the average in a single pass rather than two, filled by
`ext/BrambleKernelAbstractionsExt.jl`.
"""
_launch_fused_strain_offdiag!(out, ui, uj, hi, hj, dims, dim_i, dim_j, dev) = _throw_no_ka_vector_calculus_kernel(
    "εₕ!"
)

@inline _direction_spacing(sub, ::Backward) = backward_spacings_for_derivative(sub)
@inline _direction_spacing(sub, ::Forward) = forward_spacings_for_derivative(sub)

@inline _is_device(v::AbstractVector) = locality(typeof(v)) isa DeviceLocality

# --- The accumulating engines ------------------------------------------------------- #
#
# `_difference_engine!` (operators/difference.jl) *writes* its result, one direction at a
# time. A divergence, a curl and a Laplacian all sum several directions into one array, so
# they add instead, and the destination is zeroed once by the caller rather than per
# direction. The boundary conventions are the engines' own: a backward difference truncates
# to zero on the first slice of its direction, a forward one on the last.
#
# `h` carries a type parameter for the reason it does in `_difference_engine!`: an argument
# of function or view type that the body only forwards is not specialised on, and the
# spacing would be boxed at every grid point.
@inline function _accumulate_backward!(
        out, u, h::H, dims::NTuple{D, Int}, ::Val{DIM}, s
) where {H, D, DIM}
    li = LinearIndices(dims)
    step = _stencil_step(Val(DIM), Val(D))
    interior, _ = _stencil_ranges(axes(li), Val(DIM), Backward())

    @inbounds @simd for I in CartesianIndices(interior)
        idx, other = li[I], li[I - step]
        out[idx] += s * (u[idx] - u[other]) / _get_h_val(h, I[DIM])
    end
    return nothing
end

@inline function _accumulate_forward!(
        out, u, h::H, dims::NTuple{D, Int}, ::Val{DIM}, s
) where {H, D, DIM}
    li = LinearIndices(dims)
    step = _stencil_step(Val(DIM), Val(D))
    interior, _ = _stencil_ranges(axes(li), Val(DIM), Forward())

    @inbounds @simd for I in CartesianIndices(interior)
        idx, other = li[I], li[I + step]
        out[idx] += s * (u[other] - u[idx]) / _get_h_val(h, I[DIM])
    end
    return nothing
end

# One direction of the conservative Laplacian, fused: `D̃(D₋(u))` without the intermediate
# grid function. Reading the composition off the two engines rather than re-deriving it is
# what keeps the boundary conventions identical -- `D₋` is zero on the first slice, so the
# term it would contribute is simply absent there, and `D̃` is zero on the last, so nothing
# is written on it at all.
@inline function _accumulate_laplacian!(
        out, u, hb::HB, hs::HS, dims::NTuple{D, Int}, ::Val{DIM}
) where {HB, HS, D, DIM}
    li = LinearIndices(dims)
    step = _stencil_step(Val(DIM), Val(D))
    interior, _ = _stencil_ranges(axes(li), Val(DIM), Forward())

    @inbounds @simd for I in CartesianIndices(interior)
        idx, fwd = li[I], li[I + step]
        i = I[DIM]
        forward_flux = (u[fwd] - u[idx]) / _get_h_val(hb, i + 1)
        backward_flux = i == 1 ? zero(forward_flux) :
                        (u[idx] - u[li[I - step]]) / _get_h_val(hb, i)
        out[idx] += (forward_flux - backward_flux) / _get_h_val(hs, i)
    end
    return nothing
end

# --- The vector field, however it is spelled ---------------------------------------- #

@inline _field_components(uₕ::NTuple{D, VectorElement}) where {D} = uₕ

# `components` already answers for both kinds of grid function: the leaves of a composite,
# and the one-tuple of a scalar one -- which is the 1D case, where `∇ₕ` returns the bare
# element rather than a 1-tuple.
@inline _field_components(uₕ::VectorElement) = components(uₕ)

@inline _field_space(uₕ) = space(first(_field_components(uₕ)))

@noinline function _throw_field_arity(got::Int, D::Int, op::String)
    throw(
        DimensionMismatch(
        "$op needs one component per spatial dimension: got $got components on a $(D)D mesh."
    ),
    )
end

@inline function _check_field_arity(comps, ::Val{D}, op::String) where {D}
    length(comps) == D || _throw_field_arity(length(comps), D, op)
    return nothing
end

# --- Divergence ---------------------------------------------------------------------- #

"""
    divₕ(uₕ) -> VectorElement
    divₕ(vₕ, uₕ) -> VectorElement

Returns the discrete divergence of the vector field `uₕ`, built from the backward
differences:

```math
\\textrm{div}_h(\\textrm{u}_h)(I) = \\sum_{d=1}^{D} \\textrm{D}_{-,x_d}(\\textrm{u}_{h,d})(I)
```

`uₕ` is an `NTuple{D, VectorElement}` -- what [`∇ₕ`](@ref) returns -- or a grid function of a
[`CompositeGridSpace`](@ref) with one leaf per spatial dimension.

The backward difference is truncated to zero on the first slice of each direction, exactly as
[`D₋ₓ`](@ref) is, so the divergence there is the sum of the directions that still have a
stencil.

[`divₕ!`](@ref) writes into a destination instead, and allocates nothing.

`∇ₕ ⋅ uₕ` is the same call written with `⋅` (`LinearAlgebra.dot`, exported from Bramble)
in place of the function name: `∇ₕ ⋅ uₕ === divₕ(uₕ)`. It needs `⋅` in scope (`using
LinearAlgebra: ⋅` or Bramble's own re-export). `∇cₕ`, `∇̽ₕ`, `∇̃ₕ` and `∇₊ₕ` contract to
[`divcₕ`](@ref), [`div̽ₕ`](@ref), [`diṽₕ`](@ref) and [`div₊ₕ`](@ref) the same way.

See also: [`div₊ₕ`](@ref), [`curlₕ`](@ref), [`Δₕ`](@ref)
"""
@inline divₕ(uₕ) = divₕ!(similar(first(_field_components(uₕ))), uₕ)

"""
    divₕ!(vₕ::VectorElement, uₕ) -> VectorElement

The in-place form of [`divₕ`](@ref): the discrete divergence of `uₕ`, written into `vₕ`.

Allocates nothing, where the allocating form allocates its result. Returns `vₕ`, so it
composes: `normₕ(divₕ!(vₕ, uₕ))`.

`vₕ` is a grid function of the *scalar* space the components live on, and must not be one of
them: every stencil reads a neighbour, and aliasing would read values already overwritten.
"""
function divₕ!(vₕ::VectorElement, uₕ)
    comps = _field_components(uₕ)
    Wₕ = _field_space(uₕ)
    D = dim(mesh(Wₕ))
    _check_field_arity(comps, Val(D), "divₕ")
    _divergence!(vₕ, comps, Wₕ, Backward(), Val(D))
    return vₕ
end

"""
    div₊ₕ(uₕ) -> VectorElement
    div₊ₕ!(vₕ::VectorElement, uₕ) -> VectorElement

The forward-difference discrete divergence, ``\\sum_d \\textrm{D}_{+,x_d}(\\textrm{u}_{h,d})``.

The forward twin of [`divₕ`](@ref), standing to it as [`∇₊ₕ`](@ref) stands to [`∇ₕ`](@ref),
and truncated to zero on the *last* slice of each direction rather than the first.

`∇₊ₕ ⋅ uₕ === div₊ₕ(uₕ)`.
"""
@inline div₊ₕ(uₕ) = div₊ₕ!(similar(first(_field_components(uₕ))), uₕ)

@doc (@doc div₊ₕ)
function div₊ₕ!(vₕ::VectorElement, uₕ)
    comps = _field_components(uₕ)
    Wₕ = _field_space(uₕ)
    D = dim(mesh(Wₕ))
    _check_field_arity(comps, Val(D), "div₊ₕ")
    _divergence!(vₕ, comps, Wₕ, Forward(), Val(D))
    return vₕ
end

function _divergence!(vₕ, comps, Wₕ, dir, ::Val{D}) where {D}
    Ωₕ = mesh(Wₕ)
    dims = npoints(Ωₕ, Tuple)
    out = parent(vₕ)
    if _is_device(out)
        hs = ntuple(d -> _direction_spacing(Ωₕ(d), dir), Val(D))
        dev = ka_device(backend(Ωₕ))
        _launch_fused_divergence!(out, map(parent, comps), hs, dims, dir, dev)
    else
        fill!(out, zero(eltype(out)))
        _accumulate_direction!(vₕ, comps, Ωₕ, dims, dir, Val(D), Val(D))
    end
    return nothing
end

# One direction per rung. Recursion on `Val(d)` rather than a loop over `1:D`, because
# `Val(d)` captured in a closure boxes (gpena/Bramble.jl#146) and this is the whole reason
# the vectorial aliases are written out the way they are.
@inline function _accumulate_direction!(
        vₕ, comps, Ωₕ, dims, dir, ::Val{d}, ::Val{D}
) where {d, D}
    _accumulate_one!(parent(vₕ), parent(comps[d]), Ωₕ, dims, dir, Val(d), true)
    _accumulate_direction!(vₕ, comps, Ωₕ, dims, dir, Val(d - 1), Val(D))
    return nothing
end

@inline _accumulate_direction!(vₕ, comps, Ωₕ, dims, dir, ::Val{0}, ::Val{D}) where {D} = nothing

@inline function _accumulate_one!(out, u, Ωₕ, dims, ::Backward, ::Val{d}, s) where {d}
    return _accumulate_backward!(
        out, u, backward_spacings_for_derivative(Ωₕ(d)), dims, Val(d), s
    )
end

@inline function _accumulate_one!(out, u, Ωₕ, dims, ::Forward, ::Val{d}, s) where {d}
    return _accumulate_forward!(
        out, u, forward_spacings_for_derivative(Ωₕ(d)), dims, Val(d), s
    )
end

# --- Curl ----------------------------------------------------------------------------- #

"""
    curlₕ(uₕ) -> VectorElement or NTuple{3, VectorElement}
    curlₕ!(vₕ, uₕ) -> vₕ

Returns the discrete curl of the vector field `uₕ`, built from the backward differences.

In 2D it is the scalar

```math
\\textrm{curl}_h(\\textrm{u}_h) =
    \\textrm{D}_{-,x}(\\textrm{u}_{h,2}) - \\textrm{D}_{-,y}(\\textrm{u}_{h,1})
```

and in 3D the three-component field
``(\\partial_y u_3 - \\partial_z u_2,\\; \\partial_z u_1 - \\partial_x u_3,\\;
\\partial_x u_2 - \\partial_y u_1)``, each derivative a backward difference. There is no 1D
curl, and asking for one is an error rather than a zero.

`curlₕ!` takes a destination -- a grid function in 2D, a 3-tuple of them in 3D -- and
allocates nothing. [`curl₊ₕ`](@ref) is the forward twin.

`∇ₕ × uₕ` is the same call written with `×` (`LinearAlgebra.cross`) in place of the
function name: `∇ₕ × uₕ === curlₕ(uₕ)`, needing `×` in scope. `∇cₕ`, `∇̽ₕ` and `∇̃ₕ`
contract to [`curlcₕ`](@ref), [`curl̽ₕ`](@ref) and [`curl̃ₕ`](@ref) the same way.

See also: [`divₕ`](@ref), [`∇ₕ`](@ref)
"""
@inline curlₕ(uₕ) = _curl(uₕ, Backward(), "curlₕ")

"""
    curl₊ₕ(uₕ) -> VectorElement or NTuple{3, VectorElement}
    curl₊ₕ!(vₕ, uₕ) -> vₕ

The forward-difference discrete curl, the twin of [`curlₕ`](@ref).

`∇₊ₕ × uₕ === curl₊ₕ(uₕ)`.
"""
@inline curl₊ₕ(uₕ) = _curl(uₕ, Forward(), "curl₊ₕ")

# `op` is the public name the caller was reached by, so an arity error names it.
@inline function _curl(uₕ, dir, op::String)
    comps = _field_components(uₕ)
    D = dim(mesh(_field_space(uₕ)))
    _check_field_arity(comps, Val(D), op)
    return _curl_alloc(uₕ, comps, dir, op, Val(D))
end

@inline _curl_alloc(uₕ, comps, dir, op, ::Val{2}) = _curl!(similar(first(comps)), uₕ, dir, op)
@inline _curl_alloc(uₕ, comps, dir, op, ::Val{3}) = _curl!(
    ntuple(_ -> similar(first(comps)), Val(3)), uₕ, dir, op
)
@noinline _curl_alloc(uₕ, comps, dir, op, ::Val{D}) where {D} = throw(
    ArgumentError("the discrete curl is defined in 2D and 3D; this mesh is $(D)D.")
)

@doc (@doc curlₕ)
@inline curlₕ!(vₕ, uₕ) = _curl!(vₕ, uₕ, Backward(), "curlₕ!")

@doc (@doc curl₊ₕ)
@inline curl₊ₕ!(vₕ, uₕ) = _curl!(vₕ, uₕ, Forward(), "curl₊ₕ!")

function _curl!(vₕ::VectorElement, uₕ, dir, op::String)
    comps = _field_components(uₕ)
    Ωₕ = mesh(_field_space(uₕ))
    dims = npoints(Ωₕ, Tuple)
    _check_field_arity(comps, Val(2), op)

    out = parent(vₕ)
    if _is_device(out)
        h1 = _resolve_device_spacing(_direction_spacing(Ωₕ(1), dir))
        h2 = _resolve_device_spacing(_direction_spacing(Ωₕ(2), dir))
        dev = ka_device(backend(Ωₕ))
        _launch_fused_curl2d!(out, parent(comps[1]), parent(comps[2]), h1, h2, dims, dir, dev)
    else
        fill!(out, zero(eltype(out)))
        _accumulate_one!(out, parent(comps[2]), Ωₕ, dims, dir, Val(1), true)
        _accumulate_one!(out, parent(comps[1]), Ωₕ, dims, dir, Val(2), -true)
    end
    return vₕ
end

function _curl!(vₕ::NTuple{3, VectorElement}, uₕ, dir, op::String)
    comps = _field_components(uₕ)
    Ωₕ = mesh(_field_space(uₕ))
    dims = npoints(Ωₕ, Tuple)
    _check_field_arity(comps, Val(3), op)

    outs = map(parent, vₕ)
    if _is_device(outs[1])
        hs = ntuple(d -> _resolve_device_spacing(_direction_spacing(Ωₕ(d), dir)), Val(3))
        dev = ka_device(backend(Ωₕ))
        _launch_fused_curl3d!(
            outs[1], outs[2], outs[3], parent(comps[1]), parent(comps[2]), parent(comps[3]),
            hs[1], hs[2], hs[3], dims, dir, dev
        )
    else
        # (∂₂u₃ - ∂₃u₂, ∂₃u₁ - ∂₁u₃, ∂₁u₂ - ∂₂u₁), written out rather than looped over
        # cyclic index triples: the directions have to stay compile-time literals.
        for k in 1:3
            fill!(outs[k], zero(eltype(outs[k])))
        end
        _accumulate_one!(outs[1], parent(comps[3]), Ωₕ, dims, dir, Val(2), true)
        _accumulate_one!(outs[1], parent(comps[2]), Ωₕ, dims, dir, Val(3), -true)
        _accumulate_one!(outs[2], parent(comps[1]), Ωₕ, dims, dir, Val(3), true)
        _accumulate_one!(outs[2], parent(comps[3]), Ωₕ, dims, dir, Val(1), -true)
        _accumulate_one!(outs[3], parent(comps[2]), Ωₕ, dims, dir, Val(1), true)
        _accumulate_one!(outs[3], parent(comps[1]), Ωₕ, dims, dir, Val(2), -true)
    end
    return vₕ
end

# --- Laplacian -------------------------------------------------------------------------- #

"""
    Δₕ(uₕ::VectorElement) -> VectorElement
    Δₕ!(vₕ::VectorElement, uₕ::VectorElement) -> vₕ

Returns the conservative discrete Laplacian of the grid function `uₕ`, matrix-free:

```math
\\Delta_h(\\textrm{u}_h)(I) = \\sum_{d=1}^{D}
    \\tilde{\\textrm{D}}_{+,x_d}\\big(\\textrm{D}_{-,x_d}(\\textrm{u}_h)\\big)(I)
```

which on a non-uniform mesh is the flux-difference form

```math
\\frac{1}{h^{*}_{i}}\\left(
    \\frac{u_{i+1} - u_i}{h_{i+1}} - \\frac{u_i - u_{i-1}}{h_i}\\right),
\\qquad h^{*}_i = \\frac{h_i + h_{i+1}}{2}
```

per direction. This is the composition whose summation-by-parts identity with
[`inner₊`](@ref) holds, `innerₕ(Δₕ(uₕ), vₕ) = -inner₊(∇ₕ(uₕ), ∇ₕ(vₕ))` for grid functions
vanishing on the boundary, which is what makes it *the* discrete Laplacian rather than one of
several plausible five-point stencils.

Evaluated in a single traversal per direction rather than as two nested operator calls, so
`Δₕ!` needs no scratch grid function and allocates nothing. It agrees with the composition
`D̃ₓ(D₋ₓ(uₕ))` entry for entry, truncation at the two ends of each axis included.

See also: [`divₕ`](@ref), [`∇ₕ`](@ref), [`D̃ₓ`](@ref)
"""
@inline Δₕ(uₕ::VectorElement) = Δₕ!(similar(uₕ), uₕ)

@doc (@doc Δₕ)
function Δₕ!(vₕ::VectorElement, uₕ::VectorElement)
    _check_no_alias(vₕ, uₕ)
    Ωₕ = mesh(space(uₕ))
    dims = npoints(Ωₕ, Tuple)
    out = parent(vₕ)
    if _is_device(out)
        D = dim(Ωₕ)
        hbs = ntuple(d -> _resolve_device_spacing(backward_spacings_for_derivative(Ωₕ(d))), Val(D))
        hss = ntuple(d -> _resolve_device_spacing(star_spacings(Ωₕ(d))), Val(D))
        dev = ka_device(backend(Ωₕ))
        _launch_fused_laplacian!(out, parent(uₕ), hbs, hss, dims, dev)
    else
        fill!(out, zero(eltype(out)))
        _laplacian_direction!(out, parent(uₕ), Ωₕ, dims, Val(dim(Ωₕ)), Val(dim(Ωₕ)))
    end
    return vₕ
end

@inline function _laplacian_direction!(out, u, Ωₕ, dims, ::Val{d}, ::Val{D}) where {d, D}
    _accumulate_laplacian!(
        out, u, backward_spacings_for_derivative(Ωₕ(d)), star_spacings(Ωₕ(d)), dims, Val(d)
    )
    _laplacian_direction!(out, u, Ωₕ, dims, Val(d - 1), Val(D))
    return nothing
end

@inline _laplacian_direction!(out, u, Ωₕ, dims, ::Val{0}, ::Val{D}) where {D} = nothing

# --- Strain tensor -------------------------------------------------------------------- #
#
# gpena/Bramble.jl#234 (runtime half, v3.3.0 plan S6.7): the discrete strain tensor of a
# vector field,
#
#     εₕ(uₕ) = (∇ₕ(uₕ) + ∇ₕ(uₕ)ᵀ) / 2
#
# The gradient tensor itself needs no new code. `∇ₕ`/`∇₊ₕ` (operators/difference.jl) already
# accept a `D`-leaf composite grid function through the same generic dispatch every composite
# grid function goes through -- one leaf at a time -- and for a `D`-leaf composite on a
# `D`-dimensional mesh that componentwise gradient already *is* the gradient tensor:
# `components(∇ₕ(uₕ)[i])[j]` is `D₋ᵢ(uⱼ)`, the `(i, j)` entry. That dispatch is also the one
# `test/space/composite_operators.jl`'s "Componentwise equality"/"Vectorial forms" testsets
# exercise for leaf counts that have nothing to do with the mesh dimension (a multi-field
# space, not a vector field); giving `∇ₕ` a second, more specific method for `D`-leaf
# composites -- the only way to make it reject a mismatched leaf count -- would take over
# every composite call regardless of leaf count (Julia dispatch has no way to prefer it only
# when the leaf count happens to match `D`) and break that already-tested generic behaviour.
# So no method is added here: the vector-field reading of `∇ₕ(uₕ)`/`∇₊ₕ(uₕ)` for a `D`-leaf
# composite is the existing one, not a new one, and it does not reject other leaf counts --
# only `εₕ` below does, since it has no such pre-existing generic meaning to preserve.
#
# The strain tensor is not simply that sum, though. `D₋ⱼ(uᵢ)` and `D₋ᵢ(uⱼ)` are staggered at
# different points -- a backward difference along `xⱼ` sits half a cell along `j`, one along
# `xᵢ` sits half a cell along `i` -- so adding them where they stand would add values that do
# not belong to the same grid point. The averages `M₋ᵢ`/`M₋ⱼ` (operators/average.jl) relocate
# each term to the point the two share before the sum, which is what pins `ε_ii` to a face
# centre (offset in one direction) and `ε_ij` (`i != j`) to an edge centre (offset in two):
#
#     ε_ii(uₕ) = D₋ᵢ(uᵢ)
#     ε_ij(uₕ) = (M₋ᵢ(D₋ⱼ(uᵢ)) + M₋ⱼ(D₋ᵢ(uⱼ))) / 2                                  (i != j)
#
# the same convention `docs/src/examples/elasticity_3d.jl` hand-expands term by term (its own
# document-local `εₕ(p, i, j)` closure), which this replaces as a general operator.
#
# `divₕ` needs no extension for this. #158's `divₕ` (above) is the unstaggered
# `Σᵢ D₋ᵢ(uᵢ)`, the SBP dual `∇ₕ`/`inner₊` already close an integration-by-parts identity
# against (`test/space/discrete_calculus_identities.jl`, `test/space/sbp_identities.jl`,
# `test/space/inference_allocation.jl`); it already accepts a `D`-leaf composite and returns a
# grid function, which is everything this subplan's goal asks of `divₕ`. The elasticity
# example's own divergence term, `Σᵢ M₋ⱼM₋ₖ(D₋ᵢuᵢ)` (bringing every direction's contribution to
# the one cell-centred point before summing, `{j, k}` the two directions other than `i`), is a
# *different* object with a different staggering, needed only there; folding it into `divₕ`
# would change what `divₕ` means for every existing caller of that identity rather than extend
# it, so it is left to whatever the elasticity example itself becomes (S6.6) instead.

# The backward average, in place: `out[I] = (out[I] + out[I - eᵢ]) / 2` along `DIM`, zero on
# the first slice, exactly `_average_engine!`'s `Backward()` case (operators/average.jl)
# computes out of place. `_stencil_ranges`/`_stencil_step` are the same shared traversal
# helpers `_difference_engine!` (difference.jl) and the accumulating engines above build the
# in-place, `@inbounds @simd` engines out of; this one cannot join them, because it needs the
# *reverse* of their order.
#
# Reversed, because `out[I - eᵢ]` must still hold its pre-averaged value when `I` is visited.
# Two points that differ only along `DIM` keep their relative order under a full reversal of
# `CartesianIndices` (their linear indices differ by a fixed positive multiple of `step`
# alone, so whichever is smaller stays smaller, just later in the reversed sequence) even
# though the reversed traversal interleaves points from other lines (other coordinates)
# between them -- irrelevant here, since those touch different memory. That is what lets the
# strain tensor's off-diagonal entries average a freshly-written difference into its own
# destination without a scratch array of their own. Not `@simd`: consecutive iterations can be
# the very two points this loop's own carried dependency links.
@inline function _avg_backward_inplace!(
        out::AbstractVector, dims::NTuple{D, Int}, ::Val{DIM}
) where {D, DIM}
    li = LinearIndices(dims)
    step = _stencil_step(Val(DIM), Val(D))
    interior, boundary = _stencil_ranges(axes(li), Val(DIM), Backward())

    # The interior's first slice (`I[DIM] == 2`) reads the boundary slice
    # (`I[DIM] == 1`) as its neighbour, so the boundary is zeroed *after* the interior
    # pass has read it, not before -- zeroing it first would feed the interior loop's
    # last read a value the average operator itself put there, not the field's own.
    @inbounds for I in Iterators.reverse(CartesianIndices(interior))
        idx = li[I]
        out[idx] = (out[idx] + out[li[I - step]]) / 2
    end

    @inbounds for I in CartesianIndices(boundary)
        out[li[I]] = zero(eltype(out))
    end
    return nothing
end

# `ε_ii = D₋ᵢ(uᵢ)`, written straight into its destination's own storage: no averaging, so no
# in-place hazard to work around. On device this is exactly the same one-sided finite
# difference `D₋ₓ!`/`D₋ᵧ!`/`D₋₂!` already apply, so it reuses that launcher rather than a
# new kernel of its own.
@inline function _strain_diag!(dest, comps, Ωₕ, dims, ::Val{i}) where {i}
    out = parent(dest[i][i])
    h = backward_spacings_for_derivative(Ωₕ(i))
    if _is_device(out)
        dev = ka_device(backend(Ωₕ))
        _launch_difference_onesided!(out, parent(comps[i]), _resolve_device_spacing(h), dims, Backward(), Val(i), dev)
    else
        _difference_engine!(out, parent(comps[i]), h, dims, Backward(), Val(i))
    end
    return nothing
end

# `ε_ij = (M₋ᵢ(D₋ⱼ(uᵢ)) + M₋ⱼ(D₋ᵢ(uⱼ))) / 2`, `i != j`, computed once and copied into both
# `dest[i][j]` and `dest[j][i]` -- symmetric by construction, rather than by two equal but
# independent computations that would only agree to floating-point rounding. Each half of the
# sum is written into its own destination slot (`dest[i][j]` and `dest[j][i]` in turn), then
# averaged into itself in place, so the pair needs no scratch array beyond the two
# destinations the caller already owns.
@inline function _strain_pair!(dest, comps, Ωₕ, dims, ::Val{i}, ::Val{j}) where {i, j}
    dij, dji = dest[i][j], dest[j][i]
    out_ij = parent(dij)
    hj = backward_spacings_for_derivative(Ωₕ(j))
    hi = backward_spacings_for_derivative(Ωₕ(i))

    if _is_device(out_ij)
        dev = ka_device(backend(Ωₕ))
        _launch_fused_strain_offdiag!(
            out_ij, parent(comps[i]), parent(comps[j]), _resolve_device_spacing(hi),
            _resolve_device_spacing(hj), dims, Val(i), Val(j), dev
        )
        copyto!(parent(dji), out_ij)
    else
        _difference_engine!(out_ij, parent(comps[i]), hj, dims, Backward(), Val(j))
        _avg_backward_inplace!(out_ij, dims, Val(i))

        _difference_engine!(parent(dji), parent(comps[j]), hi, dims, Backward(), Val(i))
        _avg_backward_inplace!(parent(dji), dims, Val(j))

        dij .= (dij .+ dji) ./ 2
        dji .= dij
    end
    return nothing
end

# One row (fixed `i`) of the tensor: the diagonal entry, then every off-diagonal pair
# `(i, j)` with `j > i` (the mirror `(j, i)` is filled by the same call, `_strain_pair!`
# writing both). Recursion on `Val(d)` rather than a loop over `1:D`, for the same boxing
# reason `_accumulate_direction!` above is written this way (gpena/Bramble.jl#146).
@inline function _strain_offdiag!(dest, comps, Ωₕ, dims, ::Val{i}, ::Val{j}, ::Val{D}) where {i, j, D}
    j > i && _strain_pair!(dest, comps, Ωₕ, dims, Val(i), Val(j))
    _strain_offdiag!(dest, comps, Ωₕ, dims, Val(i), Val(j - 1), Val(D))
    return nothing
end

@inline _strain_offdiag!(dest, comps, Ωₕ, dims, ::Val{i}, ::Val{0}, ::Val{D}) where {i, D} = nothing

@inline function _strain_rows!(dest, comps, Ωₕ, dims, ::Val{i}, ::Val{D}) where {i, D}
    _strain_diag!(dest, comps, Ωₕ, dims, Val(i))
    _strain_offdiag!(dest, comps, Ωₕ, dims, Val(i), Val(D), Val(D))
    _strain_rows!(dest, comps, Ωₕ, dims, Val(i - 1), Val(D))
    return nothing
end

@inline _strain_rows!(dest, comps, Ωₕ, dims, ::Val{0}, ::Val{D}) where {D} = nothing

@inline _strain_alloc(comps, ::Val{D}) where {D} = ntuple(
    _ -> ntuple(_ -> similar(first(comps)), Val(D)), Val(D)
)

"""
    εₕ(uₕ) -> NTuple{D, NTuple{D, VectorElement}}

Returns the discrete strain tensor of the vector field `uₕ`,

```math
\\varepsilon_h(\\textrm{u}_h) = \\tfrac{1}{2}\\left(\\nabla_h \\textrm{u}_h +
    (\\nabla_h \\textrm{u}_h)^{T}\\right),
```

entry by entry

```math
\\varepsilon^{ii}_h(\\textrm{u}_h) = \\textrm{D}_{-,x_i}(\\textrm{u}_{h,i}), \\qquad
\\varepsilon^{ij}_h(\\textrm{u}_h) = \\tfrac{1}{2}\\left(
    \\textrm{M}_{-,x_i}\\big(\\textrm{D}_{-,x_j}(\\textrm{u}_{h,i})\\big) +
    \\textrm{M}_{-,x_j}\\big(\\textrm{D}_{-,x_i}(\\textrm{u}_{h,j})\\big)\\right), \\quad i \\neq j.
```

`uₕ` is an `NTuple{D, VectorElement}` -- what [`∇ₕ`](@ref) returns -- or a grid function of a
[`CompositeGridSpace`](@ref) with one leaf per spatial dimension, exactly as [`divₕ`](@ref)
takes it.

`ε_ii` sits on the face centre a backward difference along `xᵢ` alone reaches; `ε_ij`
(`i != j`) sits on the edge centre the two averages bring the two halves of the shear term
to. The result is a `D`-by-`D` nested tuple of grid functions, symmetric by construction:
`εₕ(uₕ)[i][j] === εₕ(uₕ)[j][i]`.

[`εₕ!`](@ref) writes into a preallocated `D`-by-`D` nested tuple of destinations instead, and
allocates nothing.

See also: [`∇ₕ`](@ref), [`divₕ`](@ref)
"""
function εₕ(uₕ)
    comps = _field_components(uₕ)
    Wₕ = _field_space(uₕ)
    D = dim(mesh(Wₕ))
    _check_field_arity(comps, Val(D), "εₕ")
    dest = _strain_alloc(comps, Val(D))
    return εₕ!(dest, uₕ)
end

"""
    εₕ!(dest::NTuple{D, NTuple{D, VectorElement}}, uₕ) -> dest

The in-place form of [`εₕ`](@ref): the discrete strain tensor of `uₕ`, written into `dest`.

Allocates nothing, where the allocating form allocates its `D * D` results. Returns `dest`,
so it composes.

`dest[i][j]` and `dest[j][i]` end up holding the same values for `i != j` (`εₕ` is symmetric
by construction); both are written, so either may be read.
"""
function εₕ!(dest::NTuple{D, NTuple{D, VectorElement}}, uₕ) where {D}
    comps = _field_components(uₕ)
    Wₕ = _field_space(uₕ)
    dim(mesh(Wₕ)) == D || throw(
        DimensionMismatch(
        "εₕ! destination is $(D)x$D but the mesh is $(dim(mesh(Wₕ)))D"
    ),
    )
    _check_field_arity(comps, Val(D), "εₕ!")
    Ωₕ = mesh(Wₕ)
    dims = npoints(Ωₕ, Tuple)
    _strain_rows!(dest, comps, Ωₕ, dims, Val(D), Val(D))
    return dest
end

# --- Centered vector calculus (gpena/Bramble.jl#287) ------------------------------------- #
#
# The centered family reuses the accumulating machinery above with `Centered()` as the
# direction: `_accumulate_one!` gains a `Centered` method over its own engine, and
# `divcₕ`/`curlcₕ` then go through `_divergence!`/`_curl!` unchanged. The strain tensor needs
# no averaging, since every centered difference sits on the grid point itself.
#
# There is no device kernel for any of them yet. Each entry point checks the locality of its
# destination (or, for the allocating forms, of its input) before any scalar indexing, and
# throws a named error rather than `ScalarIndexingDisallowed` partway through.

@inline _direction_spacing(sub, ::Centered) = star_spacings(sub)

@noinline function _throw_no_centered_device_kernel(fname::String)
    error(
        "$fname has no device kernel yet: the centered vector-calculus operators run on " *
        "host arrays only. Apply it to a host-backed grid function instead.",
    )
end

@inline _check_centered_host(v::VectorElement, fname) = _is_device(parent(v)) ?
                                                        _throw_no_centered_device_kernel(fname) :
                                                        nothing
@inline _check_centered_host(v::Tuple, fname) = _check_centered_host(first(v), fname)

# `out[I] += s * (u[I + eᵢ] - u[I - eᵢ]) / (2 h*ᵢ)` over the slices with a neighbour on each
# side; both end slices receive nothing, which is `Dc`'s zero there. `hs` is the averaged
# spacing `star_spacings` returns, the same view `Dc` divides by, so each term is exactly the
# value `Dc` writes.
@inline function _accumulate_centered!(
        out, u, hs::H, dims::NTuple{D, Int}, ::Val{DIM}, s
) where {H, D, DIM}
    li = LinearIndices(dims)
    step = _stencil_step(Val(DIM), Val(D))
    interior, _, _ = _centered_stencil_ranges(axes(li), Val(DIM))

    @inbounds @simd for I in CartesianIndices(interior)
        idx = li[I]
        out[idx] += s * (u[li[I + step]] - u[li[I - step]]) / (2 * _get_h_val(hs, I[DIM]))
    end
    return nothing
end

@inline function _accumulate_one!(out, u, Ωₕ, dims, ::Centered, ::Val{d}, s) where {d}
    sub = Ωₕ(d)
    _check_centered_points(sub, d)
    return _accumulate_centered!(out, u, star_spacings(sub), dims, Val(d), s)
end

"""
    ∇cₕ(uₕ::VectorElement) -> VectorElement or NTuple{D, VectorElement}

Returns the centered discrete gradient of the grid function `uₕ`, one centered difference
per direction:

```math
\\nabla_{c,h}(\\textrm{u}_h) = \\left(\\textrm{Dc}_{x_1}(\\textrm{u}_h), \\ldots,
    \\textrm{Dc}_{x_D}(\\textrm{u}_h)\\right), \\qquad
\\textrm{Dc}_{x_d}(\\textrm{u}_h)(i) = \\frac{u_{i+1} - u_{i-1}}{h_i + h_{i+1}}.
```

The same function as [`Dcₕ`](@ref), under the name the centered vector calculus family
shares. In 1D it returns the bare grid function, above a `D`-tuple. Each difference is
truncated to zero on the first and last slice of its direction.

[`∇cₕ!`](@ref) writes into a destination instead, and allocates nothing.

See also: [`divcₕ`](@ref), [`curlcₕ`](@ref), [`εcₕ`](@ref), [`∇ₕ`](@ref)
"""
const ∇cₕ = Dcₕ

"""
    ∇cₕ!(dest, uₕ::VectorElement) -> dest

The in-place form of [`∇cₕ`](@ref): the centered gradient of `uₕ`, written into `dest` --
a grid function in 1D, a `D`-tuple of them above, the shape `∇cₕ` returns.

Allocates nothing. No destination may alias `uₕ`.
"""
function ∇cₕ!(dest, uₕ::VectorElement)
    _check_centered_host(dest, "∇cₕ!")
    Ωₕ = mesh(space(uₕ))
    D = dim(Ωₕ)
    outs = dest isa VectorElement ? (dest,) : dest
    length(outs) == D || _throw_field_arity(length(outs), D, "∇cₕ!")
    _centered_gradient!(outs, parent(uₕ), Ωₕ, npoints(Ωₕ, Tuple), Centered(), Val(D))
    return dest
end

# Shared by `∇cₕ!` and `∇̽ₕ!`: `dir` is `Centered()` or `CrossWeighted()`, and
# `_direction_spacing` picks the spacing each divides by.
@inline function _centered_gradient!(outs, u, Ωₕ, dims, dir, ::Val{d}) where {d}
    sub = Ωₕ(d)
    _check_centered_points(sub, d)
    _difference_engine!(parent(outs[d]), u, _direction_spacing(sub, dir), dims, dir, Val(d))
    _centered_gradient!(outs, u, Ωₕ, dims, dir, Val(d - 1))
    return nothing
end

@inline _centered_gradient!(outs, u, Ωₕ, dims, dir, ::Val{0}) = nothing

"""
    divcₕ(uₕ) -> VectorElement
    divcₕ!(vₕ::VectorElement, uₕ) -> vₕ

Returns the centered discrete divergence of the vector field `uₕ`:

```math
\\textrm{div}_{c,h}(\\textrm{u}_h)(I) = \\sum_{d=1}^{D} \\textrm{Dc}_{x_d}(\\textrm{u}_{h,d})(I)
```

`uₕ` is spelled as for [`divₕ`](@ref): an `NTuple{D, VectorElement}`, or a grid function of a
[`CompositeGridSpace`](@ref) with one leaf per spatial dimension (in 1D, a scalar grid
function). Each centered difference is zero on the first and last slice of its direction.

For fields vanishing on the boundary it is minus the adjoint of [`∇cₕ`](@ref),
``(\\textrm{div}_{c,h} \\textrm{F}, \\textrm{u})_h = -\\sum_d (\\textrm{F}_d,
\\textrm{Dc}_{x_d} \\textrm{u})_h``, on any mesh.

`divcₕ!` writes into `vₕ`, which must not be one of the components, and allocates nothing.

`∇cₕ ⋅ uₕ === divcₕ(uₕ)`.

See also: [`divₕ`](@ref), [`curlcₕ`](@ref), [`εcₕ`](@ref)
"""
function divcₕ(uₕ)
    _check_centered_host(first(_field_components(uₕ)), "divcₕ")
    return divcₕ!(similar(first(_field_components(uₕ))), uₕ)
end

@doc (@doc divcₕ)
function divcₕ!(vₕ::VectorElement, uₕ)
    _check_centered_host(vₕ, "divcₕ!")
    comps = _field_components(uₕ)
    Wₕ = _field_space(uₕ)
    D = dim(mesh(Wₕ))
    _check_field_arity(comps, Val(D), "divcₕ")
    _divergence!(vₕ, comps, Wₕ, Centered(), Val(D))
    return vₕ
end

"""
    curlcₕ(uₕ) -> VectorElement or NTuple{3, VectorElement}
    curlcₕ!(vₕ, uₕ) -> vₕ

Returns the centered discrete curl of the vector field `uₕ`. In 2D it is the scalar

```math
\\textrm{curl}_{c,h}(\\textrm{u}_h) =
    \\textrm{Dc}_{x}(\\textrm{u}_{h,2}) - \\textrm{Dc}_{y}(\\textrm{u}_{h,1})
```

and in 3D the three-component field
``(\\partial_y u_3 - \\partial_z u_2,\\; \\partial_z u_1 - \\partial_x u_3,\\;
\\partial_x u_2 - \\partial_y u_1)``, each derivative a centered difference. There is no 1D
curl, and asking for one is an `ArgumentError`.

`curlcₕ!` takes a destination -- a grid function in 2D, a 3-tuple of them in 3D -- and
allocates nothing.

`∇cₕ × uₕ === curlcₕ(uₕ)`.

See also: [`curlₕ`](@ref), [`divcₕ`](@ref), [`∇cₕ`](@ref)
"""
function curlcₕ(uₕ)
    _check_centered_host(first(_field_components(uₕ)), "curlcₕ")
    return _curl(uₕ, Centered(), "curlcₕ")
end

@doc (@doc curlcₕ)
function curlcₕ!(vₕ, uₕ)
    _check_centered_host(vₕ, "curlcₕ!")
    return _curl!(vₕ, uₕ, Centered(), "curlcₕ!")
end

"""
    εcₕ(uₕ) -> NTuple{D, NTuple{D, VectorElement}}
    εcₕ!(dest::NTuple{D, NTuple{D, VectorElement}}, uₕ) -> dest

Returns the centered discrete strain tensor of the vector field `uₕ`, entry by entry

```math
\\varepsilon^{ii}_{c,h}(\\textrm{u}_h) = \\textrm{Dc}_{x_i}(\\textrm{u}_{h,i}), \\qquad
\\varepsilon^{ij}_{c,h}(\\textrm{u}_h) = \\tfrac{1}{2}\\left(
    \\textrm{Dc}_{x_j}(\\textrm{u}_{h,i}) + \\textrm{Dc}_{x_i}(\\textrm{u}_{h,j})\\right),
    \\quad i \\neq j.
```

Every centered difference sits on the grid point itself, so unlike [`εₕ`](@ref) no average
relocates the shear terms. `uₕ` is spelled as for [`divcₕ`](@ref). The result is symmetric
by construction: `dest[i][j]` and `dest[j][i]` hold the same values.

`εcₕ!` writes into a preallocated `D`-by-`D` nested tuple and allocates nothing.

See also: [`εₕ`](@ref), [`∇cₕ`](@ref), [`divcₕ`](@ref)
"""
function εcₕ(uₕ)
    comps = _field_components(uₕ)
    _check_centered_host(first(comps), "εcₕ")
    D = dim(mesh(_field_space(uₕ)))
    _check_field_arity(comps, Val(D), "εcₕ")
    return εcₕ!(_strain_alloc(comps, Val(D)), uₕ)
end

@doc (@doc εcₕ)
function εcₕ!(dest::NTuple{D, NTuple{D, VectorElement}}, uₕ) where {D}
    _check_centered_host(dest[1], "εcₕ!")
    comps = _field_components(uₕ)
    Ωₕ = mesh(_field_space(uₕ))
    dim(Ωₕ) == D || throw(DimensionMismatch("εcₕ! destination is $(D)x$D but the mesh is $(dim(Ωₕ))D"))
    _check_field_arity(comps, Val(D), "εcₕ!")
    _centered_strain_rows!(dest, comps, Ωₕ, npoints(Ωₕ, Tuple), Centered(), Val(D), Val(D))
    return dest
end

# Shared by `εcₕ!` and `ε̽ₕ!`, with `dir` `Centered()` or `CrossWeighted()`: both
# differences are co-located, so the strain is the symmetric part of the gradient as is.
# `ε_ij`, `i != j`: `D_j(u_i)` into `dest[i][j]`, `D_i(u_j)` into `dest[j][i]`, then their
# mean into both, so the two entries are equal bit for bit.
@inline function _centered_strain_pair!(dest, comps, Ωₕ, dims, dir, ::Val{i}, ::Val{j}) where {i, j}
    dij, dji = dest[i][j], dest[j][i]
    _difference_engine!(parent(dij), parent(comps[i]), _direction_spacing(Ωₕ(j), dir), dims, dir, Val(j))
    _difference_engine!(parent(dji), parent(comps[j]), _direction_spacing(Ωₕ(i), dir), dims, dir, Val(i))
    dij .= (dij .+ dji) ./ 2
    dji .= dij
    return nothing
end

@inline function _centered_strain_offdiag!(dest, comps, Ωₕ, dims, dir, ::Val{i}, ::Val{j}) where {i, j}
    j > i && _centered_strain_pair!(dest, comps, Ωₕ, dims, dir, Val(i), Val(j))
    _centered_strain_offdiag!(dest, comps, Ωₕ, dims, dir, Val(i), Val(j - 1))
    return nothing
end

@inline _centered_strain_offdiag!(dest, comps, Ωₕ, dims, dir, ::Val{i}, ::Val{0}) where {i} = nothing

@inline function _centered_strain_rows!(dest, comps, Ωₕ, dims, dir, ::Val{i}, ::Val{D}) where {i, D}
    sub = Ωₕ(i)
    _check_centered_points(sub, i)
    _difference_engine!(parent(dest[i][i]), parent(comps[i]), _direction_spacing(sub, dir), dims, dir, Val(i))
    _centered_strain_offdiag!(dest, comps, Ωₕ, dims, dir, Val(i), Val(D))
    _centered_strain_rows!(dest, comps, Ωₕ, dims, dir, Val(i - 1), Val(D))
    return nothing
end

@inline _centered_strain_rows!(dest, comps, Ωₕ, dims, dir, ::Val{0}, ::Val{D}) where {D} = nothing

# --- Tilde vector calculus and the forward strain (gpena/Bramble.jl#287) ------------------- #
#
# `D̃` is a forward difference over the averaged spacing `star_spacings` returns, not over the
# forward spacing. `StarForward` is the marker that lets the accumulating machinery above
# select that spacing: `_accumulate_one!` gains a method that runs the forward engine over
# `star_spacings`, and `diṽₕ`/`curl̃ₕ` then go through `_divergence!`/`_curl!` unchanged. It is
# not a `GridDirection`, since no stencil traversal of its own is needed: the traversal is
# `Forward()`'s.
#
# As for the centered family, there is no device kernel yet, and each entry point checks
# locality before any scalar indexing.

struct StarForward end

@noinline function _throw_no_star_device_kernel(fname::String)
    error(
        "$fname has no device kernel yet: the tilde vector-calculus operators and the " *
        "forward strain run on host arrays only. Apply it to a host-backed grid function " *
        "instead.",
    )
end

@inline _check_star_host(v::VectorElement, fname) = _is_device(parent(v)) ?
                                                    _throw_no_star_device_kernel(fname) :
                                                    nothing
@inline _check_star_host(v::Tuple, fname) = _check_star_host(first(v), fname)

# `out[I] += s * (u[I + eᵢ] - u[I]) / h*ᵢ`, zero on the last slice: `_accumulate_forward!`
# over the spacing `D̃` divides by, so each term is exactly the value `D̃` writes.
@inline function _accumulate_one!(out, u, Ωₕ, dims, ::StarForward, ::Val{d}, s) where {d}
    return _accumulate_forward!(out, u, star_spacings(Ωₕ(d)), dims, Val(d), s)
end

"""
    ∇̃ₕ(uₕ::VectorElement) -> VectorElement or NTuple{D, VectorElement}

Returns the tilde discrete gradient of the grid function `uₕ`, one tilde forward
difference per direction:

```math
\\tilde{\\nabla}_h(\\textrm{u}_h) = \\left(\\tilde{\\textrm{D}}_{x_1}(\\textrm{u}_h),
    \\ldots, \\tilde{\\textrm{D}}_{x_D}(\\textrm{u}_h)\\right), \\qquad
\\tilde{\\textrm{D}}_{x_d}(\\textrm{u}_h)(i) = \\frac{u_{i+1} - u_i}{(h_i + h_{i+1})/2}.
```

The same function as [`D̃ₕ`](@ref), under the name the tilde vector calculus family shares.
In 1D it returns the bare grid function, above a `D`-tuple. Each difference is truncated to
zero on the last slice of its direction.

[`∇̃ₕ!`](@ref) writes into a destination instead, and allocates nothing.

See also: [`diṽₕ`](@ref), [`curl̃ₕ`](@ref), [`∇ₕ`](@ref)
"""
const ∇̃ₕ = D̃ₕ


"""
    ∇̃ₕ!(dest, uₕ::VectorElement) -> dest

The in-place form of [`∇̃ₕ`](@ref): the tilde gradient of `uₕ`, written into `dest` -- a
grid function in 1D, a `D`-tuple of them above, the shape `∇̃ₕ` returns.

Allocates nothing. No destination may alias `uₕ`.
"""
function ∇̃ₕ!(dest, uₕ::VectorElement)
    _check_star_host(dest, "∇̃ₕ!")
    Ωₕ = mesh(space(uₕ))
    D = dim(Ωₕ)
    outs = dest isa VectorElement ? (dest,) : dest
    length(outs) == D || _throw_field_arity(length(outs), D, "∇̃ₕ!")
    _star_gradient!(outs, parent(uₕ), Ωₕ, npoints(Ωₕ, Tuple), Val(D))
    return dest
end

@inline function _star_gradient!(outs, u, Ωₕ, dims, ::Val{d}) where {d}
    _difference_engine!(parent(outs[d]), u, star_spacings(Ωₕ(d)), dims, Forward(), Val(d))
    _star_gradient!(outs, u, Ωₕ, dims, Val(d - 1))
    return nothing
end

@inline _star_gradient!(outs, u, Ωₕ, dims, ::Val{0}) = nothing

"""
    diṽₕ(uₕ) -> VectorElement
    diṽₕ!(vₕ::VectorElement, uₕ) -> vₕ

Returns the tilde discrete divergence of the vector field `uₕ`:

```math
\\tilde{\\textrm{div}}_h(\\textrm{u}_h)(I) =
    \\sum_{d=1}^{D} \\tilde{\\textrm{D}}_{x_d}(\\textrm{u}_{h,d})(I)
```

`uₕ` is spelled as for [`divₕ`](@ref): an `NTuple{D, VectorElement}`, or a grid function of a
[`CompositeGridSpace`](@ref) with one leaf per spatial dimension (in 1D, a scalar grid
function). Each tilde difference is zero on the last slice of its direction.

For fields vanishing on the boundary it pairs with the backward difference by summation by
parts, ``(\\tilde{\\textrm{div}}_h \\textrm{F}, \\textrm{v})_h =
-\\sum_d (\\textrm{F}_d, \\textrm{D}_{-,x_d} \\textrm{v})_{+,d}``.

`diṽₕ!` writes into `vₕ`, which must not be one of the components, and allocates nothing.

`∇̃ₕ ⋅ uₕ === diṽₕ(uₕ)`.

See also: [`divₕ`](@ref), [`curl̃ₕ`](@ref), [`∇̃ₕ`](@ref)
"""
function diṽₕ(uₕ)
    _check_star_host(first(_field_components(uₕ)), "diṽₕ")
    return diṽₕ!(similar(first(_field_components(uₕ))), uₕ)
end

@doc (@doc diṽₕ)
function diṽₕ!(vₕ::VectorElement, uₕ)
    _check_star_host(vₕ, "diṽₕ!")
    comps = _field_components(uₕ)
    Wₕ = _field_space(uₕ)
    D = dim(mesh(Wₕ))
    _check_field_arity(comps, Val(D), "diṽₕ")
    _divergence!(vₕ, comps, Wₕ, StarForward(), Val(D))
    return vₕ
end

"""
    curl̃ₕ(uₕ) -> VectorElement or NTuple{3, VectorElement}
    curl̃ₕ!(vₕ, uₕ) -> vₕ

Returns the tilde discrete curl of the vector field `uₕ`. In 2D it is the scalar

```math
\\tilde{\\textrm{curl}}_h(\\textrm{u}_h) =
    \\tilde{\\textrm{D}}_{x}(\\textrm{u}_{h,2}) -
    \\tilde{\\textrm{D}}_{y}(\\textrm{u}_{h,1})
```

and in 3D the three-component field
``(\\partial_y u_3 - \\partial_z u_2,\\; \\partial_z u_1 - \\partial_x u_3,\\;
\\partial_x u_2 - \\partial_y u_1)``, each derivative a tilde forward difference. There is
no 1D curl, and asking for one is an `ArgumentError`.

`curl̃ₕ!` takes a destination -- a grid function in 2D, a 3-tuple of them in 3D -- and
allocates nothing.

`∇̃ₕ × uₕ === curl̃ₕ(uₕ)`.

See also: [`curlₕ`](@ref), [`diṽₕ`](@ref), [`∇̃ₕ`](@ref)
"""
function curl̃ₕ(uₕ)
    _check_star_host(first(_field_components(uₕ)), "curl̃ₕ")
    return _curl(uₕ, StarForward(), "curl̃ₕ")
end

@doc (@doc curl̃ₕ)
function curl̃ₕ!(vₕ, uₕ)
    _check_star_host(vₕ, "curl̃ₕ!")
    return _curl!(vₕ, uₕ, StarForward(), "curl̃ₕ!")
end

# The forward average, in place: `out[I] = (out[I] + out[I + eᵢ]) / 2` along `DIM`, zero on
# the last slice, the mirror of `_avg_backward_inplace!` above. Traversed in forward order,
# so `out[I + eᵢ]` still holds its pre-averaged value when `I` is visited, and the last
# slice is zeroed only after the interior pass has read it. Not `@simd`, for the same
# carried-dependency reason.
@inline function _avg_forward_inplace!(
        out::AbstractVector, dims::NTuple{D, Int}, ::Val{DIM}
) where {D, DIM}
    li = LinearIndices(dims)
    step = _stencil_step(Val(DIM), Val(D))
    interior, boundary = _stencil_ranges(axes(li), Val(DIM), Forward())

    @inbounds for I in CartesianIndices(interior)
        idx = li[I]
        out[idx] = (out[idx] + out[li[I + step]]) / 2
    end

    @inbounds for I in CartesianIndices(boundary)
        out[li[I]] = zero(eltype(out))
    end
    return nothing
end

# `ε_ij = (M₊ᵢ(D₊ⱼ(uᵢ)) + M₊ⱼ(D₊ᵢ(uⱼ))) / 2`, `i != j`: each half written into its own slot
# and averaged in place, then their mean copied into both, so the entries are equal bit for bit.
@inline function _forward_strain_pair!(dest, comps, Ωₕ, dims, ::Val{i}, ::Val{j}) where {i, j}
    dij, dji = dest[i][j], dest[j][i]
    _difference_engine!(
        parent(dij), parent(comps[i]), forward_spacings_for_derivative(Ωₕ(j)), dims, Forward(), Val(j)
    )
    _avg_forward_inplace!(parent(dij), dims, Val(i))
    _difference_engine!(
        parent(dji), parent(comps[j]), forward_spacings_for_derivative(Ωₕ(i)), dims, Forward(), Val(i)
    )
    _avg_forward_inplace!(parent(dji), dims, Val(j))
    dij .= (dij .+ dji) ./ 2
    dji .= dij
    return nothing
end

@inline function _forward_strain_offdiag!(dest, comps, Ωₕ, dims, ::Val{i}, ::Val{j}) where {i, j}
    j > i && _forward_strain_pair!(dest, comps, Ωₕ, dims, Val(i), Val(j))
    _forward_strain_offdiag!(dest, comps, Ωₕ, dims, Val(i), Val(j - 1))
    return nothing
end

@inline _forward_strain_offdiag!(dest, comps, Ωₕ, dims, ::Val{i}, ::Val{0}) where {i} = nothing

@inline function _forward_strain_rows!(dest, comps, Ωₕ, dims, ::Val{i}, ::Val{D}) where {i, D}
    _difference_engine!(
        parent(dest[i][i]), parent(comps[i]), forward_spacings_for_derivative(Ωₕ(i)), dims, Forward(), Val(i)
    )
    _forward_strain_offdiag!(dest, comps, Ωₕ, dims, Val(i), Val(D))
    _forward_strain_rows!(dest, comps, Ωₕ, dims, Val(i - 1), Val(D))
    return nothing
end

@inline _forward_strain_rows!(dest, comps, Ωₕ, dims, ::Val{0}, ::Val{D}) where {D} = nothing

"""
    ε₊ₕ(uₕ) -> NTuple{D, NTuple{D, VectorElement}}
    ε₊ₕ!(dest::NTuple{D, NTuple{D, VectorElement}}, uₕ) -> dest

Returns the forward discrete strain tensor of the vector field `uₕ`, the forward twin of
[`εₕ`](@ref), entry by entry

```math
\\varepsilon^{ii}_{+,h}(\\textrm{u}_h) = \\textrm{D}_{+,x_i}(\\textrm{u}_{h,i}), \\qquad
\\varepsilon^{ij}_{+,h}(\\textrm{u}_h) = \\tfrac{1}{2}\\left(
    \\textrm{M}_{+,x_i}\\big(\\textrm{D}_{+,x_j}(\\textrm{u}_{h,i})\\big) +
    \\textrm{M}_{+,x_j}\\big(\\textrm{D}_{+,x_i}(\\textrm{u}_{h,j})\\big)\\right), \\quad i \\neq j.
```

`uₕ` is spelled as for [`divₕ`](@ref). Every difference and average is truncated to zero on
the last slice of its direction. The result is symmetric by construction: `dest[i][j]` and
`dest[j][i]` hold the same values.

`ε₊ₕ!` writes into a preallocated `D`-by-`D` nested tuple and allocates nothing.

See also: [`εₕ`](@ref), [`∇₊ₕ`](@ref), [`div₊ₕ`](@ref)
"""
function ε₊ₕ(uₕ)
    comps = _field_components(uₕ)
    _check_star_host(first(comps), "ε₊ₕ")
    D = dim(mesh(_field_space(uₕ)))
    _check_field_arity(comps, Val(D), "ε₊ₕ")
    return ε₊ₕ!(_strain_alloc(comps, Val(D)), uₕ)
end

@doc (@doc ε₊ₕ)
function ε₊ₕ!(dest::NTuple{D, NTuple{D, VectorElement}}, uₕ) where {D}
    _check_star_host(dest[1], "ε₊ₕ!")
    comps = _field_components(uₕ)
    Ωₕ = mesh(_field_space(uₕ))
    dim(Ωₕ) == D || throw(DimensionMismatch("ε₊ₕ! destination is $(D)x$D but the mesh is $(dim(Ωₕ))D"))
    _check_field_arity(comps, Val(D), "ε₊ₕ!")
    _forward_strain_rows!(dest, comps, Ωₕ, npoints(Ωₕ, Tuple), Val(D), Val(D))
    return dest
end

# --- Cross-weighted vector calculus (gpena/Bramble.jl#349) --------------------------------- #
#
# The cross-weighted family goes through the centered family's machinery with
# `CrossWeighted()` as the direction: `_accumulate_one!` gains a method over its own
# accumulating engine, so `div̽ₕ`/`curl̽ₕ` reach `_divergence!`/`_curl!` unchanged, and
# `∇̽ₕ!`/`ε̽ₕ!` share `∇cₕ!`'s and `εcₕ!`'s traversals. Every cross-weighted difference sits
# on the grid point itself, so, as for the centered family, nothing is averaged. Unlike it,
# nothing is truncated either: each end slice takes the one-sided difference its near side
# still defines, exactly as `D̽ₓ` does. The device story is the centered family's too: no
# kernel yet, and the same named error before any scalar indexing.

@inline _direction_spacing(sub, ::CrossWeighted) = spacings(sub)

"""
    ∇̽ₕ(uₕ::VectorElement) -> VectorElement or NTuple{D, VectorElement}

Returns the cross-weighted discrete gradient of the grid function `uₕ`, one cross-weighted
centered difference per direction:

```math
\\overset{\\times}{\\nabla}_h(\\textrm{u}_h) = \\left(
    \\overset{\\times}{\\textrm{D}}_{x_1}(\\textrm{u}_h), \\ldots,
    \\overset{\\times}{\\textrm{D}}_{x_D}(\\textrm{u}_h)\\right), \\qquad
\\overset{\\times}{\\textrm{D}}_{x_d}(\\textrm{u}_h)(i) =
    \\frac{h_i}{h_i + h_{i+1}}\\, \\textrm{D}_{-}\\textrm{u}_h(x_{i+1}) +
    \\frac{h_{i+1}}{h_i + h_{i+1}}\\, \\textrm{D}_{-}\\textrm{u}_h(x_i).
```

The same function as [`D̽ₕ`](@ref): the dispatch alias and the tuple-valued alias coincide
for this family (gpena/Bramble.jl#140), so `∇̽ₕ` is simply another name for it, under the
`∇` notation the other vectorial gradients share. In 1D it returns the bare grid function,
above a `D`-tuple. The first and last point along each direction have no truncated-boundary
convention of their own and fall back to the one-sided difference the near side still
defines, unlike [`∇cₕ`](@ref).

See also: [`D̽ₓ`](@ref), [`∇ₕ`](@ref), [`∇₊ₕ`](@ref)
"""
const ∇̽ₕ = D̽ₕ

# `out[I] += s * D̽(u)[I]` along `DIM`, over the interior and both end slices, each term the
# value `D̽`'s own engine writes: the same `_compute_difference` calls over the same
# `spacings`, only accumulated.
@inline function _accumulate_cross_weighted!(
        out, u, h::H, dims::NTuple{D, Int}, ::Val{DIM}, s
) where {H, D, DIM}
    li = LinearIndices(dims)
    step = _stencil_step(Val(DIM), Val(D))
    interior, lo, hi = _centered_stencil_ranges(axes(li), Val(DIM))
    dir = CrossWeighted()

    @inbounds @simd for I in CartesianIndices(interior)
        idx = li[I]
        out[idx] += s * _compute_difference(
            dir, Val(false), u[li[I - step]], u[idx], u[li[I + step]], h, I[DIM]
        )
    end

    @inbounds @simd for I in CartesianIndices(lo)
        idx = li[I]
        out[idx] += s * _compute_difference(dir, Val(true), u[idx], u[li[I + step]], h, I[DIM])
    end

    @inbounds @simd for I in CartesianIndices(hi)
        idx = li[I]
        out[idx] += s * _compute_difference(dir, Val(true), u[idx], u[li[I - step]], h, I[DIM])
    end
    return nothing
end

@inline function _accumulate_one!(out, u, Ωₕ, dims, ::CrossWeighted, ::Val{d}, s) where {d}
    sub = Ωₕ(d)
    _check_centered_points(sub, d)
    return _accumulate_cross_weighted!(out, u, spacings(sub), dims, Val(d), s)
end

"""
    ∇̽ₕ!(dest, uₕ::VectorElement) -> dest

The in-place form of [`∇̽ₕ`](@ref): the cross-weighted gradient of `uₕ`, written into
`dest` -- a grid function in 1D, a `D`-tuple of them above, the shape `∇̽ₕ` returns.

Allocates nothing. No destination may alias `uₕ`.
"""
function ∇̽ₕ!(dest, uₕ::VectorElement)
    _check_centered_host(dest, "∇̽ₕ!")
    Ωₕ = mesh(space(uₕ))
    D = dim(Ωₕ)
    outs = dest isa VectorElement ? (dest,) : dest
    length(outs) == D || _throw_field_arity(length(outs), D, "∇̽ₕ!")
    _centered_gradient!(outs, parent(uₕ), Ωₕ, npoints(Ωₕ, Tuple), CrossWeighted(), Val(D))
    return dest
end

"""
    div̽ₕ(uₕ) -> VectorElement
    div̽ₕ!(vₕ::VectorElement, uₕ) -> vₕ

Returns the cross-weighted discrete divergence of the vector field `uₕ`:

```math
\\overset{\\times}{\\textrm{div}}_h(\\textrm{u}_h)(I) =
    \\sum_{d=1}^{D} \\overset{\\times}{\\textrm{D}}_{x_d}(\\textrm{u}_{h,d})(I)
```

`uₕ` is spelled as for [`divₕ`](@ref): an `NTuple{D, VectorElement}`, or a grid function of a
[`CompositeGridSpace`](@ref) with one leaf per spatial dimension (in 1D, a scalar grid
function). Each cross-weighted difference is second order on a non-uniform grid and is not
truncated: on the first and last slice of its direction it is the one-sided difference the
near side still defines, as for [`D̽ₓ`](@ref).

`div̽ₕ!` writes into `vₕ`, which must not be one of the components, and allocates nothing.

`∇̽ₕ ⋅ uₕ === div̽ₕ(uₕ)`.

See also: [`divcₕ`](@ref), [`curl̽ₕ`](@ref), [`ε̽ₕ`](@ref), [`∇̽ₕ`](@ref)
"""
function div̽ₕ(uₕ)
    _check_centered_host(first(_field_components(uₕ)), "div̽ₕ")
    return div̽ₕ!(similar(first(_field_components(uₕ))), uₕ)
end

@doc (@doc div̽ₕ)
function div̽ₕ!(vₕ::VectorElement, uₕ)
    _check_centered_host(vₕ, "div̽ₕ!")
    comps = _field_components(uₕ)
    Wₕ = _field_space(uₕ)
    D = dim(mesh(Wₕ))
    _check_field_arity(comps, Val(D), "div̽ₕ")
    _divergence!(vₕ, comps, Wₕ, CrossWeighted(), Val(D))
    return vₕ
end

"""
    curl̽ₕ(uₕ) -> VectorElement or NTuple{3, VectorElement}
    curl̽ₕ!(vₕ, uₕ) -> vₕ

Returns the cross-weighted discrete curl of the vector field `uₕ`. In 2D it is the scalar

```math
\\overset{\\times}{\\textrm{curl}}_h(\\textrm{u}_h) =
    \\overset{\\times}{\\textrm{D}}_{x}(\\textrm{u}_{h,2}) -
    \\overset{\\times}{\\textrm{D}}_{y}(\\textrm{u}_{h,1})
```

and in 3D the three-component field
``(\\partial_y u_3 - \\partial_z u_2,\\; \\partial_z u_1 - \\partial_x u_3,\\;
\\partial_x u_2 - \\partial_y u_1)``, each derivative a cross-weighted centered difference.
There is no 1D curl, and asking for one is an `ArgumentError`.

`curl̽ₕ!` takes a destination -- a grid function in 2D, a 3-tuple of them in 3D -- and
allocates nothing.

`∇̽ₕ × uₕ === curl̽ₕ(uₕ)`.

See also: [`curlcₕ`](@ref), [`div̽ₕ`](@ref), [`∇̽ₕ`](@ref)
"""
function curl̽ₕ(uₕ)
    _check_centered_host(first(_field_components(uₕ)), "curl̽ₕ")
    return _curl(uₕ, CrossWeighted(), "curl̽ₕ")
end

@doc (@doc curl̽ₕ)
function curl̽ₕ!(vₕ, uₕ)
    _check_centered_host(vₕ, "curl̽ₕ!")
    return _curl!(vₕ, uₕ, CrossWeighted(), "curl̽ₕ!")
end

"""
    ε̽ₕ(uₕ) -> NTuple{D, NTuple{D, VectorElement}}
    ε̽ₕ!(dest::NTuple{D, NTuple{D, VectorElement}}, uₕ) -> dest

Returns the cross-weighted discrete strain tensor of the vector field `uₕ`, entry by entry

```math
\\overset{\\times}{\\varepsilon}^{ii}_h(\\textrm{u}_h) =
    \\overset{\\times}{\\textrm{D}}_{x_i}(\\textrm{u}_{h,i}), \\qquad
\\overset{\\times}{\\varepsilon}^{ij}_h(\\textrm{u}_h) = \\tfrac{1}{2}\\left(
    \\overset{\\times}{\\textrm{D}}_{x_j}(\\textrm{u}_{h,i}) +
    \\overset{\\times}{\\textrm{D}}_{x_i}(\\textrm{u}_{h,j})\\right), \\quad i \\neq j.
```

Every cross-weighted difference sits on the grid point itself, so, as for [`εcₕ`](@ref), no
average relocates the shear terms. `uₕ` is spelled as for [`div̽ₕ`](@ref). The result is
symmetric by construction: `dest[i][j]` and `dest[j][i]` hold the same values.

`ε̽ₕ!` writes into a preallocated `D`-by-`D` nested tuple and allocates nothing.

See also: [`εcₕ`](@ref), [`∇̽ₕ`](@ref), [`div̽ₕ`](@ref)
"""
function ε̽ₕ(uₕ)
    comps = _field_components(uₕ)
    _check_centered_host(first(comps), "ε̽ₕ")
    D = dim(mesh(_field_space(uₕ)))
    _check_field_arity(comps, Val(D), "ε̽ₕ")
    return ε̽ₕ!(_strain_alloc(comps, Val(D)), uₕ)
end

@doc (@doc ε̽ₕ)
function ε̽ₕ!(dest::NTuple{D, NTuple{D, VectorElement}}, uₕ) where {D}
    _check_centered_host(dest[1], "ε̽ₕ!")
    comps = _field_components(uₕ)
    Ωₕ = mesh(_field_space(uₕ))
    dim(Ωₕ) == D || throw(DimensionMismatch("ε̽ₕ! destination is $(D)x$D but the mesh is $(dim(Ωₕ))D"))
    _check_field_arity(comps, Val(D), "ε̽ₕ!")
    _centered_strain_rows!(dest, comps, Ωₕ, npoints(Ωₕ, Tuple), CrossWeighted(), Val(D), Val(D))
    return dest
end

# --- ⋅ and × contract a gradient alias to its divergence/curl (gpena/Bramble.jl#341) ------- #
#
# `∇ₕ ⋅ uₕ` / `∇ₕ × uₕ` read as the textbook notation for `divₕ(uₕ)` / `curlₕ(uₕ)`, and the
# same for the other four gradient aliases (`∇cₕ`/`∇̽ₕ`/`∇̃ₕ` are `const` aliases of
# `Dcₕ`/`D̽ₕ`/`D̃ₕ`, operators/difference.jl, so the methods below dispatch on those function
# types directly). `dot`/`×` are already `import`ed from LinearAlgebra by `src/Bramble.jl`.
#
# Each alias now supports `iterate`/`getindex` (gpena/Bramble.jl#340), so LinearAlgebra's
# generic `dot`/`cross` fallback would otherwise try to treat `∇ₕ` as a 3-element collection
# and zip it against `uₕ`. Dispatch on `typeof(alias)` -- a concrete singleton function type
# -- is strictly more specific than that generic `(x, y)` fallback, so these methods win
# outright and add no ambiguity: `Test.detect_ambiguities(Bramble)` is unchanged (verified in
# EVIDENCE), and neither overlaps `dot(F::NTuple, ::NormalSymbol)` (src/form/operators/normal.jl)
# or the domain/space `×(::AbstractSpaceType, ::AbstractSpaceType)` (first-argument types never
# coincide). `∇ₕ ⋅ n` and `∇ₕ × n` therefore hit no method and raise the ordinary `MethodError`.
@inline dot(::typeof(∇ₕ), uₕ) = divₕ(uₕ)
@inline ×(::typeof(∇ₕ), uₕ) = curlₕ(uₕ)

@inline dot(::typeof(∇₊ₕ), uₕ) = div₊ₕ(uₕ)
@inline ×(::typeof(∇₊ₕ), uₕ) = curl₊ₕ(uₕ)

@inline dot(::typeof(∇cₕ), uₕ) = divcₕ(uₕ)
@inline ×(::typeof(∇cₕ), uₕ) = curlcₕ(uₕ)

@inline dot(::typeof(∇̃ₕ), uₕ) = diṽₕ(uₕ)
@inline ×(::typeof(∇̃ₕ), uₕ) = curl̃ₕ(uₕ)

@inline dot(::typeof(∇̽ₕ), uₕ) = div̽ₕ(uₕ)
@inline ×(::typeof(∇̽ₕ), uₕ) = curl̽ₕ(uₕ)
