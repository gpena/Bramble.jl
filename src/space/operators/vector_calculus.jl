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

# One direction of the conservative Laplacian, fused: `D̽(D₋(u))` without the intermediate
# grid function. Reading the composition off the two engines rather than re-deriving it is
# what keeps the boundary conventions identical -- `D₋` is zero on the first slice, so the
# term it would contribute is simply absent there, and `D̽` is zero on the last, so nothing
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
    fill!(parent(vₕ), zero(eltype(parent(vₕ))))
    _accumulate_direction!(vₕ, comps, Ωₕ, dims, dir, Val(D), Val(D))
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

See also: [`divₕ`](@ref), [`∇ₕ`](@ref)
"""
@inline curlₕ(uₕ) = _curl(uₕ, Backward())

"""
    curl₊ₕ(uₕ) -> VectorElement or NTuple{3, VectorElement}
    curl₊ₕ!(vₕ, uₕ) -> vₕ

The forward-difference discrete curl, the twin of [`curlₕ`](@ref).
"""
@inline curl₊ₕ(uₕ) = _curl(uₕ, Forward())

@inline function _curl(uₕ, dir)
    comps = _field_components(uₕ)
    D = dim(mesh(_field_space(uₕ)))
    _check_field_arity(comps, Val(D), "curlₕ")
    return _curl_alloc(uₕ, comps, dir, Val(D))
end

@inline _curl_alloc(uₕ, comps, dir, ::Val{2}) = _curl!(similar(first(comps)), uₕ, dir)
@inline _curl_alloc(uₕ, comps, dir, ::Val{3}) = _curl!(
    ntuple(_ -> similar(first(comps)), Val(3)), uₕ, dir
)
@noinline _curl_alloc(uₕ, comps, dir, ::Val{D}) where {D} = throw(
    ArgumentError("the discrete curl is defined in 2D and 3D; this mesh is $(D)D.")
)

@doc (@doc curlₕ)
@inline curlₕ!(vₕ, uₕ) = _curl!(vₕ, uₕ, Backward())

@doc (@doc curl₊ₕ)
@inline curl₊ₕ!(vₕ, uₕ) = _curl!(vₕ, uₕ, Forward())

function _curl!(vₕ::VectorElement, uₕ, dir)
    comps = _field_components(uₕ)
    Ωₕ = mesh(_field_space(uₕ))
    dims = npoints(Ωₕ, Tuple)
    _check_field_arity(comps, Val(2), "curlₕ")

    out = parent(vₕ)
    fill!(out, zero(eltype(out)))
    _accumulate_one!(out, parent(comps[2]), Ωₕ, dims, dir, Val(1), true)
    _accumulate_one!(out, parent(comps[1]), Ωₕ, dims, dir, Val(2), -true)
    return vₕ
end

function _curl!(vₕ::NTuple{3, VectorElement}, uₕ, dir)
    comps = _field_components(uₕ)
    Ωₕ = mesh(_field_space(uₕ))
    dims = npoints(Ωₕ, Tuple)
    _check_field_arity(comps, Val(3), "curlₕ")

    # (∂₂u₃ - ∂₃u₂, ∂₃u₁ - ∂₁u₃, ∂₁u₂ - ∂₂u₁), written out rather than looped over cyclic
    # index triples: the directions have to stay compile-time literals.
    for k in 1:3
        fill!(parent(vₕ[k]), zero(eltype(parent(vₕ[k]))))
    end
    _accumulate_one!(parent(vₕ[1]), parent(comps[3]), Ωₕ, dims, dir, Val(2), true)
    _accumulate_one!(parent(vₕ[1]), parent(comps[2]), Ωₕ, dims, dir, Val(3), -true)
    _accumulate_one!(parent(vₕ[2]), parent(comps[1]), Ωₕ, dims, dir, Val(3), true)
    _accumulate_one!(parent(vₕ[2]), parent(comps[3]), Ωₕ, dims, dir, Val(1), -true)
    _accumulate_one!(parent(vₕ[3]), parent(comps[2]), Ωₕ, dims, dir, Val(1), true)
    _accumulate_one!(parent(vₕ[3]), parent(comps[1]), Ωₕ, dims, dir, Val(2), -true)
    return vₕ
end

# --- Laplacian -------------------------------------------------------------------------- #

"""
    Δₕ(uₕ::VectorElement) -> VectorElement
    Δₕ!(vₕ::VectorElement, uₕ::VectorElement) -> vₕ

Returns the conservative discrete Laplacian of the grid function `uₕ`, matrix-free:

```math
\\Delta_h(\\textrm{u}_h)(I) = \\sum_{d=1}^{D}
    \\overset{\\times}{\\textrm{D}}_{+,x_d}\\big(\\textrm{D}_{-,x_d}(\\textrm{u}_h)\\big)(I)
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
`D̽ₓ(D₋ₓ(uₕ))` entry for entry, truncation at the two ends of each axis included.

See also: [`divₕ`](@ref), [`∇ₕ`](@ref), [`D̽ₓ`](@ref)
"""
@inline Δₕ(uₕ::VectorElement) = Δₕ!(similar(uₕ), uₕ)

@doc (@doc Δₕ)
function Δₕ!(vₕ::VectorElement, uₕ::VectorElement)
    _check_no_alias(vₕ, uₕ)
    Ωₕ = mesh(space(uₕ))
    dims = npoints(Ωₕ, Tuple)
    out = parent(vₕ)
    fill!(out, zero(eltype(out)))
    _laplacian_direction!(out, parent(uₕ), Ωₕ, dims, Val(dim(Ωₕ)), Val(dim(Ωₕ)))
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
