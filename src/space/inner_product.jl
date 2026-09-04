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
tuple with one grid function per direction: the gradient ∇₋ₕ(uₕ) is exactly that
shape. In one dimension a one-tuple and the grid function coincide, so the scalar
form is accepted there.

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
mesh-independent. See `inner_Γ` below and point 11 of `docs/form-unlock-plan.md`.
=#

@inline function _combined_mask(Ωₕ, markers::NTuple{1, Symbol})
    return index_in_marker(Ωₕ, markers[1])
end

# `N > 1`: the stored per-marker masks are combined by union into a fresh `BitVector`, since
# a sum (unlike `Rₕ!`'s per-marker write) must not double-count a point two markers both
# cover. `N == 1` above skips this allocation, since the mesh's own stored mask can be read
# directly without being unioned with anything.
function _combined_mask(Ωₕ, markers::NTuple{N, Symbol}) where {N}
    mask = copy(index_in_marker(Ωₕ, markers[1]))
    for k in 2:N
        mask .|= index_in_marker(Ωₕ, markers[k])
    end
    return mask
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
"""
@inline function innerₕ(uₕ::VectorElement{<:ScalarGridSpace},
        vₕ::VectorElement{<:ScalarGridSpace};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {N}
    N == 0 && return _dot(uₕ.data, weights(space(uₕ), Innerh()), vₕ.data)
    mask = _combined_mask(mesh(space(uₕ)), markers)
    return _dot_masked(uₕ.data, weights(space(uₕ), Innerh()), vₕ.data, mask)
end

# Summed over the components, unrolled because `NC` is a type parameter, so this costs no
# more than writing the sum out by hand. `markers` threads through unchanged: each component
# is restricted to the same regions.
@inline function innerₕ(uₕ::VectorElement{<:CompositeGridSpace{NC}},
        vₕ::VectorElement{<:CompositeGridSpace{NC}};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {NC, N}
    uc, vc = components(uₕ), components(vₕ)
    return sum(ntuple(c -> innerₕ(uc[c], vc[c]; markers = markers), Val(NC)))
end

@noinline function innerₕ(uₕ::VectorElement{<:CompositeGridSpace{N}},
        vₕ::VectorElement{<:CompositeGridSpace{M}}) where {N, M}
    throw(DimensionMismatch(
        "innerₕ needs the same number of components on both sides; got $N and $M"))
end

"""
    inner_Γ(uₕ::VectorElement, vₕ::VectorElement, labels::Symbol...)

Placeholder for the true, ``(D-1)``-dimensional boundary integral ``\\int_\\Gamma u\\,v\\,ds``
over the mesh regions `labels` name.

**Not the same quantity as `innerₕ(uₕ, vₕ; markers = labels)`.** That is a masked sum of the
existing cell measures (a ``D``-dimensional quantity restricted to a set of points), and it
scales like `h`, vanishing under refinement: 0.125 on a 5×5 mesh of the unit square restricted
to `:bottom`. This function is meant for the mesh-independent surface integral a Neumann or
Robin term needs (1.0 on that same mesh and region), which is a genuinely different quantity,
not a bug in the other one. See point 11 of `docs/form-unlock-plan.md`.

**Not yet implemented.** No ``(D-1)``-dimensional surface quadrature weight exists anywhere in
the package today: only `Innerh` (cell measures) and `Innerplus` (directional, still
``D``-dimensional). This always throws until that weight is built; kept unexported until it
does something.
"""
function inner_Γ(uₕ::VectorElement, vₕ::VectorElement, labels::Symbol...)
    throw(ErrorException(
        "inner_Γ (the true boundary integral) is not yet implemented: no (D-1)-dimensional " *
        "surface quadrature weight exists in the package. innerₕ(uₕ, vₕ; markers = labels) " *
        "computes a related but NOT equivalent quantity: a masked sum of cell measures that " *
        "scales like h and vanishes under refinement, not a mesh-independent surface integral."))
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

################################################################################
#                 Discrete Modified L² Inner Product and Norm                  #
################################################################################

@inline function _directional_inner_plus(uₕ::VectorElement{<:ScalarGridSpace},
        vₕ::VectorElement{<:ScalarGridSpace}, _::Val{DIM};
        markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {DIM, N}
    N == 0 &&
        return _dot(uₕ.data, weights(space(uₕ), Innerplus(), DIM), vₕ.data)
    mask = _combined_mask(mesh(space(uₕ)), markers)
    return _dot_masked(
        uₕ.data, weights(space(uₕ), Innerplus(), DIM), vₕ.data, mask)
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
@inline inner₊ₓ(uₕ::VectorElement{<:ScalarGridSpace},
    vₕ::VectorElement{<:ScalarGridSpace}; markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {N} = _directional_inner_plus(
    uₕ, vₕ, Val(1); markers = markers)

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
@inline inner₊ᵧ(uₕ::VectorElement{<:ScalarGridSpace},
    vₕ::VectorElement{<:ScalarGridSpace}; markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {N} = _directional_inner_plus(
    uₕ, vₕ, Val(2); markers = markers)

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
@inline inner₊₂(uₕ::VectorElement{<:ScalarGridSpace},
    vₕ::VectorElement{<:ScalarGridSpace}; markers::NTuple{N, Symbol} = NTuple{0, Symbol}()) where {N} = _directional_inner_plus(
    uₕ, vₕ, Val(3); markers = markers)

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
        dim_u == dim_v ? dim_u :
        # The message is built here and spliced in as a String. Writing the
        # interpolation inside the quoted string would defer it to run time,
        # where dim_u and dim_v do not exist, and the call would raise
        # UndefVarError instead of the intended error.
        return :(throw(DimensionMismatch($("Dimensions $dim_u and $dim_v do not match"))))
    elseif !isnothing(dim_u)
        dim_u
    elseif !isnothing(dim_v)
        dim_v
    else
        return :(throw(ArgumentError($("Could not determine dimension from input types $u_type and $v_type"))))
    end

    # Direction count for the underlying space (fallback to 1 if unknown).
    # For tuple inputs we want the *spatial* dimension of the element type, not
    # the tuple arity (which can exceed the mesh dimension in mixed terms such as
    # `(Dx*u, Mx*u)` on 1D meshes).
    u_elem_dim = u_is_tuple ? _tuple_element_dim(u_type) : nothing
    v_elem_dim = v_is_tuple ? _tuple_element_dim(v_type) : nothing
    mesh_dim = something(u_elem_dim,
        v_elem_dim,
        (!u_is_tuple && !isnothing(dim_u)) ? dim_u : nothing,
        (!v_is_tuple && !isnothing(dim_v)) ? dim_v : nothing,
        1)

    terms = map(1:D) do i
        u_component = u_is_tuple ? :(uₕ[$i]) : :uₕ
        v_component = v_is_tuple ? :(vₕ[$i]) : :vₕ
        dir = min(i, mesh_dim) # avoid out-of-bounds when tuples are longer than spatial dim
        :(_directional_inner_plus($u_component, $v_component, Val($dir)))
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
# The squared seminorm along one direction. `d` arrives as a `Val` so the stencil step
# is built at compile time, and the spacing, weight and step are read once per direction
# rather than once per grid point.
#
# The boundary slice contributes nothing: the backward difference is truncated to zero
# there, so its square is zero. Only the interior is walked.
@inline function _seminorm_sq_along(data, space, Ωₕ, li, ::Val{d}, ::Val{D}) where {d, D}
    h = backward_spacings_for_derivative(Ωₕ(d))
    w = weights(space, Innerplus(), d)
    step = _stencil_step(Val(d), Val(D))
    interior, _ = _stencil_ranges(axes(li), Val(d), Backward())

    s = zero(eltype(data))
    @inbounds @simd for I in CartesianIndices(interior)
        idx = li[I]
        δ = (data[idx] - data[li[I - step]]) / h[I[d]]
        s = muladd(w[idx], δ * δ, s)
    end

    return s
end

# `D` is taken from the space's type parameter rather than from `dim(Ωₕ)`: building a
# `Val` out of a value returned at run time is a dynamic dispatch, which cost 224 bytes
# per call on a 2D grid.
# Summed by recursion on `Val(d)` rather than through `ntuple`: the closure `ntuple`
# needs captures four locals, and capturing them cost a small allocation per call.
@inline _sum_dirs(data, space, Ωₕ, li, ::Val{0}, ::Val{D}) where {D} = zero(eltype(data))
@inline _sum_dirs(
    data, space, Ωₕ, li, ::Val{d}, ::Val{D}) where {d, D} = _seminorm_sq_along(
    data, space, Ωₕ, li, Val(d), Val(D)) +
                                                            _sum_dirs(
    data, space, Ωₕ, li, Val(d - 1), Val(D))

@inline function _snorm₁ₕ_sq(uₕ::VectorElement{<:ScalarGridSpace{D}}) where {D}
    (; data, space) = uₕ
    Ωₕ = mesh(space)
    li = LinearIndices(npoints(Ωₕ, Tuple))
    return _sum_dirs(data, space, Ωₕ, li, Val(D), Val(D))
end

"""
    snorm₁ₕ(uₕ::VectorElement) -> Real

Returns the discrete ``H^1`` seminorm of the grid function `uₕ`,

```math
|\\textrm{u}_h|_{1h} \\vcentcolon = \\Vert \\nabla_h \\textrm{u}_h \\Vert_+
```

so that `snorm₁ₕ(uₕ) == norm₊(∇₋ₕ(uₕ))` in one, two and three dimensions. The argument is
the grid function itself; the backward gradient is taken internally, and without
materialising it, so this allocates nothing.

See also: [`norm₁ₕ`](@ref), [`norm₊`](@ref), [`∇₋ₕ`](@ref).

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
@inline norm₁ₕ(uₕ::VectorElement{<:ScalarGridSpace}) = sqrt(
    innerₕ(uₕ, uₕ) + _snorm₁ₕ_sq(uₕ))
