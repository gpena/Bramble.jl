################################################################################
#                      Discrete L² Inner Product and Norm                      #
################################################################################

"""
	innerₕ(uₕ::VectorElement, vₕ::VectorElement)

Returns the discrete ``L^2`` inner product of the grid functions `uₕ` and `vₕ`. Also accepts [MatrixElement](@ref) as any of the arguments.

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
"""
@inline innerₕ(uₕ::VectorElement, vₕ::VectorElement) = _dot(uₕ.data, weights(space(uₕ), Innerh()), vₕ.data)
@inline innerₕ(Uₕ::VecOrMatElem, Vₕ::VecOrMatElem) = _inner_product(Uₕ.data, weights(space(Uₕ), Innerh()), Vₕ.data)

"""
	normₕ(uₕ::VectorElement)

Returns the discrete ``L^2`` norm of the grid function `uₕ`, defined as

```math
\\Vert \\textrm{u}_h \\Vert_h \\vcentcolon = \\sqrt{(\\textrm{u}_h, \\textrm{u}_h)_h}
```
"""
@inline normₕ(uₕ::VectorElement) = sqrt(innerₕ(uₕ, uₕ))

################################################################################
#                 Discrete Modified L² Inner Product and Norm                  #
################################################################################

@inline function _directional_inner_plus(uₕ::VecOrMatElem, vₕ::VecOrMatElem, _::Val{DIM}) where DIM
	return _inner_product(uₕ.data, weights(space(uₕ), Innerplus(), DIM), vₕ.data)
end

"""
	inner₊ₓ(uₕ::VecOrMatElem, vₕ::VecOrMatElem)

Returns the discrete modified ``L^2`` inner product of the grid functions `uₕ` and `vₕ` associated with the first variable. It accepts arguments of type [VectorElement](@ref) or [MatrixElement](@ref), in any order.

For [VectorElement](@ref)s, it is defined as

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
"""
@inline inner₊ₓ(uₕ::VecOrMatElem, vₕ::VecOrMatElem) = _directional_inner_plus(uₕ, vₕ, Val(1))

"""
	inner₊ᵧ(uₕ::VecOrMatElem, vₕ::VecOrMatElem)

Returns the discrete modified ``L^2`` inner product of the grid functions `uₕ` and `vₕ`
associated with the second variable (y-direction). It accepts arguments of type
[VectorElement](@ref) or [MatrixElement](@ref), in any order.

For [VectorElement](@ref)s, it is defined as

  - 2D case

```math
(\\textrm{u}_h, \\textrm{v}_h)_{+y} \\vcentcolon = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y}   h_{x,i} h_{y,j+1/2}   \\textrm{u}_h(x_i,y_j) \\textrm{v}_h(x_i,y_j)
```

  - 3D case

```math
(\\textrm{u}_h, \\textrm{v}_h)_{+y} \\vcentcolon = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y}\\sum_{l=1}^{N_z}   h_{x,i+1/2} h_{y,j} h_{z,l+1/2} \\textrm{u}_h(x_i,y_j,z_l) \\textrm{v}_h(x_i,y_j,z_l).
```
"""
@inline inner₊ᵧ(uₕ::VecOrMatElem, vₕ::VecOrMatElem) = _directional_inner_plus(uₕ, vₕ, Val(2))

"""
	inner₊₂(uₕ::VecOrMatElem, vₕ::VecOrMatElem)

Returns the discrete modified ``L^2`` inner product of the grid functions `uₕ` and `vₕ` associated with the `z` variable

```math
(\\textrm{u}_h, \\textrm{v}_h)_{+z} \\vcentcolon = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y}\\sum_{l=1}^{N_z}  h_{x,i+1/2} h_{y,j+1/2} h_{z,l} \\textrm{u}_h(x_i,y_j,z_l) \\textrm{v}_h(x_i,y_j,z_l).
```
"""
@inline inner₊₂(uₕ::VecOrMatElem, vₕ::VecOrMatElem) = _directional_inner_plus(uₕ, vₕ, Val(3))

get_dimension_from_type(::Type{<:NTuple{D,Any}}) where D = D
get_dimension_from_type(::Type{<:VecOrMatElem{S}}) where S = dim(mesh_type(S))
get_dimension_from_type(::Type) = nothing

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
		dim_u == dim_v ? dim_u : return :(throw(DimensionMismatch("Dimensions $dim_u and $dim_v do not match")))
	elseif !isnothing(dim_u)
		dim_u
	elseif !isnothing(dim_v)
		dim_v
	else
		return :(throw(ArgumentError("Could not determine dimension from input types $u_type and $v_type")))
	end

	# Direction count for the underlying space (fallback to 1 if unknown).
	# For tuple inputs we want the *spatial* dimension of the element type, not
	# the tuple arity (which can exceed the mesh dimension in mixed terms such as
	# `(Dx*u, Mx*u)` on 1D meshes).
	u_elem_dim = u_is_tuple ? get_dimension_from_type(u_type.parameters[2]) : nothing
	v_elem_dim = v_is_tuple ? get_dimension_from_type(v_type.parameters[2]) : nothing
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
	inner₊(uₕ::VecOrMatElem, vₕ::VecOrMatElem, [::Type{Tuple}])
	inner₊(uₕ::NTuple{D}, vₕ::NTuple{D})

Returns the discrete modified ``L^2`` inner product of the grid functions `uₕ` and `vₕ`. It accepts arguments of type [VectorElement](@ref) or [MatrixElement](@ref), in any order.

If the `Tuple` argument is given, it returns `D`-tuple of all ``\\textrm{inner}_{x_i,+}`` applied to its input arguments, where `D` is the topological dimension of the mesh associated with the elements.

If `NTuple`s of [VectorElement](@ref) or [MatrixElement](@ref) are passed as input arguments, it returns the sum of all inner products ``(\\textrm{u}_h[i],\\textrm{v}_h[i])_{+x_i}``.

For [VectorElement](@ref)s, the definition is given by

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

See the definitions of [inner₊ₓ](@ref inner₊ₓ(uₕ::VecOrMatElem, vₕ::VecOrMatElem)), [inner₊ᵧ](@ref inner₊ᵧ(uₕ::VecOrMatElem, vₕ::VecOrMatElem)) and [inner₊₂](@ref inner₊₂(uₕ::VecOrMatElem, vₕ::VecOrMatElem)) for more details.
"""
@generated inner₊(uₕ, vₕ) = :($(_generate_inner_plus_body(uₕ, vₕ, :sum)))
@generated inner₊(uₕ, vₕ, ::Type{Tuple}) = :($(_generate_inner_plus_body(uₕ, vₕ, :tuple)))

"""
	norm₊(uₕ::VectorElement)
	norm₊(uₕ::NTuple{D,VectorElement})

Returns the discrete modified ``L^2`` norm of the grid function `uₕ`. It also accepts a `NTuple` of [VectorElement](@ref)s.

For [VectorElement](@ref)s `uₕ`, it is defined as

```math
\\Vert \\textrm{u}_h \\Vert_+ = \\sqrt{(\\textrm{u}_h,\\textrm{u}_h)_+}.
```

and for `NTuple`s of [VectorElement](@ref)s it returns

```math
\\Vert \\textrm{u}_h \\Vert_+ \\vcentcolon = \\sqrt{ \\sum_{i=1}^D(\\textrm{u}_h[i],\\textrm{u}_h[i])_{+,x_i}}.
```
"""
@inline norm₊(uₕ::Union{VectorElement,NTuple{<:Any,VectorElement}}) = sqrt(inner₊(uₕ, uₕ))

################################################################################
#                        Discrete H¹ Norm and Seminorm                         #
################################################################################
"""
	snorm₁ₕ(uₕ::VectorElement)

Returns the discrete version of the standard ``H^1`` seminorm of [VectorElement](@ref) `uₕ`.

```math
|\\textrm{u}_h|_{1h} \\vcentcolon = \\Vert \\nabla_h \\textrm{u}_h \\Vert_h
```
"""
@muladd function snorm₁ₕ(uₕ::VectorElement)
	@unpack data, space = uₕ
	Ωₕ = mesh(space)
	dims = ndofs(space, Tuple)
	D = dim(Ωₕ)

	total_seminorm_sq = 0.0
	li = LinearIndices(dims)

	@fastmath @inbounds @simd for I in CartesianIndices(dims)
		local_seminorm_sq_at_I = 0.0

		for d in 1:D
			val_at_I = data[li[I]]
			h = Base.Fix1(spacing, Ωₕ(d))

			diff_val = if I[d] > 1
				step_cartesian = CartesianIndex(ntuple(i -> i == d ? 1 : 0, D))
				val_at_prev = data[li[I - step_cartesian]]
				_compute_difference(Backward(), Val(false), val_at_I, val_at_prev, h, I[d])
			else
				_compute_difference(Backward(), Val(true), val_at_I, h, I[d])
			end

			weight_d = weights(space, Innerplus(), d)[li[I]]

			local_seminorm_sq_at_I += weight_d * diff_val^2
		end
		total_seminorm_sq += local_seminorm_sq_at_I
	end

	# returns norm₊(∇₋ₕ(uₕ))
	return sqrt(total_seminorm_sq)
end

"""
	norm₁ₕ(uₕ::VectorElement)

Returns the discrete version of the standard ``H^1`` norm of [VectorElement](@ref) `uₕ`.

```math
\\Vert \\textrm{u}_h \\Vert_{1h} \\vcentcolon = \\sqrt{ \\Vert \\textrm{u}_h \\Vert_h^2 +  \\Vert \\nabla_h \\textrm{u}_h \\Vert_h^2   }
```
"""
@inline norm₁ₕ(uₕ::VectorElement) = sqrt(normₕ(uₕ)^2 + snorm₁ₕ(uₕ)^2)