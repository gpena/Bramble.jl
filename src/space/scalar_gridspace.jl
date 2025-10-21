struct SpaceWeights{D,VT<:AbstractVector} <: BrambleType
	innerh::VT
	innerplus::NTuple{D,VT}
end

"""
	mutable struct ScalarGridSpace{D,T,VT<:AbstractVector{T},MT<:AbstractMatrix{T},MType<:AbstractMeshType{D},BackendType<:Backend{VT,MT}} <: AbstractSpaceType{1}
	const mesh::MType
	const weights::SpaceWeights{D,VT}
	const backward_difference_matrix::NTuple{D,MT}
	has_backward_difference_matrix::Bool
	const average_matrix::NTuple{D,MT}
	has_average_matrix::Bool
	const vector_buffer::GridSpaceBuffer{BackendType,VT,T}

end

Structure for a gridspace defined on a mesh.

The dictionary `weights` has the weights for several the standard discrete ``L^2``
inner product on the space of grid functions.

They are defined as follows:

  - weights[:innerₕ]

	  + 1D case

```math
(u_h, v_h)_h = \\sum_{i=1}^{N_x} |\\square_{i}| u_h(x_i) v_h(x_i)
```

	+ 2D case

```math
(u_h, v_h)_h = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y} |\\square_{i,j}| u_h(x_i,y_j) v_h(x_i,y_j)
```

	+ 3D case

```math
(u_h, v_h)_h = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y}\\sum_{l=1}^{N_z} |\\square_{i,j,l}| u_h(x_i,y_j,z_l) v_h(x_i,y_j,z_l).
```

Here, ``|\\cdot|`` denotes the measure of the set and all details on the definition of ``\\square_{i}``, ``\\square_{i,j}`` and ``\\square_{i,j,l}`` can be found in functions [cell_measure](@ref cell_measure(Ωₕ::Mesh1D, i)) (for the `1`-dimensional case) and [cell_measure](@ref cell_measure(Ωₕ::MeshnD, i)) (for the `n`-dimensional cases).

  - weights[:inner₊], weights[:inner₊ₓ], weights[:inner₊ᵧ], weights[:inner₊₂]
	These are the weights for the modified discrete ``L^2`` inner product on the space of grid functions, for each component (x, y, z).

		+ 1D case

```math
(u_h, v_h)_+ = \\sum_{i=1}^{N_x} h_{i} u_h(x_i) v_h(x_i)
```

	+ 2D case

```math
(u_h, v_h)_{+x} = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y} h_{x,i} h_{y,j+1/2} u_h(x_i,y_j) v_h(x_i,y_j)
```

```math
(u_h, v_h)_{+y} = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y} h_{x,i} h_{y,j+1/2} u_h(x_i,y_j) v_h(x_i,y_j)
```

	+ 3D case

```math
(u_h, v_h)_{+x} = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y}\\sum_{l=1}^{N_z} h_{x,i} h_{y,j+1/2} h_{z,l+1/2} u_h(x_i,y_j,z_l) v_h(x_i,y_j,z_l)
```

```math
(u_h, v_h)_{+y} = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y}\\sum_{l=1}^{N_z} h_{x,i+1/2} h_{y,j} h_{z,l+1/2} u_h(x_i,y_j,z_l) v_h(x_i,y_j,z_l)
```

```math
(u_h, v_h)_{+z} = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y}\\sum_{l=1}^{N_z} h_{x,i+1/2} h_{y,j+1/2} h_{z,l} u_h(x_i,y_j,z_l) v_h(x_i,y_j,z_l).
```
"""
mutable struct ScalarGridSpace{D,T,                               # Dimension and Element Type
							   VT<:AbstractVector{T},             # Vector Type
							   MT<:AbstractMatrix{T},             # Matrix Type
							   MType<:AbstractMeshType{D},
							   BackendType<:Backend{VT,MT}} <: AbstractSpaceType{1}
	const mesh::MType
	const weights::SpaceWeights{D,VT}
	const backward_difference_matrix::NTuple{D,MT}
	has_backward_difference_matrix::Bool
	const average_matrix::NTuple{D,MT}
	has_average_matrix::Bool
	const vector_buffer::GridSpaceBuffer{BackendType,VT,T}
end

ncomponents(::Type{<:ScalarGridSpace}) = 1

ComponentStyle(::Type) = SingleComponent()
ComponentStyle(::Type{<:ScalarGridSpace}) = SingleComponent()

@inline __vector(Ωₕ) = vector(backend(Ωₕ), npoints(Ωₕ))

function space_weights(Ωₕ::AbstractMeshType{D}) where D
	innerplus = ntuple(i -> __vector(Ωₕ), Val(D))
	innerplus_per_component = ntuple(j -> __vector(Ωₕ(j)), Val(D))

	npts_tuple = npoints(Ωₕ, Tuple)
	for i in 1:D
		for k in 1:D
			if k == i
				_innerplus_weights!(innerplus_per_component[k], Ωₕ, k)
			else
				_innerplus_mean_weights!(innerplus_per_component[k], Ωₕ, k)
			end
		end

		v = Base.ReshapedArray(innerplus[i], npts_tuple, ())
		__innerplus_weights!(v, innerplus_per_component)
	end

	inner_h_vec = __vector(Ωₕ)
	_innerh_weights!(inner_h_vec, Ωₕ)

	return SpaceWeights{D,typeof(inner_h_vec)}(inner_h_vec, innerplus)
end

function backward_difference_matrices(Ωₕ::AbstractMeshType{D}) where D
	vector = __vector(Ωₕ)

	return ntuple(i -> backward_finite_difference(Ωₕ, Val(i), vector_cache = vector), Val(D))
end

@inline average_matrices(Ωₕ::AbstractMeshType{D}) where D = ntuple(i -> _create_average_matrix(Ωₕ, i), Val(D))
@inline empty_matrix_cache(Ωₕ::AbstractMeshType{D}) where D = ntuple(i -> shift(Ωₕ, Val(1), Val(0)), Val(D))

"""
	gridspace(Ωₕ::ScalarGridSpace; cache_avg = false, cache_bwd = true)

Constructor for a [ScalarGridSpace](@ref) defined on the mesh `Ωₕ`. This builds the weights for the inner products mentioned in [ScalarGridSpace](@ref) as well as the differentiation matrices associated with the grid points of mesh `Ωₕ`.
The keyword arguments `cache_avg` and `cache_bwd` can be used to indicate if the average and backward difference matrices should be precomputed and stored in the space (default is `true` for `cache_bwd` and `false` for `cache_avg`).
"""
function gridspace(Ωₕ::AbstractMeshType{D}; cache_avg = false, cache_bwd = true) where D
	b = backend(Ωₕ)
	npts = npoints(Ωₕ)

	weights = space_weights(Ωₕ)
	diff_matrices = cache_bwd ? backward_difference_matrices(Ωₕ) : empty_matrix_cache(Ωₕ)
	avg_matrices = cache_avg ? average_matrices(Ωₕ) : empty_matrix_cache(Ωₕ)

	space_buffer = create_simple_space_buffer(b, npts, nbuffers = 1)

	MType = typeof(Ωₕ)
	T, VT, MT, BackendType = backend_types(b)

	return ScalarGridSpace{D,T,VT,MT,MType,BackendType}(Ωₕ, weights, diff_matrices, cache_bwd, avg_matrices, cache_avg, space_buffer)
end

# Accessors for GridSpaces
@inline mesh(Wₕ::ScalarGridSpace) = Wₕ.mesh
@inline backward_difference_matrix(Wₕ::ScalarGridSpace, i) = Wₕ.backward_difference_matrix[i]
@inline average_matrix(Wₕ::ScalarGridSpace, i) = Wₕ.average_matrix[i]
@inline vector_buffer(Wₕ::ScalarGridSpace) = Wₕ.vector_buffer
@inline has_backward_difference_matrix(Wₕ::ScalarGridSpace) = Wₕ.has_backward_difference_matrix
@inline has_average_matrix(Wₕ::ScalarGridSpace) = Wₕ.has_average_matrix
@inline backend(Wₕ::ScalarGridSpace) = backend(mesh(Wₕ))
@inline mesh_type(Wₕ::ScalarGridSpace) = typeof(mesh(Wₕ))
@inline mesh_type(::Type{<:ScalarGridSpace{D,T,VT,MT,MType}}) where {D,T,VT,MT,MType} = MType

"""
	weights(Wₕ::ScalarGridSpace, [::InnerProductType], [i])

Returns the weights associated with the functionspace. A second argument can be supplied detailing the type of weights.
"""
@inline weights(Wₕ::ScalarGridSpace) = Wₕ.weights
@inline weights(Wₕ::ScalarGridSpace, ::Innerh) = weights(Wₕ).innerh
@inline weights(Wₕ::ScalarGridSpace, ::Innerplus) = weights(Wₕ).innerplus
@inline weights(Wₕ::ScalarGridSpace, ::Innerh, i) = weights(Wₕ, Innerh())
@inline weights(Wₕ::ScalarGridSpace, ::Innerplus, i) = weights(Wₕ, Innerplus())[i]

@inline dim(Wₕ::ScalarGridSpace) = dim(mesh(Wₕ))
@inline dim(::Type{W}) where W<:ScalarGridSpace = dim(mesh_type(W))

"""
	ndofs(Wₕ::ScalarGridSpace)
	ndofs(Wₕ::ScalarGridSpace, [::Type{Tuple}])

Returns the total number of degrees of freedom (or a tuple with the degrees of freedom per direction) of the [ScalarGridSpace](@ref) `Wₕ`.
"""
@inline ndofs(Wₕ::ScalarGridSpace) = npoints(mesh(Wₕ))
@inline ndofs(Wₕ::ScalarGridSpace, ::Type{Tuple}) = npoints(mesh(Wₕ), Tuple)

"""
	eltype(Wₕ::ScalarGridSpace)
	eltype(::Type{<:ScalarGridSpace})

Returns the element type of the mesh associated with [ScalarGridSpace](@ref) `Wₕ`. If the input argument is a type derived from [ScalarGridSpace](@ref) then the function returns the element type of the [AbstractMeshType](@ref) associated with it.
"""
@inline eltype(Wₕ::ScalarGridSpace) = eltype(backend(Wₕ))
@inline eltype(::Type{W}) where W<:ScalarGridSpace = eltype(mesh_type(W))

"""
	_innerh_weights!(u, Ωₕ::AbstractMeshType)

Builds the weights for the standard discrete ``L^2`` inner product, ``inner_h(\\cdot, \\cdot)``, on the space of grid functions, following the order of the points provided by `indices(Ωₕ)`. The values are stored in vector `u`.
"""
function _innerh_weights!(u, Ωₕ::AbstractMeshType)
	f = Base.Fix1(cell_measure, Ωₕ)
	idxs = indices(Ωₕ)
	dims = npoints(Ωₕ, Tuple)

	v = Base.ReshapedArray(u, dims, ())
	_parallel_for!(v, idxs, f)
	# it should be 
	# _parallel_map!(f, v, idxs)
end

"""
	_innerplus_weights!(u::VT, Ωₕ, component = 1) where VT

Builds a set of weights based on the spacings, associated with the `component`-th direction, for the modified discrete ``L^2`` inner product on the space of grid functions, following the order of the points provided by `indices(Ωₕ)`. The values are stored in vector `u`.
"""
function _innerplus_weights!(u::VT, Ωₕ, component = 1) where VT
	T = eltype(VT)
	mesh_component = Ωₕ(component)

	f = Base.Fix1(spacing, mesh_component)
	idxs = indices(mesh_component)

	@inbounds @simd for idx in idxs
		i = idx[1]

		u[i] = f(i)
	end

	@inbounds u[1] = zero(T)
	return
end

"""
	_innerplus_mean_weights!(u::VT, Ωₕ, component::Int = 1) where VT

Builds a set of weights based on the half spacings, associated with the `component`-th direction, for the modified discrete ``L^2`` inner product on the space of grid functions, following the order of the [points](@ref). The values are stored in vector `u`.
for each component.
"""
function _innerplus_mean_weights!(u::VT, Ωₕ, component::Int = 1) where VT
	T = eltype(VT)
	u[1] = zero(T)
	mesh_component = Ωₕ(component)
	N = npoints(mesh_component)

	@inbounds @simd for i in 2:(N - 1)
		u[i] = half_spacing(mesh_component, i)
	end

	@inbounds u[N] = zero(T)
	return nothing
end

@inline @generated function __prod(diags::NTuple{D,VT}, I) where {D,VT}
	res = :(one(eltype(VT)))
	for i in 1:D
		res = :(@inbounds(diags[$i][I[$i]]) * $res)
	end
	return res
end

"""
	__innerplus_weights!(v, innerplus_per_component)

Builds the weights for the modified discrete ``L^2`` inner product on the space of grid functions [ScalarGridSpace](@ref). The result is stored in vector `v`.
"""
function __innerplus_weights!(v, innerplus_per_component)
	idxs = CartesianIndices(v)
	f = Base.Fix1(__prod, innerplus_per_component)
	_parallel_for!(v, idxs, f)
end