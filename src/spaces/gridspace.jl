# Getters for SpaceWeights
@inline weights_innerh(weights::SpaceWeights) = weights.innerh
@inline weights_innerplus(weights::SpaceWeights) = weights.innerplus
@inline weights_innerplus(weights::SpaceWeights, i) = weights.innerplus[i]

# Functions to help create SingleGridSpace instante
@inline _create_vector(Ωₕ) = vector(backend(Ωₕ), npoints(Ωₕ))

function create_space_weights(Ωₕ::AbstractMeshType{D}) where D
	innerplus = ntuple(i -> _create_vector(Ωₕ), Val(D))
	innerplus_per_component = ntuple(j -> _create_vector(Ωₕ(j)), Val(D))

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

	inner_h_vec = _create_vector(Ωₕ)
	build_innerh_weights!(inner_h_vec, Ωₕ)

	return SpaceWeights{D,typeof(inner_h_vec)}(inner_h_vec, innerplus)
end

function create_space_backward_diff_matrices(Ωₕ::AbstractMeshType{D}) where D
	b = backend(Ωₕ)
	npts = npoints(Ωₕ)

	return ntuple(i -> matrix(b, npts, npts), Val(D))
end

"""
	gridspace(Ωₕ::AbstractMeshType)

Constructor for a [SingleGridSpace](@ref) defined on the mesh `Ωₕ`. This builds the weights for the inner products mentioned in [GridSpace](@ref) as well as the differentiation matrices associated with the grid points of mesh `Ωₕ`.
"""
function gridspace(Ωₕ::AbstractMeshType{D}; cache_average_matrices = false, cache_backward_diff_matrices = true) where D
	b = backend(Ωₕ)
	npts = npoints(Ωₕ)

	#@info "Create weights"
	weights = create_space_weights(Ωₕ)
	diff_matrices = create_space_backward_diff_matrices(Ωₕ)
	average_matrices = create_space_backward_diff_matrices(Ωₕ)

	if cache_backward_diff_matrices
		#@info "Create differentiation matrices"

		# cache matrices in variable diff_matrices
		# diff_matrices = create_backward_diff_matrices(Wₕ; vector = _create_vector(Ωₕ))
	end

	if cache_average_matrices
		#@info "Create average matrices"

		# cache matrices in variable average_matrices
		# average_matrices = create_average_matrices(Wₕ; vector = _create_vector(Ωₕ))
	end

	space_buffer = create_simple_space_buffer(b, npts, nbuffers = 1)

	MType = typeof(Ωₕ)
	#VT = vector_type(b)
	#MT = matrix_type(b)
	_, VT, MT, _ = backend_types(b)
	BufferType = typeof(space_buffer)

	return SingleGridSpace{MType,D,VT,MT,BufferType}(Ωₕ, weights, diff_matrices, cache_backward_diff_matrices, average_matrices, cache_average_matrices, space_buffer)
end

# Accessors for SingleGridSpace
@inline mesh(Wₕ::SingleGridSpace) = Wₕ.mesh
@inline weights(Wₕ::SingleGridSpace) = Wₕ.weights
@inline backward_diff_matrix(Wₕ::SingleGridSpace, i) = Wₕ.backward_diff_matrix[i]
@inline average_matrix(Wₕ::SingleGridSpace, i) = Wₕ.average_matrix[i]
@inline vector_buffer(Wₕ::SingleGridSpace) = Wₕ.vector_buffer
@inline has_backward_diff_matrix(Wₕ::SingleGridSpace) = Wₕ.has_backward_diff_matrix
@inline has_average_matrix(Wₕ::SingleGridSpace) = Wₕ.has_average_matrix

@inline backend(Wₕ::SingleGridSpace) = backend(mesh(Wₕ))

@inline dim(Wₕ::SingleGridSpace) = dim(mesh(Wₕ))

"""
	ndofs(Wₕ::AbstractSpaceType)

Returns the number of degrees of freedom of the [GridSpace](@ref) `Wₕ`.
"""
@inline ndofs(Wₕ::SingleGridSpace) = npoints(mesh(Wₕ))

"""
	eltype(Wₕ::AbstractSpaceType)

Returns the element type of the mesh associated with [GridSpace](@ref) `Wₕ`. If the input argument is a type derived from [AbstractSpaceType](@ref) then the function returns the element type of the [AbstractMeshType](@ref) associated with it.
"""
@inline Base.eltype(Wₕ::SingleGridSpace) = eltype(backend(Wₕ))
#=
function show(io::IO, Wₕ::SingleGridSpace)
	Ωₕ = mesh(Wₕ)
	D = dim(Ωₕ)

	npts_formatted = format_with_underscores(npoints(Ωₕ))

	properties = ["Gridspace defined on a $(D)D Mesh",
		"nPoints: $npts_formatted",
		"Markers: $(keys(Ωₕ.markers))"]

	println(io, join(properties, "\n"))

	print(io, "\nSubmeshes:\n")

	direction = ["x", "y", "z"]
	properties = ["  $(direction[i]) direction | nPoints: $(npoints(Ωₕ, Tuple)[i])" for i in 1:D]

	print(io, join(properties, "\n"))
end=#

#=

"""
	gridspace(Ωₕ::AbstractMeshType)

Constructor for a [GridSpace](@ref) defined on the mesh `Ωₕ`. This builds the weights for the inner products mentioned in [GridSpace](@ref) as well as the differentiation matrices associated with the grid points of mesh `Ωₕ`.
"""
function gridspace(Ωₕ::AbstractMeshType)
	T = eltype(Ωₕ)
	innerh = _create_vector(Ωₕ)
	build_innerh_weights!(innerh, Ωₕ)

	D = dim(Ωₕ)
	npts = npoints(Ωₕ, Tuple)
	innerplus = ntuple(i -> _create_vector(Ωₕ), D)
	innerplus_per_component = ntuple(j -> _create_vector(Ωₕ(j)), D)

	for i in 1:D
		for k in 1:D
			if k == i
				_innerplus_weights!(innerplus_per_component[k], Ωₕ, k)
			else
				_innerplus_mean_weights!(innerplus_per_component[k], Ωₕ, k)
			end
		end

		v = Base.ReshapedArray(innerplus[i], npts, ())
		__innerplus_weights!(v, innerplus_per_component)
	end

	T = eltype(Ωₕ)
	type = MatrixElement{GridSpace{typeof(Ωₕ),D,T},T}
	Wₕ = GridSpace(Ωₕ, innerh, innerplus, Dict{Int,type}(), _create_vector(Ωₕ))

	# create backward difference matrices
	diff_matrices = create_backward_diff_matrices(Wₕ; vector = _create_vector(Ωₕ))

	# push diff matrices to cache
	for i in 1:D
		push!(Wₕ.diff_matrix_cache, i => diff_matrices[i])
	end

	return Wₕ
end
=#
"""
	build_innerh_weights!(u, Ωₕ::AbstractMeshType)

Builds the weights for the standard discrete ``L^2`` inner product, ``inner_h(\\cdot, \\cdot)``, on the space of grid functions, following the order of the points provided by `indices(Ωₕ)`. The values are stored in vector `u`.
"""
function build_innerh_weights!(u, Ωₕ::AbstractMeshType)
	f = Base.Fix1(_cell_measure, Ωₕ)
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

	f = Base.Fix1(spacing, Ωₕ(component))
	idxs = indices(Ωₕ(component))

	for idx in idxs
		i = idx[1]

		u[i] = f(i)
	end

	u[1] = zero(T)
	return nothing
end

"""
	_innerplus_mean_weights!(u::VT, Ωₕ, component::Int = 1) where VT

Builds a set of weights based on the half spacings, associated with the `component`-th direction, for the modified discrete ``L^2`` inner product on the space of grid functions, following the order of the [points](@ref). The values are stored in vector `u`.
for each component.
"""
function _innerplus_mean_weights!(u::VT, Ωₕ, component::Int = 1) where VT
	T = eltype(VT)
	u[1] = zero(T)
	N = npoints(Ωₕ(component))

	for i in 2:(N - 1)#indices(Ωₕ(component)) <- try
		#if i === 1 || i === N
		#	continue
		#end

		@inbounds u[i] = half_spacing(Ωₕ(component), i)
	end

	u[N] = zero(T)
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

Builds the weights for the modified discrete ``L^2`` inner product on the space of grid functions [GridSpace](@ref). The result is stored in vector `v`.
"""
function __innerplus_weights!(v, innerplus_per_component)
	idxs = CartesianIndices(v)
	f = Base.Fix1(__prod, innerplus_per_component)
	_parallel_for!(v, idxs, f)

	return nothing
end
