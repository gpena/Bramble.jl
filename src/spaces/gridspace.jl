"""
	SpaceType{MeshType}

Abstract type for grid spaces defined on meshes of type [MeshType](@ref).
"""
abstract type SpaceType{MeshType} <: BrambleType end

const SpaceCacheType = Dict{Symbol,Any}

"""
	struct GridSpace{MType,D,T}
		mesh::MType
		innerh_weights::Diagonal{T,Vector{T}}
		innerplus_weights::NTuple{D,Diagonal{T,Vector{T}}}
		cache::SpaceCacheType
	end

Structure for a gridspace defined on a mesh.

The diagonal matrix `innerh_weights` has the weights for the standard discrete ``L^2``
inner product on the space of grid functions defined as follows

  - 1D case

```math
(u_h, v_h)_h = \\sum_{i=1}^{N_x} h_{i+1/2} u_h(x_i) v_h(x_i)
```

  - 2D case

```math
(u_h, v_h)_h = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y} h_{x,i+1/2} h_{y,j+1/2} u_h(x_i,y_j) v_h(x_i,y_j)
```

  - 3D case

```math
(u_h, v_h)_h = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y}\\sum_{l=1}^{N_z} h_{x,i+1/2} h_{y,j+1/2} h_{z,l+1/2} u_h(x_i,y_j,z_l) v_h(x_i,y_j,z_l).
```

The diagonal matrix `innerplus_weights` has the weights for the modified discrete ``L^2`` inner product on the space of grid functions, for each component (x, y, z).

  - 1D case

```math
(u_h, v_h)_+ = \\sum_{i=1}^{N_x} h_{i} u_h(x_i) v_h(x_i)
```

  - 2D case

```math
(u_h, v_h)_{+x} = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y} h_{x,i} h_{y,j+1/2} u_h(x_i,y_j) v_h(x_i,y_j)
```

```math
(u_h, v_h)_{+y} = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y} h_{x,i} h_{y,j+1/2} u_h(x_i,y_j) v_h(x_i,y_j)
```

  - 3D case

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
struct GridSpace{MType,D,T} <: SpaceType{MType}
	mesh::MType
	innerh_weights::Diagonal{T,Vector{T}}
	innerplus_weights::NTuple{D,Diagonal{T,Vector{T}}}
	cache::SpaceCacheType
end

const var2symbol = ("ₓ", "ᵧ", "₂")

"""
	gridspace(Ωₕ::MeshType)

Constructor for a [GridSpace](@ref) defined on the mesh `Ωₕ`.
"""
function gridspace(Ωₕ::MType) where {D,MType<:MeshType{D}}
	innerh = _create_diagonal(Ωₕ)
	build_innerh_weights!(innerh, Ωₕ)

	innerplus = ntuple(i -> similar(innerh), D)
	diagonals = ntuple(j -> _create_diagonal(Ωₕ(j)), D)

	for i in 1:D
		for k in 1:D
			if k == i
				_innerplus_weights!(diagonals[k], Ωₕ, k)
			else
				_innerplus_mean_weights!(diagonals[k], Ωₕ, k)
			end
		end

		v = Base.ReshapedArray(innerplus[i].diag, npoints(Ωₕ, Tuple), ())
		__innerplus_weights!(v, diagonals)
	end

	Wₕ = GridSpace(Ωₕ, innerh, innerplus, SpaceCacheType())

	# create backward difference matrices
	_aux = _create_diagonal(Wₕ)
	diff_matrices = create_backward_diff_matrices(Wₕ; diagonal = _aux)

	# push diff matrices to cache
	 __push_diff_matrice_to_space(Wₕ, diff_matrices)

	return Wₕ
end

@generated function __push_diff_matrice_to_space(Wₕ::GridSpace, diff_matrices::NTuple{D, DType}) where {D, DType}
	ex = :()
	for i in 1:D
		push!(ex.args, :(push2cache!(Wₕ, get_symbol_diff_matrix(Val($i)), diff_matrices[$i])))
	end

	return ex
end

function show(io::IO, Wₕ::GridSpace)
	Ωₕ = mesh(Wₕ)
	D = dim(Ωₕ)

	properties = ["Gridspace defined on a $(D)D Mesh",
		"nPoints: $(npoints(Ωₕ))",
		"Markers: $(keys(Ωₕ.markers))"]

	println(io, join(properties, "\n"))

	print(io, "\nSubmeshes:\n")

	direction = ["x", "y", "z"]
	properties = ["  $(direction[i]) direction | nPoints: $(npoints(Ωₕ, Tuple)[i])" for i in 1:D]

	print(io, join(properties, "\n"))

	cached_variables = keys(spacecache(Wₕ))
	print(io, "\n\nCached variables: $cached_variables")

end


"""
	mesh(Wₕ::SpaceType)

Returns the `mesh` on which the [GridSpace](@ref) `Wₕ` is defined.
"""
@inline mesh(Wₕ::SpaceType{MType}) where MType = Wₕ.mesh

"""
	ndofs(Wₕ::SpaceType)

Returns the number of degrees of freedom of the [GridSpace](@ref) `Wₕ`.
"""
@inline ndofs(Wₕ::SpaceType) = npoints(mesh(Wₕ))

"""
	eltype(Wₕ::SpaceType)

Returns the element type of [GridSpace](@ref) `Wₕ`.
"""
@inline eltype(Wₕ::SpaceType) = eltype(typeof(Wₕ))
@inline eltype(::Type{<:SpaceType{MType}}) where MType = eltype(MType)

"""
	spacecache(Wₕ::SpaceType)

Returns the cache dictionary associated with [GridSpace](@ref) `Wₕ`.
"""
@inline spacecache(Wₕ::SpaceType) = Wₕ.cache

"""
	getcache(Wₕ::SpaceType, s::Symbol)

Returns the value associated with key `s` in the cache of [GridSpace](@ref) `Wₕ`.
"""
@inline getcache(Wₕ::SpaceType, s::Symbol) = (Wₕ.cache[s])::typeof(Wₕ.cache[s])

"""
	iscached(Wₕ::SpaceType, s::Symbol)

Returns `true` if the cache of the [GridSpace](@ref) `Wₕ` has a key `s`, `false` otherwise.
"""
@inline iscached(Wₕ::SpaceType, s::Symbol) = haskey(spacecache(Wₕ), s)

"""
	push2cache!(Wₕ::SpaceType, id::Symbol, item)

Adds a new entry `id => item` to the cache of the [GridSpace](@ref) `Wₕ`.
"""
@inline function push2cache!(Wₕ::SpaceType, id::Symbol, item)
	push!(spacecache(Wₕ), id => item)
end

"""
	_create_diagonal(Wₕ::SpaceType)

Returns a diagonal matrix with the same number of degrees of freedom as the [GridSpace](@ref) `Wₕ`.
"""
@inline _create_diagonal(Wₕ::SpaceType) = _create_diagonal(mesh(Wₕ))

"""
	_create_diagonal(Ωₕ::MeshType)

Returns a diagonal matrix with the same number of degrees of freedom as the mesh.
"""
@inline _create_diagonal(Ωₕ::MeshType) = Diagonal(Vector{eltype(Ωₕ)}(undef, npoints(Ωₕ)))

"""
	build_innerh_weights!(matrix::Diagonal, Ωₕ::MeshType

Builds the weights for the standard discrete ``L^2`` inner product, ``inner_h(\\cdot, \\cdot)``, on the space of grid functions, following the order of the [points](@ref). The values are stored on the diagonal of `matrix`.
"""
function build_innerh_weights!(matrix::Diagonal, Ωₕ::MeshType)
	f = Base.Fix1(cell_measure, Ωₕ)
	map!(f, matrix.diag, indices(Ωₕ))
end

"""
	_innerplus_weights!(matrix::Diagonal, Ωₕ::MeshType, component::Int = 1)

Builds a set of weights based on the spacings, associated with the `component`-th direction, for the modified discrete ``L^2`` inner product on the space of grid functions, following the order of the [points](@ref). The values are stored on the diagonal of `matrix`.
"""
function _innerplus_weights!(matrix::Diagonal{T,Vector{T}}, Ωₕ::MeshType, component::Int = 1) where T
	@assert 1 <= component <= dim(Ωₕ)

	#N = npoints(Ωₕ(component))
	_diag = matrix.diag

	_diag[1] = zero(T)
	f = Base.Fix1(spacing, Ωₕ(component))

	for i in Iterators.drop(indices(Ωₕ(component)), 1) #2:N
		_diag[i] = f(i)
	end
end

"""
	_innerplus_mean_weights!(matrix::Diagonal, Ωₕ::MeshType, component::Int = 1)

Builds a set of weights based on the half spacings, associated with the `component`-th direction, for the modified discrete ``L^2`` inner product on the space of grid functions, following the order of the [points](@ref). The values are stored on the diagonal of `matrix`.
for each component.
"""
function _innerplus_mean_weights!(matrix::Diagonal{T,Vector{T}}, Ωₕ::MeshType, component::Int = 1) where T
	@assert 1 <= component <= dim(Ωₕ)

	
	_diag = matrix.diag
	_diag[1] = zero(T)

	idxs = Iterators.drop(indices(Ωₕ(component)), 1)
	for i in Iterators.take(idxs, 1) #2:(N - 1)
		_diag[i] = half_spacing(Ωₕ(component), i)
	end

	_diag[npoints(Ωₕ(component))] = zero(T)
end
#=
@inline @generated function __prod(diags::NTuple{D}, I) where D
	res = :(1)
	for i in 1:D
		res = :(diags[$i][I[$i]] * $res)
	end
	return res
end
=#
@inline @generated function __prod(diags::NTuple{D, Diagonal{T,Vector{T}}}, I) where {D,T}
	res = :(convert(T, 1))
	for i in 1:D
		res = :(diags[$i].diag[I[$i]] * $res)
	end
	return res
end

"""
	__innerplus_weights!(v, diagonals)

Builds the weights for the modified discrete ``L^2`` inner product on the space of grid functions [GridSpace](@ref).
"""
function __innerplus_weights!(v, diagonals)
	for idx in CartesianIndices(v)
		v[idx] = __prod(diagonals, idx)
	end
end