"""
	SpaceType{MeshType}

Abstract type for spaces defined on meshes of type `MeshType`.
"""
abstract type SpaceType{MeshType} <: BrambleType end

"""
	SpaceCacheType

Dictionary of precomputed values used in the implementation of operations
on spaces.
"""
const SpaceCacheType = Dict{Symbol,Any}

"""
	GridSpace{MType, D, T}

Concrete space type for a grid defined on a mesh of type `MType`.

# Fields

  - `mesh::MType`: the mesh on which the space is defined.

  - `innerh_weights::Diagonal{T, Vector{T}}`: the weights for the standard discrete ``L^2``
	inner product on the space of grid functions:

	  + 1D: ``(u_h, v_h)_h = \\sum_{i=0}^{N} h_i u_h(x_i) v_h(x_i)``
	  + 2D: ``(u_h, v_h)_h = \\sum_{i,j=0} h_i k_j u_h(x_i,y_j) v_h(x_i,y_j)``
	  + 3D: ``(u_h, v_h)_h = \\sum_{i,j,m=0} h_i k_j l_m u_h(x_i,y_j,z_m) v_h(x_i,y_j,z_m)``
		where the ``h_i = x_i - x_{i-1}``, ``k_j = y_j - y_{j-1}`` and ``l_m = z_m - z_{m-1}``.
  - `innerplus_weights::NTuple{D, Diagonal{T, Vector{T}}}`: the weights for the
	modified discrete ``L^2`` inner product on the space of grid functions,
	for each component (x, y, z).
"""
struct GridSpace{MType,D,T} <: SpaceType{MType}
	mesh::MType
	innerh_weights::Diagonal{T,Vector{T}}
	innerplus_weights::NTuple{D,Diagonal{T,Vector{T}}}
	cache::SpaceCacheType
end

const var2symbol = ("ₓ", "ᵧ", "₂")

"""
	GridSpace(M::MeshType)

Constructor for a GridSpace on the mesh M.
"""
function GridSpace(M::MType) where {D, MType <: MeshType{D}}
	innerh = _create_diagonal(M)
	build_innerh_weights!(innerh, M)

	innerplus = ntuple(i -> similar(innerh), dim(M))
	diagonals = ntuple(j -> _create_diagonal(M(j)), dim(M))

	for i in 1:dim(M)
		for k in 1:dim(M)
			if k == i
				_innerplus_weights!(diagonals[k], M, k)
			else
				_innerplus_mean_weights!(diagonals[k], M, k)
			end
		end

		v = Base.ReshapedArray(innerplus[i].diag, npoints(M), ())
		__innerplus_weights!(v, diagonals)
	end

	S = GridSpace(M, innerh, innerplus, SpaceCacheType())

	#create backward difference matrices
	_aux = _create_diagonal(S)
	diff_matrices = create_backward_diff_matrices(S; diagonal = _aux)

	# push diff matrices to cache
	for i in 1:dim(S)
		push2cache!(S, get_symbol_diff_matrix(Val(i)), diff_matrices[i])
	end

	return S
end

"""
	mesh(S::SpaceType)

Returns the mesh on which the space is defined.
"""
@inline mesh(S::SpaceType{MType}) where MType = S.mesh

"""
	ndofs(S::SpaceType)

Returns the number of degrees of freedom of the space.
"""
@inline ndofs(S::SpaceType) = ndofs(mesh(S))

"""
	eltype(::Type{<:SpaceType{MType}})

Returns the element type of the space.
"""
@inline eltype(::Type{<:SpaceType{MType}}) where MType = eltype(MType)

"""
	eltype(S::SpaceType)

Returns the element type of the space.
"""
@inline eltype(S::SpaceType) = eltype(typeof(S))

"""
	dim(::Type{<:SpaceType{MType}})

Returns the dimensionality of the space.
"""
@inline dim(::Type{<:SpaceType{MType}}) where MType = dim(MType)

"""
	dim(S::SpaceType)

Returns the dimensionality of the space.
"""
@inline dim(S::SpaceType) = dim(typeof(S))

"""
	spacecache(S::SpaceType)

Returns the cache dictionary associated with the space.
"""
@inline spacecache(S::SpaceType) = S.cache

"""
	getcache(S::SpaceType, s::Symbol)

Returns the value associated with key `s` in the cache of the space.
"""
@inline getcache(S::SpaceType, s::Symbol) = (S.cache[s])::typeof(S.cache[s])

"""
	iscached(S::SpaceType, s::Symbol)

Returns `true` if the cache of the space has a key `s`, `false` otherwise.
"""
@inline iscached(S::SpaceType, s::Symbol) = haskey(spacecache(S), s)

"""
	push2cache!(S::SpaceType, id::Symbol, item)

Adds a new entry `id => item` to the cache of the space.
"""
@inline function push2cache!(S::SpaceType, id::Symbol, item)
	push!(spacecache(S), id => item)
end

"""
	_create_diagonal(S::SpaceType)

Returns a diagonal matrix with the same number of degrees of freedom as the space.
"""
@inline _create_diagonal(S::SpaceType) = _create_diagonal(mesh(S))

"""
	_create_diagonal(M::MeshType)

Returns a diagonal matrix with the same number of degrees of freedom as the mesh.
"""
@inline _create_diagonal(M::MeshType) = Diagonal(Vector{eltype(M)}(undef, ndofs(M)))

"""
	build_innerh_weights!(d, M)

TODO explanation
Build the weights for the standard discrete ``L^2`` inner product on the space of grid functions.
"""
function build_innerh_weights!(d::Diagonal, M::MeshType)
	f = Base.Fix1(meas_cell, M)
	map!(f, d.diag, indices(M))
end

"""
	_innerplus_weights!(D, M, component = 1)

TODO explanation
Build the weights for the modified discrete ``L^2`` inner product on the space of grid functions.
"""
function _innerplus_weights!(D::Diagonal{T,Vector{T}}, M::MeshType, component::Int = 1) where T
	N = ndofs(M(component))
	D.diag[1] = zero(T)
	d = D.diag
	f = Base.Fix1(hspace, M(component))
	for i in 2:N
		d[i] = f(i)
	end
end

"""
	_innerplus_mean_weights!(D, M, component = 1)

TODO explanation
Build the weights for the modified discrete ``L^2`` inner product on the space of grid functions,
for each component.
"""
function _innerplus_mean_weights!(D::Diagonal{T,Vector{T}}, M::MeshType, component::Int = 1) where T
	N = ndofs(M(component))
	d = D.diag
	D.diag[1] = zero(T)
	for i in 2:(N - 1)
		d[i] = hmean(M(component), i)
	end

	D.diag[N] = zero(T)
end

@inline @generated function __prod(diags::NTuple{D}, I::CartesianIndex{D}) where D
	res = :(1)
	for i in 1:D
		res = :(diags[$i][I[$i]] * $res)
	end
	return res
end

@inline @generated function __prod(diags::NTuple{D,Diagonal{T,Vector{T}}}, I::NTuple{D,Int}) where {D,T}
	res = :(convert(T, 1))
	for i in 1:D
		res = :(diags[$i].diag[I[$i]] * $res)
	end
	return res
end

"""
	__innerplus_weights!(v, diagonals)

Build the weights for the modified discrete ``L^2`` inner product on the space of grid functions.
"""
function __innerplus_weights!(v, diagonals)
	for idx in CartesianIndices(v)
		v[idx] = __prod(diagonals, Tuple(idx))
	end
end
