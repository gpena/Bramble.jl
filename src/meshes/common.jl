"""
	 MeshType{D}

Abstract type for meshes. Meshes are only parametrized by their topological dimension `D``.
"""
abstract type MeshType{D} <: BrambleType end

"""
	struct MarkerIndices{D}
		c_index::Set{CartesianIndex{D}}
		c_indices::Set{CartesianIndices{D}}
	end

	Structure to hold sets of individual `CartesianIndex` or `CartesianIndices`.
"""
struct MarkerIndices{D} <: BrambleType
	c_index::Set{CartesianIndex{D}}
	c_indices::Set{CartesianIndices{D}}
end

function boundary_symbol_to_cartesian(indices)
	D = length(size(indices))
	return boundary_symbol_to_cartesian(indices, Val(D))
end

function boundary_symbol_to_cartesian(indices, ::Val{1})
	return Dict(:left => first(indices), :right => last(indices))
end

function boundary_symbol_to_cartesian(indices, ::Val{2})
	N, M = size(indices)
	dict_2d = Dict{Symbol,CartesianIndices{2}}()
	dict_2d[:left] = CartesianIndices((1, 1:M))
	dict_2d[:right] = CartesianIndices((N, 1:M))

	dict_2d[:top] = CartesianIndices((1:N, M))
	dict_2d[:bottom] = CartesianIndices((1:N, M))

	return dict_2d
end

function boundary_symbol_to_cartesian(indices, ::Val{3})
	N, M, K = size(indices)
	dict_3d = Dict{Symbol,CartesianIndices{3}}()
	dict_3d[:left] = CartesianIndices((1:N, 1, 1:K))
	dict_3d[:right] = CartesianIndices((1:N, M, 1:K))

	dict_3d[:top] = CartesianIndices((1:N, 1:M, K))
	dict_3d[:bottom] = CartesianIndices((1:N, 1:M, 1))

	dict_3d[:front] = CartesianIndices((N, 1:M, 1:K))
	dict_3d[:back] = CartesianIndices((1, 1:M, 1:K))

	return dict_3d
end

"""
	_get_face_indices(axs, dim, is_min)

Internal helper to create CartesianIndices for a face of a multi-dimensional space.
"""
function _get_face_indices(axs::NTuple{N,AbstractUnitRange}, dim::Integer, is_min::Bool) where {N}
	idx_ranges = Any[ax for ax in axs]
	target_ax = axs[dim]
	face_index = is_min ? first(target_ax) : last(target_ax)
	idx_ranges[dim] = face_index:face_index
	return CartesianIndices(Tuple(idx_ranges))
end

"""
	boundary_symbol_to_cartesian(indices::CartesianIndices)

Compute a dictionary mapping boundary symbols (:left, :right, etc.) to the
corresponding CartesianIndex or CartesianIndices on the boundary defined
by the input `indices`.

Dispatches to specific methods based on the dimensionality of `indices`.
"""
function boundary_symbol_to_cartesian2(indices::CartesianIndices) # Type hint can be more specific
	D = ndims(indices)
	# Dispatch still works correctly as CartesianIndices are AbstractArrays
	return boundary_symbol_to_cartesian2(indices, Val(D))
end

# --- 1D Implementation ---
"""
	boundary_symbol_to_cartesian(indices::CartesianIndices{1}, ::Val{1})

1D specialization. Maps :left and :right to the first and last CartesianIndex.
"""
function boundary_symbol_to_cartesian2(indices::CartesianIndices{1}, ::Val{1})
	# axes(indices) returns a tuple like (1:10,)
	ax1 = axes(indices, 1)
	imin = first(ax1)
	imax = last(ax1)
	# Return CartesianIndex objects for consistency
	return Dict(:left => CartesianIndex(imin), :right => CartesianIndex(imax))
end

# --- 2D Implementation ---
"""
	boundary_symbol_to_cartesian(indices::CartesianIndices{2}, ::Val{2})

2D specialization. Maps :left, :right (Dim 1) and :bottom, :top (Dim 2)
to CartesianIndices representing the boundary faces.
(Remains the same, but type hint adjusted for clarity)
"""
function boundary_symbol_to_cartesian2(indices::CartesianIndices{2}, ::Val{2})
	axs              = axes(indices)
	dict_2d          = Dict{Symbol,CartesianIndices{2}}()
	dict_2d[:left]   = _get_face_indices(axs, 1, true)
	dict_2d[:right]  = _get_face_indices(axs, 1, false)
	dict_2d[:bottom] = _get_face_indices(axs, 2, true)
	dict_2d[:top]    = _get_face_indices(axs, 2, false)
	return dict_2d
end

# --- 3D Implementation ---
"""
	boundary_symbol_to_cartesian(indices::CartesianIndices{3}, ::Val{3})

3D specialization. Maps :left, :right (Dim 1), :bottom, :top (Dim 2),
and :front, :back (Dim 3) to CartesianIndices representing the boundary faces.
(Remains the same, but type hint adjusted for clarity)
"""
function boundary_symbol_to_cartesian2(indices::CartesianIndices{3}, ::Val{3})
	axs              = axes(indices)
	dict_3d          = Dict{Symbol,CartesianIndices{3}}()
	dict_3d[:left]   = _get_face_indices(axs, 1, true)
	dict_3d[:right]  = _get_face_indices(axs, 1, false)
	dict_3d[:bottom] = _get_face_indices(axs, 2, true)
	dict_3d[:top]    = _get_face_indices(axs, 2, false)
	dict_3d[:front]  = _get_face_indices(axs, 3, true)
	dict_3d[:back]   = _get_face_indices(axs, 3, false)
	return dict_3d
end

"""
	merge_consecutive_indices!(marker_data::MarkerIndices{D}) where {D}

Finds sequences of `CartesianIndex{D}` elements consecutive along any single
axis within `marker_data.c_index`. Removes these sequences (if longer than one element)
and adds the corresponding `CartesianIndices{D}` range object to
`marker_data.c_indices`.

Modifies `marker_data` in place. Uses standard `Dict` for internal grouping.

Note: This version for D > 1 iterates through each dimension. An index might be part
of multiple ranges (e.g., a corner of a 2x2 block could be part of a row and a column range).
It merges linear sequences along axes.
"""
function merge_consecutive_indices!(marker_data::MarkerIndices{D}) where {D}
	# Handle D=1 case by calling the specialized method
	if D == 1
		# This requires the MarkerIndices{1} method to be defined separately
		merge_consecutive_indices!(marker_data::MarkerIndices{1})
		return nothing
	end

	c_index_set = marker_data.c_index
	c_indices_set = marker_data.c_indices

	# Need at least 2 elements to potentially form a range
	if length(c_index_set) < 2
		return nothing
	end

	# Use master lists to collect changes across all dimensions before applying
	master_indices_to_remove = Set{CartesianIndex{D}}()
	master_ranges_to_add = Set{CartesianIndices{D}}()

	# Iterate through each dimension, trying to find runs along it
	for dim_to_vary in 1:D
		# Group indices using standard Dict
		# Key: Tuple of coordinates in dimensions *not* equal to dim_to_vary
		# Value: List of CartesianIndex sharing those fixed coordinates
		grouped_indices = Dict{NTuple{D - 1,Int},Vector{CartesianIndex{D}}}()

		for idx in c_index_set
			# Create the key tuple from coordinates in fixed dimensions
			fixed_coords = ntuple(i -> idx.I[i < dim_to_vary ? i : i + 1], D - 1)

			# Get or create the vector for this group using standard get!
			# get!(collection, key, default) -> returns value for key, adding default if key not present
			group_vec = get!(grouped_indices, fixed_coords, Vector{CartesianIndex{D}}())
			push!(group_vec, idx)
		end

		# Process each group to find runs along dim_to_vary
		for group_vec in values(grouped_indices) # Use standard values()
			if length(group_vec) < 2
				continue # Cannot form a range within this group
			end

			# Sort the group based on the coordinate in the varying dimension
			sort!(group_vec, by = ci -> ci.I[dim_to_vary])

			# --- Apply 1D merging logic along dim_to_vary ---
			i = 1
			n = length(group_vec)
			while i <= n
				start_index = group_vec[i]
				start_val = start_index.I[dim_to_vary]
				end_val = start_val
				run_end_vec_idx = i

				j = i + 1
				while j <= n && group_vec[j].I[dim_to_vary] == end_val + 1
					end_val = group_vec[j].I[dim_to_vary]
					run_end_vec_idx = j
					j += 1
				end

				run_length = run_end_vec_idx - i + 1
				if run_length > 1
					# Construct the ranges tuple for CartesianIndices
					ranges = ntuple(k -> begin
										coord_k = start_index.I[k]
										if k == dim_to_vary
											return start_val:end_val
										else
											return coord_k:coord_k
										end
									end, D)

					range_obj = CartesianIndices(ranges)
					push!(master_ranges_to_add, range_obj)

					# Mark indices in this run for removal (master list)
					for k in i:run_end_vec_idx
						push!(master_indices_to_remove, group_vec[k])
					end
				end
				i = run_end_vec_idx + 1
			end # --- End 1D merging logic ---
		end # End processing groups
	end # End looping through dimensions

	# --- Apply collected changes ---
	if !isempty(master_ranges_to_add)
		union!(c_indices_set, master_ranges_to_add)
		setdiff!(c_index_set, master_indices_to_remove) # Use the master list
	end

	return nothing
end

VecCartIndex{D} = Set{CartesianIndex{D}} where D

"""
	MeshMarkers{D}

Type of dictionary to store the `CartesianIndices` associated with a [Marker](@ref).
"""
MeshMarkers{D} = Dict{Symbol,MarkerIndices{D}} where D

struct Iterator <: BrambleType end

"""
	dim(Ωₕ::MeshType)
	dim(::Type{<:MeshType})

Returns the tolopogical dimension of `Ωₕ`.
"""
@inline dim(_::MeshType{D}) where D = D
@inline dim(::Type{<:MeshType{D}}) where D = D

"""
	indices(Ωₕ::MeshType)

Returns the `CartesianIndices` associated with the points of mesh `Ωₕ`.
"""
@inline indices(Ωₕ::MeshType{D}) where D = (Ωₕ.indices)::CartesianIndices{D}

"""
	marker(Ωₕ::MeshType, str::Symbol)

Returns the [Marker](@ref) function with label `str`.
"""
@inline marker(Ωₕ::MeshType, symbol::Symbol) = Ωₕ.markers[symbol]

function process_label_for_mesh!(markers_mesh::MeshMarkers{D}, indices, set_labels) where D
	c_indices_type = typeof(indices)
	c_index_type = eltype(indices)

	for label in set_labels
		markers_mesh[label] = MarkerIndices{D}(Set{c_index_type}(), Set{c_indices_type}())
	end
end

function _init_mesh_markers(Ωₕ::MeshType, domain_markers::DomainMarkers)
	D = dim(Ωₕ)
	idxs = indices(Ωₕ)
	markers_mesh = MeshMarkers{D}()

	process_label_for_mesh!(markers_mesh, idxs, label_symbols(domain_markers))
	process_label_for_mesh!(markers_mesh, idxs, label_tuples(domain_markers))
	process_label_for_mesh!(markers_mesh, idxs, label_conditions(domain_markers))

	return markers_mesh # Return the populated mesh markers
end

@inline _i2p(pts, idx) = pts[idx]

"""
	_i2p(pts::NTuple{D, Vector{T}}, index::CartesianIndex{D})

Returns a `D` tuple with the coordinates of the point in `pts` associated with the `CartesianIndex` given by ìndex`.
"""
@inline @generated _i2p(pts::NTuple{D,Vector{T}}, index::CartesianIndex{D}) where {D,T} = :(Base.Cartesian.@ntuple $D i->pts[i][index[i]])

# necessary?!
@inline @generated _i2ppo(pts::NTuple{D,Vector{T}}, index::CartesianIndex{D}) where {D,T} = :(Base.Cartesian.@ntuple $D i->pts[i][index[i] + 1])

"""
	mesh(Ω::Domain, npts::Int, unif::Bool)
	mesh(Ω::Domain, npts::NTuple{D}, unif::NTuple{D})

Returns a [Mesh1D](@ref) or a [MeshnD](@ref) (``D=2,3``) defined on the [Domain](@ref) `Ω`. The number of points for each coordinate projection mesh are given in the tuple `npts`. The distribution of points on the submeshes are encoded in the tuple `unif`.

For future reference, the mesh points are denoted as

	- 1D mesh, with `npts` = ``N_x``

```math
x_i, \\, i=1,\\dots,N.
```

  - 2D mesh, with `npts` = (``N_x``, ``N_y``)

```math
(x_i,y_j), \\, i=1,\\dots,N_x, \\, j=1,\\dots,N_y
```

  - 3D mesh, with `npts` = (``N_x``, ``N_y``, ``N_z``)

```math
(x_i,y_j,z_l), \\, i=1,\\dots,N_x, \\, j=1,\\dots,N_y, \\, l=1,\\dots,N_z.
```

# Examples

```@example
julia> I = interval(0, 1);
	   Ωₕ = mesh(domain(I), 10, true);
1D mesh
nPoints: 10
Markers: Dirichlet
```

```@example
julia> X = domain(interval(0, 1) × interval(4, 5));
	   Ωₕ = mesh(X, (10, 15), (true, false));
2D mesh
nPoints: 150
Markers: ["Dirichlet"]

Submeshes:
  x direction | nPoints: 10
  y direction | nPoints: 15
```
"""
function mesh(Ω::Domain, npts::NTuple{D,Int}, unif::NTuple{D,Bool}; backend = Backend()) where D
	return _mesh(Ω, npts, unif, backend)
end

function mesh(Ω::Domain{CartesianProduct{1,T}}, npts::Int, unif::Bool; backend = Backend()) where T
	return _mesh(Ω, (npts,), (unif,), backend)
end