"""
	 MeshType{D}

Abstract type for meshes. Meshes are only parametrized by their topological dimension `D``.
"""
abstract type MeshType{D} <: BrambleType end

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

using Base: axes, first, last, size, ndims # Explicit imports for clarity if needed

"""
	_get_face_indices(axs, dim, is_min)

Internal helper to create CartesianIndices for a face of a multi-dimensional space.
(Remains the same as before)
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
	merge_consecutive_indices!(marker_data::MarkerIndices{1})

Finds sequences of consecutive `CartesianIndex{1}` elements within the
`marker_data.c_index` set. Removes these sequences (if longer than one element)
and adds the corresponding `CartesianIndices{1}` range object to the
`marker_data.c_indices` set.

Modifies `marker_data` in place.

# Example

```julia
# index_set = Set([CartesianIndex(1), CartesianIndex(2), CartesianIndex(3), CartesianIndex(5), CartesianIndex(7), CartesianIndex(8)])
# indices_set = Set{CartesianIndices{1}}()
# marker_data = MarkerIndices{1}(index_set, indices_set)
# merge_consecutive_indices!(marker_data)
# # After execution:
# # marker_data.c_index == Set([CartesianIndex(5)])
# # marker_data.c_indices == Set([CartesianIndices((1:3,)), CartesianIndices((7:8,))])

```
"""
function merge_consecutive_indices!(marker_data::MarkerIndices{1})
	c_index_set = marker_data.c_index
	c_indices_set = marker_data.c_indices
	# Need at least 2 elements to form a range
	if length(c_index_set) < 2
		return nothing
	end

	# Convert Set to sorted Vector to find consecutive runs
	sorted_indices = sort(collect(c_index_set))

	indices_to_remove = Set{CartesianIndex{1}}()
	ranges_to_add = Set{CartesianIndices{1}}()

	i = 1
	n = length(sorted_indices)
	while i <= n
		# Start of a potential run
		start_index = sorted_indices[i]
		start_val = start_index.I[1]
		end_val = start_val
		run_end_vec_idx = i

		# Check for consecutive elements
		j = i + 1
		while j <= n && sorted_indices[j].I[1] == end_val + 1
			end_val = sorted_indices[j].I[1]
			run_end_vec_idx = j
			j += 1
		end

		# If the run has more than one element, it's a mergeable sequence
		run_length = run_end_vec_idx - i + 1
		if run_length > 1
			# Create the CartesianIndices range object
			range = CartesianIndices((start_val:end_val,))
			push!(ranges_to_add, range)

			# Mark all indices in this run for removal from the original set
			for k in i:run_end_vec_idx
				push!(indices_to_remove, sorted_indices[k])
			end
		end

		# Move the main index past the processed run
		i = run_end_vec_idx + 1
	end

	# Modify the original sets *after* iterating
	if !isempty(ranges_to_add)
		union!(c_indices_set, ranges_to_add)
		setdiff!(c_index_set, indices_to_remove)
	end

	return nothing
end

"""
	merge_consecutive_indices!(marker_data::MarkerIndices{D}) where {D}

Finds sequences of `CartesianIndex{D}` elements consecutive along any single
axis within `marker_data.c_index`. Removes these sequences (if longer than one element)
and adds the corresponding `CartesianIndices{D}` range object to
`marker_data.c_indices`.

Modifies `marker_data` in place.

Note: This version for D > 1 iterates through each dimension. An index might be part
of multiple ranges (e.g., a corner of a 2x2 block could be part of a row and a column range).
It merges linear sequences along axes.
"""
function merge_consecutive_indices!(marker_data::MarkerIndices{D}) where D
	c_index_set = marker_data.c_index
	c_indices_set = marker_data.c_indices

	if D == 1 # Use the specialized, simpler version for D=1
		# Redirect to the D=1 specific method for efficiency / clarity
		# This requires defining the D=1 method separately as above.
		# If defined within the same function via dispatch, Julia handles it.
		# If defined as separate functions, call the specific one:
		merge_consecutive_indices!(marker_data::MarkerIndices{1}) # Assuming specific method exists
		return nothing
	end

	# Need at least 2 elements to potentially form a range
	if length(c_index_set) < 2
		return nothing
	end

	# Keep track of indices that formed *any* range and the ranges found
	# Do modifications at the very end to avoid interference between dimensions
	master_indices_to_remove = Set{CartesianIndex{D}}()
	master_ranges_to_add = Set{CartesianIndices{D}}()

	# Iterate through each dimension, trying to find runs along it
	for dim_to_vary in 1:D
		# Group indices by the coordinates in other dimensions
		# Key: Tuple of coordinates in dimensions *not* equal to dim_to_vary
		# Value: List of CartesianIndex sharing those fixed coordinates
		grouped_indices = Dictionaries.Dictionary{NTuple{D - 1,Int},Vector{CartesianIndex{D}}}()

		for idx in c_index_set
			# Create the key tuple from coordinates in fixed dimensions
			fixed_coords = ntuple(i -> idx.I[i < dim_to_vary ? i : i + 1], D - 1)

			# Get or create the vector for this group
			group_vec = get!(grouped_indices, fixed_coords) do
				Vector{CartesianIndex{D}}()
			end
			push!(group_vec, idx)
		end

		# Process each group to find runs along dim_to_vary
		for group_vec in Dictionaries.values(grouped_indices) # Or values(grouped_indices) for standard Dict
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
					# Ensure other dimensions are indeed the same (should be by grouping, but belt-and-suspenders)
					# Check: all(k -> group_vec[j].I[k] == start_index.I[k], filter(!=(dim_to_vary), 1:D))
					end_val = group_vec[j].I[dim_to_vary]
					run_end_vec_idx = j
					j += 1
				end

				run_length = run_end_vec_idx - i + 1
				if run_length > 1
					# Construct the ranges tuple for CartesianIndices
					ranges = ntuple(k -> begin
										coord_k = start_index.I[k] # Get the coordinate for the k-th dimension
										if k == dim_to_vary
											return start_val:end_val # The range along the varying dimension
										else
											return coord_k:coord_k # Fixed dimension range (single value)
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
@inline dim(Ωₕ::MeshType{D}) where D = D
@inline dim(::Type{<:MeshType{D}}) where D = D

"""
	eltype(Ωₕ::MeshType)
	eltype(::Type{<:MeshType})

Returns the type of element of the points of the mesh.
"""
@inline eltype(Ωₕ::MeshType) = eltype(typeof(Ωₕ))
@inline eltype(Ωₕ::Type{<:MeshType}) = eltype(typeof(Ωₕ))

"""
	indices(Ωₕ::MeshType)

Returns the `CartesianIndices` associated with the points of mesh `Ωₕ`.
"""
@inline indices(Ωₕ::MeshType{D}) where D = (Ωₕ.indices)::CartesianIndices{D}

"""
	marker(Ωₕ::MeshType, str::Symbol)

Returns the [Marker](@ref) function with label `str`.
"""
@inline marker(Ωₕ::MeshType, str::Symbol) = Ωₕ.markers[str]

# investigate if this function is necessary
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

function mesh(Ω::Domain{CartesianProduct{1,T},Markers}, npts::Int, unif::Bool; backend = Backend()) where {T,Markers}
	return _mesh(Ω, (npts,), (unif,), backend)
end