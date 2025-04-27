"""
	mutable struct Mesh1D{BackendType,CartIndicesType,VectorType} <: MeshType{1}
		markers::MeshMarkers{1}
		indices::CartIndicesType
		const backend::BackendType
		pts::VectorType
	end	

Structure to create a 1D mesh with `npts` points. The points that define the grid are stored in `pts` and are identified, following the same order, with the indices in field `indices`. The variable `markers` is a dictionary that stores the indices associated with the [DomainMarkers](@ref) using [MarkerIndices](@ref).

For future reference, the `npts` entries of vector `pts` are

```math
x_i, \\, i=1,\\dots,N.
```
"""
mutable struct Mesh1D{BackendType<:Backend,CartIndicesType,VectorType<:AbstractVector} <: MeshType{1}
	markers::MeshMarkers{1}
	indices::CartIndicesType
	const backend::BackendType
	pts::VectorType
	collapsed::Bool
end

@inline is_collapsed(Ωₕ::Mesh1D) = Ωₕ.collapsed

"""
	points(Ωₕ::Mesh1D)
	points(Ωₕ::Mesh1D, i)

Returns a vector with all the points ``x_i, \\, i=1,\\dots,N`` in `Ωₕ`. A second argument can be passed. If it is an `Int` or a `CartesianIndex{1}`, it returns the `i`-th point of `Ωₕ`, ``x_i``..
"""
@inline points(Ωₕ::Mesh1D) = Ωₕ.pts
@inline function points(Ωₕ::Mesh1D, i)
	idx = CartesianIndex(i)
	@assert idx in indices(Ωₕ)

	return getindex(points(Ωₕ), idx[1])
end

"""
	points_iterator(Ωₕ::Mesh1D)

Returns an iterable object over all the points ``x_i, \\, i=1,\\dots,N`` in `Ωₕ`.
"""

@inline points_iterator(Ωₕ::Mesh1D) = (point for point in points(Ωₕ))

"""
	set_points!(Ωₕ::Mesh1D, pts)

	Overrides the points in Ωₕ.
"""
@inline set_points!(Ωₕ::Mesh1D, pts) = (Ωₕ.pts = pts)

@inline dim(_::Mesh1D) = 1
@inline dim(::Type{<:Mesh1D}) = 1

@inline eltype(_::Mesh1D{BackendType}) where BackendType = eltype(BackendType)
@inline eltype(::Type{<:Mesh1D{BackendType}}) where BackendType = eltype(BackendType)

function show(io::IO, Ωₕ::Mesh1D)
	labels = keys(markers(Ωₕ))
	labels_styled_combined = color_markers(labels)

	fields = ("Markers",)
	mlength = max_length_fields(fields)

	labels_output = style_field("Markers", labels_styled_combined, max_length = mlength)
	type_info = style_title("1D mesh")
	npoints_info = style_field("nPoints", npoints(Ωₕ), max_length = mlength)

	final_output = style_join(type_info, npoints_info, labels_output)
	print(io, final_output)
end

@inline (Ωₕ::Mesh1D)(_) = Ωₕ

"""
	merge_consecutive_indices!(marker_data::MarkerIndices{1})

Finds sequences of consecutive `CartesianIndex{1}` elements within `marker_data.c_index`. Removes these sequences (if longer than one element) and adds the corresponding `CartesianIndices{1}` range object to `marker_data.c_indices`.
"""
function merge_consecutive_indices!(marker_data::MarkerIndices{1})
	c_index_set = marker_data.c_index
	c_indices_set = marker_data.c_indices

	n = length(c_index_set)

	# Need at least 2 elements to potentially form a mergeable range
	if n < 2
		return nothing
	end

	# --- Optimization using BitSet ---
	# 1. Convert CartesianIndex values to integers in a BitSet
	#    This allocates the BitSet but avoids collect+sort.
	#    Iteration over the BitSet is fast and yields sorted integers.
	int_values_bs = BitSet(ci.I[1] for ci in c_index_set)
	# --------------------------------

	# Store results temporarily as primitive types to minimize allocations until the end
	ranges_found = Vector{Base.UnitRange{Int}}()     # Stores integer ranges like 1:3, 7:8
	vals_to_remove = Vector{Int}()             # Stores integers like 1,2,3, 7,8

	# --- Iterate efficiently through the BitSet ---
	# We need to manually handle the iterator to detect runs
	iter_state = iterate(int_values_bs)
	while iter_state !== nothing
		start_val, state = iter_state # Current value is the start of a potential run
		end_val = start_val           # Track the end of the run

		# Look ahead for consecutive values
		prev_val = start_val
		iter_state = iterate(int_values_bs, state) # Advance iterator once

		while iter_state !== nothing
			current_val, state = iter_state
			if current_val == prev_val + 1
				# Extend the run
				end_val = current_val
				prev_val = current_val
				iter_state = iterate(int_values_bs, state) # Consume this element
			else
				# Run ended (or next element is not consecutive)
				break
			end
		end
		# --- Run identified: start_val to end_val ---

		# Check if the run had more than one element
		if end_val > start_val
			push!(ranges_found, start_val:end_val)
			# Add all values in the found range to removal list
			# This avoids creating intermediate CartesianIndex objects here
			for v in start_val:end_val
				push!(vals_to_remove, v)
			end
		end
		# The outer loop continues with the state from where the inner loop broke or finished
	end
	# --- Finished iterating through BitSet ---

	# --- Apply changes if any ranges were found ---
	if !isempty(ranges_found)
		# Convert collected integer ranges to CartesianIndices objects
		# Using a generator avoids allocating an intermediate collection
		ranges_to_add = Set(CartesianIndices((r,)) for r in ranges_found)

		# Convert collected integer values to CartesianIndex objects for removal
		# Using a generator avoids allocating an intermediate collection
		indices_to_remove = Set(CartesianIndex(v) for v in vals_to_remove)

		# Modify the original marker_data sets
		union!(c_indices_set, ranges_to_add)
		setdiff!(c_index_set, indices_to_remove)
	end
	# --- End apply changes ---

	return nothing
end

"""
	npoints(Ωₕ::Mesh1D)
	npoints(Ωₕ::Mesh1D, Tuple)

Returns the number of points ``x_i`` in `Ωₕ`. If the second argument is passed, it returns the same information as a `1`-tuple.

# Example

```@example
julia> Ωₕ = mesh(domain(interval(0, 1)), 10, true);
	   npoints(Ωₕ);
10
```
"""
@inline npoints(Ωₕ::Mesh1D) = length(points(Ωₕ))
@inline npoints(Ωₕ::Mesh1D, ::Type{Tuple}) = (npoints(Ωₕ),)

"""
	hₘₐₓ(Ωₕ::Mesh1D)

Returns the maximum over the space stepsize ``h_i``of mesh `Ωₕ`

```math
h_{max} \\vcentcolon = \\max_{i=1,\\dots,N} x_i - x_{i-1}.
```
"""
@inline hₘₐₓ(Ωₕ::Mesh1D) = maximum(spacing_iterator(Ωₕ))

"""
	spacing(Ωₕ::Mesh1D, i)

Returns the space stepsize, ``h_i`` at index `i` in mesh `Ωₕ`.

```math
h_i \\vcentcolon = x_i - x_{i-1}, \\, i=2,\\dots,N
```

and ``h_1 \\vcentcolon = x_2 - x_1``.
"""
@inline function spacing(Ωₕ::Mesh1D, i)
	if topo_dim(Ωₕ) == 0
		return eltype(Ωₕ)(0.0)
	end

	pts = points(Ωₕ)
	idx = CartesianIndex(i)
	@assert idx in indices(Ωₕ)

	if idx === first(indices(Ωₕ))
		return pts[2] - pts[1]
	end

	_i = idx[1]
	_i_1 = idx[1] - 1
	return pts[_i] - pts[_i_1]
end

"""
	spacing_iterator(Ωₕ::Mesh1D)

Returns an iterator for the space stepsizes, ``h_i`` at index `i` in mesh `Ωₕ`.

```math
h_i \\vcentcolon = x_i - x_{i-1}, \\, i=2,\\dots,N
```

and ``h_1 \\vcentcolon = x_2 - x_1``.
"""
@inline spacing_iterator(Ωₕ::Mesh1D) = (spacing(Ωₕ, i) for i in eachindex(points(Ωₕ)))

"""
	half_spacing(Ωₕ::Mesh1D)

Returns the indexwise average of the space stepsize, ``h_{i+1/2}``, at index `i` in mesh `Ωₕ`.

```math
h_{i+1/2} \\vcentcolon = \\frac{h_i + h_{i+1}}{2}, \\, i=1,\\dots,N-1,
```

``h_{N+1/2} \\vcentcolon = \\frac{h_{N}}{2}`` and ``h_{1/2} \\vcentcolon = \\frac{h_1}{2}``.
"""
@inline function half_spacing(Ωₕ::Mesh1D, i)
	idx = CartesianIndex(i)
	idxs = indices(Ωₕ)

	@assert idx in idxs
	T = eltype(Ωₕ)

	if idx === first(idxs) || idx === last(idxs)
		return spacing(Ωₕ, idx) * convert(T, 0.5)::T
	end

	next = idx[1] + 1
	return (spacing(Ωₕ, next) + spacing(Ωₕ, idx)) * T(0.5)
end

"""
	half_spacing_iterator(Ωₕ::Mesh1D)

Returns an iterator for the indexwise average of the space stepsize, ``h_{i+1/2}``, at index `i` in mesh `Ωₕ`.

```math
h_{i+1/2} \\vcentcolon = \\frac{h_i + h_{i+1}}{2}, \\, i=1,\\dots,N-1,
```

``h_{N+1/2} \\vcentcolon = \\frac{h_{N}}{2}`` and ``h_{1/2} \\vcentcolon = \\frac{h_1}{2}``.
"""
@inline half_spacing_iterator(Ωₕ::Mesh1D) = (half_spacing(Ωₕ, i) for i in eachindex(points(Ωₕ)))

"""
	half_points(Ωₕ::Mesh1D, i)

Returns the average of two neighboring, ``x_{i+1/2}``, points in mesh `Ωₕ`, at index `i`.

```math
x_{i+1/2} \\vcentcolon = x_i + \\frac{h_{i+1}}{2}, \\, i=1,\\dots,N-1,
```

``x_{N+1/2} \\vcentcolon = x_{N}`` and ``x_{1/2} \\vcentcolon = x_1``.
"""
@inline function half_points(Ωₕ::Mesh1D, i)
	T = eltype(Ωₕ)

	@assert i in 1:(npoints(Ωₕ) + 1)

	if i === 1
		return points(Ωₕ, 1)
	end

	if i === npoints(Ωₕ) + 1
		return points(Ωₕ, npoints(Ωₕ))
	end

	return (points(Ωₕ, i) + points(Ωₕ, i - 1)) * convert(T, 0.5)
end

"""
	half_points_iterator(Ωₕ::Mesh1D)

Returns an iterator for the average of two neighboring, ``x_{i+1/2}``, points in mesh `Ωₕ`, at index `i`.
"""

@inline half_points_iterator(Ωₕ::Mesh1D) = (half_points(Ωₕ, i) for i in 1:(npoints(Ωₕ) + 1))

"""
	cell_measure(Ωₕ::Mesh1D, i)

Returns the measure of the cell

```math
\\square_{i} \\vcentcolon = \\left[x_i - \\frac{h_{i}}{2}, x_i + \\frac{h_{i+1}}{2} \\right]
```

at `CartesianIndex` `i` in mesh `Ωₕ`, which is given by ``h_{i+1/2}``.
"""
@inline cell_measure(Ωₕ::Mesh1D, i) = half_spacing(Ωₕ, i)

"""
	cell_measure_iterator(Ωₕ::Mesh1D)

Returns an iterator for the measure of the cells.
"""
@inline cell_measure_iterator(Ωₕ::Mesh1D) = Iterators.map(Base.Fix1(cell_measure, Ωₕ), indices(Ωₕ))

function _generate_random_points!(v)
	rand!(v)
	sort!(v)
end

@inline function _set_points!(x, I::CartesianProduct{1}, unif::Bool)
	npts = length(x)
	T = eltype(I)

	if npts == 1
		x .= zero(T)
		return nothing
	end

	x .= range(zero(T), one(T), length = npts)
	v = view(x, 2:(npts - 1))

	if !unif
		_generate_random_points!(v)
	end

	a, b = tails(I)
	@. x = a + x * (b - a)
end

"""
	generate_indices(npts::Int)

Returns a `CartesianIndices` object for a vector of length `npts`.
"""
@inline generate_indices(npts::Int) = CartesianIndices((npts,))

"""
	boundary_indices(Ωₕ::Mesh1D)

Returns the indices of the boundary points of mesh `Ωₕ`.
"""
@inline boundary_indices(Ωₕ::Mesh1D) = boundary_indices(indices(Ωₕ))
@inline boundary_indices(indices::CartesianIndices{1}) = (first(indices), last(indices))

"""
	interior_indices(Ωₕ::Mesh1D)

Returns the indices of the interior points of mesh `Ωₕ`.
"""
@inline interior_indices(Ωₕ::Mesh1D) = CartesianIndices((2:(npoints(Ωₕ) - 1),))
@inline interior_indices(indices::CartesianIndices{1}) = CartesianIndices((2:(length(indices) - 1),))

function _mesh(Ω::Domain{CartesianProduct{1,T}}, npts::Tuple{Int}, unif::Tuple{Bool}, backend) where T
	@unpack set, markers = Ω
	n_points, = npts

	if topo_dim(Ω) == 0
		n_points = 1
	end

	is_collapsed = topo_dim(set) == 0
	is_uniform, = unif

	pts = vector(backend, n_points)

	_set_points!(pts, set, is_uniform)
	idxs = generate_indices(n_points)

	mesh_markers = MeshMarkers{1}()
	mesh = Mesh1D(mesh_markers, idxs, backend, pts, is_collapsed)

	set_markers!(mesh, markers)
	return mesh
end

"""
	iterative_refinement!(Ωₕ::Mesh1D, domain_markers::DomainMarkers)

Refines the given 1D mesh `Ωₕ` by halving each existing cell, effectively doubling the number of cells and nearly doubling the number of points. It also updates the markers according to `domain_markers` after the refinement.
"""
function iterative_refinement!(Ωₕ::Mesh1D)
	if is_collapsed(Ωₕ)
		return
	end

	npts = 2 * npoints(Ωₕ) - 1

	pts = vector(backend(Ωₕ), npts)
	@views pts[1:2:end] .= points(Ωₕ)
	for i in 2:2:(npts - 1)
		pts[i] = (pts[i + 1] + pts[i - 1]) * 0.5
	end

	idxs = generate_indices(npts)

	set_indices!(Ωₕ, idxs)
	set_points!(Ωₕ, pts)
end

function iterative_refinement!(Ωₕ::Mesh1D, domain_markers::DomainMarkers)
	if is_collapsed(Ωₕ)
		return
	end

	iterative_refinement!(Ωₕ)
	set_markers!(Ωₕ, domain_markers)
end

"""
	change_points!(Ωₕ::Mesh1D, Ω::Domain{CartesianProduct{1}}, pts)

Changes the coordinates of the internal points of the `Mesh1D` object `Ωₕ` to the new coordinates specified in `pts`. The markers of the mesh are also recalculated after this change. This function assumes the points in `pts` are ordered and that the first and last of them coincide with the bounds of the `Ω`.
"""
function change_points!(Ωₕ::Mesh1D, domain_markers::DomainMarkers, pts)
	change_points!(Ωₕ, pts)
	set_markers!(Ωₕ, domain_markers)
end

function change_points!(Ωₕ::Mesh1D, pts)
	npts = npoints(Ωₕ)
	@assert npts == length(pts)

	set_points!(Ωₕ, pts)
end