"""
	$(TYPEDEF)

A mutable structure representing a 1D mesh.

This struct holds all the geometric and topological information for a one-dimensional grid. It includes the coordinates of the grid points (`pts`), the underlying geometric interval (`set`), and a dictionary of markers for labeling specific points or regions. Key geometric quantities like cell centers (`half_pts`) and cell widths (`half_spacings`) are pre-computed and stored for efficiency, which is particularly useful in numerical methods like the finite volume method.

The struct is mutable to allow for in-place modifications, such as mesh refinement.

# Fields

$(FIELDS)

For future reference, the entries of vector `pts` are

```math
x_i, \\, i=1,\\dots,N.
```
"""
mutable struct Mesh1D{BT<:Backend,CI,VT<:AbstractVector,T} <: AbstractMeshType{1}
	"the geometric domain, a 1D CartesianProduct (interval), over which the mesh is defined."
	set::CartesianProduct{1,T}
	"a dictionary mapping `Symbol` labels to `BitVector`s, marking specific points on the mesh."
	markers::MeshMarkers
	"the `CartesianIndices` of the grid, allowing for array-like iteration and indexing over the points."
	indices::CI
	"the computational backend used for linear algebra operations."
	backend::BT
	"a vector holding the coordinates of the grid points, ``x_i``."
	pts::VT
	"a vector of pre-computed cell centers (midpoints), ``x_{i+1/2}``."
	half_pts::VT
	"a vector of pre-computed cell widths, ``h_{i+1/2}``."
	half_spacings::VT
	"a boolean flag indicating if the domain is degenerate (a single point)."
	collapsed::Bool
end

@inline is_collapsed(Ωₕ::Mesh1D) = Ωₕ.collapsed

@inline points(Ωₕ::Mesh1D) = Ωₕ.pts
@inline point(Ωₕ::Mesh1D, i) = points(Ωₕ)[i]
@inline points_iterator(Ωₕ::Mesh1D) = Ωₕ.pts

@inline half_points(Ωₕ::Mesh1D) = Ωₕ.half_pts
@inline half_spacings(Ωₕ::Mesh1D) = Ωₕ.half_spacings
@inline cell_measures(Ωₕ::Mesh1D) = half_spacings(Ωₕ)

"""
	set_points!(Ωₕ::Mesh1D, pts)

Overrides the points in Ωₕ. This function recalculates the cached [half_points](@ref) and [half_spacings](@ref).
"""
@inline function set_points!(Ωₕ::Mesh1D, pts)
	# Directly update the grid point coordinates with the new vector.
	Ωₕ.pts = pts

	# Re-allocate the storage vectors for the derived geometric quantities
	# to match the size of the new points vector.
	half_points!(Ωₕ, vector(backend(Ωₕ), length(pts) + 1))
	half_spacings!(Ωₕ, vector(backend(Ωₕ), length(pts)))

	# Re-compute the cell centers (half_pts) using the new grid points.
	half_points!(half_points(Ωₕ), Ωₕ)

	# Re-compute the cell widths (half_spacings) using the new grid points.
	half_spacing!(half_spacings(Ωₕ), Ωₕ)

	# The function modifies the mesh in-place and returns nothing.
	return
end

"""
	half_points!(Ωₕ::Mesh1D, pts)

Overrides the half_points in Ωₕ.
"""
@inline half_points!(Ωₕ::Mesh1D, pts) = (Ωₕ.half_pts = pts; return)

"""
	half_spacings!(Ωₕ::Mesh1D, pts)

Overrides the spacings in Ωₕ.
"""
@inline half_spacings!(Ωₕ::Mesh1D, pts) = (Ωₕ.half_spacings = pts; return)

@inline eltype(::Mesh1D{BT}) where BT = eltype(BT)
@inline eltype(::Type{<:Mesh1D{BT}}) where BT = eltype(BT)

@inline (Ωₕ::Mesh1D)(_) = Ωₕ

@inline npoints(Ωₕ::Mesh1D) = length(points(Ωₕ))
@inline npoints(Ωₕ::Mesh1D, ::Type{Tuple}) = (npoints(Ωₕ),)

@inline hₘₐₓ(Ωₕ::Mesh1D) = maximum(spacings_iterator(Ωₕ))

@inline @inbounds function spacing(Ωₕ::Mesh1D, i::Int)
	# If the mesh is degenerate (collapsed to a single point), the spacing is zero.
	if is_collapsed(Ωₕ)
		return eltype(Ωₕ)(0.0)
	end

	# Get the vector of grid point coordinates.
	pts = points(Ωₕ)

	# Handle the boundary case for the first point (i=1). Since there is no i-1, we define its spacing as the distance to the next point.
	if i == 1
		return pts[2] - pts[1]
	end

	# For all other interior and right-boundary points, calculate the standard backward spacing.
	return pts[i] - pts[i-1]
end

@inline spacing(Ωₕ::Mesh1D, i::CartesianIndex{1}) = spacing(Ωₕ, i[1])

@inline @inbounds function forward_spacing(Ωₕ::Mesh1D, i::Int)
	# If the mesh is degenerate (a single point), the spacing is zero.
	if is_collapsed(Ωₕ)
		return zero(eltype(Ωₕ))
	end

	# Get the vector of grid point coordinates and the total number of points.
	pts = points(Ωₕ)
	N = npoints(Ωₕ)

	# Handle the boundary case for the last point (i=N). Since there is no i+1,
	# we define its spacing as the distance to the previous point (backward spacing).
	if i == N
		return pts[N] - pts[N-1]
	end

	# For all other points, calculate the standard forward spacing.
	return pts[i+1] - pts[i]
end

@inline forward_spacing(Ωₕ::Mesh1D, i::CartesianIndex{1}) = forward_spacing(Ωₕ, i[1])

@inline half_point(Ωₕ::Mesh1D, i::Int) = Ωₕ.half_pts[i]

@inline spacings_iterator(Ωₕ::Mesh1D) = (spacing(Ωₕ, i) for i in eachindex(points(Ωₕ)))
@inline forward_spacings_iterator(Ωₕ::Mesh1D) = (forward_spacing(Ωₕ, i) for i in eachindex(points(Ωₕ)))

@inline half_spacing(Ωₕ::Mesh1D, i::Int) = Ωₕ.half_spacings[i]

@inline half_spacing(Ωₕ::Mesh1D, idx::CartesianIndex{1}) = half_spacing(Ωₕ, idx[1])
@inline cell_measure(Ωₕ::Mesh1D, i) = half_spacing(Ωₕ, i)

@inline half_spacings_iterator(Ωₕ::Mesh1D) = Ωₕ.half_spacings
@inline half_points_iterator(Ωₕ::Mesh1D) = Ωₕ.half_pts
@inline cell_measures_iterator(Ωₕ::Mesh1D) = half_spacings_iterator(Ωₕ)

_generate_random_points!(v) = (rand!(v); sort!(v); return)

# Internal function to populate a vector `x` with grid point coordinates over a 1D interval `I`.
@inline @inbounds @muladd function _points!(x, I::CartesianProduct{1}, unif::Bool)
	# Get the number of points and the interval's element type and bounds.
	npts = length(x)
	T = eltype(I)
	a, b = tails(I)

	# Handle the trivial case of a single point mesh.
	if npts == 1
		x .= zero(T)
		return
	end

	# Check if the point distribution should be uniform.
	if unif
		# For a uniform grid, calculate the constant step size `h`.
		h = (b - a) / (npts - 1)
		# Populate the grid points using an arithmetic progression.
		@simd for i in eachindex(x)
			x[i] = a + (i - 1) * h
		end
	else
		# For a non-uniform grid, first generate points in the canonical interval [0, 1].
		x[1] = zero(T)
		x[npts] = one(T)

		# Generate random points for the interior of the [0, 1] interval.
		v = view(x, 2:(npts - 1))
		_generate_random_points!(v)

		# Scale and shift the points from [0, 1] to the target interval [a, b].
		@. x = a + x * (b - a)
	end
	return
end

# Calculates the "half points" (cell centers) for a 1D mesh.
@inline @muladd function half_points!(x, Ωₕ)
	n = npoints(Ωₕ)
	pts = points(Ωₕ)

	# The first and last half-points are set to the boundary points. This is a common
	# convention in finite volume methods for defining boundary control volumes.
	x[1] = pts[1]
	x[n+1] = pts[n]

	# For the interior, each half-point is the midpoint between two adjacent grid points.
	@inbounds @simd for i in 2:n
		x[i] = (pts[i] + pts[i-1]) * 0.5
	end

	return
end

# Calculates the "half spacings" (cell widths/measures) for a 1D mesh.
@inline @inbounds @muladd function half_spacing!(x, Ωₕ)
	n = npoints(Ωₕ)

	# The boundary cell widths are defined as half of the spacing of the first/last interval.
	x[1] = spacing(Ωₕ, 1) * 0.5
	x[n] = spacing(Ωₕ, n) * 0.5

	# For interior points, the cell width is the average of the spacings of the two adjacent intervals.
	# This corresponds to the distance between the cell's half-points.
	@inbounds @simd for i in 2:(n - 1)
		x[i] = (spacing(Ωₕ, i) + spacing(Ωₕ, i+1)) * 0.5
	end

	return
end

# Internal constructor function for creating a 1D mesh.
function _mesh(Ω::Domain{CartesianProduct{1,T}}, npts::Tuple{Int}, unif::Tuple{Bool}, backend) where T
	# Unpack the domain's set and markers, and the number of points.
	@unpack set, markers = Ω
	n_points, = npts

	# Check if the domain is a single point (topological dimension is 0).
	is_collapsed = topo_dim(set) == 0

	# If the domain is collapsed, force the number of points to be 1.
	if is_collapsed
		n_points = 1
	end

	# Unpack the uniformity flag.
	is_uniform, = unif

	# Allocate a vector for the grid points using the specified backend.
	pts = vector(backend, n_points)
	# Populate the vector with coordinates, either uniformly or non-uniformly.
	_points!(pts, set, is_uniform)

	# Allocate vectors for derived quantities (cell centers and widths).
	_half_pts = vector(backend, n_points + 1)
	_half_spacings = vector(backend, n_points)

	# Generate the CartesianIndices for the grid.
	idxs = generate_indices(n_points)

	# Instantiate the Mesh1D struct with initial (empty) markers.
	mesh_markers = MeshMarkers()
	mesh = Mesh1D(set, mesh_markers, idxs, backend, pts, _half_pts, _half_spacings, is_collapsed)

	# Now, calculate the derived geometric quantities for the newly created mesh.
	half_points!(half_points(mesh), mesh)
	half_spacing!(half_spacings(mesh), mesh)

	# Finally, apply the domain markers to the mesh points.
	set_markers!(mesh, markers)
	return mesh
end

# Refines a 1D mesh in-place by inserting a new point at the midpoint of each interval.
function iterative_refinement!(Ωₕ::Mesh1D)
	# Do nothing if the mesh is just a single point.
	if is_collapsed(Ωₕ)
		return
	end

	N_old = npoints(Ωₕ)

	# No intervals to refine if there's only one point.
	if N_old <= 1
		return
	end

	# Calculate the number of points in the new, refined mesh.
	N_new = 2 * N_old - 1

	# Allocate a new vector for the refined grid points.
	new_points = vector(backend(Ωₕ), N_new)
	old_points = points(Ωₕ)

	# Place the original points at the odd indices of the new array.
	@views @inbounds new_points[1:2:end] .= old_points
	# Place the new midpoints at the even indices of the new array. ✨
	@views @muladd @inbounds new_points[2:2:end] .= (old_points[1:(end - 1)] .+ old_points[2:end]) .* 0.5

	# Generate new indices for the refined mesh.
	new_indices = generate_indices(N_new)

	# Update the mesh struct with the new indices and points.
	set_indices!(Ωₕ, new_indices)
	set_points!(Ωₕ, new_points)
	return
end

# Refines a 1D mesh and then reapplies the domain markers to the new set of points.
function iterative_refinement!(Ωₕ::Mesh1D, domain_markers::DomainMarkers)
	# Do nothing if the mesh is collapsed.
	if is_collapsed(Ωₕ)
		return
	end

	# First, perform the geometric refinement of the mesh.
	iterative_refinement!(Ωₕ)
	# Then, update the markers based on the new, denser grid.
	set_markers!(Ωₕ, domain_markers)
	return
end

# Changes the grid points of a mesh and then reapplies the domain markers.
function change_points!(Ωₕ::Mesh1D, domain_markers::DomainMarkers, pts)
	# First, update the grid points and all derived geometric quantities.
	change_points!(Ωₕ, pts)
	# Then, recalculate the markers for the new point distribution.
	set_markers!(Ωₕ, domain_markers)
	return
end

# Core function to replace the grid points of a mesh with a new set of points.
function change_points!(Ωₕ::Mesh1D, pts)
	npts = npoints(Ωₕ)
	# Ensure the new points vector has the same size as the old one.
	@assert npts == length(pts) "The number of new points must match the number of points in the mesh."

	# Call the helper function that handles updating the points and all derived quantities.
	set_points!(Ωₕ, pts)
	return
end