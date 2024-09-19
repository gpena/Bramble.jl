"""
	struct Mesh1D{T} <: MeshType{1}
		markers::MeshMarkers{1}
		indices::CartesianIndices{1,Tuple{Base.OneTo{Int}}}
		pts::Vector{T}
		npts::Int
	end	

Structure to create a 1D mesh with `npts` points of type `T`. The points that define the mesh are stored in `pts` and are identified, following the same order, with the indices in `indices`. The variable `markers` stores, for each [Domain](@ref) marker, the indices satisfying ``f(x)=0``, where ``f``is the marker function.

For future reference, the `npts` entries of vector `pts` are

```math
x_i, \\, i=1,\\dots,N.
```
"""
struct Mesh1D{T} <: MeshType{1}
	markers::MeshMarkers{1}
	indices::CartesianIndices{1,Tuple{Base.OneTo{Int}}}
	pts::Vector{T}
	npts::Int
end

"""
	mesh(Ω::Domain, npts::Int, unif::Bool)

Returns a [Mesh1D](@ref) based on [Domain](@ref) `Ω` and `npts` points with uniform spacing
if `unif` is `true` (otherwise, the points are randomly generated on the domain).

For future reference, we denote the `npts` entries of vector `pts` as

```math
x_i, \\, i=1,\\dots,N.
```

# Example

```
julia> I = interval(0,1); Ωₕ = mesh(domain(I), 10, true)
1D Mesh
nPoints: 10
Markers: Dirichlet
```
"""
function mesh(Ω::Domain{CartesianProduct{1,T},MarkersType}, npts::Int, unif::Bool) where {T,MarkersType}
	R, pts, markersForMesh = create_mesh1d_basics(Ω, npts, unif)

	return Mesh1D{T}(markersForMesh, R, pts, npts)
end

mesh(Ω::Domain{CartesianProduct{1,T},Markers}, npts::NTuple{1,Int}, unif::NTuple{1,Bool}) where {T,Markers} = mesh(Ω, npts[1], unif[1])

@inline dim(Ωₕ::Mesh1D{T}) where T = 1
@inline dim(Ωₕ::Type{<:Mesh1D{T}}) where T = 1

@inline eltype(Ωₕ::Mesh1D{T}) where T = T
@inline eltype(Ωₕ::Type{<:Mesh1D{T}}) where T = T

"""
	create_mesh1d_basics(Ω::Domain, npts::Int, unif::Bool)

Creates the basic components of a 1D mesh, given a [Domain](@ref) `Ω`, the number
of points `npts` and a boolean `unif`. The points are equally spaced if `unif`
is `true` (otherwise, the points are randomly generated on the domain).
"""
function create_mesh1d_basics(Ω::Domain, npts::Int, unif::Bool)
	pts = Vector{eltype(Ω)}(undef, npts)
	createpoints!(pts, set(Ω), unif)
	R = generate_indices(npts)

	markersForMesh = MeshMarkers{1}()

	for label in labels(Ω)
		merge!(markersForMesh, Dict(label => VecCartIndex{1}()))
	end

	addmarkers!(markersForMesh, Ω, R, pts)

	return R, pts, markersForMesh
end

function show(io::IO, Ωₕ::Mesh1D)
	l = join(keys(Ωₕ.markers), ", ")
	properties = ["1D Mesh",
		"#Points: $(Ωₕ.npts)",
		"Markers: $l"]

	print(io, join(properties, "\n"))
end

@inline (Ωₕ::Mesh1D)(_) = Ωₕ

"""
	npoints(Ωₕ::Mesh1D)

Returns a 1-tuple of the number of points ``x_i`` in `Ωₕ`.

# Example

```
julia> Ωₕ = Mesh(Domain(Interval(0,1)), 10, true); npoints(Ωₕ)
(10,)
```
"""
@inline npoints(Ωₕ::Mesh1D) = (Ωₕ.npts,)

"""
	points(Ωₕ::Mesh1D)

Returns a vector with all the points ``x_i, \\, i=1,\\dots,N`` in `Ωₕ`.
"""
@inline points(Ωₕ::Mesh1D) = Ωₕ.pts

"""
	point(Ωₕ::Mesh1D, i)

Returns the `i`-th point of `Ωₕ`, ``x_i``.
"""
@inline point(Ωₕ::Mesh1D, i) = Ωₕ.pts[i]

"""
	points_iterator(Ωₕ::Mesh1D)

Returns an iterator over all points ``x_i, \\, i=1,\\dots,N`` in mesh `Ωₕ`.
"""
@inline points_iterator(Ωₕ::Mesh1D) = points(Ωₕ)

"""
	hₘₐₓ(Ωₕ::Mesh1D)

Returns the maximum over the space stepsize ``h_i``of mesh `Ωₕ`

```math
h_{max} = \\max_{i=1,\\dots,N} x_i - x_{i-1}.
```
"""
@inline hₘₐₓ(Ωₕ::Mesh1D) = maximum(spacing_iterator(Ωₕ))

"""
	spacing(Ωₕ::Mesh1D, i)

Returns the space stepsize, ``h_i`` at index `i` in mesh `Ωₕ`

```math
h_i = x_i - x_{i-1}, \\, i=2,\\dots,N
```

where ``h_1 = x_2 - x_1``.
"""
@inline function spacing(Ωₕ::Mesh1D, i)
	@assert 1 <= i <= ndofs(Ωₕ)

	if i === 1
		return Ωₕ.pts[2] - Ωₕ.pts[1]
	else
		return Ωₕ.pts[i] - Ωₕ.pts[i - 1]
	end
end

"""
	spacing_iterator(Ωₕ::Mesh1D)

Returns an iterator over all space step sizes ``h_i, \\, i=1,\\dots,N`` in mesh `Ωₕ`.
"""
@inline spacing_iterator(Ωₕ::Mesh1D) = (spacing(Ωₕ, i) for i in eachindex(points(Ωₕ)))

"""
	half_spacing(Ωₕ::Mesh1D, i)

Returns the indexwise average of the space stepsize, ``h_{i+1/2}``, at index `i` in mesh `Ωₕ`

```math
h_{i+1/2} = \\frac{h_i + h_{i+1}}{2}, \\, i=1,\\dots,N-1
```

where ``h_{N+1/2} = \\frac{h_{N}}{2}`` and ``h_{1/2} = \\frac{h_1}{2}``.
"""
@inline function half_spacing(Ωₕ::Mesh1D{T}, i) where T
	@assert 1 <= i <= ndofs(Ωₕ)

	x = zero(eltype(Ωₕ))

	if i == 1 || i == ndofs(Ωₕ)
		x = spacing(Ωₕ, i)
	else
		x = spacing(Ωₕ, i + 1) + spacing(Ωₕ, i)
	end

	return x * convert(T, 0.5)
end

"""
	half_spacing_iterator(Ωₕ::Mesh1D)

Returns an iterator over all indexwise average of the space stepsizes ``h_{i+1/2}, \\, i=1,\\dots,N`` in mesh `Ωₕ`.
"""
@inline half_spacing_iterator(Ωₕ::Mesh1D) = (half_spacing(Ωₕ, i) for i in eachindex(points(Ωₕ)))

"""
	function half_points(Ωₕ::Mesh1D, i)

Returns the average of two neighboring, ``x_{i+1/2}``, points in mesh `Ωₕ`, at index `i`

```math
x_{i+1/2} = x_i + \\frac{h_{i+1}}{2}, \\, i=1,\\dots,N-1,
```

``x_{N+1/2} = x_{N}`` and ``x_{1/2} = x_1``.
"""
@inline function half_points(Ωₕ::Mesh1D{T}, i) where T
	@assert 1 <= i <= ndofs(Ωₕ) + 1

	if i == 1
		return point(Ωₕ, 1)
	elseif i == ndofs(Ωₕ) + 1
		return point(Ωₕ, ndofs(Ωₕ))
	else
		return (point(Ωₕ, i) + point(Ωₕ, i - 1)) * convert(T, 0.5)
	end
end

"""
	half_points_iterator(Ωₕ::Mesh1D)

Returns an iterator over all points ``x_{i+1/2}, \\, i=1,\\dots,N`` in mesh `Ωₕ`.
"""
@inline half_points_iterator(Ωₕ::Mesh1D) = (half_points(Ωₕ, i) for i in 1:(ndofs(Ωₕ) + 1))

"""
	cell_measure(Ωₕ::Mesh1D, i::CartesianIndex)

Returns the measure of the cell ``\\square_{i} = [x_i - \\frac{h_{i}}{2}, x_i + \\frac{h_{i+1}}{2}]`` at `CartesianIndex` `i` in mesh `Ωₕ`, which is
given by ``h_{i+1/2}``.
"""
@inline cell_measure(Ωₕ::Mesh1D, i::CartesianIndex{1}) = half_spacing(Ωₕ, i[1])

"""
	cell_measure_iterator(Ωₕ::Mesh1D)

Returns an iterator over ``h_{i+1/2}, \\, i=1,\\dots,N`` in mesh `Ωₕ`.
"""
@inline cell_measure_iterator(Ωₕ::Mesh1D) = Iterators.map(Base.Fix1(cell_measure, Ωₕ), indices(Ωₕ))

"""
	createpoints!(x::Vector, I::CartesianProduct{1}, unif::Bool)

Overrides the components of vector `x` with uniformly (`unif = true`) or randomly distributed
(`unif = false`) points in the interval `I`.
"""
@inline function createpoints!(x::Vector{T}, I::CartesianProduct{1,T}, unif::Bool) where T
	npts = length(x)
	x .= range(zero(T), one(T), length = npts)
	v = view(x, 2:(npts - 1))

	if !unif
		rand!(v)
		sort!(v)
	end

	a, b = tails(I)
	@.. x = a + x * (b - a)
end

"""
	generate_indices(npts::Int)

Returns a `CartesianIndices` object for a vector of length `npts`.
"""
@inline generate_indices(npts::Int) = CartesianIndices((npts,))

"""
	bcindices(Ωₕ::Mesh1D)

Returns the indices of the boundary points of mesh `Ωₕ`.
"""
@inline bcindices(Ωₕ::Mesh1D) = bcindices(indices(Ωₕ))
@inline bcindices(R::CartesianIndices{1}) = (first(R), last(R))

"""
	intindices(Ωₕ::Mesh1D)

Returns the indices of the interior points of mesh `Ωₕ`.
"""
@inline intindices(Ωₕ::Mesh1D) = CartesianIndices((2:(ndofs(Ωₕ) - 1),))
@inline intindices(R::CartesianIndices{1}) = CartesianIndices((2:(length(R) - 1),))

"""
	addmarkers!(markerList::MeshMarkers{1}, Ω::Domain, R::CartesianIndices{1}, pts)

For each [Domain](@ref) marker, stores in `markerList` the indices of the points that satisfy ``f(x)=0``,
where ``f`` is the corresponding levelset function associated with the marker.
"""
function addmarkers!(markerList::MeshMarkers{1}, Ω::Domain, R::CartesianIndices{1}, pts)
	boundary = bcindices(R)

	for idx in boundary, marker in markers(Ω)
		if marker.f(_index2point(pts, idx)) ≈ 0
			push!(markerList[marker.label], idx)
		end
	end
end