"""
	struct Mesh1D{T} <: MeshType{1}
		markers::MeshMarkers{1}
		indices::CartesianIndices{1}
		pts::Vector{T}
		npts::Int
	end	

Structure to create a 1D mesh with `npts` points of type `T`. The points that define the mesh are stored in `pts` and are identified, following the same order, with the indices in `indices`. The variable `markers` stores, for each [Domain](@ref) marker, the indices satisfying ``f(x_i)=0``, where `f` is the marker's function.

For future reference, the `npts` entries of vector `pts` are

```math
x_i, \\, i=1,\\dots,N.
```
"""
struct Mesh1D{T} <: MeshType{1}
	markers::MeshMarkers{1}
	indices::CartesianIndices{1,Tuple{Base.OneTo{Int}}}
	pts::Vector{T}
end

function _mesh(Ω::Domain{CartesianProduct{1,T},MarkersType}, npts::Tuple{Int}, unif::Tuple{Bool}) where {T,MarkersType}
	pts = Vector{eltype(Ω)}(undef, npts[1])
	createpoints!(pts, set(Ω), unif[1])
	R = generate_indices(npts[1])

	markersForMesh = MeshMarkers{1}()

	for label in labels(Ω)
		merge!(markersForMesh, Dict(label => VecCartIndex{1}()))
	end

	boundary = boundary_indices(R)

	for idx in boundary, marker in markers(Ω)
		if marker.f(_i2p(pts, idx)) ≈ 0
			push!(markersForMesh[marker.label], idx)
		end
	end

	return Mesh1D{T}(markersForMesh, R, pts)
end

@inline dim(_::Mesh1D) = 1
@inline dim(::Type{<:Mesh1D{T}}) where T = 1

@inline eltype(_::Mesh1D{T}) where T = T
@inline eltype(::Type{<:Mesh1D{T}}) where T = T

function show(io::IO, Ωₕ::Mesh1D)
	l = join(keys(Ωₕ.markers), ", ")
	properties = ["1D mesh",
		"#Points: $(npoints(Ωₕ))",
		"Markers: $l"]

	print(io, join(properties, "\n"))
end

@inline (Ωₕ::Mesh1D)(_) = Ωₕ

"""
	npoints(Ωₕ::Mesh1D)
	npoints(Ωₕ::Mesh1D, Tuple)

Returns the number of points ``x_i`` in `Ωₕ`. If the second argument is passed, it returns the same information as a `1`-tuple.

# Example

```@example
julia> Ωₕ = mesh(domain(interval(0,1)), 10, true); npoints(Ωₕ)
10
```
"""
@inline npoints(Ωₕ::Mesh1D) = length(Ωₕ.pts)
@inline npoints(Ωₕ::Mesh1D, ::Type{Tuple}) = (length(Ωₕ.pts),)

"""
	points(Ωₕ::Mesh1D)
	points(Ωₕ::Mesh1D, i)
	points(Ωₕ::Mesh1D, Iterator)

Returns a vector with all the points ``x_i, \\, i=1,\\dots,N`` in `Ωₕ`. A second argument can be passed. If it is an `Int` or a `CartesianIndex{1}`, it returns the `i`-th point of `Ωₕ`, ``x_i``. If the second argument is `Iterator` then the function returns a generator iterating over the points.
"""
@inline points(Ωₕ::Mesh1D) = Ωₕ.pts
@inline points(Ωₕ::Mesh1D, ::Type{Iterator}) = (p for p in points(Ωₕ))
@inline function points(Ωₕ::Mesh1D, i)
	idx = CartesianIndex(i)
	@assert idx in indices(Ωₕ)
	return getindex(Ωₕ.pts, idx[1])
end

"""
	hₘₐₓ(Ωₕ::Mesh1D)

Returns the maximum over the space stepsize ``h_i``of mesh `Ωₕ`

```math
h_{max} \\vcentcolon = \\max_{i=1,\\dots,N} x_i - x_{i-1}.
```
"""
@inline hₘₐₓ(Ωₕ::Mesh1D) = maximum(spacing(Ωₕ, Iterator))

"""
	spacing(Ωₕ::Mesh1D, i)
	spacing(Ωₕ::Mesh1D, Iterator)

Returns the space stepsize, ``h_i`` at index `i` in mesh `Ωₕ`. If the second argument `Iterator` is supplied, the function returns a generator iterating over all spacings.

```math
h_i \\vcentcolon = x_i - x_{i-1}, \\, i=2,\\dots,N
```

and ``h_1 \\vcentcolon = x_2 - x_1``.
"""
@inline function spacing(Ωₕ::Mesh1D, i)
	idx = CartesianIndex(i)
	@assert idx in indices(Ωₕ)

	if idx === first(indices(Ωₕ))
		return Ωₕ.pts[2] - Ωₕ.pts[1]
	end

	_i = idx[1]
	_i_1 = idx[1] - 1
	return Ωₕ.pts[_i] - Ωₕ.pts[_i_1]
end

@inline spacing(Ωₕ::Mesh1D, ::Type{Iterator}) = (spacing(Ωₕ, i) for i in eachindex(points(Ωₕ)))

"""
	half_spacing(Ωₕ::Mesh1D, i)
	half_spacing(Ωₕ::Mesh1D, Iterator)

Returns the indexwise average of the space stepsize, ``h_{i+1/2}``, at index `i` in mesh `Ωₕ`. If the second argument `Iterator` is supplied, the function returns a generator iterating over all half spacings.

```math
h_{i+1/2} \\vcentcolon = \\frac{h_i + h_{i+1}}{2}, \\, i=1,\\dots,N-1,
```

``h_{N+1/2} \\vcentcolon = \\frac{h_{N}}{2}`` and ``h_{1/2} \\vcentcolon = \\frac{h_1}{2}``.
"""
@inline function half_spacing(Ωₕ::Mesh1D{T}, i) where T
	idx = CartesianIndex(i)
	@assert idx in indices(Ωₕ)

	if idx === first(indices(Ωₕ)) || idx === last(indices(Ωₕ))
		return spacing(Ωₕ, idx) * convert(T, 0.5)::T
	end

	next = idx[1] + 1
	return (spacing(Ωₕ, next) + spacing(Ωₕ, idx)) * convert(T, 0.5)::T
end

@inline half_spacing(Ωₕ::Mesh1D, ::Type{Iterator}) = (half_spacing(Ωₕ, i) for i in eachindex(points(Ωₕ)))

"""
	half_points(Ωₕ::Mesh1D, i)
	half_points(Ωₕ::Mesh1D, Iterator)

Returns the average of two neighboring, ``x_{i+1/2}``, points in mesh `Ωₕ`, at index `i`. If the second argument `Iterator` is supplied, the function returns a generator iterating over all half points.

```math
x_{i+1/2} \\vcentcolon = x_i + \\frac{h_{i+1}}{2}, \\, i=1,\\dots,N-1,
```

``x_{N+1/2} \\vcentcolon = x_{N}`` and ``x_{1/2} \\vcentcolon = x_1``.
"""
@inline function half_points(Ωₕ::Mesh1D{T}, i) where T
	indices_half_points = generate_indices(npoints(Ωₕ) + 1)

	idx = CartesianIndex(i)
	@assert idx in indices_half_points

	if idx === first(indices_half_points)
		return points(Ωₕ, 1)
	end

	if idx == last(indices_half_points)
		return points(Ωₕ, npoints(Ωₕ))
	end

	former = idx[1] - 1

	return (points(Ωₕ, idx) + points(Ωₕ, former)) * convert(T, 0.5)::T
end

@inline half_points(Ωₕ::Mesh1D, ::Type{Iterator}) = (half_points(Ωₕ, i) for i in 1:(npoints(Ωₕ) + 1))

"""
	cell_measure(Ωₕ::Mesh1D, i)
	cell_measure(Ωₕ::Mesh1D, Iterator)

Returns the measure of the cell

```math
\\square_{i} \\vcentcolon = \\left[x_i - \\frac{h_{i}}{2}, x_i + \\frac{h_{i+1}}{2} \\right]
```

at `CartesianIndex` `i` in mesh `Ωₕ`, which is given by ``h_{i+1/2}``. If the second argument `Iterator` is supplied, the function returns a generator iterating over all cell measures.
"""
@inline cell_measure(Ωₕ::Mesh1D, i) = half_spacing(Ωₕ, i)
@inline cell_measure(Ωₕ::Mesh1D, ::Type{Iterator}) = Iterators.map(Base.Fix1(cell_measure, Ωₕ), indices(Ωₕ))

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
	boundary_indices(Ωₕ::Mesh1D)

Returns the indices of the boundary points of mesh `Ωₕ`.
"""
@inline boundary_indices(Ωₕ::Mesh1D) = boundary_indices(indices(Ωₕ))
@inline boundary_indices(R::CartesianIndices{1}) = (first(R), last(R))

"""
	interior_indices(Ωₕ::Mesh1D)

Returns the indices of the interior points of mesh `Ωₕ`.
"""
@inline interior_indices(Ωₕ::Mesh1D) = CartesianIndices((2:(npoints(Ωₕ) - 1),))
@inline interior_indices(R::CartesianIndices{1}) = CartesianIndices((2:(length(R) - 1),))