"""
	struct MeshnD{D, BackendType, CartIndicesType, Mesh1DType} <: MeshType{D}
		markers::MeshMarkers{D}
		indices::CartIndicesType
		backend::BackendType
		submeshes::NTuple{D, Mesh1DType}
	end

Structure to store a cartesian nD-mesh (``2 \\leq n \\leq 3``) with points of type `T`. For efficiency, the mesh points are not stored. Instead, we store the points of the 1D meshes that make up the nD mesh. To connect both nD and 1D meshes, we use the indices in `indices`. The [Domain](@ref) markers are translated to `markers` as for [Mesh1D](@ref).
"""
mutable struct MeshnD{D,BackendType<:Backend,CartIndicesType,Mesh1DType<:MeshType{1}} <: MeshType{D}
	markers::MeshMarkers{D}
	indices::CartIndicesType
	const backend::BackendType
	submeshes::NTuple{D,Mesh1DType}
end

@generated function generate_submeshes(Ω::Domain, npts::NTuple{D,Int}, unif::NTuple{D,Bool}, backend) where D
	return :(Base.Cartesian.@ntuple $D i->_mesh(domain(projection(Ω, i)), (npts[i],), (unif[i],), backend))
end

function _mesh(Ω::Domain, npts::NTuple{D,Int}, unif::NTuple{D,Bool}, backend) where D
	@assert dim(Ω) == D

	idxs = generate_indices(npts)
	submeshes = generate_submeshes(Ω, npts, unif, backend)

	mesh_markers = MeshMarkers{D}()
	mesh = MeshnD(mesh_markers, idxs, backend, submeshes)
	set_markers!(mesh, markers(Ω))

	return mesh
end

@inline dim(_::MeshnD{D}) where D = D
@inline dim(::Type{<:MeshnD{D}}) where D = D

@inline eltype(_::MeshnD{D,BackendType}) where {D,BackendType} = eltype(BackendType)
@inline eltype(_::Type{<:MeshnD{D,BackendType}}) where {D,BackendType} = eltype(BackendType)

function show(io::IO, Ωₕ::MeshnD)
	D = dim(Ωₕ)
	properties = ["$(D)D mesh",
		"nPoints: $(npoints(Ωₕ))",
		"Markers: $(keys(Ωₕ.markers))"]

	println(io, join(properties, "\n"))

	print(io, "\nSubmeshes:\n")

	direction = ["x", "y", "z"]
	properties = ["  $(direction[i]) direction | nPoints: $(npoints(Ωₕ, Tuple)[i])" for i in 1:D]

	print(io, join(properties, "\n"))
end

"""
	(Ωₕ::MeshnD)(i)

Returns the `i`-th submesh of `Ωₕ`.
"""
@inline function (Ωₕ::MeshnD)(i)
	@assert 1 <= i <= dim(Ωₕ)
	return Ωₕ.submeshes[i]
end

"""
	points(Ωₕ::MeshnD)
	points(Ωₕ::MeshnD{D}, idx)

Returns a tuple with the points of `Ωₕ`. If the `Tuple` `idx` is passed as the second argument is passed, it returns the tuple with the point corresponding to that index.

  - 2D mesh, with `npts` = (``N_x``, ``N_y``)

```math
([x_i]_{i=1}^{N_x}, [y_j]_{j=1}^{N_y})
```

  - 3D mesh, with `npts` = (``N_x``, ``N_y``, ``N_z``)

```math
([x_i]_{i=1}^{N_x}, [y_j]_{j=1}^{N_y}, [z_l]_{l=1}^{N_z}).
```
"""
@inline @generated points(Ωₕ::MeshnD{D}) where D = :(Base.Cartesian.@ntuple $D i->points(Ωₕ(i)))
@inline @generated points(Ωₕ::MeshnD{D}, idx) where D = :(Base.Cartesian.@ntuple $D i->points(Ωₕ(i), idx[i]))

"""
	points_iterator(Ωₕ::MeshnD)

Returns an iterator over the  points of `Ωₕ`.
"""
@inline @generated points_iterator(Ωₕ::MeshnD{D}) where D = :(Base.Iterators.product((Base.Cartesian.@ntuple $D i->points_iterator(Ωₕ(i)))...))

"""
	half_points(Ωₕ::MeshnD{D}, idx)

Returns a tuple with the [half_points](@ref), for each submesh, of the points at index `idx`.

  - 2D mesh, with `idx` = ``(i,j)``

```math
(x_{i+1/2}, y_{j+1/2})
```

  - 3D mesh, with `idx` = ``(i,j,l)``

```math
(x_{i+1/2}, y_{j+1/2}, z_{l+1/2}).
```
"""
@inline @generated half_points(Ωₕ::MeshnD{D}, idx) where D = :(Base.Cartesian.@ntuple $D i->half_points(Ωₕ(i), idx[i]))

"""
	half_points_iterator(Ωₕ::MeshnD{D})

Returns an iterator for the [half_points](@ref), for each submesh.
"""
@inline @generated half_points_iterator(Ωₕ::MeshnD{D}) where D = :(Base.Iterators.product((Base.Cartesian.@ntuple $D i->half_points_iterator(Ωₕ(i)))...))

"""
	spacing(Ωₕ::MeshnD, idx::NTuple)

Returns a tuple with the [spacing](@ref), for each submesh, at index `idx`.

  - 2D mesh, with `idx` = ``(i,j)``

```math
(h_{x,i}, h_{y,j}) \\vcentcolon = (x_i - x_{i-1}, y_j - y_{j-1})
```

  - 3D mesh, with `idx` = ``(i,j,l)``

```math
(h_{x,i}, h_{y,j}, h_{z,l}) \\vcentcolon = (x_i - x_{i-1}, y_j - y_{j-1}, z_l - z_{l-1})
```
"""
@inline @generated spacing(Ωₕ::MeshnD{D}, idx) where D = :(Base.Cartesian.@ntuple $D i->spacing(Ωₕ(i), idx[i]))

"""
	spacing_iterator(Ωₕ::MeshnD)

Returns an iterator for the [spacing](@ref), for each submesh.
"""
@inline @generated spacing_iterator(Ωₕ::MeshnD{D}) where D = :(Base.Iterators.product((Base.Cartesian.@ntuple $D i->spacing_iterator(Ωₕ(i)))...))

"""
	half_spacing(Ωₕ::MeshnD, idx)

Returns a tuple with the [half_spacing](@ref), for each submesh, at index `idx`.

  - 2D mesh, with `idx` = ``(i,j)``

```math
(h_{x,i+1/2}, h_{y,j+1/2})
```

  - 3D mesh, with `idx` = ``(i,j,l)``

```math
(h_{x,i+1/2}, h_{y,j+1/2}, h_{z,l+1/2})
```
"""
@inline @generated half_spacing(Ωₕ::MeshnD{D}, idx) where D = :(Base.Cartesian.@ntuple $D i->half_spacing(Ωₕ(i), idx[i]))

"""
	half_spacing_iterator(Ωₕ::MeshnD)

Returns an iterator for the [half_spacing](@ref), for each submesh.
"""
@inline @generated half_spacing_iterator(Ωₕ::MeshnD{D}) where D = :(Base.Iterators.product((Base.Cartesian.@ntuple $D i->half_spacing_iterator(Ωₕ(i)))...))

"""
	npoints(Ωₕ::MeshnD)
	npoints(Ωₕ::MeshnD, Tuple)

Returns the number of points of mesh `Ωₕ`. If `Tuple` is passed as the second argument, it returns a tuple with the number of points of each submesh composing `Ωₕ`.
"""
@inline @generated npoints(Ωₕ::MeshnD) = :(prod(npoints(Ωₕ, Tuple)))
@inline @generated npoints(Ωₕ::MeshnD{D}, ::Type{Tuple}) where D = :(Base.Cartesian.@ntuple $D i->npoints(Ωₕ(i)))

"""
	hₘₐₓ(Ωₕ::MeshnD)

Returns the maximum diagonal of mesh `Ωₕ`.

  - 2D mesh

```math
\\max_{i,j} \\Vert (h_{x,i}, h_{y,j}) \\Vert_2
```

  - 3D mesh

```math
\\max_{i,j,l} \\Vert (h_{x,i}, h_{y,j},  h_{z,l}) \\Vert_2
```
"""
@inline function hₘₐₓ(Ωₕ::MeshnD)
	diagonals = Base.Iterators.map(h -> hypot(h...), spacing_iterator(Ωₕ))
	return maximum(diagonals)
end

"""
	cell_measure(Ωₕ::MeshnD, idx)

Returns the measure of the cell ``\\square_{idx}`` centered at the index `idx` (can be a `CartesianIndex` or a `Tuple`):

  - 2D mesh

```math
  \\square_{i,j} \\vcentcolon = \\left[x_i - \\frac{h_{x,i}}{2}, x_i + \\frac{h_{x,i+1}}{2} \\right] \\times \\left[y_j - \\frac{h_{y,j}}{2}, y_j + \\frac{h_{y,j+1}}{2} \\right]
```

with area ``h_{x,i+1/2} h_{y,j+1/2}``, where `idx` = ``(i,j)``,

  - 3D mesh

```math
\\square_{i,j,l} \\vcentcolon = \\left[x_i - \\frac{h_{x,i}}{2}, x_i + \\frac{h_{x,i+1}}{2}\\right] \\times \\left[y_j - \\frac{h_{y,j}}{2}, y_j + \\frac{h_{y,j+1}}{2}\\right] \\times \\left[z_l - \\frac{h_{z,l}}{2}, z_l + \\frac{h_{z,l+1}}{2}\\right]
```

with volume ``h_{x,i+1/2} h_{y,j+1/2} h_{z,l+1/2}``, where `idx` = ``(i,j,l)``.
"""
@inline cell_measure(Ωₕ::MeshnD, idx) = prod(half_spacing(Ωₕ, idx))

"""
	cell_measure_iterator(Ωₕ::MeshnD)

Returns an iterator for the measure of the cells ``\\square_{idx}``
"""
@inline cell_measure_iterator(Ωₕ::MeshnD) = (cell_measure(Ωₕ, idx) for idx in indices(Ωₕ))

"""
	generate_indices(nPoints::NTuple)

Returns the `CartesianIndices` of a mesh with `nPoints[i]` in each direction.
"""
@inline generate_indices(nPoints::NTuple{D,Int}) where D = CartesianIndices(ntuple(i -> 1:nPoints[i], D))

"""
	is_boundary_index(idx, idxs::CartesianIndices)

Returns true if the index `idx` is a boundary index of the mesh with indices stored in `idxs`.
"""
@generated function is_boundary_index(idx, idxs::CartesianIndices{D}) where D
	setup_expr = quote
		dims = size(idxs)
		_idx = CartesianIndex(idx)
	end

	check_expr = :(false)

	for i in 1:D
		dim_check_expr = :(_idx[$i] == 1 || _idx[$i] == dims[$i])
		check_expr = :($check_expr || $dim_check_expr)
	end

	final_expr = quote
		$setup_expr
		return $check_expr
	end

	return final_expr
end

"""
	boundary_indices(Ωₕ::MeshnD)

Returns the boundary indices of mesh `Ωₕ`.
"""
@inline boundary_indices(Ωₕ::MeshnD) = boundary_indices(indices(Ωₕ))
@inline boundary_indices(idxs::CartesianIndices{D}) where D = (idx for idx in idxs if is_boundary_index(idx, idxs))

"""
	interior_indices(Ωₕ::MeshnD)

Returns the interior indices of mesh `Ωₕ`.
"""
@inline interior_indices(Ωₕ::MeshnD) = interior_indices(indices(Ωₕ))
@inline interior_indices(indices::CartesianIndices{D}) where D = CartesianIndices(ntuple(i -> 2:(size(indices)[i] - 1), D))

function iterative_refinement!(Ωₕ::MeshnD{D}, domain_markers::DomainMarkers) where D
	for i in 1:D
		iterative_refinement!(Ωₕ(i))
	end

	npts = npoints(Ωₕ, Tuple)

	idxs = generate_indices(npts)

	set_indices!(Ωₕ, idxs)
	set_markers!(Ωₕ, domain_markers)
end

function change_points!(Ωₕ::MeshnD{D}, domain_markers::DomainMarkers, pts::NTuple{D,VecType}) where {D,VecType}
	for i in 1:D
		change_points!(Ωₕ(i), pts[i])
	end

	set_markers!(Ωₕ, domain_markers)
end