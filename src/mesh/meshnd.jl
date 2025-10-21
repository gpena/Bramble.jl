"""
	$(TYPEDEF)

Structure to store a cartesian nD-mesh (``2 \\leq n \\leq 3``). For efficiency, the mesh points are not stored. Instead, we store the points of the 1D meshes that make up the nD mesh. To connect both nD and 1D meshes, we use the indices in `indices`. The [DomainMarkers](@ref) are translated to [MeshMarkers](@ref) as for [Mesh1D](@ref).

# Fields

$(FIELDS)
"""
mutable struct MeshnD{D,BT<:Backend,CI,M1T<:AbstractMeshType{1},T} <: AbstractMeshType{D}
	"the D-dimensional CartesianProduct (hyperrectangle) defining the geometric domain."
	set::CartesianProduct{D,T}
	"a dictionary mapping `Symbol` labels to `BitVector`s, marking grid points."
	markers::MeshMarkers
	"the `CartesianIndices` for the full D-dimensional grid, allowing for multi-dimensional indexing."
	indices::CI
	"the computational backend used for linear algebra operations."
	backend::BT
	"a tuple of `D` `Mesh1D` objects, representing the grid along each spatial dimension."
	submeshes::NTuple{D,M1T}
end

"""
	submeshes(Ω::Domain, npts, unif, backend)

Creates the component 1D submeshes for a tensor-product grid.

This function takes a D-dimensional [Domain](@ref) and generates a tuple of `D` independent [Mesh1D](@ref) objects. Each submesh corresponds to one of the spatial dimensions of the original domain.

# Arguments

  - `Ω`: The D-dimensional [Domain](@ref).
  - `npts`: A tuple containing the number of points for each dimension.
  - `unif`: A tuple of booleans indicating if the grid is uniform in each dimension.
  - `backend`: The linear algebra backend.
"""
@inline function submeshes(Ω::Domain, npts, unif, backend)
	# Use ntuple for a type-stable way to generate the tuple of 1D meshes.
	# For each dimension `i` from 1 to D:
	# 1. `projection(Ω, i)` gets the i-th 1D interval from the domain's set.
	# 2. `domain(...)` wraps it in a Domain object.
	# 3. `mesh(...)` creates the corresponding Mesh1D for that dimension.
	return ntuple(i -> mesh(domain(projection(Ω, i)), npts[i], unif[i], backend = backend), Val(dim(Ω)))
end

"""
	_mesh(Ω::Domain, npts, unif, backend)

Internal constructor for a D-dimensional, tensor-product `MeshnD`.

This function orchestrates the creation of a structured multidimensional mesh. It first builds the underlying 1D submeshes for each dimension and then combines them into a single [MeshnD](@ref) object. It also handles the important edge case of "collapsed" dimensions (where an interval is just a point), forcing the number of grid points in that dimension to be 1.

# Arguments

  - `Ω`: The D-dimensional [Domain](@ref) to be meshed.
  - `npts`: An `NTuple{D, Int}` specifying the number of points in each dimension.
  - `unif`: An `NTuple{D, Bool}` specifying if the grid is uniform in each dimension.
  - `backend`: The linear algebra backend.
"""
function _mesh(Ω::Domain, npts::NTuple{D,Int}, unif::NTuple{D,Bool}, backend) where D
	# Ensure the dimension of the domain matches the length of the input tuples.
	@assert dim(Ω) == D "Domain dimension and length of npts/unif do not match."
	_set = set(Ω)

	# Adjust the number of points for any collapsed dimensions. For example, if a domain
	# is a line in 3D space, the two collapsed dimensions will have npts = 1.
	npts_with_collapsed = ntuple(i -> is_collapsed(_set(i)...) ? 1 : npts[i], Val(D))

	# Generate the CartesianIndices for the full D-dimensional grid.
	idxs = generate_indices(npts_with_collapsed)

	# Create the tuple of 1D submeshes that form the basis of the tensor-product grid.
	_submeshes = submeshes(Ω, npts_with_collapsed, unif, backend)

	# Instantiate the MeshnD object with an empty marker dictionary.
	mesh_markers = MeshMarkers()
	output_mesh = MeshnD(_set, mesh_markers, idxs, backend, _submeshes)

	# Now that the mesh object is created, populate its markers based on the domain's markers.
	set_markers!(output_mesh, markers(Ω))

	return output_mesh
end

@inline eltype(::MeshnD{D,BT}) where {D,BT} = eltype(BT)
@inline eltype(::Type{<:MeshnD{D,BT}}) where {D,BT} = eltype(BT)

"""
	(Ωₕ::MeshnD)(i)

Returns the `i`-th submesh of `Ωₕ`.
"""
@inline @inline (Ωₕ::MeshnD)(i) = Ωₕ.submeshes[i]

# A macro for functions of the form: func(Ωₕ) -> ntuple(...)
macro generate_mesh_ntuple_func(fname)
	return esc(quote
				   @inline $fname(Ωₕ::MeshnD{D}) where {D} = ntuple(i -> $fname(Ωₕ(i)), Val(D))
			   end)
end

# A macro for functions of the form: func(Ωₕ, idx) -> ntuple(...)
macro generate_mesh_ntuple_func_with_idx(fname)
	return esc(quote
				   @inline $fname(Ωₕ::MeshnD{D}, idx) where {D} = ntuple(i -> $fname(Ωₕ(i), idx[i]), Val(D))
			   end)
end

# A macro for functions of the form: func(Ωₕ) -> Iterators.product(...)
macro generate_mesh_iterator_func(fname)
	return esc(quote
				   @inline $fname(Ωₕ::MeshnD{D}) where {D} = Iterators.product(ntuple(i -> $fname(Ωₕ(i)), Val(D))...)
			   end)
end

# ntuple wrappers
@generate_mesh_ntuple_func points
@generate_mesh_ntuple_func half_points
@generate_mesh_ntuple_func half_spacings

# ntuple wrappers with an index
@generate_mesh_ntuple_func_with_idx point
@generate_mesh_ntuple_func_with_idx half_point
@generate_mesh_ntuple_func_with_idx spacing
@generate_mesh_ntuple_func_with_idx forward_spacing

@inline half_spacing(Ωₕ::MeshnD{D}, idx) where D = ntuple(i -> _apply_hs_logic(half_spacing(Ωₕ(i), idx[i])), Val(D))

# Iterator wrappers
@generate_mesh_iterator_func points_iterator
@generate_mesh_iterator_func half_points_iterator
@generate_mesh_iterator_func spacings_iterator
@generate_mesh_iterator_func forward_spacings_iterator
@generate_mesh_iterator_func half_spacings_iterator

@inline _apply_hs_logic(value::T) where T = ifelse(iszero(value), one(T), value)

@inline npoints(Ωₕ::MeshnD) = prod(npoints(Ωₕ, Tuple))
@inline npoints(Ωₕ::MeshnD{D}, ::Type{Tuple}) where D = ntuple(i -> npoints(Ωₕ(i)), Val(D))

@inline function hₘₐₓ(Ωₕ::MeshnD)
	diagonals = Iterators.map(h -> hypot(h...), spacings_iterator(Ωₕ))
	return maximum(diagonals)
end

@inline @generated function cell_measure(Ωₕ::MeshnD{D}, idx) where D
	if D == 0
		return :(1)
	end

	mesh_component_expr_1 = Expr(:call, :Ωₕ, 1)
	index_component_expr_1 = Expr(:ref, :idx, 1)
	current_product_expr = Expr(:call, :half_spacing, mesh_component_expr_1, index_component_expr_1)

	for i in 2:D
		mesh_component_expr_i = Expr(:call, :Ωₕ, i)
		index_component_expr_i = Expr(:ref, :idx, i)
		term_i_expr = Expr(:call, :half_spacing, mesh_component_expr_i, index_component_expr_i)

		current_product_expr = Expr(:call, :*, current_product_expr, term_i_expr)
	end

	return current_product_expr
end

@inline cell_measures_iterator(Ωₕ::MeshnD) = (cell_measure(Ωₕ, idx) for idx in indices(Ωₕ))

function iterative_refinement!(Ωₕ::MeshnD{D}) where D
	for i in 1:D
		iterative_refinement!(Ωₕ(i))
	end
	return
end

function iterative_refinement!(Ωₕ::MeshnD{D}, domain_markers::DomainMarkers) where D
	iterative_refinement!(Ωₕ)

	npts = npoints(Ωₕ, Tuple)
	idxs = generate_indices(npts)

	set_indices!(Ωₕ, idxs)
	set_markers!(Ωₕ, domain_markers)
	return
end

function change_points!(Ωₕ::MeshnD{D}, pts) where D
	for i in 1:D
		change_points!(Ωₕ(i), pts[i])
	end
	return
end

function change_points!(Ωₕ::MeshnD{D}, domain_markers::DomainMarkers, pts) where D
	change_points!(Ωₕ, pts)
	set_markers!(Ωₕ, domain_markers)
	return
end