@noinline _throw_domain_dim_mismatch(d::Int,
    D::Int) = throw(DimensionMismatch("the domain is $(d)-dimensional but npts and unif have length $D"))

"""
	MeshnD{D,BT,CI,M1T,T}

A structured D-dimensional tensor-product mesh (D ∈ {2,3}).

The mesh is constructed as a Cartesian product of 1D submeshes. Grid points are not
stored explicitly; they are computed on demand from the tensor-product structure.

# Type Parameters

  - `D`: Spatial dimension (2 or 3)
  - `BT <: Backend`: Linear algebra backend
  - `CI`: CartesianIndices type
  - `M1T <: AbstractMeshType{1}`: Type of 1D submeshes (typically Mesh1D)
  - `T`: Element type (Float64, Float32, etc.)

# Fields

$(FIELDS)

# Example

```julia
# Create a 2D mesh with 20×30 grid points
X = domain(interval(0, 1) × interval(0, 2))
Ωₕ = mesh(X, (20, 30), (true, false))

# Access submeshes
x_mesh = Ωₕ(1)  # 1D mesh in x-direction
y_mesh = Ωₕ(2)  # 1D mesh in y-direction

# Get a specific point
point(Ωₕ, (10, 15))  # Returns (x₁₀, y₁₅)
```

See also: [`Mesh1D`](@ref), [`submeshes`](@ref), [`mesh`](@ref)
"""
mutable struct MeshnD{D, BT <: Backend, CI <: CartesianIndices{D}, SM <: Tuple, T} <:
               AbstractMeshType{D}
    "the D-dimensional CartesianProduct (hyperrectangle) defining the geometric domain."
    set::CartesianProduct{D, T}
    "a dictionary mapping `Symbol` labels to `BitVector`s, marking grid points."
    markers::MeshMarkers
    "the `CartesianIndices` for the full D-dimensional grid, allowing for multi-dimensional indexing."
    indices::CI
    "the computational backend used for linear algebra operations."
    backend::BT
    "a tuple of `D` 1D mesh objects, representing the grid along each spatial dimension."
    submeshes::SM
end

"""
	submeshes(Ω::Domain, npts, unif, backend)

Creates the component 1D submeshes for a tensor-product grid.

This function takes a D-dimensional [`Domain`](@ref) and generates a tuple of `D` independent [`Mesh1D`](@ref) objects. Each submesh corresponds to one of the spatial dimensions of the original domain.

# Arguments

  - `Ω`: The D-dimensional [`Domain`](@ref).
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

Builds the 1D submeshes for each dimension and combines them into a single [`MeshnD`](@ref) object. A "collapsed" dimension (an interval that is just a point) forces that dimension's point count to 1.

# Arguments

  - `Ω`: The D-dimensional [`Domain`](@ref) to be meshed.
  - `npts`: An `NTuple{D, Int}` specifying the number of points in each dimension.
  - `unif`: An `NTuple{D, Bool}` specifying if the grid is uniform in each dimension.
  - `backend`: The linear algebra backend.
"""
function _mesh(Ω::Domain, npts::NTuple{D, Int}, unif::NTuple{D, Bool}, backend) where {D}
    # Ensure the dimension of the domain matches the length of the input tuples.
    dim(Ω) == D || _throw_domain_dim_mismatch(dim(Ω), D)
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

@inline eltype(::MeshnD{D, BT}) where {D, BT} = eltype(BT)
@inline eltype(::Type{<:MeshnD{D, BT}}) where {D, BT} = eltype(BT)

"""
	(Ωₕ::MeshnD)(i)

Returns the `i`-th submesh of `Ωₕ`.
"""
@inline function (Ωₕ::MeshnD{D})(i) where {D}
    @boundscheck 1 <= i <= D || throw(BoundsError(Ωₕ.submeshes, i))
    return @inbounds Ωₕ.submeshes[i]
end

#------------------------------------------------------------------------------------------#
# Macros for Boilerplate Reduction
#
# These macros generate functions that apply 1D mesh operations to all submeshes of a
# multidimensional mesh. They eliminate repetitive code and ensure type-stability.
#
# Usage patterns:
# - `@generate_mesh_ntuple_func`: For functions returning tuples of values (one per dimension)
#   Example: points(Ωₕ) returns (x_points, y_points, z_points)
#
# - `@generate_mesh_ntuple_func_with_idx`: For indexed operations returning tuples
#   Example: point(Ωₕ, idx) returns (x[idx[1]], y[idx[2]], z[idx[3]])
#
# - `@generate_mesh_iterator_func`: For functions returning Cartesian product iterators
#   Example: points_iterator(Ωₕ) returns all (x,y,z) combinations
#------------------------------------------------------------------------------------------#

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

"""
	spacings(Ωₕ::MeshnD)

Returns the per-axis backward spacings as an `NTuple{D}` of vectors, where
`spacings(Ωₕ)[d][i]` is [`spacing`](@ref)`(Ωₕ(d), i)`.

See also: [`half_spacings`](@ref), [`cell_measures`](@ref).
"""
@generate_mesh_ntuple_func spacings

# ntuple wrappers with an index
@generate_mesh_ntuple_func_with_idx point
@generate_mesh_ntuple_func_with_idx half_point
@generate_mesh_ntuple_func_with_idx spacing
@generate_mesh_ntuple_func_with_idx forward_spacing

@inline half_spacing(Ωₕ::MeshnD{D}, idx) where {D} = ntuple(
    i -> _apply_hs_logic(half_spacing(Ωₕ(i), idx[i])), Val(D))

"""
	cell_measures(Ωₕ::MeshnD)

Returns the per-axis cell widths as an `NTuple{D}` of vectors. The measure of an
individual cell is the product of its per-axis widths; see [`cell_measure`](@ref).
"""
@inline cell_measures(Ωₕ::MeshnD{D}) where {D} = ntuple(i -> cell_measures(Ωₕ(i)), Val(D))

# Iterator wrappers
@generate_mesh_iterator_func points_iterator
@generate_mesh_iterator_func half_points_iterator
@generate_mesh_iterator_func spacings_iterator
@generate_mesh_iterator_func forward_spacings_iterator
@generate_mesh_iterator_func half_spacings_iterator

@inline npoints(Ωₕ::MeshnD) = prod(npoints(Ωₕ, Tuple))
@inline npoints(Ωₕ::MeshnD{D}, ::Type{Tuple}) where {D} = ntuple(i -> npoints(Ωₕ(i)), Val(D))

# The diagonal of the largest cell. On a tensor-product mesh the spacing along axis d does
# not depend on the other coordinates, and `hypot` is increasing in each argument, so the
# maximum over the whole index set is attained at the per-axis maxima and there is no need
# to visit every cell:
#
#     max_idx ‖(h₁,ᵢ₁, …, h_D,i_D)‖₂ = hypot(hₘₐₓ(Ωₕ(1)), …, hₘₐₓ(Ωₕ(D)))
#
# Each submesh reads its own maximum off its cached spacings, which turns a pass over
# prod(Nd) cells into D lookups.
@inline hₘₐₓ(Ωₕ::MeshnD{D}) where {D} = hypot(ntuple(i -> hₘₐₓ(Ωₕ(i)), Val(D))...)

# The diagonal of the smallest cell, the counterpart of `hₘₐₓ` above and computed the same
# way: `hypot` is increasing in each argument and the per-axis index sets are independent,
# so the minimum over the whole index set sits at the per-axis minima.
#
# This is a diagonal, not an edge length. It used to be the smallest per-axis spacing,
# which made hₘₐₓ and hₘᵢₙ measure two different things: on a 33x33 grid over
# [0,1]x[0,1e-8] the pair read 0.031 and 3.1e-10, one a diagonal and one an edge. For a
# per-axis extent, ask a submesh: `hₘᵢₙ(Ωₕ(i))`.
@inline hₘᵢₙ(Ωₕ::MeshnD{D}) where {D} = hypot(ntuple(i -> hₘᵢₙ(Ωₕ(i)), Val(D))...)

@inline function cell_measure(Ωₕ::MeshnD{D}, idx) where {D}
    return prod(ntuple(i -> half_spacing(Ωₕ(i), idx[i]), Val(D)))
end

@inline Base.getindex(Ωₕ::MeshnD, idx::CartesianIndex) = point(Ωₕ, idx)
@inline Base.getindex(Ωₕ::MeshnD, idx...) = point(Ωₕ, idx)

function locate_cell(Ωₕ::MeshnD{D}, x::Tuple) where {D}
    indices_tuple = ntuple(i -> locate_cell(Ωₕ(i), x[i]), Val(D))
    return CartesianIndex(indices_tuple)
end

@inline cell_measures_iterator(Ωₕ::MeshnD) = (cell_measure(Ωₕ, idx) for idx in indices(Ωₕ))

# The geometric refinement alone, with markers left untouched — shared by both public
# methods below, neither of which wants the *other*'s marker handling as an intermediate
# step of its own.
function _refine_indices!(Ωₕ::MeshnD{D}) where {D}
    @inbounds for i in 1:D
        _refine_indices!(Ωₕ(i))
    end

    # Each submesh regenerated its own indices, but the parent holds a CartesianIndices
    # spanning the whole grid and it has to be rebuilt from the new sizes. Without this
    # the mesh is left inconsistent: npoints reports the refined count while indices
    # still spans the old one, so everything that iterates indices(Ωₕ), which is every
    # restriction and every operator, writes only the old index set and leaves the rest
    # of a grid function holding whatever was in the fresh allocation.
    set_indices!(Ωₕ, generate_indices(npoints(Ωₕ, Tuple)))
    return nothing
end

function iterative_refinement!(Ωₕ::MeshnD{D}) where {D}
    # The old markers dict is sized for the old grid and would otherwise be left silently
    # wrong rather than merely absent: `haskey(markers(Ωₕ), :boundary)` still answers
    # `true`, its BitVector still indexes without erroring (it is shorter than the new
    # point count, not longer), and every point beyond its old length reads as "not
    # boundary" — found by refining a mesh and reassembling a Poisson problem on it, where
    # the boundary rows past the old length never got constrained and the system went
    # singular with no error naming why. `:boundary`/`:interior` need no domain to
    # recompute — they come from the mesh's own shape alone — so they are rebuilt
    # unconditionally here. Anything else the mesh was carrying (a domain's own custom
    # labels) cannot be re-derived without that domain, so it is dropped rather than left
    # stale; the two-argument form re-evaluates those too, and does not route through this
    # method at all, precisely so it never triggers the warning below on its own account.
    old_markers = markers(Ωₕ)
    extra_labels = setdiff(keys(old_markers), (:boundary, :interior))
    isempty(extra_labels) ||
        @warn "iterative_refinement!(Ωₕ) refined a mesh carrying custom markers " *
              "$(Tuple(extra_labels)); those are dropped, not re-evaluated onto the new " *
              "points, because there is no domain here to re-derive them from. Call " *
              "iterative_refinement!(Ωₕ, domain_markers) instead to keep them."

    _refine_indices!(Ωₕ)
    fresh_markers = MeshMarkers()
    _ensure_geometric_markers!(fresh_markers, Ωₕ)
    Ωₕ.markers = fresh_markers
    return nothing
end

function iterative_refinement!(Ωₕ::MeshnD{D}, domain_markers::DomainMarkers) where {D}
    _refine_indices!(Ωₕ)
    # The markers are BitVectors sized to the old grid, so they are rebuilt too.
    set_markers!(Ωₕ, domain_markers)
    return nothing
end

function change_points!(Ωₕ::MeshnD{D}, pts) where {D}
    @inbounds for i in 1:D
        change_points!(Ωₕ(i), pts[i])
    end
    return
end

function change_points!(Ωₕ::MeshnD{D}, domain_markers::DomainMarkers, pts) where {D}
    change_points!(Ωₕ, pts)
    set_markers!(Ωₕ, domain_markers)
    return
end

"""
	Base.copy(Ωₕ::MeshnD)

Creates a copy of the mesh `Ωₕ`. The copy is shallow with respect to the immutable fields
(`set`, `indices`, `backend`), but deep with respect to the mutable data fields
(`submeshes`, `markers`) which are copied.
"""
function Base.copy(Ωₕ::MeshnD{D}) where {D}
    return MeshnD(Ωₕ.set,
        deepcopy(Ωₕ.markers),
        Ωₕ.indices,
        Ωₕ.backend,
        map(copy, Ωₕ.submeshes))
end

"""
	Base.show(io::IO, Ωₕ::MeshnD)

Custom display for MeshnD objects with detailed mesh information and colors.
"""
function Base.show(io::IO, Ωₕ::MeshnD{D, BT, CI, SM, T}) where {D, BT, CI, SM, T}
    pp = PrettyPrinter(io)

    if pp.compact
        # Compact display for arrays/collections
        npts_tuple = npoints(Ωₕ, Tuple)
        print(io, "MeshnD{$(D)D, ", prod(npts_tuple), " pts}")
        return
    end

    # Detailed display
    npts_tuple = npoints(Ωₕ, Tuple)
    n_total = npoints(Ωₕ)
    topodim = topo_dim(Ωₕ)

    # Check if all dimensions are collapsed (topological dimension is 0)
    collapsed = (topodim == 0)

    # Header
    print_mesh_header(pp, "MeshnD", D, T, npts_tuple)
    println(io)

    # Summary line
    print_mesh_summary(pp, npts_tuple, topodim, collapsed)

    # Domain information
    print_mesh_domain_info(pp, set(Ωₕ))

    # Spacing information
    if !collapsed
        uniform_tuple = ntuple(i -> is_uniform(Ωₕ(i)), Val(D))
        print_mesh_spacing_info(pp, uniform_tuple, hₘₐₓ(Ωₕ))
    end

    # Markers information
    print_mesh_markers(pp, markers(Ωₕ))

    # Remove trailing newline
    remove_trailing_newline(io)
end
