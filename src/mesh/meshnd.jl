@noinline _throw_domain_dim_mismatch(d::Int,
    D::Int) = throw(DimensionMismatch("the domain is $(d)-dimensional but npts and unif have length $D"))

"""
    MeshnD{D, BT, CI, SM, T} <: AbstractMeshType{D}

Structured multi-dimensional tensor-product mesh for spatial dimensions ``D \\in \\{2, 3\\}``.

Constructed as a Cartesian product of 1D submeshes ([`Mesh1D`](@ref)). Coordinate points
are evaluated on demand from the tensor-product submeshes.

# Type parameters

  - `D`: Spatial dimension (2 or 3).
  - `BT <: Backend`: Computational linear algebra backend.
  - `CI <: CartesianIndices{D}`: Cartesian index space.
  - `SM <: Tuple`: Tuple of 1D submeshes (`Mesh1D`).
  - `T`: Coordinate element type (`Float64`, `Float32`, etc.).

# Fields

  - `set`: Multi-dimensional geometric [`CartesianProduct`](@ref) domain.
  - `markers`: [`MeshMarkers`](@ref) dictionary mapping symbols to `BitVector` indicators.
  - `indices`: Multi-dimensional `CartesianIndices{D}` for the grid.
  - `backend`: Linear algebra [`Backend`](@ref).
  - `submeshes`: Tuple of `D` [`Mesh1D`](@ref) objects along each coordinate axis.

# Examples

```julia
# Create a 2D mesh with 20×30 grid points
X = domain(interval(0, 1) × interval(0, 2))
Ωₕ = mesh(X, (20, 30), (true, false))

# Access submeshes
x_mesh = Ωₕ(1)  # 1D mesh along x-axis
y_mesh = Ωₕ(2)  # 1D mesh along y-axis

# Query a specific point coordinate
point(Ωₕ, (10, 15))  # returns (x₁₀, y₁₅)
```

See also: [`Mesh1D`](@ref), [`submeshes`](@ref), [`mesh`](@ref).
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
    submeshes(Ω::Domain, npts, unif, backend) -> NTuple{D, Mesh1D}

Create the component 1D submeshes for a tensor-product grid.

Generates a tuple of `D` independent [`Mesh1D`](@ref) objects corresponding to each coordinate axis of `Ω`.

# Arguments

  - `Ω`: Multi-dimensional continuous [`Domain`](@ref).
  - `npts`: Number of points along each dimension.
  - `unif`: Flags indicating whether each axis is uniformly partitioned.
  - `backend`: Computational linear algebra [`Backend`](@ref).
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
    _mesh(Ω::Domain, npts::NTuple{D, Int}, unif::NTuple{D, Bool}, backend) -> MeshnD{D}

Internal constructor for multi-dimensional tensor-product mesh [`MeshnD`](@ref).

Builds the 1D submeshes along each axis and combines them into a [`MeshnD`](@ref). Collapsed
dimensions (degenerate single-point intervals) are forced to a point count of 1.

# Arguments

  - `Ω`: Multi-dimensional continuous [`Domain`](@ref) to discretize.
  - `npts`: Number of points in each spatial dimension.
  - `unif`: Uniformity flags for each spatial dimension.
  - `backend`: Linear algebra [`Backend`](@ref).
"""
function _mesh(Ω::Domain, npts::NTuple{D, Int}, unif::NTuple{D, Bool}, backend;
        warn_marker_mismatch::Bool = true) where {D}
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
    set_markers!(output_mesh, markers(Ω); warn_marker_mismatch)

    return output_mesh
end

@inline eltype(::MeshnD{D, BT}) where {D, BT} = eltype(BT)
@inline eltype(::Type{<:MeshnD{D, BT}}) where {D, BT} = eltype(BT)

"""
    (Ωₕ::MeshnD)(i::Integer) -> Mesh1D

Return the `i`-th 1D submesh of `Ωₕ` along coordinate axis `i`.
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
    spacings(Ωₕ::MeshnD{D}) -> NTuple{D, AbstractVector}

Return the per-axis backward spacings as an `NTuple{D}` of vectors, where
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
    cell_measures(Ωₕ::MeshnD{D}) -> NTuple{D, AbstractVector}

Return the per-axis cell widths as an `NTuple{D}` of vectors. The measure of an
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
    # Routed through the coerced, tuple-returning `half_spacing(::MeshnD, idx)` above
    # (not the raw per-submesh `half_spacing(Ωₕ(i), idx[i])`), so a collapsed axis's
    # zero half-spacing is replaced by `_apply_hs_logic` before the product, not left
    # to make the whole cell measure zero.
    return prod(half_spacing(Ωₕ, idx))
end

@inline Base.getindex(Ωₕ::MeshnD, idx::CartesianIndex) = point(Ωₕ, idx)
@inline Base.getindex(Ωₕ::MeshnD, idx...) = point(Ωₕ, idx)

function locate_cell(Ωₕ::MeshnD{D}, x::Tuple) where {D}
    indices_tuple = ntuple(i -> locate_cell(Ωₕ(i), x[i]), Val(D))
    return CartesianIndex(indices_tuple)
end

@inline cell_measures_iterator(Ωₕ::MeshnD) = (cell_measure(Ωₕ, idx) for idx in indices(Ωₕ))

# The geometric refinement alone, with markers left untouched: shared by both public
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

# A `MeshnD` always has something to refine (each axis handles its own collapse
# independently, inside `_refine_indices!` above); the rest of the refinement/
# `change_points!` plumbing is shared with `Mesh1D`, in `mesh/interface.jl`
# (gpena/Bramble.jl#68). `_nothing_to_refine`'s default (`interface.jl`) already answers
# `false` for any mesh type that does not override it, so no override is needed here.

function change_points!(Ωₕ::MeshnD{D}, pts) where {D}
    @inbounds for i in 1:D
        change_points!(Ωₕ(i), pts[i])
    end
    return
end

"""
    Base.copy(Ωₕ::MeshnD{D}) -> MeshnD{D}

Create a copy of mesh `Ωₕ`. The copy is shallow with respect to immutable fields
(`set`, `indices`, `backend`), but deep with respect to mutable data fields
(`submeshes`, `markers`).
"""
function Base.copy(Ωₕ::MeshnD{D}) where {D}
    return MeshnD(Ωₕ.set,
        deepcopy(Ωₕ.markers),
        Ωₕ.indices,
        Ωₕ.backend,
        map(copy, Ωₕ.submeshes))
end

"""
    Base.show(io::IO, Ωₕ::MeshnD) -> Nothing

Custom display for `MeshnD` objects with detailed mesh summary, domain information, and markers.
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
