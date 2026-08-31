"""
	$(TYPEDEF)

A container that stores pre-computed weight vectors for various discrete inner products on a grid space.

This struct holds the diagonal elements (weights) needed to compute different types of inner products, such as those weighted by cell measures or staggered grid spacings. By pre-computing and storing these vectors, numerical simulations can avoid costly recalculations within iterative loops.

# Fields

$(FIELDS)

For a detailed explanation of the mathematical formulas corresponding to these weights, please refer to the documentation for [`ScalarGridSpace`](@ref).
"""
struct SpaceWeights{D, VT <: AbstractVector}
    "weight vector for the standard discrete ``L^2`` inner product (`:innerₕ`), based on cell measures (``|\\square_k|``)."
    innerh::VT
    "a tuple of weight vectors for modified, staggered inner products (`:inner₊ₓ`, `:inner₊ᵧ`, etc.), with one vector for each spatial dimension."
    innerplus::NTuple{D, VT}
end

"""
	$(TYPEDEF)

Represents a function space for **scalar fields** defined on a mesh.

This structure is a cornerstone for numerical simulations, bundling a mesh with pre-computed weights for discrete inner products, lazy-initialized matrices for finite difference operators (like differentiation and averaging), and an efficient memory buffer for temporary vectors.

# Fields

$(FIELDS)

## Discrete Inner Products

The `weights` object stores vectors for different discrete ``L^2`` inner products on the space of grid functions. They are defined as follows:

### - **`:innerₕ`**: The standard discrete ``L^2`` inner product, weighted by the cell measure ``|\\square_k|``.

  - **1D case:**

```math
(u_h, v_h)_h = \\sum_{i=1}^{N_x} |\\square_{i}| u_h(x_i) v_h(x_i)
```

  - **2D case:**

```math
(u_h, v_h)_h = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y} |\\square_{i,j}| u_h(x_i,y_j) v_h(x_i,y_j)
```

  - **3D case:**

```math
(u_h, v_h)_h = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y}\\sum_{l=1}^{N_z} |\\square_{i,j,l}| u_h(x_i,y_j,z_l) v_h(x_i,y_j,z_l)
```

Here, ``|\\cdot|`` denotes the measure of the set (length, area, or volume). See [`cell_measure`](@ref) for details.

### - **`:inner₊`, `:inner₊ₓ`, `:inner₊ᵧ`, `:inner₊₂`**: Modified discrete ``L^2`` inner products, weighted by a mix of forward/backward spacings (``h_k``) and cell widths (``h_{k+1/2}``).

  - **1D case (`:inner₊`):**

```math
(u_h, v_h)_+ = \\sum_{i=1}^{N_x} h_{i} u_h(x_i) v_h(x_i)
```

  - **2D case (`:inner₊ₓ`, `:inner₊ᵧ`):**

```math
(u_h, v_h)_{+x} = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y} h_{x,i} h_{y,j+1/2} u_h(x_i,y_j) v_h(x_i,y_j)
```

```math
(u_h, v_h)_{+y} = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y} h_{x,i+1/2} h_{y,j} u_h(x_i,y_j) v_h(x_i,y_j)
```

  - **3D case (`:inner₊ₓ`, `:inner₊ᵧ`, `:inner₊₂`):**

```math
(u_h, v_h)_{+x} = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y}\\sum_{l=1}^{N_z} h_{x,i} h_{y,j+1/2} h_{z,l+1/2} u_h(x_i,y_j,z_l) v_h(x_i,y_j,z_l)
```

```math
(u_h, v_h)_{+y} = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y}\\sum_{l=1}^{N_z} h_{x,i+1/2} h_{y,j} h_{z,l+1/2} u_h(x_i,y_j,z_l) v_h(x_i,y_j,z_l)
```

```math
(u_h, v_h)_{+z} = \\sum_{i=1}^{N_x}\\sum_{j=1}^{N_y}\\sum_{l=1}^{N_z} h_{x,i+1/2} h_{y,j+1/2} h_{z,l} u_h(x_i,y_j,z_l) v_h(x_i,y_j,z_l)
```
"""
struct ScalarGridSpace{D, T,                               # Dimension and Element Type
    VT <: AbstractVector{T},             # Vector Type
    MType <: AbstractMeshType{D},
    BT <: Backend} <: AbstractSpaceType{1}
    "the underlying mesh of the grid space."
    mesh::MType
    "a [`SpaceWeights`](@ref) object holding vectors for various discrete inner products."
    weights::SpaceWeights{D, VT}
    "a [`GridSpaceBuffer`](@ref) for efficient reuse of temporary vectors, minimizing memory allocations."
    vector_buffer::GridSpaceBuffer{BT, VT, T}
end

"""
	gridspace(Ωₕ::AbstractMeshType{D}; nbuffers::Int = 1) where D

Constructor for a [`ScalarGridSpace`](@ref) defined on the mesh `Ωₕ`. This builds the weights for the inner products mentioned in [`ScalarGridSpace`](@ref) and initializes a memory pool for scratch vectors.
"""
function gridspace(Ωₕ::AbstractMeshType{D}; nbuffers::Int = 1) where {D}
    b = backend(Ωₕ)
    npts = npoints(Ωₕ)

    weights = space_weights(Ωₕ)
    space_buffer = simple_space_buffer(b, npts; nbuffers = nbuffers)

    MType = typeof(Ωₕ)
    T, VT, _, BT = backend_types(b)

    return ScalarGridSpace{D, T, VT, MType, BT}(Ωₕ, weights, space_buffer)
end

# Allocates a work vector sized to a mesh. Typed rather than generic: with an
# untyped signature this also admits spaces, for which npoints has no method.
@inline __vector(Ωₕ::AbstractMeshType) = vector(backend(Ωₕ), npoints(Ωₕ))

# One dimension has no transverse direction, so two of the four full-length vectors the
# general method builds are dead weight: the mean factor is never selected, since `k == i`
# always holds, and the product over a single factor is a copy. Filling the weight vector
# directly drops both, along with the two passes that fill them.
function space_weights(Ωₕ::AbstractMeshType{1})
    innerplus₁ = __vector(Ωₕ)
    _innerplus_weights!(innerplus₁, Ωₕ, 1)

    inner_h_vec = __vector(Ωₕ)
    _innerh_weights!(inner_h_vec, Ωₕ)

    return SpaceWeights{1, typeof(inner_h_vec)}(inner_h_vec, (innerplus₁,))
end

function space_weights(Ωₕ::AbstractMeshType{D}) where {D}
    # Initialize a tuple of D vectors. Each vector will store the final weights for one spatial direction (e.g., x, y, z).
    innerplus = ntuple(i -> __vector(Ωₕ), Val(D))

    # Per-axis factors. Neither depends on the direction `i` being assembled, so each
    # is computed once here rather than D times inside the loop below:
    #   `main[k]`  applies to the axis aligned with the difference direction,
    #   `mean[k]`  applies to every transverse axis.
    main = ntuple(k -> __vector(Ωₕ(k)), Val(D))
    mean = ntuple(k -> __vector(Ωₕ(k)), Val(D))
    for k in 1:D
        _innerplus_weights!(main[k], Ωₕ, k)
        _innerplus_mean_weights!(mean[k], Ωₕ, k)
    end

    # Retrieve the number of grid points in each dimension as a tuple (e.g., (Nx, Ny)).
    npts_tuple = npoints(Ωₕ, Tuple)

    # Assemble the weights for each difference direction `i` by taking the aligned
    # factor on axis `i` and the mean factor on all the others.
    for i in 1:D
        factors = ntuple(k -> k == i ? main[k] : mean[k], Val(D))

        # Create a D-dimensional array view of the flat `innerplus[i]` vector to
        # allow for efficient multidimensional operations.
        v = Base.ReshapedArray(innerplus[i], npts_tuple, ())

        # Combine the per-component factors into the final weight for direction 'i'.
        __innerplus_weights!(v, factors)
    end

    # --- Compute the `inner_h` weights (cell volumes) ---
    inner_h_vec = __vector(Ωₕ)
    _innerh_weights!(inner_h_vec, Ωₕ)

    # Return the computed weights wrapped in a dedicated `SpaceWeights` struct.
    return SpaceWeights{D, typeof(inner_h_vec)}(inner_h_vec, innerplus)
end

# Implementation of the interface functions for AbstractSpaceType
@inline mesh(Wₕ::ScalarGridSpace) = Wₕ.mesh
@inline vector_buffer(Wₕ::ScalarGridSpace) = Wₕ.vector_buffer
@inline backend(Wₕ::ScalarGridSpace) = backend(mesh(Wₕ))
@inline mesh_type(Wₕ::ScalarGridSpace) = typeof(mesh(Wₕ))
@inline mesh_type(::Type{<:ScalarGridSpace{
    <:Any, <:Any, <:Any, MType}}) where {MType} = MType

"""
	weights(Wₕ::ScalarGridSpace)
	weights(Wₕ::ScalarGridSpace, ::InnerProductType)
	weights(Wₕ::ScalarGridSpace, ::InnerProductType, i::Int)

Returns the pre-computed weight vectors for discrete inner products.

The weights are diagonal matrices (stored as vectors) used in computing discrete 
``L^2`` inner products. They represent cell measures or staggered grid spacings.

# Methods

1. `weights(Wₕ)` - Returns the full [`SpaceWeights`](@ref) struct
2. `weights(Wₕ, Innerh())` - Returns weights for standard ``L^2`` inner product (cell volumes)
3. `weights(Wₕ, Innerplus())` - Returns tuple of weights for modified inner products (all directions)
4. `weights(Wₕ, Innerplus(), i)` - Returns weights for modified inner product in direction `i`
5. `weights(Wₕ, Innerh(), i)` - Same as `weights(Wₕ, Innerh())`; the cell measures do not
   depend on a direction, so `i` is accepted and ignored for interface symmetry

# Examples
```julia
Wₕ = gridspace(Ωₕ)

# Get all weights
w = weights(Wₕ)  # Returns SpaceWeights{D, VT}

# Get standard L² weights
w_h = weights(Wₕ, Innerh())  # Vector of cell volumes

# Get modified inner product weights for x-direction
w_plus_x = weights(Wₕ, Innerplus(), 1)  # Vector for x-direction

# Use in inner product
result = dot(uₕ.data, w_h, vₕ.data)  # Weighted inner product
```

See also: [`SpaceWeights`](@ref), [`Innerh`](@ref), [`Innerplus`](@ref), `innerₕ`
"""
@inline weights(Wₕ::ScalarGridSpace) = Wₕ.weights
@inline weights(Wₕ::ScalarGridSpace, ::Innerh) = weights(Wₕ).innerh
@inline weights(Wₕ::ScalarGridSpace, ::Innerplus) = weights(Wₕ).innerplus
@inline weights(Wₕ::ScalarGridSpace, ::Innerh, i) = weights(Wₕ, Innerh())
@inline weights(Wₕ::ScalarGridSpace, ::Innerplus, i) = weights(Wₕ, Innerplus())[i]

"""
	dim(Wₕ::ScalarGridSpace)

Returns the spatial dimension of the function space (1, 2, or 3).

See also: [`ndofs`](@ref), [`mesh`](@ref)
"""
@inline dim(::ScalarGridSpace{D}) where {D} = D
@inline dim(::Type{<:ScalarGridSpace{D}}) where {D} = D

"""
	ndofs(Wₕ::ScalarGridSpace)
	ndofs(Wₕ::ScalarGridSpace, ::Type{Tuple})

Returns the number of degrees of freedom (grid points) in the space.

# Methods
- `ndofs(Wₕ)` - Returns total number of DOFs as an integer
- `ndofs(Wₕ, Tuple)` - Returns DOFs per dimension as a tuple (Nₓ, Nᵧ, Nᵤ)

# Example
```julia
Wₕ = gridspace(Ωₕ)
n = ndofs(Wₕ)        # Total DOFs (e.g., 10000 for 100×100 grid)
dims = ndofs(Wₕ, Tuple)  # Per dimension (e.g., (100, 100))
```

See also: [`npoints`](@ref), [`dim`](@ref)
"""
@inline ndofs(Wₕ::ScalarGridSpace) = npoints(mesh(Wₕ))
@inline ndofs(Wₕ::ScalarGridSpace, ::Type{Tuple}) = npoints(mesh(Wₕ), Tuple)

"""
	eltype(Wₕ::ScalarGridSpace)

Returns the element type of vectors in this space (e.g., `Float64`).

See also: [`backend`](@ref)
"""
@inline eltype(::ScalarGridSpace{D, T}) where {D, T} = T
@inline eltype(::Type{<:ScalarGridSpace{D, T}}) where {D, T} = T

"""
	_innerh_weights!(u, Ωₕ::AbstractMeshType)

Builds the weights for the standard discrete ``L^2`` inner product, ``inner_h(\\cdot, \\cdot)``, on the space of grid functions, following the order of the points provided by `indices(Ωₕ)`. The values are stored in vector `u`.
"""
function _innerh_weights!(u, Ωₕ::AbstractMeshType{1})
    idxs = indices(Ωₕ)
    @inbounds @simd for idx in idxs
        i = idx[1]
        u[i] = cell_measure(Ωₕ, i)
    end
    return nothing
end

function _innerh_weights!(u, Ωₕ::AbstractMeshType{D}) where {D}
    # The submeshes already hold these, so they are read rather than rebuilt: the
    # comprehension this replaces allocated one vector per axis on every call.
    cell_measures_per_component = ntuple(k -> cell_measures(Ωₕ(k)), Val(D))
    dims = npoints(Ωₕ, Tuple)
    v = Base.ReshapedArray(u, dims, ())
    __innerplus_weights!(v, cell_measures_per_component)
    return nothing
end

"""
	_innerplus_weights!(u::VT, Ωₕ, component = 1) where VT

Builds a set of weights based on the spacings, associated with the `component`-th direction, for the modified discrete ``L^2`` inner product on the space of grid functions, following the order of the points provided by `indices(Ωₕ)`. The values are stored in vector `u`.
"""
function _innerplus_weights!(u::VT, Ωₕ, component = 1) where {VT}
    T = eltype(VT)
    mesh_component = Ωₕ(component)

    # These weights are the mesh's backward spacings with the first entry zeroed, which
    # the mesh now caches, so this is a copy rather than a call per point.
    copyto!(u, spacings(mesh_component))

    @inbounds u[1] = zero(T)
    return nothing
end

"""
	_innerplus_mean_weights!(u::VT, Ωₕ, component::Int = 1) where VT

Builds a set of weights based on the half spacings, associated with the `component`-th direction, for the modified discrete ``L^2`` inner product on the space of grid functions, following the order of the [`points`](@ref). The values are stored in vector `u`.
for each component.
"""
function _innerplus_mean_weights!(u::VT, Ωₕ, component::Int = 1) where {VT}
    T = eltype(VT)
    u[1] = zero(T)
    mesh_component = Ωₕ(component)
    N = npoints(mesh_component)

    @inbounds @simd for i in 2:(N - 1)
        u[i] = half_spacing(mesh_component, i)
    end

    @inbounds u[N] = zero(T)
    return nothing
end

@inline function __prod(diags::NTuple{D, Any}, I) where {D}
    return prod(ntuple(i -> @inbounds(diags[i][I[i]]), Val(D)))
end

"""
	__innerplus_weights!(v, innerplus_per_component)

Builds the weights for the modified discrete ``L^2`` inner product on the space of grid functions [`ScalarGridSpace`](@ref). The result is stored in vector `v`.
"""
function __innerplus_weights!(v, innerplus_per_component)
    idxs = CartesianIndices(v)
    f = Base.Fix1(__prod, innerplus_per_component)
    _parallel_for!(v, idxs, f)
end
