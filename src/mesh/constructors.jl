"""
# constructors.jl

Top-level `mesh(Ω, npts, ...)` factory dispatch: the user-facing entry points that parse
positional and keyword arguments and route to each concrete mesh type's `_mesh`
constructor.

See also: [`Mesh1D`](@ref), [`MeshnD`](@ref), [`Domain`](@ref)
"""

#------------------------------------------------------------------------------------------#
# High-Level Mesh Constructor Dispatch
#------------------------------------------------------------------------------------------#

# The backend defaults to one over the domain's own element type rather than always to
# Float64, so a Float32 domain gives a Float32 mesh. The element type is a property of the
# storage, and the storage should follow the geometry it is built on; passing `backend`
# explicitly still overrides it, which is how a mesh gets a type the domain does not have.
"""
    mesh(Ω::Domain, npts::NTuple{D, Int}, unif::Union{Bool, NTuple{D, Bool}}; backend = backend(eltype(Ω))) -> AbstractMeshType{D}
    mesh(Ω::Domain{<:CartesianProduct{D}}, npts::Int, unif::Union{Bool, NTuple{D, Bool}} = true; backend = backend(eltype(Ω))) -> AbstractMeshType{D}
    mesh(Ω::Domain, npts::NTuple{D, Int}; uniform = true, backend = backend(eltype(Ω))) -> MeshnD
    mesh(X::CartesianProduct, npts, args...; kwargs...) -> AbstractMeshType

Return a [`Mesh1D`](@ref) or [`MeshnD`](@ref) (``D=2,3``) discretizing the [`Domain`](@ref) `Ω` or [`CartesianProduct`](@ref) `X`.

When passing a [`CartesianProduct`](@ref) `X` directly, `mesh` automatically constructs `domain(X)`, provisioning default `:boundary` and `:interior` geometric markers.
Passing a single integer `npts::Int` constructs an isotropic grid with `npts` points along every coordinate axis.

# Arguments

  - `Ω`: Continuous domain to discretize.
  - `X`: Continuous geometric set to discretize directly.
  - `npts`: Number of grid points along each coordinate direction, or a single integer for an isotropic grid.
  - `unif`: Boolean flag or tuple of flags specifying whether the point distribution along each axis is uniform.

# Keywords

  - `uniform`: Convenience keyword alternative to positional `unif`. Accepts a `Bool` or an `NTuple{D, Bool}`. Defaults to `true` across all axes.
  - `backend`: Linear algebra and memory storage [`Backend`](@ref). Defaults to `backend(eltype(Ω))`.
  - `warn_marker_mismatch::Bool = true`: warn if `Ω` carries a custom `:boundary`/`:interior`
    marker that disagrees with this mesh's own geometric one. The custom marker is kept
    either way; set to `false` for a deliberate redefinition you don't want flagged.

# Examples

```julia
I = interval(0.0, 1.0)
Ωₕ = mesh(domain(I), 10)                      # uniform by default
Ωₕ_nonunif = mesh(domain(I), 10, false)       # explicit non-uniform

# Direct CartesianProduct input
X = interval(0, 1) × interval(4, 5)
Ωₕ_2d = mesh(X, (10, 15))                     # uniform by default
Ωₕ_iso = mesh(X, 20)                          # isotropic 20x20 grid
Ωₕ_mixed = mesh(X, (10, 15), (true, false))
```
"""
@inline _expand_uniform(u::Bool, ::Val{D}) where {D} = ntuple(_ -> u, Val(D))
@inline _expand_uniform(u::NTuple{D, Bool}, ::Val{D}) where {D} = u

@inline mesh(X::CartesianProduct, args...; kwargs...) = mesh(domain(X), args...; kwargs...)

@inline mesh(
    Ω::Domain,
    npts::NTuple{D, Int},
    unif::Union{Bool, NTuple{D, Bool}};
    backend = backend(eltype(Ω)),
    warn_marker_mismatch::Bool = true
) where {D} = _mesh(Ω, npts, _expand_uniform(unif, Val(D)), backend; warn_marker_mismatch)

@inline mesh(
    Ω::Domain,
    npts::NTuple{D, Int};
    uniform::Union{Bool, NTuple{D, Bool}} = true,
    backend = backend(eltype(Ω)),
    warn_marker_mismatch::Bool = true
) where {D} = _mesh(Ω, npts, _expand_uniform(uniform, Val(D)), backend; warn_marker_mismatch)

@inline mesh(
    Ω::Domain{<:CartesianProduct{D}},
    npts::Int,
    unif::Union{Bool, NTuple{D, Bool}};
    backend = backend(eltype(Ω)),
    warn_marker_mismatch::Bool = true
) where {D} = mesh(Ω, ntuple(_ -> npts, Val(D)), _expand_uniform(unif, Val(D)); backend, warn_marker_mismatch)

@inline mesh(
    Ω::Domain{<:CartesianProduct{D}},
    npts::Int;
    uniform::Union{Bool, NTuple{D, Bool}} = true,
    backend = backend(eltype(Ω)),
    warn_marker_mismatch::Bool = true
) where {D} = mesh(
    Ω, ntuple(_ -> npts, Val(D)); uniform = _expand_uniform(uniform, Val(D)), backend, warn_marker_mismatch)
