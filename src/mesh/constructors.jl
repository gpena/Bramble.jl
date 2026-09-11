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
    mesh(Ω::Domain, npts::NTuple{D, Int}, unif::NTuple{D, Bool}; backend = backend(eltype(Ω))) -> AbstractMeshType{D}
    mesh(Ω::Domain{<:CartesianProduct{1}}, npts::Int, unif::Bool = true; backend = backend(eltype(Ω))) -> Mesh1D
    mesh(Ω::Domain, npts::NTuple{D, Int}; uniform = ntuple(_ -> true, Val(D)), backend = backend(eltype(Ω))) -> MeshnD

Return a [`Mesh1D`](@ref) or [`MeshnD`](@ref) (``D=2,3``) discretizing the [`Domain`](@ref) `Ω`.

# Arguments

  - `Ω`: Continuous domain to discretize.
  - `npts`: Number of grid points along each coordinate direction.
  - `unif`: Boolean flag or tuple of flags specifying whether the point distribution along each axis is uniform.

# Keywords

  - `uniform`: Convenience keyword alternative to positional `unif`. Defaults to `true` across all axes.
  - `backend`: Linear algebra and memory storage [`Backend`](@ref). Defaults to `backend(eltype(Ω))`.
  - `warn_marker_mismatch::Bool = true`: warn if `Ω` carries a custom `:boundary`/`:interior`
    marker that disagrees with this mesh's own geometric one. The custom marker is kept
    either way; set to `false` for a deliberate redefinition you don't want flagged.

# Examples

```julia
I = interval(0.0, 1.0)
Ωₕ = mesh(domain(I), 10)                      # uniform by default
Ωₕ_nonunif = mesh(domain(I), 10, false)       # explicit non-uniform

X = domain(interval(0, 1) × interval(4, 5))
Ωₕ_2d = mesh(X, (10, 15))                     # uniform by default
Ωₕ_mixed = mesh(X, (10, 15), (true, false))
```
"""
@inline mesh(
    Ω::Domain,
    npts::NTuple{D,Int},
    unif::NTuple{D,Bool};
    backend=backend(eltype(Ω)),
    warn_marker_mismatch::Bool=true,
) where {D} = _mesh(Ω, npts, unif, backend; warn_marker_mismatch)
@inline mesh(
    Ω::Domain{CartesianProduct{1,T}},
    npts::Int,
    unif::Bool;
    backend=backend(eltype(Ω)),
    warn_marker_mismatch::Bool=true,
) where {T} = _mesh(Ω, (npts,), (unif,), backend; warn_marker_mismatch)
@inline mesh(
    Ω::Domain{CartesianProduct{1,T}},
    npts::Int;
    uniform::Bool=true,
    backend=backend(eltype(Ω)),
    warn_marker_mismatch::Bool=true,
) where {T} = _mesh(Ω, (npts,), (uniform,), backend; warn_marker_mismatch)
@inline mesh(
    Ω::Domain,
    npts::NTuple{D,Int};
    uniform::NTuple{D,Bool}=ntuple(_ -> true, Val(D)),
    backend=backend(eltype(Ω)),
    warn_marker_mismatch::Bool=true,
) where {D} = _mesh(Ω, npts, uniform, backend; warn_marker_mismatch)
