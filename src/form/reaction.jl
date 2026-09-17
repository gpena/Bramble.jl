#=
# reaction.jl

Boundary flux/reaction extraction from a solved constrained problem (gpena/Bramble.jl#227).

## Mathematical background

`dirichlet_bc!` overwrites the constrained rows of `A` and `F` with identity rows, so by
the time a solution `uₕ` exists, the rows that encoded the flux are gone. But the
**unconstrained** operator and load are still recoverable -- `assemble(a)` and
`assemble(l)` with no `dirichlet` keyword give exactly those -- and the residual

```math
r = A u_h - F
```

is (to truncation error) zero on every row the mesh did not constrain, and on a
constrained row is exactly the discrete flux the imposed value had to supply there. This
is the discrete form of the divergence theorem: for `-Δu = f`, `u = g` on `Γ`,
`∫_Ω f = -∮_Γ ∂u/∂n`, so the physical outward flux `q·n = -∂u/∂n` (Fourier's law, unit
conductivity) satisfies `∮_Γ q·n = ∫_Ω f`.

## Sign convention

`reaction` returns `-r` summed over the marker: positive means flux leaving the domain
along the outward normal, so that summing `reaction` over every boundary marker recovers
the net source, `∫_Ω f`, to round-off -- not its negative. This was checked, not assumed:
see the note above `reaction` itself.

## Scaling

`r` already carries the cell-measure weighting `innerₕ`/`inner₊` impose (the same weights
`weights(space, Innerh())` returns), so `reaction`'s net (already-integrated) quantity is
`-r` summed directly, with **no** further rescaling. The pointwise density
([`reaction_density`](@ref)) is the opposite: dividing each marked entry by its own cell
measure recovers a flux **density**, comparable point to point and exportable as one.

## Overlap and components

Marker overlap reuses `_combined_mask` (space/inner_product.jl), the same union used by
`dirichlet_bc!`, so a point shared by two markers is counted once. `components` restricts
a composite space's leaves exactly as `dirichlet_components` does for `assemble`/
`dirichlet_bc!` -- 1-based positions in `leaf_spaces_offsets`, the same order `u(1)`,
`u(2)`, ... addresses.

See also: [`dirichlet_bc!`](@ref), [`assemble`](@ref), [`normal_vector`](@ref).
=#

@inline _reaction_markers(marker::Symbol) = (marker,)
@inline _reaction_markers(marker::NTuple{N, Symbol}) where {N} = marker

function _validate_reaction_spaces(a::BilinearForm, l::LinearForm, uₕ::VectorElement)
    n = ndofs(test_space(l))
    if ndofs(test_space(a)) != n || ndofs(trial_space(a)) != n
        throw(
            ArgumentError(
            "reaction: `a` and `l` must be posed on the same space; got " *
            "$(ndofs(trial_space(a)))×$(ndofs(test_space(a))) for `a` and $n for `l`.",
        ),
        )
    end
    if ndofs(space(uₕ)) != n
        throw(
            ArgumentError(
            "reaction: `uₕ` must be posed on the same space as `a`/`l`; got " *
            "$(ndofs(space(uₕ))) degrees of freedom, expected $n.",
        ),
        )
    end
    return nothing
end

# Walked once per selected leaf, unrolled by recursion over the static `leaf_spaces_offsets`
# tuple -- same idiom as `_apply_conditions!`/`_leaf_entries_impl` (dirichlet_constraints.jl):
# explicit recursion keeps leaf types concrete and avoids a closure allocation for the
# accumulator a `do`-block callback would need.
@inline _reaction_over_leaves(::Tuple{}, r, markers, components, i, acc) = acc
@inline function _reaction_over_leaves(leaves::Tuple, r, markers, components, i, acc)
    sp, offset = first(leaves)
    if _leaf_selected(components, i)
        mask = _combined_mask(mesh(sp), markers)
        acc -= _sum_masked(r, MarkedIndices(mask, offset))
    end
    return _reaction_over_leaves(Base.tail(leaves), r, markers, components, i + 1, acc)
end

@inline function _sum_masked(r::AbstractVector, idxs)
    s = zero(eltype(r))
    @inbounds for i in idxs
        s += r[i]
    end
    return s
end

"""
    reaction(A::AbstractMatrix, F::AbstractVector, uₕ::VectorElement; marker, components = nothing) -> Real

Net flux through the region(s) named by `marker`, from the **unconstrained** operator `A`
and load `F` (`assemble(a)`/`assemble(l)` with no `dirichlet` keyword -- `dirichlet_bc!`
would have already overwritten the rows this needs).

`marker` is a `Symbol` or a `Tuple` of them, exactly as `dirichlet` accepts one or several
labels; overlap between markers is counted once (`_combined_mask`, the same union
`dirichlet_bc!` uses). `components` restricts a composite space's leaves (1-based
positions in `leaf_spaces_offsets`, `nothing` meaning every leaf).

# Sign and scaling

Positive means flux leaving the domain along the outward normal: for `-Δu = f`,
`∮_Γ reaction = ∫_Ω f` to round-off. The residual already carries the cell-measure
weighting the form's inner product imposes, so no further rescaling is applied here --
see [`reaction_density`](@ref) for the pointwise quantity that does.

# Examples

```julia
Ωₕ = mesh(domain(interval(0.0, 1.0)), 41)
Wₕ = gridspace(Ωₕ)
sol(x) = sin(pi * x[1])
src(x) = pi^2 * sin(pi * x[1])

a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
l = form(Wₕ, v -> innerₕ(Rₕ(Wₕ, src), v))
A, F = assemble(a, l; dirichlet = :boundary => sol)

uₕ = element(Wₕ)
uₕ .= A \\ F

reaction(a, l, uₕ; marker = :left)    # ≈ π
reaction(a, l, uₕ; marker = :right)   # ≈ π
```

See also: [`reaction_density`](@ref), [`assemble`](@ref), [`normal_vector`](@ref).
"""
function reaction(
        A::AbstractMatrix, F::AbstractVector, uₕ::VectorElement;
        marker, components = nothing
)
    sp = space(uₕ)
    leaves = leaf_spaces_offsets(sp)
    _validate_dirichlet_components(components, length(leaves))

    r = A * parent(uₕ)
    r .-= F
    return _reaction_over_leaves(
        leaves, r, _reaction_markers(marker), components, 1, zero(eltype(r))
    )
end

"""
    reaction(a::BilinearForm, l::LinearForm, uₕ::VectorElement; marker, dirichlet_components = nothing) -> Real

Reassembles the unconstrained `a`/`l` (`assemble` with no `dirichlet` keyword) and calls
[`reaction`](@ref)`(A, F, uₕ; marker, components = dirichlet_components)`. `a` and `l` must
be posed on the same space as `uₕ`.
"""
function reaction(
        a::BilinearForm, l::LinearForm, uₕ::VectorElement;
        marker, dirichlet_components = nothing
)
    _validate_reaction_spaces(a, l, uₕ)
    A = assemble(a)
    F = assemble(l)
    return reaction(A, F, uₕ; marker = marker, components = dirichlet_components)
end

# Same recursion shape as `_reaction_over_leaves`, writing the per-point density into a
# preallocated `dens` (zero outside the marker) instead of accumulating a scalar.
@inline _reaction_density_over_leaves!(dens, ::Tuple{}, r, markers, components, i) = dens
@inline function _reaction_density_over_leaves!(dens, leaves::Tuple, r, markers, components, i)
    sp, offset = first(leaves)
    if _leaf_selected(components, i)
        mask = _combined_mask(mesh(sp), markers)
        w = weights(sp, Innerh())
        @inbounds for j in MarkedIndices(mask, offset)
            dens[j] = -r[j] / w[j - offset]
        end
    end
    return _reaction_density_over_leaves!(dens, Base.tail(leaves), r, markers, components, i + 1)
end

"""
    reaction_density(A::AbstractMatrix, F::AbstractVector, uₕ::VectorElement; marker, components = nothing) -> VectorElement

The pointwise counterpart of [`reaction`](@ref): a grid function, zero away from `marker`,
equal at each marked point to the physical flux **density** there -- `reaction`'s summand
divided by that point's own cell measure (`weights(space, Innerh())`), so that summing
`density .* weights` over the marker recovers [`reaction`](@ref)'s scalar exactly. Meant to
be exported and plotted, e.g. `export_vtk(filename, Ωₕ, "reaction" => reaction_density(...))`.
"""
function reaction_density(
        A::AbstractMatrix, F::AbstractVector, uₕ::VectorElement;
        marker, components = nothing
)
    sp = space(uₕ)
    leaves = leaf_spaces_offsets(sp)
    _validate_dirichlet_components(components, length(leaves))

    r = A * parent(uₕ)
    r .-= F
    dens = zeros(eltype(r), length(r))
    _reaction_density_over_leaves!(dens, leaves, r, _reaction_markers(marker), components, 1)
    return element(sp, dens)
end

"""
    reaction_density(a::BilinearForm, l::LinearForm, uₕ::VectorElement; marker, dirichlet_components = nothing) -> VectorElement

Reassembles the unconstrained `a`/`l` and calls [`reaction_density`](@ref)`(A, F, uₕ;
marker, components = dirichlet_components)`.
"""
function reaction_density(
        a::BilinearForm, l::LinearForm, uₕ::VectorElement;
        marker, dirichlet_components = nothing
)
    _validate_reaction_spaces(a, l, uₕ)
    A = assemble(a)
    F = assemble(l)
    return reaction_density(A, F, uₕ; marker = marker, components = dirichlet_components)
end
