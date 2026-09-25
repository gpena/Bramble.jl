# skew.jl
# Skew-symmetric split forms for nonlinear advection (gpena/Bramble.jl#165).
#
# The problem this solves. A naive discretisation of a convective term produces a matrix with
# a symmetric part, and that part does work on the solution: `d/dt ‖u‖²_H = -2 uᵀ M u` is not
# zero, so the scheme injects or drains energy that the continuous equation conserves, and a
# nonlinear problem goes unstable for reasons that have nothing to do with the physics.
#
# The usual fix is written as a split: expand the convective product into half its
# conservative form and half its advective one,
#
#     ½ ⟨∇·(w u), v⟩ + ½ ⟨w·∇u, v⟩
#
# and rely on the two halves' discrete boundary terms cancelling. What that split is *for* is
# to make the assembled operator skew-symmetric in the discrete inner product, and that can be
# said directly: the conservative half is, up to those boundary terms, minus the transpose of
# the advective one. Building it as a transpose rather than as a second stencil is exact
# rather than exact-up-to-boundaries, needs no second operator to be written or kept in step
# with the first, and gives `uᵀ M u = 0` algebraically -- at machine precision, on any mesh,
# under any boundary treatment.
#
# The transpose is available inside a form because the two slots of `innerₕ` are what decide
# rows and columns: `innerₕ(A(u), v)` assembles `H·A` and `innerₕ(u, A(v))` assembles
# `(H·A)ᵀ`. Applying the same operator to whichever symbol sits in the slot is the whole
# construction.

"""
    skew_symmetric(A) -> Function
    skew_symmetric(wₕ) -> Function

The skew-symmetric part of the operator `A`, as a form body:

```math
a(u, v) = \\tfrac{1}{2}\\,(A u, v)_h - \\tfrac{1}{2}\\,(u, A v)_h
```

The assembled matrix is ``\\tfrac{1}{2}(HA - (HA)^{\\!\\top})``, exactly skew-symmetric, so
``u^{\\!\\top} M u = 0`` to machine precision and a scheme whose mass matrix is `H` conserves
``\\Vert u \\Vert_h`` -- which is what a split form is written to achieve. It holds on any
mesh and under any boundary treatment, where the conservative/advective split it replaces
holds up to boundary terms that have to cancel.

`A` is a function of one symbolic argument: `skew_symmetric(x -> wₕ * Dcₓ(x))`. Given a grid
function or a tuple of them instead, `A` is the advection operator those coefficients define,
``\\sum_d w_d \\, \\textrm{Dc}_{x_d}``, built on the centered difference because that is the
one whose symmetric part the split is meant to remove.

The result is a two-argument function, which is what [`form`](@ref) takes:

```julia
using Bramble: skew_symmetric
a = form(Wₕ, Wₕ, skew_symmetric(wₕ))
a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + skew_symmetric(wₕ)(u, v))
```

For a nonlinear problem the coefficients are the current iterate, so the form is rebuilt per
step with `wₕ` refreshed; the construction does not care where they came from.

See also: [`innerₕ`](@ref), [`Dcₓ`](@ref)
"""
@inline function skew_symmetric(A)
    return (u, v) -> 0.5 * innerₕ(A(u), v) - 0.5 * innerₕ(u, A(v))
end

@inline skew_symmetric(wₕ::VectorElement) = skew_symmetric(_advection_operator((wₕ,)))
@inline skew_symmetric(wₕ::NTuple{D, VectorElement}) where {D} = skew_symmetric(
    _advection_operator(wₕ)
)

# `∑_d w_d Dc_{x_d}`, with the sum built by `ntuple` over `Val(D)` so each direction stays a
# compile-time literal -- `Dc(x, Val(d))` with `d` from a loop would box it (#146).
@inline function _advection_operator(wₕ::NTuple{D, VectorElement}) where {D}
    return x -> sum(ntuple(d -> wₕ[d] * Dc(x, Val(d)), Val(D)))
end
