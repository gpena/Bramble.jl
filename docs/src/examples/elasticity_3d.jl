# # Linear elasticity in three dimensions
#
# A beam clamped at one end and sagging under its own weight. It is the first example on this
# site with a *free* surface, and that is what makes it worth writing out: a scheme can be
# second order on a fully clamped problem and still be wrong by two orders of magnitude the
# moment a face is left traction-free. Every number and the plot below were produced by the code
# shown.
#
# ## Problem
#
# A solid occupying ``\Omega \subset \mathbb{R}^3`` deforms under a body force ``f``. Writing
# ``u`` for the displacement,
#
# ```math
# \varepsilon(u) = \tfrac{1}{2}\left(\nabla u + \nabla u^{T}\right), \qquad
# \sigma(u) = 2\mu\,\varepsilon(u) + \lambda\,\mathrm{tr}\,\varepsilon(u)\,I,
# ```
#
# with ``\mu`` and ``\lambda`` the Lamé constants, and equilibrium reads
#
# ```math
# -\nabla\cdot\sigma(u) = f \ \text{ in } \Omega, \qquad
# u = 0 \ \text{ on } \Gamma_{D}, \qquad
# \sigma(u)\,n = 0 \ \text{ on } \partial\Omega \setminus \Gamma_{D}.
# ```
#
# Testing against a function that vanishes on ``\Gamma_D`` and integrating by parts gives the
# energy form the discretisation reproduces. The traction-free condition disappears into it: it
# is the natural boundary condition, imposed by *not* constraining those rows.
#
# ```math
# a(u,v) = \int_\Omega 2\mu\,\varepsilon(u):\varepsilon(v)
#          + \lambda\,(\nabla\cdot u)(\nabla\cdot v) = \int_\Omega f\cdot v.
# ```
#
# ## Why the strain form
#
# There is a more convenient-looking identity,
#
# ```math
# 2\int_\Omega \varepsilon(u):\varepsilon(v) = \int_\Omega \nabla u : \nabla v
#     + \int_\Omega (\nabla\cdot u)(\nabla\cdot v),
# ```
#
# whose right-hand side is a vector Laplacian plus a divergence term, each of which is easy to
# write with the operators the [forms tutorial](../tutorials/form.md) introduces. It holds only
# up to a boundary term. For a fully clamped solid that term vanishes and the two forms are the
# same operator; on a free surface they are different operators, and the convenient one stiffens
# this beam by roughly two orders of magnitude. The strain form is the one to discretise.
#
# ## Discrete problem
#
# Every component of ``u`` sits at the same grid nodes, but a backward difference of a nodal
# function is not nodal: ``D_{-i}u_h`` approximates ``\partial_i u`` halfway between two nodes.
# Write ``M_{-i}`` for the backward average, which moves a quantity half a cell along ``i``, and
# ``(\cdot,\cdot)_S`` for the discrete inner product carrying the weight of a quantity staggered
# in the set of directions ``S``,
#
# ```math
# w_S(i) = \prod_{d \in S} h_d(i_d) \cdot \prod_{d \notin S} h_d(i_d + 1/2).
# ```
#
# The strain and divergence are then placed where a once-differenced quantity belongs, the
# averages bringing the two halves of a shear term to the edge centre they share:
#
# ```math
# \varepsilon^{ii}_h(u_h) = D_{-i} u^i_h, \qquad
# \varepsilon^{ij}_h(u_h) = \tfrac{1}{2}\left(M_{-i} D_{-j} u^i_h + M_{-j} D_{-i} u^j_h\right),
# \qquad
# \mathrm{div}_h\, u_h = \sum_{i} M_{-j} M_{-k} D_{-i} u^i_h,
# ```
#
# with ``\{j,k\}`` the two directions other than ``i``, and the discrete forms are
#
# ```math
# a_h(u_h, v_h) = 2\mu \sum_{i,j} \left(\varepsilon^{ij}_h(u_h), \varepsilon^{ij}_h(v_h)\right)_{S_{ij}}
#   + \lambda \left(\mathrm{div}_h\, u_h,\ \mathrm{div}_h\, v_h\right)_{\{1,2,3\}},
# \qquad
# l_h(v_h) = \sum_i \left(f^i_h, v^i_h\right)_h,
# ```
#
# where ``S_{ij} = \{i\}`` for ``i = j`` and ``\{i,j\}`` otherwise.
#
# ``S = \emptyset`` is [`innerₕ`](@ref), the three singletons are [`inner₊ₓ`](@ref),
# [`inner₊ᵧ`](@ref) and [`inner₊₂`](@ref), and the pairs and the triple that ``\varepsilon^{ij}_h``
# and ``\mathrm{div}_h`` need are the rest of the same [`inner₊`](@ref)`(u, v, Val(S))` family
# ([#234](https://github.com/gpena/Bramble.jl/issues/234)); the transverse-factor bug that used to
# make the singletons wrong on a free surface is fixed too
# ([#236](https://github.com/gpena/Bramble.jl/issues/236)). [`εₕ`](@ref) and [`divₕ`](@ref) build
# the placements above directly from a composite trial or test function — one call each, over
# ``u`` as a whole rather than component by component — placing ``\varepsilon^{ii}_h`` on the face
# centre normal to ``i``, ``\varepsilon^{ij}_h`` on the edge centre the pair ``\{i,j\}`` shares,
# and ``\mathrm{div}_h\,u_h`` on the cell centre every axis shares, and handing each term to
# `inner₊` with the ``S`` it needs. The discrete form is exactly `a_h` above, spelled
# `2μ * inner₊(εₕ(u), εₕ(v)) + λ * inner₊(divₕ(u), divₕ(v))`.

using Bramble
using ForwardDiff
using Random

elasticity_form(Vₕ, μ, λ) = form(
    Vₕ, Vₕ, (u, v) -> 2μ * inner₊(εₕ(u), εₕ(v)) + λ * inner₊(divₕ(u), divₕ(v)))

# The compact form above must assemble to exactly the matrix the 27-term hand expansion it       #src
# replaces would (`test/form/vector_calculus.jl` checks the same equality independently, from     #src
# its own transcription, so a discrepancy here would mean this page's own algebra is wrong, not   #src
# a shared bug). `isapprox` rather than `==`: the two sides reach the same staggered weight       #src
# through different arithmetic -- this one from `SpaceWeights` directly, the hand-expanded one    #src
# from a separately computed ratio -- so the last bit or two of a handful of entries can differ,   #src
# and `atol = 1e-12` is the bound `vector_calculus.jl` found necessary for exactly that reason.    #src
const _Dm = (D₋ₓ, D₋ᵧ, D₋₂)                                                                       #src
const _Mm = (Mₓ, Mᵧ, M₂)                                                                          #src
function _hand_stagger_ratio(Wₕ, S, scale)                                                        #src
    Ωₕ = mesh(Wₕ)                                                                                 #src
    npts = npoints(Ωₕ, Tuple)                                                                     #src
    r = fill(float(scale), npts)                                                                  #src
    for d in S                                                                                    #src
        h = [i == 1 ? 0.0 : spacing(Ωₕ(d), i) for i in 1:npts[d]]                                  #src
        ratio = h ./ [half_spacing(Ωₕ(d), i) for i in 1:npts[d]]                                   #src
        r .*= reshape(ratio, ntuple(k -> k == d ? npts[d] : 1, 3))                                 #src
    end                                                                                            #src
    cₕ = element(Wₕ)                                                                               #src
    copyto!(parent(cₕ), vec(r))                                                                    #src
    return cₕ                                                                                      #src
end                                                                                                 #src
_hand_strain(p, i, j) = i == j ? _Dm[i](p(i)) :                                                    #src
                        0.5 * _Mm[i](_Dm[j](p(i))) + 0.5 * _Mm[j](_Dm[i](p(j)))                    #src
_hand_div_term(p, i) = foldl((op, d) -> _Mm[d](op), filter(!=(i), 1:3); init = _Dm[i](p(i)))       #src
function _hand_elasticity_form(Vₕ, μ, λ)                                                           #src
    Wₕ = first(spaces(Vₕ))                                                                         #src
    cε = Dict(S => _hand_stagger_ratio(Wₕ, S, 2μ)                                                   #src
    for S in ((1,), (2,), (3,), (1, 2), (1, 3), (2, 3)))                                           #src
    cdiv = _hand_stagger_ratio(Wₕ, (1, 2, 3), λ)                                                    #src
    return form(Vₕ, Vₕ,                                                                             #src
        (p, q) -> sum(innerₕ(cε[i == j ? (i,) : minmax(i, j)] * _hand_strain(p, i, j),              #src
                          _hand_strain(q, i, j)) for i in 1:3, j in 1:3) +                                            #src
                  sum(innerₕ(cdiv * _hand_div_term(p, i), _hand_div_term(q, j))                     #src
        for i in 1:3, j in 1:3))                                                                    #src
end                                                                                                 #src
Random.seed!(20260903)                                                                             #src
Ωc_check = mesh(                                                                                    #src
    domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (5, 4, 3), (false, true, false))                #src
Vc_check = gridspace(Ωc_check)^Val(3)                                                               #src
A_compact_check = assemble(elasticity_form(Vc_check, 1.7, 0.9))                                    #src
A_hand_check = assemble(_hand_elasticity_form(Vc_check, 1.7, 0.9))                                  #src
@test isapprox(A_compact_check, A_hand_check; atol = 1.0e-12)                                      #src

lame(E, ν) = (E / (2 * (1 + ν)), E * ν / ((1 + ν) * (1 - 2ν)))
const E, ν = 1.0, 0.3
const μ, λ = lame(E, ν)

function body_force(Vₕ, f)
    (fₕ = element(Vₕ); avgₕ!(fₕ, f);
        form(Vₕ, q -> innerₕ(fₕ(1), q(1)) + innerₕ(fₕ(2), q(2)) + innerₕ(fₕ(3), q(3))))
end
nothing # hide

# ## A manufactured solution
#
# Before the beam, the interior stencils on their own. The right-hand side that belongs to a
# chosen ``u`` is ``f = -\mu\Delta u - (\lambda+\mu)\nabla(\nabla\cdot u)``, and rather than
# differentiate it by hand it is read off the exact solution with `ForwardDiff` — a wrong
# derivative in a manufactured right-hand side produces a convergence rate that looks like a
# discretisation bug.

u₁(x) = sin(π * x[1]) * sin(π * x[2]) * sin(π * x[3])
u₂(x) = sin(2π * x[1]) * sin(π * x[2]) * sin(π * x[3])
u₃(x) = sin(π * x[1]) * sin(2π * x[2]) * sin(π * x[3])
const u_exact = (u₁, u₂, u₃)

function f_exact(x)
    p = [x[1], x[2], x[3]]
    H = ntuple(c -> ForwardDiff.hessian(u_exact[c], p), 3)
    return ntuple(
        i -> -μ * (H[i][1, 1] + H[i][2, 2] + H[i][3, 3]) -
             (λ + μ) * sum(H[j][i, j] for j in 1:3), 3)
end

Random.seed!(20260903)
Ω_cube = domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
Ωc = mesh(Ω_cube, (5, 5, 5), (false, false, false))

hs, err_h, err_1h = Float64[], Float64[], Float64[]
for level in 1:3
    Wc = gridspace(Ωc)
    Vc = Wc^Val(3)
    A, F = assemble(elasticity_form(Vc, μ, λ), body_force(Vc, f_exact);
        dirichlet = dirichlet_constraints(Ω_cube, :boundary => x -> 0.0))
    uc = element(Vc)
    uc .= A \ F
    exact = element(Vc)
    Rₕ!(exact, x -> (u₁(x), u₂(x), u₃(x)))
    e = components(uc .- exact)
    push!(hs, hₘₐₓ(Ωc))
    push!(err_h, sqrt(sum(normₕ(c)^2 for c in e)))
    push!(err_1h, sqrt(sum(norm₁ₕ(c)^2 for c in e)))
    level < 3 && iterative_refinement!(Ωc)
end

order_h = log(err_h[end - 1] / err_h[end]) / log(hs[end - 1] / hs[end])
order_1h = log(err_1h[end - 1] / err_1h[end]) / log(hs[end - 1] / hs[end])
(order_h, order_1h)

# The grid is randomly perturbed and each level is a refinement of the one before, the same
# pattern the [linear Poisson example](poisson_linear.md) uses and for the same reason: a
# uniform grid can make a manufactured solution far more accurate than the scheme deserves, and
# independently drawn random grids give an ``h`` sequence too erratic to read a rate from. A
# non-uniform grid also makes the staggered weights `inner₊` reads for `εₕ`/`divₕ` non-trivial,
# rather than the trivial case a uniform mesh would exercise.
#
# Bracketed above as well as below, for the reason poisson_linear.jl gives. The ``H^1``    #src
# bracket is the looser one: three levels is what 3D affords, so the rate is read off one   #src
# pair of meshes rather than settling over several.                                         #src
@test 1.9 < order_h < 3.0                                                                   #src
@test 1.85 < order_1h < 3.0                                                                 #src
@test 1.0e-4 < err_h[end] < 1.0e-1                                                          #src

#-

include(joinpath(@__DIR__, "..", "convergence_plot.jl")) # hide
convergence_plot([(hs, err_h, "‖·‖ₕ", "#5B5FC7"), (hs, err_1h, "‖·‖₁ₕ", "#0E7C86")];
    title = "3D elasticity, clamped cube") # hide

# ## The cantilever
#
# Now the free surface. A beam ``4 \times 0.6 \times 0.4``, held at ``x = 0`` and nowhere else,
# carrying its own weight. Only the clamped face is named as a marker; every other face is left
# alone, which is what makes it traction-free.

const L, W, H = 4.0, 0.6, 0.4
const ρg = 1.0e-4

Ω = domain(box((0.0, 0.0, 0.0), (L, W, H)), :clamped => :back)

function cantilever(n)
    Ωₕ = mesh(Ω, n, (true, true, true))
    Vₕ = gridspace(Ωₕ)^Val(3)
    A, F = assemble(elasticity_form(Vₕ, μ, λ), body_force(Vₕ, x -> (0.0, 0.0, -ρg));
        dirichlet = dirichlet_constraints(Ω, :clamped => x -> 0.0))
    uₕ = element(Vₕ)
    uₕ .= A \ F
    return Ωₕ, uₕ
end

# Euler–Bernoulli gives a tip deflection ``\delta = qL^4/(8EI)`` for a uniformly loaded
# cantilever, with ``q = \rho g W H`` the weight per unit length and ``I = W H^3/12``. It is
# itself an approximation — it ignores shear and the compliance of the clamped end — so the
# ratio should approach one, not equal it.

δ_eb = (ρg * W * H) * L^4 / (8 * E * (W * H^3 / 12))
tips = Float64[]
for n in ((17, 5, 5), (25, 7, 5), (33, 9, 7))
    Ωₕ, uₕ = cantilever(n)
    nx, ny, nz = npoints(Ωₕ, Tuple)
    push!(tips, reshape(components(uₕ)[3])[nx, (ny + 1) ÷ 2, (nz + 1) ÷ 2])
end
round.(tips ./ (-δ_eb), digits = 3)

# Converging on beam theory from below. Write the shear terms with `Dₕ` and `innerₕ` instead of
# `εₕ`/`divₕ`'s staggered placements — collocating everything at the nodes rather than the
# face/edge/cell centres the discrete strain and divergence actually live on — and the same beam
# on these same three grids gives `+0.00011`, `-0.00005` and `+0.00009`: three to four orders of
# magnitude too small, and on two of the three deflecting *upward* under a downward load. That is
# not a checkerboard or an accuracy loss but an indefinite stiffness matrix, and where it comes
# from is written up in [#236](https://github.com/gpena/Bramble.jl/issues/236).
#
# The sign, then the magnitude: a beam that deflects the wrong way passes any test written  #src
# on `abs`, and was the actual failure mode of the collocated version.                      #src
@test all(<(0), tips)                                                                       #src
@test 0.85 < tips[end] / (-δ_eb) < 1.05                                                     #src
@test issorted(tips; rev = true)   # each refinement deflects further, never back            #src

# ## The deformed solid
#
# The stress to colour it by is recovered at the nodes with [`Dₕₓ`](@ref) and its siblings
# rather than the staggered differences the form uses: `Dₕ` collapses to a one-sided difference
# at the boundary instead of truncating to zero, and the boundary is the part being drawn.

const Dh = (Dₕₓ, Dₕᵧ, Dₕ₂)

function von_mises(uₕ, μ, λ)
    u = components(uₕ)
    G = ntuple(i -> ntuple(j -> parent(Dh[j](u[i])), 3), 3)
    vm = zeros(length(G[1][1]))
    for k in eachindex(vm)
        ε = ntuple(i -> ntuple(j -> 0.5 * (G[i][j][k] + G[j][i][k]), 3), 3)
        tr_ε = ε[1][1] + ε[2][2] + ε[3][3]
        σ = ntuple(i -> ntuple(j -> 2μ * ε[i][j] + (i == j ? λ * tr_ε : 0.0), 3), 3)
        p = (σ[1][1] + σ[2][2] + σ[3][3]) / 3
        vm[k] = sqrt(1.5 * sum((σ[i][j] - (i == j ? p : 0.0))^2 for i in 1:3, j in 1:3))
    end
    return vm
end

Ωₕ, uₕ = cantilever((33, 9, 7))
vm = von_mises(uₕ, μ, λ)
nothing # hide

# Beam theory predicts the largest stress at the clamped end, ``\sigma = Mc/I`` with
# ``M = qL^2/2`` and ``c = H/2``, and the field is very nearly uniaxial there, so von Mises
# should come out close to it. That is an expectation computed from outside the discretisation,
# which is what makes it worth asserting.

σ_beam = ((ρg * W * H) * L^2 / 2) * (H / 2) / (W * H^3 / 12)
round(maximum(vm) / σ_beam, digits = 3)

#-

@test 0.7 < maximum(vm) / σ_beam < 1.2                                                      #src

# Nothing here was ever solved on a deformed domain. Linear elasticity is posed on the
# *reference* configuration — the undeformed box, which is what `mesh` discretises — and the
# unknown is a displacement field over it, three scalar grid functions at the same nodes. Moving
# each node by `u(x)` happens only in the plot below. That identification of the reference and
# deformed configurations is exactly the small-strain assumption, legitimate while
# ``|\nabla u| \ll 1``, which is a claim about this solution rather than a general one:

G = ntuple(i -> ntuple(j -> parent(Dh[j](components(uₕ)[i])), 3), 3)
max_strain = maximum(maximum(abs, 0.5 .* (G[i][j] .+ G[j][i])) for i in 1:3, j in 1:3)
max_rotation = maximum(maximum(abs, 0.5 .* (G[i][j] .- G[j][i])) for i in 1:3, j in 1:3)
(round(max_strain, digits = 4), round(rad2deg(max_rotation), digits = 2))

# Strains near one percent and rotations under five degrees, so the model holds and the picture
# below is not flattering the solution. Load it far enough that either stops being small and the
# model, not the mesh, is what breaks: the repair is a finite-strain formulation, which keeps
# this same Cartesian reference box and puts the nonlinearity into ``E = \tfrac{1}{2}(F^{T}F - I)``.
#
# Asserted, because it is the condition under which the plot means what it appears to mean. #src
@test max_strain < 0.05                                                                     #src
@test max_rotation < 0.15                                                                   #src

# The solid below is drawn where the computation put it, at true scale. A block of cells is
# removed near the clamped end so that the interior plane it bordered is drawn too: the stress
# that matters in a bending beam varies through the *thickness*, tension along the top fibre and
# compression along the bottom, and on an opaque solid none of that reaches the surface.

include(joinpath(@__DIR__, "..", "deformed_plot.jl")) # hide
deformed_plot(uₕ, vm; label = "von Mises σ", scale = 1.0,
    cut = (x, y, z) -> !(x < 0.4L && y < W / 2),
    title = "Cantilever under self-weight") # hide

# The wireframe is the undeformed box. The beam sags about 6% of its length; on the cut face the
# stress runs from a maximum along the top and bottom fibres to a minimum on the neutral axis
# halfway between them, and along the length it falls from the clamped end to nearly nothing at
# the free one. That is what a cantilever does, and none of it was put in by hand — the clamped
# face is the only boundary condition this problem names.
