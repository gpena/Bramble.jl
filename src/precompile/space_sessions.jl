# precompile/space_sessions.jl: grid space construction, the two restriction operators,
# and cross-mesh interpolation (src/space/). Ordinary methods, so PrecompileTools caches
# them along with everything they call.
#
# Rₕ and avgₕ specialize on the caller's function type, so the method instances
# cached for the closures below are never reused by a user's own function. What
# this workload does cache is everything around them, which dominates: the grid
# space construction, the generated Gauss rule, the parallel-for
# skeleton and the vector element arithmetic are all closure-independent. On a
# 21-point 1D space that is a first avgₕ of 1.76 s against 0.02 s, and a first
# grid space of 0.29 s against 0.02 s.
#
# The residual per-closure cost stays. Re-measured 2026-09-04 (point 80), after points
# 51/67 moved Rₕ!/avgₕ! to named kernel structs: roughly 9 ms for Rₕ and 16 ms for avgₕ,
# a genuinely new closure each time — the reverse of the ~50 ms/~10 ms this comment
# claimed before those points landed, most likely because avgₕ! (point 51) and Rₕ!
# (point 67, modelled on it) picked up the kernel-struct fix at different times and
# this number was never revisited after the second one landed. Both are still real,
# irreducible costs: a second call with the identical closure literal at a *different*
# source line still pays the full amount, since Julia keys a closure's type by
# definition site, not text — no workload can warm a closure it does not itself write.
# A type-erasing wrapper around f would remove even that, but it also blocks
# inlining into the quadrature loop and costs about 2x at run time, which is
# the wrong trade for a time-stepping loop. See the note in avgₕ!.
function _pc_space_session(Ωₕ, f, g, marker::Symbol)
    Wₕ = gridspace(Ωₕ)

    # Grid space queries. space_weights runs once per grid space at
    # construction, so it is only reached here through gridspace itself.
    weights(Wₕ)
    weights(Wₕ, Innerh(), 1)
    weights(Wₕ, Innerplus(), 1)
    ndofs(Wₕ)
    dim(Wₕ)
    eltype(Wₕ)
    mesh(Wₕ)
    mesh_type(Wₕ)
    mesh_type(typeof(Wₕ))
    vector_gridspace(Ωₕ)
    Wₕ^2
    Wₕ^Val(2)
    Wₕ × Wₕ

    uₕ = Rₕ(Wₕ, f)
    vₕ = avgₕ(Wₕ, f)
    Rₕ!(uₕ, g)
    avgₕ!(vₕ, g)

    # The masked paths are a separate specialization, not a branch inside the ones above:
    # `project!` wraps the rule's kernel in a `_MaskedKernel` (gpena/Bramble.jl#51), so the
    # sweep is instantiated for a different callable.
    #
    # What these two lines actually buy is small, and measured rather than assumed: three
    # fresh processes each way, a masked first call goes 43.5 ms -> 40.5 ms for `Rₕ!` and
    # 41.8 ms -> 41.4 ms for `avgₕ!` (spreads ±0.2 ms), for about 0.4 s of extra
    # precompilation, itself inside the ~0.8 s run-to-run spread. So: 3 ms for `Rₕ!`, and
    # nothing measurable for `avgₕ!`.
    #
    # The other ~40 ms is irreducible for the reason given above this function: `Rₕ!` and
    # `avgₕ!` specialize on the caller's function type, and no workload can warm a closure
    # it does not itself write. Kept because it is nearly free, not because it is a fix.
    Rₕ!(uₕ, g; markers=(marker,))
    avgₕ!(vₕ, g; markers=(marker,))

    Vₕ = gridspace(Ωₕ, Val(2))
    mesh_type(Vₕ)
    ncomponents(Vₕ)
    spaces(Vₕ)
    component_range(Vₕ, 1)
    component_ranges(Vₕ)
    wₕ = Rₕ(Vₕ, (f, g))
    avgₕ(Vₕ, (f, g))
    wₕ(1)
    reshape(wₕ)
    space_type(uₕ)
    space_type(wₕ)

    # Element constructors, including the scalar fills, which take a separate
    # path from the copy of an existing coefficient vector.
    element(Wₕ)
    aₕ = element(Wₕ, 3.0)
    element(Wₕ, 2)                   # Int fill converts to eltype
    bₕ = element(Wₕ, deepcopy(parent(aₕ)))

    space(bₕ)
    # gpena/Bramble.jl#73: warms `parent`/`copyto!`, the replacements for the deprecated
    # `values`/`values!`, not the deprecated names themselves.
    copyto!(uₕ, parent(bₕ))
    copyto!(uₕ, bₕ)
    copyto!(uₕ, parent(bₕ))

    uₕ[1]
    uₕ[2] = 99.0
    uₕ[3] = 99                       # Int setindex converts

    axes(uₕ)
    reshape(uₕ)
    size(uₕ)
    firstindex(uₕ)
    lastindex(uₕ)

    uₕ .+ vₕ
    2 .* uₕ
    sum(uₕ)
    length(uₕ)
    eltype(uₕ)
    similar(uₕ)
    copy(uₕ)

    # Fused broadcasting, scalar assignment and division each lower differently.
    uₕ .= bₕ .+ aₕ .* 2.0
    uₕ .= 2
    uₕ .= bₕ ./ 2

    return Wₕ, uₕ, wₕ
end

# Cross-mesh interpolation session: pointwise interpolant, in-place and allocating
# projection, sparse matrix assembly, and form-level integration.
function _pc_interpolation_session(Ω_src, Ω_dest)
    W_src = gridspace(Ω_src)
    W_dest = gridspace(Ω_dest)

    u_src = Rₕ(W_src, x -> 1.0)
    u_dest = element(W_dest)

    interpolate_at(u_src, point(Ω_dest, first(indices(Ω_dest))))
    πₕ!(u_dest, u_src)
    πₕ(W_dest, u_src)
    interpolation_matrix(W_dest, W_src)

    assemble(form(W_dest, v -> innerₕ(πₕ(u_src), v)))
    assemble(form(W_src, W_dest, (u, v) -> innerₕ(πₕ(W_src, u), v)))
    return nothing
end
