# precompile/parallel_sessions.jl: the Parallel() execution policy, chosen once on the
# backend.
#
# The Parallel() execution policy (point 22) is chosen once, on the backend, and
# Backend{VT,MT,EP} carries it as a genuine type parameter — so a mesh, grid space or
# element built over a Parallel()-backed backend is a distinct concrete type from every
# Serial()-backed one the sessions above already warmed, sharing no method instance with
# them at all, whether or not the callee itself branches on execution_policy internally.
#
# Measured before this was added, in a fresh process: a user who calls
# `backend(policy = Parallel())` and does anything with it pays first-call latency the
# Serial() sessions above never reach —
#
#     Rₕ!                 89.0 ms  (vs 15.4 ms warmed Serial)
#     avgₕ!               76.8 ms  (vs 26.1 ms)
#     assemble! (linear)   9.3 ms  (vs 0.016 ms — 580x)
#     assemble (bilinear) 11.5 ms  (vs 0.053 ms — 217x)
#
# One 1D session is enough to specialize the shared, policy-independent cores
# (`Rₕ!`/`avgₕ!`'s kernel structs, `assemble!`/`assemble`'s `execution_policy(...) isa
# Serial` branch resolving the other way, `__innerplus_weights!`'s Parallel method) —
# it is not a second copy of the Serial sessions above, and 2D/3D and the composite/
# Dirichlet/interpolation paths are deliberately left out, the same economy the assembly
# sessions above already apply to 3D.
function _pc_parallel_policy_session(Ω1, npts::Int)
    be_par = backend(; policy=Parallel())
    Ωₕ = mesh(Ω1, npts, true; backend=be_par)
    Wₕ = gridspace(Ωₕ)

    uₕ = Rₕ(Wₕ, x -> x + 1.0)
    vₕ = avgₕ(Wₕ, x -> x + 1.0)
    Rₕ!(uₕ, x -> 2x)
    avgₕ!(vₕ, x -> 2x)

    # Masked under Parallel(): a threaded sweep over a `_MaskedKernel`. These only became
    # threaded sweeps at all in gpena/Bramble.jl#51 -- before it, a masked call silently
    # ran serially whatever the policy said. See the note in `_pc_space_session` for what
    # warming them is worth (about 3 ms on `Rₕ!`, nothing measurable on `avgₕ!`): most of
    # a masked first call is the caller's own closure, which no workload can reach.
    Rₕ!(uₕ, x -> 2x; markers=(:left,))
    avgₕ!(vₕ, x -> 2x; markers=(:left,))

    lf = form(Wₕ, v -> innerₕ(uₕ, v))
    b = zeros(eltype(Wₕ), ndofs(Wₕ))
    assemble!(b, lf)
    assemble(lf)

    bf = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v))
    A = allocate_system_matrix(bf, resolve_form_ast(bf))
    assemble!(A, bf)
    assemble(bf)
    return nothing
end
