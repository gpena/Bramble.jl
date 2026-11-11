# precompile/parallel_sessions.jl: the Parallel() execution policy, chosen once on the
# backend.
#
# The Parallel() execution policy is chosen once, on the backend, and
# Backend{VT,MT,EP} carries it as a genuine type parameter, so a mesh, grid space or
# element built over a Parallel()-backed backend is a distinct concrete type from every
# Serial()-backed one the sessions above already warmed, sharing no method instance with
# them at all, whether or not the callee itself branches on execution_policy internally.
# A user who calls `backend(policy = Parallel())` therefore pays first-call latency the
# Serial() sessions above never reach.
#
# One 1D session, and in it only the calls that reach a method the Serial() sessions do
# not: the backend, the threaded restriction sweep, and the threaded inner products
# (`_dot(::CpuThreaded, ...)`, `_threaded_dot`, `_static_partials!`, the `_static_or_serial`
# fallback). `markers` takes the masked sweep, `:x` the one direction a 1D gradient has.
# Assembly under Parallel() reaches the methods the Serial() sessions already cached.
function _pc_parallel_policy_session(Ω1, npts::Int)
    be_par = backend(; policy = Parallel())
    Wₕ = gridspace(mesh(Ω1, npts, true; backend = be_par))

    uₕ = Rₕ(Wₕ, x -> x + 1.0)
    innerₕ(uₕ, uₕ)
    inner₊(∇ₕ(uₕ), ∇ₕ(uₕ), :x; markers = (:left,))
    return nothing
end
