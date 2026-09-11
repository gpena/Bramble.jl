# precompile/utils_sessions.jl: backend-facing array construction and the weighted inner
# products (src/utils/). Ordinary methods, so PrecompileTools caches them along with
# everything they call.

# None of this is reachable from mesh construction, which only allocates vectors.
function _pc_linear_algebra(be)
    vector(be, 8)
    vector(be, 0)
    matrix(be, 4, 4)
    matrix(be, 0, 4)
    matrix(be, 4, 0)
    backend_eye(be, 4)
    backend_zeros(be, 4)
    backend_types(be)
    backend_types(typeof(be))
    vector_type(be)
    matrix_type(be)
    eltype(be)
    eltype(typeof(be))

    u = fill(1.0, 4)
    v = fill(2.0, 4)
    w = fill(0.5, 4)
    _dot(u, v, w)
    _dot_masked(u, w, v, BitVector([true, false, true, false]))

    _serial_for!(similar(u), 1:4, i -> Float64(i))
    _cpu_threaded_for!(Serial(), similar(u), 1:4, i -> Float64(i))
    _cpu_threaded_for!(Parallel(), similar(u), 1:4, i -> Float64(i))
    return nothing
end
