# precompile/geometry_sessions.jl: geometry constructors and predicates (src/geometry/)
# not exercised by the mesh/space/form sessions. Ordinary methods, so PrecompileTools
# caches them along with everything they call.

function _pc_geometry()
    I = interval(0.0, 1.0)
    R2 = I × interval(0.0, 2.0)
    R3 = R2 × interval(-1.0, 1.0)

    interval(I)
    point(0.5)
    box(0.0, 1.0)
    box((0.0, 0.0), (1.0, 1.0))
    box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

    for X in (I, R2, R3)
        dim(X)
        dim(typeof(X))
        eltype(X)
        eltype(typeof(X))
        topo_dim(X)
        center(X)
        extrema(X)
        set(X)
        point_type(X)
        boundary_symbols(X)
        for i in 1:dim(X)
            X(i)
            extrema(X, i)
            projection(X, i)
        end
        # Both display paths, which are separate methods rather than one flag-switched
        # body (gpena/Bramble.jl#45): the embeddable one-liner and the detailed block.
        sprint(show, X)
        sprint(show, MIME"text/plain"(), X)
    end
    first(I)
    last(I)
    is_collapsed(I)
    is_collapsed(0.0, 1.0)

    0.5 in I
    (0.5,) in I
    [0.5] in I
    (0.5, 1.0) in R2
    [0.5, 1.0] in R2
    return nothing
end
