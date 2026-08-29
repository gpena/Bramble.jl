#===========================================================================#
# Precompilation workload.
#
# PrecompileTools traces inference through the calls below and caches every
# method instance it reaches. The workload is therefore written as a few
# realistic end-to-end sessions rather than an enumeration of individual
# methods: building and querying a mesh over a domain already exercises
# interval, marker, backend and BrambleFunction construction transitively.
#
# Only add a call here when it is NOT reachable from one of those sessions.
#
# To skip the workload while iterating on the package (rebuilds are ~3x
# faster, first use is ~3x slower):
#
#     using Preferences, Bramble
#     set_preferences!(Bramble, "precompile_workload" => false)
#
# Preferences are tracked in the precompile cache, so the change takes effect
# on the next load without any manual cache clearing.
#===========================================================================#

const PRECOMPILE_WORKLOAD = @load_preference("precompile_workload", true)

# --- Helpers ------------------------------------------------------------- #
# These are ordinary methods, so PrecompileTools caches them along with
# everything they call.

# 1D meshes are indexed by a scalar, nD meshes by the index tuple.
function _pc_indexed(Ωₕ::AbstractMeshType{1}, idx_tup)
    i = idx_tup[1]
    point(Ωₕ, i)
    spacing(Ωₕ, i)
    forward_spacing(Ωₕ, i)
    half_spacing(Ωₕ, i)
    half_point(Ωₕ, i)
    cell_measure(Ωₕ, i)
    return nothing
end

function _pc_indexed(Ωₕ::AbstractMeshType, idx_tup)
    point(Ωₕ, idx_tup)
    spacing(Ωₕ, idx_tup)
    forward_spacing(Ωₕ, idx_tup)
    half_spacing(Ωₕ, idx_tup)
    half_point(Ωₕ, idx_tup)
    cell_measure(Ωₕ, idx_tup)
    return nothing
end

# One full pass over the mesh interface: construction, queries, iteration,
# mutation and display.
function _pc_mesh_session(Ω, npts, unif, be, label::Symbol)
    Ωₕ = mesh(Ω, npts, unif; backend = be)

    idx = first(indices(Ωₕ))
    _pc_indexed(Ωₕ, Tuple(idx))
    point(Ωₕ, idx)
    spacing(Ωₕ, idx)
    forward_spacing(Ωₕ, idx)

    dim(Ωₕ)
    dim(typeof(Ωₕ))
    eltype(Ωₕ)
    eltype(typeof(Ωₕ))
    topo_dim(Ωₕ)
    set(Ωₕ)
    backend(Ωₕ)
    markers(Ωₕ)
    npoints(Ωₕ)
    npoints(Ωₕ, Tuple)
    size(Ωₕ)
    length(Ωₕ)
    axes(Ωₕ)
    points(Ωₕ)
    half_points(Ωₕ)
    half_spacings(Ωₕ)
    hₘₐₓ(Ωₕ)
    hₘᵢₙ(Ωₕ)
    is_uniform(Ωₕ)
    is_collapsed(Ωₕ(1))

    indices(Ωₕ)
    boundary_indices(Ωₕ)
    interior_indices(Ωₕ)
    is_boundary_index(Ωₕ, idx)
    index_in_marker(Ωₕ, label)
    boundary_symbol_to_dict(indices(Ωₕ))

    for it in (points_iterator, half_points_iterator, spacings_iterator,
        forward_spacings_iterator, half_spacings_iterator, cell_measures_iterator)
        iter = it(Ωₕ)
        isempty(iter) || first(iter)
    end
    for p in points_iterator(Ωₕ)
        p
    end

    Ωₕ[idx]
    sprint(show, Ωₕ)
    sprint(show, Ωₕ; context = :compact => true)

    return Ωₕ
end

# Mesh mutation is a separate pass so the queries above stay on a pristine mesh.
function _pc_mesh_mutation(Ωₕ, dm)
    iterative_refinement!(deepcopy(Ωₕ))
    iterative_refinement!(deepcopy(Ωₕ), dm)
    pts = points(Ωₕ)
    change_points!(deepcopy(Ωₕ), pts)
    change_points!(deepcopy(Ωₕ), dm, pts)
    return nothing
end

# Backend-facing array construction and the weighted inner products. None of
# this is reachable from mesh construction, which only allocates vectors.
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
    _inner_product(u, w, v)
    _inner_product(fill(1.0, 4, 4), w, fill(2.0, 4, 4))

    _serial_for!(similar(u), 1:4, i -> Float64(i))
    _parallel_for!(similar(u), 1:4, i -> Float64(i))
    return nothing
end

# BrambleFunction call paths. Meshing reaches the 1-argument spatial call via
# condition markers, but not the tuple/vector/splatted forms or the
# time-dependent ones.
function _pc_bramble_function(X, I_time, f, ft, pt)
    bf = embed_function(X, f)
    bft = embed_function(X, I_time, ft)

    if pt isa Number
        bf(pt)
        bf((pt,))
        bf([pt])
    else
        bf(pt)
        bf(pt...)
        bf(collect(pt))
    end
    bft(0.5)(pt)
    bft(pt, 0.5)

    has_time(bf)
    has_time(typeof(bf))
    has_time(bft)
    argstype(bf.wrapped)
    codomaintype(bf.wrapped)
    embed_function(X, bf)
    sprint(show, bf)
    return nothing
end

# Geometry constructors and predicates not exercised by the sessions above.
function _pc_geometry()
    I = interval(0.0, 1.0)
    R2 = I × interval(0.0, 2.0)
    R3 = R2 × interval(-1.0, 1.0)

    interval(I)
    point(0.5)
    cartesian_product(0.0, 1.0)
    cartesian_product(I)
    cartesian_product(((0.0, 1.0),))
    cartesian_product(((0.0, 1.0), (2.0, 3.0)))
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
        tails(X)
        set(X)
        point_type(X)
        get_boundary_symbols(X)
        for i in 1:dim(X)
            X(i)
            tails(X, i)
            projection(X, i)
        end
        sprint(show, X)
    end
    first(I)
    last(I)
    is_collapsed(I)
    is_collapsed(0.0, 1.0)

    0.5 in I
    (0.5,) in I
    [0.5] in I
    SVector(0.5) in I
    (0.5, 1.0) in R2
    [0.5, 1.0] in R2
    SVector(0.5, 1.0) in R2
    return nothing
end

# Grid space construction and the two restriction operators.
#
# Rₕ and avgₕ specialise on the caller's function type, so the method instances
# cached for the closures below are never reused by a user's own function. What
# this workload does cache is everything around them, which dominates: the grid
# space and buffer construction, the generated Gauss rule, the parallel-for
# skeleton and the vector element arithmetic are all closure-independent. On a
# 21-point 1D space that is a first avgₕ of 1.76 s against 0.02 s, and a first
# grid space of 0.29 s against 0.02 s.
#
# The residual per-closure cost stays: roughly 50 ms for Rₕ and 10 ms for avgₕ.
# Embedding f in a BrambleFunction would erase the closure type and remove even
# that, but it also blocks inlining into the quadrature loop and costs about 2x
# at run time, which is the wrong trade for a time-stepping loop. See the note
# in avgₕ!.
function _pc_space_session(Ωₕ, f, g)
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

    uₕ = Rₕ(Wₕ, f)
    vₕ = avgₕ(Wₕ, f)
    Rₕ!(uₕ, g)
    avgₕ!(vₕ, g)

    Vₕ = gridspace(Ωₕ, Val(2))
    wₕ = Rₕ(Vₕ, (f, g))
    avgₕ(Vₕ, (f, g))

    # Element constructors, including the scalar fills, which take a separate
    # path from the copy of an existing coefficient vector.
    element(Wₕ)
    aₕ = element(Wₕ, 3.0)
    element(Wₕ, 2)                   # Int fill converts to eltype
    bₕ = element(Wₕ, deepcopy(values(aₕ)))

    space(bₕ)
    values!(uₕ, values(bₕ))
    copyto!(uₕ, bₕ)
    copyto!(uₕ, values(bₕ))

    uₕ[1]
    uₕ[2] = 99.0
    uₕ[3] = 99                       # Int setindex converts

    axes(uₕ)
    to_matrix(uₕ)
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

# Difference, jump and average operators, and the inner products and norms.
#
# This is the part of the space interface where precompilation pays in full.
# Rₕ and avgₕ specialise on the caller's function type, so most of what a
# workload caches for them is thrown away by a user's own function; an
# operator's method instance is fixed by the element type and the direction
# alone, and an inner product's by the element types, so nothing here is
# closure-dependent and every instance is reused verbatim.
#
# Measured on a step function that applies several operators and then takes
# the inner products and norms of the result, which is the shape a scheme
# actually has: first call 624 ms without this workload against 231 ms with it,
# in 1D and 2D together. Calling each operator and product separately at top
# level, 1.52 s against 254 ms.
#
# It is not free: the workload adds about 11 s to the package's precompile time
# and 8 MB to its cache. Set the `precompile_workload` preference to false, as
# documented at the top of this file, to skip all of it while iterating.
#
# The matrix forms of the operators are deliberately left out: they are on the
# way out of the library, and caching them would grow the image for code that
# is being removed.

# `const` so that each tuple has a concrete type and the loops below stay
# inferable, the same reason the operator config tables are `const`.
const _PC_OPS_X = (diff₋ₓ, diff₊ₓ, D₋ₓ, D₊ₓ, jump₋ₓ, jump₊ₓ, M₋ₓ, M₊ₓ, Dstar₊ₓ, Dcₓ, Dₕₓ)
const _PC_OPS_Y = (diff₋ᵧ, diff₊ᵧ, D₋ᵧ, D₊ᵧ, jump₋ᵧ, jump₊ᵧ, M₋ᵧ, M₊ᵧ, Dstar₊ᵧ, Dcᵧ, Dₕᵧ)
const _PC_OPS_Z = (diff₋₂, diff₊₂, D₋₂, D₊₂, jump₋₂, jump₊₂, M₋₂, M₊₂, Dstar₊₂, Dc₂, Dₕ₂)

# The vectorial aliases, which return a bare element in 1D and a tuple above it,
# so both returns get compiled.
const _PC_OPS_ALL = (∇₋ₕ, ∇₊ₕ, diff₋ₕ, diff₊ₕ, jump₋ₕ, jump₊ₕ, M₋ₕ, M₊ₕ, Dstar₊ₕ, Dcₕ, ∇ₕ)

# Applied with a plain loop over the tuple, which inference unrolls into a static
# call per operator. Going through `foreach` and a closure instead leaves the
# calls dynamically dispatched, and then only the dispatch site is cached and
# every alias costs its ~7 ms again on first use — measured 254 ms against
# 460 ms over the calls in this file.
function _pc_apply_each(ops, uₕ)
    for op in ops
        op(uₕ)
    end
    return nothing
end

# The directional aliases per coordinate. One method per dimension rather than a
# runtime branch, so that inference never sees D₋ᵧ applied to a 1D element.
_pc_directional_ops(uₕ, ::Val{1}) = _pc_apply_each(_PC_OPS_X, uₕ)

function _pc_directional_ops(uₕ, ::Val{2})
    _pc_directional_ops(uₕ, Val(1))
    return _pc_apply_each(_PC_OPS_Y, uₕ)
end

function _pc_directional_ops(uₕ, ::Val{3})
    _pc_directional_ops(uₕ, Val(2))
    return _pc_apply_each(_PC_OPS_Z, uₕ)
end

_pc_vectorial_ops(uₕ) = _pc_apply_each(_PC_OPS_ALL, uₕ)

# innerₕ and the norms built on it take a grid function of a scalar space; inner₊
# and norm₊ take the gradient tuple, which in 1D is the bare element.
#
# innerₕ, normₕ and _dot are all @inline, so no standalone specialisation of
# them exists to be cached: they are inlined into whatever calls them, and in a
# user's program that caller is the user's own method. Calling one at top level
# in a fresh session therefore still costs about 9 ms, and nothing this workload
# can do removes that — it is the cost of building a specialisation for a call
# that was not inlined into a method.
#
# What the calls below do cache is everything non-inline underneath: the
# directional inner-product kernels, the seminorm machinery, and the operators
# that feed them. That is where the saving is, and it is the case that matters,
# since a scheme calls these from inside its own step function rather than from
# the prompt.
function _pc_inner_products(uₕ, dim_val::Val{D}) where {D}
    innerₕ(uₕ, uₕ)
    normₕ(uₕ)
    snorm₁ₕ(uₕ)
    norm₁ₕ(uₕ)

    gₕ = ∇₋ₕ(uₕ)
    inner₊(gₕ, gₕ)
    norm₊(gₕ)
    inner₊(uₕ, uₕ)

    _pc_directional_inner(uₕ, dim_val)
    return nothing
end

function _pc_directional_inner(uₕ, ::Val{1})
    inner₊ₓ(uₕ, uₕ)
    return nothing
end
function _pc_directional_inner(uₕ, ::Val{2})
    inner₊ₓ(uₕ, uₕ)
    inner₊ᵧ(uₕ, uₕ)
    return nothing
end
function _pc_directional_inner(uₕ, ::Val{3})
    _pc_directional_inner(uₕ, Val(2))
    inner₊₂(uₕ, uₕ)
    return nothing
end

# A composite grid function takes a separate dispatch through the operators, and
# its components are scalar grid functions over a contiguous view, which is a
# distinct element type from the one above and so a distinct set of instances.
function _pc_operator_session(uₕ, cₕ, dim_val::Val)
    _pc_directional_ops(uₕ, dim_val)
    _pc_vectorial_ops(uₕ)
    _pc_inner_products(uₕ, dim_val)

    _pc_directional_ops(cₕ, dim_val)
    _pc_vectorial_ops(cₕ)

    kₕ = components(cₕ)[1]
    _pc_directional_ops(kₕ, dim_val)
    _pc_inner_products(kₕ, dim_val)
    return nothing
end

# Buffer pool. Not reachable from a grid space session: gridspace allocates its
# buffers lazily, so the lock, release and grow paths are only compiled here.
function _pc_space_buffers(be)
    vb = vector_buffer(be, 10)
    in_use(vb)
    vector(vb)
    lock!(vb)
    unlock!(vb)

    gsb = simple_space_buffer(be, 10; nbuffers = 2)
    nbuffers(gsb)
    add_buffer!(gsb)

    v, key = vector_buffer(gsb)      # hands back a locked buffer
    other = key == 1 ? 2 : 1
    lock!(gsb, other)
    unlock!(gsb, other)
    unlock!(gsb, key)

    return nothing
end

# --- Workload ------------------------------------------------------------ #

if PRECOMPILE_WORKLOAD
    @setup_workload begin
        be = backend()

        I1 = interval(0.0, 1.0)
        Ω1 = domain(I1, :left => :left, :right => :right)

        S2 = I1 × interval(0.0, 2.0)
        Ω2 = domain(S2, :wall => (:left, :right),
            :blob => x -> (x[1] - 0.5)^2 + (x[2] - 0.5)^2 < 0.25)

        S3 = box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        Ω3 = domain(S3, :boundary => get_boundary_symbols(S3))

        I_time = interval(0.0, 1.0)

        @compile_workload begin
            _pc_geometry()
            _pc_linear_algebra(be)

            # Spatial dimensions 1, 2 and 3, uniform and non-uniform.
            Ωₕ1 = _pc_mesh_session(Ω1, 5, true, be, :left)
            _pc_mesh_session(Ω1, 5, false, be, :left)
            Ωₕ2 = _pc_mesh_session(Ω2, (4, 4), (true, true), be, :wall)
            _pc_mesh_session(Ω2, (4, 4), (false, false), be, :wall)
            Ωₕ3 = _pc_mesh_session(Ω3, (3, 3, 3), (true, true, true), be, :boundary)

            _pc_mesh_mutation(Ωₕ1, markers(Ω1))
            _pc_mesh_mutation(Ωₕ2, markers(Ω2))
            _pc_mesh_mutation(Ωₕ3, markers(Ω3))
            set_points!(deepcopy(Ωₕ1), points(Ωₕ1))

            stepsize(Ωₕ1)
            stepsize(Ωₕ2)
            stepsize(Ωₕ2, 1)
            locate_cell(Ωₕ1, 0.5)
            locate_cell(Ωₕ2, (0.5, 0.5))
            locate_cell(Ωₕ2, [0.5, 0.5])
            normal_vector(Ωₕ1, :left)
            normal_vector(Ωₕ2, :top)
            normal_vector(Ωₕ3, :front)

            # Function embedding, including the time-dependent path.
            _pc_bramble_function(I1, I_time, x -> x + 1.0, (x, t) -> (x + 1.0) * t, 0.5)
            _pc_bramble_function(S2, I_time, x -> x[1] * x[2],
                (x, t) -> x[1] * x[2] * t, (0.5, 1.0))
            _pc_bramble_function(S3, I_time, x -> x[1] + x[2] + x[3],
                (x, t) -> (x[1] + x[2] + x[3]) * t, (0.5, 0.5, 0.5))

            # Grid spaces and restriction operators, in 1D, 2D and 3D.
            _, e1, c1 = _pc_space_session(Ωₕ1, x -> x + 1.0, x -> 2x)
            _, e2, c2 = _pc_space_session(Ωₕ2, x -> x[1] * x[2], x -> x[1] + x[2])
            _, e3, c3 = _pc_space_session(
                Ωₕ3, x -> x[1] * x[2] * x[3], x -> x[1] + x[2] + x[3])

            _pc_operator_session(e1, c1, Val(1))
            _pc_operator_session(e2, c2, Val(2))
            _pc_operator_session(e3, c3, Val(3))
            _pc_space_buffers(be)

            # Markers and domains, including evaluation of a space-time domain.
            for X in (I1, S2)
                m = markers(X, :f => (x -> true), :s => :left)
                d = domain(X, m)
                length(m)
                isempty(m)
                collect(labels(d))
                collect(label_symbols(d))
                collect(label_conditions(d))
                collect(marker_identifiers(d))
                center(d)
                tails(d)
                sprint(show, d)
                sprint(show, m)
            end
            collect(labels(domain(I1, I_time, :moving => ((x, t) -> x > t))(0.5)))
        end
    end
end
