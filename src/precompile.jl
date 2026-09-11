#===========================================================================#
# Precompilation workload.
#
# PrecompileTools traces inference through the calls below and caches every
# method instance it reaches. The workload is therefore written as a few
# realistic end-to-end sessions rather than an enumeration of individual
# methods: building and querying a mesh over a domain already exercises
# interval, marker and backend construction transitively.
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
#
# The session functions (`_pc_*`) are ordinary methods, split by subsystem
# under `precompile/` so a given code path or a candidate new session is easy
# to find rather than requiring a scroll through one file. They are included
# unconditionally: only the `@compile_workload` invocation below is gated by
# the preference, since the sessions themselves cost nothing to define.
#===========================================================================#

const PRECOMPILE_WORKLOAD = @load_preference("precompile_workload", true)

include("precompile/utils_sessions.jl")
include("precompile/geometry_sessions.jl")
include("precompile/mesh_sessions.jl")
include("precompile/space_sessions.jl")
include("precompile/operator_sessions.jl")
include("precompile/form_sessions.jl")
include("precompile/parallel_sessions.jl")
include("precompile/exporters_sessions.jl")

# --- Workload ------------------------------------------------------------ #

if PRECOMPILE_WORKLOAD
    @setup_workload begin
        be = backend()

        I1 = interval(0.0, 1.0)
        Ω1 = domain(I1, :left => :left, :right => :right)

        S2 = I1 × interval(0.0, 2.0)
        Ω2 = domain(
            S2,
            :wall => (:left, :right),
            :blob => x -> (x[1] - 0.5)^2 + (x[2] - 0.5)^2 < 0.25,
        )

        S3 = box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        Ω3 = domain(S3, :boundary => boundary_symbols(S3))

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

            # Grid spaces and restriction operators, in 1D, 2D and 3D.
            _, e1, c1 = _pc_space_session(Ωₕ1, x -> x + 1.0, x -> 2x, :left)
            _, e2, c2 = _pc_space_session(Ωₕ2, x -> x[1] * x[2], x -> x[1] + x[2], :wall)
            _, e3, c3 = _pc_space_session(
                Ωₕ3, x -> x[1] * x[2] * x[3], x -> x[1] + x[2] + x[3], :boundary
            )

            _pc_operator_session(e1, c1, Val(1))
            _pc_operator_session(e2, c2, Val(2))
            _pc_operator_session(e3, c3, Val(3))

            # The symbolic layer. Not reachable from the space sessions: a LazyOp tree is
            # built from IdentityOperator and the trial/test leaves, not from a grid
            # function.
            _pc_form_session(
                Ωₕ1, be, :left, x -> x + 1.0, (x, t) -> (x + 1.0) * t, I_time, Val(1)
            )
            _pc_form_session(
                Ωₕ2, be, :wall, x -> x[1] * x[2], (x, t) -> x[1] * x[2] * t, I_time, Val(2)
            )

            # Jacobian sparsity from the AST, scalar and composite, and the per-element-type
            # assembly cache (gpena/Bramble.jl#21/#95/#20). Kept to 1D, the same economy the
            # sessions above already apply.
            Wₕ_pc = gridspace(Ωₕ1)
            Vₕ_pc = gridspace(Ωₕ1, Val(2))
            _pc_jacobian_pattern_session(Wₕ_pc)
            _pc_jacobian_pattern_composite_session(Vₕ_pc)
            _pc_type_cached_assemble_session(Wₕ_pc)

            # The Parallel() execution policy (point 22), otherwise never constructed above.
            _pc_parallel_policy_session(Ω1, 5)

            # Cross-mesh interpolation sessions (1D and 2D).
            Ωₕ1_fine = mesh(Ω1, 9, true; backend=be)
            Ωₕ2_fine = mesh(Ω2, (6, 6), (true, true); backend=be)
            _pc_interpolation_session(Ωₕ1, Ωₕ1_fine)
            _pc_interpolation_session(Ωₕ2, Ωₕ2_fine)

            # Exporters session (PGFPlots 1D and 2D).
            _pc_exporters_session(Ωₕ1, Ωₕ2)

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
                extrema(d)
                sprint(show, d)
                sprint(show, m)
                sprint(show, MIME"text/plain"(), d)
                sprint(show, MIME"text/plain"(), m)
            end
            collect(labels(domain(I1, I_time, :moving => ((x, t) -> x > t))(0.5)))
        end
    end
end
