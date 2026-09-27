module SpaceAutodiffPoliciesTests

using Test
using Bramble
using Bramble: D₋ₓ, CpuPolyester, ast_sparsity_detector
using Random
using Polyester: @batch
using ForwardDiff
using DifferentiationInterface
using SparseConnectivityTracer: TracerSparsityDetector
using SparseMatrixColorings: GreedyColoringAlgorithm
using ..TestUtils: _fd, _have

# Which AD backend differentiates through Bramble under which CPU execution policy, dense
# and sparse. autodiff_backends.jl and autodiff_heavy.jl survey the backends on the default
# `Serial()` mesh only; this file crosses them with `Parallel()` and `CpuPolyester()`.
#
# The matrices below are measurements, not aspirations: every cell asserts what it was
# observed to do on four threads, `:wrong` included. A `:throws` cell that starts passing, or a `:works` cell
# that starts failing, means a backend or a sweep changed, and the matrix needs updating.
#
# Why the cells come out the way they do:
#
#   ForwardDiff, PolyesterForwardDiff  work everywhere. A `Dual` is a plain value, so each
#                         thread writes its own entries and nothing is shared.
#   ReverseDiff           throws under both threaded policies. Every tracked operation is
#                         pushed onto one shared tape, and two threads resizing it at once
#                         raise `ConcurrencyViolationError`.
#   Mooncake              throws under both. It cannot derive a rule through the `try`/`catch`
#                         in `_static_or_serial` (`CpuThreaded`), nor through the intrinsics
#                         Polyester's `@batch` lowers to (`CpuPolyester`).
#   Enzyme                works on operators and linear forms under `CpuThreaded`, and
#                         returns a wrong gradient under `CpuPolyester` without raising
#                         anything -- the one cell here that fails silently. A bilinear form
#                         over `Mₕ`-averaged coefficients (the sparse-Jacobian residual below)
#                         kills the process under both: an LLVM verifier failure in Enzyme's
#                         derivative of `_static_bands!` under `CpuThreaded`, a segfault in
#                         `_batch_bilinear_band_replay!` under `CpuPolyester`. Everything
#                         works under `Serial()`.
#
# Sparse AD is not a backend of its own: `AutoSparse` wraps a dense backend with a sparsity
# detector and a coloring. The detector side works under every policy -- `TracerSparsityDetector`'s
# tracers are plain values like `Dual`s, and `ast_sparsity_detector` never runs the residual
# at all -- so a sparse cell's verdict is its inner backend's.
#
# Two outcomes cannot be observed in this process, and run in a child `julia` instead:
#
#   `:crashes`       the child dies before it returns or throws.
#   `:throws_fresh`  the first call in a fresh process throws. It cannot run here because a
#                    task failing inside `@batch` leaves Polyester serial for the rest of the
#                    process: measured, `@batch` used threads 1-4 before one failed
#                    ReverseDiff call and thread 1 only after it. Every later `CpuPolyester`
#                    test would still pass, having quietly stopped threading. The canary at
#                    the end of this file checks that nothing here did that.
#
# `GpuKernel` has no column: nothing in the device extensions handles a `Dual` or a tracked
# type, and `pde_solve`'s rules take a `SparseMatrixCSC` only, so there is no device AD path
# to measure yet.

# The problems, and the backends differentiating them, are one quoted block so the child
# process runs exactly the code this process does, rather than a second copy of it.
const _PROBLEMS = quote
    # A seeded non-uniform mesh, reseeded per call, so every policy differentiates the same
    # grid as the `Serial()` reference.
    function dense_problems(policy)
        Random.seed!(1234)
        Ωₕ = mesh(
            domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (41, 41), (false, false);
            backend = backend(policy = policy)
        )
        Wₕ = gridspace(Ωₕ)
        uₕ(p) = Rₕ(Wₕ, x -> p[1] * sin(x[1]) + p[2] * x[2]^2)
        return (
            derivative = a -> sum(parent(D₋ₓ(Rₕ(Wₕ, x -> a * sin(x[1]) * x[2])))),
            gradient = p -> sum(parent(D₋ₓ(uₕ(p)))) + normₕ(uₕ(p))^2,
            assembly = p -> sum(abs2, assemble(form(Wₕ, v -> innerₕ(uₕ(p), v))))
        )
    end

    const POINTS = (derivative = 1.3, gradient = [1.3, 0.7], assembly = [1.3, 0.7])

    differentiate(name, f, ad, x) = name === :derivative ?
                                    DifferentiationInterface.derivative(f, ad, x) :
                                    DifferentiationInterface.gradient(f, ad, x)

    # The residual test/ext/sparse_ad_ext.jl drives Newton with: `A(u) u - F` for
    # `α(u) = 3 + 1/(1+u²)`, whose Jacobian is what a sparse backend exists to compute.
    policy_α(u) = 3 + 1 / (1 + u^2)

    function nonlinear_problem(policy)
        Random.seed!(1234)
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 300, false; backend = backend(policy = policy))
        Wₕ = gridspace(Ωₕ)
        gₕ = element(Wₕ)
        avgₕ!(gₕ, x -> exp(x[1]))
        F = assemble(form(Wₕ, v -> innerₕ(gₕ, v)))
        function diffusion_form(uₕ)
            αv = policy_α.(Mₕ(uₕ))
            return form(Wₕ, Wₕ, (U, V) -> inner₊(αv * ∇ₕ(U), ∇ₕ(V)))
        end
        function residual(u::AbstractVector{T}) where {T}
            uₕ = element(Wₕ, T)
            uₕ .= u
            return assemble(diffusion_form(uₕ); dirichlet = :boundary) * u .- F
        end
        return (
            residual = residual,
            form = diffusion_form(element(Wₕ, 0.0)),
            u = [0.1 + 0.01 * i for i in 1:ndofs(Wₕ)]
        )
    end

    detectors(prob) = (
        tracer = TracerSparsityDetector(),
        ast = ast_sparsity_detector(prob.form, U -> Mₕ(U))
    )

    sparse_backend(inner, detector) = AutoSparse(
        inner; sparsity_detector = detector, coloring_algorithm = GreedyColoringAlgorithm()
    )

    # Enzyme's two annotations are autodiff_heavy.jl's, for the reasons given there.
    make_backends() = (
        ForwardDiff = AutoForwardDiff(),
        PolyesterForwardDiff = AutoPolyesterForwardDiff(),
        ReverseDiff = AutoReverseDiff(),
        Mooncake = AutoMooncake(config = nothing),
        Enzyme = AutoEnzyme(
            mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
            function_annotation = Enzyme.Const
        )
    )
end

const _BACKEND_PACKAGES = (:ReverseDiff, :Mooncake, :Enzyme, :PolyesterForwardDiff)
const _HAVE_BACKENDS = all(_have, _BACKEND_PACKAGES)

if _HAVE_BACKENDS
    for pkg in _BACKEND_PACKAGES
        @eval import $pkg
    end
end

Core.eval(@__MODULE__, _PROBLEMS)

const POLICIES = (CpuSerial = Serial(), CpuThreaded = Parallel(), CpuPolyester = CpuPolyester())

const DENSE = (
    CpuSerial = (
        ForwardDiff = :works, PolyesterForwardDiff = :works, ReverseDiff = :works,
        Mooncake = :works, Enzyme = :works
    ),
    CpuThreaded = (
        ForwardDiff = :works, PolyesterForwardDiff = :works, ReverseDiff = :throws,
        Mooncake = :throws, Enzyme = :works
    ),
    CpuPolyester = (
        ForwardDiff = :works, PolyesterForwardDiff = :works, ReverseDiff = :throws_fresh,
        Mooncake = :throws, Enzyme = :wrong
    )
)

# The same verdict for both detectors, measured for each.
const SPARSE = (
    CpuSerial = (ForwardDiff = :works, ReverseDiff = :works, Mooncake = :works, Enzyme = :works),
    CpuThreaded = (
        ForwardDiff = :works, ReverseDiff = :throws, Mooncake = :throws, Enzyme = :crashes
    ),
    CpuPolyester = (
        ForwardDiff = :works, ReverseDiff = :throws_fresh, Mooncake = :throws, Enzyme = :crashes
    )
)

# How many distinct threads one `@batch` loop runs on.
function _polyester_threads()
    seen = zeros(Int, 64 * Threads.nthreads())
    @batch for i in eachindex(seen)
        seen[i] = Threads.threadid()
    end
    return length(unique(seen))
end

# Runs `body` after `_PROBLEMS` in a fresh `julia` with this process's project and thread
# count. `LOADED` proves the child got as far as the cell, so a child that fails to load
# cannot pass for one that crashed in it.
function _run_child(body::String)
    script = """
    using Bramble, Random, ForwardDiff, DifferentiationInterface
    using Bramble: D₋ₓ, CpuThreaded, CpuPolyester, ast_sparsity_detector
    using Polyester: @batch
    using SparseConnectivityTracer: TracerSparsityDetector
    using SparseMatrixColorings: GreedyColoringAlgorithm
    import ReverseDiff, Mooncake, Enzyme, PolyesterForwardDiff
    $(_PROBLEMS)
    println("LOADED")
    flush(stdout)
    try
        $(body)
        println("RETURNED")
    catch
        println("THREW")
    end
    """
    cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) --startup-file=no
           -t $(Threads.nthreads()) -e $script`
    out = IOBuffer()
    exited = success(pipeline(cmd; stdout = out, stderr = devnull))
    return exited, String(take!(out))
end

function _check_child(outcome, body)
    exited, out = _run_child(body)
    @test occursin("LOADED", out)
    @test !occursin("RETURNED", out)
    if outcome === :throws_fresh
        @test exited
        @test occursin("THREW", out)
    else
        @test !exited
        @test !occursin("THREW", out)
    end
end

# Three calls per `:works` cell: a race can come out right once. A `:wrong` cell returns
# without raising, and what it returns is not the gradient.
function _check_cell(outcome, run, ref; rtol, atol)
    if outcome === :works
        for _ in 1:3
            @test isapprox(run(), ref; rtol = rtol, atol = atol)
        end
    elseif outcome === :wrong
        @test !isapprox(run(), ref; rtol = rtol, atol = atol)
    else
        @test outcome === :throws
        @test_throws Exception run()
    end
end

@testset "AD backends across execution policies" begin
    if !_HAVE_BACKENDS
        @test_skip "AD backends not in this environment"
    else
        threaded = Threads.nthreads() > 1
        threaded || @test_skip "one thread: the threaded policies would run serially"
        threaded && @test _polyester_threads() > 1

        backends = make_backends()

        # References: ForwardDiff on `Serial()`, itself checked against central differences.
        dense_serial = dense_problems(Serial())
        dense_ref = (
            derivative = ForwardDiff.derivative(dense_serial.derivative, POINTS.derivative),
            gradient = ForwardDiff.gradient(dense_serial.gradient, POINTS.gradient),
            assembly = ForwardDiff.gradient(dense_serial.assembly, POINTS.assembly)
        )
        @test isapprox(
            dense_ref.derivative, _fd(dense_serial.derivative, POINTS.derivative);
            rtol = 1e-5, atol = 1e-8
        )
        for name in (:gradient, :assembly), k in 1:2

            f = t -> dense_serial[name](setindex!(copy(POINTS[name]), t, k))
            @test isapprox(dense_ref[name][k], _fd(f, POINTS[name][k]); rtol = 1e-5, atol = 1e-8)
        end

        sparse_serial = nonlinear_problem(Serial())
        J_ref = ForwardDiff.jacobian(sparse_serial.residual, sparse_serial.u)
        direction = randn(Xoshiro(7), length(sparse_serial.u))
        Jv_fd = _fd(t -> sparse_serial.residual(sparse_serial.u .+ t .* direction), 0.0)
        @test isapprox(J_ref * direction, Jv_fd; rtol = 1e-5, atol = 1e-8)

        for (pname, policy) in pairs(POLICIES)
            pname === :CpuSerial || threaded || continue

            @testset "dense, $pname" begin
                problems = dense_problems(policy)
                for (bname, outcome) in pairs(DENSE[pname])
                    @testset "$bname" begin
                        if outcome === :throws_fresh
                            _check_child(outcome, """
                                P = dense_problems($pname())
                                differentiate(:derivative, P.derivative,
                                    make_backends().$bname, POINTS.derivative)
                                """)
                        else
                            for name in keys(POINTS)
                                _check_cell(
                                    outcome,
                                    () -> differentiate(
                                        name, problems[name], backends[bname], POINTS[name]
                                    ),
                                    dense_ref[name]; rtol = 1e-8, atol = 1e-10
                                )
                            end
                        end
                    end
                end
            end

            @testset "sparse, $pname" begin
                prob = nonlinear_problem(policy)
                for (bname, outcome) in pairs(SPARSE[pname])
                    @testset "$bname" begin
                        if outcome in (:throws_fresh, :crashes)
                            # One detector: the other's verdict is the same, and each child
                            # costs a full load and compile.
                            _check_child(outcome, """
                                P = nonlinear_problem($pname())
                                ad = sparse_backend(make_backends().$bname, TracerSparsityDetector())
                                DifferentiationInterface.jacobian(P.residual, ad, P.u)
                                """)
                        else
                            for (dname, detector) in pairs(detectors(prob))
                                ad = sparse_backend(backends[bname], detector)
                                _check_cell(
                                    outcome,
                                    () -> Matrix(
                                        DifferentiationInterface.jacobian(prob.residual, ad, prob.u)
                                    ),
                                    J_ref; rtol = 1e-8, atol = 1e-8
                                )
                            end
                        end
                    end
                end
            end
        end

        # Nothing above may have left Polyester serial for the tests that follow.
        threaded && @test _polyester_threads() > 1
    end
end

end # module SpaceAutodiffPoliciesTests
