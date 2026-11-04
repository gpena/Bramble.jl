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
# Every cell listed below is pinned to work: three calls, each agreeing with ForwardDiff on
# `Serial()` to the cell's tolerance. A cell that starts failing, or returns a wrong value
# without raising, means a backend or a sweep changed, and the list needs a reason.
#
# Why the cells come out the way they do:
#
#   ForwardDiff   works everywhere. A `Dual` is a plain value, so each thread writes its own
#                 entries and nothing is shared.
#   Enzyme        differentiates both threaded policies. Enzyme never enters a `@batch` or
#                 `@threads` body: each engine hook has its own forward and reverse rule in
#                 ext/BrambleEnzymeExt.jl, and the rule runs the derivative threaded itself.
#
# Sparse AD is not a backend of its own: `AutoSparse` wraps a dense backend with a sparsity
# detector and a coloring. The detector side works under every policy -- `TracerSparsityDetector`'s
# tracers are plain values like `Dual`s, and `ast_sparsity_detector` never runs the residual
# at all -- so a sparse cell's verdict is its inner backend's.
#
# `GpuKernel` has no column: nothing in the device extensions handles a `Dual` or a tracked
# type, and `pde_solve`'s rules take a `SparseMatrixCSC` only, so there is no device AD path
# to measure yet.

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

function differentiate(name, f, ad, x)
    name === :derivative ?
    DifferentiationInterface.derivative(f, ad, x) :
    DifferentiationInterface.gradient(f, ad, x)
end

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

function sparse_backend(inner, detector)
    AutoSparse(
        inner; sparsity_detector = detector, coloring_algorithm = GreedyColoringAlgorithm()
    )
end

const POLICIES = (CpuSerial = Serial(), CpuThreaded = Parallel(), CpuPolyester = CpuPolyester())

# The pinned cells: every backend listed differentiates every problem under that policy.
const DENSE = (
    CpuSerial = (:ForwardDiff, :Enzyme),
    CpuThreaded = (:ForwardDiff, :Enzyme),
    CpuPolyester = (:ForwardDiff, :Enzyme)
)
const SPARSE = (
    CpuSerial = (:ForwardDiff, :Enzyme),
    CpuThreaded = (:ForwardDiff, :Enzyme),
    CpuPolyester = (:ForwardDiff, :Enzyme)
)

const _HAVE_ENZYME = _have(:Enzyme)
if _HAVE_ENZYME
    @eval import Enzyme
end

# How many distinct threads one `@batch` loop runs on.
function _polyester_threads()
    seen = zeros(Int, 64 * Threads.nthreads())
    @batch for i in eachindex(seen)
        seen[i] = Threads.threadid()
    end
    return length(unique(seen))
end

# Three calls per cell: a race can come out right once.
function _check_cell(run, ref; rtol, atol)
    for _ in 1:3
        @test isapprox(run(), ref; rtol = rtol, atol = atol)
    end
end

@testset "AD backends across execution policies" begin
    threaded = Threads.nthreads() > 1
    threaded || @test_skip "one thread: the threaded policies would run serially"
    threaded && @test _polyester_threads() > 1

    # Enzyme's two annotations are autodiff_heavy.jl's, for the reasons given there.
    backends = (ForwardDiff = AutoForwardDiff(),)
    if _HAVE_ENZYME
        backends = (;
            backends...,
            Enzyme = AutoEnzyme(
                mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
                function_annotation = Enzyme.Const
            )
        )
    end

    # References: ForwardDiff on `Serial()`, itself checked against central differences.
    oracle = AutoForwardDiff()
    dense_serial = dense_problems(Serial())
    dense_ref = (
        derivative = DifferentiationInterface.derivative(
            dense_serial.derivative, oracle, POINTS.derivative
        ),
        gradient = DifferentiationInterface.gradient(
            dense_serial.gradient, oracle, POINTS.gradient
        ),
        assembly = DifferentiationInterface.gradient(
            dense_serial.assembly, oracle, POINTS.assembly
        )
    )
    @test isapprox(
        dense_ref.derivative, _fd(dense_serial.derivative, POINTS.derivative);
        rtol = 1e-5, atol = 1e-8
    )
    for name in (:gradient, :assembly), k in 1:2

        f = t -> dense_serial[name](setindex!(copy(POINTS[name]), t, k))
        @test isapprox(
            dense_ref[name][k], _fd(f, POINTS[name][k]); rtol = 1e-5, atol = 1e-8
        )
    end

    sparse_serial = nonlinear_problem(Serial())
    J_ref = DifferentiationInterface.jacobian(
        sparse_serial.residual, oracle, sparse_serial.u
    )
    direction = randn(Xoshiro(7), length(sparse_serial.u))
    Jv_fd = _fd(t -> sparse_serial.residual(sparse_serial.u .+ t .* direction), 0.0)
    @test isapprox(J_ref * direction, Jv_fd; rtol = 1e-5, atol = 1e-8)

    for (pname, policy) in pairs(POLICIES)
        pname === :CpuSerial || threaded || continue

        @testset "dense, $pname" begin
            problems = dense_problems(policy)
            for bname in DENSE[pname]
                @testset "$bname" begin
                    if !haskey(backends, bname)
                        @test_skip "$bname not in this environment"
                    else
                        for name in keys(POINTS)
                            _check_cell(
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
            for bname in SPARSE[pname]
                @testset "$bname" begin
                    if !haskey(backends, bname)
                        @test_skip "$bname not in this environment"
                    else
                        for detector in values(detectors(prob))
                            ad = sparse_backend(backends[bname], detector)
                            _check_cell(
                                () -> Matrix(
                                    DifferentiationInterface.jacobian(
                                    prob.residual, ad, prob.u
                                )
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

end # module SpaceAutodiffPoliciesTests
