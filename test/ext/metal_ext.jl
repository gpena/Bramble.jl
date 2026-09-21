module ExtMetalExtTests

using Test
using Bramble
using Metal
using SparseArrays
using LinearAlgebra: I, mul!
using Bramble: Backend, vector, matrix, _backend_eye, _backend_zeros, metal_sparse_csr,
               metal_sparse_csc, host_points, host_weights, half_spacings
using ..TestUtils: _run_gpu_tests

# BrambleMetalExt's backend allocation primitives
# (`vector`/`matrix`/`_backend_eye`/`_backend_zeros`/`metal_backend`). Meshes and
# gridspaces on a Metal-backed vector are built further below, in the #307-#312
# device-quirk testsets: `mesh` construction no longer fills point coordinates with a
# scalar CPU loop (which GPUArrays refuses on a device array, "Scalar indexing is
# disallowed"). A uniform device mesh fills its four arrays in one `KernelAbstractions`
# launch (gpena/Bramble.jl#303), and a non-uniform one generates its coordinates on the
# host and transfers them in a single `copyto!` (gpena/Bramble.jl#304). What is still not
# exercised anywhere here is a full PDE assembly pipeline on a GPU-resident mesh; that
# remains a separate gap, outside the extension's own scope.
#
# `Metal.functional() && _run_gpu_tests()` gates every testset here that touches an actual
# device array: precompiling and loading `Metal` succeeds on any platform (it degrades
# gracefully rather than erroring, the same convention CUDA.jl uses), but only a real Apple
# Silicon Mac has a working device, so a host without one skips those rather than fails.
# `_run_gpu_tests()` is the second half of that gate, not a restatement of it: GitHub's
# hosted macOS runners *are* real Apple Silicon hardware, so `Metal.functional()` alone is
# not enough to keep device kernels from actually executing, unattended, in CI (see its
# definition in TestUtils.jl). The "rejects a CPU policy over device storage" testset below
# is the one exception to both halves: it checks a construction-time `ArgumentError`
# derived from type information alone, so it runs whenever Metal is loaded, regardless.

@testset "BrambleMetalExt" begin
    # A device VT (MtlVector) under a host CpuPolicy is rejected at construction
    # (gpena/Bramble.jl#296, #298): `_metal_backend` only builds `Backend{MtlVector{T},
    # MtlMatrix{T}, typeof(policy)}()`, a type-level construction that never allocates a
    # device array, so the rejection fires from `MtlVector`/`policy` type information alone.
    # That means it needs `using Metal` to be loaded (for the `MtlVector` type and the
    # `_metal_backend` method to exist) but not a functional device, so it runs outside the
    # `Metal.functional()` gate below and is exercised on any host with Metal loaded.
    @testset "metal_backend rejects a CPU policy over device storage" begin
        for cpu_policy in (CpuSerial(), CpuThreaded())
            err = try
                metal_backend(; policy = cpu_policy)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            msg = sprint(showerror, err)
            @test occursin("MtlVector", msg)
            @test occursin(string(typeof(cpu_policy)), msg)
        end
    end

    if !Metal.functional() || !_run_gpu_tests()
        @test_skip "Metal backend not exercised: Metal.functional() is false, or GPU tests are skipped in CI"
    else
        @testset "metal_backend element types" begin
            @test metal_backend() isa Backend
            # a GPU is massively parallel and cannot execute serially, so the default says
            # so (gpena/Bramble.jl#191); it used to be Serial()
            @test execution_policy(metal_backend()) === GpuAsync()
            @test execution_policy(metal_backend(Float16)) === GpuAsync()
            @test metal_backend(Float32) isa Backend
            @test metal_backend(Float16) isa Backend
            # Float64 is unsupported on Apple Silicon GPUs. The Metal-loaded method only
            # matches `T <: Union{Float16, Float32}`, so `Float64` falls through to the
            # generic "requires Metal.jl" stub in main `src/` by ordinary dispatch
            # specificity -- the same `ErrorException` as the package-not-loaded case, even
            # though Metal is loaded here; only the type is rejected.
            @test_throws ErrorException metal_backend(Float64)
        end

        @testset "vector/matrix allocation" begin
            b = metal_backend()
            v = vector(b, 6)
            @test v isa MtlVector{Float32}
            @test length(v) == 6

            M = matrix(b, 3, 4)
            @test M isa MtlMatrix{Float32}
            @test size(M) == (3, 4)

            b16 = metal_backend(Float16)
            v16 = vector(b16, 4)
            @test v16 isa MtlVector{Float16}
        end

        @testset "_backend_eye / _backend_zeros" begin
            n = 5
            E = _backend_eye(MtlMatrix{Float32}, n)
            @test E isa MtlMatrix{Float32}
            @test Array(E) == Matrix{Float32}(I, n, n)

            Z = _backend_zeros(MtlMatrix{Float32}, n)
            @test Z isa MtlMatrix{Float32}
            @test Array(Z) == zeros(Float32, n, n)
        end

        @testset "Round-trips through Array" begin
            b = metal_backend()
            data = Float32[1.0, 2.0, 3.0]
            v = vector(b, 3)
            copyto!(v, data)
            @test Array(v) == data
        end
    end
end

# ---------------------------------------------------------------------------
# Sparse CSR/CSC: construction, conversion, and SpMV/SpMM accuracy (gpena/Bramble.jl#250)
# ---------------------------------------------------------------------------
#
# Gated on `Metal.functional()` like the testset above, but the skip path here `@warn`s
# instead of only `@test_skip`ing: a silent skip is issue #84's failure mode, and this
# milestone has already shipped one silent skip that had to be fixed later, so a host
# without a functional device is loud about what it did not check.
if !Metal.functional() || !_run_gpu_tests()
    @warn "Skipping Metal sparse CSR/CSC tests: Metal.functional() is false, or GPU tests are skipped in CI"
    @test_skip "Metal sparse CSR/CSC tests not exercised: Metal.functional() is false, or GPU tests are skipped in CI"
else
    @testset "metal_sparse_csr / metal_sparse_csc: non-densifying, round-trips" begin
        A = sprand(Float32, 100, 60, 0.05)
        @test nnz(A) > 0

        Gr = metal_sparse_csr(A)
        @test nnz(Gr) == nnz(A)
        @test SparseMatrixCSC(Gr) == A

        Gc = metal_sparse_csc(A)
        @test nnz(Gc) == nnz(A)
        @test SparseMatrixCSC(Gc) == A
    end

    @testset "SpMV: mul!(y, A::CSR, x, α, β) matches CPU SparseMatrixCSC * Vector" begin
        m, n = 50, 90 # non-square
        A = sprand(Float32, m, n, 0.05)
        x = rand(Float32, n)
        G = metal_sparse_csr(A)

        # β = 0: an uninitialised destination is never read, only overwritten.
        y = MtlArray{Float32}(undef, m)
        mul!(y, G, mtl(x), 1.0f0, 0.0f0)
        @test isapprox(Array(y), A * x; atol = 1.0f-5)

        # β != 0 against a non-zero destination: `iszero(β)` is special-cased, so a test
        # that only ever passes β = 0 would not catch a destination wrongly left untouched
        # or wrongly zeroed (gpena/Bramble.jl#250, S3.2).
        y0 = rand(Float32, m)
        α, β = 2.0f0, 3.0f0
        y = mtl(copy(y0))
        mul!(y, G, mtl(x), α, β)
        @test isapprox(Array(y), α .* (A * x) .+ β .* y0; atol = 1.0f-5)
    end

    @testset "SpMM: mul!(C, A::CSR, B, α, β) for dense right-hand sides" begin
        m, n, k = 50, 90, 4 # non-square A
        A = sprand(Float32, m, n, 0.05)
        B = rand(Float32, n, k)
        G = metal_sparse_csr(A)

        C = MtlArray{Float32}(undef, m, k)
        mul!(C, G, mtl(B), 1.0f0, 0.0f0)
        @test isapprox(Array(C), A * B; atol = 1.0f-5)

        # β != 0 against a non-zero destination -- same reason as the SpMV case above.
        C0 = rand(Float32, m, k)
        α, β = 2.0f0, 3.0f0
        C = mtl(copy(C0))
        mul!(C, G, mtl(B), α, β)
        @test isapprox(Array(C), α .* (A * B) .+ β .* C0; atol = 1.0f-5)
    end

    @testset "Float16 SpMV" begin
        A = sprand(Float16, 20, 20, 0.1)
        x = rand(Float16, 20)
        G = metal_sparse_csr(A)
        y = MtlArray{Float16}(undef, 20)
        mul!(y, G, mtl(x), Float16(1.0), Float16(0.0))
        @test isapprox(Array(y), A * x; atol = Float16(1.0e-2))
    end

    @testset "CSC mul! raises ArgumentError naming the CSR conversion" begin
        m, n = 20, 20
        A = sprand(Float32, m, n, 0.1)
        Gc = metal_sparse_csc(A)

        # Split into vector and matrix `mul!` methods on purpose (gpena/Bramble.jl#250,
        # S3.2): a single `::AbstractVecOrMat` signature ties with LinearAlgebra's own
        # generic `mul!` and raises `MethodError: ... is ambiguous` instead of this
        # `ArgumentError` -- so assert the error type and message, not merely that
        # something throws.
        err = try
            mul!(MtlArray{Float32}(undef, m), Gc, mtl(rand(Float32, n)), 1.0f0, 0.0f0)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("CSR", sprint(showerror, err))

        err = try
            mul!(MtlArray{Float32}(undef, m, 3), Gc, mtl(rand(Float32, n, 3)), 1.0f0, 0.0f0)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("CSR", sprint(showerror, err))
    end

    # #250 also asks for an `ArgumentError` on `Float64`. That guard
    # (`_check_metal_sparse_eltype` inside `mul!`, ext/BrambleMetalExt.jl) is real but
    # unreachable through the normal path: `metal_sparse_csr` on a `Float64` matrix already
    # throws inside Metal.jl's own `mtl()`, because `MtlVector{Float64}` cannot be
    # constructed at all in this Metal.jl version -- confirmed directly rather than
    # contrived. So this tests the behaviour a user actually gets: refusal at construction,
    # with a message naming both `Float64` and `Float32`, rather than reaching for a way to
    # exercise the deeper, currently-unreachable guard.
    @testset "Float64 is refused before it ever reaches mul!" begin
        A64 = sprand(Float64, 10, 10, 0.3)
        err = try
            metal_sparse_csr(A64)
            nothing
        catch e
            e
        end
        @test !isnothing(err)
        msg = sprint(showerror, err)
        @test occursin("Float64", msg)
        @test occursin("Float32", msg)
    end
end

# ---------------------------------------------------------------------------
# Mesh and space quirks fixed by #307-#312 (S14 of
# .agents/plans/v3-4-0-device-quirks-and-kernels.md): one testset per issue. None of them
# uses `@allowscalar` -- every device value below reaches the host through a bulk
# transfer (`Array`, `host_points`, `host_weights`, ...) or a closed-form/host-side
# computation, exactly the paths S1-S7 added, never a per-point scalar read of a device
# array. Gated on `Metal.functional()` with a loud `@warn` skip, like the sparse
# CSR/CSC block above.
# ---------------------------------------------------------------------------
if !Metal.functional() || !_run_gpu_tests()
    @warn "Skipping mesh/space device-quirk tests: Metal.functional() is false, or GPU tests are skipped in CI"
    @test_skip "mesh/space device-quirk tests not exercised: Metal.functional() is false, or GPU tests are skipped in CI"
else
    @testset "#307: is_uniform, stepsize and show on a device mesh" begin
        b = metal_backend()

        # Uniform Float32 device mesh: the tolerance regression. The old absolute
        # tol = 1e-10 was unreachable for Float32 (eps 1.19e-7), so a uniform Float32
        # device mesh used to answer false here.
        Ω_u = mesh(domain(interval(0.0f0, 1.0f0)), 10, true; backend = b)
        @test is_uniform(Ω_u)
        @test isapprox(stepsize(Ω_u), 1.0f0 / 9; rtol = 1.0f-5)

        # A genuinely non-uniform Float32 device mesh must still answer false -- the
        # widened tolerance must not swallow real non-uniformity.
        Ω_nu = mesh(domain(interval(0.0f0, 1.0f0)), 10, false; backend = b)
        @test !is_uniform(Ω_nu)

        # Small-domain spans: 1e-3, 1e-5 and 1e-6 all answered wrongly at some point
        # during development (the span-scaled eps term against the absolute 1e-10 floor).
        for span in (1.0f-3, 1.0f-5, 1.0f-6)
            Ω_span_u = mesh(domain(interval(0.0f0, span)), 10, true; backend = b)
            @test is_uniform(Ω_span_u)

            Ω_span_nu = mesh(domain(interval(0.0f0, span)), 10, false; backend = b)
            @test !is_uniform(Ω_span_nu)
        end

        s = sprint(show, MIME"text/plain"(), Ω_u)
        @test !isempty(s)
        @test occursin("Mesh1D", s)
    end

    @testset "#308: host_points and locate_cell on a device mesh" begin
        b = metal_backend()

        # host_points is the identical object on a host mesh (===), and a genuine
        # Vector (a bulk-transferred copy, not a zero-copy alias) on a device mesh.
        Ω_host = mesh(domain(interval(0.0f0, 1.0f0)), 12, true)
        @test host_points(Ω_host) === points(Ω_host)

        Ω_dev = mesh(domain(interval(0.0f0, 1.0f0)), 12, true; backend = b)
        hp = host_points(Ω_dev)
        @test hp isa Vector
        @test length(hp) == 12

        # locate_cell must agree with searchsortedlast on the host points at every cell
        # midpoint, every grid node -- the case that was wrong: the closed-form branch
        # floored 2.9999999999999996 to the cell on the left -- one ULP either side of
        # every node, and outside both ends.
        for unif in (true, false), n in (5, 12, 33)

            Ω = mesh(domain(interval(0.0f0, 1.0f0)), n, unif; backend = b)
            pts = host_points(Ω)
            expected(x) = clamp(searchsortedlast(pts, x), 1, n - 1)

            for i in 1:(n - 1)
                mid = (pts[i] + pts[i + 1]) / 2
                @test locate_cell(Ω, mid) == expected(mid)
            end

            for i in 1:n
                x = pts[i]
                @test locate_cell(Ω, x) == expected(x)
                @test locate_cell(Ω, prevfloat(x)) == expected(prevfloat(x))
                @test locate_cell(Ω, nextfloat(x)) == expected(nextfloat(x))
            end

            @test locate_cell(Ω, pts[1] - 1.0f0) == 1
            @test locate_cell(Ω, pts[end] + 1.0f0) == n - 1
        end
    end

    @testset "#304: non-uniform device mesh built on the host and copied over" begin
        b = metal_backend()

        Ω = mesh(domain(interval(0.0f0, 1.0f0)), 16, false; backend = b)
        p = Array(points(Ω))
        @test issorted(p)
        @test p[1] == 0.0f0
        @test p[end] == 1.0f0
        @test !is_uniform(Ω)

        # Independent control: device spacings/half_points/half_spacings match a host
        # mesh given the identical coordinates via set_points!.
        n = 24
        Ω_dev = mesh(domain(interval(0.0f0, 1.0f0)), n, false; backend = b)
        coords = Array(points(Ω_dev))

        Ω_host = mesh(domain(interval(0.0f0, 1.0f0)), n, true)
        set_points!(Ω_host, coords)

        @test Array(spacings(Ω_dev)) ≈ spacings(Ω_host)
        @test Array(half_points(Ω_dev)) ≈ half_points(Ω_host)
        @test Array(half_spacings(Ω_dev)) ≈ half_spacings(Ω_host)

        # 2D, mixed uniformity.
        Ω2 = mesh(
            domain(interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0)),
            (12, 12), (true, false); backend = b
        )
        p1 = Array(points(Ω2(1)))
        p2 = Array(points(Ω2(2)))
        @test issorted(p1) && issorted(p2)
        @test is_uniform(Ω2(1))
        @test !is_uniform(Ω2(2))
    end

    @testset "#309: condition markers on device meshes match the host" begin
        b = metal_backend()

        # 1D: a selective, an empty and a total predicate give index sets IDENTICAL to
        # the host's, not merely both non-empty.
        d1 = domain(
            interval(0.0f0, 1.0f0),
            :selective => (p -> p[1] < 0.3f0),
            :empty => (p -> p[1] < 0.0f0),
            :total => (p -> true)
        )
        Ω1_dev = mesh(d1, 11, true; backend = b)
        Ω1_host = mesh(d1, 11, true)
        for label in (:selective, :empty, :total)
            @test index_in_marker(Ω1_dev, label) == index_in_marker(Ω1_host, label)
        end
        @test count(index_in_marker(Ω1_dev, :selective)) > 0
        @test count(index_in_marker(Ω1_dev, :empty)) == 0
        @test count(index_in_marker(Ω1_dev, :total)) == 11

        # 2D
        d2 = domain(
            interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0),
            :selective => (p -> p[1] < 0.3f0 && p[2] < 0.3f0),
            :empty => (p -> p[1] < 0.0f0),
            :total => (p -> true)
        )
        Ω2_dev = mesh(d2, (9, 9), (true, true); backend = b)
        Ω2_host = mesh(d2, (9, 9), (true, true))
        for label in (:selective, :empty, :total)
            @test index_in_marker(Ω2_dev, label) == index_in_marker(Ω2_host, label)
        end

        # 3D
        d3 = domain(
            interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0),
            :selective => (p -> p[1] < 0.5f0 && p[2] < 0.5f0 && p[3] < 0.5f0),
            :empty => (p -> p[1] < 0.0f0),
            :total => (p -> true)
        )
        Ω3_dev = mesh(d3, (7, 7, 7), (true, true, true); backend = b)
        Ω3_host = mesh(d3, (7, 7, 7), (true, true, true))
        for label in (:selective, :empty, :total)
            @test index_in_marker(Ω3_dev, label) == index_in_marker(Ω3_host, label)
        end
    end

    @testset "#310: SeparableWeights Array/host_weights on a device space" begin
        Ω = mesh(
            domain(interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0)),
            (5, 5), (true, true); backend = metal_backend()
        )
        W = gridspace(Ω)
        w = weights(W, Bramble.Innerh())

        a = Array(w)
        @test length(a) == 25
        @test isapprox(sum(a), 1.0f0; rtol = 1.0f-4)

        hw = host_weights(w)
        @test hw isa Bramble.SeparableWeights
        @test Array(hw) ≈ a

        err = try
            w[CartesianIndex(1, 1)]
            nothing
        catch e
            e
        end
        @test !isnothing(err)
        @test occursin("host_weights", sprint(showerror, err))
    end

    @testset "#311: inner_Γ and normal_vector on a device space" begin
        b = metal_backend()
        n = 16
        Ω_dev = mesh(
            domain(interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0)), (n, n),
            (true, true); backend = b
        )
        Ω_host = mesh(
            domain(interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0)), (n, n),
            (true, true)
        )
        W_dev = gridspace(Ω_dev)
        W_host = gridspace(Ω_host)

        # Non-constant: a constant integrand hides indexing errors.
        f(p) = 1.0f0 + p[1]^2 + 2.0f0 * p[2]

        u_dev = Rₕ(W_dev, f)
        u_host = Rₕ(W_host, f)

        for label in (:xmin, :ymax, :boundary)
            g_dev = inner_Γ(u_dev, u_dev, label)
            g_host = inner_Γ(u_host, u_host, label)
            @test isapprox(g_dev, g_host; rtol = 1.0f-4)
        end

        nu_dev = normal_vector(W_dev, :ymax)
        nu_host = normal_vector(W_host, :ymax)
        for d in 1:2
            @test Array(parent(nu_dev[d])) ≈ parent(nu_host[d])
        end

        # Outward and zero off the face: :ymax's normal is (0, 1) on the top row and
        # zero everywhere else.
        nx = Array(parent(nu_dev[1]))
        @test all(==(0.0f0), nx)

        ny = reshape(Array(parent(nu_dev[2])), n, n)
        @test all(==(0.0f0), ny[:, 1:(n - 1)])
        @test all(==(1.0f0), ny[:, n])
    end

    @testset "#312: interpolation between device spaces matches the host" begin
        b = metal_backend()

        # Non-linear: a constant or linear source would not exercise the corner
        # weights the way a genuinely curved function does.
        f(x) = x^2 - 2.0f0 * x + 1.0f0

        # Device-to-device, refinement (5 -> 9) and coarsening (9 -> 5).
        for (n_src, n_dst) in ((5, 9), (9, 5))
            Ω_src_dev = mesh(domain(interval(0.0f0, 1.0f0)), n_src, true; backend = b)
            Ω_dst_dev = mesh(domain(interval(0.0f0, 1.0f0)), n_dst, true; backend = b)
            W_src_dev = gridspace(Ω_src_dev)
            W_dst_dev = gridspace(Ω_dst_dev)

            Ω_src_host = mesh(domain(interval(0.0f0, 1.0f0)), n_src, true)
            Ω_dst_host = mesh(domain(interval(0.0f0, 1.0f0)), n_dst, true)
            W_src_host = gridspace(Ω_src_host)
            W_dst_host = gridspace(Ω_dst_host)

            u_src_dev = Rₕ(W_src_dev, f)
            u_src_host = Rₕ(W_src_host, f)

            u_dst_dev = element(W_dst_dev, 0.0f0)
            πₕ!(u_dst_dev, u_src_dev)
            u_dst_host = element(W_dst_host, 0.0f0)
            πₕ!(u_dst_host, u_src_host)

            @test Array(parent(u_dst_dev)) ≈ parent(u_dst_host)

            # host-to-device
            u_dst_h2d = element(W_dst_dev, 0.0f0)
            πₕ!(u_dst_h2d, u_src_host)
            @test Array(parent(u_dst_h2d)) ≈ parent(u_dst_host)

            # device-to-host
            u_dst_d2h = element(W_dst_host, 0.0f0)
            πₕ!(u_dst_d2h, u_src_dev)
            @test parent(u_dst_d2h) ≈ parent(u_dst_host)
        end

        # Non-uniform device source mesh: exercises locate_cell's search path, not the
        # closed-form uniform one.
        Ω_src_nu_dev = mesh(domain(interval(0.0f0, 1.0f0)), 11, false; backend = b)
        Ω_dst_dev = mesh(domain(interval(0.0f0, 1.0f0)), 15, true; backend = b)
        W_src_nu_dev = gridspace(Ω_src_nu_dev)
        W_dst_dev = gridspace(Ω_dst_dev)

        coords = Array(points(Ω_src_nu_dev))
        Ω_src_nu_host = mesh(domain(interval(0.0f0, 1.0f0)), 11, true)
        set_points!(Ω_src_nu_host, coords)
        W_src_nu_host = gridspace(Ω_src_nu_host)
        Ω_dst_host = mesh(domain(interval(0.0f0, 1.0f0)), 15, true)
        W_dst_host = gridspace(Ω_dst_host)

        u_src_nu_dev = Rₕ(W_src_nu_dev, f)
        u_src_nu_host = Rₕ(W_src_nu_host, f)

        u_dst_dev = element(W_dst_dev, 0.0f0)
        πₕ!(u_dst_dev, u_src_nu_dev)
        u_dst_host = element(W_dst_host, 0.0f0)
        πₕ!(u_dst_host, u_src_nu_host)

        @test Array(parent(u_dst_dev)) ≈ parent(u_dst_host)

        # 2D
        f2(p) = p[1]^2 - 2.0f0 * p[1] * p[2] + p[2]^2

        Ω1_2d_dev = mesh(
            domain(interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0)), (5, 5),
            (true, true); backend = b
        )
        Ω2_2d_dev = mesh(
            domain(interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0)), (9, 9),
            (true, true); backend = b
        )
        W1_2d_dev = gridspace(Ω1_2d_dev)
        W2_2d_dev = gridspace(Ω2_2d_dev)

        Ω1_2d_host = mesh(domain(interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0)), (5, 5), (true, true))
        Ω2_2d_host = mesh(domain(interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0)), (9, 9), (true, true))
        W1_2d_host = gridspace(Ω1_2d_host)
        W2_2d_host = gridspace(Ω2_2d_host)

        u1_2d_dev = Rₕ(W1_2d_dev, f2)
        u2_2d_dev = element(W2_2d_dev, 0.0f0)
        πₕ!(u2_2d_dev, u1_2d_dev)

        u1_2d_host = Rₕ(W1_2d_host, f2)
        u2_2d_host = element(W2_2d_host, 0.0f0)
        πₕ!(u2_2d_host, u1_2d_host)

        @test Array(parent(u2_2d_dev)) ≈ parent(u2_2d_host)
    end
end

end # module ExtMetalExtTests
