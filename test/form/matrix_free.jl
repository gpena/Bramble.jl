module TestFormMatrixFree

using Test
using Bramble
using Bramble: MatrixFreeOperator, VectorElement, trial_space, test_space, restrict_to, πₕ, jumpₓ, M₊ₓ, D₋ₓ, D₋ᵧ, D₊ᵧ
using LinearAlgebra: mul!, norm, dot
using Random
using ..TestUtils: WITH_SLOW_TESTS

# `matrix_free_operator` applies a bilinear form through the walk the
# assembly replays, so every check compares it against `assemble(a; dirichlet)` on the same
# vector. Meshes are non-uniform throughout: a uniform mesh hides a spacing read on the wrong
# side of a point.

const MF_SEED = 3261

_mf_agree(a, b) = isapprox(a, b; rtol = 1e-12, atol = 1e-12 * max(1.0, norm(b, Inf)))

# Inside functions so `@allocated` measures the call alone, not global-scope boxing.
_mf_alloc3(y, op, x) = (mul!(y, op, x); @allocated mul!(y, op, x))
_mf_alloc5(y, op, x) = (mul!(y, op, x, 0.5, 2.0); @allocated mul!(y, op, x, 0.5, 2.0))
_mf_alloc_times(op, x) = (op * x; @allocated op * x)

# Seeded on every call, so two backends draw the same non-uniform meshes.
function _mf_spaces(be = backend())
    Random.seed!(MF_SEED)
    return (
        gridspace(mesh(domain(interval(0.0, 1.0), :west => :left), 17, false; backend = be)),
        gridspace(mesh(
            domain(interval(0.0, 1.0) × interval(0.0, 2.0), :west => :left), (9, 11), (false, true);
            backend = be)),
        gridspace(mesh(
            domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0), :west => :left), (
                6, 7, 5), false; backend = be))
    )
end

# One space's four cases. A function of its own so each call is compiled for one concrete
# space type: looping over the 1D/2D/3D tuple inline makes inference pair a trial function of
# one dimension with a test function of another, which never happens.
function _mf_push_dim_cases!(out, W)
    D = dim(W)
    κ = Rₕ(W, x -> 1 + sum(abs2, x))
    push!(out, ("$(D)D diffusion", form(W, W, (u, v) -> innerₕ(u, v) + inner₊(κ * ∇ₕ(u), ∇ₕ(v))), :boundary))
    push!(out, ("$(D)D jump-avg", form(W, W, (u, v) -> innerₕ(jumpₓ(u), M₊ₓ(v)) + innerₕ(D₋ₓ(u), v)), nothing))
    push!(out, (
        "$(D)D restricted", form(W, W, (u, v) -> innerₕ(u, v) + innerₕ(κ * u, restrict_to(:boundary, v))), nothing))
    push!(out, ("$(D)D pair", form(W, W, (u, v) -> innerₕ(D₋ₓ(u), v) + 2.0 * innerₕ(u, D₋ₓ(v))), (:west,)))
    return out
end

# (name, form, dirichlet): variable diffusion, jump/average/difference, a region restriction,
# a transposed pair, per dimension; then composite spaces with crossed components, one on a
# single leaf object and one on two, so both halves of the pair walk run.
function _mf_cases(be = backend())
    out = Tuple{String, Bramble.BilinearForm, Union{Nothing, Symbol, Tuple{Vararg{Symbol}}}}[]
    spaces = _mf_spaces(be)
    for W in spaces
        _mf_push_dim_cases!(out, W)
    end
    W = spaces[2]
    V = W × W
    push!(out, ("composite",
        form(V, V, (u, v) -> innerₕ(u(1), v(1)) + inner₊(∇ₕ(u(2)), ∇ₕ(v(2))) + innerₕ(u(1), v(2))), :boundary))
    push!(out,
        ("composite pair",
            form(V, V, (u, v) -> innerₕ(D₋ᵧ(u(1)), v(2)) + 3.0 * innerₕ(u(2), D₋ᵧ(v(1))) + innerₕ(u(1), v(1))),
            nothing))
    W2 = gridspace(mesh(W))
    V2 = W × W2
    push!(out,
        ("two-leaf pair",
            form(V2, V2, (u, v) -> innerₕ(D₋ₓ(u(1)), v(2)) + innerₕ(u(2), D₋ₓ(v(1))) + innerₕ(u(2), v(2))),
            (:west, :boundary)))
    return out
end

_mf_op(a, dl) = dl === nothing ? matrix_free_operator(a) : matrix_free_operator(a; dirichlet = dl)
_mf_mat(a, dl) = dl === nothing ? assemble(a) : assemble(a; dirichlet = dl)

@testset "matrix-free operator (#326)" begin
    cases = _mf_cases()

    @testset "matrix-free: agrees with assemble" begin
        for (name, a, dl) in cases
            @testset "$name" begin
                A = _mf_mat(a, dl)
                op = _mf_op(a, dl)
                @test op isa MatrixFreeOperator{Float64}
                @test size(op) == size(A)
                @test eltype(op) == eltype(A)
                x = randn(size(A, 2))
                y0 = randn(size(A, 1))
                @test !iszero(A * x)
                @test _mf_agree(op * x, A * x)
                y = copy(y0)
                mul!(y, op, x)
                @test _mf_agree(y, A * x)
                y = copy(y0)
                mul!(y, op, x, 0.5, 2.0)
                @test _mf_agree(y, 0.5 * (A * x) + 2.0 * y0)
                # `β = 0` overwrites `y`: a `NaN` in it does not survive.
                y = fill(NaN, size(A, 1))
                mul!(y, op, x, -1.5, 0.0)
                @test _mf_agree(y, -1.5 * (A * x))

                uₕ = element(trial_space(a))
                parent(uₕ) .= x
                vₕ = op * uₕ
                @test vₕ isa VectorElement
                @test space(vₕ) === test_space(a)
                @test _mf_agree(parent(vₕ), A * x)
                wₕ = element(test_space(a))
                parent(wₕ) .= y0
                mul!(wₕ, op, uₕ, 0.5, 2.0)
                @test _mf_agree(parent(wₕ), 0.5 * (A * x) + 2.0 * y0)
                mul!(wₕ, op, uₕ)
                @test _mf_agree(parent(wₕ), A * x)
            end
        end
    end

    @testset "matrix-free: allocation-free mul!" begin
        for (name, a, dl) in cases
            @testset "$name" begin
                op = _mf_op(a, dl)
                x = randn(size(op, 2))
                y = randn(size(op, 1))
                @test _mf_alloc3(y, op, x) == 0
                @test _mf_alloc5(y, op, x) == 0
                # `op * x` allocates its result only, never a matrix read entry by entry.
                @test _mf_alloc_times(op, x) <= sizeof(y) + 256
                uₕ = element(trial_space(a))
                parent(uₕ) .= x
                wₕ = element(test_space(a))
                @test _mf_alloc3(wₕ, op, uₕ) == 0
                @test _mf_alloc5(wₕ, op, uₕ) == 0
            end
        end
    end

    # Every product under `CpuThreaded` equals the serial one, on every repeat: each task adds
    # only into the rows of its own band. Both backends draw the same meshes (`_mf_spaces`
    # reseeds). Run at `--threads=4` for the race to have a chance.
    @testset "matrix-free: threaded mul! race-free" begin
        # `unit` keeps one case per plan that differs under `Parallel`: Dirichlet rows on a
        # single leaf, a composite on one leaf object, and one on two leaves (the fused sweep
        # below covers 3D and the other composite). `slow` runs all of them.
        pcases = _mf_cases(backend(policy = Parallel()))
        for ((name, a, dl), (_, at, _)) in zip(cases, pcases)
            (WITH_SLOW_TESTS || name in ("2D diffusion", "composite", "two-leaf pair")) || continue
            @testset "$name" begin
                op = _mf_op(a, dl)
                opt = _mf_op(at, dl)
                @test Bramble.execution_policy(trial_space(at)) isa Parallel
                x = randn(size(op, 2))
                y0 = randn(size(op, 1))
                ref = op * x
                y = similar(ref)
                @test all(1:20) do _
                    mul!(y, opt, x)
                    return _mf_agree(y, ref)
                end
                @test _mf_agree(opt * x, ref)
                y = copy(y0)
                mul!(y, opt, x, 0.5, 2.0)
                @test _mf_agree(y, 0.5 * ref + 2.0 * y0)
                uₕ = element(trial_space(at))
                parent(uₕ) .= x
                @test _mf_agree(parent(opt * uₕ), ref)
                wₕ = element(test_space(at))
                mul!(wₕ, opt, uₕ)
                @test _mf_agree(parent(wₕ), ref)
            end
        end
        # The operator's own policy threads a serial space's product, and
        # `dirichlet_components` holds the named leaves' rows as it does serially.
        W = _mf_spaces()[2]
        V = W × W
        a = form(V, V, (u, v) -> inner₊(∇ₕ(u(1)), ∇ₕ(v(1))) + innerₕ(u(2), v(2)) + innerₕ(D₋ₓ(u(1)), v(2)))
        x = randn(ndofs(V))
        for comps in (1, 2, nothing)
            A = assemble(a; dirichlet = :boundary, dirichlet_components = comps)
            op = matrix_free_operator(a; dirichlet = :boundary, dirichlet_components = comps, policy = Parallel())
            @test _mf_agree(op * x, A * x)
        end
        # What a threaded product allocates is its task spawns, whatever the grid size. One
        # reading jitters by up to about 1 kB at four threads, so the least of five is compared.
        bytes = map((33, 3001)) do n
            Random.seed!(MF_SEED)
            Wn = gridspace(mesh(domain(interval(0.0, 1.0)), n, false; backend = backend(policy = Parallel())))
            κ = Rₕ(Wn, x -> 1 + x^2)
            op = matrix_free_operator(
                form(Wn, Wn, (u, v) -> innerₕ(u, v) + inner₊(κ * ∇ₕ(u), ∇ₕ(v))); dirichlet = :boundary)
            xn = randn(size(op, 2))
            y = similar(xn)
            return minimum(_mf_alloc3(y, op, xn) for _ in 1:5)
        end
        @test bytes[1] == bytes[2]
    end

    # The fused sweep (M1.5): one band per thread along the last axis, every term walked in
    # each, a band widened by the rows' reach and keeping only the rows it owns, in the serial
    # order. Its reach is
    # the union over every term's row offsets, fixed when the operator is built: `D₊ᵧ` and
    # `D₋ᵧ` on the test side reach one row either way. A short last axis (fewer slices than
    # threads, bands narrower than the reach) and a test-side interpolation (walked serially
    # after the bands) must still give the serial product.
    @testset "matrix-free: fused threaded sweep" begin
        Random.seed!(MF_SEED)
        Ωc = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0), :west => :left), (9, 7), (false, false))
        Ωf = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (13, 11), (false, false))
        Ωs = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (8, 3), (false, false))
        Ω3 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 4, 6), false)
        Wc, Wf, Ws, W3 = gridspace(Ωc), gridspace(Ωf), gridspace(Ωs), gridspace(Ω3)
        κ = Rₕ(Wc, x -> 1 + sum(abs2, x))
        V = Wc × Wc
        cases = (
            ("union", form(Wc, Wc, (u, v) -> innerₕ(u, D₊ᵧ(v)) + innerₕ(u, D₋ᵧ(v))), nothing, (-1, 1)),
            ("wide", form(Wc, Wc, (u, v) -> innerₕ(u, D₋ᵧ(D₋ᵧ(v))) + innerₕ(D₊ᵧ(u), v)), :boundary, (-2, 0)),
            ("pair", form(Wc, Wc, (u, v) -> innerₕ(D₊ᵧ(u), v) + 2.0 * innerₕ(u, D₊ᵧ(v))), (:west,), (0, 1)),
            ("diffusion", form(Wc, Wc, (u, v) -> innerₕ(u, v) + inner₊(κ * ∇ₕ(u), ∇ₕ(v))), :boundary, (-1, 0)),
            ("short axis", form(Ws, Ws, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))), :boundary, (-1, 0)),
            # Three slices against a reach of two: the wide term's grid has no interior to
            # peel, so its band walks every point guarded.
            ("no interior", form(Ws, Ws, (u, v) -> innerₕ(u, D₋ᵧ(D₋ᵧ(v))) + innerₕ(D₊ᵧ(u), v)), :boundary, (-2, 0)),
            ("3D", form(W3, W3, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))), :boundary, (-1, 0)),
            ("pointwise", form(Wc, Wc, (u, v) -> innerₕ(u, v) + innerₕ(κ * u, v)), nothing, (0, 0)),
            ("composite pair",
                form(V, V, (u, v) -> innerₕ(D₋ᵧ(u(1)), v(2)) + 3.0 * innerₕ(u(2), D₋ᵧ(v(1))) +
                                     inner₊(∇ₕ(u(1)), ∇ₕ(v(1)))),
                :boundary, (-1, 0)),
            ("test interpolation", form(Wc, Wf, (u, v) -> innerₕ(πₕ(u), v) + innerₕ(u, πₕ(v))), nothing, (0, 0))
        )
        for (name, a, dl, reach) in cases
            @testset "$name" begin
                op = _mf_op(a, dl)
                opt = dl === nothing ? matrix_free_operator(a; policy = Parallel()) :
                      matrix_free_operator(a; dirichlet = dl, policy = Parallel())
                @test op.plan === nothing
                @test opt.plan isa Bramble._MFFusedPlan
                @test (opt.plan.omin, opt.plan.omax) == reach
                @test opt.plan.interp == (name == "test interpolation")
                x = randn(size(op, 2))
                ref = op * x
                @test _mf_agree(ref, _mf_mat(a, dl) * x)
                # Each row receives its entries in the serial order: bitwise the serial
                # product, except where a term runs serially after the bands.
                same = name == "test interpolation" ? _mf_agree : (==)
                y = similar(ref)
                @test all(1:50) do _
                    mul!(y, opt, x)
                    return same(y, ref)
                end
                # Called from inside a user's threaded loop, each product still is.
                ys = [similar(ref) for _ in 1:8]
                Threads.@threads :static for i in 1:8
                    mul!(ys[i], opt, x)
                end
                @test all(yi -> same(yi, ref), ys)
            end
        end
        # Leaves of different sizes share no band cut: the per-unit sweep instead.
        Vm = Wc × Wf
        am = form(Vm, Vm, (u, v) -> inner₊(∇ₕ(u(1)), ∇ₕ(v(1))) + innerₕ(u(2), v(2)))
        opm = matrix_free_operator(am; policy = Parallel())
        @test opm.plan === nothing
        xm = randn(ndofs(Vm))
        @test _mf_agree(opm * xm, assemble(am) * xm)
        # A transposed pair on the larger leaf, swept in the colours of both terms' rows.
        ap = form(Vm, Vm, (u, v) -> inner₊(∇ₕ(u(1)), ∇ₕ(v(1))) + innerₕ(D₋ₓ(u(2)), v(2)) +
                                    2.0 * innerₕ(u(2), D₋ₓ(v(2))))
        opp = matrix_free_operator(ap; dirichlet = :boundary, policy = Parallel())
        @test opp.plan === nothing
        @test _mf_agree(opp * xm, assemble(ap; dirichlet = :boundary) * xm)
        # A test-side interpolation alone leaves no unit for the bands: walked whole.
        ai = form(Wc, Wf, (u, v) -> innerₕ(u, πₕ(v)))
        opi = matrix_free_operator(ai; policy = Parallel())
        @test opi.plan === nothing
        xi = randn(ndofs(Wc))
        @test !iszero(assemble(ai) * xi)
        @test _mf_agree(opi * xi, assemble(ai) * xi)
        # One parallel region per product, whatever the grid: the same bytes at two sizes
        # (the least of five readings, as above).
        bytes = map((17, 301)) do n
            Random.seed!(MF_SEED)
            Wn = gridspace(mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (n, n + 2), (false, false)))
            κn = Rₕ(Wn, x -> 1 + sum(abs2, x))
            op = matrix_free_operator(
                form(Wn, Wn, (u, v) -> innerₕ(u, v) + inner₊(κn * ∇ₕ(u), ∇ₕ(v))); policy = Parallel())
            xn = randn(size(op, 2))
            y = similar(xn)
            return minimum(_mf_alloc3(y, op, xn) for _ in 1:5)
        end
        @test bytes[1] == bytes[2]
    end

    # The form call `a(u, v)` sums `vᴴ A u` through the same walk, storing neither `A` nor
    # `A * u`. A complex `v` checks the conjugate; the pairs walk their two terms in turn.
    @testset "form call contracts vᴴAu" begin
        for (name, a, _) in cases
            (WITH_SLOW_TESTS || name in ("2D pair", "composite pair", "two-leaf pair")) || continue
            @testset "$name" begin
                A = assemble(a)
                u = randn(size(A, 2))
                v = randn(ComplexF64, size(A, 1))
                @test !iszero(A * u)
                @test isapprox(a(u, v), dot(v, A * u); rtol = 1e-12)
                uₕ = element(trial_space(a))
                parent(uₕ) .= u
                vₕ = element(test_space(a))
                parent(vₕ) .= real.(v)
                @test isapprox(a(uₕ, vₕ), dot(real.(v), A * u); rtol = 1e-12)
            end
        end
    end

    @testset "entries, Dirichlet rows, live data" begin
        W = _mf_spaces()[1]
        κ = Rₕ(W, x -> 1 + x^2)
        c = Ref(2.0)
        a = form(W, W, (u, v) -> c * innerₕ(u, v) + inner₊(κ * ∇ₕ(u), ∇ₕ(v)))
        A = assemble(a; dirichlet = :boundary)
        op = matrix_free_operator(a; dirichlet = :boundary)
        @test [op[i, j] for i in axes(A, 1), j in axes(A, 2)] ≈ Matrix(A) atol = 1e-12
        # Dirichlet rows are identity rows: `(A * x)[i] = x[i]`, columns untouched.
        x = randn(size(A, 2))
        y = op * x
        @test y[1] == x[1] && y[end] == x[end]
        # A coefficient changed in place is read by the next product, as by `assemble!`.
        c[] = 5.0
        Rₕ!(κ, x -> 3 + x)
        @test _mf_agree(op * x, assemble(a; dirichlet = :boundary) * x)
        @test_throws DimensionMismatch mul!(zeros(3), op, x)
        # Only the matrix's rows are read from a `dirichlet` pair; the values are not.
        @test _mf_agree(matrix_free_operator(a; dirichlet = :boundary => 1.0) * x, op * x)
    end

    # More test rows than trial columns: a row in Γ_D past the last column has no diagonal,
    # so the assembled matrix leaves it zero and `x` must not be read there. `x` is a view
    # into a longer buffer, so an out-of-range read would pick up the `1e6` past its end.
    @testset "rectangular Dirichlet rows" begin
        Ωc = mesh(domain(interval(0.0, 1.0)), 9, false)
        Ωf = mesh(domain(interval(0.0, 1.0)), 13, false)
        Wc, Wf = gridspace(Ωc), gridspace(Ωf)
        V = Wc × Wc
        for (name, a) in (
            ("πₕ onto finer", form(Wc, Wf, (u, v) -> innerₕ(πₕ(u), v))),
            ("scalar to composite", form(Wc, V, (u, v) -> innerₕ(u, v(1)) + innerₕ(D₋ₓ(u), v(2))))
        )
            @testset "$name" begin
                A = assemble(a; dirichlet = :boundary)
                op = matrix_free_operator(a; dirichlet = :boundary)
                @test size(op, 1) > size(op, 2)
                buf = fill(1.0e6, size(A, 1))
                buf[1:size(A, 2)] .= randn(size(A, 2))
                x = view(buf, 1:size(A, 2))
                @test _mf_agree(op * x, A * x)
                y0 = randn(size(A, 1))
                y = copy(y0)
                mul!(y, op, x, 0.5, 2.0)
                @test _mf_agree(y, 0.5 * (A * x) + 2.0 * y0)
                @test [op[i, j] for i in axes(A, 1), j in axes(A, 2)] == Matrix(A)
            end
        end
    end

    # `dirichlet_components` limits Γ_D to the named leaves, as it does in `assemble`.
    @testset "dirichlet_components" begin
        W = _mf_spaces()[1]
        V = W × W
        a = form(V, V, (u, v) -> inner₊(∇ₕ(u(1)), ∇ₕ(v(1))) + innerₕ(u(2), v(2)) + innerₕ(u(1), v(2)))
        x = randn(ndofs(V))
        for comps in (1, 2, (1, 2), nothing)
            A = assemble(a; dirichlet = :boundary, dirichlet_components = comps)
            op = matrix_free_operator(a; dirichlet = :boundary, dirichlet_components = comps)
            @test _mf_agree(op * x, A * x)
        end
        @test !(matrix_free_operator(a; dirichlet = :boundary) * x ≈
                assemble(a; dirichlet = :boundary, dirichlet_components = 2) * x)
        @test_throws ArgumentError matrix_free_operator(
            a; dirichlet = :boundary, dirichlet_components = 3)
        Ws = _mf_spaces()[1]
        as = form(Ws, Ws, (u, v) -> innerₕ(u, v))
        @test_throws ArgumentError matrix_free_operator(
            as; dirichlet = :boundary, dirichlet_components = 2)
    end

    # Both `show` forms print one line, never the dense printer's one product per entry.
    @testset "show is a one-line summary" begin
        W = _mf_spaces()[1]
        op = matrix_free_operator(form(W, W, (u, v) -> innerₕ(u, v)); dirichlet = :boundary)
        s2 = repr(op)
        s3 = repr(MIME"text/plain"(), op)
        @test s2 == s3
        @test s2 == "17×17 MatrixFreeOperator{Float64} with Dirichlet rows on (:boundary,)"
        @test sprint(print, op) == s2
    end

    @testset "GpuPolicy refused (v4.4.0)" begin
        W = _mf_spaces()[1]
        a = form(W, W, (u, v) -> innerₕ(u, v))
        err = try
            matrix_free_operator(a; policy = Bramble.GpuKernel())
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("v4.4.0", sprint(showerror, err))
    end
end

end # module
