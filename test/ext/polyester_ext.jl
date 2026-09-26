# test/ext/polyester_ext.jl: the Polyester extension (S7.2, gpena/Bramble.jl#190,
# ext/BramblePolyesterExt.jl).
#
# Gated like every other ext/*.jl file (test/runtests.jl only reaches this group under
# BRAMBLE_TEST_GROUP=ext or full). Standalone:
#
#   julia --project=test -e 'using Bramble, Test; include("test/TestUtils.jl");
#     include("test/ext/polyester_ext.jl")'
module TestPolyesterExt

using Test
using Bramble
using Bramble: CpuPolyester, Serial, execution_policy, test_space, _normalize_dirichlet,
               apply_dirichlet_conditions!, allocate_system_matrix, assemble_parallel!,
               inner₊ₓ, D₋ₓ
using Polyester
using SparseArrays
using SparseArrays: getcolptr
using LinearAlgebra: issymmetric
using Random

const ZERO_BC = :dir => (x -> 0.0)

_unit_cube(::Val{D}) where {D} = reduce(×, ntuple(_ -> interval(0.0, 1.0), Val(D)))
_sine_source(::Val{1}) = x -> sin(π * x)
_sine_source(::Val{D}) where {D} = x -> prod(sin(π * xᵢ) for xᵢ in x)

_grid(::Val{1}, Ωd, n; backend) = mesh(Ωd, n, true; backend = backend)
function _grid(::Val{D}, Ωd, n; backend) where {D}
    mesh(
        Ωd, ntuple(_ -> n, Val(D)), ntuple(_ -> true, Val(D)); backend = backend
    )
end

# One matched CpuPolyester/Parallel/Serial triple -- same domain, same mesh size, one backend
# swapped for another -- mirroring test/ext/sparse_csr_ext.jl's own `_poisson_pair`.
function _poisson_pair(dim::Val{D}, n::Integer; source = _sine_source(dim)) where {D}
    Iᴰ = _unit_cube(dim)
    Ωd = domain(Iᴰ, :dir => boundary_symbols(Iᴰ))
    Ωp = _grid(dim, Ωd, n; backend = backend(policy = Parallel()))
    Ωb = _grid(dim, Ωd, n; backend = backend(policy = CpuPolyester()))
    Ωs = _grid(dim, Ωd, n; backend = backend(policy = Serial()))
    Wp, Wb, Ws = gridspace(Ωp), gridspace(Ωb), gridspace(Ωs)

    build = (W) -> begin
        fₕ = Rₕ(W, source)
        a = form(W, W, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
        l = form(W, v -> innerₕ(fₕ, v))
        return a, l
    end
    ap, lp = build(Wp)
    ab, lb = build(Wb)
    as, ls = build(Ws)
    return (; Wp = Wp, Wb = Wb, Ws = Ws, ap = ap, lp = lp, ab = ab, lb = lb, as = as, ls = ls)
end

@testset "Polyester extension (CpuPolyester)" begin
    @testset "CpuPolyester backend and grid space (needs this extension, S7.1)" begin
        be = backend(policy = CpuPolyester())
        @test execution_policy(be) === CpuPolyester()

        Ω = domain(box((0.0, 0.0), (1.0, 1.0)))
        Ωb = mesh(Ω, (8, 7), true; backend = be)
        # `space_weights` fills through the same policy-dispatched sweep as everything
        # else (S7.1's own finding), so this is the first CpuPolyester call any program makes
        # -- it raises naming Polyester without this extension loaded.
        Wb = gridspace(Ωb)
        @test ndofs(Wb) == ndofs(gridspace(mesh(Ω, (8, 7), true)))
    end

    @testset "Rₕ!/avgₕ! agree with Parallel() and Serial(), 1D/2D/3D" begin
        for (D, n) in ((1, 21), (2, 11), (3, 6))
            p = _poisson_pair(Val(D), n)
            src = _sine_source(Val(D))

            up, ub, us = Rₕ(p.Wp, src), Rₕ(p.Wb, src), Rₕ(p.Ws, src)
            @test isapprox(parent(up), parent(ub); atol = 1.0e-12)
            @test isapprox(parent(us), parent(ub); atol = 1.0e-12)

            avgₕ!(up, src)
            avgₕ!(ub, src)
            avgₕ!(us, src)
            @test isapprox(parent(up), parent(ub); atol = 1.0e-12)
            @test isapprox(parent(us), parent(ub); atol = 1.0e-12)
        end
    end

    @testset "Bilinear assemble/assemble!/assemble_parallel! agree with Parallel(), 1D/2D/3D" begin
        for (D, n) in ((1, 21), (2, 9), (3, 5))
            p = _poisson_pair(Val(D), n)

            Ap = assemble(p.ap)
            Ab = assemble(p.ab)
            @test isapprox(Matrix(Ap), Matrix(Ab); atol = 1.0e-12)

            # `assemble!` into a matrix pre-filled with garbage: if the zeroing
            # `_assemble_bilinear!` does before dispatching to the sweep (`_zero_stored!(A)`)
            # were ever skipped for `CpuPolyester`, this would silently add the garbage into the
            # real entries instead of replacing them -- exactly the trap gpena/Bramble.jl#190
            # records from a previous attempt.
            Ab2 = allocate_system_matrix(p.ab)
            fill!(nonzeros(Ab2), 999.0)
            assemble!(Ab2, p.ab)
            @test isapprox(Matrix(Ap), Matrix(Ab2); atol = 1.0e-12)

            Ab3 = allocate_system_matrix(p.ab)
            fill!(nonzeros(Ab3), -777.0)
            assemble_parallel!(Ab3, p.ab)
            @test isapprox(Matrix(Ap), Matrix(Ab3); atol = 1.0e-12)

            Apd, Fp = assemble(p.ap, p.lp; dirichlet = ZERO_BC, symmetrize = true)

            # `assemble(a::BilinearForm, l::LinearForm; ...)` calls `assemble(l; ...)` for
            # the vector half, and `LinearForm`'s own `assemble`/`assemble!` refuse any
            # `CpuPolyester` policy unconditionally, Polyester loaded or not
            # (`src/assembly/linear.jl`'s own fail-fast, S7.1's design -- see the "integrator
            # item" testset below, and this subplan's final report). `BilinearForm`'s
            # `assemble`/`assemble!` carry no such guard, so the matrix half reaches this
            # extension's hooks the direct way; the vector half is built the way
            # `assemble_parallel!(b, ::LinearForm)` documents itself as reaching those same
            # hooks regardless of policy, then the same dirichlet/symmetrize steps
            # `assemble(a, l; ...)` itself performs are applied by hand.
            Abd = assemble(p.ab; dirichlet = ZERO_BC)
            Fb = similar(parent(element(p.Wb)))
            assemble_parallel!(Fb, p.lb)
            dirichlet_labels, dirichlet_conditions = _normalize_dirichlet(ZERO_BC)
            apply_dirichlet_conditions!(Fb, p.lb, dirichlet_conditions, dirichlet_labels, nothing)
            symmetrize!(Abd, Fb, test_space(p.ab), dirichlet_labels...; components = nothing)

            @test isapprox(Matrix(Apd), Matrix(Abd); atol = 1.0e-12)
            @test isapprox(Fp, Fb; atol = 1.0e-12)
            @test isapprox(Apd \ Fp, Abd \ Fb; atol = 1.0e-10)
        end
    end

    @testset "Linear assemble_parallel! agrees with Parallel(); assemble/assemble! (integrator item)" begin
        # `assemble_parallel!(b, ::LinearForm)` forces `_assemble_linear_parallel_core!`
        # regardless of the space's own backend policy (`src/assembly/linear.jl`'s own
        # documented contract), and that core computes its *effective* policy the same way
        # the bilinear sweep does, so `CpuPolyester` reaches this extension's
        # `_batch_linear_colour_sweep!`/`_batch_linear_band_sweep!` exactly as `Parallel()`
        # reaches `Threads.@threads`.
        for (D, n) in ((1, 21), (2, 9), (3, 5))
            p = _poisson_pair(Val(D), n)

            bp = parent(element(p.Wp))
            bb = parent(element(p.Wb))
            assemble_parallel!(bp, p.lp)
            fill!(bb, 555.0)   # the same zeroing check as the matrix case above
            assemble_parallel!(bb, p.lb)
            @test isapprox(bp, bb; atol = 1.0e-12)
        end

        # `assemble`/`assemble!` on a `LinearForm` used to throw unconditionally under
        # `CpuPolyester`: `_assemble_linear!` fast-failed on the policy before it could reach
        # `_assemble_linear_parallel_core!`, so it fired even with Polyester loaded and every
        # hook implemented, and a linear form could never be assembled under this policy at
        # all. The integrator removed that branch on 2026-09-19 (gpena/Bramble.jl#190); the
        # `else` branch dispatches on the effective policy and reaches this extension's
        # hooks, and without Polyester the hook itself still raises, naming the package, one
        # frame deeper. So these now assert agreement rather than the old failure.
        p2 = _poisson_pair(Val(2), 9)
        @test assemble(p2.lb) ≈ assemble(p2.lp)
        bb = similar(parent(element(p2.Wb)))
        bp = similar(parent(element(p2.Wp)))
        assemble!(bb, p2.lb)
        assemble!(bp, p2.lp)
        @test bb ≈ bp
    end

    @testset "innerₕ/inner₊ₓ agree with Parallel(), including masked and multi-marker _dot" begin
        for (D, n) in ((1, 21), (2, 11), (3, 6))
            p = _poisson_pair(Val(D), n)
            src = _sine_source(Val(D))
            up, ub = Rₕ(p.Wp, src), Rₕ(p.Wb, src)
            vp, vb = Rₕ(p.Wp, x -> 1.0), Rₕ(p.Wb, x -> 1.0)

            @test isapprox(innerₕ(up, vp), innerₕ(ub, vb); atol = 1.0e-12)
            @test isapprox(inner₊ₓ(up, vp), inner₊ₓ(ub, vb); atol = 1.0e-12)
        end

        # A single named region: `_dot_masked(u, v, w, mask::BitVector)`.
        S = interval(0.0, 1.0) × interval(0.0, 1.0)
        Ωd = domain(S, :bottom => :bottom, :left => :left)
        Ωp = mesh(Ωd, (12, 12), (true, true); backend = backend(policy = Parallel()))
        Ωb = mesh(Ωd, (12, 12), (true, true); backend = backend(policy = CpuPolyester()))
        Wp, Wb = gridspace(Ωp), gridspace(Ωb)
        up, ub = Rₕ(Wp, x -> x[1] + x[2]), Rₕ(Wb, x -> x[1] + x[2])
        vp, vb = Rₕ(Wp, x -> 1.0), Rₕ(Wb, x -> 1.0)

        @test isapprox(
            innerₕ(up, vp; markers = (:bottom,)), innerₕ(ub, vb; markers = (:bottom,));
            atol = 1.0e-12
        )
        # Two markers: `_dot_masked(u, v, w, mask::MarkedIndicesUnion)`.
        @test isapprox(
            innerₕ(up, vp; markers = (:bottom, :left)),
            innerₕ(ub, vb; markers = (:bottom, :left));
            atol = 1.0e-12
        )
    end

    @testset "Composite two-field form agrees with Parallel()" begin
        n1, n2 = 9, 7
        I2 = interval(0.0, 1.0) × interval(0.0, 1.0)
        Ωd = domain(I2, :dir => boundary_symbols(I2))
        Ωp = mesh(Ωd, (n1, n2), (true, true); backend = backend(policy = Parallel()))
        Ωb = mesh(Ωd, (n1, n2), (true, true); backend = backend(policy = CpuPolyester()))
        Vp, Vb = gridspace(Ωp, Val(2)), gridspace(Ωb, Val(2))

        g = (V) -> form(
            V, V,
            (u, v) -> inner₊(∇ₕ(u(1)), ∇ₕ(v(1))) + innerₕ(u(2), v(2)) +
                      innerₕ(u(1), v(2))
        )

        Ap, Ab = assemble(g(Vp)), assemble(g(Vb))
        @test isapprox(Matrix(Ap), Matrix(Ab); atol = 1.0e-12)

        Apd, Abd = assemble(g(Vp); dirichlet = (:dir,)), assemble(g(Vb); dirichlet = (:dir,))
        @test isapprox(Matrix(Apd), Matrix(Abd); atol = 1.0e-12)
    end

    @testset "Allocation: CpuPolyester against what test/space/vector_elements.jl's Parallel() accepts" begin
        # Function barriers (bramble-verification §1): the warm-up call and the measured
        # call both happen inside one function, over its own arguments.
        function _avg_allocs(u, f)
            avgₕ!(u, f)
            return @allocated avgₕ!(u, f)
        end
        function _assemble_allocs(A, a)
            assemble!(A, a)
            return @allocated assemble!(A, a)
        end
        function _dot_allocs(u, v)
            innerₕ(u, v)
            return @allocated innerₕ(u, v)
        end

        f2(x) = sin(x[1] + x[2])
        mk(be, n) = element(gridspace(mesh(
            domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (n, n); backend = be)))

        ub_small = mk(backend(policy = CpuPolyester()), 32)
        ub_large = mk(backend(policy = CpuPolyester()), 1024)

        batch_small = _avg_allocs(ub_small, f2)
        batch_large = _avg_allocs(ub_large, f2)
        @info "avgₕ! CpuPolyester allocation diagnostic: small=$batch_small large=$batch_large " *
              "nthreads=$(Threads.nthreads())"

        # The same guarantee test/space/vector_elements.jl's own "Allocation scaling"
        # testset asserts for `Parallel()`: size-independent (batch-spawn overhead, not
        # proportional to grid points) and small in absolute terms. gpena/Bramble.jl#190's
        # own recorded measurement is a naive `@batch` at 64 B/call against `Threads`' 1.6 KB
        # at 128^2 -- lower, not higher, so the same threshold applies without loosening it.
        @test batch_large < 4 * batch_small + 1     # +1 guards small == 0
        @test batch_large < 100_000                 # proportional would be tens of MB

        p = _poisson_pair(Val(2), 9)
        Ab = allocate_system_matrix(p.ab)
        assemble_allocs = _assemble_allocs(Ab, p.ab)
        @info "assemble! (CpuPolyester) allocation diagnostic: $assemble_allocs B"
        @test assemble_allocs < 100_000

        u64, v64 = Rₕ(p.Wb, x -> 1.0), Rₕ(p.Wb, x -> 2.0)
        dot_allocs = _dot_allocs(u64, v64)
        @info "innerₕ (CpuPolyester) allocation diagnostic: $dot_allocs B"
        @test dot_allocs < 100_000
    end

    @testset "Determinism across repeated runs" begin
        p = _poisson_pair(Val(2), 15)
        A1 = assemble(p.ab)
        A2 = assemble(p.ab)
        A3 = assemble(p.ab)
        @test Matrix(A1) == Matrix(A2) == Matrix(A3)

        u, v = Rₕ(p.Wb, x -> sin(x[1])), Rₕ(p.Wb, x -> cos(x[2]))
        d1, d2, d3 = innerₕ(u, v), innerₕ(u, v), innerₕ(u, v)
        @test d1 == d2 == d3
    end

    # A warmed `CpuPolyester` refill replays the form's recorded `nzval` positions instead of
    # searching (gpena/Bramble.jl#338): `_threaded_replay_policy(::CpuPolyester)` and
    # `_batch_bilinear_band_replay!`/`_batch_bilinear_colour_replay!` above. Checked the same
    # way `test/form/threaded_replay.jl` checks `CpuThreaded` -- agreement against a serial
    # `assemble` of the same non-uniform mesh, never against another threaded fill -- since
    # this extension's own `CpuPolyester` vs `Parallel()` testsets above never re-fill an
    # already-assembled matrix and so would not tell a replay from a re-search.
    @testset "Warmed CpuPolyester refill replays the recording (#338)" begin
        _replay_domains = (
            domain(interval(0.0, 1.0)),
            domain(interval(0.0, 1.0) × interval(0.0, 2.0)),
            domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0))
        )
        _replay_mesh(D, n, policy; seed) = begin
            Random.seed!(seed)
            mesh(
                _replay_domains[D], ntuple(_ -> n, D), ntuple(_ -> false, D);
                backend = backend(policy = policy)
            )
        end
        _scalar(u, v) = innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(D₋ₓ(u), v)
        _pair(u, v) = innerₕ(D₋ₓ(u), v) + innerₕ(u, D₋ₓ(v)) + innerₕ(u, v)
        _composite(u, v) = innerₕ(u(1), v(1)) + inner₊(∇ₕ(u(2)), ∇ₕ(v(2))) +
                           innerₕ(D₋ₓ(u(1)), v(2))
        sizes = (41, 13, 7)

        @testset "$(D)D, $(nm)" for D in 1:3,
            (nm, f, comps) in (
                ("scalar", _scalar, 1), ("pair", _pair, 1), ("composite", _composite, 2)
            )

            n = sizes[D]
            Ωs = _replay_mesh(D, n, Serial(); seed = 338)
            Ωb = _replay_mesh(D, n, CpuPolyester(); seed = 338)
            @test points(Ωs) == points(Ωb)
            space(Ω) = comps == 1 ? gridspace(Ω) : gridspace(Ω, Val(comps))
            as = form(space(Ωs), space(Ωs), f)
            ab = form(space(Ωb), space(Ωb), f)
            R = assemble(as)

            B = assemble(ab)
            @test getcolptr(B) == getcolptr(R) && rowvals(B) == rowvals(R)
            @test isapprox(B, R; rtol = 1e-12)

            # A warmed refill (the recording already exists): replays, not re-searches.
            fill!(nonzeros(B), NaN)
            assemble!(B, ab)
            @test getcolptr(B) == getcolptr(R) && rowvals(B) == rowvals(R)
            @test isapprox(B, R; rtol = 1e-12)
        end

        @testset "Warmed refill allocation is independent of grid size" begin
            _alloc(f::F, args...) where {F} = (f(args...); @allocated f(args...))
            sizes2 = (200, 800)
            bytes = map(sizes2) do n
                Ω = _replay_mesh(1, n, CpuPolyester(); seed = 338)
                a = form(gridspace(Ω), gridspace(Ω), _scalar)
                A = assemble(a)
                _alloc(assemble!, A, a)
            end
            @test bytes[1] == bytes[2]
        end
    end
end

# The Polyester-gated mixed-policy testset of test/form/threaded_replay.jl ("Mixed leaf
# policies: CpuThreaded beside CpuPolyester") only runs where `BramblePolyesterExt` is already
# loaded; the `unit` group deliberately never loads Polyester, so that testset never runs in
# CI on its own. Included here as a nested module so it runs wherever this file does (the
# "ext"/"full" groups).
include(joinpath(@__DIR__, "..", "form", "threaded_replay.jl"))

end # module
