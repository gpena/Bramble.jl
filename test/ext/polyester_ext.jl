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
using Bramble: CpuBatch, execution_policy
using Polyester
using SparseArrays
using LinearAlgebra: issymmetric

const ZERO_BC = :dir => (x -> 0.0)

_unit_cube(::Val{D}) where {D} = reduce(×, ntuple(_ -> interval(0.0, 1.0), Val(D)))
_sine_source(::Val{1}) = x -> sin(π * x)
_sine_source(::Val{D}) where {D} = x -> prod(sin(π * xᵢ) for xᵢ in x)

_grid(::Val{1}, Ωd, n; backend) = mesh(Ωd, n, true; backend = backend)
_grid(::Val{D}, Ωd, n; backend) where {D} = mesh(
    Ωd, ntuple(_ -> n, Val(D)), ntuple(_ -> true, Val(D)); backend = backend
)

# One matched CpuBatch/Parallel/Serial triple -- same domain, same mesh size, one backend
# swapped for another -- mirroring test/ext/sparse_csr_ext.jl's own `_poisson_pair`.
function _poisson_pair(dim::Val{D}, n::Integer; source = _sine_source(dim)) where {D}
    Iᴰ = _unit_cube(dim)
    Ωd = domain(Iᴰ, :dir => boundary_symbols(Iᴰ))
    Ωp = _grid(dim, Ωd, n; backend = backend(policy = Parallel()))
    Ωb = _grid(dim, Ωd, n; backend = backend(policy = CpuBatch()))
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

@testset "Polyester extension (CpuBatch)" begin
    @testset "CpuBatch backend and grid space (needs this extension, S7.1)" begin
        be = backend(policy = CpuBatch())
        @test execution_policy(be) === CpuBatch()

        Ω = domain(box((0.0, 0.0), (1.0, 1.0)))
        Ωb = mesh(Ω, (8, 7), true; backend = be)
        # `space_weights` fills through the same policy-dispatched sweep as everything
        # else (S7.1's own finding), so this is the first CpuBatch call any program makes
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
            # were ever skipped for `CpuBatch`, this would silently add the garbage into the
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
            Abd, Fb = assemble(p.ab, p.lb; dirichlet = ZERO_BC, symmetrize = true)
            @test isapprox(Matrix(Apd), Matrix(Abd); atol = 1.0e-12)
            @test isapprox(Fp, Fb; atol = 1.0e-12)
            @test isapprox(Apd \ Fp, Abd \ Fb; atol = 1.0e-10)
        end
    end

    @testset "Linear assemble_parallel! agrees with Parallel(); assemble/assemble! (integrator item)" begin
        # `assemble_parallel!(b, ::LinearForm)` forces `_assemble_linear_parallel_core!`
        # regardless of the space's own backend policy (`src/form/linear.jl`'s own
        # documented contract), and that core computes its *effective* policy the same way
        # the bilinear sweep does, so `CpuBatch` reaches this extension's
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

        # `assemble`/`assemble!` on a `LinearForm`, by contrast, throw unconditionally under
        # `CpuBatch` -- `src/form/linear.jl`'s `_assemble_linear!` has
        # `elseif policy isa CpuBatch; _throw_cpubatch_without_polyester(:assemble!)` ahead
        # of ever reaching `_assemble_linear_parallel_core!`, regardless of whether Polyester
        # is loaded. `BilinearForm`'s `assemble!` (`src/form/bilinear.jl`) has no such guard
        # and falls straight through to the effective-policy dispatch, which is why the
        # testset above works for the matrix but this one only documents the vector's
        # current behaviour. Reported as a blocker in this subplan's final report rather
        # than fixed here: `src/form/linear.jl` belongs to S7.1, not S7.2's OWNS.
        p2 = _poisson_pair(Val(2), 9)
        @test_throws ArgumentError assemble(p2.lb)
        @test_throws ArgumentError assemble!(similar(parent(element(p2.Wb))), p2.lb)
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
        Ωb = mesh(Ωd, (12, 12), (true, true); backend = backend(policy = CpuBatch()))
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
        Ωb = mesh(Ωd, (n1, n2), (true, true); backend = backend(policy = CpuBatch()))
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

    @testset "Allocation: CpuBatch against what test/space/vector_elements.jl's Parallel() accepts" begin
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

        ub_small = mk(backend(policy = CpuBatch()), 32)
        ub_large = mk(backend(policy = CpuBatch()), 1024)

        batch_small = _avg_allocs(ub_small, f2)
        batch_large = _avg_allocs(ub_large, f2)
        @info "avgₕ! CpuBatch allocation diagnostic: small=$batch_small large=$batch_large " *
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
        @info "assemble! (CpuBatch) allocation diagnostic: $assemble_allocs B"
        @test assemble_allocs < 100_000

        u64, v64 = Rₕ(p.Wb, x -> 1.0), Rₕ(p.Wb, x -> 2.0)
        dot_allocs = _dot_allocs(u64, v64)
        @info "innerₕ (CpuBatch) allocation diagnostic: $dot_allocs B"
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
end

end # module
