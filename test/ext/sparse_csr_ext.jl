# test/ext/sparse_csr_ext.jl: the `SparseMatrixCSR` backend extension (S3.1,
# gpena/Bramble.jl#214, ext/BrambleSparseMatricesCSRExt.jl).
#
# Gated like every other ext/*.jl file (test/runtests.jl only reaches this group under
# `BRAMBLE_TEST_GROUP=ext` or `full`). Standalone:
#
#   julia --project=test -e 'using Bramble, Test; include("test/TestUtils.jl");
#     include("test/ext/sparse_csr_ext.jl")'
module TestSparseMatricesCSRExt

using Test
using Bramble
using Bramble: matrix, backend_eye, backend_zeros
# Internal since v3.0 (gpena/Bramble.jl#211): defined and documented, not exported.
using Bramble: D₊ₓ, D₊ᵧ
using SparseArrays
using SparseMatricesCSR
using LinearAlgebra: issymmetric, I

const ZERO_BC = :dir => (x -> 0.0)

_unit_cube(::Val{D}) where {D} = reduce(×, ntuple(_ -> interval(0.0, 1.0), Val(D)))
_sine_source(::Val{1}) = x -> sin(π * x)
_sine_source(::Val{D}) where {D} = x -> prod(sin(π * xᵢ) for xᵢ in x)

_grid(::Val{1}, Ωd, n; backend) = mesh(Ωd, n, true; backend = backend)
_grid(::Val{D}, Ωd, n; backend) where {D} = mesh(
    Ωd, ntuple(_ -> n, Val(D)), ntuple(_ -> true, Val(D)); backend = backend
)

# One matched CSC/CSR pair -- same domain, same mesh sizes, same discretisation -- for the
# Poisson problem every backend file in test/ext/ shares (`SolverContracts.poisson_system`),
# just built twice: once with the default backend, once with `csr_backend()`.
function _poisson_pair(dim::Val{D}, n::Integer; source = _sine_source(dim)) where {D}
    Iᴰ = _unit_cube(dim)
    Ωd = domain(Iᴰ, :dir => boundary_symbols(Iᴰ))
    Ωc = _grid(dim, Ωd, n; backend = Bramble.backend())
    Ωr = _grid(dim, Ωd, n; backend = csr_backend())
    Wc, Wr = gridspace(Ωc), gridspace(Ωr)

    build = (W) -> begin
        fₕ = Rₕ(W, source)
        a = form(W, W, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
        l = form(W, v -> innerₕ(fₕ, v))
        return a, l
    end
    ac, lc = build(Wc)
    ar, lr = build(Wr)
    return (; Wc = Wc, Wr = Wr, ac = ac, lc = lc, ar = ar, lr = lr)
end

@testset "SparseMatricesCSR extension" begin
    @testset "csr_backend() types" begin
        be = csr_backend()
        @test matrix_type(be) === SparseMatrixCSR{1, Float64, Int}
        @test vector_type(be) === Vector{Float64}
        @test execution_policy(be) === Serial()

        be32 = csr_backend(Float32)
        @test matrix_type(be32) === SparseMatrixCSR{1, Float32, Int}
        @test vector_type(be32) === Vector{Float32}

        bep = csr_backend(; policy = Parallel())
        @test execution_policy(bep) === Parallel()

        # test/utils/backends.jl (S1.4) checks that calling `csr_backend()` *without*
        # `using SparseMatricesCSR` errors, naming the package. This file's own `using
        # SparseMatricesCSR` above loads this extension, so the same call now succeeds
        # instead of hitting `_csr_backend`'s stub in backend.jl.
        A = matrix(be, 4, 4)
        @test A isa SparseMatrixCSR{1, Float64, Int}
        @test size(A) == (4, 4)
        @test nnz(A) == 0

        I4 = backend_eye(be, 4)
        @test I4 isa SparseMatrixCSR{1, Float64, Int}
        @test Matrix(I4) == Matrix(1.0I, 4, 4)

        Z4 = backend_zeros(be, 4)
        @test Z4 isa SparseMatrixCSR{1, Float64, Int}
        @test nnz(Z4) == 0
    end

    @testset "1D/2D/3D Poisson: assemble agrees with CSC" begin
        for (D, n) in ((1, 21), (2, 9), (3, 5))
            p = _poisson_pair(Val(D), n)

            Ac, Ar = assemble(p.ac), assemble(p.ar)
            @test Ar isa SparseMatrixCSR
            @test isapprox(Matrix(Ac), Matrix(Ar); atol = 1.0e-12)
            @test nnz(Ac) == nnz(Ar)

            Acd, Fc = assemble(p.ac, p.lc; dirichlet = ZERO_BC, symmetrize = true)
            Ard, Fr = assemble(p.ar, p.lr; dirichlet = ZERO_BC, symmetrize = true)
            @test isapprox(Matrix(Acd), Matrix(Ard); atol = 1.0e-12)
            @test isapprox(Fc, Fr; atol = 1.0e-12)

            @test isapprox(Acd \ Fc, Ard \ Fr; atol = 1.0e-10)
        end
    end

    @testset "Convection-diffusion (unsymmetric): assemble agrees with CSC" begin
        n = 10
        I2 = interval(0.0, 1.0) × interval(0.0, 1.0)
        Ωd = domain(I2, :dir => boundary_symbols(I2))
        Ωc = mesh(Ωd, (n, n), (true, true))
        Ωr = mesh(Ωd, (n, n), (true, true); backend = csr_backend())
        Wc, Wr = gridspace(Ωc), gridspace(Ωr)

        build = (W) -> begin
            a = form(
                W, W,
                (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + 2.0 * innerₕ(D₊ₓ(u), v) +
                          1.0 * innerₕ(D₊ᵧ(u), v)
            )
            l = form(W, v -> innerₕ(Rₕ(W, x -> 1.0), v))
            return a, l
        end
        ac, lc = build(Wc)
        ar, lr = build(Wr)

        Ac, Fc = assemble(ac, lc; dirichlet = ZERO_BC, symmetrize = false)
        Ar, Fr = assemble(ar, lr; dirichlet = ZERO_BC, symmetrize = false)
        @test !issymmetric(Matrix(Ac))
        @test isapprox(Matrix(Ac), Matrix(Ar); atol = 1.0e-12)
        @test isapprox(Fc, Fr; atol = 1.0e-12)
        @test isapprox(Ac \ Fc, Ar \ Fr; atol = 1.0e-10)
    end

    @testset "Composite two-field form: assemble, Dirichlet and symmetrize agree with CSC" begin
        n1, n2 = 9, 7
        I2 = interval(0.0, 1.0) × interval(0.0, 1.0)
        Ωd = domain(I2, :dir => boundary_symbols(I2))
        Ωc = mesh(Ωd, (n1, n2), (true, true))
        Ωr = mesh(Ωd, (n1, n2), (true, true); backend = csr_backend())
        Vc, Vr = gridspace(Ωc, Val(2)), gridspace(Ωr, Val(2))

        g = (V) -> form(
            V, V,
            (u, v) -> inner₊(∇ₕ(u(1)), ∇ₕ(v(1))) + innerₕ(u(2), v(2)) +
                      innerₕ(u(1), v(2))
        )

        Ac, Ar = assemble(g(Vc)), assemble(g(Vr))
        @test Ar isa SparseMatrixCSR
        @test isapprox(Matrix(Ac), Matrix(Ar); atol = 1.0e-12)

        Acd = assemble(g(Vc); dirichlet = (:dir,))
        Ard = assemble(g(Vr); dirichlet = (:dir,))
        @test isapprox(Matrix(Acd), Matrix(Ard); atol = 1.0e-12)

        Fc, Fr = ones(ndofs(Vc)), ones(ndofs(Vr))
        Bc, Br = copy(Acd), copy(Ard)
        symmetrize!(Bc, Fc, Vc, :dir)
        symmetrize!(Br, Fr, Vr, :dir)
        @test isapprox(Matrix(Bc), Matrix(Br); atol = 1.0e-12)
        @test isapprox(Fc, Fr)
    end

    @testset "assemble! is allocation-free after warm-up (Serial)" begin
        # Function barrier (bramble-verification §1): `@allocated` at top level over a
        # loop/testset-local binding can misreport, so the warm-up call and the measured
        # call both happen inside one function.
        function _assemble_allocs(A, a)
            assemble!(A, a)
            return @allocated assemble!(A, a)
        end

        p = _poisson_pair(Val(2), 9)
        Ar = allocate_system_matrix(p.ar)
        @test _assemble_allocs(Ar, p.ar) == 0
    end

    @testset "assemble_parallel! matches serial" begin
        # `bilinear_execution.jl`'s band-coloured threaded sweep is typed
        # `A::SparseMatrixCSC` throughout (S1.1 leaves it that way -- see the module
        # docstring of `ext/BrambleSparseMatricesCSRExt.jl`), so `SparseMatrixCSR` falls
        # back to the ordinary serial record pass regardless of thread count. Asserted
        # under `Threads.nthreads() > 1` anyway, matching the plan's own check, since that
        # is the interesting regime for a *future* CSR-specific parallel sweep to preserve.
        p = _poisson_pair(Val(2), 9)
        A_serial = assemble(p.ar)

        A_par = allocate_system_matrix(p.ar)
        assemble_parallel!(A_par, p.ar)
        @test isapprox(Matrix(A_serial), Matrix(A_par); atol = 1.0e-12)

        if Threads.nthreads() > 1
            @test isapprox(Matrix(A_serial), Matrix(A_par); atol = 1.0e-12)
        end
    end
end

end # module
