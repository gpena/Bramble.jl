# test/ext/banded_matrices_ext.jl: the `BandedMatrix`/`BandedBlockBandedMatrix` backend
# extension (S4.2, gpena/Bramble.jl#175 #216, ext/BrambleBandedMatricesExt.jl).
#
# Gated like every other ext/*.jl file (test/runtests.jl only reaches this group under
# `BRAMBLE_TEST_GROUP=ext` or `full`). Standalone:
#
#   julia --project=test -e 'using Bramble, Test; include("test/TestUtils.jl");
#     include("test/ext/banded_matrices_ext.jl")'
module TestBandedMatricesExt

using Test
using Bramble
using Bramble: matrix, backend_eye, backend_zeros
# Internal since v3.0 (gpena/Bramble.jl#211): defined and documented, not exported.
using Bramble: D₊ₓ
using SparseArrays
using BandedMatrices
using BandedMatrices: bandwidths
using BlockBandedMatrices
using BlockBandedMatrices: blockbandwidths, subblockbandwidths
using LinearAlgebra
using LinearAlgebra: issymmetric, I, Symmetric, factorize, cholesky

const ZERO_BC = :dir => (x -> 0.0)

_unit_cube(::Val{D}) where {D} = reduce(×, ntuple(_ -> interval(0.0, 1.0), Val(D)))
_sine_source(::Val{1}) = x -> sin(π * x)
_sine_source(::Val{D}) where {D} = x -> prod(sin(π * xᵢ) for xᵢ in x)

_grid(::Val{1}, Ωd, n; backend) = mesh(Ωd, n, true; backend = backend)
_grid(::Val{D}, Ωd, n; backend) where {D} = mesh(
    Ωd, ntuple(_ -> n, Val(D)), ntuple(_ -> true, Val(D)); backend = backend
)

# One matched CSC/backend pair -- same domain, same mesh sizes, same discretisation -- for
# the Poisson problem every backend file in test/ext/ shares (mirroring
# `test/ext/sparse_csr_ext.jl`'s `_poisson_pair`), built once against the default backend and
# once against `be` (`banded_backend()` in 1D, `block_banded_backend()` in 2D/3D).
function _poisson_pair(dim::Val{D}, n::Integer, be; source = _sine_source(dim)) where {D}
    Iᴰ = _unit_cube(dim)
    Ωd = domain(Iᴰ, :dir => boundary_symbols(Iᴰ))
    Ωc = _grid(dim, Ωd, n; backend = Bramble.backend())
    Ωr = _grid(dim, Ωd, n; backend = be)
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

@testset "BandedMatrices/BlockBandedMatrices extension" begin
    @testset "banded_backend()/block_banded_backend() types" begin
        # test/utils/backends.jl (S1.4) checks that calling these *without* `using
        # BandedMatrices, BlockBandedMatrices` errors, naming the packages. This file's own
        # `using` above loads this extension, so the same calls now succeed instead of
        # hitting `_banded_backend`/`_block_banded_backend`'s stubs in backend.jl.
        be = banded_backend()
        @test matrix_type(be) === BandedMatrix{Float64, Matrix{Float64}, Base.OneTo{Int}}
        @test vector_type(be) === Vector{Float64}
        @test execution_policy(be) === Serial()

        be32 = banded_backend(Float32)
        @test matrix_type(be32) === BandedMatrix{Float32, Matrix{Float32}, Base.OneTo{Int}}

        bep = banded_backend(; policy = Parallel())
        @test execution_policy(bep) === Parallel()

        A = matrix(be, 4, 4)
        @test A isa BandedMatrix{Float64}
        @test size(A) == (4, 4)
        @test bandwidths(A) == (-1, -1)

        I4 = backend_eye(be, 4)
        @test I4 isa BandedMatrix{Float64}
        @test Matrix(I4) == Matrix(1.0I, 4, 4)

        Z4 = backend_zeros(be, 4)
        @test Z4 isa BandedMatrix{Float64}
        @test bandwidths(Z4) == (-1, -1)

        bbe = block_banded_backend()
        @test matrix_type(bbe) <: BandedBlockBandedMatrix{Float64}
        @test vector_type(bbe) === Vector{Float64}
        @test execution_policy(bbe) === Serial()

        Ab = matrix(bbe, 6, 6)
        @test Ab isa BandedBlockBandedMatrix{Float64}
        @test size(Ab) == (6, 6)

        Ib = backend_eye(bbe, 6)
        @test Matrix(Ib) == Matrix(1.0I, 6, 6)

        Zb = backend_zeros(bbe, 6)
        @test Zb isa BandedBlockBandedMatrix{Float64}
        @test iszero(Matrix(Zb))
    end

    @testset "1D Poisson: assemble agrees with CSC" begin
        p = _poisson_pair(Val(1), 21, banded_backend())

        Ac, Ar = assemble(p.ac), assemble(p.ar)
        @test Ar isa BandedMatrix
        @test isapprox(Matrix(Ac), Matrix(Ar); atol = 1.0e-12)

        Acd, Fc = assemble(p.ac, p.lc; dirichlet = ZERO_BC, symmetrize = true)
        Ard, Fr = assemble(p.ar, p.lr; dirichlet = ZERO_BC, symmetrize = true)
        @test isapprox(Matrix(Acd), Matrix(Ard); atol = 1.0e-12)
        @test isapprox(Fc, Fr; atol = 1.0e-12)
        @test isapprox(Acd \ Fc, Ard \ Fr; atol = 1.0e-10)
    end

    @testset "1D convection-diffusion (unsymmetric): assemble agrees with CSC" begin
        n = 25
        Id = interval(0.0, 1.0)
        Ωd = domain(Id, :dir => boundary_symbols(Id))
        Ωc = mesh(Ωd, n, true)
        Ωr = mesh(Ωd, n, true; backend = banded_backend())
        Wc, Wr = gridspace(Ωc), gridspace(Ωr)

        build = (W) -> begin
            a = form(W, W, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + 2.0 * innerₕ(D₊ₓ(u), v))
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

    @testset "1D innerₕ(Dcₓ(u), Dcₓ(v)) (5-point): assemble agrees with CSC" begin
        n = 21
        Id = interval(0.0, 1.0)
        Ωd = domain(Id, :dir => boundary_symbols(Id))
        Ωc = mesh(Ωd, n, true)
        Ωr = mesh(Ωd, n, true; backend = banded_backend())
        Wc, Wr = gridspace(Ωc), gridspace(Ωr)

        build = (W) -> form(W, W, (u, v) -> innerₕ(Dcₓ(u), Dcₓ(v)))
        ac = build(Wc)
        ar = build(Wr)

        Ac, Ar = assemble(ac), assemble(ar)
        @test Ar isa BandedMatrix
        @test bandwidths(Ar) == (2, 2)
        @test isapprox(Matrix(Ac), Matrix(Ar); atol = 1.0e-12)
    end

    @testset "Composite two-field form (1D): assemble, Dirichlet and symmetrize agree with CSC" begin
        n = 9
        Id = interval(0.0, 1.0)
        Ωd = domain(Id, :dir => boundary_symbols(Id))
        Ωc = mesh(Ωd, n, true)
        Ωr = mesh(Ωd, n, true; backend = banded_backend())
        Vc, Vr = gridspace(Ωc, Val(2)), gridspace(Ωr, Val(2))

        g = (V) -> form(
            V, V,
            (u, v) -> inner₊(∇ₕ(u(1)), ∇ₕ(v(1))) + innerₕ(u(2), v(2)) +
                      innerₕ(u(1), v(2))
        )

        Ac, Ar = assemble(g(Vc)), assemble(g(Vr))
        @test Ar isa BandedMatrix
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

    @testset "bandwidths(A) matches Bramble.bandwidths(a)" begin
        p = _poisson_pair(Val(1), 15, banded_backend())
        Ar = assemble(p.ar)
        @test bandwidths(Ar) == Bramble.bandwidths(p.ar)
    end

    @testset "assemble! is allocation-free after warm-up (1D banded)" begin
        # Function barrier (bramble-verification §1): `@allocated` at top level over a
        # loop/testset-local binding can misreport, so the warm-up call and the measured
        # call both happen inside one function.
        function _assemble_allocs(A, a)
            assemble!(A, a)
            return @allocated assemble!(A, a)
        end

        p = _poisson_pair(Val(1), 21, banded_backend())
        Ar = allocate_system_matrix(p.ar)
        @test _assemble_allocs(Ar, p.ar) == 0
    end

    @testset "1D solve agreement and factorisation type" begin
        p = _poisson_pair(Val(1), 21, banded_backend())
        Acd, Fc = assemble(p.ac, p.lc; dirichlet = ZERO_BC, symmetrize = true)
        Ard, Fr = assemble(p.ar, p.lr; dirichlet = ZERO_BC, symmetrize = true)

        @test isapprox(Acd \ Fc, Ard \ Fr; atol = 1.0e-10)

        # `A \ F` on a `BandedMatrix` routes to LAPACK `gbtrf!`/`gbtrs!` via
        # `BandedMatrices.jl`'s own `factorize` specialisation.
        F = factorize(Ard)
        @test F isa BandedMatrices.BandedLU

        # The Poisson form here is symmetric positive definite once Dirichlet rows/columns
        # are eliminated: `cholesky(Symmetric(A))` should hit `pbtrf!` (the banded LAPACK
        # Cholesky), keeping the `BandedMatrix` storage rather than falling back to a dense
        # factorisation -- asserted by the factor's own type parameter.
        C = cholesky(Symmetric(Ard))
        @test C isa LinearAlgebra.Cholesky{Float64, <:BandedMatrix}
        @test isapprox(C \ Fr, Ard \ Fr; atol = 1.0e-10)
    end

    @testset "2D/3D Poisson (block-banded): assemble agrees with CSC" begin
        for (D, n) in ((2, 9), (3, 5))
            p = _poisson_pair(Val(D), n, block_banded_backend())

            Ac, Ar = assemble(p.ac), assemble(p.ar)
            @test Ar isa BandedBlockBandedMatrix
            @test isapprox(Matrix(Ac), Matrix(Ar); atol = 1.0e-12)

            # `Bramble.blockbandwidths` (S4.1) reads the same bands from the AST alone;
            # the assembled matrix's own bands (`BlockBandedMatrices.jl`) must agree.
            blk_bw, sub_bw = Bramble.blockbandwidths(p.ar)
            @test (blockbandwidths(Ar), subblockbandwidths(Ar)) == (blk_bw, sub_bw)

            Acd, Fc = assemble(p.ac, p.lc; dirichlet = ZERO_BC, symmetrize = true)
            Ard, Fr = assemble(p.ar, p.lr; dirichlet = ZERO_BC, symmetrize = true)
            @test isapprox(Matrix(Acd), Matrix(Ard); atol = 1.0e-12)
            @test isapprox(Fc, Fr; atol = 1.0e-12)
            @test isapprox(Acd \ Fc, Ard \ Fr; atol = 1.0e-10)
        end
    end

    @testset "assemble! is allocation-free after warm-up (2D block-banded)" begin
        function _assemble_allocs(A, a)
            assemble!(A, a)
            return @allocated assemble!(A, a)
        end

        p = _poisson_pair(Val(2), 9, block_banded_backend())
        Ar = allocate_system_matrix(p.ar)
        @test _assemble_allocs(Ar, p.ar) == 0
    end
end

end # module
