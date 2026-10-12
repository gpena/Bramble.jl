# test/ext/sparse_csr_ext.jl: the `SparseMatrixCSR` backend extension (ext/BrambleSparseMatricesCSRExt.jl).
#
# Gated like every other ext/*.jl file (test/runtests.jl only reaches this group under
# `BRAMBLE_TEST_GROUP=ext` or `full`). Standalone:
#
#   julia --project=test -e 'using Bramble, Test; include("test/TestUtils.jl");
#     include("test/ext/sparse_csr_ext.jl")'
# test/runtests.jl includes test/ext/SolverContracts.jl before this file; included here
# too, guarded, so the file also loads standalone (as in appleaccelerate_ext.jl).
if !isdefined(Main, :ExtSolverContracts)
    include(joinpath(@__DIR__, "SolverContracts.jl"))
end

module TestSparseMatricesCSRExt

using Test
using Bramble
using Bramble: vector_type, matrix_type, execution_policy
using Bramble: matrix, backend_eye, backend_zeros, allocate_system_matrix,
               assemble_parallel!, refactor!, sparse_factorize, sparse_refactor!,
               SuiteSparseFactorization
# Internal since v3.0: defined and documented, not exported.
using Bramble: D₊ₓ, D₊ᵧ
using SparseArrays
using SparseMatricesCSR
using LinearAlgebra: issymmetric, I, Diagonal
using Random
using Sparspak, SuiteSparse, MUMPS, ILUZero
using Bramble: sparspak_solve, suitesparse_solve, mumps_solve, accelerate_solve,
               suitesparse_qr_solve, sparspak_factorize, suitesparse_factorize,
               mumps_factorize, accelerate_factorize, suitesparse_qr_factorize

if Sys.isapple()
    using AppleAccelerate
end

# `docs/src/internals/csr_solvers.md`: the `sparse_factorize`/
# `pde_solve`/`refactor!` CSC-conversion fallback for `SparseMatrixCSR`. `refactor_contract`
# is the one shared contract that takes its Poisson system `p` as an argument rather than
# building it internally (`poisson_system` always uses the default CSC backend), so it is
# directly reusable against a CSR-built `p` below; `poisson_solve_contract` is not (its
# `solve`/`solve_form` closures would still only ever see the CSC systems `poisson_system`
# builds), so the 1D/2D/3D unified-dispatcher checks it would otherwise cover are written out
# by hand against CSR-built systems instead.
using ..ExtSolverContracts: refactor_contract
using ..TestUtils: _grid, _sine_source, _unit_cube

const ZERO_BC = :dir => (x -> 0.0)

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

# The CSR half of `_poisson_pair`, assembled with Dirichlet data and symmetrized, in the
# `(; Wₕ, a, l, A, F, u_ref)` shape `SolverContracts.poisson_system` returns -- what
# `refactor_contract` expects as its `p` argument.
function _csr_poisson_system(dim::Val{D}, n::Integer; source = _sine_source(dim)) where {D}
    pair = _poisson_pair(dim, n; source = source)
    A, F = assemble(pair.ar, pair.lr; dirichlet = ZERO_BC, symmetrize = true)
    return (; Wₕ = pair.Wr, a = pair.ar, l = pair.lr, A = A, F = F, u_ref = A \ F)
end

# `assemble` infers a union that includes a dense `Matrix`, which has no `nnz`; the matrices
# counted here are always CSC or CSR, so the assertion narrows the type for JET.
function _nnz(A)
    @assert A isa Union{SparseMatrixCSC, SparseMatrixCSR}
    return nnz(A)
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
        @test _nnz(A) == 0

        I4 = backend_eye(be, 4)
        @test I4 isa SparseMatrixCSR{1, Float64, Int}
        @test Matrix(I4) == Matrix(1.0I, 4, 4)

        Z4 = backend_zeros(be, 4)
        @test Z4 isa SparseMatrixCSR{1, Float64, Int}
        @test _nnz(Z4) == 0
    end

    @testset "Poisson 1D/2D/3D: agrees with CSC" begin
        for (D, n) in ((1, 21), (2, 9), (3, 5))
            p = _poisson_pair(Val(D), n)

            Ac, Ar = assemble(p.ac), assemble(p.ar)
            @test Ar isa SparseMatrixCSR
            @test isapprox(Matrix(Ac), Matrix(Ar); atol = 1.0e-12)
            @test _nnz(Ac) == _nnz(Ar)

            Acd, Fc = assemble(p.ac, p.lc; dirichlet = ZERO_BC, symmetrize = true)
            Ard, Fr = assemble(p.ar, p.lr; dirichlet = ZERO_BC, symmetrize = true)
            @test isapprox(Matrix(Acd), Matrix(Ard); atol = 1.0e-12)
            @test isapprox(Fc, Fr; atol = 1.0e-12)

            @test isapprox(Acd \ Fc, Ard \ Fr; atol = 1.0e-10)
        end
    end

    # Unsymmetric convection-diffusion assembly agrees with CSC.
    @testset "convection-diffusion: agrees with CSC" begin
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

    # Assembly, Dirichlet and symmetrize of a composite two-field form agree with CSC.
    @testset "two-field form: agrees with CSC" begin
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

    # An `Int32` CSR target scatters into the same entries as an `Int` one
    # (gpena/Bramble.jl#469): the row search answers an `Int` position for any index type.
    @testset "Int32 CSR agrees with Int CSR" begin
        Random.seed!(469)
        Ω = mesh(
            domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 11), (false, false);
            backend = csr_backend()
        )
        W = gridspace(Ω)
        a = form(W, W, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
        A64 = allocate_system_matrix(a)
        @assert A64 isa SparseMatrixCSR{1, Float64, Int}
        A32 = SparseMatrixCSR{1}(size(A64)..., Int32.(A64.rowptr), Int32.(A64.colval),
            copy(A64.nzval))
        @test @inferred(Bramble._scatter_position(A32, 1, 1)) === 1

        entries = ((1, 1), (2, 1), (5, 6), (99, 99))
        for A in (A64, A32)
            fill!(A.nzval, 0.0)
            for (k, (i, j)) in enumerate(entries)
                Bramble.add_to_sparse!(A, i, j, Float64(k), nothing)
            end
        end
        @test A32.nzval == A64.nzval && count(!iszero, A32.nzval) == 4
        @test all(A32[i, j] == k for (k, (i, j)) in enumerate(entries))

        assemble!(A64, a)
        assemble!(A32, a)
        @test A32.rowptr == A64.rowptr && A32.colval == A64.colval
        @test isequal(A32.nzval, A64.nzval)
    end

    # assemble! is allocation-free after warm-up.
    @testset "assemble!: allocation-free (Serial)" begin
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
        # back to the ordinary serial record pass regardless of thread count.
        p = _poisson_pair(Val(2), 9)
        A_serial = assemble(p.ar)

        A_par = allocate_system_matrix(p.ar)
        assemble_parallel!(A_par, p.ar)
        @test isapprox(Matrix(A_serial), Matrix(A_par); atol = 1.0e-12)
    end

    # sparse_factorize and pde_solve accept SparseMatrixCSR through a CSC-conversion fallback.
    @testset "factorize/pde_solve accept CSR" begin
        # The unified-dispatcher half of `poisson_solve_contract` (1D/3D `pde_solve`, 2D
        # `sparse_factorize`), against CSR-built systems -- see the module-level comment on
        # why the shared contract itself isn't reusable here.
        for (D, n) in ((1, 21), (2, 12), (3, 6))
            p = _csr_poisson_system(Val(D), n)
            @test p.A isa SparseMatrixCSR

            u = pde_solve(p.A, p.F)
            @test isapprox(u, p.u_ref; atol = 1.0e-10)

            fact = sparse_factorize(p.A)
            @test fact isa SuiteSparseFactorization
            @test isapprox(fact \ p.F, p.u_ref; atol = 1.0e-10)
        end
    end

    # Factorization reuse and refactoring go through the CSC-conversion fallback.
    @testset "factorization reuse, refactor" begin
        p = _csr_poisson_system(Val(2), 9)
        refactor_contract(
            p;
            atol = 1.0e-10,
            solver = :default,
            facttype = SuiteSparseFactorization,
            factorize = sparse_factorize,
            backend_refactor! = sparse_refactor!
        )
    end

    # `sparsecsr(Val(0), ...)` stores `rowptr`/`colval` from 0 (gpena/Bramble.jl#540). The
    # matrix is unsymmetric, so a transposed conversion would also fail these checks.
    @testset "zero-based CSR: solve and factorize" begin
        rows, cols, vals = [1, 2, 2, 3], [1, 1, 2, 3], [4.0, -1.0, 4.0, 4.0]
        A0 = sparsecsr(Val(0), rows, cols, vals, 3, 3)
        @test A0.rowptr[1] == 0
        F = [1.0, 2.0, 3.0]
        ref = Matrix(sparse(rows, cols, vals, 3, 3)) \ F

        @test pde_solve(A0, F) ≈ ref
        fact = sparse_factorize(A0)
        @test fact \ F ≈ ref

        # new values, same pattern: `refactor!` must solve with the zero-based matrix's values
        vals2 = [2.0, 3.0, 5.0, 7.0]
        ref2 = Matrix(sparse(rows, cols, vals2, 3, 3)) \ F
        @test refactor!(fact, sparsecsr(Val(0), rows, cols, vals2, 3, 3)) \ F ≈ ref2

        # the one-based path is unchanged
        @test pde_solve(sparsecsr(rows, cols, vals, 3, 3), F) ≈ ref
    end

    # The Dirichlet and symmetrize methods shift a zero-based matrix's `rowptr`/`colval` by
    # one (gpena/Bramble.jl#666): read unshifted, they miss every diagonal and throw, or
    # write the neighbouring entry. The oracle is the CSC result on a non-uniform mesh.
    @testset "zero-based CSR: Dirichlet, symmetrize" begin
        Random.seed!(666)
        S = interval(0.0, 1.0) × interval(0.0, 1.0)
        Ωₕ = mesh(domain(S, :dir => x -> x[1] == 0), (6, 5), (false, false))
        W = gridspace(Ωₕ)
        a = form(W, W, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
        Ac = assemble(a)
        n = size(Ac, 1)
        r = findnz(Ac)
        A0, A1 = sparsecsr(Val(0), r..., n, n), sparsecsr(Val(1), r..., n, n)
        @test A0.rowptr[1] == 0 && A1.rowptr[1] == 1

        # scalar space: the indices path
        Rc = dirichlet_bc!(copy(Ac), Ωₕ, :dir)
        @test Rc != Ac
        dirichlet_bc!(A1, Ωₕ, :dir)
        dirichlet_bc!(A0, Ωₕ, :dir)
        @test Matrix(A0) == Matrix(Rc)
        @test Matrix(A1) == Matrix(Rc)

        Fc, F1, F0 = collect(1.0:n), collect(1.0:n), collect(1.0:n)
        symmetrize!(Rc, Fc, Ωₕ, :dir)
        symmetrize!(A1, F1, Ωₕ, :dir)
        symmetrize!(A0, F0, Ωₕ, :dir)
        @test Fc != collect(1.0:n)
        @test Matrix(A0) == Matrix(Rc) && F0 == Fc
        @test Matrix(A1) == Matrix(Rc) && F1 == Fc

        B0 = sparsecsr(Val(0), r..., n, n)
        assemble!(B0, a; dirichlet = :dir)
        @test Matrix(B0) == Matrix(assemble(a; dirichlet = :dir))

        # W -> W × W, taller than wide: the rows path, and rows past the last column
        g = form(W, W × W, (u, V) -> innerₕ(u, V(1)) + innerₕ(u, V(2)))
        Gc = assemble(g)
        q = findnz(Gc)
        G0 = sparsecsr(Val(0), q..., size(Gc)...)
        Gr = dirichlet_bc!(copy(Gc), Bramble.test_space(g), :dir)
        @test Gr != Gc
        dirichlet_bc!(G0, Bramble.test_space(g), :dir)
        @test Matrix(G0) == Matrix(Gr)

        # a zero-based row with no stored diagonal still throws, on every method
        D0 = sparsecsr(Val(0), [1, 2], [2, 1], [1.0, 1.0], 2, 2)
        m = BitVector([true, false])
        @test_throws ArgumentError Bramble._dirichlet_bc_indices!(copy(D0), m)
        @test_throws ArgumentError Bramble._dirichlet_bc_rows!(
            copy(D0), ((m, 0, nothing, true),)
        )
        @test_throws ArgumentError symmetrize!(copy(D0), ones(2), m, 0)

        # zero-based, wide: refused before the sweep reads `F` past its end
        Z0 = sparsecsr(Val(0), [1, 1, 2, 2], [1, 3, 2, 3], [2.0, 1.0, 2.0, 1.0], 2, 3)
        Z00 = copy(Z0)
        big = [1.0, 1.0, 1.0e6]
        @test_throws ArgumentError symmetrize!(
            Z0, view(big, 1:2), BitVector([false, false, true]), 0
        )
        @test Z0 == Z00
        @test big == [1.0, 1.0, 1.0e6]
    end

    # The backend solvers' `BilinearForm` methods pass the assembled matrix through
    # `_as_csc`: without it a CSR-backed form reaches the extension's `::SparseMatrixCSC`
    # method as a `MethodError`, and `qr` silently densifies it to `QRCompactWY`. The
    # oracle is a dense solve of the CSC-backed system with nonzero Dirichlet data.
    @testset "backend solvers: BilinearForm on CSR" begin
        S = interval(0.0, 1.0) × interval(0.0, 1.0)
        function mk(be)
            W = gridspace(mesh(domain(S, :boundary => boundary_symbols(S)), (9, 7), (true, true); backend = be))
            a = form(W, W, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(u, v))
            l = form(W, v -> innerₕ(Rₕ(W, x -> 1.0 + x[1] * x[2]), v))
            return a, l
        end
        ac, lc = mk(Bramble.backend())
        ar, lr = mk(csr_backend())
        bc = :boundary => x -> x[1]
        A, F = assemble(ac, lc; dirichlet = bc)
        @test assemble(ar; dirichlet = bc) isa SparseMatrixCSR
        ref = Matrix(A) \ F
        @test !(ref ≈ zero(ref))

        pairs = Any[
            ("sparspak", sparspak_solve, sparspak_factorize),
            ("suitesparse", suitesparse_solve, suitesparse_factorize),
            ("mumps", mumps_solve, mumps_factorize),
            ("suitesparse_qr", suitesparse_qr_solve, suitesparse_qr_factorize)
        ]
        Sys.isapple() && push!(pairs, ("accelerate", accelerate_solve, accelerate_factorize))
        for (name, solve, factorize) in pairs
            @testset "$name" begin
                @test parent(solve(ar, lr; dirichlet = bc)) ≈ ref
                @test factorize(ar; dirichlet = bc) \ F ≈ ref
            end
        end
        # `qr` of a CSR matrix would densify to `QRCompactWY`: the factorization is sparse
        @test suitesparse_qr_factorize(ar; dirichlet = bc) isa SparseArrays.SPQR.QRSparse
    end

    # A dense-backed form assembles into a `Matrix`: `_as_csc` makes every backend solve it.
    # With the extension loaded, the `_X_*` fallbacks reached by any other matrix type say so
    # and name the type, instead of "requires X.jl". The oracle is the dense solve.
    @testset "backend solvers: dense form (#658)" begin
        S = interval(0.0, 1.0) × interval(0.0, 1.0)
        Wd = gridspace(
            mesh(
            domain(S, :boundary => boundary_symbols(S)), (6, 7), (false, false);
            backend = Bramble.backend(matrix_type = Matrix{Float64})
        ),
        )
        a = form(Wd, Wd, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(u, v))
        l = form(Wd, v -> innerₕ(Rₕ(Wd, x -> 1.0 + x[1] * x[2]), v))
        bc = :boundary => x -> x[1]
        A, F = assemble(a, l; dirichlet = bc)
        @test A isa Matrix{Float64}
        ref = A \ F
        @test !(ref ≈ zero(ref))

        pairs = Any[
            ("sparspak", sparspak_solve, sparspak_factorize),
            ("suitesparse", suitesparse_solve, suitesparse_factorize),
            ("mumps", mumps_solve, mumps_factorize),
            ("suitesparse_qr", suitesparse_qr_solve, suitesparse_qr_factorize)
        ]
        Sys.isapple() && push!(pairs, ("accelerate", accelerate_solve, accelerate_factorize))
        for (name, solve, factorize) in pairs
            @testset "$name" begin
                @test parent(solve(a, l; dirichlet = bc)) ≈ ref
                @test factorize(a; dirichlet = bc) \ F ≈ ref
            end
        end
        @test suitesparse_qr_factorize(a; dirichlet = bc) isa SparseArrays.SPQR.QRSparse
        @test sparse_factorize(a; solver = :sparspak, dirichlet = bc) \ F ≈ ref

        # a matrix type no backend takes: the message names it, whichever backend is asked
        D = Diagonal(ones(3))
        for f in (
            () -> Bramble._sparspak_factorize(D), () -> Bramble._sparspak_solve(D, ones(3)),
            () -> Bramble._suitesparse_factorize(D), () -> Bramble._suitesparse_solve(D, ones(3)),
            () -> Bramble._mumps_factorize(D), () -> Bramble._mumps_solve(D, ones(3)),
            () -> Bramble._accelerate_factorize(D), () -> Bramble._accelerate_solve(D, ones(3)),
            () -> ilu_preconditioner(D), () -> Bramble._ilu_operator(D)
        )
            e = try
                f()
                nothing
            catch err
                err
            end
            @test e isa ArgumentError
            msg = e isa ArgumentError ? e.msg : ""
            @test occursin("takes a SparseMatrixCSC; got a $(typeof(D))", msg)
            @test !occursin("requires", msg)
        end
        @test_throws ArgumentError(
            "preconditioner = :ilu0 takes a SparseMatrixCSC; got a $(typeof(D))."
        ) Bramble._ilu_operator(D)
        @test_throws ArgumentError ilu_preconditioner(Matrix(D))
    end

    # A taller-than-wide CSR matrix zeroes the constrained rows past its last column, as
    # CSC does, instead of throwing for their missing diagonal; an in-range row with no
    # stored diagonal still throws.
    @testset "rectangular Dirichlet: agrees with CSC" begin
        mk(be, n) = gridspace(mesh(domain(interval(0.0, 1.0)), n, true; backend = be))
        csc, csr = Bramble.backend(), csr_backend()

        # rows path: W -> W × W (20 × 10)
        g = (W) -> form(W, W × W, (u, V) -> innerₕ(u, V(1)) + innerₕ(u, V(2)))
        Ac = assemble(g(mk(csc, 10)); dirichlet = :boundary)
        Ar = assemble(g(mk(csr, 10)); dirichlet = :boundary)
        @test size(Ac, 1) > size(Ac, 2)
        @test Ar isa SparseMatrixCSR
        @test Matrix(Ar) ≈ Matrix(Ac)

        # indices path: scalar πₕ, 10 -> 14 points
        h = (W1, W2) -> form(W1, W2, (u, v) -> innerₕ(πₕ(u), v))
        Bc = assemble(h(mk(csc, 10), mk(csc, 14)); dirichlet = :boundary)
        Br = assemble(h(mk(csr, 10), mk(csr, 14)); dirichlet = :boundary)
        @test size(Bc, 1) > size(Bc, 2)
        @test Br isa SparseMatrixCSR
        @test Matrix(Br) ≈ Matrix(Bc)

        # positive control: row 1 is in range and stores no diagonal
        Ad = sparsecsr([1, 2], [2, 1], [1.0, 1.0], 2, 2)
        m = BitVector([true, false])
        @test_throws ArgumentError Bramble._dirichlet_bc_indices!(copy(Ad), m)
        @test_throws ArgumentError Bramble._dirichlet_bc_rows!(
            copy(Ad), ((m, 0, nothing, true),)
        )
    end

    # `symmetrize!` refuses a non-square matrix with the same `ArgumentError` on CSR as on
    # CSC, before touching it. At mask level the CSR sweep read `F[col]` past the end of a
    # vector as long as the rows, under `@inbounds` (gpena/Bramble.jl#663).
    @testset "non-square symmetrize!: refused like CSC" begin
        mk(be) = gridspace(mesh(domain(interval(0.0, 1.0)), 7, false; backend = be))
        errs = map((Bramble.backend(), csr_backend())) do be
            W = mk(be)
            a = form(W, W × W, (u, V) -> innerₕ(u, V(1)) + innerₕ(u, V(2)))
            A = assemble(a; dirichlet = :boundary)
            F = collect(1.0:size(A, 1))
            A0, F0 = copy(A), copy(F)
            e = try
                symmetrize!(A, F, Bramble.test_space(a), :boundary)
                nothing
            catch err
                err
            end
            @test A == A0
            @test F == F0
            e
        end
        @test errs[1] isa ArgumentError
        @test errs[2] isa ArgumentError
        @test errs[1].msg == errs[2].msg

        # mask level, wide: `F` has two entries and the marked column is the third
        Ar = sparsecsr([1, 1, 2, 2], [1, 3, 2, 3], [2.0, 1.0, 2.0, 1.0], 2, 3)
        Ar0 = copy(Ar)
        big = [1.0, 1.0, 1.0e6]
        Fr = view(big, 1:2)
        @test_throws ArgumentError symmetrize!(Ar, Fr, BitVector([false, false, true]), 0)
        @test Ar == Ar0
        @test big == [1.0, 1.0, 1.0e6]
    end

    # sparse_factorize still rejects non-CSR, non-CSC types.
    @testset "other matrix types rejected" begin
        # The catch-all's guarantee for a genuinely unsupported type (dense `Matrix`,
        # test/form/sparse_solvers.jl) is unweakened by the CSR fallback: loading this
        # extension only ever widens dispatch for a `SparseMatrixCSR`, never for anything
        # else.
        @test_throws MethodError sparse_factorize(rand(4, 4))
    end
end

end # module
