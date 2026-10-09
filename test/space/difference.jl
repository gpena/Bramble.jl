module SpaceDifferenceTests

using Test
using Bramble
using Bramble: vector_type, matrix_type
using Bramble: D₋ᵧ, D₋₂, D₋ₓ, GpuKernel, set_points!
import Bramble: space, eltype, ⊗, _Eye, shift, npoints, spacing, diff₋ₓ, diff₋ᵧ, diff₋₂, diff₊ₓ, diff₊ᵧ, diff₊₂, D₊ₓ,
                D₊ᵧ, D₊₂
using Bramble: backward_difference_dim!, forward_difference_dim!
using Bramble: diff₋ₓ, diff₋ᵧ, diff₋₂, diff₊ₓ, diff₊ᵧ, diff₊₂
import SparseArrays: issparse, sprand, spdiagm, spzeros, nnz
using Bramble: forward_star_difference, centered_difference, cross_weighted_difference
using Supposition
using ..TestUtils: WITH_SLOW_TESTS, _nonuniform_points
using ..UtilsBackendsTests: MockGPUVector, MockGPUMatrix
using ..SpaceVectorElementsTests: setup_test_grid

# Backward difference operators
backward_ops(::Val{1}) = (diff₋ₓ, D₋ₓ)
backward_ops(::Val{2}) = (D₋ᵧ, diff₋ₓ, diff₋ᵧ, backward_ops(Val(1))...)
backward_ops(::Val{3}) = (D₋₂, diff₋ₓ, diff₋₂, backward_ops(Val(2))...)

# Forward difference operators
forward_ops(::Val{1}) = (diff₊ₓ, D₊ₓ)
forward_ops(::Val{2}) = (D₊ᵧ, diff₊ₓ, diff₊ᵧ, forward_ops(Val(1))...)
forward_ops(::Val{3}) = (D₊₂, diff₊ₓ, diff₊₂, forward_ops(Val(2))...)

# Compares operator application to explicit matrix-vector multiplication
function test_operator_matrix_equivalence(op_generator)
    for D in 1:3
        @testset "$(D)D" begin
            _, W, U = setup_test_grid(Val(D))
            Rₕ!(U, x -> exp(-sum(x)))

            u₁ₕ = similar(U.data)
            u₂ₕ = similar(u₁ₕ)

            for op in unique(op_generator(Val(D)))
                u₁ₕ .= op(U).data
                u₂ₕ .= op(W) * U.data
                @test u₁ₕ ≈ u₂ₕ
            end
        end
    end
end

@testset "Finite differences" begin
    import LinearAlgebra: Diagonal, UniformScaling
    import LinearAlgebra: I as identity_matrix

    # --- Common Setup for All Tests ---
    mesh1D = mesh(domain(box(0, 1)), 5, false)
    mesh2D = mesh(domain(box((0, 1), (2, 3))), (5, 4), (true, true))
    mesh3D = mesh(domain(box((0, 1, 2), (4, 5, 6))), (4, 5, 4), (true, true, true))
    T = Float64

    @testset "Helper operators" begin
        A = [1 2; 3 4]
        B = [5 6; 7 8]
        @test (A ⊗ B) == kron(A, B)

        be = backend(T)
        @test _Eye(be, 5, Val(0)) * ones(5) == ones(5)

        S_super = _Eye(be, 5, Val(1))
        S_sub = _Eye(be, 5, Val(-2))

        @test S_super == spdiagm(1 => ones(4))
        @test S_sub == spdiagm(-2 => ones(3))
        @test S_super * [1, 2, 3, 4, 5] == [2, 3, 4, 5, 0]
        @test S_sub * [1, 2, 3, 4, 5] == [0, 0, 1, 2, 3]

        # `_shift_ones` dispatches on the backend's own matrix_type: SparseMatrixCSC above,
        # a dense Matrix here, and a generic AbstractMatrix (any vendor array type, e.g. a
        # GPU array) via the scalar-indexing fallback -- MockGPUMatrix (test/utils/backends.jl,
        # already in Main by this point) stands in for that without needing real GPU hardware.
        # MockGPUVector/MockGPUMatrix answer DeviceLocality(), so this Backend now needs a
        # device policy to construct at all.
        be_dense = backend(vector_type = Vector{T}, matrix_type = Matrix{T})
        S_dense = _Eye(be_dense, 5, Val(1))
        @test S_dense isa Matrix{T}
        @test S_dense == Matrix(spdiagm(1 => ones(4)))

        be_generic = backend(vector_type = MockGPUVector{T}, matrix_type = MockGPUMatrix{T}, policy = GpuKernel())
        S_generic = _Eye(be_generic, 5, Val(-2))
        @test S_generic isa MockGPUMatrix{T}
        @test S_generic.data == Matrix(spdiagm(-2 => ones(3)))
    end

    @testset "Shift operators" begin
        for val in [-1, 1]
            name = val == 1 ? "Forward" : "Backward"
            @testset "$name Shifts" begin
                # 1D
                n = npoints(mesh1D)
                @test shift(mesh1D, Val(1), Val(val)) == spdiagm(val => ones(n - abs(val)))

                # 2D
                nx, ny = npoints(mesh2D, Tuple)
                Sₓ_expected = identity_matrix(ny) ⊗ spdiagm(val => ones(nx - abs(val)))
                Sᵧ_expected = spdiagm(val => ones(ny - abs(val))) ⊗ identity_matrix(nx)
                @test shift(mesh2D, Val(1), Val(val)) == Sₓ_expected
                @test shift(mesh2D, Val(2), Val(val)) == Sᵧ_expected

                # 3D
                nx, ny, nz = npoints(mesh3D, Tuple)
                Sₓ_3D_expected = identity_matrix(ny*nz) ⊗ spdiagm(val => ones(nx - abs(val)))
                Sᵧ_3D_expected = identity_matrix(nz) ⊗ spdiagm(val => ones(ny - abs(val))) ⊗
                                 identity_matrix(nx)
                S₂_3D_expected = spdiagm(val => ones(nz - abs(val))) ⊗ identity_matrix(nx*ny)
                @test shift(mesh3D, Val(1), Val(val)) == Sₓ_3D_expected
                @test shift(mesh3D, Val(2), Val(val)) == Sᵧ_3D_expected
                @test shift(mesh3D, Val(3), Val(val)) == S₂_3D_expected
            end
        end
    end

    @testset "Backward difference" begin
        @testset "In-place calculation" begin
            # 1D
            u_1d = T[1, 2, 4, 8, 16]
            out_1d = similar(u_1d)
            backward_difference_dim!(out_1d, u_1d, (5,), Val(1))
            @test out_1d == [1, 1, 2, 4, 8]

            # 2D
            u_2d = T[
                1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 21, 22, 23, 24, 25, 31, 32, 33, 34, 35
            ]
            out_2d = similar(u_2d)
            backward_difference_dim!(out_2d, u_2d, (5, 4), Val(1))
            @test out_2d == T[1, 1, 1, 1, 1, 11, 1, 1, 1, 1, 21, 1, 1, 1, 1, 31, 1, 1, 1, 1]
            backward_difference_dim!(out_2d, u_2d, (5, 4), Val(2))
            @test out_2d == T[
                1, 2, 3, 4, 5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10
            ]

            # 3D
            u_3d = collect(Iterators.flatten(T[i+j+k for i in 1:4, j in 1:5, k in 1:4]))
            out_3d = similar(u_3d)
            backward_difference_dim!(out_3d, u_3d, (4, 5, 4), Val(1))
            @test out_3d == T[
                3,
                1,
                1,
                1,
                4,
                1,
                1,
                1,
                5,
                1,
                1,
                1,
                6,
                1,
                1,
                1,
                7,
                1,
                1,
                1,
                4,
                1,
                1,
                1,
                5,
                1,
                1,
                1,
                6,
                1,
                1,
                1,
                7,
                1,
                1,
                1,
                8,
                1,
                1,
                1,
                5,
                1,
                1,
                1,
                6,
                1,
                1,
                1,
                7,
                1,
                1,
                1,
                8,
                1,
                1,
                1,
                9,
                1,
                1,
                1,
                6,
                1,
                1,
                1,
                7,
                1,
                1,
                1,
                8,
                1,
                1,
                1,
                9,
                1,
                1,
                1,
                10,
                1,
                1,
                1
            ]
            backward_difference_dim!(out_3d, u_3d, (4, 5, 4), Val(2))
            @test out_3d == T[
                3,
                4,
                5,
                6,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                4,
                5,
                6,
                7,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                5,
                6,
                7,
                8,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                6,
                7,
                8,
                9,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1
            ]
            backward_difference_dim!(out_3d, u_3d, (4, 5, 4), Val(3))
            @test out_3d == T[
                3,
                4,
                5,
                6,
                4,
                5,
                6,
                7,
                5,
                6,
                7,
                8,
                6,
                7,
                8,
                9,
                7,
                8,
                9,
                10,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1
            ]
        end

        @testset "In-place difference" begin
            u = T[2, 3, 5, 9, 8]
            h = Base.Fix1(spacing, mesh1D)
            out = similar(u)
            backward_difference_dim!(out, u, h, (5,), Val(1))
            expected = [
                0, (u[2]-u[1])/h(2), (u[3]-u[2])/h(3), (u[4]-u[3])/h(4), (u[5]-u[4])/h(5)
            ]
            @test out ≈ expected
        end
    end

    @testset "Forward difference" begin
        @testset "In-place calculation" begin
            # 1D
            u_1d = T[1, 2, 4, 8, 16]
            out_1d = similar(u_1d)
            forward_difference_dim!(out_1d, u_1d, (5,), Val(1))
            @test out_1d == [1, 2, 4, 8, -16]

            # 2D
            u_2d = T[
                1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 21, 22, 23, 24, 25, 31, 32, 33, 34, 35
            ]
            out_2d = similar(u_2d)
            forward_difference_dim!(out_2d, u_2d, (5, 4), Val(1))
            @test out_2d ==
                  T[1, 1, 1, 1, -5, 1, 1, 1, 1, -15, 1, 1, 1, 1, -25, 1, 1, 1, 1, -35]
            forward_difference_dim!(out_2d, u_2d, (5, 4), Val(2))
            @test out_2d == T[
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                -31,
                -32,
                -33,
                -34,
                -35
            ]

            # 3D
            u_3d = collect(Iterators.flatten(T[i+j+k for i in 1:4, j in 1:5, k in 1:4]))
            out_3d = similar(u_3d)
            forward_difference_dim!(out_3d, u_3d, (4, 5, 4), Val(1))
            @test out_3d == T[
                1,
                1,
                1,
                -6,
                1,
                1,
                1,
                -7,
                1,
                1,
                1,
                -8,
                1,
                1,
                1,
                -9,
                1,
                1,
                1,
                -10,
                1,
                1,
                1,
                -7,
                1,
                1,
                1,
                -8,
                1,
                1,
                1,
                -9,
                1,
                1,
                1,
                -10,
                1,
                1,
                1,
                -11,
                1,
                1,
                1,
                -8,
                1,
                1,
                1,
                -9,
                1,
                1,
                1,
                -10,
                1,
                1,
                1,
                -11,
                1,
                1,
                1,
                -12,
                1,
                1,
                1,
                -9,
                1,
                1,
                1,
                -10,
                1,
                1,
                1,
                -11,
                1,
                1,
                1,
                -12,
                1,
                1,
                1,
                -13
            ]
            forward_difference_dim!(out_3d, u_3d, (4, 5, 4), Val(2))
            @test out_3d == T[
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                -7,
                -8,
                -9,
                -10,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                -8,
                -9,
                -10,
                -11,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                -9,
                -10,
                -11,
                -12,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                -10,
                -11,
                -12,
                -13
            ]
            forward_difference_dim!(out_3d, u_3d, (4, 5, 4), Val(3))
            @test out_3d == T[
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                -6,
                -7,
                -8,
                -9,
                -7,
                -8,
                -9,
                -10,
                -8,
                -9,
                -10,
                -11,
                -9,
                -10,
                -11,
                -12,
                -10,
                -11,
                -12,
                -13
            ]
        end

        @testset "In-place difference" begin
            u = T[2, 3, 5, 9, 8]
            h = Base.Fix1(spacing, mesh1D)
            out = similar(u)
            N = length(u)
            forward_difference_dim!(out, u, h, (N,), Val(1))
            expected = [
                (u[2]-u[1])/h(1), (u[3]-u[2])/h(2), (u[4]-u[3])/h(3), (u[5]-u[4])/h(4), 0
            ]
            @test out ≈ expected
        end
    end

    @testset "Operator vs matrix" begin
        @testset "Backward" test_operator_matrix_equivalence(backward_ops)
        @testset "Forward" test_operator_matrix_equivalence(forward_ops)

        WITH_SLOW_TESTS && @testset "Random grids (Supposition)" begin
            positive_h = Data.Floats{Float64}(;
                minimum = 0.01, maximum = 10.0, nans = false, infs = false
            )
            field_val = Data.Floats{Float64}(;
                minimum = -100.0, maximum = 100.0, nans = false, infs = false
            )

            # 1D equivalence: matrix-free stencil loop == sparse matrix multiplication
            @check function check_operator_matrix_1d(
                    h = Data.Vectors(positive_h; min_size = 3, max_size = 25),
                    u_raw = Data.Vectors(field_val; min_size = 26, max_size = 26)
            )
                n = length(h) + 1
                pts = _nonuniform_points(h)

                Ωₕ = mesh(domain(interval(0.0, 1.0)), n, false)
                set_points!(Ωₕ, pts)
                Wₕ = gridspace(Ωₕ)

                u_vals = copy(u_raw[1:n])
                uₕ = element(Wₕ, u_vals)

                ops = (D₋ₓ, D₊ₓ, diff₋ₓ, diff₊ₓ)
                all_ok = true
                for op in ops
                    v1 = parent(op(uₕ))
                    v2 = op(Wₕ) * u_vals
                    scale = max(maximum(abs, v1), maximum(abs, v2), 1.0)
                    if !isapprox(v1, v2; atol = 1e-10 * scale, rtol = 1e-10)
                        all_ok = false
                        break
                    end
                end
                all_ok
            end

            # 2D equivalence across all coordinate difference operators
            @check function check_operator_matrix_2d(
                    hx = Data.Vectors(positive_h; min_size = 3, max_size = 7),
                    hy = Data.Vectors(positive_h; min_size = 3, max_size = 7),
                    u_raw = Data.Vectors(field_val; min_size = 64, max_size = 64)
            )
                nx = length(hx) + 1
                ny = length(hy) + 1
                pts_x = _nonuniform_points(hx)

                pts_y = _nonuniform_points(hy)

                Ωₕ = mesh(
                    domain(interval(0.0, 1.0) × interval(0.0, 1.0)),
                    (nx, ny),
                    (false, false)
                )
                set_points!(Ωₕ(1), pts_x)
                set_points!(Ωₕ(2), pts_y)
                Wₕ = gridspace(Ωₕ)

                total = nx * ny
                u_mat = reshape(copy(u_raw[1:total]), nx, ny)
                u_vec = vec(u_mat)
                uₕ = element(Wₕ, u_vec)

                ops = (D₋ₓ, D₊ₓ, diff₋ₓ, diff₊ₓ, D₋ᵧ, D₊ᵧ, diff₋ᵧ, diff₊ᵧ)
                all_ok = true
                for op in ops
                    v1 = parent(op(uₕ))
                    v2 = op(Wₕ) * u_vec
                    scale = max(maximum(abs, v1), maximum(abs, v2), 1.0)
                    if !isapprox(v1, v2; atol = 1e-10 * scale, rtol = 1e-10)
                        all_ok = false
                        break
                    end
                end
                all_ok
            end
        end
    end
end

@testset "Shift tensor product" begin
    # The `shift` docstring states the per-direction Kronecker forms that
    # `_recursive_shift` generalises. These assert them, so the docstring cannot drift
    # from the code the way the commented block it replaced could.
    # Eye/Ones (FillArrays) are gone; _Eye now builds through the mesh's own backend
    # (backend_eye/matrix_type), so the reference values below use the same backend.
    import Bramble: shift, _Eye, ⊗, backend, backend_eye

    Ωₕ1 = mesh(domain(interval(0.0, 1.0)), 5, true)
    Ωₕ2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (4, 3), (true, true))
    Ωₕ3 = mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (3, 4, 2), (true, true, true))
    be1, be2, be3 = backend(Ωₕ1), backend(Ωₕ2), backend(Ωₕ3)
    nₓ, n_y = npoints(Ωₕ2, Tuple)
    aₓ, a_y, a_z = npoints(Ωₕ3, Tuple)

    for i in (-2, -1, 1, 2)
        @testset "shift by $i" begin
            @test shift(Ωₕ1, Val(1), Val(i)) == _Eye(be1, 5, Val(i))
            @test shift(Ωₕ2, Val(1), Val(i)) ==
                  backend_eye(be2, n_y) ⊗ _Eye(be2, nₓ, Val(i))
            @test shift(Ωₕ2, Val(2), Val(i)) ==
                  _Eye(be2, n_y, Val(i)) ⊗ backend_eye(be2, nₓ)
            @test shift(Ωₕ3, Val(3), Val(i)) ==
                  _Eye(be3, a_z, Val(i)) ⊗ backend_eye(be3, aₓ * a_y)
        end
    end

    @testset "Zero shift identity" begin
        for (Ωₕ, be) in ((Ωₕ1, be1), (Ωₕ2, be2), (Ωₕ3, be3)), d in 1:dim(Ωₕ)

            @test shift(Ωₕ, Val(d), Val(0)) == backend_eye(be, npoints(Ωₕ))
        end
    end

    @testset "Truncated stencil" begin
        # n - |i| nonzeros per line of the 1D factor, so nothing wraps from the last
        # point back to the first.
        S = Matrix(shift(Ωₕ1, Val(1), Val(1)))
        @test count(!iszero, S) == 5 - 1
        @test all(S[i, i + 1] == 1.0 for i in 1:4)
        @test S[5, 1] == 0.0
    end
end

# The weighted matrices are a diagonal scaling of an unscaled one, and that scaling used to
# be written `w .* A`: a dense vector broadcast against a `SparseMatrixCSC`. The result was
# numerically right and reported the right `nnz`, `length(nzval)` and `sizeof(nzval)`, but
# it carried `rowval` and `nzval` buffers sized for the dense case, which nothing except
# `Base.summarysize` reveals. On the mesh below `D₋ₓ` held 19800 stored entries and
# 1.51 GiB. A bound per stored entry is what pins it: the storage has to follow `nnz`, not
# `nrows * ncols`.
const _BYTES_PER_STORED_ENTRY = 64

# Measuring the builder rather than what it returns, since `cross_weighted_difference` sums
# two weighted matrices and the sum rebuilds its own storage: the finished operator was
# never the oversized one, the two summands were. Behind a function barrier, because
# `@allocated` at `@testset` scope measures the enclosing closure instead.
_cross_weighted_bytes(Ωₕ) = @allocated cross_weighted_difference(Ωₕ, Val(1))

@testset "Weighted operators: storage ∝ nnz" begin
    Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (100, 100))
    n = npoints(Ωₕ)
    @test n == 10000

    operators = (
        "D₋ₓ" => D₋ₓ(Ωₕ),
        "D₊ₓ" => D₊ₓ(Ωₕ),
        "forward_star_difference" => forward_star_difference(Ωₕ, Val(1)),
        "centered_difference" => centered_difference(Ωₕ, Val(1))
    )

    for (name, A) in operators
        @testset "$name" begin
            @test nnz(A) <= 2 * n
            @test Base.summarysize(A) <= _BYTES_PER_STORED_ENTRY * nnz(A)
        end
    end

    @testset "cross_weighted_difference" begin
        cross_weighted_difference(Ωₕ, Val(1))
        @test _cross_weighted_bytes(Ωₕ) <= _BYTES_PER_STORED_ENTRY * 8 * n
    end
end

using LinearAlgebra: Diagonal
using Bramble: kronecker_operator_matrix, stencil_matrix, difference_shift, BackwardFiniteDiffOp,
               D₋, D̃ₓ, D̽ᵧ, D̃ᵧ, D̽ₓ, Dcₓ, Dcᵧ, weights, Innerh, trial_function, test_function,
               εₕ, εcₕ, ε̽ₕ, divcₕ, div̽ₕ, ∇₊ₕ, Mₓ, Mᵧ, _macro_string, _tuple_args,
               _relocate!, _subst

# The dense fallback of `stencil_matrix` (a host fill through `_HostAxisSpacings`, handed to
# the backend's matrix type in one `copyto!`) and the dense branch of `_scale_rows!` (which
# the Kronecker oracle weights through) are taken only by a dense matrix backend; the
# sparse default takes neither. Every difference family against the Kronecker construction,
# entry for entry, on a mesh non-uniform in both directions; the unscaled pair against the
# difference of two shift matrices, which is its definition.
@testset "stencil_matrix: dense vs Kronecker" begin
    Ωd = mesh(domain(box((0.0, 0.0), (1.0, 2.0))), (6, 5), (false, false);
        backend = backend(matrix_type = Matrix{Float64}))
    families = ((D₋ₓ, D₋ᵧ), (D₊ₓ, D₊ᵧ), (D̃ₓ, D̃ᵧ), (Dcₓ, Dcᵧ), (D̽ₓ, D̽ᵧ))
    for ops in families, d in 1:2

        @testset "$(ops[d])" begin
            A = ops[d](Ωd)
            @test A isa Matrix{Float64}
            @test A == kronecker_operator_matrix(Ωd, ops[d])
        end
    end
    for (op, d) in ((diff₋ₓ, 1), (diff₋ᵧ, 2))
        @test op(Ωd) isa Matrix{Float64}
        @test op(Ωd) == difference_shift(Ωd, Val(d), Val(0), Val(-1))
    end
    for (op, d) in ((diff₊ₓ, 1), (diff₊ᵧ, 2))
        @test op(Ωd) == difference_shift(Ωd, Val(d), Val(1), Val(0))
    end
    # a grid space is read as its mesh
    @test stencil_matrix(gridspace(Ωd), BackwardFiniteDiffOp{2}()) ==
          kronecker_operator_matrix(Ωd, D₋ᵧ)
end

@testset "In-place difference: size mismatch" begin
    err = try
        backward_difference_dim!(zeros(3), zeros(4), (4,), Val(1))
    catch e
        e
    end
    @test err isa DimensionMismatch
    @test err.msg == "out has 3 entries and in has 4, but the grid (4,) has 4"
end

# The device kernels' boundary index (`_stencil_boundary_dim`) must name the one slice the
# host traversal (`_stencil_ranges`) treats as the boundary: the last point for a forward
# stencil, the first for a backward one.
@testset "Boundary slice: device index vs host" begin
    for dir in (Bramble.Forward(), Bramble.Backward()), n in (2, 7)

        _, boundary = Bramble._stencil_ranges((1:n,), Val(1), dir)
        i = Bramble._stencil_boundary_dim(dir, n)
        @test boundary == (i:i,)
    end
    @test Bramble._stencil_boundary_dim(Bramble.Forward(), 7) == 7
    @test Bramble._stencil_boundary_dim(Bramble.Backward(), 7) == 1
end

# `@operator_family` expanded here, at test time, into a family of its own: in `src/` the
# macro only ever runs while Bramble precompiles. The family is the unscaled backward
# difference under another name (`_apply_spaced!` with no spacing), so what it computes can
# be checked by hand, and every keyword path is taken once: an `extra_args` tuple, prose
# given as a `*` concatenation, a note with `{direction}`/`{suffix}` placeholders, a
# vectorial keyword given and one left to its default.
module OperatorFamilyProbe
using Bramble: Bramble, VectorElement, ScalarGridSpace, CompositeGridSpace, OperatorArgument,
               Backward, _apply_spaced!, _no_spacing, _no_precheck, _dispatch_dim, _dim_index,
               _vectorial_apply, _op_mesh, dim

const PROBE_LINE = @__LINE__() + 1
Bramble.@operator_family(base=probe,
    stem=Pq,
    apply_fn=_apply_spaced!,
    extra_args=(_no_spacing, _no_precheck),
    direction=Backward(),
    dir_string="backward",
    what="probe difference",
    formula="u_i - u_{i-1}",
    trailing_note="Probed along `{direction}`, "*"subscript {suffix}.",
    vectorial_alias=Pqₕ,
    vectorial_what="probe "*"gradient")
end

@testset "@operator_family expansion" begin
    P = OperatorFamilyProbe
    nx, ny = 6, 5
    Wₕ = gridspace(mesh(domain(box((0.0, 0.0), (1.0, 2.0))), (nx, ny), (false, false)))
    uₕ = Rₕ(Wₕ, x -> sin(3x[1]) + x[1] * x[2]^2)

    # the unscaled backward difference, by hand: u_i - u_{i-1}, and u_1 on the first slice
    U = reshape(copy(parent(uₕ)), nx, ny)
    ref_x, ref_y = copy(U), copy(U)
    ref_x[2:end, :] .= U[2:end, :] .- U[1:(end - 1), :]
    ref_y[:, 2:end] .= U[:, 2:end] .- U[:, 1:(end - 1)]

    @test parent(P.Pqₓ(uₕ)) == vec(ref_x)
    @test parent(P.Pqᵧ(uₕ)) == vec(ref_y)
    vₕ = element(Wₕ)
    @test P.Pqᵧ!(vₕ, uₕ) === vₕ
    @test parent(vₕ) == vec(ref_y)
    @test parent(P.Pq(uₕ, Val(1))) == vec(ref_x)
    @test parent(P.Pq(uₕ, 2)) == vec(ref_y)
    @test parent(P.Pq(uₕ, :y)) == vec(ref_y)
    @test map(parent, P.Pqₕ(uₕ)) == (vec(ref_x), vec(ref_y))
    @test P.Pqₕ[2] === P.Pqᵧ
    @test P.Pqₕ[:z] === P.Pq₂
    @test collect(P.Pqₕ) == [P.Pqₓ, P.Pqᵧ, P.Pq₂]

    # every generated method is attributed to the macro call, not to the quote in stencil.jl
    for f in (P.Pqₓ, P.Pqᵧ!, P.Pqₕ)
        m = only(methods(f))
        @test m.line == P.PROBE_LINE
        @test endswith(String(m.file), "difference.jl")
    end

    # the prose templates, substituted per direction and folded from their `*` pieces
    # read off the module's docstring table: `Docs.doc` needs the REPL to render
    docs(name) = join(
        (join(string.(d.text))
        for d in values(Base.Docs.meta(P)[Base.Docs.Binding(P, name)].docs)), "\n")
    doc_y = docs(:Pqᵧ)
    @test occursin(
        "The `backward` probe difference along the `y` direction, ``u_i - u_{i-1}``.", doc_y)
    @test occursin("Probed along `y`, subscript ᵧ.", doc_y)
    @test occursin("The backward probe gradient of `arg` along every coordinate",
        docs(:Pqₕ))

    # the helpers' remaining arms, which no family in `src/` reaches
    @test _relocate!(:x, LineNumberNode(3, :f)) === :x
    ex = :(a + b)
    @test _relocate!(ex, nothing) === ex
    @test _subst("", "x", "ₓ") == ""
    # one pass: the text substituted for `{direction}` is not itself substituted again
    @test _subst("{direction}|{suffix}", "{suffix}", "s") == "{suffix}|s"
    @test _tuple_args(:f) == Any[:f]
    @test _tuple_args(:(f(x))) == Any[:(f(x))]
    @test _macro_string("a") == "a"
    @test _macro_string(:("a" * "b" * "c")) == "abc"
    @test_throws ErrorException _macro_string(:(uppercase("a")))
end

# The form-layer stencils of `D̃` and `D̽`, assembled under the discrete L² product, are the
# space-layer matrices scaled row by row by the quadrature weights: innerₕ(Op(u), v) = vᵀ H Op u.
@testset "Form D̃ and D̽ stencils vs space matrices" begin
    Ωₕ = mesh(domain(box((0.0, 0.0), (1.0, 2.0))), (7, 6), (false, false))
    Wₕ = gridspace(Ωₕ)
    H = Diagonal(collect(weights(Wₕ, Innerh())))
    for (op_form, op_matrix) in ((u -> D̃ₓ(u), D̃ₓ(Ωₕ)), (u -> D̽ᵧ(u), D̽ᵧ(Ωₕ)))
        A = Matrix(assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(op_form(u), v))))
        @test maximum(abs, A) > 0.1
        @test isapprox(A, Matrix(H * op_matrix); atol = 1.0e-12)
    end
end

# The builders over composite trial and test functions, against the trees written out from
# their definitions: εₕ places ε_ii as a bare backward difference and averages each cross
# difference of ε_ij once onto the shared edge; εcₕ/ε̽ₕ collocate everything, and their inner
# product keeps the upper triangle with the off-diagonal pairs doubled.
@testset "Form builders, composite trial and test" begin
    Ωₕ = mesh(domain(box((0.0, 0.0), (1.0, 2.0))), (6, 5), (false, false))
    Wₕ = gridspace(Ωₕ)
    Vₕ = Wₕ^Val(2)
    u, v = trial_function(Vₕ), test_function(Vₕ)

    @testset "∇₊ₕ" begin
        @test ∇₊ₕ(u) === ((D₊ₓ(u(1)), D₊ᵧ(u(1))), (D₊ₓ(u(2)), D₊ᵧ(u(2))))
        @test ∇₊ₕ(v) === ((D₊ₓ(v(1)), D₊ᵧ(v(1))), (D₊ₓ(v(2)), D₊ᵧ(v(2))))
        us = trial_function(Wₕ)
        @test ∇₊ₕ(us) === (D₊ₓ(us), D₊ᵧ(us))
        u1 = trial_function(gridspace(mesh(domain(interval(0.0, 1.0)), 7, false)))
        @test ∇₊ₕ(u1) === D₊ₓ(u1)
    end

    @testset "εₕ and inner₊" begin
        ε = εₕ(u)
        @test ε.entries[1][1] === (D₋ₓ(u(1)),)
        @test ε.entries[1][2] === (0.5 * Mₓ(D₋ᵧ(u(1))), 0.5 * Mᵧ(D₋ₓ(u(2))))
        @test ε.entries[2][1] === (0.5 * Mᵧ(D₋ₓ(u(2))), 0.5 * Mₓ(D₋ᵧ(u(1))))
        @test ε.entries[2][2] === (D₋ᵧ(u(2)),)

        a(w) = 0.5 * Mₓ(D₋ᵧ(w(1)))
        b(w) = 0.5 * Mᵧ(D₋ₓ(w(2)))
        S = Val((1, 2))
        hand = inner₊(D₋ₓ(u(1)), D₋ₓ(v(1)), Val((1,))) +
               inner₊(a(u), a(v), S) + inner₊(a(u), b(v), S) +
               inner₊(b(u), a(v), S) + inner₊(b(u), b(v), S) +
               inner₊(b(u), b(v), S) + inner₊(b(u), a(v), S) +
               inner₊(a(u), b(v), S) + inner₊(a(u), a(v), S) +
               inner₊(D₋ᵧ(u(2)), D₋ᵧ(v(2)), Val((2,)))
        @test inner₊(εₕ(u), εₕ(v)) === hand
    end

    @testset "divcₕ, div̽ₕ" begin
        @test divcₕ(u) === Dcₓ(u(1)) + Dcᵧ(u(2))
        @test div̽ₕ(v) === D̽ₓ(v(1)) + D̽ᵧ(v(2))
    end

    @testset "εcₕ, ε̽ₕ and innerₕ" begin
        εc = εcₕ(u)
        @test εc.entries[1][1] === (Dcₓ(u(1)),)
        @test εc.entries[1][2] === ((1 // 2) * Dcᵧ(u(1)), (1 // 2) * Dcₓ(u(2)))
        @test εc.entries[2][2] === (Dcᵧ(u(2)),)
        ε̽ = ε̽ₕ(u)
        @test ε̽.entries[2][1] === ((1 // 2) * D̽ₓ(u(2)), (1 // 2) * D̽ᵧ(u(1)))

        for (ε, name) in ((εcₕ, "Dc"), (ε̽ₕ, "D̽"))
            x, y = name * "ₓ", name * "ᵧ"
            @test Bramble.expression(innerₕ(ε(u), ε(v))) ==
                  "((((1 * innerₕ($x(u(1)), $x(v(1))) + " *
                  "2 * innerₕ(1//2 * $y(u(1)), 1//2 * $y(v(1)))) + " *
                  "2 * innerₕ(1//2 * $y(u(1)), 1//2 * $x(v(2)))) + " *
                  "2 * innerₕ(1//2 * $x(u(2)), 1//2 * $y(v(1)))) + " *
                  "2 * innerₕ(1//2 * $x(u(2)), 1//2 * $x(v(2)))) + " *
                  "1 * innerₕ($y(u(2)), $y(v(2)))"
        end
    end
end

end # module SpaceDifferenceTests
