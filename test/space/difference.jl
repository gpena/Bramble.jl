import Bramble: space, eltype, ⊗, _Eye, shift, npoints, spacing
using Bramble: backward_difference_dim!, forward_difference_dim!
import SparseArrays: issparse, sprand, spdiagm, spzeros
using Supposition

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
        be_dense = backend(vector_type = Vector{T}, matrix_type = Matrix{T})
        S_dense = _Eye(be_dense, 5, Val(1))
        @test S_dense isa Matrix{T}
        @test S_dense == Matrix(spdiagm(1 => ones(4)))

        be_generic = backend(
            vector_type = MockGPUVector{T}, matrix_type = MockGPUMatrix{T})
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
                Sₓ_3D_expected = identity_matrix(ny*nz) ⊗
                                 spdiagm(val => ones(nx - abs(val)))
                Sᵧ_3D_expected = identity_matrix(nz) ⊗ spdiagm(val => ones(ny - abs(val))) ⊗
                                 identity_matrix(nx)
                S₂_3D_expected = spdiagm(val => ones(nz - abs(val))) ⊗
                                 identity_matrix(nx*ny)
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
                1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 21, 22, 23, 24, 25, 31, 32, 33, 34, 35]
            out_2d = similar(u_2d)
            backward_difference_dim!(out_2d, u_2d, (5, 4), Val(1))
            @test out_2d == T[1, 1, 1, 1, 1, 11, 1, 1, 1, 1, 21, 1, 1, 1, 1, 31, 1, 1, 1, 1]
            backward_difference_dim!(out_2d, u_2d, (5, 4), Val(2))
            @test out_2d == T[
                1, 2, 3, 4, 5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

            # 3D
            u_3d = collect(Iterators.flatten(T[i+j+k for i in 1:4, j in 1:5, k in 1:4]))
            out_3d = similar(u_3d)
            backward_difference_dim!(out_3d, u_3d, (4, 5, 4), Val(1))
            @test out_3d ==
                  T[3, 1, 1, 1, 4, 1, 1, 1, 5, 1, 1, 1, 6, 1, 1, 1, 7, 1, 1, 1,
                4, 1, 1, 1, 5, 1, 1, 1, 6, 1, 1, 1, 7, 1, 1, 1, 8, 1, 1,
                1, 5, 1, 1, 1, 6, 1, 1, 1, 7, 1, 1, 1, 8, 1, 1, 1, 9, 1, 1,
                1, 6, 1, 1, 1, 7, 1, 1, 1, 8, 1, 1, 1, 9, 1, 1, 1, 10, 1,
                1, 1]
            backward_difference_dim!(out_3d, u_3d, (4, 5, 4), Val(2))
            @test out_3d ==
                  T[3, 4, 5, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                4, 5, 6, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 5, 6, 7, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 6, 7, 8, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1]
            backward_difference_dim!(out_3d, u_3d, (4, 5, 4), Val(3))
            @test out_3d ==
                  T[3, 4, 5, 6, 4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 10,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1]
        end

        @testset "In-place difference" begin
            u = T[2, 3, 5, 9, 8]
            h = Base.Fix1(spacing, mesh1D)
            out = similar(u)
            backward_difference_dim!(out, u, h, (5,), Val(1))
            expected = [
                0, (u[2]-u[1])/h(2), (u[3]-u[2])/h(3), (u[4]-u[3])/h(4), (u[5]-u[4])/h(5)]
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
                1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 21, 22, 23, 24, 25, 31, 32, 33, 34, 35]
            out_2d = similar(u_2d)
            forward_difference_dim!(out_2d, u_2d, (5, 4), Val(1))
            @test out_2d ==
                  T[1, 1, 1, 1, -5, 1, 1, 1, 1, -15, 1, 1, 1, 1, -25, 1, 1, 1, 1, -35]
            forward_difference_dim!(out_2d, u_2d, (5, 4), Val(2))
            @test out_2d == T[10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                10, 10, 10, 10, -31, -32, -33, -34, -35]

            # 3D
            u_3d = collect(Iterators.flatten(T[i+j+k for i in 1:4, j in 1:5, k in 1:4]))
            out_3d = similar(u_3d)
            forward_difference_dim!(out_3d, u_3d, (4, 5, 4), Val(1))
            @test out_3d ==
                  T[1, 1, 1, -6, 1, 1, 1, -7, 1, 1, 1, -8, 1, 1, 1, -9, 1, 1, 1,
                -10, 1, 1, 1, -7, 1, 1, 1, -8, 1, 1, 1, -9, 1, 1, 1, -10,
                1, 1, 1, -11, 1, 1, 1, -8, 1, 1, 1, -9, 1, 1, 1, -10, 1, 1,
                1, -11, 1, 1, 1, -12, 1, 1, 1, -9, 1, 1, 1, -10, 1, 1, 1,
                -11, 1, 1, 1, -12, 1, 1, 1, -13]
            forward_difference_dim!(out_3d, u_3d, (4, 5, 4), Val(2))
            @test out_3d ==
                  T[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -7, -8, -9,
                -10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -8, -9,
                -10, -11, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                -9, -10, -11, -12, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, -10, -11, -12, -13]
            forward_difference_dim!(out_3d, u_3d, (4, 5, 4), Val(3))
            @test out_3d ==
                  T[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, -6, -7, -8, -9, -7, -8, -9, -10, -8, -9, -10, -11, -9,
                -10, -11, -12, -10, -11, -12, -13]
        end

        @testset "In-place difference" begin
            u = T[2, 3, 5, 9, 8]
            h = Base.Fix1(spacing, mesh1D)
            out = similar(u)
            N = length(u)
            forward_difference_dim!(out, u, h, (N,), Val(1))
            expected = [
                (u[2]-u[1])/h(1), (u[3]-u[2])/h(2), (u[4]-u[3])/h(3), (u[5]-u[4])/h(4), 0]
            @test out ≈ expected
        end
    end

    @testset "Operator vs matrix" begin
        @testset "Backward" test_operator_matrix_equivalence(backward_ops)
        @testset "Forward" test_operator_matrix_equivalence(forward_ops)

        @testset "Random grids (Supposition)" begin
            positive_h = Data.Floats{Float64}(; minimum = 0.01, maximum = 10.0,
                nans = false, infs = false)
            field_val = Data.Floats{Float64}(; minimum = -100.0, maximum = 100.0,
                nans = false, infs = false)

            # 1D equivalence: matrix-free stencil loop == sparse matrix multiplication
            @check function check_operator_matrix_1d(
                    h = Data.Vectors(positive_h; min_size = 3, max_size = 25),
                    u_raw = Data.Vectors(field_val; min_size = 26, max_size = 26)
            )
                n = length(h) + 1
                pts = zeros(Float64, n)
                for i in 1:length(h)
                    pts[i + 1] = pts[i] + h[i]
                end
                pts ./= pts[end]

                Ωₕ = mesh(domain(interval(0.0, 1.0)), n, false)
                set_points!(Ωₕ, pts)
                Wₕ = gridspace(Ωₕ)

                u_vals = copy(u_raw[1:n])
                uₕ = element(Wₕ, u_vals)

                ops = (D₋ₓ, D₊ₓ, diff₋ₓ, diff₊ₓ)
                all_ok = true
                for op in ops
                    v1 = values(op(uₕ))
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
                pts_x = zeros(Float64, nx)
                for i in 1:length(hx)
                    pts_x[i + 1] = pts_x[i] + hx[i]
                end
                pts_x ./= pts_x[end]

                pts_y = zeros(Float64, ny)
                for j in 1:length(hy)
                    pts_y[j + 1] = pts_y[j] + hy[j]
                end
                pts_y ./= pts_y[end]

                Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (nx, ny),
                    (false, false))
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
                    v1 = values(op(uₕ))
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
    Ωₕ3 = mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (3, 4, 2),
        (true, true, true))
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
