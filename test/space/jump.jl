using Test
import Bramble: forward_difference, jump, jump_dim!
using LinearAlgebra: norm
using Supposition

# There is one jump, not a forward and a backward pair: the jump belongs to the interface
# between two cells rather than to a direction of travel across it. It is arithmetically
# the unscaled forward difference, u_{i+1} - u_i, and forwards to it.

jump_ops(::Val{1}) = (jumpₓ,)
jump_ops(::Val{2}) = (jumpₓ, jumpᵧ, jump_ops(Val(1))...)
jump_ops(::Val{3}) = (jumpₓ, jump₂, jump_ops(Val(2))...)

@testset "Jump operators" begin
    for D in 1:3
        @testset "$D-Dimensional Tests" begin
            dims, Wₕ, uₕ = setup_test_grid(Val(D))
            Ωₕ = mesh(Wₕ)

            vₕ = similar(uₕ)

            # Define a linear test function and project it onto the grid
            coeffs = (2.0, 3.0, 5.0)
            linear_func(x) = sum(coeffs[i] * x[i] for i in 1:D)
            Rₕ!(uₕ, linear_func)

            for i in 1:D
                # the primary applicator, out of place, against the in-place one
                res_oop = jump(uₕ, Val(i))
                jump_dim!(vₕ.data, uₕ.data, dims, Val(i))
                @test norm(values(res_oop) - values(vₕ)) < 1e-14

                # and against the unscaled forward difference it forwards to
                @test norm(res_oop .- forward_difference(uₕ, Val(i))) < 1e-14
            end

            # the definition itself, u_{i+1} - u_i, read off the values
            u = values(uₕ)
            r = reshape(values(jumpₓ(uₕ)), dims)
            ur = reshape(u, dims)
            n1 = dims[1]
            for I in CartesianIndices(r)
                if I[1] < n1
                    Inext = I + CartesianIndex(ntuple(k -> k == 1 ? 1 : 0, D))
                    @test r[I] ≈ ur[Inext] - ur[I]
                else
                    # no forward neighbour: treated as though it were zero, as in diff₊ₓ
                    @test r[I] ≈ -ur[I]
                end
            end

            # the directional aliases
            @test norm(jumpₓ(uₕ) - jump(uₕ, Val(1))) < 1e-14
            D >= 2 && @test norm(jumpᵧ(uₕ) - jump(uₕ, Val(2))) < 1e-14
            D >= 3 && @test norm(jump₂(uₕ) - jump(uₕ, Val(3))) < 1e-14

            # the vectorial alias: a bare element in 1D, a tuple above it
            jumps = jumpₕ(uₕ)
            if D == 1
                @test jumps isa VectorElement
                @test norm(jumps - jump(uₕ, Val(1))) < 1e-14
            else
                @test jumps isa NTuple{D, VectorElement}
                for i in 1:D
                    @test norm(jumps[i] - jump(uₕ, Val(i))) < 1e-14
                end
            end
        end
    end

    @testset "No directional variant" begin
        # The ₋/₊ pair was removed: a backward jump names the same interface as the
        # forward one seen from the other side, so it was the same numbers shifted by an
        # index rather than a second operator.
        for name in (:jump₋ₓ, :jump₋ᵧ, :jump₋₂, :jump₋ₕ,
            :jump₊ₓ, :jump₊ᵧ, :jump₊₂, :jump₊ₕ)
            @test !isdefined(Bramble, name)
        end
        for name in (:jumpₓ, :jumpᵧ, :jump₂, :jumpₕ)
            @test isdefined(Bramble, name)
            @test Base.isexported(Bramble, name)
        end
    end

    @testset "Operator vs matrix" begin
        test_operator_matrix_equivalence(jump_ops)
    end

    @testset "Leibniz product rule" begin
        positive_h = Data.Floats{Float64}(; minimum = 0.01, maximum = 10.0,
            nans = false, infs = false)
        field_val = Data.Floats{Float64}(; minimum = -100.0, maximum = 100.0,
            nans = false, infs = false)

        # 1D: jump(u .* v) == M₊(u) .* jump(v) .+ jump(u) .* M₊(v) across interior interfaces
        @check function check_leibniz_1d(
                h = Data.Vectors(positive_h; min_size = 2, max_size = 25),
                u_raw = Data.Vectors(field_val; min_size = 26, max_size = 26),
                v_raw = Data.Vectors(field_val; min_size = 26, max_size = 26)
        )
            n = length(h) + 1
            Ωₕ = mesh(domain(interval(0.0, 1.0)), n, false)
            Wₕ = gridspace(Ωₕ)

            u_vec = copy(u_raw[1:n])
            v_vec = copy(v_raw[1:n])
            uv_vec = u_vec .* v_vec

            uₕ = element(Wₕ, u_vec)
            vₕ = element(Wₕ, v_vec)
            uvₕ = element(Wₕ, uv_vec)

            j_uv = values(jumpₓ(uvₕ))[1:(n - 1)]
            leibniz = (values(M₊ₓ(uₕ)) .* values(jumpₓ(vₕ)) .+ values(jumpₓ(uₕ)) .* values(M₊ₓ(vₕ)))[1:(n - 1)]

            scale = max(maximum(abs, j_uv), maximum(abs, leibniz), 1.0)
            isapprox(j_uv, leibniz; atol = 1e-10 * scale, rtol = 1e-10)
        end

        # 2D: holds across every coordinate interface
        @check function check_leibniz_2d(
                hx = Data.Vectors(positive_h; min_size = 2, max_size = 8),
                hy = Data.Vectors(positive_h; min_size = 2, max_size = 8),
                u_raw = Data.Vectors(field_val; min_size = 81, max_size = 81),
                v_raw = Data.Vectors(field_val; min_size = 81, max_size = 81)
        )
            nx = length(hx) + 1
            ny = length(hy) + 1
            total = nx * ny

            Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (nx, ny),
                (false, false))
            Wₕ = gridspace(Ωₕ)

            u_vec = copy(u_raw[1:total])
            v_vec = copy(v_raw[1:total])
            uv_vec = u_vec .* v_vec

            uₕ = element(Wₕ, u_vec)
            vₕ = element(Wₕ, v_vec)
            uvₕ = element(Wₕ, uv_vec)

            j_x = reshape(values(jumpₓ(uvₕ)), nx, ny)
            leibniz_x = reshape(
                values(M₊ₓ(uₕ)) .* values(jumpₓ(vₕ)) .+
                values(jumpₓ(uₕ)) .* values(M₊ₓ(vₕ)),
                nx,
                ny)
            scale_x = max(maximum(abs, j_x[1:(nx - 1), :]), 1.0)
            ok_x = isapprox(j_x[1:(nx - 1), :], leibniz_x[1:(nx - 1), :];
                atol = 1e-10 * scale_x, rtol = 1e-10)

            j_y = reshape(values(jumpᵧ(uvₕ)), nx, ny)
            leibniz_y = reshape(
                values(M₊ᵧ(uₕ)) .* values(jumpᵧ(vₕ)) .+
                values(jumpᵧ(uₕ)) .* values(M₊ᵧ(vₕ)),
                nx,
                ny)
            scale_y = max(maximum(abs, j_y[:, 1:(ny - 1)]), 1.0)
            ok_y = isapprox(j_y[:, 1:(ny - 1)], leibniz_y[:, 1:(ny - 1)];
                atol = 1e-10 * scale_y, rtol = 1e-10)

            ok_x && ok_y
        end
    end
end
