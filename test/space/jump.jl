using Test
import Bramble: forward_difference, jump, jump_dim!
using LinearAlgebra: norm

# There is one jump, not a forward and a backward pair: the jump belongs to the interface
# between two cells rather than to a direction of travel across it. It is arithmetically
# the unscaled forward difference, u_{i+1} - u_i, and forwards to it.

jump_ops(::Val{1}) = (jumpₓ,)
jump_ops(::Val{2}) = (jumpₓ, jumpᵧ, jump_ops(Val(1))...)
jump_ops(::Val{3}) = (jumpₓ, jump₂, jump_ops(Val(2))...)

@testset "Jump Operators" begin
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

    @testset "There is no directional variant" begin
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

    @testset "Operator vs. Matrix Application" begin
        test_operator_matrix_equivalence(jump_ops)
    end
end
