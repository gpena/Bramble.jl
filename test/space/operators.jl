using Test
using Bramble
using Bramble: IdentityOperator, ZeroOperator, OperatorScale, GridFunctionScale,
               OperatorAdd, is_symbolic, space

@testset "Linear operators" begin
    for D in 1:3
        @testset "$(D)D" begin
            I = interval(0.0, 1.0)
            X = domain(reduce(×, ntuple(_ -> I, Val(D))))
            M = mesh(X, ntuple(_ -> 4, Val(D)), ntuple(_ -> false, Val(D)))
            W = gridspace(M)
            u = element(W)
            Rₕ!(u, x -> 1.0)

            x0 = IdentityOperator(W)
            @test space(x0) === W
            @test !is_symbolic(x0)

            z0 = ZeroOperator(W)
            @test space(z0) === W
            @test !is_symbolic(z0)

            # Scalar scaling
            x1 = 2 * x0
            @test x1 isa OperatorScale
            @test x1.scalar == 2
            @test x1.inner_op === x0
            @test !is_symbolic(x1)

            x1_div = x0 / 2
            @test x1_div isa OperatorScale
            @test x1_div.scalar == 0.5

            # Grid function scaling
            x2 = u * x0
            @test x2 isa GridFunctionScale
            @test x2.grid_function === u
            @test x2.inner_op === x0
            @test !is_symbolic(x2)

            # Operator addition and subtraction
            sum_op = x0 + x0
            @test sum_op isa OperatorAdd
            @test sum_op.left_op === x0
            @test sum_op.right_op === x0
            @test !is_symbolic(sum_op)

            diff_op = x0 - x0
            @test diff_op isa OperatorAdd
            @test diff_op.left_op === x0

            # String representation
            buf = IOBuffer()
            show(buf, x0)
            @test String(take!(buf)) == "I"
            show(buf, z0)
            @test String(take!(buf)) == "0"
        end
    end
end
