module SpaceAverageTests

using Test
using Bramble
using Bramble: Mᵧ, M₂, Mₓ, VectorElement
import Bramble: forward_average, backward_average, M₊ₓ, M₊ᵧ, M₊₂, M₊ₕ
import Bramble: forward_average_dim!, backward_average_dim!
using LinearAlgebra: norm
using SparseArrays: nnz
using ..SpaceVectorElementsTests: setup_test_grid
using ..SpaceDifferenceTests: test_operator_matrix_equivalence

@testset "Averaging operators" begin
    # Backward average operators
    backward_average_ops(::Val{1}) = (Mₓ,)
    backward_average_ops(::Val{2}) = (Mₓ, Mᵧ)
    backward_average_ops(::Val{3}) = (Mₓ, Mᵧ, M₂)

    # Forward average operators
    forward_average_ops(::Val{1}) = (M₊ₓ,)
    forward_average_ops(::Val{2}) = (M₊ₓ, M₊ᵧ)
    forward_average_ops(::Val{3}) = (M₊ₓ, M₊ᵧ, M₊₂)

    get_coord(pts, I::CartesianIndex{D}) where {D} = ntuple(d -> pts[d][I[d]], length(I))
    get_coord(pts, I::CartesianIndex{1}) = pts[I[1]]
    coeffs = (2.0, 3.0, 5.0)

    for D in 1:3
        @testset "$D-Dimensional Tests" begin
            dims, Wₕ, uₕ = setup_test_grid(Val(D))
            Ωₕ = mesh(Wₕ)
            pts = points(Ωₕ)
            vₕ = similar(uₕ)
            coords = Base.Fix1(get_coord, pts)

            # Define a linear test function and project it onto the grid
            linear_func(x) = sum(coeffs[i] * x[i] for i in 1:D)
            Rₕ!(uₕ, linear_func)

            @testset "Forward average (M₊)" begin
                for i in 1:D
                    # --- Calculate the analytical expected result ---
                    expected_vals = similar(uₕ.data)
                    li = LinearIndices(dims)
                    step_cartesian = CartesianIndex(ntuple(d -> d == i ? 1 : 0, D))

                    for I in CartesianIndices(dims)
                        # Interior points: f((x_i + x_{i+1})/2)
                        if I[i] < dims[i]
                            midpoint = (coords(I) .+ coords(I + step_cartesian)) ./ 2
                            expected_vals[li[I]] = linear_func(midpoint)
                        else
                            # Boundary point: f(x_N)/2
                            expected_vals[li[I]] = 0#linear_func(coords(I)) / 2
                        end
                    end

                    # Test the primary out-of-place applicator
                    res_oop = forward_average(uₕ, Val(i))
                    @test norm(res_oop.data - expected_vals) < 1e-12

                    # Test the in-place version against the out-of-place one
                    forward_average_dim!(vₕ.data, uₕ.data, dims, Val(i))
                    @test norm(res_oop.data - vₕ.data) < 1e-12
                end

                # --- Test aliases ---
                if D >= 1
                    @test norm(M₊ₓ(uₕ) - forward_average(uₕ, Val(1))) < 1e-12
                end
                if D >= 2
                    @test norm(M₊ᵧ(uₕ) - forward_average(uₕ, Val(2))) < 1e-12
                end
                if D >= 3
                    @test norm(M₊₂(uₕ) - forward_average(uₕ, Val(3))) < 1e-12
                end

                # --- Test vectorial alias ---
                averages = M₊ₕ(uₕ)
                if D == 1
                    @test averages isa VectorElement
                    @test norm(averages - forward_average(uₕ, Val(1))) < 1e-12
                else
                    @test averages isa NTuple{D, VectorElement}
                    for i in 1:D
                        @test norm(averages[i] - forward_average(uₕ, Val(i))) < 1e-12
                    end
                end
            end

            @testset "Backward average (M)" begin
                for i in 1:D
                    # --- Calculate the analytical expected result ---
                    expected_vals = similar(uₕ.data)
                    li = LinearIndices(dims)
                    step_cartesian = CartesianIndex(ntuple(d -> d == i ? 1 : 0, D))

                    for I in CartesianIndices(dims)
                        # Interior points: f((x_i + x_{i-1})/2)
                        if I[i] > 1
                            midpoint = (coords(I) .+ coords(I - step_cartesian)) ./ 2
                            expected_vals[li[I]] = linear_func(midpoint)
                        else
                            # Boundary point: f(x_1)/2
                            expected_vals[li[I]] = 0#linear_func(coords(I)) / 2
                        end
                    end

                    # (The rest of the tests in this block remain the same)
                    # Test the primary out-of-place applicator
                    res_oop = backward_average(uₕ, Val(i))
                    @test norm(res_oop.data - expected_vals) < 1e-12

                    # Test the in-place version against the out-of-place one
                    backward_average_dim!(vₕ.data, uₕ.data, dims, Val(i))
                    @test norm(res_oop.data - vₕ.data) < 1e-12
                end

                # --- Test aliases ---
                if D >= 1
                    @test norm(Mₓ(uₕ) - backward_average(uₕ, Val(1))) < 1e-12
                end
                if D >= 2
                    @test norm(Mᵧ(uₕ) - backward_average(uₕ, Val(2))) < 1e-12
                end
                if D >= 3
                    @test norm(M₂(uₕ) - backward_average(uₕ, Val(3))) < 1e-12
                end

                # --- Test vectorial alias ---
                averages = Mₕ(uₕ)
                if D == 1
                    @test averages isa VectorElement
                    @test norm(averages - backward_average(uₕ, Val(1))) < 1e-12
                else
                    @test averages isa NTuple{D, VectorElement}
                    for i in 1:D
                        @test norm(averages[i] - backward_average(uₕ, Val(i))) < 1e-12
                    end
                end
            end
        end
    end

    @testset "Operator vs matrix" begin
        @testset "Forward" test_operator_matrix_equivalence(forward_average_ops)
        @testset "Backward" test_operator_matrix_equivalence(backward_average_ops)
    end
end

# The averaging matrices carry the same weighting as the differences do, and carried the
# same defect with it: `w .* A` returned a matrix whose storage was sized for the dense
# case. See the matching testset in `test/space/difference.jl` for the measurement.
@testset "Weighted averages: storage ∝ nnz" begin
    Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (100, 100))
    n = npoints(Ωₕ)

    for (name, A) in ("M₊ₓ" => forward_average(Ωₕ, Val(1)), "Mₓ" => backward_average(Ωₕ, Val(1)))
        @testset "$name" begin
            @test nnz(A) <= 2 * n
            @test Base.summarysize(A) <= 64 * nnz(A)
        end
    end
end

using LinearAlgebra: Diagonal
using Bramble: kronecker_operator_matrix, centered_average_dim!, Mcₓ, Mcᵧ, weights, Innerh,
               D₋ᵧ, trial_function, test_function

# The dense matrix backend takes `stencil_matrix`'s dense fallback and the dense branch of
# `_scale_rows!` in the Kronecker oracle; every average family against that oracle, entry for
# entry, on a mesh non-uniform in both directions.
@testset "Averages: dense fallback vs Kronecker" begin
    Ωd = mesh(domain(box((0.0, 0.0), (1.0, 2.0))), (6, 5), (false, false);
        backend = backend(matrix_type = Matrix{Float64}))
    for ops in ((Mₓ, Mᵧ), (M₊ₓ, M₊ᵧ), (Mcₓ, Mcᵧ)), d in 1:2

        @testset "$(ops[d])" begin
            A = ops[d](Ωd)
            @test A isa Matrix{Float64}
            @test A == kronecker_operator_matrix(Ωd, ops[d])
        end
    end
end

# (u_{i-1} + 2 u_i + u_{i+1}) / 4 along the direction, zero on both end slices, written out
# by hand; the in-place form also has to survive `out === in`, which it reads from a copy.
@testset "centered_average_dim!" begin
    dims = (5, 4)
    u = Float64[i^2 + 3j for i in 1:5, j in 1:4]
    ref = (zeros(dims), zeros(dims))
    ref[1][2:4, :] .= (u[1:3, :] .+ 2 .* u[2:4, :] .+ u[3:5, :]) ./ 4
    ref[2][:, 2:3] .= (u[:, 1:2] .+ 2 .* u[:, 2:3] .+ u[:, 3:4]) ./ 4
    for d in 1:2
        out = zeros(20)
        centered_average_dim!(out, vec(u), dims, Val(d))
        @test out == vec(ref[d])
        same = vec(copy(u))
        centered_average_dim!(same, same, dims, Val(d))
        @test same == vec(ref[d])
    end

    err = try
        forward_average_dim!(zeros(5), zeros(6), (2, 3), Val(1))
    catch e
        e
    end
    @test err isa DimensionMismatch
    @test err.msg == "out has 5 entries and in has 6, but the grid (2, 3) has 6"
end

# The form-layer stencils of the forward and centered averages, assembled under the discrete
# L² product, are the space-layer matrices scaled row by row by the quadrature weights.
@testset "Form M₊, Mc stencils vs space matrices" begin
    Ωₕ = mesh(domain(box((0.0, 0.0), (1.0, 2.0))), (7, 6), (false, false))
    Wₕ = gridspace(Ωₕ)
    H = Diagonal(collect(weights(Wₕ, Innerh())))
    for (op_form, op_matrix) in ((u -> M₊ₓ(u), M₊ₓ(Ωₕ)), (u -> Mcᵧ(u), Mcᵧ(Ωₕ)))
        A = Matrix(assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(op_form(u), v))))
        @test maximum(abs, A) > 1.0e-3
        @test isapprox(A, Matrix(H * op_matrix); atol = 1.0e-14)
    end

    # composed with a difference, the nested form is the product of the two matrices
    A = Matrix(assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(M₊ₓ(D₋ᵧ(u)), v))))
    @test maximum(abs, A) > 1.0e-3
    @test isapprox(A, Matrix(H * M₊ₓ(Ωₕ) * D₋ᵧ(Ωₕ)); atol = 1.0e-12)

    # `_wraps_leaf`: an average directly over a bare trial or test leaf, and nothing deeper
    u, v = trial_function(Wₕ), test_function(Wₕ)
    @test Bramble._wraps_leaf(M₊ₓ(u))
    @test Bramble._wraps_leaf(Mcᵧ(v))
    @test !Bramble._wraps_leaf(M₊ₓ(D₋ᵧ(u)))
end

end # module SpaceAverageTests
