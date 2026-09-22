module SpaceCenteredAverageTests

using Test
using Bramble
import Bramble: centered_average, stencil_matrix, kronecker_operator_matrix, CenteredAvgOp
using Random: Xoshiro
using SparseArrays: nnz
using ..TestUtils: @test_allocs

# Random non-uniform mesh on the unit cube of dimension D.
function random_space(D; n = (9, 8, 7))
    dom = D == 1 ? domain(interval(0.0, 1.0)) :
          D == 2 ? domain(interval(0.0, 1.0) × interval(0.0, 1.0)) :
          domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    Ωₕ = D == 1 ? mesh(dom, n[1], false) : mesh(dom, n[1:D], ntuple(_ -> false, D))
    return gridspace(Ωₕ)
end

# Oracle: (u(i-1) + 2u(i) + u(i+1))/4 along d, zero on both end slices of d.
function oracle(a, dims, d)
    A = reshape(a, dims)
    out = zero(A)
    e = CartesianIndex(ntuple(k -> k == d ? 1 : 0, length(dims)))
    for I in CartesianIndices(A)
        (I[d] == 1 || I[d] == dims[d]) && continue
        out[I] = (A[I - e] + 2A[I] + A[I + e]) / 4
    end
    return vec(out)
end

@testset "Centered averages (#287)" begin
    rng = Xoshiro(287)
    ops = (Mcₓ, Mcᵧ, Mc₂)
    ops! = (Mcₓ!, Mcᵧ!, Mc₂!)

    for D in 1:3
        @testset "$(D)D" begin
            Wₕ = random_space(D)
            Ωₕ = mesh(Wₕ)
            dims = npoints(Ωₕ, Tuple)
            uₕ = element(Wₕ)
            uₕ.data .= rand(rng, length(uₕ.data))
            vₕ = similar(uₕ)

            for d in 1:D
                ref = oracle(uₕ.data, dims, d)
                @test ops[d](uₕ).data ≈ ref
                @test Mcₕ(uₕ, d).data ≈ ref
                @test centered_average(uₕ, Val(d)).data ≈ ref

                ops![d](vₕ, uₕ)
                @test vₕ.data ≈ ref
                @test_allocs ops![d](vₕ, uₕ)

                # Both end slices of direction d are zero.
                V = reshape(vₕ.data, dims)
                @test all(iszero, selectdim(V, d, 1))
                @test all(iszero, selectdim(V, d, dims[d]))

                # Matrix, Kronecker oracle and engine agree.
                A = stencil_matrix(Ωₕ, CenteredAvgOp{d}())
                K = kronecker_operator_matrix(Ωₕ, ops[d])
                @test A ≈ K
                @test nnz(A) == nnz(K)
                @test A * uₕ.data ≈ ref
                @test ops[d](Ωₕ) ≈ A
            end

            t = Mcₕ(uₕ)
            if D == 1
                @test t isa VectorElement
                @test t.data ≈ Mcₓ(uₕ).data
            else
                @test t isa NTuple{D, VectorElement}
                @test all(t[d].data ≈ ops[d](uₕ).data for d in 1:D)
            end

            @test_throws ArgumentError Mcₓ!(uₕ, uₕ)
        end
    end
end

end # module SpaceCenteredAverageTests
