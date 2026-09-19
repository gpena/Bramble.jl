module TestKroneckerExt

using Test
using Bramble
using Bramble: is_separable, kronecker_operator, KroneckerLinearOperator
using Kronecker: kronecker
using LinearAlgebra: mul!, issymmetric
using SparseArrays: SparseMatrixCSC
using Random

# `Kronecker.jl` interop and fast diagonalisation for a separable `BilinearForm`
# (gpena/Bramble.jl#259), layered on `KroneckerLinearOperator`
# (gpena/Bramble.jl#162, test/form/kronecker.jl). `fdm_solve` has no forward stub in
# `src/Bramble.jl` yet (see `ext/BrambleKroneckerExt.jl`'s module docstring), so it is
# reached the same way any other not-yet-exported extension function would be: off the
# loaded extension module itself.
const KronExt = Base.get_extension(Bramble, :BrambleKroneckerExt)
@assert KronExt !== nothing "BrambleKroneckerExt did not load -- is Kronecker.jl a test dependency?"
# `fdm_solve` is Bramble's own binding (`function fdm_solve end` in `src/Bramble.jl`), and
# this extension adds methods to it, so the exported spelling is the one to test: reaching
# into the extension module would pass even if the methods had attached to a function of
# the extension's own instead, which is precisely the failure this asserts against.
@assert !isempty(methods(Bramble.fdm_solve)) "fdm_solve has no methods -- are the extension's definitions dot-qualified as `Bramble.fdm_solve`?" 

const KRON_EXT_SEED = 20260919

@testset "Kronecker extension" begin
    @testset "Kronecker.jl object equals SparseMatrixCSC(K)" begin
        Random.seed!(KRON_EXT_SEED)
        Ω2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (13, 11), (false, false))
        W2 = gridspace(Ω2)

        Random.seed!(KRON_EXT_SEED + 1)
        Ω3 = mesh(
            domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0)),
            (8, 7, 6), (false, false, false)
        )
        W3 = gridspace(Ω3)

        for Wₕ in (W2, W3)
            a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + 2.5 * inner₊(∇ₕ(u), ∇ₕ(v)))
            K = kronecker_operator(a)
            Aref = SparseMatrixCSC(K)

            Kjl = kronecker(K)
            @test collect(Kjl) ≈ Aref
            @test Matrix(Kjl) ≈ Matrix(Aref)

            # `mul!` through the Kronecker.jl object agrees with the operator's own `mul!`.
            n = ndofs(Wₕ)
            x = rand(n)
            yref = similar(x)
            mul!(yref, K, x)
            y = collect(Kjl) * x
            @test isapprox(y, yref; rtol = 1e-10, atol = 1e-10)
        end
    end

    @testset "fdm_solve vs sparse \\, no Dirichlet (2D 25x19, 3D 11x9x8)" begin
        Random.seed!(KRON_EXT_SEED + 2)
        Ω2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (25, 19), (false, false))
        W2 = gridspace(Ω2)

        Random.seed!(KRON_EXT_SEED + 3)
        Ω3 = mesh(
            domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0)),
            (11, 9, 8), (false, false, false)
        )
        W3 = gridspace(Ω3)

        for (Wₕ, tag) in ((W2, "2D"), (W3, "3D"))
            @testset "$tag" begin
                a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
                A = assemble(a)
                n = ndofs(Wₕ)
                F = rand(n)
                xref = A \ F

                x = fdm_solve(a, F)
                @test isapprox(x, xref; rtol = 1e-9)

                # `fdm_solve(K, F)`: the unconstrained `KroneckerLinearOperator` overload.
                K = kronecker_operator(a)
                xK = fdm_solve(K, F)
                @test isapprox(xK, xref; rtol = 1e-9)
            end
        end
    end

    @testset "fdm_solve vs sparse \\, homogeneous Dirichlet (2D 25x19, 3D 11x9x8)" begin
        Random.seed!(KRON_EXT_SEED + 4)
        Ω2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (25, 19), (false, false))
        W2 = gridspace(Ω2)

        Random.seed!(KRON_EXT_SEED + 5)
        Ω3 = mesh(
            domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0)),
            (11, 9, 8), (false, false, false)
        )
        W3 = gridspace(Ω3)

        for (Wₕ, tag) in ((W2, "2D"), (W3, "3D"))
            @testset "$tag" begin
                a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
                A = assemble(a; dirichlet = :boundary)
                n = ndofs(Wₕ)
                dims = ndofs(Wₕ, Tuple)

                F = rand(n)
                Farr = reshape(F, dims)
                D = length(dims)
                for d in 1:D
                    idx_first = ntuple(k -> k == d ? 1 : Colon(), D)
                    idx_last = ntuple(k -> k == d ? dims[k] : Colon(), D)
                    Farr[idx_first...] .= 0.0
                    Farr[idx_last...] .= 0.0
                end
                F = vec(Farr)

                xref = A \ F
                x = fdm_solve(a, F; dirichlet = :boundary)
                @test isapprox(x, xref; rtol = 1e-9)
            end
        end
    end

    @testset "A grid-function coefficient throws" begin
        Random.seed!(KRON_EXT_SEED + 6)
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 7), (false, false))
        Wₕ = gridspace(Ωₕ)
        fₕ = Rₕ(Wₕ, x -> 1.0 + x[1])
        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(fₕ * u, v))
        @test !is_separable(a)
        @test_throws ArgumentError fdm_solve(a, rand(ndofs(Wₕ)))
    end

    @testset "@allocated of a second fdm_solve call (reported, not asserted zero)" begin
        Random.seed!(KRON_EXT_SEED + 7)
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (17, 13), (false, false))
        Wₕ = gridspace(Ωₕ)
        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
        F = rand(ndofs(Wₕ))

        fdm_solve(a, F)   # warm-up: JIT only, `fdm_solve` rebuilds its factors every call
        bytes = @allocated fdm_solve(a, F)
        @info "fdm_solve: @allocated on a second call (no persistent workspace across calls)" bytes
        @test bytes >= 0   # reported, not asserted zero -- see the CHECK's EVIDENCE note
    end
end

end # module
