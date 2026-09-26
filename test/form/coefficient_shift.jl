module FormCoefficientShiftTests

using Test
using Bramble
using LinearAlgebra: Diagonal, I
using Random
# Internal since v3.0 (gpena/Bramble.jl#211): defined and documented, not exported.
import Bramble: D₊ₓ, M₊ₓ, M₊ᵧ
# S3 (gpena/Bramble.jl#271, O5): the AST accessor and stencil-level entry point needed to
# assert the point-dependent path infers concretely -- neither is exported.
import Bramble: local_stencil, resolve_form_ast
using Bramble: Dcₓ, D₋ᵧ, D₋ₓ, Mᵧ, Mₓ, indices, jumpₓ

# gpena/Bramble.jl#271: a grid-function coefficient inside a shifting node (a difference,
# average, jump, or `shift_op`) must be read at the point the tap reaches, not at the point
# being visited. `D₋ₓ(cₕ * u)` assembles `H · Dx · C` -- the coefficient scales the trial
# column after the difference relabels it -- not `H · C · Dx`, which is what the pre-fix
# code produced by relabelling the coefficient's own offset along with the trial function's.
#
# Oracle (.agents/plans/coefficient-inside-difference-notes.md, "Oracle to use in tests"):
# `assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(op(u), v)))` is `H · Op`; the coefficient-inside
# spelling must equal that times `Diagonal(parent(cₕ))` on the RIGHT. Multiplying on the
# LEFT is the pre-fix reading, so every case below also asserts the two products differ --
# on a constant coefficient, or an operator the two happen to coincide on, the equality
# above would pass without the fix being exercised at all.

@testset "Coefficient inside a shifting node" begin
    Random.seed!(20260918)

    @testset "1D: every shifting node" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 7, false)   # non-uniform: false, not true
        Wₕ = gridspace(Ωₕ)
        cₕ = Rₕ(Wₕ, x -> 1 + x)                            # varies, so cᵢ ≠ cᵢ₋₁
        C = Diagonal(collect(parent(cₕ)))
        asm(f) = Matrix(assemble(form(Wₕ, Wₕ, f)))

        for (nm, op) in (
            ("D₋ₓ", D₋ₓ),
            ("D₊ₓ", D₊ₓ),
            ("Dcₓ", Dcₓ),
            ("Mₓ", Mₓ),
            ("M₊ₓ", M₊ₓ),
            ("jumpₓ", jumpₓ),
            ("shift_op", u -> Bramble.shift_op(u, 1, 1))
        )
            @testset "$nm" begin
                base = asm((u, v) -> innerₕ(op(u), v))
                got = asm((u, v) -> innerₕ(op(cₕ * u), v))

                # the oracle: coefficient-inside is H · Op · C
                @test isapprox(got, base * C; atol = 1e-12)
                # and it must actually differ from H · C · Op, the pre-fix reading --
                # otherwise this mesh and coefficient cannot tell the two readings apart
                # and the check above proves nothing
                @test !isapprox(base * C, C * base; atol = 1e-12)
            end
        end
    end

    @testset "1D: a sum holding a coefficient" begin
        # O4: `_combine_shift_traits` makes a sum point-dependent the moment either summand
        # is, which after the fix includes any summand holding a `GridFunctionScale`. The
        # translation-invariant summand (`u` alone) must keep its trial column instead of
        # being re-evaluated at the shifted point along with the coefficient-carrying one.
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 7, false)
        Wₕ = gridspace(Ωₕ)
        cₕ = Rₕ(Wₕ, x -> 1 + x)
        C = Diagonal(collect(parent(cₕ)))
        asm(f) = Matrix(assemble(form(Wₕ, Wₕ, f)))

        base = asm((u, v) -> innerₕ(D₋ₓ(u), v))
        got = asm((u, v) -> innerₕ(D₋ₓ(cₕ * u + u), v))

        @test isapprox(got, base * (C + I); atol = 1e-12)
        @test !isapprox(base * (C + I), (C + I) * base; atol = 1e-12)
    end

    @testset "2D: Dim is exercised in more than one direction" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 6), (false, false))
        Wₕ = gridspace(Ωₕ)
        cₕ = Rₕ(Wₕ, x -> 1 + x[1] + x[2])                  # varies in both directions
        C = Diagonal(collect(parent(cₕ)))
        asm(f) = Matrix(assemble(form(Wₕ, Wₕ, f)))

        for (nm, op) in (("D₋ₓ", D₋ₓ), ("D₋ᵧ", D₋ᵧ), ("Mᵧ", Mᵧ), ("M₊ᵧ", M₊ᵧ))
            @testset "$nm" begin
                base = asm((u, v) -> innerₕ(op(u), v))
                got = asm((u, v) -> innerₕ(op(cₕ * u), v))

                @test isapprox(got, base * C; atol = 1e-12)
                @test !isapprox(base * C, C * base; atol = 1e-12)
            end
        end
    end

    # gpena/Bramble.jl#271, O5: the fix makes `D₋ₓ(cₕ * u)` take the point-dependent path
    # (`shifted_inner_stencil` on `GridFunctionScale`, src/ast/common.jl) -- it re-evaluates
    # the inner stencil at the shifted point on every tap, rather than relabelling offsets
    # once the way the coefficient-outside form `cₕ * D₋ₓ(u)` does. The acceptance criterion
    # is that this cost is measured and recorded, not assumed.
    @testset "The point-dependent path's assembly cost" begin
        @testset "Type stability" begin
            Ωₕ = mesh(domain(interval(0.0, 1.0)), 7, false)
            Wₕ = gridspace(Ωₕ)
            cₕ = Rₕ(Wₕ, x -> 1 + x)
            a = form(Wₕ, Wₕ, (u, v) -> innerₕ(D₋ₓ(cₕ * u), v))

            # The trial-side AST node that `shifted_inner_stencil` recurses through
            # (a `BackwardDifference` wrapping a `GridFunctionScale`) must infer concretely
            # at the stencil-evaluation entry point, not merely run.
            trial_ast = resolve_form_ast(a).left_op
            I = CartesianIndex(4)
            lin = LinearIndices(indices(Ωₕ))[I]
            @test @inferred(local_stencil(trial_ast, Wₕ, I, nothing, lin)) isa Tuple

            # The public assembly entry point.
            @test @inferred(assemble(a)) isa AbstractMatrix
        end

        @testset "Allocation does not scale with ndofs" begin
            # `assemble!` refills a preallocated matrix and is documented (src/assembly/bilinear.jl)
            # to cost 0 bytes; the point-dependent path's extra per-tap re-evaluation must not
            # turn into extra per-tap allocation, at any grid size.
            #
            # Measured on this machine (julia --startup-file=no --project=test, Julia 1.13.0,
            # 1 thread), assembling `innerₕ(D₋ₓ(cₕ * u), v)` with `assemble!` into a
            # preallocated matrix: 0 bytes at ndofs = 65, 1025 and 16385 -- flat, not growing
            # with the grid.
            function assemble_bang_bytes(n)
                Ωₙ = mesh(domain(interval(0.0, 1.0)), n, false)
                Wₙ = gridspace(Ωₙ)
                cₙ = Rₕ(Wₙ, x -> 1 + x)
                aₙ = form(Wₙ, Wₙ, (u, v) -> innerₕ(D₋ₓ(cₙ * u), v))
                Aₙ = assemble(aₙ)   # sparsity pattern, built once
                assemble!(Aₙ, aₙ)   # warm the refill path
                return @allocated assemble!(Aₙ, aₙ)
            end

            @test assemble_bang_bytes(65) == 0
            @test assemble_bang_bytes(1025) == 0
        end

        # Not asserted here: assembly time is not something a portable, non-flaky @test can
        # pin (bramble-verification §1-2 -- absolute timings drift across runs/machines, and
        # this repo's own benchmark suite, not the test suite, is where a timing regression
        # gate belongs). Measured instead, same-run and interleaved so clock/power drift hits
        # both sides equally: `assemble!` on the same 1D mesh (coefficient `c(x) = 1 + x`),
        # batches of 5000 calls, minimum over 9 batches, two independent passes:
        #
        #   ndofs   inside (cₕ*u under D₋ₓ)   outside (cₕ*D₋ₓ(u))   ratio
        #   65      2.091e-7 s / 2.101e-7 s   1.964e-7 s / 1.914e-7 s   1.065 / 1.097
        #   1025    2.549e-6 s / 2.521e-6 s   2.319e-6 s / 2.350e-6 s   1.099 / 1.073
        #
        # i.e. the point-dependent path costs roughly 6-10% more per assembly than the
        # coefficient-outside form on this operator, at both grid sizes measured -- modest,
        # because `D₋ₓ`'s inner operand is the identity, so what the point-dependent branch
        # re-evaluates per tap is cheap. A costlier inner operand (an average, an
        # interpolation) would be expected to pay more; that is not what this form exercises.
    end
end

end # module FormCoefficientShiftTests
