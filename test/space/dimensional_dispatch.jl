module SpaceDimensionalDispatchTests

using Test
using Bramble
using Bramble: D̽ₕ
using Bramble: D₋, D̃, Dc, jump
using Bramble: Dcᵧ, Dc₂, Dcₓ, D̃ᵧ, D̃₂, D̃ₓ, D̽ᵧ, D̽₂, D̽ₓ, D₋ᵧ, D₋₂, D₋ₓ, Mᵧ, M₂, Mₓ
using Bramble: VectorElement, jumpᵧ, jump₂, jumpₓ
# Internal or `public` since v3.0 (gpena/Bramble.jl#211), entry points included.
import Bramble: diff₋, diff₋ₓ, diff₋ᵧ, diff₋₂, diff₊, diff₊ₓ, diff₊ᵧ, diff₊₂,
                D₊, D₊ₓ, D₊ᵧ, D₊₂, M₊ₕ, M₊ₓ, M₊ᵧ, M₊₂
using ..TestUtils: alloc_test

# The dimensional entry points (gpena/Bramble.jl#74): `D₋(uₕ, d)` with `d` a `Val`, an `Int`
# or a `Symbol`, over the same `Val`-parameterised base functions the subscript aliases have
# always forwarded to.
#
# Two things are worth testing and only one of them is the values. A boxed `Val` -- `Val(d)`
# built from a runtime `d`, the regression gpena/Bramble.jl#146 measured -- computes exactly
# the right numbers and merely does it through dynamic dispatch, so every value test here
# would pass with the boxing in place. The `@inferred` and allocation testsets are the ones
# that would fail, and they are why this file exists rather than three more lines in
# `difference.jl`.
#
# `M`/`M₊` are absent on purpose: the averages put their entry point on `Mₕ`/`M₊ₕ` instead of
# minting a bare `M`, which `using Bramble` would take away from a caller's mass matrix.

# (entry point, the three per-coordinate aliases it has to agree with)
const FAMILIES = (
    (diff₋, (diff₋ₓ, diff₋ᵧ, diff₋₂)),
    (diff₊, (diff₊ₓ, diff₊ᵧ, diff₊₂)),
    (D₋, (D₋ₓ, D₋ᵧ, D₋₂)),
    (D₊, (D₊ₓ, D₊ᵧ, D₊₂)),
    (D̃, (D̃ₓ, D̃ᵧ, D̃₂)),
    (Dc, (Dcₓ, Dcᵧ, Dc₂)),
    (D̽ₕ, (D̽ₓ, D̽ᵧ, D̽₂)),
    (jump, (jumpₓ, jumpᵧ, jump₂)),
    (Mₕ, (Mₓ, Mᵧ, M₂)),
    (M₊ₕ, (M₊ₓ, M₊ᵧ, M₊₂))
)

const SYMBOLS = (:x, :y, :z)

@testset "Dimensional dispatch" begin
    Ωₕ1 = mesh(domain(interval(0.0, 1.0)), 16, false)
    Ωₕ2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (8, 9), (true, false))
    Ωₕ3 = mesh(
        domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (4, 5, 6), (true, false, true)
    )
    Wₕ1, Wₕ2, Wₕ3 = gridspace(Ωₕ1), gridspace(Ωₕ2), gridspace(Ωₕ3)
    Vₕ2 = gridspace(Ωₕ2, Val(2))

    uₕ1 = Rₕ(Wₕ1, sin)
    uₕ2 = Rₕ(Wₕ2, x -> sin(x[1]) * x[2])
    uₕ3 = Rₕ(Wₕ3, x -> sin(x[1]) + x[3])
    cₕ2 = Rₕ(Vₕ2, (x -> x[1], x -> x[2]))

    @testset "agrees with the subscript aliases" begin
        for (lbl, uₕ, D) in (("1D", uₕ1, 1), ("2D", uₕ2, 2), ("3D", uₕ3, 3))
            @testset "$lbl" begin
                for (entry, aliases) in FAMILIES, d in 1:D

                    @test entry(uₕ, d) == aliases[d](uₕ)
                    @test entry(uₕ, SYMBOLS[d]) == aliases[d](uₕ)
                    @test entry(uₕ, Val(d)) == aliases[d](uₕ)
                end
            end
        end
    end

    @testset "a mesh and a space give the matrix, as the aliases do" begin
        for (entry, aliases) in FAMILIES, d in 1:2

            @test entry(Ωₕ2, d) == aliases[d](Ωₕ2)
            @test entry(Wₕ2, SYMBOLS[d]) == aliases[d](Wₕ2)
        end
    end

    @testset "composite grid functions go through the same dispatch" begin
        for (entry, aliases) in FAMILIES, d in 1:2

            @test entry(cₕ2, d) == aliases[d](cₕ2)
            @test entry(cₕ2, SYMBOLS[d]) == aliases[d](cₕ2)
        end
    end

    # `D̽ₕ`, `Mₕ` and `M₊ₕ` carry both arities: the tuple with one argument, one direction
    # with two. They coexist by arity, and this is the test that says so on purpose.
    @testset "the tuple-valued aliases keep their one-argument meaning" begin
        @test D̽ₕ(uₕ2) == (D̽ₓ(uₕ2), D̽ᵧ(uₕ2))
        @test Mₕ(uₕ2) == (Mₓ(uₕ2), Mᵧ(uₕ2))
        @test M₊ₕ(uₕ2) == (M₊ₓ(uₕ2), M₊ᵧ(uₕ2))
        @test D̽ₕ(uₕ3) == (D̽ₓ(uₕ3), D̽ᵧ(uₕ3), D̽₂(uₕ3))
        # in one dimension the tuple collapses to the entry itself, and the two arities
        # then agree on the same value rather than disagreeing on its shape
        @test Mₕ(uₕ1) == Mₓ(uₕ1) == Mₕ(uₕ1, 1)
        @test D̽ₕ(uₕ1) == D̽ₓ(uₕ1) == D̽ₕ(uₕ1, :x)
    end

    @testset "out-of-range directions throw" begin
        # the bound is the mesh's own dimension, not 3: a 1D grid has no `y`
        @test_throws ArgumentError D₋(uₕ1, 2)
        @test_throws ArgumentError D₋(uₕ1, :y)
        @test_throws ArgumentError D₋(uₕ2, 3)
        @test_throws ArgumentError D₋(uₕ2, :z)
        @test_throws ArgumentError D₋(uₕ3, 4)
        @test_throws ArgumentError D₋(uₕ3, 0)
        @test_throws ArgumentError Mₕ(uₕ2, :z)
        @test_throws ArgumentError D₋(uₕ2, :w)
        @test_throws ArgumentError jump(uₕ2, :nope)
    end

    @testset "Type stability of the Int and Symbol entry points" begin
        for (lbl, uₕ, D) in (("1D", uₕ1, 1), ("2D", uₕ2, 2), ("3D", uₕ3, 3))
            @testset "$lbl" begin
                for (entry, _) in FAMILIES, d in 1:D

                    # `Val(d)` is deliberately absent: `d` is a loop variable here, so
                    # `@inferred` would be judging the test's own `Val(d)` and not the
                    # method under it. The `Val` path is what the subscript aliases already
                    # exercise; what is new, and what could box, is the `Int` and the
                    # `Symbol`.
                    @test @inferred(entry(uₕ, d)) isa VectorElement
                    @test @inferred(entry(uₕ, SYMBOLS[d])) isa VectorElement
                end
            end
        end
        @test @inferred(D₋(cₕ2, 1)) isa VectorElement
        @test @inferred(D₋(cₕ2, :y)) isa VectorElement
    end

    # The measurement the value tests cannot make. A boxed `Val` allocates a box per call and
    # then dispatches dynamically down the whole engine call stack; the subscript alias
    # cannot box, so it is the baseline to compare against rather than a bare number that
    # would have to be updated whenever `similar` changes.
    @testset "a runtime direction costs no more than a literal one" begin
        for (lbl, uₕ, D) in (("1D", uₕ1, 1), ("2D", uₕ2, 2), ("3D", uₕ3, 3))
            @testset "$lbl" begin
                for (entry, aliases) in FAMILIES, d in 1:D

                    baseline = alloc_test(() -> aliases[d](uₕ))
                    @test alloc_test(() -> entry(uₕ, d)) == baseline
                    @test alloc_test(() -> entry(uₕ, SYMBOLS[d])) == baseline
                end
            end
        end
    end

    # What the entry points exist for, and the first dimension-agnostic expression the
    # package can write: one loop that reads the same in 1D, 2D and 3D.
    @testset "a loop over directions" begin
        for (uₕ, D, aliases) in (
            (uₕ1, 1, (D₋ₓ,)), (uₕ2, 2, (D₋ₓ, D₋ᵧ)), (uₕ3, 3, (D₋ₓ, D₋ᵧ, D₋₂))
        )
            @test sum(innerₕ(D₋(uₕ, d), D₋(uₕ, d)) for d in 1:D) ==
                  sum(innerₕ(f(uₕ), f(uₕ)) for f in aliases)
        end
    end
end

end # module
