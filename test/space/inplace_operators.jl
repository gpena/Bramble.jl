using Test
using Bramble
using Random
using Bramble: values

# The in-place forms of every directional operator.
#
# `_apply_stencil!` and `_average_engine!` were always the core of these operators; every
# allocating form was `similar(uₕ)` followed by a call to one of them. What was missing was
# a public name for that core, so a caller who already had somewhere to put the result had
# no way to say so. Each `!` form now holds the work and each allocating form is one line
# on top of it, which is also why there is no second implementation to keep in step.
#
# Per the return contract, a mutating function with a single destination returns it, so
# `D₋ₓ!(vₕ, uₕ)` gives back `vₕ` and composes: `normₕ(D₋ₓ!(vₕ, uₕ))`.

# Every family's base name, before the x/y/z suffix — the same 10 families
# `_DIFFERENCE_OP_CONFIGS`/`_AVERAGE_OP_CONFIGS` and the two "written out" generators
# (`jump.jl`, `Dc`/`Dstar₊` in `difference.jl`) build a full x/y/z alias set for
# unconditionally, regardless of what dimension a caller ever uses. Deriving the
# per-dimension list mechanically from this tuple and `_DIR_SUFFIXES`, instead of typing
# out each `(name!, name, "name")` triple by hand per `Val(D)`, is what point 78 asks
# for: the old hand-written lists quietly tested `D₊`/`M₋`/`Dstar₊`/`M₊` at only two of
# three directions and `diff₋`/`diff₊` at only one, though every omitted variant already
# resolves to a real, distinct, correct method (confirmed auditing point 66) — a gap in
# this test file's own bookkeeping, not in the operators themselves.
const _INPLACE_FAMILIES = (:D₋, :D₊, :diff₋, :diff₊, :M₋, :M₊, :jump, :Dc, :Dstar₊, :Dₕ)
const _DIR_SUFFIXES = ("ₓ", "ᵧ", "₂")

function _ops(::Val{D}) where {D}
    entries = Tuple{Function, Function, String}[]
    for dim in 1:D, fam in _INPLACE_FAMILIES

        suffix = _DIR_SUFFIXES[dim]
        name = Symbol(fam, suffix)
        push!(entries,
            (getproperty(Bramble, Symbol(name, :!)),
                getproperty(Bramble, name), string(name)))
    end
    return Tuple(entries)
end

@testset "In-place operators" begin
    @testset "Allocating agreement" begin
        Random.seed!(20260831)
        Ωs = (mesh(domain(interval(0.0, 1.0)), 9, false),
            mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (6, 7), (true, false)),
            mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (4, 5, 4),
                (false, true, false)))
        fs = (x -> x^3 + sin(4x) + 1,
            x -> exp(x[1]) * (x[2]^2 + 1),
            x -> x[1]^2 + 2x[2] + sin(x[3]) + 1)

        for D in 1:3
            @testset "$(D)D" begin
                Ωₕ = Ωs[D]
                Wₕ = gridspace(Ωₕ)
                Vₕ = gridspace(Ωₕ, Val(2))
                uₕ = Rₕ(Wₕ, fs[D])
                uv = Rₕ(Vₕ, (fs[D], fs[D]))

                for (f!, f, nm) in _ops(Val(D))
                    @testset "$nm" begin
                        vₕ = similar(uₕ)
                        returned = f!(vₕ, uₕ)
                        @test values(vₕ) == values(f(uₕ))
                        @test returned === vₕ          # single destination returns it

                        # and componentwise over a composite space
                        vv = similar(uv)
                        @test f!(vv, uv) === vv
                        @test values(vv) == values(f(uv))
                    end
                end
            end
        end
    end

    @testset "Destination overwrite" begin
        # Every one of these truncates a boundary slice to zero. If a `!` form skipped
        # those entries instead of writing them, whatever was in the destination would
        # survive — and with a fresh `similar` that is uninitialised memory, so the
        # allocating form would look right while the in-place form returned garbage at the
        # boundary. Pre-filling with a value that cannot be a correct answer catches it.
        Random.seed!(20260831)
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (6, 7), (true, false))
        Wₕ = gridspace(Ωₕ)
        uₕ = Rₕ(Wₕ, x -> exp(x[1]) * (x[2]^2 + 1))

        for (f!, f, nm) in _ops(Val(2))
            @testset "$nm" begin
                vₕ = similar(uₕ)
                values(vₕ) .= -999.0
                f!(vₕ, uₕ)
                @test values(vₕ) == values(f(uₕ))
                @test !any(==(-999.0), values(vₕ))
            end
        end
    end

    @testset "Zero allocations" begin
        # The reason the forms exist. Measured inside a function on concrete locals.
        function counts()
            Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (24, 24),
                (true, false))
            Wₕ = gridspace(Ωₕ)
            uₕ = Rₕ(Wₕ, x -> exp(x[1]) * (x[2]^2 + 1))
            vₕ = similar(uₕ)

            inplace = Int[]
            allocating = Int[]
            for (f!, f, _) in _ops(Val(2))
                f!(vₕ, uₕ)                       # warm up both paths
                f(uₕ)
                push!(inplace, @allocated f!(vₕ, uₕ))
                push!(allocating, @allocated f(uₕ))
            end
            return inplace, allocating
        end

        inplace, allocating = counts()
        @test all(iszero, inplace)
        @test all(>(0), allocating)             # the comparison is not vacuous
    end

    @testset "Composite matching" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 7, true)
        Wₕ = gridspace(Ωₕ)
        Vₕ = gridspace(Ωₕ, Val(2))
        uₕ, uv = Rₕ(Wₕ, sin), Rₕ(Vₕ, (sin, cos))

        # a scalar destination cannot take a composite result, or the reverse
        @test_throws MethodError D₋ₓ!(similar(uₕ), uv)
        @test_throws MethodError D₋ₓ!(similar(uv), uₕ)
    end
end
