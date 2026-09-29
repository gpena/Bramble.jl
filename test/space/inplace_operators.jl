module SpaceInplaceOperatorsTests

using Test
using Bramble
using Bramble: D₋ₓ!, VectorElement
using Random
using ..TestUtils: alloc_test, @test_allocs

# The in-place forms of every directional operator.
#
# `_apply_stencil!` and `_average_engine!` were always the core of these operators; every
# allocating form was `similar(uₕ)` followed by a call to one of them. What was missing was
# a public name for that core, so a caller who already had somewhere to put the result had
# no way to say so. Each `!` form now holds the work and each allocating form is one line
# on top of it, which is also why there is no second implementation to keep in step.
#
# Per the return contract, a mutating function with a single destination returns it, so

# Base names for the 10 directional operator families across spatial dimensions.
# Deriving the per-dimension list mechanically from this tuple and `_DIR_SUFFIXES`
# ensures complete and uniform test coverage across 1D, 2D, and 3D.
const _INPLACE_FAMILIES = (:D₋, :D₊, :diff₋, :diff₊, :M, :M₊, :jump, :Dc, :D̃, :D̽)
const _DIR_SUFFIXES = ("ₓ", "ᵧ", "₂")

function _ops(::Val{D}) where {D}
    entries = Tuple{Function, Function, String}[]
    for dim in 1:D, fam in _INPLACE_FAMILIES

        suffix = _DIR_SUFFIXES[dim]
        name = Symbol(fam, suffix)
        push!(
            entries,
            (
                getproperty(Bramble, Symbol(name, :!)),
                getproperty(Bramble, name),
                string(name)
            )
        )
    end
    return Tuple(entries)
end

@testset "In-place operators" begin
    @testset "Allocating agreement" begin
        Random.seed!(20260831)
        Ωs = (
            mesh(domain(interval(0.0, 1.0)), 9, false),
            mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (6, 7), (true, false)),
            mesh(
                domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))),
                (4, 5, 4),
                (false, true, false)
            )
        )
        fs = (
            x -> x^3 + sin(4x) + 1,
            x -> exp(x[1]) * (x[2]^2 + 1),
            x -> x[1]^2 + 2x[2] + sin(x[3]) + 1
        )

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
                        @test parent(vₕ) == parent(f(uₕ))
                        @test returned === vₕ          # single destination returns it

                        # and componentwise over a composite space
                        vv = similar(uv)
                        @test f!(vv, uv) === vv
                        @test parent(vv) == parent(f(uv))
                    end
                end
            end
        end
    end

    @testset "Destination overwrite" begin
        # Most of these truncate a boundary slice to zero (D̽ instead falls back to a
        # one-sided difference there, gpena/Bramble.jl#183, but still writes a real,
        # non-sentinel value). If a `!` form skipped those entries instead of writing
        # them, whatever was in the destination would survive (with a fresh `similar`
        # that is uninitialised memory), so the allocating form would look right while
        # the in-place form returned garbage at the boundary. Pre-filling with a value
        # that cannot be a correct answer catches it.
        Random.seed!(20260831)
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (6, 7), (true, false))
        Wₕ = gridspace(Ωₕ)
        uₕ = Rₕ(Wₕ, x -> exp(x[1]) * (x[2]^2 + 1))

        for (f!, f, nm) in _ops(Val(2))
            @testset "$nm" begin
                vₕ = similar(uₕ)
                parent(vₕ) .= -999.0
                f!(vₕ, uₕ)
                @test parent(vₕ) == parent(f(uₕ))
                @test !any(==(-999.0), parent(vₕ))
            end
        end
    end

    @testset "Zero allocations" begin
        # The reason the forms exist. Measured inside a function on concrete locals.
        function counts()
            Ωₕ = mesh(
                domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (24, 24), (true, false)
            )
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
            # In-place copyto! assignment checks (values!'s replacement, gpena/Bramble.jl#73)
            copyto!(vₕ, 0.0)
            copyto!(vₕ, parent(uₕ))
            push!(inplace, @allocated copyto!(vₕ, 0.0))
            push!(inplace, @allocated copyto!(vₕ, parent(uₕ)))

            return inplace, allocating
        end

        inplace, allocating = counts()
        @test all(iszero, inplace)
        @test all(>(0), allocating)             # the comparison is not vacuous
    end

    @testset "Aliasing rejected" begin
        # Every stencil here reads a neighbour of the point it writes, and the traversal
        # overwrites entries in place as it goes; calling `f!(uₕ, uₕ)` would silently
        # corrupt the result from the second write onward instead of raising an error
        # (this was the reported bug for `D₋ₓ!`). Each family must refuse aliased
        # destination and source rather than compute a wrong answer.
        Ωs = (
            mesh(domain(interval(0.0, 1.0)), 9, false),
            mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (6, 7), (true, false)),
            mesh(
                domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))),
                (4, 5, 4),
                (false, true, false)
            )
        )
        fs = (
            x -> x^3 + sin(4x) + 1,
            x -> exp(x[1]) * (x[2]^2 + 1),
            x -> x[1]^2 + 2x[2] + sin(x[3]) + 1
        )

        for D in 1:3
            @testset "$(D)D" begin
                Ωₕ = Ωs[D]
                Wₕ = gridspace(Ωₕ)
                Vₕ = gridspace(Ωₕ, Val(2))
                uₕ = Rₕ(Wₕ, fs[D])
                uv = Rₕ(Vₕ, (fs[D], fs[D]))

                for (f!, _, nm) in _ops(Val(D))
                    @testset "$nm" begin
                        # the same object
                        @test_throws ArgumentError f!(uₕ, uₕ)
                        @test_throws ArgumentError f!(uv, uv)

                        # distinct `VectorElement`s sharing the same backing array
                        shared = VectorElement(parent(uₕ), space(uₕ))
                        @test_throws ArgumentError f!(uₕ, shared)
                        @test_throws ArgumentError f!(shared, uₕ)
                    end
                end
            end
        end
    end

    @testset "Grid mismatch rejected" begin
        # The engines index both vectors under `@inbounds` with the source's grid shape, so
        # a destination of another size was written past its end: on a 4x5 source, `D₋ₓ!`
        # wrote 11 entries beyond a 9-entry view, which on a plain `Vector` corrupts
        # memory. A destination of the right length on a differently shaped grid is not a
        # memory error, but it would still receive the source's values in the wrong
        # places. Every family has to refuse all three, before writing anything.
        Ωs = (
            mesh(domain(interval(0.0, 1.0)), 9, false),
            mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (4, 5), (true, false)),
            mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (4, 5, 3), (false, true, false))
        )
        bigger = (
            mesh(domain(interval(0.0, 1.0)), 12, true),
            mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (6, 5), (true, true)),
            mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (4, 5, 4), (true, true, true))
        )
        # the same number of points, arranged differently; a 1D grid has only one shape
        permuted = (
            nothing,
            mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 4), (true, false)),
            mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (3, 5, 4), (false, true, false))
        )
        centered = (Bramble.Mcₓ!, Bramble.Mcᵧ!, Bramble.Mc₂!)
        shifts = (Bramble.S₊ₓ!, Bramble.S₋ₓ!, Bramble.S₊ᵧ!, Bramble.S₋ᵧ!, Bramble.S₊₂!,
            Bramble.S₋₂!)
        fs = (x -> x^2 + 1, x -> x[1] + 2x[2]^2 + 1, x -> x[1] * x[2] + x[3] + 1)

        # A NaN-filled destination of `len` entries, viewed from a longer array so that a
        # write past its end lands somewhere the test can see.
        nan_dest(len, Wₕ) = VectorElement(view(fill(NaN, len + 31), 1:len), Wₕ)

        # Throws an ArgumentError and leaves the whole backing array as it was.
        function rejects(f!, vₕ, uₕ)
            raw = parent(parent(vₕ))
            before = copy(raw)
            thrown = try
                f!(vₕ, uₕ)
                false
            catch err
                err isa ArgumentError
            end
            return thrown && isequal(raw, before)
        end

        for D in 1:3
            @testset "$(D)D" begin
                Wₕ, Vₕ = gridspace(Ωs[D]), gridspace(Ωs[D], Val(2))
                Bₕ, Bᵥ = gridspace(bigger[D]), gridspace(bigger[D], Val(2))
                uₕ, uv = Rₕ(Wₕ, fs[D]), Rₕ(Vₕ, (fs[D], fs[D]))
                n, m = ndofs(Wₕ), ndofs(Bₕ)

                for f! in (map(first, _ops(Val(D)))..., centered[1:D]..., shifts[1:(2D)]...)
                    @testset "$f!" begin
                        # smaller: the source's own space over too short a view
                        @test rejects(f!, nan_dest(n - 1, Wₕ), uₕ)
                        @test rejects(f!, nan_dest(2n - 1, Vₕ), uv)

                        # larger: a grid function of a bigger mesh
                        @test rejects(f!, nan_dest(m, Bₕ), uₕ)
                        @test rejects(f!, nan_dest(2m, Bᵥ), uv)

                        # equal length, different mesh shape
                        if permuted[D] !== nothing
                            Pₕ, Pᵥ = gridspace(permuted[D]), gridspace(permuted[D], Val(2))
                            @test ndofs(Pₕ) == n
                            @test rejects(f!, nan_dest(n, Pₕ), uₕ)
                            @test rejects(f!, nan_dest(2n, Pᵥ), uv)
                        end

                        # a matching destination is still accepted
                        @test f!(nan_dest(n, Wₕ), uₕ) isa VectorElement
                        @test f!(nan_dest(2n, Vₕ), uv) isa VectorElement
                    end
                end
            end
        end
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

end # module SpaceInplaceOperatorsTests
