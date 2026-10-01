module SpaceInplaceOperatorsTests

using Test
using Bramble
using Bramble: D₋ₓ!, VectorElement
using Random
using ..TestUtils: alloc_test, @test_allocs, WITH_SLOW_TESTS

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

# `unit` runs one family per engine in the grid-mismatch sweep (a backward difference, an
# average, a centered difference and the one-sided fallback `D̽`, plus the centered average
# and the x shifts below): every `!` form compiles its own rejection path over a view-backed
# destination, which is what the sweep costs. `slow` runs every family.
const _UNIT_FAMILIES = (:D₋, :M, :Dc, :D̽)

function _ops(::Val{D}, families = _INPLACE_FAMILIES) where {D}
    entries = Tuple{Function, Function, String}[]
    for dim in 1:D, fam in families

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

# One dimension of "Allocating agreement", a function of its own so the dimension, mesh and
# test function are concrete types. Indexing the three-mesh tuple with a loop counter infers
# a union of the three meshes, which JET follows into pairs that never occur (a 1D mesh
# with a 3D function).
function _agreement_case(::Val{D}, Ωₕ, fun) where {D}
    @testset "$(D)D" begin
        Wₕ = gridspace(Ωₕ)
        Vₕ = gridspace(Ωₕ, Val(2))
        uₕ = Rₕ(Wₕ, fun)
        uv = Rₕ(Vₕ, (fun, fun))

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

        _agreement_case(Val(1), Ωs[1], fs[1])
        _agreement_case(Val(2), Ωs[2], fs[2])
        _agreement_case(Val(3), Ωs[3], fs[3])
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

        families = WITH_SLOW_TESTS ? _INPLACE_FAMILIES : _UNIT_FAMILIES
        nshifts(D) = WITH_SLOW_TESTS ? 2D : 2
        for D in 1:3
            @testset "$(D)D" begin
                Wₕ, Vₕ = gridspace(Ωs[D]), gridspace(Ωs[D], Val(2))
                Bₕ, Bᵥ = gridspace(bigger[D]), gridspace(bigger[D], Val(2))
                uₕ, uv = Rₕ(Wₕ, fs[D]), Rₕ(Vₕ, (fs[D], fs[D]))
                n, m = ndofs(Wₕ), ndofs(Bₕ)

                for f! in (map(first, _ops(Val(D), families))..., centered[1:D]..., shifts[1:nshifts(D)]...)
                    @testset "$f!" begin
                        # smaller: the source's own space over too short a view
                        @test rejects(f!, nan_dest(n - 1, Wₕ), uₕ)
                        @test rejects(f!, nan_dest(2n - 1, Vₕ), uv)

                        # larger: a grid function of a bigger mesh
                        @test rejects(f!, nan_dest(m, Bₕ), uₕ)
                        @test rejects(f!, nan_dest(2m, Bᵥ), uv)

                        # equal length, different mesh shape
                        # (bound to a name so the `nothing` test narrows it for JET)
                        Ωp = permuted[D]
                        if Ωp !== nothing
                            Pₕ, Pᵥ = gridspace(Ωp), gridspace(Ωp, Val(2))
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

    @testset "Vector-calculus grid mismatch rejected" begin
        # The same failure as above, one layer up (gpena/Bramble.jl#402): every in-place
        # divergence, curl, gradient, Laplacian and strain form indexed its destinations with
        # the source's grid shape and never asked whether they were grid functions of that
        # grid. Each destination leaf has to be refused -- a short view, a bigger mesh, a
        # permuted shape -- before anything is written to any of them, and a matching
        # destination must stay allocation-free.
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
        permuted = (
            nothing,
            mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 4), (true, false)),
            mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (3, 5, 4), (false, true, false))
        )
        fs = (x -> x^2 + 1, x -> x[1] + 2x[2]^2 + 1, x -> x[1] * x[2] + x[3] + 1)

        nan_dest(len, Wₕ) = VectorElement(view(fill(NaN, len + 31), 1:len), Wₕ)
        leaves(d) = d isa VectorElement ? [d] : reduce(vcat, map(leaves, collect(d)))

        # Throws an ArgumentError and leaves every destination's backing array as it was.
        function rejects(f!, dest, uₕ)
            raws = map(v -> parent(parent(v)), leaves(dest))
            before = map(copy, raws)
            thrown = try
                f!(dest, uₕ)
                false
            catch err
                err isa ArgumentError
            end
            return thrown && all(map(isequal, raws, before))
        end

        # Destination shape: () a grid function, (k,) a k-tuple, (k, k) a k-by-k tuple.
        shape(kind, D) = kind === :scalar ? () :
                         kind === :grad ? (D == 1 ? () : (D,)) :
                         kind === :curl ? (D == 2 ? () : (3,)) : (D, D)
        build(make, sh) = sh == () ? make() :
                          length(sh) == 1 ? ntuple(_ -> make(), sh[1]) :
                          ntuple(_ -> ntuple(_ -> make(), sh[2]), sh[1])
        slots(sh) = sh == () ? [()] :
                    length(sh) == 1 ? [(i,) for i in 1:sh[1]] :
                    [(i, j) for i in 1:sh[1] for j in 1:sh[2]]
        # `dest` with the leaf at `slot` replaced by `x`
        put(dest, ::Tuple{}, x) = x
        put(dest, s::Tuple, x) = ntuple(
            k -> k == s[1] ? put(dest[k], Base.tail(s), x) : dest[k], length(dest)
        )

        forms = (
            (Bramble.divₕ!, :scalar, :field, 1:3), (Bramble.div₊ₕ!, :scalar, :field, 1:3),
            (Bramble.divcₕ!, :scalar, :field, 1:3), (Bramble.diṽₕ!, :scalar, :field, 1:3),
            (Bramble.div̽ₕ!, :scalar, :field, 1:3), (Bramble.Δₕ!, :scalar, :scalar, 1:3),
            (Bramble.curlₕ!, :curl, :field, 2:3), (Bramble.curl₊ₕ!, :curl, :field, 2:3),
            (Bramble.curlcₕ!, :curl, :field, 2:3), (Bramble.curl̃ₕ!, :curl, :field, 2:3),
            (Bramble.curl̽ₕ!, :curl, :field, 2:3),
            (Bramble.∇cₕ!, :grad, :scalar, 1:3), (Bramble.∇̃ₕ!, :grad, :scalar, 1:3),
            (Bramble.∇̽ₕ!, :grad, :scalar, 1:3),
            (Bramble.εₕ!, :strain, :field, 1:3), (Bramble.εcₕ!, :strain, :field, 1:3),
            (Bramble.ε₊ₕ!, :strain, :field, 1:3), (Bramble.ε̽ₕ!, :strain, :field, 1:3)
        )

        # `unit` keeps the first form of each kind (divergence, curl, gradient, Laplacian,
        # strain); every form compiles its own rejection path, so `slow` runs them all.
        unit_forms = (Bramble.divₕ!, Bramble.curlₕ!, Bramble.∇cₕ!, Bramble.Δₕ!, Bramble.εₕ!)
        for (f!, kind, src, dims) in forms, D in dims

            WITH_SLOW_TESTS || f! in unit_forms || continue

            @testset "$f! $(D)D" begin
                Wₕ = gridspace(Ωs[D])
                uₕ = src === :scalar ? Rₕ(Wₕ, fs[D]) :
                     Rₕ(gridspace(Ωs[D], Val(D)), ntuple(_ -> fs[D], D))
                n = ndofs(Wₕ)
                good() = nan_dest(n, Wₕ)
                wrongs = Tuple{Int, Bramble.AbstractSpaceType}[
                    (n - 1, Wₕ), (ndofs(gridspace(bigger[D])), gridspace(bigger[D]))]
                Ωp = permuted[D]  # bound to a name so the `nothing` test narrows it for JET
                if Ωp !== nothing
                    Pₕ = gridspace(Ωp)
                    @test ndofs(Pₕ) == n
                    push!(wrongs, (n, Pₕ))
                end
                sh = shape(kind, D)
                for s in slots(sh), (len, Xₕ) in wrongs

                    @test rejects(f!, put(build(good, sh), s, nan_dest(len, Xₕ)), uₕ)
                end
                @test alloc_test(f!, build(good, sh), uₕ) == 0
            end
        end

        # `Δₕ!` also takes composite elements, whose totals agree on a permuted grid, so
        # each destination leaf is checked against its source leaf.
        @testset "Δₕ! composite" begin
            Vₕ, Pᵥ = gridspace(Ωs[3], Val(2)), gridspace(permuted[3], Val(2))
            Bᵥ = gridspace(bigger[3], Val(2))
            uv = Rₕ(Vₕ, (fs[3], fs[3]))
            n = ndofs(Vₕ)
            @test ndofs(Pᵥ) == n
            @test rejects(Bramble.Δₕ!, nan_dest(n, Pᵥ), uv)
            @test rejects(Bramble.Δₕ!, nan_dest(n - 1, Vₕ), uv)
            @test rejects(Bramble.Δₕ!, nan_dest(ndofs(Bᵥ), Bᵥ), uv)
            @test alloc_test(Bramble.Δₕ!, nan_dest(n, Vₕ), uv) == 0
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
