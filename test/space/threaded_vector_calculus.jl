module SpaceThreadedVectorCalculusTests

using Test
using Bramble
using Random
using ..TestUtils: alloc_test, WITH_SLOW_TESTS, _reset_seen!, _spy, _threads_seen

# Under a `CpuThreaded` (`Parallel()`) backend the in-place vector-calculus operators -- the
# gradients, divergences, curls and strain tensors of vector_calculus.jl -- reach the banded
# stencil engines and the banded accumulating engines. Every point is
# still computed by the loop body the serial path runs, summed over the directions in the
# same order, so the answer must equal the `Serial()` one exactly (`==`), not merely to a
# tolerance. The meshes are non-uniform: on a uniform mesh a band that picked up the wrong
# spacing index would still give the right number.

const _GRADIENTS = (:∇̃ₕ!, :∇cₕ!, :∇̽ₕ!)
const _DIVERGENCES = (:divₕ!, :div₊ₕ!, :divcₕ!, :diṽₕ!, :div̽ₕ!)
const _CURLS = (:curlₕ!, :curl₊ₕ!, :curlcₕ!, :curl̃ₕ!, :curl̽ₕ!)
const _STRAINS = (:εₕ!, :ε₊ₕ!, :εcₕ!, :ε̽ₕ!)

# The operators whose stencil reaches both neighbours need three points per direction.
const _CENTERED = (:∇cₕ!, :∇̽ₕ!, :divcₕ!, :div̽ₕ!, :curlcₕ!, :curl̽ₕ!, :εcₕ!, :ε̽ₕ!)

_op(name) = getproperty(Bramble, name)

# `unit` keeps, per group, the first and last operator (the one-sided and the centered
# engine) in 2D; in 3D, where the operators cost the most to compile, the gradients' pair
# and the first divergence and curl, the strain tensors being left to 2D. 1D and `slow`
# keep them all.
function _ops(D, names)
    (WITH_SLOW_TESTS || D == 1) && return names
    D == 2 && return (first(names), last(names))
    names === _GRADIENTS && return (first(names), last(names))
    names === _STRAINS && return ()
    return (first(names),)
end

function _domain(D)
    D == 1 ? domain(interval(0.0, 1.0)) :
    D == 2 ? domain(interval(0.0, 1.0) × interval(0.0, 1.0)) :
    domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
end

# The same non-uniform mesh once per policy: the seed fixes the random points.
function _mesh(n::NTuple{D, Int}, policy; seed = 3564) where {D}
    npts = D == 1 ? n[1] : n
    unif = D == 1 ? false : ntuple(_ -> false, D)
    Random.seed!(seed)
    return mesh(_domain(D), npts, unif; backend = backend(policy = policy))
end

# One scalar function and one `D`-component field per dimension, the field's components
# distinct so a swapped component would show.
const _F = (x -> sin(3x) + x^2, x -> sin(3x[1] + 2x[2]) + x[1] * x[2],
    x -> sin(3x[1] + 2x[2] - x[3]) + x[1] * x[3])
_field(D) = ntuple(d -> (x -> sum(x) * d + sin(d * x[d]) + x[d]^2), D)

# Everything one policy produces, as a flat list of storage vectors; destinations start at
# NaN so an unwritten point cannot pass.
function _results(n::NTuple{D, Int}, policy; composite = true) where {D}
    Ωₕ = _mesh(n, policy)
    Wₕ = gridspace(Ωₕ)
    uₕ = Rₕ(Wₕ, _F[D])
    fs = _field(D)
    tup = ntuple(d -> Rₕ(Wₕ, x -> fs[d](D == 1 ? (x,) : x)), D)
    comp = Rₕ(gridspace(Ωₕ, Val(D)), D == 1 ? (x -> fs[1]((x,)),) : fs)
    fresh() = (v = similar(uₕ); parent(v) .= NaN; v)
    centered_ok = all(>=(3), n)
    out = Dict{String, Vector{Vector{Float64}}}()
    ops(names) = _ops(D, names)

    for name in ops(_GRADIENTS)
        name in _CENTERED && !centered_ok && continue
        dest = D == 1 ? fresh() : ntuple(_ -> fresh(), D)
        @test _op(name)(dest, uₕ) === dest
        out["$name"] = [copy(parent(v)) for v in (dest isa Tuple ? dest : (dest,))]
    end

    fields = composite ? (("tuple", tup), ("composite", comp)) : (("tuple", tup),)

    for name in ops(_DIVERGENCES), (label, field) in fields

        name in _CENTERED && !centered_ok && continue
        v = fresh()
        @test _op(name)(v, field) === v
        out["$name $label"] = [copy(parent(v))]
    end

    if D >= 2
        for name in ops(_CURLS), (label, field) in fields

            name in _CENTERED && !centered_ok && continue
            dest = D == 2 ? fresh() : ntuple(_ -> fresh(), 3)
            @test _op(name)(dest, field) === dest
            out["$name $label"] = [copy(parent(v)) for v in (dest isa Tuple ? dest : (dest,))]
        end
    end

    for name in ops(_STRAINS), (label, field) in fields

        name in _CENTERED && !centered_ok && continue
        dest = ntuple(_ -> ntuple(_ -> fresh(), D), D)
        @test _op(name)(dest, field) === dest
        out["$name $label"] = [copy(parent(dest[i][j])) for i in 1:D for j in 1:D]
    end
    return out
end

function _check_equal(n, policy; composite = true)
    serial = _results(n, Serial(); composite)
    other = _results(n, policy; composite)
    @test keys(serial) == keys(other)
    for (k, v) in serial
        @testset "$k n=$n" begin
            @test other[k] == v
        end
    end
end

# Grid sizes: banded axes shorter than the thread count, down to a single point, leave some
# bands empty; the operator must not notice. Centered operators skip sizes below 3.
const _SIZES = (((1,), (2,), (3,), (5,), (401,)),
    ((9, 1), (9, 2), (3, 3), (11, 7), (40, 37)),
    ((5, 4, 1), (5, 4, 2), (4, 3, 3), (9, 8, 13)))

# What `unit` keeps of `_SIZES`: per dimension, the degenerate last axis (2 points, empty
# bands) and the smallest size the centered operators accept (3 points along every axis,
# still fewer than the band count). The full sweep runs in `slow`.
const _UNIT_SIZES = (((2,), (3,)), ((9, 2), (3, 3)), ((5, 4, 2), (4, 3, 3)))

@testset "Threaded vector calculus" begin
    @testset "CpuThreaded equal to Serial, $(D)D" for D in 1:3
        # The composite field dispatches the same banded engines whatever the dimension, so
        # `unit` runs it in 1D and 2D and `slow` adds 3D, where it costs the most to compile.
        composite = WITH_SLOW_TESTS || D < 3
        foreach(n -> _check_equal(n, Parallel(); composite),
            WITH_SLOW_TESTS ? _SIZES[D] : _UNIT_SIZES[D])
    end

    # Silent on a single thread: there is nothing to band across.
    if Threads.nthreads() >= 2
        @testset "Runs on several threads" for D in 2:3
            begin
                n = D == 2 ? (64, 64) : (12, 12, 12)
                Ωₕ = _mesh(n, Parallel())
                Wₕ = gridspace(Ωₕ)
                fs = _field(D)
                tup = ntuple(d -> Rₕ(Wₕ, fs[d]), D)
                spies = map(_spy, tup)
                uₕ = Rₕ(Wₕ, _F[D])
                for name in _ops(D, _GRADIENTS)
                    dest = ntuple(_ -> similar(uₕ), D)
                    _reset_seen!()
                    _op(name)(dest, _spy(uₕ))
                    @test _threads_seen() >= 2
                end
                for name in _ops(D, _DIVERGENCES)
                    _reset_seen!()
                    _op(name)(similar(uₕ), spies)
                    @test _threads_seen() >= 2
                end
                for name in _ops(D, _CURLS)
                    dest = D == 2 ? similar(uₕ) : ntuple(_ -> similar(uₕ), 3)
                    _reset_seen!()
                    _op(name)(dest, spies)
                    @test _threads_seen() >= 2
                end
                for name in _ops(D, _STRAINS)
                    dest = ntuple(_ -> ntuple(_ -> similar(uₕ), D), D)
                    _reset_seen!()
                    _op(name)(dest, spies)
                    @test _threads_seen() >= 2
                end
            end
        end
    end

    @testset "In-place allocation is size-independent" begin
        function min_bytes(n)
            Ωₕ = _mesh((n, n), Parallel())
            Wₕ = gridspace(Ωₕ)
            tup = ntuple(d -> Rₕ(Wₕ, _field(2)[d]), 2)
            v = similar(first(tup))
            dest = ntuple(_ -> ntuple(_ -> similar(v), 2), 2)
            return (minimum(alloc_test(_op(:divₕ!), v, tup) for _ in 1:5),
                minimum(alloc_test(_op(:curlcₕ!), v, tup) for _ in 1:5),
                minimum(alloc_test(_op(:εₕ!), dest, tup) for _ in 1:5))
        end
        @test min_bytes(16) == min_bytes(160)
    end
end

end
