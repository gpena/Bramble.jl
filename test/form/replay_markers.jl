module FormReplayMarkersTests

using Test
using Bramble
using SparseArrays
using Bramble: CpuSerial, CpuThreaded, backend, assemble_add!, assemble_parallel!,
               allocate_system_matrix, restrict_to, D₋ₓ, set_points!, _marker_stamp,
               _pattern_stamp
using ..TestUtils: _fillnz!

# A refill after `markers!` never replays stale positions (gpena/Bramble.jl#465). A
# restricted term's walk reads the mesh markers, so the `nzval` positions a recording holds
# depend on them: the replay cache keys on the meshes' marker stamps as well as on `A` and
# the AST. After `markers!` the next fill re-records against the same `A`: a pattern that
# shrank refills to what a fresh `assemble` gives, one that grew throws the sparsity-pattern
# `ArgumentError`, and a label deleted since throws the unbound-label `ArgumentError`. Every
# oracle is a fresh `assemble` of a new form after the change, on a non-uniform mesh.

const _S = interval(0.0, 1.0) × interval(0.0, 1.0)
_blob(x) = (x[1] - 0.4)^2 + (x[2] - 0.5)^2 < 0.1
_blob1(x) = 0.3 < x[1] < 0.7

function _mesh2(n, pol; unif = (false, false))
    mesh(
        domain(_S, :top => :top, :blob => _blob), n, unif; backend = backend(policy = pol))
end
_mesh1(n, pol) = mesh(
    domain(interval(0.0, 1.0), :blob => _blob1), n, false; backend = backend(policy = pol))

_term(u, v) = innerₕ(D₋ₓ(restrict_to(:blob, u)), v)
_two(u, v) = _term(u(1), v(1)) + _term(u(2), v(2))

# The space, the form and the mesh `markers!` acts on. `:cross` is a composite over two
# meshes of different sizes, re-marking only the second.
function _setup(pol, case)
    Ω = _mesh2((23, 7), pol)
    case === :scalar && (W = gridspace(Ω); return W, form(W, W, _term), Ω)
    case === :pair && (W = gridspace(Ω) × gridspace(Ω); return W, form(W, W, _two), Ω)
    Ω2 = _mesh2((17, 9), pol)
    W = gridspace(Ω) × gridspace(Ω2)
    return W, form(W, W, _two), Ω2
end

# Replaces `:blob` on `Ω`: every other point of it, its complement, or no label at all.
function _remark!(Ω, change)
    m = copy(markers(Ω))
    b = m[:blob]
    if change === :shrink
        m[:blob] = b .& isodd.(eachindex(b))
    elseif change === :grow
        m[:blob] = .!b
    else
        delete!(m, :blob)
    end
    Bramble.markers!(Ω, m)
    return nothing
end

function _refill!(A, a, entry)
    entry === :assemble! && return assemble!(A, a)
    entry === :assemble_parallel! && return assemble_parallel!(A, a)
    _fillnz!(A, 0.0)
    return assemble_add!(A, a)
end

_pattern_error(e) = e isa ArgumentError && occursin("sparsity pattern", sprint(showerror, e))

function _throws_pattern(A, a, entry)
    try
        _refill!(A, a, entry)
    catch e
        return _pattern_error(e)
    end
    return false
end

# A shrink refills to a fresh `assemble`, then a grow throws, on the one mesh `Ω`. A function
# of its own, so each mesh's dimension stays concrete rather than a loop's `Union`.
function _shrink_then_grow(Ω)
    W = gridspace(Ω)
    a = form(W, W, _term)
    A = assemble(a)
    assemble!(A, a)
    _remark!(Ω, :shrink)
    fresh = assemble(form(W, W, _term))
    _fillnz!(A, NaN)
    assemble!(A, a)
    @test isapprox(A, fresh; rtol = 1e-12)
    _remark!(Ω, :grow)
    @test _throws_pattern(A, a, :assemble!)
end

@testset "Refill after markers! (#465)" begin
    @testset "$pol, $case, $path, $entry" for pol in (CpuSerial(), CpuThreaded()),
        case in (:scalar, :pair, :cross), path in (:assemble, :allocate),
        entry in (:assemble!, :assemble_add!, :assemble_parallel!)
        @testset "shrink" begin
            W, a, Ω = _setup(pol, case)
            A = path === :assemble ? assemble(a) : allocate_system_matrix(a)
            assemble!(A, a)
            before = copy(A)
            stamp = a.cache.stamp
            _remark!(Ω, :shrink)
            @test _pattern_stamp(W, W) > stamp
            fresh = assemble(form(W, W, case === :scalar ? _term : _two))
            @test fresh != before
            _fillnz!(A, NaN)
            _refill!(A, a, entry)
            @test isapprox(A, fresh; rtol = 1e-12)
            @test a.cache.stamp == _pattern_stamp(W, W)
        end

        @testset "grow" begin
            _, a, Ω = _setup(pol, case)
            A = path === :assemble ? assemble(a) : allocate_system_matrix(a)
            assemble!(A, a)
            _remark!(Ω, :grow)
            @test _throws_pattern(A, a, entry)
        end

        @testset "label deleted" begin
            _, a, Ω = _setup(pol, case)
            A = path === :assemble ? assemble(a) : allocate_system_matrix(a)
            assemble!(A, a)
            _remark!(Ω, :delete)
            @test_throws ArgumentError _refill!(A, a, entry)
        end
    end

    @testset "1D and uniform meshes, $pol" for pol in (CpuSerial(), CpuThreaded())
        _shrink_then_grow(_mesh1(31, pol))
        _shrink_then_grow(_mesh2((15, 11), pol; unif = (true, true)))
    end
end

_allocs(A, a) = @allocated assemble!(A, a)

@testset "Refill allocations around markers!" begin
    Ω = _mesh2((23, 7), CpuSerial())
    W = gridspace(Ω)
    a = form(W, W, _term)
    A = assemble(a)
    assemble!(A, a)
    _allocs(A, a)
    @test _allocs(A, a) == 0
    segments = a.cache.segments
    _remark!(Ω, :shrink)
    # One re-record: it allocates a new recording, and the refills after it replay that one.
    @test _allocs(A, a) > 0
    @test a.cache.segments !== segments
    segments = a.cache.segments
    @test _allocs(A, a) == 0
    @test a.cache.segments === segments
    @test isapprox(A, assemble(form(W, W, _term)); rtol = 1e-12)
end

@testset "set_points! keeps the stamp" begin
    Ω = _mesh1(31, CpuSerial())
    W = gridspace(Ω)
    a = form(W, W, _term)
    assemble(a)
    stamp = _marker_stamp(Ω)
    @test a.cache.stamp == stamp == _pattern_stamp(W, W)
    p = collect(points(Ω))
    p[2:(end - 1)] .= (p[2:(end - 1)] .+ p[1:(end - 2)]) ./ 2
    set_points!(Ω, p)
    @test points(Ω) == p
    @test _marker_stamp(Ω) == stamp
    @test a.cache.stamp == _pattern_stamp(W, W)
    _remark!(Ω, :shrink)
    @test _marker_stamp(Ω) > stamp
end

end
