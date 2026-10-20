module FormRepeatedAssembleTests

using Test
using Bramble
using Random
using SparseArrays
using SparseArrays: getcolptr
using Bramble: D₋ₓ, D̃ₓ, D₋ᵧ, D̃ᵧ, jumpₓ, restrict_to
using ..TestUtils: _fillnz!, alloc_test

# A repeated `assemble(a)` builds its matrix from a copy of the pattern the form recorded and
# replays the recording into it, instead of recording again (gpena/Bramble.jl#774). Every
# oracle is the first `assemble` of a new form over the same space, which records from
# scratch; meshes are non-uniform. A kept pattern must not outlive what it was recorded
# against: moved points change values only, re-marked meshes change the pattern, and a
# fill into another matrix replaces the recording.

const _DOMAINS = (domain(interval(0.0, 1.0)), domain(interval(0.0, 1.0) × interval(0.0, 1.0)))

function _mesh(D, n; seed = 774)
    Random.seed!(seed)
    return mesh(_DOMAINS[D], n, ntuple(_ -> false, D))
end

# The discrete Laplacian as nested differences, per dimension.
_lap(u, ::Val{1}) = D̃ₓ(D₋ₓ(u))
_lap(u, ::Val{2}) = D̃ₓ(D₋ₓ(u)) + D̃ᵧ(D₋ᵧ(u))

# The two forms of #774: a coefficient on one difference, and on a jump of the Laplacian.
function _forms(W, ::Val{D}) where {D}
    c = Rₕ(W, x -> 1.0 + 2.0 * first(x)^2)
    return (
        (u, v) -> innerₕ(c * D₋ₓ(u), D₋ₓ(v)),
        (u, v) -> innerₕ(c * jumpₓ(_lap(u, Val(D))), jumpₓ(_lap(v, Val(D))))
    )
end

_same_structure(A, B) = getcolptr(A) == getcolptr(B) && rowvals(A) == rowvals(B)
_agrees(A, R) = _same_structure(A, R) && isapprox(A, R; rtol = 1e-12)

# Bytes of the matrix `assemble` returns.
_bytes(A) = sizeof(nonzeros(A)) + sizeof(rowvals(A)) + sizeof(getcolptr(A))

# A function barrier, so `@allocated` sees only the call.
_assemble_bytes(a) = @allocated assemble(a)

const _SIZES = (((15,), (61,)), ((9, 11), (17, 21)))

@testset "Repeated assemble (#774)" begin
    @testset "$(D)D: equals a fresh form" for D in (1, 2)
        W = gridspace(_mesh(D, first(_SIZES[D])))
        for f in _forms(W, Val(D))
            a = form(W, W, f)
            A = assemble(a)
            R = assemble(form(W, W, f))
            @test _agrees(A, R)
            for _ in 1:3
                B = assemble(a)
                @test a.cache.has_pattern && a.cache.A_id == objectid(B)
                @test _agrees(B, R)
            end
        end
    end

    @testset "$(D)D: repeated assemble with dirichlet" for D in (1, 2)
        W = gridspace(_mesh(D, first(_SIZES[D])))
        for f in _forms(W, Val(D))
            a = form(W, W, f)
            R = assemble(form(W, W, f); dirichlet = :boundary)
            A = assemble(a; dirichlet = :boundary)
            @test _agrees(A, R)
            for _ in 1:3
                @test _agrees(assemble(a; dirichlet = :boundary), R)
            end
        end
    end

    @testset "$(D)D: matrices do not alias" for D in (1, 2)
        W = gridspace(_mesh(D, first(_SIZES[D])))
        f = last(_forms(W, Val(D)))
        a = form(W, W, f)
        A = assemble(a)
        R = copy(A)
        B = assemble(a)
        @test nonzeros(B) !== nonzeros(A) && rowvals(B) !== rowvals(A)
        @test getcolptr(B) !== getcolptr(A)
        _fillnz!(B, NaN)
        rowvals(B) .= 1
        getcolptr(B)[end] = 0
        @test _agrees(A, R)
        @test _agrees(assemble(a), R)
    end

    @testset "$(D)D: allocation tracks the matrix" for D in (1, 2)
        for n in _SIZES[D]
            W = gridspace(_mesh(D, n))
            for f in _forms(W, Val(D))
                a = form(W, W, f)
                A = assemble(a)
                assemble(a)
                b = _assemble_bytes(a)
                @test b <= 2 * _bytes(A) + 4096
                @test alloc_test(assemble!, A, a) == 0
            end
        end
    end

    @testset "$(D)D: moved points" for D in (1, 2)
        Ω = _mesh(D, first(_SIZES[D]))
        W = gridspace(Ω)
        a = form(W, W, last(_forms(W, Val(D))))
        A = assemble(a)
        assemble(a)
        moved = map(n -> collect(range(0.0, 1.0; length = n)) .^ 2, first(_SIZES[D]))
        Bramble.change_points!(Ω, D == 1 ? only(moved) : moved)
        # A form on the old space refuses, kept pattern or not.
        @test_throws ArgumentError assemble(a)
        W2 = gridspace(Ω)
        f = last(_forms(W2, Val(D)))
        a2 = form(W2, W2, f)
        assemble(a2)
        B = assemble(a2)
        @test _agrees(B, assemble(form(W2, W2, f)))
        @test !isapprox(B, A; rtol = 1e-6)
    end

    @testset "re-marked mesh re-records" begin
        blob(x) = (x[1] - 0.4)^2 + (x[2] - 0.5)^2 < 0.1
        Random.seed!(774)
        Ω = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0), :blob => blob), (23, 7),
            (false, false))
        W = gridspace(Ω)
        f(u, v) = innerₕ(D₋ₓ(restrict_to(:blob, u)), v)
        a = form(W, W, f)
        A = assemble(a)
        assemble(a)
        m = copy(markers(Ω))
        m[:blob] = m[:blob] .& isodd.(eachindex(m[:blob]))
        Bramble.markers!(Ω, m)
        B = assemble(a)
        R = assemble(form(W, W, f))
        @test nnz(R) < nnz(A)
        @test _agrees(B, R)
        @test _agrees(assemble(a), R)
    end

    @testset "$(D)D: other matrix's fill" for D in (1, 2)
        W = gridspace(_mesh(D, first(_SIZES[D])))
        f = first(_forms(W, Val(D)))
        a = form(W, W, f)
        A = assemble(a)
        assemble(a)
        n = size(A, 1)
        # A wider pattern: the recording made into it indexes its own layout.
        C = A + sparse(1:n, n:-1:1, ones(n))
        @test nnz(C) > nnz(A)
        _fillnz!(C, NaN)
        assemble!(C, a)
        @test !a.cache.has_pattern
        B = assemble(a)
        @test _agrees(B, A)
        @test _agrees(assemble(a), A)
    end
end

end # module FormRepeatedAssembleTests
