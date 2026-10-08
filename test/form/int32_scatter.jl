module FormInt32ScatterTests

using Test
using Bramble
using Random
using SparseArrays
using SparseArrays: getcolptr
using Bramble: CpuSerial, CpuThreaded, CpuPolyester, backend, assemble_add!,
               assemble_parallel!, allocate_system_matrix, D₋ₓ, πₕ, inner₊ₓ

# An `Int32` sparse target assembles to the same bits as an `Int` one (gpena/Bramble.jl#469).
# The searching sweep's position search answers an `Int` for any index type, so the write
# goes to `nzval[pos]` and never to the dense fallback `A[pos] += val`, which on a sparse
# matrix reads `pos` as a linear index. Each `Int32` fill is checked against the `Int` fill
# through the same entry point under the same policy, bitwise, never against a serial fill:
# a threaded composite refill differs from a serial one by round-off, for `Int` too.

const _SQ = domain(interval(0.0, 1.0) × interval(0.0, 1.0))
const _LINE = domain(interval(0.0, 1.0))

# The same random non-uniform mesh under any policy: the seed fixes the points.
function _mesh2(pol; seed = 469)
    Random.seed!(seed)
    return mesh(_SQ, (9, 9), (false, false); backend = backend(policy = pol))
end
function _mesh1(n, pol, seed)
    Random.seed!(seed)
    return mesh(_LINE, n, false; backend = backend(policy = pol))
end

_scalar(u, v) = innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))
_composite(u, v) = innerₕ(u(1), v(1)) + inner₊(∇ₕ(u(2)), ∇ₕ(v(2))) + innerₕ(D₋ₓ(u(1)), v(2))
_interp(u, v) = innerₕ(u, πₕ(v)) + inner₊ₓ(D₋ₓ(u), D₋ₓ(πₕ(v)))

function _form(case, pol)
    if case === :scalar
        W = gridspace(_mesh2(pol))
        return form(W, W, _scalar)
    elseif case === :composite
        W = gridspace(_mesh2(pol), Val(2))
        return form(W, W, _composite)
    end
    return form(gridspace(_mesh1(9, pol, 3)), gridspace(_mesh1(6, pol, 4)), _interp)
end

# `allocate_system_matrix` infers a union that includes a dense `Matrix`, which has no
# `nonzeros`; these matrices are always `SparseMatrixCSC`, so the assertion narrows the type.
function _csc(A)
    @assert A isa SparseMatrixCSC{Float64, Int}
    return A
end

function _refill!(A, a, entry)
    entry === :assemble! && return assemble!(A, a)
    entry === :assemble_parallel! && return assemble_parallel!(A, a)
    fill!(nonzeros(A), 0.0)
    entry === :assemble_add! && return assemble_add!(A, a)
    return Bramble._assemble_bilinear_parallel_core!(A, a.trial_space, a.test_space, a.ast)
end

function _bitwise(A32, A64)
    getcolptr(A32) == getcolptr(A64) && rowvals(A32) == rowvals(A64) &&
        isequal(nonzeros(A32), nonzeros(A64))
end

const _ENTRIES = (:assemble!, :assemble_parallel!, :assemble_add!, :core)

@testset "Int32 sparse target" begin
    @testset "the position search answers Int" begin
        A64 = _csc(allocate_system_matrix(_form(:scalar, CpuSerial())))
        A32 = SparseMatrixCSC{Float64, Int32}(A64)
        # The `Int32` answer agrees with the `Int` one in value, so only its type tells.
        @test @inferred(Bramble._scatter_position(A32, 1, 1)) === 1
        @test Bramble._scatter_position(A32, 9, 9) === Bramble._scatter_position(A64, 9, 9)
        @test Bramble._scatter_position(A32, 1, 81) === 0
        S = Bramble._ScatterCSC(getcolptr(A32), rowvals(A32), nonzeros(A32))
        @test @inferred(Bramble._scatter_position(S, 1, 1)) === 1
    end

    # Without `BramblePolyesterExt`, `CpuPolyester` can only run the one-thread fallback of
    # a test-side interpolation; the scalar and composite forms stop at a hook naming
    # Polyester (as in threaded_replay.jl).
    has_polyester = Base.get_extension(Bramble, :BramblePolyesterExt) !== nothing
    @testset "$(nameof(typeof(pol))), $case, $entry" for pol in (
            CpuSerial(), CpuThreaded(), CpuPolyester()),
        case in (:scalar, :composite, :interp),
        entry in _ENTRIES
        pol isa CpuPolyester && case !== :interp && !has_polyester && continue
        a = _form(case, pol)
        A64 = _csc(allocate_system_matrix(a))
        A32 = SparseMatrixCSC{Float64, Int32}(A64)
        # Twice: the first fill records (or searches), the second replays a recording.
        for _ in 1:2
            fill!(nonzeros(A64), NaN)
            fill!(nonzeros(A32), NaN)
            _refill!(A64, a, entry)
            _refill!(A32, a, entry)
            @test _bitwise(A32, A64)
            @test !any(isnan, nonzeros(A32))
        end
    end
end

end # module
