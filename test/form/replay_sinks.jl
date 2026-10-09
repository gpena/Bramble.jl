module FormReplaySinksTests

using Test
using Bramble
using Random
using SparseArrays
using Bramble: Serial, Parallel, backend, assemble_parallel!, D₋ₓ
using ..TestUtils: _fillnz!, alloc_test

# The replay sinks hold the matrix's storage (`_scatter_storage`) and their recorded
# positions as any `AbstractVector{Int}` (gpena/Bramble.jl#437): a sparse sink is plain
# vectors, so a threaded sweep can hand it across a task boundary. Each sink must still
# write exactly where `_scatter_add!` would: checked entry by entry on a hand-built matrix,
# on a sparse type whose positions are linear indices, without allocating into a dense
# matrix, and by a threaded refill against a serial `assemble` on a non-uniform mesh.

function _mesh(D, n, policy)
    Random.seed!(437)
    doms = (domain(interval(0.0, 1.0)), domain(interval(0.0, 1.0) × interval(0.0, 2.0)))
    return mesh(doms[D], ntuple(_ -> n, D), ntuple(_ -> false, D);
        backend = backend(policy = policy))
end

_space(Ω, c) = c == 1 ? gridspace(Ω) : gridspace(Ω, Val(c))
_scalar(u, v) = innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(D₋ₓ(u), v)

_hasmatrix(T) = any(t -> t isa Type && t <: AbstractMatrix, T.parameters)

@testset "Replay sinks hold storage (#437)" begin
    # A 3×3 tridiagonal pattern: 7 stored entries, column-major.
    A = sparse([1, 2, 1, 2, 3, 2, 3], [1, 1, 2, 2, 2, 3, 3], collect(1.0:7.0), 3, 3)
    nz = nonzeros(A)
    ptr, pos, pos_t = [1, 2], [4, 6], [6, 3]

    @testset "Sinks carry no matrix" begin
        sinks = (Bramble.ReplaySink(A, ptr, pos, 2.0),
            Bramble._PairReplaySink(A, ptr, pos, pos_t, 2.0, 3.0, 0),
            Bramble._StrideReplaySink(A, [1], [2], 1, 2.0))
        for s in sinks
            @test s.nzval === nz
            @test !_hasmatrix(typeof(s))
            @test !any(T -> T <: AbstractMatrix, fieldtypes(typeof(s)))
        end
    end

    @testset "Entries land where recorded" begin
        B = copy(A)
        s = Bramble.ReplaySink(B, ptr, pos, 2.0)
        Bramble._sink_entry!(s, 0, 0, 0.5, 2)              # nzval[6] += 2.0 * 0.5
        @test nonzeros(B) == [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 7.0]

        B = copy(A)
        p = Bramble._PairReplaySink(B, ptr, pos, pos_t, 2.0, 3.0, 0)
        Bramble._sink_entry!(p, 0, 0, 1.0, 1)              # nzval[4] += 2, nzval[6] += 3
        @test nonzeros(B) == [1.0, 2.0, 3.0, 6.0, 5.0, 9.0, 7.0]
        B = copy(A)
        Bramble._sink_entry!(Bramble._PairReplaySink(B, ptr, pos, pos_t, 2.0, 3.0, 2),
            0, 0, 1.0, 1)                                  # transposed half only
        @test nonzeros(B) == [1.0, 2.0, 3.0, 4.0, 5.0, 9.0, 7.0]

        B = copy(A)
        st = Bramble._StrideReplaySink(B, [1, 2], [3, 2], 1, 2.0)
        Bramble._sink_entry!(st, 0, 0, 1.0, 1)             # base[2] + stride[2] * 1 = 4
        @test nonzeros(B) == [1.0, 2.0, 3.0, 6.0, 5.0, 6.0, 7.0]
    end

    @testset "Positions need not be a Vector" begin
        # A view of the recording: `P` is array-parametric, and writes go to the same slots.
        B = copy(A)
        backing = [0, 1, 2, 4, 6]
        s = Bramble.ReplaySink(B, view(backing, 2:3), view(backing, 4:5), 1.0)
        @test s.positions isa SubArray
        Bramble._sink_entry!(s, 0, 0, 1.5, Bramble._sink_point!(s, 1, CartesianIndex(1)))
        @test nonzeros(B) == [1.0, 2.0, 3.0, 5.5, 5.0, 6.0, 7.0]
    end

    @testset "Dense storage is the matrix" begin
        M = zeros(3, 3)
        s = Bramble.ReplaySink(M, [1], [LinearIndices(M)[2, 3]], 2.0)
        @test s.nzval === M
        Bramble._sink_entry!(s, 0, 0, 1.25, 1)
        @test M[2, 3] == 2.5
        @test count(!iszero, M) == 1
    end

    @testset "Linear-index sparse replays" begin
        # `FixedSparseCSC` has no `nzval` position search of its own: its positions are
        # linear indices, so its storage must be the matrix, never `nonzeros(A)` (whose
        # length is only `nnz`). Recorded, then replayed, against a plain CSC fill.
        for p in (Serial(), Parallel())
            Ω = _mesh(2, 6, p)
            a = form(gridspace(Ω), gridspace(Ω), _scalar)
            R = assemble(a)
            F = SparseArrays.fixed(copy(R))
            for _ in 1:2
                assemble!(F, a)
                @test a.cache.valid && a.cache.A_id == objectid(F)
                @test nonzeros(F) == nonzeros(R)
            end
        end
    end

    @testset "Dense serial refill allocates 0 B" begin
        Ω = _mesh(2, 20, Serial())
        a = form(gridspace(Ω), gridspace(Ω), _scalar)
        R = assemble(a)
        M = Matrix(R)
        assemble!(M, a)                                    # records into `M`
        @test a.cache.A_id == objectid(M)
        @test alloc_test(assemble!, M, a) == 0
        @test M == Matrix(R)
    end

    @testset "Threaded refill matches serial" begin
        # The 2D threaded refills are threaded_replay.jl's; its 1D cases run under slow only.
        for (nm, D, n, c, f) in (("1D scalar", 1, 41, 1, _scalar),)
            R = assemble(form(_space(_mesh(D, n, Serial()), c),
                _space(_mesh(D, n, Serial()), c), f))
            Ω = _mesh(D, n, Parallel())
            a = form(_space(Ω, c), _space(Ω, c), f)
            for refill! in (assemble!, assemble_parallel!)
                A = assemble(a)
                _fillnz!(A, NaN)
                refill!(A, a)
                @test a.cache.valid && a.cache.A_id == objectid(A)
                @test rowvals(A) == rowvals(R)
                @test isapprox(A, R; rtol = 1e-12)
            end
        end
    end
end

end # module
