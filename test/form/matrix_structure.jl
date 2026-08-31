using Test
using Bramble
using Random
using SparseArrays
using Bramble: IdentityOperator, ZeroOperator, TrialFunction, TestFunction,
               IndexedTrialFunction, SourceVector, LazyOp,
               stencil_offsets, matrix_structure, local_stencil,
               MatrixStructure, DiagonalStructure, TridiagonalStructure, SparseStructure,
               shift_op, restrict_to, source_function

# Reading the sparsity pattern off an AST before assembling it.
#
# Every node reaches a fixed set of neighbours, and that set is a property of the tree
# rather than of the grid point: truncation at a boundary zeroes the coefficients and keeps
# the offsets. So the pattern — and with it the narrowest matrix type that can hold the
# operator — is known before a single entry is computed.
#
# The tests below check the prediction against two independent things: the offsets
# `local_stencil` actually produces, and the diagonals the assembled matrix actually
# occupies. A stencil offset `o` means row `i` carries an entry in column `i + o`, so the
# diagonal index to compare against is `j - i`.

# the diagonals an assembled matrix actually occupies, in the stencil's own convention
function _matrix_offsets(M)
    sort(unique(j - i for j in axes(M, 2) for i in axes(M, 1)
    if M[i, j] != 0))
end

# the offsets a stencil actually produces at one point
function _stencil_at(node, Wₕ, I, lin)
    sort(unique(first(e)
    for e in local_stencil(node, Wₕ, I, nothing, lin[I])))
end

@testset "Matrix structure from the stencil" begin
    Random.seed!(20260831)
    Ωₕ1 = mesh(domain(interval(0.0, 1.0)), 9, false)
    Ωₕ2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 6), (true, false))
    Wₕ1, Wₕ2 = gridspace(Ωₕ1), gridspace(Ωₕ2)
    id1, id2 = IdentityOperator(Wₕ1), IdentityOperator(Wₕ2)
    lin1 = LinearIndices(Bramble.indices(Ωₕ1))
    lin2 = LinearIndices(Bramble.indices(Ωₕ2))

    @testset "the prediction matches the stencil it predicts" begin
        @testset "1D" begin
            I = CartesianIndex(5)
            for (nm, node) in (("identity", id1), ("D₋ₓ", D₋ₓ(id1)), ("D₊ₓ", D₊ₓ(id1)),
                ("diff₋ₓ", D₋ₓ(id1)), ("M₋ₓ", M₋ₓ(id1)), ("M₊ₓ", M₊ₓ(id1)),
                ("jumpₓ", jumpₓ(id1)), ("Dcₓ", Dcₓ(id1)),
                ("Dstar₊ₓ", Dstar₊ₓ(id1)), ("Dₕₓ", Dₕₓ(id1)))
                @testset "$nm" begin
                    @test sort(stencil_offsets(node)) == _stencil_at(node, Wₕ1, I, lin1)
                end
            end
        end

        @testset "2D, both directions" begin
            I = CartesianIndex(3, 3)
            for (nm, node) in (("identity", id2), ("D₋ₓ", D₋ₓ(id2)), ("D₋ᵧ", D₋ᵧ(id2)),
                ("M₊ᵧ", M₊ᵧ(id2)), ("jumpᵧ", jumpᵧ(id2)), ("Dcᵧ", Dcᵧ(id2)),
                ("Dₕₓ", Dₕₓ(id2)), ("Dstar₊ᵧ", Dstar₊ᵧ(id2)))
                @testset "$nm" begin
                    @test sort(stencil_offsets(node)) == _stencil_at(node, Wₕ2, I, lin2)
                end
            end
        end
    end

    @testset "the prediction matches the assembled matrix" begin
        # The independent check: every family has a matrix form, so the predicted offsets
        # can be compared against the diagonals the matrix actually occupies rather than
        # against another prediction.
        for (nm, node, mat) in (("D₋ₓ", D₋ₓ(id1), D₋ₓ(Ωₕ1)), ("D₊ₓ", D₊ₓ(id1), D₊ₓ(Ωₕ1)),
            ("M₋ₓ", M₋ₓ(id1), M₋ₓ(Ωₕ1)), ("M₊ₓ", M₊ₓ(id1), M₊ₓ(Ωₕ1)),
            ("jumpₓ", jumpₓ(id1), jumpₓ(Ωₕ1)), ("Dcₓ", Dcₓ(id1), Dcₓ(Ωₕ1)),
            ("Dstar₊ₓ", Dstar₊ₓ(id1), Dstar₊ₓ(Ωₕ1)), ("Dₕₓ", Dₕₓ(id1), Dₕₓ(Ωₕ1)))
            @testset "$nm" begin
                predicted = sort([o[1] for o in stencil_offsets(node)])
                @test predicted == _matrix_offsets(Matrix(mat))
            end
        end
    end

    @testset "the leaves touch nothing but the point" begin
        for op in (TrialFunction{1}(), TestFunction{1}(), IndexedTrialFunction{1}(1),
            source_function(sin, Val(1)), SourceVector{1, Vector{Float64}}([1.0]),
            id1, ZeroOperator(Wₕ1))
            @test stencil_offsets(op) == [(0,)]
            @test matrix_structure(op) isa DiagonalStructure
        end

        # and in more dimensions the origin is still the origin, so still Diagonal — which
        # is the case worth catching: a mass-like term in 2D really is diagonal
        @test stencil_offsets(id2) == [(0, 0)]
        @test matrix_structure(id2) isa DiagonalStructure
        @test matrix_structure(3 * id2) isa DiagonalStructure
    end

    @testset "classification" begin
        @test matrix_structure(D₋ₓ(id1)) isa TridiagonalStructure
        @test matrix_structure(Dcₓ(id1)) isa TridiagonalStructure     # reaches ±1
        @test matrix_structure(Dₕₓ(id1)) isa TridiagonalStructure     # three point
        @test matrix_structure(D₋ₓ(id1) + D₊ₓ(id1)) isa TridiagonalStructure

        # reaching two points either side is no longer tridiagonal
        @test sort(stencil_offsets(D₋ₓ(D₋ₓ(id1)))) == [(-2,), (-1,), (0,)]
        @test matrix_structure(D₋ₓ(D₋ₓ(id1))) isa SparseStructure

        # a pure shift lands off the diagonal
        @test stencil_offsets(shift_op(id1, 1, 2)) == [(2,)]
        @test matrix_structure(shift_op(id1, 1, 2)) isa SparseStructure
        @test matrix_structure(shift_op(id1, 1, 1)) isa TridiagonalStructure
        @test matrix_structure(shift_op(id1, 1, 0)) isa DiagonalStructure

        # Above one dimension a narrow stencil is still sparse, and that is the point: a
        # five-point stencil has offsets ±1 and ±nₓ, so its band is 2nₓ+1 wide and almost
        # all zero. A banded format would store far more than a sparse one.
        @test matrix_structure(D₋ₓ(id2)) isa SparseStructure
        @test matrix_structure(Dcᵧ(id2)) isa SparseStructure
        @test matrix_structure(D₋ₓ(id2) + D₋ᵧ(id2)) isa SparseStructure
    end

    @testset "how each node changes the reach" begin
        base = stencil_offsets(D₋ₓ(id1))

        # scaling does not widen it
        @test stencil_offsets(3 * D₋ₓ(id1)) == base
        @test stencil_offsets(D₋ₓ(id1) / 4) == base
        uₕ = Rₕ(Wₕ1, x -> x + 1)
        @test stencil_offsets(uₕ * D₋ₓ(id1)) == base

        # nor does a restriction: off its region the operator contributes nothing at all,
        # so the reach is its child's wherever it contributes
        @test stencil_offsets(restrict_to(:interior, D₋ₓ(id1))) == base

        # a sum reaches the union of its addends
        @test sort(stencil_offsets(D₋ₓ(id1) + D₊ₓ(id1))) == [(-1,), (0,), (1,)]
        @test sort(stencil_offsets(Dcₓ(id1) + id1)) == [(-1,), (0,), (1,)]

        # nesting composes the two reaches
        @test sort(stencil_offsets(D₊ₓ(D₊ₓ(id1)))) == [(0,), (1,), (2,)]
        @test sort(stencil_offsets(Dₕₓ(D₋ₓ(id1)))) == [(-2,), (-1,), (0,), (1,)]

        # a shift moves the reach without widening it
        @test length(stencil_offsets(shift_op(D₋ₓ(id1), 1, 3))) == length(base)
        @test sort(stencil_offsets(shift_op(D₋ₓ(id1), 1, 3))) == [(2,), (3,)]

        # and the answer has no repeats, however the tree is built
        summed = stencil_offsets(D₋ₓ(id1) + D₋ₓ(id1))
        @test summed == unique(summed)
        @test summed == base
    end

    @testset "the offsets are the same at every grid point" begin
        # What the whole approach rests on: a truncated point keeps its offsets and zeroes
        # its coefficients, so one prediction covers the grid. If a node ever truncated by
        # dropping entries instead, the pattern would depend on position and this would
        # stop being sound.
        for node in (D₋ₓ(id1), D₊ₓ(id1), Dcₓ(id1), Dₕₓ(id1), Dstar₊ₓ(id1), jumpₓ(id1))
            predicted = sort(stencil_offsets(node))
            for i in 1:npoints(Ωₕ1)
                @test _stencil_at(node, Wₕ1, CartesianIndex(i), lin1) == predicted
            end
        end
    end

    @testset "the structure types" begin
        for T in (DiagonalStructure, TridiagonalStructure, SparseStructure)
            @test T <: MatrixStructure
            @test T() isa MatrixStructure
        end
    end
end
