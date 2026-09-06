using Test
using Bramble
using Random
using SparseArrays
using Bramble: IdentityOperator, ZeroOperator, TrialFunction, TestFunction,
               IndexedTrialFunction, SourceVector, LazyOp,
               stencil_offsets, local_stencil, shift_op, restrict_to, source_function,
               TrialFunction, TestFunction, LinearProduct, BilinearProduct

# Reading the sparsity pattern off an AST before assembling it.
#
# Every node reaches a fixed set of neighbours, and that set is a property of the tree
# rather than of the grid point: truncation at a boundary zeroes the coefficients and keeps
# the offsets. So the pattern is known before a single entry is computed, which is what lets
# the backend's matrix be preallocated with exactly that pattern: after which assembly only
# ever updates stored values instead of performing structural inserts.
#
# It deliberately stops at the pattern and does not pick a matrix type: that belongs to the
# backend, which carries it as a type parameter.
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

@testset "Stencil patterns" begin
    Random.seed!(20260831)
    Ωₕ1 = mesh(domain(interval(0.0, 1.0)), 9, false)
    Ωₕ2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 6), (true, false))
    Wₕ1, Wₕ2 = gridspace(Ωₕ1), gridspace(Ωₕ2)
    id1, id2 = IdentityOperator(Wₕ1), IdentityOperator(Wₕ2)
    lin1 = LinearIndices(Bramble.indices(Ωₕ1))
    lin2 = LinearIndices(Bramble.indices(Ωₕ2))

    @testset "Prediction match" begin
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

        @testset "2D directions" begin
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

    @testset "Matrix prediction match" begin
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

    @testset "Leaf reach" begin
        for op in (TrialFunction{1}(), TestFunction{1}(), IndexedTrialFunction{1}(1),
            source_function(sin, Val(1)), SourceVector{1, Vector{Float64}}([1.0]),
            id1, ZeroOperator(Wₕ1))
            @test stencil_offsets(op) == [(0,)]
        end
        @test stencil_offsets(id2) == [(0, 0)]
        @test stencil_offsets(3 * id2) == [(0, 0)]
    end

    @testset "Node reach bounds" begin
        # a one-sided operator reaches its own side and no further; the centered one skips
        # its centre; the cross-weighted one does not
        @test sort(stencil_offsets(D₋ₓ(id1))) == [(-1,), (0,)]
        @test sort(stencil_offsets(D₊ₓ(id1))) == [(0,), (1,)]
        @test sort(stencil_offsets(Dcₓ(id1))) == [(-1,), (1,)]
        @test sort(stencil_offsets(Dₕₓ(id1))) == [(-1,), (0,), (1,)]
        @test sort(stencil_offsets(jumpₓ(id1))) == [(0,), (1,)]

        # composing widens, and the widening is the sum of the two reaches
        @test sort(stencil_offsets(D₋ₓ(D₋ₓ(id1)))) == [(-2,), (-1,), (0,)]

        # a shift moves the reach without widening it
        @test stencil_offsets(shift_op(id1, 1, 2)) == [(2,)]
        @test length(stencil_offsets(shift_op(D₋ₓ(id1), 1, 3))) ==
              length(stencil_offsets(D₋ₓ(id1)))
    end

    @testset "Reach transformation" begin
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

    @testset "Uniform point offsets" begin
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

    @testset "Product patterns" begin
        # These are the only nodes assembly ever evaluates, and neither had a method until
        # the parallel assembly needed to ask what an assembled form reaches.
        u, v = TrialFunction{1}(), TestFunction{1}()
        uh = Rₕ(Wₕ1, sin)

        # A linear product contracts its left factor away (`multiply_stencils_linear`
        # keeps only the right offsets), so its reach is the test side's.
        @test stencil_offsets(innerₕ(uh, v)) == [(0,)]
        @test sort(stencil_offsets(innerₕ(uh, D₋ₓ(v)))) == [(-1,), (0,)]
        @test sort(stencil_offsets(inner₊(uh, D₋ₓ(v)))) == [(-1,), (0,)]

        # which is what tells a parallel assembly whether its writes can overlap: a form
        # whose reach is the origin alone scatters one value per row and needs no
        # coordination, whatever weight it carries.
        @test stencil_offsets(inner₊(uh, v)) == [(0,)]
        @test sort(stencil_offsets(innerₕ(uh, v) + innerₕ(uh, D₋ₓ(v)))) == [(-1,), (0,)]

        # A bilinear product pairs a row offset with a column offset -- but colouring, the
        # one consumer of this function (gpena/Bramble.jl#54), only ever needs the row
        # side: a write collision needs both to coincide, and colour-separated rows can't.
        # So this reduces to the test factor's reach, same as a LinearProduct already does,
        # regardless of how complex the trial factor is.
        @test sort(stencil_offsets(innerₕ(D₋ₓ(u), D₋ₓ(v)))) == [(-1,), (0,)]
    end
end
