module FormStencilPatternTests

using Test
using Bramble
# Internal names: defined and documented, not exported.
import Bramble: D₊ₓ, M₊ₓ, M₊ᵧ
using Random
using SparseArrays
using Bramble:
               IdentityOperator,
               ZeroOperator,
               TrialFunction,
               TestFunction,
               IndexedTrialFunction,
               SourceVector,
               LazyOp,
               stencil_offsets,
               local_stencil,
               shift_op,
               restrict_to,
               source_function,
               TrialFunction,
               TestFunction,
               LinearProduct,
               BilinearProduct,
               Dcᵧ,
               Dcₓ,
               D̃ᵧ,
               D̃ₓ,
               D̽ₓ,
               D₋ᵧ,
               D₋ₓ,
               Mₓ,
               jumpᵧ,
               jumpₓ,
               πₕ

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
    return sort(unique(j - i for j in axes(M, 2) for i in axes(M, 1) if M[i, j] != 0))
end

# the offsets a stencil actually produces at one point
function _stencil_at(node, Wₕ, I, lin)
    return sort(unique(first(e) for e in local_stencil(node, Wₕ, I, nothing, lin[I])))
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
            # Only the nodes "Uniform point offsets" below leaves out. That testset makes
            # this same comparison for the difference nodes, at every index rather than
            # at this one, so repeating them here would add nothing.
            for (nm, node) in (
                ("identity", id1),
                ("Mₓ", Mₓ(id1)),
                ("M₊ₓ", M₊ₓ(id1))
            )
                @testset "$nm" begin
                    @test sort(stencil_offsets(node)) == _stencil_at(node, Wₕ1, I, lin1)
                end
            end
        end

        @testset "2D directions" begin
            I = CartesianIndex(3, 3)
            for (nm, node) in (
                ("identity", id2),
                ("D₋ₓ", D₋ₓ(id2)),
                ("D₋ᵧ", D₋ᵧ(id2)),
                ("M₊ᵧ", M₊ᵧ(id2)),
                ("jumpᵧ", jumpᵧ(id2)),
                ("Dcᵧ", Dcᵧ(id2)),
                ("D̽ₓ", D̽ₓ(id2)),
                ("D̃ᵧ", D̃ᵧ(id2))
            )
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
        for (nm, node, mat) in (
            ("D₋ₓ", D₋ₓ(id1), D₋ₓ(Ωₕ1)),
            ("D₊ₓ", D₊ₓ(id1), D₊ₓ(Ωₕ1)),
            ("Mₓ", Mₓ(id1), Mₓ(Ωₕ1)),
            ("M₊ₓ", M₊ₓ(id1), M₊ₓ(Ωₕ1)),
            ("jumpₓ", jumpₓ(id1), jumpₓ(Ωₕ1)),
            ("Dcₓ", Dcₓ(id1), Dcₓ(Ωₕ1)),
            ("D̃ₓ", D̃ₓ(id1), D̃ₓ(Ωₕ1)),
            ("D̽ₓ", D̽ₓ(id1), D̽ₓ(Ωₕ1))
        )
            @testset "$nm" begin
                predicted = sort([o[1] for o in stencil_offsets(node)])
                @test predicted == _matrix_offsets(Matrix(mat))
            end
        end
    end

    @testset "Leaf reach" begin
        for op in (
            TrialFunction{1}(),
            TestFunction{1}(),
            IndexedTrialFunction{1}(1),
            source_function(sin, Val(1)),
            SourceVector{1, Vector{Float64}}([1.0]),
            id1,
            ZeroOperator(Wₕ1)
        )
            @test stencil_offsets(op) == [(0,)]
        end
        @test stencil_offsets(id2) == [(0, 0)]
        @test stencil_offsets(3 * id2) == [(0, 0)]
        @test stencil_offsets(Bramble.IndexedTestFunction{2}(2)) == [(0, 0)]
        @test stencil_offsets(Bramble.SourceConstant{2, Float64}(2.5)) == [(0, 0)]
        @test stencil_offsets(Bramble.dirac((0.3, 0.6), 2.0)) == [(0, 0)]
    end

    @testset "Interior margin" begin
        # `_stencil_margin` is how far from every face a point must be for none of a term's
        # taps to leave the grid: one step per tapped node and `|s|` per shift by `s`, added
        # up the tree, so never less than the reach `stencil_offsets` predicts (checked
        # against the matrices above), and equal to it for a leaf, which reaches nothing.
        u, v = TrialFunction{1}(), TestFunction{1}()
        uₕ = Rₕ(Wₕ1, x -> x + 1)
        _reach(op) = maximum(o -> maximum(abs, o), stencil_offsets(op))
        for op in (
            u,
            v,
            IndexedTrialFunction{1}(1),
            Bramble.IndexedTestFunction{1}(2),
            source_function(sin, Val(1)),
            SourceVector{1, Vector{Float64}}([1.0]),
            Bramble.SourceConstant{1, Float64}(2.5),
            Bramble.dirac(0.3, 2.0),
            id1,
            ZeroOperator(Wₕ1)
        )
            @test Bramble._stencil_margin(op) == _reach(op) == 0
        end

        for (op, margin) in (
            (D₋ₓ(D₊ₓ(u)), 2),
            (shift_op(D₋ₓ(u), 1, 2), 3),
            (3 * Dcₓ(u), 1),
            (uₕ * M₊ₓ(u), 1),
            (restrict_to(:interior, D̽ₓ(u)), 1),
            (innerₕ(D₋ₓ(u), D₊ₓ(D₊ₓ(v))), 2),
            (innerₕ(D₋ₓ(D₋ₓ(u)), v), 2),
            (innerₕ(u, v) + innerₕ(D₋ₓ(u), D₋ₓ(v)), 1)
        )
            @test Bramble._stencil_margin(op) == margin
            @test margin >= _reach(op)
        end

        # an interpolation names columns on another mesh, never a tap on this one, so it
        # adds no margin; a difference around it adds its own one step
        @test Bramble._stencil_margin(πₕ(u)) == 0
        @test Bramble._stencil_margin(D₋ₓ(πₕ(u))) == 1
    end

    @testset "Bilinear terms, lexicographic distance" begin
        # `bandwidths` reads every bilinear term's (trial, test) reach and turns each pair
        # into a distance in the lexicographic order. The terms come through a sum and a
        # scale, and anything that is not a bilinear term contributes none.
        u, v = TrialFunction{1}(), TestFunction{1}()
        terms = Bramble._bilinear_terms(innerₕ(D₊ₓ(u), D₋ₓ(v)) + 3 * innerₕ(u, v) + id1)
        @test length(terms) == 2
        @test eltype(terms) == Tuple{Vector{NTuple{1, Int}}, Vector{NTuple{1, Int}}}
        @test sort(terms[1][1]) == [(0,), (1,)]
        @test sort(terms[1][2]) == [(-1,), (0,)]
        @test terms[2] == ([(0,)], [(0,)])
        @test isempty(Bramble._bilinear_terms(id1))

        # strides (1, n₁): one step along x is 1, one along y is n₁, and the distance is
        # column minus row
        @test Bramble._lex_distance((1, 0), (0, 1), (1, 5)) == 1 - 5
        @test Bramble._lex_distance((0, 0), (0, 0), (1, 5)) == 0
        @test Bramble._lex_distance((-1, 2), (1, -1), (1, 7)) == -2 + 3 * 7
        @test Bramble._lex_distance((2,), (-1,), (1,)) == 3
        @test Bramble._lex_strides((5, 4, 3)) == (1, 5, 20)
        @test Bramble._lex_strides((7,)) == (1,)
        @test isconcretetype(only(Base.return_types(Bramble._lex_strides, (NTuple{3, Int},))))

        # and the bandwidth of a form on the non-uniform 2D mesh is the widest diagonal
        # the assembled matrix occupies
        a = form(Wₕ2, Wₕ2, (u, v) -> innerₕ(D₊ₓ(u), D₋ᵧ(v)) + innerₕ(u, v))
        A = assemble(a)
        offs = _matrix_offsets(A)
        @test Bramble.bandwidths(a) == (-minimum(offs), maximum(offs))
    end

    @testset "Node reach bounds" begin
        # a one-sided operator reaches its own side and no further; the centered one skips
        # its centre; the cross-weighted one does not
        @test sort(stencil_offsets(D₋ₓ(id1))) == [(-1,), (0,)]
        @test sort(stencil_offsets(D₊ₓ(id1))) == [(0,), (1,)]
        @test sort(stencil_offsets(Dcₓ(id1))) == [(-1,), (1,)]
        @test sort(stencil_offsets(D̽ₓ(id1))) == [(-1,), (0,), (1,)]
        @test sort(stencil_offsets(jumpₓ(id1))) == [(0,), (1,)]

        # composing widens, and the widening is the sum of the two reaches
        @test sort(stencil_offsets(D₋ₓ(D₋ₓ(id1)))) == [(-2,), (-1,), (0,)]

        # a shift moves the reach without widening it
        @test stencil_offsets(shift_op(id1, 1, 2)) == [(2,)]
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
        @test sort(stencil_offsets(D̽ₓ(D₋ₓ(id1)))) == [(-2,), (-1,), (0,), (1,)]

        # a shift moves the reach without widening it, and moves it exactly
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
        for node in (D₋ₓ(id1), D₊ₓ(id1), Dcₓ(id1), D̽ₓ(id1), D̃ₓ(id1), jumpₓ(id1))
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
        # one consumer of this function, only ever needs the row
        # side: a write collision needs both to coincide, and colour-separated rows can't.
        # So this reduces to the test factor's reach, same as a LinearProduct already does,
        # regardless of how complex the trial factor is.
        @test sort(stencil_offsets(innerₕ(D₋ₓ(u), D₋ₓ(v)))) == [(-1,), (0,)]
    end
end

end # module FormStencilPatternTests
