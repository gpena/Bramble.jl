using Test
using Bramble
using Bramble: IdentityOperator, ZeroOperator, TrialFunction, TestFunction,
               IndexedTrialFunction, IndexedTestFunction, LazyOp,
               BackwardDifference, ForwardDifference, BackwardAverage, ForwardAverage,
               ShiftNode, RegionRestriction, BilinearProduct, InnerH, InnerPlus,
               local_stencil, resolve_ast, restrict_to, shift_op,
               inner_plus, vectorial_avg_backward, vectorial_avg_forward,
               is_symbolic, markers

# The symbolic operator layer: averages, the shift node, region restriction, and the
# inner products that turn a pair of operators into a bilinear product.
#
# None of this was reachable by the test suite before: `average.jl`, `restriction.jl` and
# `inner.jl` had literally zero executed lines, and the coverage report said 100% because
# Julia only instruments the lines of methods it actually compiled. Probing them turned up
# three breaks straight away, all fixed here and pinned below:
#
#   - `inner₊(D₋ₓ(u), D₋ₓ(v))` threw. Above one dimension a bare `inner₊` had no method,
#     so dispatch fell through to the *numeric* inner₊ over grid functions, whose
#     @generated body then complained about types the caller never wrote.
#   - `inner₊(u, D₋ₓ(v))` threw for the same reason unless `u` was an indexed leaf.
#   - `local_stencil` on a RegionRestriction called `haskey(nothing, :boundary)` whenever
#     no marker table was passed, which every other node accepts and ignores.
#
# A stencil is a tuple of `(offset, coefficient)` pairs, offsets relative to the point.

const _ORIGIN_2D = (0, 0)

@testset "Symbolic operators" begin
    Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0), :bottom => :bottom),
        (5, 6), (true, true))
    Wₕ = gridspace(Ωₕ)
    Vₕ = gridspace(Ωₕ, Val(3))
    id = IdentityOperator(Wₕ)
    lin = LinearIndices(Bramble.indices(Ωₕ))
    interior = CartesianIndex(3, 3)
    mk = markers(Ωₕ)

    # a point that really is on the bottom edge, found rather than assumed
    bottom_idx = first(I for I in Bramble.indices(Ωₕ) if mk[:bottom][lin[I]])

    @testset "Averages" begin
        @testset "Directional nodes" begin
            for (op, T, dim) in ((M₋ₓ(id), BackwardAverage, 1), (
                M₊ₓ(id), ForwardAverage, 1),
                (M₋ᵧ(id), BackwardAverage, 2), (M₊ᵧ(id), ForwardAverage, 2),
                (M₋₂(id), BackwardAverage, 3), (M₊₂(id), ForwardAverage, 3))
                @test op isa T
                @test typeof(op).parameters[2] == dim
                @test resolve_ast(op) isa T
                @test !is_symbolic(op)
                @test is_symbolic(M₋ₓ(TrialFunction{2}()))
            end
        end

        @testset "Stencil evaluation" begin
            # an average is the mean of the point and its neighbour: two half weights,
            # one at the origin and one a step away in the direction it averages over
            for (op, offset) in ((M₋ₓ(id), (-1, 0)), (M₊ₓ(id), (1, 0)),
                (M₋ᵧ(id), (0, -1)), (M₊ᵧ(id), (0, 1)))
                st = local_stencil(op, Wₕ, interior, nothing, lin[interior])
                @test length(st) == 2
                @test sum(last, st) ≈ 1.0            # an average preserves constants
                @test all(≈(0.5) ∘ last, st)
                @test Set(first.(st)) == Set([_ORIGIN_2D, offset])
            end
        end

        @testset "Vector forms" begin
            @test M₋ₕ(id) === vectorial_avg_backward(id)
            @test M₊ₕ(id) === vectorial_avg_forward(id)
            @test M₋ₕ(id) isa NTuple{2, BackwardAverage}
            @test M₊ₕ(id) isa NTuple{2, ForwardAverage}
            @test M₋ₕ(id)[1] === M₋ₓ(id)
            @test M₋ₕ(id)[2] === M₋ᵧ(id)

            # in one dimension it is the node itself, as the gradients are
            id1 = IdentityOperator(gridspace(mesh(domain(interval(0.0, 1.0)), 7, true)))
            @test !(M₋ₕ(id1) isa Tuple)
            @test !(M₊ₕ(id1) isa Tuple)
        end
    end

    @testset "Shift node" begin
        for (dim, amount, offset) in ((1, 1, (1, 0)), (1, -1, (-1, 0)),
            (2, 1, (0, 1)), (2, -2, (0, -2)))
            op = shift_op(id, dim, amount)
            @test op isa ShiftNode
            st = local_stencil(op, Wₕ, interior, nothing, lin[interior])
            @test st == ((offset, 1.0),)          # a pure relabelling, weight untouched
        end

        # a zero shift is the identity, and shifting composes with what it wraps
        @test local_stencil(shift_op(id, 1, 0), Wₕ, interior, nothing, lin[interior]) ==
              local_stencil(id, Wₕ, interior, nothing, lin[interior])
        @test resolve_ast(shift_op(id, 1, 1)) isa ShiftNode
    end

    @testset "Region restriction" begin
        @test restrict_to(:bottom, id) isa RegionRestriction
        @test resolve_ast(restrict_to(:bottom, id)) isa RegionRestriction

        inner_st = local_stencil(id, Wₕ, bottom_idx, mk, lin[bottom_idx])

        @testset "Stencil retention" begin
            r = restrict_to(:bottom, id)
            @test local_stencil(r, Wₕ, bottom_idx, mk, lin[bottom_idx]) == inner_st
            @test local_stencil(r, Wₕ, interior, mk, lin[interior]) == ()
        end

        @testset ":interior vs :boundary" begin
            r = restrict_to(:interior, id)
            @test local_stencil(r, Wₕ, interior, mk, lin[interior]) ==
                  local_stencil(id, Wₕ, interior, nothing, lin[interior])
            if haskey(mk, :boundary)
                @test local_stencil(r, Wₕ, bottom_idx, mk, lin[bottom_idx]) == ()
            end
        end

        @testset "Absent marker table" begin
            # Every other node takes `markers` and ignores it, so callers with nothing to
            # restrict by pass `nothing`. This used to be `haskey(::Nothing, ::Symbol)`.
            # Nothing marked means `:interior` is the whole grid and every named region is
            # empty, the same answer a table simply missing the key already gave.
            @test local_stencil(restrict_to(:interior, id), Wₕ, interior, nothing,
                lin[interior]) == local_stencil(id, Wₕ, interior, nothing, lin[interior])
            @test local_stencil(restrict_to(:bottom, id), Wₕ, bottom_idx, nothing,
                lin[bottom_idx]) == ()
            @test local_stencil(restrict_to(:nosuchregion, id), Wₕ, interior, nothing,
                lin[interior]) == ()

            # and a table without the key behaves the same way
            @test local_stencil(restrict_to(:nosuchregion, id), Wₕ, interior, mk,
                lin[interior]) == ()
        end

        @testset "Custom :interior marker is honoured, not overridden by !:boundary (#66)" begin
            # `:interior` used to be computed as `!_is_marked(markers, :boundary, ...)`
            # unconditionally, discarding whatever a real marker table's own `:interior`
            # entry said -- even a deliberately redefined one, despite mesh/marker.jl
            # warning the caller that a custom definition wins.
            S1 = interval(0.0, 1.0)
            Ωc = domain(S1, :interior => (x -> x[1] > 0.5))
            Ωch = mesh(Ωc, 5, true; warn_marker_mismatch = false)
            custom_interior = markers(Ωch)[:interior]

            # Deliberately not the complement of :boundary, so reading :interior directly
            # and computing "not :boundary" give different answers -- the only way to tell
            # the fix from the bug.
            @test custom_interior != .!markers(Ωch)[:boundary]

            Wc = gridspace(Ωch)
            a = form(Wc, Wc, (u, v) -> innerₕ(restrict_to(:interior, u), v))
            A = Matrix(assemble(a))
            n = size(A, 1)
            @test findall(!iszero, [A[i, i] for i in 1:n]) == findall(custom_interior)

            # The default (geometric, unmarked) case must still behave exactly as before:
            # there, :interior IS defined as !:boundary by construction
            # (`_ensure_geometric_markers!`), so reading it directly agrees numerically
            # with the old computation -- this fix changes which entry is read, not what a
            # mesh with no custom marker computes.
            Ωd = mesh(domain(S1), 5, true)
            @test markers(Ωd)[:interior] == .!markers(Ωd)[:boundary]
            Wd = gridspace(Ωd)
            ad = form(Wd, Wd, (u, v) -> innerₕ(restrict_to(:interior, u), v))
            Ad = Matrix(assemble(ad))
            nd = size(Ad, 1)
            @test findall(!iszero, [Ad[i, i] for i in 1:nd]) ==
                  findall(markers(Ωd)[:interior])
        end

        @testset "Operator composition" begin
            r = restrict_to(:bottom, D₋ₓ(id))
            @test local_stencil(r, Wₕ, bottom_idx, mk, lin[bottom_idx]) ==
                  local_stencil(D₋ₓ(id), Wₕ, bottom_idx, mk, lin[bottom_idx])
            @test local_stencil(r, Wₕ, interior, mk, lin[interior]) == ()
        end
    end

    @testset "Inner products" begin
        u1, v1 = TrialFunction{1}(), TestFunction{1}()
        u2, v2 = TrialFunction{2}(), TestFunction{2}()

        weight(p) = typeof(p).parameters[2]

        @testset "innerₕ weights" begin
            @test weight(innerₕ(u2, v2)) === InnerH
            @test weight(innerₕ(D₋ₓ(u2), D₋ᵧ(v2))) === InnerH   # no direction to clash
            @test innerₕ(u2, v2) isa BilinearProduct
        end

        @testset "1D inner₊" begin
            # there is only one direction to name
            for p in (inner₊(u1, v1), inner₊(D₋ₓ(u1), v1), inner₊(D₋ₓ(u1), D₋ₓ(v1)))
                @test weight(p) === InnerPlus{1}
            end
        end

        @testset "nD inner₊ direction inference" begin
            for (D, dim) in ((D₋ₓ, 1), (D₋ᵧ, 2), (D₋₂, 3))
                @test weight(inner₊(D(u2), D(v2))) === InnerPlus{dim}
                @test weight(inner₊(u2, D(v2))) === InnerPlus{dim}     # the common form
                @test weight(inner₊(D(u2), v2)) === InnerPlus{dim}
            end

            # not restricted to the indexed leaves: a plain trial function reads the
            # direction off the difference exactly as an indexed one does
            p, q = IndexedTrialFunction{2}(1), IndexedTestFunction{2}(2)
            @test weight(inner₊(p, D₋ₓ(q))) === InnerPlus{1}
            @test weight(inner₊(D₋ᵧ(p), q)) === InnerPlus{2}
        end

        @testset "Missing direction error" begin
            @test_throws ArgumentError inner₊(u2, v2)
            @test_throws ArgumentError inner₊(D₋ₓ(u2), D₋ᵧ(v2))
            @test_throws ArgumentError inner₊(M₋ₓ(u2), M₋ₓ(v2))

            # the message has to name the way out, since the failure is a usage error
            msg = try
                inner₊(u2, v2)
            catch e
                sprint(showerror, e)
            end
            @test occursin("inner₊ₓ", msg)
            @test occursin("2 dimensions", msg)
        end

        @testset "Explicit directions" begin
            for (f, dim) in ((inner₊ₓ, 1), (inner₊ᵧ, 2), (inner₊₂, 3))
                @test weight(f(u2, v2)) === InnerPlus{dim}
                @test weight(f(M₋ₓ(u2), M₋ₓ(v2))) === InnerPlus{dim}
            end
        end

        @testset "Gradient tuple sum" begin
            g = inner₊(∇₋ₕ(u2), ∇₋ₕ(v2))
            @test g === inner_plus(∇₋ₕ(u2), ∇₋ₕ(v2))
            @test g isa Bramble.OperatorAdd          # one product per direction, summed

            # There is deliberately no innerₕ over gradient tuples: InnerH carries a single
            # weight, so the sum has nothing to infer and is written out at the call site.
            @test_throws MethodError innerₕ(∇₋ₕ(u2), ∇₋ₕ(v2))
            @test innerₕ(∇₋ₕ(u2)[1], ∇₋ₕ(v2)[1]) + innerₕ(∇₋ₕ(u2)[2], ∇₋ₕ(v2)[2]) isa
                  Bramble.OperatorAdd
        end

        @testset "Left operands" begin
            uₕ = Rₕ(Wₕ, x -> x[1])
            for l in (3.0, (x -> x[1]), uₕ)
                @test innerₕ(l, v2) isa LazyOp
                @test inner₊ₓ(l, v2) isa LazyOp
                @test inner₊ᵧ(l, v2) isa LazyOp
            end
            @test inner₊(3.0, v2) isa LazyOp
            @test inner₊((x -> x[1]), v2) isa LazyOp
        end
    end

    @testset "Composite space nodes" begin
        # The nodes carry the space only through their dimension, so a composite space is
        # not a different case for them, but nothing had checked, and the difference and
        # average families are what a coupled form is written from.
        idv = IdentityOperator(Vₕ)
        @test Bramble.space(idv) === Vₕ

        for f in (D₋ₓ, D₊ₓ, D₋ᵧ, D₊ᵧ, M₋ₓ, M₊ₓ, M₋ᵧ, M₊ᵧ)
            @test f(idv) isa LazyOp{2}
            @test resolve_ast(f(idv)) isa LazyOp{2}
        end
        @test ∇₋ₕ(idv) isa NTuple{2, BackwardDifference}
        @test ∇₊ₕ(idv) isa NTuple{2, ForwardDifference}
        @test M₋ₕ(idv) isa NTuple{2, BackwardAverage}
        @test restrict_to(:bottom, idv) isa RegionRestriction
        @test shift_op(idv, 1, 1) isa ShiftNode

        # and the stencils evaluate against the composite space unchanged: the offsets are
        # in grid coordinates, which the components share
        linv = LinearIndices(Bramble.indices(mesh(Vₕ)))
        for f in (D₋ₓ, M₋ₓ, M₊ᵧ)
            @test local_stencil(f(idv), Vₕ, interior, nothing, linv[interior]) ==
                  local_stencil(f(id), Wₕ, interior, nothing, lin[interior])
        end
    end

    @testset "Zero & identity nodes" begin
        z = ZeroOperator(Wₕ)
        @test z isa LazyOp{2}
        @test sprint(show, z) == "0"
        @test sprint(show, id) == "I"
        @test local_stencil(id, Wₕ, interior, nothing, lin[interior]) ==
              ((_ORIGIN_2D, 1.0),)
    end
end
