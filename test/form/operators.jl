module FormOperatorsTests

using Test
using Bramble
# Internal names: defined and documented, not exported.
import Bramble: D₊ₓ, D₊ᵧ, ∇₊ₕ, M₊ₓ, M₊ᵧ, M₊₂, M₊ₕ
using Bramble:
               IdentityOperator,
               ZeroOperator,
               TrialFunction,
               TestFunction,
               IndexedTrialFunction,
               IndexedTestFunction,
               LazyOp,
               BackwardDifference,
               ForwardDifference,
               BackwardAverage,
               ForwardAverage,
               ShiftNode,
               RegionRestriction,
               BilinearProduct,
               InnerH,
               InnerPlus,
               local_stencil,
               resolve_ast,
               restrict_to,
               shift_op,
               inner_plus,
               is_symbolic,
               markers,
               D₋ᵧ,
               D₋₂,
               D₋ₓ,
               Mᵧ,
               M₂,
               Mₓ,
               inner₊ᵧ,
               inner₊₂,
               inner₊ₓ,
               S₊,
               S₋,
               S₊ₓ,
               S₋ₓ,
               S₊ᵧ,
               S₋ᵧ,
               S₊₂,
               jumpₓ,
               forward_shift,
               backward_shift,
               Dcₓ,
               D̃ₓ,
               D̽ₓ,
               Mcₓ
using LinearAlgebra: I, norm
using ..TestUtils: WITH_SLOW_TESTS

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
    Ωₕ = mesh(
        domain(interval(0.0, 1.0) × interval(0.0, 1.0), :bottom => :bottom),
        (5, 6),
        (true, true)
    )
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
            for (op, T, dim) in (
                (Mₓ(id), BackwardAverage, 1),
                (M₊ₓ(id), ForwardAverage, 1),
                (Mᵧ(id), BackwardAverage, 2),
                (M₊ᵧ(id), ForwardAverage, 2),
                (M₂(id), BackwardAverage, 3),
                (M₊₂(id), ForwardAverage, 3)
            )
                @test op isa T
                @test typeof(op).parameters[2] == dim
                @test resolve_ast(op) isa T
                @test !is_symbolic(op)
                @test is_symbolic(Mₓ(TrialFunction{2}()))
            end
        end

        @testset "Stencil evaluation" begin
            # an average is the mean of the point and its neighbour: two half weights,
            # one at the origin and one a step away in the direction it averages over
            for (op, offset) in (
                (Mₓ(id), (-1, 0)), (M₊ₓ(id), (1, 0)), (Mᵧ(id), (0, -1)), (M₊ᵧ(id), (0, 1))
            )
                st = local_stencil(op, Wₕ, interior, nothing, lin[interior])
                @test length(st) == 2
                @test sum(last, st) ≈ 1.0            # an average preserves constants
                @test all(≈(0.5) ∘ last, st)
                @test Set(first.(st)) == Set([_ORIGIN_2D, offset])
            end
        end

        @testset "Vector forms" begin
            # `vectorial_avg_backward`/`vectorial_avg_forward` are `Mₕ`/`M₊ₕ` under
            # another name. The direction argument the same names also take is the one thing
            # that is new.
            @test Mₕ(id) === (Mₕ(id, Val(1)), Mₕ(id, Val(2)))
            @test M₊ₕ(id) === (M₊ₕ(id, Val(1)), M₊ₕ(id, Val(2)))
            @test Mₕ(id) isa NTuple{2, BackwardAverage}
            @test M₊ₕ(id) isa NTuple{2, ForwardAverage}
            @test Mₕ(id)[1] === Mₓ(id)
            @test Mₕ(id)[2] === Mᵧ(id)

            # in one dimension it is the node itself, as the gradients are
            id1 = IdentityOperator(gridspace(mesh(domain(interval(0.0, 1.0)), 7, true)))
            @test !(Mₕ(id1) isa Tuple)
            @test !(M₊ₕ(id1) isa Tuple)
        end
    end

    @testset "Shift node" begin
        for (dim, amount, offset) in ((1, 1, (1, 0)), (1, -1, (-1, 0)), (2, 1, (0, 1)), (2, -2, (0, -2)))
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
            # restrict by pass `nothing`, not `haskey(::Nothing, ::Symbol)`.
            # Nothing marked means `:interior` is the whole grid and every named region is
            # empty, the same answer a table simply missing the key already gave.
            @test local_stencil(
                restrict_to(:interior, id), Wₕ, interior, nothing, lin[interior]
            ) == local_stencil(id, Wₕ, interior, nothing, lin[interior])
            @test local_stencil(
                restrict_to(:bottom, id), Wₕ, bottom_idx, nothing, lin[bottom_idx]
            ) == ()
            @test local_stencil(
                restrict_to(:nosuchregion, id), Wₕ, interior, nothing, lin[interior]
            ) == ()

            # and a table without the key behaves the same way
            @test local_stencil(
                restrict_to(:nosuchregion, id), Wₕ, interior, mk, lin[interior]
            ) == ()
        end

        # A custom :interior marker is honoured, not overridden by !:boundary.
        @testset "custom :interior marker kept (#66)" begin
            # `:interior` must not be computed as the complement of `:boundary`
            # unconditionally, which would discard a real marker table's own `:interior`
            # entry, even a deliberately redefined one. mesh/marker.jl
            # warns the caller that a custom definition wins.
            S1 = interval(0.0, 1.0)
            Ωc = domain(S1, :interior => (x -> x[1] > 0.5))
            Ωch = mesh(Ωc, 5, true; warn_marker_mismatch = false)
            custom_interior = markers(Ωch)[:interior]

            # Deliberately not the complement of the boundary marker, so reading the interior marker directly
            # and computing "not boundary" give different answers.
            @test custom_interior != .!markers(Ωch)[:boundary]

            Wc = gridspace(Ωch)
            a = form(Wc, Wc, (u, v) -> innerₕ(restrict_to(:interior, u), v))
            A = Matrix(assemble(a))
            n = size(A, 1)
            @test findall(!iszero, [A[i, i] for i in 1:n]) == findall(custom_interior)

            # The default (geometric, unmarked) case reads :interior directly and agrees with !:boundary,
            # which `_ensure_geometric_markers!` defines it as by construction.
            Ωd = mesh(domain(S1), 5, true)
            @test markers(Ωd)[:interior] == .!markers(Ωd)[:boundary]
            Wd = gridspace(Ωd)
            ad = form(Wd, Wd, (u, v) -> innerₕ(restrict_to(:interior, u), v))
            Ad = Matrix(assemble(ad))
            nd = size(Ad, 1)
            @test findall(!iszero, [Ad[i, i] for i in 1:nd]) == findall(markers(Ωd)[:interior])
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
            @test_throws ArgumentError inner₊(Mₓ(u2), Mₓ(v2))

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
                @test weight(f(Mₓ(u2), Mₓ(v2))) === InnerPlus{dim}
            end
        end

        @testset "Gradient tuple sum" begin
            g = inner₊(∇ₕ(u2), ∇ₕ(v2))
            @test g === inner_plus(∇ₕ(u2), ∇ₕ(v2))
            @test g isa Bramble.OperatorAdd          # one product per direction, summed

            # There is deliberately no innerₕ over gradient tuples: InnerH carries a single
            # weight, so the sum has nothing to infer and is written out at the call site.
            # `invokelatest` keeps JET from proving the call always throws and reporting it
            # at the enclosing testset: the missing method is what this line asserts.
            @test_throws MethodError Base.invokelatest(innerₕ, ∇ₕ(u2), ∇ₕ(v2))
            @test innerₕ(∇ₕ(u2)[1], ∇ₕ(v2)[1]) + innerₕ(∇ₕ(u2)[2], ∇ₕ(v2)[2]) isa
                  Bramble.OperatorAdd
        end
    end

    @testset "Composite space nodes" begin
        # The nodes carry the space only through their dimension, so a composite space is
        # not a different case for them, but nothing had checked, and the difference and
        # average families are what a coupled form is written from.
        idv = IdentityOperator(Vₕ)
        @test Bramble.space(idv) === Vₕ

        for f in (D₋ₓ, D₊ₓ, D₋ᵧ, D₊ᵧ, Mₓ, M₊ₓ, Mᵧ, M₊ᵧ)
            @test f(idv) isa LazyOp{2}
            @test resolve_ast(f(idv)) isa LazyOp{2}
        end
        @test ∇ₕ(idv) isa NTuple{2, BackwardDifference}
        @test ∇₊ₕ(idv) isa NTuple{2, ForwardDifference}
        @test Mₕ(idv) isa NTuple{2, BackwardAverage}
        @test restrict_to(:bottom, idv) isa RegionRestriction
        @test shift_op(idv, 1, 1) isa ShiftNode

        # and the stencils evaluate against the composite space unchanged: the offsets are
        # in grid coordinates, which the components share
        linv = LinearIndices(Bramble.indices(mesh(Vₕ)))
        for f in (D₋ₓ, Mₓ, M₊ᵧ)
            @test local_stencil(f(idv), Vₕ, interior, nothing, linv[interior]) ==
                  local_stencil(f(id), Wₕ, interior, nothing, lin[interior])
        end
    end

    @testset "Zero & identity nodes" begin
        z = ZeroOperator(Wₕ)
        @test z isa LazyOp{2}
        @test sprint(show, z) == "0"
        @test sprint(show, id) == "I"
    end
end

# The public index shifts on a symbolic operand build `ShiftNode`s,
# and a shift composed with another stencil assembles what the grid functions compute. The
# oracle is the matrix product `H · S · Op` of the space-layer matrices, `H` the `innerₕ`
# weights. The boundary is the hard case: `S₊ₓ(D₋ₓ(u))` must not keep the `-u_n/h` tap of `D₋ₓ`
# at the clamped last point, where the shift reads 0.
@testset "shift node: composed stencils" begin
    box(D) = D == 1 ? interval(0.0, 1.0) :
             D == 2 ? interval(0.0, 1.0) × interval(0.0, 2.0) :
             interval(0.0, 1.0) × interval(0.0, 2.0) × interval(0.0, 3.0)
    mat(W, f) = f === identity ? Matrix(1.0I, ndofs(W), ndofs(W)) : Matrix(f(W))

    @testset "nodes" begin
        W = gridspace(mesh(domain(box(2)), (5, 6), (false, false)))
        u = Bramble.trial_function(W)
        @test @inferred(S₊ₓ(u)) isa ShiftNode{2, 1}
        @test S₊ₓ(u).shift_amount == 1
        @test S₋ᵧ(u) isa ShiftNode{2, 2}
        @test S₋ᵧ(u).shift_amount == -1
        @test S₊(u, Val(2)) === S₊ᵧ(u) === forward_shift(u, Val(2))
        @test S₋(u, Val(1)) === S₋ₓ(u) === backward_shift(u, Val(1))
        @test S₊ₕ(u) === (S₊ₓ(u), S₊ᵧ(u))
        @test S₋ₕ(u) === (S₋ₓ(u), S₋ᵧ(u))
        @test S₊ₕ[1](u) === S₊ₓ(u)
        # the node `shift_op` builds, with the direction a type
        @test S₊ₓ(D₋ₓ(u)) === shift_op(D₋ₓ(u), 1, 1)
        u1 = Bramble.trial_function(gridspace(mesh(domain(box(1)), 5, false)))
        @test S₊ₕ(u1) === S₊ₓ(u1)
        @test S₋ₕ(u1) === S₋ₓ(u1)
        # a direction the operand does not have is refused, as on a grid function
        @test_throws ArgumentError S₊ᵧ(u1)
        @test_throws ArgumentError S₋(u1, Val(2))
        @test_throws ArgumentError S₊₂(u)
        @test_throws ArgumentError S₋(D₋ₓ(u), Val(3))
    end

    @testset "$(D)D" for D in 1:3
        n = ntuple(i -> 4 + i, D)
        W = gridspace(mesh(domain(box(D)), n, ntuple(_ -> false, D)))
        uh = Rₕ(W, x -> 1 + sum(abs2, x) + prod(x))
        x = parent(uh)
        H = Matrix(assemble(form(W, W, (u, v) -> innerₕ(u, v))))
        inners = D == 1 ? (identity, D₋ₓ, M₊ₓ, jumpₓ) :
                 (identity, D₋ₓ, M₊ₓ, jumpₓ, D₋ᵧ)
        @testset "$(S)∘$(op)" for d in 1:D,
            (S, Sb) in ((S₊ₕ[d], forward_shift), (S₋ₕ[d], backward_shift)),
            op in inners
            SO = Matrix(Sb(W, Val(d))) * mat(W, op)
            a = form(W, W, (u, v) -> innerₕ(S(op(u)), v))
            A = assemble(a)
            @test isapprox(Matrix(A), H * SO; atol = 1e-10)
            # the grid-function computation
            @test isapprox(A * x, H * parent(S(op(uh))); atol = 1e-10)
            # the rest compiles a form per case, so it runs on the two taps that reach
            # back onto the grid: a difference and a jump
            op in (D₋ₓ, jumpₓ) || continue
            # the refill and the matrix-free product
            A.nzval .= 0
            assemble!(A, a)
            @test isapprox(Matrix(A), H * SO; atol = 1e-10)
            @test isapprox(matrix_free_operator(a) * x, A * x; atol = 1e-10)
            # on the test side of a linear form, the transpose
            l = assemble(form(W, v -> innerₕ(uh, S(op(v)))))
            @test isapprox(l, SO' * H * x; atol = 1e-10)
        end
    end

    # `shift_op` builds the same node, so it has the same boundary: two points along, the
    # last two rows read 0 rather than a clamped re-evaluation of `D₋ₓ`
    @testset "shift_op by two" begin
        W = gridspace(mesh(domain(box(1)), 7, false))
        H = Matrix(assemble(form(W, W, (u, v) -> innerₕ(u, v))))
        A = assemble(form(W, W, (u, v) -> innerₕ(shift_op(D₋ₓ(u), 1, 2), v)))
        S2 = Matrix(forward_shift(W, Val(1)))^2
        @test isapprox(Matrix(A), H * S2 * Matrix(D₋ₓ(W)); atol = 1e-10)
    end
end

# A tapping node over an operand that carries a grid-function coefficient, `c*u`,
# `c*D₋ₓ(u)` or `D₋ₓ(c*u)`: every difference, average, jump and shift family, on non-uniform
# meshes. The last operand is point-dependent (the coefficient varies) without being a
# `GridFunctionScale` itself, and a tap must relabel its offsets when re-evaluating it at the neighbour, or
# `D₊ₓ(D₋ₓ(c*u))` puts the neighbour's stencil on the point's own columns. The oracle is
# the grid-function computation, one basis vector per column, so the whole matrix is checked against it.
@testset "tap over a coefficient operand" begin
    box(D) = D == 1 ? interval(0.0, 1.0) :
             D == 2 ? interval(0.0, 1.0) × interval(0.0, 2.0) :
             interval(0.0, 1.0) × interval(0.0, 2.0) × interval(0.0, 3.0)
    # Every tap compiles its own form, matrix, refill and product per operand and dimension,
    # which is what the sweep costs. `unit` keeps one tap per kind (a backward and a forward
    # difference, a jump, a forward average, a shift) on all three operands in every
    # dimension; `slow` runs the whole family.
    taps = WITH_SLOW_TESTS ?
           (D₋ₓ, D₊ₓ, Dcₓ, D̃ₓ, D̽ₓ, jumpₓ, Mₓ, M₊ₓ, Mcₓ, S₊ₓ, S₋ₓ) :
           (D₋ₓ, D₊ₓ, jumpₓ, M₊ₓ, S₊ₓ)

    @testset "$(D)D" for D in 1:3
        n = ntuple(i -> 4 + i, D)
        W = gridspace(mesh(domain(box(D)), n, ntuple(_ -> false, D)))
        c = Rₕ(W, x -> 2 + sum(x) + prod(x))
        uh = Rₕ(W, x -> 1 + sum(abs2, x))
        x = parent(uh)
        H = Matrix(assemble(form(W, W, (u, v) -> innerₕ(u, v))))
        # each operand as a form operator and as the grid function it computes
        cu = ("c*u", u -> c * u, z -> element(W, parent(c) .* parent(z)))
        cdu = ("c*D₋ₓ(u)", u -> c * D₋ₓ(u), z -> element(W, parent(c) .* parent(D₋ₓ(z))))
        dcu = ("D₋ₓ(c*u)", u -> D₋ₓ(c * u), z -> D₋ₓ(element(W, parent(c) .* parent(z))))
        cases = Any[(t, o...) for o in (cu, cdu, dcu) for t in taps]  # Any: every tap and operand is its own closure type
        D >= 2 && push!(cases, (S₊ᵧ ∘ D₋ᵧ, cu...), (D₊ᵧ ∘ S₋ᵧ, cu...))
        basis(j) = element(W, [i == j ? 1.0 : 0.0 for i in 1:ndofs(W)])
        @testset "$(tap)∘$(nm)" for (tap, nm, opd, grid) in cases
            G = reduce(hcat, [parent(tap(grid(basis(j)))) for j in 1:ndofs(W)])
            a = form(W, W, (u, v) -> innerₕ(tap(opd(u)), v))
            A = assemble(a)
            # `rtol` as well: the mesh is random, and next to a short cell a double
            # difference's entries grow far past 1
            @test isapprox(Matrix(A), H * G; atol = 1e-9, rtol = 1e-10)
            g = parent(tap(grid(uh)))
            Ax = A * x
            # The product cancels large entries down to O(1), so a fixed `atol` misses by
            # rounding alone: bound it by eps()·‖|A||x|‖. The miss measured at most 1.6 of
            # that over 3000 random meshes per dimension and every case here; 16 leaves 10×.
            @test isapprox(Ax, H * g; atol = 16 * eps() * norm(abs.(A) * abs.(x)), rtol = 1e-10)
            # Evaluating twice must give the same bits: a miss here is nondeterminism (state
            # left by earlier tests, or a race), not a tolerance that is too tight.
            @test parent(tap(grid(uh))) == g
            @test A * x == Ax
            A.nzval .= 0
            assemble!(A, a)
            @test isapprox(Matrix(A), H * G; atol = 1e-9, rtol = 1e-10)
            # the matrix-free product compiles its own walk, so only on the operand that
            # cancels like `A * x`, so it takes the same rounding bound
            nm == "D₋ₓ(c*u)" || continue
            @test isapprox(matrix_free_operator(a) * x, H * G * x; atol = 16 * eps() * norm(abs.(A) * abs.(x)), rtol = 1e-10)
        end
    end
end

end # module FormOperatorsTests
