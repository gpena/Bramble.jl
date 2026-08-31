using Test
using Bramble
using Random
using Bramble: IdentityOperator, TrialFunction, IndexedTrialFunction, IndexedTestFunction,
               LazyOp, JumpNode, CenteredDifference, StarDifference,
               CrossWeightedDifference, ExtendedDifferenceNode,
               local_stencil, resolve_ast, is_symbolic, get_innermost_dim,
               find_trial_component, find_test_component, values, indices, restrict_to

# The four remaining operator families, as symbolic nodes: the jump, the centered
# difference, the starred forward difference and the cross-weighted centered difference.
#
# The space layer had all four on grid functions; the form layer had none of them, so a
# form could only be written from one-sided differences and averages. Three of the four
# have no *matrix* in the space layer either — `Dcₓ(Ωₕ)` is a MethodError, only `Dcₓ(uₕ)`
# exists — so these nodes are the only route from those operators into an assembled form.
#
# Which is what the central test below rests on. A node is right exactly when applying its
# stencil to a grid function reproduces what the space-layer operator returns for the same
# grid function, and those operators are already tested against their definitions. So the
# stencils are checked against trusted numbers rather than against the algebra that
# produced them.

# Apply a symbolic stencil across the grid, the way assembly will: sum `coefficient ×
# u[I + offset]` over the stencil at each point. Also counts any live coefficient pointing
# off the grid, which must never happen — a truncated point zeroes its coefficients and
# keeps its offsets, so an out-of-range offset must always carry a zero.
function apply_stencil(node, Wₕ, uₕ)
    Ωₕ = mesh(Wₕ)
    idx = indices(Ωₕ)
    lin = LinearIndices(idx)
    u = values(uₕ)
    out = zeros(eltype(u), length(u))
    escaped = 0
    for I in idx
        acc = zero(eltype(u))
        for (off, c) in local_stencil(node, Wₕ, I, nothing, lin[I])
            J = CartesianIndex(Tuple(I) .+ off)
            if J in idx
                acc += c * u[lin[J]]
            elseif c != 0
                escaped += 1
            end
        end
        out[lin[I]] = acc
    end
    return out, escaped
end

@testset "Extended symbolic operators" begin
    @testset "each stencil reproduces its space-layer operator" begin
        Random.seed!(20260831)

        @testset "1D" begin
            for unif in (true, false)
                Ωₕ = mesh(domain(interval(0.0, 1.0)), 9, unif)
                Wₕ = gridspace(Ωₕ)
                id = IdentityOperator(Wₕ)
                uₕ = Rₕ(Wₕ, x -> x^3 + sin(3x) + 1)
                # the four new families, and the four that were already here — with
                # `get_derivative_matrix_and_scale` gone, this is what pins the one-sided
                # nodes to the operators they stand for
                for (node, op) in ((jumpₓ(id), jumpₓ), (Dcₓ(id), Dcₓ),
                    (Dstar₊ₓ(id), Dstar₊ₓ), (Dₕₓ(id), Dₕₓ),
                    (D₋ₓ(id), D₋ₓ), (D₊ₓ(id), D₊ₓ), (M₋ₓ(id), M₋ₓ), (M₊ₓ(id), M₊ₓ))
                    got, escaped = apply_stencil(node, Wₕ, uₕ)
                    @test got ≈ values(op(uₕ)) rtol=1e-12
                    @test escaped == 0
                end
            end
        end

        @testset "2D" begin
            Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (6, 7),
                (true, false))
            Wₕ = gridspace(Ωₕ)
            id = IdentityOperator(Wₕ)
            uₕ = Rₕ(Wₕ, x -> exp(x[1]) * (x[2]^2 + 1))
            for (node, op) in ((jumpₓ(id), jumpₓ), (jumpᵧ(id), jumpᵧ),
                (Dcₓ(id), Dcₓ), (Dcᵧ(id), Dcᵧ),
                (Dstar₊ₓ(id), Dstar₊ₓ), (Dstar₊ᵧ(id), Dstar₊ᵧ),
                (Dₕₓ(id), Dₕₓ), (Dₕᵧ(id), Dₕᵧ))
                got, escaped = apply_stencil(node, Wₕ, uₕ)
                @test got ≈ values(op(uₕ)) rtol=1e-12
                @test escaped == 0
            end
        end

        @testset "3D, every direction" begin
            Ωₕ = mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (4, 5, 4),
                (false, true, false))
            Wₕ = gridspace(Ωₕ)
            id = IdentityOperator(Wₕ)
            uₕ = Rₕ(Wₕ, x -> x[1]^2 + 2x[2] + sin(x[3]) + 1)
            for (node, op) in ((jump₂(id), jump₂), (Dc₂(id), Dc₂),
                (Dstar₊₂(id), Dstar₊₂), (Dₕ₂(id), Dₕ₂))
                got, escaped = apply_stencil(node, Wₕ, uₕ)
                @test got ≈ values(op(uₕ)) rtol=1e-12
                @test escaped == 0
            end
        end
    end

    @testset "the boundary conventions differ, and deliberately" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 5, false)
        Wₕ = gridspace(Ωₕ)
        id = IdentityOperator(Wₕ)
        n = npoints(Ωₕ)
        at_end = CartesianIndex(n)
        at_start = CartesianIndex(1)
        coeffs(node, I, l) = last.(local_stencil(node, Wₕ, I, nothing, l))

        # the jump does NOT truncate: the missing u_{n+1} is taken as zero, so -uₙ
        # survives. That is what makes it agree with its matrix, whose last row is
        # [0 … 0 -1].
        jump_end = local_stencil(jumpₓ(id), Wₕ, at_end, nothing, n)
        @test any(!iszero, coeffs(jumpₓ(id), at_end, n))
        @test sum(c for (o, c) in jump_end if o == (0,)) == -1.0
        @test all(iszero(c) for (o, c) in jump_end if o == (1,))
        @test Matrix(jumpₓ(Ωₕ))[n, n] == -1.0            # and the matrix agrees

        # the scaled differences do truncate, at whichever ends they need a neighbour
        @test all(iszero, coeffs(Dstar₊ₓ(id), at_end, n))
        for node in (Dcₓ(id), Dₕₓ(id))
            @test all(iszero, coeffs(node, at_end, n))
            @test all(iszero, coeffs(node, at_start, 1))
        end

        # a starred difference is fine at the first point — it only reaches forward
        @test any(!iszero, coeffs(Dstar₊ₓ(id), at_start, 1))
    end

    @testset "the stencils have the shape the formulas do" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 7, false)
        Wₕ = gridspace(Ωₕ)
        id = IdentityOperator(Wₕ)
        I = CartesianIndex(4)
        offsets(node) = sort(collect(Set(first.(local_stencil(node, Wₕ, I, nothing, 4)))))

        @test offsets(jumpₓ(id)) == [(0,), (1,)]          # two point, forward
        @test offsets(Dstar₊ₓ(id)) == [(0,), (1,)]        # two point, forward
        @test offsets(Dcₓ(id)) == [(-1,), (1,)]           # two point, skipping the centre
        @test offsets(Dₕₓ(id)) == [(-1,), (0,), (1,)]     # three point

        # a constant differences to zero under all four, and the jump too
        cₕ = Rₕ(Wₕ, x -> 3.0)
        for node in (Dcₓ(id), Dstar₊ₓ(id), Dₕₓ(id), jumpₓ(id))
            got, _ = apply_stencil(node, Wₕ, cₕ)
            # the jump keeps -uₙ at the far end, so only the interior is zero there
            @test all(≈(0.0; atol = 1e-11), got[1:(end - 1)])
        end
    end

    @testset "composing with the other nodes" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (6, 6), (true, false))
        Wₕ = gridspace(Ωₕ)
        id = IdentityOperator(Wₕ)
        uₕ = Rₕ(Wₕ, x -> x[1] * x[2] + 1)

        # each wraps whatever it is given, so a difference of an average is a difference
        # of an average — checked against doing the two in turn on grid functions
        got, escaped = apply_stencil(Dcₓ(M₋ᵧ(id)), Wₕ, uₕ)
        @test escaped == 0
        @test got ≈ values(Dcₓ(M₋ᵧ(uₕ))) rtol=1e-12

        got2, escaped2 = apply_stencil(jumpᵧ(D₋ₓ(id)), Wₕ, uₕ)
        @test escaped2 == 0
        @test got2 ≈ values(jumpᵧ(D₋ₓ(uₕ))) rtol=1e-12

        # and scaling passes straight through
        got3, _ = apply_stencil(3 * Dₕₓ(id), Wₕ, uₕ)
        @test got3 ≈ 3 .* values(Dₕₓ(uₕ)) rtol=1e-12
    end

    @testset "the vector forms" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 6), (true, false))
        Wₕ = gridspace(Ωₕ)
        id = IdentityOperator(Wₕ)

        @test jumpₕ(id) isa NTuple{2, JumpNode}
        @test Dcₕ(id) isa NTuple{2, CenteredDifference}
        @test Dstar₊ₕ(id) isa NTuple{2, StarDifference}
        @test ∇ₕ(id) isa NTuple{2, CrossWeightedDifference}

        @test jumpₕ(id)[1] === jumpₓ(id)
        @test jumpₕ(id)[2] === jumpᵧ(id)
        @test Dcₕ(id)[2] === Dcᵧ(id)
        @test ∇ₕ(id)[1] === Dₕₓ(id)

        # in one dimension the node itself, not a one-element tuple, as ∇₋ₕ already does
        Ω1 = mesh(domain(interval(0.0, 1.0)), 7, true)
        id1 = IdentityOperator(gridspace(Ω1))
        for f in (jumpₕ, Dcₕ, Dstar₊ₕ, ∇ₕ)
            @test !(f(id1) isa Tuple)
        end
        @test jumpₕ(id1) === jumpₓ(id1)
        @test ∇ₕ(id1) === Dₕₓ(id1)
    end

    @testset "the traits every node answers" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 5), (true, true))
        Wₕ = gridspace(Ωₕ)
        id = IdentityOperator(Wₕ)
        u = TrialFunction{2}()

        for (f, T, dim) in ((jumpₓ, JumpNode, 1), (jumpᵧ, JumpNode, 2),
            (Dcₓ, CenteredDifference, 1), (Dcᵧ, CenteredDifference, 2),
            (Dstar₊ₓ, StarDifference, 1), (Dstar₊₂, StarDifference, 3),
            (Dₕₓ, CrossWeightedDifference, 1), (Dₕᵧ, CrossWeightedDifference, 2))
            node = f(id)
            @test node isa T
            @test node isa LazyOp{2}
            @test get_innermost_dim(node) == dim
            @test resolve_ast(node) isa T
            @test !is_symbolic(node)
            @test is_symbolic(f(u))            # symbolic through the wrapper
        end

        # Every node that names a direction answers for it, not only the differences. The
        # averages, the shift and the restriction had no method until the precompilation
        # workload called the trait across every node kind and met a MethodError on the
        # averages — the kind of gap nothing else was going to find, since the trait has no
        # caller in the unlocked code.
        for (f, dim) in ((M₋ₓ, 1), (M₊ₓ, 1), (M₋ᵧ, 2), (M₊ᵧ, 2), (M₋₂, 3), (M₊₂, 3))
            @test get_innermost_dim(f(id)) == dim
        end
        @test get_innermost_dim(Bramble.shift_op(id, 2, 3)) == 2
        @test get_innermost_dim(restrict_to(:interior, M₋ᵧ(id))) == 2
        @test get_innermost_dim(restrict_to(:bottom, D₋ₓ(id))) == 1
        @test M₋ₓ(id) isa Bramble.AverageNode
        @test M₊ᵧ(id) isa Bramble.AverageNode

        # the three without a matrix form are grouped, so anything reading only the
        # direction covers all of them
        @test Dcₓ(id) isa ExtendedDifferenceNode
        @test Dstar₊ᵧ(id) isa ExtendedDifferenceNode
        @test Dₕₓ(id) isa ExtendedDifferenceNode
        @test !(jumpₓ(id) isa ExtendedDifferenceNode)   # the jump has a matrix

        # and the block walk reaches its leaf through every one of them
        p, q = IndexedTrialFunction{2}(2), IndexedTestFunction{2}(3)
        for f in (jumpₓ, Dcₓ, Dstar₊ₓ, Dₕₓ)
            @test find_trial_component(f(p)) == 2
            @test find_test_component(f(q)) == 3
            @test find_trial_component(restrict_to(:interior, f(p))) == 2
        end
    end

    @testset "over a composite space" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 5), (true, false))
        Wₕ = gridspace(Ωₕ)
        Vₕ = gridspace(Ωₕ, Val(3))
        idv = IdentityOperator(Vₕ)
        id = IdentityOperator(Wₕ)

        for f in (jumpₓ, jumpᵧ, Dcₓ, Dcᵧ, Dstar₊ₓ, Dₕₓ, Dₕᵧ)
            @test f(idv) isa LazyOp{2}
            @test resolve_ast(f(idv)) isa LazyOp{2}
        end
        @test ∇ₕ(idv) isa NTuple{2, CrossWeightedDifference}
        @test jumpₕ(idv) isa NTuple{2, JumpNode}

        # the offsets are grid coordinates, which the components share, so the stencil is
        # the same one the scalar space gives
        lin = LinearIndices(indices(mesh(Vₕ)))
        I = CartesianIndex(3, 3)
        for f in (jumpₓ, Dcₓ, Dstar₊ᵧ, Dₕₓ)
            @test local_stencil(f(idv), Vₕ, I, nothing, lin[I]) ==
                  local_stencil(f(id), Wₕ, I, nothing, lin[I])
        end
    end
end
