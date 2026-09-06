using Test
using Bramble
# Named, not bare: a bare `using LinearAlgebra` here would export its own `×` (cross
# product) into this shared Main session, ambiguous with Bramble's own `×` for every
# file included afterward -- the same trap flagged elsewhere in this suite.
using LinearAlgebra: Diagonal, dot, I
using SparseArrays
using ForwardDiff
using Bramble: CompositeGridSpace, form, assemble, assemble!, assemble_parallel!,
               allocate_system_matrix, weights, Innerh, Innerplus, TrialFunction,
               TestFunction, InterpolationNode, AbsoluteColumn, _trial_column,
               _all_trial_interpolated, _check_interp_spaces, stencil_shift_trait,
               TranslationInvariantStencil, PointDependentStencil, shifted_inner_stencil,
               shift_stencil, local_stencil, markers, LinearProduct, shift_op, jumpₓ,
               resolve_form_ast

# Standalone runner fallback
if !@isdefined(alloc_test)
    @inline function alloc_test(f::F, args...; kwargs...) where {F}
        f(args...; kwargs...)
        return @allocated(f(args...; kwargs...))
    end
end

if !@isdefined(var"@test_allocs")
    macro test_allocs(call_expr)
        if Meta.isexpr(call_expr, :call)
            fn = call_expr.args[1]
            args = call_expr.args[2:end]
            quote
                @test alloc_test($(esc(fn)), $(map(esc, args)...)) == 0
            end
        else
            quote
                let
                    $(esc(call_expr))
                    @test (@allocated $(esc(call_expr))) == 0
                end
            end
        end
    end
end

# The interpolation operator: `πₕ(Wsrc, u)` over a trial function, as opposed to
# `πₕ(uₕ)` over a grid function whose values are known (test/form/interpolation.jl).
#
# What makes this its own mechanism rather than another wrapper is that its stencil entries
# name **absolute columns** of `Wsrc` (the `2ᴰ` corners `locate_cell` picks out for the point
# being visited), where every other node's entries are offsets from that point. Two
# consequences, and this file exists to hold both to their word:
#
#  1. The assembled block is exactly `interpolation_matrix`'s own `P`, times whatever the
#     surrounding operators and quadrature weight amount to. That is a matrix identity, so it
#     is testable to the last bit rather than by "is finite and the right shape", and it is
#     checked against `interpolation_matrix` rather than against a hand-written expectation,
#     since the two share `_interp_cell_frac` and would have to *both* be wrong to agree with
#     a wrong third answer.
#
#  2. An outer operator can no longer produce a neighbour's contribution by relabelling
#     offsets (`shift_stencil`), which is what every difference, average, shift and jump did
#     unconditionally. It has to evaluate the interpolation again at the neighbour's own
#     point. `shifted_inner_stencil` is the one place that choice is made, so the identities
#     for `D₋ₓ(πₕ(…))` and friends are what test the re-evaluation path, and a direct
#     comparison against `shift_stencil` is what tests that the ordinary path did not change.

# The two quadrature weights as matrices, so a form's assembly can be written as the matrix
# product it is meant to equal.
Hh(W) = Diagonal(collect(weights(W, Innerh())))
Hp(W, d) = Diagonal(collect(weights(W, Innerplus(), d)))

@testset "Interpolation operator" begin
    @testset "1D matrix identities" begin
        Ω = domain(interval(0.0, 1.0))
        Wt = gridspace(mesh(Ω, 11, true))      # test space: the mesh integrated over
        Ws = gridspace(mesh(Ω, 7, true))       # source space: where the unknown lives
        P = interpolation_matrix(Wt, Ws)
        Dx, Mx = D₋ₓ(Wt), M₋ₓ(Wt)
        # `Hₕ · P`, the cross-mesh mass matrix: the term interpolation on trial spaces exists for
        A = assemble(form(Ws, Wt, (u, v) -> innerₕ(πₕ(Ws, u), v)))
        @test size(A) == (ndofs(Wt), ndofs(Ws))
        @test A ≈ Hh(Wt) * P

        # `Dₓᵀ H₊ Dₓ P`: an operator *outside* the interpolation, differencing on the mesh
        # being integrated over. This is the re-evaluation path: relabelling the absolute
        # columns would have named the wrong ones, or failed to name any.
        A = assemble(form(Ws, Wt, (u, v) -> inner₊ₓ(D₋ₓ(πₕ(Ws, u)), D₋ₓ(v))))
        @test A ≈ Dx' * Hp(Wt, 1) * Dx * P

        # the average, whose mask is ½ rather than 1/h: a different weight over the same shift
        A = assemble(form(Ws, Wt, (u, v) -> inner₊ₓ(M₋ₓ(πₕ(Ws, u)), v)))
        @test A ≈ Hp(Wt, 1) * Mx * P

        # `D₊ₓ`, which shifts the other way
        A = assemble(form(Ws, Wt, (u, v) -> innerₕ(D₊ₓ(πₕ(Ws, u)), v)))
        @test A ≈ Hh(Wt) * D₊ₓ(Wt) * P

        # a shift by more than one point, which is the `Int` delta rather than a `Val`
        A = assemble(form(Ws, Wt, (u, v) -> innerₕ(shift_op(πₕ(Ws, u), 1, 2), v)))
        Pshift = interpolation_matrix(Wt, Ws)
        # row I reads the interpolant two points along, clamped at the far end
        pts = points(mesh(Wt))
        Pexpect = zero(Matrix(P))
        for i in eachindex(pts)
            j = min(i + 2, length(pts))
            Pexpect[i, :] .= Matrix(Pshift)[j, :]
        end
        @test A ≈ Hh(Wt) * Pexpect

        # a jump, whose reach is a spacing rather than a mask
        A = assemble(form(Ws, Wt, (u, v) -> innerₕ(jumpₓ(πₕ(Ws, u)), v)))
        @test A ≈ Hh(Wt) * Matrix(jumpₓ(Wt)) * P

        # composition of the two kinds of coefficient: a constant, a live `Ref`, a sum
        β = Ref(3.0)
        a = form(Ws, Wt,
            (u, v) -> 2.0 * innerₕ(πₕ(Ws, u), v) +
                      β * inner₊ₓ(D₋ₓ(πₕ(Ws, u)), D₋ₓ(v)))
        A = assemble(a)
        @test A ≈ 2.0 * Hh(Wt) * P + 3.0 * Dx' * Hp(Wt, 1) * Dx * P
        # the `Ref` stays live, and refilling costs nothing
        β[] = -1.0
        @test_allocs assemble!(A, a)
        @test A ≈ 2.0 * Hh(Wt) * P - Dx' * Hp(Wt, 1) * Dx * P

        # a grid-function coefficient on the interpolated side, read on the *test* mesh
        cₕ = Rₕ(Wt, x -> 1 + x^2)
        A = assemble(form(Ws, Wt, (u, v) -> innerₕ(cₕ * πₕ(Ws, u), v)))
        @test A ≈ Hh(Wt) * Diagonal(values(cₕ)) * P
    end

    @testset "Higher dimensions" begin
        # the `2ᴰ`-corner blend, where the 1D case only exercises two corners
        for (nt, ns) in (((7, 5), (4, 3)), ((5, 4, 3), (3, 3, 2)))
            D = length(nt)
            Ω = D == 2 ? domain(box((0.0, 0.0), (1.0, 1.0))) :
                domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
            Wt = gridspace(mesh(Ω, nt, ntuple(_ -> true, D)))
            Ws = gridspace(mesh(Ω, ns, ntuple(_ -> true, D)))
            P = interpolation_matrix(Wt, Ws)
            @test nnz(P) == ndofs(Wt) * 2^D

            A = assemble(form(Ws, Wt, (u, v) -> innerₕ(πₕ(Ws, u), v)))
            @test size(A) == (ndofs(Wt), ndofs(Ws))
            @test A ≈ Hh(Wt) * P

            A = assemble(form(Ws, Wt, (u, v) -> inner₊ₓ(D₋ₓ(πₕ(Ws, u)), D₋ₓ(v))))
            Dx = D₋ₓ(Wt)
            @test A ≈ Dx' * Hp(Wt, 1) * Dx * P
        end
    end

    @testset "Non-uniform meshes" begin
        # `_interp_cell_frac` reads the mesh's own point coordinates rather than assuming a
        # step, so a non-uniform pair is not a special case, but it is the case that would
        # catch an implementation that quietly divided by a nominal `h`.
        Ω = domain(interval(0.0, 1.0))
        Wt = gridspace(mesh(Ω, 13, false))
        Ws = gridspace(mesh(Ω, 8, false))
        P = interpolation_matrix(Wt, Ws)
        @test assemble(form(Ws, Wt, (u, v) -> innerₕ(πₕ(Ws, u), v))) ≈ Hh(Wt) * P

        Dx = D₋ₓ(Wt)
        A = assemble(form(Ws, Wt, (u, v) -> inner₊ₓ(D₋ₓ(πₕ(Ws, u)), D₋ₓ(v))))
        @test A ≈ Dx' * Hp(Wt, 1) * Dx * P
    end

    @testset "Same-mesh interpolation is the identity" begin
        # interpolating a space onto its own mesh must reproduce the un-interpolated form
        # exactly: every point sits on a corner, so `P` is `I`. Worth pinning because it is
        # the one case where the two code paths are answerable to the same number.
        Wₕ = gridspace(mesh(domain(interval(0.0, 1.0)), 9, false))
        @test interpolation_matrix(Wₕ, Wₕ) ≈ I

        plain = assemble(form(Wₕ, Wₕ, (u, v) -> inner₊ₓ(D₋ₓ(u), D₋ₓ(v)) + innerₕ(u, v)))
        interp = assemble(form(Wₕ, Wₕ,
            (u, v) -> inner₊ₓ(D₋ₓ(πₕ(Wₕ, u)), D₋ₓ(v)) + innerₕ(πₕ(Wₕ, u), v)))
        @test interp ≈ plain
    end

    @testset "Action on a grid function" begin
        # the assembled matrix applied to a source vector is the weighted interpolant: the
        # numeric `πₕ` (space/operators/interpolation.jl) computing the same thing by an
        # entirely different route, which is the check that the two layers agree.
        Ω = domain(interval(0.0, 1.0))
        Wt = gridspace(mesh(Ω, 11, true))
        Ws = gridspace(mesh(Ω, 7, true))
        A = assemble(form(Ws, Wt, (u, v) -> innerₕ(πₕ(Ws, u), v)))

        for f in (x -> sin(3x) + 1, x -> 2x - 1, x -> exp(-x))
            uₛ = Rₕ(Ws, f)
            @test A * values(uₛ) ≈ collect(weights(Wt, Innerh())) .* values(πₕ(Wt, uₛ))
        end

        # and the interpolant is exact on an affine function, so there the action is the
        # weighted restriction of `f` itself: an oracle that does not go through `πₕ` at all
        uₛ = Rₕ(Ws, x -> 2x - 1)
        @test A * values(uₛ) ≈ collect(weights(Wt, Innerh())) .* values(Rₕ(Wt, x -> 2x - 1))
    end

    @testset "Composite blocks" begin
        # the motivating case: two leaves over different meshes, coupled. Cross-mesh terms refuse
        # this by name unless something says how to map between the two index spaces; `πₕ` is
        # what specifies that mapping.
        Ω = domain(interval(0.0, 1.0))
        Wbig = gridspace(mesh(Ω, 9, true))
        Wsmall = gridspace(mesh(Ω, 5, true))
        Vh = CompositeGridSpace((Wbig, Wsmall))
        nb, ns = ndofs(Wbig), ndofs(Wsmall)

        # trial on the small leaf, test on the big one: the direction that used to throw from
        # inside `sparse!`
        A = assemble(form(Vh, Vh, (u, v) -> innerₕ(πₕ(Wsmall, u(2)), v(1))))
        @test size(A) == (nb + ns, nb + ns)
        @test A[1:nb, (nb + 1):(nb + ns)] ≈ Hh(Wbig) * interpolation_matrix(Wbig, Wsmall)
        # and nothing landed anywhere else: the direction that used to write silently wrong
        # columns would have shown up here
        @test iszero(A[1:nb, 1:nb])
        @test iszero(A[(nb + 1):(nb + ns), :])

        # trial on the big leaf, test on the small one: the direction that used to assemble
        # in-range but wrong
        A = assemble(form(Vh, Vh, (u, v) -> innerₕ(πₕ(Wbig, u(1)), v(2))))
        @test A[(nb + 1):(nb + ns), 1:nb] ≈ Hh(Wsmall) * interpolation_matrix(Wsmall, Wbig)
        @test iszero(A[1:nb, :])
        @test iszero(A[(nb + 1):(nb + ns), (nb + 1):(nb + ns)])

        # an interpolating off-diagonal term alongside ordinary diagonal ones: the diagonal
        # blocks must be untouched by the exemption the interpolating term gets
        a = form(Vh, Vh,
            (u, v) -> innerₕ(u(1), v(1)) + innerₕ(u(2), v(2)) +
                      innerₕ(πₕ(Wsmall, u(2)), v(1)))
        A = assemble(a)
        @test A[1:nb, 1:nb] ≈ Hh(Wbig)
        @test A[(nb + 1):(nb + ns), (nb + 1):(nb + ns)] ≈ Hh(Wsmall)
        @test A[1:nb, (nb + 1):(nb + ns)] ≈ Hh(Wbig) * interpolation_matrix(Wbig, Wsmall)
        @test_allocs assemble!(A, a)

        # an operator outside the interpolation, on a composite block
        Dx = D₋ₓ(Wbig)
        A = assemble(form(Vh, Vh, (u, v) -> inner₊ₓ(D₋ₓ(πₕ(Wsmall, u(2))), D₋ₓ(v(1)))))
        @test A[1:nb, (nb + 1):(nb + ns)] ≈
              Dx' * Hp(Wbig, 1) * Dx * interpolation_matrix(Wbig, Wsmall)
    end

    @testset "Serial and parallel agree" begin
        Ω = domain(interval(0.0, 1.0))
        Wt = gridspace(mesh(Ω, 21, true))
        Ws = gridspace(mesh(Ω, 9, true))
        a = form(Ws, Wt,
            (u, v) -> inner₊ₓ(D₋ₓ(πₕ(Ws, u)), D₋ₓ(v)) + innerₕ(πₕ(Ws, u), v))

        As = assemble(a)
        Ap = allocate_system_matrix(a)
        assemble_parallel!(Ap, a)
        @test As ≈ Ap

        # Bilinear colouring (`_colour_strides(stencil_offsets(ast))`) reads only the *test*
        # side of the stencil (disjoint rows imply disjoint entries regardless of what
        # columns do), which is why an interpolation on the trial side needs no colouring
        # change. This is what would fail if that reasoning were wrong: two threads racing
        # on one entry would give a wrong sum, not an error.
        assemble_parallel!(Ap, a)
        @test As ≈ Ap
        @test_allocs assemble!(As, a)
    end

    @testset "Refusals" begin
        Ω = domain(interval(0.0, 1.0))
        Wt = gridspace(mesh(Ω, 11, true))
        Ws = gridspace(mesh(Ω, 7, true))

        # an operator *inside* the interpolation would difference on the source mesh, which
        # is a different operator from `D₋ₓ(πₕ(…))` and is not implemented; refused rather
        # than quietly treated as the one that is
        @test_throws ArgumentError πₕ(Ws, D₋ₓ(TrialFunction{1}()))
        @test_throws ArgumentError πₕ(Ws, M₋ₓ(TrialFunction{1}()))
        # and the test function has nothing to interpolate: the rows are the mesh being
        # integrated over, so interpolating them is not a thing to ask for
        @test_throws ArgumentError πₕ(Ws, TestFunction{1}())

        # `πₕ` given a space that is not the trial function's: the columns it names are
        # numbered in that space, so pairing it with the wrong one writes into the wrong part
        # of the block: the same silent-wrong answer cross-mesh blocks refuse, so it is refused too
        @test_throws ArgumentError assemble(form(Ws, Wt, (u, v) -> innerₕ(πₕ(Wt, u), v)))
        @test_throws ArgumentError allocate_system_matrix(form(
            Ws, Wt, (u, v) -> innerₕ(πₕ(Wt, u), v)))

        Wbig = gridspace(mesh(Ω, 9, true))
        Wsmall = gridspace(mesh(Ω, 5, true))
        Vh = CompositeGridSpace((Wbig, Wsmall))
        # leaf 2 is the small space; naming the big one interpolates from the wrong leaf
        @test_throws ArgumentError assemble(form(
            Vh, Vh, (u, v) -> innerₕ(πₕ(Wbig, u(2)), v(1))))
        @test_throws ArgumentError assemble_parallel!(
            spzeros(ndofs(Vh), ndofs(Vh)),
            form(Vh, Vh, (u, v) -> innerₕ(πₕ(Wbig, u(2)), v(1))))

        # a cross-mesh block with *no* interpolation is still refused, at every entry point:
        # the exemption is for the term that says how to map, not for cross-mesh generally
        for g in ((u, v) -> innerₕ(u(2), v(1)), (u, v) -> innerₕ(u(1), v(2)))
            @test_throws ArgumentError assemble(form(Vh, Vh, g))
            @test_throws ArgumentError allocate_system_matrix(form(Vh, Vh, g))
        end
        @test_throws ArgumentError assemble(form(Ws, Wt, (u, v) -> innerₕ(u, v)))

        # a sum in which only one summand interpolates does not exempt the other
        @test_throws ArgumentError assemble(form(Vh, Vh,
            (u, v) -> innerₕ(πₕ(Wsmall, u(2)), v(1)) + innerₕ(u(1), v(2))))
    end

    @testset "Mixed sums are refused" begin
        # A term contributing *both* absolute columns and ordinary offsets: the offsets are
        # still read out of the index space being walked, so the two leaves still have to
        # share one. The first version of this file asked only whether an interpolation
        # appeared anywhere in the term, which exempted the bare `u` along with it; and in
        # the direction where the trial space is the larger one, the bare `u`'s column landed
        # *in range* and simply wrong. Measured at 0.25 absolute error on a 5×9 block, with
        # no error raised: this is the silent-wrong answer cross-mesh blocks refuse.
        Ω = domain(interval(0.0, 1.0))
        Wbig = gridspace(mesh(Ω, 9, true))
        Wsml = gridspace(mesh(Ω, 5, true))
        Wt = gridspace(mesh(Ω, 11, true))

        # the silent direction, and the loud one
        @test_throws ArgumentError assemble(form(
            Wbig, Wsml, (u, v) -> innerₕ(πₕ(Wbig, u) + u, v)))
        @test_throws ArgumentError assemble(form(
            Wsml, Wt, (u, v) -> innerₕ(πₕ(Wsml, u) + u, v)))
        # under an outer operator, and with the plain factor on the left
        @test_throws ArgumentError assemble(form(
            Wbig, Wsml, (u, v) -> inner₊ₓ(D₋ₓ(πₕ(Wbig, u) + u), D₋ₓ(v))))
        @test_throws ArgumentError assemble(form(
            Wbig, Wsml, (u, v) -> innerₕ(u + πₕ(Wbig, u), v)))
        # and at the pattern entry point, not only at assembly
        @test_throws ArgumentError allocate_system_matrix(form(
            Wbig, Wsml, (u, v) -> innerₕ(πₕ(Wbig, u) + u, v)))

        # two interpolations in one term are both checked, so one from the wrong space is
        # caught wherever it sits
        @test_throws ArgumentError assemble(form(
            Wbig, Wt, (u, v) -> innerₕ(πₕ(Wbig, u) + πₕ(Wsml, u), v)))
        @test_throws ArgumentError assemble(form(
            Wbig, Wt, (u, v) -> innerₕ(πₕ(Wbig, u), v) +
                                inner₊ₓ(D₋ₓ(πₕ(Wsml, u)), D₋ₓ(v))))

        # what must keep working: a sum of interpolations from the *same* space, and a mix on
        # a single mesh, where the offsets are meaningful and `P` is the identity
        P = interpolation_matrix(Wt, Wbig)
        A = assemble(form(Wbig, Wt, (u, v) -> innerₕ(πₕ(Wbig, u) + πₕ(Wbig, u), v)))
        @test A ≈ 2 * Hh(Wt) * P
        A = assemble(form(Wt, Wt, (u, v) -> innerₕ(πₕ(Wt, u) + u, v)))
        @test A ≈ 2 * Hh(Wt)
    end

    @testset "Traits and stencils" begin
        Ω = domain(interval(0.0, 1.0))
        Wt = gridspace(mesh(Ω, 11, true))
        Ws = gridspace(mesh(Ω, 7, true))
        u, v = TrialFunction{1}(), TestFunction{1}()
        node = πₕ(Ws, u)

        @test node isa InterpolationNode

        # whether *every* trial column the term contributes is an absolute one, under any
        # tower of wrappers. Not "does an interpolation appear anywhere" (see the mixed-sum
        # testset below for why that distinction is the whole ballgame).
        @test _all_trial_interpolated(node)
        @test _all_trial_interpolated(D₋ₓ(M₋ₓ(node)))
        @test _all_trial_interpolated(2.0 * node)
        @test _all_trial_interpolated(node + node)
        @test _all_trial_interpolated(innerₕ(node, v))
        @test !_all_trial_interpolated(u)
        @test !_all_trial_interpolated(D₋ₓ(u))
        @test !_all_trial_interpolated(innerₕ(u, v))
        # one summand interpolating is not enough, in either position
        @test !_all_trial_interpolated(node + u)
        @test !_all_trial_interpolated(u + node)
        @test !_all_trial_interpolated(D₋ₓ(node + u))
        @test !_all_trial_interpolated(innerₕ(node + u, v))
        # a node contributing no trial column at all answers vacuously: there is nothing
        # there that would need a mesh correspondence
        @test _all_trial_interpolated(v)
        @test _all_trial_interpolated(πₕ(Rₕ(Ws, x -> x)))
        # a linear product contracts its left factor away, so it contributes no column
        @test _all_trial_interpolated(innerₕ(πₕ(Rₕ(Ws, x -> x)), v))
        @test innerₕ(πₕ(Rₕ(Ws, x -> x)), v) isa LinearProduct

        # every interpolation is validated against the leaf it writes into, not just the
        # first one a walk finds
        @test _check_interp_spaces(node, Ws) === nothing
        @test _check_interp_spaces(D₋ₓ(node), Ws) === nothing
        @test _check_interp_spaces(u, Ws) === nothing
        @test_throws ArgumentError _check_interp_spaces(node, Wt)
        @test_throws ArgumentError _check_interp_spaces(D₋ₓ(M₋ₓ(node)), Wt)
        @test_throws ArgumentError _check_interp_spaces(πₕ(Wt, u) + node, Ws)
        @test_throws ArgumentError _check_interp_spaces(node + πₕ(Wt, u), Ws)

        # the shift trait: an interpolation cannot be relabelled, a trial function can
        @test stencil_shift_trait(node) isa PointDependentStencil
        @test stencil_shift_trait(D₋ₓ(node)) isa PointDependentStencil
        @test stencil_shift_trait(u) isa TranslationInvariantStencil
        @test stencil_shift_trait(D₋ₓ(M₋ₓ(u))) isa TranslationInvariantStencil
        @test stencil_shift_trait(u + node) isa PointDependentStencil
        @test stencil_shift_trait(u + D₋ₓ(u)) isa TranslationInvariantStencil

        # `shifted_inner_stencil` must be exactly `shift_stencil` on the translation-invariant
        # path: every operator in the package goes through it now, so this is what says the
        # refactor changed nothing for them
        Ωₕ = mesh(Wt)
        mk = markers(Ωₕ)
        I = CartesianIndex(5)
        for op in (u, D₋ₓ(u), M₋ₓ(D₊ₓ(u)), 2.0 * u)
            inner = local_stencil(op, Wt, I, mk, 5)
            for δ in (Val(-1), Val(1), 2)
                @test shifted_inner_stencil(op, inner, Wt, I, mk, Val(1), δ) ==
                      shift_stencil(inner, Val(1), δ)
            end
        end

        # the interpolation's own stencil: `2ᴰ` entries naming absolute columns, weights
        # summing to one, which is what keeps the interpolant from overshooting
        st = local_stencil(node, Wt, I, mk, 5)
        @test length(st) == 2
        @test all(e -> e[1] isa AbsoluteColumn, st)
        @test sum(e -> e[2], st) ≈ 1
        # and they are the columns `interpolation_matrix` puts in that row
        P = interpolation_matrix(Wt, Ws)
        @test sort([e[1].col for e in st]) == sort(findnz(P[5, :])[1])
        for e in st
            @test P[5, e[1].col] ≈ e[2]
        end

        # `_trial_column` resolves the two kinds of entry: an offset against the index space
        # being walked, dropped when it falls off the grid, and an absolute column taken as
        # it stands
        li = LinearIndices(indices(Ωₕ))
        @test _trial_column(li, I, (0,)) == 5
        @test _trial_column(li, I, (-1,)) == 4
        @test _trial_column(li, CartesianIndex(1), (-1,)) == 0
        @test _trial_column(li, CartesianIndex(ndofs(Wt)), (1,)) == 0
        @test _trial_column(li, I, AbsoluteColumn(3)) == 3
        # an absolute column is not bounds-checked against the walked space, on purpose: it
        # numbers the *other* one, which may be larger
        @test _trial_column(li, CartesianIndex(1), AbsoluteColumn(ndofs(Wt) + 4)) ==
              ndofs(Wt) + 4
    end

    @testset "Element type" begin
        # the interpolation's weights come from the mesh's own coordinates, so a `Float32`
        # space must not be widened to `Float64` by them; the same rule every other stencil
        # leaf follows (test/form/common.jl, "Element type preservation")
        for T in (Float32, Float64)
            Ω = domain(interval(T(0), T(1)))
            Ωt, Ωs = mesh(Ω, 11, true), mesh(Ω, 7, true)
            @test eltype(Ωt) === T
            Wt, Ws = gridspace(Ωt), gridspace(Ωs)
            @test eltype(interpolation_matrix(Wt, Ws)) === T

            for f in ((u, v) -> innerₕ(πₕ(Ws, u), v),
                (u, v) -> inner₊ₓ(D₋ₓ(πₕ(Ws, u)), D₋ₓ(v)),
                (u, v) -> innerₕ(M₋ₓ(πₕ(Ws, u)), v))
                @test eltype(assemble(form(Ws, Wt, f))) === T
            end
        end
    end

    @testset "Differentiation" begin
        # A coefficient in the integrand, differentiated through the assembly: the block is
        # `H·diag(c)·P`, so `d sum(A) / d cᵢ` is `Hᵢᵢ` times row `i` of `P` summed. Checked
        # against that, not against itself, so a gradient of the wrong thing cannot pass.
        #
        # What has to hold for this to work is the element-type-from-the-data rule: the
        # interpolation's own weights come from the mesh's coordinates and are ordinary
        # floats, so they must promote against a `Dual` coefficient rather than pin the
        # matrix to `Float64`. (The mesh geometry itself is not the differentiation variable
        # here: `interpolation_matrix` over a real-valued mesh stays real-valued.)
        Ω = domain(interval(0.0, 1.0))
        Wt = gridspace(mesh(Ω, 11, true))
        Ws = gridspace(mesh(Ω, 7, true))
        P = interpolation_matrix(Wt, Ws)
        c0 = fill(1.0, ndofs(Wt))
        want = collect(weights(Wt, Innerh())) .* vec(sum(Matrix(P), dims = 2))

        f(w) = form(Ws, Wt, (u, v) -> innerₕ(Bramble.element(Wt, w) * πₕ(Ws, u), v))
        @test ForwardDiff.gradient(w -> sum(assemble(f(w))), c0) ≈ want

        # the matrix has to be able to hold a `Dual` at all
        @test eltype(assemble(f(ForwardDiff.Dual.(c0, 1.0)))) <: ForwardDiff.Dual

        # and the in-place path, which allocates the matrix separately
        @test ForwardDiff.gradient(c0) do w
            a = f(w)
            A = allocate_system_matrix(a)
            assemble!(A, a)
            sum(A)
        end ≈ want

        # through the re-evaluation path, where the outer difference evaluates the
        # interpolation twice rather than relabelling it.
        #
        # Contracted against two fixed vectors rather than summed: `sum(Dₓᵀ H₊ diag(c) Dₓ P)`
        # is identically zero whatever `c` is, because each column of `Dₓᵀ …` telescopes away,
        # so a gradient of it is zero for reasons that have nothing to do with this operator
        # and would pass against any implementation at all. `rᵀ A s` does not degenerate:
        # `A = Dₓᵀ H₊ diag(c) Dₓ P`, so `d(rᵀAs)/dcᵢ = (Dₓr)ᵢ · H₊ᵢᵢ · (DₓPs)ᵢ`.
        fd(w) = form(Ws, Wt,
            (u, v) -> inner₊ₓ(Bramble.element(Wt, w) * D₋ₓ(πₕ(Ws, u)), D₋ₓ(v)))
        r = [inv(1.0 + i) for i in 1:ndofs(Wt)]
        sv = [cospi(i / 5) for i in 1:ndofs(Ws)]
        Dx = D₋ₓ(Wt)
        want_d = (Dx * r) .* collect(weights(Wt, Innerplus(), 1)) .* (Dx * (P * sv))
        @test ForwardDiff.gradient(w -> dot(r, assemble(fd(w)) * sv), c0) ≈ want_d
        # and the oracle is not itself zero, which is the mistake above
        @test maximum(abs, want_d) > 0.1
    end

    @testset "Resolved AST round-trips" begin
        # `resolve_form_ast` rebuilds the tree once, at `form(...)`, and every wrapper has to
        # rebuild itself with the resolved inner operator, including this one, or the node
        # would be dropped and the term would silently assemble as if un-interpolated
        Ω = domain(interval(0.0, 1.0))
        Wt = gridspace(mesh(Ω, 11, true))
        Ws = gridspace(mesh(Ω, 7, true))
        cₕ = Rₕ(Wt, x -> 1 + x)
        a = form(Ws, Wt, (u, v) -> innerₕ(cₕ * πₕ(Ws, u), v))
        @test _all_trial_interpolated(a.ast)
        @test _check_interp_spaces(a.ast, Ws) === nothing
        @test_throws ArgumentError _check_interp_spaces(a.ast, Wt)
        @test assemble(a) ≈ Hh(Wt) * Diagonal(values(cₕ)) * interpolation_matrix(Wt, Ws)
    end
end
