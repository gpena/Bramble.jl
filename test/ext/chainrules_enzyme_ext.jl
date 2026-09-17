module ExtChainRulesEnzymeExtTests

using Test
using Bramble
using ChainRulesCore
using SparseArrays: SparseMatrixCSC, nnz
using LinearAlgebra: I
using ..TestUtils: _fd, _have

# Enzyme composition for `BrambleChainRulesExt`'s `pde_solve` rrule. Behind the "ad"/"full"
# groups, same reasoning and the same `_have`/`@test_skip` idiom `autodiff_heavy.jl` already
# uses for Enzyme/Mooncake: neither is a `test/Project.toml` dependency (kept out of the
# ordinary dependency graph so an every-push run never pays their ~30 s first-call
# compilation), and `Weekly.yml`'s "add the expensive differentiation backends" step installs
# both at CI runtime for exactly this group.
#
# `test/ext/chainrules_ext.jl` already checks the rrule's own math against finite differences
# without any AD package; what is left to check here is that a real reverse-mode backend
# actually *reaches* that rule and gets the right answer end to end, through
# `assemble(...; dirichlet = θ) -> pde_solve -> J`, including the Dirichlet-value case
# `chainrules_ext.jl` could not check against `ForwardDiff` (UMFPACK's `Float64`-only sparse
# factorisation, documented in `docs/src/tutorials/autodiff.md` §5) -- reverse-mode through
# the hand-written adjoint has no such restriction, since it never asks UMFPACK to factor
# anything but a plain `Float64` matrix.
#
# Enzyme reaches `pde_solve` through `BrambleEnzymeExt`'s own native `EnzymeRules` rule now,
# not through `Enzyme.@import_rrule` (gpena/Bramble.jl#240): the import bridge corrupts the
# `SparseMatrixCSC` shadow whenever the cotangent carries an explicit zero -- `nzval` is left
# shorter than `colptr`/`rowval` claim -- and a homogeneous Dirichlet problem produces such a
# zero routinely, so the gradient came back confidently wrong rather than failing. The
# `nnz`/buffer-consistency check below pins that, and no test in this file may call
# `@import_rrule`: doing so defines a second rule for the same signature.


@testset "BrambleChainRulesExt + Enzyme" begin
    @testset "Enzyme, through BrambleEnzymeExt's native rule" begin
        if _have(:Enzyme)
            @eval import Enzyme

            Ωₕ = Bramble.mesh(Bramble.domain(Bramble.interval(0.0, 1.0)), 21, true)
            Wₕ = gridspace(Ωₕ)
            a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
            fₕ = Rₕ(Wₕ, x -> pi^2 * sinpi(x[1]))
            l_fixed = form(Wₕ, v -> innerₕ(fₕ, v))

            function loss(θ::Real)
                gₕ = Bramble.element(Wₕ, zero(θ))
                l = form(Wₕ, v -> innerₕ(gₕ, v))
                A, F = assemble(a, l; dirichlet = :boundary => x -> θ)
                u = Bramble.pde_solve(A, F)
                return sum(abs2, u)
            end

            θ0 = 0.7
            d_fd = _fd(loss, θ0)

            # Closures reaching a grid space/mesh need `set_runtime_activity` -- the same
            # documented Enzyme quirk `autodiff_heavy.jl`'s own Enzyme testset already pins
            # for the ordinary (non-solve) path, not a defect this rule introduces.
            # `Enzyme.Const(loss)`: the closure captures `Wₕ`/`a` (a grid space and a
            # bilinear form), which Enzyme cannot prove read-only on its own -- the same
            # `function_annotation = Const` requirement `autodiff_heavy.jl`'s own Enzyme
            # testset documents, spelled through Enzyme's direct API rather than
            # `DifferentiationInterface`'s keyword.
            mode = Enzyme.set_runtime_activity(Enzyme.Reverse)
            d_enzyme = Enzyme.gradient(mode, Enzyme.Const(loss), θ0)[1]
            @test d_enzyme≈d_fd rtol=1e-4

            # `symmetrize = true` rewrites both `A` and `F` after assembly, restoring
            # symmetry once the constrained rows are cleared: the adjoint has to stay correct
            # across that rewrite too.
            function loss_symmetrized(θ::Real)
                gₕ = Bramble.element(Wₕ, zero(θ))
                l = form(Wₕ, v -> innerₕ(gₕ, v))
                A, F = assemble(
                    a, l; dirichlet = :boundary => x -> θ, symmetrize = true
                )
                return sum(abs2, Bramble.pde_solve(A, F))
            end
            @test Enzyme.gradient(mode, Enzyme.Const(loss_symmetrized), θ0)[1] ≈
                  _fd(loss_symmetrized, θ0) rtol=1e-4

            # A gradient with respect to the *operator's own* coefficient: `θ` scales the
            # bilinear form itself, so it reaches `assemble`'s recording engine rather than
            # only `F`, and the form is rebuilt inside the differentiated closure. This used
            # to raise `IllegalTypeAnalysisException` -- `simplify_ast` decided whether to
            # elide a scaling by reading the coefficient's *value*, which put a `Union` of
            # three `BilinearForm` types (`ZeroOperator`/`OperatorScale`/the bare product)
            # into `form`'s return type whenever the compiler could not fold the comparison.
            # Restricting those rules to `Integer` coefficients (gpena/Bramble.jl#240) leaves
            # one concrete type, and the gradient compiles.
            #
            # `Wₕ`/`l_fixed` are testset locals, so the closure captures them at concrete
            # types and this is the fully inferred call site, not the dynamically dispatched
            # one that happened to compile before.
            function loss_coeff(θ::Real)
                aθ = form(Wₕ, Wₕ, (u, v) -> θ * inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
                A, F = assemble(aθ, l_fixed; dirichlet = :boundary => x -> 0.0)
                return sum(abs2, Bramble.pde_solve(A, F))
            end
            @test Enzyme.gradient(mode, Enzyme.Const(loss_coeff), θ0)[1] ≈
                  _fd(loss_coeff, θ0) rtol=1e-4

            # A sum of terms, each with its own runtime coefficient -- the shape an inverse
            # problem actually has. This one needed the *factoring* rule (`c*A + c*B ->
            # c*(A+B)`) restricted too: it decides by comparing the two coefficients, and
            # comparing two runtime numbers leaves the same kind of `Union` behind.
            function loss_coeff_sum(θ::Real)
                aθ = form(
                    Wₕ, Wₕ, (u, v) -> θ * inner₊(∇₋ₕ(u), ∇₋ₕ(v)) + (1 - θ) * innerₕ(u, v)
                )
                A, F = assemble(aθ, l_fixed; dirichlet = :boundary => x -> 0.0)
                return sum(abs2, Bramble.pde_solve(A, F))
            end
            @test Enzyme.gradient(mode, Enzyme.Const(loss_coeff_sum), θ0)[1] ≈
                  _fd(loss_coeff_sum, θ0) rtol=1e-4

            # The gradient is not merely *a* number Enzyme was willing to produce: it has to
            # be the one the adjoint says it is. `θ * a(u, v)` scales `A` by `θ`, so
            # `u(θ) = u(1)/θ` and `J(θ) = ‖u(1)‖²/θ²`, whose derivative is `-2J(θ)/θ`
            # in closed form -- checked here against the analytic value, not only against the
            # finite difference above, which shares the same forward code path.
            @test Enzyme.gradient(mode, Enzyme.Const(loss_coeff), θ0)[1] ≈
                  -2 * loss_coeff(θ0) / θ0 rtol=1e-8

            # Higher dimensions, and a coefficient that varies in space
            # (gpena/Bramble.jl#249). Both used to raise `EnzymeNoTypeError`: the assembly
            # traversal read a stencil entry's offsets and its weight out of the same mixed
            # `Int`/`Float64` tuple, which Enzyme cannot type in a function reading both
            # halves that is not inlined into the differentiated one. `_visit_entries`
            # (form/bilinear_traversal.jl) reads them from `entry_offsets`/`entry_weights`
            # instead. A 1D stiffness term is *not* a regression test for this: it is small
            # enough that the guarded walk it selects compiled either way -- the failures
            # started at 2D stiffness, at 3D, and at a 1D sum of three terms.
            #
            # Kept small (9 points per side in 2D, 5 in 3D): what is being pinned is that
            # the gradient compiles and is right, and every one of these pays Enzyme's
            # compilation on first call.
            I01 = Bramble.interval(0.0, 1.0)
            Ω2 = Bramble.mesh(Bramble.domain(I01 × I01), (9, 9), (true, true))
            W2 = gridspace(Ω2)
            l2 = form(W2, v -> innerₕ(Rₕ(W2, x -> 1.0), v))
            Ω3 = Bramble.mesh(Bramble.domain(I01 × I01 × I01), (5, 5, 5), (true, true, true))
            W3 = gridspace(Ω3)
            l3 = form(W3, v -> innerₕ(Rₕ(W3, x -> 1.0), v))

            function loss_stiff_2d(θ::Real)
                aθ = form(W2, W2, (u, v) -> θ * inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
                A, F = assemble(aθ, l2; dirichlet = :boundary => x -> 0.0)
                return sum(abs2, Bramble.pde_solve(A, F))
            end
            @test Enzyme.gradient(mode, Enzyme.Const(loss_stiff_2d), θ0)[1] ≈
                  _fd(loss_stiff_2d, θ0) rtol=1e-3

            function loss_stiff_3d(θ::Real)
                aθ = form(W3, W3, (u, v) -> θ * inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
                A, F = assemble(aθ, l3; dirichlet = :boundary => x -> 0.0)
                return sum(abs2, Bramble.pde_solve(A, F))
            end
            @test Enzyme.gradient(mode, Enzyme.Const(loss_stiff_3d), θ0)[1] ≈
                  _fd(loss_stiff_3d, θ0) rtol=1e-3

            # A `VectorElement` coefficient inside the form, which is what recovering a
            # diffusion *field* needs: the active values reach the walk through
            # `GridFunctionScale`'s vector rather than through an `OperatorScale`'s number.
            function loss_field_2d(θ::Real)
                κₕ = Bramble.element(W2, θ)
                aκ = form(W2, W2, (u, v) -> inner₊(κₕ * ∇₋ₕ(u), ∇₋ₕ(v)))
                A, F = assemble(aκ, l2; dirichlet = :boundary => x -> 0.0)
                return sum(abs2, Bramble.pde_solve(A, F))
            end
            @test Enzyme.gradient(mode, Enzyme.Const(loss_field_2d), θ0)[1] ≈
                  _fd(loss_field_2d, θ0) rtol=1e-3

            # Two *distinct* coefficient fields on same-shaped terms in one sum. This is the
            # ordinary way to write a two-material model, and it raised
            # `IllegalTypeAnalysisException` until the like-term rule stopped being decided by
            # a run-time comparison (gpena/Bramble.jl#240): `_ast_equal` walked the two
            # `GridFunctionScale` subtrees field by field, the compiler could not fold the
            # result, and `form` inferred as `Union{..., OperatorAdd}, {..., OperatorScale}}`.
            # The rule is gated on `_statically_equal` now, so a data-carrying term is left as
            # written and `form` has one concrete type. Measured on this exact loss: the
            # exception before, this gradient after.
            #
            # Not a duplicate of `loss_field_2d` above: that one has a single coefficient
            # field, so the sum -- and with it the rule that used to fire here -- never enters.
            function loss_two_fields_2d(θ::Real)
                g₁ = Bramble.element(W2, θ)
                g₂ = Bramble.element(W2, 2θ)
                a = form(W2, W2, (u, v) -> innerₕ(g₁ * u, v) + innerₕ(g₂ * u, v))
                A, F = assemble(a, l2; dirichlet = :boundary => x -> 0.0)
                return sum(abs2, Bramble.pde_solve(A, F))
            end
            @test Enzyme.gradient(mode, Enzyme.Const(loss_two_fields_2d), θ0)[1] ≈
                  _fd(loss_two_fields_2d, θ0) rtol=1e-3

            # A 1D sum of three terms: the other shape that used to fail, at the same
            # stencil width as 2D stiffness but reached by summing rather than by dimension.
            function loss_sum_three(θ::Real)
                aθ = form(
                    Wₕ,
                    Wₕ,
                    (u, v) -> θ * (innerₕ(u, v) + inner₊(∇₋ₕ(u), ∇₋ₕ(v)) +
                                   inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
                )
                A, F = assemble(aθ, l_fixed; dirichlet = :boundary => x -> 0.0)
                return sum(abs2, Bramble.pde_solve(A, F))
            end
            @test Enzyme.gradient(mode, Enzyme.Const(loss_sum_three), θ0)[1] ≈
                  _fd(loss_sum_three, θ0) rtol=1e-3

            # The defect the native rule exists to avoid, pinned at the level it actually
            # showed up: `@import_rrule`'s bridge merged the rrule's returned
            # `SparseMatrixCSC` into Enzyme's shadow by dropping the cotangent's explicit
            # zeros from `nzval` alone, leaving `colptr`/`rowval` claiming more entries than
            # `nzval` holds. `Matrix(shadow)` then trips SparseArrays' own `_goodbuffers`
            # assertion, and any gradient read off that shadow is silently wrong. A
            # homogeneous Dirichlet problem supplies such a zero for free: a constrained
            # row's solution entry is exactly its boundary value, so `u[j] == 0` makes a
            # whole column of `-λuᵀ` exactly zero.
            aθ0 = form(Wₕ, Wₕ, (u, v) -> 0.7 * inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
            A0, F0 = assemble(aθ0, l_fixed; dirichlet = :boundary => x -> 0.0)
            @test count(iszero, A0 \ F0) > 0          # the zero that triggered it exists here
            dA = SparseMatrixCSC(
                size(A0, 1), size(A0, 2), copy(A0.colptr), copy(A0.rowval), zeros(nnz(A0))
            )
            solve_loss(Ain, Fin) = sum(abs2, Bramble.pde_solve(Ain, Fin))
            Enzyme.autodiff(
                mode, Enzyme.Const(solve_loss), Enzyme.Active,
                Enzyme.Duplicated(copy(A0), dA), Enzyme.Const(F0)
            )
            @test length(dA.nzval) == dA.colptr[end] - 1
            @test length(dA.rowval) == dA.colptr[end] - 1
            @test Matrix(dA) isa Matrix                # the `_goodbuffers` assertion itself

            # The shadow must also hold the right numbers, not merely a well-formed shape:
            # compared against `BrambleChainRulesExt`'s own pullback, which
            # `test/ext/chainrules_ext.jl` checks against finite differences independently.
            _, pullback = ChainRulesCore.rrule(Bramble.pde_solve, A0, F0)
            _, Ā_rrule, _ = pullback(2 .* (A0 \ F0))
            @test Matrix(dA)≈Matrix(Ā_rrule) rtol=1e-10

            # Pinned failure: handed `A \ F` directly rather than `pde_solve`, Enzyme has no
            # rule to reach and tries to trace into CHOLMOD/UMFPACK's own internals, raising
            # `IllegalTypeAnalysisException` on a `Union` type it cannot strictly alias -- if
            # a future Enzyme release handles that on its own, this test starts failing and
            # says so, the same "pin it so a fix is noticed" discipline `autodiff_heavy.jl`
            # already follows for its own two annotations.
            function loss_no_rule(θ::Real)
                A, F = assemble(a, form(Wₕ, v -> innerₕ(Bramble.element(Wₕ, zero(θ)), v));
                    dirichlet = :boundary => x -> θ)
                return sum(abs2, A \ F)
            end
            @test_throws Exception Enzyme.gradient(mode, Enzyme.Const(loss_no_rule), θ0)
        else
            @test_skip "Enzyme not in this environment"
        end
    end

    @testset "Mooncake: pinned as currently unsupported" begin
        if _have(:Mooncake)
            @eval import Mooncake

            # `@from_rrule` itself only *registers* the bridge, and `build_rrule` only
            # *plans* it -- both succeed regardless. The failure surfaces the first time the
            # rule actually runs and Mooncake needs a real tangent value for the
            # `SparseMatrixCSC` argument: `increment_and_get_rdata!` has no method for that
            # type combination. Pinned rather than skipped: a future `Mooncake.jl` release
            # adding sparse-array tangent support turns this into a silent pass, which is the
            # point of `pde_solve`'s own docstring saying "revisit if a future release adds
            # sparse-array tangent support" -- a newly-failing `@test_throws` (because it
            # stops throwing) is exactly the signal that revisit is due.
            @eval Mooncake.@from_rrule(Mooncake.DefaultCtx,
                Tuple{typeof(Bramble.pde_solve), SparseMatrixCSC, AbstractVector},
                false)
            n = 4
            A = SparseMatrixCSC(Matrix(2.0I, n, n))
            f(F) = sum(abs2, Bramble.pde_solve(A, F))
            rule = Mooncake.build_rrule(f, ones(n))
            @test_throws Exception Mooncake.value_and_gradient!!(rule, f, ones(n))
        else
            @test_skip "Mooncake not in this environment"
        end
    end
end

end # module
