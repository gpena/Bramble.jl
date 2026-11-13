module ExtEnzymeGmgSolveTests

using Test
using Bramble
using LinearAlgebra: I, dot, norm
using Random: Random
using ForwardDiff
import DifferentiationInterface as DI
using ..TestUtils: _have

# Enzyme through `BrambleEnzymeExt`'s `gmg_solve` rules (#1168), split out of
# chainrules_enzyme_ext.jl: the nested Enzyme calls of the coefficient pins took that file's
# child past the 10 GB memory cap. Enzyme is not a test/Project.toml dependency; without it
# each testset skips.

# A coefficient that is a `Real` holding a `Float64`, for the `gmg_solve` rule's shadow walk.
struct WrappedCoefficient <: Real
    κ::Float64
end

# The `build`s of the `gmg_solve` coefficient pins, one closure type per kind: the rule's
# nested Enzyme calls compile once per type, and the Enzyme child has a 10 GB cap.
gmg_scalar(c) = V -> form(V, V, (u, v) -> innerₕ(u, v) + c * inner₊(∇ₕ(u), ∇ₕ(v)))
function gmg_grid(c)
    V -> (κ = element(V, typeof(c));
        fill!(parent(κ), c);
        form(V, V, (u, v) -> innerₕ(u, v) + inner₊(κ * ∇ₕ(u), ∇ₕ(v))))
end
gmg_mixed(c, γ) = V -> form(V, V, (u, v) -> (1 + c) * innerₕ(u, v) + γ[] * inner₊(∇ₕ(u), ∇ₕ(v)))
gmg_mixed_ref(cr, γ) = V -> form(V, V,
    (u, v) -> (1 + cr[]) * innerₕ(u, v) + γ[] * inner₊(∇ₕ(u), ∇ₕ(v)))
# A `Ref` in a global that is not `const`, as a script holds one: a `build` capturing it
# reaches the forward rule with a shadow `===` its primal.
gmg_mixed_γ = Ref(0.7)
# A constant `build` bound to a `const` global reaches the forward rule as a constant.
const GMG_GLOBAL_BUILD = gmg_mixed(0.3, Ref(0.7))
# A Robin `build` on ghost spaces, whose coefficient the rules refuse to differentiate.
gmg_robin_relations(u) = map(F -> F => u + ∂ₙ(u) => 0.0, (:xmin, :xmax, :ymin, :ymax))
function gmg_robin(c)
    V -> (C = constrained(gridspace(Bramble.mesh(V); ghosts = :boundary),
            gmg_robin_relations);
        form(C, C, (u, v) -> innerₕ(-1.0 * Δₕ(u), v) + c * innerₕ(u, v)))
end
gmg_tuple(ps) = V -> form(V, V, (u, v) -> innerₕ(u, v) + ps[1][1] * inner₊(∇ₕ(u), ∇ₕ(v)))
gmg_wrapped(cw) = V -> form(V, V, (u, v) -> innerₕ(u, v) + cw.κ * inner₊(∇ₕ(u), ∇ₕ(v)))
gmg_square(γ) = V -> form(V, V, (u, v) -> innerₕ(u, v) + γ[]^2 * inner₊(∇ₕ(u), ∇ₕ(v)))

# A point mass and a point advection at the node `GMG_POINT[1]`, scaled by the `Ref`s the form
# holds, `GMG_POINT[2]` and `GMG_POINT[3]`. A named `build` without captures reaches the
# `gmg_solve` rule as a constant.
const GMG_POINT = (Ref((0.0, 0.0)), Ref(0.0), Ref(0.0))
gmg_point_χ(x) = (c = GMG_POINT[1][];
    abs(x[1] - c[1]) < 1e-12 && abs(x[2] - c[2]) < 1e-12 ? 1.0 : 0.0)
function gmg_point_build(V)
    form(V,
        V,
        (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)) +
                  GMG_POINT[2] * innerₕ(Rₕ(V, gmg_point_χ) * u, v) +
                  GMG_POINT[3] *
                  innerₕ(Rₕ(V, gmg_point_χ) * Bramble.D₋ₓ(u), v))
end

@testset "BrambleEnzymeExt: gmg_solve" begin
    # `gmg_solve` in its right-hand side (#1168). `build` is called inside the differentiated
    # function, as a user writes it, so the rule sits on `gmg_solve` and Enzyme never enters
    # the cycles. The oracle is the direct solve of the finest matrix. Uniform and graded
    # meshes, both modes, a `VectorElement` load, forward width 2.
    @testset "Enzyme: gmg_solve rules" begin
        if _have(:Enzyme)
            @eval import Enzyme
            rev = Enzyme.set_runtime_activity(Enzyme.Reverse)
            fwd = Enzyme.set_runtime_activity(Enzyme.Forward)
            ez(m) = DI.AutoEnzyme(; mode = Enzyme.set_runtime_activity(m),
                function_annotation = Enzyme.Const)
            bld(V) = form(V, V, (u, v) -> innerₕ(u, v) + 0.7 * inner₊(∇ₕ(u), ∇ₕ(v)))

            for (dims, unif) in (((17, 17), true), ((9, 9), false))
                Random.seed!(1234)
                Ωg = Bramble.mesh(Bramble.domain(Bramble.interval(0.0, 1.0) ×
                                                 Bramble.interval(0.0, 1.0)), dims,
                    (unif, unif))
                Wg = gridspace(Ωg)
                ng = ndofs(Wg)
                A = Matrix(assemble(bld(Wg)))
                b₀ = [1.0 + sin(i) for i in 1:ng]
                ḃ1 = [cos(k) for k in 1:ng]
                ḃ2 = [sin(3k) for k in 1:ng]
                w = [sin(0.3k) + 0.1 for k in 1:ng]
                kw = (; tol = 1e-11)
                solve(b) = gmg_solve(bld, Ωg, b; kw...)
                loss(b) = sum(w .* parent(solve(b)))
                # The closure `build` is made inside the loss, a constant all the same.
                loss_in(b) = sum(w .* parent(gmg_solve(V -> bld(V), Ωg, b; kw...)))
                b̄ = A' \ w
                @test first(Enzyme.gradient(rev, Enzyme.Const(loss), b₀)) ≈ b̄ rtol=1e-8
                @test first(Enzyme.gradient(rev, Enzyme.Const(loss_in), b₀)) ≈ b̄ rtol=1e-8

                # The cotangent adds to what `b̄` holds, the load is left alone, and a load
                # used twice sums its two pullbacks.
                bd = fill(0.5, ng)
                b = copy(b₀)
                Enzyme.autodiff(rev, Enzyme.Const(loss), Enzyme.Active, Enzyme.Duplicated(b, bd))
                @test bd ≈ 0.5 .+ b̄ rtol=1e-8
                @test b == b₀
                twice(b) = loss(b) + sum(parent(solve(b)))
                bd = zeros(ng)
                Enzyme.autodiff(rev, Enzyme.Const(twice), Enzyme.Active,
                    Enzyme.Duplicated(copy(b₀), bd))
                @test bd ≈ A' \ (w .+ 1) rtol=1e-8

                # A `VectorElement` load: its shadow is one too, and gets the cotangent.
                bₑ = element(Wg, copy(b₀))
                bₑd = element(Wg, zeros(ng))
                Enzyme.autodiff(rev, Enzyme.Const(loss), Enzyme.Active,
                    Enzyme.Duplicated(bₑ, bₑd))
                @test parent(bₑd) ≈ b̄ rtol=1e-8
                u̇ₑ = only(Enzyme.autodiff(fwd, Enzyme.Const(solve), Enzyme.Duplicated,
                    Enzyme.Duplicated(element(Wg, copy(b₀)), element(Wg, copy(ḃ1)))))
                @test u̇ₑ isa Bramble.VectorElement
                @test parent(u̇ₑ) ≈ A \ ḃ1 rtol=1e-8

                # Forward: the tangent is the solve of `ḃ`, a `VectorElement`; a `Const` load
                # has a zero one; a batch of two solves each lane.
                u̇ = only(Enzyme.autodiff(fwd, Enzyme.Const(solve), Enzyme.Duplicated,
                    Enzyme.Duplicated(copy(b₀), ḃ1)))
                @test u̇ isa Bramble.VectorElement
                @test parent(u̇) ≈ A \ ḃ1 rtol=1e-8
                @test parent(only(Enzyme.autodiff(fwd, Enzyme.Const(solve),
                    Enzyme.Duplicated, Enzyme.Const(b₀)))) == zeros(ng)
                u̇s = only(Enzyme.autodiff(fwd, Enzyme.Const(solve), Enzyme.BatchDuplicated,
                    Enzyme.BatchDuplicated(copy(b₀), (ḃ1, ḃ2))))
                @test parent(u̇s[1]) ≈ A \ ḃ1 rtol=1e-8
                @test parent(u̇s[2]) ≈ A \ ḃ2 rtol=1e-8
                @test_throws ArgumentError "batched" Enzyme.autodiff(rev,
                    Enzyme.Const(loss), Enzyme.Active,
                    Enzyme.BatchDuplicated(copy(b₀), (zeros(ng), zeros(ng))))

                # A quadratic loss in a scaled load, through DI: the exact derivative is
                # `2(1 + q) S`. A coefficient that moves is differentiated too, in the testset
                # "Enzyme: gmg_solve in its coefficients".
                S = sum(abs2, A \ b₀)
                for m in (Enzyme.Reverse, Enzyme.Forward)
                    @test only(DI.gradient(
                        q -> sum(abs2, parent(gmg_solve(bld, Ωg, (1 + q[1]) .* b₀; kw...))),
                        ez(m), [0.4])) ≈ 2.8 * S rtol=1e-8
                end
            end

            # A `Ref` set between two solves. Read by `build` (`γ[]`), the form holds its value,
            # and the pullback of each solve is the adjoint of the operator it solved with: the
            # loss is quadratic in `θ`, so `d/dθ = 2 loss(θ) / θ`. Held by the form (`γ`), it
            # is read at each product, and reverse refuses the changed operator by name.
            Ωr = Bramble.mesh(
                Bramble.domain(Bramble.interval(0.0, 1.0) ×
                               Bramble.interval(0.0, 1.0)), (9, 9), (false, false))
            xr = [sin(0.7k) for k in 1:npoints(Ωr)]
            γ = Ref(1.0)
            by_value(V) = form(V, V, (u, v) -> innerₕ(u, v) + γ[] * inner₊(∇ₕ(u), ∇ₕ(v)))
            by_ref(V) = form(V, V, (u, v) -> innerₕ(u, v) + γ * inner₊(∇ₕ(u), ∇ₕ(v)))
            function loss_ref(θ, bld)
                γ[] = 1.0
                u = gmg_solve(bld, Ωr, θ .* xr; tol = 1e-10)
                γ[] = 5.0
                return sum(abs2, parent(gmg_solve(bld, Ωr, parent(u); tol = 1e-10)))
            end
            exact = 2 * loss_ref(0.7, by_value) / 0.7
            @test first(Enzyme.autodiff(rev, Enzyme.Const(loss_ref), Enzyme.Active,
                Enzyme.Active(0.7), Enzyme.Const(by_value)))[1] ≈ exact rtol=1e-7
            @test only(Enzyme.autodiff(fwd, Enzyme.Const(loss_ref), Enzyme.Duplicated,
                Enzyme.Duplicated(0.7, 1.0), Enzyme.Const(by_value))) ≈ exact rtol=1e-7
            @test_throws ArgumentError "changed" Enzyme.autodiff(rev, Enzyme.Const(loss_ref),
                Enzyme.Active, Enzyme.Active(0.7), Enzyme.Const(by_ref))
            @test only(Enzyme.autodiff(fwd, Enzyme.Const(loss_ref), Enzyme.Duplicated,
                Enzyme.Duplicated(0.7, 1.0), Enzyme.Const(by_ref))) ≈
                  2 * loss_ref(0.7, by_ref) / 0.7 rtol=1e-7
            # A change the form holds, tuned to leave `ΔA w = 0` for a fixed `w`: a point mass
            # and a point advection at the centre node, the second scaled to cancel the first
            # there. Only the whole matrix sees it.
            Ωt = Bramble.mesh(
                Bramble.domain(Bramble.interval(0.0, 1.0) ×
                               Bramble.interval(0.0, 1.0)), (17, 17), (true, true))
            Wt = gridspace(Ωt)
            nt = ndofs(Wt)
            Xt, Yt = parent(Rₕ(Wt, x -> x[1])), parent(Rₕ(Wt, x -> x[2]))
            xc, yc = sort(unique(Xt))[9], sort(unique(Yt))[9]
            GMG_POINT[1][] = (xc, yc)
            jc = findfirst(k -> Xt[k] == xc && Yt[k] == yc, 1:nt)
            wt = [cos(1.3k) for k in 1:nt]
            s₁ = assemble(form(Wt, Wt, (u, v) -> innerₕ(Rₕ(Wt, gmg_point_χ) * u, v))) * wt
            s₂ = assemble(form(Wt, Wt,
                (u, v) -> innerₕ(Rₕ(Wt, gmg_point_χ) * Bramble.D₋ₓ(u), v))) * wt
            t₂ = -s₁[jc] / s₂[jc]
            @test norm(s₁ + t₂ * s₂) < 1e-12 * norm(s₁)
            xt = [sin(0.7k) for k in 1:nt]
            function loss_point(θ)
                GMG_POINT[2][] = 0.0
                GMG_POINT[3][] = 0.0
                u = gmg_solve(gmg_point_build, Ωt, θ .* xt; tol = 1e-10)
                GMG_POINT[2][] = 30.0
                GMG_POINT[3][] = 30.0 * t₂
                return sum(abs2, parent(u))
            end
            @test_throws ArgumentError "changed" Enzyme.autodiff(rev, Enzyme.Const(loss_point),
                Enzyme.Active, Enzyme.Active(0.7))

            # An advection term makes `A` non-symmetric, whose adjoint is not `gmg_solve`:
            # reverse refuses it by name.
            Wa = gridspace(Ωr)
            adv(V) = form(V, V, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)) +
                                          innerₕ(Bramble.D₋ₓ(u), v))
            Aadv = Matrix(assemble(adv(Wa)))
            @test maximum(abs, Aadv - Aadv') > 1e-3
            @test_throws ArgumentError "symmetric" Enzyme.autodiff(rev,
                Enzyme.Const(b -> sum(parent(gmg_solve(adv, Ωr, b)))), Enzyme.Active,
                Enzyme.Duplicated(copy(xr), zeros(length(xr))))
            # The check is exact, on the assembled finest matrix. Four advection coefficients
            # tuned so that `vᵀ(A − Aᵀ)w` vanishes on four random pairs of probes (those of
            # `Xoshiro(1168)`), and a weak advection whose probe gap sits under `√eps` of a
            # scale that grows with the mesh, are refused all the same.
            Ωu = Bramble.mesh(
                Bramble.domain(Bramble.interval(0.0, 1.0) ×
                               Bramble.interval(0.0, 1.0)), (17, 17), (true, true))
            Wu = gridspace(Ωu)
            nu = ndofs(Wu)
            cs = (Ref(0.0), Ref(0.0), Ref(0.0), Ref(0.0))
            terms = (V -> form(V, V, (u, v) -> innerₕ(Bramble.D₋ᵧ(u), v)),
                V -> form(V, V, (u, v) -> innerₕ(Rₕ(V, x -> x[1]) * Bramble.D₋ₓ(u), v)),
                V -> form(V, V, (u, v) -> innerₕ(Rₕ(V, x -> x[2]) * Bramble.D₋ᵧ(u), v)),
                V -> form(V, V, (u, v) -> innerₕ(Rₕ(V, x -> x[1] * x[2]) * Bramble.D₋ₓ(u), v)))
            four(V) = form(V,
                V,
                (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)) +
                          innerₕ(Bramble.D₋ₓ(u), v) +
                          cs[1][] * innerₕ(Bramble.D₋ᵧ(u), v) +
                          cs[2][] * innerₕ(Rₕ(V, x -> x[1]) * Bramble.D₋ₓ(u), v) +
                          cs[3][] * innerₕ(Rₕ(V, x -> x[2]) * Bramble.D₋ᵧ(u), v) +
                          cs[4][] *
                          innerₕ(Rₕ(V, x -> x[1] * x[2]) * Bramble.D₋ₓ(u), v))
            rng = Random.Xoshiro(1168)
            pairs = [(randn(rng, nu), randn(rng, nu)) for _ in 1:4]
            gaps(M) = [dot(v, (M - M') * w) for (v, w) in pairs]
            A₀ = Matrix(assemble(four(Wu)))
            G = reduce(hcat, [gaps(Matrix(assemble(t(Wu)))) for t in terms])
            foreach(((c, x),) -> c[] = x, zip(cs, G \ -gaps(A₀)))
            A₄ = Matrix(assemble(four(Wu)))
            @test maximum(abs, gaps(A₄)) < 1e-10 * norm(A₄) * nu
            @test maximum(abs, A₄ - A₄') > 1e-2
            @test_throws ArgumentError "symmetric" Enzyme.autodiff(rev,
                Enzyme.Const(b -> sum(parent(gmg_solve(four, Ωu, b; tol = 1e-10)))),
                Enzyme.Active, Enzyme.Duplicated(ones(nu), zeros(nu)))
            Ωw = Bramble.mesh(
                Bramble.domain(Bramble.interval(0.0, 1.0) ×
                               Bramble.interval(0.0, 1.0)), (33, 33), (true, true))
            weak(V) = form(V, V, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)) +
                                           1e-4 * innerₕ(Bramble.D₋ₓ(u), v))
            nw = npoints(Ωw)
            @test_throws ArgumentError "symmetric" Enzyme.autodiff(rev,
                Enzyme.Const(b -> sum(parent(gmg_solve(weak, Ωw, b; tol = 1e-10)))),
                Enzyme.Active, Enzyme.Duplicated(ones(nw), zeros(nw)))

            # A `build` capturing a constant `κ`: the tangent is `A⁻¹ ḃ` and, in reverse, where
            # Enzyme passes such a `build` as `Active`, the pullback is `A⁻ᴴ w`, the
            # cotangent of `κ` dropped by Enzyme.
            κ = 0.7
            Aκ = Matrix(assemble(gmg_scalar(κ)(Wa)))
            solve_κ(b) = gmg_solve(gmg_scalar(κ), Ωr, b; tol = 1e-10)
            ḃr = [cos(k) for k in 1:ndofs(Wa)]
            wr = [sin(0.3k) + 0.1 for k in 1:ndofs(Wa)]
            @test parent(only(Enzyme.autodiff(fwd, Enzyme.Const(solve_κ), Enzyme.Duplicated,
                Enzyme.Duplicated(copy(xr), ḃr)))) ≈ Aκ \ ḃr rtol=1e-7
            b̄κ = zeros(length(xr))
            Enzyme.autodiff(rev, Enzyme.Const(b -> sum(wr .* parent(solve_κ(b)))), Enzyme.Active,
                Enzyme.Duplicated(copy(xr), b̄κ))
            @test b̄κ ≈ Aκ' \ wr rtol=1e-7
            # A moving coefficient is differentiated, against ForwardDiff of the direct solve
            # of the same form with the coefficient a plain scalar: in forward mode even when
            # its tangent equals its value, and inside a vector of tuples or a user's
            # `struct <: Real` in both modes.
            fd_c(c₀) = ForwardDiff.derivative(
                c -> sum(abs2, pde_solve(assemble(gmg_scalar(c)(Wa)), xr)), c₀)
            loss_c(c) = sum(abs2, parent(gmg_solve(gmg_scalar(c), Ωr, xr; tol = 1e-10)))
            for c in (1.0, 2.0)
                @test abs(fd_c(c)) > 1e-2
                @test only(Enzyme.autodiff(fwd, Enzyme.Const(loss_c), Enzyme.Duplicated,
                    Enzyme.Duplicated(c, c))) ≈ c * fd_c(c) rtol=1e-7
            end
            loss_t(q) = sum(abs2,
                parent(gmg_solve(gmg_tuple([(q[1], 0.0)]), Ωr, xr; tol = 1e-10)))
            loss_w(q) = sum(abs2,
                parent(gmg_solve(gmg_wrapped(WrappedCoefficient(q[1])), Ωr, xr; tol = 1e-10)))
            for lossq in (loss_t, loss_w)
                @test only(Enzyme.autodiff(fwd, Enzyme.Const(lossq), Enzyme.Duplicated,
                    Enzyme.Duplicated([0.7], [1.0]))) ≈ fd_c(0.7) rtol=1e-7
                q̄ = [0.0]
                Enzyme.autodiff(rev, Enzyme.Const(lossq), Enzyme.Active,
                    Enzyme.Duplicated([0.7], q̄))
                @test only(q̄) ≈ fd_c(0.7) rtol=1e-7
            end
            # A constant scalar captured beside a constant `Ref`, whose shadow is the primal's
            # own, leaves the tangent `A⁻¹ ḃ`.
            @test gmg_mixed_γ[] == κ
            solve_γκ(b) = gmg_solve(gmg_mixed(0.0, gmg_mixed_γ), Ωr, b; tol = 1e-10)
            @test parent(only(Enzyme.autodiff(fwd, Enzyme.Const(solve_γκ), Enzyme.Duplicated,
                Enzyme.Duplicated(copy(xr), ḃr)))) ≈ Aκ \ ḃr rtol=1e-7
            # With the scalar nonzero, forward mode cannot tell this constant `build` from
            # an active one whose tangent equals its value, and refuses it by name; each way
            # out the refusal names gives `A⁻¹ ḃ` or, in reverse, `A⁻ᴴ w`.
            Am = Matrix(assemble(gmg_mixed(0.3, gmg_mixed_γ)(Wa)))
            solve_m(b) = gmg_solve(gmg_mixed(0.3, gmg_mixed_γ), Ωr, b; tol = 1e-10)
            @test_throws ArgumentError "undetermined" Enzyme.autodiff(fwd,
                Enzyme.Const(solve_m), Enzyme.Duplicated, Enzyme.Duplicated(copy(xr), ḃr))
            @test_throws ArgumentError "undetermined" Enzyme.autodiff(fwd,
                Enzyme.Const(solve_m), Enzyme.BatchDuplicated,
                Enzyme.BatchDuplicated(copy(xr), (ḃr, wr)))
            b̄m = zeros(length(xr))
            Enzyme.autodiff(rev, Enzyme.Const(b -> sum(wr .* parent(solve_m(b)))),
                Enzyme.Active, Enzyme.Duplicated(copy(xr), b̄m))
            @test b̄m ≈ Am' \ wr rtol=1e-7
            @test ForwardDiff.derivative(t -> parent(solve_m(xr .+ t .* ḃr)), 0.0) ≈
                  Am \ ḃr rtol=1e-7
            solve_g(b) = gmg_solve(GMG_GLOBAL_BUILD, Ωr, b; tol = 1e-10)
            @test parent(only(Enzyme.autodiff(fwd, Enzyme.Const(solve_g), Enzyme.Duplicated,
                Enzyme.Duplicated(copy(xr), ḃr)))) ≈ Am \ ḃr rtol=1e-7
            solve_r(b) = gmg_solve(gmg_mixed_ref(Ref(0.3), gmg_mixed_γ), Ωr, b;
                tol = 1e-10)
            u̇r = only(Enzyme.autodiff(fwd, Enzyme.Const(solve_r), Enzyme.BatchDuplicated,
                Enzyme.BatchDuplicated(copy(xr), (ḃr, wr))))
            @test parent(u̇r[1]) ≈ Am \ ḃr rtol=1e-7
            @test parent(u̇r[2]) ≈ Am \ wr rtol=1e-7
            # A `build` on ghost spaces with an active coefficient is refused in both modes.
            loss_robin(c) = sum(abs2, parent(gmg_solve(gmg_robin(c), Ωr, xr; tol = 1e-10)))
            @test_throws ArgumentError "ghost spaces" Enzyme.autodiff(fwd,
                Enzyme.Const(loss_robin), Enzyme.Duplicated, Enzyme.Duplicated(1.0, 1.0))
            @test_throws ArgumentError "ghost spaces" Enzyme.autodiff(rev,
                Enzyme.Const(loss_robin), Enzyme.Active, Enzyme.Active(1.0))
        else
            @test_skip "Enzyme not in this environment"
        end
    end

    # `gmg_solve` in the coefficients `build` captures (#1168): a nested Enzyme call on the
    # finest product gives `Ȧ u` and `−λᴴ (∂A/∂θ) u`. The oracle is ForwardDiff of the direct
    # solve of the same form, on a uniform and a graded mesh, with the load and the
    # coefficient active together.
    @testset "Enzyme: gmg_solve in its coefficients" begin
        if _have(:Enzyme)
            @eval import Enzyme
            rev = Enzyme.set_runtime_activity(Enzyme.Reverse)
            fwd = Enzyme.set_runtime_activity(Enzyme.Forward)
            ez(m) = DI.AutoEnzyme(; mode = Enzyme.set_runtime_activity(m),
                function_annotation = Enzyme.Const)
            for (dims, unif) in (((17, 17), true), ((9, 9), false))
                Random.seed!(1234)
                Ω = Bramble.mesh(
                    Bramble.domain(Bramble.interval(0.0, 1.0) ×
                                   Bramble.interval(0.0, 1.0)), dims, (unif, unif))
                W = gridspace(Ω)
                b₀ = [1.0 + sin(i) for i in 1:npoints(Ω)]
                for (name, mk) in (("scalar", gmg_scalar), ("grid function", gmg_grid))
                    # `q = (c, s)`: the coefficient and a scale of the load.
                    direct(q) = sum(abs2, pde_solve(assemble(mk(q[1])(W)), q[2] .* b₀))
                    ref = ForwardDiff.gradient(direct, [0.7, 1.3])
                    @test minimum(abs, ref) > 1e-2
                    loss(q) = sum(abs2,
                        parent(gmg_solve(mk(q[1]), Ω, q[2] .* b₀; tol = 1e-11)))
                    for m in (Enzyme.Reverse, Enzyme.Forward)
                        @testset "$name $m unif=$unif" begin
                            @test DI.gradient(loss, ez(m), [0.7, 1.3]) ≈ ref rtol=1e-7
                        end
                    end
                end
            end
            # A coefficient in a `Ref` that `build` reads: reverse differentiates the
            # operator the solve used, though the `Ref` holds another value by then.
            Random.seed!(1234)
            Ωr = Bramble.mesh(
                Bramble.domain(Bramble.interval(0.0, 1.0) ×
                               Bramble.interval(0.0, 1.0)), (9, 9), (false, false))
            Wr = gridspace(Ωr)
            xr = [sin(0.7k) for k in 1:npoints(Ωr)]
            ref_sq = ForwardDiff.derivative(
                γ -> sum(abs2, pde_solve(assemble(gmg_scalar(γ^2)(Wr)), xr)), 0.8)
            @test abs(ref_sq) > 1e-2
            function loss_sq(γ)
                u = gmg_solve(gmg_square(γ), Ωr, xr; tol = 1e-10)
                γ[] = 5.0
                return sum(abs2, parent(u))
            end
            γ̄ = Ref(0.0)
            Enzyme.autodiff(rev, Enzyme.Const(loss_sq), Enzyme.Active,
                Enzyme.Duplicated(Ref(0.8), γ̄))
            @test γ̄[] ≈ ref_sq rtol=1e-7
            # A `build` capturing an active scalar beside an active `Ref`, the scalar at 0.0:
            # its derivative is not lost to a shadow that compares equal to the primal.
            ref_mix = ForwardDiff.gradient(
                p -> sum(abs2, pde_solve(
                    assemble(form(Wr, Wr,
                        (u, v) -> (1 + p[1]) * innerₕ(u, v) + p[2] * inner₊(∇ₕ(u), ∇ₕ(v)))),
                    xr)),
                [0.0, 0.6])
            @test minimum(abs, ref_mix) > 1e-2
            loss_mix(c, γ) = sum(abs2, parent(gmg_solve(gmg_mixed(c, γ), Ωr, xr; tol = 1e-10)))
            γ̄ = Ref(0.0)
            c̄ = first(Enzyme.autodiff(rev, Enzyme.Const(loss_mix), Enzyme.Active,
                Enzyme.Active(0.0), Enzyme.Duplicated(Ref(0.6), γ̄)))[1]
            @test [c̄, γ̄[]] ≈ ref_mix rtol=1e-7
            # Forward width 2: one lane per coefficient.
            ṡ = only(Enzyme.autodiff(fwd, Enzyme.Const(loss_mix), Enzyme.BatchDuplicated,
                Enzyme.BatchDuplicated(0.0, (1.0, 0.0)),
                Enzyme.BatchDuplicated(Ref(0.6), (Ref(0.0), Ref(1.0)))))
            @test collect(values(ṡ)) ≈ ref_mix rtol=1e-7
            # The scalar active beside a constant `Ref`: a tangent equal to its value
            # reaches forward mode as a constant `build` does and is refused by name; any
            # other is differentiated, directly and through DI.
            γc = Ref(0.6)
            ref_c = ForwardDiff.derivative(
                c -> sum(abs2, pde_solve(assemble(gmg_mixed(c, γc)(Wr)), xr)), 0.3)
            @test abs(ref_c) > 1e-2
            loss_c(c) = sum(abs2, parent(gmg_solve(gmg_mixed(c, γc), Ωr, xr; tol = 1e-10)))
            @test_throws ArgumentError "undetermined" Enzyme.autodiff(fwd,
                Enzyme.Const(loss_c), Enzyme.Duplicated, Enzyme.Duplicated(0.3, 0.3))
            @test only(Enzyme.autodiff(fwd, Enzyme.Const(loss_c), Enzyme.Duplicated,
                Enzyme.Duplicated(0.3, 1.0))) ≈ ref_c rtol=1e-7
            @test DI.derivative(loss_c, ez(Enzyme.Forward), 0.3) ≈ ref_c rtol=1e-7
        else
            @test_skip "Enzyme not in this environment"
        end
    end
end

end # module
