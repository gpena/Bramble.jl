module ExtChainRulesEnzymeExtTests

using Test
using Bramble
using ChainRulesCore
using SparseArrays: SparseMatrixCSC, nnz, nonzeros, sparse, spdiagm
using LinearAlgebra: I, NoPivot, dot, ldiv!, lu, norm
using Random: Random
using Polyester
using ForwardDiff
using SuiteSparse
using Bramble: CpuSerial, CpuThreaded, CpuPolyester
using Kronecker: Kronecker
import DifferentiationInterface as DI
using ..TestUtils: _fd, _have

# Enzyme composition for `BrambleChainRulesExt`'s `pde_solve` rrule. Behind the "ad"/"full"
# groups, same reasoning and the same `_have`/`@test_skip` idiom `autodiff_heavy.jl` already
# uses for Enzyme: it is not a `test/Project.toml` dependency (kept out of the ordinary
# dependency graph so an every-push run never pays its ~30 s first-call compilation), and
# `Weekly.yml`'s "add the expensive differentiation backends" step installs it at CI runtime
# for exactly this group.
#
# `test/ext/chainrules_ext.jl` already checks the rrule's own math against finite differences
# without any AD package; what is left to check here is that a real reverse-mode backend
# actually *reaches* that rule and gets the right answer end to end, through
# `assemble(...; dirichlet = θ) -> pde_solve -> J`, including the Dirichlet-value case
# `chainrules_ext.jl` could not check against `ForwardDiff` (UMFPACK's `Float64`-only sparse
# factorisation, documented in `docs/notes/tutorials/autodiff.md` §5) -- reverse-mode through
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

# The 2D losses, as the source of a child process.
const CHILD_LOSSES_2D = """
I01 = Bramble.interval(0.0, 1.0)
Ω2 = Bramble.mesh(Bramble.domain(I01 × I01), (9, 9), (true, true))
W2 = gridspace(Ω2)
l2 = form(W2, v -> innerₕ(Rₕ(W2, x -> 1.0), v))
function loss_stiff_2d(θ::Real)
    aθ = form(W2, W2, (u, v) -> θ * inner₊(∇ₕ(u), ∇ₕ(v)))
    A, F = assemble(aθ, l2; dirichlet = :boundary => x -> 0.0)
    return sum(abs2, Bramble.pde_solve(A, F))
end
function loss_field_2d(θ::Real)
    κₕ = Bramble.element(W2, θ)
    aκ = form(W2, W2, (u, v) -> inner₊(κₕ * ∇ₕ(u), ∇ₕ(v)))
    A, F = assemble(aκ, l2; dirichlet = :boundary => x -> 0.0)
    return sum(abs2, Bramble.pde_solve(A, F))
end
function loss_two_fields_2d(θ::Real)
    g₁ = Bramble.element(W2, θ)
    g₂ = Bramble.element(W2, 2θ)
    a = form(W2, W2, (u, v) -> innerₕ(g₁ * u, v) + innerₕ(g₂ * u, v))
    A, F = assemble(a, l2; dirichlet = :boundary => x -> 0.0)
    return sum(abs2, Bramble.pde_solve(A, F))
end
"""

# `loss_no_rule`, for `_child_gradient`: `A \ F` on the assembled system, no `pde_solve`.
const CHILD_LOSS_NO_RULE = """
Ω1 = Bramble.mesh(Bramble.domain(Bramble.interval(0.0, 1.0)), 21, true)
W1 = gridspace(Ω1)
a1 = form(W1, W1, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
function loss_no_rule(θ::Real)
    gₕ = Bramble.element(W1, zero(θ))
    A, F = assemble(a1, form(W1, v -> innerₕ(gₕ, v)); dirichlet = :boundary => x -> θ)
    return sum(abs2, A \\ F)
end
"""

const ENZYME_TOML = "[deps]\nEnzyme = \"7da242da-08ed-463a-9acd-ee780be4f1d9\"\n"

# `name`'s direct Enzyme gradient (`name` is defined by the source `losses`) at 0.7 and a
# central difference of it, from a child Julia process with the parent's project, thread
# count, bounds checking and optimisation level (Enzyme comes from a temporary environment,
# as in `enzyme_policy_cases.jl`). Returns the exit code, the two values (`NaN` when the
# child died before printing them) and the child's whole output.
function _child_gradient(name::AbstractString; losses::AbstractString = CHILD_LOSSES_2D)
    src = """
        _e = mktempdir()
        write(joinpath(_e, "Project.toml"), $(repr(ENZYME_TOML)))
        push!(LOAD_PATH, _e)
        using Bramble, Enzyme
        $losses
        f = $name
        g = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(f), 0.7)[1]
        h = 1e-6
        println("GRADIENT ", g, " ", (f(0.7 + h) - f(0.7 - h)) / 2h)
        """
    opt = max(Base.JLOptions().opt_level, 0)
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) --threads=$(Threads.nthreads()) --optimize=$opt -e $src`
    out = IOBuffer()
    p = run(pipeline(ignorestatus(cmd); stdout = out, stderr = out))
    text = String(take!(out))
    m = match(r"^GRADIENT (\S+) (\S+)$"m, text)
    grad, fd = m === nothing ? (NaN, NaN) : parse.(Float64, m.captures)
    return p.exitcode, grad, fd, text
end

# A factorization that solves with the matrix it holds and has no adjoint solve, standing for
# a backend whose `fact'` is unavailable (Sparspak, MUMPS, Accelerate).
struct NoAdjointFactorization{T} <: Bramble.SparspakFactorization{T}
    A::SparseMatrixCSC{T, Int}
end
Base.size(f::NoAdjointFactorization) = size(f.A)
Base.:\(f::NoAdjointFactorization, b::AbstractVector) = f.A \ b

# A coefficient that is a `Real` holding a `Float64`, for the `gmg_solve` rule's shadow walk.
struct WrappedCoefficient <: Real
    κ::Float64
end

# A point mass and a point advection at the node `GMG_POINT[1]`, scaled by the `Ref`s the form
# holds, `GMG_POINT[2]` and `GMG_POINT[3]`. A named `build` without captures reaches the
# `gmg_solve` rule as a constant.
const GMG_POINT = (Ref((0.0, 0.0)), Ref(0.0), Ref(0.0))
gmg_point_χ(x) = (c = GMG_POINT[1][];
abs(x[1] - c[1]) < 1e-12 && abs(x[2] - c[2]) < 1e-12 ? 1.0 : 0.0)
gmg_point_build(V) = form(V, V, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)) +
                                         GMG_POINT[2] * innerₕ(Rₕ(V, gmg_point_χ) * u, v) +
                                         GMG_POINT[3] *
                                         innerₕ(Rₕ(V, gmg_point_χ) * Bramble.D₋ₓ(u), v))

@testset "BrambleChainRulesExt + Enzyme" begin
    @testset "Enzyme forward: pde_solve rule" begin
        if _have(:Enzyme)
            @eval import Enzyme
            fwd = Enzyme.set_runtime_activity(Enzyme.Forward)
            ext = Base.get_extension(Bramble, :BrambleEnzymeExt)

            # The rule's own cases on a small system, against the tangent in closed form
            # (`u̇ = A⁻¹(Ḟ − Ȧ u)`, dense) and, where the `Dual` solve supports the keyword,
            # against ForwardDiff. `Ȧ` shares `A`'s pattern, as an Enzyme shadow does.
            n = 6
            A0 = sparse(spdiagm(-1 => fill(-1.0, n - 1), 0 => fill(2.5, n),
                1 => fill(-1.2, n - 1)))
            F0 = collect(1.0:n)
            dA = sparse(spdiagm(-1 => cos.(1.0:(n - 1)), 0 => sin.(1.0:n),
                1 => cos.(2.0:n)))
            dA2 = 0.5 .* dA .+ sparse(spdiagm(0 => fill(0.1, n)))
            dF = cos.(1.0:n)
            dF2 = sin.(2.0:(n + 1))
            u0 = Matrix(A0) \ F0
            tangent(dA, dF) = Matrix(A0) \ (dF - dA * u0)

            solve_plain(A, F) = Bramble.pde_solve(A, F)
            @test only(Enzyme.autodiff(fwd, Enzyme.Const(solve_plain), Enzyme.Duplicated,
                Enzyme.Duplicated(copy(A0), copy(dA)), Enzyme.Duplicated(copy(F0), copy(dF)))) ≈
                  tangent(dA, dF) rtol=1e-12

            # One of the two inputs constant: the tangent is the other's alone.
            @test only(Enzyme.autodiff(fwd, Enzyme.Const(solve_plain), Enzyme.Duplicated,
                Enzyme.Const(A0), Enzyme.Duplicated(copy(F0), copy(dF)))) ≈
                  tangent(0 .* dA, dF) rtol=1e-12
            @test only(Enzyme.autodiff(fwd, Enzyme.Const(solve_plain), Enzyme.Duplicated,
                Enzyme.Duplicated(copy(A0), copy(dA)), Enzyme.Const(F0))) ≈
                  tangent(dA, 0 .* dF) rtol=1e-12

            # The primal comes back with the tangent, as `ForwardWithPrimal` asks.
            der, val = Enzyme.autodiff(Enzyme.set_runtime_activity(Enzyme.ForwardWithPrimal),
                Enzyme.Const(solve_plain), Enzyme.Duplicated, Enzyme.Duplicated(copy(A0), copy(dA)),
                Enzyme.Duplicated(copy(F0), copy(dF)))
            @test val ≈ u0 rtol=1e-12
            @test der ≈ tangent(dA, dF) rtol=1e-12

            # Width 2: two tangents, one factorisation. The count is the rule's own entry
            # counter, read around the call.
            factors() = ext._rule_entries(:pde_solve_factor)
            before = factors()
            bat = only(Enzyme.autodiff(fwd, Enzyme.Const(solve_plain), Enzyme.BatchDuplicated,
                Enzyme.BatchDuplicated(copy(A0), (copy(dA), copy(dA2))),
                Enzyme.BatchDuplicated(copy(F0), (copy(dF), copy(dF2)))))
            @test factors() - before == 1
            @test collect(values(bat)) ≈ [tangent(dA, dF), tangent(dA2, dF2)] rtol=1e-12
            bat_c = only(Enzyme.autodiff(fwd, Enzyme.Const(solve_plain), Enzyme.BatchDuplicated,
                Enzyme.Const(A0), Enzyme.BatchDuplicated(copy(F0), (copy(dF), copy(dF2)))))
            @test collect(values(bat_c)) ≈ [tangent(0 .* dA, dF), tangent(0 .* dA, dF2)] rtol=1e-12

            # Keywords: no factorisation is kept, and each tangent is a solve with the same
            # keywords. `mean_zero` borders a singular Neumann matrix; its tangent equals
            # ForwardDiff's bordered solve.
            An = sparse(spdiagm(-1 => fill(-1.0, n - 1), 0 => [1.0; fill(2.0, n - 2); 1.0],
                1 => fill(-1.0, n - 1)))
            Fn = F0 .- sum(F0) / n
            dAn = sparse(spdiagm(0 => sin.(1.0:n))) + 0.1 * (sparse(spdiagm(0 => cos.(1.0:n))) * An)
            w = ones(n)
            solve_mz(A, F) = Bramble.pde_solve(A, F; mean_zero = w)
            ref_mz = ForwardDiff.derivative(
                t -> Bramble.pde_solve(An + t * dAn, Fn + t * dF; mean_zero = w), 0.0)
            before = factors()
            got_mz = only(Enzyme.autodiff(fwd, Enzyme.Const(solve_mz), Enzyme.Duplicated,
                Enzyme.Duplicated(copy(An), copy(dAn)), Enzyme.Duplicated(copy(Fn), copy(dF))))
            @test got_mz ≈ ref_mz rtol=1e-10
            @test factors() == before

            solve_ss(A, F) = Bramble.pde_solve(A, F; solver = :suitesparse, sym = :auto)
            @test only(Enzyme.autodiff(fwd, Enzyme.Const(solve_ss), Enzyme.Duplicated,
                Enzyme.Duplicated(copy(A0), copy(dA)), Enzyme.Duplicated(copy(F0), copy(dF)))) ≈
                  tangent(dA, dF) rtol=1e-10
            bat_ss = only(Enzyme.autodiff(fwd, Enzyme.Const(solve_ss), Enzyme.BatchDuplicated,
                Enzyme.BatchDuplicated(copy(A0), (copy(dA), copy(dA2))),
                Enzyme.BatchDuplicated(copy(F0), (copy(dF), copy(dF2)))))
            @test collect(values(bat_ss)) ≈ [tangent(dA, dF), tangent(dA2, dF2)] rtol=1e-10

            # A rectangular `A` (a `:spqr` least-squares solve) is refused, as in reverse.
            solve_rect(A, F) = Bramble.pde_solve(A, F; solver = :spqr)
            Ar = sparse([1.0 0.0; 0.0 2.0; 1.0 1.0])
            @test_throws ArgumentError Enzyme.autodiff(fwd, Enzyme.Const(solve_rect),
                Enzyme.Duplicated, Enzyme.Duplicated(copy(Ar), copy(Ar)),
                Enzyme.Duplicated([1.0, 2.0, 0.0], zeros(3)))

            # Through `assemble`, a field coefficient in `A` and a parameter in `F`, one
            # right-hand side and a width-2 batch, under the three CPU policies.
            for policy in (CpuSerial(), CpuThreaded(), CpuPolyester())
                Wp = gridspace(mesh(domain(interval(0.0, 1.0)), 17, false;
                    backend = backend(policy = policy)))
                B = [sin(i * j) for i in 1:ndofs(Wp), j in 1:2]
                function loss(q)
                    κ = element(Wp, eltype(q))
                    parent(κ) .= 2.0 .+ B * q
                    a = form(Wp, Wp, (U, V) -> inner₊(κ * ∇ₕ(U), ∇ₕ(V)))
                    l = form(Wp, v -> innerₕ(x -> 1.0 + x[1], v))
                    AF = assemble(a, l; dirichlet = :boundary => x -> 0.0)
                    return sum(abs2, Bramble.pde_solve(AF[1], q[2] .* AF[2]))
                end
                q0 = [0.4, 0.7]
                ref = ForwardDiff.gradient(loss, q0)
                one_dir = [only(Enzyme.autodiff(fwd, Enzyme.Const(loss),
                               Enzyme.Duplicated(copy(q0), Float64.(1:2 .== k)))) for k in 1:2]
                @test one_dir ≈ ref rtol=1e-8
                batched = only(Enzyme.autodiff(fwd, Enzyme.Const(loss),
                    Enzyme.BatchDuplicated(copy(q0), ([1.0, 0.0], [0.0, 1.0]))))
                @test collect(values(batched)) ≈ ref rtol=1e-8
            end
        else
            @test_skip "Enzyme not in this environment"
        end
    end

    @testset "Enzyme: native BrambleEnzymeExt rule" begin
        if _have(:Enzyme)
            @eval import Enzyme

            Ωₕ = Bramble.mesh(Bramble.domain(Bramble.interval(0.0, 1.0)), 21, true)
            Wₕ = gridspace(Ωₕ)
            a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
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

            # Each gradient goes through DifferentiationInterface and through a direct Enzyme
            # call, and the two must agree: DI's `function_annotation = Duplicated` handles the
            # closures here that capture grid spaces and forms. `θ` is a scalar, so
            # `derivative`. Calls DI cannot express stay direct, with a comment saying why.
            di = DI.AutoEnzyme(; mode, function_annotation = Enzyme.Duplicated)
            function grad(f, θ)
                d_direct = Enzyme.gradient(mode, Enzyme.Const(f), θ)[1]
                d_di = DI.derivative(f, di, θ)
                @test d_di ≈ d_direct rtol=1e-13
                return d_di
            end

            d_enzyme = grad(loss, θ0)
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
            d_sym = grad(loss_symmetrized, θ0)
            @test d_sym ≈ _fd(loss_symmetrized, θ0) rtol=1e-4

            # Keywords on the solve: the rule takes `; kwargs...` and forwards them, so
            # `pde_solve(A, F; solver, sym)` differentiates through the rule, and its adjoint is
            # solved by `pde_solve` with the same keywords. Without `; kwargs...` the keyword
            # call bypasses the rule and Enzyme differentiates UMFPACK. It is the same loss as
            # `loss`, so its gradient must equal `loss`'s as well.
            function loss_kw(θ::Real)
                gₕ = Bramble.element(Wₕ, zero(θ))
                l = form(Wₕ, v -> innerₕ(gₕ, v))
                A, F = assemble(a, l; dirichlet = :boundary => x -> θ)
                return sum(abs2, Bramble.pde_solve(A, F; solver = :default, sym = :auto))
            end
            d_kw = grad(loss_kw, θ0)
            @test d_kw ≈ _fd(loss_kw, θ0) rtol=1e-4
            @test d_kw ≈ d_enzyme rtol=1e-10

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
                aθ = form(Wₕ, Wₕ, (u, v) -> θ * inner₊(∇ₕ(u), ∇ₕ(v)))
                A, F = assemble(aθ, l_fixed; dirichlet = :boundary => x -> 0.0)
                return sum(abs2, Bramble.pde_solve(A, F))
            end
            d_coeff = grad(loss_coeff, θ0)
            @test d_coeff ≈ _fd(loss_coeff, θ0) rtol=1e-4

            # A sum of terms, each with its own runtime coefficient -- the shape an inverse
            # problem actually has. No rule compares the two coefficients, so `form` has one
            # concrete type and the gradient goes through both terms.
            function loss_coeff_sum(θ::Real)
                aθ = form(
                    Wₕ, Wₕ, (u, v) -> θ * inner₊(∇ₕ(u), ∇ₕ(v)) + (1 - θ) * innerₕ(u, v)
                )
                A, F = assemble(aθ, l_fixed; dirichlet = :boundary => x -> 0.0)
                return sum(abs2, Bramble.pde_solve(A, F))
            end
            d_sum = grad(loss_coeff_sum, θ0)
            @test d_sum ≈ _fd(loss_coeff_sum, θ0) rtol=1e-4

            # The gradient is not merely *a* number Enzyme was willing to produce: it has to
            # be the one the adjoint says it is. `θ * a(u, v)` scales `A` by `θ`, so
            # `u(θ) = u(1)/θ` and `J(θ) = ‖u(1)‖²/θ²`, whose derivative is `-2J(θ)/θ`
            # in closed form -- checked here against the analytic value, not only against the
            # finite difference above, which shares the same forward code path.
            @test d_coeff ≈ -2 * loss_coeff(θ0) / θ0 rtol=1e-8

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
            Ω3 = Bramble.mesh(Bramble.domain(I01 × I01 × I01), (5, 5, 5), (true, true, true))
            W3 = gridspace(Ω3)
            l3 = form(W3, v -> innerₕ(Rₕ(W3, x -> 1.0), v))

            function loss_stiff_3d(θ::Real)
                aθ = form(W3, W3, (u, v) -> θ * inner₊(∇ₕ(u), ∇ₕ(v)))
                A, F = assemble(aθ, l3; dirichlet = :boundary => x -> 0.0)
                return sum(abs2, Bramble.pde_solve(A, F))
            end
            d_3d = grad(loss_stiff_3d, θ0)
            @test d_3d ≈ _fd(loss_stiff_3d, θ0) rtol=1e-3

            # The 2D cases are stiffness, a `VectorElement` coefficient inside the form (what
            # recovering a diffusion *field* needs), and two *distinct* coefficient fields on
            # same-shaped terms in one sum (once an `IllegalTypeAnalysisException`, fixed by
            # removing the like-term rule, gpena/Bramble.jl#771). They then aborted Enzyme's
            # compiler with no Julia exception (the LLVM assertion `needsCacheWholeAllocation`,
            # reached through `_dirichlet_probed_eltype` in `assemble`), until
            # gpena/Bramble.jl#1126's fix: `src/mesh/interface.jl` no longer indexes the tuple
            # of faces at run time. They abort no more and the gradient matches a central
            # difference. Each loss still runs in a child, so a regression that brings the
            # abort back fails the test and does not end the whole process.
            for name in ("loss_stiff_2d", "loss_field_2d", "loss_two_fields_2d")
                code, d_ad, d_cd, _ = _child_gradient(name)
                @test code == 0 && isapprox(d_ad, d_cd; rtol = 1e-3)
            end

            # A 1D sum of three terms: the other shape that used to fail, at the same
            # stencil width as 2D stiffness but reached by summing rather than by dimension.
            function loss_sum_three(θ::Real)
                aθ = form(
                    Wₕ,
                    Wₕ,
                    (u, v) -> θ * (innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)) +
                                   inner₊(∇ₕ(u), ∇ₕ(v)))
                )
                A, F = assemble(aθ, l_fixed; dirichlet = :boundary => x -> 0.0)
                return sum(abs2, Bramble.pde_solve(A, F))
            end
            d_three = grad(loss_sum_three, θ0)
            @test d_three ≈ _fd(loss_sum_three, θ0) rtol=1e-3

            # The defect the native rule exists to avoid, pinned at the level it actually
            # showed up: `@import_rrule`'s bridge merged the rrule's returned
            # `SparseMatrixCSC` into Enzyme's shadow by dropping the cotangent's explicit
            # zeros from `nzval` alone, leaving `colptr`/`rowval` claiming more entries than
            # `nzval` holds. `Matrix(shadow)` then trips SparseArrays' own `_goodbuffers`
            # assertion, and any gradient read off that shadow is silently wrong. A
            # homogeneous Dirichlet problem supplies such a zero for free: a constrained
            # row's solution entry is exactly its boundary value, so `u[j] == 0` makes a
            # whole column of `-λuᵀ` exactly zero.
            aθ0 = form(Wₕ, Wₕ, (u, v) -> 0.7 * inner₊(∇ₕ(u), ∇ₕ(v)))
            A0, F0 = assemble(aθ0, l_fixed; dirichlet = :boundary => x -> 0.0)
            @test count(iszero, A0 \ F0) > 0          # the zero that triggered it exists here
            dA = SparseMatrixCSC(
                size(A0, 1), size(A0, 2), copy(A0.colptr), copy(A0.rowval), zeros(nnz(A0))
            )
            # Kept a direct `Enzyme.autodiff`: DI's gradient returns a vector, with no sparse
            # `Duplicated` shadow of `A` to write the pattern's cotangent into.
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

            # Complex data: the adjoint conjugates `u` (`conj(u[j])`, as the rrule does). For a
            # real loss, the shadow `G` satisfies `dJ = Re(Σ conj(G) .* dA)` on the stored
            # pattern, so one central difference along a fixed direction `Ev` on that pattern
            # checks the shadow.
            Ac = sparse(ComplexF64[4 + 0.5im 1im 0 0; -1im 4 1 0; 0 1 4 - 0.3im 1; 0 0 1 3])
            Fc = ComplexF64[1, 2, 3, 4]
            dAc = Enzyme.make_zero(Ac)
            Enzyme.autodiff(
                mode, Enzyme.Const(solve_loss), Enzyme.Active,
                Enzyme.Duplicated(copy(Ac), dAc), Enzyme.Const(Fc)
            )
            Ev = ComplexF64.(cos.(1:nnz(Ac)), sin.(1:nnz(Ac)))
            shifted(s) = SparseMatrixCSC(
                size(Ac, 1), size(Ac, 2), Ac.colptr, Ac.rowval, Ac.nzval .+ s .* Ev
            )
            h = 1e-6
            fd_dir = (solve_loss(shifted(h), Fc) - solve_loss(shifted(-h), Fc)) / (2h)
            @test sum(real.(conj.(dAc.nzval) .* Ev)) ≈ fd_dir rtol=1e-6

            # Pinned failure: handed `A \ F` directly rather than `pde_solve`, Enzyme has no
            # rule to reach and traces into CHOLMOD's own internals. On Enzyme 0.13.218 the
            # compiler dies in LLVM's verifier (`function failed verification`, `PHI nodes
            # not grouped at top of basic block` in `preprocess_julia__extract_args_*`),
            # which aborts the process, so no exception is thrown to catch: gpena/Bramble.jl
            # #1135, reported upstream as https://github.com/EnzymeAD/Enzyme.jl/issues/3961.
            # The loss runs in a child with the parent's `--check-bounds` and `--optimize`
            # (`Base.julia_cmd()` carries the first; `_child_gradient` the second), the flags
            # the weekly ad-4 shard sees. `@test_broken` on the child's success turns an
            # upstream fix into an unexpected pass, which says so.
            code, d_ad, d_cd, text = _child_gradient("loss_no_rule"; losses = CHILD_LOSS_NO_RULE)
            @test code == 0 || occursin("function failed verification", text)
            @test_broken code == 0 && isapprox(d_ad, d_cd; rtol = 1e-3)

            # With keywords the rule keeps no factorisation, so it keeps `A` itself when the
            # caller overwrites it after the solve. `u = diag(2, 4) \ [θ, 1]`, then `A` doubles:
            # `d(sum(u))/dθ = 1/2`, where the doubled matrix would give 1/4. Direct Enzyme
            # calls, since DI's closures would not overwrite the solve's own `A`.
            function loss_overwritten(θ)
                A = sparse([2.0 0.0; 0.0 4.0])
                u = Bramble.pde_solve(A, [θ, 1.0]; solver = :suitesparse)
                nonzeros(A) .*= 2
                return sum(u)
            end
            @test Enzyme.gradient(mode, Enzyme.Const(loss_overwritten), 0.3)[1] ≈ 0.5

            # A rectangular `A` (a `:spqr` least-squares solve) is refused.
            loss_rect(θ) = sum(Bramble.pde_solve(sparse([1.0 0.0; 0.0 2.0; 1.0 1.0]),
                [θ, 2.0, 0.0]; solver = :spqr))
            @test_throws ArgumentError Enzyme.gradient(mode, Enzyme.Const(loss_rect), 0.3)
        else
            @test_skip "Enzyme not in this environment"
        end
    end

    @testset "Enzyme: kept factorization rules" begin
        if _have(:Enzyme)
            @eval import Enzyme
            rev = Enzyme.set_runtime_activity(Enzyme.Reverse)
            fwd = Enzyme.set_runtime_activity(Enzyme.Forward)
            sparse_factorize = Bramble.sparse_factorize

            # A non-symmetric `A₀`, so `fact \ ū` where `fact' \ ū` belongs changes the
            # gradient, and a symmetric positive definite `As` for the CHOLMOD factor.
            Wk = gridspace(Bramble.mesh(Bramble.domain(Bramble.interval(0.0, 1.0)), 9, true))
            n = ndofs(Wk)
            A₀ = sparse(spdiagm(-1 => fill(-1.0, n - 1), 0 => fill(2.5, n),
                1 => fill(-1.2, n - 1)))
            As = sparse(spdiagm(-1 => fill(-1.0, n - 1), 0 => fill(2.5, n),
                1 => fill(-1.0, n - 1)))
            c = [sin(2i) for i in 1:n]
            F₀ = [1.0 + sin(i) for i in 1:n]
            Ḟ1 = [cos(k) for k in 1:n]
            Ḟ2 = [sin(3k) for k in 1:n]
            @test maximum(abs, A₀ \ c - A₀' \ c) > 1e-3

            solve_vec(fact, F) = Bramble.pde_solve(fact, F)
            function ldiv_vec(fact, F)
                u = element(Wk)
                ldiv!(u, fact, F)
                return copy(parent(u))
            end
            solve_loss(fact, F) = sum(c .* solve_vec(fact, F))
            ldiv_loss(fact, F) = sum(c .* ldiv_vec(fact, F))

            for (A, fact) in ((A₀, sparse_factorize(A₀)), (As, sparse_factorize(As; sym = :spd))),
                (solver, loss) in ((solve_vec, solve_loss), (ldiv_vec, ldiv_loss))

                # Reverse: `F̄ = A⁻ᵀ c`, with `Const` and `Duplicated` loads. The primal load is
                # left alone.
                Fd = zeros(n)
                F = copy(F₀)
                Enzyme.autodiff(rev, Enzyme.Const(loss), Enzyme.Active, Enzyme.Const(fact),
                    Enzyme.Duplicated(F, Fd))
                @test Fd ≈ A' \ c rtol=1e-10
                @test F == F₀
                @test Enzyme.autodiff(rev, Enzyme.Const(loss), Enzyme.Active,
                    Enzyme.Const(fact), Enzyme.Const(F)) isa Tuple

                # Forward, width 1 and width 2, and a load with no tangent.
                u̇ = only(Enzyme.autodiff(fwd, Enzyme.Const(solver), Enzyme.Duplicated,
                    Enzyme.Const(fact), Enzyme.Duplicated(copy(F₀), Ḟ1)))
                @test u̇ ≈ A \ Ḟ1 rtol=1e-10
                bat = only(Enzyme.autodiff(fwd, Enzyme.Const(solver), Enzyme.BatchDuplicated,
                    Enzyme.Const(fact), Enzyme.BatchDuplicated(copy(F₀), (Ḟ1, Ḟ2))))
                @test collect(values(bat)) ≈ [A \ Ḟ1, A \ Ḟ2] rtol=1e-10
                u̇ = only(Enzyme.autodiff(fwd, Enzyme.Const(solver), Enzyme.Duplicated,
                    Enzyme.Const(fact), Enzyme.Const(copy(F₀))))
                @test u̇ == zeros(n)

                # In place on the load itself, `ldiv!(uₕ, fact, parent(uₕ))`: the destination
                # and the load share one shadow, so clearing it must come before `F̄ += λ`.
                alias_loss(F) = (u = element(Wk); parent(u) .= F; ldiv!(u, fact, parent(u));
                                 sum(c .* parent(u)))
                Fd = zeros(n)
                Enzyme.autodiff(rev, Enzyme.Const(alias_loss), Enzyme.Active,
                    Enzyme.Duplicated(copy(F₀), Fd))
                @test Fd ≈ A' \ c rtol=1e-10
            end

            # `ldiv!` is in place: with the destination an argument, the cotangent it holds
            # after the call is read and then cleared, and the primal solution is written.
            fact = sparse_factorize(A₀)
            dest_loss(x, fact, F) = (ldiv!(x, fact, F); sum(c .* parent(x)))
            x = element(Wk)
            dx = element(Wk)
            parent(dx) .= 0
            Fd = zeros(n)
            Enzyme.autodiff(rev, Enzyme.Const(dest_loss), Enzyme.Active,
                Enzyme.Duplicated(x, dx), Enzyme.Const(fact), Enzyme.Duplicated(copy(F₀), Fd))
            @test Fd ≈ A₀' \ c rtol=1e-10
            @test all(iszero, parent(dx))
            @test parent(x) ≈ A₀ \ F₀ rtol=1e-10

            # Forward with the destination an argument: its tangent is written even when the
            # return is `Const`, so a shadow that starts as junk ends as `A₀ \ Ḟ`.
            dx = element(Wk)
            parent(dx) .= 7.0
            Enzyme.autodiff(fwd, Enzyme.Const(ldiv!), Enzyme.Const,
                Enzyme.Duplicated(element(Wk), dx), Enzyme.Const(fact),
                Enzyme.Duplicated(copy(F₀), Ḟ1))
            @test parent(dx) ≈ A₀ \ Ḟ1 rtol=1e-10
            parent(dx) .= 7.0
            Enzyme.autodiff(fwd, Enzyme.Const(ldiv!), Enzyme.Const,
                Enzyme.Duplicated(element(Wk), dx), Enzyme.Const(fact), Enzyme.Const(F₀))
            @test all(iszero, parent(dx))

            # `sparse_factorize` called directly: a matrix without a shadow factors in both
            # modes, and a matrix with one is refused, naming the rule that carries it, since
            # a gradient that drops the matrix's derivative would be silently wrong.
            @test Enzyme.autodiff(rev, Enzyme.Const(sparse_factorize), Enzyme.Const,
                Enzyme.Const(A₀)) == ((nothing,),)
            @test Enzyme.autodiff(fwd, Enzyme.Const(sparse_factorize), Enzyme.Const,
                Enzyme.Const(A₀)) == ()
            @test_throws ArgumentError "pde_solve(A, F)" Enzyme.autodiff(
                rev, Enzyme.Const(sparse_factorize), Enzyme.Const,
                Enzyme.Duplicated(copy(A₀), Enzyme.make_zero(A₀)))
            @test_throws ArgumentError "pde_solve(A, F)" Enzyme.autodiff(
                fwd, Enzyme.Const(sparse_factorize), Enzyme.Const,
                Enzyme.Duplicated(copy(A₀), copy(A₀)))

            # A factorization with no adjoint solve is refused in reverse, by name, and solves
            # in forward.
            nofact = NoAdjointFactorization(A₀)
            @test_throws ArgumentError "NoAdjointFactorization" Enzyme.autodiff(
                rev, Enzyme.Const(solve_loss), Enzyme.Active, Enzyme.Const(nofact),
                Enzyme.Duplicated(copy(F₀), zeros(n)))
            u̇ = only(Enzyme.autodiff(fwd, Enzyme.Const(solve_vec), Enzyme.Duplicated,
                Enzyme.Const(nofact), Enzyme.Duplicated(copy(F₀), Ḟ1)))
            @test u̇ ≈ A₀ \ Ḟ1 rtol=1e-10

            # A factorization with a shadow of its own, here a dense `lu` Enzyme can
            # differentiate, is refused by name in both modes and for both solves: the rules
            # treat it as a constant and would drop its derivative. A shadow that is the
            # factorization itself, as runtime activity gives a constant, is no direction.
            Ad = Matrix(A₀)
            Ȧd = zero(Ad)
            Ȧd[2, 3] = 1.0
            act_solve(A, F) = sum(c .* Bramble.pde_solve(lu(A, NoPivot()), F))
            act_ldiv(A, F) = (u = element(Wk); ldiv!(u, lu(A, NoPivot()), F);
                              sum(c .* parent(u)))
            for f in (act_solve, act_ldiv)
                @test_throws ArgumentError "pde_solve(A, F)" Enzyme.autodiff(
                    rev, Enzyme.Const(f), Enzyme.Active,
                    Enzyme.Duplicated(copy(Ad), zero(Ad)), Enzyme.Const(F₀))
                @test_throws ArgumentError "pde_solve(A, F)" Enzyme.autodiff(
                    fwd, Enzyme.Const(f), Enzyme.Duplicated,
                    Enzyme.Duplicated(copy(Ad), Ȧd), Enzyme.Const(F₀))
            end
            luA = lu(Ad, NoPivot())
            for (solver, loss) in ((solve_vec, solve_loss), (ldiv_vec, ldiv_loss))
                Fd = zeros(n)
                Enzyme.autodiff(rev, Enzyme.Const(loss), Enzyme.Active,
                    Enzyme.Duplicated(luA, luA), Enzyme.Duplicated(copy(F₀), Fd))
                @test Fd ≈ A₀' \ c rtol=1e-10
                u̇ = only(Enzyme.autodiff(fwd, Enzyme.Const(solver), Enzyme.Duplicated,
                    Enzyme.Duplicated(luA, luA), Enzyme.Duplicated(copy(F₀), Ḟ1)))
                @test u̇ ≈ A₀ \ Ḟ1 rtol=1e-10
            end

            # Through `assemble`, with the matrix factored once outside the differentiated
            # function: the load is the differentiated input, on a uniform and a non-uniform
            # mesh. The reference is ForwardDiff of `pde_solve(A, F)`.
            for unif in (true, false)
                Wp = gridspace(Bramble.mesh(Bramble.domain(Bramble.interval(0.0, 1.0)), 17, unif))
                bf(p) = (κ = element(Wp, eltype(p)); parent(κ) .= 2.0 .+ p[1];
                         form(Wp, Wp, (U, V) -> inner₊(κ * ∇ₕ(U), ∇ₕ(V)) + innerₕ(U, V)))
                lf = form(Wp, v -> innerₕ(x -> 1.0 + x[1], v))
                sys(p) = ((A, F) = assemble(bf(p), lf; dirichlet = :boundary => x -> 0.0);
                          (A, (1 + p[2]) .* F))
                p0 = [0.5, 0.25]
                Ap = first(sys(p0))
                g = ForwardDiff.gradient(q -> sum(abs2, Bramble.pde_solve(Ap, last(sys(q)))), p0)
                # `A` is fixed, so only the load's parameter moves the solution.
                @test g[1] == 0 && abs(g[2]) > 1e-6
                fa = sparse_factorize(Ap)
                kept(q) = sum(abs2, Bramble.pde_solve(fa, last(sys(q))))
                inplace(q) = (u = element(Wp, eltype(q)); ldiv!(u, fa, last(sys(q)));
                              sum(abs2, parent(u)))
                for m in (Enzyme.Reverse, Enzyme.Forward), f in (kept, inplace)
                    ad = DI.AutoEnzyme(; mode = Enzyme.set_runtime_activity(m),
                        function_annotation = Enzyme.Const)
                    @test DI.gradient(f, ad, p0) ≈ g rtol=1e-8
                end
            end
        else
            @test_skip "Enzyme not in this environment"
        end
    end

    # `fdm_solve` in its right-hand side (#1167). The form is built inside the differentiated
    # function, as a user writes it, so the rule sits on `fdm_solve` and not on a call below
    # it that Enzyme would enter first. Uniform and graded meshes, with and without
    # `dirichlet`, both modes. A coefficient or mesh that moves is refused by name.
    @testset "Enzyme: fdm_solve rules" begin
        if _have(:Enzyme)
            @eval import Enzyme
            rev = Enzyme.set_runtime_activity(Enzyme.Reverse)
            fwd = Enzyme.set_runtime_activity(Enzyme.Forward)
            ez(m) = DI.AutoEnzyme(; mode = Enzyme.set_runtime_activity(m),
                function_annotation = Enzyme.Const)

            for (dims, unif) in (((17, 17), true), ((9, 9), false)),
                dir in (nothing, :boundary)

                Wf = gridspace(Bramble.mesh(Bramble.domain(Bramble.interval(0.0, 1.0) ×
                                                           Bramble.interval(0.0, 1.0)),
                    dims, (unif, unif)))
                nf = ndofs(Wf)
                # `F` is zero on the boundary under `dirichlet`, as `fdm_solve` requires.
                edge = falses(dims)
                edge[[1, end], :] .= true
                edge[:, [1, end]] .= true
                bnd = dir === nothing ? Int[] : findall(vec(edge))
                F₀ = [1.0 + sin(i) for i in 1:nf]
                F₀[bnd] .= 0
                Ḟ1 = [cos(k) for k in 1:nf]
                Ḟ1[bnd] .= 0
                Ḟ2 = [sin(3k) for k in 1:nf]
                Ḟ2[bnd] .= 0
                w = [sin(0.3k) + 0.1 for k in 1:nf]
                af(c) = form(Wf, Wf, (U, V) -> c * inner₊(∇ₕ(U), ∇ₕ(V)) + innerₕ(U, V))
                a = af(1.3)
                kw = dir === nothing ? (;) : (; dirichlet = dir)
                A = Matrix(assemble(a))
                u₀ = fdm_solve(a, F₀; kw...)
                # `F̄ = A⁻ᵀ w`, zero on the boundary under `dirichlet`: the interior solve is
                # embedded back with zero, whatever `F` holds there.
                intr = setdiff(1:nf, bnd)
                F̄ref = zeros(nf)
                F̄ref[intr] = A[intr, intr]' \ w[intr]

                # The form built inside.
                inside(F) = sum(w .* fdm_solve(af(1.3), F; kw...))
                solve_in(F) = fdm_solve(af(1.3), F; kw...)
                # The form built outside, constant.
                outside(F) = sum(w .* fdm_solve(a, F; kw...))
                @test first(Enzyme.gradient(rev, Enzyme.Const(inside), F₀)) ≈ F̄ref rtol=1e-8
                @test first(Enzyme.gradient(rev, Enzyme.Const(outside), F₀)) ≈ F̄ref rtol=1e-8

                # The cotangent adds to what `F̄` holds, the primal load is left alone, and a
                # load used twice sums its two pullbacks.
                Fd = fill(0.5, nf)
                F = copy(F₀)
                Enzyme.autodiff(rev, Enzyme.Const(inside), Enzyme.Active,
                    Enzyme.Duplicated(F, Fd))
                @test Fd ≈ 0.5 .+ F̄ref rtol=1e-8
                @test F == F₀
                twice(F) = sum(w .* fdm_solve(a, F; kw...)) + sum(fdm_solve(a, F; kw...))
                Fd = zeros(nf)
                Enzyme.autodiff(rev, Enzyme.Const(twice), Enzyme.Active,
                    Enzyme.Duplicated(copy(F₀), Fd))
                F̄two = zeros(nf)
                F̄two[intr] = A[intr, intr]' \ (w[intr] .+ 1)
                @test Fd ≈ F̄two rtol=1e-8
                # No cotangent reaches a `Const` load.
                @test Enzyme.autodiff(rev, Enzyme.Const(outside), Enzyme.Active,
                    Enzyme.Const(F₀)) isa Tuple

                # Forward: the tangent is the same solve of `Ḟ`, a `Const` load has none, and
                # a batch is refused by name, as Enzyme aborts on a batched call to a rule
                # whose return type is not concrete.
                u̇ = only(Enzyme.autodiff(fwd, Enzyme.Const(solve_in), Enzyme.Duplicated,
                    Enzyme.Duplicated(copy(F₀), Ḟ1)))
                @test u̇ ≈ fdm_solve(a, Ḟ1; kw...) rtol=1e-10
                @test only(Enzyme.autodiff(fwd, Enzyme.Const(solve_in), Enzyme.Duplicated,
                    Enzyme.Const(F₀))) == zeros(nf)
                @test_throws ArgumentError "batched" Enzyme.autodiff(fwd,
                    Enzyme.Const(solve_in), Enzyme.BatchDuplicated,
                    Enzyme.BatchDuplicated(copy(F₀), (Ḟ1, Ḟ2)))
                @test_throws ArgumentError "batched" Enzyme.autodiff(rev,
                    Enzyme.Const(inside), Enzyme.Active,
                    Enzyme.BatchDuplicated(copy(F₀), (zeros(nf), zeros(nf))))

                # The gradient of a quadratic loss in a scaled load, through DI as a user
                # writes it: the exact derivative is `2(1 + q) S`.
                S = sum(abs2, u₀)
                for m in (Enzyme.Reverse, Enzyme.Forward)
                    @test only(DI.gradient(
                        q -> sum(abs2, fdm_solve(af(1.3), (1 + q[1]) .* F₀; kw...)),
                        ez(m), [0.4])) ≈ 2.8 * S rtol=1e-8
                    # A coefficient that moves is refused, naming both calls.
                    err = try
                        DI.gradient(q -> sum(abs2, fdm_solve(af(q[1]), F₀; kw...)), ez(m), [1.3])
                        nothing
                    catch e
                        e
                    end
                    @test err isa ArgumentError
                    msg = err isa ArgumentError ? err.msg : ""
                    @test occursin("fdm_solve", msg) && occursin("pde_solve(assemble(a, l)..., F)", msg)
                end
            end

            # A `Ref` coefficient set between two solves, the time-loop pattern of
            # `BilinearForm`'s docstring: the pullback of each solve is the adjoint of the
            # operator it solved with, not of the one the coefficient holds at the reverse
            # sweep. `γ` is data, and the loss is quadratic in `θ`, so `d/dθ = 2 loss(θ) / θ`.
            Wr = gridspace(Bramble.mesh(Bramble.domain(Bramble.interval(0.0, 1.0) ×
                                                       Bramble.interval(0.0, 1.0)),
                (9, 9), (false, false)))
            xr = [sin(0.7k) for k in 1:ndofs(Wr)]
            γ = Ref(1.0)
            ar = form(Wr, Wr, (U, V) -> γ * inner₊(∇ₕ(U), ∇ₕ(V)) + innerₕ(U, V))
            function loss_ref(θ)
                γ[] = 1.0
                u = fdm_solve(ar, θ .* xr)
                γ[] = 5.0
                return sum(abs2, fdm_solve(ar, u))
            end
            exact = 2 * loss_ref(0.7) / 0.7
            @test first(Enzyme.gradient(rev, Enzyme.Const(loss_ref), 0.7)) ≈ exact rtol=1e-8
            @test only(Enzyme.autodiff(fwd, Enzyme.Const(loss_ref), Enzyme.Duplicated,
                Enzyme.Duplicated(0.7, 1.0))) ≈ exact rtol=1e-8

            # A form `fdm_solve` always refuses (1D, on a composite space, a ghost form with
            # `dirichlet`): inference proves the call only throws, and Enzyme must still
            # report the named `ArgumentError`, not stop in LLVM on a call with a rule. The forms
            # are module globals, as there Enzyme reaches the call with a rule it must not have.
            W1 = gridspace(Bramble.mesh(Bramble.domain(Bramble.interval(0.0, 1.0)), 11, true))
            global refused_1d = form(W1, W1, (U, V) -> inner₊(∇ₕ(U), ∇ₕ(V)) + innerₕ(U, V))
            Wq = gridspace(Bramble.mesh(Bramble.domain(Bramble.interval(0.0, 1.0) ×
                                                       Bramble.interval(0.0, 1.0)),
                (6, 5), (true, true)))
            Cq = Wq × Wq
            global refused_composite = form(Cq, Cq, (U, V) -> innerₕ(U, V))
            Cb = constrained(gridspace(Bramble.mesh(Bramble.domain(Bramble.interval(0.0, 1.0) ×
                                                                   Bramble.interval(0.0, 1.0)),
                    (9, 8), (true, false)); ghosts = :boundary),
                u -> (:xmin => 2.0 * u + ∂ₙ(u) => 0.0, :xmax => 0.5 * u + 3.0 * ∂ₙ(u) => 1.0,
                    :ymin => u + ∂ₙ(u) => 0.0, :ymax => 4.0 * u + ∂ₙ(u) => 0.0))
            global refused_ghost = form(Cb, Cb, (u, v) -> innerₕ(u, v) + 1.3 * inner₊(∇ₕ(u), ∇ₕ(v)))
            solve_1d(F) = fdm_solve(refused_1d, F)
            solve_composite(F) = fdm_solve(refused_composite, F)
            solve_ghost(F) = fdm_solve(refused_ghost, F; dirichlet = :boundary)
            loss_1d(F) = sum(solve_1d(F))
            loss_composite(F) = sum(solve_composite(F))
            loss_ghost(F) = sum(solve_ghost(F))
            for (solver, loss, nx) in ((solve_1d, loss_1d, ndofs(W1)),
                (solve_composite, loss_composite, ndofs(Cq)),
                (solve_ghost, loss_ghost, ndofs(Cb)))

                @test_throws ArgumentError Enzyme.autodiff(rev, Enzyme.Const(loss),
                    Enzyme.Active, Enzyme.Duplicated(ones(nx), zeros(nx)))
                @test_throws ArgumentError Enzyme.autodiff(fwd, Enzyme.Const(solver),
                    Enzyme.Duplicated, Enzyme.Duplicated(ones(nx), ones(nx)))
            end

            # A form with an advection term takes the generalised Schur route, whose system is
            # not symmetric: forward solves the tangent, and reverse is refused by name, since
            # `fdm_solve(a, ū)` is not its adjoint.
            Wc = gridspace(Bramble.mesh(Bramble.domain(Bramble.interval(0.0, 1.0) ×
                                                       Bramble.interval(0.0, 1.0)),
                (9, 9), (false, false)))
            nc = ndofs(Wc)
            adv = form(Wc, Wc, (U, V) -> innerₕ(U, V) + inner₊(∇ₕ(U), ∇ₕ(V)) +
                                         innerₕ(Bramble.D₋ₓ(U), V))
            Aadv = Matrix(assemble(adv))
            Fc = [1.0 + sin(i) for i in 1:nc]
            Ḟc = [cos(k) for k in 1:nc]
            @test maximum(abs, Aadv \ Ḟc - Aadv' \ Ḟc) > 1e-3
            adv_solve(F) = fdm_solve(adv, F)
            @test only(Enzyme.autodiff(fwd, Enzyme.Const(adv_solve), Enzyme.Duplicated,
                Enzyme.Duplicated(copy(Fc), Ḟc))) ≈ Aadv \ Ḟc rtol=1e-8
            @test_throws ArgumentError "Schur" Enzyme.autodiff(rev,
                Enzyme.Const(F -> sum(adv_solve(F))), Enzyme.Active,
                Enzyme.Duplicated(copy(Fc), zeros(nc)))
        else
            @test_skip "Enzyme not in this environment"
        end
    end

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
                # `2(1 + q) S`. A coefficient that moves gives `build` a shadow, refused by
                # name.
                S = sum(abs2, A \ b₀)
                for m in (Enzyme.Reverse, Enzyme.Forward)
                    @test only(DI.gradient(
                        q -> sum(abs2, parent(gmg_solve(bld, Ωg, (1 + q[1]) .* b₀; kw...))),
                        ez(m), [0.4])) ≈ 2.8 * S rtol=1e-8
                    @test_throws ArgumentError "`build` has a shadow" DI.gradient(
                        q -> sum(abs2, parent(gmg_solve(
                            V -> form(V, V, (u, v) -> q[1] * innerₕ(u, v) +
                                                     inner₊(∇ₕ(u), ∇ₕ(v))),
                            Ωg, b₀; kw...))),
                        ez(m), [1.3])
                end
            end

            # A `Ref` set between two solves. Read by `build` (`γ[]`), the form holds its value,
            # and the pullback of each solve is the adjoint of the operator it solved with: the
            # loss is quadratic in `θ`, so `d/dθ = 2 loss(θ) / θ`. Held by the form (`γ`), it
            # is read at each product, and reverse refuses the changed operator by name.
            Ωr = Bramble.mesh(Bramble.domain(Bramble.interval(0.0, 1.0) ×
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
            Ωt = Bramble.mesh(Bramble.domain(Bramble.interval(0.0, 1.0) ×
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
            Ωu = Bramble.mesh(Bramble.domain(Bramble.interval(0.0, 1.0) ×
                                             Bramble.interval(0.0, 1.0)), (17, 17), (true, true))
            Wu = gridspace(Ωu)
            nu = ndofs(Wu)
            cs = (Ref(0.0), Ref(0.0), Ref(0.0), Ref(0.0))
            terms = (V -> form(V, V, (u, v) -> innerₕ(Bramble.D₋ᵧ(u), v)),
                V -> form(V, V, (u, v) -> innerₕ(Rₕ(V, x -> x[1]) * Bramble.D₋ₓ(u), v)),
                V -> form(V, V, (u, v) -> innerₕ(Rₕ(V, x -> x[2]) * Bramble.D₋ᵧ(u), v)),
                V -> form(V, V, (u, v) -> innerₕ(Rₕ(V, x -> x[1] * x[2]) * Bramble.D₋ₓ(u), v)))
            four(V) = form(V, V, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)) +
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
            Ωw = Bramble.mesh(Bramble.domain(Bramble.interval(0.0, 1.0) ×
                                             Bramble.interval(0.0, 1.0)), (33, 33), (true, true))
            weak(V) = form(V, V, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)) +
                                           1e-4 * innerₕ(Bramble.D₋ₓ(u), v))
            nw = npoints(Ωw)
            @test_throws ArgumentError "symmetric" Enzyme.autodiff(rev,
                Enzyme.Const(b -> sum(parent(gmg_solve(weak, Ωw, b; tol = 1e-10)))),
                Enzyme.Active, Enzyme.Duplicated(ones(nw), zeros(nw)))

            # A `build` capturing a scalar. Forward reads its shadow's data: a constant `κ`
            # leaves the tangent `A⁻¹ ḃ`, and a moving `c` is refused by name even when its
            # tangent equals its value. Reverse gets such a `build` as `Active`, which cannot be
            # told constant, and refuses it by name.
            κ = 0.7
            Aκ = Matrix(assemble(form(Wa, Wa, (u, v) -> innerₕ(u, v) + κ * inner₊(∇ₕ(u), ∇ₕ(v)))))
            solve_κ(b) = gmg_solve(V -> form(V, V, (u, v) -> innerₕ(u, v) +
                                                            κ * inner₊(∇ₕ(u), ∇ₕ(v))), Ωr, b;
                tol = 1e-10)
            ḃr = [cos(k) for k in 1:ndofs(Wa)]
            @test parent(only(Enzyme.autodiff(fwd, Enzyme.Const(solve_κ), Enzyme.Duplicated,
                Enzyme.Duplicated(copy(xr), ḃr)))) ≈ Aκ \ ḃr rtol=1e-7
            @test_throws ArgumentError "captures scalar values" Enzyme.autodiff(rev,
                Enzyme.Const(b -> sum(parent(solve_κ(b)))), Enzyme.Active,
                Enzyme.Duplicated(copy(xr), zeros(length(xr))))
            loss_c(c) = sum(abs2, parent(gmg_solve(V -> form(V, V, (u, v) -> c * innerₕ(u, v) +
                                                                          inner₊(∇ₕ(u), ∇ₕ(v))),
                Ωr, xr; tol = 1e-10)))
            for c in (1.0, 2.0)
                @test_throws ArgumentError "`build` has a shadow" Enzyme.autodiff(fwd,
                    Enzyme.Const(loss_c), Enzyme.Duplicated, Enzyme.Duplicated(c, c))
            end
            # The shadow is read with its primal. An active coefficient inside a vector of
            # tuples is a derivative the rule cannot compute, refused by name, as `pde_solve`
            # would carry it; a constant `κ` captured beside a `Ref`, whose shadow is the
            # primal's own, leaves the tangent `A⁻¹ ḃ`.
            loss_t(q) = (ps = [(q[1], 0.0)];
            sum(abs2, parent(gmg_solve(V -> form(V, V, (u, v) -> ps[1][1] * innerₕ(u, v) +
                                                                inner₊(∇ₕ(u), ∇ₕ(v))),
                Ωr, xr; tol = 1e-10))))
            @test_throws ArgumentError "`build` has a shadow" Enzyme.autodiff(fwd,
                Enzyme.Const(loss_t), Enzyme.Duplicated, Enzyme.Duplicated([0.7], [1.0]))
            loss_w(q) = (cw = WrappedCoefficient(q[1]);
            sum(abs2, parent(gmg_solve(V -> form(V, V, (u, v) -> cw.κ * innerₕ(u, v) +
                                                                inner₊(∇ₕ(u), ∇ₕ(v))),
                Ωr, xr; tol = 1e-10))))
            @test_throws ArgumentError "`build` has a shadow" Enzyme.autodiff(fwd,
                Enzyme.Const(loss_w), Enzyme.Duplicated, Enzyme.Duplicated([0.7], [1.0]))
            γκ = Ref(1.0)
            solve_γκ(b) = gmg_solve(V -> form(V, V, (u, v) -> κ * γκ[] * inner₊(∇ₕ(u), ∇ₕ(v)) +
                                                             innerₕ(u, v)), Ωr, b; tol = 1e-10)
            @test parent(only(Enzyme.autodiff(fwd, Enzyme.Const(solve_γκ), Enzyme.Duplicated,
                Enzyme.Duplicated(copy(xr), ḃr)))) ≈ Aκ \ ḃr rtol=1e-7
        else
            @test_skip "Enzyme not in this environment"
        end
    end

    @testset "Enzyme: pde_solve(::RankOneUpdate) rules" begin
        if _have(:Enzyme)
            @eval import Enzyme
            rev = Enzyme.set_runtime_activity(Enzyme.Reverse)
            fwd = Enzyme.set_runtime_activity(Enzyme.Forward)
            ROU = Bramble.RankOneUpdate

            # A non-symmetric `A` with `wₛ ≠ wₜ`, so a swapped border or a missing transpose
            # changes the gradient. Every input is a float of one vector `p`: `A`'s stored
            # entries, `σ`, `wₛ`, `wₜ`, `F`, and the references are ForwardDiff of the dense
            # solve of the same system, with the border `[K w; wᵀ 0]` for `mean_zero`.
            n = 6
            A0 = sparse(spdiagm(-1 => fill(-1.0, n - 1), 0 => fill(2.5, n),
                1 => fill(-1.2, n - 1)))
            nA = nnz(A0)
            σ0 = 0.8
            ws0 = [1.0 + 0.1 * i for i in 1:n]
            wt0 = [cos(i) for i in 1:n]
            F0 = [1.0 + sin(i) for i in 1:n]
            c = [sin(2i) for i in 1:n]
            wmz = [1.0 + 0.2 * cos(i) for i in 1:n]
            p0 = [nonzeros(A0); σ0; ws0; wt0; F0]
            ia = 1:nA
            iσ = nA + 1
            iws = nA .+ (2:(n + 1))
            iwt = nA .+ ((n + 2):(2n + 1))
            iF = nA .+ ((2n + 2):(3n + 1))
            function dense(p, border)
                T = eltype(p)
                Ad = zeros(T, n, n)
                k = 0
                for j in 1:n, i in A0.rowval[A0.colptr[j]:(A0.colptr[j + 1] - 1)]
                    Ad[i, j] = p[k += 1]
                end
                Kd = Ad + p[iσ] * p[iws] * transpose(p[iwt])
                border === nothing && return Kd \ p[iF]
                return ([Kd border; transpose(border) 0] \ [p[iF]; 0])[1:n]
            end
            sparsify(vals) = SparseMatrixCSC(n, n, A0.colptr, A0.rowval, vals)
            function rou(p)
                K = ROU(sparsify(p[ia]), p[iσ], p[iws], p[iwt])
                return K, p[iF]
            end
            # The shadow of `K` for one tangent or cotangent vector `v` of the layout of `p`.
            shadow(v) = ROU(sparsify(v[ia]), v[iσ], v[iws], v[iwt])
            unshadow(Kd, Fd) = [nonzeros(Kd.A); Kd.σ; Kd.ws; Kd.wt; Fd]

            for border in (nothing, wmz)
                kw = border === nothing ? (;) : (; mean_zero = border)
                solve(K, F) = Bramble.pde_solve(K, F; kw...)
                loss(K, F) = sum(c .* solve(K, F))
                ref_grad = ForwardDiff.gradient(p -> sum(c .* dense(p, border)), p0)

                # Reverse: the cotangent of every field at once, `K` and `F` both active, then
                # `K` alone (`F` constant), then `F` alone (`K` constant).
                K, F = rou(p0)
                Kd = shadow(zeros(length(p0)))
                Fd = zeros(n)
                Enzyme.autodiff(rev, Enzyme.Const(loss), Enzyme.Active,
                    Enzyme.Duplicated(K, Kd), Enzyme.Duplicated(F, Fd))
                @test unshadow(Kd, Fd) ≈ ref_grad rtol=1e-8
                @test unshadow(K, F) == p0     # the primal is not touched

                K, F = rou(p0)
                Kd = shadow(zeros(length(p0)))
                Enzyme.autodiff(rev, Enzyme.Const(loss), Enzyme.Active,
                    Enzyme.Duplicated(K, Kd), Enzyme.Const(F))
                @test unshadow(Kd, zeros(n))[1:(end - n)] ≈ ref_grad[1:(end - n)] rtol=1e-8

                K, F = rou(p0)
                Fd = zeros(n)
                Enzyme.autodiff(rev, Enzyme.Const(loss), Enzyme.Active,
                    Enzyme.Const(K), Enzyme.Duplicated(F, Fd))
                @test Fd ≈ ref_grad[iF] rtol=1e-8

                # Forward, width 1 and width 2: `J ṗ` for two directions.
                J = ForwardDiff.jacobian(p -> dense(p, border), p0)
                ṗ1 = [cos(k) for k in 1:length(p0)]
                ṗ2 = [sin(3k) for k in 1:length(p0)]
                K, F = rou(p0)
                u̇ = only(Enzyme.autodiff(fwd, Enzyme.Const(solve), Enzyme.Duplicated,
                    Enzyme.Duplicated(K, shadow(ṗ1)), Enzyme.Duplicated(F, ṗ1[iF])))
                @test u̇ ≈ J * ṗ1 rtol=1e-8

                # `K` with every field perturbed, `A` alone, then `F` alone.
                u̇ = only(Enzyme.autodiff(fwd, Enzyme.Const(solve), Enzyme.Duplicated,
                    Enzyme.Duplicated(K, shadow(ṗ1)), Enzyme.Const(F)))
                ṗA = [ṗ1[ia]; zeros(length(p0) - nA)]
                u̇ = only(Enzyme.autodiff(fwd, Enzyme.Const(solve), Enzyme.Duplicated,
                    Enzyme.Duplicated(K, shadow(ṗA)), Enzyme.Const(F)))
                @test u̇ ≈ J * ṗA rtol=1e-8
                u̇ = only(Enzyme.autodiff(fwd, Enzyme.Const(solve), Enzyme.Duplicated,
                    Enzyme.Const(K), Enzyme.Duplicated(F, ṗ1[iF])))
                @test u̇ ≈ J[:, iF] * ṗ1[iF] rtol=1e-8

                bat = only(Enzyme.autodiff(fwd, Enzyme.Const(solve), Enzyme.BatchDuplicated,
                    Enzyme.BatchDuplicated(K, (shadow(ṗ1), shadow(ṗ2))),
                    Enzyme.BatchDuplicated(F, (ṗ1[iF], ṗ2[iF]))))
                @test collect(values(bat)) ≈ [J * ṗ1, J * ṗ2] rtol=1e-8
            end

            # A `K.A` that is not a `SparseMatrixCSC` is refused by name, in both modes, and
            # reverse mode does not batch: the rule's own messages, not any `ArgumentError`.
            Kw = ROU(adjoint(copy(A0)), σ0, ws0, wt0)
            Kwd = ROU(adjoint(0 .* A0), 0.0, zeros(n), zeros(n))
            solve_w(K, F) = sum(Bramble.pde_solve(K, F))
            @test_throws ArgumentError "needs `K.A` to be a `SparseMatrixCSC`" Enzyme.autodiff(
                rev, Enzyme.Const(solve_w), Enzyme.Active, Enzyme.Duplicated(Kw, Kwd),
                Enzyme.Const(F0))
            @test_throws ArgumentError "needs `K.A` to be a `SparseMatrixCSC`" Enzyme.autodiff(
                fwd, Enzyme.Const(solve_w), Enzyme.Duplicated, Enzyme.Duplicated(Kw, Kwd),
                Enzyme.Const(F0))
            @test_throws ArgumentError "does not support batched" Enzyme.autodiff(rev,
                Enzyme.Const(solve_w), Enzyme.Active,
                Enzyme.BatchDuplicated(ROU(copy(A0), σ0, ws0, wt0),
                    (shadow(zeros(length(p0))), shadow(zeros(length(p0))))),
                Enzyme.Const(F0))

            # Through `assemble`, with a constant border `w` captured by the closure and used
            # as `wₛ` and `wₜ`: under runtime activity Enzyme gives such a field its own array
            # as its shadow, which is a zero tangent (forward) and not ours to write
            # (reverse). The reference is ForwardDiff of the plain solve of `A + w wᵀ`.
            for unif in (true, false)
                Wp = gridspace(mesh(domain(interval(0.0, 1.0)), 17, unif))
                np = ndofs(Wp)
                wp = ones(np)
                Fp = [1.0 + sin(i) for i in 1:np]
                mat(q) = (κ = element(Wp, eltype(q)); parent(κ) .= 2.0 .+ q[1];
                          assemble(form(Wp, Wp, (U, V) -> inner₊(κ * ∇ₕ(U), ∇ₕ(V)))))
                f(q) = sum(abs2, Bramble.pde_solve(ROU(mat(q), 1.0, wp, wp), (1 + q[2]) .* Fp))
                ref(q) = sum(abs2, Bramble.pde_solve(mat(q) + sparse(wp * wp'), (1 + q[2]) .* Fp))
                q0 = [0.5, 0.25]
                g = ForwardDiff.gradient(ref, q0)
                @test minimum(abs, g) > 1e-6 * maximum(abs, g)
                @test f(q0) ≈ ref(q0) rtol=1e-9
                for m in (Enzyme.Reverse, Enzyme.Forward)
                    ad = DI.AutoEnzyme(; mode = Enzyme.set_runtime_activity(m),
                        function_annotation = Enzyme.Const)
                    @test DI.gradient(f, ad, q0) ≈ g rtol=1e-8
                end
            end
        else
            @test_skip "Enzyme not in this environment"
        end
    end
end

end # module
