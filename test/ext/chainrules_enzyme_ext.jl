module ExtChainRulesEnzymeExtTests

using Test
using Bramble
using ChainRulesCore
using SparseArrays: SparseMatrixCSC, nnz, nonzeros, sparse, spdiagm
using LinearAlgebra: I
using Polyester
using ForwardDiff
using SuiteSparse
using Bramble: CpuSerial, CpuThreaded, CpuPolyester
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
end

end # module
