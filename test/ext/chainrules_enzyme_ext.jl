module ExtChainRulesEnzymeExtTests

using Test
using Bramble
using ChainRulesCore
using SparseArrays: SparseMatrixCSC, nnz, nonzeros, sparse
using LinearAlgebra: I
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

const ENZYME_TOML = "[deps]\nEnzyme = \"7da242da-08ed-463a-9acd-ee780be4f1d9\"\n"

# `name`'s direct Enzyme gradient at 0.7 and a central difference of it, from a child Julia
# process with the parent's project, thread count and optimisation level (Enzyme comes from
# a temporary environment, as in `enzyme_policy_cases.jl`). Returns the exit code, the two
# values (`NaN` when the child died before printing them) and the child's whole output.
function _child_gradient(name::AbstractString)
    src = """
        _e = mktempdir()
        write(joinpath(_e, "Project.toml"), $(repr(ENZYME_TOML)))
        push!(LOAD_PATH, _e)
        using Bramble, Enzyme
        $CHILD_LOSSES_2D
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
            # removing the like-term rule, gpena/Bramble.jl#771). Enzyme's compiler now aborts
            # on all three, with no Julia exception. The LLVM assertion is
            # `needsCacheWholeAllocation` (GradientUtils.cpp:10433), reached through
            # `_dirichlet_probed_eltype` in `assemble`, and it is the same one that
            # `docs/notes/internals/enzyme_policies.md` records for a keyword `innerₕ` call.
            # The limit is Enzyme's and the plan accepts it. The same file at `HEAD`, before
            # the keyword rule existed, aborts at `loss_stiff_2d` identically, and the 3D
            # stiffness loss above compiles. An abort ends the whole process, so each loss runs
            # in a child and `@test_broken` pins it. A fixed Enzyme turns that into an
            # unexpected pass, which says so.
            for name in ("loss_stiff_2d", "loss_field_2d", "loss_two_fields_2d")
                code, d_ad, d_cd, text = _child_gradient(name)
                @test code == 0 || occursin("needsCacheWholeAllocation", text)
                @test_broken code == 0 && isapprox(d_ad, d_cd; rtol = 1e-3)
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
            # Kept a direct Enzyme call, since the pin is on Enzyme's own exception.
            @test_throws Exception Enzyme.gradient(mode, Enzyme.Const(loss_no_rule), θ0)

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
