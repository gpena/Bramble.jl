module ExtChainRulesEnzymeExtTests

using Test
using Bramble
using ChainRulesCore
using SparseArrays: SparseMatrixCSC
using LinearAlgebra: I

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

_have(mod::Symbol) = Base.identify_package(String(mod)) !== nothing
_central_diff(f, x, h = 1e-6) = (f(x + h) - f(x - h)) / 2h

@testset "BrambleChainRulesExt + Enzyme" begin
    @testset "Enzyme, with @import_rrule" begin
        if _have(:Enzyme)
            @eval import Enzyme
            @eval Enzyme.@import_rrule(typeof(Bramble.pde_solve), SparseMatrixCSC, AbstractVector)

            Ωₕ = Bramble.mesh(Bramble.domain(Bramble.interval(0.0, 1.0)), 21, true)
            Wₕ = gridspace(Ωₕ)
            a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))

            function loss(θ::Real)
                fₕ = Bramble.element(Wₕ, typeof(θ))
                l = form(Wₕ, v -> innerₕ(fₕ, v))
                A, F = assemble(a, l; dirichlet = :boundary => x -> θ)
                u = Bramble.pde_solve(A, F)
                return sum(abs2, u)
            end

            θ0 = 0.7
            d_fd = _central_diff(loss, θ0)

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

            # Pinned failure: without `@import_rrule`, Enzyme tries to trace into `lu`'s
            # CHOLMOD/UMFPACK internals directly and raises `IllegalTypeAnalysisException` on
            # a `Union` type it cannot strictly alias -- if a future Enzyme release fixes
            # this on its own, this test starts failing and says so, the same "pin it so a
            # fix is noticed" discipline `autodiff_heavy.jl` already follows for its own two
            # annotations.
            function loss_no_bridge(θ::Real)
                A, F = assemble(a, form(Wₕ, v -> innerₕ(Bramble.element(Wₕ, typeof(θ)), v));
                    dirichlet = :boundary => x -> θ)
                return sum(abs2, A \ F)
            end
            @test_throws Exception Enzyme.gradient(mode, Enzyme.Const(loss_no_bridge), θ0)
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
            @eval Mooncake.@from_rrule(
                Mooncake.DefaultCtx,
                Tuple{typeof(Bramble.pde_solve), SparseMatrixCSC, AbstractVector},
                false
            )
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
