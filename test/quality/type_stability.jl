module QualityTypeStabilityTests

using Test
using JET
using Bramble
using Bramble: Dcₕ, D̃ₕ, D̽ₕ, matrix_free_operator, pde_solve
using Bramble: D₋ₓ, D₋ᵧ, D₋₂, jumpₓ, jumpᵧ, jump₂, Mₓ, Mᵧ, M₂
using Bramble: spacings, half_spacings, cell_measures, points
# Internal since v3.0: defined and documented, not exported.
import Bramble: diff₋ₓ, diff₋ᵧ, diff₋₂, diff₋ₕ, diff₊ₓ, diff₊ᵧ, diff₊₂, diff₊ₕ, D₊ₓ, D₊ᵧ, D₊₂, ∇₊ₕ,
                M₊ₓ, M₊ᵧ, M₊₂, M₊ₕ
using ForwardDiff
using LinearAlgebra: mul!
using ..TestUtils: _fd, _have, _run_gpu_tests, _check_eoc, _tri, _nonuniform_points,
                   _zero_boundary!, _matches_fd, alloc_test

# JET's optimisation analysis over every public path: construction, operator application,
# assembly, solve, the matrix-free apply and ForwardDiff assembly, in 1D, 2D and 3D and on a
# composite space. `@test_opt` sees runtime dispatch anywhere inside the call, not only an
# unstable return type, which is what `@inferred` checks. A call pinned here is not also
# pinned by `@inferred` elsewhere in test/ (test/quality/type_stability.jl is the owner).
#
# Meshes are non-uniform throughout: a uniform mesh takes a different spacing path.
#
# `target_modules = (Bramble,)` restricts reports to frames inside Bramble. Dispatch inside
# Base, SparseArrays or ForwardDiff is theirs to fix, not a Bramble instability.

const TM = (Bramble,)

function _meshes()
    (mesh(domain(interval(0.0, 1.0)), 17, false),
        mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (8, 9), (false, true)),
        mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (4, 5, 6), false))
end

_diffusion(W) = form(W, W, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
_load(W) = form(W, v -> innerₕ(1.0, v))

# Functions of the ForwardDiff parameter, so the dual type flows through assembly.
_fd_bilinear(W, s) = assemble(form(W, W, (u, v) -> innerₕ(u, v) + Ref(s) * inner₊(∇ₕ(u), ∇ₕ(v))))
_fd_linear(W, s) = assemble(form(W, v -> innerₕ(1.0, v) + Ref(s) * innerₕ(x -> 1.0 + x[1], v)))

# Oracle controls: a local function with a deliberate runtime dispatch (a `Vector{Any}`
# element added to an Int), and a closure returning through a `Ref{Any}`.
module Control
g(v) = v[1] + 1
const R = Ref{Any}(1.0)
h(x) = R[]
end

@testset "Type stability (JET)" begin
    Ωₕ1, Ωₕ2, Ωₕ3 = _meshes()
    Wₕ1, Wₕ2, Wₕ3 = gridspace(Ωₕ1), gridspace(Ωₕ2), gridspace(Ωₕ3)
    uₕ1 = Rₕ(Wₕ1, sin)
    uₕ2 = Rₕ(Wₕ2, x -> sin(x[1]) * x[2])
    uₕ3 = Rₕ(Wₕ3, x -> sin(x[1]) + x[3])

    @testset "Construction" begin
        @test_opt target_modules = TM mesh(domain(interval(0.0, 1.0)), 17, false)
        @test_opt target_modules = TM mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (8, 9), (false, true))
        for Ωₕ in (Ωₕ1, Ωₕ2, Ωₕ3)
            @test_opt target_modules = TM gridspace(Ωₕ)
            @test_opt target_modules = TM spacings(Ωₕ)
            @test_opt target_modules = TM half_spacings(Ωₕ)
            @test_opt target_modules = TM cell_measures(Ωₕ)
            @test_opt target_modules = TM points(Ωₕ)
        end
        for Wₕ in (Wₕ1, Wₕ2, Wₕ3)
            @test_opt target_modules = TM element(Wₕ)
            @test_opt target_modules = TM element(Wₕ, 0.0)
            @test_opt target_modules = TM Rₕ(Wₕ, x -> x[1])
            @test_opt target_modules = TM avgₕ(Wₕ, x -> x[1])
            @test_opt target_modules = TM _diffusion(Wₕ)
            @test_opt target_modules = TM _load(Wₕ)
        end
        # `reassemble` left at its default must fold into `Val(false)`, whatever
        # the (non-constant) Dirichlet labels.
        sd_closure(a, l, d::Symbol) = semidiscretize(a, l; dirichlet = d)
        sd_args = (typeof(_diffusion(Wₕ1)), typeof(_load(Wₕ1)), Symbol)
        @test isconcretetype(only(Base.return_types(sd_closure, sd_args)))
        @test_opt target_modules = TM sd_closure(_diffusion(Wₕ1), _load(Wₕ1), :boundary)
    end

    @testset "Operator application" begin
        for uₕ in (uₕ1, uₕ2, uₕ3)
            @test_opt target_modules = TM ∇ₕ(uₕ)
            @test_opt target_modules = TM Δₕ(uₕ)
            @test_opt target_modules = TM innerₕ(uₕ, uₕ)
            @test_opt target_modules = TM inner₊(uₕ, uₕ)
            @test_opt target_modules = TM normₕ(uₕ)
            @test_opt target_modules = TM snorm₁ₕ(uₕ)
            @test_opt target_modules = TM norm₁ₕ(uₕ)
        end
        # The scalar operators, one direction per dimension.
        for (uₕ, ops) in ((uₕ1, (diff₋ₓ, diff₊ₓ, D₋ₓ, D₊ₓ, jumpₓ, Mₓ, M₊ₓ)),
                (uₕ2, (diff₋ᵧ, diff₊ᵧ, D₋ᵧ, D₊ᵧ, jumpᵧ, Mᵧ, M₊ᵧ)),
                (uₕ3, (diff₋₂, diff₊₂, D₋₂, D₊₂, jump₂, M₂, M₊₂))),
            op in ops

            @test_opt target_modules = TM op(uₕ)
        end
        # Moved from test/space/inference_allocation.jl: the 2D/3D
        # vectorial aliases once built their methods from `ntuple(i -> op(u, Val(i)))`,
        # boxing `i` and dispatching dynamically down the difference engine.
        for op in (∇ₕ, ∇₊ₕ, diff₋ₕ, diff₊ₕ, jumpₕ, Mₕ, M₊ₕ, D̃ₕ, Dcₕ, D̽ₕ), vₕ in (uₕ2, uₕ3)

            @test_opt target_modules = TM op(vₕ)
        end
    end

    @testset "Assembly" begin
        for Wₕ in (Wₕ1, Wₕ2, Wₕ3)
            aₕ = _diffusion(Wₕ)
            lₕ = _load(Wₕ)
            A = assemble(aₕ)
            @test_opt target_modules = TM assemble(aₕ)
            @test_opt target_modules = TM assemble(lₕ)
            @test_opt target_modules = TM assemble!(A, aₕ)
        end
    end

    @testset "Solve" begin
        for Wₕ in (Wₕ1, Wₕ2, Wₕ3)
            A = assemble(_diffusion(Wₕ))
            F = assemble(_load(Wₕ))
            @test_opt target_modules = TM pde_solve(A, F)
        end
    end

    @testset "Matrix-free apply" begin
        for Wₕ in (Wₕ1, Wₕ2, Wₕ3)
            a = _diffusion(Wₕ)
            op = matrix_free_operator(a)
            x = ones(ndofs(Wₕ))
            y = similar(x)
            @test_opt target_modules = TM matrix_free_operator(a)
            @test_opt target_modules = TM mul!(y, op, x)
        end
    end

    @testset "ForwardDiff assembly" begin
        d = ForwardDiff.Dual(2.0, 1.0)
        for Wₕ in (Wₕ1, Wₕ2, Wₕ3)
            @test_opt target_modules = TM _fd_bilinear(Wₕ, d)
            @test_opt target_modules = TM _fd_linear(Wₕ, d)
        end
    end

    @testset "Composite spaces" begin
        Vₕ = Wₕ2 × Wₕ2
        vₕ = Rₕ(Vₕ, (x -> x[1], x -> x[2]))
        aᵥ = form(Vₕ, Vₕ, (u, v) -> innerₕ(u(1), v(1)) + inner₊(∇ₕ(u(2)), ∇ₕ(v(2))))
        A = assemble(aᵥ)
        op = matrix_free_operator(aᵥ)
        x = ones(ndofs(Vₕ))
        y = similar(x)
        @test_opt target_modules = TM Rₕ(Vₕ, (x -> x[1], x -> x[2]))
        @test_opt target_modules = TM element(Vₕ)
        @test_opt target_modules = TM innerₕ(vₕ, vₕ)
        @test_opt target_modules = TM assemble(aᵥ)
        @test_opt target_modules = TM assemble!(A, aᵥ)
        @test_opt target_modules = TM mul!(y, op, x)
    end

    # Not among the eight testsets the CHECK names: it proves the harness above can fail.
    @testset "Oracle control" begin
        # Mechanics: JET's optimisation analysis reports a known runtime dispatch.
        @test !isempty(JET.get_reports(
            JET.report_opt(Control.g, (Vector{Any},); target_modules = (Control,))))
        # Scope: `target_modules = (Bramble,)` keeps dispatch inside Bramble's own frames.
        # A closure with an `Any` return drives `Rₕ` into dispatch in
        # `_restriction_eltype`, `element` and `_serial_for!`, all Bramble frames, and the
        # filter the testsets above use reports it.
        @test !isempty(JET.get_reports(
            JET.report_opt(Rₕ, (typeof(Wₕ1), typeof(Control.h)); target_modules = TM)))
    end

    @testset "Test helpers" begin
        @test @inferred(_fd(sin, 1.0)) isa Float64
        @test @inferred(_have(:Test)) isa Bool
        @test @inferred(_run_gpu_tests()) isa Bool
        @test @inferred(_tri(5)) isa AbstractMatrix{Float64}
        @test @inferred(_nonuniform_points([0.1, 0.3, 0.2])) isa Vector{Float64}
        @test @inferred(_zero_boundary!(ones(3, 3))) isa Matrix{Float64}
        @test @inferred(_matches_fd(sin)) isa Bool
        @test @inferred(alloc_test(sin, 1.0)) isa Int
        @test @inferred(_check_eoc(n -> (1.0 / n^2, 1.0 / n), [4, 8, 16])) isa Vector{Float64}
    end

    # gpena/Bramble.jl#811. `constrained` builds one space type per call: the relations are
    # a tuple of concrete types, and whether a datum depends on time is read off its type.
    # The cases of the ghost precompile workload, and data in time.
    @testset "Constrained ghost spaces" begin
        W1 = gridspace(Ωₕ1; ghosts = :boundary)
        W2 = gridspace(Ωₕ2; ghosts = :boundary)
        K = [2.0 0.5; 0.5 1.0]
        F4 = (:xmin, :xmax, :ymin, :ymax)
        robin(u) = (:xmin => 2.0 * u + ∂ₙ(u) => 0.5, :xmax => ∂ₙ(u) => x -> x[1])
        mixed(u) = (:xmin => u => 0.0, :xmin => ∂ₙ(u) => 1.0,
            :xmax => 1.5 * u + ∂ₙₙ(u) => 1.0)
        timed(u) = (:xmin => ∂ₙ(u) => (x, t) -> sin(t), :xmax => u + ∂ₙ(u) => 2.0)
        @test @inferred(constrained(W1, robin)) isa Bramble.GhostGridSpace
        @test @inferred(constrained(W1, mixed)) isa Bramble.GhostGridSpace
        C = @inferred constrained(W1, timed)
        @test Bramble._is_time_dependent(C)
        @test @inferred(constrained(W2, u -> map(F -> F => 2.0 * u + ∂ₙ(u) => 0.5, F4))) isa
              Bramble.GhostGridSpace
        @test @inferred(constrained(W2,
            u -> map(F -> F => 2.0 * u + ∂ₙ(u; K) => 1.0, F4))) isa Bramble.GhostGridSpace
        # On a composite, whether a facet has a relation reading several components is a
        # value of the space, not part of its type.
        one_each(u) = (:xmin => ∂ₙ(u(1)) => 1.0, :xmax => ∂ₙ(u(1)) => 0.0,
            :xmin => ∂ₙ(u(2)) => 0.0, :xmax => u(2) + ∂ₙ(u(2)) => 0.0)
        @test @inferred(constrained(W1 × W1, one_each)) isa
              Bramble.ConstrainedCompositeSpace
        one_each2(u) = (one_each(u)..., :ymin => ∂ₙ(u(1)) => 0.0, :ymax => ∂ₙ(u(1)) => 0.0,
            :ymin => ∂ₙ(u(2)) => 0.0, :ymax => ∂ₙ(u(2)) => 0.0)
        @test @inferred(constrained(W2 × W2, one_each2)) isa
              Bramble.ConstrainedCompositeSpace
        traction(u) = map(F -> F => ∂ₙ(u(1); K) + ∂ₙ(u(2); K) => 0.0, F4)
        sym(u) = (traction(u)..., map(F -> F => ∂ₙ(u(1); K) - ∂ₙ(u(2); K) => 0.0, F4)...)
        T2 = @inferred constrained(W2 × W2, sym)
        @test any(any, T2.facets)
    end

    # gpena/Bramble.jl#549. The band sweeps pick one colour or two from `prod(strides)`, a
    # runtime value; a tuple of one or two ranges made that slot a Union. `@inferred` and
    # JET pass either way (the return type is concrete and the split is not dispatch), so
    # only the unoptimised slot and SSA types show it. Both assemblies band through
    # `_sweep_coloured!`, one instance per point functor; a Union with a range or a tuple of
    # ranges among its arms is the pattern, whichever step or colour value carries it.
    @testset "Sweep band slots" begin
        ranged(S) = S <: AbstractRange || S <: Tuple{Vararg{AbstractRange}}
        seen(ct) = filter(T -> T isa Type, [ct.slottypes; ct.ssavaluetypes])
        bad(ct) = filter(T -> T isa Union && any(ranged, Base.uniontypes(T)), seen(ct))
        slots(f, types) = first(code_typed(f, types; optimize = false)).first
        a = form(Wₕ2, Wₕ2, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
        l = form(Wₕ2, v -> innerₕ(x -> 1.0, D₋ₓ(v)))
        grid = Bramble.indices(Ωₕ2)
        lin = LinearIndices(grid)
        ab, am = Bramble._bind_walk(a.ast, Wₕ2)
        lb, lm = Bramble._bind_walk(l.ast, Wₕ2)
        pb = Bramble._BilinearPoint(assemble(a), Wₕ2, ab, lin, am, 0, 0, true)
        pl = Bramble._LinearPoint(zeros(ndofs(Wₕ2)), Wₕ2, lb, lin, lm, 0, true)
        sweep(p) = slots(Bramble._sweep_coloured!,
            (Bramble.CpuThreaded, typeof(p), typeof(grid), NTuple{2, Int}))
        cb, cl = sweep(pb), sweep(pl)
        # the slots are those of the banding body, not of a forwarder
        @test any(T -> T <: StepRange{Int, Int}, seen(cb))
        @test isempty(bad(cb))
        @test isempty(bad(cl))
        # Control: the pattern #549 removed is flagged by the same filter.
        old(s, n) = (b = prod(s) == 1 ? (1:1:n,) : (1:2:n, 2:2:n); sum(sum, b))
        @test !isempty(bad(slots(old, (NTuple{2, Int}, Int))))
    end
end

end # module QualityTypeStabilityTests
