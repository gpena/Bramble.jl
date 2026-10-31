module FormSemidiscreteTests

using Test
using Bramble
using Bramble: matrix_type
using SparseArrays
using SciMLBase: SciMLBase # loads the `ode_problem` extension
using ForwardDiff: ForwardDiff, Dual, value
using FiniteDiff: FiniteDiff
using Bramble:
               D₋ₓ,
               Mₓ,
               allocate_system_matrix,
               jacobian_pattern,
               jacobian_prototype,
               mass_matrix,
               operator_matrix,
               semidiscretize_rhs,
               set_points!,
               type_cached_assemble!
using ..TestUtils: WITH_AD_TESTS, _sd_problem, _Const, _x1, _graded_mesh

# `semidiscretize` and the residual it returns (src/problems/semidiscrete.jl) need no SciMLBase:
# a `Semidiscretization` is a callable with the `(du, u, p, t)` signature plus two matrices.
# Everything here therefore belongs in the always-run unit group, and only the
# `ODEFunction`/`ODEProblem`/`LinearProblem` wrapping is left to test/ext/sciml_ext.jl.
#
# Its second order in space, stepped with backward Euler written out by hand, is in
# test/convergence/heat_1d.jl.

# Allocation assertions go behind a function barrier: measured at `@testset` scope they
# would count the closure the testset body becomes (bramble-verification §1).
_sd_residual_allocs(sd, du, u, t) = @allocated sd(du, u, nothing, t)
_sd_jacobian_allocs(J, sd, u, t) = @allocated Bramble.jacobian!(J, sd, u, nothing, t)
_sd_rhs_allocs(rhs, du, u, t) = @allocated rhs(du, u, nothing, t)
_sd_negate_allocs(J, A) = @allocated Bramble._negate_into!(J, A)

@testset "semidiscretize" begin
    Ωₕ, Wₕ, fₕ, a, l = _sd_problem(21)
    n = ndofs(Wₕ)
    I = Bramble.interval(0.0, 1.0)
    Rₕ!(fₕ, _Const(1.0))

    # The constraint carrier is chosen once, by shape and arity.
    @testset "constraint carrier: shape and arity" begin
        @test semidiscretize(a, l).constraints isa Bramble.NoConstraints
        @test semidiscretize(a, l; dirichlet = :boundary).constraints isa Bramble.LabelsOnly
        @test semidiscretize(a, l; dirichlet = :boundary => _Const(0.0)).constraints isa
              Bramble.StaticConstraints

        # A time domain is not what marks constraints as time dependent -- the conditions'
        # arity is, the same test `dirichlet_constraints` validates with.
        bcs_t = dirichlet_constraints(Ωₕ, I, :boundary => (x, t) -> 2t)
        @test semidiscretize(a, l; dirichlet = bcs_t).constraints isa
              Bramble.TimeDependentConstraints

        # A three-argument (x, t, p) condition is its own carrier, not folded into the
        # (x, t) one: `_dirichlet_is_time_param_dependent` checks a stricter arity.
        bcs_tp = dirichlet_constraints(Ωₕ, I, :boundary => (x, t, p) -> p[1] * t)
        @test semidiscretize(a, l; dirichlet = bcs_tp).constraints isa
              Bramble.TimeParamDependentConstraints

        bcs_x = dirichlet_constraints(Ωₕ, :boundary => _Const(1.0))
        @test semidiscretize(a, l; dirichlet = bcs_x).constraints isa
              Bramble.StaticConstraints

        # Evaluated markers hold `Fix2(g, t)`, which takes one argument: static, not timed.
        sd_eval = semidiscretize(a, l; dirichlet = bcs_t(0.5))
        @test sd_eval.constraints isa Bramble.StaticConstraints
        @test sd_eval(zeros(n), zeros(n), nothing, 0.3)[1] == 1.0

        # A `Fix` is classified by the function it wraps: `Fix2(g, c)` with `g(x, c, t)`
        # and `Fix{3}(h3, c)` with `h3(x, t, c)` are both (x, t) conditions.
        g = (x, c, t) -> c * t
        sd_fix2 = semidiscretize(
            a, l; dirichlet = dirichlet_constraints(Ωₕ, I, :boundary => Base.Fix2(g, 3.0))
        )
        @test sd_fix2.constraints isa Bramble.TimeDependentConstraints
        @test sd_fix2(zeros(n), zeros(n), nothing, 0.5)[1] == 1.5
        h3 = (x, t, c) -> c * t
        sd_fix3 = semidiscretize(
            a, l; dirichlet = dirichlet_constraints(Ωₕ, I, :boundary => Base.Fix{3}(h3, 3.0))
        )
        @test sd_fix3.constraints isa Bramble.TimeDependentConstraints
        @test sd_fix3(zeros(n), zeros(n), nothing, 0.5)[1] == 1.5

        # A defaulted time argument stays time dependent: the lowest-arity rule would
        # freeze it at its default.
        bcs_d = dirichlet_constraints(Ωₕ, I, :boundary => (x, t = 0.0) -> 1 + t)
        sd_d = semidiscretize(a, l; dirichlet = bcs_d)
        @test sd_d.constraints isa Bramble.TimeDependentConstraints
        @test sd_d(zeros(n), zeros(n), nothing, 0.5)[1] == 1.5
    end

    @testset "mass matrix carries the algebraic rows" begin
        M₀ = assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v)))
        sd = semidiscretize(a, l; dirichlet = :boundary)
        M = mass_matrix(sd)

        @test iszero(M[1, 1])
        @test iszero(M[n, n])
        @test all(M[i, i] == M₀[i, i] for i in 2:(n - 1))
        # The cleared diagonal stays stored: the mass matrix is handed over once, as a
        # constant, and a solver factorising `M/γ - J` reads its pattern.
        @test nnz(M) == nnz(M₀)
        @test M[1, 2] == 0

        # Unconstrained problems keep the mass matrix whole.
        @test mass_matrix(semidiscretize(a, l)) == M₀
    end

    @testset "operator matrix is the assembled form" begin
        sd = semidiscretize(a, l; dirichlet = :boundary)
        @test operator_matrix(sd) == assemble(a; dirichlet = :boundary)
        @test Bramble.space(sd) === Wₕ
    end

    @testset "residual, Jacobian, and allocations" begin
        bcs = dirichlet_constraints(Ωₕ, I, :boundary => (x, t) -> 2t)
        for dirichlet in (nothing, :boundary, :boundary => _Const(0.5), bcs)
            sd = semidiscretize(a, l; dirichlet = dirichlet)
            A = operator_matrix(sd)
            u = collect(range(0.25, 1.75; length = n))
            du = zeros(n)
            t = 0.3

            # F(t) read back through the residual itself, at u = 0.
            F = zeros(n)
            sd(F, zeros(n), nothing, t)
            sd(du, u, nothing, t)
            @test du ≈ F - A * u

            J = jacobian_prototype(sd)
            @test J == A
            Bramble.jacobian!(J, sd, u, nothing, t)
            @test J == -A

            @test _sd_residual_allocs(sd, du, u, t) == 0
            @test _sd_jacobian_allocs(J, sd, u, t) == 0
        end
    end

    # The fast path writes `-nonzeros(A)` straight into `nonzeros(J)`, so it must see the
    # whole pattern: here the row sequences match but the column breaks do not (#703).
    @testset "_negate_into!: CSC colptr (#703)" begin
        A = sparse([1, 2, 3], [1, 2, 3], [1.0, 2.0, 3.0], 3, 3)
        J = SparseMatrixCSC(3, 3, [1, 4, 4, 4], [1, 2, 3], zeros(3))
        @test rowvals(J) == rowvals(A)
        @test Bramble._negate_into!(J, A) === J
        @test Matrix(J) == -Matrix(A)

        S = copy(A)
        @test Bramble._negate_into!(S, A) === S
        @test S == -A
        @test _sd_negate_allocs(S, A) == 0
    end

    @testset "boundary values reach the residual" begin
        # Labels without values constrain to zero; `assemble!` cannot express that, so the
        # rows are cleared instead.
        sd_zero = semidiscretize(a, l; dirichlet = :boundary)
        F = zeros(n)
        sd_zero(F, zeros(n), nothing, 0.0)
        @test iszero(F[1]) && iszero(F[n])

        sd_static = semidiscretize(a, l; dirichlet = :boundary => _Const(7.0))
        sd_static(F, zeros(n), nothing, 0.0)
        @test F[1] ≈ 7.0 && F[n] ≈ 7.0

        bcs = dirichlet_constraints(Ωₕ, I, :boundary => (x, t) -> 1 + 3t)
        sd_time = semidiscretize(a, l; dirichlet = bcs)
        sd_time(F, zeros(n), nothing, 0.0)
        @test F[1] ≈ 1.0
        sd_time(F, zeros(n), nothing, 2.0)
        @test F[1] ≈ 7.0
        @test F[n] ≈ 7.0
    end

    # Boundary values given as (x, t, p) receive the parameter p.
    @testset "boundary values: (x, t, p) gets p" begin
        bcs = dirichlet_constraints(Ωₕ, I, :boundary => (x, t, p) -> p[1] + p[2] * t)
        sd = semidiscretize(a, l; dirichlet = bcs)
        F = zeros(n)

        sd(F, zeros(n), (1.0, 3.0), 0.0)
        @test F[1] ≈ 1.0 && F[n] ≈ 1.0
        sd(F, zeros(n), (1.0, 3.0), 2.0)
        @test F[1] ≈ 7.0 && F[n] ≈ 7.0

        # A different `p` at the same `t` -- confirms `p` is read fresh every call, not
        # captured once at construction.
        sd(F, zeros(n), (2.0, 1.0), 2.0)
        @test F[1] ≈ 4.0 && F[n] ≈ 4.0

        # `dirichlet_bc!`'s own initial-condition consistency step takes the same `p`.
        u0 = zeros(n)
        Bramble.dirichlet_bc!(u0, sd, 2.0, (1.0, 3.0))
        @test u0[1] ≈ 7.0 && u0[n] ≈ 7.0
    end

    # Mixing (x, t) and (x, t, p) arity across labels is rejected.
    @testset "boundary arity: mixed is rejected" begin
        Ωₕ2 = Bramble.mesh(
            Bramble.domain(Bramble.interval(0.0, 1.0), :left => :left, :right => :right), 11
        )
        @test_throws ArgumentError dirichlet_constraints(
            Ωₕ2, I, :left => (x, t) -> t, :right => (x, t, p) -> p * t
        )
    end

    # A time-dependent condition may sit beside a spatial one, and the keyword, `I`-less
    # constraint routes and `ode_problem` agree. The (x, t) and (x, t, p) arities still do
    # not mix: `semidiscretize` refuses that itself.
    @testset "boundary arity: spatial beside timed" begin
        Ωₕ2 = Bramble.mesh(
            Bramble.domain(Bramble.interval(0.0, 1.0), :left => :left, :right => :right), 11
        )
        W2 = gridspace(Ωₕ2)
        a2 = Bramble.form(
            W2, W2, (u, v) -> Bramble.inner₊(Bramble.D₋ₓ(u), Bramble.D₋ₓ(v))
        )
        l2 = Bramble.form(W2, v -> Bramble.innerₕ(1.0, v))
        n2 = ndofs(W2)
        mixed = (:left => (x, t) -> t, :right => x -> 3.0)
        for dirichlet in (mixed, dirichlet_constraints(Ωₕ2, mixed...))
            sd = semidiscretize(a2, l2; dirichlet = dirichlet)
            @test sd.constraints isa Bramble.TimeDependentConstraints
            du = zeros(n2)
            sd(du, zeros(n2), nothing, 2.0)
            @test du[1] ≈ 2.0
            @test du[n2] ≈ 3.0
            sd(du, zeros(n2), nothing, 5.0)
            @test du[1] ≈ 5.0
            @test du[n2] ≈ 3.0
            u = zeros(n2)
            @test dirichlet_bc!(u, sd, 2.0) === u
            @test u[1] ≈ 2.0
            @test u[n2] ≈ 3.0
        end

        # With no spatial condition the `I` route builds the same set, and agrees too.
        timed = (:left => (x, t) -> t, :right => (x, t) -> 3.0)
        for dirichlet in (timed, dirichlet_constraints(Ωₕ2, timed...), dirichlet_constraints(Ωₕ2, I, timed...))
            sd = semidiscretize(a2, l2; dirichlet = dirichlet)
            @test sd.constraints isa Bramble.TimeDependentConstraints
            du = zeros(n2)
            sd(du, zeros(n2), nothing, 2.0)
            @test du[1] ≈ 2.0
            @test du[n2] ≈ 3.0
        end

        # `ode_problem` forwards the keyword to `semidiscretize`.
        prob = Bramble.ode_problem(a2, l2, zeros(n2), (0.0, 1.0); dirichlet = mixed)
        du = zeros(n2)
        prob.f(du, zeros(n2), prob.p, 2.0)
        @test du[1] ≈ 2.0
        @test du[n2] ≈ 3.0

        # A spatial condition beside an (x, t, p) one reaches the parametric carrier.
        sd_p = semidiscretize(
            a2, l2; dirichlet = (:left => (x, t, p) -> p * t, :right => x -> 3.0)
        )
        @test sd_p.constraints isa Bramble.TimeParamDependentConstraints
        du = zeros(n2)
        sd_p(du, zeros(n2), 2.0, 4.0)
        @test du[1] ≈ 8.0
        @test du[n2] ≈ 3.0

        # An (x, t) condition beside an (x, t, p) one is refused at construction.
        @test_throws ArgumentError semidiscretize(
            a2, l2; dirichlet = (:left => (x, t) -> t, :right => (x, t, p) -> p)
        )
        @test_throws ArgumentError Bramble.ode_problem(
            a2, l2, zeros(n2), (0.0, 1.0);
            dirichlet = (:left => (x, t) -> t, :right => (x, t, p) -> p)
        )
    end

    # update_coefficients! runs before each assembly.
    @testset "update_coefficients!: before assembly" begin
        seen = Float64[]
        sd = semidiscretize(
            a, l; (update_coefficients!) = t -> (push!(seen, t); Rₕ!(fₕ, x -> t))
        )
        F = zeros(n)
        sd(F, zeros(n), nothing, 2.5)
        @test seen == [2.5]
        # `l` is `innerₕ(fₕ, v)`, so an fₕ set to t scales the assembled source by t.
        G = zeros(n)
        Rₕ!(fₕ, _Const(1.0))
        assemble!(G, l)
        @test F ≈ 2.5 .* G
    end

    # update_coefficients! given as (t, p) receives the parameter p.
    @testset "update_coefficients!: (t, p) gets p" begin
        seen = Tuple{Float64, Float64}[]
        sd = semidiscretize(
            a, l; (update_coefficients!) = (t, p) -> (push!(seen, (t, p)); Rₕ!(fₕ, x -> t * p))
        )
        F = zeros(n)
        sd(F, zeros(n), 3.0, 2.5)
        @test seen == [(2.5, 3.0)]
        G = zeros(n)
        Rₕ!(fₕ, _Const(1.0))
        assemble!(G, l)
        @test F ≈ (2.5 * 3.0) .* G

        # The existing one-argument form still dispatches exactly as before: a raw closure
        # is stored (never wrapped in `ParametricUpdate`), so this is not merely "the p case
        # still works with p = nothing" but the untouched original code path.
        seen1 = Float64[]
        sd1 = semidiscretize(a, l; (update_coefficients!) = t -> push!(seen1, t))
        sd1(zeros(n), zeros(n), 99.0, 1.5)
        @test seen1 == [1.5]
        @test !(sd1.update_coefficients isa Bramble.ParametricUpdate)

        # A `Fix1` hook is classified by the function it wraps: `setindex!(α, t)` takes
        # one argument more than the hook, so the hook is one-argument and stays raw.
        α = Ref(0.0)
        sdα = semidiscretize(a, l; (update_coefficients!) = Base.Fix1(setindex!, α))
        @test !(sdα.update_coefficients isa Bramble.ParametricUpdate)
        sdα(zeros(n), zeros(n), nothing, 0.5)
        @test α[] == 0.5

        # The hook prefers its one-argument form: `(t, p = 1.0) -> ...` is called as
        # `f(t)` and ignores `p`.
        β = Ref(0.0)
        sdq = semidiscretize(a, l; (update_coefficients!) = (t, p = 1.0) -> (β[] = p))
        @test !(sdq.update_coefficients isa Bramble.ParametricUpdate)
        sdq(zeros(n), zeros(n), 0.25, 0.5)
        @test β[] == 1.0
    end

    @testset "state exposes u to the forms" begin
        uₕ = Bramble.element(Wₕ, 0.0)
        sd = semidiscretize(a, l; state = uₕ)
        u = collect(range(1.0, 2.0; length = n))
        sd(zeros(n), u, nothing, 0.0)
        @test parent(uₕ) == u
    end

    @testset "reassemble refills the operator" begin
        α = Ref(1.0)
        # The `Ref` itself goes in the expression, not `α[]`: dereferencing evaluates
        # once, when the form is built, and the coefficient stops being live.
        aα = form(Wₕ, Wₕ, (u, v) -> α * innerₕ(u, v))
        sd = semidiscretize(aα, l; reassemble = true, (update_coefficients!) = t -> (α[] = t))
        A₁ = copy(operator_matrix(sd))
        sd(zeros(n), zeros(n), nothing, 3.0)
        @test operator_matrix(sd) ≈ 3 .* A₁

        # Without `reassemble` the matrix assembled at construction is reused as is.
        α[] = 1.0
        sd_fixed = semidiscretize(
            aα, l; reassemble = false, (update_coefficients!) = t -> (α[] = t)
        )
        A₂ = copy(operator_matrix(sd_fixed))
        sd_fixed(zeros(n), zeros(n), nothing, 3.0)
        @test operator_matrix(sd_fixed) == A₂
    end

    @testset "type-cached (build-based) operator" begin
        # A genuinely time-dependent coefficient α(t) = 1 + t, refilled into a live buffer --
        # not baked into the closure -- the way the docstring's own example does it.
        build_calls = Ref(0)
        function build_scaled_mass(t)
            build_calls[] += 1
            αₕ = Bramble.element(Wₕ, typeof(t))
            aα = form(Wₕ, Wₕ, (u, v) -> αₕ * innerₕ(u, v))
            refill!(t) = (fill!(parent(αₕ), one(typeof(t)) + t); nothing)
            return aα, refill!
        end

        sd = semidiscretize(build_scaled_mass, l)
        # `build` runs once at construction (probing at the space's element type), and
        # defaults to `reassemble = true` -- a `build`-based operator exists specifically to
        # be rebuilt.
        @test build_calls[] == 1
        @test occursin("every step", sprint(show, MIME"text/plain"(), sd))

        A₀ = assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v)))
        # At u = 0 the residual is the source alone, whatever α(t): take a nonzero state.
        u0 = collect(range(0.2, 1.7; length = n))
        du64 = zeros(n)
        sd(du64, u0, nothing, 3.0)
        @test operator_matrix(sd) ≈ 4 .* A₀      # α(3.0) = 1 + 3 = 4
        @test build_calls[] == 1                 # same element type: no rebuild, only refill

        # The probe's entry is stored concretely, and its matrix is `operator_matrix(sd)`
        # itself, so a step at the space's element type infers and allocates nothing (#554).
        # `dirichlet = :boundary` makes the keyword call carry data, so a dynamically
        # dispatched `assemble!` could not pass as 0 B by accident.
        sd_bc = semidiscretize(build_scaled_mass, l; dirichlet = :boundary)
        @test sd_bc.operator.primary[3] === operator_matrix(sd_bc)
        @test isempty(sd_bc.operator.cache)
        R = only(
            Base.return_types(
            Bramble._reassemble_operator!,
            (typeof(sd_bc.operator), typeof(sd_bc), Vector{Float64}, Float64)
        ),
        )
        @test isconcretetype(R)
        du_bc = zeros(n)
        J_bc = jacobian_prototype(sd_bc)
        _sd_residual_allocs(sd_bc, du_bc, u0, 3.0)
        @test _sd_residual_allocs(sd_bc, du_bc, u0, 3.0) == 0
        _sd_jacobian_allocs(J_bc, sd_bc, u0, 3.0)
        @test _sd_jacobian_allocs(J_bc, sd_bc, u0, 3.0) == 0
        @test J_bc ≈ -operator_matrix(sd_bc)
        @test operator_matrix(sd_bc)[2, 2] ≈ 4 * A₀[2, 2]   # an interior row, refilled
        # `semidiscretize_rhs` takes no Dirichlet labels, so the RHS runs on `sd` above.
        rhs = semidiscretize_rhs(sd)
        _sd_rhs_allocs(rhs, du_bc, u0, 3.0)
        @test _sd_rhs_allocs(rhs, du_bc, u0, 3.0) == 0
        @test isempty(sd.operator.cache)

        if WITH_AD_TESTS
            # A `ForwardDiff.Dual` `t` -- the same thing a Rosenbrock stepper's `tgrad` reaches
            # for -- rebuilds once at that new element type, and never throws `InexactError`
            # trying to write a `Dual` into the `Float64` matrix the classic `BilinearForm` path
            # would have kept.
            t_dual = Dual(3.0, 1.0)
            u_dual = Dual.(u0, 0.0)
            du_dual = similar(u_dual)
            sd(du_dual, u_dual, nothing, t_dual)
            @test build_calls[] == 2
            @test value.(du_dual) ≈ du64

            # Calling again at the already-seen `Dual` type refills without rebuilding.
            sd(du_dual, u_dual, nothing, t_dual)
            @test build_calls[] == 2

            # The `Float64` path stays exactly as before, unaffected by the Dual excursion in
            # between: same cached entry, refilled at its own `t` rather than rebuilt.
            sd(zeros(n), zeros(n), nothing, 1.0)
            @test operator_matrix(sd) ≈ 2 .* A₀      # α(1.0) = 1 + 1 = 2
            @test build_calls[] == 2
        end
    end

    @testset "type-cached: Float32 space (#555)" begin
        Ω32 = Bramble.mesh(
            Bramble.domain(Bramble.interval(0.0f0, 1.0f0)),
            9,
            true;
            backend = backend(Float32)
        )
        W32 = gridspace(Ω32)
        n32 = ndofs(W32)
        l32 = form(W32, v -> innerₕ(_Const(1.0f0), v))
        A₀32 = assemble(form(W32, W32, (u, v) -> innerₕ(u, v)))
        # Nonzero, so `du = F - A u` sees `A`, not the source alone.
        u32 = collect(range(0.2f0, 1.7f0; length = n32))

        # A build whose coefficient takes the element type of `t`, and one that ignores it:
        # both are probed once, at `zero(Float32)`, and a Float32 step hits that entry.
        typed_calls = Ref(0)
        function build_typed(t)
            typed_calls[] += 1
            αₕ = Bramble.element(W32, typeof(t))
            aα = form(W32, W32, (u, v) -> αₕ * innerₕ(u, v))
            return aα, s -> (fill!(parent(αₕ), 1 + s); nothing)
        end
        fixed_calls = Ref(0)
        function build_fixed(t)
            fixed_calls[] += 1
            αₕ = Bramble.element(W32)
            aα = form(W32, W32, (u, v) -> αₕ * innerₕ(u, v))
            return aα, s -> (fill!(parent(αₕ), 1 + s); nothing)
        end

        for (build_fn, calls) in ((build_typed, typed_calls), (build_fixed, fixed_calls))
            sd = semidiscretize(build_fn, l32)
            @test calls[] == 1
            @test eltype(operator_matrix(sd)) == Float32
            @test eltype(jacobian_prototype(sd)) == Float32
            du = zeros(Float32, n32)
            sd(du, u32, nothing, 3.0f0)
            @test calls[] == 1                      # same element type: refill, no rebuild
            @test operator_matrix(sd) ≈ 4 .* A₀32   # α(3) = 1 + 3, not the probe's α(0)
            @test du ≈ sd.source_vector .- (4 .* A₀32) * u32
        end

        # The documented mixed case: a Float64 time over a Float32 space reaches a second
        # element type, so it builds once more (the time span should be Float32).
        sd = semidiscretize(build_typed, l32)
        calls_before = typed_calls[]
        du32 = zeros(Float32, n32)
        sd(du32, u32, nothing, 3.0f0)
        du_mixed = zeros(Float32, n32)
        sd(du_mixed, u32, nothing, 3.0)
        @test typed_calls[] == calls_before + 1
        @test du_mixed ≈ du32
    end

    @testset "consistent initial conditions" begin
        bcs = dirichlet_constraints(Ωₕ, I, :boundary => (x, t) -> 5 + t)
        sd = semidiscretize(a, l; dirichlet = bcs)
        u = fill(-1.0, n)
        @test dirichlet_bc!(u, sd, 2.0) === u
        @test u[1] ≈ 7.0 && u[n] ≈ 7.0
        @test u[5] == -1.0

        sd_zero = semidiscretize(a, l; dirichlet = :boundary)
        v = fill(-1.0, n)
        dirichlet_bc!(v, sd_zero, 0.0)
        @test iszero(v[1]) && iszero(v[n]) && v[5] == -1.0

        # Nothing to impose leaves the vector alone.
        w = fill(-1.0, n)
        @test dirichlet_bc!(w, semidiscretize(a, l), 0.0) == fill(-1.0, n)
    end

    @testset "display" begin
        bcs = dirichlet_constraints(Ωₕ, I, :boundary => (x, t) -> 0.0)
        sd = semidiscretize(a, l; dirichlet = bcs)

        compact = sprint(show, sd)
        @test compact == "Semidiscretization{$n dofs, 1 constrained label}"
        @test summary(sd) == compact
        # The default would spell out every nested type parameter.
        @test length(compact) < 80

        block = sprint(show, MIME"text/plain"(), sd)
        @test occursin("Semidiscretization", block)
        @test occursin("g(x, t) on :boundary", block)
        @test occursin("Reassembled: once", block)
        # `display` adds the newline; the body must not.
        @test !endswith(block, "\n")

        @test occursin(
            "zero on :boundary",
            sprint(show, MIME"text/plain"(), semidiscretize(a, l; dirichlet = :boundary))
        )
        @test occursin(
            "g(x) on :boundary",
            sprint(
                show,
                MIME"text/plain"(),
                semidiscretize(a, l; dirichlet = :boundary => _Const(0.0))
            )
        )
        @test occursin(
            "Constraints: none", sprint(show, MIME"text/plain"(), semidiscretize(a, l))
        )
        @test occursin(
            "every step",
            sprint(show, MIME"text/plain"(), semidiscretize(a, l; reassemble = true))
        )
        @test occursin("constrained labels", sprint(show, semidiscretize(a, l)))
    end

    @testset "mismatched spaces are rejected" begin
        _, Wc, _, ac, _ = _sd_problem(9)
        @test_throws ArgumentError semidiscretize(ac, l)
    end
end

# The matrix-type seam extended to the consumers of assembly ):
# `semidiscretize`, `mass_matrix` and `operator_matrix` read whatever matrix type the form's
# own backend produces -- the `Semidiscretization{...,MT,MM,...}` parameters carried for
# the operator and mass matrices -- rather than assuming `SparseMatrixCSC`. A dense
# `Matrix{Float64}` backend must semidiscretise to the exact same numbers as the default CSC
# backend, and the residual built from it must agree too.
@testset "Dense backend" begin
    I = Bramble.interval(0.0, 1.0)
    Ωc = Bramble.mesh(Bramble.domain(I), 21, false)
    Ωd = Bramble.mesh(Bramble.domain(I), 21, false; backend = backend(matrix_type = Matrix{Float64}))
    set_points!(Ωd, points(Ωc))
    Wc, Wd = gridspace(Ωc), gridspace(Ωd)

    function _sd_forms(W)
        fₕ = Rₕ(W, _Const(1.0))
        a = form(W, W, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
        l = form(W, v -> innerₕ(fₕ, v))
        return a, l
    end
    ac, lc = _sd_forms(Wc)
    ad, ld = _sd_forms(Wd)

    bcs_c = dirichlet_constraints(Ωc, I, :boundary => (x, t) -> 0.0)
    bcs_d = dirichlet_constraints(Ωd, I, :boundary => (x, t) -> 0.0)

    sdc = semidiscretize(ac, lc; dirichlet = bcs_c)
    sdd = semidiscretize(ad, ld; dirichlet = bcs_d)

    # operator_matrix and mass_matrix agree with the CSC backend.
    @testset "operator/mass matrix: match CSC" begin
        @test operator_matrix(sdc) isa SparseMatrixCSC
        @test operator_matrix(sdd) isa Matrix{Float64}
        @test isapprox(Matrix(operator_matrix(sdc)), operator_matrix(sdd); atol = 1e-13)
        @test mass_matrix(sdc) isa SparseMatrixCSC
        @test mass_matrix(sdd) isa Matrix{Float64}
        @test isapprox(Matrix(mass_matrix(sdc)), mass_matrix(sdd); atol = 1e-13)
    end

    @testset "the residual agrees with CSC" begin
        n = ndofs(Wc)
        u = collect(range(0.25, 1.75; length = n))
        duc, dud = zeros(n), zeros(n)
        t = 0.3
        sdc(duc, u, nothing, t)
        sdd(dud, u, nothing, t)
        @test duc ≈ dud
    end

    @testset "assemble_add! agrees with CSC" begin
        mc = form(Wc, Wc, (u, v) -> innerₕ(u, v))
        md = form(Wd, Wd, (u, v) -> innerₕ(u, v))

        Ac = allocate_system_matrix(ac)
        Ad = allocate_system_matrix(ad)
        assemble_add!(Ac, mc, 2.0)
        assemble_add!(Ac, ac, 0.5)
        assemble_add!(Ad, md, 2.0)
        assemble_add!(Ad, ad, 0.5)
        @test isapprox(Matrix(Ac), Ad; atol = 1e-13)
    end

    @testset "type_cached_assemble! agrees with CSC" begin
        cache_c, cache_d = Dict{DataType, Any}(), Dict{DataType, Any}()
        build(a) = uₕ -> (a, _ -> nothing)
        Bc = type_cached_assemble!(build(ac), cache_c, Bramble.element(Wc, 0.0))
        Bd = type_cached_assemble!(build(ad), cache_d, Bramble.element(Wd, 0.0))
        @test isapprox(Matrix(Bc), Bd; atol = 1e-13)
    end

    # jacobian_pattern keeps the same nonzero count.
    @testset "jacobian_pattern: same nnz" begin
        Pc = jacobian_pattern(ac)
        Pd = jacobian_pattern(ad)
        @test Pc isa SparseMatrixCSC
        @test count(!iszero, Matrix(Pc)) == count(!iszero, Matrix(Pd))
    end
end

# Pin `rhs = semidiscretize_rhs(...)` and its residual call as inferred and
# allocation-free -- the call a time-stepping loop makes every step. Allocation checks go behind a function
# barrier (bramble-verification §1); `@inferred` reads no global binding here, so it is
# left in the barrier alongside the warm-up rather than pulled out to top level.
@testset "rhs: inferred, no allocs (#283)" begin
    function _rhs_alloc(rhs, du, u, t)
        rhs(du, u, nothing, t)                # cold: records
        @inferred rhs(du, u, nothing, t)
        return @allocated rhs(du, u, nothing, t)
    end

    function _rhs_problem(dims)
        Ωₕ = dims == 1 ?
             Bramble.mesh(Bramble.domain(Bramble.interval(0.0, 1.0)), 21) :
             Bramble.mesh(
            Bramble.domain(Bramble.interval(0.0, 1.0) × Bramble.interval(0.0, 1.0)),
            (9, 11),
            (true, true)
        )
        Wₕ = gridspace(Ωₕ)
        fₕ = Bramble.element(Wₕ, 0.0)
        Rₕ!(fₕ, _Const(1.0))
        K = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
        l = form(Wₕ, v -> innerₕ(fₕ, v))
        return Wₕ, K, l
    end

    for (lbl, dims) in (("1D", 1), ("2D", 2))
        @testset "$lbl" begin
            Wₕ, K, l = _rhs_problem(dims)
            n = ndofs(Wₕ)

            @testset "semidiscretize_rhs" begin
                rhs = semidiscretize_rhs(semidiscretize(K, l))
                du = zeros(n)
                u = collect(range(0.2, 1.7; length = n))
                @test _rhs_alloc(rhs, du, u, 0.0) == 0
            end
        end
    end
end

@testset "semidiscretize on a composite space" begin
    Ωₕ = Bramble.mesh(Bramble.domain(Bramble.interval(0.0, 1.0)), 11)
    Wₕ = gridspace(Ωₕ)
    Vₕ = vector_gridspace(Ωₕ, Val(2))
    n = ndofs(Vₕ)
    nleaf = ndofs(Wₕ)

    cₕ = Rₕ(Wₕ, _Const(1.0))
    a = form(Vₕ, Vₕ, (u, v) -> inner₊(∇ₕ(u(1)), ∇ₕ(v(1))) + inner₊(∇ₕ(u(2)), ∇ₕ(v(2))))
    # Distinct values per component, so a residual that mixed the blocks could not pass.
    l = form(Vₕ, v -> innerₕ(cₕ, v(1)) + 3 * innerₕ(cₕ, v(2)))

    sd = semidiscretize(a, l; dirichlet = :boundary => _Const(0.0))
    u = collect(range(0.5, 2.5; length = n))
    du = zeros(n)
    F = zeros(n)
    sd(F, zeros(n), nothing, 0.0)
    sd(du, u, nothing, 0.0)

    @test du ≈ F - operator_matrix(sd) * u
    @test F[2:(nleaf - 1)] ≈ F[(nleaf + 2):(2nleaf - 1)] ./ 3
    # cₕ ≡ 1: each interior load entry is its quadrature weight, three times it on leaf 2
    w = collect(Bramble.weights(Wₕ, Bramble.Innerh()))
    @test F[2:(nleaf - 1)] ≈ w[2:(nleaf - 1)]
    @test F[(nleaf + 2):(2nleaf - 1)] ≈ 3 .* w[2:(nleaf - 1)]
    @test _sd_residual_allocs(sd, du, u, 0.0) == 0

    M = mass_matrix(sd)
    @test iszero(M[1, 1]) && iszero(M[nleaf, nleaf])
    @test iszero(M[nleaf + 1, nleaf + 1]) && iszero(M[n, n])

    # dirichlet_components binds the labels to one leaf.
    @testset "dirichlet_components: one leaf" begin
        sd₁ = semidiscretize(a, l; dirichlet = :boundary => _Const(0.0), dirichlet_components = 1)
        M₁ = mass_matrix(sd₁)
        @test iszero(M₁[1, 1]) && iszero(M₁[nleaf, nleaf])
        @test !iszero(M₁[nleaf + 1, nleaf + 1]) && !iszero(M₁[n, n])
    end
end

# The `Semidiscretization` type is concrete at the call site for each constraint carrier,
# and the leaf-by-leaf mesh check infers `Bool` across leaves of unlike types (#981).
@testset "semidiscretize: concrete type (#981)" begin
    Ωₕ, Wₕ, fₕ, a, l = _sd_problem(9)
    I = Bramble.interval(0.0, 1.0)
    sd_closure(a, l, b) = semidiscretize(a, l; dirichlet = b)
    for b in (
        dirichlet_constraints(Ωₕ, :boundary => x -> 0.0),
        dirichlet_constraints(Ωₕ, I, :boundary => (x, t) -> t),
        dirichlet_constraints(Ωₕ, I, :boundary => (x, t, p) -> p[1] * t)
    )
        R = only(Base.return_types(sd_closure, (typeof(a), typeof(l), typeof(b))))
        @test isconcretetype(R)
        @test R === typeof(sd_closure(a, l, b))
    end

    D = Bramble.domain(Bramble.interval(0.0, 1.0))
    Wp = gridspace(Bramble.mesh(D, 7, false; backend = Bramble.backend(policy = Parallel())))
    @test typeof(Bramble.mesh(Wp)) != typeof(Bramble.mesh(Wₕ))
    V = Wₕ × Wp
    @test @inferred(Bramble._same_leaf_meshes(V, V)) === true
    @test @inferred(Bramble._same_leaf_meshes(V, Wp × Wₕ)) === false
end

# Every form handed to `semidiscretize` is posed on `l`'s space: `mass` of another
# size (#705), and `a` or `mass` on another mesh with as many points (#706), are
# refused at construction, while two `gridspace` calls over one mesh still pass.
@testset "forms on other spaces (#705, #706)" begin
    D = Bramble.domain(Bramble.interval(0.0, 1.0))
    Ωᵤ = Bramble.mesh(D, 11, true)
    Ωₙ = Bramble.mesh(D, 11, false)
    @test Bramble.points(Ωᵤ) != Bramble.points(Ωₙ)
    W = gridspace(Ωₙ)
    W₇ = gridspace(Bramble.mesh(D, 7, false))
    Wᵤ = gridspace(Ωᵤ)
    stiff(X) = form(X, X, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
    mass(X) = form(X, X, (u, v) -> innerₕ(u, v))
    a = stiff(W)
    l = form(W, v -> innerₕ(_Const(1.0), v))
    errmsg(f) =
        try
            f()
            ""
        catch err
            err isa ArgumentError ? sprint(showerror, err) : rethrow()
        end

    @testset "mass of another size" begin
        msg = errmsg(() -> semidiscretize(a, l; mass = mass(W₇)))
        @test occursin("`mass`", msg) && occursin("7×7", msg) && occursin("11", msg)
        build(t) = (a, _ -> nothing)
        msg = errmsg(() -> semidiscretize(build, l; mass = mass(W₇)))
        @test occursin("`mass`", msg)
    end

    @testset "same ndofs, other points" begin
        @test_throws ArgumentError semidiscretize(stiff(Wᵤ), l)
        @test occursin("mesh", errmsg(() -> semidiscretize(stiff(Wᵤ), l)))
        @test occursin("`mass`", errmsg(() -> semidiscretize(a, l; mass = mass(Wᵤ))))
        buildᵤ(t) = (stiff(Wᵤ), _ -> nothing)
        @test_throws ArgumentError semidiscretize(buildᵤ, l)
        # trial space alone on the other mesh
        aₘ = form(Wᵤ, W, (u, v) -> innerₕ(u, v))
        @test_throws ArgumentError semidiscretize(aₘ, l)
    end

    @testset "composite: checked leaf by leaf" begin
        V = W × W
        Vₘ = W × Wᵤ
        lᵥ = form(V, v -> innerₕ(_Const(1.0), v(1)) + innerₕ(_Const(1.0), v(2)))
        aᵥ(X) = form(X, X, (u, v) -> innerₕ(u(1), v(1)) + innerₕ(u(2), v(2)))
        @test semidiscretize(aᵥ(V), lᵥ) isa Bramble.Semidiscretization
        @test occursin("mesh", errmsg(() -> semidiscretize(aᵥ(Vₘ), lᵥ)))
        @test occursin("`mass`", errmsg(() -> semidiscretize(aᵥ(V), lᵥ; mass = aᵥ(Vₘ))))
        # one leaf of 22 points against two of 11: same ndofs, other layout
        W₂₂ = gridspace(Bramble.mesh(D, 22, true))
        @test_throws ArgumentError semidiscretize(stiff(W₂₂), lᵥ)
    end

    @testset "one mesh, two gridspaces, rebuilt mesh" begin
        W₁, W₂ = gridspace(Ωₙ), gridspace(Ωₙ)
        sd = semidiscretize(stiff(W₁), form(W₂, v -> innerₕ(_Const(1.0), v)); mass = mass(W₂))
        @test size(mass_matrix(sd)) == (11, 11)
        Wᵣ = gridspace(Bramble.mesh(D, 11, true))
        @test semidiscretize(stiff(Wᵣ), form(Wᵤ, v -> innerₕ(_Const(1.0), v))) isa
              Bramble.Semidiscretization
    end
end

# `semidiscretize_rhs`: `du = M⁻¹(F(t) - A u)` with `M`'s diagonal
# folded in once, instead of a solver factorising `M` at every step. Checked against
# `weights(Wₕ).innerh`, the space's own independently-computed `L²` weight vector -- not
# against `mass_matrix(sd)`'s diagonal, which would just check the implementation agrees
# with itself.
# semidiscretize_rhs builds a matrix-free explicit right-hand side.
@testset "semidiscretize_rhs: matrix-free RHS" begin
    Ωₕ, Wₕ, fₕ, a, l = _sd_problem(21)
    Rₕ!(fₕ, _Const(1.0))
    n = ndofs(Wₕ)
    h = Bramble.weights(Wₕ).innerh

    sd = semidiscretize(a, l)
    rhs = semidiscretize_rhs(sd)
    @test rhs isa Bramble.SemidiscretizeRHS

    u = collect(range(0.2, 1.7; length = n))
    du_rhs, du_sd = zeros(n), zeros(n)
    t = 0.37
    rhs(du_rhs, u, nothing, t)
    sd(du_sd, u, nothing, t)
    @test du_rhs≈du_sd ./ h atol=1e-12 rtol=1e-12

    # Any Dirichlet row makes M singular there, so NoConstraints is required.
    @testset "rhs: requires NoConstraints" begin
        sd_static = semidiscretize(a, l; dirichlet = :boundary => _Const(0.0))
        @test_throws ArgumentError semidiscretize_rhs(sd_static)

        sd_labels = semidiscretize(a, l; dirichlet = :boundary)
        @test_throws ArgumentError semidiscretize_rhs(sd_labels)

        I = Bramble.interval(0.0, 1.0)
        bcs_t = dirichlet_constraints(Ωₕ, I, :boundary => (x, t) -> 0.0)
        sd_time = semidiscretize(a, l; dirichlet = bcs_t)
        @test_throws ArgumentError semidiscretize_rhs(sd_time)
    end

    @testset "requires a diagonal mass matrix" begin
        # A genuine coupling term: `Mₓ(v)` reaches `v`'s neighbour, so column `i` of the
        # assembled mass form has an off-diagonal entry.
        mass_coupled = form(Wₕ, Wₕ, (u, v) -> innerₕ(D₋ₓ(u), Mₓ(v)))
        sd_coupled = semidiscretize(a, l; mass = mass_coupled)
        @test_throws ArgumentError semidiscretize_rhs(sd_coupled)
    end

    # The mass diagonal must be invertible (nonzero).
    @testset "rhs: requires nonzero diagonal" begin
        # Diagonal, but exactly zero at the left boundary point (x = 0) -- no coupling
        # term, so this does not hit the non-diagonal case above; it is its own check.
        cₕ = Rₕ(Wₕ, _x1)
        mass_zero = form(Wₕ, Wₕ, (u, v) -> innerₕ(cₕ * u, v))
        sd_zero = semidiscretize(a, l; mass = mass_zero)
        @test iszero(mass_matrix(sd_zero)[1, 1])
        @test_throws ArgumentError semidiscretize_rhs(sd_zero)
    end
end

# The dense-backend paths of `semidiscretize_rhs` (`_diagonal_or_throw` on a plain
# `Matrix`) and `(x, t, p)` initial constraints, on a smoothly graded non-uniform mesh. The
# oracle for the mass diagonal is the space's own `L²` weight vector, not the assembled
# matrix.
@testset "rhs on a dense backend" begin
    I = Bramble.interval(0.0, 1.0)
    Ωd = Bramble.mesh(
        Bramble.domain(I), 13, false; backend = backend(matrix_type = Matrix{Float64})
    )
    xs = [ξ + 0.1 * sinpi(2ξ) for ξ in range(0.0, 1.0; length = 13)]
    set_points!(Ωd, xs)
    Wd = gridspace(Ωd)
    n = ndofs(Wd)
    w = Bramble.weights(Wd).innerh
    @test !all(≈(w[2]), w[2:(n - 1)])   # the mesh really is non-uniform

    fₕ = Rₕ(Wd, _Const(1.0))
    K = form(Wd, Wd, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
    l = form(Wd, v -> innerₕ(fₕ, v))
    bcs = dirichlet_constraints(Ωd, I, :boundary => (x, t, p) -> p[1] * t + x[1])

    sd_bc = semidiscretize(K, l; dirichlet = bcs)
    @test sd_bc.constraints isa Bramble.TimeParamDependentConstraints

    @testset "dirichlet_bc!: (x, t, p) gets p" begin
        u = fill(-1.0, n)
        @test dirichlet_bc!(u, sd_bc, 0.5, [3.0]) === u
        @test u[1] ≈ 1.5 && u[n] ≈ 2.5
        @test all(==(-1.0), u[2:(n - 1)])
    end

    @testset "semidiscretize_rhs: dense mass" begin
        sd = semidiscretize(K, l)
        @test mass_matrix(sd) isa Matrix{Float64}
        rhs = semidiscretize_rhs(sd)
        u = collect(range(0.2, 1.7; length = n))
        du_rhs, du_sd = zeros(n), zeros(n)
        rhs(du_rhs, u, nothing, 0.3)
        sd(du_sd, u, nothing, 0.3)
        @test du_rhs≈du_sd ./ w atol=1e-12 rtol=1e-12

        mass_coupled = form(Wd, Wd, (u, v) -> innerₕ(D₋ₓ(u), Mₓ(v)))
        @test_throws ArgumentError semidiscretize_rhs(semidiscretize(K, l; mass = mass_coupled))
    end
end

# The SciMLBase/SciMLSensitivity entry points are fallbacks that only run while the
# extension is not loaded, which the test environment always loads. A child process on the
# root project (no extension) runs them; `Base.julia_cmd()` carries this process's coverage
# flag, so its hits count.
function _sd_child_messages(code)
    root = pkgdir(Bramble)
    cmd = `$(Base.julia_cmd()) --project=$root --startup-file=no --threads=1 -e $code`
    return split(readchomp(pipeline(cmd; stderr = devnull)), '\n')
end

# A `Dual` `u` the state cannot hold goes to a twin of the semidiscretization at its type: a
# state buffer of that type, the forms rebound to it and their own operator matrix, so the
# AD Jacobian carries `-(∂A/∂u) u` (#985). The oracle is a central difference of the Float64
# residual, and the Picard `-A` is checked to differ from it, so a twin reading a stale
# state could not pass. Graded 2D mesh, its boundary constrained to data.
_sd_jac(sd, u, t) = ForwardDiff.jacobian((du, v) -> sd(du, v, nothing, t), zero(u), u)
function _sd_jac_fd(sd, u, t)
    n = length(u)
    return FiniteDiff.finite_difference_jacobian(v -> sd(zeros(n), v, nothing, t), u,
        Val(:central))
end

@testset "Dual state: exact AD Jacobian (#985)" begin
    Ωₕ = _graded_mesh((6, 7))
    Wₕ = gridspace(Ωₕ)
    n = ndofs(Wₕ)
    u0 = collect(range(0.5, 2.0; length = n))
    bc = :boundary => x -> 1.0 + x[2]

    @testset "operator reads the state, reassembled" begin
        sₕ = Bramble.element(Wₕ, 3.0)
        a = form(Wₕ, Wₕ, (u, v) -> inner₊(sₕ * ∇ₕ(u), ∇ₕ(v)))
        l = form(Wₕ, v -> innerₕ(x -> 1.0 + x[1], v))
        sd = semidiscretize(a, l; state = sₕ, reassemble = true, dirichlet = bc)
        # one function object, so both Jacobian calls below share one Dual type
        f! = (du, v) -> sd(du, v, nothing, 0.0)
        J = ForwardDiff.jacobian(f!, zeros(n), u0)
        D = only(keys(sd.twins))
        # the user's state never receives a Dual
        @test eltype(parent(sₕ)) === Float64
        @test all(==(3.0), parent(sₕ))

        Jfd = _sd_jac_fd(sd, u0, 0.0)
        @test isapprox(J, Jfd; rtol = 1e-6)
        # the Picard `-A` at u0 is not the Jacobian
        sd(zeros(n), u0, nothing, 0.0)
        @test !isapprox(J, -Matrix(operator_matrix(sd)); rtol = 1e-3)

        # a Dual residual's values are the Float64 residual's
        ud = D.(u0)
        rd = sd(zeros(D, n), ud, nothing, 0.0)
        @test value.(rd) ≈ sd(zeros(n), u0, nothing, 0.0) rtol = 1e-14

        # built once per element type: a second Jacobian reuses the twin and its matrix
        twin = sd.twins[D]
        Atwin = operator_matrix(twin)
        @test length(sd.twins) == 1
        @test ForwardDiff.jacobian(f!, zeros(n), u0) == J
        @test length(sd.twins) == 1
        @test sd.twins[D] === twin
        @test operator_matrix(twin) === Atwin
        @test eltype(Atwin) === D

        # the Float64 residual still allocates nothing beside the twin
        du, u = zeros(n), copy(u0)
        sd(du, u, nothing, 0.0)
        @test _sd_residual_allocs(sd, du, u, 0.0) == 0
        @test parent(sₕ) == u
    end

    # Without `reassemble` the operator never followed the state: the twin keeps `A`, and
    # only the source, which reads the state here, is rebound.
    @testset "source reads the state, operator fixed" begin
        sₕ = Bramble.element(Wₕ, 2.0)
        a = form(Wₕ, Wₕ, (u, v) -> inner₊(sₕ * ∇ₕ(u), ∇ₕ(v)))
        l = form(Wₕ, v -> innerₕ(sₕ, v))
        sd = semidiscretize(a, l; state = sₕ)
        J = _sd_jac(sd, u0, 0.0)
        @test isapprox(J, _sd_jac_fd(sd, u0, 0.0); rtol = 1e-6)
        @test !isapprox(J, -Matrix(operator_matrix(sd)); rtol = 1e-3)
        twin = only(values(sd.twins))
        @test operator_matrix(twin) === operator_matrix(sd)
        @test twin.operator === sd.operator
        @test twin.source !== sd.source
    end

    # A plain number type always takes the `copyto!` path: a Float64 `u` into a Float32
    # state is converted there, as before, with no twin and no allocation.
    @testset "plain u wider than the state: copied" begin
        sₕ = Bramble.element(Wₕ, Float32)
        fill!(parent(sₕ), 1.0f0)
        a = form(Wₕ, Wₕ, (u, v) -> inner₊(sₕ * ∇ₕ(u), ∇ₕ(v)))
        l = form(Wₕ, v -> innerₕ(x -> 1.0 + x[1], v))
        sd = semidiscretize(a, l; state = sₕ, reassemble = true)
        du, u = zeros(n), copy(u0)
        sd(du, u, nothing, 0.0)
        @test parent(sₕ) == Float32.(u)
        @test isempty(sd.twins)
        @test _sd_residual_allocs(sd, du, u, 0.0) == 0
    end

    # A state no form reads in its expression cannot follow a Dual: refused, never the
    # Picard `-A` passed off as the Jacobian.
    @testset "state the forms do not rebind: refused" begin
        sₕ = Bramble.element(Wₕ, 1.0)
        a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
        l = form(Wₕ, v -> innerₕ(x -> 1.0 + x[1], v))
        sd = semidiscretize(a, l; state = sₕ, reassemble = true)
        @test_throws ArgumentError _sd_jac(sd, u0, 0.0)
        @test isempty(sd.twins)

        # a `build`-based operator is opaque to the rebind
        mk(t) = (form(Wₕ, Wₕ, (u, v) -> inner₊(sₕ * ∇ₕ(u), ∇ₕ(v))), t -> nothing)
        sdb = semidiscretize(mk, l; state = sₕ)
        err = try
            _sd_jac(sdb, u0, 0.0)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("build", sprint(showerror, err))

        # a hook may read the user's state, which a Dual `u` never reaches: refused
        fₕ = Bramble.element(Wₕ, 0.0)
        ah = form(Wₕ, Wₕ, (u, v) -> inner₊(sₕ * ∇ₕ(u), ∇ₕ(v)))
        sdh = semidiscretize(ah, form(Wₕ, v -> innerₕ(fₕ, v)); state = sₕ,
            reassemble = true, (update_coefficients!) = t -> (parent(fₕ) .= parent(sₕ) .^ 2))
        err = try
            _sd_jac(sdh, u0, 0.0)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("update_coefficients!", sprint(showerror, err))
        @test isempty(sdh.twins)

        # a composite state is read through component views, which are not rebound
        Vₕ = Wₕ × Wₕ
        cₕ = Bramble.element(Vₕ, 1.0)
        av = form(Vₕ, Vₕ, (u, v) -> inner₊(∇ₕ(u(1)), ∇ₕ(v(1))) + inner₊(∇ₕ(u(2)), ∇ₕ(v(2))))
        lv = form(Vₕ, v -> innerₕ(cₕ(1), v(1)) + 100 * innerₕ(cₕ(2), v(2)))
        sdv = semidiscretize(av, lv; state = cₕ)
        uv = collect(range(0.5, 2.0; length = ndofs(Vₕ)))
        @test_throws ArgumentError _sd_jac(sdv, uv, 0.0)
        @test all(==(1.0), parent(cₕ))
    end
end

@testset "SciML entry points without the extension" begin
    # a child process script: the shared test helpers are not defined there
    code = """
    using Bramble
    Ωₕ = Bramble.mesh(Bramble.domain(Bramble.interval(0.0, 1.0)), 5)
    W = Bramble.gridspace(Ωₕ)
    a = Bramble.form(W, W, (u, v) -> Bramble.inner₊(Bramble.∇ₕ(u), Bramble.∇ₕ(v)))
    l = Bramble.form(W, v -> Bramble.innerₕ(Bramble.Rₕ(W, x -> 1.0), v))
    sd = Bramble.semidiscretize(a, l)
    u0 = zeros(Bramble.ndofs(W))
    calls = (
        () -> Bramble.ode_function(sd),
        () -> Bramble.ode_problem(sd, u0, (0.0, 1.0)),
        () -> Bramble.linear_problem(a, l),
        () -> Bramble.nonlinear_problem((r, u, p) -> r, u0),
        () -> Bramble.adjoint_sensitivities(nothing, nothing),
    )
    for f in calls
        try
            f()
            println("RETURNED")
        catch e
            println(first(split(sprint(showerror, e), '.')))
        end
    end
    """
    @test _sd_child_messages(code) == [
        "ode_function requires SciMLBase",
        "ode_problem requires SciMLBase",
        "linear_problem requires SciMLBase",
        "nonlinear_problem requires SciMLBase",
        "adjoint_sensitivities requires SciMLSensitivity"
    ]
end

end # module FormSemidiscreteTests
