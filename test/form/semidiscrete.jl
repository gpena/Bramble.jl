module FormSemidiscreteTests

using Test
using Bramble
using Bramble: matrix_type
using SparseArrays
using ForwardDiff: Dual, value
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
using ..TestUtils: _check_eoc

# `semidiscretize` and the residual it returns (src/problems/semidiscrete.jl) need no SciMLBase:
# a `Semidiscretization` is a callable with the `(du, u, p, t)` signature plus two matrices.
# Everything here therefore belongs in the always-run unit group, and only the
# `ODEFunction`/`ODEProblem`/`LinearProblem` wrapping is left to test/ext/sciml_ext.jl.
#
# The integration test at the bottom steps the system with backward Euler written out by
# hand, which keeps the mathematical claim -- that this semidiscretisation is second order
# in space -- independent of any solver package.

const _SD_T = 0.1

_sd_uex(x, t) = exp(-t) * sinpi(x[1])
_sd_src(x, t) = (pi^2 - 1) * exp(-t) * sinpi(x[1])

# Allocation assertions go behind a function barrier: measured at `@testset` scope they
# would count the closure the testset body becomes (bramble-verification §1).
_sd_residual_allocs(sd, du, u, t) = @allocated sd(du, u, nothing, t)
_sd_jacobian_allocs(J, sd, u, t) = @allocated Bramble.jacobian!(J, sd, u, nothing, t)
_sd_rhs_allocs(rhs, du, u, t) = @allocated rhs(du, u, nothing, t)

function _sd_problem(n)
    Ωₕ = Bramble.mesh(Bramble.domain(Bramble.interval(0.0, 1.0)), n)
    Wₕ = gridspace(Ωₕ)
    fₕ = Bramble.element(Wₕ, 0.0)
    a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
    l = form(Wₕ, v -> innerₕ(fₕ, v))
    return Ωₕ, Wₕ, fₕ, a, l
end

@testset "semidiscretize" begin
    Ωₕ, Wₕ, fₕ, a, l = _sd_problem(21)
    n = ndofs(Wₕ)
    I = Bramble.interval(0.0, 1.0)
    Rₕ!(fₕ, x -> 1.0)

    @testset "constraint carrier is chosen once, by shape and arity" begin
        @test semidiscretize(a, l).constraints isa Bramble.NoConstraints
        @test semidiscretize(a, l; dirichlet = :boundary).constraints isa Bramble.LabelsOnly
        @test semidiscretize(a, l; dirichlet = :boundary => x -> 0.0).constraints isa
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

        bcs_x = dirichlet_constraints(Ωₕ, :boundary => x -> 1.0)
        @test semidiscretize(a, l; dirichlet = bcs_x).constraints isa
              Bramble.StaticConstraints
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
        for dirichlet in (nothing, :boundary, :boundary => x -> 0.5, bcs)
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

    @testset "boundary values reach the residual" begin
        # Labels without values constrain to zero; `assemble!` cannot express that, so the
        # rows are cleared instead.
        sd_zero = semidiscretize(a, l; dirichlet = :boundary)
        F = zeros(n)
        sd_zero(F, zeros(n), nothing, 0.0)
        @test iszero(F[1]) && iszero(F[n])

        sd_static = semidiscretize(a, l; dirichlet = :boundary => x -> 7.0)
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

    @testset "boundary values reach p when given (x, t, p)" begin
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

    @testset "mixed (x, t) / (x, t, p) arity across labels is rejected" begin
        Ωₕ2 = Bramble.mesh(
            Bramble.domain(Bramble.interval(0.0, 1.0), :left => :left, :right => :right), 11
        )
        @test_throws ErrorException dirichlet_constraints(
            Ωₕ2, I, :left => (x, t) -> t, :right => (x, t, p) -> p * t
        )
    end

    @testset "update_coefficients! runs before each assembly" begin
        seen = Float64[]
        sd = semidiscretize(
            a, l; (update_coefficients!) = t -> (push!(seen, t); Rₕ!(fₕ, x -> t))
        )
        F = zeros(n)
        sd(F, zeros(n), nothing, 2.5)
        @test seen == [2.5]
        # `l` is `innerₕ(fₕ, v)`, so an fₕ set to t scales the assembled source by t.
        G = zeros(n)
        Rₕ!(fₕ, x -> 1.0)
        assemble!(G, l)
        @test F ≈ 2.5 .* G
    end

    @testset "update_coefficients! reaches p when given (t, p)" begin
        seen = Tuple{Float64, Float64}[]
        sd = semidiscretize(
            a, l; (update_coefficients!) = (t, p) -> (push!(seen, (t, p)); Rₕ!(fₕ, x -> t * p))
        )
        F = zeros(n)
        sd(F, zeros(n), 3.0, 2.5)
        @test seen == [(2.5, 3.0)]
        G = zeros(n)
        Rₕ!(fₕ, x -> 1.0)
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
        # `build` runs once at construction (probing the `Float64` pattern), and defaults to
        # `reassemble = true` -- a `build`-based operator exists specifically to be rebuilt.
        @test build_calls[] == 1
        @test occursin("every step", sprint(show, MIME"text/plain"(), sd))

        A₀ = assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v)))
        sd(zeros(n), zeros(n), nothing, 3.0)
        @test operator_matrix(sd) ≈ 4 .* A₀      # α(3.0) = 1 + 3 = 4
        @test build_calls[] == 1                 # same element type: no rebuild, only refill

        # A `ForwardDiff.Dual` `t` -- the same thing a Rosenbrock stepper's `tgrad` reaches
        # for -- rebuilds once at that new element type, and never throws `InexactError`
        # trying to write a `Dual` into the `Float64` matrix the classic `BilinearForm` path
        # would have kept.
        t_dual = Dual(3.0, 1.0)
        u_dual = Dual.(zeros(n), 0.0)
        du_dual = similar(u_dual)
        sd(du_dual, u_dual, nothing, t_dual)
        @test build_calls[] == 2
        @test all(isfinite, value.(du_dual))

        # Calling again at the already-seen `Dual` type refills without rebuilding.
        sd(du_dual, u_dual, nothing, t_dual)
        @test build_calls[] == 2

        # The `Float64` path stays exactly as before, unaffected by the Dual excursion in
        # between: same cached entry, refilled at its own `t` rather than rebuilt.
        sd(zeros(n), zeros(n), nothing, 1.0)
        @test operator_matrix(sd) ≈ 2 .* A₀      # α(1.0) = 1 + 1 = 2
        @test build_calls[] == 2
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
        # The default would spell out every nested type parameter (gpena/Bramble.jl#17).
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
                semidiscretize(a, l; dirichlet = :boundary => x -> 0.0)
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

# The matrix-type seam extended to the consumers of assembly (S1.2, gpena/Bramble.jl#12):
# `semidiscretize`, `mass_matrix` and `operator_matrix` read whatever matrix type the form's
# own backend produces -- the `Semidiscretization{...,MT,...}` parameter already carried for
# both matrices -- rather than assuming `SparseMatrixCSC`. A dense `Matrix{Float64}` backend
# must semidiscretise to the exact same numbers as the default CSC backend, and the residual
# built from it must agree too.
@testset "Dense backend" begin
    I = Bramble.interval(0.0, 1.0)
    Ωc = Bramble.mesh(Bramble.domain(I), 21, false)
    Ωd = Bramble.mesh(Bramble.domain(I), 21, false; backend = backend(matrix_type = Matrix{Float64}))
    set_points!(Ωd, points(Ωc))
    Wc, Wd = gridspace(Ωc), gridspace(Ωd)

    function _sd_forms(W)
        fₕ = Rₕ(W, x -> 1.0)
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

    @testset "operator_matrix and mass_matrix agree with CSC" begin
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
        cache_c, cache_d = Dict(), Dict()
        build(a) = uₕ -> (a, _ -> nothing)
        Bc = type_cached_assemble!(build(ac), cache_c, Bramble.element(Wc, 0.0))
        Bd = type_cached_assemble!(build(ad), cache_d, Bramble.element(Wd, 0.0))
        @test isapprox(Matrix(Bc), Bd; atol = 1e-13)
    end

    @testset "jacobian_pattern keeps the same nonzero count" begin
        Pc = jacobian_pattern(ac)
        Pd = jacobian_pattern(ad)
        @test Pc isa SparseMatrixCSC
        @test count(!iszero, Matrix(Pc)) == count(!iszero, Matrix(Pd))
    end
end

# gpena/Bramble.jl#283: pin `sd2 = semidiscretize_second_order(...)`, `rhs =
# semidiscretize_rhs(...)` and their residual calls as inferred and allocation-free -- the
# calls a time-stepping loop makes every step. Allocation checks go behind a function
# barrier (bramble-verification §1); `@inferred` reads no global binding here, so it is
# left in the barrier alongside the warm-up rather than pulled out to top level.
@testset "sd2 and rhs are inferred and allocation-free (#283)" begin
    function _sd2_alloc(sd2, dv, v, u, t)
        sd2(dv, v, u, nothing, t)             # cold: records
        @inferred sd2(dv, v, u, nothing, t)
        return @allocated sd2(dv, v, u, nothing, t)
    end

    function _rhs_alloc(rhs, du, u, t)
        rhs(du, u, nothing, t)                # cold: records
        @inferred rhs(du, u, nothing, t)
        return @allocated rhs(du, u, nothing, t)
    end

    function _sd2_problem(dims)
        Ωₕ = dims == 1 ?
             Bramble.mesh(Bramble.domain(Bramble.interval(0.0, 1.0)), 21) :
             Bramble.mesh(
            Bramble.domain(Bramble.interval(0.0, 1.0) × Bramble.interval(0.0, 1.0)),
            (9, 11),
            (true, true)
        )
        Wₕ = gridspace(Ωₕ)
        fₕ = Bramble.element(Wₕ, 0.0)
        Rₕ!(fₕ, x -> 1.0)
        K = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
        l = form(Wₕ, v -> innerₕ(fₕ, v))
        return Wₕ, K, l
    end

    for (lbl, dims) in (("1D", 1), ("2D", 2))
        @testset "$lbl" begin
            Wₕ, K, l = _sd2_problem(dims)
            n = ndofs(Wₕ)

            @testset "semidiscretize_second_order, dirichlet = $(repr(dirichlet))" for dirichlet in (
                :boundary, nothing
            )
                sd2 = semidiscretize_second_order(K, l; dirichlet = dirichlet)
                dv = zeros(n)
                v = collect(range(0.1, 0.9; length = n))
                u = collect(range(0.25, 1.75; length = n))
                @test _sd2_alloc(sd2, dv, v, u, 0.0) == 0
            end

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

    cₕ = Rₕ(Wₕ, x -> 1.0)
    a = form(Vₕ, Vₕ, (u, v) -> inner₊(∇ₕ(u(1)), ∇ₕ(v(1))) + inner₊(∇ₕ(u(2)), ∇ₕ(v(2))))
    # Distinct values per component, so a residual that mixed the blocks could not pass.
    l = form(Vₕ, v -> innerₕ(cₕ, v(1)) + 3 * innerₕ(cₕ, v(2)))

    sd = semidiscretize(a, l; dirichlet = :boundary => x -> 0.0)
    u = collect(range(0.5, 2.5; length = n))
    du = zeros(n)
    F = zeros(n)
    sd(F, zeros(n), nothing, 0.0)
    sd(du, u, nothing, 0.0)

    @test du ≈ F - operator_matrix(sd) * u
    @test F[2:(nleaf - 1)] ≈ F[(nleaf + 2):(2nleaf - 1)] ./ 3
    @test _sd_residual_allocs(sd, du, u, 0.0) == 0

    M = mass_matrix(sd)
    @test iszero(M[1, 1]) && iszero(M[nleaf, nleaf])
    @test iszero(M[nleaf + 1, nleaf + 1]) && iszero(M[n, n])

    @testset "dirichlet_components binds the labels to one leaf" begin
        sd₁ = semidiscretize(a, l; dirichlet = :boundary => x -> 0.0, dirichlet_components = 1)
        M₁ = mass_matrix(sd₁)
        @test iszero(M₁[1, 1]) && iszero(M₁[nleaf, nleaf])
        @test !iszero(M₁[nleaf + 1, nleaf + 1]) && !iszero(M₁[n, n])
    end
end

# The semidiscretisation is second order in space. Backward Euler is written out here rather
# than imported so the claim rests on nothing but Bramble: on a constrained row `M` is zero
# and `A` is `eₖ`, so `(M + Δt A) u = M u + Δt F` reduces to `u[i] = g(x_i, t)` and the
# boundary condition is imposed by the same step that advances the interior. `Δt = h²` keeps
# the first-order time error at the order being measured.
@testset "semidiscretize: order of convergence (backward Euler)" begin
    function step_to(n)
        Ωₕ, Wₕ, fₕ, a, l = _sd_problem(n)
        h = hₘₐₓ(Ωₕ)
        bcs = dirichlet_constraints(
            Ωₕ, Bramble.interval(0.0, _SD_T), :boundary => (x, t) -> 0.0
        )
        sd = semidiscretize(
            a, l; dirichlet = bcs, (update_coefficients!) = t -> Rₕ!(fₕ, x -> _sd_src(x, t))
        )

        M = mass_matrix(sd)
        nsteps = ceil(Int, _SD_T / h^2)
        Δt = _SD_T / nsteps
        K = M + Δt * operator_matrix(sd)

        u = collect(parent(Rₕ(Wₕ, x -> _sd_uex(x, 0.0))))
        dirichlet_bc!(u, sd, 0.0)
        F = similar(u)
        zero_u = zero(u)
        for k in 1:nsteps
            t = k * Δt
            sd(F, zero_u, nothing, t)      # F(t), the residual at u = 0
            u = K \ (M * u + Δt * F)
        end

        uₕ = Bramble.element(Wₕ)
        parent(uₕ) .= u
        return normₕ(Rₕ(Wₕ, x -> _sd_uex(x, _SD_T)) - uₕ), h
    end

    eoc = _check_eoc(step_to, (11, 21, 41, 81))
    @test last(eoc) > 1.95
end

# `semidiscretize_rhs` (gpena/Bramble.jl#163): `du = M⁻¹(F(t) - A u)` with `M`'s diagonal
# folded in once, instead of a solver factorising `M` at every step. Checked against
# `weights(Wₕ).innerh`, the space's own independently-computed `L²` weight vector -- not
# against `mass_matrix(sd)`'s diagonal, which would just check the implementation agrees
# with itself.
@testset "semidiscretize_rhs: matrix-free explicit right-hand side" begin
    Ωₕ, Wₕ, fₕ, a, l = _sd_problem(21)
    Rₕ!(fₕ, x -> 1.0)
    n = ndofs(Wₕ)
    h = Bramble.weights(Wₕ).innerh

    sd = semidiscretize(a, l)
    @test sd.constraints isa Bramble.NoConstraints
    rhs = semidiscretize_rhs(sd)
    @test rhs isa Bramble.SemidiscretizeRHS

    u = collect(range(0.2, 1.7; length = n))
    du_rhs, du_sd = zeros(n), zeros(n)
    t = 0.37
    rhs(du_rhs, u, nothing, t)
    sd(du_sd, u, nothing, t)
    @test du_rhs≈du_sd ./ h atol=1e-12 rtol=1e-12
    @test _sd_rhs_allocs(rhs, du_rhs, u, t) == 0

    @testset "requires NoConstraints: any Dirichlet row makes M singular there" begin
        sd_static = semidiscretize(a, l; dirichlet = :boundary => x -> 0.0)
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

    @testset "requires an invertible (nonzero) diagonal" begin
        # Diagonal, but exactly zero at the left boundary point (x = 0) -- no coupling
        # term, so this does not hit the non-diagonal case above; it is its own check.
        cₕ = Rₕ(Wₕ, x -> x[1])
        mass_zero = form(Wₕ, Wₕ, (u, v) -> innerₕ(cₕ * u, v))
        sd_zero = semidiscretize(a, l; mass = mass_zero)
        @test iszero(mass_matrix(sd_zero)[1, 1])
        @test_throws ArgumentError semidiscretize_rhs(sd_zero)
    end
end

end # module FormSemidiscreteTests
