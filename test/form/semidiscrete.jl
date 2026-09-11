using Test
using Bramble
using SparseArrays

# `semidiscretize` and the residual it returns (src/form/semidiscrete.jl) need no SciMLBase:
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

function _sd_problem(n)
    Ωₕ = Bramble.mesh(Bramble.domain(Bramble.interval(0.0, 1.0)), n)
    Wₕ = gridspace(Ωₕ)
    fₕ = Bramble.element(Wₕ, 0.0)
    a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))
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
        @test semidiscretize(a, l; dirichlet=:boundary).constraints isa Bramble.LabelsOnly
        @test semidiscretize(a, l; dirichlet=:boundary => x -> 0.0).constraints isa
            Bramble.StaticConstraints

        # A time domain is not what marks constraints as time dependent -- the conditions'
        # arity is, the same test `dirichlet_constraints` validates with.
        bcs_t = dirichlet_constraints(Ωₕ, I, :boundary => (x, t) -> 2t)
        @test semidiscretize(a, l; dirichlet=bcs_t).constraints isa
            Bramble.TimeDependentConstraints

        bcs_x = dirichlet_constraints(Ωₕ, :boundary => x -> 1.0)
        @test semidiscretize(a, l; dirichlet=bcs_x).constraints isa
            Bramble.StaticConstraints
    end

    @testset "mass matrix carries the algebraic rows" begin
        M₀ = assemble(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v)))
        sd = semidiscretize(a, l; dirichlet=:boundary)
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
        sd = semidiscretize(a, l; dirichlet=:boundary)
        @test operator_matrix(sd) == assemble(a; dirichlet=:boundary)
        @test Bramble.space(sd) === Wₕ
    end

    @testset "residual, Jacobian, and allocations" begin
        bcs = dirichlet_constraints(Ωₕ, I, :boundary => (x, t) -> 2t)
        for dirichlet in (nothing, :boundary, :boundary => x -> 0.5, bcs)
            sd = semidiscretize(a, l; dirichlet=dirichlet)
            A = operator_matrix(sd)
            u = collect(range(0.25, 1.75; length=n))
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
        sd_zero = semidiscretize(a, l; dirichlet=:boundary)
        F = zeros(n)
        sd_zero(F, zeros(n), nothing, 0.0)
        @test iszero(F[1]) && iszero(F[n])

        sd_static = semidiscretize(a, l; dirichlet=:boundary => x -> 7.0)
        sd_static(F, zeros(n), nothing, 0.0)
        @test F[1] ≈ 7.0 && F[n] ≈ 7.0

        bcs = dirichlet_constraints(Ωₕ, I, :boundary => (x, t) -> 1 + 3t)
        sd_time = semidiscretize(a, l; dirichlet=bcs)
        sd_time(F, zeros(n), nothing, 0.0)
        @test F[1] ≈ 1.0
        sd_time(F, zeros(n), nothing, 2.0)
        @test F[1] ≈ 7.0
        @test F[n] ≈ 7.0
    end

    @testset "update_coefficients! runs before each assembly" begin
        seen = Float64[]
        sd = semidiscretize(
            a, l; (update_coefficients!)=t -> (push!(seen, t); Rₕ!(fₕ, x -> t))
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

    @testset "state exposes u to the forms" begin
        uₕ = Bramble.element(Wₕ, 0.0)
        sd = semidiscretize(a, l; state=uₕ)
        u = collect(range(1.0, 2.0; length=n))
        sd(zeros(n), u, nothing, 0.0)
        @test parent(uₕ) == u
    end

    @testset "reassemble refills the operator" begin
        α = Ref(1.0)
        # The `Ref` itself goes in the expression, not `α[]`: dereferencing evaluates
        # once, when the form is built, and the coefficient stops being live.
        aα = form(Wₕ, Wₕ, (u, v) -> α * innerₕ(u, v))
        sd = semidiscretize(aα, l; reassemble=true, (update_coefficients!)=t -> (α[] = t))
        A₁ = copy(operator_matrix(sd))
        sd(zeros(n), zeros(n), nothing, 3.0)
        @test operator_matrix(sd) ≈ 3 .* A₁

        # Without `reassemble` the matrix assembled at construction is reused as is.
        α[] = 1.0
        sd_fixed = semidiscretize(
            aα, l; reassemble=false, (update_coefficients!)=t -> (α[] = t)
        )
        A₂ = copy(operator_matrix(sd_fixed))
        sd_fixed(zeros(n), zeros(n), nothing, 3.0)
        @test operator_matrix(sd_fixed) == A₂
    end

    @testset "consistent initial conditions" begin
        bcs = dirichlet_constraints(Ωₕ, I, :boundary => (x, t) -> 5 + t)
        sd = semidiscretize(a, l; dirichlet=bcs)
        u = fill(-1.0, n)
        @test dirichlet_bc!(u, sd, 2.0) === u
        @test u[1] ≈ 7.0 && u[n] ≈ 7.0
        @test u[5] == -1.0

        sd_zero = semidiscretize(a, l; dirichlet=:boundary)
        v = fill(-1.0, n)
        dirichlet_bc!(v, sd_zero, 0.0)
        @test iszero(v[1]) && iszero(v[n]) && v[5] == -1.0

        # Nothing to impose leaves the vector alone.
        w = fill(-1.0, n)
        @test dirichlet_bc!(w, semidiscretize(a, l), 0.0) == fill(-1.0, n)
    end

    @testset "display" begin
        bcs = dirichlet_constraints(Ωₕ, I, :boundary => (x, t) -> 0.0)
        sd = semidiscretize(a, l; dirichlet=bcs)

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
            sprint(show, MIME"text/plain"(), semidiscretize(a, l; dirichlet=:boundary)),
        )
        @test occursin(
            "g(x) on :boundary",
            sprint(
                show,
                MIME"text/plain"(),
                semidiscretize(a, l; dirichlet=:boundary => x -> 0.0),
            ),
        )
        @test occursin(
            "Constraints: none", sprint(show, MIME"text/plain"(), semidiscretize(a, l))
        )
        @test occursin(
            "every step",
            sprint(show, MIME"text/plain"(), semidiscretize(a, l; reassemble=true)),
        )
        @test occursin("constrained labels", sprint(show, semidiscretize(a, l)))
    end

    @testset "mismatched spaces are rejected" begin
        _, Wc, _, ac, _ = _sd_problem(9)
        @test_throws ArgumentError semidiscretize(ac, l)
    end
end

@testset "semidiscretize on a composite space" begin
    Ωₕ = Bramble.mesh(Bramble.domain(Bramble.interval(0.0, 1.0)), 11)
    Wₕ = gridspace(Ωₕ)
    Vₕ = vector_gridspace(Ωₕ, Val(2))
    n = ndofs(Vₕ)
    nleaf = ndofs(Wₕ)

    cₕ = Rₕ(Wₕ, x -> 1.0)
    a = form(Vₕ, Vₕ, (u, v) -> inner₊(∇₋ₕ(u(1)), ∇₋ₕ(v(1))) + inner₊(∇₋ₕ(u(2)), ∇₋ₕ(v(2))))
    # Distinct values per component, so a residual that mixed the blocks could not pass.
    l = form(Vₕ, v -> innerₕ(cₕ, v(1)) + 3 * innerₕ(cₕ, v(2)))

    sd = semidiscretize(a, l; dirichlet=:boundary => x -> 0.0)
    u = collect(range(0.5, 2.5; length=n))
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
        sd₁ = semidiscretize(a, l; dirichlet=:boundary => x -> 0.0, dirichlet_components=1)
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
            a, l; dirichlet=bcs, (update_coefficients!)=t -> Rₕ!(fₕ, x -> _sd_src(x, t))
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

    errors = Float64[]
    spacings = Float64[]
    for n in (11, 21, 41, 81)
        e, h = step_to(n)
        push!(errors, e)
        push!(spacings, h)
    end

    eoc = [
        log(errors[i] / errors[i + 1]) / log(spacings[i] / spacings[i + 1]) for
        i in 1:(length(errors) - 1)
    ]
    @test all(>(1.9), eoc)
    @test last(eoc) > 1.95
    @test issorted(errors; rev=true)
end
