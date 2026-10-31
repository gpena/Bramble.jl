# ghost_semidiscrete.jl: `semidiscretize` and `ode_problem` on a constrained ghost space,
# with time-dependent boundary data `(x, t) -> ...` entering through `lift`
# (gpena/Bramble.jl#776). Every reference is a space rebuilt with the data frozen at `t`,
# `x -> g(x, t)`, assembled by the steady `assemble(a, l)`, which shares no code with the
# carrier's `d(t)`. The operator reads the ghosts (`-Δₕ`), so `A_pad d(t)` matters. Seeded
# non-uniform meshes only. `CpuPolyester` is in test/ext/ghost_semidiscrete_polyester.jl.
module GhostSemidiscreteTests

using Test
using Bramble
import Bramble
using Bramble: CpuSerial, CpuThreaded
using LinearAlgebra: norm, isdiag, diag
using ForwardDiff: ForwardDiff
using FiniteDiff: FiniteDiff
using SciMLBase: solve
using OrdinaryDiffEqBDF: FBDF

using ..TestUtils: ghost_mesh

const TIMES = (0.0, 0.4, 1.3)

alloc(f::F) where {F} = (f(); f(); @allocated f())

# The datum as given (`t === nothing`) or frozen at `t`.
at(f, ::Nothing) = f
at(f::Number, t) = f
at(f, t) = x -> f(x, t)

const gb = (x, t) -> t * x[2]
const gc = (x, t) -> sin(t) * x[1]
rel2(t) = u -> (:xmin => ∂ₙ(u) => 0.0, :xmax => u + ∂ₙ(u) => at(gb, t),
    :ymin => u => at(gc, t), :ymax => ∂ₙ(u) => 1.0)
const GHOSTS2 = (:xmin, :xmax, :ymax)

const g1 = (x, t) -> cos(t) + x[1]
rel1(t) = u -> (:xmin => u => at(g1, t), :xmax => 2.0 * u + ∂ₙ(u) => at(gb1, t))
const gb1 = (x, t) -> exp(-t)

laplacian(u, v) = innerₕ(-Δₕ(u), v)
source(v) = innerₕ(x -> 1.0 + x[1], v)

# A band function that does nothing, as wide as the load's band: one task spawn.
noop_band!(args...) = nothing

relerr(a, b) = norm(a - b, Inf) / max(norm(b, Inf), eps(real(eltype(b))))

# The semidiscretization on `W` with time data, and the steady pair at `t` on `W` frozen.
function build(W, rels; f = laplacian, g = source, kwargs...)
    C = constrained(W, rels(nothing))
    sd = semidiscretize(form(C, C, f), form(C, g); kwargs...)
    frozen(t) = (Ct = constrained(W, rels(t)); assemble(form(Ct, Ct, f), form(Ct, g)))
    return C, sd, frozen
end

# `sd(du, u, nothing, t)` against `F − A u` of the frozen pair at each time.
function residual_matches(sd, C, frozen; tol = 1e-12, times = TIMES)
    T = eltype(Bramble.closure(C)[2])
    u = T.(collect(range(-1, 2; length = Bramble.ndofs(C))) .^ 2)
    du = similar(u)
    for t in times
        A, F = frozen(T(t))
        sd(du, u, nothing, T(t))
        r = F - A * u
        @test relerr(du, r) <= tol
    end
    return u, du
end

@testset "Ghost semidiscretization (#776)" begin
    @testset "2D residual against frozen data" begin
        W = gridspace(ghost_mesh(CpuSerial(), (6, 7)); ghosts = GHOSTS2)
        C, sd, frozen = build(W, rel2)
        residual_matches(sd, C, frozen)
        a = form(C, C, laplacian)
        @test Bramble.operator_matrix(sd) == assemble(a)
        M = Bramble.mass_matrix(sd)
        @test isdiag(M)
        @test all(>(0), diag(M))
        @test M == assemble(form(C, C, (u, v) -> innerₕ(u, v)))
        # `A_pad d(t)` matters here: the data at two times give different residuals.
        u = ones(Bramble.ndofs(C))
        @test norm(sd(similar(u), u, nothing, 0.0) - sd(similar(u), u, nothing, 1.3)) > 0.1
    end

    @testset "Static data" begin
        W = gridspace(ghost_mesh(CpuSerial(), (9, 8); seed = 3); ghosts = :boundary)
        rs = t -> u -> (:xmin => ∂ₙ(u) => 1.0, :xmax => u + ∂ₙ(u) => x -> x[2],
            :ymin => ∂ₙ(u) => 0.0, :ymax => 2.0 * u + ∂ₙ(u) => 1.0)
        C, sd, frozen = build(W, rs)
        residual_matches(sd, C, frozen)
        u, du = ones(Bramble.ndofs(C)), zeros(Bramble.ndofs(C))
        @test alloc(() -> sd(du, u, nothing, 0.5)) == 0
        # a mass reading the ghosts is accepted: the data do not depend on time
        sdm = semidiscretize(form(C, C, laplacian), form(C, source);
            mass = form(C, C, (u, v) -> innerₕ(u, v) + 0.01 * innerₕ(Δₕ(u), v)))
        @test !isdiag(Bramble.mass_matrix(sdm))
    end

    @testset "1D" begin
        W = gridspace(ghost_mesh(CpuSerial(), (13,)); ghosts = :xmax)
        C, sd, frozen = build(W, rel1)
        residual_matches(sd, C, frozen)
    end

    @testset "Float32 space" begin
        W = gridspace(ghost_mesh(CpuSerial(), (6, 7); T = Float32); ghosts = GHOSTS2)
        C, sd, frozen = build(W, rel2)
        u, du = residual_matches(sd, C, frozen; tol = 1e-5)
        @test eltype(du) == Float32
        @test eltype(Bramble.mass_matrix(sd)) == Float32
        @test alloc(() -> sd(du, u, nothing, 0.3f0)) == 0
    end

    @testset "Residual allocates nothing" begin
        for ns in ((6, 7), (30, 33))
            W = gridspace(ghost_mesh(CpuSerial(), ns); ghosts = GHOSTS2)
            C, sd, _ = build(W, rel2)
            u = ones(Bramble.ndofs(C))
            du = similar(u)
            @test alloc(() -> sd(du, u, nothing, 0.3)) == 0
        end
    end

    @testset "jacobian! is -A" begin
        W = gridspace(ghost_mesh(CpuSerial(), (6, 7)); ghosts = GHOSTS2)
        C, sd, _ = build(W, rel2)
        J = Bramble.jacobian_prototype(sd)
        u = ones(Bramble.ndofs(C))
        Bramble.jacobian!(J, sd, u, nothing, 0.7)
        @test J == -assemble(form(C, C, laplacian))
    end

    # The operator changes with `t` through `update_coefficients!`: the source must lift
    # with `A_pad` at `t`, not at the step before.
    @testset "reassemble with update_coefficients!" begin
        W = gridspace(ghost_mesh(CpuSerial(), (6, 7)); ghosts = GHOSTS2)
        c = Ref(0.0)
        f = (u, v) -> innerₕ(-Δₕ(u), v) + innerₕ(c * Δₕ(u), v)
        C = constrained(W, rel2(nothing))
        sd = semidiscretize(form(C, C, f), form(C, source);
            (update_coefficients!) = t -> (c[] = t / 4), reassemble = true)
        u = collect(range(-1, 2; length = Bramble.ndofs(C)))
        du = similar(u)
        for t in TIMES
            sd(du, u, nothing, t)
            cr = Ref(t / 4)
            Ct = constrained(W, rel2(t))
            A, F = assemble(form(Ct, Ct, (u, v) -> innerₕ(-Δₕ(u), v) +
                                                   innerₕ(cr * Δₕ(u), v)), form(Ct, source))
            @test relerr(du, F - A * u) <= 1e-12
        end
        @test alloc(() -> sd(du, u, nothing, 0.3)) == 0
    end

    # A Rosenbrock `tgrad` differentiates the residual through `t`: a Dual time takes
    # buffers of its own type and gives ∂ₜ of the source.
    @testset "Dual time" begin
        W = gridspace(ghost_mesh(CpuSerial(), (6, 7)); ghosts = GHOSTS2)
        C, sd, _ = build(W, rel2)
        u = collect(range(-1, 2; length = Bramble.ndofs(C)))
        t = 0.4
        r(t) = sd(zeros(typeof(t), length(u)), u, nothing, t)
        dr = ForwardDiff.derivative(r, t)
        h = 1e-6
        @test relerr(dr, (r(t + h) - r(t - h)) / 2h) <= 1e-6
        @test norm(dr) > 0.1
        rd = r(ForwardDiff.Dual(t, 1.0))
        @test ForwardDiff.value.(rd) ≈ r(t) rtol = 1e-14
        # a state held at Float64 cannot follow a Dual time
        sₕ = element(Bramble.padded_space(C))
        sds = semidiscretize(form(C, C, laplacian), form(C, source); state = sₕ)
        err = try
            sds(zeros(ForwardDiff.Dual{Nothing, Float64, 1}, length(u)), u, nothing,
                ForwardDiff.Dual{Nothing}(t, 1.0))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("tgrad", sprint(showerror, err))
    end

    # A Dual `u` lifts into a padded state of its own type, read by the forms rebound to
    # it, whose own `A_pad` the load reads (#985). The data depend on time, so `A_pad d(t)`
    # carries a u-derivative a twin with only its matrix swapped would drop. The oracle is a
    # central difference of the Float64 residual; the Picard `-A` is checked to differ.
    @testset "Dual state: exact AD Jacobian (#985)" begin
        W = gridspace(ghost_mesh(CpuSerial(), (6, 7)); ghosts = GHOSTS2)
        C = constrained(W, rel2(nothing))
        sₕ = element(Bramble.padded_space(C), 3.0)
        a = form(C, C, (u, v) -> inner₊(sₕ * ∇ₕ(u), ∇ₕ(v)))
        sd = semidiscretize(a, form(C, source); state = sₕ, reassemble = true)
        m = Bramble.ndofs(C)
        u0 = collect(range(0.5, 2.0; length = m))
        t = 0.4
        f! = (du, u) -> sd(du, u, nothing, t)
        J = ForwardDiff.jacobian(f!, zeros(m), u0)
        D = only(keys(sd.twins))
        # the user's state never receives a Dual
        @test eltype(parent(sₕ)) === Float64
        @test all(==(3.0), parent(sₕ))
        Jfd = FiniteDiff.finite_difference_jacobian(u -> sd(zeros(m), u, nothing, t), u0,
            Val(:central))
        @test relerr(J, Jfd) <= 1e-6
        sd(zeros(m), u0, nothing, t)
        @test relerr(J, -Matrix(Bramble.operator_matrix(sd))) > 1e-3
        rd = sd(zeros(D, m), D.(u0), nothing, t)
        @test relerr(ForwardDiff.value.(rd), sd(zeros(m), u0, nothing, t)) <= 1e-14

        # built once: a second Jacobian reuses the twin, its matrix and its padded state
        twin = sd.twins[D]
        Atwin = Bramble.operator_matrix(twin)
        @test eltype(Atwin) === D
        @test twin.state.v !== sₕ
        @test ForwardDiff.jacobian(f!, zeros(m), u0) == J
        @test length(sd.twins) == 1
        @test sd.twins[D] === twin
        @test Bramble.operator_matrix(twin) === Atwin

        # `jacobian!` lifts `u` into the user's state: a Dual one is refused, naming `u`
        Jp = Bramble.jacobian_prototype(sd)
        s0 = copy(parent(sₕ))
        err = try
            Bramble.jacobian!(Jp, sd, D.(u0), nothing, t)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        msg = sprint(showerror, err)
        @test occursin("a u of element type Dual", msg)
        @test !occursin("ForwardDiff time", msg)
        @test parent(sₕ) == s0

        # the Float64 residual still allocates nothing beside the twin
        u, du = copy(u0), zeros(m)
        @test alloc(() -> sd(du, u, nothing, t)) == 0
        @test parent(sₕ) ≈ parent(Bramble.lift(C, u, t)) rtol = 1e-14
    end

    # Without `reassemble` the twin keeps the operator, whose `A_pad` the load reads, and
    # only the source, reading the state here, is rebound.
    @testset "Dual state: source reads it (#985)" begin
        W = gridspace(ghost_mesh(CpuSerial(), (6, 7)); ghosts = GHOSTS2)
        C = constrained(W, rel2(nothing))
        sₕ = element(Bramble.padded_space(C), 2.0)
        sd = semidiscretize(form(C, C, laplacian), form(C, v -> innerₕ(sₕ, v));
            state = sₕ)
        m = Bramble.ndofs(C)
        u0 = collect(range(0.5, 2.0; length = m))
        t = 0.4
        J = ForwardDiff.jacobian((du, u) -> sd(du, u, nothing, t), zeros(m), u0)
        Jfd = FiniteDiff.finite_difference_jacobian(u -> sd(zeros(m), u, nothing, t), u0,
            Val(:central))
        @test relerr(J, Jfd) <= 1e-6
        @test relerr(J, -Matrix(Bramble.operator_matrix(sd))) > 1e-3
        twin = only(values(sd.twins))
        @test twin.operator === sd.operator
        @test Bramble.operator_matrix(twin) === Bramble.operator_matrix(sd)
    end

    @testset "show names the relations" begin
        W = gridspace(ghost_mesh(CpuSerial(), (6, 7)); ghosts = GHOSTS2)
        _, sd, _ = build(W, rel2)
        @test occursin("Constraints: relations",
            sprint(show, MIME"text/plain"(), sd))
    end

    @testset "state receives lift(C, u, t)" begin
        W = gridspace(ghost_mesh(CpuSerial(), (6, 7)); ghosts = GHOSTS2)
        C = constrained(W, rel2(nothing))
        sₕ = element(Bramble.padded_space(C))
        sd = semidiscretize(form(C, C, laplacian), form(C, source); state = sₕ)
        u = collect(range(-1, 2; length = Bramble.ndofs(C)))
        sd(similar(u), u, nothing, 0.4)
        @test parent(sₕ) ≈ parent(Bramble.lift(C, u, 0.4)) rtol = 1e-14
        Wp = gridspace(ghost_mesh(CpuSerial(), (6, 7)))
        @test_throws ArgumentError semidiscretize(form(C, C, laplacian), form(C, source);
            state = element(Wp))
        err = try
            semidiscretize(form(C, C, laplacian), form(C, source); state = element(Wp))
        catch e
            e
        end
        @test occursin("got a ", sprint(showerror, err))
    end

    # Static data: as `t → ∞` the solution tends to the steady `A u = F`.
    @testset "FBDF tends to the steady solution" begin
        W = gridspace(ghost_mesh(CpuSerial(), (7, 8); seed = 5); ghosts = GHOSTS2)
        rs = u -> (:xmin => ∂ₙ(u) => 0.0, :xmax => u + ∂ₙ(u) => x -> x[2],
            :ymin => u => x -> x[1], :ymax => ∂ₙ(u) => 1.0)
        C = constrained(W, rs)
        a, l = form(C, C, laplacian), form(C, source)
        A, F = assemble(a, l)
        prob = ode_problem(a, l, zeros(Bramble.ndofs(C)), (0.0, 80.0))
        sol = solve(prob, FBDF(); reltol = 1e-10, abstol = 1e-12)
        @test relerr(sol.u[end], A \ F) <= 1e-6
    end

    # `mass` reaches `semidiscretize` through the ghost pair: the problem's mass matrix is
    # `assemble(m)`, not the default `innerₕ(u, v)` one.
    @testset "ode_problem forwards mass (ghost)" begin
        W = gridspace(ghost_mesh(CpuSerial(), (6, 7)); ghosts = GHOSTS2)
        C = constrained(W, rel2(nothing))
        a, l = form(C, C, laplacian), form(C, source)
        m = form(C, C, (u, v) -> 3.0 * innerₕ(u, v))
        prob = ode_problem(a, l, zeros(Bramble.ndofs(C)), (0.0, 1.0); mass = m)
        @test prob.f.mass_matrix == assemble(m)
        @test prob.f.mass_matrix != Bramble.mass_matrix(semidiscretize(a, l))
    end

    @testset "Refusals" begin
        Ω = ghost_mesh(CpuSerial(), (6, 7))
        W = gridspace(Ω; ghosts = GHOSTS2)
        C = constrained(W, rel2(nothing))
        a, l = form(C, C, laplacian), form(C, source)
        says(f, s) =
            try
                f()
                false
            catch e
                e isa ArgumentError && occursin(s, sprint(showerror, e))
            end
        @test_throws ArgumentError semidiscretize(a, l; dirichlet = :xmin)
        @test says(() -> semidiscretize(a, l; dirichlet = :xmin), "u => g(x, t)")
        @test_throws ArgumentError semidiscretize(a, l; dirichlet_components = 1)
        # ghosts kept as unknowns
        @test says(() -> semidiscretize(form(W, W, laplacian), form(W, source)),
            "relation in constrained")
        @test_throws ArgumentError ode_problem(form(W, W, laplacian), form(W, source),
            zeros(Bramble.ndofs(W)), (0.0, 1.0))
        # a time-dependent operator built by a function
        P = gridspace(Ω)
        @test says(() -> semidiscretize(t -> (form(C, C, laplacian), t -> nothing), l),
            "reassemble = true")
        # plain and ghost forms together
        @test_throws ArgumentError semidiscretize(form(P, P, laplacian), l)
        @test_throws ArgumentError semidiscretize(a, form(P, source))
        @test says(() -> semidiscretize(a, form(P, source)), "l::LinearForm")
        # a plain-space mass with a ghost operator
        @test says(() -> semidiscretize(a, l; mass = form(P, P, (u, v) -> innerₕ(u, v))),
            "mass::BilinearForm")
        @test_throws "on every side or on none" ode_problem(form(P, P, laplacian), l,
            zeros(Bramble.ndofs(C)), (0.0, 1.0))
        @test_throws "on every side or on none" ode_problem(a, form(P, source),
            zeros(Bramble.ndofs(C)), (0.0, 1.0))
        # forms on another constrained space of the same mesh
        C2 = constrained(W, rel2(nothing))
        @test_throws ArgumentError semidiscretize(form(C2, C2, laplacian), l)
        # a mass reading the ghosts at :xmax, whose data depend on time
        mΔ = form(C, C, (u, v) -> innerₕ(u, v) + 0.01 * innerₕ(Δₕ(u), v))
        @test says(() -> semidiscretize(a, l; mass = mΔ), "d′(t)")
        sd = semidiscretize(a, l)
        @test says(() -> Bramble.semidiscretize_rhs(sd), "ode_problem(sd")
    end

    # Departure 9: `CpuThreaded` spawns its tasks on each band pass; the residual equals
    # serial and allocates no more than the plain semidiscretization's plus one spawn per
    # pass it adds (two data groups, `L g`, the load), the same at two sizes.
    @testset "CpuThreaded" begin
        figures = map(((6, 7), (30, 33))) do ns
            Ws = gridspace(ghost_mesh(CpuSerial(), ns); ghosts = GHOSTS2)
            _, sds, _ = build(Ws, rel2)
            Wt = gridspace(ghost_mesh(CpuThreaded(), ns); ghosts = GHOSTS2)
            C, sdt, _ = build(Wt, rel2)
            u = collect(range(-1, 2; length = Bramble.ndofs(C)))
            dus, dut = similar(u), similar(u)
            sds(dus, u, nothing, 0.4)
            sdt(dut, u, nothing, 0.4)
            @test relerr(dut, dus) <= 1e-13
            P = gridspace(ghost_mesh(CpuThreaded(), ns))
            sdp = semidiscretize(form(P, P, laplacian), form(P, source))
            up = ones(Bramble.ndofs(P))
            dup = similar(up)
            x = zeros(3)
            spawn = alloc(() -> Bramble._run_bands!(CpuThreaded(), noop_band!, x, x, x, x,
                x, x, x, x, x, x, x))
            (ghost = alloc(() -> sdt(dut, u, nothing, 0.3)),
                plain = alloc(() -> sdp(dup, up, nothing, 0.3)), spawn)
        end
        f1, f2 = figures
        @test f1.ghost == f2.ghost
        # Under CPU contention a pass can settle, for the rest of the process, into
        # re-boxing its two band closures in every task: `nthreads × 2 × closure` bytes
        # more, the same at every size. Which passes do so varies from run to run. The
        # residual's five closures are 256 (the load), 128 (`lift`'s load), 80 (`L g`) and
        # 64 + 64 (the data groups) bytes. On a macOS runner at 3 threads, ghost was 7808
        # against 6272 when idle (3 × 2 × 256). Under 8 spinners locally at 4 threads, it
        # was 11696 against 8112 (4 × 2 × (256 + 128 + 64)), with plain 3808 against 1760.
        @test f1.ghost <= f1.plain + 4 * f1.spawn + 2 * Threads.nthreads() * 592
    end

    # The residual refreshes the operator before the source, so a `build` whose `refill!`
    # sets a coefficient the source also reads gives `F(t)` at `t`, not at the previous
    # call's time; `semidiscretize_rhs` keeps the same order and agrees with it.
    @testset "semidiscretize_rhs after another t" begin
        P = gridspace(ghost_mesh(CpuSerial(), (9, 8)))
        αₕ = element(P)
        mk(t) = (form(P, P, (u, v) -> laplacian(u, v) + innerₕ(αₕ * u, v)),
            t -> (Rₕ!(αₕ, x -> 1 + t * x[1]); nothing))
        l = form(P, v -> innerₕ(αₕ, v))
        sd = semidiscretize(mk, l)
        rhs = Bramble.semidiscretize_rhs(sd)
        M = Bramble.mass_matrix(sd)
        n = Bramble.ndofs(P)
        u = sin.(1:n)
        sd(zeros(n), u, nothing, 0.0)
        a = sd(zeros(n), u, nothing, 0.9)
        sd(zeros(n), u, nothing, 0.0)
        b = M * rhs(zeros(n), u, nothing, 0.9)
        Rₕ!(αₕ, x -> 1 + 0.9 * x[1])
        r = assemble(l) - assemble(mk(0.9)[1]) * u
        @test relerr(a, r) <= 1e-12
        @test relerr(b, r) <= 1e-12
    end
end

end
