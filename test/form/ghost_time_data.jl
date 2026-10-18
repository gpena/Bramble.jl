module GhostTimeDataTests

# Relation data that depend on time, `(x, t) -> …` (gpena/Bramble.jl#776): the closure
# keeps its `E` and its affine part becomes `d(t) = L g(t)`, which `_lift_affine!` evaluates
# with no block solve and no allocation, and `lift(Cₕ, u, t)` lifts with. Every reference
# is a space rebuilt with the data frozen at `t`, `x -> g(x, t)`, which takes the static
# path's block solves and shares no code with the map. Seeded random meshes only; 3D runs
# nightly (`WITH_3D_TESTS`). `CpuPolyester` is in test/ext/ghost_polyester.jl.

using Test
using Bramble
import Bramble
using Bramble: CpuSerial, CpuThreaded
using LinearAlgebra: norm
using SparseArrays: nonzeros
using ForwardDiff: ForwardDiff

using ..TestUtils: WITH_3D_TESTS, ghost_mesh

const K = [2.0 1.0; 1.0 2.0]
const TIMES = (0.0, 0.3, 1.7)

alloc(f::F) where {F} = (f(); f(); @allocated f())

# The datum `f` as given (`t === nothing`) or frozen at `t`, a function of the point.
at(f, ::Nothing) = f
at(f::Number, t) = f
at(f, t) = x -> f(x, t)

relerr(a, b) = norm(a - b, Inf) / max(norm(b, Inf), eps(real(eltype(b))))

# `rels(t)` builds the relations with the data as given (`nothing`) or frozen at `t`: the
# space with time data and its frozen references, on the ghost space `W`.
spaces(W, rels) = (constrained(W, rels(nothing)), t -> constrained(W, rels(t)))

# Against the frozen spaces at each time: `E`, `d(t)` from `_lift_affine!`, `lift(C, u, t)`
# and `lift!(uₕ, C, u, t)`.
function matches_frozen(C, Cfix; tol = 1e-13, times = TIMES)
    n = Bramble.ndofs(C)
    u = collect(range(-1, 2; length = n)) .^ 2
    u = convert(Vector{eltype(Bramble.closure(C)[2])}, u)
    d = similar(Bramble.closure(C)[2])
    uₕ = Bramble.lift(C, u, first(times))
    for t in times
        Ct = Cfix(t)
        Et, dt = Bramble.closure(Ct)
        @test Bramble.ndofs(Ct) == n
        @test Bramble.closure(C)[1] == Et
        @test relerr(Bramble._lift_affine!(d, C, t), dt) <= tol
        Ut = parent(Bramble.lift(Ct, u))
        U = parent(Bramble.lift(C, u, t))
        @test relerr(U, Ut) <= tol
        @test parent(Bramble.lift!(uₕ, C, u, t)) == U
    end
    @test any(t -> norm(Bramble.closure(Cfix(t))[2] - Bramble.closure(Cfix(0.0))[2]) > 0.1,
        times)   # the data do change `d`
    return nothing
end

function rels2d(t)
    u -> (:xmin => ∂ₙ(u) => at((x, s) -> sin(s) + x[2] * cos(s), t),
        :xmax => u + 0.5 * ∂ₙ(u) => at((x, s) -> s * x[2], t),
        :ymin => u => at((x, s) -> exp(-s) * x[1], t),
        :ymax => ∂ₙₙ(u) => 1.0)
end

@testset "2D Neumann, Robin, ∂ₙₙ, Dirichlet" begin
    W = gridspace(ghost_mesh(CpuSerial(), (6, 7)); ghosts = (:xmin, :xmax, :ymax))
    C, Cfix = spaces(W, rels2d)
    @test C.td isa Bramble._TimeData
    @test relerr(Bramble.closure(C)[2], Bramble.closure(Cfix(0.0))[2]) <= 1e-13
    matches_frozen(C, Cfix)
end

@testset "1D Neumann and Robin in time" begin
    W = gridspace(ghost_mesh(CpuSerial(), (9,)); ghosts = :boundary)
    rels(t) = u -> (:xmin => ∂ₙ(u) => at((x, s) -> cos(3s) + x[1], t),
        :xmax => 2.0 * u + ∂ₙ(u) => at((x, s) -> s^2 - x[1], t))
    matches_frozen(spaces(W, rels)...)
    # One datum in time, the other static, and a Dirichlet relation in time.
    W1 = gridspace(ghost_mesh(CpuSerial(), (9,)); ghosts = (:xmax,))
    rels1(t) = u -> (:xmin => u => at((x, s) -> 1 + s, t), :xmax => ∂ₙ(u) => x -> x[1])
    matches_frozen(spaces(W1, rels1)...)
end

@testset "Conormal corner blocks in time" begin
    W = gridspace(ghost_mesh(CpuSerial(), (7, 6)); ghosts = :boundary)
    rels(t) = u -> (:xmin => ∂ₙ(u; K) => at((x, s) -> sin(s) * x[2], t),
        :xmax => u + ∂ₙ(u; K) => at((x, s) -> cos(s) + x[2], t),
        :ymin => ∂ₙ(u; K) => 0.5,
        :ymax => 2.0 * u + ∂ₙ(u; K) => at((x, s) -> s * x[1]^2, t))
    C, Cfix = spaces(W, rels)
    matches_frozen(C, Cfix; tol = 1e-12)
end

@testset "Dirichlet in time at a shared corner" begin
    # Two Dirichlet facets meet at (0, 0), where both data are `s`. The ghost relations on
    # xmax and ymax read the fixed nodes of xmin and ymin, and xmin carries ghosts that
    # its own ghost relation solves for, reading the fixed nodes it is beside.
    W = gridspace(ghost_mesh(CpuSerial(), (6, 7)); ghosts = (:xmin, :xmax, :ymax))
    rels(t) = u -> (:xmin => u => at((x, s) -> s + x[2]^2, t),
        :ymin => u => at((x, s) -> s + x[1], t),
        :xmin => ∂ₙ(u) => at((x, s) -> s * x[2], t),
        :xmax => ∂ₙ(u) => at((x, s) -> exp(s) * x[2], t),
        :ymax => u + ∂ₙ(u) => 1.0)
    matches_frozen(spaces(W, rels)...)
end

@testset "Static data keep the static path" begin
    W = gridspace(ghost_mesh(CpuSerial(), (6, 7)); ghosts = (:xmin, :xmax, :ymax))
    C = constrained(W, rels2d(0.4))
    @test C.td === nothing
    u = collect(1.0:Bramble.ndofs(C))
    U = parent(Bramble.lift(C, u))
    @test parent(Bramble.lift(C, u, 0.0)) == U
    @test parent(Bramble.lift(C, u, 2.5)) == U
    d = similar(Bramble.closure(C)[2])
    @test Bramble._lift_affine!(d, C, 7.0) == Bramble.closure(C)[2]
    @test parent(Bramble.lift!(Bramble.lift(C, u), C, u, 1.0)) == U
end

@testset "Without a time, the load refuses" begin
    W = gridspace(ghost_mesh(CpuSerial(), (6, 7)); ghosts = (:xmin, :xmax, :ymax))
    C, Cfix = spaces(W, rels2d)
    u = ones(Bramble.ndofs(C))
    # The message names the call made, `lift(Cₕ, u, t)` and `semidiscretize`.
    names_fix(e, call) = e isa ArgumentError &&
                         startswith(sprint(showerror, e), "ArgumentError: $call has no") &&
                         occursin("lift(Cₕ, u, t)", sprint(showerror, e)) &&
                         occursin("semidiscretize", sprint(showerror, e))
    refused(f, call) =
        try
            f()
            false
        catch e
            names_fix(e, call)
        end
    @test refused(() -> Bramble.lift(C), "lift(Wₕ)")
    @test refused(() -> Bramble.lift(C, u), "lift(Wₕ, u)")
    @test refused(() -> Bramble.lift!(Bramble.lift(C, u, 0.0), C, u), "lift!(uₕ, Wₕ, u)")
    a = form(C, C, (w, v) -> inner₊(∇ₕ(w), ∇ₕ(v)))
    l = form(C, v -> innerₕ(x -> 1.0, v))
    @test refused(() -> assemble(a, l), "assemble(a, l)")
    A0 = assemble(a)
    F0 = zeros(Bramble.ndofs(C))
    @test refused(() -> assemble!(A0, F0, a, l), "assemble!(A, F, a, l)")
    @test refused(() -> assemble(l), "assemble(l)")
    @test refused(() -> assemble!(zeros(Bramble.ndofs(C)), l), "assemble!(b, l)")
    @test_throws ArgumentError constrained(C, rels2d(nothing))
    # `form` and `assemble(a)` read `E` only: the frozen space's matrix, bit for bit.
    Ct = Cfix(0.8)
    @test A0 == assemble(form(Ct, Ct, (w, v) -> inner₊(∇ₕ(w), ∇ₕ(v))))
    @test assemble!(copy(A0), a) == A0
end

@testset "The load's lift reads every d(t) column" begin
    # Data that vanish at t = 0 leave `d(0) = 0` on their facets; the form's `A_pad d` map
    # must still cover every column `d(t)` can touch, or the load at a later `t` drops it.
    W = gridspace(ghost_mesh(CpuSerial(), (6, 7)); ghosts = (:xmin, :xmax, :ymax))
    rels(t) = u -> (:xmin => ∂ₙ(u) => at((x, s) -> s * (1 + x[2]), t),
        :xmax => u + ∂ₙ(u) => at((x, s) -> sin(s), t),
        :ymin => u => at((x, s) -> s * x[1], t),
        :ymax => ∂ₙ(u) => 0.0)
    C, Cfix = spaces(W, rels)
    @test iszero(Bramble.closure(C)[2])
    f(w, v) = innerₕ(-Δₕ(w), v)
    g(v) = innerₕ(x -> 1.0 + x[1], v)
    a, l = form(C, C, f), form(C, g)
    A = assemble(a)
    for t in (0.5, 1.3)
        Ct = Cfix(t)
        At, Ft = assemble(form(Ct, Ct, f), form(Ct, g))
        @test A == At
        Bramble._lift_affine!(C.d, C, t)          # what a load at `t` lifts with
        F = zeros(Bramble.ndofs(C))
        Bramble._assemble_ghost_load!(F, a, l)
        @test relerr(F, Ft) <= 1e-12
    end
end

# A datum closing over a number, as a frequency read from a parameter.
wave(ω) = (x, s) -> cos(ω * s) + x[2]

@testset "Threaded d(t) equals serial" begin
    rels(t) = u -> (:xmin => ∂ₙ(u; K) => at((x, s) -> sin(s) * x[2], t),
        :xmax => u + ∂ₙ(u; K) => at(wave(2.0), t),
        :ymin => ∂ₙ(u; K) => 0.5,
        :ymax => u => at((x, s) -> s * x[1], t),
        :ymax => ∂ₙ(u; K) => at((x, s) -> s - x[1], t))
    C0 = constrained(gridspace(ghost_mesh(CpuSerial(), (9, 11)); ghosts = :boundary),
        rels(nothing))
    C1 = constrained(gridspace(ghost_mesh(CpuThreaded(), (9, 11)); ghosts = :boundary),
        rels(nothing))
    @test Bramble.execution_policy(Bramble.padded_space(C1)) == CpuThreaded()
    u = collect(range(0.0, 1.0; length = Bramble.ndofs(C0)))
    d0, d1 = similar(C0.d), similar(C1.d)
    for t in TIMES
        @test Bramble._lift_affine!(d1, C1, t) == Bramble._lift_affine!(d0, C0, t)
        @test parent(Bramble.lift(C1, u, t)) == parent(Bramble.lift(C0, u, t))
    end
end

@testset "Float32 space with time data" begin
    rels(t) = u -> (:xmin => ∂ₙ(u) => at((x, s) -> sin(s) + x[2] * cos(s), t),
        :xmax => u + 0.5f0 * ∂ₙ(u) => at((x, s) -> s * x[2], t),
        :ymin => u => at((x, s) -> exp(-s) * x[1], t),
        :ymax => ∂ₙₙ(u) => 1.0f0)
    W = gridspace(ghost_mesh(CpuSerial(), (6, 7); T = Float32);
        ghosts = (:xmin, :xmax, :ymax))
    C, Cfix = spaces(W, rels)
    @test eltype(C.d) == Float32 && eltype(C.td.g) == Float32
    matches_frozen(C, t -> Cfix(Float32(t)); tol = 1.0f-5, times = Float32.(TIMES))
    d = similar(C.d)
    let d = d, C = C
        @test alloc(() -> Bramble._lift_affine!(d, C, 0.4f0)) == 0
    end
end

@testset "_lift_affine! infers and allocates 0 B" begin
    W = gridspace(ghost_mesh(CpuSerial(), (6, 7)); ghosts = (:xmin, :xmax, :ymax))
    C = constrained(W, rels2d(nothing))
    d = similar(C.d)
    @test (@inferred Bramble._lift_affine!(d, C, 0.4)) === d
    u = ones(Bramble.ndofs(C))
    @test (@inferred Bramble.lift(C, u, 0.4)) isa Bramble.VectorElement
    U = Bramble.lift(C, u, 0.0)
    let d = d, C = C, U = U, u = u
        @test alloc(() -> Bramble._lift_affine!(d, C, 0.4)) == 0
        @test alloc(() -> Bramble.lift!(U, C, u, 0.4)) == 0
    end
    @test_throws DimensionMismatch Bramble._lift_affine!(zeros(3), C, 0.4)
end

@testset "A Dual time gives ∂d/∂t" begin
    # `d` is linear in the data, so `∂d/∂t` is the `d` of the space whose data are `∂ₜg`
    # and whose static data are 0: the frozen reference of the derivative.
    W = gridspace(ghost_mesh(CpuSerial(), (6, 7)); ghosts = (:xmin, :xmax, :ymax))
    rels(g, h, k) = u -> (:xmin => ∂ₙ(u) => g, :xmax => u + ∂ₙ(u) => h,
        :ymin => u => k, :ymax => ∂ₙ(u) => 0.0)
    C = constrained(W, rels((x, t) -> sin(t) * (1 + x[2]), (x, t) -> t^2 * x[2],
        (x, t) -> exp(-t) * x[1]))
    t0 = 0.7
    ref = Bramble.closure(constrained(W, rels(x -> cos(t0) * (1 + x[2]),
        x -> 2t0 * x[2], x -> -exp(-t0) * x[1])))[2]
    u = collect(range(-1.0, 1.0; length = Bramble.ndofs(C)))
    dlift = ForwardDiff.derivative(t -> parent(Bramble.lift(C, u, t)), t0)
    @test relerr(dlift, ref) <= 1e-12
    daff = ForwardDiff.derivative(t -> Bramble._lift_affine!(similar(C.d, typeof(t)), C, t),
        t0)
    @test relerr(daff, ref) <= 1e-12
    # The stored buffer still serves a plain time afterwards.
    @test relerr(Bramble._lift_affine!(similar(C.d), C, t0),
        parent(Bramble.lift(C, zeros(Bramble.ndofs(C)), t0))) == 0
end

@testset "Concurrent lifts at two times" begin
    W = gridspace(ghost_mesh(CpuSerial(), (30, 33)); ghosts = (:xmin, :xmax, :ymax))
    C = constrained(W, u -> (:xmin => ∂ₙ(u) => (x, t) -> t * (1 + x[2]),
        :xmax => ∂ₙ(u) => 0.0, :ymax => ∂ₙ(u) => (x, t) -> t * x[1]))
    u = zeros(Bramble.ndofs(C))
    refs = map(t -> parent(Bramble.lift(C, u, t)), (1.0, 2.0))
    @test refs[1] != refs[2]
    bad = Threads.Atomic{Int}(0)
    @sync for k in 1:2
        Threads.@spawn begin
            t = Float64(k)
            U = Bramble.lift(C, u, t)
            g = Bramble._affine_buffer(C)
            for _ in 1:300
                parent(Bramble.lift(C, u, t)) == refs[k] || Threads.atomic_add!(bad, 1)
                Bramble.lift!(U, C, u, t, g)
                parent(U) == refs[k] || Threads.atomic_add!(bad, 1)
            end
        end
    end
    @test bad[] == 0
    # A caller's buffer of the wrong length is refused.
    @test_throws DimensionMismatch Bramble._lift_affine!(similar(C.d), C, 1.0, zeros(2))
end

@testset "Data not finite at t = 0" begin
    W = gridspace(ghost_mesh(CpuSerial(), (6, 7)); ghosts = (:xmin, :xmax, :ymax))
    # The same datum on two facets meeting at a corner agrees there, `Inf` at `t = 0` too.
    rels(t) = u -> (:xmin => u => at((x, s) -> 1 / s, t),
        :ymin => u => at((x, s) -> 1 / s, t),
        :xmin => ∂ₙ(u) => 0.0, :xmax => ∂ₙ(u) => 0.0, :ymax => ∂ₙ(u) => 0.0)
    C, Cfix = spaces(W, rels)
    u = ones(Bramble.ndofs(C))
    U, Ut = parent(Bramble.lift(C, u, 1.5)), parent(Bramble.lift(Cfix(1.5), u))
    @test relerr(U, Ut) <= 1e-13
    # A datum not finite where it meets a finite one is named, not called a disagreement.
    msg = try
        constrained(W,
            u -> (:xmin => u => (x, s) -> 1 / s, :ymin => u => (x, s) -> s,
                :xmin => ∂ₙ(u) => 0.0, :xmax => ∂ₙ(u) => 0.0, :ymax => ∂ₙ(u) => 0.0))
        ""
    catch e
        e isa ArgumentError ? sprint(showerror, e) : "other"
    end
    @test occursin("on :xmin is not finite at t = 0", msg)
    # A datum of the wrong kind names the forms data take, `(x, t)` among them.
    msg = try
        constrained(W, u -> (:xmin => ∂ₙ(u) => "g",))
        ""
    catch e
        e isa ArgumentError ? sprint(showerror, e) : "other"
    end
    @test occursin("(x, t) ->", msg)
end

# 3D, nightly.
WITH_3D_TESTS && @testset "3D Neumann, Robin, Dirichlet" begin
    W = gridspace(ghost_mesh(CpuSerial(), (4, 5, 4)); ghosts = (:xmin, :xmax, :ymax, :zmax))
    rels(t) = u -> (:xmin => ∂ₙ(u) => at((x, s) -> sin(s) + x[2] * x[3], t),
        :xmax => u + ∂ₙ(u) => at((x, s) -> s * x[3], t),
        :ymin => u => at((x, s) -> s + x[1], t),
        :zmin => u => at((x, s) -> s + x[1], t),
        :ymax => ∂ₙ(u) => 0.25,
        :zmax => ∂ₙₙ(u) => at((x, s) -> cos(s) * x[2], t))
    matches_frozen(spaces(W, rels)...; tol = 1e-12)
end

end
