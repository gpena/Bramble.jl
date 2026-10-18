module AllocationsTests

# Every warm in-place call on the ghost path allocates nothing on `CpuSerial`
# (gpena/Bramble.jl#785): the matrix refill `assemble!(A, a)`, the pair refill
# `assemble!(A, F, a, l)` whose load reads `A_pad d ≠ 0`, `Bramble.lift!`, the load refill
# `assemble!(b, l)` with and without `dirichlet`, and `dirichlet_bc!` and `symmetrize!` on a
# ghost space. `CpuThreaded` spawns its tasks on every banded pass, so there each call is
# bounded by its padded or plain counterpart plus one task spawn per band pass of the ghost
# layer, and gives the same figure at two grid sizes. The last two testsets refill a form
# whose coefficient changed on a ghost space, and padded forms under the corner rule. The
# plain-and-ghost warm refill with `dirichlet` is in test/form/two_spaces.jl,
# and `CpuPolyester` is in test/ext/ghost_polyester.jl, where Polyester is loaded.

using Test
using Bramble
import Bramble
using Bramble: CpuSerial, CpuThreaded, D₋ₓ, D₋ᵧ, D̃ₓ, D̃ᵧ, D̽ₓ, D̽ᵧ, Dcₓ, Dcᵧ
using LinearAlgebra: norm
using SparseArrays: nonzeros

using ..TestUtils: WITH_3D_TESTS, ghost_mesh, padded

const K = [2.0 1.0; 1.0 2.0]

# The bytes of the third call of `f`, the first two compiling and warming it.
alloc(f::F) where {F} = (f(); f(); @allocated f())

laplacian(u, v) = innerₕ(-Δₕ(u), v)
complex_laplacian(u, v) = (1.0 + 0.5im) * innerₕ(-Δₕ(u), v)
source(v) = innerₕ(x -> 1 + x[1], v)          # of the mesh's element type
# Forms and loads whose element types widen each other on a Float32 mesh.
complex32_laplacian(u, v) = (1.0f0 + 0.5f0im) * innerₕ(-Δₕ(u), v)
wide_laplacian(u, v) = innerₕ(-2.0 * Δₕ(u), v)
source64(v) = innerₕ(x -> 1.0 + x[1], v)
complex32_source(v) = innerₕ(x -> ComplexF32(1 + x[1], 1), v)

function robin2(u)
    (:xmin => ∂ₙ(u) => 1.0, :xmax => u + ∂ₙ(u) => x -> x[2],
        :ymin => ∂ₙ(u) => 0.0, :ymax => 2.0 * u + ∂ₙ(u) => 1.0)
end
function conormal2(u)
    (:xmin => ∂ₙ(u; K) => 1.0, :xmax => u + ∂ₙ(u; K) => x -> x[2],
        :ymin => ∂ₙ(u; K) => 0.0, :ymax => 2.0 * u + ∂ₙ(u; K) => 1.0)
end
dirichlet2(u) = (:xmin => ∂ₙ(u) => 1.0, :xmax => ∂ₙₙ(u) => x -> x[1],
    :ymin => u => x -> x[1], :ymax => ∂ₙ(u) => 1.0)
robin1(u) = (:xmin => u + ∂ₙ(u) => 2.0, :xmax => ∂ₙ(u) => x -> x[1])
neumann0(u) = map(F -> F => ∂ₙ(u) => 0.0, (:xmin, :xmax, :ymin, :ymax))
function robin3(u)
    (:xmin => ∂ₙ(u) => 1.0, :xmax => u + ∂ₙ(u) => x -> x[3],
        :ymin => ∂ₙ(u) => 0.0, :ymax => ∂ₙ(u) => 0.0, :zmin => u => x -> x[1],
        :zmax => ∂ₙ(u) => 1.0)
end

# The pair refill, `lift!` and the matrix refill on `constrained(gridspace(Ω; ghosts),
# rel)`, warm, with the values each leaves behind and the references they must equal.
function warm_pair(pol, ns, ghosts, rel; f = laplacian, g = source, T = Float64)
    C = constrained(gridspace(ghost_mesh(pol, ns; T); ghosts), rel)
    a, l = form(C, C, f), form(C, g)
    A0, F0 = assemble(a, l)
    A, F = copy(A0), copy(F0)
    fill!(nonzeros(A), 0)
    fill!(F, 0)
    pair = alloc(() -> assemble!(A, F, a, l))
    u = A0 \ F0
    U0 = Bramble.lift(C, u)
    U = Bramble.lift(C, zero(u))
    lift = alloc(() -> Bramble.lift!(U, C, u))
    A2 = copy(A0)
    fill!(nonzeros(A2), 0)
    matrix = alloc(() -> assemble!(A2, a))
    return (; C, a, l, A0, F0, A, F, A2, u, U0, U, pair, lift, matrix)
end

# `F = E_vᵀ (F_pad − A_pad d)` and `U = E u + d` from the padded forms assembled on their
# own and the closure, sharing nothing with the banded passes.
function closure_reference(r, f, g)
    E, d = Bramble.closure(r.C)
    P = Bramble.padded_space(r.C)
    Ap = assemble(form(P, P, f))
    Fp = assemble(form(P, g))
    return transpose(E) * (Fp - Ap * d), E * r.u + d
end

relerr(X, Y) = norm(X - Y, Inf) / norm(Y, Inf)

# The serial pair refill equals `assemble(a, l)` bit for bit after three refills (a stale
# `F_pad − A_pad d` in the padded load would subtract the lift twice), the load and the lift
# equal the closure's own algebra, and every warm call allocates nothing.
function check_serial(ns, ghosts, rel; f = laplacian, g = source, T = Float64,
        tol = 1e-12)
    r = warm_pair(CpuSerial(), ns, ghosts, rel; f, g, T)
    @test any(!iszero, last(Bramble.closure(r.C)))     # the load reads `A_pad d`
    @test r.A == r.A0
    @test r.F == r.F0
    @test r.A2 == r.A0
    @test parent(r.U) == parent(r.U0)
    Fref, Uref = closure_reference(r, f, g)
    @test relerr(r.F, Fref) <= tol
    @test relerr(parent(r.U), Uref) <= tol
    @test r.pair == 0
    @test r.lift == 0
    @test r.matrix == 0
    return r
end

# A homogeneous Neumann space, so `d = 0` and its rows are the physical nodes: the load
# refills alone, and `dirichlet` replaces rows.
function warm_rows(pol, ns)
    C = constrained(gridspace(ghost_mesh(pol, ns); ghosts = :boundary), neumann0)
    a, l = form(C, C, laplacian), form(C, source)
    A, F = assemble(a, l)
    g = x -> x[2]
    A2, F2 = copy(A), copy(F)
    out = (
        matrix = alloc(() -> assemble!(A, a; dirichlet = :xmin)),
        load = alloc(() -> assemble!(F, l)),
        load_dirichlet = alloc(() -> assemble!(F, l; dirichlet = :xmin => g)),
        pair_symmetrize = alloc(() -> assemble!(A2, F2, a, l; dirichlet = :xmin => g,
            symmetrize = true)),
        dirichlet_bc = alloc(() -> dirichlet_bc!(A, C, :xmin)),
        symmetrize = alloc(() -> symmetrize!(A, F, C, :xmin)),
        marker_bc = alloc(() -> dirichlet_bc!(A, C, :boundary)),
        matrix_marker = alloc(() -> assemble!(A, a; dirichlet = :boundary)),
        load_marker = alloc(() -> assemble!(F, l; dirichlet = :boundary => g)))
    A3, F3 = assemble(a, l; dirichlet = :xmin => g, symmetrize = true)
    return out, (A2, F2), (A3, F3)
end

# A band function that does nothing, with as many arguments as the widest ghost band
# (`_lifted_load_band!`): its warm `_run_bands!` is one task spawn of the policy.
noop_band!(args...) = nothing

function threaded_figures(ns)
    pol = CpuThreaded()
    r = warm_pair(pol, ns, :boundary, robin2)
    x = zeros(3)
    spawn = alloc(() -> Bramble._run_bands!(pol, noop_band!, x, x, x, x, x, x, x, x, x, x,
        x))
    a, l = r.a, r.l
    padded_matrix = alloc(() -> assemble!(a.A_pad, a.padded))
    padded_load = alloc(() -> assemble!(l.F_pad, l.padded))
    rows, _, _ = warm_rows(pol, ns)
    Ω = ghost_mesh(pol, ns)
    W = gridspace(Ω)
    Ap, Fp = assemble(form(W, W, laplacian), form(W, source))
    plain_bc = alloc(() -> dirichlet_bc!(Ap, W, :boundary))
    plain_symmetrize = alloc(() -> symmetrize!(Ap, Fp, W, :boundary))
    return (; spawn, padded_matrix, padded_load, plain_bc, plain_symmetrize, r.pair, r.lift,
        r.matrix, rows.load, rows.dirichlet_bc, rows.symmetrize, rows.marker_bc,
        rows_matrix = rows.matrix)
end

@testset "Warm ghost calls allocate (#785)" begin
    @testset "2D Robin pair and lift!" begin
        check_serial((9, 11), :boundary, robin2)
        check_serial((30, 33), :boundary, robin2)
    end

    @testset "2D conormal corners" begin
        check_serial((9, 11), :boundary, conormal2)
    end

    @testset "2D named facets, Dirichlet u => g" begin
        check_serial((9, 11), (:xmin, :xmax, :ymax), dirichlet2)
    end

    @testset "1D Robin pair and lift!" begin
        check_serial((17,), :boundary, robin1)
    end

    @testset "Float32 pair and lift!" begin
        r = check_serial((9, 11), :boundary, robin2; T = Float32, tol = 1e-4)
        @test eltype(r.F) == Float32
        @test eltype(r.U) == Float32
    end

    # A complex form with a real load: `F_pad − A_pad d` is complex, and the load forms it
    # inside its single banded pass, with no buffer to allocate.
    @testset "Complex form, real load" begin
        r = check_serial((9, 11), :boundary, robin2; f = complex_laplacian)
        @test eltype(r.F) == ComplexF64
        @test eltype(r.l.F_pad) == Float64
        @test norm(imag(r.F)) > 1e-6 * norm(r.F)          # the complex lift reaches F
        @test eltype(r.U) == ComplexF64
    end

    # On a Float32 mesh, form and load types that widen each other: a ComplexF32 form
    # with a Float64 load, and a Float64-coefficient form with a ComplexF32 load, both
    # giving ComplexF64.
    @testset "Form and load widen each other" begin
        for (f, g) in ((complex32_laplacian, source64), (wide_laplacian, complex32_source))
            r = check_serial((9, 11), :boundary, robin2; f, g, T = Float32, tol = 1e-5)
            @test eltype(r.F) == ComplexF64
            @test eltype(r.F) != eltype(r.l.F_pad)
            @test eltype(r.F) != eltype(r.a.A_pad)
        end
    end

    WITH_3D_TESTS && @testset "3D Robin and Dirichlet relations" begin
        check_serial((5, 6, 4), (:xmin, :xmax, :ymin, :ymax, :zmax), robin3)
    end

    @testset "Rows, load, dirichlet_bc!, symmetrize!" begin
        for ns in ((9, 11), (30, 33))
            out, (A2, F2), (A3, F3) = warm_rows(CpuSerial(), ns)
            for (k, n) in pairs(out)
                @test n == 0
            end
            @test A2 == A3                  # the warm pair with rows replaced, symmetrized
            @test F2 == F3
        end
    end

    # `CpuThreaded` allocates its tasks: each ghost call is bounded by its padded or plain
    # counterpart plus one spawn per band pass it adds (the reduction 1, the load 1,
    # `lift!` 1), and nothing grows with the grid.
    @testset "CpuThreaded bound" begin
        f1 = threaded_figures((9, 11))
        f2 = threaded_figures((30, 33))
        @test f1 == f2
        s = f1.spawn
        @test f1.matrix <= f1.padded_matrix + s
        @test f1.rows_matrix <= f1.padded_matrix + s
        @test f1.load <= f1.padded_load + s
        @test f1.pair <= f1.padded_matrix + f1.padded_load + 2s
        @test f1.lift <= s
        @test f1.dirichlet_bc <= f1.plain_bc
        @test f1.marker_bc <= f1.plain_bc
        @test f1.symmetrize <= f1.plain_symmetrize
    end

    @testset "Refusals of the in-place calls" begin
        r = warm_pair(CpuSerial(), (9, 11), :boundary, robin2)
        C, a, l, A, F, u = r.C, r.a, r.l, r.A, r.F, r.u
        # another space, the same padded size: `U` must live on the padded space of `C`
        C2 = constrained(gridspace(ghost_mesh(CpuSerial(), (9, 11)); ghosts = :boundary),
            robin2)
        @test_throws ArgumentError Bramble.lift!(Bramble.lift(C2, u), C, u)
        @test_throws ArgumentError Bramble.lift!(r.U, C, u[1:(end - 1)])
        msg = try
            Bramble.lift!(r.U, C, u[1:(end - 1)])
        catch e
            sprint(showerror, e)
        end
        @test occursin("lift!(uₕ, Wₕ, u)", msg)
        @test_throws ArgumentError Bramble.lift!(r.U, C, complex.(u))
        @test_throws ArgumentError assemble!(A, zeros(Float32, length(F)), a, l)
        @test_throws DimensionMismatch assemble!(A, zeros(length(F) + 1), a, l)
        @test_throws ArgumentError assemble!(A[:, 1:(end - 1)], F, a, l)
        @test_throws ArgumentError assemble!(A, F, a, l; symmetrize = true)
        @test_throws ArgumentError assemble!(F, l)              # `d ≠ 0` needs the pair
        # the refusals left the system as it was
        @test A == r.A0
        @test F == r.F0
    end
end

@testset "Warm refill follows a coefficient" begin
    for ns in ((11,), (9, 10))
        Wg = gridspace(ghost_mesh(CpuSerial(), ns); ghosts = :boundary)
        β = Ref(1.0)
        a = form(Wg, Wg, (u, v) -> inner₊(β * ∇ₕ(u), ∇ₕ(v)) + innerₕ(Dcₓ(u), v))
        A = assemble(a)
        A0 = copy(A)
        β[] = 3.0
        assemble!(A, a)
        assemble!(A, a)
        @test (@allocated assemble!(A, a)) == 0
        @test A ≈ 3 * A0 - 2 * assemble(form(Wg, Wg, (u, v) -> innerₕ(Dcₓ(u), v)))
        # a repeated assemble gives an equal, separate matrix
        B = assemble(a)
        @test B == A && B !== A
        l = form(Wg, v -> innerₕ(x -> 1.0 + x[1], v))
        b = assemble(l)
        assemble!(b, l)
        @test (@allocated assemble!(b, l)) == 0
    end
end

@testset "Warm refill, padded corner rule" begin
    P = padded(ghost_mesh(CpuSerial(), (9, 9)))
    Kc = [1.0 0.5; 0.5 2.0]
    L(u) = Kc[1, 1] * D̃ₓ(D₋ₓ(u)) + 2 * Kc[1, 2] * Dcₓ(Dcᵧ(u)) + Kc[2, 2] * D̃ᵧ(D₋ᵧ(u))
    a = form(P, P, (u, v) -> innerₕ(L(u), v))
    A = assemble(a)
    @test alloc(() -> assemble!(A, a)) == 0
    b = form(P, P, (u, v) -> innerₕ(D̽ₓ(D̽ᵧ(u)), D̽ᵧ(D̽ₓ(v))))
    B = assemble(b)
    @test alloc(() -> assemble!(B, b)) == 0
    c = 1.0 .+ 0.5 .* sin.(1:Bramble.npoints(Bramble.mesh(P)))
    for X in (u -> Dcₓ(c * Dcᵧ(u)), u -> Dcₓ(Dcᵧ(u) + 2.0 * D₋ᵧ(u)))
        s = form(P, P, (u, v) -> innerₕ(X(u), v))
        S = assemble(s)
        @test alloc(() -> assemble!(S, s)) == 0
    end
end

end
