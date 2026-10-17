module PoliciesTests

# The reduction `E_vᵀ A_pad E_u`, the load `E_vᵀ(F_pad − A_pad d_u)` and `lift`'s `E u + d`
# run in the bands of the mesh's policy (gpena/Bramble.jl#785): each band owns whole output
# entries and adds their terms in a fixed order, so `CpuThreaded` gives the serial values
# bitwise and a repeated run gives the same bits (a racing scatter would give a
# nondeterministic sum). The unit group runs this file at 4 threads; `CpuPolyester` is in
# test/ext/ghost_polyester.jl, where Polyester is loaded.

using Test
using Bramble
import Bramble
using Bramble: CpuSerial, CpuThreaded
using LinearAlgebra: norm
using SparseArrays: nonzeros

using ..TestUtils: WITH_3D_TESTS, ghost_mesh

const K = [2.0 1.0; 1.0 2.0]

laplacian(u, v) = innerₕ(-Δₕ(u), v)
source(v) = innerₕ(x -> 1.0 + x[1], v)

# Matrix, load, refilled matrix, solution and its lift on `space(Ω)` under `pol`.
function solve_by(pol, ns, space; f = laplacian, kwargs...)
    C = space(ghost_mesh(pol, ns))
    a, l = form(C, C, f), form(C, source)
    A, F = assemble(a, l; kwargs...)
    A2 = copy(A)
    fill!(nonzeros(A2), 0.0)
    assemble!(A2, a; (k => v for (k, v) in kwargs if k === :dirichlet)...)
    u = pde_solve(A, F)
    return (A = A, F = F, A2 = A2, u = u, U = parent(Bramble.lift(C, u)), C = C, a = a)
end

# Every result of `solve_by` under `CpuThreaded` equals the serial one bit for bit.
function same_as_serial(ns, space; kwargs...)
    r0 = solve_by(CpuSerial(), ns, space; kwargs...)
    r1 = solve_by(CpuThreaded(), ns, space; kwargs...)
    @test any(!iszero, last(Bramble.closure(r0.C)))       # the load reads `A_pad d`
    for k in (:A, :F, :A2, :u, :U)
        @test getfield(r1, k) == getfield(r0, k)
    end
    # The serial values themselves are the closure's: `E_vᵀ A_pad E_u` and `E u + d`.
    E, d = Bramble.closure(r0.C)
    isempty(kwargs) && @test norm(r0.A - E' * r0.a.A_pad * E, Inf) <= 1e-12 * norm(r0.A, Inf)
    @test norm(r0.U - (E * r0.u + d), Inf) <= 1e-12 * norm(r0.U, Inf)
    return r0
end

# A vector that marks, per thread, every task that reads or writes it: a band function that
# runs on more than one thread marks more than one, a path left serial marks one.
struct ThreadMarks{T, V <: AbstractVector{T}} <: AbstractVector{T}
    data::V
    seen::Vector{Bool}
end
ThreadMarks(v::AbstractVector) = ThreadMarks(v, fill(false, Threads.maxthreadid()))
Base.size(r::ThreadMarks) = size(r.data)
Base.IndexStyle(::Type{<:ThreadMarks}) = IndexLinear()
@inline function Base.getindex(r::ThreadMarks, i::Int)
    @inbounds r.seen[Threads.threadid()] = true
    return r.data[i]
end
@inline function Base.setindex!(r::ThreadMarks, x, i::Int)
    @inbounds r.seen[Threads.threadid()] = true
    return setindex!(r.data, x, i)
end
nthreads_seen(r::ThreadMarks) = count(r.seen)

# The form `a` with its map's arrays marked, so that each path's band function reads one.
function marked_form(a)
    marks = (k, v) -> v isa AbstractVector && k !== :A_pad ? ThreadMarks(v) : v
    fields = map(k -> marks(k, getfield(a, k)), fieldnames(typeof(a)))
    return Bramble.GhostBilinearForm(fields...)
end

# The threads that ran the reduction, the two passes of the load and `lift` under `pol`,
# their values checked against the unmarked paths'.
function band_threads(pol)
    C = constrained(gridspace(ghost_mesh(pol, (9, 11)); ghosts = :boundary), robin_conormal)
    a, l = form(C, C, laplacian), form(C, source)
    A, F = assemble(a, l)
    am = marked_form(a)
    Am = copy(A)
    fill!(nonzeros(Am), 0.0)
    Bramble._assemble_ghost!(Am, am)
    Fm = ThreadMarks(zeros(length(F)))
    Bramble._assemble_ghost_load!(Fm, am, l)
    u = A \ F
    um = ThreadMarks(copy(u))
    U = parent(Bramble.lift(C, um))
    @test Am == A
    @test Fm.data == F
    @test U == parent(Bramble.lift(C, u))
    return (policy = Bramble._ghost_policy(C), reduction = nthreads_seen(am.coef),
        lift_load = nthreads_seen(am.liftcol), load = nthreads_seen(Fm),
        lift = nthreads_seen(um))
end

function robin_conormal(u)
    (:xmin => ∂ₙ(u; K) => 1.0, :xmax => u + ∂ₙ(u; K) => x -> x[2],
        :ymin => ∂ₙ(u; K) => 0.0, :ymax => 2.0 * u + ∂ₙ(u; K) => 1.0)
end
function named_dirichlet(u)
    (:xmin => ∂ₙ(u) => 1.0, :xmax => ∂ₙₙ(u) => x -> x[1],
        :ymin => u => x -> x[1], :ymax => ∂ₙ(u) => 1.0)
end

@testset "Ghost bands under CpuThreaded (#785)" begin
    @test Threads.nthreads() >= 2

    @testset "2D Robin with conormal corners" begin
        same_as_serial((9, 11),
            Ω -> constrained(gridspace(Ω; ghosts = :boundary), robin_conormal))
    end

    @testset "2D named facets, ∂ₙₙ, Dirichlet relation" begin
        same_as_serial((9, 11),
            Ω -> constrained(gridspace(Ω; ghosts = (:xmin, :xmax, :ymax)), named_dirichlet))
    end

    @testset "2D Dirichlet by row replacement" begin
        rel = u -> (:xmin => ∂ₙ(u) => 0.5, :xmax => u + ∂ₙ(u) => x -> x[2],
            :ymin => ∂ₙ(u) => 0.0, :ymax => ∂ₙ(u) => 1.0)
        space = Ω -> constrained(gridspace(Ω; ghosts = :boundary), rel)
        for symmetrize in (false, true)
            same_as_serial((9, 11), space; dirichlet = :xmin => (x -> x[2]), symmetrize)
        end
    end

    @testset "1D Robin and Dirichlet relations" begin
        rel = u -> (:xmin => u + ∂ₙ(u) => 2.0, :xmax => u => 1.0)
        same_as_serial((17,), Ω -> constrained(gridspace(Ω; ghosts = :xmin), rel))
    end

    # 3D: nightly (`WITH_3D_TESTS`).
    WITH_3D_TESTS && @testset "3D Robin and Dirichlet relations" begin
        rel = u -> (:xmin => ∂ₙ(u) => 1.0, :xmax => u + ∂ₙ(u) => x -> x[3],
            :ymin => ∂ₙ(u) => 0.0, :ymax => ∂ₙ(u) => 0.0, :zmin => u => x -> x[1],
            :zmax => ∂ₙ(u) => 1.0)
        ghosts = (:xmin, :xmax, :ymin, :ymax, :zmax)
        same_as_serial((5, 6, 4), Ω -> constrained(gridspace(Ω; ghosts), rel))
    end

    @testset "Load with d = 0 refills, CpuThreaded" begin
        rel = u -> map(F -> F => ∂ₙ(u; K) => 0.0, (:xmin, :xmax, :ymin, :ymax))
        b = map((CpuSerial(), CpuThreaded())) do pol
            C = constrained(gridspace(ghost_mesh(pol, (9, 11)); ghosts = :boundary), rel)
            l = form(C, source)
            b = assemble(l)
            fill!(b, 0.0)
            assemble!(b, l)
            E = first(Bramble.closure(C))
            @test norm(b - E' * l.F_pad, Inf) <= 1e-13 * norm(b, Inf)
            return b
        end
        @test norm(b[1], Inf) > 0
        @test b[2] == b[1]
    end

    @testset "Repeated threaded runs: the same bits" begin
        space = Ω -> constrained(gridspace(Ω; ghosts = :boundary), robin_conormal)
        r0 = solve_by(CpuThreaded(), (30, 32), space)
        for _ in 1:3
            r = solve_by(CpuThreaded(), (30, 32), space)
            @test r.A == r0.A
            @test r.F == r0.F
            @test r.A2 == r0.A2
            @test r.U == r0.U
        end
    end

    # A path that ignores the mesh's policy runs on the calling thread alone.
    @testset "Each path runs on several threads" begin
        r = band_threads(CpuSerial())
        @test r.policy isa CpuSerial
        @test (r.reduction, r.lift_load, r.load, r.lift) == (1, 1, 1, 1)
        r = band_threads(CpuThreaded())
        @test r.policy isa CpuThreaded
        @test r.reduction >= 2
        @test r.lift_load >= 2
        @test r.load >= 2
        @test r.lift >= 2
    end

    @testset "Every band count covers every entry once" begin
        C = constrained(gridspace(ghost_mesh(CpuSerial(), (9, 11)); ghosts = :boundary),
            robin_conormal)
        a, l = form(C, C, laplacian), form(C, source)
        A, F = assemble(a, l)
        E, d = Bramble.closure(C)
        u = collect(range(-1.0, 2.0; length = Bramble.ndofs(C)))
        U0 = parent(Bramble.lift(C, u))
        Et = C.Et
        for nb in (1, 2, 3, 7, length(d) + 3)
            nz = fill(NaN, length(nonzeros(A)))
            Bramble._serial_bands!(Bramble._reduce_band!, nb, nz, nonzeros(a.A_pad),
                a.dst, a.src, a.coef)
            @test nz == nonzeros(A)
            U = fill(NaN, length(d))
            Bramble._serial_bands!(Bramble._lift_band!, nb, U, d, Et.colptr, Et.rowval,
                Et.nzval, u)
            @test U == U0
            y = fill(NaN, Bramble.ndofs(C))
            Bramble._serial_bands!(Bramble._csc_columns_band!, nb, y, E.colptr, E.rowval,
                E.nzval, U0)
            @test norm(y - E' * U0, Inf) <= 1e-13 * norm(y, Inf)
        end
    end

    @testset "assemble_parallel! names assemble!" begin
        C = constrained(gridspace(ghost_mesh(CpuThreaded(), (6, 7)); ghosts = :boundary),
            robin_conormal)
        a, l = form(C, C, laplacian), form(C, source)
        A = assemble(a)
        @test_throws ArgumentError Bramble.assemble_parallel!(A, a)
        err = try
            Bramble.assemble_parallel!(A, a)
        catch e
            e
        end
        @test err isa ArgumentError && occursin("call assemble! instead", err.msg)
        @test_throws ArgumentError Bramble.assemble_parallel!(zeros(Bramble.ndofs(C)), l)
    end
end

end
