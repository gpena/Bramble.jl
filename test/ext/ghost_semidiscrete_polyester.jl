# test/ext/ghost_semidiscrete_polyester.jl: the residual of a semidiscretization on a
# constrained ghost space with time-dependent data, on a `CpuPolyester` mesh, equals the
# serial one and allocates nothing warm (gpena/Bramble.jl#776, the Conventions' policy rule).
#
# Gated like every other ext/*.jl file. Standalone:
#
#   julia --project=test -t 2 -e 'using Bramble, Test; include("test/TestUtils.jl");
#     include("test/ext/ghost_semidiscrete_polyester.jl")'
module GhostSemidiscretePolyesterTests

using Test
using Bramble
using Bramble: CpuPolyester, CpuSerial
using LinearAlgebra: norm
using SparseArrays: SparseMatrixCSC
using Polyester

alloc(f::F) where {F} = (f(); f(); @allocated f())

function smesh(pol, ns::NTuple{2, Int}; T = Float64)
    Bramble._seed_mesh1d_rng!(11)
    be = backend(vector_type = Vector{T}, matrix_type = SparseMatrixCSC{T, Int},
        policy = pol)
    Ω = mesh(domain(interval(T(0), T(1)) × interval(T(0), T(2))), ns, (false, false);
        backend = be)
    Bramble._unseed_mesh1d_rng!()
    return Ω
end

function rel(u)
    (:xmin => ∂ₙ(u) => 0.0, :xmax => u + ∂ₙ(u) => (x, t) -> t * x[2],
        :ymin => u => (x, t) -> sin(t) * x[1], :ymax => ∂ₙ(u) => 1.0)
end

# The semidiscretization under `pol`; with `c`, an operator refilled at each step.
function ghost_sd(pol, ns; T = Float64, c = nothing)
    C = constrained(gridspace(smesh(pol, ns; T); ghosts = (:xmin, :xmax, :ymax)), rel)
    l = form(C, v -> innerₕ(x -> 1.0 + x[1], v))
    c === nothing && return C, semidiscretize(form(C, C, (u, v) -> innerₕ(-Δₕ(u), v)), l)
    a = form(C, C, (u, v) -> innerₕ(-Δₕ(u), v) + innerₕ(c * u, v))
    return C, semidiscretize(a, l; (update_coefficients!) = t -> (c[] = t),
        reassemble = true)
end

@testset "Polyester ghost residual (#776)" begin
    Threads.nthreads() >= 2 ||
        @warn "one thread: the Polyester split is a single inline chunk"
    for ns in ((6, 7), (30, 33)), reassemble in (false, true)

        @testset "$(ns), reassemble = $(reassemble)" begin
            sds = map((CpuSerial(), CpuPolyester())) do pol
                ghost_sd(pol, ns; c = reassemble ? Ref(0.0) : nothing)
            end
            (C, sds_), (_, sdp) = sds
            u = collect(range(-1, 2; length = Bramble.ndofs(C)))
            dus, dup = similar(u), similar(u)
            for t in (0.0, 0.4, 1.3)
                sds_(dus, u, nothing, t)
                sdp(dup, u, nothing, t)
                @test norm(dup - dus, Inf) <= 1e-13 * norm(dus, Inf)
            end
            @test alloc(() -> sdp(dup, u, nothing, 0.3)) == 0
        end
    end

    @testset "Float32" begin
        C, sdp = ghost_sd(CpuPolyester(), (6, 7); T = Float32)
        _, sds = ghost_sd(CpuSerial(), (6, 7); T = Float32)
        u = Float32.(collect(range(-1, 2; length = Bramble.ndofs(C))))
        dus, dup = similar(u), similar(u)
        sds(dus, u, nothing, 0.4f0)
        sdp(dup, u, nothing, 0.4f0)
        @test norm(dup - dus, Inf) <= 1.0f-5 * norm(dus, Inf)
        @test alloc(() -> sdp(dup, u, nothing, 0.3f0)) == 0
    end
end

end
