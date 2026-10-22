module FormTwoSpacesTests

# One test body per behaviour, run over a plain space with row Dirichlet and over the ghost
# space of the same problem (`u => g` and `∂ₙₙ(u) => h` on every Dirichlet facet), and
# asserting once that the two systems agree (gpena/Bramble.jl#786). With the ghost relation
# beside it the fixed node's row is gone and its ghost feeds nothing else, so the reduced
# system is the plain symmetric elimination. Mixed problems put `u => g` on some facets and
# Neumann or Robin on the others (`α * u + β * ∂ₙ(u) => r`, with `∂ₙ(u) => r` for Neumann). The meshes are seeded random ones, checked non-uniform,
# and the dense reference shares no code with Bramble.

using Test
using Bramble
import Bramble
using LinearAlgebra: Diagonal, issymmetric, norm
using SparseArrays: sparse, spdiagm
import ..TestUtils

isdefined(Main, :TwoSpacesCases) ||
    Base.include(Main, joinpath(@__DIR__, "TwoSpacesCases.jl"))
using Main.TwoSpacesCases: PROBLEMS, mesh_axes, node_solution, problem_data, random_mesh

const tol = 1e-11

relerr(A, B) = norm(A - B) / max(norm(B), eps())

function build(prob)
    Ω = random_mesh(prob.D, prob.n0, 0)
    return Ω, TestUtils.plain_and_ghost_spaces(Ω, problem_data(prob.D; mixed = prob.mixed))
end

# --- Behaviours of one case --------------------------------------------------------------- #

# The system on the free nodes: the plain one by `assemble` with `dirichlet` and
# `symmetrize`, then cut to the free nodes; the ghost one as it is.
function reduced_system(c)
    if c.route === :rows
        A, F = assemble(c.a, c.l; dirichlet = c.pairs, symmetrize = true)
        return A[c.free, c.free], F[c.free]
    end
    return assemble(c.a, c.l)
end

# `dirichlet_bc!` then `symmetrize!` by hand, the data written into the load between them.
function applied_by_hand(c)
    A, F = assemble(c.a, c.l)
    sp = c.space isa Bramble.ScalarGridSpace ? Bramble.mesh(c.space) : c.space
    dirichlet_bc!(A, c.space, c.labels...)
    F[c.fixed] .= c.gfixed
    rows_only = copy(A)
    symmetrize!(A, F, sp, c.labels...)
    return rows_only, A, F
end

# The plain forms refill a matrix and a load one at a time; the ghost ones, whose load can
# read the matrix through the closure's affine part, refill the pair.
function initial_system(c)
    c.space isa Bramble.ScalarGridSpace &&
        return assemble(c.a; dirichlet = c.labels[1]), assemble(c.l; dirichlet = c.pairs)
    c.route === :rows && return assemble(c.a, c.l; dirichlet = c.pairs, symmetrize = true)
    return assemble(c.a, c.l)
end

function refill!(c, A, F)
    if c.space isa Bramble.ScalarGridSpace
        assemble!(A, c.a; dirichlet = c.labels[1])
        assemble!(F, c.l; dirichlet = c.pairs)
    elseif c.route === :rows
        assemble!(A, F, c.a, c.l; dirichlet = c.pairs, symmetrize = true)
    else
        assemble!(A, F, c.a, c.l)
    end
    return nothing
end

# Warm refill in a function, so only the call is measured.
function warm_bytes(c)
    A, F = initial_system(c)
    refill!(c, A, F)
    A0, F0 = copy(A), copy(F)
    bytes = @allocated refill!(c, A, F)
    return bytes, A == A0 && F == F0
end

# The 1D stiffness and lumped weights of the points `x`, the reference for the pure problems.
function stiffness_weights(x)
    h = diff(x)
    n = length(x)
    K = zeros(n, n)
    for i in 1:(n - 1)
        K[i:(i + 1), i:(i + 1)] .+= [1 -1; -1 1] ./ h[i]
    end
    w = [h[1] / 2; (h[1:(end - 1)] .+ h[2:end]) ./ 2; h[end] / 2]
    return K, Diagonal(w)
end

# `-Δ` of the pure problem on the free nodes by dense Kronecker sums, and its load with the
# fixed nodes lifted: nothing here calls Bramble's operators.
function dense_reference(Ω, c, data)
    axs = mesh_axes(Ω)
    KW = [stiffness_weights(collect(x)) for x in axs]
    Afull = length(axs) == 1 ? KW[1][1] : kron(KW[2][2], KW[1][1]) + kron(KW[2][1], KW[1][2])
    pts = vec([ntuple(d -> axs[d][I[d]], length(axs)) for I in CartesianIndices(c.dims)])
    w = length(axs) == 1 ? diag_of(KW[1][2]) : vec([a * b for a in diag_of(KW[1][2]),
    b in diag_of(KW[2][2])])
    Afull = Matrix(Afull)
    F = w .* data.f.(pts) .- Afull[:, c.fixed] * c.gfixed
    return Afull[c.free, c.free], F[c.free]
end

diag_of(D) = D.diag

@testset "The meshes are non-uniform" begin
    for prob in PROBLEMS, level in 0:1

        for x in mesh_axes(random_mesh(prob.D, prob.n0, level))
            h = diff(collect(x))
            @test maximum(h) / minimum(h) > 1.3
        end
    end
end

@testset "Assembly: $(prob.name)" for prob in PROBLEMS
    Ω, cases = build(prob)
    systems = map(reduced_system, cases)
    for (c, (A, F)) in zip(cases, systems)
        @test size(A, 1) == length(c.free) == length(F)
        @test norm(A - A') <= 1e-12 * norm(A)
    end
    @test relerr(systems[1][1], systems[2][1]) <= tol
    @test relerr(systems[1][2], systems[2][2]) <= tol
    if !prob.mixed
        A0, F0 = dense_reference(Ω, cases[1], problem_data(prob.D; mixed = false))
        for (A, F) in systems
            @test relerr(Matrix(A), A0) <= tol
            @test relerr(F, F0) <= tol
        end
    end
end

@testset "dirichlet_bc! and symmetrize!: $(prob.name)" for prob in PROBLEMS
    _, cases = build(prob)
    plain, ghost = cases
    rows_only, A, F = applied_by_hand(plain)
    A1, F1 = assemble(plain.a, plain.l; dirichlet = plain.pairs, symmetrize = true)
    @test A == A1
    @test F == F1
    # rows replaced: not symmetric until `symmetrize!` eliminates the columns
    @test !issymmetric(rows_only)
    @test norm(A - A') <= 1e-12 * norm(A)
    for k in eachindex(plain.fixed)        # the Dirichlet rows are identity rows
        i = plain.fixed[k]
        @test A[i, i] == 1 && count(!iszero, A[i, :]) == 1
        @test F[i] == plain.gfixed[k]
    end
    # the same system, from `symmetrize!` and from the ghost relations
    @test relerr(A[plain.free, plain.free], reduced_system(ghost)[1]) <= tol
    @test relerr(F[plain.free], reduced_system(ghost)[2]) <= tol
    # the ghost space leaves no Dirichlet node among its unknowns
    @test Bramble.ndofs(ghost.space) == length(ghost.free)
end

@testset "Warm refill is 0 B: $(prob.name)" for prob in PROBLEMS
    _, cases = build(prob)
    for c in cases
        bytes, same = warm_bytes(c)
        @test bytes == 0
        @test same
    end
end

@testset "pde_solve: $(prob.name)" for prob in PROBLEMS
    Ω, cases = build(prob)
    data = problem_data(prob.D; mixed = prob.mixed)
    sols = map(node_solution, cases)
    @test relerr(sols[1], sols[2]) <= 1e-9
    for (c, u) in zip(cases, sols)
        A, F = reduced_system(c)
        @test relerr(Matrix(A) \ F, u[c.free]) <= 1e-9              # a dense solve
        @test relerr(u[c.fixed], c.gfixed) <= 1e-13                 # the data sit at the facets
    end
end

end # module FormTwoSpacesTests
