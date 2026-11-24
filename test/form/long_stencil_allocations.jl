module FormLongStencilAllocationsTests

# A stencil of more than `_INLINED_ENTRIES` (32) entries used to leave the unrolled paths:
# `(left..., right...)` over an operand that long runs generically, and `map` over a tuple
# that long collects into a heap vector (gpena/Bramble.jl#853). A warm `assemble!` of a deep
# `Δₕ` allocated up to hundreds of MB per call and spent nearly all of its time concatenating
# stencils. The forms below put a stencil past 32 entries on each of the two paths, and the
# matrix is compared with the runtime composition so a fast path that drops a term fails too.

using Test
using Bramble
using Random
using LinearAlgebra: dot
using ..TestUtils: @test_allocs, WITH_3D_TESTS

function graded_mesh(dims, n)
    Random.seed!(774)
    sets = (interval(0.0, 1.0), interval(0.0, 1.0) × interval(0.0, 1.0),
        box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    return mesh(domain(sets[dims]), n, ntuple(_ -> false, dims))
end

@testset "Δₕ³ in 2D: 216 entries (#853)" begin
    Wₕ = gridspace(graded_mesh(2, (9, 8)))
    a = form(Wₕ, Wₕ, (u, v) -> innerₕ(Δₕ(Δₕ(Δₕ(u))), v))
    A = assemble(a)
    @test_allocs Bramble.assemble!(A, a)
    u = element(Wₕ, rand(ndofs(Wₕ)))
    w = element(Wₕ, rand(ndofs(Wₕ)))
    @test dot(parent(w), A*parent(u))≈innerₕ(Δₕ(Δₕ(Δₕ(u))), w) rtol=1e-10 atol=1e-12
end

@testset "Δₕ² on both sides in 2D (#853)" begin
    Wₕ = gridspace(graded_mesh(2, (11, 10)))
    a = form(Wₕ, Wₕ, (u, v) -> innerₕ(Δₕ(Δₕ(u)), Δₕ(Δₕ(v))))
    A = assemble(a)
    @test_allocs Bramble.assemble!(A, a)
    u = element(Wₕ, rand(ndofs(Wₕ)))
    w = element(Wₕ, rand(ndofs(Wₕ)))
    @test dot(parent(w), A*parent(u))≈innerₕ(Δₕ(Δₕ(u)), Δₕ(Δₕ(w))) rtol=1e-10 atol=1e-12
end

@testset "linear form of 40 products in 2D (#853)" begin
    Wₕ = gridspace(graded_mesh(2, (9, 11)))
    f = x -> 1.0 + x[1]
    l = form(Wₕ, v -> foldl(+, ntuple(k -> (k + 0.5) * innerₕ(f, v), Val(40))))
    F = assemble(l)
    @test_allocs Bramble.assemble!(F, l)
    @test F≈sum(k -> k + 0.5, 1:40) .* assemble(form(Wₕ, v -> innerₕ(f, v))) rtol=1e-12
end

# Restricted products are walked one at a time (`_each_walk_unit`); past 32 the walk used
# to `map` over them (320 B at 33), and the marker check splatted their labels generically
# (4608 B at 40, 27904 B at 65, #853).
@testset "$n restricted products, linear (#853)" for n in (33, 40, 65)
    Wₕ = gridspace(graded_mesh(2, (9, 11)))
    f = x -> 1.0 + x[1]
    r(v) = Bramble.restrict_to(:boundary, v)
    l = form(Wₕ, v -> foldl(+, ntuple(k -> (k + 0.5) * innerₕ(f, r(v)), Val(n))))
    F = assemble(l)
    @test_allocs Bramble.assemble!(F, l)
    @test F≈sum(k -> k + 0.5, 1:n) .* assemble(form(Wₕ, v -> innerₕ(f, r(v)))) rtol=1e-12
end

WITH_3D_TESTS && @testset "Δₕ² in 3D: 81 entries (#853)" begin
    Wₕ = gridspace(graded_mesh(3, (7, 6, 5)))
    a = form(Wₕ, Wₕ, (u, v) -> innerₕ(Δₕ(Δₕ(u)), v))
    A = assemble(a)
    @test_allocs Bramble.assemble!(A, a)
    u = element(Wₕ, rand(ndofs(Wₕ)))
    w = element(Wₕ, rand(ndofs(Wₕ)))
    @test dot(parent(w), A*parent(u))≈innerₕ(Δₕ(Δₕ(u)), w) rtol=1e-10 atol=1e-12
end

end # module
