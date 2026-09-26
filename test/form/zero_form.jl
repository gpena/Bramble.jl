module FormZeroFormTests

using Test
using Bramble
using Bramble: ZeroOperator, resolve_form_ast, allocate_system_matrix, assemble_parallel!
using ..TestUtils: alloc_test

# A bilinear form whose whole AST simplifies to zero.
#
# `0 * a`, or a runtime integer `k = 0`, collapses the AST to a bare `ZeroOperator`. Its
# stencil used to be the one linear-form entry `((offset, 0),)`, which the bilinear walk
# destructures as `(off_u, off_v, weight)`: `allocate_system_matrix` died with a
# `BoundsError` in `_visit_entries`. The answer is a zero matrix of the right size, on the
# serial and threaded paths alike, and refilling it in place stays allocation-free.

function _check_zero(a, nrows, ncols)
    @test resolve_form_ast(a) isa ZeroOperator
    A = assemble(a)
    @test size(A) == (nrows, ncols)
    @test iszero(A)
    assemble!(A, a)
    @test iszero(A)
    @test alloc_test(assemble!, A, a) == 0
    Apar = allocate_system_matrix(a)
    assemble_parallel!(Apar, a)
    @test size(Apar) == (nrows, ncols)
    @test iszero(Apar)
    return nothing
end

@testset "A form that simplifies to zero assembles to a zero matrix" begin
    cases = (
        (domain(interval(0.0, 1.0) × interval(0.0, 2.0)), (7, 6)),
        (domain(interval(0.0, 1.0) × interval(0.0, 2.0) × interval(0.0, 3.0)), (5, 6, 4))
    )

    @testset "Scalar space, $(length(sz))D, non-uniform" for (S, sz) in cases
        D = length(sz)
        Wₕ = gridspace(mesh(S, sz, ntuple(_ -> false, D)))
        N = ndofs(Wₕ)
        ηₓ = η[1]
        m = (:boundary,)

        # a literal zero
        _check_zero(form(Wₕ, Wₕ, (u, v) -> 0 * inner_Γ(u, v; markers = m)), N, N)
        _check_zero(form(Wₕ, Wₕ, (u, v) -> 0 * inner₊(∇ₕ(u), ∇ₕ(v))), N, N)

        # a runtime integer zero, as a scale and inside a normal-component term
        k = Ref(0)[]
        _check_zero(form(Wₕ, Wₕ, (u, v) -> k * inner_Γ(u, v; markers = m)), N, N)
        _check_zero(form(Wₕ, Wₕ, (u, v) -> inner_Γ(k * (u * ηₓ), v; markers = m)), N, N)

        # a runtime Float64 zero is kept as a scale, not elided: stored zeros, same answer
        κ = Ref(0.0)[]
        a = form(Wₕ, Wₕ, (u, v) -> inner_Γ(κ * (u * ηₓ), v; markers = m))
        A = assemble(a)
        @test size(A) == (N, N)
        @test iszero(A)
    end

    @testset "Composite space, $(length(sz))D, non-uniform" for (S, sz) in cases
        D = length(sz)
        Ωₕ = mesh(S, sz, ntuple(_ -> false, D))
        Wₕ = gridspace(Ωₕ)
        Vₕ = gridspace(Ωₕ, Val(D))
        NV, NW = ndofs(Vₕ), ndofs(Wₕ)
        k = Ref(0)[]

        # every block of a coupled form zero: a diagonal and a crossed block
        _check_zero(
            form(Vₕ, Vₕ, (u, v) -> 0 * (innerₕ(u(1), v(1)) + innerₕ(u(2), v(D)))), NV, NV
        )
        _check_zero(form(Vₕ, Vₕ, (u, v) -> k * innerₕ(u(1), v(2))), NV, NV)
        # a scalar trial against a composite test, and the mirror
        _check_zero(form(Wₕ, Vₕ, (q, v) -> 0 * innerₕ(q(1), v(D))), NV, NW)
        _check_zero(form(Vₕ, Wₕ, (v, q) -> k * innerₕ(v(D), q(1))), NW, NV)
    end
end

end # module FormZeroFormTests
