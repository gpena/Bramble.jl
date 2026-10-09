module TestFormKroneckerBlock

using Test
using Bramble
using Bramble: is_separable, kronecker_operator, KroneckerBlockOperator
using Bramble: D₋ₓ, D₋ᵧ, inner₊ₓ
using LinearAlgebra: issymmetric, mul!
using SparseArrays: SparseMatrixCSC, nnz, findnz
using Random
using Polyester

# `kronecker_operator` on a composite space whose leaves share one mesh: one
# `KroneckerLinearOperator` per nonzero (test leaf, trial leaf) block, wrapped in a
# `KroneckerBlockOperator`. Every check compares it against `assemble(a)` on a graded mesh,
# where a factor built from the wrong axis or offset gives other numbers.

const KB_SEED = 20261004

# Uniform, then moved by `change_points!` to `t^(1 + d/4)` along axis `d`, so no two axes
# share their nodes.
function _kb_graded_mesh(n::NTuple{D, Int}, be = backend()) where {D}
    Ωₕ = mesh(domain(reduce(×, ntuple(_ -> interval(0.0, 1.0), D))), n, ntuple(_ -> false, D);
        backend = be)
    Bramble.change_points!(Ωₕ, ntuple(d -> range(0.0, 1.0; length = n[d]) .^ (1 + 0.25d), D))
    return Ωₕ
end

# Two diagonal blocks of different families and one advection-type off-diagonal block.
_kb_coupled(u, v) = innerₕ(u(1), v(1)) + inner₊ₓ(D₋ₓ(u(2)), D₋ₓ(v(2))) + innerₕ(u(2), v(2)) +
                    innerₕ(D₋ₓ(u(1)), v(2))

# Same stored pattern and nnz as `assemble(a)`, entries equal to rounding.
function _kb_same_matrix(K, A)
    B = SparseMatrixCSC(K)
    nnz(B) == nnz(A) && findnz(B)[1:2] == findnz(A)[1:2] || return false
    return maximum(abs, B - A; init = 0.0) <= 1e-13 * maximum(abs, A)
end

# A vector indexed from 0, standing in for an `OffsetVector` (not a test dependency).
struct _KBZeroBased <: AbstractVector{Float64}
    data::Vector{Float64}
end
Base.size(v::_KBZeroBased) = size(v.data)
Base.axes(v::_KBZeroBased) = (0:(length(v.data) - 1),)
Base.getindex(v::_KBZeroBased, i::Int) = v.data[i + 1]
Base.setindex!(v::_KBZeroBased, a, i::Int) = (v.data[i + 1] = a)

_kb_alloc5(y, K, x, α, β) = @allocated mul!(y, K, x, α, β)
_kb_alloc3(y, K, x) = @allocated mul!(y, K, x)

# The checks of "matches assemble" for one mesh size, behind a function barrier: inlined in
# the loop over 2D and 3D sizes, the space is a 2D/3D union and inference pairs a 3D trial
# function with a 2D test function.
function _kb_matches_assemble(n::NTuple{D, Int}) where {D}
    V = gridspace(_kb_graded_mesh(n), Val(2))
    a = form(V, V, _kb_coupled)
    @test is_separable(a)
    K = kronecker_operator(a)
    A = assemble(a)
    @test K isa KroneckerBlockOperator
    @test length(K.blocks) == 3
    @test size(K) == size(A)
    @test eltype(K) === Float64
    @test _kb_same_matrix(K, A)
    @test K[2 + ndofs(V) ÷ 2, 3] == A[2 + ndofs(V) ÷ 2, 3]
    @test K[3, 2 + ndofs(V) ÷ 2] == 0
    x = rand(MersenneTwister(KB_SEED), size(A, 2))
    y0 = rand(MersenneTwister(KB_SEED + 1), size(A, 1))
    @test isapprox(K * x, A * x; rtol = 1e-12)
    @test isapprox(mul!(copy(y0), K, x, 0.5, -3.0), 0.5 * (A * x) - 3.0 * y0;
        rtol = 1e-12)
    @test mul!(fill(NaN, size(A, 1)), K, x, 1.0, 0.0) == K * x
    @test !issymmetric(K)
end

@testset "Kronecker blocks" begin
    @testset "matches assemble, 2D and 3D" begin
        for n in ((9, 7), (6, 5, 7))
            _kb_matches_assemble(n)
        end
    end

    # A test leaf no term reaches gets no block, and its rows are still `β * y` (zero for
    # `β = 0`, whatever `y` held).
    @testset "rows no block reaches" begin
        V = gridspace(_kb_graded_mesh((9, 7)), Val(2))
        a = form(V, V, (u, v) -> innerₕ(D₋ᵧ(u(2)), v(1)))
        K = kronecker_operator(a)
        A = assemble(a)
        @test length(K.blocks) == 1
        @test _kb_same_matrix(K, A)
        x = rand(MersenneTwister(KB_SEED), size(A, 2))
        y0 = rand(MersenneTwister(KB_SEED + 1), size(A, 1))
        @test isapprox(mul!(copy(y0), K, x, 2.0, -0.5), 2.0 * (A * x) - 0.5 * y0;
            rtol = 1e-12)
        y = mul!(fill(NaN, size(A, 1)), K, x, 1.0, 0.0)
        @test !any(isnan, y)
        @test isapprox(y, A * x; rtol = 1e-12)
    end

    # A term naming no component is the same integrand on every diagonal block.
    @testset "unnamed term on every leaf" begin
        V = gridspace(_kb_graded_mesh((9, 7)), Val(3))
        a = form(V, V, (u, v) -> innerₕ(u, v) + 2.0 * innerₕ(D₋ₓ(u(3)), D₋ₓ(v(3))))
        K = kronecker_operator(a)
        A = assemble(a)
        @test length(K.blocks) == 3
        @test _kb_same_matrix(K, A)
        @test issymmetric(K) && issymmetric(A)
    end

    # Symmetric: every diagonal block symmetric and each off-diagonal block the transpose of
    # its mirror; one coefficient changed in a mirror breaks it.
    @testset "issymmetric" begin
        V = gridspace(_kb_graded_mesh((9, 7)), Val(2))
        sym(c) = (u, v) -> innerₕ(u(1), v(1)) + innerₕ(D₋ₓ(u(2)), D₋ₓ(v(2))) +
                           innerₕ(D₋ᵧ(u(1)), D₋ᵧ(v(2))) + c * innerₕ(D₋ᵧ(u(2)), D₋ᵧ(v(1)))
        a = form(V, V, sym(1.0))
        @test issymmetric(assemble(a))
        @test issymmetric(kronecker_operator(a))
        b = form(V, V, sym(2.0))
        @test !issymmetric(assemble(b))
        @test !issymmetric(kronecker_operator(b))
    end

    # The refusals: a block that does not project, and leaves on different meshes.
    @testset "refusals" begin
        Ωₕ = _kb_graded_mesh((9, 7))
        W = gridspace(Ωₕ)
        V = gridspace(Ωₕ, Val(2))
        g = Rₕ(W, x -> x[1] + x[2])
        a = form(V, V, (u, v) -> innerₕ(u(1), v(1)) + innerₕ(g * u(2), v(2)))
        @test !is_separable(a)
        @test_throws ArgumentError kronecker_operator(a)
        err = try
            kronecker_operator(a)
        catch e
            e
        end
        @test occursin("trial component 2, test component 2", err.msg)
        @test occursin("GridFunctionScale", err.msg)

        W2 = gridspace(_kb_graded_mesh((9, 7)))
        U = W × W2
        b = form(U, U, (u, v) -> innerₕ(u(1), v(1)) + innerₕ(u(2), v(2)))
        @test !is_separable(b)
        @test_throws ArgumentError kronecker_operator(b)
        err = try
            kronecker_operator(b)
        catch e
            e
        end
        @test occursin("trial component 2", err.msg)
    end

    # A grid-function coefficient read in two blocks warns once; a `Ref` stays live.
    @testset "coefficients" begin
        Ωₕ = _kb_graded_mesh((9, 7))
        W = gridspace(Ωₕ)
        V = gridspace(Ωₕ, Val(2))
        fx = Rₕ(W, x -> 1 + x[1])
        a = form(V, V, (u, v) -> innerₕ(fx * u(1), v(1)) + innerₕ(fx * u(2), v(2)))
        K = @test_logs (:warn, r"read once") kronecker_operator(a)
        @test _kb_same_matrix(K, assemble(a))
        c = Ref(2.0)
        b = form(V, V, (u, v) -> innerₕ(u(1), v(1)) + c * innerₕ(D₋ₓ(u(1)), v(2)))
        Kc = @test_logs min_level = Base.CoreLogging.Warn kronecker_operator(b)
        c[] = 7.0
        x = rand(MersenneTwister(KB_SEED), size(Kc, 2))
        @test isapprox(Kc * x, assemble(b) * x; rtol = 1e-12)
    end

    # The kernels index from 1: a vector indexed otherwise is refused, not read wrongly.
    @testset "offset-indexed vectors refused" begin
        Ωₕ = _kb_graded_mesh((9, 7))
        W = gridspace(Ωₕ)
        V = gridspace(Ωₕ, Val(2))
        Ks = (kronecker_operator(form(W, W, (u, v) -> innerₕ(D₋ₓ(u), D₋ₓ(v)))),
            kronecker_operator(form(V, V, _kb_coupled)))
        for K in Ks
            n = size(K, 1)
            x = rand(MersenneTwister(KB_SEED), n)
            @test_throws ArgumentError mul!(zeros(n), K, _KBZeroBased(x), 1.0, 0.0)
            @test_throws ArgumentError mul!(_KBZeroBased(zeros(n)), K, x, 1.0, 0.0)
            @test_throws ArgumentError mul!(zeros(n), K, _KBZeroBased(x))
        end
    end

    # The same product bit for bit under every host policy, and 0 B warm under `CpuSerial`
    # and `CpuPolyester`, 3- and 5-argument.
    @testset "policies: bitwise and bytes" begin
        for n in ((9, 7), (6, 5, 7))
            Ks = map((Bramble.CpuSerial(), Bramble.CpuThreaded(), Bramble.CpuPolyester())) do P
                V = gridspace(_kb_graded_mesh(n, backend(; policy = P)), Val(2))
                return kronecker_operator(form(V, V, _kb_coupled))
            end
            x = rand(MersenneTwister(KB_SEED), size(Ks[1], 2))
            y0 = rand(MersenneTwister(KB_SEED + 1), size(Ks[1], 1))
            ys = map(K -> mul!(copy(y0), K, x, 0.5, -3.0), Ks)
            @test ys[2] == ys[1] && ys[3] == ys[1]
            for K in (Ks[1], Ks[3])
                y = copy(y0)
                _kb_alloc5(y, K, x, 0.5, -3.0)
                @test _kb_alloc5(y, K, x, 0.5, -3.0) == 0
                _kb_alloc3(y, K, x)
                @test _kb_alloc3(y, K, x) == 0
            end
        end
    end
end

end # module
