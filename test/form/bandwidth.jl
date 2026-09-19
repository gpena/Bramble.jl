module TestFormBandwidth

using Test
using Bramble
using SparseArrays
using Random
using Bramble: bandwidths, blockbandwidths, inner₊

# Reading a form's matrix bandwidth off its AST, before assembling anything.
#
# `bandwidths`/`blockbandwidths` predict the diagonals an assembled matrix occupies from
# the resolved AST's per-axis offset reach alone (`stencil_offsets`) and the mesh's point
# counts. The tests below check that prediction against the one thing it does not read:
# the diagonals the assembled `SparseMatrixCSC` actually stores a nonzero value in.

# The truth: `(l, u) = (max(i - j), max(j - i))` over stored entries with a nonzero value,
# the standard lower/upper bandwidth convention (`BandedMatrices.bandwidths`).
function _true_bandwidths(A::SparseMatrixCSC)
    l = 0
    u = 0
    rows = rowvals(A)
    vals = nonzeros(A)
    for j in axes(A, 2)
        for k in nzrange(A, j)
            iszero(vals[k]) && continue
            i = rows[k]
            l = max(l, i - j)
            u = max(u, j - i)
        end
    end
    return (l, u)
end

# The block truth: the same matrix, read as blocks of `nsub` rows/columns each (a block's
# index is the last-axis coordinate, its within-block index the lexicographic index over
# the remaining axes) -- the `BandedBlockBandedMatrix` convention `blockbandwidths` follows.
function _true_blockbandwidths(A::SparseMatrixCSC, nsub::Int)
    n = size(A, 1)
    @assert size(A, 2) == n && mod(n, nsub) == 0
    l_blk = 0
    u_blk = 0
    l_sub = 0
    u_sub = 0
    rows = rowvals(A)
    vals = nonzeros(A)
    for j in axes(A, 2)
        bj, sj = divrem(j - 1, nsub)
        for k in nzrange(A, j)
            iszero(vals[k]) && continue
            i = rows[k]
            bi, si = divrem(i - 1, nsub)
            l_blk = max(l_blk, bi - bj)
            u_blk = max(u_blk, bj - bi)
            l_sub = max(l_sub, si - sj)
            u_sub = max(u_sub, sj - si)
        end
    end
    return ((l_blk, u_blk), (l_sub, u_sub))
end

# One family of forms, generic in the grid space so the same set runs in 1D, 2D and 3D:
# every operator here is either dimension-agnostic (`innerₕ`) or names the x-direction
# alone, which every dimension has.
function _bandwidth_test_forms()
    (
        ("innerₕ(u,v)", (u, v) -> innerₕ(u, v)),
        ("innerₕ(D₋ₓ(u), v)", (u, v) -> innerₕ(D₋ₓ(u), v)),
        ("inner₊(∇ₕ(u), ∇ₕ(v))", (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v))),
        ("innerₕ(Dcₓ(u), Dcₓ(v))", (u, v) -> innerₕ(Dcₓ(u), Dcₓ(v))),
        ("innerₕ(Dₕₓ(u), v)", (u, v) -> innerₕ(Dₕₓ(u), v)),
        ("innerₕ(jumpₓ(u), v)", (u, v) -> innerₕ(jumpₓ(u), v)),
        ("innerₕ(Mₓ(u), v)", (u, v) -> innerₕ(Mₓ(u), v))
    )
end

@testset "Bandwidths from the AST" begin
    Random.seed!(20260919)

    Ωₕ1 = mesh(domain(interval(0.0, 1.0)), 9, false)
    Ωₕ2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (7, 6), (false, false))
    Ωₕ3 = mesh(
        domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0)),
        (5, 4, 3),
        (false, false, false)
    )

    Wₕ1, Wₕ2, Wₕ3 = gridspace(Ωₕ1), gridspace(Ωₕ2), gridspace(Ωₕ3)

    @testset "Against the assembled matrix" begin
        for (dname, Wₕ) in (("1D", Wₕ1), ("2D", Wₕ2), ("3D", Wₕ3))
            @testset "$dname" begin
                n = npoints(mesh(Wₕ), Tuple)
                D = length(n)
                for (nm, fbuild) in _bandwidth_test_forms()
                    @testset "$nm" begin
                        a = form(Wₕ, Wₕ, fbuild)
                        A = assemble(a)
                        @test bandwidths(a) == _true_bandwidths(A)

                        if D == 1
                            @test blockbandwidths(a) == ((0, 0), bandwidths(a))
                        else
                            nsub = prod(n[1:(D - 1)])
                            @test blockbandwidths(a) == _true_blockbandwidths(A, nsub)
                        end
                    end
                end
            end
        end
    end

    @testset "Positive control: different true bandwidths give different answers" begin
        # A wider stencil widens the prediction, checked against the same-shape narrower
        # one rather than against a hard-coded number, so the test still means something
        # if the discretisation constants above change.
        a_narrow = form(Wₕ1, Wₕ1, (u, v) -> innerₕ(u, v))
        a_wide = form(Wₕ1, Wₕ1, (u, v) -> innerₕ(Dₕₓ(D₋ₓ(u)), v))
        @test bandwidths(a_narrow) != bandwidths(a_wide)
        @test bandwidths(a_narrow) == _true_bandwidths(assemble(a_narrow))
        @test bandwidths(a_wide) == _true_bandwidths(assemble(a_wide))

        a2_narrow = form(Wₕ2, Wₕ2, (u, v) -> innerₕ(u, v))
        a2_wide = form(Wₕ2, Wₕ2, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
        @test blockbandwidths(a2_narrow) != blockbandwidths(a2_wide)
    end

    @testset "ArgumentError cases" begin
        @testset "Composite trial/test space" begin
            Vₕ1 = vector_gridspace(Ωₕ1, 2)
            a = form(Vₕ1, Vₕ1, (u, v) -> innerₕ(u(1), v(1)) + innerₕ(u(2), v(2)))
            @test_throws ArgumentError bandwidths(a)
            @test_throws ArgumentError blockbandwidths(a)

            # Positive control: the same shape of form on a non-composite (leaf-only)
            # space answers rather than throwing.
            a_leaf = form(Wₕ1, Wₕ1, (u, v) -> innerₕ(u, v))
            @test bandwidths(a_leaf) isa Tuple{Int, Int}
        end

        @testset "Interpolation" begin
            Ωₕ_src = mesh(domain(interval(0.0, 1.0)), 5, true)
            Ωₕ_dst = mesh(domain(interval(0.0, 1.0)), 9, true)
            Wₕ_src, Wₕ_dst = gridspace(Ωₕ_src), gridspace(Ωₕ_dst)
            a = form(Wₕ_src, Wₕ_dst, (u, v) -> innerₕ(πₕ(u), v))
            @test_throws ArgumentError bandwidths(a)
            @test_throws ArgumentError blockbandwidths(a)
        end

        @testset "Cross-mesh without interpolation" begin
            Ωₕ_a = mesh(domain(interval(0.0, 1.0)), 5, true)
            Ωₕ_b = mesh(domain(interval(0.0, 1.0)), 9, true)
            Wₕ_a, Wₕ_b = gridspace(Ωₕ_a), gridspace(Ωₕ_b)
            a = form(Wₕ_a, Wₕ_b, (u, v) -> innerₕ(u, v))
            @test_throws ArgumentError bandwidths(a)
        end
    end
end

end # module TestFormBandwidth
