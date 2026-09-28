module TestFormMultigrid

using Test
using Bramble
using Bramble: GeometricMeshHierarchy, set_markers!, spacings, interpolation_matrix, CpuPolyester
using LinearAlgebra: dot, norm
using SparseArrays: sparse
using ForwardDiff: ForwardDiff
using Random

# Geometric multigrid (gpena/Bramble.jl#329). Meshes are non-uniform throughout: on a uniform
# mesh rebuilding each level from the domain would nest too, and hide a hierarchy that does
# not take every other point.

const MG_SEED = 3291

# The markers a mesh gets from `Ω` at its own points, for comparison with the carried ones.
function _mg_reevaluated_markers(Ωₕ, Ω)
    Ωr = copy(Ωₕ)
    set_markers!(Ωr, markers(Ω); warn_marker_mismatch = false)
    return markers(Ωr)
end

_mg_axes(p::AbstractVector{<:Number}) = (p,)
_mg_axes(p::Tuple) = p

# A vector indexed from 0, to check that the transfers refuse offset axes.
struct _MgZeroBased <: AbstractVector{Float64}
    p::Vector{Float64}
end
Base.size(v::_MgZeroBased) = size(v.p)
Base.axes(v::_MgZeroBased) = (Base.IdentityUnitRange(0:(length(v.p) - 1)),)
Base.getindex(v::_MgZeroBased, i::Int) = v.p[i + 1]
Base.setindex!(v::_MgZeroBased, x, i::Int) = (v.p[i + 1] = x)

# Allocation of a warmed transfer, behind a function barrier.
_mg_palloc(xf, H, l, xc) = (prolongate!(xf, H, l, xc); @allocated prolongate!(xf, H, l, xc))
_mg_calloc(xc, H, l, xf) = (coarsen!(xc, H, l, xf); @allocated coarsen!(xc, H, l, xf))

# The oracle `P`. `interpolation_matrix` does not take a collapsed axis, so there `P` is the
# Kronecker product of the per-axis matrices, a 1×1 identity on the collapsed axis.
function _mg_oracle(Ωf, Ωc)
    all(>(1), npoints(Ωf, Tuple)) || return reduce(
        kron, reverse(ntuple(
            d -> npoints(Ωf(d)) == 1 ? sparse(ones(1, 1)) :
                 interpolation_matrix(gridspace(Ωf(d)), gridspace(Ωc(d))),
            length(npoints(Ωf, Tuple)))))
    return interpolation_matrix(gridspace(Ωf), gridspace(Ωc))
end

_mg_agree(a, b) = isapprox(a, b; rtol = 1e-13, atol = 1e-13)

# Non-uniform meshes in 1D, 2D and 3D, and two with a collapsed axis, with a level count each.
function _mg_transfer_meshes(bk = backend())
    Random.seed!(MG_SEED)
    unit(a = 0.0, b = 1.0) = interval(a, b)
    return (
        (mesh(domain(unit()), 33, false; backend = bk), 4),
        (mesh(domain(unit() × unit(0.0, 2.0)), (17, 9), false; backend = bk), 3),
        (mesh(domain(unit() × unit(-1.0, 1.0) × unit(0.0, 2.0)), (9, 5, 9), false; backend = bk), 3),
        (mesh(domain(unit() × unit(0.5, 0.5)), (17, 4), false; backend = bk), 3),
        (mesh(domain(unit() × unit(0.5, 0.5) × unit()), (9, 4, 5), false; backend = bk), 2)
    )
end

@testset "gmg: nested hierarchy" begin
    Random.seed!(MG_SEED)
    cases = (
        (domain(interval(0.0, 1.0), :inlet => :left, :near => x -> x[1] < 0.3), 33, 4),
        (
            domain(interval(0.0, 1.0) × interval(0.0, 2.0), :walls => (:top, :bottom),
                :blob => x -> (x[1] - 0.5)^2 + (x[2] - 1.0)^2 < 0.2),
            (17, 33),
            3),
        (
            domain(interval(0.0, 1.0) × interval(-1.0, 1.0) × interval(0.0, 2.0),
                :inlet => :left, :half => x -> x[3] < 1.0),
            (9, 5, 17),
            3)
    )
    for (Ω, n, L) in cases
        Ωf = mesh(Ω, n, false)
        H = GeometricMeshHierarchy(Ωf, L)
        @test length(H) == L && lastindex(H) == L && firstindex(H) == 1
        @test H[end] === Ωf
        @test all(Ωₗ -> Ωₗ isa typeof(Ωf), H)
        @test collect(H) == [H[l] for l in 1:L]
        for l in 1:(L - 1)
            pc, pf = _mg_axes(points(H[l])), _mg_axes(points(H[l + 1]))
            # Exact equality: the coarse points are copies of the fine ones, not recomputed.
            @test all(d -> pc[d] == pf[d][1:2:end], eachindex(pc))
            @test npoints(H[l], Tuple) == map(k -> (k - 1) ÷ 2 + 1, npoints(H[l + 1], Tuple))
            # Cached metrics follow the new points, not the uniform ones built first.
            @test all(d -> _mg_axes(spacings(H[l]))[d] ≈ [pc[d][2] - pc[d][1]; diff(pc[d])], eachindex(pc))
        end
        @test npoints(H[1], Tuple) == map(k -> (k - 1) ÷ 2^(L - 1) + 1, npoints(Ωf, Tuple))
        # Custom markers carry over, and agree with re-evaluating the domain at each level.
        for l in 1:L
            @test markers(H[l]) == _mg_reevaluated_markers(H[l], Ω)
        end
        @test keys(markers(H[1])) == keys(markers(Ωf))
        # A space on the coarsest level sees its own points.
        @test ndofs(gridspace(H[1])) == npoints(H[1])
    end

    # The element type and backend of the finest mesh are kept on every level.
    Ω32 = mesh(domain(interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0)), (9, 17), false)
    H32 = GeometricMeshHierarchy(Ω32, 4)
    @test all(Ωₗ -> eltype(Ωₗ) == Float32 && backend(Ωₗ) == backend(Ω32), H32)
    @test npoints(H32[1], Tuple) == (2, 3)

    # One level is the mesh alone.
    Ω1 = mesh(domain(interval(0.0, 1.0)), 10, false)
    @test only(collect(GeometricMeshHierarchy(Ω1, 1))) === Ω1

    # A collapsed axis stays one point.
    Ωc = mesh(domain(interval(0.0, 1.0) × interval(0.5, 0.5)), (9, 4), false)
    @test npoints(GeometricMeshHierarchy(Ωc, 3)[1], Tuple) == (3, 1)

    # (n - 1) not divisible by 2^(levels - 1) on some axis, and levels < 1.
    @test_throws ArgumentError GeometricMeshHierarchy(mesh(domain(interval(0.0, 1.0)), 18, false), 2)
    @test_throws ArgumentError GeometricMeshHierarchy(
        mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (17, 11), false), 3)
    @test_throws ArgumentError GeometricMeshHierarchy(
        mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 9, 7), false), 3)
    @test_throws ArgumentError GeometricMeshHierarchy(Ω1, 0)
    Ω33 = mesh(domain(interval(0.0, 1.0)), 33, false)
    @test_throws ArgumentError GeometricMeshHierarchy(Ω33, 65)
    @test eachindex(GeometricMeshHierarchy(Ω33, 2)) == 1:2

    @test sprint(show, GeometricMeshHierarchy(Ω32, 3)) ==
          "GeometricMeshHierarchy{2D, 3 levels, (3, 5) to (9, 17) pts}"
end

# `P` is `interpolation_matrix` of the fine space from the coarse one, never built by the
# transfers; `coarsen!` is `Pᵀ` without scaling.
@testset "gmg: prolongation and transpose" begin
    for (Ωf, L) in _mg_transfer_meshes()
        H = GeometricMeshHierarchy(Ωf, L)
        for l in 2:L
            Wf, Wc = gridspace(H[l]), gridspace(H[l - 1])
            P = _mg_oracle(H[l], H[l - 1])
            xc, yf = randn(ndofs(Wc)), randn(ndofs(Wf))
            xf, yc = fill(NaN, ndofs(Wf)), fill(NaN, ndofs(Wc))
            @test prolongate!(xf, H, l, xc) === xf
            @test _mg_agree(xf, P * xc)
            y0 = copy(yf)
            @test coarsen!(yc, H, l, yf) === yc
            @test _mg_agree(yc, P' * yf)
            # ⟨P xc, yf⟩ = ⟨xc, Pᵀ yf⟩, and the fine input is left as it was.
            @test isapprox(dot(xf, yf), dot(xc, yc); rtol = 1e-13)
            @test yf == y0
            @test _mg_palloc(xf, H, l, xc) == 0
            @test _mg_calloc(yc, H, l, yf) == 0
            # Grid functions give the same as their raw storage.
            uc, uf = element(Wc), element(Wf)
            parent(uc) .= xc
            @test prolongate!(uf, H, l, uc) === uf && parent(uf) == xf
            parent(uf) .= yf
            @test parent(coarsen!(uc, H, l, uf)) == yc
            # A view as the destination, as a column of a matrix.
            M = zeros(ndofs(Wf), 2)
            prolongate!(view(M, :, 2), H, l, xc)
            @test M[:, 2] == xf && iszero(M[:, 1])
        end
        # Two levels composed are the product of the two matrices.
        if L >= 3
            P32, P21 = _mg_oracle(H[3], H[2]), _mg_oracle(H[2], H[1])
            x1 = randn(npoints(H[1]))
            x2, x3 = zeros(npoints(H[2])), zeros(npoints(H[3]))
            prolongate!(x3, H, 3, prolongate!(x2, H, 2, x1))
            @test _mg_agree(x3, P32 * (P21 * x1))
        end
    end

    # Interpolation reproduces the multilinear function x₁ + 2 x₂ x₃ - x₂ exactly.
    Ωf, L = _mg_transfer_meshes()[3]
    H = GeometricMeshHierarchy(Ωf, L)
    f(p) = p[1] + 2 * p[2] * p[3] - p[2]
    grid(Ω) = vec([f((a, b, c)) for a in points(Ω(1)), b in points(Ω(2)), c in points(Ω(3))])
    xf = zeros(npoints(H[3]))
    @test _mg_agree(prolongate!(xf, H, 3, grid(H[2])), grid(H[3]))

    # The unscaled transpose is Galerkin for the 1D stiffness on a non-uniform mesh: the
    # forms carry the discrete measure, so Pᵀ A_f P is the coarse form's matrix.
    Ω1, _ = _mg_transfer_meshes()[1]
    H = GeometricMeshHierarchy(Ω1, 3)
    stiff(Ω) = (W = gridspace(Ω); assemble(form(W, W, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))))
    for l in 2:3
        n = npoints(H[l - 1])
        PA = zeros(npoints(H[l]), n)
        for j in 1:n
            e = zeros(n)
            e[j] = 1
            prolongate!(view(PA, :, j), H, l, e)
        end
        Ac = stiff(H[l - 1])
        @test norm(PA' * stiff(H[l]) * PA - Ac, Inf) < 1e-12 * norm(Ac, Inf)
    end

    # Dual numbers flow through both transfers: the Jacobians are P and Pᵀ.
    Ωf, L = _mg_transfer_meshes()[2]
    H = GeometricMeshHierarchy(Ωf, L)
    P = _mg_oracle(H[3], H[2])
    Jp = ForwardDiff.jacobian(x -> prolongate!(similar(x, npoints(H[3])), H, 3, x), randn(npoints(H[2])))
    Jc = ForwardDiff.jacobian(y -> coarsen!(similar(y, npoints(H[2])), H, 3, y), randn(npoints(H[3])))
    @test _mg_agree(Jp, Matrix(P)) && _mg_agree(Jc, Matrix(P'))

    # Float32 levels transfer in Float32.
    Ω32 = mesh(domain(interval(0.0f0, 1.0f0) × interval(0.0f0, 1.0f0)), (9, 5), false)
    H32 = GeometricMeshHierarchy(Ω32, 2)
    P32 = interpolation_matrix(gridspace(H32[2]), gridspace(H32[1]))
    x32 = rand(Float32, npoints(H32[1]))
    @test isapprox(prolongate!(zeros(Float32, npoints(H32[2])), H32, 2, x32), P32 * x32; rtol = 1.0f-5)

    # Wrong lengths, a level with no coarser one or none at all, and offset axes.
    Ωf, L = _mg_transfer_meshes()[2]
    H = GeometricMeshHierarchy(Ωf, L)
    nf, nc = npoints(H[2]), npoints(H[1])
    for (a, b) in ((zeros(nf + 1), zeros(nc)), (zeros(nf), zeros(nc - 1)), (zeros(nc), zeros(nf)))
        @test_throws DimensionMismatch prolongate!(a, H, 2, b)
        @test_throws DimensionMismatch coarsen!(b, H, 2, a)
    end
    for l in (0, 1, L + 1)
        @test_throws ArgumentError prolongate!(zeros(nf), H, l, zeros(nc))
        @test_throws ArgumentError coarsen!(zeros(nc), H, l, zeros(nf))
    end
    @test_throws ArgumentError prolongate!(_MgZeroBased(zeros(nf)), H, 2, zeros(nc))
    @test_throws ArgumentError prolongate!(zeros(nf), H, 2, _MgZeroBased(zeros(nc)))
    @test_throws ArgumentError coarsen!(_MgZeroBased(zeros(nc)), H, 2, zeros(nf))
    @test_throws ArgumentError coarsen!(zeros(nc), H, 2, _MgZeroBased(zeros(nf)))
    buf = zeros(nf + nc)
    @test_throws ArgumentError prolongate!(view(buf, 1:nf), H, 2, view(buf, 4:(3 + nc)))
    @test_throws ArgumentError coarsen!(view(buf, 4:(3 + nc)), H, 2, view(buf, 1:nf))
    @test_throws ArgumentError prolongate!(zeros(1), GeometricMeshHierarchy(Ω1, 1), 1, zeros(1))
end

# The threaded and Polyester sweeps write every point once, so they equal the serial ones
# bitwise, on every repeat. Run at `--threads=4` for a race to have a chance. `CpuPolyester`
# needs Polyester, which the `unit` group does not load.
@testset "gmg: threaded transfers agree" begin
    policies = Any[Parallel()]
    Base.get_extension(Bramble, :BramblePolyesterExt) !== nothing && push!(policies, CpuPolyester())
    for policy in policies
        for ((Ωs, L), (Ωt, _)) in zip(_mg_transfer_meshes(), _mg_transfer_meshes(backend(; policy)))
            @test Bramble.execution_policy(Ωt) == policy
            @test _mg_axes(points(Ωt)) == _mg_axes(points(Ωs))
            Hs, Ht = GeometricMeshHierarchy(Ωs, L), GeometricMeshHierarchy(Ωt, L)
            for l in 2:L
                xc, yf = randn(npoints(Hs[l - 1])), randn(npoints(Hs[l]))
                xf, yc = prolongate!(zeros(length(yf)), Hs, l, xc), coarsen!(zeros(length(xc)), Hs, l, yf)
                xt, yt = similar(xf), similar(yc)
                @test all(1:20) do _
                    prolongate!(xt, Ht, l, xc)
                    coarsen!(yt, Ht, l, yf)
                    return xt == xf && yt == yc
                end
            end
        end
    end
    # What a threaded transfer allocates is its task spawns, whatever the grid size.
    bytes = map((17, 257)) do n
        Random.seed!(MG_SEED)
        Ω = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (n, n), false;
            backend = backend(policy = Parallel()))
        H = GeometricMeshHierarchy(Ω, 2)
        xc, xf = randn(npoints(H[1])), zeros(npoints(H[2]))
        return (_mg_palloc(xf, H, 2, xc), _mg_calloc(xc, H, 2, xf))
    end
    @test bytes[1] == bytes[2]
end

end # module TestFormMultigrid
