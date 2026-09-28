module TestFormMultigrid

using Test
using Bramble
using Bramble: GeometricMeshHierarchy, set_markers!, spacings, interpolation_matrix, CpuPolyester
using Bramble: AbstractSmoother, JacobiSmoother, ChebyshevSmoother, RedBlackGaussSeidel, max_eigenvalue_estimate,
               trial_space, D₋ₓ, D₋ᵧ, M₊ᵧ
using LinearAlgebra: LinearAlgebra, dot, norm, diag, LowerTriangular
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

# Smoothers (#329): mass plus variable diffusion on non-uniform meshes, with and without
# Dirichlet rows, against dense references built from `assemble`.
function _mg_smoother_form(D, n; T = Float64)
    Random.seed!(MG_SEED)
    I = interval(T(0), T(1))
    Ω = D == 1 ? I : D == 2 ? I × interval(T(0), T(2)) : I × I × interval(T(-1), T(1))
    W = gridspace(mesh(domain(Ω), n, ntuple(_ -> false, D)))
    κ = Rₕ(W, x -> 1 + sum(abs2, x))
    return form(W, W, (u, v) -> innerₕ(u, v) + inner₊(κ * ∇ₕ(u), ∇ₕ(v)))
end

const _MG_SMOOTHER_SHAPES = ((1, 33), (2, (17, 13)), (3, (9, 7, 9)))

# The colour of each unknown: the parity of its zero-based index sum.
_mg_colours(dims) = [mod(sum(Tuple(I)) - length(dims), 2) for I in CartesianIndices(dims)][:]

# `sweeps` of forward Gauss-Seidel in the red-then-black ordering: the lower triangle of the
# permuted matrix, solved densely.
function _mg_rbgs_dense(A, x, b, dims, sweeps)
    c = _mg_colours(dims)
    p = [findall(==(0), c); findall(==(1), c)]
    Ap = A[p, p]
    xp, bp = x[p], b[p]
    for _ in 1:sweeps
        xp += LowerTriangular(Ap) \ (bp - Ap * xp)
    end
    y = similar(x)
    y[p] = xp
    return y
end

# The Chebyshev smoother from its error polynomial: x - x⋆ ↦ T_k((θ - D⁻¹A)/δ) / T_k(θ/δ) (x - x⋆).
function _mg_cheb_dense(A, x, b, λmin, λmax, k)
    θ, δ = (λmax + λmin) / 2, (λmax - λmin) / 2
    Id = Matrix{Float64}(LinearAlgebra.I, size(A))
    t = (θ * Id - A ./ diag(A)) / δ
    T0, T1 = Id, t
    for _ in 2:k
        T0, T1 = T1, 2 * t * T1 - T0
    end
    xs = A \ b
    return xs + (T1 / cosh(k * acosh(θ / δ))) * (x - xs)
end

# The highest-frequency grid function, (-1)^(sum of indices).
_mg_checkerboard(dims) = [(-1.0)^sum(Tuple(I)) for I in CartesianIndices(dims)][:]

_mg_salloc(s, x, b) = (smooth!(s, x, b); @allocated smooth!(s, x, b))

@testset "gmg: smoothers damp high frequencies" begin
    for (D, n) in _MG_SMOOTHER_SHAPES, dl in (nothing, :boundary)

        a = _mg_smoother_form(D, n)
        W = trial_space(a)
        dims = npoints(mesh(W), Tuple)
        kw = dl === nothing ? (;) : (; dirichlet = dl)
        op = matrix_free_operator(a; kw...)
        A = Matrix(assemble(a; kw...))
        Random.seed!(MG_SEED + D)
        x0, b = randn(ndofs(W)), randn(ndofs(W))

        x = copy(x0)
        @test smooth!(jacobi_smoother(op; ω = 0.7, sweeps = 2), x, b) === x
        y = x0 + 0.7 * (b - A * x0) ./ diag(A)
        @test _mg_agree(x, y + 0.7 * (b - A * y) ./ diag(A))
        @test _mg_agree(smooth!(jacobi_smoother(a; kw..., ω = 0.7, sweeps = 2), copy(x0), b), x)

        s = chebyshev_smoother(op; degree = 3)
        @test s.λmax ≈ max_eigenvalue_estimate(op; preconditioner = jacobi_preconditioner(op))
        @test s.λmin ≈ s.λmax / 4
        @test _mg_agree(smooth!(s, copy(x0), b), _mg_cheb_dense(A, x0, b, s.λmin, s.λmax, 3))

        for sweeps in (1, 2)
            x = smooth!(red_black_gauss_seidel(a; kw..., sweeps), copy(x0), b)
            @test _mg_agree(x, _mg_rbgs_dense(A, x0, b, dims, sweeps))
            # A Dirichlet row is the identity row, so Gauss-Seidel solves it exactly.
            if dl !== nothing
                bnd = findall(i -> A[i, i] == 1 && count(!iszero, A[i, :]) == 1, 1:ndofs(W))
                @test !isempty(bnd) && _mg_agree(x[bnd], b[bnd])
            end
        end
        # A black half-sweep solves the black equations: their residual vanishes.
        x = smooth!(red_black_gauss_seidel(op), copy(x0), b)
        @test norm((b - A * x)[_mg_colours(dims) .== 1], Inf) < 1e-10 * norm(b, Inf)

        # The checkerboard is near the top of the spectrum of D⁻¹A, λ ≈ 2: a Jacobi sweep
        # multiplies it by about |1 - 2ω|, 1/3 at ω = 2/3 and (2D - 1)/(2D + 1) at the
        # default; Chebyshev and red-black remove more than half of its residual.
        hf = _mg_checkerboard(dims)
        ratio(sm) = norm(A * smooth!(sm, copy(hf), zeros(ndofs(W)))) / norm(A * hf)
        @test ratio(jacobi_smoother(op; ω = 2 / 3)) < 0.4
        @test ratio(jacobi_smoother(op)) < (2D - 1) / (2D + 1) + 0.05
        @test ratio(chebyshev_smoother(op)) < 0.5
        @test ratio(red_black_gauss_seidel(op)) < 0.5
        for sm in (jacobi_smoother(op), chebyshev_smoother(op), red_black_gauss_seidel(op))
            @test _mg_salloc(sm, copy(x0), b) == 0
            # A `VectorElement` iterate and right-hand side act on their storage.
            uₕ, bₕ = element(W, copy(x0)), element(W, b)
            @test smooth!(sm, uₕ, bₕ) === uₕ
            @test parent(uₕ) == smooth!(sm, copy(x0), b)
        end
    end

    # Transposed difference pairs and region restrictions couple neighbours of opposite
    # colour only, so red-black takes them.
    W = trial_space(_mg_smoother_form(2, (9, 7)))
    for a in (form(W, W, (u, v) -> innerₕ(u, v) + innerₕ(D₋ₓ(u), v) + 2.0 * innerₕ(u, D₋ₓ(v))),
        form(W, W, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(u, Bramble.restrict_to(:boundary, v))))
        x0, b = randn(ndofs(W)), randn(ndofs(W))
        @test _mg_agree(smooth!(red_black_gauss_seidel(a), copy(x0), b),
            _mg_rbgs_dense(Matrix(assemble(a)), x0, b, npoints(mesh(W), Tuple), 1))
    end

    # Float32 stays Float32 and allocation-free.
    a32 = _mg_smoother_form(2, (17, 13); T = Float32)
    n32 = ndofs(trial_space(a32))
    x32, b32 = randn(Float32, n32), zeros(Float32, n32)
    hf32 = Float32.(_mg_checkerboard((17, 13)))
    A32 = assemble(a32)
    for sm in (jacobi_smoother(a32; ω = 2 / 3), chebyshev_smoother(a32), red_black_gauss_seidel(a32))
        @test eltype(sm) === Float32
        @test sm.inv_diagonal isa Vector{Float32}
        @test eltype(smooth!(sm, copy(hf32), b32)) === Float32
        @test norm(A32 * smooth!(sm, copy(hf32), b32)) < 0.5f0 * norm(A32 * hf32)
        @test _mg_salloc(sm, x32, b32) == 0
    end

    # A mixed difference couples diagonal neighbours, which share a colour.
    W = trial_space(_mg_smoother_form(2, (9, 7)))
    box = form(W, W, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(D₋ᵧ(D₋ₓ(u)), D₋ᵧ(D₋ₓ(v))))
    @test_throws ArgumentError red_black_gauss_seidel(box)
    @test_throws ArgumentError red_black_gauss_seidel(matrix_free_operator(box))
    @test_throws ArgumentError red_black_gauss_seidel(form(W, W, (u, v) -> innerₕ(D₋ₓ(D₋ₓ(u)), v)))
    # An average across y of a difference along x couples diagonal neighbours too, but on a
    # collapsed y axis they do not exist, so the same form is taken there. (A difference
    # along a collapsed axis divides by its zero spacing, hence the average.)
    avg(W) = form(W, W, (u, v) -> innerₕ(u, v) + innerₕ(D₋ₓ(u), D₋ₓ(v)) + innerₕ(M₊ᵧ(D₋ₓ(u)), M₊ᵧ(D₋ₓ(v))))
    @test_throws ArgumentError red_black_gauss_seidel(avg(W))
    Random.seed!(MG_SEED)
    Wc = gridspace(mesh(domain(interval(0.0, 1.0) × interval(0.5, 0.5)), (17, 4), false))
    @test npoints(mesh(Wc), Tuple) == (17, 1)
    boxc = avg(Wc)
    x0, b = randn(17), randn(17)
    @test _mg_agree(smooth!(red_black_gauss_seidel(boxc), copy(x0), b),
        _mg_rbgs_dense(Matrix(assemble(boxc)), x0, b, (17, 1), 1))
    # A composite space is refused; Jacobi and Chebyshev take it.
    V = W × W
    c = form(V, V, (u, v) -> innerₕ(u(1), v(1)) + inner₊(∇ₕ(u(2)), ∇ₕ(v(2))) + innerₕ(u(2), v(2)))
    @test_throws ArgumentError red_black_gauss_seidel(c)
    Ac = Matrix(assemble(c))
    x0, b = randn(ndofs(V)), randn(ndofs(V))
    @test _mg_agree(smooth!(jacobi_smoother(c), copy(x0), b), x0 + 0.8 * (b - Ac * x0) ./ diag(Ac))

    a = _mg_smoother_form(2, (9, 7))
    n = ndofs(trial_space(a))
    for sm in (jacobi_smoother(a), chebyshev_smoother(a; λmax = 2.5), red_black_gauss_seidel(a))
        @test sm isa AbstractSmoother{Float64}
        @test size(sm) == (n, n)
        @test_throws DimensionMismatch smooth!(sm, zeros(n + 1), zeros(n))
        @test_throws DimensionMismatch smooth!(sm, zeros(n), zeros(n - 1))
        x = zeros(n)
        @test_throws ArgumentError smooth!(sm, x, x)
        buf = zeros(2n)
        @test_throws ArgumentError smooth!(sm, view(buf, 1:n), view(buf, 2:(n + 1)))
        @test_throws ArgumentError smooth!(sm, _MgZeroBased(zeros(n)), zeros(n))
    end
    @test jacobi_smoother(a) isa JacobiSmoother
    @test chebyshev_smoother(a) isa ChebyshevSmoother
    @test red_black_gauss_seidel(a) isa RedBlackGaussSeidel
    @test_throws ArgumentError jacobi_smoother(a; sweeps = 0)
    @test_throws ArgumentError jacobi_smoother(a; ω = 0)
    @test_throws ArgumentError jacobi_smoother(a; ω = Inf)
    @test_throws ArgumentError chebyshev_smoother(a; degree = 0)
    @test_throws ArgumentError chebyshev_smoother(a; λmax = -1.0)
    @test_throws ArgumentError red_black_gauss_seidel(a; sweeps = 0)
    Wr = gridspace(mesh(domain(interval(0.0, 1.0)), 9, false))
    rect = form(Wr, trial_space(_mg_smoother_form(1, 5)), (u, v) -> innerₕ(u, v))
    @test_throws DimensionMismatch jacobi_smoother(rect)
end

end # module TestFormMultigrid
