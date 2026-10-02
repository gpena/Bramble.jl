# gmg_cycles.jl: the cycles of geometric multigrid and the
# preconditioner and solver built on them.
#
# Every level is rediscretised: `build(gridspace(H[l]))` gives the level's form, applied
# matrix-free, because a form embeds its spaces and cannot move between levels, and the
# Galerkin product `Pᵀ A P` would need the matrices matrix-free application avoids. The
# coarsest level alone is assembled, densely, and factorised once; the default hierarchy
# coarsens until an axis would drop below three points, so it is small by construction.
#
# A cycle recurses on the level index and works on vectors stored per level in the
# preconditioner: `x[l]` the correction, `b[l]` the restricted residual, `r[l]` the residual
# (and then the prolongated correction). The finest level's iterate and right-hand side are
# the caller's, so `x[L]` is empty unless the hierarchy has one level. A cycle allocates
# nothing on a serial policy.
#
# Post-smoothing is the adjoint of pre-smoothing (`_gmg_post_smooth!`: red-black runs
# reversed), and restriction is `Pᵀ` (`coarsen!`), so with `ν₁ == ν₂` the V- and W-cycles
# from a zero guess are symmetric operators, which conjugate gradients needs.

"""
    GMGPreconditioner{T, H, Op, S, F, V <: AbstractVector{T}} <: AbstractMatrixFreePreconditioner{T}

A geometric multigrid cycle as a preconditioner: `ldiv!(y, P, x)` sets `y` to one cycle for
`A y = x` from `y = 0`, where `A` is the finest level's form. It holds the
[`GeometricMeshHierarchy`](@ref), one [`MatrixFreeOperator`](@ref) and one smoother per
level, the dense factorisation of the coarsest level's matrix, and the per-level work
vectors, so an application allocates nothing on a [`CpuSerial`](@ref) policy and one `P`
must not be applied from two tasks at once. Build one with [`gmg_preconditioner`](@ref); the
cycles themselves are [`v_cycle!`](@ref), [`w_cycle!`](@ref) and [`fmg!`](@ref).

# Type parameters
- `T`: The element type of `A`.
- `H`: The hierarchy type.
- `Op`: The operator type of the levels.
- `S`: The smoother type of the levels.
- `F`: The type of the coarsest level's factorisation.
- `V`: The vector type of the work vectors.

See also: [`gmg_preconditioner`](@ref), [`gmg_solve`](@ref),
[`AbstractMatrixFreePreconditioner`](@ref).
"""
struct GMGPreconditioner{T, H <: GeometricMeshHierarchy, Op, S, F, V <: AbstractVector{T}} <:
       AbstractMatrixFreePreconditioner{T}
    hierarchy::H
    ops::Vector{Op}
    smoothers::Vector{S}
    coarse::F
    x::Vector{V}
    b::Vector{V}
    r::Vector{V}
    cycle::Symbol
    ν₁::Int
    ν₂::Int
end

Base.size(P::GMGPreconditioner) = size(last(P.ops))

function Base.show(io::IO, P::GMGPreconditioner)
    H = P.hierarchy
    L = length(H)
    print(
        io, "GMGPreconditioner{", P.cycle, "(", P.ν₁, ",", P.ν₂, "), ", L, " level",
        L == 1 ? "" : "s", ", ", npoints(H[1], Tuple), " to ", npoints(H[end], Tuple),
        " pts}"
    )
    return nothing
end

# The largest dense coarsest level `gmg_preconditioner` factorises.
const _GMG_MAX_COARSE_DOFS = 4096

_gmg_default_smoother(op) = chebyshev_smoother(op)

# The most levels coarsening by 2 allows with at least three points left on every axis that
# has more than one.
function _gmg_default_levels(Ωₕ::AbstractMeshType)
    n = npoints(Ωₕ, Tuple)
    any(>(1), n) || return 1
    L = 1
    while all(k -> k == 1 || (trailing_zeros(k - 1) >= L && (k - 1) >> L >= 2), n)
        L += 1
    end
    return L
end

"""
    gmg_preconditioner(build, Ωₕ::AbstractMeshType; levels = nothing, cycle = :V, ν₁ = 2, ν₂ = 2, smoother = op -> chebyshev_smoother(op)) -> GMGPreconditioner

The geometric multigrid preconditioner of the form `build(gridspace(Ωₕ))`. `ldiv!(y, P, x)`
runs one cycle for `A y = x` from `y = 0` over the hierarchy
[`GeometricMeshHierarchy`](@ref)`(Ωₕ, levels)`. Each level `l` is rediscretised, so its
form is `build(gridspace(H[l]))`, applied through [`matrix_free_operator`](@ref) under the
backend's execution policy, and smoothed by `smoother(op)`. Levels are joined by
[`prolongate!`](@ref) and [`coarsen!`](@ref) (`Pᵀ`, unscaled, since Bramble's forms carry
the discrete measure). The coarsest level's matrix is assembled, stored dense and
LU-factorised once, and solved exactly.

A V-cycle on level `l` smooths `ν₁` times, restricts the residual, runs one cycle on level
`l - 1` from zero (two for a W-cycle), adds the prolongated correction and smooths `ν₂`
times with the adjoint smoother (a [`RedBlackGaussSeidel`](@ref) runs black then red). With
`ν₁ == ν₂` and an SPD `A`, the V- and W-cycles are symmetric positive definite, and
conjugate gradients may use them (`Pl = P` in `LinearSolve`'s `KrylovJL_CG`). The full
multigrid cycle `:FMG` ([`fmg!`](@ref)) is a linear operator but not symmetric.

`build(W)` must return a square [`BilinearForm`](@ref) on the scalar grid space `W` it is
given, with no Dirichlet rows (a form carries none; `A` is `assemble(build(W))`). Conjugate
gradients needs `A` SPD, such as mass plus diffusion with natural boundary conditions. A
Dirichlet problem needs its constrained rows eliminated or lifted into the right-hand side
before a form without them is passed here. Every grid function in the form (a coefficient
`Rₕ(W, κ)`, say) must be built from `W` inside `build`: one built on the finest space and
captured is accepted on every level and gives a wrong coarse operator.

On meshes whose cells have bounded aspect ratio, CG iteration counts do not grow with the
mesh: with the defaults, mass plus variable diffusion took 6 iterations (relative residual
`1e-8`) at every size from 2D 33² to 513², and 7 from 3D 17³ to 129³, on uniform points each
jittered by up to `±0.3h`; the W-cycle took 5 and full multigrid 4 at 65². The smoothers
are point smoothers, and they stall where cells are stretched (see
[`AbstractSmoother`](@ref)). On the random meshes of `mesh(…, false)`, whose largest cell
aspect ratio grows with `n` (96 at 33², 52600 at 513²), CG took 14–25, 26–37 and 32–116
iterations at 2D 33², 65² and 129² over four draws, and 87 at 513² in one. A random base
mesh refined with [`iterative_refinement!`](@ref) keeps its aspect ratio, but each stretched
base cell becomes a block of stretched cells, and iterations still grow per level: 11, 16, 21, 26, 26 from
17² to 257² from a 2D base of 9² (aspect ratio 10.9); 23 to 56 over 33² to 257² from a
base of 17² (aspect ratio 43); 14 to 29 over 9³ to 65³ from a 3D base of 5³ (aspect ratio
15); 21 to 41 over 17³ to 65³ from a base of 9³ (aspect ratio 36). Red-black Gauss-Seidel
gave the same counts.

Construction builds one form, operator and smoother per level (the default smoother runs
[`max_eigenvalue_estimate`](@ref)'s 20 products per level), assembles the coarsest matrix,
and allocates three work vectors per level. The work vectors live in `P`, so one
preconditioner must not be applied from two tasks at once. On a threaded policy an
application allocates only the task launches of its products with `A` and its transfers:
that grows with the number of products (a W-cycle doubles the cycles per level) but not with
the grid size.

# Arguments
- `build`: A function of a [`ScalarGridSpace`](@ref) `W` returning a square
  [`BilinearForm`](@ref) on `W`.
- `Ωₕ`: The finest mesh.

# Keywords
- `levels`: The number of levels, `Ωₕ` included, an integer (`nothing` by default, for as many as
  coarsening by 2 allows while every axis with more than one point keeps at least three, so
  33² gives 5 levels down to 3², and 97 points, with `96 = 3 ⋅ 2⁵`, give 6 down to 4).
- `cycle`: `:V`, `:W` or `:FMG`, the cycle `ldiv!` runs (`:V` by default).
- `ν₁`, `ν₂`: The number of [`smooth!`](@ref) calls before and after the coarse
  correction (`2` each by default).
- `smoother`: A function of a level's [`MatrixFreeOperator`](@ref) returning its
  [`AbstractSmoother`](@ref) (default: `op -> chebyshev_smoother(op)`, degree 2, which damped
  most per product with `A` in the two-grid measurements of [`chebyshev_smoother`](@ref)). A
  smoother other than the three Bramble provides is post-smoothed with `smooth!` as well,
  so it must be self-adjoint in the `A` inner product for the cycle to be symmetric.

# Returns
- [`GMGPreconditioner`](@ref).

# Throws
- `ArgumentError`: `cycle` is not `:V`, `:W` or `:FMG`, `ν₁` or `ν₂` is negative, or
  `ν₁ + ν₂ == 0` (no smoothing leaves the cycle singular).
- `ArgumentError`: `levels` is not an integer, or is invalid for `Ωₕ` (see
  [`GeometricMeshHierarchy`](@ref)).
- `ArgumentError`: `build(W)` does not return a square form on `W`.
- `ArgumentError`: `smoother(op)` does not return an [`AbstractSmoother`](@ref).
- `ArgumentError`: the coarsest level has more than 4096 points: `Ωₕ`'s point counts minus
  one need a larger power of two as a factor.
- `ArgumentError`: the backend's policy is a [`GpuPolicy`](@ref); device execution is
  tracked on milestone v4.4.0.

# Examples
The V-cycle is symmetric.
```jldoctest
using Bramble, LinearAlgebra
Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (33, 33), (true, true))
P = gmg_preconditioner(W -> form(W, W, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))), Ωₕ)
x, y = rand(33^2), rand(33^2)
(length(P.hierarchy), dot(P \\ x, y) ≈ dot(x, P \\ y))

# output
(5, true)
```

See also: [`GMGPreconditioner`](@ref), [`gmg_solve`](@ref), [`v_cycle!`](@ref),
[`GeometricMeshHierarchy`](@ref).
"""
function gmg_preconditioner(
        build, Ωₕ::AbstractMeshType; levels = nothing, cycle::Symbol = :V,
        ν₁::Integer = 2, ν₂::Integer = 2, smoother = _gmg_default_smoother
)
    cycle in (:V, :W, :FMG) ||
        throw(ArgumentError("gmg_preconditioner needs cycle :V, :W or :FMG, got :$cycle"))
    (ν₁ >= 0 && ν₂ >= 0 && ν₁ + ν₂ >= 1) || throw(
        ArgumentError("gmg_preconditioner needs ν₁, ν₂ >= 0 and ν₁ + ν₂ >= 1, got $ν₁ and $ν₂"),
    )
    levels === nothing || levels isa Integer ||
        throw(
            ArgumentError("gmg_preconditioner needs an integer number of levels, got $levels"),
        )
    L = levels === nothing ? _gmg_default_levels(Ωₕ) : Int(levels)
    H = GeometricMeshHierarchy(Ωₕ, L)
    nc = npoints(H[1])
    nc <= _GMG_MAX_COARSE_DOFS || _throw_gmg_coarse(H)
    ops = [matrix_free_operator(_gmg_level_form(build, H[l], l)) for l in 1:L]
    smoothers = [_gmg_level_smoother(smoother, op, l) for (l, op) in enumerate(ops)]
    T = eltype(first(ops))
    coarse = lu(Matrix{T}(assemble(first(ops).form)))
    vec_of(n) = zeros(T, n)
    x = [vec_of(l < L || L == 1 ? npoints(H[l]) : 0) for l in 1:L]
    b = [vec_of(npoints(H[l])) for l in 1:L]
    r = [vec_of(l > 1 ? npoints(H[l]) : 0) for l in 1:L]
    return GMGPreconditioner{
        T, typeof(H), eltype(ops), eltype(smoothers), typeof(coarse), eltype(x)}(
        H, ops, smoothers, coarse, x, b, r, cycle, Int(ν₁), Int(ν₂)
    )
end

# The form `build` returns on level `l`, checked to be square on that level's mesh.
function _gmg_level_form(build, Ωl, l)
    a = build(gridspace(Ωl))
    a isa BilinearForm || throw(
        ArgumentError(
        "gmg_preconditioner: build(W) must return a BilinearForm, got a " *
        "$(nameof(typeof(a))) on level $l",
    ),
    )
    Wu, Wv = trial_space(a), test_space(a)
    ok = Wu isa ScalarGridSpace && Wv isa ScalarGridSpace && mesh(Wu) === Ωl &&
         mesh(Wv) === Ωl
    ok || throw(
        ArgumentError(
        "gmg_preconditioner: build(W) must return a form whose trial and test spaces are " *
        "scalar grid spaces on the mesh of W (level $l)",
    ),
    )
    return a
end

function _gmg_level_smoother(smoother, op, l)
    s = smoother(op)
    s isa AbstractSmoother || throw(
        ArgumentError(
        "gmg_preconditioner: smoother(op) must return an AbstractSmoother, got a " *
        "$(nameof(typeof(s))) on level $l",
    ),
    )
    return s
end

@noinline function _throw_gmg_coarse(H)
    throw(
        ArgumentError(
        "gmg_preconditioner: the coarsest of $(length(H)) levels has " *
        "$(npoints(H[1], Tuple)) points, more than the $_GMG_MAX_COARSE_DOFS it " *
        "factorises densely. Choose point counts n with n - 1 divisible by a larger " *
        "power of two on every axis with more than one point, so that more levels are " *
        "possible, or ask for more levels if the hierarchy allows them.",
    ),
    )
end

# --- Cycles ------------------------------------------------------------------------ #

"""
    v_cycle!(x, P::GMGPreconditioner, b) -> x

One V-cycle for `A x = b` from the iterate `x`, in place, with `A` the finest level's form
of `P`: `ν₁` smoothing steps, the coarse correction by one V-cycle on the next level from
zero (an exact solve on the coarsest), then `ν₂` adjoint smoothing steps. Allocates nothing
on a [`CpuSerial`](@ref) policy. The work vectors live in `P`, so it must not run on one `P`
from two tasks at once.

# Arguments
- `x`: The iterate, overwritten: a vector of length `ndofs`, or a [`VectorElement`](@ref).
- `P`: The multigrid preconditioner.
- `b`: The right-hand side, not modified: a vector of length `ndofs`, or a
  [`VectorElement`](@ref). It must not alias `x`.

# Returns
- `x`, the iterate after the cycle.

# Throws
- `DimensionMismatch`: `x` or `b` has the wrong length.
- `ArgumentError`: `x` or `b` is not 1-based, or the two may alias.

# Examples
Each V-cycle divides the residual by more than five.
```jldoctest
using Bramble, LinearAlgebra
using Bramble: v_cycle!
Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (33, 33), (true, true))
build(W) = form(W, W, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
P = gmg_preconditioner(build, Ωₕ)
A = assemble(build(gridspace(Ωₕ)))
b, x = rand(33^2), zeros(33^2)
r₀ = norm(b)
v_cycle!(x, P, b)
r₁ = norm(b - A * x)
v_cycle!(x, P, b)
(r₁ < r₀ / 5, norm(b - A * x) < r₁ / 5)

# output
(true, true)
```

See also: [`w_cycle!`](@ref), [`fmg!`](@ref), [`gmg_preconditioner`](@ref).
"""
function v_cycle!(x::AbstractVector, P::GMGPreconditioner, b::AbstractVector)
    xd, bd = _gmg_cycle_data(:v_cycle!, P, x, b)
    _gmg_cycle!(P, length(P.hierarchy), xd, bd, 1)
    return x
end

"""
    w_cycle!(x, P::GMGPreconditioner, b) -> x

One W-cycle for `A x = b` from the iterate `x`, in place: as [`v_cycle!`](@ref), but the
coarse correction on every level below the finest runs two cycles in succession (the second
from the first's result) instead of one. Allocates nothing on a [`CpuSerial`](@ref) policy.

# Arguments
- `x`: The iterate, overwritten: a vector of length `ndofs`, or a [`VectorElement`](@ref).
- `P`: The multigrid preconditioner.
- `b`: The right-hand side, not modified: a vector of length `ndofs`, or a
  [`VectorElement`](@ref). It must not alias `x`.

# Returns
- `x`, the iterate after the cycle.

# Throws
- `DimensionMismatch`: `x` or `b` has the wrong length.
- `ArgumentError`: `x` or `b` is not 1-based, or the two may alias.

# Examples
On two levels the coarse solve is exact, so the W- and V-cycles agree.
```jldoctest
using Bramble
using Bramble: v_cycle!, w_cycle!
Ωₕ = mesh(domain(interval(0.0, 1.0)), 17, true)
P = gmg_preconditioner(W -> form(W, W, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))), Ωₕ; levels = 2)
b = rand(17)
w_cycle!(zeros(17), P, b) ≈ v_cycle!(zeros(17), P, b)

# output
true
```

See also: [`v_cycle!`](@ref), [`fmg!`](@ref), [`gmg_preconditioner`](@ref).
"""
function w_cycle!(x::AbstractVector, P::GMGPreconditioner, b::AbstractVector)
    xd, bd = _gmg_cycle_data(:w_cycle!, P, x, b)
    _gmg_cycle!(P, length(P.hierarchy), xd, bd, 2)
    return x
end

"""
    fmg!(x, P::GMGPreconditioner, b) -> x

Full multigrid for `A x = b`, overwriting `x` (its input is not used): `b` is restricted by
[`coarsen!`](@ref) to every level, the coarsest is solved exactly, and on each finer level
the prolongated solution of the level below starts one V-cycle. The result is linear in `b`
but not symmetric. Allocates nothing on a [`CpuSerial`](@ref) policy.

# Arguments
- `x`: The solution, overwritten: a vector of length `ndofs`, or a [`VectorElement`](@ref).
- `P`: The multigrid preconditioner.
- `b`: The right-hand side, not modified: a vector of length `ndofs`, or a
  [`VectorElement`](@ref). It must not alias `x`.

# Returns
- `x`, the full multigrid approximation of `A⁻¹ b`.

# Throws
- `DimensionMismatch`: `x` or `b` has the wrong length.
- `ArgumentError`: `x` or `b` is not 1-based, or the two may alias.

# Examples
On a smooth right-hand side, one full multigrid cycle leaves a residual more than ten times
smaller than one V-cycle from zero does.
```jldoctest
using Bramble, LinearAlgebra
using Bramble: v_cycle!, fmg!
Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (33, 33), (true, true))
build(W) = form(W, W, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
P = gmg_preconditioner(build, Ωₕ)
A = assemble(build(gridspace(Ωₕ)))
b = A * vec([sin(3x) * cos(2y) for x in range(0, 1, 33), y in range(0, 1, 33)])
r_fmg = norm(b - A * fmg!(zeros(33^2), P, b))
r_v = norm(b - A * v_cycle!(zeros(33^2), P, b))
r_fmg < r_v / 10

# output
true
```

See also: [`v_cycle!`](@ref), [`w_cycle!`](@ref), [`gmg_preconditioner`](@ref).
"""
function fmg!(x::AbstractVector, P::GMGPreconditioner, b::AbstractVector)
    xd, bd = _gmg_cycle_data(:fmg!, P, x, b)
    _gmg_fmg!(P, xd, bd)
    return x
end

function ldiv!(y::AbstractVector, P::GMGPreconditioner, x::AbstractVector)
    n = first(size(P))
    (length(x) == n && length(y) == n) || _throw_gmg_dimmismatch(:ldiv!, P, y, x)
    yd, xd = _mf_data(y), _mf_data(x)
    Base.require_one_based_indexing(yd, xd)
    L = length(P.hierarchy)
    bL = P.b[L]
    copyto!(bL, xd)
    if P.cycle === :FMG
        _gmg_fmg!(P, yd, bL)
    else
        fill!(yd, zero(eltype(yd)))
        _gmg_cycle!(P, L, yd, bL, P.cycle === :W ? 2 : 1)
    end
    return y
end

ldiv!(P::GMGPreconditioner, x::AbstractVector) = ldiv!(x, P, x)

# The raw storage of the finest iterate and right-hand side, once their lengths, indexing
# and aliasing are checked.
@inline function _gmg_cycle_data(fname, P, x, b)
    n = first(size(P))
    (length(x) == n && length(b) == n) || _throw_gmg_dimmismatch(fname, P, x, b)
    xd, bd = _mf_data(x), _mf_data(b)
    Base.require_one_based_indexing(xd, bd)
    Base.mightalias(xd, bd) && throw(ArgumentError("$fname: x and b must not alias"))
    return xd, bd
end

@noinline function _throw_gmg_dimmismatch(fname, P, x, b)
    throw(
        DimensionMismatch(
        "$fname: GMGPreconditioner of size $(size(P)) got vectors of length $(length(x)) " *
        "and $(length(b))",
    ),
    )
end

# One cycle on level `l` for `A_l x = b` from the iterate `x`, with `γ` coarse cycles per
# level (`1` a V-cycle, `2` a W-cycle). The coarsest level is solved exactly.
function _gmg_cycle!(P::GMGPreconditioner, l::Int, x, b, γ::Int)
    if l == 1
        _gmg_coarse_solve!(P, x, b)
        return x
    end
    s, op, r = P.smoothers[l], P.ops[l], P.r[l]
    xc, bc = P.x[l - 1], P.b[l - 1]
    for _ in 1:(P.ν₁)
        smooth!(s, x, b)
    end
    _residual!(r, op, x, b)
    coarsen!(bc, P.hierarchy, l, r)
    fill!(xc, zero(eltype(xc)))
    for _ in 1:γ
        _gmg_cycle!(P, l - 1, xc, bc, γ)
    end
    prolongate!(r, P.hierarchy, l, xc)
    x .+= r
    for _ in 1:(P.ν₂)
        _gmg_post_smooth!(s, x, b)
    end
    return x
end

# The adjoint of `smooth!` in the `A` inner product: Jacobi and Chebyshev are self-adjoint,
# red-black reverses its colour order.
@inline _gmg_post_smooth!(s::AbstractSmoother, x, b) = smooth!(s, x, b)
@inline _gmg_post_smooth!(s::RedBlackGaussSeidel, x, b) = smooth!(s, x, b; reverse = true)

# `x = F⁻¹ x` in place for a dense `F = lu(A)`: LAPACK's row swaps in order, then the unit lower
# and the upper triangular solves. `ldiv!(F, x)` calls LAPACK's `getrs!`, which boxes six
# `Ref`s per call that Julia 1.12 keeps at `--optimize=1`, so the cycles would allocate there;
# the coarsest level is small by construction (`_GMG_MAX_COARSE_DOFS`), so the loops cost nothing.
function _gmg_lu_ldiv!(F, x::AbstractVector)
    A, ipiv = F.factors, F.ipiv
    n = length(x)
    for i in 1:n
        j = ipiv[i]
        j == i || ((x[i], x[j]) = (x[j], x[i]))
    end
    for j in 1:n, i in (j + 1):n

        x[i] -= A[i, j] * x[j]
    end
    for j in n:-1:1
        x[j] /= A[j, j]
        for i in 1:(j - 1)
            x[i] -= A[i, j] * x[j]
        end
    end
    return x
end

# `x = A₁⁻¹ b` on the coarsest level, through the dense factorisation, solved in `P.x[1]`.
@inline function _gmg_coarse_solve!(P::GMGPreconditioner, x, b)
    x₁ = P.x[1]
    copyto!(x₁, b)
    _gmg_lu_ldiv!(P.coarse, x₁)
    x === x₁ || copyto!(x, x₁)
    return x
end

function _gmg_fmg!(P::GMGPreconditioner, x, b)
    H = P.hierarchy
    L = length(H)
    L == 1 && return _gmg_coarse_solve!(P, x, b)
    coarsen!(P.b[L - 1], H, L, b)
    for l in (L - 1):-1:2
        coarsen!(P.b[l - 1], H, l, P.b[l])
    end
    _gmg_coarse_solve!(P, P.x[1], P.b[1])
    for l in 2:L
        xl, bl = l == L ? (x, b) : (P.x[l], P.b[l])
        prolongate!(xl, H, l, P.x[l - 1])
        _gmg_cycle!(P, l, xl, bl, 1)
    end
    return x
end

# --- Solver ------------------------------------------------------------------------- #

"""
    gmg_solve(build, Ωₕ::AbstractMeshType, b; levels = nothing, cycle = :V, ν₁ = 2, ν₂ = 2, smoother = op -> chebyshev_smoother(op), tol = nothing, maxiters = 100) -> VectorElement

The solution of `A u = b` by multigrid iteration, with `A` the matrix of
`build(gridspace(Ωₕ))`: from `u = 0`, cycles of [`gmg_preconditioner`](@ref)`(build, Ωₕ;
levels, cycle, ν₁, ν₂, smoother)` repeat until the residual satisfies
`‖b - A u‖ ≤ tol ‖b‖` (2-norm). A `:V` or `:W` cycle is repeated as itself; `:FMG` runs
[`fmg!`](@ref) once and V-cycles after it. Each cycle costs one more product with `A` for
the residual.

This is a stationary iteration: its contraction per cycle is the cycle's, so where point
smoothers stall (stretched cells, see [`gmg_preconditioner`](@ref)), conjugate gradients
preconditioned by the V-cycle converges in fewer cycles. `A` needs no symmetry here, but the
cycle must contract. As in [`gmg_preconditioner`](@ref), every grid function in the form
must be built from `W` inside `build`.

The residual floors at a multiple of the rounding error of the element type `T` of `A`: in
`Float32` it stalled near `1e-5` relative at 65², so `tol` must stay above that. The default
`√eps(T)` (`1.5e-8` in `Float64`, `3.5e-4` in `Float32`) does.

# Arguments
- `build`: A function of a [`ScalarGridSpace`](@ref) `W` returning a square
  [`BilinearForm`](@ref) on `W`, as in [`gmg_preconditioner`](@ref).
- `Ωₕ`: The finest mesh.
- `b`: The right-hand side: a vector of length `npoints(Ωₕ)` or a [`VectorElement`](@ref),
  such as `assemble(l)` for a linear form `l`.

# Keywords
- `levels`, `cycle`, `ν₁`, `ν₂`, `smoother`: As in [`gmg_preconditioner`](@ref).
- `tol`: The relative residual reached (default: `nothing`, for `√eps(T)` with `T` the
  element type of `A`).
- `maxiters`: The largest number of cycles (default: `100`).

# Returns
- [`VectorElement`](@ref): The solution, an element of the trial space of the finest form.

# Throws
- As [`gmg_preconditioner`](@ref).
- `DimensionMismatch`: `b` has the wrong length.
- `ArgumentError`: `b` is not 1-based, `tol` is not positive, or `maxiters < 1`.
- `ErrorException`: the residual is above `tol ‖b‖` after `maxiters` cycles; the message
  names `tol` and `T`, since a `tol` near the rounding floor of `T` is never reached.

# Examples
```jldoctest
using Bramble, LinearAlgebra
Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (33, 33), (true, true))
build(W) = form(W, W, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
b = rand(33^2)
uₕ = gmg_solve(build, Ωₕ, b; tol = 1e-10)
norm(assemble(build(gridspace(Ωₕ))) * parent(uₕ) - b) <= 1e-10 * norm(b)

# output
true
```

See also: [`gmg_preconditioner`](@ref), [`v_cycle!`](@ref), [`fmg!`](@ref).
"""
function gmg_solve(
        build, Ωₕ::AbstractMeshType, b::AbstractVector; levels = nothing,
        cycle::Symbol = :V, ν₁::Integer = 2, ν₂::Integer = 2,
        smoother = _gmg_default_smoother, tol = nothing, maxiters::Integer = 100
)
    (tol === nothing || (tol isa Real && tol > 0)) ||
        throw(ArgumentError("gmg_solve needs tol > 0, got $tol"))
    maxiters >= 1 || throw(ArgumentError("gmg_solve needs maxiters >= 1, got $maxiters"))
    P = gmg_preconditioner(build, Ωₕ; levels, cycle, ν₁, ν₂, smoother)
    op = last(P.ops)
    T = eltype(op)
    τ = tol === nothing ? sqrt(eps(real(T))) : tol
    n = size(op, 1)
    length(b) == n || throw(
        DimensionMismatch("gmg_solve: a right-hand side of length $(length(b)) for $n unknowns"),
    )
    bd = _mf_data(b)
    Base.require_one_based_indexing(bd)
    uₕ = element(trial_space(op.form), zero(promote_type(eltype(P), eltype(bd))))
    x = parent(uₕ)
    r = similar(x)
    target = τ * norm(bd)
    iszero(target) && return uₕ
    for k in 1:maxiters
        if cycle === :FMG && k == 1
            _gmg_fmg!(P, x, bd)
        else
            _gmg_cycle!(P, length(P.hierarchy), x, bd, cycle === :W ? 2 : 1)
        end
        _residual!(r, op, x, bd)
        norm(r) <= target && return uₕ
    end
    return error(
        "gmg_solve: relative residual $(norm(r) / norm(bd)) after $maxiters cycles, above " *
        "tol = $τ for element type $T. A tol near the rounding floor of $T is never " *
        "reached; otherwise the cycle contracts slowly on this mesh, and conjugate " *
        "gradients preconditioned by gmg_preconditioner converges in fewer cycles.",
    )
end
