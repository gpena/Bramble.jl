# gmg_smoothers.jl: the smoothers of geometric multigrid (gpena/Bramble.jl#329). Each one
# updates an iterate `x` towards `A x = b` in place through products with a
# `MatrixFreeOperator` and the Jacobi diagonal (`jacobi_preconditioner`), so no level's
# matrix is ever built: damped Jacobi, the Chebyshev recurrence of `chebyshev.jl` from a
# nonzero guess, and red-black Gauss-Seidel.
#
# Red-black Gauss-Seidel is two masked Jacobi updates per sweep: a residual through `mul!`,
# then an update of one colour's points only. A row cannot be read off the walk, which
# scatters each stencil entry to its row, so a lexicographic Gauss-Seidel is not available
# matrix-free; the masked update is exact Gauss-Seidel in the red-then-black ordering when no
# entry of `A` couples two distinct points of one colour, which the constructor checks on the
# form's stencil offsets.

"""
    AbstractSmoother{T}

A multigrid smoother for the matrix `A` of a [`BilinearForm`](@ref): [`smooth!`](@ref)`(s, x,
b)` moves the iterate `x` towards the solution of `A x = b` in place, damping the
high-frequency part of its error. A subtype stores its operator, the Jacobi diagonal and its
work vectors, so `smooth!` allocates nothing on a serial policy, and one smoother must not be
used from two tasks at once.

All three smoothers are point smoothers: each unknown is updated from its own equation. They
damp the high frequencies of an isotropic problem, but where cells are stretched the error
that oscillates only along the long side of a cell is smooth for them and oscillatory for the
coarse grid, and neither removes it. With one application before and after an exact Galerkin
coarse correction, on mass plus variable diffusion, the two-grid contraction factors quoted
for each smoother (uniform meshes) rose to 0.74–0.93 on a 2D mesh of 33² graded 7.4-fold
along each axis (cell aspect ratio 6.9), and to 0.68–0.995 on the random non-uniform
meshes of `mesh(…, false)` (2D 33², 3D 9³, six seeds), tracking the largest cell aspect
ratio: 0.69 at 36, 0.995 at 27400. This held for every smoother and degree; in 1D, where no
cell is stretched, they did not change.

# Type parameters
- `T`: The element type of `A`.

See also: [`JacobiSmoother`](@ref), [`ChebyshevSmoother`](@ref),
[`RedBlackGaussSeidel`](@ref), [`smooth!`](@ref).
"""
abstract type AbstractSmoother{T} end

Base.eltype(::Type{<:AbstractSmoother{T}}) where {T} = T
Base.size(s::AbstractSmoother) = size(s.op)

"""
    JacobiSmoother{T, Op, V <: AbstractVector{T}} <: AbstractSmoother{T}

Damped Jacobi: one sweep of [`smooth!`](@ref) sets `x ← x + ω D⁻¹ (b - A x)`, with `D` the
diagonal of `A`. A sweep costs one product with `A` and allocates nothing: the residual
vector is stored in the smoother, so one smoother must not be used from two tasks at once.
Build one with [`jacobi_smoother`](@ref).

# Type parameters
- `T`: The element type of `A`.
- `Op`: The operator type, a [`MatrixFreeOperator`](@ref).
- `V`: The vector type of the reciprocal diagonal and the residual.

See also: [`jacobi_smoother`](@ref), [`AbstractSmoother`](@ref).
"""
struct JacobiSmoother{T, Op, V <: AbstractVector{T}} <: AbstractSmoother{T}
    op::Op
    inv_diagonal::V
    ω::T
    sweeps::Int
    r::V
end

"""
    ChebyshevSmoother{T, Op, V <: AbstractVector{T}} <: AbstractSmoother{T}

Chebyshev smoothing: [`smooth!`](@ref) runs `degree` steps of Jacobi-scaled Chebyshev
iteration from the iterate passed in, on the interval ``[λ_{\\max} / 4, λ_{\\max}]`` of the
spectrum of `D⁻¹A` (the recurrence of [`ChebyshevPreconditioner`](@ref)). The error
`x - A⁻¹ b` is multiplied by ``T_k((θ - D⁻¹A) / δ) / T_k(θ / δ)``, with ``θ = 5 λ_{\\max} /
8``, ``δ = 3 λ_{\\max} / 8`` and ``k`` the degree, whose modulus on the interval is at most
``1 / T_k(5 / 3)``. An application costs `degree` products with `A` and allocates nothing:
the two work vectors are stored in the smoother, so one smoother must not be used from two
tasks at once. Build one with [`chebyshev_smoother`](@ref).

# Type parameters
- `T`: The element type of `A`.
- `Op`: The operator type, a [`MatrixFreeOperator`](@ref).
- `V`: The vector type of the reciprocal diagonal and the work vectors.

See also: [`chebyshev_smoother`](@ref), [`AbstractSmoother`](@ref).
"""
struct ChebyshevSmoother{T, Op, V <: AbstractVector{T}} <: AbstractSmoother{T}
    op::Op
    inv_diagonal::V
    λmin::T
    λmax::T
    degree::Int
    r::V
    d::V
end

"""
    RedBlackGaussSeidel{T, Op, V <: AbstractVector{T}, D} <: AbstractSmoother{T}

Red-black Gauss-Seidel on a scalar grid space: the grid points are coloured by the parity of
the sum of their (zero-based) indices, red when it is even, so the first point is red. One
sweep of [`smooth!`](@ref) updates the red points, then the black ones, each by
`x[i] ← x[i] + (b - A x)[i] / A[i, i]` with the residual of the current iterate. As no entry
of `A` couples two distinct points of one colour (checked by
[`red_black_gauss_seidel`](@ref)), each half-sweep is exact Gauss-Seidel on its colour, and
a sweep equals forward Gauss-Seidel in the red-then-black ordering. A sweep costs two
products with `A` and allocates nothing: the residual vector is stored in the smoother, so
one smoother must not be used from two tasks at once. The order is always red then black, so
a cycle that pre- and post-smooths with it is not symmetric; the two-grid factors quoted
below are for that order on both sides.

# Type parameters
- `T`: The element type of `A`.
- `Op`: The operator type, a [`MatrixFreeOperator`](@ref).
- `V`: The vector type of the reciprocal diagonal and the residual.
- `D`: The dimension of the grid.

See also: [`red_black_gauss_seidel`](@ref), [`AbstractSmoother`](@ref).
"""
struct RedBlackGaussSeidel{T, Op, V <: AbstractVector{T}, D} <: AbstractSmoother{T}
    op::Op
    inv_diagonal::V
    dims::NTuple{D, Int}
    sweeps::Int
    r::V
end

# --- Construction ------------------------------------------------------------------- #

"""
    jacobi_smoother(a::BilinearForm; dirichlet = nothing, dirichlet_components = nothing, policy = execution_policy(trial_space(a)), ω = nothing, sweeps = 1) -> JacobiSmoother
    jacobi_smoother(op::MatrixFreeOperator; ω = nothing, sweeps = 1) -> JacobiSmoother

The damped Jacobi smoother of `A = assemble(a; dirichlet, dirichlet_components)`, or of the
matrix `op` stands for: each sweep sets `x ← x + ω D⁻¹ (b - A x)`, with `D = diag(A)` read
off one stencil walk as by [`jacobi_preconditioner`](@ref). Construction allocates two
vectors of length `ndofs` and no product with `A`, so it is cheap enough to build once per
multigrid level. A Dirichlet row of `A` is the identity row, so a sweep moves `x[i]` a
fraction `ω` of the way to `b[i]` there.

For a diagonally dominant form, such as second-order diffusion plus mass, the spectrum of
`D⁻¹A` lies in `(0, 2]`, and a sweep multiplies an eigencomponent of the error by `1 - ωλ`.
In `d` dimensions the components a grid coarsened by 2 cannot represent have `λ` in about
`[1/d, 2]` (the 3-, 5- and 7-point Laplacians), and the default `ω = 2d / (2d + 1)` (`2/3`,
`4/5`, `6/7`) is the weight that minimises `|1 - ωλ|` over that band. With one sweep before
and after an exact Galerkin coarse correction, on mass plus variable diffusion, it gave
two-grid contraction factors 0.36 (2D, 33²) and 0.51 (3D, 9³), against 0.44 and 0.56 for
`ω = 2/3`; in 1D the two are the same weight (0.11 on 65 points, uniform, graded or
random). Mesh non-uniformity with large cell aspect ratios slows every point smoother (see
[`AbstractSmoother`](@ref)).

# Arguments
- `a`: A square bilinear form ([`ndofs`](@ref) of trial and test space equal), on scalar or
  composite spaces.
- `op`: A square [`MatrixFreeOperator`](@ref); its Dirichlet rows are kept.

# Keywords
- `dirichlet`, `dirichlet_components`, `policy`: As in [`matrix_free_operator`](@ref).
- `ω`: The damping weight (default: `nothing`, for `2d / (2d + 1)` with `d` the dimension
  of the test space).
- `sweeps`: The number of sweeps per [`smooth!`](@ref) (default: `1`, so that a multigrid
  cycle's pre- and post-smoothing counts are sweep counts).

# Returns
- [`JacobiSmoother`](@ref).

# Throws
- `ArgumentError`: `sweeps < 1`, or `ω` not finite and positive.
- `ArgumentError`: `policy` is a [`GpuPolicy`](@ref); device execution is tracked on
  milestone v4.4.0.
- `ArgumentError`: `dirichlet_components` names a leaf the test space does not have.
- `DimensionMismatch`: `A` is not square.

# Examples
In 1D the default weight is `2/3`.
```jldoctest
using Bramble, LinearAlgebra
Wₕ = gridspace(mesh(domain(interval(0.0, 1.0)), 11, false))
a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
A = assemble(a)
x, b = rand(ndofs(Wₕ)), rand(ndofs(Wₕ))
x₀ = copy(x)
smooth!(jacobi_smoother(a), x, b) ≈ x₀ .+ (2 / 3) .* (b .- A * x₀) ./ diag(A)

# output
true
```

See also: [`JacobiSmoother`](@ref), [`smooth!`](@ref), [`chebyshev_smoother`](@ref),
[`red_black_gauss_seidel`](@ref).
"""
function jacobi_smoother(
        a::BilinearForm; dirichlet = nothing, dirichlet_components = nothing,
        policy = execution_policy(trial_space(a)), ω = nothing, sweeps::Integer = 1
)
    op = matrix_free_operator(
        a; dirichlet = dirichlet, dirichlet_components = dirichlet_components, policy = policy
    )
    return jacobi_smoother(op; ω = ω, sweeps = sweeps)
end

function jacobi_smoother(
        op::MatrixFreeOperator{T}; ω = nothing, sweeps::Integer = 1
) where {T}
    _check_sweeps(:jacobi_smoother, sweeps)
    D = dim(test_space(op.form))
    w = ω === nothing ? T(2D // (2D + 1)) : T(ω)
    (isfinite(w) && w > 0) ||
        throw(ArgumentError("jacobi_smoother needs a finite ω > 0, got $ω"))
    J = jacobi_preconditioner(op)
    d = J.inv_diagonal
    return JacobiSmoother{T, typeof(op), typeof(d)}(op, d, w, Int(sweeps), similar(d))
end

"""
    chebyshev_smoother(a::BilinearForm; dirichlet = nothing, dirichlet_components = nothing, policy = execution_policy(trial_space(a)), degree = 2, λmax = nothing) -> ChebyshevSmoother
    chebyshev_smoother(op::MatrixFreeOperator; degree = 2, λmax = nothing) -> ChebyshevSmoother

The Chebyshev smoother of `A = assemble(a; dirichlet, dirichlet_components)`, or of the
matrix `op` stands for: [`smooth!`](@ref) runs `degree` steps of the Jacobi-scaled
Chebyshev recurrence of [`chebyshev_preconditioner`](@ref) from the iterate passed in, on
the interval `[λmax / 4, λmax]` of the spectrum of `D⁻¹A`, `D = diag(A)`. The interval is the
upper part of the spectrum a coarser level cannot represent, so the smoother damps all of it
and leaves the rest to the coarse correction. `A` must be SPD with every eigenvalue of
`D⁻¹A` at most `λmax`; one above the interval is amplified.

Each step damps the upper spectrum by more than a Jacobi sweep does for the same product: on
the interval the error contracts by at most `1 / T_k(5/3)` in `k` products, 0.22 at degree 2.
With one application before and after an exact Galerkin coarse correction, on mass plus
variable diffusion on uniform meshes, degree 2 gave two-grid contraction factors 0.055 (1D,
65 points), 0.088 (2D, 33²) and 0.20 (3D, 9³) for four products with `A` per cycle, against
0.062, 0.20 and 0.33 for two sweeps of Jacobi with `ω = 2/3` at the same cost; degree 3 gave
0.034, 0.064 and 0.098 for six. Per product, degree 2 is the fastest or tied in every
dimension, and on a graded 2D mesh too (0.82, against 0.74 at degree 3), so it is the
default.

Construction allocates four vectors of length `ndofs` and, when `λmax` is not given, runs
[`max_eigenvalue_estimate`](@ref)'s 20 products with `A` and allocates its two vectors.

# Arguments
- `a`: A square bilinear form ([`ndofs`](@ref) of trial and test space equal), on scalar or
  composite spaces, whose matrix is SPD.
- `op`: A square [`MatrixFreeOperator`](@ref); its Dirichlet rows are kept.

# Keywords
- `dirichlet`, `dirichlet_components`, `policy`: As in [`matrix_free_operator`](@ref).
- `degree`: The number of Chebyshev steps, and of products with `A`, per
  [`smooth!`](@ref) (default: `2`).
- `λmax`: The top of the interval, an upper bound for the spectrum of `D⁻¹A`, used as given
  (default: `nothing`, for [`max_eigenvalue_estimate`](@ref)`(op; preconditioner = J)` with
  `J` the Jacobi preconditioner of `op`).

# Returns
- [`ChebyshevSmoother`](@ref).

# Throws
- `ArgumentError`: `degree < 1`, or `λmax` not finite and positive.
- `ArgumentError`: `policy` is a [`GpuPolicy`](@ref); device execution is tracked on
  milestone v4.4.0.
- `ArgumentError`: `dirichlet_components` names a leaf the test space does not have.
- `DimensionMismatch`: `A` is not square.

# Examples
At degree 1 the smoother is damped Jacobi with `ω = 1 / θ = 8 / (5 λmax)`.
```jldoctest
using Bramble
Wₕ = gridspace(mesh(domain(interval(0.0, 1.0)), 11, false))
a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
x, b = rand(ndofs(Wₕ)), rand(ndofs(Wₕ))
y = copy(x)
smooth!(chebyshev_smoother(a; degree = 1, λmax = 2.0), x, b) ≈ smooth!(jacobi_smoother(a; ω = 0.8), y, b)

# output
true
```

See also: [`ChebyshevSmoother`](@ref), [`smooth!`](@ref), [`chebyshev_preconditioner`](@ref).
"""
function chebyshev_smoother(
        a::BilinearForm; dirichlet = nothing, dirichlet_components = nothing,
        policy = execution_policy(trial_space(a)), degree::Integer = 2, λmax = nothing
)
    op = matrix_free_operator(
        a; dirichlet = dirichlet, dirichlet_components = dirichlet_components, policy = policy
    )
    return chebyshev_smoother(op; degree = degree, λmax = λmax)
end

function chebyshev_smoother(
        op::MatrixFreeOperator{T}; degree::Integer = 2, λmax = nothing
) where {T}
    degree >= 1 || throw(ArgumentError("chebyshev_smoother needs degree >= 1, got $degree"))
    J = jacobi_preconditioner(op)
    λ = λmax === nothing ? max_eigenvalue_estimate(op; preconditioner = J) : T(λmax)
    (isfinite(λ) && λ > 0) ||
        throw(ArgumentError("chebyshev_smoother needs a finite λmax > 0, got $λ"))
    d = J.inv_diagonal
    return ChebyshevSmoother{T, typeof(op), typeof(d)}(
        op, d, λ / 4, λ, Int(degree), similar(d), similar(d)
    )
end

"""
    red_black_gauss_seidel(a::BilinearForm; dirichlet = nothing, policy = execution_policy(trial_space(a)), sweeps = 1) -> RedBlackGaussSeidel
    red_black_gauss_seidel(op::MatrixFreeOperator; sweeps = 1) -> RedBlackGaussSeidel

The red-black Gauss-Seidel smoother of `A = assemble(a; dirichlet)`, or of the matrix `op`
stands for (see [`RedBlackGaussSeidel`](@ref)): each sweep updates the points whose
zero-based indices sum to an even number, then the rest, by a masked Jacobi step with the
current residual, which is exact Gauss-Seidel on each colour because no entry of `A` couples
two distinct points of one colour. That holds for the 3-, 5- and 7-point forms of
second-order diffusion and for mass terms; a form with a mixed difference such as
`D₋ᵧ(D₋ₓ(u))` (a 9-point box) couples diagonal neighbours, which share a colour, and is
refused. The check reads the offsets of every term's trial and test sides
(`stencil_offsets`) and ignores collapsed axes (one point), where no entry lands off
the axis. A Dirichlet row of `A` is the identity row, so a sweep sets `x[i] = b[i]` there.

Construction allocates two vectors of length `ndofs` and the small offset lists, and no
product with `A`, so it is cheap enough to build once per multigrid level. Red-black is
defined on one grid: a composite space, whose leaves couple at a shared point, is refused.
With one sweep before and after an exact Galerkin coarse correction, on mass plus variable
diffusion on uniform meshes, it gave two-grid contraction factors 0.002 (1D, 65 points),
0.063 (2D, 33²) and 0.18 (3D, 9³) for four products with `A` per cycle, close to degree-2
[`chebyshev_smoother`](@ref) at the same cost. The default `sweeps = 1` makes a multigrid
cycle's pre- and post-smoothing counts sweep counts.

# Arguments
- `a`: A square bilinear form on a scalar grid space, with the same mesh for trial and test
  space and no interpolation (`πₕ`).
- `op`: A [`MatrixFreeOperator`](@ref) of such a form; its Dirichlet rows are kept.

# Keywords
- `dirichlet`, `policy`: As in [`matrix_free_operator`](@ref).
- `sweeps`: The number of red-then-black sweeps per [`smooth!`](@ref) (default: `1`).

# Returns
- [`RedBlackGaussSeidel`](@ref).

# Throws
- `ArgumentError`: `sweeps < 1`.
- `ArgumentError`: the form couples two distinct points of one colour, is on a composite
  space, has an interpolation, or its trial and test meshes differ in shape.
- `ArgumentError`: `policy` is a [`GpuPolicy`](@ref); device execution is tracked on
  milestone v4.4.0.
- `DimensionMismatch`: `A` is not square.

# Examples
After a sweep the residual vanishes at the black points, whose equations the black half has
just solved exactly.
```jldoctest
using Bramble
Wₕ = gridspace(mesh(domain(interval(0.0, 1.0)), 11, false))
a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
x, b = rand(ndofs(Wₕ)), rand(ndofs(Wₕ))
smooth!(red_black_gauss_seidel(a), x, b)
maximum(abs, (b - assemble(a) * x)[2:2:end]) < 1e-12

# output
true
```

See also: [`RedBlackGaussSeidel`](@ref), [`smooth!`](@ref), [`jacobi_smoother`](@ref).
"""
function red_black_gauss_seidel(
        a::BilinearForm; dirichlet = nothing, policy = execution_policy(trial_space(a)),
        sweeps::Integer = 1
)
    op = matrix_free_operator(a; dirichlet = dirichlet, policy = policy)
    return red_black_gauss_seidel(op; sweeps = sweeps)
end

function red_black_gauss_seidel(op::MatrixFreeOperator{T}; sweeps::Integer = 1) where {T}
    _check_sweeps(:red_black_gauss_seidel, sweeps)
    a = op.form
    Wu, Wv = trial_space(a), test_space(a)
    (Wu isa ScalarGridSpace && Wv isa ScalarGridSpace) || throw(
        ArgumentError(
        "red_black_gauss_seidel needs a form on scalar grid spaces: the leaves of a " *
        "composite space couple at a shared point, which has one colour",
    ),
    )
    dims = npoints(mesh(Wv), Tuple)
    dims == npoints(mesh(Wu), Tuple) || throw(
        ArgumentError(
        "red_black_gauss_seidel needs trial and test meshes of one shape, got " *
        "$(npoints(mesh(Wu), Tuple)) and $dims",
    ),
    )
    foreach(t -> _rb_check_term(t, dims), _summands(a.ast))
    J = jacobi_preconditioner(op)
    d = J.inv_diagonal
    return RedBlackGaussSeidel{T, typeof(op), typeof(d), length(dims)}(
        op, d, dims, Int(sweeps), similar(d)
    )
end

@inline function _check_sweeps(fname, sweeps)
    sweeps >= 1 || throw(ArgumentError("$fname needs sweeps >= 1, got $sweeps"))
    return nothing
end

# A term of the form, `⟨L u, M v⟩` possibly scaled: at a point `I` its entries sit at rows
# `I + m` and columns `I + l`, `m` and `l` in the offsets of `M` and `L`, so it couples
# points `l - m` apart. An offset along a collapsed axis never lands, and is dropped.
function _rb_check_term(t, dims::NTuple{D, Int}) where {D}
    p = _bare_product(t)
    p isa BilinearProduct || _throw_rb_term(p)
    (_has_test_interp(p) || _has_trial_interp(p)) && throw(
        ArgumentError(
        "red_black_gauss_seidel does not take a form with an interpolation (πₕ): its " *
        "entries have no grid offset to colour",
    ),
    )
    for l in stencil_offsets(p.left_op), m in stencil_offsets(p.right_op)

        o = ntuple(k -> l[k] - m[k], Val(D))
        # an offset reaching past the mesh on some axis never lands, collapsed axes included
        any(k -> abs(o[k]) >= dims[k], 1:D) && continue
        (iseven(sum(o)) && any(!iszero, o)) && _throw_rb_coupling(o)
    end
    return nothing
end

@noinline function _throw_rb_term(p)
    throw(ArgumentError("red_black_gauss_seidel cannot read the offsets of a $(nameof(typeof(p))) term"))
end

@noinline function _throw_rb_coupling(o)
    throw(
        ArgumentError(
        "red_black_gauss_seidel needs a form that couples no two points of one colour, but " *
        "a term couples points $o apart (a mixed difference, say, makes a 9-point box): " *
        "use jacobi_smoother or chebyshev_smoother",
    ),
    )
end

# --- Application -------------------------------------------------------------------- #

"""
    smooth!(s::AbstractSmoother, x, b) -> x

Apply the smoother `s` to the iterate `x` for `A x = b`, in place: the sweeps of a
[`JacobiSmoother`](@ref) or [`RedBlackGaussSeidel`](@ref), the Chebyshev steps of a
[`ChebyshevSmoother`](@ref). Allocates nothing on a [`CpuSerial`](@ref) policy; on a threaded
one, only the task launches of each product with `A`. The work vectors live in `s`, so one
smoother must not be used from two tasks at once.

# Arguments
- `s`: The smoother.
- `x`: The iterate, overwritten: a vector of length `ndofs`, or a [`VectorElement`](@ref).
- `b`: The right-hand side, not modified: a vector of length `ndofs`, or a
  [`VectorElement`](@ref). It must not alias `x`.

# Returns
- `x`, the smoothed iterate.

# Throws
- `DimensionMismatch`: `x` or `b` has the wrong length.
- `ArgumentError`: `x` or `b` is not 1-based, or the two may alias.

# Examples
One damped Jacobi sweep with `ω = 1/2` on the 1D Laplacian with its highest-frequency error
`(-1)^i`: away from the ends `D⁻¹A` doubles it, so the sweep removes it there exactly.
```jldoctest
using Bramble
Wₕ = gridspace(mesh(domain(interval(0.0, 1.0)), 11, true))
a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
x = [(-1.0)^i for i in 1:11]
smooth!(jacobi_smoother(a; ω = 0.5), x, zeros(11))
maximum(abs, x[2:10]) < 1e-12

# output
true
```

See also: [`AbstractSmoother`](@ref), [`jacobi_smoother`](@ref), [`chebyshev_smoother`](@ref),
[`red_black_gauss_seidel`](@ref).
"""
function smooth! end

function smooth!(s::JacobiSmoother, x::AbstractVector, b::AbstractVector)
    xd, bd = _smoother_data(s, x, b)
    op, dinv, r, ω = s.op, s.inv_diagonal, s.r, s.ω
    for _ in 1:(s.sweeps)
        _residual!(r, op, xd, bd)
        xd .+= ω .* dinv .* r
    end
    return x
end

function smooth!(s::ChebyshevSmoother, x::AbstractVector, b::AbstractVector)
    xd, bd = _smoother_data(s, x, b)
    _chebyshev!(xd, s.op, bd, s.inv_diagonal, s.r, s.d, s.λmin, s.λmax, s.degree, false)
    return x
end

function smooth!(s::RedBlackGaussSeidel, x::AbstractVector, b::AbstractVector)
    xd, bd = _smoother_data(s, x, b)
    op, dinv, r = s.op, s.inv_diagonal, s.r
    for _ in 1:(s.sweeps), colour in (0, 1)

        _residual!(r, op, xd, bd)
        _rb_update!(xd, dinv, r, s.dims, colour)
    end
    return x
end

@inline function _residual!(r, op, xd, bd)
    copyto!(r, bd)
    mul!(r, op, xd, -1, 1)
    return r
end

# `x[i] += r[i] / A[i, i]` at the points of `colour` (`0` red, `1` black): along the first
# axis every other point, starting where the parity of the zero-based index sum matches.
@inline function _rb_update!(xd, dinv, r, dims::NTuple{D, Int}, colour::Int) where {D}
    n₁ = dims[1]
    rest = Base.tail(dims)
    lin = LinearIndices(rest)
    @inbounds for J in CartesianIndices(rest)
        base = n₁ * (lin[J] - 1)
        first_i = 1 + mod(sum(Tuple(J); init = 0) - length(rest) + colour, 2)
        for i in first_i:2:n₁
            q = base + i
            xd[q] += dinv[q] * r[q]
        end
    end
    return nothing
end

# The raw storage of `x` and `b`, once their lengths, indexing and aliasing are checked.
@inline function _smoother_data(s::AbstractSmoother, x, b)
    n = length(s.r)
    (length(x) == n && length(b) == n) || _throw_smoother_dimmismatch(s, x, b)
    xd, bd = _mf_data(x), _mf_data(b)
    Base.require_one_based_indexing(xd, bd)
    Base.mightalias(xd, bd) && throw(ArgumentError("smooth!: x and b must not alias"))
    return xd, bd
end

@noinline function _throw_smoother_dimmismatch(s, x, b)
    throw(
        DimensionMismatch(
        "$(nameof(typeof(s))) of size $(size(s)) cannot smooth an iterate of length " *
        "$(length(x)) against a right-hand side of length $(length(b))",
    ),
    )
end
