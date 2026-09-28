# chebyshev.jl: Chebyshev polynomial preconditioning of a `MatrixFreeOperator`
# (gpena/Bramble.jl#327). The preconditioner applies a fixed polynomial in `D⁻¹A`, with `D` the
# Jacobi diagonal, through the three-term Chebyshev recurrence (`_chebyshev!`), so it needs only
# products with `A`; the top of its interval comes from power iteration on the same operator
# (`max_eigenvalue_estimate`). `_chebyshev!` also takes a nonzero initial guess, which is the
# smoother form multigrid reuses.
#
# The Jacobi scaling is what makes the polynomial pay on a non-uniform mesh: there the top of
# the spectrum of `A` is a few outliers from the smallest cells, which CG resolves in a few
# iterations anyway, and a polynomial on `[λmax / ratio, λmax]` of `A` leaves the bulk below
# the interval untouched. On the 1D check mesh (257 points) it took CG from 965 to 512
# iterations at degree 4, where the scaled polynomial takes it to 147; for a diagonally
# dominant form, such as second-order diffusion plus mass, `D⁻¹A` has its spectrum in `(0, 2]`
# whatever the cell sizes.

"""
    ChebyshevPreconditioner{T, Op, V <: AbstractVector{T}} <: AbstractMatrixFreePreconditioner{T}

The Chebyshev preconditioner of an SPD operator `A`, scaled by its diagonal `D`:
`ldiv!(y, P, x)` sets `y = p(D⁻¹A) D⁻¹ x`, the result of `degree` steps of Jacobi-preconditioned
Chebyshev iteration for `A y = x` from `y = 0` on the interval ``[λ_{\\min}, λ_{\\max}]``. The
residual polynomial ``1 - λ p(λ)`` is the scaled Chebyshev polynomial
``T_k((θ - λ) / δ) / T_k(θ / δ)``, with ``θ = (λ_{\\max} + λ_{\\min}) / 2``,
``δ = (λ_{\\max} - λ_{\\min}) / 2`` and ``k`` the degree, whose modulus is below `1` on
``(0, λ_{\\max}]``. So `P⁻¹` is a fixed SPD operator whenever the spectrum of `D⁻¹A` lies in
``(0, λ_{\\max}]``, and plain conjugate gradients stays valid with it. Build one with
[`chebyshev_preconditioner`](@ref).

With Dirichlet rows `A` is not symmetric (only its rows are replaced), and neither is `P⁻¹`:
the Dirichlet part of `x` feeds the interior part of `y`. Conjugate gradients with `P` is then
valid only for a right-hand side and starting guess that vanish on the Dirichlet rows; for any
other it can report convergence on a wrong solution.

An application costs `degree - 1` products with `A` and allocates nothing: the two work
vectors are stored in `P`, so one `P` must not be applied from two tasks at once.

# Type parameters
- `T`: The element type of `A`.
- `Op`: The operator type, a [`MatrixFreeOperator`](@ref).
- `V`: The vector type of the reciprocal diagonal and the work vectors.

See also: [`chebyshev_preconditioner`](@ref), [`max_eigenvalue_estimate`](@ref),
[`AbstractMatrixFreePreconditioner`](@ref).
"""
struct ChebyshevPreconditioner{T, Op, V <: AbstractVector{T}} <: AbstractMatrixFreePreconditioner{T}
    op::Op
    inv_diagonal::V
    λmin::T
    λmax::T
    degree::Int
    r::V
    d::V
end

Base.size(P::ChebyshevPreconditioner) = size(P.op)

"""
    chebyshev_preconditioner(a::BilinearForm; dirichlet = nothing, dirichlet_components = nothing, policy = execution_policy(trial_space(a)), degree = 4, λmax = nothing, ratio = 30) -> ChebyshevPreconditioner
    chebyshev_preconditioner(op::MatrixFreeOperator; degree = 4, λmax = nothing, ratio = 30) -> ChebyshevPreconditioner

The Chebyshev preconditioner of `A = assemble(a; dirichlet, dirichlet_components)`, or of the
matrix `op` stands for: a fixed polynomial in `D⁻¹A`, with `D = diag(A)`, on the interval
`[λmax / ratio, λmax]`. It is applied through products with the matrix-free operator, and `D`
is read off one stencil walk as by [`jacobi_preconditioner`](@ref), so `A` is never built.
`A` must be symmetric positive definite, and every eigenvalue of `D⁻¹A` at most `λmax`: one
above it can make the preconditioner indefinite. With `dirichlet`, use it in conjugate
gradients only for a right-hand side that vanishes on the Dirichlet rows (see
[`ChebyshevPreconditioner`](@ref)).

Eigenvalues below `λmax / ratio` are damped less, not lost, so `ratio` trades how far the
polynomial reaches into the low spectrum against how flat it is on its interval. Construction
allocates three vectors of length `ndofs` and, when `λmax` is not given, the two of
[`max_eigenvalue_estimate`](@ref).

# Arguments
- `a`: A square bilinear form ([`ndofs`](@ref) of trial and test space equal), on scalar or
  composite spaces, whose matrix is SPD.
- `op`: A square [`MatrixFreeOperator`](@ref); its Dirichlet rows are kept.

# Keywords
- `dirichlet`, `dirichlet_components`, `policy`: As in [`matrix_free_operator`](@ref).
- `degree`: The number of Chebyshev steps, the degree of the residual polynomial; an
  application costs `degree - 1` products with `A` (default: `4`).
- `λmax`: The top of the interval, an upper bound for the spectrum of `D⁻¹A`, used as given
  (default: `nothing`, for [`max_eigenvalue_estimate`](@ref)`(op; preconditioner = J)` with
  `J` the Jacobi preconditioner of `op`, which carries its own safety factor).
- `ratio`: The ratio of the top of the interval to its bottom (default: `30`).

# Returns
- [`ChebyshevPreconditioner`](@ref): `ldiv!(y, P, x)` gives `p(D⁻¹A) D⁻¹ x`.

# Throws
- `ArgumentError`: `degree < 1`, `ratio <= 1`, or `λmax` not finite and positive.
- `ArgumentError`: `policy` is a [`GpuPolicy`](@ref); device execution is tracked on
  milestone v4.4.0.
- `ArgumentError`: `dirichlet_components` names a leaf the test space does not have.
- `DimensionMismatch`: `A` is not square.

# Examples
With `degree = 1` the polynomial is the constant ``1 / θ``, so `P` is Jacobi scaled by
``2 / (λ_{\\max} + λ_{\\min})``.
```jldoctest
using Bramble, LinearAlgebra
Wₕ = gridspace(mesh(domain(interval(0.0, 1.0)), 11, false))
a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
P = chebyshev_preconditioner(a; degree = 1, λmax = 2.0, ratio = 3)
x = rand(ndofs(Wₕ))
P \\ x ≈ (x ./ diag(assemble(a))) ./ (4 / 3)

# output
true
```

See also: [`ChebyshevPreconditioner`](@ref), [`max_eigenvalue_estimate`](@ref),
[`jacobi_preconditioner`](@ref).
"""
function chebyshev_preconditioner(
        a::BilinearForm; dirichlet = nothing, dirichlet_components = nothing,
        policy = execution_policy(trial_space(a)), degree::Integer = 4, λmax = nothing,
        ratio::Real = 30
)
    op = matrix_free_operator(
        a; dirichlet = dirichlet, dirichlet_components = dirichlet_components, policy = policy
    )
    return chebyshev_preconditioner(op; degree = degree, λmax = λmax, ratio = ratio)
end

function chebyshev_preconditioner(
        op::MatrixFreeOperator{T}; degree::Integer = 4, λmax = nothing, ratio::Real = 30
) where {T}
    n = op.nrows
    n == op.ncols || _throw_chebyshev_nonsquare(op)
    degree >= 1 || throw(ArgumentError("chebyshev_preconditioner needs degree >= 1, got $degree"))
    ratio > 1 || throw(ArgumentError("chebyshev_preconditioner needs ratio > 1, got $ratio"))
    J = jacobi_preconditioner(op)
    λ = λmax === nothing ? max_eigenvalue_estimate(op; preconditioner = J) : T(λmax)
    (isfinite(λ) && λ > 0) ||
        throw(ArgumentError("chebyshev_preconditioner needs a finite λmax > 0, got $λ"))
    return ChebyshevPreconditioner{T, typeof(op), Vector{T}}(
        op, J.inv_diagonal, λ / T(ratio), λ, Int(degree), zeros(T, n), zeros(T, n)
    )
end

@noinline function _throw_chebyshev_nonsquare(op)
    throw(DimensionMismatch("Chebyshev preconditioning needs a square operator, got $(size(op))"))
end

@noinline function _throw_chebyshev_dimmismatch(P, x, y)
    throw(
        DimensionMismatch(
        "ChebyshevPreconditioner of size $(size(P)) cannot divide a vector of length " *
        "$(length(x)) into one of length $(length(y))",
    ),
    )
end

function ldiv!(y::AbstractVector, P::ChebyshevPreconditioner, x::AbstractVector)
    n = length(P.r)
    (length(x) == n && length(y) == n) || _throw_chebyshev_dimmismatch(P, x, y)
    yd = _mf_data(y)
    xd = _mf_data(x)
    Base.require_one_based_indexing(yd, xd)
    _chebyshev!(yd, P.op, xd, P.inv_diagonal, P.r, P.d, P.λmin, P.λmax, P.degree, true)
    return y
end

ldiv!(P::ChebyshevPreconditioner, x::AbstractVector) = ldiv!(x, P, x)

"""
    _chebyshev!(y, A, b, dinv, r, d, λmin, λmax, degree, zero_guess::Bool) -> y

`degree` steps of Chebyshev iteration for `A y = b`, preconditioned by the reciprocal
diagonal `dinv`, on the interval `[λmin, λmax]` holding the spectrum of `Diagonal(dinv) * A`
(Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., Algorithm 12.1, with the
preconditioned residual in place of the residual), overwriting `y`. From `y = 0`
(`zero_guess`) the result is `p(D⁻¹A) D⁻¹ b` and `y` may alias `b`; otherwise the iteration
starts from the `y` passed in, which costs one more product and makes it a smoother. `r` and
`d` are work vectors of the length of `b` (the residual and the update), neither aliasing `y`
or `b`. Allocates nothing when `mul!(r, A, d, -1, 1)` does not.
"""
function _chebyshev!(y, A, b, dinv, r, d, λmin, λmax, degree::Int, zero_guess::Bool)
    θ = (λmax + λmin) / 2
    δ = (λmax - λmin) / 2
    σ = θ / δ
    ρ = inv(σ)
    copyto!(r, b)
    zero_guess || mul!(r, A, y, -1, 1)
    d .= dinv .* r ./ θ
    zero_guess ? copyto!(y, d) : (y .+= d)
    for _ in 2:degree
        mul!(r, A, d, -1, 1)
        ρ₊ = inv(2σ - ρ)
        c = ρ₊ * ρ
        e = 2ρ₊ / δ
        d .= c .* d .+ e .* dinv .* r
        y .+= d
        ρ = ρ₊
    end
    return y
end

"""
    max_eigenvalue_estimate(op::MatrixFreeOperator; iterations = 20, preconditioner = nothing) -> T

An upper estimate of the largest eigenvalue of the SPD matrix `A` that `op` stands for, or of
`P⁻¹A` for an SPD `preconditioner` `P`, by power iteration on the matrix-free operator:
`iterations` products `w = A v` (then `w = P⁻¹ w`) from a fixed pseudo-random start, each
normalising `v = w / ‖w‖`, return `1.1 ‖w‖` for the last `w`.

For a symmetric `A`, `‖A v‖ ≤ λ_max` for every unit `v`, so the bare power iterate is a lower
bound, and it approaches `λ_max` slowly when the top eigenvalues cluster, as they do for a
discrete diffusion operator. The factor `1.1` lifts it above `λ_max` once power iteration has
come within 9%. On variable-coefficient diffusion on non-uniform meshes in one to three
dimensions (up to 65² and 17³ points, five meshes each) the default `iterations` came within
8%, scaled by Jacobi or not, so the estimate landed between `1.02 λ_max` and `1.1 λ_max`.
Overshooting is the side a Chebyshev preconditioner or smoother must err on: an eigenvalue
above its interval can make it indefinite, while an interval a tenth too wide only shifts
the polynomial slightly. With a `preconditioner` the operator `P⁻¹A` is not symmetric, only
similar to a symmetric one, so `‖P⁻¹A v‖` may pass its top eigenvalue and the estimate
overshoot a little more. Allocates two vectors of length `ndofs`.

# Arguments
- `op`: A square [`MatrixFreeOperator`](@ref) of an SPD matrix; its Dirichlet rows are kept.

# Keywords
- `iterations`: The number of products with `A` (default: `20`).
- `preconditioner`: `nothing`, or an [`AbstractMatrixFreePreconditioner`](@ref) `P`
  applied in place after each product, to estimate the top of the spectrum of `P⁻¹A`
  (default: `nothing`).

# Returns
- `T`: The estimate, in the element type of `op`.

# Throws
- `ArgumentError`: `iterations < 1`.
- `DimensionMismatch`: `op` is not square.

# Examples
```jldoctest
using Bramble, LinearAlgebra
Wₕ = gridspace(mesh(domain(interval(0.0, 1.0)), 33, false))
a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
λ = eigmax(Symmetric(Matrix(assemble(a))))
λ <= Bramble.max_eigenvalue_estimate(matrix_free_operator(a)) < 1.2λ

# output
true
```

See also: [`chebyshev_preconditioner`](@ref), [`matrix_free_operator`](@ref).
"""
function max_eigenvalue_estimate(
        op::MatrixFreeOperator{T}; iterations::Integer = 20, preconditioner = nothing
) where {T}
    n = op.nrows
    n == op.ncols || _throw_chebyshev_nonsquare(op)
    iterations >= 1 ||
        throw(ArgumentError("max_eigenvalue_estimate needs iterations >= 1, got $iterations"))
    v = rand!(Random.Xoshiro(327), zeros(T, n))
    v .-= one(T) / 2
    v ./= norm(v)
    w = similar(v)
    μ = zero(T)
    for _ in 1:iterations
        mul!(w, op, v)
        preconditioner === nothing || ldiv!(preconditioner, w)
        μ = norm(w)
        iszero(μ) && return μ
        v .= w ./ μ
    end
    return T(11 // 10) * μ
end
