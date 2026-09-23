# ext/BrambleKroneckerExt.jl: `Kronecker.jl` interop and fast diagonalisation for a
# separable `BilinearForm` (S5.2, gpena/Bramble.jl#259, .agents/plans/v3-3-0-memory-
# scaling.md), layered on top of the dependency-free `KroneckerLinearOperator` S5.1 built in
# `src/form/kronecker.jl`.
#
# Two independent pieces, both read `KroneckerLinearOperator`'s own `terms` (each a
# `KroneckerTerm{D}` of `scales` and `factors`, `src/form/kronecker.jl`) rather than
# re-walking the form's AST:
#
#   1. `Kronecker.kronecker(K)`: the same object as `SparseMatrixCSC(K)`
#      (`src/form/kronecker.jl`), built from `Kronecker.jl`'s own `⊗` instead of `kron` --
#      for a single term this stays the lazy `KroneckerProduct` `Kronecker.jl` itself
#      returns from `⊗`; summing more than one term falls back to `Kronecker.jl`'s own
#      `AbstractMatrix` `+`, which materialises (verified directly: `A ⊗ B` agrees with
#      `kron(A, B)`, not `kron(B, A)`, so no axis-order flip is needed against
#      `SparseMatrixCSC(K)`'s own `foldl(kron, reverse(factors))` convention).
#   2. `fdm_solve`: a direct solve for the separable, constant-coefficient system
#      `assemble(a) \ F` (optionally with homogeneous Dirichlet on the whole mesh
#      boundary) by fast diagonalisation -- see the derivation comment below.
#
# `Kronecker.kronecker` is `Kronecker.jl`'s own generic function, extended here like any
# other package extension method. `fdm_solve` is Bramble's: `src/Bramble.jl` declares
# `function fdm_solve end` and exports it, the same shape `_sparspak_factorize` has for the
# Sparspak extension, and the methods below attach to that binding. They must therefore be
# written `function Bramble.fdm_solve(...)`, dot-qualified: an unqualified
# `function fdm_solve(...)` alongside `using Bramble: Bramble` defines a *different*
# function local to this module, which then answers nobody's call to `Bramble.fdm_solve`.
# That is what happened here first, and it is silent -- the extension loads, the tests that
# reach it through `Base.get_extension` pass, and only the exported spelling stays empty.
module BrambleKroneckerExt

using Bramble:
               Bramble,
               BilinearForm,
               is_separable,
               kronecker_operator,
               KroneckerLinearOperator,
               domain,
               interval,
               ×,
               mesh,
               gridspace,
               boundary_symbols,
               form,
               assemble,
               Rₕ,
               inner₊,
               ∇ₕ,
               innerₕ
using Kronecker: Kronecker, ⊗
using LinearAlgebra: Diagonal, Symmetric, eigen, mul!
using SparseArrays: SparseMatrixCSC
using PrecompileTools: @setup_workload, @compile_workload

# --- 1. Conversion to a Kronecker.jl object ------------------------------------------ #

# One term's coefficient times its Kronecker product, last axis leftmost -- the same
# convention `SparseMatrixCSC(K)` uses (`src/form/kronecker.jl`), with `⊗` standing in for
# `kron` (they agree, module docstring above).
@inline function _kron_jl_term(term)
    c = Bramble._kron_coeff(term.scales)
    return c * foldl(⊗, reverse(term.factors))
end

"""
    Kronecker.kronecker(K::KroneckerLinearOperator) -> AbstractMatrix

Convert `K` (see [`kronecker_operator`](@ref)) into the equivalent `Kronecker.jl` object: the
sum, over `K`'s terms, of `coefficient * (F_D ⊗ ... ⊗ F_1)` built with `Kronecker.jl`'s own
`⊗`. A single-term `K` (for instance a bare `innerₕ(u, v)`) stays the lazy
`Kronecker.KroneckerProduct` `⊗` itself returns; summing two or more terms falls back to
`Kronecker.jl`'s own `AbstractMatrix` addition, which materialises a plain `Matrix` -- the
same thing calling `+` on two `Kronecker.jl` objects of unrelated shape does anywhere else,
not a limitation specific to this method.

`collect(Kronecker.kronecker(K))` and `Matrix(Kronecker.kronecker(K))` agree with
`SparseMatrixCSC(K)` (`src/form/kronecker.jl`) -- both are the same sum of Kronecker
products, read off the same `K.terms`.

# Examples

```julia
Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 7), (false, false))
Wₕ = gridspace(Ωₕ)
K = kronecker_operator(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))))
collect(Kronecker.kronecker(K)) ≈ SparseMatrixCSC(K)
```

See also: [`kronecker_operator`](@ref), [`KroneckerLinearOperator`](@ref).
"""
function Kronecker.kronecker(K::KroneckerLinearOperator)
    terms = K.terms
    result = _kron_jl_term(terms[1])
    for term in Base.tail(terms)
        result = result + _kron_jl_term(term)
    end
    return result
end

# --- 2. Fast diagonalisation ---------------------------------------------------------- #
#
# Derivation.
#
# `kronecker_operator` (`src/form/kronecker.jl`) recognises exactly two term shapes, so a
# separable `a`'s assembled matrix is always
#
#     Σ_d c_d * (H_D ⊗ ... ⊗ A_d ⊗ ... ⊗ H_1)   +   c_m * (H_D ⊗ ... ⊗ H_1)
#
# one addend per differentiated axis `d` (`inner₊(∇ₕ(u), ∇ₕ(v))`'s own per-axis addend,
# `A_d` the assembled 1D `inner₊(D₋ₓ(u), D₋ₓ(v))` stiffness) plus, optionally, one mass
# addend (`innerₕ(u, v)`, `H_c` the diagonal cell-measure matrix on every axis `c`). For
# each axis, solve the generalised eigenproblem
#
#     A_d Q_d = H_d Q_d Λ_d,     Q_d' H_d Q_d = I     (Λ_d diagonal)
#
# `LinearAlgebra.eigen(Symmetric(A_d), Symmetric(H_d))` normalises exactly this way (LAPACK's
# symmetric-definite driver, `sygvd`), which is what makes the next step work without an
# extra `H_d` anywhere. Substitute `x = (Q_D ⊗ ... ⊗ Q_1) y` into the directional term and
# left-multiply by `(Q_D ⊗ ... ⊗ Q_1)'`:
#
#     (Q_D ⊗ ... ⊗ Q_1)' (H_D ⊗ ... ⊗ A_d ⊗ ... ⊗ H_1) (Q_D ⊗ ... ⊗ Q_1)
#         = (Q_D' H_D Q_D) ⊗ ... ⊗ (Q_d' A_d Q_d) ⊗ ... ⊗ (Q_1' H_1 Q_1)
#         = I ⊗ ... ⊗ Λ_d ⊗ ... ⊗ I
#
# using `Q_c' H_c Q_c = I` on every axis `c ≠ d` regardless of whether `c` is ever
# differentiated, and the mass addend becomes `I ⊗ ... ⊗ I` outright by the same identity on
# every axis. So in the `y`-basis the whole operator is diagonal, entry `j = (j_1, ..., j_D)`
# reading `Λ_total[j] = c_m + Σ_d c_d * Λ_d[j_d]`, and
#
#     y = ((Q_D ⊗ ... ⊗ Q_1)' F) ./ Λ_total,     x = (Q_D ⊗ ... ⊗ Q_1) y
#
# applied axis by axis (sum factorisation) rather than by ever forming `Q_D ⊗ ... ⊗ Q_1`.
#
# Homogeneous Dirichlet (`dirichlet = :boundary`). `:boundary` marks every point with at
# least one coordinate on an axis's first or last index, so its complement -- the interior --
# is exactly the tensor product, over every axis, of that axis's own interior points
# (`2:end-1`): a point is interior in the domain iff it is interior along *every* axis. For a
# zero boundary value, the interior rows/columns of the unconstrained `A_d`/`H_d` are exactly
# the reduced operator (a boundary column's contribution is its entry times the -- zero --
# boundary unknown, which drops out), so the derivation above runs unchanged on the
# `(n_d - 2)`-sized interior restriction of every per-axis factor, and the full solution is
# the reduced one embedded back with zero at every boundary dof.

@noinline function _throw_fdm_not_separable(a)
    throw(
        ArgumentError(
        "fdm_solve needs a separable, constant-coefficient BilinearForm (see " *
        "`is_separable`): a grid-function coefficient, a region restriction, an " *
        "interpolation, a surface weight, a composite space, or a 1D mesh cannot be " *
        "fast-diagonalised. Got $(typeof(a)).",
    ),
    )
end

@noinline function _throw_fdm_untouched_axis(d::Int)
    throw(
        ArgumentError(
        "fdm_solve needs every axis to carry its own directional term (what ∇ₕ(u)/∇ₕ(v) " *
        "expand into, one term per axis); axis $d has none of its own, only mass " *
        "(innerₕ) contributions, so it cannot be assigned a 1D stiffness factor.",
    ),
    )
end

@noinline function _throw_fdm_bad_dirichlet(dirichlet)
    throw(
        ArgumentError(
        "fdm_solve only supports dirichlet = nothing (unconstrained) or " *
        "dirichlet = :boundary (homogeneous Dirichlet on the whole mesh boundary); got " *
        "$(repr(dirichlet)).",
    ),
    )
end

@noinline function _throw_fdm_length_mismatch(n::Int, m::Int)
    throw(DimensionMismatch("fdm_solve: F has length $m, the operator needs $n"))
end

# Classifies `K`'s own terms once: the per-axis mass vector `H_d`, the per-axis assembled 1D
# stiffness `A_d` (only ever read for an axis some term actually touches), that axis's
# summed directional coefficient, and the summed mass-only coefficient. Reuses
# `KroneckerLinearOperator`'s own factors (built once by `kronecker_operator`,
# gpena/Bramble.jl#162) rather than re-walking the form's AST: `_separable_axis`
# (`src/form/kronecker.jl`) already guarantees a mass term's factor is `Diagonal` on every
# axis and a directional term's is `Diagonal` on every axis but the one it differentiates,
# so which factor is which is read off its type alone.
#
# A device-backed `K` (gpena/Bramble.jl#323) holds `_KronDeviceDiagonal`/`_KronDeviceSparse`
# factors (`src/form/kronecker.jl`); `_fdm_host_factor` brings each back to the host as a
# `Diagonal`/`SparseMatrixCSC` first. They are the 1D factors, O(n_d) per axis, and the
# eigendecomposition below is a host LAPACK call anyway, so this copy is negligible.
_fdm_host_factor(F) = F
_fdm_host_factor(F::Bramble._KronDeviceDiagonal) = Diagonal(Array(F.diag))
function _fdm_host_factor(F::Bramble._KronDeviceSparse)
    m = length(F.colptr) - 1
    return SparseMatrixCSC(m, m, Vector{Int}(Array(F.colptr)), Vector{Int}(Array(F.rowval)),
        Array(F.nzval))
end

function _fdm_axis_data(K::KroneckerLinearOperator{T, D}) where {T, D}
    H = Vector{Any}(undef, D)
    A = Vector{Any}(undef, D)
    touched = falses(D)
    coeff_axis = zeros(T, D)
    coeff_mass = zero(T)
    for term in K.terms
        factors = map(_fdm_host_factor, term.factors)
        c = Bramble._kron_coeff(term.scales)
        axis = 0
        for d in 1:D
            if factors[d] isa Diagonal
                H[d] = factors[d].diag
            else
                axis = d
            end
        end
        if axis == 0
            coeff_mass += T(c)  # keep `T`: a Float64 grid would not reach a Float32-only device
        else
            A[axis] = factors[axis]
            coeff_axis[axis] += c
            touched[axis] = true
        end
    end
    all(touched) || _throw_fdm_untouched_axis(findfirst(!, touched))
    return H, A, coeff_axis, coeff_mass
end

# The generalised eigendecomposition per axis and the combined eigenvalue grid
# `Λ_total[j] = coeff_mass + Σ_d coeff_axis[d] * Λ_d[j_d]` the derivation above needs. `H`
# and `A` are already restricted to the interior when `dirichlet = :boundary` called this.
function _fdm_eigendecompose(H, A, coeff_axis, coeff_mass::T, dims::NTuple{D, Int}) where {T, D}
    decomps = ntuple(Val(D)) do d
        eigen(Symmetric(Matrix(A[d])), Symmetric(Matrix(Diagonal(H[d]))))
    end
    Q = ntuple(d -> decomps[d].vectors, Val(D))
    Λ = fill(coeff_mass, dims)
    for d in 1:D
        shape = ntuple(k -> k == d ? dims[d] : 1, Val(D))
        Λ .+= coeff_axis[d] .* reshape(decomps[d].values, shape)
    end
    return Q, Λ
end

# `Y = M *_d X`: dense matrix `M` applied along axis `d` of the `D`-array `X`, viewed as
# `(pre, dims[d], post)` (`pre = prod(dims[1:d-1])`, `post = prod(dims[d+1:end])`) and
# right-multiplying each `pre x dims[d]` slab by `M'`, so `Y[i, j, k] = Σ_l M[j, l]
# X[i, l, k]`. Dense `reshape`/matrix-multiply rather than `kronecker.jl`'s own
# zero-allocation `_kron_apply_mode!`: `Q_d` is a full (not diagonal or sparse) matrix here,
# and this runs once per `fdm_solve` call, not once per Krylov iteration, so it is not on the
# path that contract measures allocation-free.
function _fdm_apply_mode(X::Array{T}, M::AbstractMatrix, d::Int) where {T}
    dims = size(X)
    pre = prod(dims[1:(d - 1)]; init = 1)
    m = dims[d]
    post = prod(dims[(d + 1):end]; init = 1)
    X3 = reshape(X, pre, m, post)
    Y3 = Array{T}(undef, pre, size(M, 1), post)
    Mt = transpose(M)
    for k in 1:post
        @views Y3[:, :, k] = X3[:, :, k] * Mt
    end
    newdims = ntuple(i -> i == d ? size(M, 1) : dims[i], length(dims))
    return reshape(Y3, newdims)
end

# Device path (gpena/Bramble.jl#323): `X` and `M` are both device arrays, so the mode
# product is one dense device matmul with no host round trip. Axis `d` is brought to the
# front with `permutedims` (a device kernel), `M * X2` runs on the `(m, pre * post)`
# matricisation, and `permutedims` puts the axis back. For `d == 1` no permutation is
# needed at all.
function _fdm_apply_mode(X::AbstractArray{T}, M::AbstractMatrix, d::Int) where {T}
    dims = size(X)
    pre = prod(dims[1:(d - 1)]; init = 1)
    m = dims[d]
    post = prod(dims[(d + 1):end]; init = 1)
    mo = size(M, 1)
    newdims = ntuple(i -> i == d ? mo : dims[i], length(dims))
    if pre == 1
        Y2 = similar(X, T, mo, post)
        mul!(Y2, M, reshape(X, m, post))
        return reshape(Y2, newdims)
    end
    Xp = permutedims(reshape(X, pre, m, post), (2, 1, 3))
    Y2 = similar(X, T, mo, pre * post)
    mul!(Y2, M, reshape(Xp, m, pre * post))
    return reshape(permutedims(reshape(Y2, mo, pre, post), (2, 1, 3)), newdims)
end

# Copies a host matrix/array to storage like the device vector `F` (`similar` + `copyto!`,
# so no GPU package is named here); the identity for a host `F`.
_fdm_to_storage(::Array, A::AbstractArray) = A
function _fdm_to_storage(F::AbstractArray, A::AbstractArray{T}) where {T}
    B = similar(F, T, size(A))
    copyto!(B, Array(A))
    return B
end

# Sum factorisation: `F` into the eigenbasis axis by axis (`Q_d'`), divide by the combined
# eigenvalue grid, transform back (`Q_d`) -- the three steps the derivation comment ends on.
function _fdm_apply(Q::NTuple{D}, Λ::AbstractArray{T, D}, F::AbstractVector, dims::NTuple{D, Int}) where {T, D}
    # Host `F`: `Q`, `Λ` are used as they are. Device `F`: `Q_d`, `Q_d'` (materialised, so
    # the device matmul never sees a lazy `Transpose`) and `Λ` are copied to the device once
    # per call, and every step below stays device-resident.
    Xv = similar(F, T, length(F))
    Xv .= F
    X = reshape(Xv, dims)
    for d in 1:D
        Qt = Xv isa Array ? transpose(Q[d]) : _fdm_to_storage(Xv, Matrix(transpose(Q[d])))
        X = _fdm_apply_mode(X, Qt, d)
    end
    Y = X ./ _fdm_to_storage(Xv, Λ)
    for d in 1:D
        Y = _fdm_apply_mode(Y, _fdm_to_storage(Xv, Q[d]), d)
    end
    return vec(Y)
end

function _fdm_solve_core(K::KroneckerLinearOperator{T, D}, F::AbstractVector, dirichlet) where {T, D}
    length(F) == K.n || _throw_fdm_length_mismatch(K.n, length(F))
    (dirichlet === nothing || dirichlet === :boundary) || _throw_fdm_bad_dirichlet(dirichlet)

    H, A, coeff_axis, coeff_mass = _fdm_axis_data(K)
    dims_full = K.dims

    if dirichlet === :boundary
        rng = ntuple(d -> 2:(dims_full[d] - 1), Val(D))
        dims_solve = ntuple(d -> dims_full[d] - 2, Val(D))
        H = ntuple(d -> H[d][rng[d]], Val(D))
        A = ntuple(d -> A[d][rng[d], rng[d]], Val(D))
        # A view plus broadcast, not `getindex` with ranges: no scalar indexing on a
        # device `F` (gpena/Bramble.jl#323), and the same values on the host.
        Fint = similar(F, T, prod(dims_solve))
        reshape(Fint, dims_solve) .= view(reshape(F, dims_full), rng...)
    else
        dims_solve = dims_full
        Fint = F
    end

    Q, Λ = _fdm_eigendecompose(H, A, coeff_axis, coeff_mass, dims_solve)
    xint = _fdm_apply(Q, Λ, Fint, dims_solve)

    dirichlet === nothing && return xint

    x = fill!(similar(xint, T, K.n), zero(T))
    view(reshape(x, dims_full), rng...) .= reshape(xint, dims_solve)
    return x
end

"""
    fdm_solve(a::BilinearForm, F::AbstractVector; dirichlet = nothing) -> Vector

Solve the separable, constant-coefficient system `assemble(a) \\ F` (or, with
`dirichlet = :boundary`, `assemble(a; dirichlet = :boundary) \\ F` for an `F` that is
already zero on the boundary) by fast diagonalisation instead of a general sparse
factorisation -- see the derivation comment above `fdm_solve` in
`ext/BrambleKroneckerExt.jl`. `a` must be [`is_separable`](@ref) and have every axis
carry its own directional term (what `∇ₕ(u)`/`∇ₕ(v)` expand into).

`dirichlet`:

  - `nothing` (the default): the unconstrained system.
  - `:boundary`: homogeneous Dirichlet on the whole mesh boundary. `F` must already carry
    zero at every boundary dof (matching what `assemble(a; dirichlet = :boundary) \\ F`
    would require of its own right-hand side); the interior solve is embedded back with
    zero on the boundary.

# Throws

  - `ArgumentError`: `a` is not separable, an axis has no directional term of its own, or
    `dirichlet` is neither `nothing` nor `:boundary`.
  - `DimensionMismatch`: `F`'s length does not match `a`'s number of degrees of freedom.

# Examples

```julia
Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (25, 19), (false, false))
Wₕ = gridspace(Ωₕ)
a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
F = rand(ndofs(Wₕ))
fdm_solve(a, F) ≈ assemble(a) \\ F
```

See also: [`is_separable`](@ref), [`kronecker_operator`](@ref).
"""
function Bramble.fdm_solve(a::BilinearForm, F::AbstractVector; dirichlet = nothing)
    is_separable(a) || _throw_fdm_not_separable(a)
    K = kronecker_operator(a)
    return _fdm_solve_core(K, F, dirichlet)
end

"""
    fdm_solve(K::KroneckerLinearOperator, F::AbstractVector) -> Vector

Unconstrained fast-diagonalisation solve directly from an already-built
[`KroneckerLinearOperator`](@ref) (see [`kronecker_operator`](@ref)), reusing its own
factors rather than rebuilding them from a `BilinearForm`. `K` carries no boundary
constraint of its own (see its docstring), so only the unconstrained case is available
here; call the `BilinearForm` method with `dirichlet = :boundary` for homogeneous
Dirichlet.

# Throws

  - `ArgumentError`: some axis of `K` has no directional term of its own.
  - `DimensionMismatch`: `F`'s length does not match `K`'s size.

See also: [`fdm_solve(::BilinearForm, ::AbstractVector)`](@ref).
"""
function Bramble.fdm_solve(K::KroneckerLinearOperator, F::AbstractVector)
    return _fdm_solve_core(K, F, nothing)
end

# Warms this extension's entry points -- `Kronecker.kronecker` and both `fdm_solve` calls
# (unconstrained and `dirichlet = :boundary`) -- on a 2D separable, constant-coefficient
# form, only reachable once `Kronecker` is loaded so only this extension's own precompile
# pass reaches them. Not named in gpena/Bramble.jl#259; added for gpena/Bramble.jl#284.
if Bramble.PRECOMPILE_WORKLOAD
    @setup_workload begin
        Ω = domain(interval(0.0, 1.0) × interval(0.0, 1.0), :boundary =>
            boundary_symbols(interval(0.0, 1.0) × interval(0.0, 1.0)))
        Ωₕ = mesh(Ω, (8, 8), (false, false))
        Wₕ = gridspace(Ωₕ)
        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
        fₕ = Rₕ(Wₕ, x -> 1.0)
        l = form(Wₕ, v -> innerₕ(fₕ, v))
        F = assemble(l)

        @compile_workload begin
            K = kronecker_operator(a)
            Kronecker.kronecker(K)
            Bramble.fdm_solve(a, F)
            Bramble.fdm_solve(a, F; dirichlet = :boundary)
        end
    end
end

end # module BrambleKroneckerExt
