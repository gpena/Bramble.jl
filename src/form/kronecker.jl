# kronecker.jl
#
# `is_separable` and `KroneckerLinearOperator`: a matrix-free operator for a `BilinearForm`
# whose assembled matrix is an exact sum of Kronecker products of one-dimensional factors
# (gpena/Bramble.jl#162), so a 200^3 problem stores `3 * 200` numbers per factor instead of
# an `8_000_000^2`-entry sparse matrix.
#
# What this file recognises is deliberately narrow: a term is separable here only when it is
# `innerₕ(u, v)` (identity, i.e. a mass factor, on every axis) or `inner₊` of a backward
# difference along one axis on both sides (what `∇ₕ(u)`/`∇ₕ(v)` expand into, one term per
# axis, once `form` has resolved and simplified the AST -- see `simplifier.jl` and
# `operators/inner.jl`'s `inner_plus`). Both shapes factor as `H_D ⊗ ... ⊗ A_d ⊗ ... ⊗ H_1`:
# a difference/mass matrix on the touched axis, the plain mass (cell-measure) matrix on
# every other axis. A scalar coefficient (literal or `Ref`) wrapping a term does not break
# this -- it factors out of the whole Kronecker product -- so it is stripped and carried
# separately rather than being part of the shape match.
#
# Everything else is refused rather than approximated: a `GridFunctionScale` coefficient (no
# tensor structure), a `RegionRestriction` (Dirichlet rows included -- S5.2 handles
# constraints through the `Kronecker.jl` extension, not here), an `InterpolationNode`
# (cross-mesh, no per-axis submesh), an `InnerGamma` surface weight (a `(D-1)`-dimensional
# integral, no full-`D` factorisation), a composite space (leaves can have different
# meshes), a 1D mesh (nothing to factor), and any node this file does not explicitly
# recognise (forward/centered/star/cross-weighted differences, averages, jumps, a mixed
# multi-axis composition). A false negative here only forgoes the fast path; a false
# positive would build an operator that silently computes the wrong matrix-vector product.
#
# Dirichlet rows are out of scope for this operator: it has no boundary constraint of its
# own. `bramble-plan`'s v3.3.0 subplan S5.2 layers that on top, through the `Kronecker.jl`
# extension.

# --- Flattening a sum into (coefficient, term) pairs -------------------------------- #

"""
    _kron_leaves(op, scales::Tuple) -> Tuple

Flatten `op`'s top-level `OperatorAdd` sum into `(scales, term)` pairs, one per addend,
pushing every scalar factor found along the way -- a literal number or a `Ref` -- into
`scales` instead of leaving it wrapped around the sum. `simplify_ast` (`simplifier.jl`)
already lifts a term's own scalar all the way out (`⟨c * u, v⟩ -> c * ⟨u, v⟩`) and factors a
shared one out of a sum (`c*A + c*B -> c*(A+B)`), so a coefficient can sit above several
addends at once; this walk is what puts it back beside each one without rebuilding the AST.

`_separable_axis` classifies `term` alone; [`is_separable`](@ref) and
[`kronecker_operator`](@ref) multiply by `scales` (via `_kron_coeff`) at `mul!` time, the
same way a live `Ref` coefficient stays live through `assemble!`.
"""
@inline _kron_leaves(op::OperatorAdd, scales::Tuple) = (
    _kron_leaves(op.left_op, scales)..., _kron_leaves(op.right_op, scales)...
)
@inline _kron_leaves(op::OperatorScale, scales::Tuple) = _kron_leaves(op.inner_op, (scales..., op.scalar))
@inline _kron_leaves(op::LazyOp, scales::Tuple) = ((scales, op),)

@inline _kron_coeff_factor(c::Number) = c
@inline _kron_coeff_factor(c::Base.RefValue{<:Number}) = c[]

# `init = 1.0` rather than `true` (the usual empty-product identity elsewhere in this
# package): every use multiplies straight into a `Float64` accumulator, and an empty
# `scales` tuple is the common case (a term with no wrapping scalar at all).
@inline _kron_coeff(scales::Tuple) = prod(_kron_coeff_factor, scales; init = 1.0)

# --- Term classification ------------------------------------------------------------- #

"""
    _separable_axis(term) -> Union{Nothing, Some{Union{Int, Nothing}}}

The axis a recognised separable `term` acts along, wrapped in `Some` to tell "valid, no
axis" apart from "not recognised": `Some(nothing)` for a mass term (`innerₕ(u, v)`,
identity on every axis) and `Some(d)` for a single-direction term (`inner₊` of a backward
difference along axis `d` on both sides -- what `∇ₕ` expands to, one term per axis). Plain
`nothing` for anything else.

Conservative by construction: only the two `BilinearProduct` shapes below have a method:
a bare mass product over plain (non-indexed) trial/test functions, and a directional
product over a `BackwardDifference` wrapping each. Every other node -- including a
`GridFunctionScale`, `RegionRestriction`, `InterpolationNode`, an `InnerGamma` weight, an
indexed (composite) trial or test function, or any other difference/average/jump family --
falls through to the `LazyOp` fallback and answers `nothing`, never a guess.
"""
_separable_axis(::LazyOp) = nothing

@inline function _separable_axis(
        ::BilinearProduct{D, InnerH, <:TrialFunction{D}, <:TestFunction{D}}
) where {D}
    return Some(nothing)
end

@inline function _separable_axis(
        ::BilinearProduct{
        D, InnerPlus{Dim}, <:BackwardDifference{D, Dim, <:TrialFunction{D}},
        <:BackwardDifference{D, Dim, <:TestFunction{D}}
}
) where {D, Dim}
    return Some(Dim)
end

"""
    is_separable(a::BilinearForm) -> Bool

Whether `a`'s resolved AST is a sum of terms each expressible as a Kronecker product of
one-dimensional factors, `H_D ⊗ ... ⊗ A_d ⊗ ... ⊗ H_1`, over a `MeshnD`.

`true` requires every one of the following:

  - `a`'s trial and test space are both a (non-composite) [`ScalarGridSpace`](@ref) sharing
    one mesh, and that mesh is at least two-dimensional (a 1D mesh has nothing to factor).
  - Every addend of the resolved AST, after stripping any constant (literal or `Ref`)
    scalar coefficient, is `innerₕ(u, v)` or `inner₊` of a `D₋` backward difference along
    one axis on both sides -- what `innerₕ(u, v)` and `inner₊(∇ₕ(u), ∇ₕ(v))` resolve to.

A grid-function coefficient, a region restriction (Dirichlet included), an interpolation, a
surface (`InnerGamma`) weight, or any operator family this file does not explicitly
recognise (forward/centered/star/cross-weighted differences, averages, jumps, a mixed
multi-axis composition) all answer `false` -- conservatively: a false negative only forgoes
the Kronecker fast path, so this never claims separability it cannot back up with factors.

# Examples

```julia
Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 7), (false, false))
Wₕ = gridspace(Ωₕ)
is_separable(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))))  # true

fₕ = Rₕ(Wₕ, x -> 1 + x[1])
is_separable(form(Wₕ, Wₕ, (u, v) -> innerₕ(fₕ * u, v)))  # false: grid-function coefficient
```

See also: [`kronecker_operator`](@ref), [`KroneckerLinearOperator`](@ref).
"""
function is_separable(a::BilinearForm{D}) where {D}
    D == 1 && return false
    Wu = trial_space(a)
    Wv = test_space(a)
    Wu isa ScalarGridSpace || return false
    Wv isa ScalarGridSpace || return false
    mesh(Wu) === mesh(Wv) || return false
    for (_, term) in _kron_leaves(resolve_form_ast(a), ())
        _separable_axis(term) === nothing && return false
    end
    return true
end

# --- KroneckerTerm: one addend's per-axis factors ------------------------------------ #

"""
    KroneckerTerm{D, S <: Tuple, F <: Tuple, L}

One separable addend's `D` one-dimensional factor matrices and the (possibly still-`Ref`)
scalar coefficients multiplying it: [`KroneckerLinearOperator`](@ref)'s building block, not
exported. `factors[d]` is either the assembled 1D difference matrix on the touched axis or
the diagonal mass matrix on every other axis, in the order [`kronecker_operator`](@ref)
built them; `scales` is read at `mul!` time through `_kron_coeff` so a `Ref` coefficient
stays live, matching a `BilinearForm`'s own contract. `line` is the host `mul!`'s own copy
of a sparse axis-1 factor (see `_kron_line_operator`), `nothing` for every other term and
on a device-backed operator.
"""
struct KroneckerTerm{D, S <: Tuple, F <: Tuple, L}
    scales::S
    factors::F
    line::L
end

"""
    KroneckerLinearOperator{T, D, TermsT <: Tuple}

A matrix-free linear operator for a separable [`BilinearForm`](@ref) (see
[`is_separable`](@ref)): the sum, over its terms, of a Kronecker product of `D`
one-dimensional factor matrices, applied in one fused pass over the grid
(`LinearAlgebra.mul!(y, K, x)`, or the five-argument `mul!(y, K, x, α, β)` computing
`α * K * x + β * y`) rather than ever materialising the `D`-dimensional matrix: every term
has at most one non-diagonal factor, so each entry of `y` is its own term-weighted
combination of `x` at that grid point and its neighbours along each axis, written once.
For a `200^3` mesh the factors together hold `O(200)` numbers per axis instead of the
assembled matrix's `O(200^3)` stored entries.

Build one with [`kronecker_operator`](@ref). Subtypes `AbstractMatrix{T}` so it plugs into
`LinearProblem`/`KrylovJL_CG` (`LinearSolve.jl`) the same way an assembled matrix does, and
supports `size`, `eltype`, `getindex`, `Base.:*`, three- and five-argument `mul!`, `LinearAlgebra.issymmetric`, and
`SparseMatrixCSC(K)` (an explicit `kron` of the factors, for testing and inspection -- the
very matrix this operator avoids forming).

`K` holds no work buffers: it stores only its `D` one-dimensional factors, so it is
immutable after construction and safe to share across threads (concurrent `mul!` calls on
one `K` never race). The fused pass needs no scratch either: `mul!` allocates nothing on
the host, and is generic over the element type, so ForwardDiff `Dual`s pass through. The
`scratch = (b1, b2)` keyword that `mul!(y, K, x; scratch)` and
`mul!(y, K, x, α, β; scratch)` accept is kept so existing callers still work, and is
ignored.

On a device-backed form (gpena/Bramble.jl#323) the factors are built on the host and then
moved to the space backend's device storage, so `mul!` with device `x`/`y` runs entirely on
the device, as one `KernelAbstractions` kernel with one work item per entry of `y`
(`using KernelAbstractions` required). `getindex` and `SparseMatrixCSC(K)` stay host-only
and throw an `ArgumentError` on such an operator.

Dirichlet rows are out of scope: this operator carries no boundary constraint of its own.
`bramble-plan`'s v3.3.0 subplan S5.2 layers that on top, through the `Kronecker.jl`
extension and its fast-diagonalisation solve.

See also: [`is_separable`](@ref), [`kronecker_operator`](@ref).
"""
struct KroneckerLinearOperator{T, D, TermsT <: Tuple} <: AbstractMatrix{T}
    terms::TermsT
    dims::NTuple{D, Int}
    n::Int
end

@noinline function _throw_not_separable_dim(D::Int)
    throw(
        ArgumentError(
        "kronecker_operator needs at least two dimensions to factor a Kronecker product " *
        "from; got a $(D)D form, which has nothing to factor.",
    ),
    )
end

@noinline function _throw_not_separable_space(Wu, Wv)
    throw(
        ArgumentError(
        "kronecker_operator only supports a scalar (non-composite) trial and test space " *
        "sharing one mesh; got $(typeof(Wu)) and $(typeof(Wv)).",
    ),
    )
end

@noinline function _throw_not_separable_term(term)
    throw(
        ArgumentError(
        "kronecker_operator: the term $(typeof(term)) is not one of the recognised " *
        "separable shapes (innerₕ(u, v), or inner₊ of a backward difference along one " *
        "axis on both sides -- what ∇ₕ(u)/∇ₕ(v) expand into). A grid-function " *
        "coefficient, a region restriction (Dirichlet included), an interpolation, a " *
        "surface weight, a composite space, or a mixed/forward/centered/averaged/jump " *
        "operator all fall outside what this file builds Kronecker factors for.",
    ),
    )
end

"""
    kronecker_operator(a::BilinearForm) -> KroneckerLinearOperator

Build a matrix-free [`KroneckerLinearOperator`](@ref) for the separable bilinear form `a`
(see [`is_separable`](@ref)), without ever assembling the `D`-dimensional matrix.

For each axis `d`, the per-axis mass factor is the diagonal matrix of `d`'s cell measures
(`weights(gridspace(Ωₕ(d)), Innerh())`); the factor on a term's touched axis is instead the
assembled 1D operator `assemble(form(Wₕd, Wₕd, (u, v) -> inner₊(D₋ₓ(u), D₋ₓ(v))))` over
`Wₕd = gridspace(Ωₕ(d))`, cached across terms that share an axis. Factors are always built on
the host (a device mesh through its host mirror) and then converted to the storage
`backend(trial_space(a))` uses, so a device-backed form yields device-resident factors.

# Throws

  - `ArgumentError`: `a` is not separable, naming the offending term (or dimension, or
    space) -- the same check [`is_separable`](@ref) runs, made specific.

# Examples

```julia
Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 7), (false, false))
Wₕ = gridspace(Ωₕ)
a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))
K = kronecker_operator(a)
x = rand(ndofs(Wₕ))
K * x ≈ assemble(a) * x
```

See also: [`is_separable`](@ref), [`KroneckerLinearOperator`](@ref).
"""
function kronecker_operator(a::BilinearForm{D}) where {D}
    D == 1 && _throw_not_separable_dim(D)
    Wu = trial_space(a)
    Wv = test_space(a)
    (Wu isa ScalarGridSpace && Wv isa ScalarGridSpace) || _throw_not_separable_space(Wu, Wv)
    mesh(Wu) === mesh(Wv) || _throw_not_separable_space(Wu, Wv)

    # Factors are always built on the host -- a device-backed mesh's per-axis spaces would
    # otherwise be scalar-indexed by `weights`/`assemble` -- and only then moved to the
    # storage `Wu`'s backend chooses (`_kron_to_storage`). On a host mesh
    # `_host_mirror_mesh` returns the mesh itself and `_kron_to_storage` is the identity,
    # so the host operator is the same object graph it always was.
    be = backend(Wu)
    Ωₕ = _host_mirror_mesh(mesh(Wu))
    axis_spaces = ntuple(d -> gridspace(Ωₕ(d)), Val(D))
    mass_vecs = ntuple(d -> weights(axis_spaces[d], Innerh()), Val(D))

    # Cached across terms that touch the same axis, since two directional terms along the
    # same axis (rare, but not disallowed) would otherwise assemble the identical 1D
    # operator twice.
    diff_mats = Dict{Int, Any}()

    leaves = _kron_leaves(resolve_form_ast(a), ())
    terms = map(leaves) do leaf
        scales, term = leaf
        axis_opt = _separable_axis(term)
        axis_opt === nothing && _throw_not_separable_term(term)
        axis = something(axis_opt)
        factors = ntuple(Val(D)) do d
            if axis !== nothing && d == axis
                get!(diff_mats, d) do
                    Wd = axis_spaces[d]
                    assemble(form(Wd, Wd, (u, v) -> inner₊(D₋ₓ(u), D₋ₓ(v))))
                end
            else
                Diagonal(mass_vecs[d])
            end
        end
        sfactors = map(F -> _kron_to_storage(locality(be), be, F), factors)
        line = _kron_line_operator(sfactors[1])
        KroneckerTerm{D, typeof(scales), typeof(sfactors), typeof(line)}(scales, sfactors, line)
    end

    dims = ndofs(Wu, Tuple)
    n = ndofs(Wu)
    T = eltype(mass_vecs[1])
    return KroneckerLinearOperator{T, D, typeof(terms)}(terms, dims, n)
end

# --- Device-resident factors (gpena/Bramble.jl#323) ---------------------------------- #
#
# On a device-backed space the host-built factors move to device storage, each wrapped in a
# type of its own so `mul!` dispatches on the factor, never on a GPU array type (this file
# names no GPU package). The sparse factor keeps its CSC arrays separately
# (`Int32` indices) because a kernel is handed the raw arrays, never a struct nesting a
# device array -- see `docs/src/internals/gpu.md`. Every 1D factor here is symmetric, so
# column `j` of the CSC storage is also row `j`: the kernel gathers row `j` from column `j`
# without a transpose.

struct _KronDeviceDiagonal{V <: AbstractVector}
    diag::V
end

struct _KronDeviceSparse{V <: AbstractVector, IV <: AbstractVector}
    colptr::IV
    rowval::IV
    nzval::V
end

@inline _kron_to_storage(::HostLocality, be, F) = F

function _kron_device_vector(be, h::AbstractVector)
    v = vector(be, length(h))
    copyto!(v, Array(h))  # `h` may be a lazy `SeparableWeights`: tabulate it on the host
    return v
end

function _kron_device_index(like::AbstractVector, h::AbstractVector)
    v = similar(like, Int32, length(h))
    copyto!(v, Int32.(h))
    return v
end

function _kron_to_storage(::DeviceLocality, be, F::Diagonal)
    return _KronDeviceDiagonal(_kron_device_vector(be, F.diag))
end

function _kron_to_storage(::DeviceLocality, be, F::SparseMatrixCSC)
    nz = _kron_device_vector(be, nonzeros(F))
    return _KronDeviceSparse(
        _kron_device_index(nz, SparseArrays.getcolptr(F)), _kron_device_index(nz, rowvals(F)), nz
    )
end

@noinline function _throw_kron_device_entry()
    throw(
        ArgumentError(
        "this KroneckerLinearOperator holds device-resident factors; reading single " *
        "entries (getindex) or materialising SparseMatrixCSC(K) is host-only. Build the " *
        "operator from a host-backed form for inspection.",
    ),
    )
end

@inline _kron_entry(F::AbstractMatrix, i::Int, j::Int) = F[i, j]
_kron_entry(::Union{_KronDeviceDiagonal, _KronDeviceSparse}, ::Int, ::Int) = _throw_kron_device_entry()

# --- Fused application: one pass over `y` -------------------------------------------- #
#
# `kronecker_operator` builds every term with at most one non-diagonal factor: the mass
# term has a `Diagonal` on every axis, and a directional term along axis `d` has the 1D
# difference matrix on `d` and a `Diagonal` everywhere else. So entry `I = (i_1, ..., i_D)`
# of `K * x` is
#
#     sum_t c_t * prod_{e != d_t} H_e[i_e] * sum_j A_{d_t}[i_{d_t}, j] * x[I with i_{d_t} = j]
#
# (the inner sum is just `x[I]` for the mass term): one read of `x` around `I` per term,
# weighted by products of per-axis diagonal entries. `mul!` evaluates that directly and
# writes each entry of `y` once, instead of sweeping the whole array once per axis per term.
# Which factor of a term is the sparse one is read off the factor types, so the whole
# evaluation is resolved at compile time; the tuples are peeled recursively for the same
# reason `_fold_taps` (`stencil_eval.jl`) peels its taps.
#
# Host: one grid line along axis 1 at a time (`off + 1:off + m`, contiguous). Per term, the
# diagonal factors on axes `2:D` fold into one scalar line weight, and the term adds a
# `@simd` pass over the line -- one per stored entry of the sparse factor's row when that
# factor sits on an axis `e >= 2` (reading the neighbouring line `stride_e` away), or one
# tridiagonal sweep along the line when it sits on axis 1 (`_KronTridiag`). The line is the
# only part of `y` those passes revisit, so `y` goes through memory once. Every 1D factor
# is symmetric, so column `j` of the CSC storage is row `j` (the same fact the device kernel
# relies on).

@inline _kron_diag(F::Diagonal) = F.diag
# The mass factor's diagonal is a one-axis `SeparableWeights`: index its own vector rather
# than going through its `CartesianIndex` `getindex` in the inner loop.
@inline _kron_diag(F::Diagonal{<:Any, <:SeparableWeights{1}}) = F.diag.factors[1]

# One axis `e >= 2` of a term on the line whose axis-`e` index is `i`: a diagonal factor
# contributes its entry to the line weight; the sparse factor contributes weight one and is
# returned, with `i` and the axis stride, as the line's neighbour axis.
@inline _kron_line_split(F::Diagonal, i::Int, ::Int) = (_kron_diag(F)[i], nothing)
@inline _kron_line_split(F::SparseMatrixCSC, i::Int, stride::Int) = (one(eltype(F)), (F, i, stride))

# At most one sparse factor per term, so at most one side is ever not `nothing`; two sparse
# factors throw, since `kronecker_operator` never builds such a term.
@noinline function _throw_kron_two_sparse()
    throw(
        ArgumentError(
        "KroneckerLinearOperator term has more than one non-diagonal factor; " *
        "kronecker_operator only builds terms with at most one.",
    ),
    )
end

@inline _kron_pick(::Nothing, ::Nothing) = nothing
@inline _kron_pick(a, ::Nothing) = a
@inline _kron_pick(::Nothing, b) = b
_kron_pick(::Tuple, ::Tuple) = _throw_kron_two_sparse()

@inline _kron_line_fold(::Tuple{}, ::Tuple{}, ::Tuple{}) = (true, nothing)
@inline function _kron_line_fold(Fs::Tuple, Js::Tuple, ss::Tuple)
    w, sp = _kron_line_split(Fs[1], Js[1], ss[1])
    wr, spr = _kron_line_fold(Base.tail(Fs), Base.tail(Js), Base.tail(ss))
    return (w * wr, _kron_pick(sp, spr))
end

# A host sparse axis-1 factor stored as its diagonal and sub-diagonal when it is
# tridiagonal -- which the `inner₊(D₋ₓ(u), D₋ₓ(v))` matrix `kronecker_operator` builds is --
# so the axis-1 term runs as a `@simd` loop along the line rather than a CSC row gather,
# which took 18-20 ms of a 23-26 ms 2D 3000^2 / 3D 200^3 `Float32` `mul!` (2026-09-24). Any
# other sparsity keeps the CSC matrix itself. `sub[i] = F[i + 1, i]`, which equals
# `F[i, i + 1]` since every factor is symmetric.
struct _KronTridiag{V <: AbstractVector}
    dg::V
    sub::V
end

_kron_line_operator(::Any) = nothing
function _kron_line_operator(F::SparseMatrixCSC)
    m = size(F, 1)
    rows = rowvals(F)
    for j in 1:m
        for k in nzrange(F, j)
            abs(rows[k] - j) <= 1 || return F
        end
    end
    return _KronTridiag([F[i, i] for i in 1:m], [F[i + 1, i] for i in 1:(m - 1)])
end

# Mass term: diagonal on axis 1, no neighbour axis.
@inline function _kron_line!(y, F1::Diagonal, ::Nothing, ::Nothing, x, s, off::Int, m::Int)
    h = _kron_diag(F1)
    @inbounds @simd for i in 1:m
        y[off + i] += (s * h[i]) * x[off + i]
    end
    return y
end

# Directional term along an axis `e >= 2`: one pass per stored entry of row `ie`.
@inline function _kron_line!(y, F1::Diagonal, ::Nothing, sp::Tuple, x, s, off::Int, m::Int)
    F, ie, stride = sp
    h = _kron_diag(F1)
    rows = rowvals(F)
    vals = nonzeros(F)
    @inbounds for k in nzrange(F, ie)
        c = s * vals[k]
        xoff = off + (rows[k] - ie) * stride
        @simd for i in 1:m
            y[off + i] += (c * h[i]) * x[xoff + i]
        end
    end
    return y
end

# Directional term along axis 1, tridiagonal factor: ends by hand, interior as one loop.
@inline function _kron_line!(y, ::SparseMatrixCSC, L::_KronTridiag, ::Nothing, x, s, off::Int, m::Int)
    dg, sub = L.dg, L.sub
    @inbounds if m == 1
        y[off + 1] += s * (dg[1] * x[off + 1])
    else
        y[off + 1] += s * (dg[1] * x[off + 1] + sub[1] * x[off + 2])
        @simd for i in 2:(m - 1)
            y[off + i] += s * (sub[i - 1] * x[off + i - 1] + dg[i] * x[off + i] + sub[i] * x[off + i + 1])
        end
        y[off + m] += s * (sub[m - 1] * x[off + m - 1] + dg[m] * x[off + m])
    end
    return y
end

# Directional term along axis 1, any other sparsity: gather row `i` of the factor.
@inline function _kron_line!(y, ::SparseMatrixCSC, F1::SparseMatrixCSC, ::Nothing, x, s, off::Int, m::Int)
    rows = rowvals(F1)
    vals = nonzeros(F1)
    @inbounds for i in 1:m
        acc = zero(eltype(y))
        for k in nzrange(F1, i)
            acc += vals[k] * x[off + rows[k]]
        end
        y[off + i] += s * acc
    end
    return y
end

@inline _kron_line_terms!(y, ::Tuple{}, ::Tuple{}, x, Js::Tuple, ss::Tuple, off::Int, m::Int) = y
@inline function _kron_line_terms!(
        y, terms::Tuple{KroneckerTerm, Vararg{KroneckerTerm}}, cs::Tuple, x, Js::Tuple, ss::Tuple,
        off::Int, m::Int
)
    F = terms[1].factors
    w, sp = _kron_line_fold(Base.tail(F), Js, ss)
    _kron_line!(y, F[1], terms[1].line, sp, x, cs[1] * w, off, m)
    return _kron_line_terms!(y, Base.tail(terms), Base.tail(cs), x, Js, ss, off, m)
end

# `β == 0` overwrites the line (a `NaN` in `y` does not survive), `β == 1` leaves it.
@inline function _kron_line_init!(y, β, off::Int, m::Int)
    if iszero(β)
        @inbounds @simd for i in 1:m
            y[off + i] = zero(eltype(y))
        end
    elseif !isone(β)
        @inbounds @simd for i in 1:m
            y[off + i] *= β
        end
    end
    return y
end

function _kron_fused!(::HostLocality, y, K::KroneckerLinearOperator, x, cs::Tuple, β)
    dims = K.dims
    m = dims[1]
    ss = Base.front(cumprod(dims))::Tuple{Vararg{Int}}  # strides of axes 2:D
    for (o, J) in enumerate(CartesianIndices(Base.tail(dims)))
        off = (o - 1) * m
        _kron_line_init!(y, β, off, m)
        _kron_line_terms!(y, K.terms, cs, x, Tuple(J), ss, off, m)
    end
    return y
end

# Device: one work item per entry of `y`, in `BrambleKernelAbstractionsExt`. The kernel is
# handed raw arrays -- a diagonal factor as its vector, a sparse one as its
# `(colptr, rowval, nzval)` tuple -- because a struct nesting a device array fails
# `KernelAbstractions` kernel compilation (`docs/src/internals/gpu.md`).
@inline _kron_raw(F::_KronDeviceDiagonal) = F.diag
@inline _kron_raw(F::_KronDeviceSparse) = (F.colptr, F.rowval, F.nzval)

"""
    _launch_kron_fused!(y, x, dims, strides, cs, facs, β, βzero) -> Nothing

Device `y = sum_t cs[t] * (term t of a KroneckerLinearOperator) * x + β * y` in one kernel,
one work item per entry of `y` (so no write conflicts and no atomics). `dims` and
`strides` are the grid shape and the column-major stride of each axis; `facs[t][e]` is term
`t`'s axis-`e` factor as raw arrays, a diagonal as its vector and a symmetric sparse factor
as its `(colptr, rowval, nzval)` CSC tuple, at most one of the latter per term. `βzero`
drops the `β * y` read, so a `NaN` in `y` does not survive.

Requires `using KernelAbstractions`; the real method is supplied by
`BrambleKernelAbstractionsExt`.

# Throws
- `ErrorException`: if `KernelAbstractions` is not loaded.
"""
function _launch_kron_fused!(y, x, dims, strides, cs, facs, β, βzero)
    return error(
        "_launch_kron_fused! has no method loaded. Add `using KernelAbstractions` " *
        "before applying a device-backed KroneckerLinearOperator.",
    )
end

function _kron_fused!(::DeviceLocality, y, K::KroneckerLinearOperator, x, cs::Tuple, β)
    dims = K.dims
    strides = (1, Base.front(cumprod(dims))...)
    facs = map(t -> map(_kron_raw, t.factors), K.terms)
    _launch_kron_fused!(y, x, dims, strides, cs, facs, convert(eltype(y), β), iszero(β))
    return y
end

# A term's coefficient `α * c_t` in the operator's own float type when both are plain
# floats: `_kron_coeff` is a `Float64`, which would put `Float32` host loops in `Float64`
# and would not compile in a kernel on a device with no double precision. Anything else (a
# `Dual` `Ref` coefficient, say) is kept as it is.
@inline _kron_scalar(::Type{T}, c::AbstractFloat) where {T <: AbstractFloat} = convert(T, c)
@inline _kron_scalar(::Type, c) = c

@noinline function _throw_kron_dimmismatch(K::KroneckerLinearOperator, x, y)
    throw(
        DimensionMismatch(
        "KroneckerLinearOperator of size $(size(K)) cannot multiply a vector of length " *
        "$(length(x)) into one of length $(length(y))",
    ),
    )
end

# The three-argument method below is ambiguous against `ReverseDiff.mul!(::TrackedArray,
# ::AbstractMatrix, ::TrackedArray{V, D, 1})` whenever `ReverseDiff` is loaded alongside Bramble: `KroneckerLinearOperator <:
# AbstractMatrix`, so it satisfies ReverseDiff's unconstrained middle argument, while a
# `TrackedVector` (`TrackedArray{V, D, 1}`) satisfies this method's `AbstractVector` on both
# `y` and `x` -- the classic diagonal clash where each method wins on a different argument
# and neither dominates. Unlike the `*` method removed below, this one cannot simply be
# deleted: `mul!` is the primitive `AbstractMatrix` operations are built from, not something
# derived from a richer method the way `K * x` is derived from this `mul!`. Narrowing `y`/`x`
# to something other than `AbstractVector` was considered and rejected: this operator's own
# docstring commits it to plugging into `LinearProblem`/`KrylovJL_CG` "the same way an
# assembled matrix does", and those callers are entitled to pass any `AbstractVector` (a
# view, a solver's own work buffer), not just `Vector`. Resolved in
# `ext/BrambleReverseDiffExt.jl` (gpena/Bramble.jl#295), a weak dependency on `ReverseDiff`
# that defines the disambiguating `mul!(::ReverseDiff.TrackedArray, ::KroneckerLinearOperator,
# ::ReverseDiff.TrackedArray)`, forwarding to `ReverseDiff.record_mul!` so the reverse pass
# stays correct.
#
# `scratch` is accepted and ignored: the fused pass above needs no work vectors, so `mul!`
# allocates nothing on the host with or without it, and callers that pass one keep working.
#
# Five-argument form, `y = α * K * x + β * y`, with `LinearAlgebra`'s semantics: `β == 0`
# (including `false`) overwrites `y`, so a `NaN` already in `y` does not survive; otherwise
# each line of `y` is scaled by `β` before the terms (each scaled by `α`) accumulate into it.
# Without this method `mul!(y, K, x, α, β)` falls to `LinearAlgebra`'s generic `O(n^2)`
# `getindex` loop. `α`/`β` may be `Int` (`semidiscretize_rhs` passes `-1, 1`).
function mul!(
        y::AbstractVector, K::KroneckerLinearOperator{T}, x::AbstractVector, α::Number,
        β::Number; scratch = nothing
) where {T}
    n = K.n
    (length(x) == n && length(y) == n) || _throw_kron_dimmismatch(K, x, y)
    cs = map(t -> _kron_scalar(T, α * _kron_coeff(t.scales)), K.terms)
    _kron_fused!(locality(typeof(y)), y, K, x, cs, β)
    return y
end

# The three-argument form is the `α = true, β = false` case: one code path.
function mul!(
        y::AbstractVector, K::KroneckerLinearOperator, x::AbstractVector; scratch = nothing
)
    return mul!(y, K, x, true, false; scratch = scratch)
end

# Only the one-argument method. `AbstractArray` derives `size(A, i)` from it, and defining
# that second method here invalidates every existing caller of the generic one, which the
# invalidation gate (gpena/Bramble.jl#198) rejects.
Base.size(K::KroneckerLinearOperator) = (K.n, K.n)
Base.eltype(::KroneckerLinearOperator{T}) where {T} = T

"""
    getindex(K::KroneckerLinearOperator, i::Int, j::Int) -> Number

The `(i, j)` entry of the assembled matrix `K` stands for, read off the Kronecker
structure directly (`sum` over terms of the coefficient times the product of each factor's
own `(i_d, j_d)` entry) rather than through `mul!` -- `AbstractMatrix`'s minimal interface,
so `K` prints and indexes like the matrix it factors.
"""
function Base.getindex(K::KroneckerLinearOperator{T, D}, i::Int, j::Int) where {T, D}
    @boundscheck checkbounds(K, i, j)
    Ic = CartesianIndices(K.dims)[i]
    Jc = CartesianIndices(K.dims)[j]
    total = zero(T)
    for term in K.terms
        c = _kron_coeff(term.scales)
        p = one(T)
        for d in 1:D
            p *= _kron_entry(term.factors[d], Ic[d], Jc[d])
        end
        total += c * p
    end
    return total
end

# No `*` method of its own. `KroneckerLinearOperator <: AbstractMatrix`, so `LinearAlgebra`
# already derives `K * x` from the `mul!` above, and defining the two-argument form here was
# ambiguous against any package that dispatches `*` on its own vector type -- `NamedDims`'
# `*(::AbstractMatrix, ::NamedDimsArray{_, _, 1})` among them, which the extension ambiguity
# gate reports whenever such a package is loaded alongside Bramble.

"""
    issymmetric(K::KroneckerLinearOperator) -> Bool

Always `true`: [`kronecker_operator`](@ref) only ever builds a term from `innerₕ(u, v)` or
`inner₊` of the *same* backward difference on the trial and the test side, so every
Kronecker factor -- and therefore every term, and their sum -- is symmetric.
"""
issymmetric(::KroneckerLinearOperator) = true

_kron_as_sparse(F::SparseMatrixCSC) = F
_kron_as_sparse(F::Diagonal) = sparse(F)
_kron_as_sparse(::Union{_KronDeviceDiagonal, _KronDeviceSparse}) = _throw_kron_device_entry()

"""
    SparseMatrixCSC(K::KroneckerLinearOperator) -> SparseMatrixCSC

Materialise `K` as an explicit sparse matrix: the sum, over its terms, of the coefficient
times the Kronecker product of its `D` one-dimensional factors, last axis leftmost
(`A_2D = H_y ⊗ A_x + A_y ⊗ H_x`, matching gpena/Bramble.jl#162's own formula). For testing
and inspection only -- this is exactly the `D`-dimensional matrix [`kronecker_operator`](@ref)
is built to avoid forming.

Built as a single `sparse(I, J, V, n, n)` call over every term's `findnz` triplets (`V`
pre-scaled by that term's coefficient) rather than summing each term's Kronecker product
into an accumulator one term at a time, which reallocates the whole `n x n` matrix per term.
"""
function SparseArrays.SparseMatrixCSC(K::KroneckerLinearOperator{T}) where {T}
    I = Int[]
    J = Int[]
    V = T[]
    for term in K.terms
        c = _kron_coeff(term.scales)
        Aterm = foldl(kron, reverse(map(_kron_as_sparse, term.factors)))
        i, j, v = SparseArrays.findnz(Aterm)
        append!(I, i)
        append!(J, j)
        append!(V, c .* v)
    end
    return sparse(I, J, V, K.n, K.n)
end
