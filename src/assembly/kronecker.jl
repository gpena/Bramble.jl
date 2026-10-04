# kronecker.jl
#
# `is_separable` and `KroneckerLinearOperator`: a matrix-free operator for a `BilinearForm`
# whose assembled matrix is an exact sum of Kronecker products of one-dimensional factors
# (gpena/Bramble.jl#162), so a 200^3 problem stores `3 * 200` numbers per factor instead of
# an `8_000_000^2`-entry sparse matrix.
#
# What factors: a term is separable when the walk in `kronecker_projection.jl`
# (`_kron_project`) projects every node of it onto each axis of the mesh. That covers the
# difference, average, jump and shift nodes along any axis and chains of them, so mixed
# derivatives and advection; an `:interior` restriction; `innerₕ`, `inner₊` and `inner_Γ`
# weights (one Kronecker term per face for the last); grid-function coefficients varying
# along one axis; and plain numbers inside a side. Each factor is the 1D form assembled on
# that axis's own submesh, so a factor need be neither symmetric nor diagonal, and a term
# may have several non-diagonal factors. A scalar coefficient (literal or `Ref`) wrapping a
# term factors out of the whole Kronecker product, so it is stripped and carried separately
# (`_kron_leaves`) and a `Ref` stays live. `kronecker_block.jl` extends this to composite
# spaces whose leaves share one mesh.
#
# What does not factor is a grid-function coefficient varying along several axes or living
# on another mesh, a `Ref` merged inside a side, a region other than `:interior` (Dirichlet
# rows included; `fdm_solve`'s `dirichlet` keyword handles those, not this operator), an
# `InterpolationNode`, a 1D mesh (nothing to factor), a trial and test space on different
# meshes, and the star and cross-weighted differences. Anything the walk has no method for
# is refused rather than approximated. A false negative only forgoes the fast path, while a
# false positive would build an operator that silently computes the wrong matrix-vector
# product.
#
# Dirichlet rows are out of scope for this operator: it has no boundary constraint of its
# own. The `Kronecker.jl` extension adds homogeneous Dirichlet conditions on the whole
# boundary to `fdm_solve`, through its `dirichlet = :boundary` keyword.

# --- Flattening a sum into (coefficient, term) pairs -------------------------------- #

"""
    _kron_leaves(op, scales::Tuple) -> Tuple

Flatten `op`'s top-level `OperatorAdd` sum into `(scales, term)` pairs, one per addend,
pushing every scalar factor found along the way -- a literal number or a `Ref` -- into
`scales` instead of leaving it wrapped around the sum. `simplify_ast` (`simplifier.jl`)
already lifts a term's own scalar all the way out (`⟨c * u, v⟩ -> c * ⟨u, v⟩`) and factors a
shared one out of a sum (`c*A + c*B -> c*(A+B)`), so a coefficient can sit above several
addends at once; this walk is what puts it back beside each one without rebuilding the AST.

`_kron_project` factors `term` alone; [`kronecker_operator`](@ref) multiplies by `scales`
(via `_kron_coeff`) at `mul!` time, the same way a live `Ref` coefficient stays live
through `assemble!`.
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

# Whether a projected term reads a grid function's values, which `_kron_project` copies into
# the factors once: a later edit to that grid function is not seen. A constant (`Number`)
# coefficient is not one.
_kron_reads_coef(op::LazyOp) = hasproperty(op, :inner_op) && _kron_reads_coef(op.inner_op)
_kron_reads_coef(op::BilinearProduct) = _kron_reads_coef(op.left_op) || _kron_reads_coef(op.right_op)
_kron_reads_coef(op::OperatorAdd) = _kron_reads_coef(op.left_op) || _kron_reads_coef(op.right_op)
function _kron_reads_coef(op::GridFunctionScale)
    return !(op.grid_function isa Number) || _kron_reads_coef(op.inner_op)
end

# The node of one side of a product that `_kron_split` has no projection for, by name: the
# innermost node whose own operand still splits.
function _kron_refused_split(op::OperatorAdd)
    _kron_split(op.left_op) === nothing && return _kron_refused_split(op.left_op)
    return _kron_refused_split(op.right_op)
end
function _kron_refused_split(op::LazyOp)
    if hasproperty(op, :inner_op) && _kron_split(op.inner_op) === nothing
        return _kron_refused_split(op.inner_op)
    end
    op isa RegionRestriction && return "RegionRestriction($(repr(op.region)))"
    return string(nameof(typeof(op)))
end

"""
    _kron_refused_node(term, Ωₕ) -> String

What in `term`, a leaf `_kron_project(term, Ωₕ)` answers `nothing` for, has no per-axis
projection, for the error `kronecker_operator` throws.
"""
_kron_refused_node(term::LazyOp, ::Any) = "the node $(nameof(typeof(term)))"
function _kron_refused_node(term::BilinearProduct{D, I}, Ωₕ) where {D, I}
    _kron_inners(I, Ωₕ) === nothing && return "the inner-product weight $(nameof(I))"
    for side in (term.left_op, term.right_op)
        chains = _kron_split(side)
        chains === nothing && return "the node $(_kron_refused_split(side))"
        _kron_coefs(chains, Ωₕ) === nothing &&
            return "the node GridFunctionScale (a coefficient varying along more than one " *
                   "axis, or on another mesh)"
    end
    return "the node RegionRestriction(:interior) (the mesh's :interior marker is not the " *
           "product of its axes' own)"
end

"""
    is_separable(a::BilinearForm) -> Bool

Whether `a`'s resolved AST is a sum of terms each expressible as a sum of Kronecker products
of one-dimensional factors over a `MeshnD`: every addend, its constant (literal or `Ref`)
scalar coefficients stripped, has a projection onto the mesh's axes (`_kron_project`).

`a`'s trial and test space must both be a [`ScalarGridSpace`](@ref) sharing one mesh, at
least two-dimensional (a 1D mesh has nothing to factor), or composite spaces whose leaves
all live on one such mesh; every (test leaf, trial leaf) block of `a` must then be
separable in this sense.

What factors:

  - the difference, average, jump and shift families along any axis, and chains of them
    (`D₋ₓ(D₋ᵧ(u))`), so mixed derivatives and advection terms, whose factors need not be
    symmetric;
  - `innerₕ`, `inner₊` and `inner_Γ` weights, the last as one term per face;
  - a restriction to `:interior`, provided the mesh's `:interior` marker is the geometric
    one (the product of its axes' own interiors);
  - a grid-function coefficient that varies along one axis only, and a plain number inside a
    side;
  - a scalar or `Ref` coefficient around a term.

What does not, and answers `false`:

  - a grid-function coefficient varying along several axes, or living on another mesh;
  - a `Ref` merged inside a side with other terms;
  - a region restriction other than `:interior`, including Dirichlet rows;
  - an interpolation, a 1D mesh, trial and test spaces on different meshes, composite
    leaves on different meshes;
  - the star and cross-weighted differences, and any node without a projection.

A false negative only forgoes the Kronecker fast path, so this never claims separability it
cannot back up with factors.

Stale spaces throw. If `a`'s spaces were built before an in-place mutation of their mesh,
`is_separable`, [`kronecker_operator`](@ref) and `fdm_solve(a, F)` throw the space's
stale-weights `ArgumentError`, exactly as `assemble(a)` does, rather than answer.

# Examples

```julia
Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 7), (false, false))
Wₕ = gridspace(Ωₕ)
is_separable(form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))))  # true

gₕ = Rₕ(Wₕ, x -> 1 + x[1])
is_separable(form(Wₕ, Wₕ, (u, v) -> innerₕ(gₕ * u, v)))  # true: varies along one axis

fₕ = Rₕ(Wₕ, x -> 1 + x[1] * x[2])
is_separable(form(Wₕ, Wₕ, (u, v) -> innerₕ(fₕ * u, v)))  # false: varies along both axes
```

See also: [`kronecker_operator`](@ref), [`KroneckerLinearOperator`](@ref).
"""
function is_separable(a::BilinearForm{D}) where {D}
    D == 1 && return false
    return _kron_separable(trial_space(a), test_space(a), a)
end

# By the trial and test space: two scalar spaces here, composite ones (one block per leaf
# pair) in `kronecker_block.jl`, anything else is not separable.
_kron_separable(::Any, ::Any, ::BilinearForm) = false
function _kron_separable(Wu::ScalarGridSpace, Wv::ScalarGridSpace, a::BilinearForm)
    mesh(Wu) === mesh(Wv) || return false
    _kron_check_spaces(a)
    Ωₕ = _host_mirror_mesh(mesh(Wu))
    return all(l -> _kron_project(l[2], Ωₕ) !== nothing, _kron_leaves(resolve_form_ast(a), ()))
end

# --- KroneckerTerm: one addend's per-axis factors ------------------------------------ #

"""
    KroneckerTerm{D, S <: Tuple, F <: Tuple, L, R}

One separable addend's `D` one-dimensional factor matrices and the (possibly still-`Ref`)
scalar coefficients multiplying it: [`KroneckerLinearOperator`](@ref)'s building block, not
exported. Build one with `_kron_term`, never directly. `factors[d]` is the axis-`d` factor
(a `Diagonal`, or any other matrix), in the order the caller gave them; `scales` is read at
`mul!` time through `_kron_coeff` so a `Ref` coefficient stays live, matching a
`BilinearForm`'s own contract. The host `mul!` reads only `rows` and `line`: `rows[d]` is
`factors[d]` itself when it is a `Diagonal` or a symmetric `SparseMatrixCSC`, and otherwise
a CSC of its transpose (column `i` of `rows[d]` is row `i` of the factor); `line` is the
axis-1 factor as the line kernels apply it (see `_kron_line_operator`). `symmetric` records
whether every factor is exactly symmetric. On a device-backed operator `line` and `rows`
are `nothing`.
"""
struct KroneckerTerm{D, S <: Tuple, F <: Tuple, L, R}
    scales::S
    factors::F
    line::L
    rows::R
    symmetric::Bool
end

# A factor's rows as the host kernels gather them: a `Diagonal` as it is, a symmetric CSC
# factor as itself (column `i` is row `i`), any other factor as a CSC of its transpose.
_kron_rows(F::Diagonal, ::Bool) = F
_kron_rows(F::SparseMatrixCSC, sym::Bool) = sym ? F : copy(transpose(F))
_kron_rows(F::AbstractMatrix, sym::Bool) = _kron_rows(sparse(F), sym)

"""
    _kron_term(scales::Tuple, factors::NTuple{D, AbstractMatrix}) -> KroneckerTerm{D}

The one way to build a host [`KroneckerTerm`](@ref): `scales` times the Kronecker product
of `factors` (axis 1 first). Each factor is checked for exact symmetry once; a non-diagonal
factor that is not symmetric gets its rows stored as the CSC of its transpose, so the host
kernel gathers row `i` without assuming symmetry, and a symmetric tridiagonal axis-1 factor
becomes a `_KronTridiag`. Any number of factors may be non-diagonal.
"""
function _kron_term(scales::Tuple, factors::NTuple{D, AbstractMatrix}) where {D}
    for d in 1:D
        size(factors[d], 1) == size(factors[d], 2) || _throw_kron_nonsquare(d, size(factors[d]))
    end
    sym = map(F -> F isa Diagonal || issymmetric(F), factors)
    rows = map(_kron_rows, factors, sym)
    line = _kron_line_operator(rows[1], sym[1])
    return KroneckerTerm{D, typeof(scales), typeof(factors), typeof(line), typeof(rows)}(
        scales, factors, line, rows, all(sym)
    )
end

"""
    KroneckerLinearOperator{T, D, TermsT <: Tuple, P <: ExecutionPolicy, M}

A matrix-free linear operator for a separable [`BilinearForm`](@ref) (see
[`is_separable`](@ref)): the sum, over its terms, of a Kronecker product of `D`
one-dimensional factor matrices, applied in one fused pass over the grid
(`LinearAlgebra.mul!(y, K, x)`, or the five-argument `mul!(y, K, x, α, β)` computing
`α * K * x + β * y`) rather than ever materialising the `D`-dimensional matrix: each entry
of `y` is its own term-weighted combination of `x` at that grid point and its neighbours
along the axes where a term's factor is not diagonal, written once.
For a `200^3` mesh the factors together hold `O(200)` numbers per axis instead of the
assembled matrix's `O(200^3)` stored entries.

A term's factors need be neither diagonal nor symmetric, and a term may have non-diagonal
factors on several axes. A mixed derivative `D₋ₓ(D₋ᵧ(u))` has non-diagonal factors on two
axes, and an advection term has a non-symmetric one. A factor that is not symmetric is
applied through the CSC of its transpose, so the product is exact for it. Which forms have
such factors is [`is_separable`](@ref)'s list.

Build one with [`kronecker_operator`](@ref). Subtypes `AbstractMatrix{T}` so it plugs into
`LinearProblem`/`KrylovJL_CG` (`LinearSolve.jl`) the same way an assembled matrix does, and
supports `size`, `eltype`, `getindex`, `Base.:*`, three- and five-argument `mul!`,
`LinearAlgebra.issymmetric`, and `SparseMatrixCSC(K)` (an explicit `kron` of the factors,
for testing and inspection: the very matrix this operator avoids forming). `issymmetric(K)`
is computed from the factors, not assumed: it is `false` as soon as one factor of one term
is not symmetric, even for a form whose matrix happens to be symmetric. `mul!` refuses
vectors that are not 1-based (an `OffsetArray`, say), which would otherwise give a wrong
product silently.

`K` holds no work buffers: it stores only its `D` one-dimensional factors and its execution
policy, so it is immutable after construction and safe to share across threads (concurrent
`mul!` calls on one `K` never race). The fused pass needs no scratch either. A serial host
`mul!` allocates nothing, and so does a [`CpuPolyester`](@ref) product: each task rebuilds
the light structs around plain arrays, so Polyester's argument box stays on the stack. `mul!` is generic over the element type, so ForwardDiff `Dual`s pass through. The
`scratch = (b1, b2)` keyword that `mul!(y, K, x; scratch)` and
`mul!(y, K, x, α, β; scratch)` accept is kept so existing callers still work, and is
ignored.

`P` is the execution policy of the space the operator was built on
(`execution_policy(backend(trial_space(a)))`), kept as the `policy` field. On the host, a
[`CpuThreaded`](@ref) operator runs the grid lines along axis 1 in one contiguous chunk per
thread under `Threads.@threads :static` (serially when nested in a threaded loop), a
[`CpuPolyester`](@ref) one as `Polyester.@batch` tasks (`using Polyester` required); every
other policy runs them serially. Each line writes its own slice of `y`, so all give the same
result bit for bit. An operator built directly from
its terms, `KroneckerLinearOperator{T, D, TermsT}(terms, dims, n)`, is [`CpuSerial`](@ref).

On a device-backed form (gpena/Bramble.jl#323) the factors are built on the host and then
moved to the space backend's device storage, so `mul!` with device `x`/`y` runs entirely on
the device, as one `KernelAbstractions` kernel with one work item per entry of `y`
(`using KernelAbstractions` required). The kernel reads at most one non-diagonal factor
per term and that factor's columns as its rows, so `kronecker_operator` refuses, on such a
form, a term with non-diagonal factors on several axes and a term with a non-symmetric
factor: build those on a host backend. `getindex` and `SparseMatrixCSC(K)` stay host-only
and throw an `ArgumentError` on such an operator.

The factors are read from the mesh once, when the operator is built (gpena/Bramble.jl#442):
`K` keeps that mesh (`M` is its type) and its version then. After an in-place mutation of
the mesh (`set_points!`, `change_points!`, `iterative_refinement!`), `mul!`, `getindex`,
`SparseMatrixCSC(K)`, `fdm_solve(K, F)` and `Kronecker.kronecker(K)` throw an
`ArgumentError`, as a space's stale weights do; build the operator again on the mutated
mesh. An operator built directly from
its terms has no mesh (`M` is `Nothing`) and never goes stale.

Dirichlet rows are out of scope: this operator carries no boundary constraint of its own.
The `Kronecker.jl` extension's fast-diagonalisation solve, `fdm_solve`, imposes homogeneous
Dirichlet conditions on the whole boundary through its `dirichlet = :boundary` keyword.

See also: [`is_separable`](@ref), [`kronecker_operator`](@ref).
"""
struct KroneckerLinearOperator{T, D, TermsT <: Tuple, P <: ExecutionPolicy, M} <:
       AbstractMatrix{T}
    terms::TermsT
    dims::NTuple{D, Int}
    n::Int
    policy::P
    # The mesh the factors were built from and its `_mesh_version` then, compared on every
    # call (`_kron_check_fresh`); `nothing` for an operator built from hand-made terms.
    mesh::M
    version::Int
    # The line kernels index under `@inbounds`, trusting that every axis-`d` factor is
    # `dims[d] × dims[d]` and that `n == prod(dims)`: checked here, once, not in `mul!`.
    function KroneckerLinearOperator{T, D, TermsT, P, M}(
            terms, dims, n, policy, mesh, version
    ) where {T, D, TermsT <: Tuple, P <: ExecutionPolicy, M}
        n == prod(dims) || _throw_kron_size(n, dims)
        foreach(t -> _kron_check_factors(t.factors, dims), terms)
        return new{T, D, TermsT, P, M}(terms, dims, n, policy, mesh, version)
    end
end

# Without a mesh: hand-made terms, which no mesh mutation can make stale.
function KroneckerLinearOperator{T, D, TermsT, P}(
        terms, dims, n, policy
) where {T, D, TermsT <: Tuple, P <: ExecutionPolicy}
    return KroneckerLinearOperator{T, D, TermsT, P, Nothing}(terms, dims, n, policy, nothing, 0)
end

# Whether `K`'s factors still describe its mesh: one integer comparison against the mesh's
# current `_mesh_version`, the one a space's `weights` makes (`scalar_gridspace.jl`).
@inline _kron_is_fresh(K::KroneckerLinearOperator) = _kron_is_fresh(K.mesh, K.version)
@inline _kron_is_fresh(::Nothing, ::Int) = true
@inline _kron_is_fresh(Ω::AbstractMeshType, version::Int) = version == _mesh_version(Ω)
@inline function _kron_check_fresh(K)
    _kron_is_fresh(K) || _throw_kron_stale()
    return nothing
end

# The form's own spaces must not be stale either: each leaf's `weights` throws the
# stale-weights error `assemble(a)` throws for them, before any factor is built.
function _kron_check_spaces(a::BilinearForm)
    for W in (trial_space(a), test_space(a))
        foreach(l -> weights(first(l)), leaf_spaces_offsets(W))
    end
    return nothing
end

# The stale-weights error of `_throw_stale_weights` (`scalar_gridspace.jl`), with this
# operator's remedy: its factors are the weights it would otherwise read stale.
@noinline function _throw_kron_stale()
    throw(
        ArgumentError(
        "KroneckerLinearOperator's factors were computed from its mesh before an in-place " *
        "mutation (set_points!, change_points!, or iterative_refinement!) changed it -- " *
        "every product, entry and solve through this operator would silently use factors " *
        "for a mesh that no longer exists. Build the operator again: gridspace on the " *
        "mutated mesh, then form and kronecker_operator.",
    ),
    )
end

@noinline function _throw_kron_nonsquare(d::Int, sz)
    throw(ArgumentError("_kron_term: the axis-$d factor is $(sz[1]) × $(sz[2]); every " *
                        "Kronecker factor must be square."))
end

@noinline function _throw_kron_size(n, dims)
    throw(DimensionMismatch("KroneckerLinearOperator: n = $n does not equal prod(dims) = " *
                            "$(prod(dims)) for dims = $dims."))
end

@noinline function _throw_kron_factor_size(d::Int, sz, m)
    throw(DimensionMismatch("KroneckerLinearOperator: a term's axis-$d factor is " *
                            "$(sz[1]) × $(sz[2]), but the grid has $m points along axis $d."))
end

# Every factor `d` of one term is `dims[d] × dims[d]`.
function _kron_check_factors(factors::Tuple, dims::Tuple)
    length(factors) == length(dims) ||
        throw(DimensionMismatch("KroneckerLinearOperator: a term has $(length(factors)) " *
                                "factors for a $(length(dims))-dimensional grid."))
    for d in eachindex(dims)
        sz = _kron_factor_size(factors[d])
        sz == (dims[d], dims[d]) || _throw_kron_factor_size(d, sz, dims[d])
    end
    return nothing
end

# Without a policy the operator runs serially: what every operator did before the policy was
# a parameter.
function KroneckerLinearOperator{T, D, TermsT}(terms, dims, n) where {T, D, TermsT <: Tuple}
    return KroneckerLinearOperator{T, D, TermsT, CpuSerial}(terms, dims, n, CpuSerial())
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

# `block` names the composite form's block the term is in (`kronecker_block.jl`), or is
# empty for a scalar form.
@noinline function _throw_not_separable_term(term, Ωₕ, block::String = "")
    throw(
        ArgumentError(
        "kronecker_operator: the term $(nameof(typeof(term)))$block has no Kronecker factors: " *
        "$(_kron_refused_node(term, Ωₕ)) has no projection onto the mesh's axes. See " *
        "`is_separable` for what factors.",
    ),
    )
end

"""
    _kron_check_device(term::KroneckerTerm) -> Nothing

Throw an `ArgumentError` naming the shape when the device kernel (`_launch_kron_fused!`)
cannot apply the host-built `term`: it reads at most one non-diagonal factor per term, and
reads that factor's rows from its columns, so the factor must be symmetric. `nothing`
otherwise.
"""
function _kron_check_device(t::KroneckerTerm)
    axes = Tuple(d for d in eachindex(t.factors) if !(t.factors[d] isa Diagonal))
    length(axes) > 1 && throw(
        ArgumentError(
        "kronecker_operator: a term has non-diagonal factors on axes $axes; the device " *
        "kernel applies at most one per term. Build the operator on a host backend.",
    ),
    )
    for d in axes
        issymmetric(t.factors[d]) || throw(
            ArgumentError(
            "kronecker_operator: a term's axis-$d factor is not symmetric; the device " *
            "kernel reads a factor's rows from its columns. Build the operator on a host " *
            "backend.",
        ),
        )
    end
    return nothing
end

# One projected factor as `kronecker_operator` stores it: a diagonal one as a `Diagonal`,
# the axis's own mass factor `H` itself when it equals it (so a mass axis keeps the lazy
# weights it always had), and any other factor as it is.
function _kron_factor(F::SparseMatrixCSC, H::Diagonal)
    h = zeros(eltype(F), size(F, 1))
    rows, vals = rowvals(F), nonzeros(F)
    for j in axes(F, 2), k in nzrange(F, j)

        rows[k] == j || return F
        h[j] = vals[k]
    end
    return isequal(h, _kron_diag(H)) ? H : Diagonal(h)
end

# The first factor in `seen` equal to `F`, or `F` itself, pushed: terms that project to the
# same 1D matrix on an axis share one object.
function _kron_cached!(seen::Vector{Any}, F)
    for G in seen
        typeof(G) === typeof(F) && G == F && return G
    end
    push!(seen, F)
    return F
end

@noinline function _warn_kron_coefficient()
    @warn "kronecker_operator: a grid-function coefficient was read once, now, into the " *
          "Kronecker factors; a later `Rₕ!` (or any other edit) to it is not seen by this " *
          "operator. Use a `Ref` scalar for a coefficient that changes, or rebuild the " *
          "operator."
    return nothing
end

"""
    kronecker_operator(a::BilinearForm) -> Union{KroneckerLinearOperator, KroneckerBlockOperator}

Build a matrix-free [`KroneckerLinearOperator`](@ref) for the separable bilinear form `a`
(see [`is_separable`](@ref)), without ever assembling the `D`-dimensional matrix.

Each term's factor on axis `d` is the 1D form it projects to there, assembled on
`gridspace(Ωₕ(d))` (`_kron_project`): a diagonal factor is stored as a `Diagonal` (the
axis's cell measures `weights(gridspace(Ωₕ(d)), Innerh())` themselves for a mass factor),
and terms sharing a factor on an axis share one matrix. Factors are always built on the host
(a device mesh through its host mirror) and then converted to the storage
`backend(trial_space(a))` uses, so a device-backed form yields device-resident factors. A
scalar or `Ref` coefficient around a term stays live, read at every `mul!`.

!!! warning

    A grid-function coefficient (an `Rₕ` multiplying a side, varying along one axis) is
    read once, when the operator is built, and copied into the factors; a warning says so.
    A later edit to it is not seen. Use a `Ref` for a coefficient that changes, or
    [`matrix_free_operator`](@ref), which reads the coefficient live at every product. On a
    device-backed form a coefficient stored on the device is refused, since the factors are
    built on the host.

The factors also describe the mesh as it is now: if the mesh is mutated in place after this
call, the operator throws instead of applying (see [`KroneckerLinearOperator`](@ref)).

On a composite trial or test space whose leaves all share one mesh, the result is a
`Bramble.KroneckerBlockOperator` (not exported): one `KroneckerLinearOperator` per nonzero
(test leaf, trial leaf) block, placed at that block's rows and columns, with the same
`size`, `getindex`, `mul!`, `SparseMatrixCSC` and `issymmetric` interface.

# Throws

  - `ArgumentError`: `a` is not separable, naming the offending node and, on a composite
    space, its block (or the dimension, or the spaces) -- the same check
    [`is_separable`](@ref) runs, made specific; or `a` is device-backed and a term has more
    than one non-diagonal factor or a non-symmetric one, which the device kernel cannot
    apply (`_kron_check_device`).

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
    return _kron_operator(trial_space(a), test_space(a), a)
end

# By the trial and test space, as `_kron_separable`: two scalar spaces here, composite ones
# in `kronecker_block.jl`, anything else refused.
_kron_operator(Wu, Wv, ::BilinearForm) = _throw_not_separable_space(Wu, Wv)

function _kron_operator(Wu::ScalarGridSpace, Wv::ScalarGridSpace, a::BilinearForm)
    mesh(Wu) === mesh(Wv) || _throw_not_separable_space(Wu, Wv)
    _kron_check_spaces(a)
    cache = _kron_cache(mesh(Wu))
    K, reads_coef = _kron_build(cache, backend(Wu), ndofs(Wu, Tuple),
        _kron_leaves(resolve_form_ast(a), ()), "")
    reads_coef && _warn_kron_coefficient()
    return K
end

# What every operator of one `kronecker_operator` call shares: the mesh and its version
# (recorded in each operator, `_kron_check_fresh`), the host mesh the factors are built on,
# each axis's mass factor, and each axis's factors built so far (`_kron_cached!`).
# Factors are always built on the host -- a device-backed mesh's per-axis spaces would
# otherwise be scalar-indexed by `weights`/`assemble` -- and only then moved to the storage
# the backend chooses (`_kron_to_storage`). On a host mesh `_host_mirror_mesh` returns the
# mesh itself and `_kron_to_storage` is the identity.
function _kron_cache(Ω::AbstractMeshType{D}) where {D}
    Ωₕ = _host_mirror_mesh(Ω)
    version = _mesh_version(Ω)
    mass = ntuple(d -> Diagonal(weights(_kron_axis_space(Ωₕ, d), Innerh())), Val(D))
    seen = ntuple(_ -> Any[], Val(D))
    return (; Ω, version, Ωₕ, mass, seen)
end

"""
    _kron_build(cache, be, dims, leaves, block::String) -> (KroneckerLinearOperator, Bool)

The operator on backend `be` and a grid of `dims` points summing `leaves`, `(scales, term)`
pairs from `_kron_leaves`, each term projected on `cache.Ωₕ`, and whether a term read a
grid-function coefficient (the caller warns once). Throws naming `block` (see
`_throw_not_separable_term`) when a term does not project.
"""
function _kron_build(cache, be, dims::NTuple{D, Int}, leaves, block::String) where {D}
    loc = locality(be)
    (; Ωₕ, mass, seen) = cache
    T = eltype(mass[1])
    reads_coef = false
    terms = ()
    for (scales, term) in leaves
        P = _kron_project(term, Ωₕ)
        P === nothing && _throw_not_separable_term(term, Ωₕ, block)
        reads_coef |= _kron_reads_coef(term)
        for projected in P
            factors = ntuple(d -> _kron_cached!(seen[d], _kron_factor(projected[d], mass[d])), Val(D))
            T = promote_type(T, map(eltype, factors)...)
            t = _kron_term(scales, factors)
            loc isa DeviceLocality && _kron_check_device(t)
            terms = (terms..., _kron_to_storage(loc, be, t))
        end
    end
    policy = execution_policy(be)
    K = KroneckerLinearOperator{T, D, typeof(terms), typeof(policy), typeof(cache.Ω)}(
        terms, dims, prod(dims), policy, cache.Ω, cache.version)
    return K, reads_coef
end

# --- Device-resident factors (gpena/Bramble.jl#323) ---------------------------------- #
#
# On a device-backed space the host-built factors move to device storage, each wrapped in a
# type of its own so `mul!` dispatches on the factor, never on a GPU array type (this file
# names no GPU package). The sparse factor keeps its CSC arrays separately
# (`Int32` indices) because a kernel is handed the raw arrays, never a struct nesting a
# device array -- see `docs/notes/internals/gpu.md`. `_kron_check_device` admits only a
# symmetric sparse factor, so column `j` of the CSC storage is also row `j`: the kernel
# gathers row `j` from column `j` without a transpose.

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

# A host term's factors moved to the device; the host kernels' `line` and `rows` stay
# behind.
function _kron_to_storage(loc::DeviceLocality, be, t::KroneckerTerm{D}) where {D}
    F = map(f -> _kron_to_storage(loc, be, f), t.factors)
    return KroneckerTerm{D, typeof(t.scales), typeof(F), Nothing, Nothing}(
        t.scales, F, nothing, nothing, t.symmetric
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

_kron_factor_size(F::AbstractMatrix) = size(F)
_kron_factor_size(F::_KronDeviceDiagonal) = (length(F.diag), length(F.diag))
# Built from a square host factor (`_kron_term` checked it): the column count is the size.
_kron_factor_size(F::_KronDeviceSparse) = (length(F.colptr) - 1, length(F.colptr) - 1)
_kron_entry(::Union{_KronDeviceDiagonal, _KronDeviceSparse}, ::Int, ::Int) = _throw_kron_device_entry()

# --- Fused application: one pass over `y` -------------------------------------------- #
#
# Entry `I = (i_1, ..., i_D)` of `K * x` is
#
#     sum_t c_t * sum_{j_1, ..., j_D} prod_e A^t_e[i_e, j_e] * x[j_1, ..., j_D]
#
# and on the host `mul!` evaluates it one grid line along axis 1 at a time
# (`off + 1:off + m`, contiguous), writing each entry of `y` once. Per term, the diagonal factors on axes
# `2:D` fold into one scalar line weight. Each non-diagonal factor on an axis `e >= 2`
# contributes the stored entries of its row `i_e`: the term loops over the product of those
# rows, reading the neighbour line `sum_e (j_e - i_e) * stride_e` away with the product of
# the entries as its weight, and applies the axis-1 factor to each neighbour line with a
# `@simd` pass (diagonal), a tridiagonal sweep (`_KronTridiag`) or a row gather. The line is
# the only part of `y` those passes revisit, so `y` goes through memory once and the product
# needs no scratch. Which factors are non-diagonal is read off the factor types, so the
# whole evaluation is resolved at compile time; the tuples are peeled recursively for the
# same reason `_fold_taps` (`stencil_eval.jl`) peels its taps. The mass shape (every factor
# diagonal) and the single-sparse shapes keep methods of their own. The kernels read a
# term's `rows`, never its `factors`: column `i` of `rows[e]` is row `i` of the factor
# whether or not it is symmetric (`_kron_term`).

@inline _kron_diag(F::Diagonal) = F.diag
# The mass factor's diagonal is a one-axis `SeparableWeights`: index its own vector rather
# than going through its `CartesianIndex` `getindex` in the inner loop.
@inline _kron_diag(F::Diagonal{<:Any, <:SeparableWeights{1}}) = F.diag.factors[1]

# A CSC factor rebuilt around the arrays `Polyester.@batch` hands each task
# (`_kron_host_rebuild`): the line kernels read only these three fields, which a
# `SparseMatrixCSC` has under the same names, so both run the same kernel.
struct _KronCSC{CP <: AbstractVector, RV <: AbstractVector, NV <: AbstractVector}
    colptr::CP
    rowval::RV
    nzval::NV
end

const _KronCSCLike = Union{SparseMatrixCSC, _KronCSC}

# One axis `e >= 2` of a term on the line whose axis-`e` index is `i`: a diagonal factor
# contributes its entry to the line weight; a sparse one contributes weight one and is
# returned, with `i` and the axis stride, as one of the line's neighbour axes.
@inline _kron_line_split(F::Diagonal, i::Int, ::Int) = (_kron_diag(F)[i], nothing)
@inline _kron_line_split(F::_KronCSCLike, i::Int, stride::Int) = (one(eltype(F.nzval)), (F, i, stride))

# Two or more neighbour axes of one term, each an `(F, i, stride)` tuple. One neighbour axis
# stays a bare tuple so the single-sparse shapes keep their own methods.
struct _KronNeighbours{T <: Tuple}
    axes::T
end

@inline _kron_pick(::Nothing, ::Nothing) = nothing
@inline _kron_pick(a, ::Nothing) = a
@inline _kron_pick(::Nothing, b) = b
@inline _kron_pick(a::Tuple, b::Tuple) = _KronNeighbours((a, b))
@inline _kron_pick(a::Tuple, b::_KronNeighbours) = _KronNeighbours((a, b.axes...))

@inline _kron_line_fold(::Tuple{}, ::Tuple{}, ::Tuple{}) = (true, nothing)
@inline function _kron_line_fold(Fs::Tuple, Js::Tuple, ss::Tuple)
    w, sp = _kron_line_split(Fs[1], Js[1], ss[1])
    wr, spr = _kron_line_fold(Base.tail(Fs), Base.tail(Js), Base.tail(ss))
    return (w * wr, _kron_pick(sp, spr))
end

# A host symmetric axis-1 factor stored as its diagonal and sub-diagonal when it is
# tridiagonal -- which the `inner₊(D₋ₓ(u), D₋ₓ(v))` matrix `kronecker_operator` builds is --
# so the axis-1 term runs as a `@simd` loop along the line rather than a CSC row gather,
# which took 18-20 ms of a 23-26 ms 2D 3000^2 / 3D 200^3 `Float32` `mul!` (2026-09-24). Any
# other sparsity, and any non-symmetric factor, keeps its rows (`_kron_rows`) for the
# gather. `sub[i] = F[i + 1, i]`, which equals `F[i, i + 1]` since the factor is symmetric.
struct _KronTridiag{V <: AbstractVector}
    dg::V
    sub::V
end

_kron_line_operator(::Diagonal, ::Bool) = nothing
function _kron_line_operator(R::SparseMatrixCSC, sym::Bool)
    sym || return R
    m = size(R, 1)
    rows = rowvals(R)
    for j in 1:m
        for k in nzrange(R, j)
            abs(rows[k] - j) <= 1 || return R
        end
    end
    return _KronTridiag([R[i, i] for i in 1:m], [R[i + 1, i] for i in 1:(m - 1)])
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
    rows = F.rowval
    vals = F.nzval
    @inbounds for k in F.colptr[ie]:(F.colptr[ie + 1] - 1)
        c = s * vals[k]
        xoff = off + (rows[k] - ie) * stride
        @simd for i in 1:m
            y[off + i] += (c * h[i]) * x[xoff + i]
        end
    end
    return y
end

# Directional term along axis 1, tridiagonal factor: ends by hand, interior as one loop.
# The axis-1 factor itself is never read here or below: `line` carries it.
@inline function _kron_line!(y, ::Any, L::_KronTridiag, ::Nothing, x, s, off::Int, m::Int)
    dg, sub = L.dg, L.sub
    m == 0 && return y
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
@inline function _kron_line!(y, ::Any, F1::_KronCSCLike, ::Nothing, x, s, off::Int, m::Int)
    rows = F1.rowval
    vals = F1.nzval
    @inbounds for i in 1:m
        acc = zero(eltype(y))
        for k in F1.colptr[i]:(F1.colptr[i + 1] - 1)
            acc += vals[k] * x[off + rows[k]]
        end
        y[off + i] += s * acc
    end
    return y
end

# A non-diagonal axis-1 factor with one neighbour axis, or any axis-1 factor with several:
# the axis-1 factor applied to every neighbour line of the product of the neighbour rows.
@inline function _kron_line!(
        y, F1, L::Union{_KronTridiag, _KronCSCLike}, sp::Tuple, x, s, off::Int, m::Int
)
    return _kron_line_product!(y, F1, L, (sp,), x, s, off, off, m)
end
@inline function _kron_line!(y, F1, L, nb::_KronNeighbours, x, s, off::Int, m::Int)
    return _kron_line_product!(y, F1, L, nb.axes, x, s, off, off, m)
end

# Peel one neighbour axis per level: each stored entry of its row `ie` scales the weight and
# moves the read offset by `(j - ie) * stride`; with none left, apply the axis-1 factor.
@inline function _kron_line_product!(
        y, F1, L, ::Tuple{}, x, c, off::Int, xoff::Int, m::Int
)
    return _kron_line_apply!(y, F1, L, x, c, off, xoff, m)
end
@inline function _kron_line_product!(y, F1, L, nbs::Tuple, x, c, off::Int, xoff::Int, m::Int)
    F, ie, stride = nbs[1]
    rows = F.rowval
    vals = F.nzval
    @inbounds for k in F.colptr[ie]:(F.colptr[ie + 1] - 1)
        _kron_line_product!(y, F1, L, Base.tail(nbs), x, c * vals[k], off,
            xoff + (rows[k] - ie) * stride, m)
    end
    return y
end

# `y[off + i] += c * (A_1 * x[xoff .+ (1:m)])[i]` for the axis-1 factor `A_1`, in the same
# arithmetic as the single-line methods above.
@inline function _kron_line_apply!(y, F1::Diagonal, ::Nothing, x, c, off::Int, xoff::Int, m::Int)
    h = _kron_diag(F1)
    @inbounds @simd for i in 1:m
        y[off + i] += (c * h[i]) * x[xoff + i]
    end
    return y
end
@inline function _kron_line_apply!(y, ::Any, L::_KronTridiag, x, c, off::Int, xoff::Int, m::Int)
    dg, sub = L.dg, L.sub
    m == 0 && return y
    @inbounds if m == 1
        y[off + 1] += c * (dg[1] * x[xoff + 1])
    else
        y[off + 1] += c * (dg[1] * x[xoff + 1] + sub[1] * x[xoff + 2])
        @simd for i in 2:(m - 1)
            y[off + i] += c * (sub[i - 1] * x[xoff + i - 1] + dg[i] * x[xoff + i] + sub[i] * x[xoff + i + 1])
        end
        y[off + m] += c * (sub[m - 1] * x[xoff + m - 1] + dg[m] * x[xoff + m])
    end
    return y
end
@inline function _kron_line_apply!(y, ::Any, F1::_KronCSCLike, x, c, off::Int, xoff::Int, m::Int)
    rows = F1.rowval
    vals = F1.nzval
    @inbounds for i in 1:m
        acc = zero(eltype(y))
        for k in F1.colptr[i]:(F1.colptr[i + 1] - 1)
            acc += vals[k] * x[xoff + rows[k]]
        end
        y[off + i] += c * acc
    end
    return y
end

@inline _kron_line_terms!(y, ::Tuple{}, ::Tuple{}, x, Js::Tuple, ss::Tuple, off::Int, m::Int) = y
@inline function _kron_line_terms!(
        y, terms::Tuple{KroneckerTerm, Vararg{KroneckerTerm}}, cs::Tuple, x, Js::Tuple, ss::Tuple,
        off::Int, m::Int
)
    # A host term's `rows` is a `Tuple`; the assertion keeps inference off the device
    # terms' `nothing`.
    F = terms[1].rows::Tuple
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

# Under `CpuThreaded` the lines are cut into one contiguous chunk per thread, run in one
# `Threads.@threads :static` loop through `_static_or_serial`, so a call nested in a user's
# threaded loop runs the same chunks in order on the calling task. Each line writes only its
# own slice of `y` and reads `x`, so the result is bitwise the serial loop's.
@noinline function _kron_fused!(
        ::HostLocality, y, K::KroneckerLinearOperator{<:Any, <:Any, <:Tuple, CpuThreaded}, x,
        cs::Tuple, β
)
    dims = K.dims
    ss = Base.front(cumprod(dims))::Tuple{Vararg{Int}}
    lines = CartesianIndices(Base.tail(dims))
    _static_or_serial(_static_bands!, _serial_bands!, _kron_line_band!, Threads.nthreads(),
        y, K.terms, cs, x, β, ss, lines, dims[1])
    return y
end

# Lines `_band_range(1:length(lines), nbands, b)` of the product: band `b` of `nbands`.
@inline function _kron_line_band!(y, terms, cs, x, β, ss, lines, m::Int, nbands::Int, b::Int)
    for o in _band_range(1:length(lines), nbands, b)
        off = (o - 1) * m
        _kron_line_init!(y, β, off, m)
        _kron_line_terms!(y, terms, cs, x, Tuple(@inbounds lines[o]), ss, off, m)
    end
    return nothing
end

# Under `CpuPolyester` the lines run as `Polyester.@batch` tasks (`_batch_kron_lines!`,
# filled by `BramblePolyesterExt`). Each line writes only its own slice of `y` and reads `x`,
# so the result is bitwise the serial loop's.
@noinline function _kron_fused!(
        ::HostLocality, y, K::KroneckerLinearOperator{<:Any, <:Any, <:Tuple, CpuPolyester}, x,
        cs::Tuple, β
)
    dims = K.dims
    ss = Base.front(cumprod(dims))::Tuple{Vararg{Int}}
    lines = CartesianIndices(Base.tail(dims))
    _late(_batch_kron_lines!, y, K.terms, cs, x, β, ss, lines, dims[1])
    return y
end

# What a term's line kernels read (its `rows` and `line`), as plain arrays only: a diagonal
# as its vector, a CSC as `(colptr, rowval, nzval)`, a `_KronTridiag` as `(dg, sub)`.
# `@batch` turns every array in these nested tuples into a `PtrArray` under its own
# `GC.@preserve`, and each task puts the light structs back around them with
# `_kron_host_rebuild` (its `factors` are the rebuilt rows, which the kernels never read);
# the coefficients stay out, since `cs` carries them.
_kron_host_raw(::Nothing) = nothing
_kron_host_raw(F::Diagonal) = _kron_diag(F)
_kron_host_raw(F::SparseMatrixCSC) = (F.colptr, F.rowval, F.nzval)
_kron_host_raw(L::_KronTridiag) = (L.dg, L.sub)
_kron_host_raw(t::KroneckerTerm) = (map(_kron_host_raw, t.rows), _kron_host_raw(t.line))

@inline _kron_host_rebuild(::Nothing) = nothing
@inline _kron_host_rebuild(v::AbstractVector) = Diagonal(v)
@inline _kron_host_rebuild(r::NTuple{3, AbstractVector}) = _KronCSC(r...)
@inline _kron_host_rebuild(r::NTuple{2, AbstractVector}) = _KronTridiag(r...)
@inline function _kron_host_rebuild(r::Tuple{Tuple, Any})
    F = map(_kron_host_rebuild, r[1])
    L = _kron_host_rebuild(r[2])
    return KroneckerTerm{length(F), Tuple{}, typeof(F), typeof(L), typeof(F)}((), F, L, F, false)
end

"""
    _batch_kron_lines!(y, terms, cs, x, β, ss, lines, m) -> Nothing

[`CpuPolyester`](@ref)'s host `KroneckerLinearOperator` product, filled by
`BramblePolyesterExt`: line `o` of `lines` (the `CartesianIndices` of axes `2:D`) starts at
offset `(o - 1) * m` and gets `_kron_line_init!` then `_kron_line_terms!`, one line per
`Polyester.@batch` iteration. The only `src/` method errors naming Polyester.
"""
@noinline function _batch_kron_lines!(y, terms, cs, x, β, ss, lines, m)
    return _throw_cpubatch_without_polyester(:_batch_kron_lines!)
end

# Device: one work item per entry of `y`, in `BrambleKernelAbstractionsExt`. The kernel is
# handed raw arrays -- a diagonal factor as its vector, a sparse one as its
# `(colptr, rowval, nzval)` tuple -- because a struct nesting a device array fails
# `KernelAbstractions` kernel compilation (`docs/notes/internals/gpu.md`).
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
# deleted, because `mul!` is the primitive `AbstractMatrix` operations are built from, not something
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
# `scratch` is accepted and ignored: the fused pass above needs no work vectors, so a serial
# host `mul!` allocates nothing with or without it, and callers that pass one keep working.
# Under `CpuPolyester` a warm product allocates nothing either: only plain arrays and isbits
# values cross `@batch` (`_kron_host_raw`), so Polyester keeps its argument box on the stack.
#
# Five-argument form, `y = α * K * x + β * y`, with `LinearAlgebra`'s semantics: `β == 0`
# (including `false`) overwrites `y`, so a `NaN` already in `y` does not survive; otherwise
# each line of `y` is scaled by `β` before the terms (each scaled by `α`) accumulate into it.
# Without this method `mul!(y, K, x, α, β)` falls to `LinearAlgebra`'s generic `O(n^2)`
# `getindex` loop. `α`/`β` may be `Int` (`semidiscretize_rhs` passes `-1, 1`).
function mul!(
        y::AbstractVector, K::KroneckerLinearOperator, x::AbstractVector, α::Number,
        β::Number; scratch = nothing
)
    # The line kernels index `y` and `x` from 1 under `@inbounds`.
    # Before the size check (a refined mesh changes the sizes, and staleness is the cause),
    # and before any kernel launches or any `@batch` task starts.
    _kron_check_fresh(K)
    Base.require_one_based_indexing(y, x)
    n = K.n
    (length(x) == n && length(y) == n) || _throw_kron_dimmismatch(K, x, y)
    return _kron_apply!(locality(typeof(y)), y, K, x, α, β)
end

# The product with the locality given rather than read off `y`: a composite operator
# (`kronecker_block.jl`) hands each block a view, whose type says host even on a device.
function _kron_apply!(loc, y, K::KroneckerLinearOperator{T}, x, α, β) where {T}
    cs = map(t -> _kron_scalar(T, α * _kron_coeff(t.scales)), K.terms)
    _kron_fused!(loc, y, K, x, cs, β)
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
    _kron_check_fresh(K)
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

# A stale operator has no entries to show (`getindex` throws): its summary and why.
function Base.show(io::IO, m::MIME"text/plain", K::KroneckerLinearOperator)
    _kron_is_fresh(K) && return invoke(show, Tuple{IO, MIME"text/plain", AbstractMatrix}, io, m, K)
    return _kron_show_stale(io, K)
end

function _kron_show_stale(io::IO, K)
    summary(io, K)
    print(io,
        ":\n  stale: its mesh was mutated in place (set_points!, change_points!, or " *
        "iterative_refinement!) after it was built; no entries to show.")
    return nothing
end

# No `*` method of its own. `KroneckerLinearOperator <: AbstractMatrix`, so `LinearAlgebra`
# already derives `K * x` from the `mul!` above, and defining the two-argument form here was
# ambiguous against any package that dispatches `*` on its own vector type -- `NamedDims`'
# `*(::AbstractMatrix, ::NamedDimsArray{_, _, 1})` among them, which the extension ambiguity
# gate reports whenever such a package is loaded alongside Bramble.

"""
    issymmetric(K::KroneckerLinearOperator) -> Bool

`true` when every factor of every term is exactly symmetric (checked once, when
`_kron_term` built the term), so every term and their sum are symmetric. `false` otherwise,
even in the rare case where non-symmetric terms happen to sum to a symmetric matrix: the
check reads the factors, never the materialised matrix.
"""
issymmetric(K::KroneckerLinearOperator) = all(t -> t.symmetric, K.terms)

_kron_as_sparse(F::SparseMatrixCSC) = F
_kron_as_sparse(F::Diagonal) = sparse(F)
_kron_as_sparse(F::AbstractMatrix) = sparse(F)
_kron_as_sparse(::Union{_KronDeviceDiagonal, _KronDeviceSparse}) = _throw_kron_device_entry()

"""
    SparseMatrixCSC(K::KroneckerLinearOperator) -> SparseMatrixCSC

Materialise `K` as an explicit sparse matrix: the sum, over its terms, of the coefficient
times the Kronecker product of its `D` one-dimensional factors, last axis leftmost
(`A_2D = H_y ⊗ A_x + A_y ⊗ H_x`, matching gpena/Bramble.jl#162's own formula). For testing
and inspection only -- this is exactly the `D`-dimensional matrix [`kronecker_operator`](@ref)
is built to avoid forming.

The stored pattern is the one `dropzeros` leaves of `assemble(a)`: `assemble` keeps the
explicit zeros of its stencil, this matrix does not, so the two agree entry for entry but
their `nnz` can differ.

Built as a single `sparse(I, J, V, n, n)` call over every term's `findnz` triplets (`V`
pre-scaled by that term's coefficient) rather than summing each term's Kronecker product
into an accumulator one term at a time, which reallocates the whole `n x n` matrix per term.
"""
function SparseArrays.SparseMatrixCSC(K::KroneckerLinearOperator{T}) where {T}
    _kron_check_fresh(K)
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
