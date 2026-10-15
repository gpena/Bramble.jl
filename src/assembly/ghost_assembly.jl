# ghost_assembly.jl: the assembly of forms posed on a `GhostGridSpace`, which eliminates
# the ghost values through the space's closure so that the assembled matrix acts on the
# space's degrees of freedom.
#
# A form on ghost spaces is the form on their padded spaces (the same placeholders, the
# same AST) wrapped with the closures `(E, d)` of both sides. `assemble` fills the padded
# matrix `A_pad` and reduces it to `E_vᵀ A_pad E_u` in DOF order. The load of a linear form
# is `E_vᵀ (F_pad − A_pad d_u)`, so it needs the bilinear form and `assemble(l)` alone
# refuses a closure with `d ≠ 0`. A map from `A_pad`'s stored values to the reduced
# matrix's, built once with the form, does the reduction, so a warm `assemble!` allocates
# nothing at all. Why not substitute `E` at each stencil tap? It would rewrite the guarded
# walk, the pattern recorder and the replay for no measured gain (design D1).
#
# Out of scope on a ghost space, and refused with an `ArgumentError`: `semidiscretize`,
# `ode_problem`, `matrix_free_operator`, `kronecker_operator`, device backends, matrix
# types other than `SparseMatrixCSC`, composite spaces with a ghost leaf, AD element types,
# and `dirichlet` (row replacement on the reduced system is not implemented yet).
#
# See also: `GhostGridSpace`, `BilinearForm`, `LinearForm`

"""
    GhostBilinearForm

A bilinear form on two [`GhostGridSpace`](@ref)s, built by `form(Wₕ, Vₕ, f)`: the form on
their padded spaces, the padded matrix it fills, and the map that reduces that matrix to
`E_vᵀ A_pad E_u`, the matrix on the degrees of freedom.
"""
struct GhostBilinearForm{D, F <: BilinearForm{D}, WT <: GhostGridSpace,
    VT <: GhostGridSpace, MT <: SparseMatrixCSC, CT}
    "the form on the padded spaces."
    padded::F
    trial_space::WT
    test_space::VT
    "the padded matrix, filled by every assembly."
    A_pad::MT
    "the stored position in `A_pad` of each contribution."
    src::Vector{Int}
    "the stored position in the reduced matrix of each contribution."
    dst::Vector{Int}
    "the closure weight `E_v[i, r] E_u[j, c]` of each contribution."
    coef::Vector{CT}
    "the reduced pattern."
    colptr::Vector{Int}
    rowval::Vector{Int}
end

"""
    GhostLinearForm

A linear form on a [`GhostGridSpace`](@ref), built by `form(Vₕ, f)`: the form on the padded
space and the padded load vector it fills.
"""
struct GhostLinearForm{D, F <: LinearForm{D}, VT <: GhostGridSpace, BT <: AbstractVector}
    "the form on the padded space."
    padded::F
    test_space::VT
    "the padded load vector, filled by every assembly."
    F_pad::BT
end

trial_space(a::GhostBilinearForm) = a.trial_space
test_space(a::GhostBilinearForm) = a.test_space
test_space(l::GhostLinearForm) = l.test_space

function Base.show(io::IO, a::GhostBilinearForm{D}) where {D}
    print(io, "GhostBilinearForm{$(D)D}(", a.trial_space, " → ", a.test_space, ")")
    return nothing
end

function Base.show(io::IO, l::GhostLinearForm{D}) where {D}
    print(io, "GhostLinearForm{$(D)D}(", l.test_space, ")")
    return nothing
end

# --- Construction -------------------------------------------------------------------- #

function form(Wₕ::GhostGridSpace, Vₕ::GhostGridSpace, f)
    _check_ghost_host(Wₕ)
    _check_ghost_host(Vₕ)
    _check_same_ghosts(Wₕ, Vₕ)
    a = form(padded_space(Wₕ), padded_space(Vₕ), f)
    _check_ghost_eltype(_matrix_eltype(a, a.ast))
    A_pad = allocate_system_matrix(a)::SparseMatrixCSC  # `_check_ghost_host`
    src, dst, coef, colptr, rowval = _ghost_reduction_map(A_pad, Vₕ.E, Wₕ.E)
    return GhostBilinearForm(a, Wₕ, Vₕ, A_pad, src, dst, coef, colptr, rowval)
end

function form(Vₕ::GhostGridSpace, f)
    _check_ghost_host(Vₕ)
    l = form(padded_space(Vₕ), f)
    T = _assembled_eltype(l.ast, padded_space(Vₕ))
    _check_ghost_eltype(T)
    return GhostLinearForm(l, Vₕ, Vector{T}(undef, ndofs(padded_space(Vₕ))))
end

form(::GhostGridSpace, Vₕ, f) = _throw_ghost_mixed()
form(Wₕ, ::GhostGridSpace, f) = _throw_ghost_mixed()

@noinline function _throw_ghost_mixed()
    throw(
        ArgumentError(
        "a form takes ghost spaces on both sides or on neither: the reduction to the " *
        "degrees of freedom needs a closure on each. Build both spaces with " *
        "gridspace(Ωₕ; ghosts = ...).",
    ),
    )
end

# A composite space with a ghost leaf would need a closure per block.
_leaf_spaces_offsets(::GhostGridSpace, ::Int) = _throw_ghost_composite()
first_space(::GhostGridSpace) = _throw_ghost_composite()

@noinline function _throw_ghost_composite()
    throw(
        ArgumentError(
        "a composite space with a ghost-space component is not supported: forms assemble " *
        "on a scalar GhostGridSpace only.",
    ),
    )
end

# Padded index `k` names the same point on both sides only when both pad the same facets:
# `(:ymin,)` and `(:ymax,)` pad grids of one size whose rows are shifted by one node.
function _check_same_ghosts(Wₕ::GhostGridSpace, Vₕ::GhostGridSpace)
    Gw, Gv = mesh(padded_space(Wₕ)), mesh(padded_space(Vₕ))
    Gw === Gv || Gw.active == Gv.active || _throw_ghost_sets(Gw.active, Gv.active)
    return nothing
end

@noinline function _throw_ghost_sets(aw, av)
    throw(
        ArgumentError(
        "the trial and test ghost spaces ghost different facets ($aw and $av as per-axis " *
        "(min, max) flags), so a padded index names different points on each side. " *
        "Forms coupling different ghost sets are not supported: build both spaces with " *
        "the same `ghosts`.",
    ),
    )
end

function _check_ghost_host(Wₕ::GhostGridSpace)
    P = padded_space(Wₕ)
    (locality(backend(P)) isa DeviceLocality || execution_policy(P) isa GpuPolicy) &&
        _throw_ghost_unsupported("a device backend")
    matrix_type(backend(P)) <: SparseMatrixCSC ||
        _throw_ghost_unsupported("the matrix type $(matrix_type(backend(P)))")
    return nothing
end

# Plain real and complex floating-point entries only: AD types are out of scope.
_check_ghost_eltype(::Type{<:Union{AbstractFloat, Complex{<:AbstractFloat}}}) = nothing
_check_ghost_eltype(T::Type) = _throw_ghost_unsupported("the element type $T (AD)")

@noinline function _throw_ghost_unsupported(what)
    throw(ArgumentError("forms on a ghost space do not support $what yet."))
end

# The map of `A = E_vᵀ A_pad E_u`: `A[r, c] = Σ E_v[i, r] A_pad[i, j] E_u[j, c]`, one
# contribution per stored `(i, j)` of `A_pad` and pair of stored entries in rows `i` of
# `E_v` and `j` of `E_u`. The reduced pattern holds every such `(r, c)`.
# Closures of different element types (a Float32 and a Float64 space) give weights of the
# promoted type.
function _ghost_reduction_map(A_pad::SparseMatrixCSC, Ev::SparseMatrixCSC,
        Eu::SparseMatrixCSC)
    Evt = sparse(transpose(Ev))
    Eut = sparse(transpose(Eu))
    src, rs, cs = Int[], Int[], Int[]
    coef = promote_type(eltype(Ev), eltype(Eu))[]
    rv, ev, eu = rowvals(A_pad), nonzeros(Evt), nonzeros(Eut)
    for j in axes(A_pad, 2), k in nzrange(A_pad, j)

        i = rv[k]
        for p in nzrange(Evt, i), q in nzrange(Eut, j)

            push!(src, k)
            push!(rs, rowvals(Evt)[p])
            push!(cs, rowvals(Eut)[q])
            push!(coef, ev[p] * eu[q])
        end
    end
    S = sparse(rs, cs, trues(length(rs)), size(Ev, 2), size(Eu, 2), |)
    srv = rowvals(S)
    dst = similar(src)
    for t in eachindex(dst)
        range = nzrange(S, cs[t])
        dst[t] = first(range) - 1 + searchsortedfirst(view(srv, range), rs[t])
    end
    return src, dst, coef, copy(SparseArrays.getcolptr(S)), copy(srv)
end

# --- Bilinear assembly --------------------------------------------------------------- #

function assemble(a::GhostBilinearForm; dirichlet = nothing, dirichlet_components = nothing)
    _check_ghost_dirichlet(dirichlet, dirichlet_components)
    return _assemble_ghost!(_reduced_matrix(a.A_pad, a), a)
end

# A matrix of the reduced pattern, of the type of `A_pad`'s entries times the closure
# weights; `_assemble_ghost!` zeroes it.
function _reduced_matrix(A_pad::SparseMatrixCSC, a::GhostBilinearForm)
    T = promote_type(eltype(A_pad), eltype(a.coef))
    return SparseMatrixCSC(ndofs(a.test_space), ndofs(a.trial_space), copy(a.colptr),
        copy(a.rowval), Vector{T}(undef, length(a.rowval)))
end

function assemble!(A::AbstractMatrix, a::GhostBilinearForm; dirichlet = nothing,
        dirichlet_components = nothing)
    _check_ghost_dirichlet(dirichlet, dirichlet_components)
    A isa SparseMatrixCSC && size(A) == (ndofs(a.test_space), ndofs(a.trial_space)) &&
    SparseArrays.getcolptr(A) == a.colptr && rowvals(A) == a.rowval ||
        _throw_ghost_pattern()
    return _assemble_ghost!(A, a)
end

function _assemble_ghost!(A::SparseMatrixCSC, a::GhostBilinearForm)
    assemble!(a.A_pad, a.padded)
    nz = nonzeros(A)
    pv = nonzeros(a.A_pad)
    fill!(nz, zero(eltype(nz)))
    src, dst, coef = a.src, a.dst, a.coef
    @inbounds for t in eachindex(src)
        nz[dst[t]] += coef[t] * pv[src[t]]
    end
    return A
end

@noinline function _throw_ghost_pattern()
    throw(
        ArgumentError(
        "assemble! on a ghost-space form fills a SparseMatrixCSC with the pattern of the " *
        "matrix assemble(a) returns; call A = assemble(a) once and refill that.",
    ),
    )
end

_check_ghost_dirichlet(::Nothing, ::Nothing) = nothing
function _check_ghost_dirichlet(dirichlet, components)
    _throw_ghost_unsupported("dirichlet (row replacement on the reduced system)")
end

# --- Linear assembly ----------------------------------------------------------------- #

function assemble(l::GhostLinearForm; dirichlet = nothing, dirichlet_components = nothing)
    b = similar(l.F_pad, ndofs(l.test_space))
    return assemble!(b, l; dirichlet = dirichlet, dirichlet_components = dirichlet_components)
end

function assemble!(b::AbstractVector, l::GhostLinearForm; dirichlet = nothing,
        dirichlet_components = nothing)
    _check_ghost_dirichlet(dirichlet, dirichlet_components)
    any(!iszero, l.test_space.d) && _throw_ghost_lift()
    _check_linear_length(b, l.test_space)
    assemble!(l.F_pad, l.padded)
    mul!(b, transpose(l.test_space.E), l.F_pad)
    return b
end

@noinline function _throw_ghost_lift()
    throw(
        ArgumentError(
        "the closure of this ghost space has an affine part d ≠ 0, so its load is " *
        "Eᵀ(F − A d) and needs the bilinear form: assemble both with assemble(a, l).",
    ),
    )
end

function assemble(a::GhostBilinearForm, l::GhostLinearForm; dirichlet = nothing,
        dirichlet_components = nothing, symmetrize::Bool = false)
    _check_ghost_dirichlet(dirichlet, dirichlet_components)
    symmetrize && _throw_symmetrize_without_dirichlet()
    size(l.F_pad, 1) == size(a.A_pad, 1) && ndofs(l.test_space) == ndofs(a.test_space) ||
        throw(DimensionMismatch("the forms' test spaces differ: $(a.test_space) and " *
                                "$(l.test_space)"))
    A = assemble(a)
    F = similar(l.F_pad, promote_type(eltype(l.F_pad), eltype(A)), ndofs(l.test_space))
    return A, _assemble_ghost_load!(F, a, l)
end

# `E_vᵀ (F_pad − A_pad d_u)`, `A_pad` as the last assembly of `a` left it. The lift goes
# into `F_pad` in place when its type holds `A_pad d_u`, and into a promoted copy otherwise
# (a complex form with a real load).
function _assemble_ghost_load!(F::AbstractVector, a::GhostBilinearForm, l::GhostLinearForm)
    assemble!(l.F_pad, l.padded)
    d = a.trial_space.d
    Eᵀ = transpose(l.test_space.E)
    if any(!iszero, d)
        mul!(F, Eᵀ, _subtract_lift(l.F_pad, a.A_pad, d))
    else
        mul!(F, Eᵀ, l.F_pad)
    end
    return F
end

function _subtract_lift(F_pad::AbstractVector, A_pad::SparseMatrixCSC, d::AbstractVector)
    T = promote_type(eltype(F_pad), eltype(A_pad), eltype(d))
    Fp = T === eltype(F_pad) ? F_pad : copyto!(similar(F_pad, T), F_pad)
    return mul!(Fp, A_pad, d, -1, 1)
end

# --- Refused paths ------------------------------------------------------------------- #

semidiscretize(::GhostBilinearForm, l; kwargs...) = _throw_ghost_unsupported("semidiscretize")
function semidiscretize(::Union{BilinearForm, Function}, ::GhostLinearForm; kwargs...)
    _throw_ghost_unsupported("semidiscretize")
end
function ode_problem(::GhostBilinearForm, l, u₀, tspan; kwargs...)
    _throw_ghost_unsupported("ode_problem")
end
function ode_problem(::BilinearForm, ::GhostLinearForm, u₀, tspan; kwargs...)
    _throw_ghost_unsupported("ode_problem")
end
function matrix_free_operator(::GhostBilinearForm; kwargs...)
    _throw_ghost_unsupported("matrix_free_operator")
end
kronecker_operator(::GhostBilinearForm) = _throw_ghost_unsupported("kronecker_operator")
function assemble_parallel!(::AbstractVecOrMat, ::Union{GhostBilinearForm, GhostLinearForm})
    _throw_ghost_unsupported("assemble_parallel!")
end
(::GhostLinearForm)(v) = _throw_ghost_unsupported("contracting a linear form")
function evaluate!(::AbstractVector, ::GhostLinearForm, v)
    _throw_ghost_unsupported("evaluate!")
end

# The AST of the form on the padded spaces.
resolve_form_ast(a::Union{GhostBilinearForm, GhostLinearForm}) = resolve_form_ast(a.padded)
