# ghost_contract.jl: `a(u, v)`, `l(v)` and `evaluate!` on forms posed on `GhostGridSpace`s
# or on composites of them, with the meanings they have on the reduced system `assemble(a)`,
# `assemble(l)`: the contraction `vᴴ A u` of `A = E_vᵀ A_pad E_u`, and `Fᵀ v` of
# `F = E_vᵀ F_pad` (#781). One method each serves both kinds of form, whose sides
# (`GhostGridSpace`, `_CompositeClosure`) have the same fields.
#
# Neither assembles the reduced system. `a(u, v)` contracts the form on the padded spaces
# with `u` and `v` lifted through the closures' linear parts, `(E_v v)ᴴ A_pad (E_u u)`
# (bilinear.jl's walk, no matrix), each lifted value summed where the walk reads it.
# `l(v)` fills the padded load `F_pad` the form stores and sums `F_padᵀ (E_v v)` over
# the columns of `E_v`.
#
# On a pair whose trial closure has an affine part `d ≠ 0` the load is
# `E_vᵀ (F_pad − A_pad d)` and needs the bilinear form. A load alone sees its test space
# only, so `l(v)` and `evaluate!` refuse a test space with `d ≠ 0` naming `assemble(a, l)`,
# as `assemble!(b, l)` does, which is exact when both sides are one space. `a(u, v)` is the
# contraction of `assemble(a)`, the linear part, whatever `d` is. `reaction(a, l, u)`
# refuses: the plain rows it reads are the relations'. So does `a(u, v)` on a polygon
# form that reads a conflict node, whose substitution lives in the reduction map alone.

"""
    (a::GhostBilinearForm)(u, v) -> Number
    (a::GhostCompositeBilinearForm)(u, v) -> Number

The form on ghost spaces, or on composites of them, evaluated at a trial function `u` and a
test function `v` of its degrees of freedom: `vᴴ A u` for `A = assemble(a)`, without `A`.
It lifts `u` and `v` to the padded spaces through the closures' linear parts and contracts
them with the form there, `(E_v v)ᴴ A_pad (E_u u)`, each padded value summed where the walk
reads it, so the call allocates only the one cell its sum runs in, as the plain form's
does. The affine part `d`
of the closures does not enter, as it does not enter `assemble(a)`.

`u` and `v` are [`VectorElement`](@ref)s or plain vectors, of lengths `ndofs` of the trial
and test spaces; a `DimensionMismatch` is thrown otherwise. On the spaces of
`polygon_spaces`, a form that reads a node two interior nodes extend differently (a
V-notch) throws an `ArgumentError`: only `assemble(a)` substitutes that node per equation.
"""
function (a::_AnyGhostBilinearForm)(u, v)
    ud, vd = _mf_data(u), _mf_data(v)
    Wₕ, Vₕ = a.trial_space, a.test_space
    _reads_conflict(a) && _throw_polygon_conflict("the contraction a(u, v)")
    (length(ud) == ndofs(Wₕ) && length(vd) == ndofs(Vₕ)) ||
        _throw_ghost_contraction_length(a, ud, vd)
    Base.require_one_based_indexing(ud, vd)
    return a.padded(_ClosureApplied{false}(Wₕ.Et, ud), _ClosureApplied{true}(Vₕ.Et, vd))
end

# The padded vector `E x` (`C = false`) or `conj(E) x` (`C = true`), each entry summed from
# column `i` of the stored `Eᵀ` when the walk reads it, so the contraction stores no padded
# vector. The conjugate is the test side's: the walk's `conj(V[i])` is then
# `Σ E_v[i, r] conj(v[r])`, which makes `conj(V)ᵀ A_pad U` equal `vᴴ E_vᵀ A_pad E_u u` for
# a complex closure too.
struct _ClosureApplied{C, T, IV <: AbstractVector{Int}, NV <: AbstractVector,
    XV <: AbstractVector} <: AbstractVector{T}
    colptr::IV
    rowval::IV
    nz::NV
    x::XV
end

function _ClosureApplied{C}(Et::SparseMatrixCSC, x::AbstractVector) where {C}
    T = promote_type(eltype(Et), eltype(x))
    return _ClosureApplied{C, T, typeof(rowvals(Et)), typeof(nonzeros(Et)), typeof(x)}(
        SparseArrays.getcolptr(Et), rowvals(Et), nonzeros(Et), x)
end

Base.size(y::_ClosureApplied) = (length(y.colptr) - 1,)

# The caller checked `x` against the closure's size, so the reads of `x` are in bounds.
@inline function Base.getindex(y::_ClosureApplied{C, T}, i::Int) where {C, T}
    @boundscheck checkbounds(y, i)
    s = zero(T)
    @inbounds for k in y.colptr[i]:(y.colptr[i + 1] - 1)
        e = C ? conj(y.nz[k]) : y.nz[k]
        s = convert(T, s + e * y.x[y.rowval[k]])
    end
    return s
end

@noinline function _throw_ghost_contraction_length(a::_AnyGhostBilinearForm, ud, vd)
    throw(
        DimensionMismatch(
        "a bilinear form on $(_ghost_kind(a.test_space, true)) with " *
        "$(ndofs(a.test_space))×$(ndofs(a.trial_space)) degrees of freedom cannot " *
        "contract a trial vector of length $(length(ud)) with a test vector of length " *
        "$(length(vd))",
    ),
    )
end

# The kind of space a side is, for the messages: one, or two (`plural`).
_ghost_kind(::GhostGridSpace, plural::Bool) = plural ? "ghost spaces" : "a ghost space"
function _ghost_kind(::_CompositeClosure, plural::Bool)
    return plural ? "composites of ghost spaces" : "a composite of ghost spaces"
end

"""
    (l::GhostLinearForm)(vₕ::VectorElement) -> Number
    (l::GhostCompositeLinearForm)(vₕ::VectorElement) -> Number

The linear form on a ghost space, or on a composite of them, evaluated at `vₕ`, an element
of its degrees of freedom: `Fᵀ v` for `F = assemble(l)`, without `F`. It fills the padded
load `F_pad` the form stores and sums `F_padᵀ (E v)`, allocating nothing; as
`assemble!(b, l)`, it is not safe to call concurrently on one form.

The load of a pair whose trial space has an affine part `d ≠ 0` is
`E_vᵀ (F_pad − A_pad d)`, which needs the bilinear form, `assemble(a, l)`. The call alone
refuses a test space with `d ≠ 0` with an `ArgumentError` naming `assemble(a, l)`, which is
exact when both sides are one space: on a pair whose trial space has data and whose test
space has none, it gives `E_vᵀ F_pad`, without the lift. On a space whose relation data
depend on time it refuses naming `semidiscretize`. A `vₕ` of the wrong length throws a
`DimensionMismatch`.
"""
function (l::_AnyGhostLinearForm)(vₕ::VectorElement)
    _check_ghost_load(l, "l(v)")
    vd = parent(vₕ)
    E = l.test_space.E
    length(vd) == size(E, 2) || _throw_ghost_load_length(l, vd)
    assemble!(l.F_pad, l.padded)
    T = promote_type(eltype(l.F_pad), eltype(E), eltype(vd))
    return _closure_contract(zero(T), SparseArrays.getcolptr(E), rowvals(E), nonzeros(E),
        l.F_pad, vd)
end

@noinline function (l::_AnyGhostLinearForm)(::AbstractVector)
    throw(
        ArgumentError(
        "a linear form contracts against an element of its test space, not a bare vector. " *
        "Name the space first, with l(element(test_space(l), v)).",
    ),
    )
end

# `Σ_r v[r] Σ_i E[i, r] F[i]`, column `r` of `E` at a time, without conjugating, as the
# plain `l(v)`.
function _closure_contract(s::T, colptr, rowval, nz, F, v) where {T}
    @inbounds for r in eachindex(v)
        w = zero(T)
        for k in colptr[r]:(colptr[r + 1] - 1)
            w = convert(T, w + nz[k] * F[rowval[k]])
        end
        s = convert(T, s + w * v[r])
    end
    return s
end

"""
    evaluate!(scratch::AbstractVector, l::GhostLinearForm, vₕ::VectorElement) -> Number
    evaluate!(scratch::AbstractVector, l::GhostCompositeLinearForm, vₕ::VectorElement) -> Number

Write the reduced load `assemble(l)` into `scratch`, of length `ndofs(test_space(l))`, and
return its contraction with `vₕ`, the value `l(vₕ)`. It refuses as `l(vₕ)` does: with an
`ArgumentError` naming `assemble(a, l)` on a test space whose closure has an affine part
`d ≠ 0`.
"""
function evaluate!(scratch::AbstractVector, l::_AnyGhostLinearForm, vₕ::VectorElement)
    _check_ghost_load(l, "evaluate!")
    length(parent(vₕ)) == ndofs(l.test_space) || _throw_ghost_load_length(l, parent(vₕ))
    assemble!(scratch, l)
    return _contract_values(scratch, parent(vₕ))
end

# The load alone exists only for a static test closure with `d = 0` (`assemble!(b, l)`).
function _check_ghost_load(l::_AnyGhostLinearForm, call::String)
    _is_time_dependent(l.test_space) && _throw_ghost_time(call)
    any(!iszero, l.test_space.d) && _throw_ghost_lift()
    return nothing
end

@noinline function _throw_ghost_load_length(l::_AnyGhostLinearForm, vd)
    throw(
        DimensionMismatch(
        "a linear form on $(_ghost_kind(l.test_space, false)) with " *
        "$(ndofs(l.test_space)) degrees of freedom cannot contract a vector of length " *
        "$(length(vd))",
    ),
    )
end

"""
    reaction(a::GhostBilinearForm, l::GhostLinearForm, uₕ::VectorElement; marker, dirichlet_components = nothing)

Refuses with an `ArgumentError`. A reaction reads the rows of the unconstrained plain
system at the marked nodes, and a ghost space has replaced those rows by its relations.
Pose `a` and `l` on the plain space `gridspace(Ωₕ)`, fill an element of it with the
physical values of `Bramble.lift(C, u)`, and call [`reaction`](@ref) there.
"""
function reaction(::GhostBilinearForm, ::GhostLinearForm, ::VectorElement;
        marker, dirichlet_components = nothing)
    return _throw_ghost_reaction()
end

@noinline function _throw_ghost_reaction()
    throw(
        ArgumentError(
        "reaction reads the rows of the unconstrained plain system at the marked nodes, " *
        "and on a ghost space the relations have replaced them. Assemble the form on the " *
        "plain space gridspace(Ωₕ), fill an element of it with the physical values of " *
        "Bramble.lift(C, u), and call reaction(a, l, uₕ; marker) there.",
    ),
    )
end
