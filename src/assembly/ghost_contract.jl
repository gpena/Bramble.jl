# ghost_contract.jl: `a(u, v)`, `l(v)` and `evaluate!` on forms posed on `GhostGridSpace`s,
# with the meanings they have on the reduced system `assemble(a)`, `assemble(l)`: the
# contraction `vᴴ A u` of `A = E_vᵀ A_pad E_u`, and `Fᵀ v` of `F = E_vᵀ F_pad` (#781).
#
# Neither assembles the reduced system. `a(u, v)` contracts the form on the padded spaces
# with `u` and `v` lifted through the closures' linear parts, `(E_v v)ᴴ A_pad (E_u u)`
# (bilinear.jl's walk, no matrix), each lifted value summed where the walk reads it. `l(v)` fills the padded load
# `F_pad` the form stores and sums `F_padᵀ (E_v v)` over the columns of `E_v`.
#
# On a closure with an affine part `d ≠ 0` the load is `E_vᵀ (F_pad − A_pad d)` and needs
# the bilinear form, so `l(v)` and `evaluate!` refuse naming `assemble(a, l)`, as
# `assemble!(b, l)` does. `a(u, v)` is the contraction of `assemble(a)`, the linear part,
# whatever `d` is.

"""
    (a::GhostBilinearForm)(u, v) -> Number

The form on ghost spaces evaluated at a trial function `u` and a test function `v` of its
degrees of freedom: `vᴴ A u` for `A = assemble(a)`, without `A`. It lifts `u` and `v` to the
padded spaces through the closures' linear parts and contracts them with the form there,
`(E_v v)ᴴ A_pad (E_u u)`, each padded value summed where the walk reads it, so the call
allocates only the one cell its sum runs in, as the plain form's does. The affine part `d`
of the closures does not enter, as it does not enter `assemble(a)`.

`u` and `v` are [`VectorElement`](@ref)s or plain vectors, of lengths `ndofs` of the trial
and test spaces; a `DimensionMismatch` is thrown otherwise.
"""
function (a::GhostBilinearForm)(u, v)
    ud, vd = _mf_data(u), _mf_data(v)
    Wₕ, Vₕ = a.trial_space, a.test_space
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

@noinline function _throw_ghost_contraction_length(a::GhostBilinearForm, ud, vd)
    throw(
        DimensionMismatch(
        "a bilinear form on ghost spaces with $(ndofs(a.test_space))×" *
        "$(ndofs(a.trial_space)) degrees of freedom cannot contract a trial vector of " *
        "length $(length(ud)) with a test vector of length $(length(vd))",
    ),
    )
end

"""
    (l::GhostLinearForm)(vₕ::VectorElement) -> Number

The linear form on a ghost space evaluated at `vₕ`, an element of its degrees of freedom:
`Fᵀ v` for `F = assemble(l)`, without `F`. It fills the padded load `F_pad` the form
stores and sums `F_padᵀ (E v)`, allocating nothing; as `assemble!(b, l)`, it is not safe
to call concurrently on one form.

On a closure with an affine part `d ≠ 0` the load is `Eᵀ (F_pad − A_pad d)`, which needs
the bilinear form: the call refuses with an `ArgumentError` naming `assemble(a, l)`. On a
space whose relation data depend on time it refuses naming `semidiscretize`. A `vₕ` of the
wrong length throws a `DimensionMismatch`.
"""
function (l::GhostLinearForm)(vₕ::VectorElement)
    _check_ghost_load(l, "l(v)")
    vd = parent(vₕ)
    E = l.test_space.E
    length(vd) == size(E, 2) || _throw_ghost_load_length(l, vd)
    assemble!(l.F_pad, l.padded)
    T = promote_type(eltype(l.F_pad), eltype(E), eltype(vd))
    return _closure_contract(zero(T), SparseArrays.getcolptr(E), rowvals(E), nonzeros(E),
        l.F_pad, vd)
end

@noinline function (l::GhostLinearForm)(::AbstractVector)
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

Write the reduced load `assemble(l)` into `scratch`, of length `ndofs(test_space(l))`, and
return its contraction with `vₕ`, the value `l(vₕ)`. It refuses as `l(vₕ)` does: with an
`ArgumentError` naming `assemble(a, l)` on a closure with an affine part `d ≠ 0`.
"""
function evaluate!(scratch::AbstractVector, l::GhostLinearForm, vₕ::VectorElement)
    _check_ghost_load(l, "evaluate!")
    length(parent(vₕ)) == ndofs(l.test_space) || _throw_ghost_load_length(l, parent(vₕ))
    assemble!(scratch, l)
    return _contract_values(scratch, parent(vₕ))
end

# The load alone exists only for a static closure with `d = 0` (`assemble!(b, l)`).
function _check_ghost_load(l::GhostLinearForm, call::String)
    _is_time_dependent(l.test_space) && _throw_ghost_time(call)
    any(!iszero, l.test_space.d) && _throw_ghost_lift()
    return nothing
end

@noinline function _throw_ghost_load_length(l::GhostLinearForm, vd)
    throw(
        DimensionMismatch(
        "a linear form on a ghost space with $(ndofs(l.test_space)) degrees of freedom " *
        "cannot contract a vector of length $(length(vd))",
    ),
    )
end
