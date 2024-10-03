test_space(b::BilinearFormType) = b.space1
trial_space(b::BilinearFormType) = b.space2

BilinearForm(f::F, s1::S1Type, s2::S2Type) where {S1Type, S2Type, F<:Function} = BilinearForm{S1Type,S2Type,F}(s1, s2, f)

function assemble(m::BilinearFormType, bcs::DirichletBCs) 
    M = assemble(m)
    apply_dirichlet_bc!(M, bcs, mesh(m.space1))

    return M
end

function assemble!(A::AbstractMatrix, m::BilinearFormType, bcs::DirichletBCs) 
    assemble!(A, m)
    apply_dirichlet_bc!(A, bcs, mesh(m.space1))
end










## implementation of mass operator with scaling: BilinearForm((U, V) -> innerₕ(uh*U, V), Wh, Wh)
function Mass(s1::SpaceType, s2::SpaceType) 
    vec_ones = Vector{eltype(s1)}(undef, ndofs(s1))
    #vec_ones .= zero(eltype(s1))
    vec_ones .= one(eltype(s1))
    #vec_ones = vec(__ones(eltype(s1), ndofs(s1)))

    BilinearMass(s1, s2, vec_ones)
end

update!(m::BilinearMass, v::AbstractFloat) = ( fill!(m.vec, v) )
update!(m::BilinearMass, v::VectorElement) = ( copyto!(m.vec, v) )

function assemble(m::BilinearMass) 
    E = sparse(Eye(ndofs(m.space1)))
    assemble!(E, m)

    return E
end

@inline assemble!(A::SparseMatrixCSC, m::BilinearMass) = ( A.nzval .= m.vec .* innerh_weights(m.space1) )
 

## implementation of stiffness operator with scaling
function Diff(s1::SpaceType, s2::SpaceType)
    id = elements(s1)
    T = eltype(s1)
    D = dim(mesh(s1))
    Uh = ∇₋ₕ(id)
    Vh = ∇₋ₕ(elements(s2))
    E = zero(eltype(s1))*id.values #spzeros(eltype(s1), ndofs(s1), ndofs(s2))
    S = D == 1 ? BilinearDiff{typeof(s1),typeof(s2), D, T}(s1, s2, (Uh,), (Vh,), E, falses(ndofs(s1))) : BilinearDiff{typeof(s1),typeof(s2), D, T}(s1, s2, Uh, Vh, E, falses(ndofs(s1)))
    update!(S, one(eltype(s1)))
    return S
end

_stiffness_update!(M::SparseMatrixCSC, v::VectorElement) = ( spmatmul_rvec!(M, v.values) )
_stiffness_update!(M::SparseMatrixCSC, v::AbstractFloat) = ( M.nzval .*= v )

function update!(m::BilinearDiff{S1, S2, D}, v::Union{AbstractFloat, VectorElement}) where {S1, S2, D}
    update!(m, ntuple(i-> v, D))
end


function update!(m::BilinearDiff{S1, S2, D, MType}, v::NTuple{D,T}) where {S1, S2, D, T, MType} 
    for i in 1:D
        ftranspose!(m.aux[i].values, m.grad[i].values, identity)

        _stiffness_update!(m.aux[i].values, v[i])
        x = innerplus_weights(m.space1, Val(i))
        spmatmul_rvec!(m.aux[i].values, x)
    end
end

#=
function assemble(m::BilinearDiff{S1, S2, D}) where {S1, S2, D}
    A = spzeros(ndofs(m.space1), ndofs(m.space2))

    @inbounds @simd for i in 1:D
        A += m.aux[i].values * m.grad[i].values
    end

    return A
end
=#

function assemble(m::BilinearDiff{S1, S2, 1}) where {S1, S2}
    return m.aux[1].values * m.grad[1].values
end

function assemble(m::BilinearDiff{S1, S2, 2}) where {S1, S2}
    return m.aux[1].values * m.grad[1].values + m.aux[2].values * m.grad[2].values
end

function assemble(m::BilinearDiff{S1, S2, 3}) where {S1, S2}
    return m.aux[1].values * m.grad[1].values + m.aux[2].values * m.grad[2].values + m.aux[3].values * m.grad[3].values
end


function assemble!(A::SparseMatrixCSC{T,Int}, m::BilinearDiff{S1, S2, D, T}) where {S1, S2, D, T} 
    A.nzval .= zero(T)
    M1 = ntuple(d->m.aux[d].values, D)
    M2 = ntuple(d->m.grad[d].values, D)

    _spmatmuladd!(A, M1, M2, buffer_sparse = m.buffer_sparse, buffer_bitvector = m.buffer_bitvector)
end

#=
function spgemm_gustavson!(out, _A::NTuple{1}, _B::NTuple{1})
    A = Tensor(_A[1])
    B = Tensor(_B[1])
    z = fill_value(A) * fill_value(B) + false
    C = Tensor(Dense(SparseList(Finch.Element(z))))
    w = Tensor(SparseByteMap(Finch.Element(z)))

    @finch begin
        C .= 0
        for j=_
            w .= 0
            for k=_, i=_; w[i] += A[i, k] * B[k, j] end
            for i=_; C[i, j] = w[i] end
        end
    end

    Finch.copyto!(out, C)
end

function spgemm_gustavson!(out, _A::NTuple{2}, _B::NTuple{2})
    A1 = Tensor(_A[1])
    B1 = Tensor(_B[1])
    A2 = Tensor(_A[2])
    B2 = Tensor(_B[2])
    z = fill_value(A1) * fill_value(B1) + fill_value(A2) * fill_value(B2) + false
    C = Tensor(Dense(SparseList(Finch.Element(z))))
    w = Tensor(SparseByteMap(Finch.Element(z)))

    @finch begin
        C .= 0
        for j=_
            w .= 0
            for k=_, i=_; w[i] += A1[i, k] * B1[k, j] + A2[i, k] * B2[k, j] end
            for i=_; C[i, j] = w[i] end
        end
    end

    Finch.copyto!(out, C)
end

function spgemm_gustavson!(out, _A::NTuple{3}, _B::NTuple{3})
    A1 = _A[1]
    B1 = _B[1]
    A2 = _A[2]
    B2 = _B[2]
    A3 = _A[3]
    B3 = _B[3]
    z = fill_value(A1) * fill_value(B1) + fill_value(A2) * fill_value(B2) + fill_value(A3) * fill_value(B3) + false
    C = Tensor(Dense(SparseList(Finch.Element(z))))
    w = Tensor(SparseByteMap(Finch.Element(z)))

    @finch begin
        C .= 0
        for j=_
            w .= 0
            for k=_, i=_; w[i] += A1[i, k] * B1[k, j] + A2[i, k] * B2[k, j] + A3[i, k] * B3[k, j] end
            for i=_; C[i, j] = w[i] end
        end
    end

    Finch.copyto!(out, C)
end

function assemble2!(A::AbstractSparseMatrix, m::BilinearDiff{S1, S2, D, T}) where {S1, S2, D, T} 
    A.nzval .= zero(T)
    M1 = ntuple(d->m.aux[d].values, D)
    M2 = ntuple(d->m.grad[d].values, D)

    spgemm_gustavson!(A, M1, M2)
    #_spmatmuladd!(A, M1, M2, buffer_sparse = m.buffer_sparse, buffer_bitvector = m.buffer_bitvector)
end
=#


## implementation of advection operator with scaling
#=
function Advection(s1::SpaceType, s2::SpaceType)
    E = spzeros(eltype(s1), ndofs(s1), ndofs(s2))
    S = BilinearDiff{typeof(s1),typeof(s2), dim(s1), eltype(s1)}(s1, s2, ∇₋ₕ(elements(s1)), ∇₋ₕ(elements(s2)), E, falses(ndofs(s1)))
    update!(S, one(eltype(s1)))
    return S
end

_stiffness_update!(M::AbstractSparseMatrix, v::VectorElement) = ( spmatmul_rvec!(M, v.values) )
_stiffness_update!(M::AbstractSparseMatrix, v::AbstractFloat) = ( @turbo M.nzval .*= v )

function update!(m::BilinearDiff{S1, S2, D, T}, v::Union{AbstractFloat, VectorElement}) where {S1, S2, D, T}
    update!(m, ntuple(i-> v, D))
end

function update!(m::BilinearDiff, v::NTuple{D,Union{AbstractFloat, VectorElement}}) where D 
    for i in 1:D
        ftranspose!(m.aux[i].values, m.grad[i].values, identity)

        _stiffness_update!(m.aux[i].values, v[i])

        spmatmul_rvec!(m.aux[i].values, innerplus_weights(m.space1, Val(i)).diag)
    end
end

function assemble(m::BilinearDiff{S1, S2, D, T}) where {S1, S2, D, T}
    A = spzeros(ndofs(m.space1), ndofs(m.space2))

    @inbounds for i in 1:D
        A += m.aux[i].values * m.grad[i].values
    end

    return A
end

function assemble!(A::AbstractSparseMatrix, m::BilinearDiff{S1, S2, D, T}) where {S1, S2, D, T} 
    A.nzval .= zero(T)
    M1 = ntuple(d->m.aux[d].values, D)
    M2 = ntuple(d->m.grad[d].values, D)

    _spmatmuladd!(A, M1, M2, buffer_sparse = m.buffer_sparse, buffer_bitvector = m.buffer_bitvector)
end

=#
















ScalingType = Union{T,VecOrMatElem} where T<:AbstractFloat

_mat_scale!(x::AbstractVector, y::AbstractFloat) = _mat_scale!(x, y, Val{!isequal(y, 1.0)}())
_mat_scale!(x, y::AbstractFloat, ::Val{true}) = ( x .*= y )
_mat_scale!(_, _, ::Val{false}) = nothing
_mat_scale!(x::AbstractVector,y::VecOrMatElem) = ( x .*= y.values )


# special implementation of recurrent bilinear forms for GridSpaces
mass(U::VecOrMatElem, V::VecOrMatElem; scaling::ScalingType = 1.0) = mass(U, V, Val(dim(mesh(space(U)))), scaling)

function mass(U::VecOrMatElem,_,::Val{1}, scaling::ScalingType)
    u = _create_vector(test_space(U))
    #D = Diagonal(Vector{eltype(U)}(undef, ndofs(mesh(U))))
    @. u = hmean_extended(mesh(test_space(U)))
    _mat_scale!(u, scaling)

    return D
end

stiffness(U::VecOrMatElem, V::VecOrMatElem; scaling::ScalingType = 1.0) = stiffness(U, V, Val(dim(mesh(space(U)))), scaling)

function stiffness(U::VecOrMatElem, _, ::Val{1}, scaling::ScalingType)
    # implements a faster version of innerplus(U,V) = (D_x(U),D_x(V))_+ with scaling
    # in 1D
    # the result will be tridiagonal
    _h = copy(collect(hspaceit(mesh(U))))
    _x = similar(_h)
   
    map!(inv, _x, _h)

    _mat_scale!(_x, scaling)

    @views x = _x[2:end]
    @views b = [x[1]; x[1:end-1] + x[2:end]; x[end]]

    return spdiagm(0 => b, -1 => -x, 1 => -x)
end


_update_advection_vec!(b::AbstractVector,a::AbstractVector,::Val{true}) = ( @views b[2:end-1] .= a[2:end-1] .- a[3:end])
_update_advection_vec!(_, _, ::Val{false}) = nothing

advection(U::VecOrMatElem, V::VecOrMatElem; scaling::ScalingType = 1.0) = advection(U, V, Val(dim(mesh(space(U)))), scaling)

function advection(U::VecOrMatElem, _, ::Val{1}, scaling::ScalingType)
    N = ndofs(space(U))
    T = eltype(U)
    b = zeros(T, N)
    a = ones(T, N)

    _mat_scale!(a, scaling)

    _update_advection_vec!(b, a, Val(scaling isa VectorElement))

    b[1] = -a[2]
    b[end] = a[end]

    @views aux = a[2:end]

    return UniformScaling(convert(T, 0.5))*spdiagm(0 => b, -1 => aux, 1 => -aux)
end