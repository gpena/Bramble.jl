@generated function sub2ind(dims::NTuple{D}, I::CartesianIndex{D}) where D
    ex = :(I[$D] - 1)
    for i = (D - 1):-1:1
        ex = :(I[$i] - 1 + dims[$i] * $ex)
    end
    return :($ex + 1)
end

function _spmatmul!(C::SparseMatrixCSC{Tv,Ti}, A::SparseMatrixCSC{Tv,Ti}, B::SparseMatrixCSC{Tv,Ti}; xb::BitVector = falses(size(A)[1])) where {Tv,Ti}
    mA, nA = size(A)
    nB = size(B, 2)

    nnzC = min(SparseArrays.estimate_mulsize(mA, nnz(A), nA, nnz(B), nB) * 11 ÷ 10 + mA, mA*nB)
    colptrC = C.colptr
    rowvalC = C.rowval
    nzvalC = C.nzval
    resize!(colptrC, nB+1)
    resize!(rowvalC, nnzC)
    resize!(nzvalC, nnzC)

    @inbounds begin
        ip = 1
        for i in 1:nB
            if ip + mA - 1 > nnzC
                nnzC += max(mA, nnzC>>2)
                resize!(rowvalC, nnzC)
                resize!(nzvalC, nnzC)
            end
            colptrC[i] = ip
            ip = SparseArrays.spcolmul!(rowvalC, nzvalC, xb, i, ip, A, B)
        end
        colptrC[nB+1] = ip
    end

    resize!(rowvalC, ip - 1)
    resize!(nzvalC, ip - 1)
end

function _spmatmuladd!(C::SparseMatrixCSC{Tv,Ti}, A::NTuple{D,SparseMatrixCSC{Tv,Ti}}, B::NTuple{D,SparseMatrixCSC{Tv,Ti}}; buffer_sparse::SparseMatrixCSC = spzeros(Tv, size(C)), buffer_bitvector::BitVector = falses(size(C)[1])) where {D, Tv,Ti}
    mC, _ = size(C)
    rows = rowvals(C)

    for d in 1:D
        _spmatmul!(buffer_sparse, A[d], B[d], xb = buffer_bitvector)

        @inbounds for j = 1:mC
            @simd for i in nzrange(C, j) # can be made parallel
                C.nzval[i] += buffer_sparse[rows[i],j]
            end
        end

        buffer_bitvector .= false
    end
end