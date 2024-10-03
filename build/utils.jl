#=function spmatmul_rvec!(M::AbstractSparseMatrix, v::AbstractVector) 
    @assert length(v) == size(M)[1] == size(M)[2]

    m, _ = size(M)

    for j = 1:m
        vj = v[j]
        @simd for i in nzrange(M, j)
             M.nzval[i] *= vj
        end
    end
end


function spmatmul_lvec!(M::AbstractSparseMatrix, v::AbstractVector) 
    @assert length(v) == size(M)[1] == size(M)[2]
    rows = rowvals(M)
    m, _ = size(M)

    for j = 1:m
        vj = v[rows[j]]
        @simd for i in nzrange(M, j)
             M.nzval[i] *= vj
        end
    end
end
=#