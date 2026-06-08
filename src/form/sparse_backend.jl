# Shared sparse backend traits and default CSC implementation

"""
    AbstractSparseKernel

Abstract base type for sparse backend kernels.
"""
abstract type AbstractSparseKernel end

"""
    CSCSparseKernel <: AbstractSparseKernel

CSC sparse matrix backend kernel implementing the sparse interface for standard Julia SparseMatrixCSC matrices.
"""
struct CSCSparseKernel <: AbstractSparseKernel end

"""
    sparse_kernel_for(b)

Selects and returns the appropriate sparse backend kernel based on the backend `b`.
"""
@inline function sparse_kernel_for(b)
	_sparse_kernel_for(matrix_type(b))
end

@inline _sparse_kernel_for(::Type{<:SparseMatrixCSC}) = CSCSparseKernel()
@inline function _sparse_kernel_for(mt)
	error("No sparse kernel implemented for matrix type $(mt). Provide a backend-specific kernel implementing the sparse backend interface.")
end

# --- Sparse construction helpers ---
@inline function backend_sparse(::CSCSparseKernel, op, rows::Int, cols::Int)
	return ensure_sparse(op, rows, cols)
end

"""
    ensure_sparse(Op, rows, cols)

Converts the operator or function `Op` to a sparse matrix CSC of size `(rows, cols)`.
"""
function ensure_sparse(Op, rows, cols)

	if Op isa SparseMatrixCSC
		return Op
	elseif Op isa Diagonal
		return sparse(Op)
	elseif Op isa UniformScaling
		return sparse(Op, rows, cols)
	elseif Op isa VectorElement
		return _sparse_diag_ref(Op.data)
	elseif Op isa AbstractVector
		return _sparse_diag_ref(Op)
	else
		return sparse(Op)
	end
end

@inline backend_rowvals(::CSCSparseKernel, A::SparseMatrixCSC) = rowvals(A)
@inline backend_nonzeros(::CSCSparseKernel, A::SparseMatrixCSC) = nonzeros(A)
@inline backend_nzrange(::CSCSparseKernel, A::SparseMatrixCSC, j) = nzrange(A, j)

@inline function backend_matrix_from_pattern(::CSCSparseKernel, rows::Int, cols::Int, colptr, rowval, nzval)
	return SparseMatrixCSC(rows, cols, colptr, rowval, nzval)
end

@inline backend_nnz(::CSCSparseKernel, A::SparseMatrixCSC) = nnz(A)

"""
    compute_pattern(kernel::CSCSparseKernel, terms, rows::Int, cols::Int)

Computes the overall sparsity pattern for a set of bilinear terms, returning a sparse matrix of booleans.
"""
function compute_pattern(kernel::CSCSparseKernel, terms, rows::Int, cols::Int)

	function get_term_pattern(t)
		SB = t.B
		SC = t.C
		tc = transpose(SC) * SB
		return SparseMatrixCSC(tc.m, tc.n, tc.colptr, tc.rowval, ones(Bool, nnz(tc)))
	end

	overall_pattern = spzeros(Bool, rows, cols)
	for t in terms
		p = get_term_pattern(t)
		overall_pattern = overall_pattern + p
	end
	map!(x -> true, nonzeros(overall_pattern), nonzeros(overall_pattern))
	return overall_pattern
end

# Helper for diagonal sparse ref (copied from existing code)
@inline function _sparse_diag_ref(v::AbstractVector)
	n = length(v)
	colptr = collect(1:(n + 1))
	rowval = collect(1:n)
	return SparseMatrixCSC(n, n, colptr, rowval, v)
end
