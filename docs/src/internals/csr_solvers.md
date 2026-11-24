```@meta
CurrentModule = Bramble
```

# CSR direct solve

No backend solver has a native `SparseMatrixCSR` path. `pde_solve`, `sparse_factorize` and
`refactor!` convert a CSR matrix to `SparseMatrixCSC` first, a re-layout of the triplets that
costs O(nnz) and reads the index base off `rowptr`, so a zero-based matrix converts too. The
function is `_csr_to_csc` in `ext/BrambleSparseMatricesCSRExt.jl`, which adds the
`SparseMatrixCSR` methods of these entry points and of `accelerate_factorize`.

## Refilling a `VectorElement` from a factorization

```@docs
ldiv!(::VectorElement, ::Factorization, ::AbstractVector)
```
