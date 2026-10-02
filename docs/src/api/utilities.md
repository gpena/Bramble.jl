```@meta
CollapsedDocStrings = false
CurrentModule = Bramble
```

# Utilities

## Linear algebra backends

The backend allocation hooks (`backend_eye`, `backend_zeros`, `ka_device`,
`supports_undef_construction`) are private; see the
[utilities internals page](../internals/utils.md).

```@docs
backend
Locality
HostLocality
DeviceLocality
locality
ExecutionPolicy
CpuPolicy
CpuSerial
CpuThreaded
CpuPolyester
GpuOffload
GpuPolicy
GpuKernel
Serial
Parallel
execution_policy
vector
matrix
vector_type
matrix_type
backend_types
metal_sparse_csr
metal_sparse_csc
gpu_backend
metal_backend
csr_backend
sparse_refactor!
PRECOMPILE_WORKLOAD
```

### Deprecated

```@docs
CpuBatch
GpuAsync
```
