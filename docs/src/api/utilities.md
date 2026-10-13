```@meta
CollapsedDocStrings = false
CurrentModule = Bramble
```

# Utilities

## Linear algebra backends

```@docs
backend
Backend
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
supports_undef_construction
backend_eye
backend_zeros
ka_device
ka_synchronize
metal_sparse_csr
metal_sparse_csc
gpu_backend
metal_backend
csr_backend
profile_backends
BackendProfile
sparse_refactor!
PRECOMPILE_WORKLOAD
```

### Deprecated

```@docs
CpuBatch
GpuAsync
```
