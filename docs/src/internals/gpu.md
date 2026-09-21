```@meta
CollapsedDocStrings = false
CurrentModule = Bramble
```

# GPU acceleration

The [backend tutorial](../tutorials/backend.md) covers how to *choose* a GPU backend
([`metal_backend`](@ref), [`gpu_backend`](@ref)) and what it changes about calling
`Rₕ!`/`avgₕ!`/`assemble`. This page covers the machinery underneath: how a device kernel
gets from one source file to every backend, how a form gets assembled into device memory,
and the traps that machinery sprang while it was being built
([gpena/Bramble.jl#94](https://github.com/gpena/Bramble.jl/issues/94),
[#174](https://github.com/gpena/Bramble.jl/issues/174),
[#250](https://github.com/gpena/Bramble.jl/issues/250)).

**No performance claim is made anywhere on this page.** No benchmark in this milestone
measured the Metal backend against the CPU backend, so nothing here should be read as a
speedup.

## Storage decides locality; nothing declares it

A [`Backend`](@ref) is `Backend{VT, MT, EP}`: a vector type, a matrix type and an
[`ExecutionPolicy`](@ref), carried as type parameters alone. What decides whether a piece
of work can run on a device is not the policy -- it is [`locality`](@ref), and locality is
derived from the array type, never declared by a caller or a policy's name:

```julia
Bramble.locality(::Type{<:AbstractArray}) = Bramble.HostLocality()   # the fallback
Bramble.locality(::Type{<:MtlVector}) = Bramble.DeviceLocality()     # BrambleMetalExt's addition
```

[`CpuPolicy`](@ref) claims [`HostLocality`](@ref); [`GpuPolicy`](@ref) claims
[`DeviceLocality`](@ref). [`Backend`](@ref)'s inner constructor requires the vector type,
the matrix type and the policy to all agree on locality, and rejects the combination
outright otherwise
([gpena/Bramble.jl#298](https://github.com/gpena/Bramble.jl/issues/298)): a `GpuPolicy`
paired with host storage, or a `CpuPolicy` paired with device storage, is refused at
`Backend{...}()` rather than accepted and left to fail the first time a sweep scalar-indexes
device memory several frames later.

**This is the single most load-bearing rule for anyone adding a device method.** Every
device-specific method in the package keys on `::DeviceLocality`, never on `::GpuPolicy`:

```julia
@noinline _sweep_for!(::DeviceLocality, policy::GpuPolicy, v, idxs, f) = _gpu_for!(policy, v, idxs, f)
```

A method written `f(::GpuPolicy, ...)` instead of `f(::DeviceLocality, ::GpuPolicy, ...)`
is silently unreachable: `locality` is checked on the *storage*, and the dispatch tables
this package builds (`_sweep_for!`/`_sweep_scatter_for!` in
`src/utils/linear_algebra.jl`, `_resolve_device_mirror`/`_flush_device_scatter!`/
`_zero_stored!` in `src/form/bilinear_traversal.jl`) all branch on locality first, policy
second. Keying on the policy alone skips that branch and never gets called.

## The extension-contract idiom

A package extension cannot introduce a new binding into the parent module -- it can only
add methods to names the parent already declared. Every device seam in this package
follows the same shape because of that constraint: `src/` binds a stub function that
raises a helpful error naming the package to load, and the extension supplies the real
method once loaded. This is the same idiom `metal_backend`/`_metal_backend` and
`export_vtk`/`_export_vtk` already used before this milestone; [`ka_device`](@ref) (listed
in the [API reference](../api.md)) and [`ka_synchronize`](@ref) below are the two new
stubs it adds, and `_gpu_for!`/`_gpu_scatter_for!` and the `_launch_*!` family (declared
throughout `src/space/operators/` and `src/Bramble.jl`) follow the identical pattern for
the individual kernels.

```@autodocs
Modules = [Bramble]
Public = true
Private = false
Filter = x -> x == Bramble.ka_synchronize
Pages = ["utils/device_kernels.jl", ]
```

## What a new GPU backend has to implement

This is the entire return on building this milestone on `KernelAbstractions.jl` rather
than directly on `Metal.jl`, and it is what v3.5.0's CUDA/ROCm/oneAPI extensions
(gpena/Bramble.jl#11) depend on: **a new backend inherits every kernel in
`ext/BrambleKernelAbstractionsExt.jl` by supplying one [`ka_device`](@ref) method**,
naming its own `KernelAbstractions.Backend`. `ext/BrambleMetalExt.jl`'s entire
contribution to the kernel substrate is one line:

```julia
ka_device(::Backend{<:MtlVector, MT, EP}) where {MT, EP} = MetalBackend()
```

Nothing else about a `@kernel` in the KernelAbstractions extension changes: every one of
them is written against `KernelAbstractions.Backend` alone, launched as
`some_kernel!(dev)(args...; ndrange = ...)` where `dev = ka_device(backend)` (or
`get_backend(array)` for a kernel reached from an existing device array), so a future
`ka_device(::Backend{<:CuVector, MT, EP}) where {MT, EP} = CUDABackend()` picks up the
mesh, grid-space, difference, average and sparse SpMV/SpMM kernels with no further wiring.

A backend also needs `locality` and `_gpu_functional` methods for its own storage and
device probe (`BrambleMetalExt` supplies both, alongside `ka_device`), and typically its
own `vector`/`matrix`/`_backend_eye`/`_backend_zeros` allocation methods and a
`metal_backend`-style constructor -- but none of that touches the kernels themselves.
Once `ka_device` answers for a storage type, every `@kernel` in the extension is already
available on it.

## Where the kernels live, and where they do not

Every `@kernel` in the package lives in `ext/BrambleKernelAbstractionsExt.jl`. The file's
own header states the rule it holds itself to: written against
`KernelAbstractions.Backend` alone, never against `Metal.MtlVector` or any other concrete
device array type. The file mentions `Mtl`/`Metal` only inside comments stating this
agnosticism rule -- there is no executable reference to Metal anywhere in it. It carries:

- Mesh coordinate kernels (`_uniform_points_kernel!`, `_half_points_kernel!`,
  `_spacing_kernel!`, `_half_spacing_kernel!`, `_refine_indices_kernel!`), one per
  `src/mesh/mesh1d.jl` CPU loop they mirror.
- The generic `_gpu_for!`/`_gpu_scatter_for!` seam that `_sweep_for!`/`_sweep_scatter_for!`
  (`src/utils/linear_algebra.jl`) dispatch into for any `DeviceLocality` destination under
  a `GpuPolicy`.
- [`Rₕ!`](@ref)'s and `avgₕ!`'s device kernels, in 1D and `D >= 2` forms.
- The difference, jump and average operator kernels, one body each covering every
  dimension via a bits `NTuple{D,Int}` and `@index(Global, Cartesian)`.
- The row-parallel sparse SpMV/SpMM kernels (below).
- [`ka_synchronize`](@ref)'s real method, `synchronize(get_backend(x))`.

`ext/BrambleMetalExt.jl` holds only what is genuinely Metal-specific: the `ka_device` and
`locality` methods above, the Metal-side `vector`/`matrix`/`_backend_eye`/
`_backend_zeros` allocators, the `MetalSparseMatrixCSR`/`MetalSparseMatrixCSC` types
(placeholders behind an `isdefined(Metal, :MtlSparseMatrixCSR)` alias hook, since tagged
Metal.jl does not yet ship them -- JuliaGPU/Metal.jl#909), their `mul!` methods, and
`_allocate_from_pattern` for a Metal matrix type. None of these are `@kernel`s: they are
ordinary Julia methods that construct Metal-specific storage and forward the array
fields into the backend-agnostic kernels above.

## A struct nesting a device array will not compile

Every kernel above takes its arrays -- destinations, mesh coordinate vectors, an `NTuple`
of quadrature nodes/weights -- as separate, top-level kernel arguments, or as a `Tuple` of
them, and never nested inside a wrapper struct passed as one argument. Checked against a
real Metal device while designing this: a struct holding an `MtlVector` field fails to
compile the moment it is a kernel argument, with

```
GPUCompiler.KernelError: passing non-bitstype argument
```

however small the wrapper. A `Tuple` of genuine arrays is fine -- `Adapt.jl` converts it
element-wise -- which is why the `D >= 2` restriction and cell-average kernels take
`pts::NTuple{D}`/`x::NTuple{D}` rather than the mesh itself, and why the difference
kernels take `h` already resolved to a plain array or `nothing`
(`Bramble._resolve_device_spacing`) rather than the `StarSpacings` wrapper that holds it
on the host.

## Sparse device matrices: row-parallel SpMV and SpMM

A device CSR matrix (`MetalSparseMatrixCSR` today) is multiplied with two kernels,
`_launch_spmv_csr!` and `_launch_spmm_csr!`, both in the KernelAbstractions extension and
both written against the matrix's raw `rowPtr`/`colVal`/`nzVal` arrays -- never the struct
itself, for the same non-bitstype reason as above. Each work item owns one output row
(SpMV) or one entry of a row (SpMM):

```julia
@kernel function _spmv_csr_kernel!(y, @Const(rowPtr), @Const(colVal), @Const(nzVal), @Const(x), α, β)
    row = @index(Global)
    acc = zero(eltype(y))
    for k in rowPtr[row]:(rowPtr[row + 1] - 1)
        acc += nzVal[k] * x[colVal[k]]
    end
    y[row] = iszero(β) ? α * acc : α * acc + β * y[row]
end
```

One work item never writes an entry another work item also writes, so there are no
atomics anywhere in either kernel. `ext/BrambleMetalExt.jl`'s `mul!(y, A::MetalSparseMatrixCSR, x, α, β)`
does the dimension checks and forwards `A.rowPtr`, `A.colVal`, `A.nzVal` into
`_launch_spmv_csr!`; `MetalSparseMatrixCSC` has no `mul!` at all, matching
JuliaGPU/Metal.jl#909's own convention that a row-parallel kernel needs row-major storage
-- its `mul!` methods raise, naming `metal_sparse_csr` as the fix.

`_allocate_from_pattern` for a Metal matrix type (`ext/BrambleMetalExt.jl`) builds the
system matrix's sparsity host-side with `SparseArrays.sparse!` -- the same combiner the
`SparseMatrixCSC` method already uses -- and transfers it once with `metal_sparse_csr`.
The matrix is *born* device-resident; nothing scatters into it element by element on the
device.

## Form assembly on a device

The sparsity pattern is discovered host-side, exactly as for any other backend, and a
refill sweep runs afterward for every subsequent `assemble!`. Both walks evaluate the
form's AST through `local_stencil`, which reads a leaf space's weights and its mesh's
spacings one grid point at a time -- reading `MtlVector` element by element is exactly the
`Scalar indexing is disallowed` failure this milestone hit repeatedly elsewhere. Both
walks route around it the same way: `host_weights` returns a `ScalarGridSpace` that
mirrors the *whole* space -- mesh and weights together -- to the host in one bulk transfer
per call, and the walk reads that mirror instead of the device space. On a host backend
`host_weights` is a no-op (`host_weights(Wc) === Wc`, not a copy).

Scatter never writes a device array element by element either. `_scatter_add!` for a
`DeviceLocality` matrix accumulates into a host-resident mirror of the matrix's
`nzval`/`colval`/`rowptr` (`_resolve_device_mirror`, `src/form/bilinear_traversal.jl`),
resolved once before a sweep starts and threaded through every scatter call by hand rather
than looked up again per entry. `_flush_device_scatter!` ends the sweep with one
`copyto!(A.nzVal, mirror.nzval)` -- a single bulk transfer, not one write per nonzero.

## Traps worth knowing before touching any of this

- **A struct nesting a device array fails kernel compilation.** Covered above; pass plain
  arrays or a `Tuple` of them, never a mesh or a wrapper struct.
- **Write `/ 2`, not `* 0.5`.** A `Float64` literal forces double-precision arithmetic
  inside a kernel, and Apple Silicon GPUs do not support `Float64` at all. Dividing by the
  `Int` literal `2` promotes to the array's own float type instead.
- **`copyto!(::MtlArray{Bool}, ::BitVector)` trips the scalar-indexing guard** -- a
  `BitVector`'s packed storage is not a bulk-copyable source for Metal. Convert with
  `Vector{Bool}(mask)` first.
- **A host-side accessor left alone when the layer below it moved to device storage blocks
  the layer above it.** This shape bit the milestone four times: `_probe_point` (which
  `Rₕ`'s multi-dimensional probe called through `point(::MeshnD, ...)`), `spacing`/
  `forward_spacing` (called from `_stencil_weights` while building operator matrices), and
  `SeparableWeights`'s `__prod` (reached from both the pattern walk and the assembly
  sweep). Each fix was the same: replace the per-element read with one bulk transfer per
  call -- never one per grid point.
- **An asynchronous device write with nothing ordering it against a later write to the
  same buffer.** `assemble` returned before a device write had actually landed, and no
  `CHECK` in the milestone caught it: the milestone's own 33-point grid was too small and
  too fast to lose the race, and the bug only surfaced when a much larger, repeated
  full-stack assembly hit it (`n = 513`, then `n = 1025`). The first diagnosis -- the
  scatter flush's `copyto!` was unsynchronized -- was wrong; adding `ka_synchronize` there
  did not fix it, and neither did an extra `Metal.synchronize()` right after `assemble`
  returned. The actual second writer was `_zero_stored!`'s generic `fill!(nonzeros(A), 0)`,
  which is an asynchronous kernel on a device matrix with nothing ordering it against the
  later scatter flush; a late-landing zero-fill silently wiped entries the flush had
  already written. The fix removed the second writer instead of ordering the two:
  `_zero_stored!` for a `DeviceLocality` matrix now zeroes the host-resident mirror alone
  (`fill!(mirror.nzval, zero(...))`) rather than `A.nzVal` itself, since
  `_flush_device_scatter!` already overwrites every stored entry from the mirror
  unconditionally. Any future device write needs the same discipline: order it against
  whatever else touches the same buffer, or remove the second writer, and do not trust a
  small, single-run `CHECK` to have exercised the timing at all.

## Current limits

A masked [`Rₕ!`](@ref)/`avgₕ!` call (`markers` non-empty) on a `GpuPolicy` backend raises
rather than running, on either processor -- there is no device kernel for it yet
([gpena/Bramble.jl#297](https://github.com/gpena/Bramble.jl/issues/297), planned for
v3.5.0). A masked reduction (`innerₕ`/`normₕ` with a mask) does run on a device today; a
masked *projection* does not, and those are different mechanisms -- the mask folds into a
reduction in the first case, but would need folding into a per-point projection kernel in
the second.
