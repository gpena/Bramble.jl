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

**No general performance claim is made on this page.** Every figure below is a specific
measurement, cited beside the script, sizes and machine state it came from, on one Apple M2
host; none should be read as a speedup beyond those conditions.

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
using Bramble: GpuPolicy

@noinline _sweep_for!(::DeviceLocality, policy::GpuPolicy, v, idxs, f) = _gpu_for!(policy, v, idxs, f)
```

A method written `f(::GpuPolicy, ...)` instead of `f(::DeviceLocality, ::GpuPolicy, ...)`
is silently unreachable: `locality` is checked on the *storage*, and the dispatch tables
this package builds (`_sweep_for!`/`_sweep_scatter_for!` in
`src/utils/linear_algebra.jl`, `_scatter_position`/`_scatter_add!`/`_flush_device_scatter!`/
`_zero_stored!` in `src/assembly/bilinear_traversal.jl`) all branch on locality first, policy
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
Public = false
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

**What a device sparse type must provide.** A device-resident CSR matrix that wants to
plug into bilinear form assembly (gpena/Bramble.jl#313) carries a field named `mirror`,
holding `rowptr`/`colval`/`nzval` host-side `Vector`s: `rowptr`/`colval` are host copies of
the matrix's own CSR structure, carried in the matrix's own index type `Ti` rather than
widened to `Int` -- on hardware where the device and host share physical memory (Apple
Silicon's unified DRAM), an `Int`-hardcoded mirror would cost more host memory than the
device arrays it stages for the moment `Ti` is narrower than `Int` -- and `nzval` is what a
scatter into the matrix actually accumulates into (the device array itself is left alone
until the sweep finishes). `src/assembly/bilinear_traversal.jl`'s `_scatter_position`/
`_scatter_add!` read `A.mirror` straight off the matrix under `::DeviceLocality`, duck-typed
rather than dispatched on a concrete type -- this file has no dependency on Metal or
`GPUArrays` and cannot name `MetalSparseMatrixCSR` -- so any type providing this one field,
of this shape, is a sparsity-search-and-scatter-ready backend with no further code in
`src/assembly/` at all. `MetalSparseMatrixCSR` (`ext/BrambleMetalExt.jl`) builds its `mirror`
once, when the matrix itself is built, straight from the host `Vector{Ti}`s already on hand
before they are uploaded to the device -- no conversion, no separate transfer.

## Where the kernels live, and where they do not

Every `@kernel` in the package lives in `ext/BrambleKernelAbstractionsExt.jl`. The file's
own header states the rule it holds itself to: written against
`KernelAbstractions.Backend` alone, never against `Metal.MtlVector` or any other concrete
device array type. The file mentions `Mtl`/`Metal` only inside comments stating this
agnosticism rule -- there is no executable reference to Metal anywhere in it. It carries:

- Mesh coordinate kernels (`_half_points_kernel!`,
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
device. `metal_sparse_csr` also builds the matrix's `mirror` field at this point (below),
straight from the host CSR arrays already in hand.

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
`DeviceLocality` matrix accumulates into `A`'s own `mirror` field
(`src/assembly/bilinear_traversal.jl`) -- read straight off the matrix, never resolved from a
cache -- and `_flush_device_scatter!` ends the sweep with one
`copyto!(A.nzVal, A.mirror.nzval)`, a single bulk transfer, not one write per nonzero. See
"What a device sparse type must provide" above for the field's shape.

**Read the mirror for what it actually is: the fill step performs no device compute.**
Every scatter-add above happens on the CPU, into the mirror's plain `Vector` storage, on
every backend. The device sparse matrix's own storage is touched exactly once per
assembly, by the bulk `copyto!` at the end. "Assembling on a device-backed space" is CPU
assembly plus an upload -- there is no device-side fill step to have sped up or slowed
down. Measured on this host (`innerₕ(D₋ₓ(U), D₋ₓ(V))`, `BenchmarkTools`, one warmed
process): device `assemble`/`assemble!` are 2-8x slower than the host path, in 1D, 2D and
3D, both forms, with no offsetting benefit anywhere in the number, because the device
never does anything the host wasn't already going to do. Full detail and the table:
[gpena/Bramble.jl#317](https://github.com/gpena/Bramble.jl/issues/317#issuecomment-5764380143).
Treat this path as a correctness fallback -- something that produces *a* device-resident
matrix for a consumer that needs entries, not a performance path. #317 has decided that the
fill stays on the host and is not moved to the device ("The device architecture decision",
below).

## The device architecture decision

[gpena/Bramble.jl#317](https://github.com/gpena/Bramble.jl/issues/317) asked whether large
problems on a device should get full device assembly (a form's stencil traversal compiled to
a device kernel, filling the matrix in device memory) or matrix-free operators. Its second
amendment reframed the question for a second-order finite-difference package: a five-point
apply and the assembled SpMV do about the same arithmetic on about the same data, so there
is no high-order sum-factorisation gain to collect, and matrix-free buys memory rather than
flops. The question is therefore a memory question: at what grid size does storing the
matrix stop being affordable, what still needs matrix entries, and does the assembled path
need any device-assembly machinery below that size?

**Decision.** Host assembly plus one upload stays the path for everything that needs
matrix entries. Matrix-free [`KroneckerLinearOperator`](@ref) covers the operator applies it
can express (the separable forms below). Full device assembly -- the atomic-scatter,
warp-interleaved-table architecture recorded on #317 -- is **not built**. The rule was fixed
before the numbers came in: if the assembled matrix fits device memory at every size the
measurement reaches, and a consumer of entries assembles at most once per nonlinear solve,
this is the direction; otherwise the decision would name the size where the matrix stops
fitting and select a device scatter instead. The matrix fitted at every size measured, so
the first branch holds. The evidence follows.

### What needs matrix entries rather than an operator's action

Four kinds of Bramble algorithm read the entries of a matrix instead of applying an
operator:

- **AlgebraicMultigrid** ([`amg_preconditioner`](@ref), `BrambleAlgebraicMultigridExt`):
  the hierarchy's setup coarsens from the graph and values of `A`. The `BilinearForm`
  method assembles the full form it is given; the `AbstractMatrix` method accepts any matrix,
  so a coarser or lower-order proxy could be passed in, but Bramble builds no such proxy
  today and none has been measured.
- **ILUZero** ([`ilu_preconditioner`](@ref), `BrambleILUZeroExt`): ILU(0) factors reuse
  `A`'s own sparsity pattern, so it needs that pattern and its values. The same two methods
  as for AMG, the same absence of a proxy.
- **The sparse direct solvers** (`sparse_factorize`, `pde_solve`, `refactor!`): SuiteSparse,
  MUMPS, Sparspak and Apple Accelerate, which is `pde_solve`'s macOS default for a symmetric
  system once `AppleAccelerate.jl` is loaded (`_default_wants_accelerate`,
  `src/solvers/pde_solve.jl`). A direct factorisation needs the exact operator being solved,
  so no proxy applies, and it has no matrix-free fallback at all. MUMPS, Accelerate and
  Sparspak are pinned to a host `SparseMatrixCSC` ([CSR in the solvers](csr_solvers.md)), so
  every one of them consumes a host matrix whichever backend the form was built on.
- **Eigenvalue work.** Bramble has no eigenvalue solver of its own. The one eigensolve in the
  package is [`fdm_solve`](@ref)'s per-axis `LAPACK` call on the 1D Kronecker factors
  (below), which needs only those small factors, never the assembled matrix. An eigenvalue
  computation a user runs on an assembled matrix is an entries consumer like the others.

All four build their object -- a hierarchy, incomplete factors, a factorisation -- once at
setup and then apply it many times. That is the assembly frequency the decision rule asks
about: the entries path runs when a preconditioner or factorisation is (re)built, not on
every operator apply.

### What each design covers, and what the host path stays responsible for

**Matrix-free Kronecker** (`src/assembly/kronecker.jl`) covers exactly what
[`is_separable`](@ref) accepts: a non-composite [`ScalarGridSpace`](@ref) on a mesh of
dimension `D >= 2`, trial and test sharing one mesh, and a sum of mass terms `innerₕ(u, v)`
and one-axis backward-difference stiffness terms `inner₊(D₋ₓ(u), D₋ₓ(v))` (what
`inner₊(∇ₕ(u), ∇ₕ(v))` expands to), each under at most a constant or `Ref` scalar. Everything
else is refused rather than approximated: grid-function coefficients, region restrictions
(so Dirichlet rows), interpolation across meshes, surface (`InnerGamma`) weights, composite
spaces, 1D meshes, and every other difference, average or jump family. The operator has no
boundary constraint of its own; [`fdm_solve`](@ref)'s `dirichlet = :boundary` branch
restricts to the interior around it (below).

**Host assembly plus upload** stays responsible for everything else: every non-separable
form, every Dirichlet-constrained system (`apply_dirichlet_labels!` runs as its own pass
after the scatter), every composite space, and every matrix an entries consumer above
needs, separable or not. It is also the correctness reference for the Kronecker path. One
limit is worth stating here because it bounds the fallback on Metal: assembling a
*composite* form on a Metal-backed space fails today, before any scatter, with
`scalar getindex on a device-backed SeparableWeights is not supported` (checked on this host
for `W × W` in 1D). That is a separate defect in the element-type probe, not a consequence of
this decision; a composite system on Metal has no working assembled path until it is fixed.

### The evidence

**Memory and throughput: assembled CSR against Kronecker on Metal.**
`benchmark/assembled_vs_matrixfree.jl --full` (S6.1 of the v3.14.0 plan, commit `9051ac58`)
compared, for the same separable operator `innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v))`, a host
`assemble` uploaded once with `metal_sparse_csr` against a Metal-backed
`kronecker_operator`, both `Float32`, after checking host-against-device agreement for both
at every size. Apple M2, recommended working set 12.71 GB, AC power, load 2.42-3.28 against a
threshold of 4.0, `--threads=4`:

| Case | N (dofs) | CSR bytes | Kronecker bytes | CSR/Kron | CSR share of working set | SpMV (ms) | Kron apply (ms) | SpMV/Kron |
|---|---|---|---|---|---|---|---|---|
| 2D 500² | 250,000 | 16,976,008 | 35,976 | 471.87 | 0.13% | 0.6747 | 0.6507 | 1.037 |
| 2D 1500² | 2,250,000 | 152,928,008 | 107,976 | 1416.31 | 1.20% | 2.0782 | 1.3717 | 1.515 |
| 2D 3000² | 9,000,000 | 611,856,008 | 215,976 | 2832.98 | 4.81% | 7.645 | 4.7157 | 1.621 |
| 3D 60³ | 216,000 | 19,612,808 | 7,164 | 2737.69 | 0.15% | 0.9352 | 0.6802 | 1.375 |
| 3D 120³ | 1,728,000 | 157,939,208 | 14,364 | 10995.5 | 1.24% | 2.1991 | 1.8521 | 1.187 |
| 3D 200³ | 8,000,000 | 733,120,008 | 23,964 | 30592.6 | 5.77% | 9.5458 | 7.7066 | 1.239 |

- **There is no memory crossover in the measured range.** The assembled matrix never
  exceeds 5.77% of the device's recommended working set, at the largest size measured (3D
  200³, 8,000,000 dofs). The device CSR's own bytes are what the table counts; the host
  mirror that the fill path accumulates into holds roughly a second copy in the same unified
  DRAM (#316's own description), which at most doubles the 5.77% figure -- still a small
  fraction of the working set. Kronecker factors are 472x to 30,593x smaller than the matrix.
- **Matrix-free costs no throughput.** The second amendment kept the throughput
  measurement only as the check that matrix-free does not cost throughput at low order. The
  Kronecker apply was faster than the assembled SpMV at every size: 1.04-1.62x in 2D, growing
  with size, and 1.19-1.38x in 3D, where the margin does not grow monotonically.

**Assembly frequency: assemble once against re-assemble every step.** The same run timed
both regimes over 100 steps, computed from one measured build and one measured apply per
arm (`build + 100 * apply`, and `100 * (build + apply)`):

| Case | CSR build (ms) | Kron build (ms) | Once + 100 applies, CSR/Kron | Re-assemble every step, CSR/Kron |
|---|---|---|---|---|
| 2D 500² | 30.85 | 3.05 | 1.443 | 8.528 |
| 2D 1500² | 290.22 | 3.12 | 3.550 | 65.098 |
| 2D 3000² | 1735.69 | 3.60 | 5.262 | 209.681 |
| 3D 60³ | 40.25 | 4.16 | 1.853 | 8.513 |
| 3D 120³ | 362.77 | 4.62 | 3.069 | 56.426 |
| 3D 200³ | 3035.24 | 4.19 | 5.149 | 255.844 |

The CSR build grows with the problem (31 ms to 1736 ms in 2D, 40 ms to 3035 ms in 3D) while
the Kronecker build stays at 3-5 ms, so re-assembling every step is where an assembled
operator hurts most, and for a separable operator the Kronecker path removes that cost
outright. The CSR "build" here is a full `assemble` plus upload, not a refill: an
`assemble!` into an existing matrix replays recorded positions (below) and costs a small
fraction of a first assembly, so the re-assemble column is an upper bound on the CSR side.

**The scatter-position table.**
[gpena/Bramble.jl#318](https://github.com/gpena/Bramble.jl/issues/318) asked for scatter
positions to be recorded once and indexed on every refill instead of searched. That is now
true on every backend, as a host-side record/replay rather than a device table:
`CpuSerial`/`CpuThreaded` since gpena/Bramble.jl#338, `CpuPolyester` since commit
`d47d3376`, and Metal since commit `208b23de`, where the first `assemble` searches the
mirror's CSR layout once and every later `assemble!` (and `assemble_add!`) replays the
recorded positions into the mirror before the single bulk flush -- no device scatter
kernel. `benchmark/scatter_table.jl` (S5.1, commit `fb839b94`; AC power, load 3.01,
`--threads=4`, every row agreeing with a fresh assemble) measured the refill at the minimum of
40 `assemble!` calls; at 1D `n = 2049` CpuSerial took 0.00621 ms, CpuThreaded 0.05117 ms,
CpuPolyester 0.01196 ms and Metal 1.29733 ms, and on a non-uniform 2D `129 × 97` grid
0.07325, 0.14017, 0.06304 and 2.53271 ms respectively, against first assemblies of 0.0859 ms
(1D, CpuSerial) and 6.298 ms (2D, CpuSerial). The recorded positions cost 16,984 bytes
against a 114,880-byte matrix at 1D `n = 2049` (ratio 0.148 for CpuSerial; 0.086 for the
threaded policies, whose matrix measures 196,792 bytes) and 996,280 bytes against 1,094,080 on the 2D grid
(ratio 0.91 for CpuSerial, 0.42 threaded). The Metal rows in that run predate commit
`208b23de` and searched on every refill; their 64-byte cache figure is the absence of a
table, not its cost. Commit `208b23de`'s own review measured the Metal replay on a
four-term 2D `257²` form at 2.94-3.0 ms against 5.0-6.7 ms for the search it replaced.

This is a host-side refill speedup, largely orthogonal to the device-assembly question: it
makes the entries path cheaper to re-run, which is the path this decision keeps. Two parts of
#318 were not built. The negative sentinel for constrained entries has nothing to remove:
Dirichlet conditions already run as a separate pass after the scatter
(`apply_dirichlet_labels!`), so no scatter carries a boundary branch. And the measurement of
atomic accumulation against a colour-partitioned scatter asked for in #318's comment
compares two *device* scatter kernels, neither of which exists or is being built.

### Shared Metal storage is abandoned

[gpena/Bramble.jl#316](https://github.com/gpena/Bramble.jl/issues/316) proposed allocating
`MetalSparseMatrixCSR`'s buffers in `Metal.SharedStorage` so the host scatter writes device
values in place and the mirror disappears. It is abandoned. The step it removes, the one bulk
upload at the end of a fill, is about 33 µs against a 2,040 µs gap between Metal and serial
`assemble!` at 1D `N = 200,001` -- roughly 1.6% of the measured gap, because every
scatter-add already runs on the CPU whatever the storage mode. Its memory argument (one copy
of the values instead of two) is weaker too: #317's own reframing, and this decision, leave
the assembled device matrix as a setup-time object for an entries consumer, and the
measurements above show it fitting with a wide margin. #316 stays open until its owner
decides whether to close it as not planned.

### If the direction needs revisiting

Two measured facts would change this decision: a workload whose assembled matrix stops
fitting device memory at a size beyond those above, or one that needs a large
device-resident matrix held continuously and rebuilt often (the scenario #316's own
abandonment names as the one where its memory argument returns). Either should be recorded
as a fresh issue with its own numbers, rather than by reopening #317. A broader matrix-free path
is also recorded on #317 without being scoped: `assemble` on a device-backed form returning
a matrix-free operator for any form the already-fused device kernels can express, not only
the separable ones, with host assembly plus upload as the fallback for the rest. The architecture that
would then be weighed is kept below, in "Appendix: device assembly, if this decision is ever
revisited".

## Matrix-free Kronecker operators on a device

[gpena/Bramble.jl#323](https://github.com/gpena/Bramble.jl/issues/323) is the first
concrete step toward the matrix-free-on-device direction the previous section selects for
separable applies: a [`KroneckerLinearOperator`](@ref) built from a device-backed form, and
[`fdm_solve`](@ref) on top of it, both apply on the device with no host round trip once
built. The split below is decided and built, not a target.

**`KroneckerLinearOperator`.** [`kronecker_operator`](@ref) always builds the 1D mass and
difference factors on the host mirror of the mesh (`_host_mirror_mesh`), the same way it
does for a host-backed form -- a device-backed axis space cannot be scalar-indexed to
assemble a 1D factor. Each factor is then moved to the backend's storage once, by
`_kron_to_storage`: a `Diagonal` mass factor becomes a `_KronDeviceDiagonal` holding one
device vector, and a `SparseMatrixCSC` difference factor becomes a `_KronDeviceSparse`
holding its `colptr`/`rowval`/`nzval` as three separate device arrays -- never the matrix
struct itself, for the same non-bitstype-kernel-argument reason as "A struct nesting a
device array will not compile" above. `mul!` no longer sweeps the grid once per axis per
term ([gpena/Bramble.jl#323](https://github.com/gpena/Bramble.jl/issues/323) shipped that
version; a later change replaced it): every term `kronecker_operator` builds has at most
one non-diagonal factor, so `mul!` now applies the whole operator in a single fused pass,
writing each entry of `y` once. On the host that pass walks grid lines along axis 1,
folding the other axes' diagonal entries into one scalar per line, and a term whose
non-diagonal factor sits on axis 1 runs as one `@simd` sweep along the line
(`_KronTridiag`, since the `inner₊(D₋ₓ(u), D₋ₓ(v))` factor `kronecker_operator` builds is
always tridiagonal). On the device the whole operator runs as one `KernelAbstractions`
kernel, one work item per entry of `y` and no write conflicts or atomics
(`_launch_kron_fused!`, which replaced the earlier per-axis-mode kernel), using `Int32`
index arithmetic to recover each work item's grid index from `dims`/
`strides` -- 64-bit integer division is emulated on Apple GPUs. Neither path leaves the
device between terms, and `src/assembly/kronecker.jl` names no GPU package anywhere in this
machinery -- the stub `_launch_kron_fused!` in `src/` throws unless
`BrambleKernelAbstractionsExt` has supplied the real method, the same extension-contract
idiom as everywhere else on this page. `mul!` allocates nothing on the host either way, and
the `scratch` keyword some callers still pass is accepted and ignored: the fused pass needs
no work buffers.

**`fdm_solve`.** The 1D factors' generalised eigendecomposition (`_fdm_eigendecompose`)
stays on the host: it is a small, per-axis LAPACK call regardless of `K`'s own storage, so
a device-backed `K`'s factors are copied back to the host first (`_fdm_host_factor`) before
the eigensolve runs. Everything downstream of that eigensolve follows the storage of the
right-hand side `F` instead: `_fdm_apply` copies the eigenvector matrices `Q_d`, their
transposes and the combined eigenvalue grid `Λ` to `F`'s storage once per call
(`_fdm_to_storage`), then runs the sum-factorisation as dense device matmuls
(`_fdm_apply_mode`, `mul!` against a `(m, pre * post)` matricisation, with `permutedims` to
bring the contracted axis to the front when it is not already axis 1) and a device
broadcast for the division by `Λ`. This holds for both `dirichlet` branches:
`dirichlet = nothing` runs the sum-factorisation over the whole grid, and
`dirichlet = :boundary` first restricts `F` to the interior with a view and a broadcast
(not `getindex` on a range, which would scalar-index a device array) before the
eigensolve, then embeds the interior solution back into a zero-filled `K.n`-length result
with the same view-and-broadcast pattern afterward -- no scalar indexing appears on either
branch's device path.

**Measured throughput.** `benchmark/kronecker_device.jl` timed the host and Metal backends
back to back on uniform meshes, `Float32`, 4 threads, AC power, load 2.4:

| Case | Operation | Host | Metal | Ratio (host/Metal) |
|---|---|---|---|---|
| 2D 3000x3000 | `mul!` | 5.97 ms | 4.28 ms | 1.39 |
| 2D 3000x3000 | 30 CG-shaped iterations | 523.9 ms | 304.5 ms | 1.72 |
| 2D 3000x3000 | `fdm_solve` | 5382.2 ms | 2088.6 ms | 2.58 |
| 3D 200x200x200 | `mul!` | 7.86 ms | 7.08 ms | 1.11 |
| 3D 200x200x200 | 30 CG-shaped iterations | 577.2 ms | 369.0 ms | 1.56 |
| 3D 200x200x200 | `fdm_solve` | 486.1 ms | 122.2 ms | 3.98 |

The `fdm_solve` host eigendecomposition alone (the LAPACK call above, unaffected by
backend) took 1821.3 ms in the 2D case and 4.99 ms in the 3D case, out of the totals above.
Before the fused pass replaced the per-axis sweep, the same benchmark measured `mul!` at
41.49 ms host / 41.12 ms Metal in 2D and 43.43 ms host / 77.39 ms Metal in 3D, and the
30-iteration CG loop at 1686.1 ms host / 1416.5 ms Metal in 2D and 1755.7 ms host / 2484.1
ms Metal in 3D. For comparison, #323's hand-rolled matrix-free CG loop -- calling `Δₕ!`
directly rather than going through `mul!(y, K, x, ...)` -- measured 1.55x (2D) and 1.16x
(3D) Serial/Metal, at an absolute Metal time of 867.1 ms (2D) and 1144.0 ms (3D). The CG
loop through `K` now beats that hand-rolled loop on both counts: a higher Serial/Metal
ratio (1.72x and 1.56x) and a lower absolute Metal time (304.5 ms and 369.0 ms).

**The plain conclusion.** The fused pass turns the device `mul!` from roughly break-even
(2D) or slower-on-device (3D) into a device win, and that carries through to an iterative
solve: 30 CG-shaped iterations through `K` now gain 1.72x in 2D and 1.56x in 3D, and
`fdm_solve` gains 2.6-4.0x. No cause for the `fdm_solve` gain is established beyond what
the eigendecomposition split above already shows. For `mul!` itself, the one cause
measured is the switch to `Int32` index arithmetic in the device kernel, which took its
time from 10.8/18.4 ms (2D/3D) to 4.3/7.1 ms; no other cause was measured, so none is
claimed.

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
  `_zero_stored!` for a `DeviceLocality` matrix now zeroes `A`'s own mirror alone
  (`fill!(A.mirror.nzval, zero(...))`) rather than `A.nzVal` itself, since
  `_flush_device_scatter!` already overwrites every stored entry from the mirror
  unconditionally. Any future device write needs the same discipline: order it against
  whatever else touches the same buffer, or remove the second writer, and do not trust a
  small, single-run `CHECK` to have exercised the timing at all.

## Masked projection

A masked [`Rₕ!`](@ref)/`avgₕ!` call (`markers` non-empty) on a device backend runs the same
projection kernel as the unmasked call, launched over an index list rather than the whole
grid ([gpena/Bramble.jl#297](https://github.com/gpena/Bramble.jl/issues/297)). The host
path folds the mask into its kernel, but that kernel closes over the mesh's `BitVector`s,
and a struct nesting an array is not a kernel argument (see above). So each call gathers
the marked linear indices on the host from the mesh's own masks (the union, for several
markers), uploads them one way as an `Int32` vector, and passes that vector to the kernel as
a top-level argument of its own: thread `j` fills grid index `sel[j]`. The destination is
zeroed on the device first, so off-region entries end up zero, as on the host. A marker
holding no point launches nothing.

Nothing is cached on the device. A persistent device copy of the markers would need
invalidating in `set_markers!`: the staleness hazard gpena/Bramble.jl#313 removed along
with the mirror cache. Launching over the list rather than over every point with a
predicate is also the cheaper choice for the usual case, a `:boundary` marker, which covers
a vanishing fraction of the grid.

## `GpuOffload`: per-call device routing without a device-typed space

[`GpuOffload`](@ref) ([gpena/Bramble.jl#324](https://github.com/gpena/Bramble.jl/issues/324))
is a different shape from everything above: a [`CpuPolicy`](@ref) -- `locality` answers
[`HostLocality`](@ref) for it, and the `Backend` it configures keeps host storage -- that
wraps an inner `CpuPolicy` alongside a device `Backend`. It routes only `Rₕ!`/`avgₕ!`'s fill
step through that device backend (`_offload_project!`,
`src/space/operators/projection.jl`); every other operation on the space runs under the
wrapped inner policy exactly as it would without `GpuOffload` in play, since
[`execution_policy`](@ref) on a `GpuOffload`-configured `Backend` returns the inner policy
instance, not the wrapper. There is deliberately no persistent device state: each call
uploads the mesh axis it needs and copies the result back, with nothing cached between
calls (see "Masked projection" above for why a persistent device copy of anything
mesh-derived is a staleness hazard this package has already removed once, #313).

**Measured, honestly: three wins and one loss, not a universal speedup.**
`benchmark/gpu_offload.jl` (commit `2c7217e6`) re-measured the issue's own exploratory table
back to back, `CpuThreaded()` against `GpuOffload(metal_backend(), CpuThreaded())`, on the
same four rows, correctness gated before timing (`rtol = atol = 1f-5`, `Float64` host vs
`Float32` device). Apple M2, AC power, load 3.09/4.0, `--threads=4`:

| Case | Host (`CpuThreaded`) | `GpuOffload` | Ratio |
|---|---|---|---|
| `Rₕ!` 1D, n=10,000,000 | faster | slower | **0.86x -- offload loses** |
| `Rₕ!` 2D, 3000x3000 | -- | -- | 1.79x |
| `avgₕ!` 1D, n=10,000,000 | -- | -- | 3.27x |
| `avgₕ!` 2D, 3000x3000 | -- | -- | 10.86x |

Three of the four rows win clearly, close to the issue's own exploratory figures
(1.98x/9.87x/20.52x for the same three rows) -- but `Rₕ!` 1D at ten million points *loses*,
0.86x, host wins. The reason is the same no-hidden-cache design named above: `Rₕ!` is one
cheap function evaluation per point, already fast on four CPU threads, and does not
amortise the per-call mesh-axis upload plus full result copy-back that `GpuOffload` pays on
every call with nothing cached. `avgₕ!`'s per-cell quadrature is expensive enough on the
host that the same round trip is dwarfed instead, which is why its two rows show the
largest wins in the table. Read this policy as a win for expensive per-point/per-cell work
on a large grid, not as something to reach for by default -- measure the specific call
before choosing it over the wrapped inner policy alone.

## A device kernel launch perturbs the global RNG stream

Any code that mixes `rand()`/a seeded RNG with a device kernel launch -- Metal today, a
future CUDA/ROCm/oneAPI backend later -- has to reckon with a hazard that has nothing to
do with Bramble's own logic: **launching a kernel consumes a draw from Julia's global RNG
as a side effect of the launch itself.** Seeding `Random.default_rng()` immediately before
code that both calls `rand()` and launches a device kernel does not leave the stream in
the state that code might expect afterward, even though nothing the caller wrote touched
`Random` at all. Confirmed empirically while investigating
[gpena/Bramble.jl#320](https://github.com/gpena/Bramble.jl/issues/320): seed, draw twice as
a control; seed again, launch a device kernel, draw once more -- the post-kernel draw does
not match the control's second draw.

**Source, pinned down by reading the installed package code (not by launching a real
kernel):** it is Metal.jl, not KernelAbstractions.jl. KernelAbstractions.jl 0.9.42's own
source has no `Random`/`rand` use anywhere in its launch path -- the only hit anywhere in
its `src/` is a `rand(1024)` inside a docstring example. Metal.jl does the drawing itself,
unconditionally, on every kernel dispatch:

```julia
# Metal.jl src/compiler/execution.jl, function `launch`
kernel_state = KernelState(Random.rand(UInt32), buf_ptr, exc_ptr)
```

`Random.rand(UInt32)` here has no explicit RNG argument, so it reads
`Random.default_rng()` -- Julia's global stream. The draw happens before the function's
first branch (the `kernel.loggingEnabled` / `precompiling` check a few lines below), so it
is not conditional on logging, precompilation, or anything about the kernel body: **every
launch costs exactly one `UInt32` draw, a fixed amount, not a variable one.** The result
seeds `KernelState.random_seed` (`src/device/runtime.jl`), which is what Metal.jl's
on-device `rand()` support (`src/device/random.jl`) uses to seed each thread's own
generator -- the global-stream draw is not accidental leakage, it is how Metal.jl gives
every kernel launch an unpredictable device-side seed by default. Checked across the three
Metal.jl releases installed on this host (`~/.julia/packages/Metal/`) -- v1.9.3, v1.10.0
(what this repository's `Manifest.toml` resolves to) and v1.11.1 -- and the same
`Random.rand(UInt32)` call sits in the same spot in `launch` in all three, so this is a
stable, versioned design choice, not a fluke of one release. Every kernel `Metal.jl`'s
`KernelAbstractions.jl` backend launches (`MetalKernels.jl`'s `(::KA.Kernel{MetalBackend})`
method) goes through this same `HostKernel` call and therefore this same `launch` function
-- there is no path through `MetalBackend` that avoids it.

Because the behaviour is consistent and clearly intentional (seeding a documented
on-device RNG feature) rather than an oversight, it is not obviously a defect
KernelAbstractions.jl or Metal.jl would want to change on request; this section only
identifies where the draw happens; it does not file anything upstream, and no one should
read this as Bramble's assessment of whether an issue is warranted -- that's a judgment
call for a human maintainer with the exact file/line above in hand
(`Metal.jl` `src/compiler/execution.jl`, function `launch`, the `KernelState(...)` line).

**What Bramble provides, and its limits.** `_seed_mesh1d_rng!`/`_unseed_mesh1d_rng!`
in `src/mesh/mesh1d.jl` exist so Bramble's own non-uniform mesh generation can opt out of
the global stream entirely and draw from an isolated package-local `Xoshiro` instead,
immune to whatever a device kernel launch does to `Random.default_rng()`. These are
internal functions, not public API: they are not something user code is meant to call
directly today. They were added for Bramble's own reproducibility needs (a host and device
build of the "same" non-uniform mesh producing the same interior coordinates for a given
seed) and for the test suite that checks it, not as a general-purpose answer to this
hazard.

Outside that one internal use, this is a general hazard for *any* code mixing `rand()`
with a device kernel launch, Bramble's own or not, and Bramble cannot fix it on the
user's behalf: the perturbation originates a layer below Bramble, in the backend package
that actually launches kernels. A seeded cross-backend comparison written against
Bramble's public API -- or against any other package that launches device kernels -- needs
its own isolated RNG (seeded independently of `Random.default_rng()`) for any `rand()` call
whose result must not depend on what kernel launches happened to run first.

## Appendix: device assembly, if this decision is ever revisited

**Nothing in this appendix is being acted on.** "The device architecture decision" above
selects host assembly plus upload and does not build device assembly. What follows is the
architecture recorded on [gpena/Bramble.jl#317](https://github.com/gpena/Bramble.jl/issues/317)
(the comment "Stated target architecture for the GPU path", 2026-09-21), kept because the
engineering knowledge in it would still apply if a fresh issue ever reopened the question.

Across Metal, CUDA and ROCm, device assembly would follow the architecture
[cuda-dolfinx](https://github.com/bpachev/cuda-dolfinx) uses for DOLFINx:

1. Mesh geometry, coefficients, constraint markers and the CSR structure would live
   entirely on the device once uploaded; the host would never read or write matrix values
   during assembly. The current mirror path deliberately keeps that structure
   host-readable instead.
2. Scatter positions would be precomputed into a table and indexed by a device scatter
   kernel. The host-side half of this exists already as the record/replay described under
   "The scatter-position table" above; a device version would lay the table out
   warp-interleaved, as cuda-dolfinx does, to keep its accesses coalesced.
3. Constrained entries would be carried in that table as a negative sentinel, so the
   scatter kernel has no boundary-condition branch. Bramble's Dirichlet pass already runs
   after the scatter, so this would only matter to a device scatter that folded the
   constraint in.
4. Accumulation would be atomic, or colour-partitioned -- which one is a measurement
   #318's comment asks for, since on a tensor-product grid with a fixed-width stencil the
   colouring is a small strided partition fixed at compile time, not a graph algorithm.

Four qualifications travel with the architecture, each one a defect if it were copied
without them:

- **The table's interleave stride is a backend property, not the literal `32`.**
  cuda-dolfinx hardcodes `warpSize` to keep the table's accesses coalesced. Apple
  SIMD-groups and NVIDIA warps are both 32 wide, but AMD wavefronts are 64 on CDNA and 32
  on RDNA -- hardcoding 32 would silently lose the coalescing on exactly the backend most
  likely to need it.
- **Metal's `Float32` atomic support has to be verified before any design leans on it.**
  The atomic scatter rests entirely on a `Float32` atomic add; CUDA and ROCm have had it
  for years, but what Metal.jl and Atomix expose through KernelAbstractions for Metal is
  narrower and arrived later. If it turns out to be missing or slow, the design does not
  degrade gracefully on its own -- the fallback is cuda-dolfinx's own rowwise variant,
  which inverts the map so each thread owns a row and gathers its contributions by
  recomputing them, avoiding atomics entirely.
- **The runtime source generation cuda-dolfinx relies on is a workaround for its own
  toolchain, and must not be copied.** cuda-dolfinx templates CUDA C into strings and
  compiles them with NVRTC because FFCx hands it C kernels to wrap. Bramble does not have
  that problem: one `@kernel` written against `KernelAbstractions.Backend` already covers
  every device ("Where the kernels live, and where they do not", above) -- the thing
  cuda-dolfinx's code generation works hardest for, this package already has for free.
- **The diagonal/off-diagonal block split waits for v3.6.0.** It exists in cuda-dolfinx
  only because PETSc matrices are MPI-distributed; Bramble has no MPI-distributed matrix
  yet, so there is nothing for that split to do until one lands.

Following this end to end would also mean compiling a form's AST, user coefficient
functions included, to device code -- the largest single piece of the work, and the reason
#317 asked for the measurement before building any of it.
