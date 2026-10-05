```@meta
CurrentModule = Bramble
```

# [Backends and execution policies](@id tutorial_backend)

**What you will learn.** What a backend is, how to choose between serial and threaded execution, and how large a problem must be before threading pays off.

**What you need first.** The [mesh tutorial](@ref tutorial_mesh) and the [space tutorial](@ref tutorial_space), for meshes and grid spaces, and the [form tutorial](@ref tutorial_form), for assembling a system.

**Where next.** The [solvers tutorial](solvers.md) shows what to do with the linear system once it is assembled.

Every block below runs when this page is built, except the ones that need an optional package, which are marked as plain code.

---

## What a backend carries

Every mesh, grid space and form carries a **backend**: a compile-time choice of which vector and matrix types to allocate, and whether the operations that can thread do so. It is chosen once, when a mesh is built, and inherited by everything made from that mesh. Ask for the default one:

```@example backend
using Bramble
using Bramble: vector_type, matrix_type, backend_types, execution_policy

be = backend()
vector_type(be), matrix_type(be), execution_policy(be)
```

A [`Backend`](@ref) is only three type parameters: a vector type, a matrix type and an [`ExecutionPolicy`](@ref). It has no fields, so making one costs nothing. The default stores dense `Vector{Float64}` and sparse `SparseMatrixCSC{Float64,Int}`, and runs every operation as a plain loop under the [`Serial`](@ref) policy.

[`backend`](@ref) takes a single element type, or the two storage types as keywords:

```@example backend
f32 = backend(Float32)
dense = backend(vector_type = Vector{Float64}, matrix_type = Matrix{Float64})
backend_types(f32), matrix_type(dense)
```

[`mesh`](@ref) calls `backend(eltype(Ω))` when you give it no backend, so a mesh over `Float32` points already gets a `Float32` backend.

## Attach a backend to a mesh

Pass the backend when the mesh is built, and every object made from the mesh carries it:

```@example backend
Ω = domain(interval(0.0, 1.0))
Ωₕ = mesh(Ω, 21; backend = backend(Float64))
Wₕ = gridspace(Ωₕ)

backend(Ωₕ) === backend(Float64), backend(Wₕ) === backend(Ωₕ)
```

---

## [Choose a policy](@id backend_policies)

An **execution policy** says whether the operations that can thread, namely `Rₕ!`, `avgₕ!`, the quadrature weights of a grid space and form assembly, run in a plain loop or on several threads. The two names you will use are `Serial()` and `Parallel()`. The same code runs under both; only the backend differs. Build one mesh of each kind:

```@example backend
Ω_big = domain(interval(0.0, 1.0))
Wₛ = gridspace(mesh(Ω_big, 100_000; backend = backend(policy = Serial())))
Wₚ = gridspace(mesh(Ω_big, 100_000; backend = backend(policy = Parallel())))

execution_policy(Wₛ), execution_policy(Wₚ)
```

Now restrict the same function to both and compare:

```@example backend
uₛ = Rₕ(Wₛ, sin)
uₚ = Rₕ(Wₚ, sin)

parent(uₛ) == parent(uₚ)
```

The two policies give identical values. Only the schedule of the work differs, and you never change the call. Assembly reads the policy the same way:

```@example backend
l = form(Wₚ, v -> innerₕ(uₚ, v))
b = assemble(l)      # threads, because Wₚ's backend says Parallel()
nothing # hide
```

!!! tip "Try this"
    Build the same two spaces with 100 points instead of 100,000. The values still agree, but `Parallel()` now starts threads for work that is too small to repay them. The next sections say where threading starts to pay and how to find that size on your machine.

There is **no automatic size threshold**. A `Parallel()` backend threads every eligible call, however small and however many times a loop repeats it. A threshold would have to be tuned per operation, because `Rₕ!`'s crossover is not `avgₕ!`'s and neither is assembly's, and the caller could neither see nor override it. So the choice is yours:

- Use `Serial()`, the default, for small calls that repeat, such as an operator inside a time loop, where starting threads would cost more than the work.
- Use `Parallel()` once one call is expensive enough to repay the threads, such as a one-shot restriction or assembly over a large mesh.

Do not call `assemble_parallel!` to get parallel behaviour. It threads regardless of the backend, so it is only for a deliberate comparison. Build the backend you want and call the ordinary `assemble`, as the [form tutorial](@ref tutorial_form) does.

### Crossovers

A **crossover** is the smallest problem size at which a parallel policy beats `CpuSerial`
twice running. Below it, starting threads costs more than the work saved. The crossover
differs widely between operations (restriction, cell averaging, assembly and inner
products each have their own), so a figure for one does not transfer to another, and it
depends on the machine, the thread count and the power state. `CpuPolyester`, whose threads
start more cheaply, crosses over at smaller sizes than `CpuThreaded`.

### See where each policy wins here

[`Bramble.profile_backends`](@ref) gives a quick first answer for your machine and thread count. It times one sweep kernel, the loop that the weight and scatter builds go through, under each policy at sizes from `2^10` to `2^22`. It does not time assembly, so a policy that wins there is a good first choice, not a guarantee for your form. It lists `Serial()` and `Parallel()`, adds `CpuPolyester()` once `using Polyester` has loaded its extension, and adds a `Float32` `GpuKernel()` row after `using Metal` on a functional device. The sweep takes a fraction of a second on the CPU policies. With Metal loaded, the first call also compiles the GPU kernel and takes a few seconds. Call it by hand, once per session: neither `backend` nor `gridspace` calls it. The block is not run when this page is built, because the figures differ from machine to machine:

```julia
using Bramble

Bramble.profile_backends()
```

The result prints a table of times, the size from which each policy stays clearly faster than `Serial()`, and the `backend(policy = ...)` expression that selects it.

### Measure your own crossovers

To get these figures for your own machine, at the thread count you plan to use, run the crossover script from a checkout of the repository:

```bash
julia --threads=N --project=benchmark benchmark/policy_crossover.jl
```

It sweeps every workload above, and a few more, in 1D, 2D and 3D, and prints one `CROSSOVER | ...` line per workload and dimension. Each line gives the smallest size at which `CpuThreaded` and `CpuPolyester` start beating `CpuSerial`. Those lines count total degrees of freedom per dimension. Pass `--smoke` for a quick check with a few tiny sizes, `--max-dofs N` to raise the default cap of 1e6 degrees of freedom, or `--out file.md` to write the tables to a Markdown file.

### The policy hierarchy

`Serial` and `Parallel` are aliases: `Serial === CpuSerial` and `Parallel === CpuThreaded`. They sit in this hierarchy:

```
ExecutionPolicy
└── CpuPolicy
    ├── CpuSerial      (Serial)    -- one CPU thread
    ├── CpuThreaded    (Parallel)  -- Base.Threads.@threads
    └── CpuPolyester                -- Polyester.jl's @batch
```

Both spellings work everywhere. [`CpuPolyester`](@ref) is a threaded policy with a lower per-call overhead, and it crosses over at smaller sizes (see Crossovers above). It needs `using Polyester` before its first sweep, or the call errors and names `Polyester.jl`. The older name `CpuBatch` remains as a deprecated alias of `CpuPolyester`.

---

## Reference

### Custom storage

Any array type the package knows nothing about counts as host memory, so a custom wrapper array needs no registration to work as a `vector_type`. It must still be a `DenseVector`, be indexable on the host, and be constructible as `T(undef, n)` (or `T(n)`, see `supports_undef_construction`). Host storage admits [`CpuSerial`](@ref), [`CpuThreaded`](@ref) and [`CpuPolyester`](@ref). The [`Backend`](@ref) constructor rejects any other pairing, and a matrix type that disagrees with the vector type, before the backend exists.

### Introspection

[`vector_type`](@ref)`(be)` and [`matrix_type`](@ref)`(be)` give the two storage types, and [`backend_types`](@ref)`(be)` gives `(eltype, VT, MT, typeof(be))` at once. [`execution_policy`](@ref)`(x)` works on a backend, a mesh or a grid space, and answers what a threading-capable call on it would do.

### CSC or CSR storage

[`csr_backend`](@ref) swaps the default `SparseMatrixCSC{Float64,Int}` for the one-based `SparseMatrixCSR{1,T,Int}` of `SparseMatricesCSR.jl`. It needs that package loaded and throws without it.

```julia
using Bramble, SparseMatricesCSR

be = csr_backend()                 # Float64, Serial()
matrix_type(be)                    # SparseMatrixCSR{1, Float64, Int}
```

A finite-difference stencil is assembled row by row, which CSR storage reaches without the column scatter a CSC assembly needs ([#214](https://github.com/gpena/Bramble.jl/issues/214)). What follows from that:

- **Assembly costs about the same** in both layouts, and neither allocates in `assemble!`.
- **CSR stores the same matrix in less memory**, and the gap grows from 1D to 3D.
- **CSR loses the direct solve**, because `SparseMatricesCSR.jl` has no native CSR solve: `\` goes through a transposed factorization of a reinterpreted LU.

Reach for CSR when memory is the constraint and assembly dominates. Keep CSC when `A \ F` is on the critical path. `benchmark/backends.jl` measures both on your machine.

[`Bramble.profile_backends`](@ref) does not cover this choice: it times execution policies,
not matrix storage. Its docstring shows how to time `assemble!` under both storages on your
own form.

### A Polyester policy

```julia
using Bramble, Polyester

be = backend(policy = CpuPolyester())
Ωₕ = mesh(domain(interval(0.0, 1.0)), 100_000; backend = be)
execution_policy(gridspace(Ωₕ))   # CpuPolyester()
```

The policy type ships with `Bramble.jl`, but its sweeps live in the `BramblePolyesterExt` extension, which loads with `using Polyester`. After that you call `assemble`, `Rₕ!` and `avgₕ!` exactly as before.

### The precompilation workload

`Bramble.jl` precompiles a workload over 1D, 2D and 3D meshes. It makes the package build slower and the time to first result shorter. To skip it while developing the package itself:

```julia
using Preferences, Bramble
set_preferences!(Bramble, "precompile_workload" => false)
```

Julia tracks preferences in the precompilation cache, so the change applies on the next `using Bramble`. Restore the default with `delete_preferences!(Bramble, "precompile_workload"; force = true)`. The setting is written to `LocalPreferences.toml` next to your active project.
