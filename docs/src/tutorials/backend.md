```@meta
CurrentModule = Bramble
```

# [Backend tutorial](@id tutorial_backend)

Every mesh, grid space and form in `Bramble.jl` carries a **backend**: a compile-time
configuration saying which vector and matrix types to allocate, and whether
threading-capable operations run serially or in parallel. Chosen once, when a mesh is
built, and inherited by everything constructed from it afterwards.

In this tutorial, you will learn how to:
1. Build a [`Backend`](@ref) with [`backend`](@ref), choosing its vector, matrix and
   element types.
2. Attach one to a [`mesh`](@ref), and see it propagate to grid spaces and forms.
3. Choose [`Serial`](@ref) or [`Parallel`](@ref), what it actually changes, and when it
   pays.
4. Call the ordinary entry points — `Rₕ!`, `avgₕ!`, `assemble!`, `assemble` — the same
   way regardless of which policy the backend carries.
5. Build a Metal GPU backend.

---

## 1. What a backend is

A [`Backend`](@ref) is `Backend{VT, MT, EP}`: a vector type, a matrix type, and an
[`ExecutionPolicy`](@ref), carried as type parameters on a struct with no fields at all.
Building one costs nothing at runtime — the types alone tell every downstream `vector`/
`matrix` allocation and every threading-capable loop what to construct and how to run.

```@example backend
using Bramble

be = backend()
```

```@example backend
vector_type(be), matrix_type(be), execution_policy(be)
```

The default is dense `Vector{Float64}`, sparse `SparseMatrixCSC{Float64,Int}`, and the
[`Serial`](@ref) policy — a plain loop, unconditionally, for every operation that could
otherwise thread.

## 2. Choosing element, vector and matrix types

[`backend`](@ref) takes `vector_type`/`matrix_type` keywords directly, or a single
element type as its first argument:

```@example backend
using SparseArrays

f32 = backend(Float32)                                    # Vector{Float32}, sparse Float32 matrix
sparse_state = backend(vector_type = SparseVector{Float64, Int},
    matrix_type = SparseMatrixCSC{Float64, Int})
vector_type(sparse_state)
```

`backend(T)` is what [`mesh`](@ref) uses when no backend is given at all — `mesh`
defaults to `backend(eltype(Ω))`, so a mesh built over `Float32` points already gets a
`Float32` backend without asking for one explicitly.

## 3. Attaching a backend to a mesh

```@example backend
Ω = domain(interval(0.0, 1.0))
Ωₕ = mesh(Ω, 21; backend = backend(Float64))
backend(Ωₕ) === backend(Float64)
```

Everything built from `Ωₕ` afterwards — [`gridspace`](@ref), its `VectorElement`s, and
any [`form`](@ref) over it — carries the same backend, so choosing one at mesh
construction is choosing it for the whole computation downstream:

```@example backend
Wₕ = gridspace(Ωₕ)
backend(Wₕ) === backend(Ωₕ)
```

## 4. `Serial()` or `Parallel()`, chosen once

`Rₕ!`, `avgₕ!`, a grid space's own quadrature-weight construction, and form assembly
(`assemble!`/`assemble`, for both a `LinearForm` and a `BilinearForm`) all read
[`execution_policy`](@ref) off the mesh or space they are given, and take the serial or
threaded branch accordingly — nothing else about how you call them changes:

```@example backend
Ωₕ_par = mesh(Ω, 100_000; backend = backend(policy = Parallel()))
Wₕ_par = gridspace(Ωₕ_par)
execution_policy(Wₕ_par)
```

There is **no automatic size threshold**. A `Parallel()` backend threads every eligible
call, however small, however many times a loop repeats it. That is deliberate: a
threshold would have to be tuned per operation — `Rₕ!`'s crossover is not `avgₕ!`'s, and
neither is assembly's — tuning the caller cannot see or override. Asking for `Parallel()`
and getting it, every time, is the point.

So the choice belongs to you, not a heuristic:
- **`Serial()`** (the default) for small, frequently repeated calls — a per-step
  operation inside a time loop, where the cost of spawning threads would dwarf the work
  itself.
- **`Parallel()`** once a single call is expensive enough on its own that spawning tasks
  pays for itself — a one-shot restriction or assembly over a large mesh.

The [benchmarks page](../benchmarks.md) carries the actual measurements across sizes —
that is what should decide, not a guess.

## 5. One interface, governed by the backend

Call `assemble!`/`assemble`, `Rₕ!`/`avgₕ!` the same way regardless of which policy the
backend carries — that is the entire reason to choose a policy once, on the backend,
rather than per call:

```@example backend
f = Rₕ(Wₕ_par, sin)
l = form(Wₕ_par, v -> innerₕ(f, v))
b = assemble(l)      # threads, because Wₕ_par's backend says Parallel()
nothing # hide
```

Do not reach for `assemble_parallel!` (or a hand-picked threaded variant) to get
parallel behaviour — that bypasses whatever the backend says and threads regardless,
which defeats the point of choosing a policy at all. It still exists, but only as a
deliberate, explicit override: a one-off forced comparison, or a benchmark that wants
the threaded path irrespective of the ambient backend. Build the backend you want and
call the ordinary entry point; see the [forms tutorial](form.md) for the full picture
of how assembly uses it.

## 6. A GPU backend (Metal)

```julia
using Bramble, Metal

gpu = metal_backend()                    # Float32, Serial()
gpu_par = metal_backend(Float16; policy = Parallel())
```

`Float64` is not supported on Apple Silicon GPUs — use `Float32` or `Float16`.
[`metal_backend`](@ref) requires `Metal.jl` loaded alongside `Bramble.jl`; without it,
it throws.

## 7. Introspection

- [`vector_type`](@ref)`(be)`, [`matrix_type`](@ref)`(be)` — the two type parameters.
- [`backend_types`](@ref)`(be)` — `(eltype, VT, MT, typeof(be))`, all four at once.
- [`execution_policy`](@ref)`(x)` — works on a `Backend`, a mesh, or a grid space, and
  always answers the same question: what would a threading-capable call on this do.

```@example backend
backend_types(be)
```
