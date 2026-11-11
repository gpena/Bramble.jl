```@meta
CurrentModule = Bramble
```

# [Assemble on several threads](@id howto_backends)

**Goal.** Make `assemble` use several threads, and check that it returns what the serial
path returns.

**You need.** A mesh and a form ([forms tutorial](@ref tutorial_form)). Start Julia with
more than one thread, for example `julia --threads=4`, or the threaded policy runs on one.

## The backend carries the policy

A [`Backend`](@ref) fixes the vector type, the matrix type and an
[`ExecutionPolicy`](@ref). The mesh holds it, and every grid space and form made from the
mesh inherits it. The default policy is [`Serial`](@ref), so nothing threads until you ask.

```@example backends
using Bramble
using Bramble: execution_policy

Ω = domain(interval(0.0, 1.0))
execution_policy(mesh(Ω, 33)), execution_policy(gridspace(mesh(Ω, 33)))
```

## Ask for threads

Pass a backend built with [`Parallel`](@ref) when you create the mesh:

```@example backends
Ωₛ = mesh(Ω, 33; backend = backend(policy = Serial()))
Ωₚ = mesh(Ω, 33; backend = backend(policy = Parallel()))

execution_policy(gridspace(Ωₛ)), execution_policy(gridspace(Ωₚ))
```

The same works in any dimension. [`Serial`](@ref) and [`Parallel`](@ref) are the short names
of [`CpuSerial`](@ref) and [`CpuThreaded`](@ref), which use `Base.Threads.@threads`. The
call sites do not change: `assemble`, `assemble!`, `Rₕ!`, `avgₕ!`, `innerₕ` and the other
operations that can thread read the policy from the space.

## Check that threads change nothing

Assemble one form under both policies and compare. The form needs a two-dimensional mesh
large enough that every thread has rows to fill:

```@example backends
using SparseArrays

Ω² = domain(interval(0.0, 1.0) × interval(0.0, 1.0))

function assemble_pair(policy)
    Ωₕ = mesh(Ω², (65, 65); backend = backend(policy = policy))
    Wₕ = gridspace(Ωₕ)
    fₕ = Rₕ(Wₕ, x -> sin(π * x[1]) * cos(π * x[2]))
    a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + innerₕ(u, v))
    l = form(Wₕ, v -> innerₕ(fₕ, v))
    return assemble(a), assemble(l)
end

Aₛ, bₛ = assemble_pair(Serial())
Aₚ, bₚ = assemble_pair(Parallel())

@assert nnz(Aₛ) > 0 && sum(abs, bₛ) > 0    # control: both are nonzero
@assert Aₚ == Aₛ
@assert bₚ ≈ bₛ
Aₚ == Aₛ, bₚ ≈ bₛ
```

The matrix is identical entry by entry. Reductions such as `innerₕ` add their terms in
another order under a threaded policy, so compare vectors and scalars built from them with
`≈`.

## Use the Polyester policy

[`CpuPolyester`](@ref) is a third policy, built on `Polyester.jl`'s `@batch`. The type
ships with Bramble, but its sweeps live in an extension that loads with `using Polyester`.
Without it, the first threaded call throws an error that names the package.

```julia
using Bramble, Polyester

be = backend(policy = CpuPolyester())
Ωₕ = mesh(domain(interval(0.0, 1.0)), 100_000; backend = be)
execution_policy(gridspace(Ωₕ))   # CpuPolyester()
```

## Choose between the policies

There is no size threshold. A `Parallel()` backend threads every call that can thread, small
ones included, so starting threads can cost more than the work it saves.

- Keep `Serial()` for small calls that repeat, such as an operator inside a time loop.
- Use `Parallel()` or `CpuPolyester()` when one call is large, such as assembling over a
  fine mesh. `CpuPolyester()` starts threads more cheaply and pays off on smaller problems.

[`Bramble.profile_backends`](@ref) times a sweep kernel under each policy on your machine
and lists the size from which each one stays ahead. It does not time assembly, so treat
its answer as a first guess. Call it once per session, by hand.

!!! warning "Use `assemble`, not `assemble_parallel!`"
    [`assemble_parallel!`](@ref) threads whatever the backend says. It exists for a
    deliberate comparison. Build the backend you want and call `assemble`.

## See also

- [`backend`](@ref), [`Backend`](@ref) and [`execution_policy`](@ref) for the full
  signatures, including the storage types.
- [Utilities reference](@ref "Linear algebra backends") for the policy hierarchy.
