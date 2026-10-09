# Performance: type stability, allocations, threads

## Measuring allocations

`@allocated` at top level over global variables reports bytes a function call does not
allocate (48 B against a true 0 B). Measure inside a warmed function:

```julia
function allocs(f!, args...)
    f!(args...)                                   # compile and warm
    return @allocated f!(args...)
end
@assert allocs(Rₕ!, uₕ, x -> sin(x[1])) == 0
```

A fixed allocation that does not grow with the number of unknowns usually means something is
rebuilt on every call (a form, a closure, a new matrix); one that grows like `8 * ndofs` is a
real array allocation.

Time code the same way: inside a function, over `const` globals or arguments, after a warm-up
call. Compare variants in one process, alternating them; separate Julia processes differ by
10 to 30 percent on the same code.

## In-place routines and reuse

- Every allocating operator has an in-place form ending in `!` (`Rₕ!`, `avgₕ!`,
  `assemble!`); it writes into and returns its first argument. Use the in-place form in loops.
- Build a form once. `A = Bramble.allocate_system_matrix(a)` fixes the sparsity pattern;
  `assemble!(A, a; dirichlet = bcs)` then refills it with no allocation. Changing data the
  form captures (a grid function's values, a `Ref`) is seen by the next `assemble!`.
- `assemble_add!(A, a, α)` accumulates `α` times a form onto an existing matrix, for
  operators such as `M/Δt + K` in a time step.
- Grid-function components are views: `uₓ, uᵧ = components(uₕ)` and `uₕ(1)` share memory with
  `uₕ`, so `uₓ .= 0` writes into `uₕ`. `parent(uₕ)` is the flat coefficient vector.

## Type stability in user code

- **Dispatch, don't branch on runtime values that pick a type.** A function returning `T` in
  one branch and `NTuple{D,T}` in another is unstable; dispatch on `Val(D)` instead.
- **Integer coefficients in forms must be literals**: a runtime `Int` makes the form's type
  depend on the value. Use `float(n)` or a `Ref`.
- **A direction known only at run time**: index `∇ₕ[d]`, or pass it as an argument
  (`Bramble.D₋(uₕ, d)`); never build `Val(d)` from a runtime `d` in a hot loop.
- **Closures over loop variables** capture boxed variables and run slowly: hoist the closure
  out of the loop, or bind copies with `let`.
- **Element types come from the data.** Allocate with `element(Wₕ, eltype(x))` or
  `similar(uₕ)`, not `zeros(Float64, n)`, so dual numbers and `Float32` pass through.

Check a call with `@code_warntype` or `@inferred`; Julia 1.13 accepts type-annotated
arguments there (`@code_warntype f(::Vector{Float64})`).

## Threads

- Start Julia with an explicit thread count (`--threads=4`). `--threads=auto` on a CPU with
  efficiency cores can be slower than fewer threads, because static partitions wait for the
  slowest core.
- `backend(; policy = Parallel())` threads assembly and the operators with `Threads.@threads`;
  `Bramble.CpuPolyester()` uses Polyester (`using Polyester`) for lower overhead on small loops.
  `Bramble.profile_backends()` shows which policy wins at which size on this machine.
- Never index per-thread buffers by `Threads.threadid()`: tasks migrate. Size them with
  `Threads.maxthreadid()` or give each chunk of work its own buffer.

## Solvers

`A \ F` is fine for moderate sizes. For repeated solves, factorise once and reuse:
`fact = Bramble.sparse_factorize(A)`, then `Bramble.refactor!(fact, A)` after refilling `A`
with the same pattern. `amg_preconditioner(A)` and `ilu_preconditioner(A)` (each needs its
package) feed iterative solvers. For a Laplacian-like form on a tensor mesh, factorise with
`fact = fdm_factorize(a)` and solve each right-hand side with `fdm_solve!(x, fact, F)`: no
matrix is assembled and a solve allocates nothing (`using Kronecker`;
`reference/forms-assembly.md`).
