# Measurements moved out of docstrings

Docstrings render in the API pages, so their machine-dependent figures moved here too.

## assemble docstring (`src/assembly/bilinear.jl`)

```
**Call this once, then assemble into what it returns.** Building the sparsity pattern is the
larger part of the work (at 250,000 degrees of freedom it is 9,700 us and 52 MB against 1,500 us
and zero allocations to refill the matrix), and the pattern does not change between assemblies.
```

## pde_solve docstring (`src/solvers/pde_solve.jl`)

```
on the symmetric factorisations it reaches (SPD/Cholesky, LDLᵀ): measured on this host against
  forms assembled on real 2D grid spaces, `:default` was a 1.2-1.3x win on a symmetric
  Poisson-plus-mass system and a 2.3-3.6x **loss** on an unsymmetric convection-diffusion one
  (gpena/Bramble.jl#246).
```

## SeparableWeights docstring (`src/space/scalar_gridspace.jl`)

```
(gpena/Bramble.jl#115, #234;
under 1 MB of weights on a 100³ mesh):
```

## CpuThreaded docstring (`src/utils/backend.jl`)

```
Where "small" ends was measured per workload rather than assumed (gpena/Bramble.jl#299,
`benchmark/polyester_crossover.jl`, commit 4b76d62b, on the Apple M2 host this milestone's
other measurements were taken on, `--threads=4`, AC power): the smallest grid size at which
this policy beats `CpuSerial` twice running is 64-96 points per axis for unmasked `Rₕ!`, 256
for masked `Rₕ!` (an O(perimeter) write against the mesh's `:boundary` marker, not O(n^D)),
and 24-32 for `avgₕ!` at `nq = 3`. Below those sizes `CpuSerial` is faster; the crossover
differs by an order of magnitude between workloads, so a number from one does not transfer to
another.

`innerₕ`/`normₕ` (`_dot`/`_dot_masked`, `src/utils/linear_algebra.jl`) now thread under this
policy too (gpena/Bramble.jl#301, commits ad3ab31a and 237ca306): the crossover against
`CpuSerial` was measured the same way (gpena/Bramble.jl#301, `benchmark/polyester_crossover.jl`,
commit d6a1fc10, same Apple M2 host, `--threads=4`, AC power) at 100,000-300,000 elements across
four runs -- the "twice running" confirmation rule held in each run, but the exact crossing
point moved within that range from one run to the next (300k, 300k, 100k, 100k), the same
run-to-run jitter the other workloads above show near their own crossing point.

`D₋ₓ!` (in-place stencil) and the warmed broadcast axpy `vₕ .= a .* uₕ .+ wₕ` into a
`VectorElement` now thread under this policy too (gpena/Bramble.jl#356, #357, commits
af19afea, 96e93c02, f7a8798f, 90f262e2): the crossover against `CpuSerial` was measured the
same way (`benchmark/policy_crossover.jl`, commit 90f262e2, same Apple M2 host, `--threads=4`,
AC power) across two runs. `D₋ₓ!`'s crossover held in both runs at 300,000 DOFs in 1D, 300,304
in 2D, and 300,763 in 3D. Broadcast axpy's crossover held in both runs at 300,304 DOFs in 2D
and 300,763 in 3D, but in 1D it moved between the two runs (100,000 and 300,000 DOFs), the same
run-to-run jitter `innerₕ` shows above. Below these sizes `CpuSerial` is faster for both
workloads.
```

## CpuPolyester docstring (`src/utils/backend.jl`)

```
The crossover against `CpuSerial` was measured per workload, closing the last open acceptance
criterion of gpena/Bramble.jl#190 (gpena/Bramble.jl#299, `benchmark/polyester_crossover.jl`,
commit 4b76d62b, same Apple M2 host, `--threads=4` and AC power as [`CpuThreaded`](@ref)'s
figures): the smallest grid size at which this policy beats `CpuSerial` twice running is 8-24
points per axis for unmasked `Rₕ!`, 16 for masked `Rₕ!`, 8 for `avgₕ!` at `nq = 3`, and 1,000
elements for `innerₕ`/`_dot`. Every one of these crossovers falls one to two orders of
magnitude below [`CpuThreaded`](@ref)'s own crossover for the same workload (gpena/Bramble.jl#301
measured [`CpuThreaded`](@ref)'s `_dot` crossover at 100,000-300,000 elements), and this policy
beats [`CpuThreaded`](@ref) at every crossover measured.

`D₋ₓ!` and the warmed broadcast axpy `vₕ .= a .* uₕ .+ wₕ` also thread under this policy
(gpena/Bramble.jl#356, #357, commits af19afea, 96e93c02, f7a8798f, 90f262e2), measured the
same way (`benchmark/policy_crossover.jl`, commit 90f262e2, same Apple M2 host, `--threads=4`,
AC power) across two runs: `D₋ₓ!`'s crossover against `CpuSerial` held at 3,000 DOFs in 1D and
10,648 in 3D in both runs, and moved between 1,024 and 10,000 DOFs in 2D from one run to the
next. Broadcast axpy's crossover held at 10,000 DOFs in 1D, 300,304 in 2D, and 97,336 in 3D in
both runs. Against [`CpuThreaded`](@ref) directly, this policy beats it from 100 DOFs in 1D/2D
and 125 in 3D for both workloads in both runs, again one to two orders of magnitude below
[`CpuThreaded`](@ref)'s own crossover against `CpuSerial` for the same workload.
```

