# #1126: Enzyme reverse abort in the 2D Dirichlet assembly

Reduction of gpena/Bramble.jl#1126 (Enzyme reverse of a 2D `assemble(a, l; dirichlet)` ends
the process with `needsCacheWholeAllocation`). The reproducer is
`1126_dirichlet_reverse.jl`, next to this note. It needs Bramble and nothing else from the
repository.

Reproduces: `julia --project=. --threads=1 --optimize=1 --check-bounds=auto docs/notes/internals/ad_reductions/1126_dirichlet_reverse.jl` from the repository root dies with `Assertion failed: (false && "caching potentially capturing/offset of allocation"), function needsCacheWholeAllocation, file .../Enzyme/GradientUtils.cpp, line 10570` and exit 134 (SIGABRT), in 10 s under `CpuSerial`. It differentiates `Bramble.apply_dirichlet_conditions!` on a vector, with the linear form built inside the differentiated function, on a 9 by 9 uniform mesh. The argument `assemble` runs the issue's `assemble(a, l; dirichlet = :boundary => x -> 0.0)` instead and aborts the same way in 23 s.

3D: not affected. `assemble(a, l; dirichlet)` on a 5 by 5 by 5 non-uniform mesh under `CpuSerial` differentiates in reverse in 164 s, and the gradient matches ForwardDiff to 1e-12 relative (`apply3d` and `assemble3d` of the reproducer both finish with a gradient). The 3D failure #1126 feared does not exist at this commit; what #1126 leaves open in 3D is only the threaded policies, which this reduction did not run.

Fix site: `src/mesh/interface.jl`, function `_geometric_face_bits` (line 172), with `_marks_only_face` (line 160) and `_each_marker_point` (line 146) sharing its idiom `boundary_indices(Ωₕ)[2axis - 2 + side]`. It is reached from `src/assembly/linear.jl` `apply_dirichlet_conditions!` through `dirichlet_bc!`, `_apply_conditions!` and `index_in_marker`, and Enzyme upstream owns the abort itself (the assertion in `needsCacheWholeAllocation` where it should raise an error).

Compile: 10 s and 0.82 GB peak footprint for the `apply` reproducer up to the abort, 23 s and 1.02 GB for `assemble`; the fixed 2D `assemble` (`fixed_assemble`) compiles and differentiates in 66 s and 1.6 GB, the 3D one in 164 s and 2.5 GB.

Enzyme: 0.13.218 (Julia 1.13.3, macOS arm64, 1 thread, `--optimize=1`, `--check-bounds=auto`).

## What aborts

The log's `origInst` is `%"new::Tuple146" = call ... @malloc(i64 128), !enzyme_fromstack`
and the `cur` instruction a `getelementptr` into it with a run-time index. 128 bytes is four
`CartesianIndices{2}` of 32 bytes each, the `NTuple{4, CartesianIndices{2}}` that
`boundary_indices(Ωₕ)` returns. The debug location of the `getelementptr` is
`_geometric_face_bits`, `src/mesh/interface.jl:174`:

```julia
face = boundary_indices(Ωₕ)[2axis - 2 + side]
```

`index_in_marker` calls `_geometric_face_bits` when `_stored_marker` answers `nothing`, a face
name such as `:xmin` that no stored marker holds. For `:boundary` the stored marker answers,
so the branch never runs, but Enzyme compiles and differentiates it. `boundary_indices`
builds the tuple of all `2D` faces on the stack and the run-time index picks one. The likely
reading is that Enzyme has to cache that face's loop bounds for the reverse pass, which is an
offset into the stack allocation, and `needsCacheWholeAllocation` asserts instead of handling
it. The evidence for the cause is the `origInst` and `cur` lines and that redefining the
function removes the abort, not a trace of the cache request.

## Reduction

Each row is one process under `CpuSerial`, a 9 by 9 non-uniform mesh unless stated,
`Enzyme.autodiff` of a scalar function of three parameters, set_runtime_activity reverse.

| Differentiated function | Outcome |
|---|---|
| `assemble(a)`, no Dirichlet | gradient |
| `dirichlet_bc!(A, W, :boundary)` after assembling `A` | not run alone |
| `Bramble._dirichlet_bc_indices!(v, Ωₕ, mask, f)` with a global mask | gradient |
| `index_in_marker(Ωₕ, :boundary)` inside, 2D and 1D | gradient |
| `index_in_marker` with the label read from the stored condition | gradient, 2D and 1D |
| `_geometric_face_bits(Ωₕ, :xmin)` alone | gradient, 2D and 1D |
| the same tuple-then-dynamic-index pattern on a bare `CartesianIndices`, no Bramble | gradient |
| `dirichlet_bc!(v, space, conds, :boundary)`, global conditions | gradient |
| `dirichlet_bc!(v, test_space(lf), conds, :boundary)`, `lf` built inside | gradient |
| `apply_dirichlet_conditions!(v, LF, conds, labels, nothing)`, `LF` a global form | gradient |
| the same with the form built inside, with or without a coefficient `κ` in it | abort |
| `assemble(l_κ; dirichlet)` with `l_κ = innerₕ(κ, v)`, 2D | abort |
| the same on a 1D mesh | gradient |
| `apply_dirichlet_conditions!` with the form built inside, uniform mesh | abort |
| `assemble(a, l; dirichlet)` with `_geometric_face_bits` and `_marks_only_face` redefined | gradient, 66 s |

So the abort needs a 2D mesh and a form built inside the differentiated function, which the
test's `build_assemble_dirichlet_2d` does in `f`. A non-uniform mesh is not needed, and
neither is the parameter inside the form. The same idiom exists in 1D and 3D, in
`boundary_indices` for `CartesianIndices{1}` and `{3}`, and both gave gradients; why, I did
not find, so the table records an observation and no rule about the tuple's size. The matrix
half of `assemble(a, l; dirichlet)` was not run on its own before the fix. With the fix the
whole call differentiates, so no second site was left behind it.

## Code change that would remove it

In `src/mesh/interface.jl`, build the one face from the index ranges and do not index a tuple
of all faces with a run-time value. The reproducer's `fixed` and `fixed_assemble` variants
redefine the two functions this way (the faces and their order are the ones
`boundary_indices` gives):

```julia
@inline function _face_cart(Ωₕ::AbstractMeshType{D}, axis, side) where {D}
    ci = indices(Ωₕ)
    return CartesianIndices(ntuple(
        d -> d == axis ? (side == 1 ? (1:1) : (size(ci, d):size(ci, d))) : (1:size(ci, d)),
        Val(D)))
end
```

`_geometric_face_bits`, `_marks_only_face` and `_each_marker_point` then call
`_face_cart(Ωₕ, axis, side)` where they call `boundary_indices(Ωₕ)[2axis - 2 + side]`. With the
first two redefined, `assemble(a, l; dirichlet)` in 2D under `CpuSerial` on the test cell's
non-uniform mesh gives `[-2394.7267043599404, 17603.624152858876, 8649.085318703092]`
against ForwardDiff's `[-2394.7267043599427, 17603.62415285889, 8649.085318703093]`.
`_each_marker_point` was not redefined: the semidiscretisation residual uses it, not this
call, so its use of the idiom is unchecked.

## For the fix subplan

- The cause is Bramble code that Enzyme cannot differentiate, so a Bramble change is in
  scope: the one above, in `src/mesh/interface.jl`. Test it first with the reproducer's
  default variant (it must stop aborting), then with the 2D cell of
  `test/space/ad_coverage.jl`, `assemble_dirichlet_2d`, whose `:aborts` expectation for
  `CpuSerial` reverse then changes to `:ok`.
- An upstream Enzyme report (gate item G3) is a second route: `needsCacheWholeAllocation`
  asserts on a stack tuple indexed with a run-time value, where it should raise an error. The
  Bramble-free tuple pattern did not abort alone, so the report would need the Bramble
  reproducer or a smaller one this reduction did not find.
- The threaded policies were not run. The issue's CpuThreaded reverse compile of 22 minutes
  is a separate measurement (S4), and may or may not share this site.
- 3D needs no work for `CpuSerial` reverse. Its cell can be added to the page as working,
  with the 164 s and 2.5 GB compile above.

## Commands run

All through `.claude/scripts/julia_slot.sh 1`, on a `git archive` copy of `1e770658` (so the
uncommitted edits to `src/assembly/` and `src/space/` in the tree could not move the result),
with the tree's `Manifest.toml` (Enzyme 0.13.218) and Enzyme from a temporary environment:

```text
julia --project=<copy> --threads=1 --optimize=1 --check-bounds=auto <script>
```

The reduction table came from a script with one `CASE` line per function (the abort ends the
process, so each aborting row is its own run). The reproducer was run from that copy once per
variant: `apply` 134 (10 s, 0.82 GB), `assemble` 134 (23 s, 1.02 GB), `apply1d`, `apply3d`,
`fixed` (gradient, 9 s), `fixed_assemble` (66 s, 1.6 GB) and `assemble3d` (164 s, 2.5 GB), all
exit 0. The ForwardDiff comparison used a temporary environment with ForwardDiff added.
