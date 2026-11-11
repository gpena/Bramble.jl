# #1171: Enzyme on the geometric inputs

Measurement for gpena/Bramble.jl#1171 (with #1170 and #1172): the seven cells S3.1 added to
`test/space/ad_coverage.jl` (groups `:inputs` and `:polygon`), run as the coverage test runs
them: one child process per policy and group of cells, Enzyme from a temporary
environment, `set_runtime_activity`, `function_annotation = Const`, the reference a central
difference or ForwardDiff on the same policy. There is no reproducer script: the cells are
the reproducer, run with `BRAMBLE_AD_COVERAGE_CHILD=<policy>:<group>[:<cell>[:<kind>[:<mode>]]]`,
one process after another through `.claude/scripts/julia_slot.sh 2`, `--optimize=1`.

Enzyme: 0.13.219, 2026-11-11, macbook (Julia 1.13.3, arm64), 2 threads. Figures were taken
under load (another session ran Julia in the main checkout, load average 3 to 5), so the
times are upper bounds.

Compile s is the wall time from the previous `CELL` line of the same process (the first
cell of a process: from its start, with about 70 s of package loading included; a second
mesh kind of the same cell adds under 1 s, written `<1`). Peak GB is the process's peak
memory footprint from `/usr/bin/time -l`, shared by the cells of that process. A `Cell`
line gives `<name> | <policy> | <kind> | <mode> | <status> | <relerr> | <compile s> | <peak GB>`.
Status `throws` carries the error's name.

Cell: dirac_location | CpuSerial | uniform | rev | ok | 1.1e-13 | 82 | 2.27
Cell: dirac_location | CpuSerial | nonuniform | rev | ok | 1.2e-13 | <1 | 2.27
Cell: dirac_location | CpuSerial | uniform | fwd | ok | 1.1e-13 | 120 | 2.31
Cell: dirac_location | CpuSerial | nonuniform | fwd | ok | 1.2e-13 | <1 | 2.31
Cell: dirac_strength | CpuSerial | uniform | rev | ok | 0 | 81 | 2.28
Cell: dirac_strength | CpuSerial | nonuniform | rev | ok | 0 | <1 | 2.28
Cell: dirac_strength | CpuSerial | uniform | fwd | ok | 0 | 110 | 2.37
Cell: dirac_strength | CpuSerial | nonuniform | fwd | ok | 0 | <1 | 2.37
Cell: polygon_vertices | CpuSerial | uniform | rev | throws IllegalTypeAnalysisException | NaN | 43 | 3.49
Cell: polygon_vertices | CpuSerial | uniform | fwd | throws IllegalTypeAnalysisException | NaN | 32 | 3.49
Cell: polygon_vertices | CpuSerial | nonuniform | rev | throws IllegalTypeAnalysisException | NaN | 10 | 3.49
Cell: polygon_vertices | CpuSerial | nonuniform | fwd | throws IllegalTypeAnalysisException | NaN | 6 | 3.49
Cell: polygon_mesh_inside | CpuSerial | uniform | rev | throws IllegalTypeAnalysisException | NaN | 392 | 6.24
Cell: polygon_mesh_inside | CpuSerial | nonuniform | rev | throws IllegalTypeAnalysisException | NaN | 75 | 6.24
Cell: polygon_mesh_inside | CpuSerial | uniform | fwd | throws IllegalTypeAnalysisException | NaN | 165 | 3.48
Cell: polygon_mesh_inside | CpuSerial | nonuniform | fwd | throws IllegalTypeAnalysisException | NaN | 193 | 3.45
Cell: levelset_parameter | CpuSerial | uniform | rev | silent_zero (first run only, see below) | 1.0 | 53 | 2.95
Cell: levelset_parameter | CpuSerial | uniform | fwd | silent_zero | 1.0 | 22 | 2.95
Cell: levelset_parameter | CpuSerial | nonuniform | rev | silent_zero | 1.0 | <1 | 2.95
Cell: levelset_parameter | CpuSerial | nonuniform | fwd | silent_zero | 1.0 | <1 | 2.95
Cell: polygon_datum | CpuSerial | uniform | rev | ok | 1.8e-16 | 300 | 4.60
Cell: polygon_datum | CpuSerial | uniform | fwd | throws MethodError | NaN | 86 | 4.60
Cell: polygon_datum | CpuSerial | nonuniform | rev | ok | 2.6e-16 | <1 | 4.60
Cell: polygon_datum | CpuSerial | nonuniform | fwd | throws MethodError | NaN | <1 | 4.60
Cell: polygon_coefficient | CpuSerial | uniform | rev | ok | 6.6e-16 | 248 | 4.27
Cell: polygon_coefficient | CpuSerial | uniform | fwd | throws MethodError | NaN | 88 | 4.27
Cell: polygon_coefficient | CpuSerial | nonuniform | rev | ok | 3.1e-16 | <1 | 4.27
Cell: polygon_coefficient | CpuSerial | nonuniform | fwd | throws MethodError | NaN | <1 | 4.27
Cell: polygon_datum | CpuPolyester | nonuniform | rev | ok | 2.6e-16 | 287 | 5.71
Cell: polygon_datum | CpuPolyester | nonuniform | fwd | throws MethodError | NaN | 116 | 5.71
Cell: polygon_coefficient | CpuPolyester | nonuniform | rev | ok | 1.5e-16 | 293 | 5.94
Cell: polygon_coefficient | CpuPolyester | nonuniform | fwd | throws MethodError | NaN | 115 | 5.94
Cell: polygon_datum | CpuThreaded | nonuniform | rev | ok | 2.6e-16 | 289 | 6.04
Cell: polygon_datum | CpuThreaded | nonuniform | fwd | throws MethodError | NaN | 147 | 6.04
Cell: polygon_coefficient | CpuThreaded | nonuniform | rev | ok | 1.5e-16 | 294 | 6.47
Cell: polygon_coefficient | CpuThreaded | nonuniform | fwd | throws MethodError | NaN | 148 | 6.47

No cell reached 10 minutes or 10 GB (the largest, `polygon_coefficient` under `CpuThreaded`,
took 442 s and 6.47 GB in one process), so no `_slow` group was needed. The threaded
policies were run on the non-uniform mesh only, for the two polygon `works` cells.

`levelset_parameter` is a silent zero in the one fresh process that ran it: Enzyme returned a
gradient of zeros in both modes (relative error 1.0 against the central difference), because
the `Float64` grid carries no derivative, while ForwardDiff refuses the `Dual` with an
`ArgumentError`. Four later fresh processes (the whole `inputs` group, then the cell alone,
three times) aborted in LLVM at the uniform reverse call with `Instruction does not dominate
all uses!` (`%loopLimit_cache`) and `signal 6`, with the cell and the file unchanged. The
verdict is not stable, so the cell is `:skip` in `ad_coverage.jl`, in a group of its own
(`:levelset`) so that an abort ends no other cell. The polygon
`works` cells differentiate in reverse under all three policies (relative error below
7e-16 against ForwardDiff on the same policy) and fail in forward under all three: Enzyme
has no `forward` rule for `deepcopy` (a `MethodError` on `FwdConfigWidth`). The vertices
and the mesh built inside `f` meet the same `IllegalTypeAnalysisException` in both modes,
as expected, and `dirac` differentiates in both modes where ForwardDiff has no method.

## Group runs

The groups the parent runs, each as one child process, `--optimize=1`, 2 threads, all exits
normal with every non-skip `CELL` line equal to its pin:

- `CpuSerial:inputs`: 611 s, 8.88 GB (close to the 10 GB cap), four cells.
- `CpuSerial:levelset`: 79 s, 2.29 GB, the two ForwardDiff lines only (Enzyme is `:skip`).
- `CpuSerial:polygon`: 581 s, 7.21 GB.
- `CpuThreaded:polygon`: 777 s, 9.80 GB (0.2 GB under the cap).
- `CpuPolyester:polygon`: 696 s, 8.99 GB.

The `polygon` group under `CpuThreaded` is the one to split if the cap kills it under load.
