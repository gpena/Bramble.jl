#===========================================================================#
# stencil_matrix vs. the Kronecker construction (gpena/Bramble.jl#185)
#
# `D₋ₓ`, `Dcₓ` and `Mₓ` now build their matrix in one pass over the grid
# (`Bramble.stencil_matrix`, `src/space/operators/stencil.jl`) instead of composing it out
# of Kronecker products of 1D shift matrices and then scaling by a dense weight vector
# (`Bramble.kronecker_operator_matrix`, kept in `src/space/operators/shift.jl` as the
# oracle `stencil_matrix` is checked against, `test/space/operators.jl`). This script times
# the two side by side and reports allocations and bytes, not just wall-clock: the
# Kronecker path's own weighting step (`Vector .* SparseMatrixCSC`) is a measured
# `SparseArrays` memory bug independent of Bramble, allocating backing storage sized for
# the *dense* result rather than the sparse pattern (~1.5 GiB instead of ~400 KB for
# `D₋ₓ` on a 100×100 mesh) -- the number this script exists to make visible at the sizes
# the milestone cares about.
#
# Usage:
#     julia --project=benchmark benchmark/operator_matrices.jl
#===========================================================================#

using BenchmarkTools
using Bramble
# Internal since gpena/Bramble.jl#185: the retained oracle, not exported.
import Bramble: kronecker_operator_matrix

const CASES = (
    ("1000 (1D)", mesh(domain(interval(0.0, 1.0)), 1000, true)),
    ("300² (2D)", mesh(domain(box((0.0, 0.0), (1.0, 1.0))), (300, 300), true)),
    ("60³ (3D)", mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (60, 60, 60), true))
)

const FAMILIES = (("D₋ₓ", D₋ₓ), ("Dcₓ", Dcₓ), ("Mₓ", Mₓ))

function _row(label, mesh_label, op, Ωₕ)
    old = @benchmark $kronecker_operator_matrix($Ωₕ, $op)
    new = @benchmark $op($Ωₕ)
    return (
        label = label,
        mesh = mesh_label,
        old_time = minimum(old.times),
        old_allocs = old.allocs,
        old_memory = old.memory,
        new_time = minimum(new.times),
        new_allocs = new.allocs,
        new_memory = new.memory
    )
end

function main()
    rows = [_row(fname, mlabel, op, Ωₕ) for (mlabel, Ωₕ) in CASES for (fname, op) in FAMILIES]

    header = rpad("operator", 8) * rpad("mesh", 12) * rpad("old time", 12) *
             rpad("new time", 12) * rpad("old allocs", 12) * rpad("new allocs", 12) *
             rpad("old bytes", 14) * "new bytes"
    println(header)
    println("-"^length(header))
    for r in rows
        println(
            rpad(r.label, 8), rpad(r.mesh, 12),
            rpad(BenchmarkTools.prettytime(r.old_time), 12),
            rpad(BenchmarkTools.prettytime(r.new_time), 12),
            rpad(string(r.old_allocs), 12), rpad(string(r.new_allocs), 12),
            rpad(BenchmarkTools.prettymemory(r.old_memory), 14),
            BenchmarkTools.prettymemory(r.new_memory)
        )
    end
    return rows
end

main()
