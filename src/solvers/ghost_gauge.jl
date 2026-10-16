# ghost_gauge.jl: `ghost_kernel(A, Wₕ)`, the sparse basis of the ghost-only kernel of a matrix
# assembled on a ghost space, which `pde_solve(A, F; gauge)` imposes as a constraint. Filled
# in by the ghost-dynamics plan (gpena/Bramble.jl#777).
