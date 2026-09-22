# precompile/solver_sessions.jl: the direct-solve entry points `pde_solve` reaches without
# any solver package loaded -- `A \ F` (`:default`), `:spqr` (`SparseArrays.SPQR`, no
# extension needed) -- and the two SPQR functions `:spqr` wraps
# (`suitesparse_qr_factorize`/`suitesparse_qr_solve`, src/solvers/suitesparse_solver.jl) --
# so a fresh process's very first `pde_solve(A, F)` after `assemble` compiles nothing.
#
# `suitesparse_factorize`/`suitesparse_solve`/`sparse_factorize`/`refactor!` are out of
# scope: they dispatch into `BrambleSuiteSparseExt`, only loaded once the caller does `using
# SuiteSparse`, so the package's own workload cannot reach them (gpena/Bramble.jl#283).
#
# One tiny 1D mesh: `pde_solve`/`suitesparse_qr_*` only dispatch on
# `A::SparseMatrixCSC{Float64,Int}`/`F::Vector{Float64}`, not on mesh dimension, so a 2D
# shape would compile the same methods again for no new coverage.
function _pc_solver_session()
    Ω = domain(interval(0.0, 1.0), :boundary => boundary_symbols(interval(0.0, 1.0)))
    Ωₕ = mesh(Ω, 5, false)
    Wₕ = gridspace(Ωₕ)
    a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
    fₕ = Rₕ(Wₕ, x -> 1.0)
    l = form(Wₕ, v -> innerₕ(fₕ, v))

    A = assemble(a; dirichlet = :boundary)
    F = assemble(l; dirichlet = :boundary => x -> 0.0)

    suitesparse_qr_factorize(A)
    suitesparse_qr_solve(A, F)
    qr(A) \ F  # the `\` on a QRSparse directly, same call `suitesparse_qr_solve` makes internally
    pde_solve(A, F)
    pde_solve(A, F; solver = :spqr)
    return nothing
end
