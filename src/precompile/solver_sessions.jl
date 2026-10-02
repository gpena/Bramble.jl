# precompile/solver_sessions.jl: the direct-solve entry points `pde_solve` reaches without
# any solver package loaded -- `A \ F` (`:default`), `:spqr` (`SparseArrays.SPQR`, no
# extension needed) -- and the two SPQR functions `:spqr` wraps
# (`suitesparse_qr_factorize`/`suitesparse_qr_solve`, src/solvers/suitesparse_solver.jl) --
# so a fresh process's very first `pde_solve(A, F)` after `assemble` compiles nothing.
#
# `suitesparse_factorize`/`suitesparse_solve`/`sparse_factorize`/`refactor!` are out of
# scope: they dispatch into `BrambleSuiteSparseExt`, only loaded once the caller does `using
# SuiteSparse`, so the package's own workload cannot reach them.
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

    # The calls above run as static calls inside this function, so the compiler inlines
    # straight to each callee's body and never caches the entry signature itself as a
    # standalone method instance -- the positional wrapper `pde_solve(A, F)`, and the
    # `Core.kwcall` generated for `pde_solve(A, F; solver = ...)`. A REPL caller reaches
    # Bramble through ordinary dynamic dispatch, which resolves those entry signatures
    # first, so `precompile` here forces exactly that lookup to be cached too.
    precompile(pde_solve, (typeof(A), typeof(F)))
    precompile(
        Core.kwcall,
        (NamedTuple{(:solver,), Tuple{Symbol}}, typeof(pde_solve), typeof(A), typeof(F))
    )
    precompile(suitesparse_qr_factorize, (typeof(A),))
    precompile(suitesparse_qr_solve, (typeof(A), typeof(F)))

    # The matrix-free path, which nothing above reaches: the
    # operator's apply and the Jacobi and Chebyshev preconditioners in 1D, 2D and 3D on
    # non-uniform meshes, and a GMG V-cycle on the uniform 2D mesh it supports.
    for (X, n) in (
        (interval(0.0, 1.0), 6),
        (interval(0.0, 1.0) × interval(0.0, 1.0), (5, 4)),
        (box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), (4, 3, 3))
    )
        _pc_matrix_free_session(gridspace(mesh(domain(X), n, map(_ -> false, n))))
    end
    Ω₂ₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (9, 9), (true, true))
    P = gmg_preconditioner(_pc_mass_stiffness, Ω₂ₕ)
    P \ ones(npoints(Ω₂ₕ))
    return nothing
end

_pc_mass_stiffness(Wₕ) = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) + inner₊(∇ₕ(u), ∇ₕ(v)))

function _pc_matrix_free_session(Wₕ)
    a = _pc_mass_stiffness(Wₕ)
    x = ones(ndofs(Wₕ))
    y = similar(x)
    mul!(y, matrix_free_operator(a; dirichlet = :boundary), x)
    jacobi_preconditioner(a; dirichlet = :boundary) \ x
    chebyshev_preconditioner(a; degree = 2) \ x
    chebyshev_preconditioner(a; dirichlet = :boundary, degree = 2) \ x
    return y
end
