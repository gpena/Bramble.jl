"""
	solve(A, F, Solver = LUFactorization(); prec = Diagonal(A))

Solves the linear system `A * u = F` using the solver `Solver` and preconditioner `prec`.

# Arguments

  - `A`: The matrix, a `AbstractMatrix`.
  - `F`: The right hand side, a `AbstractVector`.
  - `Solver`: The solver to use, a `LinearSolve.Solver`. Defaults to `LUFactorization()`.
  - `prec`: The preconditioner to use, a `AbstractMatrix`. Defaults to `Diagonal(A)`.

# Output

  - `u`: The solution, a `VectorElement`.
"""
function Bramble.solve(A::AbstractMatrix, F::AbstractVector, Solver = LUFactorization(); prec = Diagonal(A))
	prob = LinearProblem(A, F)
	solution = LinearSolve.solve(prob, Solver, Pl = prec)

	return typeof(F)(solution.u)
end

"""
	solve!(u, A, F, Solver = LUFactorization(); prec = Diagonal(A))

Solves the linear system `A * u = F` using the solver `Solver` and preconditioner `prec` and stores the result in `u`.

# Arguments

  - `u`: The output, a `VectorElement`.
  - `A`: The matrix, a `AbstractMatrix`.
  - `F`: The right hand side, a `AbstractVector`.
  - `Solver`: The solver to use, a `LinearSolve.Solver`. Defaults to `LUFactorization()`.
  - `prec`: The preconditioner to use, a `AbstractMatrix`. Defaults to `Diagonal(A)`.
"""
function solve!(u::VectorElement, A::AbstractMatrix, F::AbstractVector, Solver = LUFactorization(); prec = Diagonal(A))
	solution = Bramble.solve(A, F, Solver, prec = prec)
	copyto!(u, solution)
end

#Bramble.solve!(u::VectorElement, A::AbstractMatrix, F::AbstractVector) = ( copyto!(u, Bramble.solve(A, F)) )
#=
function Bramble.solve(A::AbstractMatrix, F::AbstractVector, Solver = KrylovJL_GMRES(); prec = nothing, droptol::AbstractFloat = 0.0001)
	prob = LinearProblem(A, F)

	P = A

	if !isnothing(prec)
		P = prec
	end

	fact = ilu(P, τ = droptol)
	solution = LinearSolve.solve(prob, Solver, Pl = fact)
	return solution.u
end
=#
#=
function solve_gmres(A::AbstractMatrix, F::AbstractVector; prec = nothing, droptol::AbstractFloat = 0.0001)
	prob = LinearProblem(A, F)

	P = A

	if !isnothing(prec)
		P = prec
	end

	fact = ilu(P, τ = droptol)
	solution = LinearSolve.solve(prob, KrylovJL_GMRES(), Pl = fact)
	return solution.u
end

function solve_gmres!(u::VectorElement, A::AbstractMatrix, F::AbstractVector; prec = nothing, droptol::AbstractFloat = 0.0001)
	copyto!(u, solve_gmres(A, F, droptol = droptol, prec = prec))
end
=#
#=
function solve_gmres!(u::VectorElement, A::AbstractMatrix, F::AbstractVector; prec = nothing, droptol::AbstractFloat = 0.0001)
	copyto!(u, solve_gmres(A, F, droptol = droptol, prec = prec))
end
=#
#solve_direct(A::AbstractMatrix, F::AbstractVector) = A\F
#solve_direct!(u::VectorElement, A::AbstractMatrix, F::AbstractVector) = ( copyto!(u, solve_direct(A, F)) )
