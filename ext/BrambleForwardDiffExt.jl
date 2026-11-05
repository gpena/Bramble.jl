module BrambleForwardDiffExt

using Bramble: Bramble
using ForwardDiff: ForwardDiff

# A `ForwardDiff.Dual` is linear in its parts (value and partials), so a Sparspak solve may
# split a `Dual` load into one solve per part (src/solvers/sparspak_solver.jl).
Bramble._linear_in_parts(::Type{<:ForwardDiff.Dual}) = true

end # module
