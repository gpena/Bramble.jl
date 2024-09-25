module Bramble

if Sys.isapple()
	# Apple: Load Apple Accelerate
	try
		using AppleAccelerate
		@info "Compiled with Apple Accelerate support on macOS"
	catch e
		@warn "Not an Apple machine, falling back to default BLAS/LAPACK"
	end
end

if Sys.iswindows()
	try
		#using MKLSparse
		using MKL
		@info "Compiled with MKL support on Windows"
	catch e
		@warn "Not an Intel machine, falling back to default BLAS/LAPACK"
	end
end

using DocStringExtensions
using InteractiveUtils: @code_warntype, @code_llvm, @code_native
import Base: eltype, similar, length, copyto!, isapprox, isequal, IndexStyle, axes, materialize!
import Base: show, getindex, setindex!, IndexStyle, iterate, size, ndims, diff, firstindex, lastindex
#import Base: map, map!,
import Base: *, +, -, /
import Random: rand!

using FunctionWrappers
import FunctionWrappers: FunctionWrapper
using LazyArrays
using FastBroadcast: @..
using LinearAlgebra
using SparseArrays, FillArrays

import LinearSolve: LinearProblem, solve, KrylovJL_GMRES, LinearSolve, LUFactorization
import IncompleteLU: ilu

using Cubature
import Integrals: solve, IntegralFunction, IntegralProblem, QuadGKJL, CubatureJLh
using WriteVTK

abstract type BrambleType end

# Domain/Interval handling functions
export interval, cartesianproduct
export domain, ×, create_markers, markers, labels

# Mesh handling
export mesh, hₘₐₓ

# Space handling
export gridspace, element
export Rₕ, Rₕ!, avgₕ, avgₕ!

export innerₕ
export inner₊, inner₊ₓ, inner₊ᵧ, inner₊₂
export snorm₁ₕ, norm₁ₕ, norm₊, normₕ

export diff₋ₓ, diff₋ᵧ, diff₋₂, diff₋
export D₋ₓ, D₋ᵧ, D₋₂, ∇ₕ
export diffₓ, diffᵧ, diff₂, diff
export jumpₓ, jumpᵧ, jump₂, jump
export Mₕₓ, Mₕᵧ, Mₕ₂, Mₕ

export solve, solve!

# Forms exports
export BilinearForm, LinearForm, assemble, assemble!, Mass, Diff, update!
export dirichletbcs
export mass, stiffness, advection
#=
# Exporters
export ExporterVTK, addScalarDataset!, datasets, save2file, close
=#

include("geometry/sets.jl")
include("geometry/domains.jl")

include("meshes/common.jl")
include("meshes/mesh1d.jl")
include("meshes/meshnd.jl")
include("meshes/function.jl")


include("spaces/gridspace.jl")
include("spaces/vectorelements.jl")
include("spaces/matrixelements.jl")
include("spaces/difference_utils.jl")
include("spaces/backward_difference.jl")
include("spaces/backward_finite_difference.jl")
include("spaces/forward_difference.jl")
include("spaces/jump.jl")
include("spaces/average.jl")

include("spaces/inner_product.jl")
include("spaces/linearalg.jl")


include("forms/types.jl")
include("forms/utils.jl")
include("forms/dirichletbcs.jl")
include("forms/bilinearforms.jl")
include("forms/linearforms.jl")
include("forms/assembler.jl")
#=
include("problems/types.jl")
include("problems/laplacian.jl")

include("exporters/types.jl")
include("exporters/exporter_vtk.jl")
=#

#include("precompile.jl")
end
