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
		using MKL
		@info "Compiled with MKL support on Windows"
	catch e
		@warn "Not an Intel machine, falling back to default BLAS/LAPACK"
	end
end

import Base: eltype, similar, length, copyto!, axes, materialize!
import Base: show, getindex, setindex!, iterate, size, ndims, firstindex, lastindex
import Base: map, map!, first, last
import Base: *, +, -, /, ^
import Random: rand!

using FunctionWrappers: FunctionWrapper

using FastBroadcast: @..
using LinearAlgebra: Diagonal, mul!, I
import LinearAlgebra: ⋅
using SparseArrays: spdiagm, SparseMatrixCSC, AbstractSparseMatrix
using FillArrays: Ones, Eye

using Cubature
using Integrals: solve, IntegralFunction, IntegralProblem, QuadGKJL, CubatureJLh
using WriteVTK

abstract type BrambleType end

# domain/interval handling functions
export interval, cartesianproduct, first, last, set
export domain, ×, create_markers, markers, labels, @embed

# Mesh handling
export mesh, hₘₐₓ, points, Iterator, iterative_refinement

# Space handling
export gridspace, element
export Rₕ, Rₕ!, avgₕ, avgₕ!

export innerₕ, innerₕ!
export inner₊, inner₊ₓ, inner₊ᵧ, inner₊₂
export snorm₁ₕ, norm₁ₕ, norm₊, normₕ

export diff₋ₓ, diff₋ᵧ, diff₋₂, diff₋ₕ
export D₋ₓ, D₋ᵧ, D₋₂, ∇₋ₕ
export diffₓ, diffᵧ, diff₂, diffₕ
export jumpₓ, jumpᵧ, jump₂, jumpₕ
export M₋ₕₓ, M₋ₕᵧ, M₋ₕ₂, M₋ₕ

export ⋅

# Forms exports
export AutoDetect, DefaultAssembly, InPlaceAssembly, OperatorsAssembly
export form
export assemble, assemble!
export constraints, symmetrize!

#=
# Exporters
export ExporterVTK, addScalarDataset!, datasets, save2file, close
=#

include("utils/bramblefunction.jl")
include("utils/linearalgebra.jl")

include("geometry/sets.jl")
include("geometry/domains.jl")

include("meshes/common.jl")
include("meshes/mesh1d.jl")
include("meshes/meshnd.jl")

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
include("spaces/operators.jl")


include("forms/constraints.jl")
include("forms/bilinearforms.jl")
include("forms/linearforms.jl")

#=
include("exporters/types.jl")
include("exporters/exporter_vtk.jl")

=#
#include("precompile.jl")
end
