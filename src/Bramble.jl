module Bramble

using StyledStrings: styled, @styled_str

using Base: remove_linenums!

import Base: eltype, similar, length, copyto!, axes, materialize!
import Base: show, first, last, getindex, setindex!, iterate, size, ndims, firstindex, lastindex
#import Base: map, map!, first, last
#import Base: *, +, -, /, ^
using Random: rand!

using SparseArrays: SparseMatrixCSC#, AbstractSparseMatrix, spdiagm

using FunctionWrappers: FunctionWrapper
using Lazy: @forward

using OrderedCollections: LittleDict, OrderedDict, freeze
using OhMyThreads: tforeach, tmap!
#using Base.Threads: @threads
using UnPack: @unpack

using LinearAlgebra: norm#Diagonal, mul!, I
#=import LinearAlgebra: ⋅

using FillArrays: Ones, Eye

using Cubature
using Integrals: solve, IntegralFunction, IntegralProblem, QuadGKJL, CubatureJLh
using WriteVTK
=#
abstract type BrambleType end

# domain/interval handling functions
export box, interval, ×
export domain, markers, labels

# Mesh handling
export mesh, hₘₐₓ, iterative_refinement!, change_points!

# Space handling
export gridspace, element
#=
export Rₕ, Rₕ!, avgₕ, avgₕ!
export ndofs

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
=#
#=
# Exporters
export ExporterVTK, addScalarDataset!, datasets, save2file, close
=#

include("utils/style.jl")
include("utils/backend.jl")
#include("utils/linearalgebra.jl")

include("geometry/sets.jl")
include("utils/bramblefunction.jl")

include("geometry/markers.jl")
include("geometry/domains.jl")

include("meshes/common.jl")
include("meshes/markers.jl")
include("meshes/mesh1d.jl")
include("meshes/meshnd.jl")

include("spaces/buffer.jl")
include("spaces/types.jl")
include("spaces/gridspace.jl")
include("spaces/vectorelements.jl")

#=include("spaces/matrixelements.jl")
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
=#
#=
include("exporters/types.jl")
include("exporters/exporter_vtk.jl")

=#
#include("precompile.jl")
end
