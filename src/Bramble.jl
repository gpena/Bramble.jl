module Bramble

#using StyledStrings: styled, @styled_str

using DocStringExtensions
using Base: remove_linenums!

import Base: eltype, similar, length, copyto!, axes, materialize!
import Base: show, first, last, getindex, setindex!, iterate, size, ndims, firstindex, lastindex
import Base: *, +, -, /, ^, @propagate_inbounds, @_inline_meta

using Random: rand!

using SparseArrays: SparseMatrixCSC, AbstractSparseMatrix, spdiagm, rowvals, nnz, dropzeros!, nzrange, spzeros, nonzeros

using FunctionWrappers: FunctionWrapper
using MacroTools: @capture, isexpr

using StaticArrays: @SVector, SVector

using OrderedCollections: LittleDict, OrderedDict, freeze
using Base.Threads: @threads
using UnPack: @unpack
using MuladdMacro: @muladd

using LinearAlgebra: norm, Diagonal, mul!, I, dot, transpose, Transpose
import LinearAlgebra: ⋅

using FillArrays: Ones, Eye

using Cubature
using Integrals: solve, IntegralFunction, IntegralProblem, QuadGKJL, CubatureJLh, BatchIntegralFunction
#using WriteVTK

# domain/interval handling functions
export box, interval, ×, dim, topo_dim, tails
export domain, markers, labels

# Mesh handling
export mesh, hₘₐₓ, iterative_refinement!, change_points, npoints, points, set

# Space handling
export gridspace, element, space
export Rₕ, Rₕ!, avgₕ, avgₕ!
export ndofs

export innerₕ
export inner₊, inner₊ₓ, inner₊ᵧ, inner₊₂
export snorm₁ₕ, norm₁ₕ, norm₊, normₕ

export diff₋ₓ, diff₋ᵧ, diff₋₂, diff₋ₕ
export diff₊ₓ, diff₊ᵧ, diff₊₂, diff₊ₕ

export D₋ₓ, D₋ᵧ, D₋₂, ∇₋ₕ
export D₊ₓ, D₊ᵧ, D₊₂, ∇₊ₕ

export jump₋ₓ, jump₋ᵧ, jump₋₂, jump₋ₕ
export jump₊ₓ, jump₊ᵧ, jump₊₂, jump₊ₕ

export M₋ₓ, M₋ᵧ, M₋₂, M₋ₕ
export M₊ₓ, M₊ᵧ, M₊₂, M₊ₕ

export ⋅

export dirichlet_constraints
export form, assemble, assemble!

#=
# Exporters
export ExporterVTK, addScalarDataset!, datasets, save2file, close
=#

include("utils/macros.jl")
include("utils/backend.jl")
include("utils/linear_algebra.jl")

include("geometry/pretty_print.jl")
include("geometry/set.jl")
include("utils/bramble_function.jl")

include("geometry/marker.jl")
include("geometry/domain.jl")

include("mesh/common.jl")
include("mesh/marker.jl")
include("mesh/pretty_print.jl")
include("mesh/mesh1d.jl")
include("mesh/meshnd.jl")

include("space/buffer.jl")
include("space/gridspace.jl")
include("space/scalar_gridspace.jl")
include("space/vector_gridspace.jl")
include("space/vectorelement.jl")

include("space/matrixelement.jl")
include("space/operators/shift.jl")
include("space/operators/difference.jl")
include("space/operators/jump.jl")
include("space/operators/average.jl")
include("space/operators/linear_operators.jl")
include("space/inner_product.jl")

include("form/dirichlet_constraints.jl")
include("form/bilinear_form.jl")
include("form/linear_form.jl")

#=
include("exporter/types.jl")
include("exporter/exporter_vtk.jl")
=#

#include("precompile.jl")
end
