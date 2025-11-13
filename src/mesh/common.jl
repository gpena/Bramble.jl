"""
# common.jl

This file provides the common interface and utilities for all mesh types in Bramble.

**Organization:**

The previous monolithic `common.jl` file has been refactored into focused modules:

- `dimension_traits.jl`: Dimension types for compile-time dispatch
- `index_generation.jl`: CartesianIndices generation and manipulation
- `mesh_interface.jl`: AbstractMeshType interface definitions

This file now serves as the orchestrator, including the component modules.

See also: [`Mesh1D`](@ref), [`MeshnD`](@ref), [`Domain`](@ref)
"""

# Include refactored components
include("dimension_traits.jl")
include("index_generation.jl")
include("mesh_interface.jl")
include("mesh_common_methods.jl")