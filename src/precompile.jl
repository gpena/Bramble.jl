import PrecompileTools: @compile_workload, @setup_workload

@info "Start precompilation..."
include("utils/precompile.jl")
include("geometry/precompile.jl")
include("mesh/precompile.jl")
#include("space/precompile.jl")
#include("form/precompile.jl")

@info "Precompilation finished."
