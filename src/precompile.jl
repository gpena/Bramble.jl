import PrecompileTools: @compile_workload, @setup_workload

# Load precompilation configuration
include("precompile_config.jl")

# Only run precompilation if not in dev mode
if !BRAMBLE_DEV_MODE
	include("utils/precompile.jl")
	include("geometry/precompile.jl")
	include("mesh/precompile.jl")
	include("space/precompile.jl")
	include("form/precompile.jl")
else
	if BRAMBLE_PRECOMPILE_VERBOSE
		@info "Skipping precompilation (BRAMBLE_DEV_MODE=true)"
	end
end
