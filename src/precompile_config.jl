"""
# precompile_config.jl

Configuration for selective precompilation to reduce compilation time.

## Environment Variables

- `BRAMBLE_EXTENDED_PRECOMPILE`: Set to "true" to enable extended precompilation
- `BRAMBLE_DEV_MODE`: Set to "true" to skip all precompilation

## Precompilation Levels

**Essential** (always enabled, ~15-20s):
- 1D meshes and spaces
- Float64 only
- Most common operators
- Basic forms

**Extended** (opt-in, additional ~30s):
- 2D and 3D meshes
- Float32 type coverage
- Edge cases and rare operators
- Full type coverage

## Usage

### Production / Normal Use
```bash
# Default: Essential precompilation only
julia -e 'using Bramble'
```

### Full Precompilation (CI, releases)
```bash
export BRAMBLE_EXTENDED_PRECOMPILE=true
julia -e 'using Bramble'
```

### Development Mode (skip precompilation)
```bash
export BRAMBLE_DEV_MODE=true
julia -e 'using Bramble'  # Very fast, but first-use will be slower
```

## Implementation

Each module's precompile.jl should check these flags:
```julia
@setup_workload begin
	# Essential workload (always run)
	@compile_workload begin
		# 1D, Float64, common paths
	end
	
	# Extended workload (conditional)
	if BRAMBLE_EXTENDED_PRECOMPILE
		@compile_workload begin
			# 2D, 3D, Float32, edge cases
		end
	end
end
```
"""

# Check for development mode (skip all precompilation)
const BRAMBLE_DEV_MODE = get(ENV, "BRAMBLE_DEV_MODE", "false") == "true"

# Check for extended precompilation (slower but comprehensive)
const BRAMBLE_EXTENDED_PRECOMPILE = get(ENV, "BRAMBLE_EXTENDED_PRECOMPILE", "false") == "true"

# Print configuration on load (can be disabled)
const BRAMBLE_PRECOMPILE_VERBOSE = get(ENV, "BRAMBLE_PRECOMPILE_VERBOSE", "false") == "true"

if BRAMBLE_PRECOMPILE_VERBOSE
	@info "Bramble Precompilation Configuration" begin
		DEV_MODE = BRAMBLE_DEV_MODE
		EXTENDED = BRAMBLE_EXTENDED_PRECOMPILE
		MODE = BRAMBLE_DEV_MODE ? "Development (no precompile)" :
			   BRAMBLE_EXTENDED_PRECOMPILE ? "Extended" :
			   "Essential"
	end
end
