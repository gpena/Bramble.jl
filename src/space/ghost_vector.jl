# ghost_vector.jl: `GhostVector`, the facet-only storage of a ghost space's values in a
# time loop: the physical values together with the ghost values beyond the named facets,
# `update_ghosts!` refreshing the latter from the former, and `physical`/`ghost_values`
# reading the two parts. Filled in by the ghost-dynamics plan (gpena/Bramble.jl#776).
