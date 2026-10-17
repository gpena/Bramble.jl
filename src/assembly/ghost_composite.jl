# Composite spaces with ghost components: filled by subplan X4 (#781).
# Until then a composite space with a ghost leaf is refused.

# A composite space with a ghost leaf would need a closure per block.
_leaf_spaces_offsets(::GhostGridSpace, ::Int) = _throw_ghost_composite()
first_space(::GhostGridSpace) = _throw_ghost_composite()
