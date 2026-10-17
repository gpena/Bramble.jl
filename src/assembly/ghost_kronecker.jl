# Kronecker operator of a ghost-space form: filled by subplan X2 (#780).
# Until then `kronecker_operator` refuses a ghost-space form.

kronecker_operator(::GhostBilinearForm) = _throw_ghost_unsupported("kronecker_operator")
