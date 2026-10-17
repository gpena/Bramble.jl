# Matrix-free application of a ghost-space form: filled by subplan X1 (#780).
# Until then `matrix_free_operator` refuses a ghost-space form.

function matrix_free_operator(::GhostBilinearForm; kwargs...)
    _throw_ghost_unsupported("matrix_free_operator")
end
