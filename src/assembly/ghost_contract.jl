# `evaluate!`, `l(v)` and `a(u, v)` on ghost-space forms: filled by subplan X5 (#781).
# Until then `l(v)` and `evaluate!` refuse a ghost-space linear form.

(::GhostLinearForm)(v) = _throw_ghost_unsupported("contracting a linear form")
function evaluate!(::AbstractVector, ::GhostLinearForm, v)
    _throw_ghost_unsupported("evaluate!")
end
