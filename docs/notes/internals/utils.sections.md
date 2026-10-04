# Passages moved out of `docs/src/internals/utils.md`

Private since gpena/Bramble.jl#339, but still an extension contract rather than an internal:
`BrambleKernelAbstractionsExt` implements them by their qualified `Bramble.` names with
`KernelAbstractions.@kernel`s. `src/utils/linear_algebra.jl` carries error-only stubs that
throw when no device extension is loaded. The "Linear algebra" block above filters both out so
they are listed here.

