# Passages moved out of `docs/src/internals/space.md`

Removing the division
speeds up the serial matrix-free product most, since there the weight read is a larger share
of each point's work than in assembly; the measured figures are in gpena/Bramble.jl#428.

weight by `CartesianIndex` (or a linear index converted to one) at every point -- and
repeat the `#310` device-factor guard wherever a factor is indexed directly. For scattered

