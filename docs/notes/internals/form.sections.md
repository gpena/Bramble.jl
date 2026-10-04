# GPU passages moved out of `docs/src/internals/form.md`

`BramblePolyesterExt`'s own replay batch functions. Host and device matrices and leaves both
replay ([gpena/Bramble.jl#318](https://github.com/gpena/Bramble.jl/issues/318)): a device
matrix replays positions recorded against its host `mirror`, a device leaf replays over
`host_weights`, and a `GpuPolicy` leaf's effective policy is `CpuThreaded`. A leaf whose

