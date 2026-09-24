# ext/BrambleKernelAbstractionsExt.jl: the home for every device kernel in Bramble
# (gpena/Bramble.jl#174, S0.1 of .agents/plans/metal-and-apple-silicon-acceleration.md).
#
# Every `@kernel` the rest of the milestone writes lives here, and every one of them is
# written against `KernelAbstractions.Backend` alone -- never against `Metal.MtlVector` or
# any other concrete device array type. That is what makes them GPU-agnostic: a new GPU
# backend (CUDA, ROCm, oneAPI, ...) inherits every kernel in this file for free, simply by
# supplying one `ka_device` method (`src/utils/device_kernels.jl`) that names its own
# `KernelAbstractions.Backend`. `BrambleMetalExt` does exactly that for `Metal.MtlVector`.
#
# KernelAbstractions' own CPU backend is out of scope on purpose (#174 criterion (c) --
# see `.agents/plans/ka-cpu-measurement.md`): the CPU sweeps in
# `src/utils/linear_algebra.jl` stay `@simd`/`Threads.@threads`/`Polyester.@batch`, untouched
# by this file.
#
# S2.1 (gpena/Bramble.jl#94, #174): mesh coordinate kernels. Every kernel below fills a
# device-backed `Mesh1D` vector (`pts`, `half_pts`, `spacings`, `half_spacings`, or the
# refined-points buffer) with exactly the arithmetic the CPU loop in `src/mesh/mesh1d.jl`
# already performs -- with one deliberate exception: every `* 0.5` becomes `/ 2`, since a
# `Float64` literal would force `Float64` arithmetic inside the kernel, and Apple Silicon
# GPUs do not support double precision at all (`metal_backend`'s own docstring). Dividing by
# the `Int` literal `2` promotes to the array's own float type instead, matching what the
# CPU path computes to within rounding, never forcing `Float64`.
#
# No `@compile_workload` here, for the same reason `BrambleMetalExt` has none: a kernel
# needs a real device to precompile against.
module BrambleKernelAbstractionsExt

using Bramble: Bramble
using KernelAbstractions: KernelAbstractions, @kernel, @index, @Const, synchronize, get_backend

import Bramble:
                _launch_uniform_mesh1d_init!,
                _launch_half_points!,
                _launch_spacing!,
                _launch_half_spacing!,
                _launch_nonuniform_mesh1d_metrics!,
                _launch_refine_indices!,
                _gpu_for!,
                _gpu_scatter_for!,
                _launch_restriction!,
                _launch_restriction_scatter!,
                _launch_restriction_nd!,
                _launch_restriction_scatter_nd!,
                _launch_cell_average!,
                _launch_cell_average_scatter!,
                _launch_cell_average_nd!,
                _launch_cell_average_scatter_nd!,
                _launch_difference_onesided!,
                _launch_difference_centered!,
                _launch_average_engine!,
                _launch_spmv_csr!,
                _launch_spmm_csr!,
                _launch_kron_sparse_mode!,
                _launch_fused_divergence!,
                _launch_fused_curl2d!,
                _launch_fused_curl3d!,
                _launch_fused_laplacian!,
                _launch_fused_strain_offdiag!,
                ka_synchronize

# ---------------------------------------------------------------------------
# Fused uniform mesh init (gpena/Bramble.jl#303, S9 of
# .agents/plans/v3-4-0-device-quirks-and-kernels.md): one kernel over `1:(n + 1)` fills
# points, spacings, half points and half spacings together, replacing a uniform point fill
# plus the three sequential launches below (`_launch_spacing!`, `_launch_half_points!`,
# `_launch_half_spacing!`) for a uniform device mesh. Every entry is closed-form arithmetic
# on `a`, `h` and `n` alone -- unlike `_half_points_kernel!`/`_half_spacing_kernel!` below,
# which read `pts`/`h` back from global memory, this kernel reads nothing. `src/mesh/mesh1d.jl`
# dispatches only the uniform, non-collapsed, device-backed, `n >= 2` case here; a
# non-uniform or host-backed mesh keeps going through the three kernels below.
# ---------------------------------------------------------------------------

@kernel function _uniform_mesh1d_init_kernel!(pts, half_pts, spacings, half_spacings, a, h, n)
    i = @index(Global)
    @inbounds begin
        if i <= n
            pts[i] = a + (i - 1) * h
            spacings[i] = h
            half_spacings[i] = (i == 1 || i == n) ? h / 2 : h
        end
        if i == 1
            half_pts[i] = a
        elseif i == n + 1
            half_pts[i] = a + (n - 1) * h
        else
            half_pts[i] = a + (2i - 3) * h / 2
        end
    end
end

function _launch_uniform_mesh1d_init!(
        pts::AbstractVector, half_pts::AbstractVector, spacings::AbstractVector,
        half_spacings::AbstractVector, a, h, n::Int, dev
)
    _uniform_mesh1d_init_kernel!(dev)(pts, half_pts, spacings, half_spacings, a, h, n; ndrange = n + 1)
    return nothing
end

# ---------------------------------------------------------------------------
# `half_points!` (src/mesh/mesh1d.jl:347-352 is the CPU original)
# ---------------------------------------------------------------------------

@kernel function _half_points_kernel!(x, @Const(pts), n)
    i = @index(Global)
    @inbounds if i == 1
        x[i] = pts[1]
    elseif i == n + 1
        x[i] = pts[n]
    else
        x[i] = (pts[i] + pts[i - 1]) / 2
    end
end

function _launch_half_points!(x::AbstractVector, pts, n::Int, dev)
    _half_points_kernel!(dev)(x, pts, n; ndrange = n + 1)
    return nothing
end

# ---------------------------------------------------------------------------
# `spacing!` (src/mesh/mesh1d.jl:375-378 is the CPU original)
# ---------------------------------------------------------------------------

@kernel function _spacing_kernel!(x, @Const(pts), n)
    i = @index(Global)
    @inbounds if i == 1
        x[i] = pts[2] - pts[1]
    else
        x[i] = pts[i] - pts[i - 1]
    end
end

function _launch_spacing!(x::AbstractVector, pts, n::Int, dev)
    _spacing_kernel!(dev)(x, pts, n; ndrange = n)
    return nothing
end

# ---------------------------------------------------------------------------
# `half_spacing!` (src/mesh/mesh1d.jl:388-394 is the CPU original)
# ---------------------------------------------------------------------------

@kernel function _half_spacing_kernel!(x, @Const(h), n)
    i = @index(Global)
    @inbounds if i == 1
        x[i] = h[1] / 2
    elseif i == n
        x[i] = h[n] / 2
    else
        x[i] = (h[i] + h[i + 1]) / 2
    end
end

function _launch_half_spacing!(x::AbstractVector, h, n::Int, dev)
    _half_spacing_kernel!(dev)(x, h, n; ndrange = n)
    return nothing
end

# ---------------------------------------------------------------------------
# Fused non-uniform mesh metrics (gpena/Bramble.jl#305, S10 of
# .agents/plans/v3-4-0-device-quirks-and-kernels.md): one kernel over `1:(n + 1)` work
# items fills `spacings`, `half_pts` and `half_spacings` together from `pts` alone,
# replacing the three sequential launches above (`_launch_spacing!`, `_launch_half_points!`,
# `_launch_half_spacing!`) for a non-uniform device mesh. Each thread reads only its own
# 3-point local stencil of `pts` -- unlike `_half_spacing_kernel!` above, which reads
# `spacings` back from global memory, the interior half spacing here recomputes both
# adjacent spacings from `pts` directly, so `spacings` never has to be written to memory
# before this kernel can read it back. `src/mesh/mesh1d.jl` dispatches a non-uniform,
# non-collapsed, device-backed, `n >= 2` mesh here, at both construction and
# `set_points!`; the formula is correct for any coordinates, not only genuinely
# non-uniform ones, which is what lets `set_points!` (with no uniformity flag to branch
# on) use it unconditionally for a device destination.
# ---------------------------------------------------------------------------

@kernel function _nonuniform_mesh1d_metrics_kernel!(half_pts, spacings, half_spacings, @Const(pts), n)
    i = @index(Global)
    @inbounds begin
        if i <= n
            if i == 1
                h = pts[2] - pts[1]
                spacings[i] = h
                half_spacings[i] = h / 2
                half_pts[i] = pts[1]
            elseif i == n
                h = pts[i] - pts[i - 1]
                spacings[i] = h
                half_spacings[i] = h / 2
                half_pts[i] = (pts[i] + pts[i - 1]) / 2
            else
                # Sum the two already-rounded adjacent spacings and halve, the same
                # association `half_spacing!` (src/mesh/mesh1d.jl) uses --
                # `(spacing(i) + spacing(i + 1)) * 0.5` -- rather than the cheaper,
                # marginally more accurate two-point difference `(pts[i + 1] - pts[i - 1]) / 2`
                # this kernel used before. A fused kernel that changes results is not a
                # fusion: cross-backend bit-reproducibility is worth more here than the
                # half-ULP it costs.
                back = pts[i] - pts[i - 1]
                fwd = pts[i + 1] - pts[i]
                spacings[i] = back
                half_spacings[i] = (back + fwd) / 2
                half_pts[i] = (pts[i] + pts[i - 1]) / 2
            end
        end
        if i == n + 1
            half_pts[i] = pts[n]
        end
    end
end

function _launch_nonuniform_mesh1d_metrics!(
        half_pts::AbstractVector, spacings::AbstractVector, half_spacings::AbstractVector,
        pts::AbstractVector, n::Int, dev
)
    _nonuniform_mesh1d_metrics_kernel!(dev)(half_pts, spacings, half_spacings, pts, n; ndrange = n + 1)
    return nothing
end

# ---------------------------------------------------------------------------
# `_refine_indices!` (src/mesh/mesh1d.jl:486-489 is the CPU original)
# ---------------------------------------------------------------------------

@kernel function _refine_indices_kernel!(new_points, @Const(old_points), N_old)
    i = @index(Global)
    @inbounds begin
        new_points[2i - 1] = old_points[i]
        if i < N_old
            new_points[2i] = (old_points[i] + old_points[i + 1]) / 2
        end
    end
end

function _launch_refine_indices!(new_points::AbstractVector, old_points, N_old::Int, dev)
    _refine_indices_kernel!(dev)(new_points, old_points, N_old; ndrange = N_old)
    return nothing
end

# ---------------------------------------------------------------------------
# S2.3 (gpena/Bramble.jl#94, #174): the `GpuPolicy` device sweep seam
# (`src/utils/linear_algebra.jl`), and the two operators that evaluate a
# user-supplied function on the device -- `Rₕ!` (`src/space/operators/restriction.jl`) and
# `avgₕ!` (`src/space/operators/cell_average.jl`).
#
# Every kernel below takes its arrays -- the destination(s), the mesh's coordinate vector,
# a fixed-size `NTuple` of quadrature nodes/weights -- as separate, top-level kernel
# arguments, and the caller's function `f`/`g` as another top-level argument of its own.
# None of them are ever nested inside a wrapper struct passed as a single argument: checked
# against a real Metal device while designing this, a struct holding an `MtlVector` field
# fails to compile with `GPUCompiler.KernelError: passing non-bitstype argument` the moment
# it is a kernel argument, however small the wrapper -- only a *direct* array argument (or
# a `Tuple` of them, which `Adapt.jl` does convert element-wise) is adapted to the device's
# own array type before the kernel runs. This is why `_rule_kernel`'s `_RₕKernel`/
# `_AvgKernel` (which close over the whole mesh) are CPU-only, and the device paths below
# are written from scratch against the mesh's own arrays instead.
# ---------------------------------------------------------------------------

# The context every device-kernel launch below adds to whatever the compiler actually
# raised, so a closure that boxes a capture or allocates unexpectedly gets pointed at the
# reason rather than a bare wall of `GPUCompiler`/`KernelAbstractions` frames. Confirmed
# against a real Metal device: a closure boxing a `mutable struct` field fails to compile
# with a `GPUCompiler.InvalidIRError` reading "unsupported use of an undefined name" /
# "unsupported dynamic function invocation" for each offending call.
@noinline function _wrap_device_kernel_error(err, opname::String)
    return error(
        "$opname could not compile the supplied function to run on the device. A GPU " *
        "kernel evaluates it directly, so it must be free of heap allocation, dynamic " *
        "dispatch and boxed captures (a value from a Ref, a mutable struct field, or a " *
        "global variable) -- this is inherent to GPU compilation, not specific to " *
        "$opname. The underlying compiler error is:\n\n$(sprint(showerror, err))",
    )
end

# --- the generic `_sweep_for!`/`_sweep_scatter_for!` seam ------------------ #
#
# `_sweep_for!`/`_sweep_scatter_for!` (`src/utils/linear_algebra.jl`, gpena/Bramble.jl#298)
# check locality before reaching here: only a `DeviceLocality` destination paired with a
# `GpuPolicy` dispatches to `_gpu_for!`/`_gpu_scatter_for!` below, which keep their own names
# and signatures -- the rename only touched the seam above them.

@kernel function _generic_for_kernel!(v, @Const(idxs), f)
    i = @index(Global)
    idx = @inbounds idxs[i]
    @inbounds v[idx] = f(idx)
end

function _gpu_for!(policy, v::AbstractArray, idxs, f)
    dev = get_backend(v)
    try
        _generic_for_kernel!(dev)(v, idxs, f; ndrange = length(idxs))
    catch err
        _wrap_device_kernel_error(err, "_gpu_for!")
    end
    return nothing
end

@kernel function _generic_scatter_kernel!(mats, @Const(idxs), g)
    i = @index(Global)
    idx = @inbounds idxs[i]
    vals = g(idx)
    Bramble._write_components!(mats, vals, idx)
end

function _gpu_scatter_for!(policy, mats::Tuple, idxs, g)
    dev = get_backend(mats[1])
    try
        _generic_scatter_kernel!(dev)(mats, idxs, g; ndrange = length(idxs))
    catch err
        _wrap_device_kernel_error(err, "_gpu_scatter_for!")
    end
    return nothing
end

# --- `Rₕ!` (src/space/operators/restriction.jl is the CPU original) --------------------- #

@kernel function _restriction_kernel!(v, @Const(pts), f)
    i = @index(Global)
    @inbounds v[i] = f(pts[i])
end

function _launch_restriction!(v::AbstractVector, pts::AbstractVector, f, dev)
    try
        _restriction_kernel!(dev)(v, pts, f; ndrange = length(v))
    catch err
        _wrap_device_kernel_error(err, "Rₕ!")
    end
    return nothing
end

@kernel function _restriction_scatter_kernel!(mats, @Const(pts), f)
    i = @index(Global)
    vals = f(@inbounds pts[i])
    Bramble._write_components!(mats, vals, i)
end

function _launch_restriction_scatter!(mats::Tuple, pts::AbstractVector, f, dev)
    try
        _restriction_scatter_kernel!(dev)(mats, pts, f; ndrange = length(pts))
    catch err
        _wrap_device_kernel_error(err, "Rₕ!")
    end
    return nothing
end

# `D >= 2` counterparts (gpena/Bramble.jl#94, #174, S2.3): `pts` is a `Tuple` of `D`
# per-axis coordinate vectors (`points(Ωₕ::MeshnD)`), passed whole as one top-level kernel
# argument -- `Adapt.jl` converts a `Tuple` of arrays element-wise, confirmed against a
# real Metal device while designing this, unlike a struct nesting the same arrays. `idxs`
# (`indices(Ωₕ)`) is a bits `CartesianIndices`, so `idxs[i]` and `pts[d][I[d]]` are both
# ordinary arithmetic, not scalar array indexing. `f` still receives an `NTuple{D}`, per
# `Rₕ!`'s own docstring.
@kernel function _restriction_nd_kernel!(v, @Const(pts::NTuple{D}), @Const(idxs), f) where {D}
    i = @index(Global)
    I = @inbounds idxs[i]
    pt = ntuple(d -> (@inbounds pts[d][I[d]]), Val(D))
    @inbounds v[i] = f(pt)
end

function _launch_restriction_nd!(v::AbstractVector, pts::Tuple, idxs, f, dev)
    try
        _restriction_nd_kernel!(dev)(v, pts, idxs, f; ndrange = length(v))
    catch err
        _wrap_device_kernel_error(err, "Rₕ!")
    end
    return nothing
end

@kernel function _restriction_scatter_nd_kernel!(mats, @Const(pts::NTuple{D}), @Const(idxs), f) where {D}
    i = @index(Global)
    I = @inbounds idxs[i]
    pt = ntuple(d -> (@inbounds pts[d][I[d]]), Val(D))
    vals = f(pt)
    Bramble._write_components!(mats, vals, i)
end

function _launch_restriction_scatter_nd!(mats::Tuple, pts::Tuple, idxs, f, dev)
    try
        _restriction_scatter_nd_kernel!(dev)(mats, pts, idxs, f; ndrange = length(idxs))
    catch err
        _wrap_device_kernel_error(err, "Rₕ!")
    end
    return nothing
end

# --- `avgₕ!` (src/space/operators/cell_average.jl is the CPU original) ------------------ #
#
# `Bramble._cell_average` is the exact quadrature the CPU sweep runs -- called here rather
# than duplicated, so the device and host answers stay identical by construction, not by
# two implementations agreeing.

@kernel function _cell_average_kernel!(v, @Const(x), nodes, wts, f)
    i = @index(Global)
    @inbounds v[i] = Bramble._cell_average(f, x, i, nodes, wts)
end

function _launch_cell_average!(v::AbstractVector, x::AbstractVector, nodes, wts, f, dev)
    try
        _cell_average_kernel!(dev)(v, x, nodes, wts, f; ndrange = length(v))
    catch err
        _wrap_device_kernel_error(err, "avgₕ!")
    end
    return nothing
end

@kernel function _cell_average_scatter_kernel!(mats, @Const(x), nodes, wts, f)
    i = @index(Global)
    vals = Bramble._cell_average(f, x, i, nodes, wts)
    Bramble._write_components!(mats, vals, i)
end

function _launch_cell_average_scatter!(mats::Tuple, x::AbstractVector, nodes, wts, f, dev)
    n = length(x) - 1
    try
        _cell_average_scatter_kernel!(dev)(mats, x, nodes, wts, f; ndrange = n)
    catch err
        _wrap_device_kernel_error(err, "avgₕ!")
    end
    return nothing
end

# `D >= 2` counterparts: `half_points(Ωₕ::MeshnD) -> NTuple{D,AbstractVector}` is exactly
# the `x` shape `Bramble._cell_average`'s 2D/3D methods already take, so the kernel calls
# it unchanged with `x` (a `Tuple` of top-level device arrays) and `I` (a `CartesianIndex{D}`
# read from the bits `idxs` argument) -- same non-nesting rule as `_restriction_nd_kernel!`.
@kernel function _cell_average_nd_kernel!(v, @Const(x::NTuple{D}), @Const(idxs), nodes, wts, f) where {D}
    i = @index(Global)
    I = @inbounds idxs[i]
    @inbounds v[i] = Bramble._cell_average(f, x, I, nodes, wts)
end

function _launch_cell_average_nd!(v::AbstractVector, x::Tuple, idxs, nodes, wts, f, dev)
    try
        _cell_average_nd_kernel!(dev)(v, x, idxs, nodes, wts, f; ndrange = length(v))
    catch err
        _wrap_device_kernel_error(err, "avgₕ!")
    end
    return nothing
end

@kernel function _cell_average_scatter_nd_kernel!(mats, @Const(x::NTuple{D}), @Const(idxs), nodes, wts, f) where {D}
    i = @index(Global)
    I = @inbounds idxs[i]
    vals = Bramble._cell_average(f, x, I, nodes, wts)
    Bramble._write_components!(mats, vals, i)
end

function _launch_cell_average_scatter_nd!(mats::Tuple, x::Tuple, idxs, nodes, wts, f, dev)
    try
        _cell_average_scatter_nd_kernel!(dev)(mats, x, idxs, nodes, wts, f; ndrange = length(idxs))
    catch err
        _wrap_device_kernel_error(err, "avgₕ!")
    end
    return nothing
end

# --- Difference, jump and average operators (src/space/operators/difference.jl and
# src/space/operators/average.jl are the CPU originals) ---------------------------------- #
#
# S2.4 (gpena/Bramble.jl#94, #174): `jump.jl` needs nothing of its own here -- `jump_dim!`/
# `jump!` forward straight into `forward_difference_dim!`/`forward_difference!`, so the
# one-sided launcher below already covers it.
#
# Revised by S12 (gpena/Bramble.jl#306, #302): through S11 every kernel here built a
# `CartesianIndex` via `@index(Global, Cartesian)`, then converted it back to a linear index
# with `LinearIndices(dims)` -- once for the point itself, once more for its neighbour -- and
# branched on `I[DIM]` with a data-dependent `if`/`elseif`, which is exactly the boundary
# divergence #306 measured splitting Metal's memory transactions and dropping bus throughput
# from ~74 GB/s to ~18 GB/s. Every kernel below instead launches over a flat linear
# `ndrange = length(out)`, takes the *scalar* stride along `DIM` (`_axis_stride`, computed
# once on the host before the launch, from `dims` and `DIM` -- both compile-time `Val`s, so
# the loop inside it unrolls to nothing at runtime) and derives the point's own coordinate
# along `DIM` from one `fld`/`mod` pair, never a `CartesianIndex`. The boundary test becomes
# one index comparison, and the branch itself is replaced by `ifelse` (a predicated select,
# not a divergent instruction stream): both the interior and boundary values are computed
# unconditionally -- reading a clamped, always-in-bounds neighbour index when the true one
# would fall off the grid -- and `ifelse` picks the one that matters. Every SIMD-group thread
# now runs the identical instruction sequence regardless of where it sits on the grid.
#
# Every kernel still calls the very `Bramble._compute_difference`/`Bramble._compute_average`
# methods the CPU sweep calls, so the device and host answers stay identical by construction
# for every family except the two named in the reciprocal-spacing note below. `h` arrives
# already resolved to either `nothing` or a plain top-level array by
# `Bramble._resolve_device_spacing` (`difference.jl`): never the `StarSpacings` wrapper,
# which -- like any struct nesting a device array -- fails kernel compilation.
#
# --- Threadgroup shape, measured and rejected -------------------------------------------
#
# The stride/branch-free rewrite above leaves this kernel at ~20% of the 73.5 GB/s Metal
# peak (measured on a 3000x3000 mesh: ~4.9 ms on a uniform mesh, ~4.67 ms on the non-uniform
# one `benchmark/gpu_stencils.jl`'s own workgroup-shape section reruns) with the GPU busy
# essentially the whole wall-clock time -- not launch overhead, and not (per the device-side
# profile) idle time either, so #306/#302's own remaining proposal was worth checking:
# "configure threadgroups for optimal SIMD coalescing (e.g. (16, 16) or (32, 8))". Measured
# directly, `Metal.@bprofile` device-side busy time, three repeated in-process runs, same
# non-uniform 3000x3000 case:
#
#   flat `ndrange = length(out)`, default workgroupsize (shipped):  ~4.67 ms/launch
#   flat `ndrange = length(out)`, explicit workgroupsize 256/512/1024: ~4.67 ms/launch (no
#     measurable difference from the default -- KernelAbstractions' own choice already
#     matches whatever these three do)
#   2D `ndrange = dims`, workgroupsize (16, 16):  ~8.53 ms/launch (1.8x SLOWER)
#   2D `ndrange = dims`, workgroupsize (32, 8):   ~8.49 ms/launch (1.8x SLOWER)
#   2D `ndrange = dims`, default workgroupsize:   ~8.57 ms/launch (1.8x SLOWER)
#
# All three were stable to <0.01 ms across the three runs -- not noise. `@index(Global)`
# under a 2D `ndrange` was confirmed separately to return the identical column-major linear
# index a flat `ndrange` gives (so the flat-index stride arithmetic above did not have to
# change to try this; only the launch's `ndrange`/`workgroupsize` did), which rules out a
# correctness difference explaining the gap -- both shapes were checked against the CPU
# reference and agree to the same tolerance. The 2D dispatch is simply slower on this
# device: shipped stays the flat `ndrange`, default `workgroupsize`, exactly as it already
# was before this was tried. #302's own hand-written 2D kernel measurement (293.5 us at
# 1024x1024, predicting ~2.5 ms scaled to 3000x3000) does not reproduce here, on Metal.jl
# 1.10 and this M2 -- reported as measured, not assumed.
#
# --- 64-bit index arithmetic, the actual bottleneck the threadgroup-shape and reciprocal-
#     spacing measurements above were both taken *underneath* --------------------------- #
#
# A control settles what the ~20%-of-peak figure above means: a plain KernelAbstractions
# copy kernel, same element count, same flat `ndrange`, same `@index(Global)` pattern,
# reaches ~98% of this machine's own measured ceiling (a KA copy kernel here: 72.35 GB/s,
# not #306's 73.5 GB/s Metal spec figure -- see the note on percentages below). So neither
# kernel launch overhead nor KA's indexing pattern in general explains the gap: the copy
# kernel pays the same overhead and still saturates. The one thing the difference kernel
# does that the copy kernel does not is recover the axis coordinate from the flat index
# with `i_dim = mod(fld(idx - 1, s), n) + 1` -- one integer division and one modulo, every
# thread, every launch -- and `idx`, `s`, `n` are all Julia's default `Int` (`Int64`).
# Apple GPUs are 32-bit-native; Metal.jl 1.10 shipping `UInt16` variants of every thread-
# and grid-indexing intrinsic, and gpena/Bramble.jl#319 independently arguing for `Int32`
# sparse indices on this same hardware, are both symptoms of the same fact: 64-bit integer
# division on this hardware is not native-width arithmetic.
#
# Measured directly (three repeated `Metal.@bprofile` device-side-busy-time runs, same
# non-uniform 3000x3000 case, `D₋ₓ!`): doing the *exact same* `fld`/`mod` recovery in
# `Int32` instead of `Int64` -- convert `idx` once at kernel entry, take `n`/`s` in as
# `Int32` from the launcher so nothing widens back inside the kernel, convert the *result*
# back to `Int` only where array indexing needs it -- took the kernel from 4.67-5.00 ms/launch
# to 1.155-1.203 ms/launch, stable to within 0.05 ms across two of the three runs (the first
# run's outlier reads as compile/warm-up noise inside the profiling window, matching the
# pattern seen elsewhere in this file). Two diagnostics taken first, to make sure the right
# thing was being isolated before reaching for this fix: a scalar-`h`/no-index-recovery
# variant (not shippable -- boundary correctness dropped on purpose) reached 0.77-0.87 ms,
# and a full-size reciprocal array read directly by the flat index, no div/mod at all
# (also a diagnostic: it does not amortise a per-launch construction cost, measured
# separately at 1.2-3.7 ms and unstable run to run -- rejected for that reason) reached
# 1.16 ms. The `Int32` fix lands in between the two diagnostics, with none of either one's
# downsides: no extra array, no extra memory traffic, no per-launch construction to pay for,
# and the same win on every axis (`DIM = 1`, `2` or `3`) rather than only the cheap one.
# Correctness: bitwise identical to the `Int64` version on the same input (`max|Δ| = 0.0`),
# and within `rtol = 1f-5` of the CPU reference, exactly as the `Int64` version was.
#
# `benchmark/gpu_stencils.jl`'s own `_run_int32_index` section reruns this comparison.
#
# One reading note, since a percentage invites misreading it as an absolute: the "fraction
# of peak" figures anywhere in this file or its benchmarks use whichever peak was measured
# on THIS host (a plain KA copy kernel, ~72.35 GB/s), not the 73.5 GB/s Metal hardware
# spec figure #306 quotes -- and even that is a floor, not a ceiling: a kernel whose
# neighbour reads overlap between adjacent threads (as every stencil's do) can measure
# *above* either figure, because some of those reads hit a cache line an adjacent thread
# already pulled rather than round-tripping DRAM. Read the ratios in this file as "how much
# closer to what a copy achieves", not as a literal fraction of an DRAM bandwidth ceiling.
#
# --- Reciprocal spacing (gpena/Bramble.jl#306 item 4) ------------------------------------
#
# Measured before the `Int32` fix above (so under a much larger, now-removed constant
# factor): once the stride/branch-free rewrite is in place, a per-thread
# `(cur - other) / h[i_dim]` against `(cur - other) * invh[i_dim]` -- `invh` a bulk
# `inv.(h)` computed once, up front -- gave a stable 1.03x, small because the `Int64`
# div/mod dominated everything else at the time. Kept after the `Int32` fix regardless: it
# is still strictly cheaper (one multiply against one divide, same memory traffic, same
# array, no new state), so there is no reason to divide once multiplying by a fresh
# reciprocal is already correct and in place. `invh` is recomputed fresh, once per launch,
# from whatever `h` this call was given (`_reciprocal_spacing` below), never cached on the
# mesh across calls, for the reasons given at S12's `inv_spacings` design note (`h` is a
# per-*axis* array, and a mesh-level cache would not reach `D₊`'s or `Dc`'s `h` either,
# which are not the mesh's own cached `spacings` field).
#
# Applied to the two families #302/#306 actually measured as bottlenecks (`D₋ₓ`/`D₊ₓ`, the
# one-sided finite differences, and `Dcₓ`, `Centered`): `CrossWeighted` (`Dₕ`) reads two
# distinct raw spacings and combines them in a weighted average that is not a single
# reciprocal multiply, so its interior branch keeps calling `_compute_difference` unchanged
# (still a division, functionally identical to before this file) -- it still gets the
# `Int32` fix, independent of this choice. The average engine has no spacing at all (it
# divides by the literal constant 2) and needs none of this.
# ---------------------------------------------------------------------------

@inline _reciprocal_spacing(::Nothing) = nothing
@inline _reciprocal_spacing(h::AbstractVector) = inv.(h)

# The linear stride along axis `DIM` of a column-major `dims::NTuple{D,Int}` array: 1 for
# `DIM == 1`, `dims[1]` for `DIM == 2`, `dims[1] * dims[2]` for `DIM == 3` -- exactly the
# formula #306's proposal 2 gives. `D` and `DIM` are both `Val`s here, so this loop unrolls
# to a handful of multiplications at compile time; it runs once on the host before a launch,
# never per thread inside a kernel. Returned as `Int32` directly (see the file-level note
# above): every kernel below takes `n`/`s` already narrowed, so nothing widens them back.
#
# `Int32(s)` throws (an ordinary `InexactError`, not a cryptic on-device one) once the
# stride itself -- the product of every axis before `DIM`, which for a 3D mesh's third axis
# is `dims[1] * dims[2]` -- exceeds `typemax(Int32)` (~2.1 billion): a mesh that large is far
# past anything this milestone's hardware (or a Float32 field's memory footprint) supports
# today, so this is a real but currently unreachable limit, named here rather than left to
# surface as an unexplained conversion error deep in a kernel launch.
@inline function _axis_stride(dims::NTuple{D, Int}, ::Val{DIM}) where {D, DIM}
    s = 1
    for d in 1:(DIM - 1)
        s *= dims[d]
    end
    s <= typemax(Int32) || error(
        "mesh too large for the Int32 device-index fast path: the linear stride along " *
        "axis $DIM is $s, past typemax(Int32) = $(typemax(Int32))",
    )
    return Int32(s)
end

# A compile-time choice between the two neighbour steps a one-sided stencil can take,
# resolved from `dir`'s own (concrete, zero-field) type -- never a per-thread branch on
# data, unlike the boundary test below.
@inline _signed_stride(::Bramble.Forward, s::Int32) = s
@inline _signed_stride(::Bramble.Backward, s::Int32) = -s

# The point's own coordinate along `DIM`, recovered from the flat `idx` in `Int32`
# arithmetic (the file-level note above this section is the why): `idx` itself stays
# whatever type `@index(Global)` gives (used for array indexing, where its width is not the
# bottleneck), only the division/modulo operands are narrowed. Returns an `Int32`; callers
# convert to `Int` only where `Bramble._compute_difference`/`_compute_average` (typed on a
# plain `Int` index, and never actually reading it in the boundary methods -- see below)
# need one.
@inline function _axis_coord32(idx, n::Int32, s::Int32)
    idx32 = Int32(idx)
    return mod(fld(idx32 - Int32(1), s), n) + Int32(1)
end

@kernel function _difference_onesided_kernel!(
        out, @Const(in_ref), h, invh, n::Int32, s::Int32, boundary_idx::Int32, dir, ::Val{DIM}
) where {DIM}
    idx = @index(Global)
    @inbounds begin
        i_dim32 = _axis_coord32(idx, n, s)
        is_boundary = i_dim32 == boundary_idx
        other_idx = ifelse(is_boundary, idx, idx + Int(_signed_stride(dir, s)))
        cur = in_ref[idx]
        other = in_ref[other_idx]
        i_dim = Int(i_dim32)
        interior_val = _onesided_interior(dir, cur, other, h, invh, i_dim)
        boundary_val = Bramble._compute_difference(dir, Val(true), cur, h, i_dim)
        out[idx] = ifelse(is_boundary, boundary_val, interior_val)
    end
end

# Unscaled (`h === nothing`, so `invh === nothing` too, `_reciprocal_spacing` below):
# no division ever appeared here, so this falls straight back to `_compute_difference`.
@inline _onesided_interior(
    dir::Bramble.GridDirection, cur, other, ::Nothing, ::Nothing, i) = Bramble._compute_difference(
    dir, Val(false), cur, other, nothing, i)
# Scaled: multiply by the reciprocal instead of dividing by `h` (see the file-level note
# above this kernel's definitions).
@inline _onesided_interior(::Bramble.Forward, cur, other, h, invh::AbstractVector, i) = (other - cur) *
                                                                                        (@inbounds invh[i])
@inline _onesided_interior(::Bramble.Backward, cur, other, h, invh::AbstractVector, i) = (cur - other) *
                                                                                         (@inbounds invh[i])

function _launch_difference_onesided!(
        out::AbstractVector, in_ref, h, dims::Tuple, dir, dim_val::Val{DIM}, dev) where {DIM}
    n = dims[DIM]
    s = _axis_stride(dims, dim_val)
    invh = _reciprocal_spacing(h)
    boundary_idx = Int32(Bramble._stencil_boundary_dim(dir, n))
    try
        _difference_onesided_kernel!(dev)(
            out, in_ref, h, invh, Int32(n), s, boundary_idx, dir, dim_val; ndrange = length(out)
        )
    catch err
        _wrap_device_kernel_error(err, "difference operator")
    end
    return nothing
end

@kernel function _difference_centered_kernel!(
        out, @Const(in_ref), h, invh, n::Int32, s::Int32, dir, ::Val{DIM}
) where {DIM}
    idx = @index(Global)
    @inbounds begin
        i_dim32 = _axis_coord32(idx, n, s)
        is_lo = i_dim32 == Int32(1)
        is_hi = i_dim32 == n
        sInt = Int(s)
        back_idx = ifelse(is_lo, idx, idx - sInt)
        fwd_idx = ifelse(is_hi, idx, idx + sInt)
        cur = in_ref[idx]
        back = in_ref[back_idx]
        fwd = in_ref[fwd_idx]
        i_dim = Int(i_dim32)
        # `CrossWeighted`'s interior formula reads `h[i]` *and* `h[i + 1]`
        # (`_compute_difference`, difference.jl): at `i_dim == n` that second read is out of
        # bounds, even though the result is about to be discarded by the `ifelse` below --
        # every thread still evaluates it speculatively, branch-free. `i_interior` clamps to
        # `n - 1` only for that speculative read; `i_dim` itself (unclamped) is still what
        # `lo_val`/`hi_val` and the destination index use.
        i_interior = ifelse(is_hi, Int(n) - 1, i_dim)
        interior_val = _centered_interior(dir, back, cur, fwd, h, invh, i_interior)
        lo_val = Bramble._compute_difference(dir, Val(true), cur, fwd, h, i_dim)
        hi_val = Bramble._compute_difference(dir, Val(true), cur, back, h, i_dim)
        out[idx] = ifelse(is_lo, lo_val, ifelse(is_hi, hi_val, interior_val))
    end
end

# `Centered` (`Dc`) divides by `2 * h[i]`, `h` here already the averaged *star* spacing
# (`star_spacings`, resolved before this launch): one multiply by `invh[i] / 2` replaces it.
@inline _centered_interior(dir::Bramble.Centered, back, cur, fwd, h, invh::AbstractVector, i) = (fwd - back) *
                                                                                                (@inbounds invh[i]) / 2
# `CrossWeighted` (`Dₕ`) reads two distinct raw spacings and combines them in a weighted
# average that is not a single reciprocal multiply (see the file-level note above); kept on
# `_compute_difference`, unchanged.
@inline _centered_interior(dir::Bramble.CrossWeighted, back, cur, fwd, h, invh, i) = Bramble._compute_difference(
    dir, Val(false), back, cur, fwd, h, i)

function _launch_difference_centered!(
        out::AbstractVector, in_ref, h, dims::Tuple, dir, dim_val::Val{DIM}, dev) where {DIM}
    n = dims[DIM]
    s = _axis_stride(dims, dim_val)
    invh = _reciprocal_spacing(h)
    try
        _difference_centered_kernel!(dev)(out, in_ref, h, invh, Int32(n), s, dir, dim_val; ndrange = length(out))
    catch err
        _wrap_device_kernel_error(err, "centered difference operator")
    end
    return nothing
end

@kernel function _average_kernel!(
        out, @Const(in_ref), n::Int32, s::Int32, boundary_idx::Int32, dir, ::Val{DIM}) where {DIM}
    idx = @index(Global)
    @inbounds begin
        i_dim32 = _axis_coord32(idx, n, s)
        is_boundary = i_dim32 == boundary_idx
        other_idx = ifelse(is_boundary, idx, idx + Int(_signed_stride(dir, s)))
        cur = in_ref[idx]
        other = in_ref[other_idx]
        interior_val = Bramble._compute_average(dir, Val(false), cur, other)
        boundary_val = Bramble._compute_average(dir, Val(true), cur)
        out[idx] = ifelse(is_boundary, boundary_val, interior_val)
    end
end

function _launch_average_engine!(out::AbstractVector, in_ref, dims::Tuple, dir, dim_val::Val{DIM}, dev) where {DIM}
    n = dims[DIM]
    s = _axis_stride(dims, dim_val)
    boundary_idx = Int32(Bramble._stencil_boundary_dim(dir, n))
    try
        _average_kernel!(dev)(out, in_ref, Int32(n), s, boundary_idx, dir, dim_val; ndrange = length(out))
    catch err
        _wrap_device_kernel_error(err, "average operator")
    end
    return nothing
end

# --- Row-parallel SpMV/SpMM for a device CSR matrix (gpena/Bramble.jl#250, #174, S3.2) -- #
#
# `BrambleMetalExt`'s `mul!` methods for `MetalSparseMatrixCSR` call `_launch_spmv_csr!`/
# `_launch_spmm_csr!` with the matrix's raw `rowPtr`/`colVal`/`nzVal` arrays -- never the
# struct itself, per this file's own header comment -- so these kernels are written against
# `KernelAbstractions.Backend` alone and never see `Metal.MtlVector`. One work item owns one
# output row (SpMV) or one output entry of a row (SpMM), so there are no write conflicts and
# no atomics; the device is read off the destination array with `get_backend`, the same
# idiom `_gpu_for!` above uses, rather than threaded through as its own argument.

@kernel function _spmv_csr_kernel!(y, @Const(rowPtr), @Const(colVal), @Const(nzVal), @Const(x), α, β)
    row = @index(Global)
    @inbounds begin
        acc = zero(eltype(y))
        for k in rowPtr[row]:(rowPtr[row + 1] - 1)
            acc += nzVal[k] * x[colVal[k]]
        end
        y[row] = iszero(β) ? α * acc : α * acc + β * y[row]
    end
end

function _launch_spmv_csr!(y::AbstractVector, rowPtr, colVal, nzVal, x::AbstractVector, α, β)
    dev = get_backend(y)
    try
        _spmv_csr_kernel!(dev)(y, rowPtr, colVal, nzVal, x, α, β; ndrange = length(y))
    catch err
        _wrap_device_kernel_error(err, "Metal sparse mul! (SpMV)")
    end
    return nothing
end

# `KroneckerLinearOperator` mode contraction by a symmetric sparse 1D factor
# (gpena/Bramble.jl#323, `src/form/kronecker.jl`). One work item per output entry of the flat
# `pre x m x post` array: output `(i, j, k)` gathers column `j` of the CSC storage, which
# equals row `j` because every factor is symmetric, so no write conflicts and no atomics.
@kernel function _kron_sparse_mode_kernel!(Y, @Const(colptr), @Const(rowval), @Const(nzval), @Const(X), pre, m)
    g = @index(Global)
    @inbounds begin
        g0 = g - 1
        i = g0 % pre
        r = g0 ÷ pre
        j = r % m + 1
        base = (r ÷ m) * pre * m + i + 1
        acc = zero(eltype(Y))
        for p in colptr[j]:(colptr[j + 1] - Int32(1))
            acc += nzval[p] * X[base + (rowval[p] - 1) * pre]
        end
        Y[g] = acc
    end
end

function _launch_kron_sparse_mode!(
        Y::AbstractVector, colptr, rowval, nzval, X::AbstractVector, pre::Int, m::Int, post::Int)
    dev = get_backend(Y)
    try
        _kron_sparse_mode_kernel!(dev)(Y, colptr, rowval, nzval, X, pre, m; ndrange = pre * m * post)
    catch err
        _wrap_device_kernel_error(err, "KroneckerLinearOperator mul!")
    end
    return nothing
end

@kernel function _spmm_csr_kernel!(C, @Const(rowPtr), @Const(colVal), @Const(nzVal), @Const(B), α, β)
    I = @index(Global, Cartesian)
    row = I[1]
    col = I[2]
    @inbounds begin
        acc = zero(eltype(C))
        for k in rowPtr[row]:(rowPtr[row + 1] - 1)
            acc += nzVal[k] * B[colVal[k], col]
        end
        C[row, col] = iszero(β) ? α * acc : α * acc + β * C[row, col]
    end
end

function _launch_spmm_csr!(C::AbstractMatrix, rowPtr, colVal, nzVal, B::AbstractMatrix, α, β)
    dev = get_backend(C)
    try
        _spmm_csr_kernel!(dev)(C, rowPtr, colVal, nzVal, B, α, β; ndrange = size(C))
    catch err
        _wrap_device_kernel_error(err, "Metal sparse mul! (SpMM)")
    end
    return nothing
end

# --- Fused vector-calculus kernels (gpena/Bramble.jl#306, #302, S12 part 4) -------------- #
#
# `src/space/operators/vector_calculus.jl`'s CPU engines accumulate one spatial direction
# (or, for the strain tensor's off-diagonal entries, one difference and the average composed
# onto it) per pass over the whole grid, scalar-indexing `out[idx] += ...` as they go -- which
# a device array refuses outright, and which even where it would not throw would round-trip
# global memory once per direction instead of once. Every kernel below instead reads each
# component array exactly once per grid point and accumulates every direction's contribution
# in registers before the one write to `out`, using the same direct-stride, branch-free-select
# idiom the difference/average kernels above do: `_onesided_term` is that idiom's shared
# building block (one truncatable one-sided term, scaled by a reciprocal spacing), and every
# kernel here is a small, explicit composition of it -- never a generic `NTuple{D}` loop
# hidden behind an extra abstraction, since `D` is always 1, 2 or 3.
#
# `h`/`hs`/`hbs`/`hss` arrive already resolved to plain top-level arrays by
# `Bramble._resolve_device_spacing`, exactly as the difference kernels' `h` does; the
# reciprocal is computed once per launch, from whatever array arrives, for the same reason
# and with the same measured payoff as the difference kernels' own reciprocal-spacing note
# above (this array is per-*axis*, not per-*grid-point*, so the extra broadcast is tiny next
# to the kernel it feeds). `n`/`s` are `Int32` throughout, for the same reason and with the
# same measured payoff as the difference kernels' own 64-bit-index-arithmetic note above:
# every one of these kernels does the identical `fld`/`mod` recovery per direction, so the
# fix is not specific to the two-array difference kernel it was isolated on.

@inline function _onesided_term(dir, u, invh, n::Int32, s::Int32, idx)
    @inbounds begin
        i_dim32 = _axis_coord32(idx, n, s)
        is_boundary = i_dim32 == Int32(Bramble._stencil_boundary_dim(dir, Int(n)))
        other_idx = ifelse(is_boundary, idx, idx + Int(_signed_stride(dir, s)))
        cur = u[idx]
        other = u[other_idx]
        return ifelse(is_boundary, zero(cur), _onesided_pure(dir, cur, other) * invh[Int(i_dim32)])
    end
end

@inline _onesided_pure(::Bramble.Forward, cur, other) = other - cur
@inline _onesided_pure(::Bramble.Backward, cur, other) = cur - other

# --- Divergence: divₕ (Backward) / div₊ₕ (Forward) --------------------------------------- #

@kernel function _fused_divergence1d_kernel!(out, @Const(u1), invh1, n1::Int32, s1::Int32, dir)
    idx = @index(Global)
    @inbounds out[idx] = _onesided_term(dir, u1, invh1, n1, s1, idx)
end

@kernel function _fused_divergence2d_kernel!(
        out, @Const(u1), @Const(u2), invh1, invh2, n1::Int32, n2::Int32, s1::Int32, s2::Int32, dir
)
    idx = @index(Global)
    @inbounds out[idx] = _onesided_term(dir, u1, invh1, n1, s1, idx) +
                         _onesided_term(dir, u2, invh2, n2, s2, idx)
end

@kernel function _fused_divergence3d_kernel!(
        out, @Const(u1), @Const(u2), @Const(u3), invh1, invh2, invh3, n1::Int32, n2::Int32, n3::Int32,
        s1::Int32, s2::Int32, s3::Int32, dir
)
    idx = @index(Global)
    @inbounds out[idx] = _onesided_term(dir, u1, invh1, n1, s1, idx) +
                         _onesided_term(dir, u2, invh2, n2, s2, idx) +
                         _onesided_term(dir, u3, invh3, n3, s3, idx)
end

function _launch_fused_divergence!(out::AbstractVector, comps::Tuple, hs::Tuple, dims::Tuple, dir, dev)
    D = length(dims)
    strides = ntuple(d -> _axis_stride(dims, Val(d)), Val(D))
    ns32 = ntuple(d -> Int32(dims[d]), Val(D))
    invhs = map(h -> inv.(h), hs)
    try
        if D == 1
            _fused_divergence1d_kernel!(dev)(out, comps[1], invhs[1], ns32[1], strides[1], dir; ndrange = length(out))
        elseif D == 2
            _fused_divergence2d_kernel!(dev)(
                out, comps[1], comps[2], invhs[1], invhs[2], ns32[1], ns32[2], strides[1], strides[2], dir;
                ndrange = length(out)
            )
        else
            _fused_divergence3d_kernel!(
                dev)(
                out, comps[1], comps[2], comps[3], invhs[1], invhs[2], invhs[3], ns32[1], ns32[2], ns32[3],
                strides[1], strides[2], strides[3], dir; ndrange = length(out)
            )
        end
    catch err
        _wrap_device_kernel_error(err, "divₕ!/div₊ₕ!")
    end
    return nothing
end

# --- Curl: curlₕ (Backward) / curl₊ₕ (Forward) -------------------------------------------- #

@kernel function _fused_curl2d_kernel!(
        out, @Const(u1), @Const(u2), invh1, invh2, n1::Int32, n2::Int32, s1::Int32, s2::Int32, dir
)
    idx = @index(Global)
    @inbounds out[idx] = _onesided_term(dir, u2, invh1, n1, s1, idx) -
                         _onesided_term(dir, u1, invh2, n2, s2, idx)
end

function _launch_fused_curl2d!(out::AbstractVector, u1, u2, h1, h2, dims::Tuple, dir, dev)
    s1 = _axis_stride(dims, Val(1))
    s2 = _axis_stride(dims, Val(2))
    invh1 = inv.(h1)
    invh2 = inv.(h2)
    try
        _fused_curl2d_kernel!(dev)(
            out, u1, u2, invh1, invh2, Int32(dims[1]), Int32(dims[2]), s1, s2, dir; ndrange = length(out)
        )
    catch err
        _wrap_device_kernel_error(err, "curlₕ!/curl₊ₕ!")
    end
    return nothing
end

@kernel function _fused_curl3d_kernel!(
        out1, out2, out3, @Const(u1), @Const(u2), @Const(u3), invh1, invh2, invh3,
        n1::Int32, n2::Int32, n3::Int32, s1::Int32, s2::Int32, s3::Int32, dir
)
    idx = @index(Global)
    @inbounds begin
        out1[idx] = _onesided_term(dir, u3, invh2, n2, s2, idx) - _onesided_term(dir, u2, invh3, n3, s3, idx)
        out2[idx] = _onesided_term(dir, u1, invh3, n3, s3, idx) - _onesided_term(dir, u3, invh1, n1, s1, idx)
        out3[idx] = _onesided_term(dir, u2, invh1, n1, s1, idx) - _onesided_term(dir, u1, invh2, n2, s2, idx)
    end
end

function _launch_fused_curl3d!(out1, out2, out3, u1, u2, u3, h1, h2, h3, dims::Tuple, dir, dev)
    s1 = _axis_stride(dims, Val(1))
    s2 = _axis_stride(dims, Val(2))
    s3 = _axis_stride(dims, Val(3))
    invh1, invh2, invh3 = inv.(h1), inv.(h2), inv.(h3)
    try
        _fused_curl3d_kernel!(dev)(
            out1, out2, out3, u1, u2, u3, invh1, invh2, invh3, Int32(dims[1]), Int32(dims[2]), Int32(dims[3]),
            s1, s2, s3, dir; ndrange = length(out1)
        )
    catch err
        _wrap_device_kernel_error(err, "curlₕ!/curl₊ₕ!")
    end
    return nothing
end

# --- Laplacian: Δₕ ------------------------------------------------------------------------ #
#
# One direction's flux-difference term, speculatively evaluated at every point and masked
# rather than branched: `mask_hi` truncates the whole term to zero at the last slice along
# this direction (matching `_accumulate_laplacian!`'s CPU sweep, which never visits it at
# all), and every neighbour index and every `h`/`hs` index used only by a speculative read is
# clamped to a safe in-bounds value with `ifelse` when the true one would fall outside its
# own array -- `hs` (`star_spacings`, resolved) has `n - 1` entries, one fewer than `hb`, so
# it needs its own clamp at `i_dim == n`, not just `hb`'s.
#
# Division, not the reciprocal-multiply the difference/average/divergence/curl kernels above
# use: measured directly on a mildly non-uniform 1D mesh (spacing ratio ~2.8, nothing
# pathological), multiplying by `1 ./ hb` and `1 ./ hs` disagreed with the CPU (dividing)
# reference by up to 0.85% relative -- two orders of magnitude past `rtol = 1f-5` -- at
# ordinary interior points, not just where a spacing happens to be tiny. The Laplacian
# subtracts two flux terms that are individually much larger than their difference (a
# forward and a backward first difference, each divided by a spacing on the order of `1/n`,
# so each flux is `O(n)` while a smooth field's second difference is `O(1)`): that is
# catastrophic cancellation by construction, and it amplifies the one-ULP disagreement
# between `x / h` and `x * (1 / h)` into a difference orders of magnitude larger than either
# value alone. The difference/average/divergence/curl kernels have no such cancellation (each
# writes one first difference, not the difference of two comparably-sized ones), which is
# where their own reciprocal-spacing note's measurement actually applies -- it does not
# transfer here, and re-measuring on the Laplacian's own shape says so.
@inline function _laplacian_term(u, cur, hb, hs, n::Int32, s::Int32, idx)
    @inbounds begin
        i_dim32 = _axis_coord32(idx, n, s)
        mask_hi = i_dim32 != n
        mask_lo = i_dim32 != Int32(1)
        sInt = Int(s)
        fwd_idx = ifelse(mask_hi, idx + sInt, idx)
        back_idx = ifelse(mask_lo, idx - sInt, idx)
        i_dim = Int(i_dim32)
        hb_fwd_i = ifelse(mask_hi, i_dim + 1, i_dim)
        hs_i = ifelse(mask_hi, i_dim, Int(n) - 1) # `hs` has n - 1 entries; clamp only matters when discarded
        fwd_val = u[fwd_idx]
        back_val = u[back_idx]
        forward_flux = (fwd_val - cur) / hb[hb_fwd_i]
        backward_flux = mask_lo * (cur - back_val) / hb[i_dim]
        return mask_hi * (forward_flux - backward_flux) / hs[hs_i]
    end
end

@kernel function _fused_laplacian1d_kernel!(out, @Const(u), hb1, hs1, n1::Int32, s1::Int32)
    idx = @index(Global)
    @inbounds begin
        cur = u[idx]
        out[idx] = _laplacian_term(u, cur, hb1, hs1, n1, s1, idx)
    end
end

@kernel function _fused_laplacian2d_kernel!(
        out, @Const(u), hb1, hs1, hb2, hs2, n1::Int32, n2::Int32, s1::Int32, s2::Int32
)
    idx = @index(Global)
    @inbounds begin
        cur = u[idx]
        out[idx] = _laplacian_term(u, cur, hb1, hs1, n1, s1, idx) +
                   _laplacian_term(u, cur, hb2, hs2, n2, s2, idx)
    end
end

@kernel function _fused_laplacian3d_kernel!(
        out, @Const(u), hb1, hs1, hb2, hs2, hb3, hs3,
        n1::Int32, n2::Int32, n3::Int32, s1::Int32, s2::Int32, s3::Int32
)
    idx = @index(Global)
    @inbounds begin
        cur = u[idx]
        out[idx] = _laplacian_term(u, cur, hb1, hs1, n1, s1, idx) +
                   _laplacian_term(u, cur, hb2, hs2, n2, s2, idx) +
                   _laplacian_term(u, cur, hb3, hs3, n3, s3, idx)
    end
end

function _launch_fused_laplacian!(out::AbstractVector, u, hbs::Tuple, hss::Tuple, dims::Tuple, dev)
    D = length(dims)
    strides = ntuple(d -> _axis_stride(dims, Val(d)), Val(D))
    ns32 = ntuple(d -> Int32(dims[d]), Val(D))
    try
        if D == 1
            _fused_laplacian1d_kernel!(dev)(
                out, u, hbs[1], hss[1], ns32[1], strides[1]; ndrange = length(out)
            )
        elseif D == 2
            _fused_laplacian2d_kernel!(
                dev)(
                out, u, hbs[1], hss[1], hbs[2], hss[2], ns32[1], ns32[2], strides[1], strides[2];
                ndrange = length(out)
            )
        else
            _fused_laplacian3d_kernel!(
                dev)(
                out, u, hbs[1], hss[1], hbs[2], hss[2], hbs[3], hss[3],
                ns32[1], ns32[2], ns32[3], strides[1], strides[2], strides[3]; ndrange = length(out)
            )
        end
    catch err
        _wrap_device_kernel_error(err, "Δₕ!")
    end
    return nothing
end

# --- Strain tensor: εₕ, off-diagonal entries ---------------------------------------------- #
#
# `ε_ij = (M₋ᵢ(D₋ⱼ(uᵢ)) + M₋ⱼ(D₋ᵢ(uⱼ))) / 2`, `i != j` (the CPU docstring on `εₕ`,
# `src/space/operators/vector_calculus.jl`, has the full derivation). Composed directly
# rather than as a difference kernel followed by an average kernel: `D₋ⱼ(uᵢ)` truncates to
# zero exactly when its own `j`-coordinate is 1 (`mask_j`), and that truncation is shared by
# both points the backward average at `i`-coordinate reads (shifting along `i` never changes
# the `j`-coordinate, since `i != j`) -- so the average's own truncation (`mask_i`, at
# `i`-coordinate 1) and the difference's (`mask_j`) combine into the single product
# `mask_i & mask_j` gating the whole term, with no separate pass needed. `shift_i`/`shift_j`
# zero out (rather than branch away) whichever neighbour offset a mask disallows, so every
# one of `ui`/`uj`'s four reads stays in bounds regardless of where `idx` sits on the grid.
# `ε_ii` (the diagonal) needs none of this: it is exactly `D₋ᵢ(uᵢ)`, so it reuses
# `_launch_difference_onesided!` above rather than a kernel of its own
# (`vector_calculus.jl`'s `_strain_diag!`).

@kernel function _fused_strain_offdiag_kernel!(
        out, @Const(ui), @Const(uj), invh_i, invh_j, ni::Int32, nj::Int32, si::Int32, sj::Int32
)
    idx = @index(Global)
    @inbounds begin
        i_i32 = _axis_coord32(idx, ni, si)
        i_j32 = _axis_coord32(idx, nj, sj)
        mask_i = i_i32 != Int32(1)
        mask_j = i_j32 != Int32(1)
        together = mask_i & mask_j
        siInt, sjInt = Int(si), Int(sj)
        shift_i = ifelse(mask_i, siInt, 0)
        shift_j = ifelse(mask_j, sjInt, 0)
        idx_mi = idx - shift_i
        idx_mj = idx - shift_j
        idx_mimj = idx - shift_i - shift_j
        i_i, i_j = Int(i_i32), Int(i_j32)

        ui_00 = ui[idx]
        ui_mi = ui[idx_mi]
        ui_mj = ui[idx_mj]
        ui_mimj = ui[idx_mimj]
        uj_00 = uj[idx]
        uj_mi = uj[idx_mi]
        uj_mj = uj[idx_mj]
        uj_mimj = uj[idx_mimj]

        term1 = ifelse(together, ((ui_00 - ui_mj) + (ui_mi - ui_mimj)) * invh_j[i_j] / 2, zero(ui_00))
        term2 = ifelse(together, ((uj_00 - uj_mi) + (uj_mj - uj_mimj)) * invh_i[i_i] / 2, zero(uj_00))
        out[idx] = (term1 + term2) / 2
    end
end

function _launch_fused_strain_offdiag!(
        out::AbstractVector, ui, uj, hi, hj, dims::Tuple, dim_i::Val{I}, dim_j::Val{J}, dev
) where {I, J}
    si = _axis_stride(dims, dim_i)
    sj = _axis_stride(dims, dim_j)
    invh_i = inv.(hi)
    invh_j = inv.(hj)
    try
        _fused_strain_offdiag_kernel!(dev)(
            out, ui, uj, invh_i, invh_j, Int32(dims[I]), Int32(dims[J]), si, sj; ndrange = length(out)
        )
    catch err
        _wrap_device_kernel_error(err, "εₕ!")
    end
    return nothing
end

# --- Device-write synchronisation (gpena/Bramble.jl#94, S4.2; revised #302, #306, S11) -- #
#
# Through S10, every launcher above called `synchronize(dev)` right after launching, so
# each device kernel paid a host round-trip before the next one could even be enqueued --
# exactly what `GpuAsync` (`src/utils/backend.jl`) claims not to do. S11 removes that call
# from every launcher in this file: a kernel launch now only enqueues onto the device's own
# command queue and returns, so a chain of them (`D₋ₓ` into `D₋ᵧ` into a sum, say) pipelines
# instead of blocking after each step. Kernels enqueued on the same queue still run in that
# queue's order, so one kernel reading what an earlier one wrote (the entire point of
# chaining operators) needs no synchronisation between them -- only code that leaves the
# queue and touches the array some other way needs a barrier first.
#
# `GpuAsync` is the only `GpuPolicy` that exists today, and no `_launch_*!` here is ever
# reached under anything else, so there is deliberately no policy argument threaded through
# to branch on: adding one now would be conditional logic with nothing to condition on.
# `ka_synchronize` below is that barrier, kept for the two kinds of caller that still need
# one:
#
#   - a genuine host boundary: converting a device array to a host one (`Array(...)`,
#     `host_points`, ...), a host-side reduction or assertion, or anything else that reads
#     the array outside the device's own command queue;
#   - a write that reaches device memory through something other than a `@kernel` launch on
#     that queue -- a plain `copyto!`, which queues a transfer exactly like a kernel launch
#     does but is not itself one of the launches this file just stopped synchronising.
#     `_flush_device_scatter!` and `_zero_stored!` (`src/form/bilinear_traversal.jl`) are
#     this second kind: `_flush_device_scatter!` ends a device-resident matrix's assembly
#     with `copyto!(A.nzVal, mirror.nzval)` and calls `ka_synchronize` right after, exactly
#     as it already did before S11 -- that call was never one of the ones removed above, and
#     it is what keeps `assemble`/`assemble!` from returning before the write lands (S4.2's
#     race, found at `n = 513` over repeated assemblies, invisible at a small `CHECK` size).
#     S11's own check script re-runs that exact shape at n = 513, 1025 and 2049 over 40
#     assemblies each, since removing synchronisation elsewhere is precisely the change that
#     could resurrect it if this file's other launchers were what had been masking it.
#
# A host reduction is a boundary of the first kind without any extra call needed: `_dot`'s
# device method (`src/utils/linear_algebra.jl`) is `sum(u .* v .* w)`, and fetching a
# `GPUArrays` reduction's result to a host scalar already forces the device to finish
# everything queued before it -- `innerₕ`/`normₕ` synchronise by returning a plain number,
# not by this file calling anything.
function ka_synchronize(x::AbstractArray)
    synchronize(get_backend(x))
    return nothing
end

end # module BrambleKernelAbstractionsExt
