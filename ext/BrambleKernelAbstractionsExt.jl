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
                _launch_uniform_points!,
                _launch_half_points!,
                _launch_spacing!,
                _launch_half_spacing!,
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
                _launch_average_engine!

# ---------------------------------------------------------------------------
# `_points!` uniform branch (src/mesh/mesh1d.jl:322-324 is the CPU original)
# ---------------------------------------------------------------------------

@kernel function _uniform_points_kernel!(x, a, h)
    i = @index(Global)
    @inbounds x[i] = a + (i - 1) * h
end

function _launch_uniform_points!(x::AbstractVector, a, h, dev)
    _uniform_points_kernel!(dev)(x, a, h; ndrange = length(x))
    synchronize(dev)
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
    synchronize(dev)
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
    synchronize(dev)
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
    synchronize(dev)
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
    synchronize(dev)
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
        synchronize(dev)
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
        synchronize(dev)
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
        synchronize(dev)
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
        synchronize(dev)
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
        synchronize(dev)
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
        synchronize(dev)
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
        synchronize(dev)
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
        synchronize(dev)
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
        synchronize(dev)
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
        synchronize(dev)
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
# Every kernel calls the very `Bramble._compute_difference`/`Bramble._compute_average`
# methods the CPU sweep calls, once per grid point, so the device and host answers stay
# identical by construction -- the same reasoning `avgₕ!`'s kernels above rely on for
# `_cell_average`. `h` arrives already resolved to either `nothing` or a plain top-level
# array by `Bramble._resolve_device_spacing` (`difference.jl`): never the `StarSpacings`
# wrapper, which -- like any struct nesting a device array -- fails kernel compilation.
#
# `D >= 1` in one kernel each, following this module's own guidance for the `D >= 2`
# restriction/cell-average kernels above: `dims` is a bits `NTuple{D,Int}`, and
# `@index(Global, Cartesian)` gives the `CartesianIndex{D}` the CPU engine's own
# `CartesianIndices(dims)` loop walks, so one kernel body serves every dimension the mesh
# has rather than a 1D/nD split.
# ---------------------------------------------------------------------------

@kernel function _difference_onesided_kernel!(
        out, @Const(in_ref), h, dims::NTuple{D, Int}, dir, ::Val{DIM}
) where {D, DIM}
    I = @index(Global, Cartesian)
    li = LinearIndices(dims)
    n = dims[DIM]
    @inbounds begin
        idx = li[I]
        if I[DIM] == Bramble._stencil_boundary_dim(dir, n)
            out[idx] = Bramble._compute_difference(dir, Val(true), in_ref[idx], h, I[DIM])
        else
            step = Bramble._stencil_step(Val(DIM), Val(D))
            other = li[Bramble._neighbour(dir, I, step)]
            out[idx] = Bramble._compute_difference(dir, Val(false), in_ref[idx], in_ref[other], h, I[DIM])
        end
    end
end

function _launch_difference_onesided!(out::AbstractVector, in_ref, h, dims::Tuple, dir, dim_val::Val, dev)
    try
        _difference_onesided_kernel!(dev)(out, in_ref, h, dims, dir, dim_val; ndrange = dims)
        synchronize(dev)
    catch err
        _wrap_device_kernel_error(err, "difference operator")
    end
    return nothing
end

@kernel function _difference_centered_kernel!(
        out, @Const(in_ref), h, dims::NTuple{D, Int}, dir, ::Val{DIM}
) where {D, DIM}
    I = @index(Global, Cartesian)
    li = LinearIndices(dims)
    n = dims[DIM]
    i = I[DIM]
    step = Bramble._stencil_step(Val(DIM), Val(D))
    @inbounds begin
        idx = li[I]
        if i == 1
            fwd = li[I + step]
            out[idx] = Bramble._compute_difference(dir, Val(true), in_ref[idx], in_ref[fwd], h, i)
        elseif i == n
            back = li[I - step]
            out[idx] = Bramble._compute_difference(dir, Val(true), in_ref[idx], in_ref[back], h, i)
        else
            back = li[I - step]
            fwd = li[I + step]
            out[idx] = Bramble._compute_difference(dir, Val(false), in_ref[back], in_ref[idx], in_ref[fwd], h, i)
        end
    end
end

function _launch_difference_centered!(out::AbstractVector, in_ref, h, dims::Tuple, dir, dim_val::Val, dev)
    try
        _difference_centered_kernel!(dev)(out, in_ref, h, dims, dir, dim_val; ndrange = dims)
        synchronize(dev)
    catch err
        _wrap_device_kernel_error(err, "centered difference operator")
    end
    return nothing
end

@kernel function _average_kernel!(out, @Const(in_ref), dims::NTuple{D, Int}, dir, ::Val{DIM}) where {D, DIM}
    I = @index(Global, Cartesian)
    li = LinearIndices(dims)
    n = dims[DIM]
    @inbounds begin
        idx = li[I]
        if I[DIM] == Bramble._stencil_boundary_dim(dir, n)
            out[idx] = Bramble._compute_average(dir, Val(true), in_ref[idx])
        else
            step = Bramble._stencil_step(Val(DIM), Val(D))
            other = li[Bramble._neighbour(dir, I, step)]
            out[idx] = Bramble._compute_average(dir, Val(false), in_ref[idx], in_ref[other])
        end
    end
end

function _launch_average_engine!(out::AbstractVector, in_ref, dims::Tuple, dir, dim_val::Val, dev)
    try
        _average_kernel!(dev)(out, in_ref, dims, dir, dim_val; ndrange = dims)
        synchronize(dev)
    catch err
        _wrap_device_kernel_error(err, "average operator")
    end
    return nothing
end

end # module BrambleKernelAbstractionsExt
