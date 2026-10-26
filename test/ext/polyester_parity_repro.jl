# test/ext/polyester_parity_repro.jl: a deterministic reproduction of gpena/Bramble.jl#942,
# CpuPolyester's 1D reductions differing from CpuThreaded's in the last bit. The cause is
# in docs/notes/polyester-parity-942.md.
#
#   julia --project=test -t 4 -e 'using Bramble, Polyester;
#     include("test/ext/polyester_parity_repro.jl"); @show Repro942.run(20)'
#
# The `@simd` band body the reductions share (`_separable_line_band`) has two compiled
# specializations: on `Vector`s, run by `CpuThreaded` and by a nested `CpuPolyester` call,
# cached in Bramble's package image; and on the `PtrArray`s `@batch` makes, run by a
# top-level `CpuPolyester` call, cached in `BramblePolyesterExt`'s image. A process at
# `--optimize=1` (the ext group) loads the newest image built at its level or above, so it
# can run an `-O2` Bramble image, vectorized, beside an `-O1` extension image, not.
#
# `run` builds that state in a fresh depot and runs the reductions in a child process at
# the ext group's flags (`--optimize=1`, this process's thread count), whatever this
# process's own flags: a child at `-O2` finds or builds an `-O2` Bramble image, which is
# copied into the depot so that it is the newest; a child at `-O1` compiles the extension's
# image into it at `-O1`. The images the last child loaded are checked, so the state cannot
# silently differ. Every child gets the flags `.claude/scripts/test.sh` gives the tests
# (`_FLAGS`), never this process's bounds checks, coverage or debug level, which would ask
# for other images. Loading an image touches it, which is what decides the next process's
# choice, so `run` puts back the modification times of every Bramble and extension image
# in the depots it found. The depot is deleted before `run` returns. A call takes about a
# minute: the extension's image is compiled every time.
module Repro942

using Bramble

# `@batch` converts the arrays to `PtrArray`s only when it finds a free worker.
const NTHREADS = 2

const _EXT = "BramblePolyesterExt"

# Prints a line `IMAGE <name> <opt level> <path>` for each of the two packages' images.
const _IMAGES = """
function _report_images()
    for (k, v) in Base.pkgorigins
        k.name in ("Bramble", "$_EXT") || continue
        io = open(v.cachepath)
        Base.isvalid_cache_header(io)
        flags = Base.parse_cache_header(io, v.cachepath)[end]
        close(io)
        println("IMAGE ", k.name, " ", Base.CacheFlags(flags).opt_level, " ", v.cachepath)
    end
end
"""

# The three reductions #942 saw fail, on the non-uniform (1001,) grid, `n` times.
const _REDUCE = """
using Bramble, Polyester, Random
using Bramble: CpuPolyester, CpuThreaded
$_IMAGES
function _space(policy)
    X = interval(0.0, 1.0)
    Ω = domain(X, markers(X, :a => x -> x[1] < 0.6, :b => x -> x[end] > 0.3))
    Random.seed!(473)
    return gridspace(mesh(Ω, (1001,), false; backend = backend(policy = policy)))
end
_f(x) = x[1] + sum(x) / 3 + x[1] * x[end] / 7
const REDUCTIONS = (u -> innerₕ(u, u), u -> normₕ(u), u -> inner₊(u, u, 1))
ut = Rₕ(_space(CpuThreaded()), _f)
up = Rₕ(_space(CpuPolyester()), _f)
parent(up) == parent(ut) || error("the two spaces' data differ")
ts = map(r -> r(ut), REDUCTIONS)
n = parse(Int, ARGS[1])
mismatches = count(_ -> map(r -> r(up), REDUCTIONS) != ts, 1:n)
_report_images()
println("RESULT ", n, " ", mismatches)
"""

# The flags of the tests' process under `test.sh` (`Pkg.test` with `--threads=4
# --startup-file=no --optimize=1`), bar the optimisation level and the threads, spelled out
# at their defaults so that none is inherited from this process.
const _FLAGS = `--startup-file=no --check-bounds=auto --code-coverage=none
                --track-allocation=none --debug-info=1 --inline=yes --pkgimages=yes`

function _child(code, depot, opt, threads, args...)
    exe = joinpath(Sys.BINDIR, Base.julia_exename())
    cmd = `$exe $_FLAGS --project=$(Base.active_project()) --optimize=$opt
           --threads=$threads -e $code $args`
    env = ("JULIA_DEPOT_PATH" => join([depot; DEPOT_PATH], ':'),)
    log = joinpath(depot, "child.log")
    try
        return readchomp(pipeline(addenv(cmd, env...); stderr = log))
    catch e
        e isa ProcessFailedException || rethrow()
        error("Repro942: a child at -O$opt failed:\n", read(log, String))
    end
end

function _images(out)
    lines = filter(startswith("IMAGE "), split(out, '\n'))
    return Dict(String(f[2]) => (parse(Int, f[3]), String(f[4]))
    for f in split.(lines, ' '; limit = 4))
end

_compiled(depot, name) = joinpath(depot, "compiled", "v$(VERSION.major).$(VERSION.minor)", name)

# Copies an image (`.ji` and its library) into `depot`, which makes it the newest. An image
# the child built there itself (no `-O2` image existed) is only touched.
function _install(path, depot, name)
    dir = mkpath(_compiled(depot, name))
    lib = Base.ocachefile_from_cachefile(path)
    if dirname(realpath(path)) == realpath(dir)
        touch(lib)
        touch(path)
    else
        cp(lib, joinpath(dir, basename(lib)); force = true)
        cp(path, joinpath(dir, basename(path)); force = true)
    end
    return nothing
end

# The access and modification times, to the nanosecond, of every Bramble and extension
# image file in `depots`. `jl_stat` fills a libuv `uv_stat_t`: twelve 64-bit fields, then
# `st_atim` and `st_mtim` as (seconds, nanoseconds) pairs. Only the low 32 bits of a
# nanoseconds word hold the value (on macOS the high half read 1, not 0), so they are
# masked.
_ns(w::Int64) = Int64(w % UInt32)

function _image_times(depots)
    times = Dict{String, NTuple{4, Int64}}()
    buf = zeros(Int64, ccall(:jl_sizeof_stat, Int32, ()) ÷ 8)
    for d in depots, name in ("Bramble", _EXT)

        dir = _compiled(d, name)
        isdir(dir) || continue
        for f in readdir(dir; join = true)
            isfile(f) || continue
            ccall(:jl_stat, Int32, (Cstring, Ptr{Int64}), f, buf) == 0 || continue
            times[f] = (buf[13], _ns(buf[14]), buf[15], _ns(buf[16]))
        end
    end
    return times
end

# Puts the times back with `utimensat`, whose two `timespec`s are (seconds, nanoseconds).
function _restore_times(times)
    at_fdcwd = Sys.isapple() ? Cint(-2) : Cint(-100)
    for (f, t) in times
        isfile(f) || continue
        ts = collect(t)
        ccall(:utimensat, Cint, (Cint, Cstring, Ptr{Int64}, Cint), at_fdcwd, f, ts, 0) == 0 ||
            @warn "Repro942: could not restore the times of $f" Libc.errno()
    end
    return nothing
end

"""
    run(n::Integer) -> (; ran, mismatches)

Runs the three `(1001,)` reductions `n` times under `CpuPolyester` in a child process that
loads an `-O2` Bramble image and an `-O1` `BramblePolyesterExt` image, and counts the runs in
which any of them differs from `CpuThreaded`'s value. Throws an `ArgumentError` below
`NTHREADS` threads, and an `ErrorException` if the child did not load those images.
"""
function run(n::Integer)
    Threads.nthreads() >= NTHREADS || throw(ArgumentError(
        "Repro942 needs $NTHREADS threads, this process has $(Threads.nthreads())"))
    depot = mktempdir()
    times = _image_times(DEPOT_PATH)
    try
        # An `-O2` Bramble image, copied in so that it is the newest one at any level.
        o2 = _images(_child("using Bramble; $_IMAGES; _report_images()", depot, 2, 1))
        _install(o2["Bramble"][2], depot, "Bramble")
        # The extension's image, compiled at `-O1` against that Bramble image.
        _child("""
            using Bramble, Polyester
            k = only(k for k in keys(Base.pkgorigins) if k.name == "$_EXT")
            Base.compilecache(k, Base.pkgorigins[k].path)
            """, depot, 1, 1)
        out = _child(_REDUCE, depot, 1, Threads.nthreads(), string(n))
        img = _images(out)
        get(img, "Bramble", (0, ""))[1] >= 2 && get(img, _EXT, (0, ""))[1] == 1 ||
            error("Repro942: the child loaded $img, not an -O2 Bramble image and an -O1 \
                   $_EXT image")
        f = split(only(filter(startswith("RESULT "), split(out, '\n'))))
        return (; ran = parse(Int, f[2]), mismatches = parse(Int, f[3]))
    finally
        _restore_times(times)
        rm(depot; recursive = true, force = true)
    end
end

end # module Repro942
