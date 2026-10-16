#===========================================================================#
# Ghost spaces against plain spaces on every CPU policy (gpena/Bramble.jl#785, item 4).
#
# Included by `benchmarks.jl` as `SUITE["ghost vs plain"]`, so the per-commit baseline
# carries it and `--compare` reads it like every other key. For each policy (serial,
# threaded, Polyester) and dimension (1D to 3D), on the seeded non-uniform meshes the ghost
# tests use, the same operator `innerₕ(-Δₕ(u), v)` is posed twice:
#
#   - plain: `gridspace(Ωₕ)` with Dirichlet rows, `assemble(a, l; dirichlet = ...)`;
#   - ghost: `gridspace(Ωₕ; ghosts = :boundary)` through `constrained`, Neumann on the
#     minimum facets and Robin on the maximum ones, so the ghost layer carries the
#     boundary data and `Δₕ` reads it.
#
# The group is flat, keyed `<policy> <D>D, <case>`, as the printer and `--compare` expect.
# The timed steps are construction (space, and for the ghost side `constrained`), `form`,
# `assemble`, the warm refills (`assemble!` of the matrix, of the load, and on the ghost
# side the pair: a load alone refuses on a closure with data), `lift` (a warm `Bramble.lift!`) and `pde_solve`. The plain side has no
# lift: its unknowns are the grid function.
#
# The sizes are small enough for the whole group to take a few minutes. Needs
# `using Polyester` (loaded by `benchmarks.jl`) for the `CpuPolyester()` rows.
#===========================================================================#

const _GVP_SIZES = Dict(1 => 100_000, 2 => (300, 300), 3 => (30, 30, 30))
const _GVP_POLICIES = (
    "serial" => () -> Bramble.CpuSerial(),
    "threaded" => () -> Bramble.CpuThreaded(),
    "polyester" => () -> Bramble.CpuPolyester()
)

# A seeded mesh with random (non-uniform) axes, on the policy `pol`.
function _gvp_mesh(D, pol)
    Bramble._seed_mesh1d_rng!(11)
    Ωd = domain(foldl(×, ntuple(d -> interval(0.0, Float64(d)), D)))
    ns = _GVP_SIZES[D]
    Ωₕ = mesh(Ωd, ns, D == 1 ? false : ntuple(_ -> false, D); backend = backend(policy = pol))
    Bramble._unseed_mesh1d_rng!()
    return Ωₕ
end

# Neumann on every minimum facet, Robin on every maximum one.
function _gvp_relations(u)
    (:xmin => ∂ₙ(u) => 0.0, :xmax => u + ∂ₙ(u) => 1.0,
        :ymin => ∂ₙ(u) => 0.0, :ymax => u + ∂ₙ(u) => 1.0,
        :zmin => ∂ₙ(u) => 0.0, :zmax => u + ∂ₙ(u) => 1.0)
end
_gvp_relations(D, u) = Tuple(r for r in _gvp_relations(u)[1:(2D)])
_gvp_constrain(Wg, D) = constrained(Wg, u -> _gvp_relations(D, u))

_gvp_a(W) = form(W, W, (u, v) -> innerₕ(-Δₕ(u), v))
_gvp_l(W) = form(W, v -> innerₕ(x -> 1.0, v))

# Heavy steps take few samples; light ones are left to BenchmarkTools.
function _ghost_vs_plain_group()
    g = BenchmarkGroup()
    for (pname, mkpol) in _GVP_POLICIES, D in 1:3
        Ωₕ = _gvp_mesh(D, mkpol())
        tag = "$pname $(D)D"
        dir = :boundary => x -> 0.0

        # plain
        Wp = gridspace(Ωₕ)
        ap, lp = _gvp_a(Wp), _gvp_l(Wp)
        Ap, Fp = assemble(ap, lp; dirichlet = dir)
        Ap2, Fp2 = copy(Ap), copy(Fp)
        assemble!(Ap2, ap)
        assemble!(Fp2, lp)
        g["$tag, plain construction"] = @benchmarkable gridspace($Ωₕ)
        g["$tag, plain form"] = @benchmarkable _gvp_a($Wp)
        g["$tag, plain assemble"] = @benchmarkable assemble($ap, $lp; dirichlet = $dir) samples=5 evals=1
        g["$tag, plain assemble! matrix"] = @benchmarkable assemble!($Ap2, $ap)
        g["$tag, plain assemble! load"] = @benchmarkable assemble!($Fp2, $lp)
        g["$tag, plain pde_solve"] = @benchmarkable pde_solve($Ap, $Fp) samples=3 evals=1

        # ghost
        Wg = gridspace(Ωₕ; ghosts = :boundary)
        Cg = _gvp_constrain(Wg, D)
        ag, lg = _gvp_a(Cg), _gvp_l(Cg)
        Ag, Fg = assemble(ag, lg)
        Ag2, Fg2 = copy(Ag), copy(Fg)
        assemble!(Ag2, Fg2, ag, lg)
        assemble!(Ag2, ag)
        ug = pde_solve(Ag, Fg)
        uₕ = Bramble.lift(Cg, ug)
        g["$tag, ghost construction"] = @benchmarkable gridspace($Ωₕ; ghosts = :boundary)
        g["$tag, ghost constrained"] = @benchmarkable _gvp_constrain($Wg, $D) samples=5 evals=1
        g["$tag, ghost form"] = @benchmarkable _gvp_a($Cg)
        g["$tag, ghost assemble"] = @benchmarkable assemble($ag, $lg) samples=5 evals=1
        g["$tag, ghost assemble! matrix"] = @benchmarkable assemble!($Ag2, $ag)
        g["$tag, ghost assemble! pair"] = @benchmarkable assemble!($Ag2, $Fg2, $ag, $lg)
        g["$tag, ghost lift"] = @benchmarkable Bramble.lift($Cg, $ug)
        g["$tag, ghost lift!"] = @benchmarkable Bramble.lift!($uₕ, $Cg, $ug)
        g["$tag, ghost pde_solve"] = @benchmarkable pde_solve($Ag, $Fg) samples=3 evals=1
    end
    return g
end
