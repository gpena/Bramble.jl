# The problems test/form/two_spaces.jl runs over a plain space and over the ghost space of
# the same problem, and test/convergence/two_spaces.jl refines. Loaded into `Main` by both,
# never by the package.
module TwoSpacesCases

using Bramble
import Bramble

# The manufactured solution and its second derivatives, one per axis.
function manufactured(D)
    D == 1 && return (u = p -> sin(p[1]) + p[1]^2, du = p -> (cos(p[1]) + 2p[1],),
        d2u = p -> (-sin(p[1]) + 2,))
    return (u = p -> sin(p[1]) * cos(p[2]) + p[1] * p[2],
        du = p -> (cos(p[1]) * cos(p[2]) + p[2], -sin(p[1]) * sin(p[2]) + p[1]),
        d2u = p -> (-sin(p[1]) * cos(p[2]), -sin(p[1]) * cos(p[2])))
end

# `α u + β ∂ₙ(u) = r` on `facet`, with `r` from the manufactured solution.
function robin_on(ex, facet, α, β)
    s = String(facet)
    d = findfirst(==(s[1]), "xyz")
    η = endswith(s, "min") ? -1.0 : 1.0
    return (facet, α, β, p -> α * ex.u(p) + β * η * ex.du(p)[d])
end

function problem_data(D; mixed)
    ex = manufactured(D)
    f = p -> -sum(ex.d2u(p))
    h = (F, p) -> ex.d2u(p)[findfirst(==(String(F)[1]), "xyz")]
    mixed || return (; f, g = ex.u, h, dirichlet = D == 1 ? (:xmin, :xmax) :
                                                   (:xmin, :xmax, :ymin, :ymax))
    D == 1 && return (; f, g = ex.u, h, dirichlet = (:xmin,),
        robin = (robin_on(ex, :xmax, 2.0, 1.0),))
    return (; f, g = ex.u, h, dirichlet = (:xmin, :ymax),
        robin = (robin_on(ex, :xmax, 2.0, 1.0), robin_on(ex, :ymin, 0.0, 1.0)))
end

# A seeded random mesh of `n0` points per axis on the unit box, halved `level` times.
function random_mesh(D, n0, level; seed = 11)
    Bramble._seed_mesh1d_rng!(seed)
    Ω = try
        D == 1 ? mesh(domain(interval(0.0, 1.0)), n0, false) :
        mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (n0, n0), (false, false))
    finally
        Bramble._unseed_mesh1d_rng!()
    end
    foreach(_ -> iterative_refinement!(Ω), 1:level)
    return Ω
end

mesh_axes(Ω) = Bramble.points(Ω) isa Tuple ? Bramble.points(Ω) : (Bramble.points(Ω),)

const PROBLEMS = [
    (name = "1D Dirichlet", D = 1, n0 = 11, mixed = false),
    (name = "1D Dirichlet and Robin", D = 1, n0 = 11, mixed = true),
    (name = "2D Dirichlet", D = 2, n0 = 6, mixed = false),
    (name = "2D Dirichlet, Robin and Neumann", D = 2, n0 = 6, mixed = true)
]

# The solution at every node, mesh order, by `pde_solve`: the plain system carries its
# Dirichlet nodes, the ghost one is lifted by `Bramble.lift`.
function node_solution(c)
    if c.route === :rows
        return pde_solve(assemble(c.a, c.l; dirichlet = c.pairs, symmetrize = true)...)
    end
    ub = pde_solve(assemble(c.a, c.l)...)
    uh = parent(Bramble.lift(c.space, ub))
    G = Bramble.mesh(c.space)
    plin = LinearIndices(Bramble.npoints(G, Tuple))
    return vec([uh[plin[Bramble.padded_index(G, I)]] for I in CartesianIndices(c.dims)])
end

end # module TwoSpacesCases
