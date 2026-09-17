# The contract every sparse direct solver backend satisfies, written once.
#
# suitesparse_ext.jl, mumps_ext.jl, sparspak_ext.jl and appleaccelerate_ext.jl shared one
# skeleton almost verbatim: the same 1D/2D/3D Poisson fixture, the same solve-and-compare
# sequence, the same factorize/ldiv!/refactor! walk, and the same validation testset. What
# actually differs between them is which keyword each entry point accepts -- `sym = :spd`
# for three of them, nothing at all for Sparspak, whose methods take no keywords
# (src/solvers/sparspak_solver.jl:46,81) -- so every entry point arrives here as a CLOSURE
# with those keywords already applied, and nothing below ever asks which backend it is
# running against.
#
# Nothing here names an ext package, and its own dependencies are unconditional. It is
# included from the `ext` group only because nothing else needs it.
#
# Reached as `using ..ExtSolverContracts: ...`, relying on test/runtests.jl including this
# file first -- the same ordering contract autodiff_backends.jl -> autodiff_heavy.jl uses
# (STANDARDS.md, Tests). To run one backend file standalone:
#
#   julia --project=test -e 'using Bramble, Test; include("test/TestUtils.jl");
#     include("test/ext/SolverContracts.jl"); include("test/ext/suitesparse_ext.jl")'
module ExtSolverContracts

using Test
using Bramble
using LinearAlgebra: ldiv!
using SparseArrays: SparseMatrixCSC, spzeros

export ZERO_BC, poisson_system, convection_diffusion_system,
       poisson_solve_contract, refactor_contract, unsymmetric_refactor_contract,
       validation_contract

# Written inline at every call site before, which is one closure type -- and so one
# `assemble` specialization -- per site.
const ZERO_BC = :boundary => (x -> 0.0)

# Capability presence by dispatch rather than by `isnothing` branches in the contract
# bodies: a backend either hands over an entry point or hands over `nothing`.
@inline if_supported(body, ::Nothing) = nothing
@inline if_supported(body, capability) = body(capability)

# ---------------------------------------------------------------------- fixtures

_unit_cube(::Val{D}) where {D} = reduce(×, ntuple(_ -> interval(0.0, 1.0), Val(D)))

_sine_source(::Val{1}) = x -> sin(π * x)
_sine_source(::Val{D}) where {D} = x -> prod(sin(π * xᵢ) for xᵢ in x)

_grid(::Val{1}, Ωd, n) = mesh(Ωd, n, true)
_grid(::Val{D}, Ωd, n) where {D} = mesh(Ωd, ntuple(_ -> n, Val(D)), ntuple(_ -> true, Val(D)))

"""
    poisson_system(Val(D), n; source, symmetrize = true)

`-Δu = f` with homogeneous Dirichlet data on the unit `D`-cube, `n` points per direction,
in the one discretisation all four backend files use. Returns
`(; Wₕ, a, l, A, F, u_ref)`, where `u_ref = A \\ F` is the backslash answer every backend is
compared against -- computed here rather than respelled at each assertion.
"""
function poisson_system(
        dim::Val{D}, n::Integer; source = _sine_source(dim), symmetrize::Bool = true
) where {D}
    Iᴰ = _unit_cube(dim)
    Ωd = domain(Iᴰ, :boundary => boundary_symbols(Iᴰ))
    Wₕ = gridspace(_grid(dim, Ωd, n))
    a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)))
    fₕ = Rₕ(Wₕ, source)
    l = form(Wₕ, v -> innerₕ(fₕ, v))
    A, F = assemble(a, l; dirichlet = ZERO_BC, symmetrize = symmetrize)
    return (; Wₕ = Wₕ, a = a, l = l, A = A, F = F, u_ref = A \ F)
end

"""
    convection_diffusion_system(n; βx = 2.0, βy = 1.0)

`-Δu + β⋅∇u = 1` on the unit square: the genuinely unsymmetric system. `β` is a parameter
because MUMPS runs this at `(5.0, 2.0)` where the others use `(2.0, 1.0)` -- a difference
in the problem, not in what is asserted about it.
"""
function convection_diffusion_system(n::Integer; βx = 2.0, βy = 1.0)
    I2 = interval(0.0, 1.0) × interval(0.0, 1.0)
    Ωd = domain(I2, :boundary => boundary_symbols(I2))
    Wₕ = gridspace(mesh(Ωd, (n, n), (true, true)))
    a = form(
        Wₕ, Wₕ,
        (u, v) -> inner₊(∇ₕ(u), ∇ₕ(v)) + βx * innerₕ(D₊ₓ(u), v) + βy * innerₕ(D₊ᵧ(u), v)
    )
    l = form(Wₕ, v -> innerₕ(Rₕ(Wₕ, x -> 1.0), v))
    A, F = assemble(a, l; dirichlet = ZERO_BC, symmetrize = false)
    return (; Wₕ = Wₕ, a = a, l = l, A = A, F = F, u_ref = A \ F)
end

# ---------------------------------------------------------------------- contracts
#
# None of these open a `@testset`: the caller keeps its own literal title, so the suite's
# reported testset tree is unchanged by the move and every failure still names the backend
# through the testset it is nested in rather than through a loop variable.

"""
    poisson_solve_contract(; atol, solver, facttype, solve, solve_form,
                             factorize_form = nothing, unified_kwargs = (;))

Poisson in 1D, 2D and 3D through every solve entry point: the unified `pde_solve`
dispatcher in 1D and 3D, the backend's own matrix method in 2D, its `(a, l)` form method
(which must answer with a `VectorElement`), and, where the backend has one, its form
factorisation.

The closures carry each backend's own keywords, so `sym` never appears below:
`(A, F) -> suitesparse_solve(A, F; sym = :spd)` against `(A, F) -> sparspak_solve(A, F)`.
`unified_kwargs` is what the *dispatcher* is handed, which is a separate question from what
the backend method accepts -- the dispatcher drops a `sym` Sparspak has no method for.
"""
function poisson_solve_contract(
        ; atol::Real, solver::Symbol, facttype::Type, solve, solve_form,
        factorize_form = nothing, unified_kwargs::NamedTuple = (;)
)
    p1 = poisson_system(Val(1), 21)
    @test isapprox(
        pde_solve(p1.A, p1.F; solver = solver, unified_kwargs...), p1.u_ref; atol = atol
    )

    p2 = poisson_system(Val(2), 12)
    @test isapprox(solve(p2.A, p2.F), p2.u_ref; atol = atol)

    u_elem = solve_form(p2.a, p2.l)
    @test u_elem isa Bramble.VectorElement
    @test isapprox(parent(u_elem), p2.u_ref; atol = atol)

    if_supported(factorize_form) do factorize_a
        fact = factorize_a(p2.a)
        @test fact isa facttype
        @test isapprox(fact \ p2.F, p2.u_ref; atol = atol)
    end

    p3 = poisson_system(Val(3), 6)
    @test isapprox(pde_solve(p3.A, p3.F; solver = solver), p3.u_ref; atol = atol)
    return nothing
end

"""
    refactor_contract(p; atol, solver, facttype, factorize, backend_refactor!,
                        unified_kwargs = (;))

`size`, every `ldiv!` form (including the `VectorElement` destination), a complex
right-hand side, and every route to a numeric refactorisation of a matrix whose
sparsity pattern has not changed: the unique `refactor!` driver, the backend's own
`X_refactor!`, the unified `sparse_factorize`, and the `sparse_refactor!` alias -- plus the
rejection of a dense matrix. `p` comes from `poisson_system`, passed in because the
backends do not agree on which system they run this on.

Deliberately does not refactor this factorisation from a `BilinearForm`: see
`unsymmetric_refactor_contract`.
"""
function refactor_contract(
        p; atol::Real, solver::Symbol, facttype::Type, factorize, backend_refactor!,
        unified_kwargs::NamedTuple = (;)
)
    A, F = p.A, p.F

    fact = factorize(A)
    @test size(fact) == (size(A, 1), size(A, 2))
    @test size(fact, 1) == size(A, 1)

    b = copy(F)
    x = similar(b)
    ldiv!(x, fact, b)
    @test isapprox(x, p.u_ref; atol = atol)

    b_in_place = copy(F)
    ldiv!(fact, b_in_place)
    @test isapprox(b_in_place, x; atol = atol)

    # gpena/Bramble.jl#261: a `VectorElement` destination was ambiguous between Bramble's
    # `ldiv!(::VectorElement, ::Factorization, ::AbstractVector)` and each extension's
    # three-argument `ldiv!` on an `AbstractVector` destination -- neither more specific
    # than the other, since `VectorElement <: AbstractVector` and every concrete
    # factorisation is `<: Factorization`. Asserted here rather than once per backend file,
    # since the ambiguity was the same in all four.
    uₕ = element(p.Wₕ)
    @test ldiv!(uₕ, fact, F) === uₕ
    @test isapprox(parent(uₕ), p.u_ref; atol = atol)

    # gpena/Bramble.jl#261, the same shape on the other operator: a complex right-hand side
    # was ambiguous between each extension's `\(::Factorization, ::AbstractVector)` and
    # `LinearAlgebra`'s `\(::Factorization{T}, ::Vector{Complex{T}})`. The answer is the
    # real solve applied to each part.
    F_complex = complex.(F, 2 .* F)
    u_complex = fact \ F_complex
    @test u_complex isa Vector{ComplexF64}
    @test isapprox(real(u_complex), p.u_ref; atol = atol)
    @test isapprox(imag(u_complex), A \ (2 .* F); atol = atol)

    # same sparsity pattern, different values, through the unique `refactor!` driver
    A_mod = copy(A)
    A_mod[1, 1] += 5.0
    u_mod = A_mod \ F
    refactor!(fact, A_mod)
    @test isapprox(fact \ F, u_mod; atol = atol)

    fact_unified = sparse_factorize(A; solver = solver, unified_kwargs...)
    @test fact_unified isa facttype
    @test isapprox(fact_unified \ F, p.u_ref; atol = atol)
    refactor!(fact_unified, A_mod)
    @test isapprox(fact_unified \ F, u_mod; atol = atol)

    backend_refactor!(fact, A_mod)
    @test isapprox(fact \ F, u_mod; atol = atol)
    sparse_refactor!(fact_unified, A_mod)
    @test isapprox(fact_unified \ F, u_mod; atol = atol)

    @test_throws ArgumentError refactor!(fact, Matrix(A_mod))
    @test_throws ArgumentError sparse_refactor!(fact, Matrix(A_mod))
    return nothing
end

"""
    unsymmetric_refactor_contract(p; atol, factorize)

The `BilinearForm` refactoring routes, on a factorisation that can legally take the matrix
they produce.

This is separate from `refactor_contract` rather than three more lines of it because
`refactor!(fact, a; dirichlet = ...)` assembles *without* symmetrising
(src/solvers/sparse_solvers.jl:100), so the matrix it hands back is not symmetric. Feeding
that to an SPD-flagged CHOLMOD/Accelerate/MUMPS factorisation is not a valid operation,
which is why suitesparse_ext.jl and appleaccelerate_ext.jl only ever did this on their LU
factorisation. `factorize` is therefore the unsymmetric-flagged constructor for the three
backends that have a symmetry flag, and plain `sparspak_factorize` for the one that does
not.
"""
function unsymmetric_refactor_contract(p; atol::Real, factorize)
    A_u, F_u = assemble(p.a, p.l; dirichlet = ZERO_BC, symmetrize = false)
    u_ref_u = A_u \ F_u

    fact = factorize(A_u)
    @test isapprox(fact \ F_u, u_ref_u; atol = atol)

    A_u_mod = copy(A_u)
    A_u_mod[1, 1] += 3.0
    refactor!(fact, A_u_mod)
    @test isapprox(fact \ F_u, A_u_mod \ F_u; atol = atol)

    # straight from the BilinearForm, and through the alias
    refactor!(fact, p.a; dirichlet = ZERO_BC)
    @test isapprox(fact \ F_u, u_ref_u; atol = atol)
    sparse_refactor!(fact, p.a; dirichlet = ZERO_BC)
    @test isapprox(fact \ F_u, u_ref_u; atol = atol)
    return nothing
end

"""
    validation_contract(p; factorize, backend_refactor!, invalid_sym_solve = nothing)

A rectangular matrix, both `ldiv!` dimension mismatches, and a size mismatch in
`X_refactor!`. `invalid_sym_solve` is `nothing` for Sparspak and only for Sparspak: its
entry points take no keywords, so there is no invalid symmetry option for it to reject.
"""
function validation_contract(p; factorize, backend_refactor!, invalid_sym_solve = nothing)
    A, F = p.A, p.F
    n = length(F)

    if_supported(invalid_sym_solve) do bad_solve
        @test_throws ArgumentError bad_solve(A, F)
    end

    @test_throws DimensionMismatch factorize(spzeros(5, 4))

    fact = factorize(A)
    @test_throws DimensionMismatch ldiv!(fact, rand(n + 1))
    @test_throws DimensionMismatch ldiv!(rand(n + 1), fact, F)
    @test_throws DimensionMismatch backend_refactor!(fact, spzeros(n + 1, n + 1))
    return nothing
end

end # module ExtSolverContracts
