module QualityAlloccheckTests

using Test
using Bramble
using AllocCheck
using Bramble:
               ×,
               _boundary_symbol_alias,
               _dot,
               _expand_uniform,
               boundary_indices,
               diff₋ₓ!,
               diff₊ₓ!,
               diff₋ᵧ!,
               diff₊ᵧ!,
               diff₋₂!,
               diff₊₂!,
               hₘₐₓ,
               hₘᵢₙ,
               is_collapsed,
               locate_cell,
               normal_vector,
               spacings,
               TrialFunction,
               TestFunction,
               πₕ!

# Static allocation verification (gpena/Bramble.jl#118).
#
# `@test_allocs` in `test/TestUtils.jl` counts bytes for one call with one set of values.
# `check_allocs` compiles a method for a given signature and inspects the IR, so it covers
# every branch of that specialisation rather than the one the sample values took. The two
# are kept side by side rather than one replacing the other, because measurement shows each
# sees what the other misses:
#
#   - Runtime counting sees allocations made by the *caller* -- a boxed capture, a keyword
#     call over a global -- which never appear in the callee's IR. The boxing regression
#     recorded at `test/runtests.jl:34` allocated 23,824 B against 0 B for its fix and drew
#     no report from either AllocCheck or JET.
#   - Static checking reports branches the call never took. Measured here on this tree:
#
#         function               runtime   check_allocs
#         assemble!(A, a)          0 B      24 reports
#         jacobian!(J, sd, ...)    0 B       8 reports
#         set_points!(Ωₕ, pts)     0 B       6 reports
#         copyto!(uₕ, v)           0 B       3 reports
#
#     Each of those is a path the compiled method keeps and the call never enters: for
#     `assemble!`, the first-assembly recording in `src/form/bilinear_execution.jl`, which
#     allocates once by design and is replaced by the replay plan on every call after; for
#     the others, a `Base` copy or resize branch. They are documented at the end of this
#     file rather than asserted, since the guarantee they would break is the runtime one,
#     which is asserted where it belongs.
#
# So what is asserted here is the stronger claim, on the kernels that satisfy it: no
# allocating instruction survives anywhere in the compiled method, for any input of these
# types.

# Names the allocating instructions rather than reporting `false`, so a failure says which
# instruction appeared and where.
function _alloc_report(f, types)
    allocs = check_allocs(f, types)
    isempty(allocs) && return ""
    lines = [first(split(string(a), "\n")) for a in Iterators.take(allocs, 5)]
    return string(length(allocs), " allocating instruction(s): ", join(lines, "; "))
end

@testset "AllocCheck static verification" begin
    # Uniform meshes throughout: `mesh(Ω, n, false)` draws its points with `rand!`, and
    # nothing here reads a coordinate value, so the random stream would only make the
    # setup irreproducible for no gain.
    Ωₕ1 = mesh(domain(interval(0.0, 1.0)), 32, true)
    Ωₕ2 = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (8, 9), (true, true))
    Ωₕ3 = mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (4, 5, 6), (true, true, true))

    Wₕ1, Wₕ2, Wₕ3 = gridspace(Ωₕ1), gridspace(Ωₕ2), gridspace(Ωₕ3)
    Vₕ2 = gridspace(Ωₕ2, Val(2))

    uₕ1 = Rₕ(Wₕ1, sin)
    uₕ2 = Rₕ(Wₕ2, x -> sin(x[1]) * x[2])
    uₕ3 = Rₕ(Wₕ3, x -> sin(x[1]) + x[3])
    cₕ2 = Rₕ(Vₕ2, (x -> x[1], x -> x[2]))
    vₕ1, vₕ2, vₕ3, dₕ2 = similar(uₕ1), similar(uₕ2), similar(uₕ3), similar(cₕ2)

    @testset "Geometry" begin
        X1 = interval(0.0, 1.0)
        X2 = interval(0.0, 1.0) × interval(0.0, 1.0)
        X3 = box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

        @test _alloc_report(interval, (Float64, Float64)) == ""
        @test _alloc_report(point, (Float64,)) == ""
        @test _alloc_report(box, (Tuple{Float64, Float64}, Tuple{Float64, Float64})) == ""

        for (lbl, X) in (("1D", X1), ("2D", X2), ("3D", X3))
            @testset "$lbl" begin
                @test _alloc_report(center, (typeof(X),)) == ""
                @test _alloc_report(extrema, (typeof(X),)) == ""
                @test _alloc_report(dim, (typeof(X),)) == ""
                @test _alloc_report(topo_dim, (typeof(X),)) == ""
                @test _alloc_report(projection, (typeof(X), Int)) == ""
                @test _alloc_report(is_collapsed, (typeof(X), Int)) == ""
            end
        end
    end

    @testset "Uniformity expansion" begin
        @test _alloc_report(_expand_uniform, (Bool, Val{1})) == ""
        @test _alloc_report(_expand_uniform, (Bool, Val{2})) == ""
        @test _alloc_report(_expand_uniform, (Bool, Val{3})) == ""
        @test _alloc_report(_expand_uniform, (NTuple{2, Bool}, Val{2})) == ""
        @test _alloc_report(_expand_uniform, (NTuple{3, Bool}, Val{3})) == ""
    end

    @testset "Boundary symbol aliases" begin
        @test _alloc_report(_boundary_symbol_alias, (Val{1}, Symbol)) == ""
        @test _alloc_report(_boundary_symbol_alias, (Val{2}, Symbol)) == ""
        @test _alloc_report(_boundary_symbol_alias, (Val{3}, Symbol)) == ""
    end

    @testset "Form component operations (gpena/Bramble.jl#153)" begin
        @test _alloc_report(components, (TrialFunction{2, 2},)) == ""
        @test _alloc_report(components, (TestFunction{2, 2},)) == ""
        @test _alloc_report(getindex, (TrialFunction{2, 2}, Int)) == ""
        @test _alloc_report(getindex, (TestFunction{2, 2}, Int)) == ""
    end

    @testset "Mesh queries" begin
        for (lbl, Ωₕ, I) in (
            ("1D", Ωₕ1, Int), ("2D", Ωₕ2, CartesianIndex{2}), ("3D", Ωₕ3, CartesianIndex{3})
        )
            @testset "$lbl" begin
                @test _alloc_report(point, (typeof(Ωₕ), I)) == ""
                @test _alloc_report(spacing, (typeof(Ωₕ), I)) == ""
                @test _alloc_report(half_spacing, (typeof(Ωₕ), I)) == ""
                @test _alloc_report(cell_measure, (typeof(Ωₕ), I)) == ""
                @test _alloc_report(hₘₐₓ, (typeof(Ωₕ),)) == ""
                @test _alloc_report(hₘᵢₙ, (typeof(Ωₕ),)) == ""
                @test _alloc_report(npoints, (typeof(Ωₕ),)) == ""
                @test _alloc_report(spacings, (typeof(Ωₕ),)) == ""
                @test _alloc_report(normal_vector, (typeof(Ωₕ), Symbol)) == ""
                @test _alloc_report(boundary_indices, (typeof(Ωₕ),)) == ""
            end
        end

        @test _alloc_report(locate_cell, (typeof(Ωₕ1), Float64)) == ""
    end

    # Every exported in-place directional operator, on the lowest dimension that defines it.
    # These are the kernels the whole package's zero-allocation promise rests on: each is a
    # bare loop over a stencil, and each has at some point boxed a callable and allocated per
    # grid point (`test/space/inference_allocation.jl` records the instances).
    @testset "In-place directional operators" begin
        for (suffix, dst, src) in (
            ("ₓ", vₕ1, uₕ1), ("ᵧ", vₕ2, uₕ2), ("₂", vₕ3, uₕ3)
        )
            @testset "$suffix" begin
                for stem in (
                    "D₋", "D₊", "Dc", "Dstar₊", "Dₕ", "M₋", "M₊", "diff₋", "diff₊", "jump"
                )
                    op = getfield(Bramble, Symbol(stem, suffix, "!"))
                    @test _alloc_report(op, (typeof(dst), typeof(src))) == ""
                end
            end
        end

        # A composite space routes each component through the same kernel; the routing is
        # where gpena/Bramble.jl#64 put a `Core.Box`.
        @test _alloc_report(D₋ₓ!, (typeof(dₕ2), typeof(cₕ2))) == ""
        @test _alloc_report(M₋ₓ!, (typeof(dₕ2), typeof(cₕ2))) == ""
    end

    @testset "Restriction, averaging and interpolation" begin
        for (lbl, uₕ) in (("1D", uₕ1), ("2D", uₕ2), ("3D", uₕ3))
            @testset "$lbl" begin
                @test _alloc_report(Rₕ!, (typeof(uₕ), typeof(sin))) == ""
                @test _alloc_report(avgₕ!, (typeof(uₕ), typeof(sin))) == ""
            end
        end

        @test _alloc_report(Rₕ!, (typeof(cₕ2), typeof((sin, cos)))) == ""
        @test _alloc_report(copyto!, (typeof(uₕ1), Float64)) == ""

        Wₕ_coarse = gridspace(mesh(domain(interval(0.0, 1.0)), 16, true))
        @test _alloc_report(πₕ!, (typeof(element(Wₕ_coarse)), typeof(uₕ1))) == ""
    end

    @testset "Inner products and norms" begin
        for (lbl, uₕ) in (("1D", uₕ1), ("2D", uₕ2), ("3D", uₕ3))
            @testset "$lbl" begin
                @test _alloc_report(innerₕ, (typeof(uₕ), typeof(uₕ))) == ""
                @test _alloc_report(inner₊, (typeof(uₕ), typeof(uₕ))) == ""
                @test _alloc_report(normₕ, (typeof(uₕ),)) == ""
                @test _alloc_report(snorm₁ₕ, (typeof(uₕ),)) == ""
                @test _alloc_report(norm₁ₕ, (typeof(uₕ),)) == ""
            end
        end

        # A composite space carries `innerₕ`/`normₕ` (block by block) and not the seminorms,
        # so only the two it defines are checked here.
        @testset "composite" begin
            @test _alloc_report(innerₕ, (typeof(cₕ2), typeof(cₕ2))) == ""
            @test _alloc_report(normₕ, (typeof(cₕ2),)) == ""
        end

        @test _alloc_report(_dot, (Vector{Float64}, Vector{Float64}, Vector{Float64})) == ""
    end

    @testset "Forms and semidiscretisation" begin
        l = form(Wₕ1, v -> innerₕ(uₕ1, v))
        a = form(Wₕ1, Wₕ1, (u, v) -> innerₕ(u, v) + inner₊ₓ(D₋ₓ(u), D₋ₓ(v)))
        b = assemble(l)
        scratch = zeros(ndofs(Wₕ1))

        @test _alloc_report(assemble!, (typeof(b), typeof(l))) == ""
        @test _alloc_report(Bramble.evaluate!, (typeof(scratch), typeof(l), typeof(uₕ1))) ==
              ""

        # The residual a time integrator calls once per stage, on the matching element type.
        # `src/form/semidiscrete.jl` documents this as 0 bytes; here it is the stronger
        # statement, that no branch of that specialisation can allocate at all.
        sd = semidiscretize(a, l; dirichlet = :boundary)
        u = collect(range(0.25, 1.75; length = ndofs(Wₕ1)))
        du = zeros(ndofs(Wₕ1))
        @test _alloc_report(sd, (typeof(du), typeof(u), Nothing, Float64)) == ""
    end

    # Cold-branch reports, kept as a record of what static checking says about paths whose
    # runtime cost is measured at 0 B elsewhere in the suite. They are not assertions: the
    # counts belong to `Base` internals and will move with the Julia version. What is
    # asserted about these calls is the runtime figure, in the files that own them
    # (`test/form/interpolation_operator.jl`, `test/form/semidiscrete.jl`,
    # `test/mesh/inference_allocation.jl`, `test/space/inference_allocation.jl`).
    #
    #   assemble!(A, a)   24 reports   first-assembly recording (`_try_diagonal_segment`)
    #   jacobian!(...)     8 reports   `Base` vector allocation outside the refresh path
    #   set_points!(...)   6 reports   `Base` copy branch
    #   copyto!(uₕ, v)     3 reports   `Base` copy branch
    #
    # Two more allocate at runtime as well, by design, and so are not candidates at all:
    # `assemble_parallel!` (1,632 B of task-spawn machinery) and `iterative_refinement!`
    # (5,472 B, which builds a new marker dictionary).
end

end # module QualityAlloccheckTests
