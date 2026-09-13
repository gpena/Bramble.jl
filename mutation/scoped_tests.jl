#===========================================================================#
# Scoped test entrypoint for mutation testing (issue #121).
#
# `test/runtests.jl` in full takes ~6 minutes even with the Quality group
# skipped — far too slow to run once per mutant across hundreds of mutation
# sites. This file includes only the test files that directly exercise the
# four kernels mutation-tested here:
#
#   src/form/operators/difference.jl      → grad_backward, grad_forward,
#                                            Dcₕ, Dstar₊ₕ, ∇ₕ
#   src/space/operators/difference.jl     → forward_star_difference,
#                                            centered_difference,
#                                            cross_weighted_difference
#   src/space/operators/cell_average.jl   → avgₕ, avgₕ!
#   src/form/dirichlet_constraints.jl     → dirichlet_constraints,
#                                            dirichlet_bc!, symmetrize!
#
# A mutation site outside what these files cover is reported `no_coverage`
# by Gremlins — informational, not a false negative: it does not count for
# or against the mutation score, it flags where coverage would need
# broadening (e.g. into test/form/{linear,bilinear}.jl, which exercise the
# same kernels through full form assembly) to get a verdict on that site.
#
# Kept in sync by hand with test/runtests.jl's helper preamble — only the
# helpers the included files actually call are copied here.
#===========================================================================#

if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    test_dir = @__DIR__
    bramble_dir = abspath(joinpath(test_dir, "../"))
    Pkg.activate(joinpath(bramble_dir, "test"))
    Pkg.develop(; path = bramble_dir)
    Pkg.instantiate()
end

using Test
using Bramble
using Bramble: set, cell_measures
using ForwardDiff

@inline function alloc_test(f::F, args...; kwargs...) where {F}
    f(args...; kwargs...)
    return @allocated(f(args...; kwargs...))
end

macro test_allocs(call_expr)
    if Meta.isexpr(call_expr, :call)
        fn = call_expr.args[1]
        args = call_expr.args[2:end]
        quote
            @test alloc_test($(esc(fn)), $(map(esc, args)...)) == 0
        end
    else
        quote
            let
                $(esc(call_expr))
                @test (@allocated $(esc(call_expr))) == 0
            end
        end
    end
end

_fd(f, a; h = 1e-6) = (f(a + h) - f(a - h)) / (2h)

function _nonuniform_points(h::AbstractVector{<:Real})
    pts = zeros(Float64, length(h) + 1)
    for (i, hᵢ) in enumerate(h)
        pts[i + 1] = pts[i] + hᵢ
    end
    pts ./= pts[end]
    return pts
end

function _zero_boundary!(a::AbstractArray{<:Real, N}) where {N}
    for d in 1:N
        selectdim(a, d, 1) .= 0
        selectdim(a, d, size(a, d)) .= 0
    end
    return a
end

function _matches_fd(f, a = 1.3; rtol = 1e-5)
    return isapprox(ForwardDiff.derivative(f, a), _fd(f, a); rtol = rtol)
end

@testset verbose=true "Mutation-scoped kernels" begin
    @testset "Grid spaces" begin
        # Pulls in `MockGPUVector` and Bramble's own `vector` accessor, both of which
        # test/space/difference.jl and test/space/gridspaces.jl assume are already in
        # scope — true in the full suite (test/utils/backends.jl runs earlier under
        # "Core library" > "Utilities"), not true when this file is scoped down.
        include(joinpath(@__DIR__, "..", "test", "utils", "backends.jl"))
        include(joinpath(@__DIR__, "..", "test", "space", "gridspaces.jl"))
        include(joinpath(@__DIR__, "..", "test", "space", "vector_elements.jl"))
    end

    @testset "Difference operators (space)" begin
        include(joinpath(@__DIR__, "..", "test", "space", "difference.jl"))
        include(joinpath(@__DIR__, "..", "test", "space", "star_difference.jl"))
        include(joinpath(@__DIR__, "..", "test", "space", "centered_difference.jl"))
        include(joinpath(@__DIR__, "..", "test", "space", "cross_weighted_difference.jl"))
        include(joinpath(@__DIR__, "..", "test", "space", "sbp_identities.jl"))
        include(joinpath(@__DIR__, "..", "test", "space", "element_type.jl"))
        include(joinpath(@__DIR__, "..", "test", "space", "autodiff.jl"))
        include(joinpath(@__DIR__, "..", "test", "space", "inference_allocation.jl"))
    end

    @testset "Difference operators (form) and Dirichlet constraints" begin
        include(joinpath(@__DIR__, "..", "test", "form", "difference_ast.jl"))
        include(joinpath(@__DIR__, "..", "test", "form", "extended_operators.jl"))
        include(joinpath(@__DIR__, "..", "test", "form", "dirichlet_constraints.jl"))
    end
end
