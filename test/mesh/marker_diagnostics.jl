module MarkerDiagnosticsTests

using Test
using Bramble
using Bramble: index_in_marker, markers
using ..TestUtils: @test_allocs

@testset "Marker Predicate and Label Error Diagnostics (#224)" begin
    @testset "Non-Bool Predicate Validation at Domain Construction" begin
        # 1. Float-returning level-set predicate
        err_float = try
            domain(interval(0.0, 1.0) × interval(0.0, 1.0), :empty => x -> x[1] - 42.0)
            nothing
        catch e
            e
        end
        @test err_float isa ArgumentError
        msg_float = sprint(showerror, err_float)
        @test occursin("Marker predicate for label :empty", msg_float)
        @test occursin("Float64", msg_float)
        @test occursin("expected a Bool", msg_float)
        @test occursin("x -> predicate(x) <= 0", msg_float)

        # 2. Integer-returning predicate
        err_int = try
            domain(interval(0.0, 1.0), :bad_int => x -> 42)
            nothing
        catch e
            e
        end
        @test err_int isa ArgumentError
        msg_int = sprint(showerror, err_int)
        @test occursin("Marker predicate for label :bad_int", msg_int)
        @test occursin("Int", msg_int)
        @test occursin("expected a Bool", msg_int)

        # 3. String-returning predicate
        err_str = try
            domain(interval(0.0, 1.0), :bad_str => x -> "boundary")
            nothing
        catch e
            e
        end
        @test err_str isa ArgumentError
        msg_str = sprint(showerror, err_str)
        @test occursin("Marker predicate for label :bad_str", msg_str)
        @test occursin("String", msg_str)

        # 4. Valid Bool predicate works without error
        dom_valid = domain(interval(0.0, 1.0), :valid => x -> x[1] < 0.5)
        @test :valid in labels(dom_valid)
    end

    @testset "Misspelled Boundary Symbols in Domain Construction" begin
        # 1. Typo in 1D boundary symbol
        err_1d = try
            domain(interval(0.0, 1.0), :inlet => :lefft)
            nothing
        catch e
            e
        end
        @test err_1d isa ArgumentError
        msg_1d = sprint(showerror, err_1d)
        @test occursin("Unknown boundary symbol :lefft for marker :inlet", msg_1d)
        @test occursin(":left", msg_1d)
        @test occursin(":right", msg_1d)
        @test occursin(":xmin", msg_1d)
        @test occursin(":xmax", msg_1d)

        # 2. Typo in 2D tuple boundary symbols
        err_2d = try
            domain(interval(0.0, 1.0) × interval(0.0, 1.0), :walls => (:top, :botttom))
            nothing
        catch e
            e
        end
        @test err_2d isa ArgumentError
        msg_2d = sprint(showerror, err_2d)
        @test occursin("Unknown boundary symbol :botttom for marker :walls", msg_2d)
        @test occursin(":bottom", msg_2d)
        @test occursin(":top", msg_2d)
    end

    @testset "Informative Label Lookup Error in index_in_marker" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0), :inlet => :left), 11)
        err = try
            index_in_marker(Ωₕ, :lefft)
            nothing
        catch e
            e
        end
        @test err isa KeyError
        msg = sprint(showerror, err)
        @test occursin("key :lefft not found", msg)
        @test occursin("Available marker labels on this mesh are:", msg)
        @test occursin(":boundary", msg)
        @test occursin(":inlet", msg)
        @test occursin(":interior", msg)
    end

    @testset "index_in_marker single-probe lookup (#335)" begin
        S = domain(interval(0.0, 1.0) × interval(0.0, 1.0), :inlet => :left, :left => :left)
        Ωₕ = mesh(S, (6, 5), (true, true))

        direct = index_in_marker(Ωₕ, :left)
        @test direct === markers(Ωₕ)[:left]

        alias = index_in_marker(Ωₕ, :xmin)
        @test alias === direct

        @test_allocs index_in_marker(Ωₕ, :left)
        @test_allocs index_in_marker(Ωₕ, :xmin)

        err = try
            index_in_marker(Ωₕ, :lefft)
            nothing
        catch e
            e
        end
        @test err isa KeyError
        msg = sprint(showerror, err)
        @test occursin("key :lefft not found", msg)
        @test occursin("Available marker labels on this mesh are:", msg)
    end

    @testset "Informative Diagnostics in dirichlet_bc! and dirichlet_constraints" begin
        Ω = domain(interval(0.0, 1.0), :inlet => :left, :outlet => :right)
        Ωₕ = mesh(Ω, 11)
        Wₕ = gridspace(Ωₕ)
        A = zeros(11, 11)

        # 1. dirichlet_bc! with misspelled label
        err_bc = try
            dirichlet_bc!(A, Ωₕ, :lefft)
            nothing
        catch e
            e
        end
        @test err_bc isa KeyError
        msg_bc = sprint(showerror, err_bc)
        @test occursin("key :lefft not found", msg_bc)
        @test occursin("Available marker labels on this mesh are:", msg_bc)
        @test occursin(":inlet", msg_bc)
        @test occursin(":outlet", msg_bc)

        # 2. dirichlet_constraints with misspelled label
        err_dc = try
            dirichlet_constraints(Ωₕ, :lefft => x -> 0.0)
            nothing
        catch e
            e
        end
        @test err_dc isa ArgumentError
        msg_dc = sprint(showerror, err_dc)
        @test occursin("dirichlet_constraints: label `:lefft` is not registered", msg_dc)
        @test occursin("Known labels:", msg_dc)
        @test occursin("inlet", msg_dc)
        @test occursin("outlet", msg_dc)
    end

    @testset "Informative Diagnostics in innerₕ and Form Assembly Markers" begin
        Ω = domain(interval(0.0, 1.0), :inlet => :left)
        Ωₕ = mesh(Ω, 11)
        Wₕ = gridspace(Ωₕ)
        uₕ = Rₕ(Wₕ, x -> 1.0)

        # 1. innerₕ with nonexistent marker keyword
        err_inner = try
            innerₕ(uₕ, uₕ; markers = (:unknown_label,))
            nothing
        catch e
            e
        end
        @test err_inner isa KeyError
        msg_inner = sprint(showerror, err_inner)
        @test occursin("key :unknown_label not found", msg_inner)
        @test occursin("Available marker labels on this mesh are:", msg_inner)
        @test occursin(":inlet", msg_inner)

        # 2. Form assembly with nonexistent marker keyword
        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v; markers = (:unknown_label,)))
        err_form = try
            assemble(a)
            nothing
        catch e
            e
        end
        @test err_form isa ArgumentError
        msg_form = sprint(showerror, err_form)
        @test occursin("the marker :unknown_label is not defined on the form's space", msg_form)
        @test occursin("Available marker labels on this space are:", msg_form)
        @test occursin(":inlet", msg_form)
    end
end

end # module
