module FormExpressionTests

using Test
using Bramble
using Bramble: restrict_to, dirac, D₋ᵧ, D₋₂, D₋ₓ, Mₓ, jumpₓ

# `expression(form)`/`expression(ast::LazyOp)` (src/ast/expression.jl, issue #274) render a
# resolved form AST in Bramble's own operator notation. Every string asserted below was
# observed by actually running the corresponding `expression(...)` call in a REPL, not
# guessed from the rendering conventions alone.

@testset "Expression rendering (#274)" begin
    @testset "1D scalar form" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 5)
        Wₕ = gridspace(Ωₕ)

        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(D₋ₓ(u), D₋ₓ(v)))
        @test expression(a) == "innerₕ(D₋ₓ(u), D₋ₓ(v))"

        # a literal number coefficient, the way test/form/linear.jl's "Source variants"
        # testset builds a linear form's source term
        l = form(Wₕ, v -> innerₕ(3.0, v))
        @test expression(l) == "innerₕ(3, v)"
    end

    @testset "2D scalar form" begin
        # A Laplacian-like sum: two BilinearProducts of different node families
        # (D₋ₓ/D₋ᵧ) combined through OperatorAdd.
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (5, 5), (true, true))
        Wₕ = gridspace(Ωₕ)

        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(D₋ₓ(u), D₋ₓ(v)) + innerₕ(D₋ᵧ(u), D₋ᵧ(v)))
        @test expression(a) == "innerₕ(D₋ₓ(u), D₋ₓ(v)) + innerₕ(D₋ᵧ(u), D₋ᵧ(v))"
    end

    @testset "3D scalar form" begin
        # The third spatial subscript renders as `₂`, not `z`/`₃`
        # (`Bramble._BRAMBLE_var2symbol = ("ₓ", "ᵧ", "₂")`).
        Ωₕ = mesh(
            domain(interval(0.0, 1.0) × interval(0.0, 1.0) × interval(0.0, 1.0)),
            (4, 4, 4),
            (true, true, true)
        )
        Wₕ = gridspace(Ωₕ)

        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(D₋ₓ(u), D₋ₓ(v)) + innerₕ(D₋₂(u), D₋₂(v)))
        @test expression(a) == "innerₕ(D₋ₓ(u), D₋ₓ(v)) + innerₕ(D₋₂(u), D₋₂(v))"
    end

    @testset "Composite space, indexed components" begin
        # Same composite-space spelling as test/form/symmetry.jl's "Composite space,
        # indexed components" testset: `u(1)`/`v(1)` dispatch to IndexedTrialFunction/
        # IndexedTestFunction.
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 5)
        Wₕ = gridspace(Ωₕ)
        Vₕ = Wₕ × Wₕ

        f = form(Vₕ, Vₕ, (u, v) -> innerₕ(u(1), v(1)) + innerₕ(u(2), v(2)))
        @test expression(f) == "innerₕ(u(1), v(1)) + innerₕ(u(2), v(2))"
    end

    @testset "Mixed operator nesting" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 5)
        Wₕ = gridspace(Ωₕ)

        # A difference wrapped in an average, summed with a jump, inside an inner
        # product whose other side carries a region restriction: five node families
        # (difference, average, jump, OperatorAdd, restriction) nested in one expression,
        # purely to confirm correctly-ordered, correctly-parenthesized rendering.
        mix = form(
            Wₕ, Wₕ, (u, v) -> innerₕ(Mₓ(D₋ₓ(u)) + jumpₓ(u), restrict_to(:boundary, v))
        )
        @test expression(mix) == "innerₕ(Mₓ(D₋ₓ(u)) + jumpₓ(u), Rₕ(:boundary, v))"
    end

    @testset "DiracSource" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 5)
        Wₕ = gridspace(Ωₕ)

        l = form(Wₕ, v -> innerₕ(dirac(0.3, 2.0), v))
        s = expression(l)
        @test s == "innerₕ(dirac((0.3,), 2.0), v)"
        @test occursin("dirac", s)
        @test occursin("0.3", s)
        @test occursin("2.0", s)
    end

    @testset "Negative scalar (OperatorAdd subtraction)" begin
        # S1.1's subtraction detection: a negative OperatorScale folded into `OperatorAdd`
        # renders with a literal " - ", not " + -2 * ".
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 5)
        Wₕ = gridspace(Ωₕ)

        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(u, v) - 2 * innerₕ(D₋ₓ(u), D₋ₓ(v)))
        s = expression(a)
        @test s == "innerₕ(u, v) - 2 * innerₕ(D₋ₓ(u), D₋ₓ(v))"
        @test occursin(" - ", s)
        @test !occursin(" + -2 * ", s)
        @test !occursin("+ -", s)
    end

    @testset "Show integration" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 5)
        Wₕ = gridspace(Ωₕ)

        a = form(Wₕ, Wₕ, (u, v) -> innerₕ(D₋ₓ(u), D₋ₓ(v)))
        sa = sprint(show, MIME"text/plain"(), a)
        @test occursin("Expression", sa)
        @test occursin("Expression: $(expression(a))", sa)

        l = form(Wₕ, v -> innerₕ(3.0, v))
        sl = sprint(show, MIME"text/plain"(), l)
        @test occursin("Expression", sl)
        @test occursin("Expression: $(expression(l))", sl)
    end
end

# println("OK-S1.7")

end # module FormExpressionTests
