import Bramble: points

import Bramble: points, BilinearForm, LinearForm, DirichletConstraint, trial_space, test_space, set, element, Rₕ!, D₋ₓ, M₋ₕ, innerₕ, inner₊, form, assemble, assemble!, dirichlet_constraints

import SparseArrays: issparse

import LinearAlgebra: diag, issymmetric, norm

@testset "BilinearForm Construction and Accessors" begin
	N = 5;
	I = interval(-1.0, 4.0);

	X = domain(I, markers(I, :bc => x -> x[1]-4 < 0));
	Mh = mesh(X, N, false);

	Wh = gridspace(Mh);

	a(U, V) = innerₕ(U, V);
	bform = form(Wh, Wh, a);
	A = assemble(bform)

	bform2 = form(Wh, Wh, (U, V) -> inner₊(D₋ₓ(U), D₋ₓ(V)));
	assemble(bform2)

	bform3 = form(Wh, Wh, (U, V) -> inner₊(M₋ₕ(U), D₋ₓ(V)));
	assemble(bform3)

	gh = element(Wh);
	Rₕ!(gh, x->sin(x[1]));

	l(V) = innerₕ(gh, V);
	lform = form(Wh, l);
	F = assemble(lform)

	u = A\F

	@test u ≈ sin.(points(Mh))
end

@testset "Form Module Coverage Tests" begin
	@testset "BilinearForm: Basic API Coverage" begin
		# Test 1D case which is well supported
		N = 10
		I = interval(0.0, 1.0)
		X = domain(I, markers(I, :boundary => x -> abs(x[1] - 0.5) > 0.48))
		Mh = mesh(X, N, false)
		Wh = gridspace(Mh)

		@testset "Form construction and accessors" begin
			# Test basic construction
			a = form(Wh, Wh, (u, v) -> innerₕ(u, v))
			@test a isa BilinearForm
			@test trial_space(a) === Wh
			@test test_space(a) === Wh

			# Test with different spaces (still same underlying space)
			Vh = gridspace(Mh)
			b = form(Wh, Vh, (u, v) -> innerₕ(u, v))
			@test trial_space(b) === Wh
			@test test_space(b) === Vh
		end

		@testset "Callable interface" begin
			a = form(Wh, Wh, (u, v) -> innerₕ(u, v))

			uₕ = element(Wh)
			vₕ = element(Wh)

			Rₕ!(uₕ, x -> sin(2π*x[1]))
			Rₕ!(vₕ, x -> cos(2π*x[1]))

			# Direct evaluation
			result = a(uₕ, vₕ)
			@test result isa Number
			@test isfinite(result)
		end

		@testset "Assembly variants" begin
			a = form(Wh, Wh, (u, v) -> inner₊(D₋ₓ(u), D₋ₓ(v)))

			# Test basic assembly
			A1 = assemble(a)
			@test A1 isa AbstractMatrix
			@test size(A1) == (ndofs(Wh), ndofs(Wh))
			@test issparse(A1)

			# Test with boundary conditions (single label)
			A2 = assemble(a, dirichlet_labels = :boundary)
			@test A2 isa AbstractMatrix
			@test size(A2) == size(A1)
			@test A1 != A2  # Should be different with BCs

			# Test with tuple of labels (even if just one)
			A3 = assemble(a, dirichlet_labels = (:boundary,))
			@test A3 isa AbstractMatrix
			@test size(A3) == size(A1)

			# Test with nothing (explicit)
			A4 = assemble(a, dirichlet_labels = nothing)
			@test A4 ≈ A1  # Should be same as no BCs
		end

		@testset "In-place assembly" begin
			a = form(Wh, Wh, (u, v) -> innerₕ(u, v))

			A = assemble(a)
			A_copy = similar(A)
			fill!(A_copy, 0)

			assemble!(A_copy, a)
			@test A_copy ≈ A
		end

		@testset "Different bilinear forms" begin
			# Mass matrix
			a_mass = form(Wh, Wh, (u, v) -> innerₕ(u, v))
			M = assemble(a_mass)
			@test all(diag(M) .> 0)  # Positive diagonal
			@test issymmetric(M)

			# Stiffness matrix
			a_stiff = form(Wh, Wh, (u, v) -> inner₊(D₋ₓ(u), D₋ₓ(v)))
			K = assemble(a_stiff)
			@test issparse(K)

			# Mixed form
			a_mixed = form(Wh, Wh, (u, v) -> inner₊(M₋ₕ(u), D₋ₓ(v)))
			C = assemble(a_mixed)
			@test size(C) == (ndofs(Wh), ndofs(Wh))
		end
	end

	@testset "LinearForm: Basic API Coverage" begin
		N = 10
		I = interval(0.0, 1.0)
		X = domain(I, markers(I, :left => x -> x[1] < 0.05, :right => x -> x[1] > 0.95))
		Mh = mesh(X, N, false)
		Wh = gridspace(Mh)

		@testset "Form construction and accessors" begin
			fh = element(Wh)
			Rₕ!(fh, x -> exp(x[1]))

			l = form(Wh, v -> innerₕ(fh, v))
			@test l isa LinearForm
			@test test_space(l) === Wh
		end

		@testset "Callable interface" begin
			fh = element(Wh)
			Rₕ!(fh, x -> x[1]^2)

			l = form(Wh, v -> innerₕ(fh, v))

			vₕ = element(Wh)
			Rₕ!(vₕ, x -> 1.0)

			result = l(vₕ)
			@test result isa Number
			@test isfinite(result)
			@test result > 0  # Positive integrand
		end

		@testset "Assembly without BCs" begin
			fh = element(Wh)
			Rₕ!(fh, x -> sin(π*x[1]))

			l = form(Wh, v -> innerₕ(fh, v))
			F = assemble(l)

			@test F isa AbstractVector
			@test length(F) == ndofs(Wh)
		end

		@testset "Assembly with Dirichlet conditions" begin
			fh = element(Wh)
			Rₕ!(fh, x -> 1.0)

			l = form(Wh, v -> innerₕ(fh, v))

			# Create boundary conditions
			bcs = dirichlet_constraints(set(X), :left => x -> 0.0, :right => x -> 1.0)

			# Assemble with conditions
			F1 = assemble(l, bcs)
			@test F1 isa AbstractVector
			@test length(F1) == ndofs(Wh)

			# Test with dirichlet_labels parameter
			F2 = assemble(l, bcs, dirichlet_labels = :left)
			@test F2 isa AbstractVector
			@test length(F2) == ndofs(Wh)

			# Compare: should be different
			@test F1 != F2
		end

		@testset "In-place assembly" begin
			fh = element(Wh)
			Rₕ!(fh, x -> 2.0)

			l = form(Wh, v -> innerₕ(fh, v))

			F = assemble(l)
			F_copy = similar(F)
			fill!(F_copy, 0)

			assemble!(F_copy, l)
			@test F_copy ≈ F
		end

		@testset "Different linear forms" begin
			# L2 load vector
			fh = element(Wh)
			Rₕ!(fh, x -> exp(-x[1]))
			l1 = form(Wh, v -> innerₕ(fh, v))
			F1 = assemble(l1)
			@test all(F1 .> 0)

			# H1 load vector
			gh = element(Wh)
			Rₕ!(gh, x -> x[1])
			l2 = form(Wh, v -> inner₊(D₋ₓ(gh), D₋ₓ(v)))
			F2 = assemble(l2)
			@test length(F2) == ndofs(Wh)
		end
	end

	@testset "Dirichlet Constraints: Additional Coverage" begin
		I = interval(0.0, 1.0)
		X = domain(I, markers(I,
							  :left => x -> x[1] < 0.05,
							  :right => x -> x[1] > 0.95,
							  :interior => x -> 0.4 < x[1] < 0.6))

		@testset "Multiple boundary markers" begin
			# Test with multiple markers
			bcs = dirichlet_constraints(set(X),
										:left => x -> 0.0,
										:right => x -> 1.0)

			@test bcs isa DirichletConstraint
			@test length(label_conditions(bcs)) == 2
		end

		@testset "Different value functions" begin
			# Constant value
			bcs1 = dirichlet_constraints(set(X), :left => x -> 1.0)
			@test bcs1 isa DirichletConstraint

			# Linear value
			bcs2 = dirichlet_constraints(set(X), :right => x -> 2 * x[1])
			@test bcs2 isa DirichletConstraint

			# Nonlinear value
			bcs3 = dirichlet_constraints(set(X), :interior => x -> sin(π * x[1]))
			@test bcs3 isa DirichletConstraint
		end

		@testset "Integration with forms" begin
			Mh = mesh(X, 10, false)
			Wh = gridspace(Mh)

			# Create form and conditions
			a = form(Wh, Wh, (u, v) -> inner₊(D₋ₓ(u), D₋ₓ(v)))
			bcs = dirichlet_constraints(set(X), :left => x -> 0.0, :right => x -> 0.0)

			# Assemble with conditions
			A = assemble(a, dirichlet_labels = :left)
			@test A isa AbstractMatrix

			# Assemble linear form with conditions
			fh = element(Wh)
			Rₕ!(fh, x -> 1.0)
			l = form(Wh, v -> innerₕ(fh, v))
			F = assemble(l, bcs)
			@test F isa AbstractVector
		end
	end

	@testset "Complete FEM Workflow Coverage" begin
		# Test complete workflow to ensure all parts work together
		N = 20
		I = interval(0.0, 1.0)
		X = domain(I, markers(I, :boundary => x -> x[1] < 0.01 || x[1] > 0.99))
		Mh = mesh(X, N, false)
		Wh = gridspace(Mh)

		# Define problem: -u'' = f with u(0) = u(1) = 0
		# Solution: u(x) = x(1-x)
		# RHS: f(x) = 2

		# Assembly
		a = form(Wh, Wh, (u, v) -> inner₊(D₋ₓ(u), D₋ₓ(v)))
		A = assemble(a, dirichlet_labels = :boundary)

		fh = element(Wh)
		Rₕ!(fh, x -> 2.0)
		l = form(Wh, v -> innerₕ(fh, v))

		bcs = dirichlet_constraints(set(X), :boundary => x -> 0.0)
		F = assemble(l, bcs, dirichlet_labels = :boundary)

		# Solve
		u = A \ F

		# Check solution
		@test length(u) == ndofs(Wh)
		@test all(isfinite.(u))

		# Solution should be approximately parabolic
		pts = points(Mh)
		exact = [x * (1 - x) for x in pts]
		error = norm(u - exact, Inf)
		@test error < 0.01  # Should be accurate for N=20
	end
end
