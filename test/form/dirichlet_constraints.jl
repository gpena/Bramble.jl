import Bramble: CartesianProduct, DirichletConstraint, label_conditions, embed_function, symbols, labels, DomainMarkers, tuples, conditions, identifier, EvaluatedDomainMarkers, BrambleFunction, label, markers, point, index_in_marker

@testset "Dirichlet Constraints Tests" begin
	# --- Setup ---
	I = interval(0.0, 1.0)
	Ω = I × I

	@testset "Boundary Constraints" begin
		# Define some functions to use as boundary conditions
		f1 = x -> x[1]^2 + x[2]
		f2 = x -> 2 * x[2]

		# Define a time-dependent function: f(x, t)
		f_t = (x, t) -> x[1] * t

		# Wrap them using the embed_function(macro
		bf1 = embed_function(Ω, f1)
		bf2 = embed_function(Ω, f2)
		bf_t = embed_function(Ω, I, f_t)

		# --- Tests ---

		@testset "dirichlet_constraints constructor" begin
			bcs = dirichlet_constraints(Ω, :gamma_1 => x->bf1(x), :gamma_2 => x->bf2(x))

			@test bcs isa DirichletConstraint
			@test length(label_conditions(bcs)) == 2
		end

		@testset "Time-dependent functor" begin
			# Create a time-dependent constraint
			bcs_t = dirichlet_constraints(Ω, I, :time_dep_bc => (x, t) -> f_t(x, t))

			function_markers = bcs_t.conditions
			function_snapshot = first(function_markers)
			@test identifier(function_snapshot) isa BrambleFunction
			@test length(function_markers) == 1

			# The new marker should contain a non-time-dependent BrambleFunction
			@test label(function_snapshot) == :time_dep_bc

			# The new function should be equivalent to `x -> f_t(x, 0.5)`
			x_point = (10.0, 5.0)
			t_point = 0.5

			@test identifier(function_snapshot)(t_point)(x_point) == f_t(x_point, t_point)
		end
	end

	@testset "Lazy Time Evaluation of DomainMarkers" begin
		original_markers = markers(Ω, I, :moving_front => (x, t) -> x[1] > t, :moving_back => (x, t) -> x[1] < t)
		lazy_markers_at_t = EvaluatedDomainMarkers(original_markers, 0.75)

		@test lazy_markers_at_t isa EvaluatedDomainMarkers
		@test lazy_markers_at_t.evaluation_time == 0.75

		@test symbols(lazy_markers_at_t) === symbols(original_markers)
		evaluated_conditions = collect(conditions(lazy_markers_at_t))

		@test length(evaluated_conditions) == 2
		for marker in evaluated_conditions
			new_bf = identifier(marker)
			@test new_bf isa BrambleFunction

			if label(marker) == :moving_front
				# The new function should be equivalent to `x -> x[1] > 0.75`
				@test new_bf(0.8, 0.8) == true
				@test new_bf(0.7, 0.7) == false
			end
		end
	end
end