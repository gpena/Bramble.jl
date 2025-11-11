#=
# bilinear_form.jl

This file implements bilinear forms for finite element assembly.

## Mathematical Background

A bilinear form is a mapping:
```math
a : W_h × V_h → ℝ
```

where W_h and V_h are the trial and test spaces respectively.

## Usage Pattern

```julia
# Define spaces
Wₕ = gridspace(Ωₕ)  # Trial space
Vₕ = gridspace(Ωₕ)  # Test space (often same as trial)

# Create bilinear form (e.g., stiffness matrix)
a = form(Wₕ, Vₕ, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))

# Assemble system matrix
A = assemble(a)

# With Dirichlet conditions
A = assemble(a, dirichlet_labels=:boundary)
```

## Performance Considerations

- Assembly uses FunctionWrappers for type stability
- Specialized sparse matrix handling for boundary conditions
- Bit manipulation for fast Dirichlet index processing

See also: [`BilinearForm`](@ref), [`assemble`](@ref), [`form`](@ref)
=#

"""
	BilinearFormType

Abstract type for bilinear forms.
"""
abstract type BilinearFormType end

"""
	struct BilinearForm{TrialType,TestType,F} <: BilinearFormType
		trial_space::TrialType
		test_space::TestType
		form_expression::F
	end

Structure to store the data associated with a bilinear form

```math
\\begin{array}{rcll}
a \\colon & W_h \\times V_h &\\longrightarrow &\\mathbb{R} \\\\
		  &       (u,v)     &\\longmapsto & a(u,v).
\\end{array}
```

The field `form_expression` has the expression of the form and the remaining fields
store the trial and test spaces ``W_h`` and ``V_h``.

# Fields

  - `trial_space::TrialType` - The trial space W_h
  - `test_space::TestType` - The test space V_h
  - `form_expression::F` - Function defining a(u,v)

# Example

```julia
# Stiffness matrix for Poisson equation
Wₕ = gridspace(Ωₕ)
a = form(Wₕ, Wₕ, (u, v) -> inner₊(∇₋ₕ(u), ∇₋ₕ(v)))

# Assemble
A = assemble(a)

# With Dirichlet boundary conditions
A = assemble(a, dirichlet_labels = :boundary)

# Evaluate form on specific functions
uₕ = element(Wₕ)
vₕ = element(Wₕ)
result = a(uₕ, vₕ)  # Returns scalar
```

See also: [`form`](@ref), [`assemble`](@ref), [`trial_space`](@ref), [`test_space`](@ref)
"""
struct BilinearForm{TrialType,TestType,F} <: BilinearFormType
	trial_space::TrialType
	test_space::TestType
	form_expression::F
end

"""
	trial_space(a::BilinearForm)

Returns the trial space of a bilinear form.
"""
trial_space(a::BilinearFormType) = a.trial_space

"""
	test_space(a::BilinearForm)

Returns the test space of a bilinear form.
"""
test_space(a::BilinearFormType) = a.test_space

"""
	form(Wₕ::AbstractSpaceType, Vₕ::AbstractSpaceType, f)

Returns a bilinear form from a given expression and trial and test spaces.
"""
@inline form(Wₕ::AbstractSpaceType, Vₕ::AbstractSpaceType, f::F) where F = BilinearForm{typeof(Wₕ),typeof(Vₕ),F}(Wₕ, Vₕ, f)

"""
	(a::BilinearForm)(u, v)

Callable interface for evaluating a bilinear form on given functions.

Returns the scalar value a(u,v) where u and v are elements from the trial
and test spaces respectively.
"""
@inline (a::BilinearForm)(u, v) = a.form_expression(u, v)

"""
	_assemble(a::BilinearForm)

Helper function. Returns the assembled matrix of a bilinear form.
"""
_assemble(a::BilinearForm) = a(elements(test_space(a)), elements(trial_space(a)))

"""
	_assemble!(A::AbstractMatrix, a::BilinearForm)

Helper function. Copies the assembled matrix of a bilinear form to a given matrix.
"""
_assemble!(A::AbstractMatrix, a::BilinearForm) = (copyto!(A, assemble(a)))

"""
	assemble(a::BilinearForm, [dirichlet_labels])

Returns the assembled matrix of a bilinear form with imposed constraints.
"""
function assemble(a::BilinearForm; dirichlet_labels = nothing)
	_validate_dirichlet_labels(dirichlet_labels)
	A = _assemble(a)
	if dirichlet_labels !== nothing
		if dirichlet_labels isa Symbol
			dirichlet_bc!(A, mesh(trial_space(a)), dirichlet_labels)
		elseif dirichlet_labels isa Tuple
			if !isempty(dirichlet_labels)
				dirichlet_bc!(A, mesh(trial_space(a)), dirichlet_labels...)
			end
		end
	end
	return A
end

"""
	assemble!(A::AbstractMatrix, a::BilinearFormType [dirichlet_labels])

Copies the assembled matrix of a bilinear form and imposes the Dirichlet constraints to a given matrix `A`.
"""
function assemble!(A::AbstractMatrix, a::BilinearForm; dirichlet_labels = nothing)
	_validate_dirichlet_labels(dirichlet_labels)
	_assemble!(A, a)
	if dirichlet_labels !== nothing
		if dirichlet_labels isa Symbol
			dirichlet_bc!(A, mesh(trial_space(a)), dirichlet_labels)
		elseif dirichlet_labels isa Tuple
			if !isempty(dirichlet_labels)
				dirichlet_bc!(A, mesh(trial_space(a)), dirichlet_labels...)
			end
		end
	end
end