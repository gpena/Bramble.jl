"""
	BilinearFormType

Abstract type for bilinear forms.
"""
abstract type BilinearFormType <: BrambleType end

"""
	struct BilinearForm{TrialType,TestType,F} <: BilinearFormType
		trial_space::TrialType
		test_space::TestType
		form_expr::F
	end

Structure to store the data associated with a bilinear form

```math
\\begin{array}{rcll}
a \\colon & W_h \\times V_h &\\longrightarrow &\\mathbb{R} \\\\
		  &       (u,v)     &\\longmapsto & a(u,v).
\\end{array}
```

The field `form_expr` has the expression of the form and the remaining fields store the trial and test spaces ``W_h`` and ``V_h``.
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
	A = _assemble(a)
	if dirichlet_labels !== nothing
		if dirichlet_labels isa Symbol
			dirichlet_bc!(A, mesh(trial_space(a)), dirichlet_labels)
		elseif dirichlet_labels isa Tuple
			if !isempty(dirichlet_labels)
				dirichlet_bc!(A, mesh(trial_space(a)), dirichlet_labels...)
			end
		else
			error("dirichlet_labels must be nothing, a Symbol, or a Tuple of Symbols")
		end
	end
	return A
end

"""
	assemble!(A::AbstractMatrix, a::BilinearFormType [dirichlet_labels])

Copies the assembled matrix of a bilinear form and imposes the Dirichlet constraints to a given matrix `A`.
"""
function assemble!(A::AbstractMatrix, a::BilinearForm; dirichlet_labels = nothing)
	_assemble!(A, a)
	if dirichlet_labels !== nothing
		if dirichlet_labels isa Symbol
			dirichlet_bc!(A, mesh(trial_space(a)), dirichlet_labels)
		elseif dirichlet_labels isa Tuple
			if !isempty(dirichlet_labels)
				dirichlet_bc!(A, mesh(trial_space(a)), dirichlet_labels...)
			end
		else
			error("dirichlet_labels must be nothing, a Symbol, or a Tuple of Symbols")
		end
	end
end

