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

The field `form_expr` has the expression of the form and the remaining fields store the trial and test spaces `` W_h`` and ``V_h``.
"""
struct BilinearForm{TrialType,TestType,F} <: BilinearFormType
	trial_space::TrialType
	test_space::TestType
	form_expr::F
end

"""
	trialspace(a::BilinearForm)

Returns the trial space of a bilinear form.
"""
trialspace(a::BilinearFormType) = a.trial_space

"""
	testspace(a::BilinearForm)

Returns the test space of a bilinear form.
"""
testspace(a::BilinearFormType) = a.test_space

"""
	form(Wₕ::SpaceType, Vₕ::SpaceType, f)

Returns a bilinear form from a given expression and trial and test spaces.
"""
form(Wₕ::SpaceType, Vₕ::SpaceType, f::F) where F = BilinearForm{typeof(Wₕ),typeof(Vₕ),F}(Wₕ, Vₕ, f)

"""
	assemble(a::BilinearForm)

Returns the assembled matrix of a bilinear form.
"""
assemble(a::BilinearForm) = a.form_expr(elements(testspace(a)), elements(trialspace(a)))

"""
	assemble!(A::AbstractMatrix, a::BilinearForm)

Copies the assembled matrix of a bilinear form to a given matrix.
"""
assemble!(A::AbstractMatrix, a::BilinearForm) = (copyto!(A, assemble(a)))

"""
	assemble(a::BilinearForm, bcs::Constraints)

Returns the assembled matrix of a bilinear form with imposed constraints.
"""
function assemble(a::BilinearForm, bcs::Constraints)
	A = assemble(a)

	if constraint_type(bcs) == :dirichlet
		apply_dirichlet_bc!(A, bcs, mesh(trialspace(a)))
	end

	return A
end

"""
	assemble!(A::AbstractMatrix, a::BilinearFormType, bcs::Constraints)

Copies the assembled matrix of a bilinear form with imposed constraints to a given matrix.
"""
function assemble!(A::AbstractMatrix, a::BilinearForm, bcs::Constraints)
	assemble!(A, a)

	if constraint_type(bcs) == :dirichlet
		apply_dirichlet_bc!(A, bcs, mesh(trialspace(a)))
	end
end