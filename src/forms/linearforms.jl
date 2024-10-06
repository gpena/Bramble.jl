"""
	LinearFormType

Abstract type for linear forms.
"""
abstract type LinearFormType <: BrambleType end

"""
	struct LinearForm{TestType,F} <: LinearFormType
		test_space::TestType
		form_expr::F
	end

Structure to store the data associated with a llinear form

```math
\\begin{array}{rcll}
l \\colon & W_h &\\longrightarrow &\\mathbb{R} \\\\
		  &  v  &\\longmapsto & l(v).
\\end{array}
```

The field `form_expr` has the expression of the form and the remaining field stores the test space ``W_h``.
"""
struct LinearForm{TestType,F} <: LinearFormType
	test_space::TestType
	form_expr::F
end

"""
	testspace(a::LinearForm)

Returns the test space of a linear form.
"""
testspace(l::LinearForm) = l.test_space

"""
	form(Wₕ::SType, f::F)

Returns a linear form from a given expression and a test space.
"""
form(Wₕ::SpaceType, f::F) where F = LinearForm{typeof(Wₕ),F}(Wₕ, f)

"""
	assemble(l::LinearForm)

Returns the assembled linear form as a vector.
"""
function assemble(l::LinearForm)
	z = elements(testspace(l))

	return l.form_expr(z)
end

"""
	assemble!(x::AbstractVector, l::LinearForm)

In-place assemble of a linear form into a given vector.
"""
function assemble!(x::AbstractVector, l::LinearForm)
	x .= l.form_expr(elements(testspace(l)))
end

"""
	assemble!(x::VectorElement, l::LinearForm)

In-place assemble of a linear form into a given [VectorElement](@ref).
"""
function assemble!(x::VectorElement, l::LinearForm)
	x .= l.form_expr(elements(testspace(l)))
end

"""
	assemble(l::LinearForm, bcs::Constraints)

Returns the assembled linear form with imposed constraints as a vector of numbers.
"""
function assemble(l::LinearForm, bcs::Constraints)
	vec = assemble(l)

	if constraint_type(bcs) == :dirichlet
		apply_dirichlet_bc!(vec, bcs, mesh(testspace(l)))
	end

	return vec
end

"""
	assemble!(vec::AbstractVector, l::LinearForm, bcs::Constraints)

In-place assemble of a linear form with imposed constraints into a given vector.
"""
function assemble!(vec::AbstractVector, l::LinearForm, bcs::Constraints)
	assemble!(vec, l)

	if constraint_type(bcs) == :dirichlet
		apply_dirichlet_bc!(vec, bcs, mesh(testspace(l)))
	end
end
