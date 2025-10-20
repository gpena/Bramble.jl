"""
	LinearFormType

Abstract type for linear forms.
"""
abstract type LinearFormType <: BrambleType end

struct DefaultAssembly <: BrambleType end
struct InPlaceAssembly <: BrambleType end
struct OperatorsAssembly <: BrambleType end
struct AutoDetect <: BrambleType end

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
struct LinearForm{TestType,F,AssemblyType} <: LinearFormType
	test_space::TestType
	form_expr::F
	strategy::AssemblyType
end

"""
	test_space(a::LinearForm)

Returns the test space of a linear form.
"""
test_space(l::LinearForm) = l.test_space

"""
	form(Wₕ::SType, f::F)

Returns a linear form from a given expression and a test space.
"""
@inline function form(Wₕ::AbstractSpaceType, _f::F; strategy = DefaultAssembly(), verbose::Bool = false) where F
	available_strategy = []
	#=
		try
			#@show "test inplace"
			z = IdentityOperator(Wₕ)
			x = element(Wₕ)
			_f(x, z)
			#@show available_strategy
			push!(available_strategy, InPlaceAssembly())
			#@show available_strategy
		catch
			try
				#@show "test operators"
				z = IdentityOperator(Wₕ)
				_f(z)
				push!(available_strategy, OperatorsAssembly())
			catch #e2
				#@show e2
			end
		end=#
	#@show available_strategy
	if length(available_strategy) >= 1 && strategy isa AutoDetect
		strat = available_strategy[end]
		f = _form_expr2fwrapper(_f, Wₕ, strat)

		if verbose
			println("Linear form will be assembled using $(strat) strategy.")
		end
		return LinearForm{typeof(Wₕ),typeof(f),typeof(strat)}(Wₕ, f, strat)
	end

	f = _form_expr2fwrapper(_f, Wₕ, strategy)

	if verbose
		println("Linear form will be assembled using $(strategy) strategy.")
	end
	return LinearForm{typeof(Wₕ),typeof(f),typeof(strategy)}(Wₕ, f, strategy)
end

_form_expr2fwrapper(f, S, _::DefaultAssembly) = FunctionWrapper{Vector{eltype(S)},Tuple{typeof(elements(S))}}(f)
_form_expr2fwrapper(f, S, _::OperatorsAssembly) = FunctionWrapper{Vector{eltype(S)},Tuple{typeof(IdentityOperator(S))}}(f)
_form_expr2fwrapper(f, S, _::InPlaceAssembly) = FunctionWrapper{Nothing,Tuple{Vector{eltype(S)},typeof(IdentityOperator(S))}}(f)

"""
	assemble(l::LinearForm)

Returns the assembled linear form as a vector.
"""
@inline assemble(l::LinearForm) = _assemble(l, l.strategy)
@inline _assemble(l::LinearForm, ::DefaultAssembly) = l.form_expr(elements(test_space(l)))
#@inline _assemble(l::LinearForm, ::OperatorsAssembly) = l.form_expr(IdentityOperator(test_space(l)))
#@inline _assemble(l::LinearForm, ::InPlaceAssembly) = @error "Please call `assemble!(x, l)` instead of `assemble(l)`"
#@inline _assemble(l::LinearForm, ::AutoDetect) = @error "The linearform shouldn't have the AutoDetect strategy"

"""
	_assemble!(x::AbstractVector, l::LinearForm)

In-place assemble of a linear form into a given vector.
"""
@inline _assemble!(x::AbstractVector, l::LinearForm) = _assemble!(x, l, l.strategy)

@inline _assemble!(x::AbstractVector, l::LinearForm, ::DefaultAssembly) = x .= l.form_expr(elements(test_space(l)))
#@inline _assemble!(x::AbstractVector, l::LinearForm, ::OperatorsAssembly) = x .= l.form_expr(IdentityOperator(test_space(l)))
#@inline _assemble!(x::AbstractVector, l::LinearForm, ::InPlaceAssembly) = l.form_expr(x, IdentityOperator(test_space(l)))

"""
	_assemble!(x::VectorElement, l::LinearForm)

In-place assemble of a linear form into a given [VectorElement](@ref).
"""
@inline _assemble!(x::VectorElement, l::LinearForm) = (assemble!(x.values, l); return nothing)

"""
	assemble(l::LinearForm, dirichlet_conditions::DomainMarkers, [dirichlet_labels])

Returns the assembled linear form with imposed constraints as a vector of numbers.
"""
function assemble(l::LinearForm, dirichlet_conditions::DomainMarkers; dirichlet_labels = nothing)
	vec = assemble(l)

	Wₕ = test_space(l)
	Ωₕ = mesh(Wₕ)
	dict_mesh_markers = markers(Ωₕ)

	if dirichlet_labels !== nothing
		if dirichlet_labels isa Symbol
			dirichlet_bc!(vec, mesh(test_space(l)), dirichlet_conditions, dirichlet_labels)
		elseif dirichlet_labels isa Tuple
			if !isempty(dirichlet_labels)
				dirichlet_bc!(vec, mesh(test_space(l)), dirichlet_conditions, dirichlet_labels...)
			end
		else
			error("dirichlet_labels must be nothing, a Symbol, or a Tuple of Symbols")
		end
	end
	return vec
end

"""
	assemble!(vec::AbstractVector, l::LinearForm; dirichlet_conditions::DomainMarkers, [dirichlet_labels])

In-place assemble of a linear form with imposed constraints into a given vector.
"""
function assemble!(vec::AbstractVector, l::LinearForm; dirichlet_conditions::DomainMarkers = dirichlet_constraints(test_space(l)), dirichlet_labels = nothing)
	_assemble!(vec, l)

	Wₕ = test_space(l)
	Ωₕ = mesh(Wₕ)
	dict_mesh_markers = markers(Ωₕ)

	if dirichlet_labels !== nothing && !isempty(dict_mesh_markers)
		if dirichlet_labels isa Symbol
			dirichlet_bc!(vec, mesh(test_space(l)), dirichlet_conditions, dirichlet_labels)
		elseif dirichlet_labels isa Tuple
			if !isempty(dirichlet_labels)
				dirichlet_bc!(vec, mesh(test_space(l)), dirichlet_conditions, dirichlet_labels...)
			end
		else
			error("dirichlet_labels must be nothing, a Symbol, or a Tuple of Symbols")
		end
	end
	return nothing
end
