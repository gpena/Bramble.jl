#=
# linear_form.jl

This file implements linear forms for finite element assembly.

## Mathematical Background

A linear form is a mapping:
```math
l : W_h → ℝ
```

where W_h is the test space.

## Usage Pattern

```julia
# Define space
Wₕ = gridspace(Ωₕ)

# Create linear form (e.g., load vector)
l = form(Wₕ, v -> innerₕ(f, v))

# Assemble load vector
F = assemble(l)

# With Dirichlet conditions
F = assemble(l, dirichlet_conditions, dirichlet_labels=:boundary)
```

## Assembly Strategies

- `DefaultAssembly`: Standard approach using VectorElement
- `InPlaceAssembly`: In-place computation for reduced allocations
- `OperatorsAssembly`: Uses IdentityOperator instead of VectorElement
- `AutoDetect`: Automatic strategy selection (experimental)

See also: [`LinearForm`](@ref), [`assemble`](@ref), [`form`](@ref)
=#

"""
	LinearFormType

Abstract type for linear forms.
"""
abstract type LinearFormType end

"""
	DefaultAssembly

Assembly strategy that evaluates the linear form by passing a [VectorElement](@ref)
to the form expression. This is the standard and most flexible approach.

# Example

```julia
l = form(Wₕ, v -> innerₕ(f, v), strategy = DefaultAssembly())
```

See also: [`InPlaceAssembly`](@ref), [`OperatorsAssembly`](@ref), [`AutoDetect`](@ref)
"""
struct DefaultAssembly end

"""
	InPlaceAssembly

Assembly strategy for in-place computation. The form expression should accept
two arguments: `(output_vector, operator)` and modify `output_vector` directly.

This strategy is useful for reducing allocations in iterative solvers.

See also: [`DefaultAssembly`](@ref), [`assemble!`](@ref)
"""
struct InPlaceAssembly end

"""
	OperatorsAssembly

Assembly strategy that passes an [`IdentityOperator`](@ref) to the form expression
instead of a [VectorElement](@ref). Useful for operator-based formulations.

See also: [`DefaultAssembly`](@ref), [`IdentityOperator`](@ref)
"""
struct OperatorsAssembly end

"""
	AutoDetect

Automatic detection of the most appropriate assembly strategy based on the form
expression signature. Currently not fully implemented.

See also: [`DefaultAssembly`](@ref)
"""
struct AutoDetect end

"""
	struct LinearForm{TestType,F} <: LinearFormType
		test_space::TestType
		form_expr::F
	end

Structure to store the data associated with a linear form

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
	form(Wₕ::AbstractSpaceType, f; strategy=DefaultAssembly(), verbose=false)

Returns a linear form from a given expression `f` and a test space `Wₕ`.

# Arguments

  - `Wₕ::AbstractSpaceType`: The test space
  - `f`: Function defining the linear form expression
  - `strategy`: Assembly strategy (default: `DefaultAssembly()`)
  - `verbose::Bool`: If `true`, logs the selected assembly strategy

# Returns

A `LinearForm` object ready for assembly.

# Example

```julia
Wₕ = gridspace(Ωₕ)
f = x -> sin(π*x[1])
l = form(Wₕ, v -> innerₕ(f, v), verbose = true)
```
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
		@info "Linear form will be assembled using $(strategy) strategy."
	end
	return LinearForm{typeof(Wₕ),typeof(f),typeof(strategy)}(Wₕ, f, strategy)
end

"""
	(l::LinearForm)(u)

Callable interface for evaluating a linear form on given elements.

Returns the scalar value l(u) where u is an element from the test space.
"""
#TODO: improve this to actually evaluate the form on u
@inline (l::LinearForm)(u::VectorElement) = dot(l.form_expr(elements(test_space(l))), u.data)
@inline (l::LinearForm)(u::MatrixElement) = l.form_expr(u)

"""
	_form_expr2fwrapper(f, S, strategy)

Internal helper to wrap form expressions in FunctionWrappers for type stability.

Converts the user-provided function `f` into a type-stable FunctionWrapper based on
the selected assembly strategy. This eliminates dynamic dispatch and improves performance.

# Arguments

  - `f`: The form expression function
  - `S`: The test space
  - `strategy`: The assembly strategy (determines wrapper signature)

# Returns

A FunctionWrapper with appropriate type signature for the given strategy.
"""
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
	_assemble!(uₕ::VectorElement, l::LinearForm)

In-place assemble of a linear form into a given [VectorElement](@ref).
"""
@inline _assemble!(uₕ::VectorElement, l::LinearForm) = (assemble!(values(uₕ), l); return nothing)

"""
	assemble(l::LinearForm, dirichlet_conditions::DomainMarkers, [dirichlet_labels])

Returns the assembled linear form with imposed constraints as a vector of numbers.
"""
function assemble(l::LinearForm, dirichlet_conditions::DomainMarkers; dirichlet_labels = nothing)
	_validate_dirichlet_labels(dirichlet_labels)
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
		end
	end
	return vec
end

"""
	assemble!(vec::AbstractVector, l::LinearForm; dirichlet_conditions::DomainMarkers, [dirichlet_labels])

In-place assemble of a linear form with imposed constraints into a given vector.
"""
function assemble!(vec::AbstractVector, l::LinearForm; dirichlet_conditions::DomainMarkers = dirichlet_constraints(test_space(l)), dirichlet_labels = nothing)
	_validate_dirichlet_labels(dirichlet_labels)
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
		end
	end
	return nothing
end
