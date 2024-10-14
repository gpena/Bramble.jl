@inline _scalar2wrapper(::Type{T}, α::ElType) where {T,ElType<:Number} = FunctionWrapper{T,Tuple{}}(() -> convert(T, α)::T)
@inline _scalar2wrapper(::Type{T}, α) where T = FunctionWrapper{typeof(α),Tuple{}}(() -> α)

@inline _scalar2wrapper(::Type{T}, op, α::Tuple, β) where T = FunctionWrapper{typeof(α),Tuple{}}(() -> broadcast(op, α, β))
@inline _scalar2wrapper(::Type{T}, op, α, β::Tuple) where T = FunctionWrapper{typeof(β),Tuple{}}(() -> broadcast(op, α, β))

@inline _scalar2wrapper(::Type{T}, op, α::Number, β::Number) where T = FunctionWrapper{T,Tuple{}}(() -> broadcast(op, convert(T, α)::T, β))
@inline _scalar2wrapper(::Type{T}, op, α::VectorElement, β::Number) where T = FunctionWrapper{typeof(α),Tuple{}}(() -> broadcast(op, α, β))
@inline _scalar2wrapper(::Type{T}, op, α::Number, β::VectorElement) where T = FunctionWrapper{typeof(β),Tuple{}}(() -> broadcast(op, α, β))
@inline _scalar2wrapper(::Type{T}, op, α::VectorElement, β::VectorElement) where T = FunctionWrapper{typeof(α),Tuple{}}(() -> broadcast(op, α, β))

_process_scalar(f::FunctionWrapper{T,Tuple{}}) where T<:Number = "$(f())"
_process_scalar(f::FunctionWrapper{CoType,Tuple{}}) where CoType = "uₕ"


abstract type OperatorType <: BrambleType end
abstract type ScalarOperatorType <: OperatorType end
abstract type VectorOperatorType <: OperatorType end

@inline *(op::OperatorType, α) = α * op
@inline eltype(op::OperatorType) = eltype(space(op))
@inline space(op::OperatorType) = op.space

@inline function *(α::_T, op::OP) where {_T,OP<:OperatorType}
	T = eltype(op)

	if α isa Number && α == 0
		return ZeroOperator(space(op))
	end

	if α isa Number && α == 1
		return op
	end

	S = typeof(space(op))
	operator_cotype = codomaintype(op)

	if _T <: Number && operator_cotype <: Number
		f = _scalar2wrapper(T, α)
		return ScaledOperator{typeof(op),S,typeof(f)}(op.space, f, op)
	end

	if _T <: VectorElement || operator_cotype <: VectorElement
		f = _scalar2wrapper(T, *, α, 1)
		return ScaledOperator{typeof(op),S,typeof(f)}(op.space, f, op)
	end

	@error "Don't know how to handle this expression"
end

#########################################
#                                       #
#             Zero Operator             #
#                                       #
#########################################

struct ZeroOperator{S} <: OperatorType
	space::S
end

show(io::IO, _::ZeroOperator) = print(io, "0")

# rules for zero operator
@inline *(α, op::ZeroOperator) = op
@inline ⋅(α, op::ZeroOperator) = op
scalar(op::ZeroOperator) = zero(eltype(space(op)))




#########################################
#                                       #
#            Vector Operator            #
#                                       #
#########################################
# (op1,op2,op3) -> tuple(operators)
# define composite space: S × S × S
struct VectorOperator{S,CompType} <: OperatorType
	space::S
	component_operators::CompType
end

show(io::IO, op::VectorOperator) = print(io, "0")




#########################################
#                                       #
#           Identity Operator           #
#                                       #
#########################################

struct IdentityOperator{S} <: OperatorType
	space::S
end

show(io::IO, _::IdentityOperator) = print(io, "I")
scalar(op::IdentityOperator) = one(eltype(space(op)))

@inline function ⋅(α, op::IdentityOperator) 
	if α isa Number
		return ScaledOperator(op.space, α, op)
	end

	if α isa VectorElement
		return ScaledOperator(op.space, α, op)
	end

	if α isa Tuple
		@show "Tuple"

		#elem = ntuple(i -> op, length(α))
		#return VectorOperator(space(op), elem)
	end

	@error "Don't know how to handle this expression"
end







#########################################
#                                       #
#           Scaling Operator            #
#                                       #
#########################################
struct ScaledOperator{OP,S,V} <: OperatorType
	space::S
	scalar::V
	operator::OP
end

@inline codomaintype(_::ScaledOperator{OP,S,V}) where {OP,S,V} = codomaintype(V)
@inline scalar(op::ScaledOperator) = op.scalar()
@inline parent_operator(op::ScaledOperator) = op.operator

show(io::IO, op::ScaledOperator{OP}) where OP<:IdentityOperator = print(io, "$(_process_scalar(op.scalar)) * I")
show(io::IO, op::ScaledOperator{OP1}) where {OP2<:IdentityOperator,OP1<:ScaledOperator{OP2}} = print(io, "$(_process_scalar(op.scalar)) * I")
show(io::IO, op::ScaledOperator{OP}) where OP<:OperatorType = print(io, "$(_process_scalar(op.scalar)) * ($(op.operator))")

# rules for scaled operators
## α * ZeroOperator
ScaledOperator(S::SpaceType, α, op::ZeroOperator) = op

## α * op
function ScaledOperator(S::SpaceType, α, op = IdentityOperator(S))
	T = eltype(S)

	if α isa Number && α == 1
		return op
	end

	if α isa Number && α == 0
		return ZeroOperator(S)
	end 

	f = _scalar2wrapper(T, α)
	return ScaledOperator{typeof(op), typeof(S), typeof(f)}(S, f, op)
end

@inline *(α, op::IdentityOperator) = ScaledOperator(space(op), α)
@inline *(op::IdentityOperator, α) = α * op

@inline function *(α::_T, op::OP) where {_T,OP<:ScaledOperator}
	T = eltype(op)

	if α isa Number && α == 0
		return ZeroOperator(space(op))
	end

	if α isa Number && α == 1
		return op
	end

	S = typeof(space(op))
	operator_cotype = codomaintype(op)

	if _T <: Number && operator_cotype <: Number
		f = _scalar2wrapper(T, α * scalar(op))
		return ScaledOperator{typeof(op.operator),S,typeof(f)}(op.space, f, op.operator)
	end

	if _T <: VectorElement || operator_cotype <: VectorElement
		f = _scalar2wrapper(T, *, α, scalar(op))
		return ScaledOperator{typeof(op.operator),S,typeof(f)}(op.space, f, op.operator)
	end

	@error "Don't know how to handle this expression"
end










#########################################
#                                       #
#           Gradient Operators          #
#                                       #
#########################################
struct GradientOperator{S} <: OperatorType
	space::S
end

ScaledGradientOperator{S,V} = ScaledOperator{GradientOperator{S},S,V}

scalar(op::GradientOperator) = one(eltype(space(op)))
codomaintype(op::GradientOperator) = eltype(space(op))

show(io::IO, _::GradientOperator) = print(io, "∇vₕ")
show(io::IO, op::ScaledOperator{OP}) where OP<:GradientOperator = print(io, "$(_process_scalar(op.scalar)) * $(op.operator)")
show(io::IO, op::ScaledOperator{OP1}) where {OP<:GradientOperator,OP1<:ScaledOperator{OP}} = print(io, "$(op.operator)")

@inline codomaintype(_::ScaledGradientOperator{S,V}) where {S,V} = codomaintype(V)
@inline scalar(op::ScaledGradientOperator) = op.scalar()
@inline parent_operator(op::ScaledGradientOperator) = op.operator


# rules for (scaled) gradient operators
ScaledGradientOperator(_::SpaceType, _, op::ZeroOperator) = op
ScaledGradientOperator(S::SpaceType, α) = ScaledGradientOperator(S, α, GradientOperator(S))

@inline ∇₋ₕ(op::IdentityOperator) = GradientOperator(op.space)













#########################################
#                                       #
#         Add/Subtract Operators        #
#                                       #
#########################################

struct AddOperator{OP1,OP2,S} <: OperatorType
	space::S
	operator1::OP1
	operator2::OP2
	scalar::Int        # op1 + scalar*op2, scalar = 1 or -1
end

@inline function codomaintype(op::AddOperator) 
	@assert codomaintype(op.operator1) == codomaintype(op.operator2)
	return codomaintype(op.operator1)
end

@inline scalar(op::AddOperator) = op.scalar
@inline first(op::AddOperator) = op.operator1
@inline second(op::AddOperator) = op.operator2

show(io::IO, op::AddOperator) = print(io, "$(op.operator1) $(scalar(op) == 1 ? "+" : "-") $(op.operator2)")

+(op1::OP1, op2::OP2) where {OP1<:OperatorType, OP2<:OperatorType} = AddOperator{typeof(op1),typeof(op2),typeof(space(op1))}(space(op1), op1, op2, 1)
+(op1::OP1, _::ZeroOperator) where OP1<:OperatorType  = op1
+(_::ZeroOperator, op2::OP2) where OP2<:OperatorType  = op2

-(op1::OP1, op2::OP2) where {OP1<:OperatorType, OP2<:OperatorType} = AddOperator(space(op1), op1, op2, -1)
-(op1::OP1, _::ZeroOperator) where {OP1<:OperatorType} = op1
-(_::ZeroOperator, op2::OP2) where {OP2<:OperatorType} = ScaledOperator(space(op2), -1, op2)
















###################################
#                                 #
#   Specialized Inner Products    #
#                                 #
###################################

############### innerₕ ############
innerₕ(l::OperatorType, uₕ::VectorElement) = innerₕ(uₕ, l)

@inline innerₕ(uₕ::VectorElement, _::IdentityOperator) = innerh_weights(space(uₕ)) .* uₕ.values
@inline function innerₕ!(vₕ::AbstractVector, uₕ::VectorElement, _::IdentityOperator)
	@assert length(vₕ) == length(uₕ.values)
	vₕ .+= innerh_weights(space(uₕ)) .* uₕ.values
end

@inline function innerₕ(uₕ::VectorElement, l::ScaledOperator)
	return innerh_weights(space(uₕ)) .* l.scalar() .* uₕ.values
end

@inline function innerₕ!(vₕ::AbstractVector, uₕ::VectorElement, l::ScaledOperator)
	vₕ .+= innerh_weights(space(uₕ)) .* l.scalar() .* uₕ.values
end

############### inner₊ ############
inner₊(l::OperatorType, uₕ::VectorElement) = inner₊(uₕ, l)

@inline inner₊(uₕ::VectorElement, _::IdentityOperator) = innerplus_weights(space(uₕ), Val(1)) .* uₕ.values
@inline function inner₊!(vₕ::AbstractVector, uₕ::VectorElement, _::IdentityOperator)
	@assert length(vₕ) == length(uₕ.values)
	vₕ .+= innerplus_weights(space(uₕ), Val(1)) .* uₕ.values
end

@inline inner₊(uₕ::VectorElement, l::ScaledOperator) = innerplus_weights(space(uₕ), Val(1)) .* l.scalar() .* uₕ.values

@inline function inner₊!(vₕ::AbstractVector, uₕ::VectorElement, l::ScaledOperator)
	x = innerplus_weights(space(uₕ), Val(1))
	α = l.scalar()
	u = uₕ.values
	@.. vₕ += x * α * u
	nothing
end

@inline function inner₊(uₕ::VectorElement, l::GradientOperator)
	res = similar(uₕ.values)
	res .= 0
	inner₊!(res, uₕ, l)

	return res
end

function inner₊!(vₕ::AbstractVector, uₕ::VectorElement, l::GradientOperator)
	W = l.space
	D = dim(mesh(W))

	for i in 1:D
		x = innerplus_weights(W, i)
		@.. W.vec_cache = x * uₕ.values
		mul!(vₕ, transpose(W.diff_matrix_cache[i].values), W.vec_cache, 1, 1)
	end

	nothing
end

@inline function inner₊(uₕ::NTuple{D,VectorElement}, l::GradientOperator) where D
	res = similar(uₕ.values)
	res .= 0
	inner₊!(res, uₕ, l)

	return res
end

function inner₊!(vₕ::AbstractVector, uₕ::NTuple{D,VectorElement}, l::GradientOperator) where D
	W = l.space

	for i in 1:D
		x = innerplus_weights(W, i)
		@.. W.vec_cache = x * uₕ[i].values
		mul!(vₕ, transpose(W.diff_matrix_cache[i].values), W.vec_cache, 1, 1)
	end

	return nothing
end