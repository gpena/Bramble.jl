ops(::Val{1}) = (diff₋ₓ, diffₓ, jumpₓ, M₋ₕₓ, D₋ₓ)

function ops(::Val{2})
	ops2 = (D₋ᵧ, diffᵧ, diff₋ₓ, diff₋ᵧ, M₋ₕᵧ)
	return (ops2..., ops(Val(1))...)
end

function ops(::Val{3})
	ops2 = (D₋₂, diff₂, diff₋ₓ, diff₋₂, M₋ₕ₂)
	return (ops2..., ops(Val(2))...)
end

function matrix_element_tests(::Val{D}) where D
	dims, Wₕ, uₕ = __init(Val(D))

	test_function = @embed(mesh(Wₕ), x -> exp(-sum(x)))
	Rₕ!(uₕ, test_function)

	u₁ₕ = similar(uₕ.values)
	u₂ₕ = similar(u₁ₕ)

	gen_ops = ops(Val(D))

	for op in gen_ops
		u₁ₕ .= op(uₕ).values
		dd1 = reshape(u₁ₕ, dims)

		u₂ₕ .= op(Wₕ).values * uₕ.values
		dd2 = reshape(u₂ₕ, dims)
		@test(@views validate_equal(dd1, dd2))
	end
end

for i in 1:3
	matrix_element_tests(Val(i))
end