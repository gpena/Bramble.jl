"""
    jumpₓ(S::SpaceType)

Returns the jump matrix for the space `S` in the x-direction.

**Inputs:**

  - `S`: a space
"""
@inline jumpₓ(S::SpaceType) = shiftₓ(mesh(S), Val(dim(mesh(S))), Val(0)) -  shiftₓ(mesh(S), Val(dim(mesh(S))), Val(1))

"""
    jumpᵧ(S::SpaceType)

Returns the jump matrix for the space `S` in the y-direction.

**Inputs:**

  - `S`: a space
"""
@inline jumpᵧ(S::SpaceType) = shiftᵧ(mesh(S), Val(dim(mesh(S))), Val(0)) -  shiftᵧ(mesh(S), Val(dim(mesh(S))), Val(1))

"""
    jump₂(S::SpaceType)

Returns the jump matrix for the space `S` in the z-direction.

**Inputs:**

  - `S`: a space
"""
@inline jump₂(S::SpaceType) = shift₂(mesh(S), Val(dim(mesh(S))), Val(0)) -  shift₂(mesh(S), Val(dim(mesh(S))), Val(1))

@inline jumpᵢ(S::SpaceType, ::Val{1}) = jumpₓ(S)
@inline jumpᵢ(S::SpaceType, ::Val{2}) = jumpᵧ(S)
@inline jumpᵢ(S::SpaceType, ::Val{3}) = jump₂(S)

@inline jump(u::VectorElement) = ntuple(i-> element(space(u), jumpᵢ(space(u), Val(i))*u.values), dim(u))
@inline jumpₓ(u::VectorElement) = element(space(u), jumpₓ(space(u))*u.values)
@inline jumpᵧ(u::VectorElement) = element(space(u), jumpᵧ(space(u))*u.values)
@inline jump₂(u::VectorElement) = element(space(u), jump₂(space(u))*u.values)

@inline jump(u::MatrixElement) = ntuple(i-> elements(space(u), jumpᵢ(space(u), Val(i))*u.values), dim(u))
@inline jumpₓ(u::MatrixElement) = elements(space(u), jumpₓ(space(u))*u.values)
@inline jumpᵧ(u::MatrixElement) = elements(space(u), jumpᵧ(space(u))*u.values)
@inline jump₂(u::MatrixElement) = elements(space(u), jump₂(space(u))*u.values)

jump(M::MeshType) = (@assert dim(M) == 1; return jumpₓ(M))
jump(S::SpaceType) = (@assert dim(mesh(S)) == 1; return jumpₓ(S))
