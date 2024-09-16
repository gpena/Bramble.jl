using Test

using LinearAlgebra: norm, \

using Bramble


#Aqua.test_all(Bramble)

validate_zero(u) = norm(u, Inf) < 1e-9
validate_equal(u,v) = validate_zero(u.-v)

function leastsquares(x::AbstractVector, y::AbstractVector)
    A = [sum(x .* x) sum(x);
         sum(x) length(x)]

    b = [sum(x .* y); sum(y)]

    res = A \ b
    return res[1], res[2]
end


sep = "######"

#include("quality.jl")

@time @testset "\n$sep Sets and Domains $sep" begin
    include("sets.jl")
    include("domains.jl")
end

@time @testset "\n$sep Meshes $sep" begin
  include("mesh1d.jl")
  include("meshnd.jl")
end


@time @testset "\n$sep Grid spaces $sep" begin
    include("gridspaces.jl")
    include("vectorelements.jl")
    include("matrixelements.jl")
end



@time @testset "\n$sep Forms $sep" begin
    include("bilinearforms.jl")
end

@time @testset "\n$sep 1D Problems $sep" begin
    include("problems/1d/laplacian_uniform.jl")
    include("problems/1d/laplacian_nonuniform.jl")
    #include("problems/1d/laplacian_nonlinear.jl")
    #include("problems/1d/advection.jl")
    #include("problems/1d/advectionstab.jl")
    #include("problems/1d/wave.jl")
end

@time @testset "\n$sep 2D Problems $sep" begin
    include("problems/2d/laplacian_uniform.jl")
end

@time @testset "\n$sep 3D Problems $sep" begin
    include("problems/3d/laplacian_uniform.jl")
end
