struct Laplacian{DomainType, F<:Function,G<:Function}
    X::DomainType
    u::F
    f::G
end

