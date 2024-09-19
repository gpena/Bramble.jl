using Bramble

function domain_tests()
    M = [-3.0 10.0; 70.0 100.0; -15.0 -1.0]

    I = interval(M[1, 1], M[1, 2])

    f1(x) = x - 0.0
    f3(x) = x - 0.5
    domain_markers = create_markers("Dirichlet" => f1, "Neumann" => f3)
    
    I2 = interval(M[2, 1], M[2, 2])
    I3 = interval(M[3, 1], M[3, 2])

    for D in 1:3
        X = domain(I, domain_markers)

        @test(Bramble.dim(X) === 1)

        if D == 2
            X = domain(I × I2, domain_markers)
            @test(Bramble.dim(X) === 2)
        elseif D == 3
            X = domain(I × I2 × I3, domain_markers)
            @test(Bramble.dim(X) === 3)
        end

        for i = 1:D
            Pi = Bramble.projection(X, i)
            @test validate_equal(Pi.data[1][1], M[i, 1])
            @test validate_equal(Pi.data[1][2], M[i, 2])
        end
    end
end

domain_tests()