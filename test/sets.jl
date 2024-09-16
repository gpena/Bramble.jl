using Bramble
using Bramble: projection, dim, CartesianProduct

function set_tests()
    I = Interval(-3.0, 10.0)

    @test validate_equal(I.data[1][1], -3.0)
    @test validate_equal(I.data[1][2], 10.0)
    @test dim(I) == 1

    I2 = Interval(70.0, 100.0)

    test_sets = [a × b for a in [I] for b in [I2]]
    for set in test_sets
        Ω2_x = projection(set, 1)
        Ω2_y = projection(set, 2)

        @test validate_equal(Ω2_y.data[1][1], 70.0)
        @test validate_equal(Ω2_x.data[1][1], -3.0)
        @test validate_equal(Ω2_y.data[1][2], 100.0)
        @test validate_equal(Ω2_x.data[1][2], 10.0)
    end

    I3 = Interval(-15.0, -1.0)

    S(x) = (x, CartesianProduct(x))
    test_sets = (a × b × c for a in S(I) for b in S(I2) for c in S(I3))
    for set in test_sets
        Ω3_x = projection(set, 1)
        Ω3_y = projection(set, 2)
        Ω3_z = projection(set, 3)

        @test validate_equal(Ω3_x.data[1][1], -3.0)
        @test validate_equal(Ω3_x.data[1][2], 10.0)
        @test validate_equal(Ω3_y.data[1][1], 70.0)
        @test validate_equal(Ω3_y.data[1][2], 100.0)
        @test validate_equal(Ω3_z.data[1][1], -15.0)
        @test validate_equal(Ω3_z.data[1][2], -1.0)
    end
end

set_tests()