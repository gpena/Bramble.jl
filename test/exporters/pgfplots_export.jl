using Test
using Bramble

# Plain text, so unlike the VTK export test, the byte content itself is exactly what is
# under test: there is no library on the other end whose correctness can be assumed.
# Two facts matter most: the header/column layout for the 1D table format, and the
# blank-line placement for the 2D `surf`/`mesh` format, which is the single most common
# mistake when writing this by hand (miss it, and pgfplots connects points across rows
# that should not be connected, into a shredded zigzag instead of a surface).

@testset "PGFPlots export" begin
    @testset "1D format" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 4, true)
        Wₕ = gridspace(Ωₕ)
        uₕ = Rₕ(Wₕ, x -> x)
        vₕ = Rₕ(Wₕ, x -> 2x)

        mktempdir() do dir
            f = export_pgfplots(joinpath(dir, "t"), Ωₕ, "u" => uₕ, "v" => vₕ)
            @test isfile(f)
            @test endswith(f, ".dat")

            lines = readlines(f)
            @test lines[1] == "x u v"
            @test length(lines) == 1 + ndofs(Wₕ)

            rows = [parse.(Float64, split(l)) for l in lines[2:end]]
            xs = points(Ωₕ)
            for (i, row) in enumerate(rows)
                @test row == [xs[i], xs[i], 2xs[i]]
            end
        end
    end

    @testset "1D composite" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 4, true)
        Wₕ = gridspace(Ωₕ)
        Vₕ = Wₕ^Val(2)
        cₕ = Rₕ(Vₕ, x -> (x, 10x))

        mktempdir() do dir
            f = export_pgfplots(joinpath(dir, "t"), Ωₕ, "u" => cₕ)
            lines = readlines(f)
            @test lines[1] == "x u_1 u_2"
            row2 = parse.(Float64, split(lines[2]))
            @test row2[2] ≈ row2[1]           # u_1 = x
            @test row2[3] ≈ 10 * row2[1]       # u_2 = 10x
        end
    end

    @testset "1D wrong length" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 5, true)
        mktempdir() do dir
            @test_throws "has length 3, but the mesh has 5 points" export_pgfplots(
                joinpath(dir, "t"), Ωₕ, "u" => [1.0, 2.0, 3.0])
        end
    end

    @testset "2D scanlines" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (3, 4), (true, true))
        Wₕ = gridspace(Ωₕ)
        uₕ = Rₕ(Wₕ, x -> x[1] + 10x[2])

        mktempdir() do dir
            f = export_pgfplots(joinpath(dir, "t"), Ωₕ, "u" => uₕ)
            lines = readlines(f)

            @test startswith(lines[1], "%")   # a comment line, which pgfplots skips

            # 3 scan lines of 4 rows each, separated by exactly one blank line, with no
            # leading or trailing blank line.
            body = lines[2:end]
            @test count(==(""), body) == 2
            @test body[1] != ""
            @test body[end] != ""

            blanks = findall(==(""), body)
            @test blanks == [5, 10]   # after row 4 and after row 9 of a 4-row scan line

            # every value on a blank-separated block shares the same x (the outer,
            # blank-line-grouped coordinate), and every block's y values sweep the full
            # inner axis in order.
            x, y = points(Ωₕ)
            blocks = [body[1:4], body[6:9], body[11:14]]
            for (k, block) in enumerate(blocks)
                rows = [parse.(Float64, split(l)) for l in block]
                @test all(r -> r[1] ≈ x[k], rows)
                @test [r[2] for r in rows] ≈ y
                @test [r[3] for r in rows] ≈ [x[k] + 10y[j] for j in eachindex(y)]
            end
        end
    end

    @testset "2D field constraints" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (3, 4), (true, true))
        Wₕ = gridspace(Ωₕ)
        uₕ = Rₕ(Wₕ, x -> x[1])
        Vₕ = Wₕ^Val(2)
        cₕ = Rₕ(Vₕ, x -> (x[1], x[2]))

        mktempdir() do dir
            @test_throws "one scalar field per 2D file" export_pgfplots(
                joinpath(dir, "t"), Ωₕ, "u" => uₕ, "v" => uₕ)
            @test_throws "one scalar field per 2D file" export_pgfplots(
                joinpath(dir, "t"), Ωₕ, "velocity" => cₕ)
        end
    end

    @testset "3D refusal" begin
        Ωₕ = mesh(domain(box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))), (3, 3, 3),
            (true, true, true))
        Wₕ = gridspace(Ωₕ)
        uₕ = Rₕ(Wₕ, x -> x[1])
        mktempdir() do dir
            @test_throws "use export_vtk instead" export_pgfplots(
                joinpath(dir, "t"), Ωₕ, "u" => uₕ)
        end
    end

    @testset "Single-element shorthand" begin
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 4, true)
        Wₕ = gridspace(Ωₕ)
        uₕ = Rₕ(Wₕ, sin)
        mktempdir() do dir
            f = export_pgfplots(joinpath(dir, "t"), uₕ)
            @test readlines(f)[1] == "x u"
        end
    end

    @testset "Filename already has a recognised extension" begin
        # Every other testset writes to an extension-less path, always taking
        # `_pgf_filename`'s "append .dat" branch; a name that already ends in one of the
        # recognised extensions must come back unchanged instead of doubly-suffixed.
        Ωₕ = mesh(domain(interval(0.0, 1.0)), 4, true)
        Wₕ = gridspace(Ωₕ)
        uₕ = Rₕ(Wₕ, sin)
        mktempdir() do dir
            for ext in (".dat", ".txt", ".tsv", ".csv")
                path = joinpath(dir, "t" * ext)
                f = export_pgfplots(path, Ωₕ, "u" => uₕ)
                @test f == path
            end
        end
    end

    @testset "2D field data as a plain array, not a VectorElement" begin
        # Every other 2D testset passes a VectorElement; `_pgf_grid` also accepts a plain
        # matrix (already shaped like the grid) or a plain vector (reshaped to it), each
        # with its own size/length check.
        Ωₕ = mesh(domain(interval(0.0, 1.0) × interval(0.0, 1.0)), (3, 4), (true, true))
        x, y = points(Ωₕ)
        Z = [xi + 10yj for xi in x, yj in y]   # already (nx, ny)-shaped

        mktempdir() do dir
            f_mat = export_pgfplots(joinpath(dir, "m"), Ωₕ, "u" => Z)
            f_vec = export_pgfplots(joinpath(dir, "v"), Ωₕ, "u" => vec(Z))
            @test read(f_mat, String) == read(f_vec, String)

            # and matches the VectorElement route for the same underlying values
            Wₕ = gridspace(Ωₕ)
            uₕ = Rₕ(Wₕ, x -> x[1] + 10x[2])
            f_ref = export_pgfplots(joinpath(dir, "r"), Ωₕ, "u" => uₕ)
            @test read(f_mat, String) == read(f_ref, String)

            @test_throws "has size (2, 4), but the mesh has (3, 4) points" export_pgfplots(
                joinpath(dir, "bad"), Ωₕ, "u" => Z[1:2, :])
            @test_throws "has length 11, but the mesh has 12 points" export_pgfplots(
                joinpath(dir, "bad2"), Ωₕ, "u" => vec(Z)[1:11])
        end
    end
end
