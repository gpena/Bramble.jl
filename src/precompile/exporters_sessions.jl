# precompile/exporters_sessions.jl: PGFPlots export (src/exporters/).

function _pc_exporters_session(Ω1, Ω2)
    W1 = gridspace(Ω1)
    W2 = gridspace(Ω2)
    u1 = Rₕ(W1, x -> 1.0)
    u2 = Rₕ(W2, x -> 1.0)
    mktempdir() do d
        export_pgfplots(joinpath(d, "out1.dat"), Ω1, "u" => u1)
        return export_pgfplots(joinpath(d, "out2.dat"), Ω2, "u" => u2)
    end
    return nothing
end
