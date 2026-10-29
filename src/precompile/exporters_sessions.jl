# precompile/exporters_sessions.jl: PGFPlots export (src/exporters/).

function _pc_exporters_session(Ω1, Ω2)
    W1 = gridspace(Ω1)
    W2 = gridspace(Ω2)
    u1 = element(W1, 1.0)
    u2 = element(W2, 1.0)
    # A composite element, the PDE-system case: `_pgf_columns`' composite method.
    uv = Rₕ(W1^2, x -> (x, x^2))
    mktempdir() do d
        export_pgfplots(joinpath(d, "out1.dat"), Ω1, "u" => u1)
        export_pgfplots(joinpath(d, "out2.dat"), Ω2, "u" => u2)
        return export_pgfplots(joinpath(d, "out3.dat"), uv)
    end
    return nothing
end
