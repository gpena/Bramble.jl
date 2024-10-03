abstract type Exporter{SpaceType} <: BrambleType end

#=
Template for Exporter structs

struct ExporterSomething{SType,EltType} <: Exporter{SType}
    space::SType
    filename::String
    export_dir::String
    time_dependent::Bool
    scalar_datasets::Dict{String,EltType}
    time_level::Ref{Int}
end
=#


space(exporter::ExportType) where {ExportType <: Exporter} = exporter.space
datasets(exporter::ExportType) where {ExportType <: Exporter} = exporter.scalar_datasets
mesh(exporter::ExportType) where {ExportType <: Exporter} = exporter |> space |> mesh
filename(exporter::ExportType) where {ExportType <: Exporter} = exporter.filename
dir(exporter::ExportType) where {ExportType <: Exporter} = exporter.export_dir

fullPath(exporter::ExportType) where {ExportType <: Exporter} = dir(exporter)*filename(exporter)

isTimeDependent(exporter::ExportType) where {ExportType <: Exporter} = exporter.time_dependent
timeLevel(exporter::ExportType) where {ExportType <: Exporter} = isTimeDependent(exporter) ? exporter.time_level[] : 0
incrementTimestep!(exporter::ExportType) where {ExportType <: Exporter} = (exporter.time_level[] += 1)
