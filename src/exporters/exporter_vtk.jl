abstract type Exporter{SpaceType} end

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

space(exporter::ExportType) where {ExportType<:Exporter}    = exporter.space
datasets(exporter::ExportType) where {ExportType<:Exporter} = exporter.scalar_datasets
mesh(exporter::ExportType) where {ExportType<:Exporter}     = exporter |> space |> mesh
filename(exporter::ExportType) where {ExportType<:Exporter} = exporter.filename
dir(exporter::ExportType) where {ExportType<:Exporter}      = exporter.export_dir

fullPath(exporter::ExportType) where {ExportType<:Exporter} = dir(exporter) * filename(exporter)

isTimeDependent(exporter::ExportType) where {ExportType<:Exporter} = exporter.time_dependent
timeLevel(exporter::ExportType) where {ExportType<:Exporter} = isTimeDependent(exporter) ? exporter.time_level[] : 0
incrementTimestep!(exporter::ExportType) where {ExportType<:Exporter} = (exporter.time_level[] += 1)

struct ExporterVTK{SType,EltType} <: Exporter{SType}
	space::SType
	filename::String
	export_dir::String
	time_dependent::Bool
	scalar_datasets::Dict{String,EltType}
	pvd::WriteVTK.CollectionFile
	time_level::Ref{Int}
end

#space(exporter::ExporterVTK) = exporter.space
#datasets(exporter::ExporterVTK) = exporter.scalar_datasets

function ExporterVTK(space::SpaceType, filename = "data", export_dir = "./"; time::Bool = false) where {SpaceType}
	M = mesh(space)
	Eltype = typeof(reshape(Element(space).values, npoints(M)))

	return ExporterVTK{typeof(space),Eltype}(space, filename, export_dir, time, Dict(), paraview_collection(export_dir * filename), Ref(0))
end

function addScalarDataset!(exporter::ExporterVTK, name::String, dataset::EltType) where {EltType<:AbstractVector}
	exporter.scalar_datasets[name] = reshape(dataset, npoints(mesh(exporter)))
end

function addScalarDataset!(exporter::ExporterVTK, name::String, dataset::EltType) where {EltType<:VectorElement}
	addScalarDataset!(exporter, name, dataset.values)
end

function addScalarDataset!(exporter::ExporterVTK, name::String, dataset::EltType) where {EltType<:AbstractArray}
	exporter.scalar_datasets[name] = dataset
end

function addScalarDataset!(exporter::ExporterVTK, name::String, f::FType) where {FType<:Function}
	Wh = space(exporter)
	M = mesh(Wh)

	s = Element(Wh).values
	v = reshape(s, npoints(M))

	mesh_points = points(M)
	for idx in indices(M)
		pts = _i2p(mesh_points, idx)
		v[idx] = f(pts)
	end

	addScalarDataset!(exporter, name, v)
end

_exporter_mesh_pts(exporter::ExporterVTK) = _exporter_mesh_pts(exporter, Val(dim(mesh(space(exporter)))))
_exporter_mesh_pts(exporter::ExporterVTK, ::Val{1}) = error("ExporterVTK: not implemented for dimension 1")
_exporter_mesh_pts(exporter::ExporterVTK, ::Val{2}) = ((exporter |> mesh |> points)..., [zero(eltype(space(exporter)))])
_exporter_mesh_pts(exporter::ExporterVTK, ::Val{3}) = exporter |> mesh |> points

function save2file(exporter::ExporterVTK; filename::String = filename(exporter), export_dir::String = dir(exporter), time::AbstractFloat = 0.0)
	save2file(exporter, Val(dim(mesh(exporter))), Val(isTimeDependent(exporter)), filename, export_dir, time)
end

save2file(_::ExporterVTK, ::Val{1}, ::Val{false}, _::String, _::String, _::AbstractFloat) = error("ExporterVTK: not implemented for dimension 1")
save2file(_::ExporterVTK, ::Val{1}, ::Val{true}, _::String, _::String, _::AbstractFloat) = error("ExporterVTK: not implemented for dimension 1")

function save2file(exporter::ExporterVTK, ::Val{D}, ::Val{false}, filename::String, export_dir::String, _::AbstractFloat) where D
	export_file = export_dir * filename

	exporter_mesh_pts = _exporter_mesh_pts(exporter)

	vtk = vtk_grid(export_file, exporter_mesh_pts...)
	for key in exporter.scalar_datasets
		vtk[first(key)] = last(key)
	end

	#=
	for key in exporter.vector_datasets
		vtk[first(key)] = last(key)
	end
	=#

	vtk_save(vtk)
end

function save2file(exporter::ExporterVTK, ::Val{D}, ::Val{true}, filename::String, export_dir::String, time::AbstractFloat) where D
	exporter_mesh_pts = _exporter_mesh_pts(exporter)
	incrementTimestep!(exporter)

	fname = fullPath(exporter) * "_$(timeLevel(exporter))"

	vtk = vtk_grid(fname, exporter_mesh_pts...)
	for key in exporter.scalar_datasets
		vtk[first(key)] = last(key)
	end

	#=
	for key in exporter.vector_datasets
		vtk[first(key)] = last(key)
	end
	=#

	exporter.pvd[time] = vtk
end

function close(exporter::ExporterVTK)
	vtk_save(exporter.pvd)
end