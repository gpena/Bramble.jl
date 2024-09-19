var documenterSearchIndex = {"docs":
[{"location":"api/","page":"API","title":"API","text":"CollapsedDocStrings = false","category":"page"},{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Documentation for Bramble.jl's public API.","category":"page"},{"location":"api/#Geometry","page":"API","title":"Geometry","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"cartesianproduct\ninterval\n×\ndomain\ncreate_markers\nmarkers\nlabels","category":"page"},{"location":"api/#Bramble.cartesianproduct","page":"API","title":"Bramble.cartesianproduct","text":"cartesianproduct(data::NTuple)\n\nReturns a CartesianProduct from a tuple of intervals.\n\n\n\n\n\ncartesianproduct(x, y)\n\nReturns a 1D CartesianProduct from two scalars x and y, where x and y are, respectively, the lower and upper bounds of the interval.\n\nExample\n\njulia> cartesianproduct(0, 1)\nType: Float64 \n Dim: 1 \n Set: [0.0, 1.0]\n\n\n\n\n\n","category":"function"},{"location":"api/#Bramble.interval","page":"API","title":"Bramble.interval","text":"interval(x, y)\n\nReturns a 1D CartesianProduct set from two scalars x and y, where x and y are, respectively, the lower and upper bounds of the interval.\n\nExample\n\njulia> interval(0, 1)\nCartesianProduct{1,Float64}((0.0,1.0))\n\n\n\n\n\n","category":"function"},{"location":"api/#Bramble.:×","page":"API","title":"Bramble.:×","text":"×(X::CartesianProduct, Y::CartesianProduct)\n\nReturns the cartesian product of two CartesianProduct X and Y as a CartesianProduct.\n\nExample\n\njulia> X = cartesianproduct(0, 1); Y = cartesianproduct(2, 3);\n\t   X × Y;\nType: Float64 \n Dim: 2 \n Set: [0.0, 1.0] × [2.0, 3.0]\n\n\n\n\n\n","category":"function"},{"location":"api/#Bramble.domain","page":"API","title":"Bramble.domain","text":"Domain(Ω::CartesianProduct)\n\nCreates a Domain from a CartesianProduct assuming the single Marker \"Dirichlet\" => x -> zero(eltype(x)).\n\nExample\n\njulia> domain(Interval(0,1))\nType: Float64 \n Dim: 1 \n Set: [0.0, 1.0]\n\nBoundary markers: Dirichlet\n\n\n\n\n\nDomain(Ω::CartesianProduct, markers::MarkersType)\n\nCreates a Domain from a CartesianProduct assuming the single Marker \"Dirichlet\" => x -> zero(eltype(x)).\n\nExample\n\njulia> m = markers( \"Dirichlet\" => (x -> x-1), \"Neumann\" => (x -> x-0) ); Domain(Interval(0,1), m)\nType: Float64 \n Dim: 1 \n Set: [0.0, 1.0]\n\nBoundary markers: Dirichlet, Neumann\n\n\n\n\n\n","category":"function"},{"location":"api/#Bramble.create_markers","page":"API","title":"Bramble.create_markers","text":"create_markers(m::MarkerType...)\n\nConverts several Pair{String,F} (\"label\" => func) to domain Markers to be passed in the construction of a Domain Ω.\n\nExample\n\njulia> create_markers( \"Dirichlet\" => (x -> x-1), \"Neumann\" => (x -> x-0) )\n\n\n\n\n\n","category":"function"},{"location":"api/#Bramble.markers","page":"API","title":"Bramble.markers","text":"markers(Ω::Domain)\n\nReturns a generator with the Markers associated with a Domain Ω.\n\n\n\n\n\n","category":"function"},{"location":"api/#Bramble.labels","page":"API","title":"Bramble.labels","text":"labels(Ω::Domain)\n\nReturns a generator with the labels of the Markers associated with a Domain Ω.\n\n\n\n\n\n","category":"function"},{"location":"api/#Mesh","page":"API","title":"Mesh","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"mesh\nhₘₐₓ","category":"page"},{"location":"api/#Bramble.mesh","page":"API","title":"Bramble.mesh","text":"mesh(Ω::Domain, npts::Int, unif::Bool)\n\nReturns a Mesh1D based on Domain Ω and npts points with uniform spacing if unif is true (otherwise, the points are randomly generated on the domain).\n\nFor future reference, we denote the npts entries of vector pts as\n\nx_i  i=1dotsN\n\nExample\n\njulia> I = interval(0,1); Ωₕ = mesh(domain(I), 10, true)\n1D Mesh\nnPoints: 10\nMarkers: Dirichlet\n\n\n\n\n\nmesh(Ω::Domain, npts::NTuple{D,Int}, unif::NTuple{D,Bool})\n\nReturns a MeshnD (1 leq n leq 3) from the Domain Ω. The number of points for each coordinate projection mesh are given in the tuple npts. The distribution of points on the submeshes are encoded in the tuple unif. The Domain markers are translated to markers as for Mesh1D.\n\nFor future reference, the mesh points are denoted as\n\n2D mesh, with npts = (N_x, N_y)\n\n(x_iy_j)  i=1dotsN_x  j=1dotsN_y\n\n3D mesh, with npts = (N_x, N_y, N_z)\n\n(x_iy_jz_l)  i=1dotsN_x  j=1dotsN_y  l=1dotsN_z\n\nExample\n\njulia> X = Domain(Interval(0,1) × Interval(4,5)); Ωₕ = Mesh(X, (10, 15), (true, false))\n2D Mesh\nnPoints: 150\nMarkers: [\"Dirichlet\"]\n\nSubmeshes:\n  x direction | nPoints: 10\n  y direction | nPoints: 15\n\n\n\n\n\n","category":"function"},{"location":"api/#Bramble.hₘₐₓ","page":"API","title":"Bramble.hₘₐₓ","text":"hₘₐₓ(Ωₕ::Mesh1D)\n\nReturns the maximum over the space stepsize h_iof mesh Ωₕ\n\nh_max = max_i=1dotsN x_i - x_i-1\n\n\n\n\n\nhₘₐₓ(Ωₕ::MeshnD)\n\nReturns the maximum diagonal of mesh Ωₕ\n\n2D mesh\n\nmax_ij Vert (h_xi h_yj) Vert_2\n\n3D mesh\n\nmax_ijl Vert (h_xi h_yj  h_zl) Vert_2\n\n\n\n\n\n","category":"function"},{"location":"api/#Space","page":"API","title":"Space","text":"","category":"section"},{"location":"examples/#Linear-Poisson-equation","page":"Linear Poisson equation","title":"Linear Poisson equation","text":"","category":"section"},{"location":"#Bramble.jl","page":"Home","title":"Bramble.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This documentation is for Bramble.jl, a Julia library implementing discretization methods to solve partial differential equations using finite differences on nonuniform grids.","category":"page"},{"location":"","page":"Home","title":"Home","text":"For more information on the types of discretizations encompassed by Bramble.jl, please consult the papers","category":"page"},{"location":"","page":"Home","title":"Home","text":"J. A. Ferreira and R. D. Grigorieff, On the supraconvergence of elliptic finite difference schemes, Applied Numerical Mathematics 28 (1998), pp. 275-292\nS. Barbeiro, J. A. Ferreira and R. D. Grigorieff, Supraconvergence of a finite difference scheme for solutions in H^s(0L), IMA Journal of Numerical Analysis 25.4 (2005), pp. 797–811\nJ. A. Ferreira and R. D. Grigorieff, Supraconvergence and Supercloseness of a Scheme for Elliptic Equations on Nonuniform Grids, Numerical Functional Analysis and Optimization 27.5-6 (2006), pp. 539–564","category":"page"},{"location":"internals/","page":"Internal","title":"Internal","text":"CollapsedDocStrings = false","category":"page"},{"location":"internals/#Internal","page":"Internal","title":"Internal","text":"","category":"section"},{"location":"internals/","page":"Internal","title":"Internal","text":"Documentation for Bramble.jl's functions.","category":"page"},{"location":"internals/#Geometry","page":"Internal","title":"Geometry","text":"","category":"section"},{"location":"internals/","page":"Internal","title":"Internal","text":"Modules = [Bramble]\nPublic = false\nPages = [\"sets.jl\", \"domains.jl\"]","category":"page"},{"location":"internals/#Bramble.CartesianProduct","page":"Internal","title":"Bramble.CartesianProduct","text":"struct CartesianProduct{D,T}\n\tdata::NTuple{D,Tuple{T,T}}\nend\n\nType for storage of cartesian products of D intervals having elements of type T.\n\n\n\n\n\n","category":"type"},{"location":"internals/#Base.eltype-Union{Tuple{Bramble.CartesianProduct{D, T}}, Tuple{T}, Tuple{D}} where {D, T}","page":"Internal","title":"Base.eltype","text":"eltype(X::CartesianProduct)\n\nReturns the element type of a CartesianProduct.\n\nExample\n\njulia> X = cartesianproduct(0, 1); eltype(X)\nFloat64\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.dim-Union{Tuple{Bramble.CartesianProduct{D}}, Tuple{D}} where D","page":"Internal","title":"Bramble.dim","text":"dim(X::CartesianProduct)\n\nReturns the topological dimension of a CartesianProduct.\n\nExample\n\njulia> X = cartesianproduct(0, 1); dim(X)\n1\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.projection-Tuple{Bramble.CartesianProduct, Any}","page":"Internal","title":"Bramble.projection","text":"projection(X::CartesianProduct, i)\n\nReturns the i-th 1D CartesianProduct of the CartesianProduct X.\n\nExample\n\njulia> X = cartesianproduct(0, 1) × cartesianproduct(4, 5); projection(X, 1)\nType: Float64 \n Dim: 1 \n Set: [0.0, 1.0]\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.tails-Tuple{Bramble.CartesianProduct, Any}","page":"Internal","title":"Bramble.tails","text":"tails(X::CartesianProduct, i)\n\nReturns a tuple with the 1D CartesianProduct of the i-th interval of the CartesianProduct X.\n\nExample\n\njulia> X = cartesianproduct(0, 1) × cartesianproduct(4, 5); tails(X,1)\n(0.0, 1.0)\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.tails-Union{Tuple{Bramble.CartesianProduct{D}}, Tuple{D}} where D","page":"Internal","title":"Bramble.tails","text":"tails(X::CartesianProduct)\n\nReturns a tuple of tuples with 1D CartesianProducts that make up the CartesianProduct X.\n\nExample\n\njulia> X = cartesianproduct(0, 1) × cartesianproduct(4, 5); tails(X)\n((0.0, 1.0), (4.0, 5.0))\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.Domain","page":"Internal","title":"Bramble.Domain","text":"struct Domain{SetType, MarkersType}\n\tset::SetType\n\tmarkers::MarkersType\nend\n\nStructure to represent a domain composed of a CartesianProduct and a set of Markers.\n\n\n\n\n\n","category":"type"},{"location":"internals/#Bramble.DomainBaseType","page":"Internal","title":"Bramble.DomainBaseType","text":"DomainBaseType\n\nAn abstract type for representing domains.\n\n\n\n\n\n","category":"type"},{"location":"internals/#Bramble.Marker","page":"Internal","title":"Bramble.Marker","text":"struct Marker{F<:Function}\n\tlabel::String\n\tf::F\nend\n\nStructure to implement markers for a portion of a domain. Each Marker is composed of a label and a levelset function that identifies a portion of the domain.\n\n\n\n\n\n","category":"type"},{"location":"internals/#Base.eltype-Tuple{Bramble.Domain}","page":"Internal","title":"Base.eltype","text":"eltype(Ω::Domain)\n\nReturns the element type of a Domain Ω.\n\nExample\n\njulia> eltype(Domain(I × I))\nFloat64\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.dim-Tuple{Bramble.Domain}","page":"Internal","title":"Bramble.dim","text":"dim(Ω::DomainBaseType)\n\nReturns the topological dimension of a Domain Ω.\n\nExample\n\njulia> I = Interval(0.0, 1.0); dim(Domain(I × I))\n2\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.marker_funcs-Tuple{Bramble.Domain}","page":"Internal","title":"Bramble.marker_funcs","text":"marker_funcs(Ω::Domain)\n\nReturns a generator with the Markers levelset functions associated with a Domain Ω.\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.projection-Tuple{Bramble.Domain, Int64}","page":"Internal","title":"Bramble.projection","text":"projection(Ω::Domain, i::Int)\n\nReturns the CartesianProduct of the i-th projection of the set of the Domain Ω. \n\nFor example, projection(Domain(I × I), 1) will return I.\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.set-Tuple{Bramble.Domain}","page":"Internal","title":"Bramble.set","text":"set(Ω::Domain)\n\nReturns the CartesianProduct associated with a Domain Ω.\n\n\n\n\n\n","category":"method"},{"location":"internals/#Mesh","page":"Internal","title":"Mesh","text":"","category":"section"},{"location":"internals/#1D-Meshes","page":"Internal","title":"1D Meshes","text":"","category":"section"},{"location":"internals/","page":"Internal","title":"Internal","text":"Modules = [Bramble]\nPublic = false\nPages = [\"mesh1d.jl\"]","category":"page"},{"location":"internals/#Bramble.Mesh1D","page":"Internal","title":"Bramble.Mesh1D","text":"struct Mesh1D{T} <: MeshType{1}\n\tmarkers::MeshMarkers{1}\n\tindices::CartesianIndices{1,Tuple{Base.OneTo{Int}}}\n\tpts::Vector{T}\n\tnpts::Int\nend\n\nStructure to create a 1D mesh with npts points of type T. The points that define the mesh are stored in pts and are identified, following the same order, with the indices in indices. The variable markers stores, for each Domain marker, the indices satisfying f(x)=0, where fis the marker function.\n\nFor future reference, the npts entries of vector pts are\n\nx_i  i=1dotsN\n\n\n\n\n\n","category":"type"},{"location":"internals/#Bramble.addmarkers!-Tuple{Dict{String, Set{CartesianIndex{1}}}, Bramble.Domain, CartesianIndices{1, R} where R<:Tuple{OrdinalRange{Int64, Int64}}, Any}","page":"Internal","title":"Bramble.addmarkers!","text":"addmarkers!(markerList::MeshMarkers{1}, Ω::Domain, R::CartesianIndices{1}, pts)\n\nFor each Domain marker, stores in markerList the indices of the points that satisfy f(x)=0, where f is the corresponding levelset function associated with the marker.\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.bcindices-Tuple{Bramble.Mesh1D}","page":"Internal","title":"Bramble.bcindices","text":"bcindices(Ωₕ::Mesh1D)\n\nReturns the indices of the boundary points of mesh Ωₕ.\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.cell_measure-Tuple{Bramble.Mesh1D, CartesianIndex{1}}","page":"Internal","title":"Bramble.cell_measure","text":"cell_measure(Ωₕ::Mesh1D, i::CartesianIndex)\n\nReturns the measure of the cell square_i = x_i - frach_i2 x_i + frach_i+12 at CartesianIndex i in mesh Ωₕ, which is given by h_i+12.\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.cell_measure_iterator-Tuple{Bramble.Mesh1D}","page":"Internal","title":"Bramble.cell_measure_iterator","text":"cell_measure_iterator(Ωₕ::Mesh1D)\n\nReturns an iterator over h_i+12  i=1dotsN in mesh Ωₕ.\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.create_mesh1d_basics-Tuple{Bramble.Domain, Int64, Bool}","page":"Internal","title":"Bramble.create_mesh1d_basics","text":"create_mesh1d_basics(Ω::Domain, npts::Int, unif::Bool)\n\nCreates the basic components of a 1D mesh, given a Domain Ω, the number of points npts and a boolean unif. The points are equally spaced if unif is true (otherwise, the points are randomly generated on the domain).\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.createpoints!-Union{Tuple{T}, Tuple{Vector{T}, Bramble.CartesianProduct{1, T}, Bool}} where T","page":"Internal","title":"Bramble.createpoints!","text":"createpoints!(x::Vector, I::CartesianProduct{1}, unif::Bool)\n\nOverrides the components of vector x with uniformly (unif = true) or randomly distributed (unif = false) points in the interval I.\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.generate_indices-Tuple{Int64}","page":"Internal","title":"Bramble.generate_indices","text":"generate_indices(npts::Int)\n\nReturns a CartesianIndices object for a vector of length npts.\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.half_points-Union{Tuple{T}, Tuple{Bramble.Mesh1D{T}, Any}} where T","page":"Internal","title":"Bramble.half_points","text":"function half_points(Ωₕ::Mesh1D, i)\n\nReturns the average of two neighboring, x_i+12, points in mesh Ωₕ, at index i\n\nx_i+12 = x_i + frach_i+12  i=1dotsN-1\n\nx_N+12 = x_N and x_12 = x_1.\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.half_points_iterator-Tuple{Bramble.Mesh1D}","page":"Internal","title":"Bramble.half_points_iterator","text":"half_points_iterator(Ωₕ::Mesh1D)\n\nReturns an iterator over all points x_i+12  i=1dotsN in mesh Ωₕ.\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.half_spacing-Union{Tuple{T}, Tuple{Bramble.Mesh1D{T}, Any}} where T","page":"Internal","title":"Bramble.half_spacing","text":"half_spacing(Ωₕ::Mesh1D, i)\n\nReturns the indexwise average of the space stepsize, h_i+12, at index i in mesh Ωₕ\n\nh_i+12 = frach_i + h_i+12  i=1dotsN-1\n\nwhere h_N+12 = frach_N2 and h_12 = frach_12.\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.half_spacing_iterator-Tuple{Bramble.Mesh1D}","page":"Internal","title":"Bramble.half_spacing_iterator","text":"half_spacing_iterator(Ωₕ::Mesh1D)\n\nReturns an iterator over all indexwise average of the space stepsizes h_i+12  i=1dotsN in mesh Ωₕ.\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.intindices-Tuple{Bramble.Mesh1D}","page":"Internal","title":"Bramble.intindices","text":"intindices(Ωₕ::Mesh1D)\n\nReturns the indices of the interior points of mesh Ωₕ.\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.npoints-Tuple{Bramble.Mesh1D}","page":"Internal","title":"Bramble.npoints","text":"npoints(Ωₕ::Mesh1D)\n\nReturns a 1-tuple of the number of points x_i in Ωₕ.\n\nExample\n\njulia> Ωₕ = Mesh(Domain(Interval(0,1)), 10, true); npoints(Ωₕ)\n(10,)\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.point-Tuple{Bramble.Mesh1D, Any}","page":"Internal","title":"Bramble.point","text":"point(Ωₕ::Mesh1D, i)\n\nReturns the i-th point of Ωₕ, x_i.\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.points-Tuple{Bramble.Mesh1D}","page":"Internal","title":"Bramble.points","text":"points(Ωₕ::Mesh1D)\n\nReturns a vector with all the points x_i  i=1dotsN in Ωₕ.\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.points_iterator-Tuple{Bramble.Mesh1D}","page":"Internal","title":"Bramble.points_iterator","text":"points_iterator(Ωₕ::Mesh1D)\n\nReturns an iterator over all points x_i  i=1dotsN in mesh Ωₕ.\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.spacing-Tuple{Bramble.Mesh1D, Any}","page":"Internal","title":"Bramble.spacing","text":"spacing(Ωₕ::Mesh1D, i)\n\nReturns the space stepsize, h_i at index i in mesh Ωₕ\n\nh_i = x_i - x_i-1  i=2dotsN\n\nwhere h_1 = x_2 - x_1.\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.spacing_iterator-Tuple{Bramble.Mesh1D}","page":"Internal","title":"Bramble.spacing_iterator","text":"spacing_iterator(Ωₕ::Mesh1D)\n\nReturns an iterator over all space step sizes h_i  i=1dotsN in mesh Ωₕ.\n\n\n\n\n\n","category":"method"},{"location":"internals/#nD-Meshes,-n2,3","page":"Internal","title":"nD Meshes, n=23","text":"","category":"section"},{"location":"internals/","page":"Internal","title":"Internal","text":"Modules = [Bramble]\nPublic = false\nPages = [\"meshnd.jl\"]","category":"page"},{"location":"internals/#Bramble.MeshnD","page":"Internal","title":"Bramble.MeshnD","text":"struct MeshnD{n,T} <: MeshType{n}\n\tmarkers::MeshMarkers{n}\n\tindices::CartesianIndices{n,NTuple{n,UnitRange{Int}}}\n\tnpts::Int\n\tsubmeshes::NTuple{n,Mesh1D{T}}\nend\n\nType to store a cartesian nD-mesh (2 leq n leq 3) with prod(npts) points of type T. For efficiency, the mesh points are not stored. Instead, we store the points of the 1D meshes that make up the nD mesh. To connect both nD and 1D meshes, we use the indices in indices. The Domain markers are translated to markers as for Mesh1D.\n\n\n\n\n\n","category":"type"},{"location":"internals/#Bramble.MeshnD-Tuple{Any}","page":"Internal","title":"Bramble.MeshnD","text":"(Ωₕ::MeshnD)(i)\n\nReturns the i-th submesh of Ωₕ.\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.addmarkers!-Union{Tuple{D}, Tuple{Dict{String, Set{CartesianIndex{D}}}, Bramble.Domain, Tuple{Vararg{Bramble.Mesh1D, D}}}} where D","page":"Internal","title":"Bramble.addmarkers!","text":"addmarkers!(mrks::MeshMarkers, Ω::Domain, submeshes::NTuple{D,Mesh1D})\n\nAdds the markers of Domain to the markers of the mesh, using the submeshes` in each coordinate direction.\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.boundary_indices-Tuple{Bramble.MeshnD}","page":"Internal","title":"Bramble.boundary_indices","text":"boundary_indices(Ωₕ::MeshnD)\n\nReturns the boundary indices of mesh Ωₕ.\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.cell_measure-Union{Tuple{T}, Tuple{D}, Tuple{Bramble.MeshnD{D, T}, CartesianIndex{D}}} where {D, T}","page":"Internal","title":"Bramble.cell_measure","text":"cell_measure(Ωₕ::MeshnD, idx::CartesianIndex)\n\nReturns the measure of the cell square_idx centered at the index idx\n\n2D mesh, square_ij = x_i - frach_xi2 x_i + frach_xi+12 times y_j - frach_yj2 y_j + frach_yj+12 is\n\nh_xi+12 h_yj+12\n\nwhere idx = (ij),\n\n3D mesh, square_ijl = x_i - frach_xi2 x_i + frach_xi+12 times y_j - frach_yj2 y_j + frach_yj+12 times z_l - frach_zl2 z_l + frach_zl+12 is\n\nh_xi+12 h_yj+12 h_zl+12\n\nwhere idx = (ijl).\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.generate_indices-Union{Tuple{Tuple{Vararg{Int64, D}}}, Tuple{D}} where D","page":"Internal","title":"Bramble.generate_indices","text":"generate_indices(nPoints::NTuple)\n\nReturns the CartesianIndices indices of a mesh with nPoints[i] in each direction.\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.half_points-Union{Tuple{D}, Tuple{Bramble.MeshnD{D}, Tuple{Vararg{Int64, D}}}} where D","page":"Internal","title":"Bramble.half_points","text":"half_points(Ωₕ::MeshnD, idx::NTuple)\n\nReturns a tuple with the half_points, for each submesh, of the points at index idx\n\n2D mesh, with idx = (ij)\n\n(x_i+12 y_j+12)\n\n3D mesh, with idx = (ijl)\n\n(x_i+12 y_j+12 z_l+12)\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.half_points_iterator-Union{Tuple{Bramble.MeshnD{D}}, Tuple{D}} where D","page":"Internal","title":"Bramble.half_points_iterator","text":"half_points_iterator(Ωₕ::MeshnD)\n\nReturns an iterator over all half_points points in mesh Ωₕ\n\n2D mesh\n\n(x_i+12 y_j+12) i = 1N_x j = 1N_y\n\n3D mesh\n\n(x_i+12 y_j+12 z_l+12) i = 1N_x j = 1N_y l = 1N_z\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.half_spacing-Union{Tuple{D}, Tuple{Bramble.MeshnD{D}, Tuple{Vararg{Int64, D}}}} where D","page":"Internal","title":"Bramble.half_spacing","text":"half_spacing(Ωₕ::MeshnD, idx::NTuple)\n\nReturns a tuple with the half_spacing, for each submesh, at index idx\n\n2D mesh, with idx = (ij)\n\n(h_xi+12 h_yj+12)\n\n3D mesh, with idx = (ijl)\n\n(h_xi+12 h_yj+12 h_zl+12)\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.half_spacing_iterator-Union{Tuple{Bramble.MeshnD{D}}, Tuple{D}} where D","page":"Internal","title":"Bramble.half_spacing_iterator","text":"half_spacing_iterator(Ωₕ::MeshnD)\n\nReturns an iterator over all mean space step sizes half_spacing in mesh Ωₕ\n\n2D mesh\n\n(h_xi+12 h_yj+12) i = 1N_x j = 1N_y\n\n3D mesh\n\n(h_xi+12 h_yj+12 h_zl+12) i = 1N_x j = 1N_y l = 1N_z\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.interior_indices-Union{Tuple{CartesianIndices{D, R} where R<:Tuple{Vararg{OrdinalRange{Int64, Int64}, D}}}, Tuple{D}} where D","page":"Internal","title":"Bramble.interior_indices","text":"interior_indices(R::CartesianIndices)\n\nReturns the interior indices of mesh Ωₕ.\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.is_boundary_index-Union{Tuple{D}, Tuple{CartesianIndex{D}, CartesianIndices{D, R} where R<:Tuple{Vararg{OrdinalRange{Int64, Int64}, D}}}} where D","page":"Internal","title":"Bramble.is_boundary_index","text":"is_boundary_index(idx::CartesianIndex, R::CartesianIndices)\n\nReturns true if the index idx is a boundary index of the mesh with indices stored in R.\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.npoints-Union{Tuple{Bramble.MeshnD{D, T}}, Tuple{T}, Tuple{D}} where {D, T}","page":"Internal","title":"Bramble.npoints","text":"npoints(Ωₕ::MeshnD)\n\nReturns a tuple with the number of points of mesh Ωₕ, in each coordinate direction.\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.point-Union{Tuple{D}, Tuple{Bramble.MeshnD{D}, Tuple{Vararg{Int64, D}}}} where D","page":"Internal","title":"Bramble.point","text":"point(Ωₕ::MeshnD, idx::NTuple)\n\nReturns the point at index idx of Ωₕ as a tuple\n\n2D mesh, with idx = (ij)\n\n(x_i y_j)\n\n3D mesh, with idx = (ijl)\n\n(x_i y_j z_l)\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.points-Union{Tuple{Bramble.MeshnD{D}}, Tuple{D}} where D","page":"Internal","title":"Bramble.points","text":"points(Ωₕ::MeshnD)\n\nReturns a tuple with the points of Ωₕ\n\n2D mesh, with npts = (N_x, N_y)\n\n(x_i_i=1^N_x y_j_j=1^N_y)\n\n3D mesh, with npts = (N_x, N_y, N_z)\n\n(x_i_i=1^N_x y_j_j=1^N_y z_l_l=1^N_z)\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.points_iterator-Union{Tuple{Bramble.MeshnD{D}}, Tuple{D}} where D","page":"Internal","title":"Bramble.points_iterator","text":"points_iterator(Ωₕ::MeshnD)\n\nReturns an iterator over all points in mesh Ωₕ\n\n2D mesh\n\n(x_i y_j) i = 1N_x j = 1N_y\n\n3D mesh\n\n(x_i y_j z_l) i = 1N_x j = 1N_y l = 1N_z\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.spacing-Union{Tuple{D}, Tuple{Bramble.MeshnD{D}, Tuple{Vararg{Int64, D}}}} where D","page":"Internal","title":"Bramble.spacing","text":"spacing(Ωₕ::MeshnD, idx::NTuple)\n\nReturns a tuple with the spacing, for each submesh, at index idx\n\n2D mesh, with idx = (ij)\n\n(h_xi h_yj) = (x_i - x_i-1 y_j - y_j-1)\n\n3D mesh, with idx = (ijl)\n\n(h_xi h_yj h_zl) = (x_i - x_i-1 y_j - y_j-1 z_l - z_l-1)\n\n\n\n\n\n","category":"method"},{"location":"internals/#Bramble.spacing_iterator-Union{Tuple{Bramble.MeshnD{D}}, Tuple{D}} where D","page":"Internal","title":"Bramble.spacing_iterator","text":"spacing_iterator(Ωₕ::MeshnD{D})\n\nReturns an iterator over all space step sizes spacing in mesh Ωₕ\n\n2D mesh\n\n(h_xi h_yj) i = 1N_x j = 1N_y\n\n3D mesh\n\n(h_xi h_yj h_zl) i = 1N_x j = 1N_y l = 1N_z\n\n\n\n\n\n","category":"method"}]
}
