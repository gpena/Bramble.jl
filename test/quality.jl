using Bramble
using Aqua
#=
@testset "Aqua" begin
    Aqua.find_persistent_tasks_deps(Bramble)
    Aqua.test_ambiguities(Bramble, recursive = false)
    #Aqua.test_deps_compat(Bramble)
    Aqua.test_piracies(Bramble)
    Aqua.test_project_extras(Bramble)
    #Aqua.test_stale_deps(Bramble)
    Aqua.test_unbound_args(Bramble)
    Aqua.test_undefined_exports(Bramble)
end
=#


