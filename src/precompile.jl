import PrecompileTools: @compile_workload, @setup_workload

__unif_choices() = (true, false)

function generate_unifs(::Val{1})
    gen = ((a,) for a in __unif_choices())
    return Tuple(gen)
end

generate_unifs(::Val{2}) = ((true, true),)

generate_unifs(::Val{3}) = ((true, true, true),)

generate_unifs() = ntuple(i->generate_unifs(Val(i)), 3)

## Domain precompilation
@compile_workload begin
    I0 = Interval(-3.0, 10.0)
    Ω0 = CartesianProduct(I0)

    eltype(I0), eltype(Ω0)
    dim(I0)

    npts = ((2,), (2, 2), (2, 2, 2))
    unif = generate_unifs()

    for i in 1:3
        for u in unif[i]
            Ii = ntuple(j->I0, i)
            Ω = reduce(×, Ii)
            projection(Ω, 1)
            
            X = Domain(Ω)
            set(X), projection(X, 1)
        end
    end
end


## Mesh precompilation
@compile_workload begin
    I0 = Interval(-3.0, 10.0)
    Ω0 = CartesianProduct(I0)

    eltype(I0), eltype(Ω0)
    dim(I0)

    npts = ((2,), (2, 2), (2, 2, 2))
    unif = generate_unifs()

    for i in 1:3
        for u in unif[i]
            X = Domain(reduce(×, ntuple(j->I0, i)))
            M = Mesh(X, npts[i], u)
            
            eltype(M), eltype(typeof(M)), dim(M)

            indices(M), npoints(M), M(1), ndofs(M),  hₘₐₓ(M)
            
            if i == 1 
                points(M,1), hspace(M,1), hmean(M,1), xmean(M,1)
            else
                points(M,npts[i]), hspace(M,npts[i]), hmean(M,npts[i]), xmean(M,npts[i])
            end

            c = CartesianIndex(npts[i]...)
            meas_cell(M,c)

            bcindices(M)
            intindices(M)

            pointsit(M), hspaceit(M), hmeanit(M), xmeanit(M), meas_cellit(M)
        end
    end
end


## Space compilation
@compile_workload begin
    I0 = Interval(-3.0, 10.0)

    npts = ((2,), (2, 2), (2, 2, 2))
    unif = generate_unifs()
    T = Float64

    for i in 1:3
        for u in unif[i]
            X = Domain(reduce(×, ntuple(j->I0, i)))
            M = Mesh(X, npts[i], u)

            Wh = GridSpace(M)

            dim(Wh), eltype(Wh), eltype(typeof(Wh))
            mesh(Wh), ndofs(Wh)

            uh = Element(Wh, 1.0)
            uh = Element(Wh, 1.0)
            wh = Element(Wh)
            dim(uh), eltype(uh), eltype(typeof(uh))
            length(u)

            Uh = Bramble.Elements(Wh)

            u1h = similar(uh.values)
            u2h = similar(u1h)
            copyto!(u1h, u2h)

            uh .= 0.0

            f(x) = sum(x)
            Rₕ(Wh, f)
            Rₕ!(uh, f)

            operators = ((D₋ₓ, jumpₓ, Mₕₓ, diff₋ₓ, Mₕ),
                        (D₋ₓ, D₋ᵧ, jumpₓ, jumpᵧ, diff₋ₓ, diff₋ᵧ, Mₕₓ, Mₕᵧ),
                        (D₋ₓ, D₋ᵧ, D₋₂, jumpₓ, jumpᵧ, jump₂, diff₋ₓ, diff₋ᵧ, diff₋₂, Mₕₓ, Mₕᵧ, Mₕ₂))
                        
            for op in operators[i]
                u1h .= op(uh)[1]
                u2h .= op(Wh)*uh.values
                op(Uh)
            end

            for op in (+, -, *, /, ^)
                res = op(uh, uh)
                map(op, uh, uh)
            end
            z = ∇ₕ(uh)
            normₕ(uh), snorm₁ₕ(uh), norm₁ₕ(uh), norm₊(z)

            #jump(z), diff(z), Mₕ(z)

            if i == 1
                continue
            end

            exporter0 = ExporterVTK(Wh, "dados", "./", time = true)
            addScalarDataset!(exporter0, "sol_numerica", uh)
            datasets(exporter0)
            save2file(exporter0; filename = "dados", export_dir = "./")
            

            exporter = ExporterVTK(Wh, "dados", "./", time = true)
            addScalarDataset!(exporter, "sol_numerica1", uh)
            addScalarDataset!(exporter, "sol_numerica2", uh.values)
            addScalarDataset!(exporter, "sol_exata", x -> x[1])
            save2file(exporter)
            addScalarDataset!(exporter, "sol_numerica1", 0.0*uh)
            addScalarDataset!(exporter, "sol_numerica1", similar(uh).values)
            addScalarDataset!(exporter, "sol_exata", x -> 0.0)
            save2file(exporter)
            Bramble.close(exporter)

            rm(fullPath(exporter)*"_1.vtr")
            rm(fullPath(exporter)*"_2.vtr")
            rm(fullPath(exporter)*".pvd")
            

        end
    end
end




## Special laplacian problem precompilation
@compile_workload begin
    I = Interval(0.0, 1.0)
    npts = ((10,), (10, 10), (10, 10, 10))
    unif = generate_unifs()

    for i in 1:3
        for u in unif[i]
            Ii = ntuple(j->I, i)
            Ω = Domain(reduce(×, Ii))

            sol(x) = exp(sum(x))
            g(x) = -3.0*sol(x)

            problem = Laplacian(Ω, sol, g)
            solveLaplacian(problem, npts[i], u)
        end
    end
end













@compile_workload begin
    I0 = Interval(-3.0, 10.0)

    npts = ((2,), (2, 2), (2, 2, 2))
    unif = generate_unifs()

    for i in 1:3
        for u in unif[i]
            X = Domain(reduce(×, ntuple(j->I0, i)))
            M = Mesh(X, npts[i], u)

            Wh = GridSpace(M)

            uh = Element(Wh, 1.0)

            bc = dirichletbcs("Dirichlet" => x -> maximum(abs.(x.-0.5)) - 0.5)
            bc1 = dirichletbcs("Dirichlet" => x -> maximum(abs.(x.-0.5)) - 0.5, "Neumann" => x -> maximum(abs.(x.-0.5)) - 0.5)
            dirichletbcs(x->x[1])

            if i == 1
                c1(U, V) = advection(U, V; scaling = 3.0) + stiffness(U, V; scaling = 1.0) 
                b1 = BilinearForm(c1, Wh, Wh)
                assemble(b1, bc)
                l1 = LinearForm(V->mass(uh, V), Wh)
                assemble(l1, bc)

                c2(U, V) = stiffness(U, V; scaling = uh) + advection(U, V; scaling = uh)
                assemble(BilinearForm(c2, Wh, Wh), bc)
                l2 = LinearForm(V->stiffness(uh, V), Wh)
                assemble(l2, bc)

                c3(U, V) = inner₊(1.0 * D₋ₓ(U) - 2.0 * Mₕ(U) + Mₕ(U), D₋ₓ(V)) + innerₕ(uh * jump(D₋ₓ(U)), jump(D₋ₓ(V)))
                assemble(BilinearForm(c3, Wh, Wh), bc)
            end
            

            a = BilinearForm( (u,v) -> innerₕ(u,v) + inner₊(∇ₕ(u),∇ₕ(v)), Wh, Wh)
            assemble(a)
            Ah = assemble(a, bc)
            assemble!(Ah, a, bc)
            LinearAlgebra.factorize(Ah)

            if i >= 2
                b = BilinearForm( (u,v) -> inner₊ₓ(Mₕₓ(u),Mₕₓ(v)) + inner₊ₓ(jumpₓ(Mₕₓ(u)),diffₓ(Mₕₓ(v))) + inner₊ᵧ(jumpᵧ(Mₕᵧ(u)), diffᵧ(Mₕᵧ(v))) , Wh, Wh)
                assemble(b)
                Ah = assemble(b, bc)
                assemble!(Ah, b, bc)
            end

            if i == 3
                c = BilinearForm( (u,v) -> inner₊ₓ(Mₕₓ(u),Mₕₓ(v)) + inner₊ₓ(jumpₓ(Mₕₓ(u)),diffₓ(Mₕₓ(v))) + inner₊ᵧ(jumpᵧ(Mₕᵧ(u)),diffᵧ(Mₕᵧ(v))) + inner₊₂(jump₂(Mₕ₂(u)),diff₂(Mₕ₂(v))), Wh, Wh)
                assemble(c)
                Ah = assemble(c, bc)
                assemble!(Ah, c, bc)
            end
            
            rhs = avgₕ(Wh, x->x[1])
            avgₕ!(rhs, x->x[1])
            l = LinearForm(v -> innerₕ(rhs, v), Wh)
            Fh = assemble(l, bc)
            assemble(l)
            assemble!(Fh, l, bc)

            fact = ilu(Ah, τ = 0.0001)
            solve!(uh, Ah, Fh, KrylovJL_GMRES(), prec = fact)
            solve!(uh, Ah, Fh, KrylovJL_GMRES())
            solve!(uh, Ah, Fh)
            Rₕ(Wh, x->x[1])
            Rₕ!(uh, x->x[1])
        end
    end
end


## Special linear and bilinear forms precompilation
@compile_workload begin
    I = Interval(0.0, 1.0)
    npts = ((4,), (4, 4), (4, 4, 4))
    unif = generate_unifs()

    for i in 1:3
        for u in unif[i]
            Ii = ntuple(j->I, i)
            X = Domain(reduce(×, Ii))
            Wh = Mesh(X, npts[i], u) |> GridSpace
            uh = Element(Wh, 1.0)
            bc = dirichletbcs("Dirichlet" => x -> maximum(abs.(x.-0.5)) - 0.5)

            m = Mass(Wh,Wh)
            lm = Mass(Wh)
            s = Diff(Wh,Wh)

            Ah = assemble(m)
            Fh = assemble(lm)

            linearforms = (lm,)
            bilinearforms = (s, m)

            forms = (linearforms..., bilinearforms...)
            containers = ntuple(i-> forms[i] in linearforms ? Fh : Ah, length(forms))

            for (form,container) in zip(forms, containers)
                update!(form, 1.0)
                update!(form, uh)
                assemble!(container, form)
                assemble!(container, form, bc)

                if form == s
                    update!(form, ntuple(_ -> 1.0, i))
                    update!(form, ntuple(_ -> uh, i))
                end
            end
        end
    end
end
